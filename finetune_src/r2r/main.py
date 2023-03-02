import os
import json
import time
from collections import defaultdict

import torch

from utils.misc import set_random_seed
from utils.logger import write_to_record_file
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init import get_tokenizer

from r2r.agent_cmt import Seq2SeqCMTAgent
from r2r.agent_scanme import ScanmeAgent
from r2r.agent_tdstp import TDSTPAgent
from r2r.agent_hamt_scanme import HamtScanmeAgent

from r2r.data_utils import ImageFeaturesDB, construct_instrs
from r2r.env import R2RBatch, R2RBackBatch
from r2r.env_tdstp import R2RBatch as R2RBatch_tdstp


from r2r.parser import parse_args
from engine import ClassicTrainer


def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    if args.dataset == 'r2r_back':
        dataset_class = R2RBackBatch
    else:
        if args.agent_name in ['tdstp', 'scanme']:
            dataset_class = R2RBatch_tdstp
        elif args.agent_name in ['seq2seq', 'hamt', 'hamt_scanme']:
            dataset_class = R2RBatch
        else:
            raise NotImplementedError

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes

    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], tokenizer=tok, max_instr_len=args.max_instr_len
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train'
    )
    
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], tokenizer=tok, max_instr_len=args.max_instr_len
        )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name='aug'
        )
    else:
        aug_env = None

    val_env_names = ['val_train_seen', 'val_seen']
    if args.test or args.dataset != 'r4r':
        val_env_names.append('val_unseen')
    else:   # val_unseen of r4r is too large to evaluate in training
        val_env_names.append('val_unseen_sampled')

    if args.submit:
        if args.dataset == 'r2r':
            val_env_names.append('test')
        elif args.dataset == 'rxr':
            val_env_names.extend(['test_challenge_public', 'test_standard_public'])
    
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split
        )
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def valid(args, agent, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    # for cnt in range(5):
    for env_name, env in val_envs.items():
        if os.path.exists(os.path.join(args.pred_dir, "submit_%s.json" % env_name)):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results()
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, rank=rank)

    if args.agent_name == 'scanme':  # tdstp_scanme
        agent_class = ScanmeAgent
    elif args.agent_name == 'tdstp':
        agent_class = TDSTPAgent
    elif args.agent_name == 'hamt_scanme':
        agent_class = HamtScanmeAgent
    elif args.agent_name in ['seq2seq', 'hamt']:  # HAMT
        agent_class = Seq2SeqCMTAgent
    else:
        raise NotImplementedError
    listner = agent_class(args, train_env, rank=rank)

    if not args.test:
        trainer = ClassicTrainer()
        trainer.train(args, listner, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        valid(args, listner, train_env, val_envs, rank=rank)            

if __name__ == '__main__':
    main()
