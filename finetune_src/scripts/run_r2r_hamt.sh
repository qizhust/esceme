#!/bin/bash

ob_type=pano
feedback=sample

features=vitbase_r2rfte2e
ft_dim=768

ngpus=1
seed=0

outdir=../datasets/R2R/finetune/hamt/vitbase_r2rfte2e-finetune_seed${seed}
rootdir=../datasets

flag="--root_dir ../datasets
      --output_dir ${outdir}
      --root_dir ${rootdir}

      --dataset r2r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}
      
      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4
      
      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 100000
      --log_every 2000
      --batch_size 8
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5"

# train
python r2r/main.py $flag --eval_first \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/trained_models_released/vitbase-6tasks-pretrain-e2e/model_step_22000.pt
       
# inference
python r2r/main.py $flag \
      --resume_file $outdir/ckpts/best_val_unseen \
      --test --submit
