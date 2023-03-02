#!/bin/bash

features=vitbase_r2rfte2e
ft_dim=768

ngpus=1
seed=0

outdir=../datasets/CVDN/finetune/hamt
rootdir=../datasets

flag="--output_dir ${outdir}
      --root_dir ${rootdir}

      --dataset cvdn
      --use_player_path

      --ob_type pano
      
      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4
      
      --hist_enc_pano
      --hist_pano_num_layers 2
      
      --no_lang_ca

      --features ${features}
      --feedback sample

      --max_action_len 30
      --max_instr_len 100

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 100000
      --log_every 2000
      --batch_size 4
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5"

# train
python cvdn/main.py $flag \
      --bert_ckpt_file ../datasets/R2R/trained_models_released/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
      --eval_first

# inference
python cvdn/main.py $flag \
      --resume_file ${outdir}/ckpts/best_val_unseen \
      --test --submit
