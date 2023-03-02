#!/bin/bash

features=vitbase_r2rfte2e
ft_dim=768

ngpus=1
seed=0

outdir=../datasets/CVDN/finetune/hamt_esceme
rootdir=../datasets


flag="--root_dir ${rootdir}
      --output_dir ${outdir}

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

# inference
python cvdn/sr_curve.py $flag \
      --resume_file ${outdir}/ckpts/best_val_unseen \
      --test --submit
