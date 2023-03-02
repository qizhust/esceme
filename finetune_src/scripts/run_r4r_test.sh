#!/bin/bash

features=vitbase_r2rfte2e
ft_dim=768
feedback=sample
ngpus=1

outdir=../datasets/R4R/finetune/hamt_esceme
rootdir=../datasets

flag="--dataset r4r
      --output_dir ${outdir}
      --root_dir ${rootdir}

      --seed 0
      --ngpus ${ngpus}

      --no_lang_ca
      --ob_type pano
      --hist_enc_pano
      --hist_pano_num_layers 2

      --features ${features}
      --feedback ${feedback}

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 100000
      --log_every 2000
      --optim adamW

      --ml_weight 0.2
      
      --batch_size 8
      --feat_dropout 0.4
      --dropout 0.5"

# inference
python r2r/main.py $flag --agent_name hamt_scanme --pooling max \
      --resume_file ${outdir}/ckpts/best_val_unseen_sampled \
      --test --submit
