#!/bin/bash

ob_type=pano
pooling=max
feedback=sample

features=vitbase_r2rfte2e
ft_dim=768

ngpus=1
seed=0

outdir=../datasets/R2R/finetune/hamt_esceme/${ob_type}_${pooling}_seed${seed}
rootdir=../datasets

flag="--output_dir ${outdir}
      --root_dir ${rootdir}

      --dataset r2r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}
      --pooling ${pooling}
      
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

      --lr 5e-6
      --iters 100000
      --log_every 2000
      --batch_size 8
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5"
   
# inference
python r2r/main.py $flag --agent_name hamt_scanme \
      --resume_file /project/qiz/datasets/vln/R2R/finetune/hamt_node_e2e/pano_max_5e6_seed4/ckpts/best_val_unseen \
      --test --submit
                                                                                                                                                                                                                                                                                                                                                                          