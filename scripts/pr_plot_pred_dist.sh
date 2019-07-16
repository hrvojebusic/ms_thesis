#!/usr/bin/env bash
python ../peer_review/utils/plot_pred_dist.py \
 --data_root ../data \
 --in_dataset cifar10 \
 --out_datasets svhn imagenet lsun \
 --model_chp $1 \
 --out_folder $2 \
 --cuda \
 --gpu_index $3
