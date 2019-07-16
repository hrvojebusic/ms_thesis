#!/usr/bin/env bash
python ../peer_review/test_performance.py \
 --data_root ../data \
 --data_normalization \
 --in_dataset cifar10 \
 --out_datasets svhn imagenet lsun \
 --out_folder $1 \
 --name $2 \
 --chp_folder $3 \
 --chp_prefix $4 \
 --cuda \
 --gpu_index $5 \
 | tee $1/$2_log.txt