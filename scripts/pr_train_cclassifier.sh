#!/usr/bin/env bash
export save=../results/peer_review/${RANDOM}
mkdir -p $save
python ../peer_review/train_cclassifier.py \
 --dataset cifar10 \
 --data_normalization \
 --data_root ../data \
 --out_folder $save \
 --batch_size 64 \
 --kl_beta 0.1 \
 --cuda \
 --gpu_index 0 \
 | tee $save/log.txt
