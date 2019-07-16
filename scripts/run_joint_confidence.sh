#!/usr/bin/env bash
export save=../results/confidence_classifier/${RANDOM}/
mkdir -p $save
python ../confidence_classifier/run_joint_confidence.py \
    --dataroot ../data \
    --dataset cifar10 \
    --outf $save \
    | tee $save/log.txt
