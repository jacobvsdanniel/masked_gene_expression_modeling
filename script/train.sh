#!/bin/bash

set -eux

gpu=$1
model=$2
src=$3
seed1=$4
seed2=$5

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

mss=${model}__${src}__seed${seed}

python train.py \
--task train \
--cell_gene_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/macrophage/hvg/hvg_index.npy \
--cell_meta_file /volume/penghsuanli-genome2-nas2/macrophage/raw/metadata.csv \
--source ${src} \
--test_file /volume/penghsuanli-genome2-nas2/macrophage/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/macrophage/train/${gpu}/${mss}.pt \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size 1 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/train/gpu${gpu}__${mss}.txt

done

