#!/bin/bash

set -eux

model=$1
batch=$2

python train_autoencoder.py \
--cell_gene_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/macrophage/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/macrophage/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/macrophage/train_autoencoder/${model}.pt \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size ${batch} \
--model ${model} \
2>&1 | tee log/train_autoencoder/${model}__batch${batch}.txt

