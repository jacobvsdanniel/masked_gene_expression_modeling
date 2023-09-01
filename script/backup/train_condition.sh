#!/bin/bash

set -eux

condition=$1
model_type=$2
layers=$3
dimension=$4

model=${model_type}__${layers}__${dimension}
condition_model=${condition}__${model}

python train_condition.py \
--condition ${condition} \
--cell_meta_file /volume/penghsuanli-genome2-nas2/macrophage/raw/metadata.csv \
--cell_gene_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/macrophage/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/macrophage/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/macrophage/train_condition/${condition_model}.pt \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size 1 \
--model ${model} \
2>&1 | tee log/train_condition/${condition_model}.txt

