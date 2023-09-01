#!/bin/bash

set -eux

model=$1

python train.py \
--cell_meta_file /volume/penghsuanli-genome2-nas2/macrophage/raw/metadata.csv \
--index_seq_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_index.csv \
--exp_seq_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_logexp.npy \
--hvg_index_file /volume/penghsuanli-genome2-nas2/macrophage/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/macrophage/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/macrophage/train/${model}.pt \
--weight_dir /volume/penghsuanli-genome2-nas2/macrophage/train/weight \
--model ${model} \
2>&1 | tee log/test/${model}.txt

