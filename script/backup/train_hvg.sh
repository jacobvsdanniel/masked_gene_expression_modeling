#!/bin/bash

set -eux

model=model_transformer__1__100

python train.py \
--cell_gene_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/macrophage/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/macrophage/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/macrophage/train/${model}__hvg.pt \
--backward hvg \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size 1 \
--model ${model} \
2>&1 | tee log/train/${model}__hvg.txt

