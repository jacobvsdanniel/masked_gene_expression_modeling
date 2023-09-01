#!/bin/bash

set -eux

model_type=$1
layers=$2
dimension=$3
neighbor=$4
seed=42

model=${model_type}__${layers}__${dimension}
mns=${model}__neighbor${neighbor}__seed${seed}

python train_neighbor.py \
--task train \
--cell_gene_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_logexp.npy \
--cell_neighbor_file /volume/penghsuanli-genome2-nas2/macrophage/emb/cell_hvgpca_neighbor.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/macrophage/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/macrophage/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/macrophage/train/${mns}.pt \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size 1 \
--model ${model} \
--neighbor_weight ${neighbor} \
--seed ${seed} \
2>&1 | tee log/train/${mns}.txt

