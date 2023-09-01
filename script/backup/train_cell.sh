#!/bin/bash

set -eux

model_type=model_transformer
layers=$1
dimension=$2
seed=42

model=${model_type}__${layers}__${dimension}
ms=${model}__${seed}

python train_cell.py \
--cell_emb_file /volume/penghsuanli-genome2-nas2/macrophage/emb/cell_umap.npy \
--cell_gene_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/macrophage/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/macrophage/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/macrophage/train/cell/${ms}.pt \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size 1 \
--model ${model} \
--backward all \
--seed ${seed} \
2>&1 | tee log/cell/train/${ms}.txt

