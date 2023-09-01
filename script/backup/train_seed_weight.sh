#!/bin/bash

set -eux

model=model_transformer__1__100
seed=$1
ms=${model}__seed${seed}

python train.py \
--cell_gene_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/macrophage/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/macrophage/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/macrophage/train/${ms}.pt \
--weight_dir /volume/penghsuanli-genome2-nas2/macrophage/train/weight/cell/${ms} \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/test/${ms}.txt

