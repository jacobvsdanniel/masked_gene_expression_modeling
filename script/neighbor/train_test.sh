#!/bin/bash

set -eux

seed1=$1
seed2=$2
model=model_transformer__1__100
neighbor=1

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

mns=${model}__neighbor${neighbor}__seed${seed}

python train_neighbor.py \
--task test \
--cell_gene_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/macrophage/seq/gene_seq_logexp.npy \
--cell_neighbor_file /volume/penghsuanli-genome2-nas2/macrophage/emb/cell_hvgpca_neighbor.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/macrophage/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/macrophage/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/macrophage/train/${mns}.pt \
--weight_dir /volume/penghsuanli-genome2-nas2/macrophage/train/weight/cell/${mns} \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/test/${mns}.txt

done

