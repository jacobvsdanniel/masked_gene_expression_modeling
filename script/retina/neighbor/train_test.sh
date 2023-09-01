#!/bin/bash

set -eux

gpu=$1
model=$2
neighbor=1
src=$3
seed1=$4
seed2=$5

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

mns=${model}__neighbor${neighbor}__seed${seed}

python train_neighbor.py \
--task test \
--cell_gene_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_logexp.npy \
--cell_neighbor_file /volume/penghsuanli-genome2-nas2/retina/${src}/emb/cell_hvgpca_neighbor.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/retina/${src}/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/${gpu}/${mns}.pt \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/retina/test/gpu${gpu}__${mns}__${src}.txt

done

