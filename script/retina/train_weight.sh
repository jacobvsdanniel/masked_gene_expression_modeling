#!/bin/bash

set -eux

gpu=$1
model=$2
attn=$3
src=$4
seed1=$5
seed2=$6

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

ms=${model}__seed${seed}
mas=${model}__attn${attn}__seed${seed}

python train.py \
--task weight \
--cell_gene_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/retina/${src}/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/${gpu}/${ms}.pt \
--weight_dir /volume/penghsuanli-genome2-nas2/retina/${src}/train/weight/cell/${gpu}/${mas} \
--attn_layer ${attn} \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/retina/weight/gpu${gpu}__${mas}__${src}.txt

done

