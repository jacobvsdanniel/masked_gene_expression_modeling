#!/bin/bash

set -eux

gpu=$1
model=$2
src=$3
seed1=$4
seed2=$5

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

mss=${model}__${src}__seed${seed}

python train.py \
--task test \
--cell_gene_file /volume/penghsuanli-genome2-nas2/brain2770/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/brain2770/seq/gene_seq_logexp.npy \
--cell_meta_file /volume/penghsuanli-genome2-nas2/brain2770/raw/metadata.csv \
--source ${src} \
--test_file /volume/penghsuanli-genome2-nas2/brain2770/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/brain2770/train/${gpu}/${mss}.pt \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/brain2770/test/gpu${gpu}__${mss}.txt

done

