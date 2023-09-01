#!/bin/bash

set -eux

gpu=$1
model=$2
# src=$3
attn=$4
seed1=$5
seed2=$6

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do
for src in TJ WC WJ TC-WC TJ-WJ TC-TJ-WC-WJ
do

mss=${model}__${src}__seed${seed}
msas=${model}__${src}__attn${attn}__seed${seed}

python train.py \
--task weight \
--cell_gene_file /volume/penghsuanli-genome2-nas2/brain2770/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/brain2770/seq/gene_seq_logexp.npy \
--cell_meta_file /volume/penghsuanli-genome2-nas2/brain2770/raw/metadata.csv \
--source ${src} \
--test_file /volume/penghsuanli-genome2-nas2/brain2770/train/test.json \
--train_test_split merge \
--model_file /volume/penghsuanli-genome2-nas2/brain2770/train/${gpu}/${mss}.pt \
--weight_dir /volume/penghsuanli-genome2-nas2/brain2770/train/weight/cell/${gpu}/${msas} \
--attn_layer ${attn} \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/brain2770/weight/gpu${gpu}__${msas}.txt

done
done

