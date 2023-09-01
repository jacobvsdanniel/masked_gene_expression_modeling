#!/bin/bash

set -eux

gpu=2080
model=$1
attn=last
seed1=42
seed2=42

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do
for src in TC-TJ-WC-WJ TC-WC TJ-WJ TC TJ WC WJ
do

mss=${model}__${src}__seed${seed}

python train.py \
--task train \
--cell_gene_file /volume/penghsuanli-genome2-nas2/brain2770/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/brain2770/seq/gene_seq_logexp.npy \
--cell_meta_file /volume/penghsuanli-genome2-nas2/brain2770/raw/metadata.csv \
--source ${src} \
--test_file /volume/penghsuanli-genome2-nas2/brain2770/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/brain2770/train/${gpu}/${mss}.pt \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size 1 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/brain2770/train/gpu${gpu}__${mss}.txt

msas=${model}__${src}__attn${attn}__seed${seed}

python train.py \
--task weight \
--cell_gene_file /volume/penghsuanli-genome2-nas2/brain2770/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/brain2770/seq/gene_seq_logexp.npy \
--cell_meta_file /volume/penghsuanli-genome2-nas2/brain2770/raw/metadata.csv \
--source ${src} \
--test_file /volume/penghsuanli-genome2-nas2/brain2770/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/brain2770/train/${gpu}/${mss}.pt \
--weight_dir /volume/penghsuanli-genome2-nas2/brain2770/train/weight/cell/${gpu}/${msas} \
--attn_layer ${attn} \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/brain2770/weight/gpu${gpu}__${msas}.txt

done
done

