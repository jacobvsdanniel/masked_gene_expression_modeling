#!/bin/bash

set -eux

model=$1
attn=$2
src=$3
seed1=$4
seed2=$5

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

mts=${model}__tf707__seed${seed}
mtas=${model}__tf707__attn${attn}__seed${seed}

python train_tf.py \
--task weight \
--cell_gene_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/retina/${src}/hvg/hvg_index.npy \
--tf_file /volume/penghsuanli-genome2-nas2/retina/${src}/tf/tf_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/${mts}.pt \
--weight_dir /volume/penghsuanli-genome2-nas2/retina/${src}/train/weight/cell/${mtas} \
--attn_layer ${attn} \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/retina/weight/${mtas}__${src}.txt

done

