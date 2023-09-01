#!/bin/bash

set -eux

model=$1
src=$2
seed1=$3
seed2=$4

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

mts=${model}__tf707__seed${seed}

python train_tf.py \
--task test \
--cell_gene_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/retina/${src}/hvg/hvg_index.npy \
--tf_file /volume/penghsuanli-genome2-nas2/retina/${src}/tf/tf_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/${mts}.pt \
--weight_dir /volume/penghsuanli-genome2-nas2/retina/${src}/train/weight/cell/${mts} \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/retina/test/${mts}__${src}.txt

done

