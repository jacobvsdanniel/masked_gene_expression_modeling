#!/bin/bash

set -eux

gpu=2080
model=$1
src=$2
attn=last
seed1=42
seed2=42

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do
for tf in HV
do
for nontfs in 500 1000
do

dataset=${src}__${tf}TF__${nontfs}nonTFs

seq=/volume/penghsuanli-genome2-nas2/beeline/seq/${dataset}
mkdir -p ${seq}

train=/volume/penghsuanli-genome2-nas2/beeline/train/${dataset}
mkdir -p ${train}/${gpu}
mkdir -p ${train}/weight/cell/${gpu}

log=/volume/penghsuanli-genome2-nas2/sc/log/beeline/${dataset}
mkdir -p ${log}/train
mkdir -p ${log}/weight

ms=${model}__seed${seed}

python train.py \
--task train \
--cell_gene_file ${seq}/gene_index_seq.csv \
--cell_exp_file ${seq}/gene_logexp_seq.npy \
--train_test_split merge \
--test_file ${train}/test.json \
--model_file ${train}/${gpu}/${ms}.pt \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size 1 \
--model ${model} \
--seed ${seed} \
2>&1 | tee ${log}/train/gpu${gpu}__${ms}.txt

mas=${model}__attn${attn}__seed${seed}

python train.py \
--task weight \
--cell_gene_file ${seq}/gene_index_seq.csv \
--cell_exp_file ${seq}/gene_logexp_seq.npy \
--train_test_split merge \
--test_file ${train}/test.json \
--model_file ${train}/${gpu}/${ms}.pt \
--weight_dir ${train}/weight/cell/${gpu}/${mas} \
--attn_layer ${attn} \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee ${log}/weight/gpu${gpu}__${mas}.txt

done
done
done

