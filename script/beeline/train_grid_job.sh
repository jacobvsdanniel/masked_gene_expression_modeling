#!/bin/bash

set -eux

. /volume/penghsuanli-genome2-nas2/venv_dir/sc/bin/activate
cd /volume/penghsuanli-genome2-nas2/sc

gpu=2080
src=$1
attn=last
seed=42

for tf in HV
do
for nontfs in 500 1000
do
for layer in 1 2 4
do
for dim in 20 30 40 60 80 100
do
for head in 1 5
do
for ff in 20 30 40 60 80 100
do

dataset=${src}__${tf}TF__${nontfs}nonTFs
model=model_transformer__${layer}__${dim}__${head}__${ff}

seq=/volume/penghsuanli-genome2-nas2/beeline/seq/${dataset}
train=/volume/penghsuanli-genome2-nas2/beeline/train/${dataset}
mkdir -p ${train}/${gpu}
mkdir -p ${train}/weight/cell/${gpu}
log=/volume/penghsuanli-genome2-nas2/sc/log/beeline/${dataset}
mkdir -p ${log}/train
mkdir -p ${log}/weight

ms=${model}__seed${seed}

# check if training is complete
train_log_file=${log}/train/gpu${gpu}__${ms}.txt
if [ -f "${train_log_file}" ]; then
    train_last_log=$( tail -n 1 ${train_log_file} )
    if [[ ${train_last_log} == *"Best test result"* ]]; then
        echo "skip ${train_log_file}"
        continue
    fi
fi

python train.py \
--task train \
--cell_gene_file ${seq}/gene_index_seq.csv \
--cell_exp_file ${seq}/gene_logexp_seq.npy \
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
done
done
done

