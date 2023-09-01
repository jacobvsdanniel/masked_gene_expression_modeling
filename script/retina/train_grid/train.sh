#!/bin/bash

set -eux

. /volume/penghsuanli-genome2-nas2/venv_dir/sc/bin/activate
cd /volume/penghsuanli-genome2-nas2/sc

gpu=2080
seed=42
attn=last
train_test_split=split
dim=$1

for layer in 1 2 4
do
for dim in ${dim}
do
for head in 1 5
do
for ff in 20 30 40 60 80 100
do
for src in lukowski3101 menon3101
do

data_dir=/volume/penghsuanli-genome2-nas2/retina/${src}
mkdir -p ${data_dir}/train/${gpu}
log_dir=/volume/penghsuanli-genome2-nas2/sc/log/retina/${src}/train
mkdir -p ${log_dir}

model=model_transformer__${layer}__${dim}__${head}__${ff}
mst=${model}__seed${seed}__${train_test_split}Test

# check if training is complete
log_file=${log_dir}/gpu${gpu}__${mst}.txt
if [ -f "${log_file}" ]; then
    log_line=$( tail -n 1 ${log_file} )
    if [[ ${log_line} == *"Best test result"* ]]; then
        echo "skip ${log_file}"
        continue
    fi
fi

python train.py \
--task train \
--cell_gene_file ${data_dir}/seq/gene_seq_index.csv \
--cell_exp_file ${data_dir}/seq/gene_seq_logexp.npy \
--hvg_file ${data_dir}/hvg/hvg_index.npy \
--train_test_split ${train_test_split} \
--test_file ${data_dir}/train/test.json \
--model_file ${data_dir}/train/${gpu}/${mst}.pt \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size 1 \
--model ${model} \
--seed ${seed} \
2>&1 | tee ${log_file}

done
done
done
done
done

