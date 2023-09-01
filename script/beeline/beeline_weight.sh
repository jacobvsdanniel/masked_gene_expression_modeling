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
log=/volume/penghsuanli-genome2-nas2/sc/log/beeline/${dataset}/weight/cell_to_all
mkdir -p ${log}

mas=${model}__attn${attn}__seed${seed}

python beeline.py \
--task collect_weight \
--processes 9 \
--model ${gpu}/${mas} \
--source ${src} \
--tf ${tf} \
--nontfs ${nontfs} \
2>&1 | tee ${log}/gpu${gpu}__${mas}.txt

done
done
done

