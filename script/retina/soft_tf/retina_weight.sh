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

mas=${model}__attn${attn}__seed${seed}

python retina.py \
--model ${gpu}/${mas} \
--source ${src} \
--processes 9 \
2>&1 | tee log/retina/weight/cell_to_all/gpu${gpu}__${mas}__${src}.txt

done

