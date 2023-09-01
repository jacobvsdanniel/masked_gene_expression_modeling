#!/bin/bash

set -eux

gpu=$1
model=$2
attn=$3
src=$4
seed1=$5
seed2=$6
pool=$7

ma=${model}__attn${attn}

python retina.py \
--model ${gpu}/${ma} \
--source ${src} \
--seed1 ${seed1} \
--seed2 ${seed2} \
--pool ${pool} \
2>&1 | tee log/retina/weight/pool_seeds/gpu${gpu}__${ma}__seed${pool}-${seed1}-${seed2}__${src}.txt

