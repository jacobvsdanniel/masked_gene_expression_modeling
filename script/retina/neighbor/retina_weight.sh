#!/bin/bash

set -eux

gpu=$1
model=$2
neighbor=1
attn=$3
src=$4
seed1=$5
seed2=$6

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

mnas=${model}__neighbor${neighbor}__attn${attn}__seed${seed}

python retina.py \
--model ${gpu}/${mnas} \
--source ${src} \
--processes 9 \
2>&1 | tee log/retina/weight/cell_to_all/gpu${gpu}__${mnas}__${src}.txt

done

