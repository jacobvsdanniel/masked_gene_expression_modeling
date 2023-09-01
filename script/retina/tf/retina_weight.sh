#!/bin/bash

set -eux

model=$1
attn=$2
src=$3
seed1=$4
seed2=$5

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

mtas=${model}__tf707__attn${attn}__seed${seed}

python retina.py \
--model ${mtas} \
--source ${src} \
--processes 9 \
2>&1 | tee log/retina/weight/cell_to_all/${mtas}__${src}.txt

done

