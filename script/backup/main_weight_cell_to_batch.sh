#!/bin/bash

set -eux

model=$1
src=$2
i=$3
j=$4

python main.py \
--model ${model} \
--source ${src} \
--start ${i} \
--end ${j} \
2>&1 | tee log/weight/cell_to_batch/${model}_${src}_${i}_${j}.txt

