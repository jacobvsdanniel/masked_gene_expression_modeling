#!/bin/bash

set -eux

model=$1
src=$2

python main.py \
--model ${model} \
--source ${src} \
2>&1 | tee log/weight/batch_to_all/${model}_${src}.txt

