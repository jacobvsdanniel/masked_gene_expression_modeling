#!/bin/bash

set -eux

model=model_transformer__1__100__neighbor1
seed1=$1
seed2=$2
src=$3

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

python main.py \
--model ${model}__seed${seed} \
--source ${src} \
2>&1 | tee log/weight/batch_to_all/${model}__seed${seed}__${src}.txt

done

