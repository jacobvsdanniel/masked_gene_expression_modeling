#!/bin/bash

set -eux

model=model_transformer__1__100__neighbor1
seed1=$1
seed2=$2

src=$3
i=$4
j=$5

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do

python main.py \
--model ${model}__seed${seed} \
--source ${src} \
--start ${i} \
--end ${j} \
2>&1 | tee log/weight/cell_to_batch/${model}__seed${seed}__${src}__cell${i}__cell${j}.txt

done

