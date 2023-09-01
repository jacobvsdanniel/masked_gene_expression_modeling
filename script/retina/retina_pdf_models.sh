#!/bin/bash

set -eux

gpu=2080
model_prefix=model_transformer
attn=last
seed=42

for model_suffix in 1__100__1__100 1__100__1__30 1__100__1__40 1__100__5__100 2__80__1__80 4__20__1__20
do
for src in lukowski3101 menon3101
do

model=${model_prefix}__${model_suffix}
mas=${model}__attn${attn}__seed${seed}

python retina.py \
--model ${gpu}/${mas} \
--source ${src} \
2>&1 | tee log/retina/weight/pdf/gpu${gpu}__${mas}__${src}.txt

done
done

