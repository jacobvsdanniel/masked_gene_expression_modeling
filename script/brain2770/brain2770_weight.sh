#!/bin/bash

set -eux

gpu=2080
model=$1
attn=last
seed1=42
seed2=42

for (( seed=${seed1}; seed<=${seed2}; seed++ ))
do
for src in TC-TJ-WC-WJ TC-WC TJ-WJ TC TJ WC WJ
do

msas=${model}__${src}__attn${attn}__seed${seed}

python brain2770.py \
--model ${gpu}/${msas} \
--source ${src} \
--processes 9 \
2>&1 | tee log/brain2770/weight/cell_to_all/gpu${gpu}__${msas}__${src}.txt

done
done

