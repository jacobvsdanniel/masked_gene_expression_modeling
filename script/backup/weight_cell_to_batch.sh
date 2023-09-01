#!/bin/bash

set -eux

src=$1
i=$2
j=$3

python main.py \
--source ${src} \
--start ${i} \
--end ${j} \
2>&1 | tee log/weight/${src}_${i}_${j}.txt

