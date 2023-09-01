#!/bin/bash

set -eux

s1=$1
s2=$2

python main.py \
--s1 ${s1} \
--s2 ${s2} \
2>&1 | tee log/weight/${s1}_${s2}.txt

