#!/bin/bash

set -eux

src=$1
model_list_file=loss/model_list.txt
model1=$2
model2=$3

python retina.py \
--source ${src} \
--model_list_file ${model_list_file} \
--model1 ${model1} \
--model2 ${model2} \
--processes 9 \
2>&1 | tee log/retina/weight/cell_to_all/models__${src}__${model1}__${model2}.txt

