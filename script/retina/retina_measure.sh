#!/bin/bash

set -eux

gpu=$1
gold=$2
auto=$3
measure_csv_file=$4

prefix=/volume/penghsuanli-genome2-nas2/retina
suffix=train/weight/all/${gpu}

measure_file_prefix=measure

python retina.py \
--gold ${prefix}/${gold}/${suffix} \
--auto ${prefix}/${auto}/${suffix} \
--measure_csv_file ${measure_file_prefix}/${measure_csv_file} \
2>&1 | tee log/retina/measure/${gold}_${auto}_${measure_csv_file}.txt

