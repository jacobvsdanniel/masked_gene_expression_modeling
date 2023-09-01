#!/bin/bash

set -eux

src_prefix=/volume/macrophagenn/beeline_benchmark
tgt_prefix=/volume/penghsuanli-genome2-nas2/beeline
log_dir=/volume/penghsuanli-genome2-nas2/sc/log/beeline/extract_seq
mkdir -p ${log_dir}

for src in hESC hHep mDC mESC mHSC-E mHSC-GM mHSC-L
do
for tf in HV
do
for nontfs in 500 1000
do

dataset=${src}__${tf}TF__${nontfs}nonTFs

raw_dir=${tgt_prefix}/raw/${dataset}
seq_dir=${tgt_prefix}/seq/${dataset}
train_dir=${tgt_prefix}/train/${dataset}
mkdir -p ${raw_dir}
mkdir -p ${seq_dir}
mkdir -p ${train_dir}

cp_src=${src_prefix}/${tf}TF+HVG/${src}/${src}_${nontfs}-GEP.csv
cp_tgt=${raw_dir}/GEP.csv
cp ${cp_src} ${cp_tgt}

python beeline.py \
--task extract_seq \
--source ${src} \
--tf ${tf} \
--nontfs ${nontfs} \
2>&1 | tee ${log_dir}/${dataset}.txt

done
done
done

