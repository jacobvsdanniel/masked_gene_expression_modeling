#!/bin/bash

set -eux

# for src in hESC hHep mDC mESC mHSC-E mHSC-GM mHSC-L
for src in mESC
do
for tf in HV
do
for nontfs in 500 1000
do
for config in 2__80__1__80
do

# train/hESC__HVTF__1000nonTFs/weight/all/2080/model_transformer__2__80__1__80__attnlast__seed42
dataset=${src}__${tf}TF__${nontfs}nonTFs
src_prefix=/volume/penghsuanli-genome2-nas2/beeline/train/${dataset}/weight/all/2080
tgt_prefix=/volume/macrophagenn/beeline_benchmark/weight/${dataset}
mkdir -p ${tgt_prefix}

mas=model_transformer__${config}__attnlast__seed42
cp_src=${src_prefix}/${mas}
cp_tgt=${tgt_prefix}/${mas}

cp -r ${cp_src} ${cp_tgt}

done
done
done
done

