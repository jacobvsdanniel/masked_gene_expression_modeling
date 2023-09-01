#!/bin/bash

set -eux

src_prefix=/volume/penghsuanli-genome2-nas2/brain2770/train/weight/all/2080
tgt_prefix=/volume/macrophagenn/scRNAseq/brain/brain2770

for m in 1__100__1__100 1__100__1__30 1__100__1__40 1__100__5__100 2__80__1__80 4__20__1__20
do
for s in TC TJ WC WJ TC-WC TJ-WJ TC-TJ-WC-WJ
do

dir=model_transformer__${m}__${s}__attnlast__seed42__${s}
src=${src_prefix}/${dir}
tgt=${tgt_prefix}/${dir}

cp -r ${src} ${tgt}

done
done

