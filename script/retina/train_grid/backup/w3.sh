#!/bin/bash

set -eux

gpu=2080
seed=42
attn=last

for layer in 1 2 4
do
for dim in 40
do
for head in 1 5
do
for ff in 20 30 40 60 80 100
do
for src in lukowski3101 menon3101
do

model=model_transformer__${layer}__${dim}__${head}__${ff}
ms=${model}__seed${seed}

: <<'END'
python train.py \
--task train \
--cell_gene_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/retina/${src}/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/${gpu}/${ms}.pt \
--train_steps 100000 \
--log_steps 1000 \
--test_steps 1000 \
--batch_size 1 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/retina/train/gpu${gpu}__${ms}__${src}.txt
END

mas=${model}__attn${attn}__seed${seed}

python train.py \
--task weight \
--cell_gene_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_index.csv \
--cell_exp_file /volume/penghsuanli-genome2-nas2/retina/${src}/seq/gene_seq_logexp.npy \
--hvg_file /volume/penghsuanli-genome2-nas2/retina/${src}/hvg/hvg_index.npy \
--test_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/test.json \
--model_file /volume/penghsuanli-genome2-nas2/retina/${src}/train/${gpu}/${ms}.pt \
--weight_dir /volume/penghsuanli-genome2-nas2/retina/${src}/train/weight/cell/${gpu}/${mas} \
--attn_layer ${attn} \
--batch_size 4 \
--model ${model} \
--seed ${seed} \
2>&1 | tee log/retina/weight/gpu${gpu}__${mas}__${src}.txt

done
done
done
done
done

