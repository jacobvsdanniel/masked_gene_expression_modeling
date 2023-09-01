#!/bin/bash

set -eux

src=$1

python main.py \
--source ${src} \
2>&1 | tee log/weight/${src}_hvg.txt

