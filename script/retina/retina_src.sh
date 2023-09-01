#!/bin/bash

set -eux

src=$1

python retina.py \
--source ${src} \
2>&1 | tee log/retina/tmp_${src}.txt

