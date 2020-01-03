#!/bin/bash
source /home/vz/miniconda3/bin/activate base
source activate cn27
export LC_CTYPE=en_US.UTF-8
python three_step_decoding.py --test-file $1 --output-file $2
source /home/vz/miniconda3/bin/activate ak
