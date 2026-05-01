#!/bin/bash

CFG=test_128k_qwen.yaml
CFG=test_128k_nano_v2.yaml
#CFG=test_256k_qwen.yaml
#CFG=test_256k.yaml
CFG=test_128k_qwen.yaml
#CFG=test_128k_qwen_cp.yaml
CFG=test_128k_granite30b_cp4.yaml
CFG=test_128k_granite30b_cp8.yaml
CFG=test_128k_granite30b_cp16.yaml
CFG=test_256k_granite30b_cp8.yaml
CFG=test_256k_granite30b_cp16.yaml
CFG=test_256k_granite30b_cp32.yaml
CFG=test_256k_granite30b_cp16_256mult.yaml
CFG=test_512k_granite30b_cp32.yaml
CFG=test_64k_granite_moe_3b_pad64.yaml
CFG=test_40k_granite_moe_3b_pad256.yaml
CFG=test_128k_granite_8b_pad256.yaml
CFG=test_128k_granite_moe_3b_pad256.yaml
CFG=test_128k_granite30b_cp4.yaml
CFG=test_math_8k.yaml
CFG=test_math_8k_cp2.yaml
CFG=test_math_8k_g338binst.yaml
CFG=test_math_128k.yaml
CFG=test_math_64k.yaml
CFG=test_granite_3b_math_128k.yaml
CFG=test_granite_3b_math_128k_cp4.yaml
BASE_PATH=src/nemotron/recipes/super3/stage1_sft/config/data_prep
CFG_PATH=${BASE_PATH}/${CFG}
CFG_PATH=/mnt/home/bathen/src/github.com/Nemotron/src/nemotron/recipes/granite30/stage1_sft/config/data_prep/tokenization_32k_granite_8b.yaml
python src/nemotron/recipes/super3/stage1_sft/data_prep.py --config $CFG_PATH

