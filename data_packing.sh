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
BASE_PATH=src/nemotron/recipes/super3/stage1_sft/config/data_prep
python src/nemotron/recipes/super3/stage1_sft/data_prep.py --config ${BASE_PATH}/${CFG}

