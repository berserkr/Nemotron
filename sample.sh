#!/bin/bash

python scripts/presample_blend.py \
      --input /mnt/vast/proj/checkpoints/bathen/datasets/sft/granite-4.2-sft-datasets-r260414a/blends/blend_sampling_15m_balanced.json \
      --output /mnt/vast/proj/checkpoints/bathen/datasets/sft/granite-4.2-sft-datasets-r260414a/v1_sampled_15m \
      --seed 42 \
      --workers 40

