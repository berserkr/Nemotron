## Validated Training Configurations

| Model | TP | PP | EP | CP | Seq Len | Nodes | GPUs | rotary_base | LR | Status |
|-------|----|----|----|----|---------|-------|------|-------------|-----|--------|
| MoE 3B instruct | 1 | 1 | 4 | 1 | 8k | 1 | 4 | default (10M) | 5e-6 | loss 1.37 -> 0.65 |
| MoE 3B base | 1 | 1 | 8 | 4 | 128k | 8 | 32 | 500000 | 1e-6 | stable past warmup |
| Granite 8B dense | 4 | 8 | - | 16 | 256k | 128 | 512 | default | 5e-6 | stable |
| Granite 8B dense | 4 | 4 | - | 32 | 512k | 128 | 512 | default | 5e-6 | stable (low LR post-warmup) |
| Granite 30B dense | 4 | 4 | - | 16 | 256k | 128 | 512 | default | 5e-6 | stable |
| Nemotron Super 120B | 4 | 4 | 4 | - | - | 16 | 64 | default | - | export only (--not-strict) |

## Known Failures

| Model | TP | PP | EP | CP | Seq Len | Issue |
|-------|----|----|----|----|---------|-------|
| MoE 3B any | 2 | 1 | 8 | 4 | 128k | loss=26 with TP=2 + vocab resize (bridge bug) |
| MoE 3B any | 1 | 1 | 2 | 2 | 8k | NaN step 1 (MoE + small EP + CP interaction) |

## Key Lessons

- **MoE + CP**: only works with larger EP (8+), fails with EP=2
- **MoE + TP=2**: broken with non-original vocab size (49152) — use TP=1 or keep original vocab
- **MoE base model at long context**: needs `rotary_base: 500000` override (original 10k too low for >40k positions)
- **Granite multipliers**: embedding x12, residual x0.22, logits /6, attention x0.015625 — bridge bakes these into weights
- **Nemotron Super export**: MTP weights missing from RL checkpoint, use `--not-strict` to export backbone only
- **NVLink errors**: hardware issue on GB200 clusters, use pre-flight checks and `--exclude` bad nodes
