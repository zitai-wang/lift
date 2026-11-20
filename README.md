# LIFT

LIFT studies reasoning-vector transfer and recovery from language models to vision-language models. The released code contains the method framework and the main experiment path only; datasets, checkpoints, generated chains of thought, vectors, logs, and result tables are intentionally excluded.

## What is included

- Self-CoT generation and reasoning-vector extraction.
- Static attention/FFN vector injection and layer-wise evaluation.
- Trainable recovery with random, unimodal-extracted, or multimodal-extracted initialization.
- CE-only, alignment-only, and joint recovery objectives.
- Qwen/Qwen-VL and InternLM/InternVL model wrappers.
- Dataset adapters for GSM8K, CommonsenseQA, StrategyQA, MathVista, MathVision, MMMU, and ScienceQA.

The central implementation is in `src/shift_encoder.py`, `src/shift_model.py`, and `src/recovery_ablation.py`.

## Setup

```bash
conda create -n lift python=3.10 -y
conda activate lift
pip install -r requirements.txt
```

Download model weights and datasets separately. Pass machine-specific locations through Hydra overrides (`model_path=...`, `data.source=...`) or set `LIFT_MODEL_PATH`. Outputs default to `./results`; set `LIFT_OUTPUT_DIR` to change this.

## Main pipeline

```bash
python scripts/run_static_pipeline.py \
  --model-name qwen2.5-7b-instruct \
  --model-path /path/to/model \
  --dataset gsm8k \
  --data-source /path/to/gsm8k \
  --method ffn \
  --devices 0
```

This runs Self-CoT generation, vector extraction, a baseline, and a layer-wise vector-injection evaluation. Add `--dry-run` to inspect commands without launching models.

Trainable recovery is configured in `src/config/train.yaml`:

```bash
python src/train.py --config-name train \
  model_name=qwen2.5-vl-7b-instruct \
  model_path=/path/to/model \
  data.name=mathvista \
  data.source=/path/to/mathvista \
  init_mode=multimodal_extracted \
  recovery_vector_init.multimodal_extracted.path=/path/to/vector.pt \
  loss_mode=ce_align \
  only_shift_at_layer=9
```

Use `init_mode=random|multimodal_extracted|unimodal_extracted` and `loss_mode=ce_only|align_only|ce_align` for the main recovery ablations.

## Repository policy

Large artifacts and local machine state are ignored by design. Do not commit datasets, model weights, extracted vectors, generated outputs, logs, or benchmark results.

## Acknowledgement

This implementation builds on the [MimIC](https://arxiv.org/abs/2504.08851) codebase; the released structure keeps the relevant model-hook and testbed components needed by LIFT.
