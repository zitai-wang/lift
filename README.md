# LIFT

Official implementation of **LIFT**, a framework for studying how reasoning representations transfer from language models (LLMs) to vision-language models (VLMs), how these representations degrade after multimodal extension, and whether the degraded multimodal signal can be recovered through training.

This public repository contains the method framework and the key experiment path. Datasets, model weights, generated chains of thought, extracted vectors, checkpoints, logs, and result tables are intentionally excluded.

## Overview

LIFT is organized around three connected questions:

1. **Cross-model and cross-modal transfer.** Extract a reasoning vector from an LLM or VLM and inject it into a target VLM. This tests whether a reasoning representation learned in the language model remains useful after multimodal extension.
2. **Multimodal self-extraction.** Extract a vector from a VLM and inject it back into the same VLM. This tests whether the multimodal model still contains a directly usable internal reasoning signal.
3. **Trainable recovery.** Initialize a trainable vector from a multimodal extracted vector and optimize it with task and/or representation supervision. This tests whether a weak multimodal signal can be reorganized and recovered.

The intended interpretation is not that a trainable vector is always stronger. LIFT treats training as a **signal-recovery mechanism** whose effectiveness can vary across tasks.

## Method at a glance

The complete experiment path is:

```text
questions
   │
   ▼
generate model-specific Self-CoT
   │
   ▼
record attention/FFN activations
   │
   ▼
aggregate a layer-wise reasoning vector
   │
   ├── static injection ──► layer-wise evaluation
   │
   └── trainable initialization
          │
          ├── CE supervision
          ├── hidden-state alignment
          └── CE + alignment
                  │
                  ▼
             recovered vector
```

Vector injection is implemented with forward hooks, so the backbone architecture does not need to be rewritten. The framework supports attention-side and FFN-side interventions, static scales, learnable scales, single-layer injection, layer combinations, and dense layer injection.

## Main components

| File | Purpose |
| --- | --- |
| `src/shift_encoder.py` | Hook registration, activation recording, attention/FFN vector injection, static and learnable shift modules. |
| `src/shift_model.py` | Teacher/student forwards, CE/KL/alignment objectives, optimization, validation, checkpointing, and recovery summaries. |
| `src/recovery_ablation.py` | Recovery initialization, vector-shape/layer matching, loss-mode resolution, and vector statistics. |
| `src/generate_self_cot.py` | Generate model-specific reasoning traces and retain correct traces for extraction/training. |
| `src/extract_cot_vector.py` | Extract and aggregate reasoning vectors from Qwen-family models. |
| `src/extract_cot_vector_internlm.py` | InternLM extraction path. |
| `src/extract_cot_vector_internvl.py` | InternVL extraction path. |
| `src/eval_baseline.py` | Evaluation without vector intervention. |
| `src/eval_base.py` | Base hidden-state vector evaluation. |
| `src/eval_licv.py` | FFN/LICV-style vector evaluation. |
| `src/eval_mimic.py` | Attention/MimIC-style vector evaluation. |
| `src/train.py` | Entry point for trainable recovery and LoRA baselines. |
| `scripts/run_static_pipeline.py` | End-to-end Self-CoT → extraction → baseline → layer sweep pipeline. |

## Repository layout

```text
lift/
├── scripts/
│   ├── run_static_pipeline.py
│   ├── run_eval_layers_base.py
│   ├── run_eval_layers_licv.py
│   └── run_eval_layers_mimic.py
├── src/
│   ├── config/
│   │   ├── encoder/          # intervention modules/objectives
│   │   ├── peft/             # optimization and LoRA settings
│   │   ├── generate_self_cot.yaml
│   │   ├── extract_cot_vector.yaml
│   │   ├── eval_*.yaml
│   │   └── train.yaml
│   ├── dataset_utils/        # benchmark adapters
│   ├── *_model_wrapper.py    # model-specific prompting/generation
│   ├── shift_encoder.py
│   ├── shift_model.py
│   └── recovery_ablation.py
└── testbed/                  # minimal hook/model utilities
```

## Supported models

The public model registry currently includes:

- `qwen2.5-7b`
- `qwen2.5-7b-instruct`
- `qwen2.5-math-7b`
- `qwen2.5-math-7b-instruct`
- `qwen2.5-vl-7b-instruct`
- `qwen3-8b`
- `llama-3.1-8b-instruct`
- `llama-3.1-70b-instruct`
- `internlm2_5-7b-chat`
- `internvl2_5-8b`

The main maintained paths are Qwen/Qwen-VL and InternLM/InternVL. A local checkpoint directory can be supplied with `model_path=...`. If `model_path` is omitted, the corresponding Hugging Face model ID is used.

## Supported datasets

Adapters are included for:

- GSM8K
- CommonsenseQA
- StrategyQA
- MATH and MMLU-Pro
- MathVista
- MathVision
- MMMU
- ScienceQA

Dataset locations are never hard-coded. Supply them with `data.source=/path/to/dataset`. Some adapters can also load a Hugging Face dataset ID, such as `MathLLMs/MathVision` or `MMMU/MMMU`.

For local text benchmarks, the expected default layout is:

```text
data/
├── gsm8k/
│   ├── train-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
├── commonsenseqa/
│   ├── train-00000-of-00001.parquet
│   └── validation-00000-of-00001.parquet
└── strategyqa/
    ├── train-*.parquet
    └── test-*.parquet
```

For multimodal benchmarks, images must remain accessible from the records returned by the dataset loader. Dataset-specific split names can be selected with `data.query_split=...` and `data.support_split=...`.

## Installation

Python 3.10 is recommended.

```bash
git clone https://github.com/zitai-wang/lift.git
cd lift

conda create -n lift python=3.10 -y
conda activate lift
pip install -r requirements.txt
```

The default environment uses PyTorch, Transformers, PEFT, Hydra, PyTorch Lightning, and DeepSpeed. Flash Attention is optional; disable the relevant config option if it is unavailable on your hardware.

## Paths and device selection

Machine-specific values are supplied at runtime:

```bash
# Optional default model source.
export LIFT_MODEL_PATH=/path/to/model

# Optional output root. Defaults to ./results.
export LIFT_OUTPUT_DIR=/path/to/outputs
```

Alternatively, pass `model_path=...` and `data.source=...` as Hydra overrides. GPU selection is explicit:

```bash
CUDA_VISIBLE_DEVICES=0 python ... devices=0
```

After `CUDA_VISIBLE_DEVICES` is set, the code uses logical `cuda:0`; no physical GPU index is embedded in the source code.

## Quick start: complete static pipeline

The convenience pipeline currently supports Qwen/Qwen-VL with GSM8K, CommonsenseQA, and StrategyQA.

```bash
python scripts/run_static_pipeline.py \
  --model-name qwen2.5-7b-instruct \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --data-source /path/to/gsm8k \
  --method ffn \
  --devices 0 \
  --generate-samples 500 \
  --eval-samples 1319 \
  --start-layer 0 \
  --end-layer 28
```

This command performs four stages:

1. Evaluate the no-vector baseline.
2. Generate Self-CoT traces and a correct-only subset.
3. Extract a layer-wise vector from the correct traces.
4. Sweep vector injection across the requested layers.

Available methods are:

| `--method` | Intervention | Entrypoint |
| --- | --- | --- |
| `baseline` | No vector injection | `src/eval_baseline.py` |
| `base` | Base hidden-state vector | `src/eval_base.py` |
| `ffn` | FFN/LICV-style vector | `src/eval_licv.py` |
| `attn` | Attention/MimIC-style vector | `src/eval_mimic.py` |

Useful pipeline options:

```text
--dry-run          print commands without loading models
--skip-generate    skip the generation stage
--skip-extract     skip the extraction stage
--skip-eval        stop before evaluation
--skip-baseline    omit the no-vector baseline stage
--no-layerwise     evaluate only one layer
--single-layer N   layer used with --no-layerwise
--run-tag NAME     set a deterministic output directory name
```

The `--skip-*` switches are intended for custom orchestration. When a downstream stage needs a skipped artifact, place that artifact at the path printed by `--dry-run` or invoke the stage manually with an explicit path.

## Running each stage manually

### 1. Generate Self-CoT

```bash
python src/generate_self_cot.py --config-name generate_self_cot \
  model_name=qwen2.5-7b-instruct \
  model_path=/path/to/model \
  data.name=gsm8k \
  data.source=/path/to/gsm8k \
  data.num_query_samples=500 \
  output_path=results/self_cot/gsm8k.json \
  batch_size=8
```

Two files are produced:

- `gsm8k.json`: all generated traces and correctness metadata.
- `gsm8k_correct_only.json`: traces retained for vector extraction or recovery training.

### 2. Extract a reasoning vector

```bash
python src/extract_cot_vector.py --config-name extract_cot_vector \
  model_name=qwen2.5-7b-instruct \
  model_path=/path/to/model \
  data.name=gsm8k \
  data.source=/path/to/gsm8k \
  data.use_self_cot=true \
  data.self_cot_path=results/self_cot/gsm8k_correct_only.json \
  output_path=results/vectors/qwen_gsm8k.pt
```

Use `src/extract_cot_vector_internlm.py` or `src/extract_cot_vector_internvl.py` for the corresponding model family.

The saved payload contains the encoder metadata and available layer-wise vectors, including attention, FFN, or hidden-state vectors depending on the extraction strategy.

### 3. Evaluate a vector

Single-layer FFN injection:

```bash
python src/eval_licv.py --config-name eval_ffn \
  +encoder=licv +peft=licv \
  model_name=qwen2.5-7b-instruct \
  model_path=/path/to/model \
  data.name=gsm8k \
  data.source=/path/to/gsm8k \
  extracted_cot_vector_path=results/vectors/qwen_gsm8k.pt \
  only_shift_at_layer=12 \
  static_shift_scale=1.0 \
  resume=false
```

For a layer sweep, use the corresponding script in `scripts/` and set `start_layer` and `end_layer`.

## Trainable recovery

Trainable recovery optimizes an injected vector while keeping the intended recovery semantics explicit. The main configuration is `src/config/train.yaml`.

```bash
python src/train.py --config-name train \
  model_name=qwen2.5-vl-7b-instruct \
  model_path=/path/to/Qwen2.5-VL-7B-Instruct \
  data.name=mathvista \
  data.source=/path/to/mathvista \
  data.num_query_samples=500 \
  data.use_self_cot=true \
  data.self_cot_path=/path/to/mathvista_correct_only.json \
  init_mode=multimodal_extracted \
  recovery_vector_init.multimodal_extracted.path=/path/to/multimodal_vector.pt \
  recovery_vector_init.multimodal_extracted.source_model_name=qwen2.5-vl-7b-instruct \
  recovery_vector_init.multimodal_extracted.source_dataset_name=mathvista \
  loss_mode=ce_align \
  only_shift_at_layer=9 \
  runname=mathvista-recovery-l9
```

### Initialization ablation

| `init_mode` | Meaning | Required configuration |
| --- | --- | --- |
| `random` | Small Gaussian initialization controlled by `random_init_std`. | None |
| `multimodal_extracted` | Initialize from a vector extracted by a VLM. | `recovery_vector_init.multimodal_extracted.path` |
| `unimodal_extracted` | Initialize from a vector extracted by an LLM. | `recovery_vector_init.unimodal_extracted.path` |

Example random initialization:

```bash
python src/train.py --config-name train \
  init_mode=random random_init_std=1e-3 \
  loss_mode=ce_align \
  model_name=qwen2.5-vl-7b-instruct \
  model_path=/path/to/model \
  data.name=mathvista data.source=/path/to/mathvista
```

### Loss ablation

| `loss_mode` | Effective objective |
| --- | --- |
| `ce_only` | Answer/task cross-entropy only. |
| `align_only` | Hidden-state alignment only. |
| `ce_align` | Joint cross-entropy and alignment. |

The base weights are controlled by `ce_loss_weight` and `align_loss_weight`. Selecting a loss mode automatically disables the unused term. `align_num_tokens` controls how many answer-side tokens participate in hidden-state alignment.

### Layer modes

```text
layer_train_mode=single_layer   use only_shift_at_layer=N
layer_train_mode=combinations   use shift_layers_combinations=[[3,7],[3,7,13]]
layer_train_mode=dense          intervene in all configured layers
```

## Output structure

The static pipeline writes:

```text
results/static_pipeline/<run-tag>/
├── logs/
│   ├── 00_baseline.log
│   ├── 01_generate.log
│   ├── 02_extract.log
│   └── 03_eval_layerwise.log
├── records/
├── self_cot_data.json
├── self_cot_data_correct_only.json
├── <run-tag>.pt
└── run_summary.json
```

Recovery training writes under:

```text
results/ckpt/<runname>-<model>-<dataset>-<sample-count>/
├── epoch-*/
├── best-epoch-*/
├── final/
├── recovery_metrics/
├── config.json
└── recovery_ablation_summary.json
```

`recovery_ablation_summary.json` records initialization provenance, effective loss weights, vector statistics, validation metrics, and the final checkpoint location.

## Adding a dataset

1. Add a class under `src/dataset_utils/` that inherits `DatasetBase`.
2. Define `support_datasets`, `_support_set`, and `_query_set`.
3. Implement the prompt/collation, answer extraction, and evaluation methods.
4. Register the class in `src/dataset_utils/__init__.py`.
5. Pass the dataset root through `data.source`; do not add machine-specific paths to source files.

The existing GSM8K adapter is the simplest text-only reference, while MathVista, MathVision, MMMU, and ScienceQA show multimodal loading and collation.

## Adding a model

1. Add its public model ID to `MODEL_REGISTRY` in `src/utils.py`.
2. Implement a wrapper derived from `testbed.models.model_base.ModelBase` if its prompt or generation API differs.
3. Add the wrapper construction to the relevant generation, extraction, training, and evaluation entrypoints.
4. Add the layer/module patterns required by `src/shift_encoder.py`.
5. Verify extraction and injection with a small sample and a single layer before launching a sweep.

## Reproducibility notes

- Set `data.seed` for dataset sampling and keep the Self-CoT correct-only file fixed across comparisons.
- Compare vectors with the same extraction sample set and aggregation strategy.
- Report the source model, source dataset, target model, injection type, layer, and scale together.
- For recovery experiments, record both `init_mode` and `loss_mode`; neither should be inferred from the checkpoint filename alone.
- Use `--dry-run` to archive the exact commands before expensive runs.
- Baseline and vector evaluations should use identical generation settings.

## Troubleshooting

**Hydra reports an unknown key.** Use `+key=value` to append a new key and `++key=value` when a key may or may not already exist.

**A local checkpoint is not found.** Pass `model_path=/absolute/path/to/checkpoint` or set `LIFT_MODEL_PATH`. If neither is set, the code attempts to use the Hugging Face model ID.

**CUDA out of memory.** Reduce `batch_size`, enable gradient checkpointing, shorten `data.max_seq_len`, reduce image resolution, or evaluate fewer layers at a time.

**The extracted vector has the wrong shape.** Ensure that source and target layer dimensions are compatible and that the requested `only_shift_at_layer` matches the saved layer axis. Recovery initialization validates and reports both shapes.

**A recovery run has no alignment signal.** Confirm that the selected encoder records hidden states and that `loss_mode` includes alignment.

## Repository policy

The following artifacts are excluded by `.gitignore` and must not be committed:

- datasets and cached dataset shards;
- model weights and PEFT checkpoints;
- extracted `.pt` vectors;
- generated Self-CoT files;
- experiment logs and evaluation records;
- spreadsheets, slides, and paper PDFs;
- local IDE, Hydra, Lightning, and Hugging Face caches.

## Acknowledgement

This implementation builds on the [MimIC](https://arxiv.org/abs/2504.08851) codebase. The public repository retains the hook and testbed components required by LIFT while adding reasoning-vector transfer, multimodal self-extraction, and trainable recovery workflows.
