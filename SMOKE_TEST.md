# MiniMind smoke test

This repository has been verified with a tiny CPU-only smoke workflow. The smoke data is intentionally small and only checks that training, saving, loading, and generation run end to end. It is not meant to produce a useful model.

## Environment

```bash
conda create -n minimind python=3.10 -y
conda activate minimind
pip install -r requirements.txt
```

The local verification used the existing `minimind` conda environment with Python 3.10.

## Verified commands

Run from the `trainer/` directory for training scripts:

```bash
HF_HOME=../.hf_cache HF_DATASETS_CACHE=../.hf_cache/datasets TRANSFORMERS_CACHE=../.hf_cache/transformers \
python train_pretrain.py --device cpu --dtype float16 --epochs 1 --batch_size 2 --accumulation_steps 1 --num_workers 0 --hidden_size 64 --num_hidden_layers 1 --max_seq_len 32 --log_interval 1 --save_interval 999 --save_weight smoke_pretrain --data_path ../dataset/smoke/pretrain_smoke.jsonl --from_weight none

HF_HOME=../.hf_cache HF_DATASETS_CACHE=../.hf_cache/datasets TRANSFORMERS_CACHE=../.hf_cache/transformers \
python train_full_sft.py --device cpu --dtype float16 --epochs 1 --batch_size 1 --accumulation_steps 1 --num_workers 0 --hidden_size 64 --num_hidden_layers 1 --max_seq_len 64 --log_interval 1 --save_interval 999 --save_weight smoke_sft --data_path ../dataset/smoke/sft_smoke.jsonl --from_weight smoke_pretrain

HF_HOME=../.hf_cache HF_DATASETS_CACHE=../.hf_cache/datasets TRANSFORMERS_CACHE=../.hf_cache/transformers \
python train_dpo.py --device cpu --dtype float16 --epochs 1 --batch_size 1 --accumulation_steps 1 --num_workers 0 --hidden_size 64 --num_hidden_layers 1 --max_seq_len 64 --log_interval 1 --save_interval 999 --save_weight smoke_dpo --data_path ../dataset/smoke/dpo_smoke.jsonl --from_weight smoke_sft
```

Run inference from the repository root:

```bash
printf "0\n" | HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache/datasets TRANSFORMERS_CACHE=.hf_cache/transformers \
python eval_llm.py --device cpu --load_from model --save_dir out --weight smoke_pretrain --hidden_size 64 --num_hidden_layers 1 --max_new_tokens 8 --show_speed 0
```

For real training, replace files under `dataset/smoke/` with the full MiniMind datasets and use the default model sizes on a CUDA GPU.
