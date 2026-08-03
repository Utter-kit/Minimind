# MiniMind 实习项目版

这是我基于开源项目 MiniMind 整理和跑通的一个小型大语言模型训练项目。
我把项目改成了更适合学习和实习展示的版本：不用一开始就下载很大的数据集，也可以先用小样例数据把训练、微调、偏好对齐和推理流程跑通。

原项目地址：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)

## 项目目标

这个项目主要是为了理解一个语言模型从 0 到能对话的大致流程：

1. Pretrain：让模型学习基础语言规律
2. SFT：让模型学习按照指令回答问题
3. DPO：让模型更偏向好的回答
4. Eval：加载训练后的权重并进行简单推理

我目前已经用 CPU 小模型跑通了完整 smoke test，证明代码、环境、数据格式和训练脚本可以正常运行。

## 我的工作

- 搭建并检查了 Python 3.10 的 MiniMind 环境
- 修复了数据加载相关问题
- 新增了 `dataset/smoke/` 小样例数据
- 跑通了 Pretrain、SFT、DPO 三个训练阶段
- 跑通了本地推理脚本 `eval_llm.py`
- 整理了快速验证说明：[SMOKE_TEST.md](./SMOKE_TEST.md)

## 环境准备

推荐使用 conda：

```bash
conda create -n minimind python=3.10 -y
conda activate minimind
pip install -r requirements.txt
```

如果有 NVIDIA GPU，训练会快很多。没有 GPU 也可以先用 smoke test 验证流程。

## 快速运行

训练脚本建议在 `trainer/` 目录下运行。

### 1. 预训练

```bash
cd trainer
python train_pretrain.py \
  --device cpu \
  --dtype float16 \
  --epochs 1 \
  --batch_size 2 \
  --accumulation_steps 1 \
  --num_workers 0 \
  --hidden_size 64 \
  --num_hidden_layers 1 \
  --max_seq_len 32 \
  --log_interval 1 \
  --save_interval 999 \
  --save_weight smoke_pretrain \
  --data_path ../dataset/smoke/pretrain_smoke.jsonl \
  --from_weight none
```

### 2. SFT 微调

```bash
python train_full_sft.py \
  --device cpu \
  --dtype float16 \
  --epochs 1 \
  --batch_size 1 \
  --accumulation_steps 1 \
  --num_workers 0 \
  --hidden_size 64 \
  --num_hidden_layers 1 \
  --max_seq_len 64 \
  --log_interval 1 \
  --save_interval 999 \
  --save_weight smoke_sft \
  --data_path ../dataset/smoke/sft_smoke.jsonl \
  --from_weight smoke_pretrain
```

### 3. DPO 偏好对齐

```bash
python train_dpo.py \
  --device cpu \
  --dtype float16 \
  --epochs 1 \
  --batch_size 1 \
  --accumulation_steps 1 \
  --num_workers 0 \
  --hidden_size 64 \
  --num_hidden_layers 1 \
  --max_seq_len 64 \
  --log_interval 1 \
  --save_interval 999 \
  --save_weight smoke_dpo \
  --data_path ../dataset/smoke/dpo_smoke.jsonl \
  --from_weight smoke_sft
```

### 4. 推理测试

回到项目根目录：

```bash
cd ..
printf "0\n" | python eval_llm.py \
  --device cpu \
  --load_from model \
  --save_dir out \
  --weight smoke_pretrain \
  --hidden_size 64 \
  --num_hidden_layers 1 \
  --max_new_tokens 8 \
  --show_speed 0
```

注意：smoke test 数据非常小，模型输出不代表真实效果，只用于验证流程能跑通。

## 数据说明

项目里准备了 3 份小数据：

- `dataset/smoke/pretrain_smoke.jsonl`：预训练文本数据
- `dataset/smoke/sft_smoke.jsonl`：问答微调数据
- `dataset/smoke/dpo_smoke.jsonl`：好回答/坏回答偏好数据

正式训练时，可以把这些小数据替换成 MiniMind 官方数据集。

## 项目结构

```text
.
├── dataset/
│   ├── lm_dataset.py
│   └── smoke/
├── model/
│   ├── model_minimind.py
│   └── tokenizer.json
├── trainer/
│   ├── train_pretrain.py
│   ├── train_full_sft.py
│   └── train_dpo.py
├── eval_llm.py
├── requirements.txt
└── SMOKE_TEST.md
```

## 我学到的内容

- Transformer 模型的基本结构
- Tokenizer 如何把文本转成 token
- 预训练、SFT、DPO 的区别
- PyTorch 训练循环的基本流程
- 如何保存和加载模型权重
- 如何用小数据做工程 smoke test

## 适合实习面试怎么讲

这个项目可以这样介绍：

> 我基于 MiniMind 做了一个小型 LLM 训练流程复现项目。重点不是训练一个很强的模型，而是把 Pretrain、SFT、DPO 和推理这条链路实际跑通。我自己整理了最小样例数据，修复了数据加载问题，并写了 smoke test 文档，保证别人 clone 后可以先用 CPU 快速验证流程。

可以重点讲：

- 为什么要先做 smoke test
- SFT 数据和 DPO 数据格式有什么区别
- Loss 在训练里代表什么
- 为什么小数据训练出来的模型效果不好，但仍然有工程验证价值
- 如果有 GPU 和完整数据，下一步如何扩大训练

## 后续计划

- 下载完整 MiniMind 数据集进行正式训练
- 用 GPU 训练更大的 hidden size 和更多层数
- 记录 loss 曲线
- 对比 SFT 前后的回答效果
- 尝试 LoRA 微调一个特定领域的小模型

## License

本项目基于原 MiniMind 项目整理，遵循 Apache 2.0 License。
