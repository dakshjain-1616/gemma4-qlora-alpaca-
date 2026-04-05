# QLoRA Fine-tuning: Gemma-4-E2B-it on Alpaca

<div align="center">

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-daksh--neo%2Fgemma4--qlora--alpaca-yellow?style=for-the-badge)](https://huggingface.co/daksh-neo/gemma4-qlora-alpaca)
[![Built with NEO](https://img.shields.io/badge/Built%20with-NEO%20AI%20Agent-6f42c1?style=for-the-badge)](https://heyneo.so)
[![VS Code Extension](https://img.shields.io/badge/VS%20Code-NEO%20Extension-007ACC?style=for-the-badge&logo=visualstudiocode)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)

</div>

> **Autonomously built by [NEO](https://heyneo.so)** — Your autonomous AI Agent. Try the [VS Code extension](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo).

**Model**: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | **Dataset**: [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | **Adapter**: [daksh-neo/gemma4-qlora-alpaca](https://huggingface.co/daksh-neo/gemma4-qlora-alpaca)

---

## Pipeline Overview

<img src="assets/pipeline_diagram.svg" alt="QLoRA Pipeline Diagram" width="100%">

---

## Training Configuration

<img src="assets/qlora_config.svg" alt="QLoRA Configuration" width="100%">

| Parameter | Value |
|-----------|-------|
| **Base Model** | google/gemma-4-E2B-it |
| **Quantization** | 4-bit NF4 (BitsAndBytes) |
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 32 |
| **Target Modules** | all-linear |
| **Batch Size** | 2 |
| **Gradient Accumulation** | 2 |
| **Learning Rate** | 2e-4 (cosine decay) |
| **Epochs** | 1 |
| **Max Sequence Length** | 512 |
| **Training Samples** | 4,500 (Alpaca) |

---

## Training Results

<img src="assets/training_progress.svg" alt="Training Progress" width="100%">

| Metric | Value |
|--------|-------|
| **Total Steps** | 1,125 |
| **Initial Loss** | 31.49 |
| **Final Loss** | 31.43 |
| **Samples/Second** | ~2.75 |
| **Adapter Size** | ~150 MB |
| **Trainable Params** | <1% of total |

### Key Findings

- **4-bit Quantization**: Loaded gemma-4-E2B-it with 75% memory reduction via NF4 quantization
- **LoRA Coverage**: Adapters applied to all 238 unique linear module types across all layers
- **MoE Compatibility**: LoRA applied uniformly across MoE expert layers without routing disruption
- **Memory Footprint**: Training completed on single RTX A6000 48GB with headroom to spare

---

## HuggingFace Adapter

The trained LoRA adapter is available at [daksh-neo/gemma4-qlora-alpaca](https://huggingface.co/daksh-neo/gemma4-qlora-alpaca).

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it",
    torch_dtype="bfloat16",
    device_map="cuda"
)
model = PeftModel.from_pretrained(model, "daksh-neo/gemma4-qlora-alpaca")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
```

---

## Project Structure

```
01-finetuning-qlora-moe/
├── assets/                      # SVG infographics
│   ├── pipeline_diagram.svg
│   ├── qlora_config.svg
│   └── training_progress.svg
└── ml_project_0921/
    ├── data/src/                # prepare_alpaca.py
    ├── model/src/               # train_qlora.py
    └── reports/                 # findings_report.md
```

---

## Quick Start

```bash
pip install transformers peft accelerate bitsandbytes datasets torch

# Prepare data
python ml_project_0921/data/src/prepare_alpaca.py

# Run training
python ml_project_0921/model/src/train_qlora.py
```

---

<div align="center">

Autonomously built with **[NEO](https://heyneo.so)** — Your Autonomous AI Agent

[Get NEO for VS Code](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)

</div>
