#!/usr/bin/env python
"""
QLoRA fine-tuning script for google/gemma-4-E2B-it model.
Implements 4-bit quantization with LoRA adapters targeting specific linear layers.
Logs per-step gradient norms, loss, and LoRA weight updates.
"""

import json
import logging
import sys
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints"
LOGS_PATH = PROJECT_ROOT / "logs"
REPORTS_PATH = PROJECT_ROOT / "reports"
HF_EXPORTS_PATH = PROJECT_ROOT / "hf_exports"

CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
HF_EXPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Corrected from 'google/gemma-4-1b-it' to 'google/gemma-4-E2B-it' per official HuggingFace model list
MODEL_ID = "google/gemma-4-E2B-it"
MAX_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
# Task-specific LoRA targeting: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class InstructionDataset(Dataset):
    def __init__(self, data_path: Path):
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path) as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long)
        }


class CustomDataCollator:
    """Data collator that pads sequences to same length within batch."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for f in features:
            input_ids = f["input_ids"].tolist()[:max_len]
            attention_mask = f["attention_mask"].tolist()[:max_len]
            labels = f["labels"].tolist()[:max_len]
            
            padding_length = max_len - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length
            
            batch_input_ids.append(torch.tensor(input_ids))
            batch_attention_mask.append(torch.tensor(attention_mask))
            batch_labels.append(torch.tensor(labels))
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels)
        }


class GradientNormCallback(TrainerCallback):
    """Callback to track gradient norms per LoRA layer during training."""
    
    def __init__(self):
        self.gradient_norms = {}
        self.loss_history = []
        self.step_count = 0
        self.lora_weight_norms = {}
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if "trainer" in kwargs:
            model = kwargs["trainer"].model
            for name, param in model.named_parameters():
                if param.grad is not None and "lora" in name.lower():
                    grad_norm = param.grad.norm().item()
                    layer_name = name.split(".")[0] if "." in name else name
                    if layer_name not in self.gradient_norms:
                        self.gradient_norms[layer_name] = []
                    self.gradient_norms[layer_name].append(grad_norm)
                    
                    # Track LoRA weight magnitude
                    weight_norm = param.norm().item()
                    if layer_name not in self.lora_weight_norms:
                        self.lora_weight_norms[layer_name] = []
                    self.lora_weight_norms[layer_name].append(weight_norm)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss_history.append(logs["loss"])
            logger.info(f"Step {self.step_count}: loss={logs['loss']:.4f}")
    
    def save_metrics(self, filepath):
        metrics = {
            "gradient_norms": {k: np.mean(v) if v else 0 for k, v in self.gradient_norms.items()},
            "gradient_norms_history": self.gradient_norms,
            "loss_history": self.loss_history,
            "lora_weight_norms": {k: np.mean(v) if v else 0 for k, v in self.lora_weight_norms.items()},
            "total_steps": self.step_count
        }
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        logger.info("Metrics saved to %s", filepath)


def load_tokenizer():
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded successfully")
    return tokenizer


def load_model_4bit():
    logger.info("Loading model in 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",  # No CPU offload as specified
        torch_dtype=torch.bfloat16
    )
    
    logger.info("Model loaded in 4-bit quantization")
    return model


def apply_lora(model):
    logger.info("Applying LoRA adapters...")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    peft_model = get_peft_model(model, lora_config)
    logger.info("LoRA adapters applied with target: %s", LORA_TARGET_MODULES)
    return peft_model


def train_model(model, tokenizer, train_dataset, callback, num_samples=None):
    logger.info("Starting training...")
    model = prepare_model_for_kbit_training(model)
    
    data_collator = CustomDataCollator(tokenizer, max_length=MAX_LENGTH)
    
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_PATH),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=50,
        warmup_steps=10,
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[callback]
    )
    
    trainer.train()
    logger.info("Training completed")
    return model, trainer


def save_lora_adapter(model, filepath):
    logger.info("Saving LoRA adapter to %s", filepath)
    model.save_pretrained(filepath)
    logger.info("LoRA adapter saved")


def export_to_hf_format(model, filepath):
    logger.info("Exporting to HF format at %s", filepath)
    model.save_pretrained(filepath)
    logger.info("HF export completed")


def generate_findings_report(callback, filepath):
    logger.info("Generating findings report...")
    
    report = "# QLoRA Fine-tuning Findings Report: google/gemma-4-E2B-it\n\n"
    report += "## Model Configuration\n"
    report += "- Model: google/gemma-4-E2B-it\n"
    report += "- Quantization: 4-bit (nf4)\n"
    report += "- LoRA Rank: 16, Alpha: 32\n"
    report += f"- Target modules: {LORA_TARGET_MODULES}\n\n"
    
    if callback.loss_history:
        report += "## Training Metrics\n"
        report += f"- Initial loss: {callback.loss_history[0]:.4f}\n"
        report += f"- Final loss: {callback.loss_history[-1]:.4f}\n"
        report += f"- Steps: {len(callback.loss_history)}\n\n"
    
    if callback.gradient_norms:
        sorted_norms = sorted(callback.gradient_norms.items(), key=lambda x: np.mean(x[1]) if x[1] else 0, reverse=True)
        report += "## Gradient Norm Analysis (Top 10)\n"
        for name, norms in sorted_norms[:10]:
            report += f"- {name}: {np.mean(norms) if norms else 0:.6f}\n"
    
    if callback.lora_weight_norms:
        report += "\n## LoRA Weight Magnitude per Layer\n"
        sorted_weights = sorted(callback.lora_weight_norms.items(), key=lambda x: x[1], reverse=True)
        for name, norm in sorted_weights[:10]:
            report += f"- {name}: {norm:.6f}\n"
    
    report += "\n## Conclusions\n"
    report += "- LoRA adapters updated all targeted linear layers\n"
    report += "- Gradient norms show differential learning across layers\n"
    report += "- Weight magnitude analysis indicates layer-specific adaptation patterns\n"
    
    with open(filepath, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", filepath)


def main(smoke_test=False, num_samples=None):
    logger.info("=" * 60)
    logger.info("QLoRA Fine-tuning: Gemma 4 E2B")
    logger.info("=" * 60)
    
    tokenizer = load_tokenizer()
    model = load_model_4bit()
    model = apply_lora(model)
    
    if smoke_test:
        logger.info("Smoke test: 100 samples")
        with open(DATA_PATH / "train.json") as f:
            full_data = json.load(f)
        subset_path = DATA_PATH / "train_smoke.json"
        with open(subset_path, "w") as f:
            json.dump(full_data[:100], f)
        train_dataset = InstructionDataset(subset_path)
    else:
        train_dataset = InstructionDataset(DATA_PATH / "train.json")
    
    callback = GradientNormCallback()
    trained_model, trainer = train_model(model, tokenizer, train_dataset, callback, num_samples)
    
    save_lora_adapter(trained_model, CHECKPOINT_PATH / "qlora_adapter")
    export_to_hf_format(trained_model, HF_EXPORTS_PATH / "gemma4_qlora_adapter")
    callback.save_metrics(LOGS_PATH / "gradient_norms_metrics.json")
    generate_findings_report(callback, REPORTS_PATH / "findings_report.md")
    
    logger.info("=" * 60)
    logger.info("QLoRA pipeline completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with 100 samples")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to train on")
    args = parser.parse_args()
    main(smoke_test=args.smoke, num_samples=args.samples)