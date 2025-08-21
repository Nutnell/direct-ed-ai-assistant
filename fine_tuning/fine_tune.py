# fine_tuning/fine_tune.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
output_dir = "fine_tuning/results/llama-3-8b-instruct-direct-ed"
dataset_path = "fine_tuning/dataset.jsonl"


# Load the Dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train")
print("Dataset loaded successfully.")


# Load the Base Model & Tokenizer
print(f"Loading base model: {base_model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
print("Base model loaded successfully.")

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded and configured.")


# Configure PEFT
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
print("LoRA configured.")


# Define Training Arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="linear",
)
print("Training arguments set.")


#Initialize and Start Training
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
)
print("Trainer initialized. Starting the fine-tuning process...")
trainer.train()
print("Training complete.")


# Save the Final Model
trainer.model.save_pretrained(output_dir)
print(f"Fine-tuned model adapter saved to {output_dir}")