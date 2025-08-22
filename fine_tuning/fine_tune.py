import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- Configuration ---
base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
output_dir = "/data/fine_tuning"
dataset_path = "dataset.jsonl"

# --- Initialize model and tokenizer variables ---
model = None
tokenizer = None

# --- Training Logic ---
# Check if a fine-tuned model adapter already exists
if not os.path.exists(os.path.join(output_dir, 'adapter_config.json')):
    print("No fine-tuned model found. Starting training...")

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Load base model for training
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Training args
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="linear",
        push_to_hub=True,
        hub_model_id="Nutnell/DirectEd-AI",
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text", # Ensure your dataset has a 'text' column
        args=training_arguments,
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained adapter
    trainer.model.save_pretrained(output_dir)
    print(f"Fine-tuned model adapter saved to {output_dir}")
    
    model = trainer.model

# --- Inference Logic ---
# If training did not run, load the existing model
else:
    print("Found existing fine-tuned model. Loading for inference...")
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    # Apply the PEFT adapter
    model = PeftModel.from_pretrained(base_model, output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)


# --- Create Inference Pipeline ---
print("Setting up inference pipeline...")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
print("Inference pipeline ready.")


# --- FastAPI App ---

# PYDANTIC MODEL FOR THE REQUEST BODY
class GenerateRequest(BaseModel):
    prompt: str

app = FastAPI(title="Fine-tuned LLaMA API")

@app.get("/")
def home():
    return {"status": "ok", "message": "Fine-tuned LLaMA is ready."}

@app.post("/generate")
def generate(request: GenerateRequest):
    # Access the prompt from the request object
    formatted_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{request.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    outputs = pipe(formatted_prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return {"response": outputs[0]["generated_text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)