import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, LoraConfig
from trl import SFTTrainer

# --- Configuration ---
base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
adapter_model_id = "Nutnell/DirectEd-AI"  # Your adapter on the Hub
new_dataset_path = "new_training_data.jsonl"  # Path to your new data
new_output_dir = "./DirectEd-AI-v2"  # Local directory to save the new version

# --- Load Model and Tokenizer ---
print("Loading base model on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="cpu",       # Force CPU
    torch_dtype=torch.float32,  # Ensure full precision (no fp16/bfloat16)
    trust_remote_code=True,
)

print(f"Loading adapter: {adapter_model_id}...")
# Apply your fine-tuned adapter from the Hub
model = PeftModel.from_pretrained(base_model, adapter_model_id)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Load New Data ---
print(f"Loading new dataset from {new_dataset_path}...")
new_dataset = load_dataset("json", data_files=new_dataset_path, split="train")

# --- Continue Training ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

training_args = TrainingArguments(
    output_dir=new_output_dir,
    num_train_epochs=1,  # Increase if dataset is small
    per_device_train_batch_size=1,  # Keep tiny for CPU
    gradient_accumulation_steps=4,  # Simulate bigger batch
    learning_rate=5e-5,  # Slightly lower for stability
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",  # CPU-safe optimizer
    fp16=False,           # Disable mixed precision (CPU only supports fp32)
)

trainer = SFTTrainer(
    model=model,
    train_dataset=new_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_args,
)

print("Starting continued fine-tuning (CPU, may be slow)...")
trainer.train()
print("Training complete.")

# Save the new adapter locally
trainer.model.save_pretrained(new_output_dir)
print(f"New model adapter saved to {new_output_dir}")

# Optional: Push to Hugging Face Hub
# trainer.model.push_to_hub("Nutnell/DirectEd-AI-v2", private=False, commit_message="Continued training on CPU")
