# fine_tuning/test_finetune.py

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
adapter_path = "fine_tuning/results/llama-3-8b-instruct-direct-ed"

# Load the Tokenizer and Model
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print("Base model and tokenizer loaded.")

# Merge the Base Model with the LoRA Adapter
print(f"Loading LoRA adapter from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()
print("Model and adapter merged successfully.")

prompt_text = "Explain the difference between MLOps and LLMOps in a simple way for a beginner."

messages = [
    {"role": "user", "content": prompt_text},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\n--- Testing with prompt ---\n{prompt_text}\n")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=250,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("--- Model Response ---")

assistant_response = response.split("assistant")[1].strip()
print(assistant_response)