import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Paths
base_model_path = "meta-llama/Llama-2-7b-chat-hf"
lora_model_path = "./llm-recipe" 

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Quantization config for 4-bit inference
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load base model + LoRA adapter
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quant_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()

# Prompt
prompt = "<recipe_generation> Can I haveHoney Almond Velvet Latte <recipe_generation> 1. Place Cup"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
