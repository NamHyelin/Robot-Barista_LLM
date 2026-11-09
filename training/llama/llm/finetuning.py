import os, sys
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model

ROOT = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.append(os.path.join(ROOT, "src", "task_planning"))

# from rag import RAGPromptGenerator   # ← import your RAG helper

# your HuggingFace token (if needed)
token = os.getenv("HF_TOKEN")

# ─── 1) Build the RAG prompt generator ─────────────────────────────────────────
'''
rag_generator = RAGPromptGenerator(
    recipe_file="mega_coffee_data/drink_recipe.json"
)
'''

# ─── 2) Load & interleave your three JSON datasets ─────────────────────────────
data1 = load_dataset("json", data_files="mega_coffee_data/drink_recipe.json")["train"]
data2 = load_dataset("json", data_files="mega_coffee_data/modified_order_recipe.json")["train"]
data3 = load_dataset("json", data_files="mega_coffee_data/order_recipe.json")["train"]
data4 = load_dataset("json", data_files="mega_coffee_data/combined_training_data.json")["train"]


combined_data = Dataset.from_dict({
    "prompt":   data1["prompt"]   + data2["prompt"]  + data3["prompt"] + data4["prompt"],
    "response": data1["response"] + data2["response"] + data3["response"] + data4["response"],
})

'''
def make_rag_prompt(example):
    return {
        "prompt": rag_generator.generate_rag_prompt(example["prompt"])
    }

# map with batched=False since generate_rag_prompt is per-example
combined_data = combined_data.map(make_rag_prompt, batched=False)
'''
# ─── 4) Shuffle & split into train / test ───────────────────────────────────────
dataset = combined_data.train_test_split(test_size=0.1, seed=42)

# ─── 5) Load tokenizer & model ─────────────────────────────────────────────────
model_path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer  = AutoTokenizer.from_pretrained(model_path, token=token)

# optional 8‑bit quantization config (unused in this script, but kept for reference)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    token=token,
    device_map="auto"
)

# ensure a pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── 6) Add your special marker and resize embeddings ───────────────────────────
special_token = "<recipe_generation>"

# ─── 7) Apply LoRA adapters ────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ─── 8) Tokenization / preprocessing ────────────────────────────────────────────
MAX_LENGTH = 512

def preprocess_function(examples):
    input_ids_list      = []
    attention_mask_list = []
    labels_list         = []

    for prompt_text, response_text in zip(examples["prompt"], examples["response"]):
        # wrap with special token
        prompt_part = f"{special_token} {prompt_text} {special_token}"
        full_text   = prompt_part + " " + response_text

        tokenized       = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
        input_ids       = tokenized["input_ids"]
        attention_mask  = tokenized["attention_mask"]

        # mask out prompt portion in labels
        tokenized_prompt = tokenizer(
            prompt_part,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False
        )
        prompt_len = len(tokenized_prompt["input_ids"])

        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids":      input_ids_list,
        "attention_mask": attention_mask_list,
        "labels":         labels_list,
    }

train_ds = dataset["train"]
for i, example in enumerate(train_ds):
    prompt   = example["prompt"]
    response = example["response"]
    print(f"{i:4d} ▶ PROMPT: {prompt}")
    print(f"     RESPONSE: {response}")
    print("-" * 80)

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ─── 9) Training setup ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./models/llama2-mega",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    fp16=True,
    overwrite_output_dir=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# ─── 10) Launch training ───────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./models/llama2-mega")
    tokenizer.save_pretrained("./models/llama2-mega")
