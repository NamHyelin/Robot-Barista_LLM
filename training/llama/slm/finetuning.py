import os

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model

token = os.getenv("HF_TOKEN")

data1 = load_dataset("json", data_files="mega_coffee_data/drink_recipe.json")["train"]
data2 = load_dataset("json", data_files="mega_coffee_data/modified_order_recipe.json")["train"]
data3 = load_dataset("json", data_files="mega_coffee_data/order_recipe.json")["train"]
# data4 = load_dataset("json", data_files="mega_coffee_data/combined_training_data.json")["train"]

combined_data = Dataset.from_dict({
    "prompt":   data1["prompt"]   + data2["prompt"]  + data3["prompt"],
    "response": data1["response"] + data2["response"] + data3["response"],
})

# Shuffle and split into train/test
dataset = combined_data.train_test_split(test_size=0.1, seed=42)
first_example = dataset['train'][0]

# Print the actual content of the 'prompt' and 'response'
# print("Prompt:", first_example['prompt'])
# print("Response:", first_example['response'])

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    token=token,
    device_map="auto"
)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    # Option 1: Use the EOS token as the pad token.
    tokenizer.pad_token = tokenizer.eos_token


special_token = "<recipe_generation>"


lora_config = LoraConfig(
    r=8,                  
    lora_alpha=32,        
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.05,    
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

MAX_LENGTH = 128

def preprocess_function(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for prompt_text, response_text in zip(examples["prompt"], examples["response"]):
        prompt_part = f"{special_token} {prompt_text} {special_token}"
        full_text = prompt_part + " " + response_text

        tokenized = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        tokenized_prompt = tokenizer(prompt_part, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
        prompt_len = len(tokenized_prompt["input_ids"])

        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)

training_args = TrainingArguments(
    output_dir="./models/tiny-llama-mega",
    # keep at 5 epochs (tweak up to 7–8 if under‑fitting)
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    eval_strategy="steps",
    # evaluate every 100 steps to catch over‑fitting early
    eval_steps=100,
    save_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    fp16=True,
    # automatically reload best checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # keep only 3 most recent checkpoints
    save_total_limit=3,
    overwrite_output_dir=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

trainer.save_model("./models/tiny-llama-mega")
tokenizer.save_pretrained("./models/tiny-llama-mega")
