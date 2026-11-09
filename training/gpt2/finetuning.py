import os
import json
import random
import time

from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset


from google import genai
from google.genai.types import HttpOptions, Part

# Setup tokenizer and model as before
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
special_token = "<recipe_generation>"

# Add the special token to the tokenizer's vocabulary
num_added_tokens = tokenizer.add_tokens([special_token])
print(f"Added {num_added_tokens} token(s): {special_token}")

# Resize the model's token embeddings to accommodate the new token
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(example):
    combined = f"{special_token} {example['prompt']} {special_token} {example['response']}"
    tokenized = tokenizer(combined, truncation=True, padding="max_length", max_length=128)

    special_token_id = tokenizer.convert_tokens_to_ids(special_token)
    input_ids = tokenized["input_ids"]

    try:
        first_index = input_ids.index(special_token_id)
        # Look for the next occurrence after first_index
        second_index = input_ids.index(special_token_id, first_index + 1)
    except ValueError:
        # If not found, default to using the whole sequence for loss.
        tokenized["labels"] = input_ids.copy()
        return tokenized

    labels = [-100] * (second_index + 1) + input_ids[second_index + 1:]
    labels = labels[:len(input_ids)]
    tokenized["labels"] = labels

    return tokenized

# 1. Load your saved JSON file into a Python list of dictionaries
with open("data_collection.json", "r") as f:
    distillation_data = json.load(f)

# Create the dataset from the list of dictionaries
dataset = Dataset.from_list(distillation_data)
tokenized_dataset = dataset.map(tokenize_function, batched=False)

print(dataset[2])

# Use the default data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gpt2_recipe",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
print("Per-device train batch size:", trainer.args.per_device_train_batch_size)

trainer.train()

model.save_pretrained("./gpt2_recipe")
tokenizer.save_pretrained("./gpt2_recipe")