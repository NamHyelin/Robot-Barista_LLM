# inference_server.py
from flask import Flask, request, jsonify
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)

# 1) Load your fine-tuned LoRA model + tokenizer once on startup
BASE = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_DIR = "./models/llama2-mega"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, torch_dtype=torch.float16)
model.eval()

@app.route("/generate", methods=["POST"])
def generate():
    payload = request.get_json()
    prompt = payload["prompt"]

    # tokenize
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # run generation and measure only model time
    start_gen = time.time()
    with torch.no_grad():
        # initialize
        generated = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # autoregressively generate up to 128 tokens
        for _ in range(128):
            outputs = model(generated, attention_mask=attention_mask)
            logits  = outputs.logits[:, -1, :] / 1.0  # temperature=1.0

            # greedy decoding (set DO_SAMPLE logic here if needed)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.squeeze().item()

            # 2) Decode from a single-element list
            token_str = tokenizer.decode([token_id], skip_special_tokens=True).strip()

            # 3) Append and check termination
            generated = torch.cat([generated, next_token], dim=-1)
            if token_str == "Done" or token_id == tokenizer.eos_token_id:
                break

            # extend attention mask if present
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token, device=device)],
                    dim=-1
                )

        out_ids = generated
    gen_time = time.time() - start_gen

    # decode and count
    gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    num_tokens = out_ids.shape[-1] - inputs["input_ids"].shape[-1]

    return jsonify({
        "generated_text": gen_text,
        "gen_time": gen_time,
        "num_tokens": num_tokens
    })

if __name__ == "__main__":
    # listen on your allocated IP and port
    app.run(host="165.132.40.52", port=5001)
