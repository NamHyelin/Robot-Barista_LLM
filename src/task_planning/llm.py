import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import difflib
#from rag import RAGPromptGenerator
# from transformers import BitsAndBytesConfig
# quant_cfg = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="bfloat16",  # 안되면 "float16"
#     bnb_4bit_use_double_quant=True,
# )
app = Flask(__name__)
import torch, gc
gc.collect(); torch.cuda.empty_cache()


model_name = "./models/llama2-mega"
offload_path = r'D:\hf_offload'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": "cuda:0"}) #{"": "cuda:0"} #, quantization_config=quant_cfg 이건 원래 코드에는 없었음.
#rag = RAGPromptGenerator(recipe_file="mega_coffee_data/drink_recipe.json")

def llm_verification(draft_distribution, draft_token_id, generated, allowed_tokens):
    # candidate_lookup = tokenizer.decode(generated[0].tolist()).strip().split("\n")[0]
    # prefix_text = rag.generate_rag_prompt(candidate_lookup)

    # prefix_tokens = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(generated.device)
    # generated = torch.cat((prefix_tokens, generated), dim=1)
    
    banned_words = ["in", "into", "In"]
    banned_token_ids = []
    for word in banned_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if tokens:
            banned_token_ids.extend(tokens)

    with torch.no_grad():
        outputs = model(generated)

    logits = outputs.logits  
    next_token_logits = logits[0, -1, :]
    for token_id in banned_token_ids:
        next_token_logits[token_id] = -float('Inf')
    
    allowed_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
    allowed_token_ids = torch.tensor(allowed_tokens, device=next_token_logits.device)
    allowed_mask[allowed_token_ids] = True
    
    disallowed_value = float('-inf')
    masked_logits = torch.where(allowed_mask, next_token_logits,
                                 torch.tensor(disallowed_value, device=next_token_logits.device))
    
    target_distribution = torch.softmax(masked_logits, dim=-1)
    
    # Ensure the draft distribution is a numpy array
    if not isinstance(draft_distribution, np.ndarray):
        draft_distribution = np.array(draft_distribution)
    
    target_distribution = target_distribution.cpu().numpy()
    x_d = draft_distribution[draft_token_id]
    y_d = target_distribution[draft_token_id]
    
    if x_d <= y_d or np.random.rand() < y_d / x_d:
        accepted = True
        result_token_id = draft_token_id
    else:
        diff = np.maximum(target_distribution - draft_distribution, 0)
        total_diff = np.sum(diff)
        if total_diff == 0:
            result_token_id = draft_token_id  
        else:
            normalized_diff = diff / total_diff
            result_token_id = np.random.choice(len(draft_distribution), p=normalized_diff)
            accepted = False
    
    return result_token_id, accepted

@app.route('/llm_verification', methods=['POST'])
def call_llm_verification():
    data = request.get_json()
    draft_distribution = data['draft_distribution']
    draft_token_id = data['draft_token_id']
    generated_list = data['generated']
    allowed_tokens = data['allowed_tokens']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Convert the received list of token IDs back into a tensor.
    generated = torch.tensor([generated_list], device=device)
    start = time.time() #######
    result_token_id, accepted = llm_verification(draft_distribution, draft_token_id, generated, allowed_tokens)
    end = time.time() #######
    llm_time = end-start #######
    return jsonify({'result_token_id': result_token_id, 'accepted': accepted, 'llm_time': llm_time})  #######

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
