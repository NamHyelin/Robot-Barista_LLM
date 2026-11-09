import time
import json
import re
import torch
import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from slm import slm_inference

model_dir = "./models/tiny-llama-mega"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"

LLM_SERVER_HOST = "165.132.40.52"
LLM_SERVER_PORT = 5001
SERVER_URL      = f"http://{LLM_SERVER_HOST}:{LLM_SERVER_PORT}/llm_verification"

allowed_tokens = [262, 263, 265, 269, 273, 274, 276, 280, 286, 290, 293, 295, 298, 301, 304, 306, 310, 311, 314, 315, 316, 317, 319, 329, 335, 341, 343, 345, 347, 349, 350, 355, 360, 365, 367, 371, 374, 377, 379, 380, 381, 385, 402, 407, 411, 412, 438, 446, 447, 454, 468, 488, 496, 505, 508, 521, 524, 528, 542, 549, 598, 600, 603, 612, 617, 624, 625, 672, 678, 679, 719, 728, 763, 831, 837, 853, 907, 923, 932, 964, 968, 1109, 1113, 1133, 1219, 1236, 1281, 1302, 1398, 1532, 1559, 1581, 1610, 1617, 1633, 1648, 1701, 1704, 1725, 1760, 1763, 1773, 1789, 1797, 1815, 1816, 1862, 1878, 1920, 1943, 1999, 2049, 2078, 2139, 2142, 2148, 2163, 2181, 2326, 2442, 2610, 2646, 2696, 2753, 2878, 2911, 3104, 3113, 3118, 3164, 3189, 3445, 3462, 3478, 3712, 3826, 3833, 3848, 3905, 3938, 3973, 4003, 4088, 4094, 4116, 4161, 4227, 4326, 4524, 4628, 4764, 4798, 4805, 4989, 4992, 5342, 5391, 5617, 5642, 6038, 6054, 6235, 6324, 6527, 6556, 6561, 6781, 6803, 6983, 7021, 7053, 7141, 7254, 7347, 7375, 7420, 7537, 7646, 7933, 8142, 8195, 8296, 8533, 8836, 8887, 9216, 9243, 9683, 9878, 9892, 10173, 10293, 10322, 10484, 10492, 10765, 10924, 11179, 11220, 11790, 12113, 12569, 13231, 13749, 14225, 14890, 14954, 15043, 15327, 15392, 15484, 15774, 16242, 16344, 16668, 17169, 17278, 17827, 18002, 18254, 18345, 19493, 19698, 20447, 20559, 21144, 21353, 22531, 22780, 23167, 23429, 23816, 25529, 25606, 25679, 26163, 26494, 26731, 27274, 27810, 28311, 28684, 29316, 29399, 29871, 29872, 29874, 29877, 29880, 29881, 29884, 29888, 29889, 29892, 29893, 29895, 29896, 29899, 29906, 29907, 29915, 29920, 29924, 29929, 29933, 29940, 29941, 29945, 29946, 29947, 29953, 29955, 29973]

def detokenize(token_id):
    if isinstance(token_id, int):
        return tokenizer.decode(token_id)
    else:
        return tokenizer.decode(token_id[0], skip_special_tokens=True)

# Initialize Flask
app = Flask(__name__)

def detokenize(token_id):
    """Decode a single token ID or batch element."""
    if isinstance(token_id, int):
        return tokenizer.decode(token_id, skip_special_tokens=True)
    return tokenizer.decode(token_id[0], skip_special_tokens=True)

# Allowed tokens list (copy from local configuration)
allowed_tokens = [262, 263, 265, 269, 273, 274, 276, 280, 286, 290, 293, 295, 298, 301, 304, 306, 310, 311, 314, 315, 316, 317, 319, 329, 335, 341, 343, 345, 347, 349, 350, 355, 360, 365, 367, 371, 374, 377, 379, 380, 381, 385, 402, 407, 411, 412, 438, 446, 447, 454, 468, 488, 496, 505, 508, 521, 524, 528, 542, 549, 598, 600, 603, 612, 617, 624, 625, 672, 678, 679, 719, 728, 763, 831, 837, 853, 907, 923, 932, 964, 968, 1109, 1113, 1133, 1219, 1236, 1281, 1302, 1398, 1532, 1559, 1581, 1610, 1617, 1633, 1648, 1701, 1704, 1725, 1760, 1763, 1773, 1789, 1797, 1815, 1816, 1862, 1878, 1920, 1943, 1999, 2049, 2078, 2139, 2142, 2148, 2163, 2181, 2326, 2442, 2610, 2646, 2696, 2753, 2878, 2911, 3104, 3113, 3118, 3164, 3189, 3445, 3462, 3478, 3712, 3826, 3833, 3848, 3905, 3938, 3973, 4003, 4088, 4094, 4116, 4161, 4227, 4326, 4524, 4628, 4764, 4798, 4805, 4989, 4992, 5342, 5391, 5617, 5642, 6038, 6054, 6235, 6324, 6527, 6556, 6561, 6781, 6803, 6983, 7021, 7053, 7141, 7254, 7347, 7375, 7420, 7537, 7646, 7933, 8142, 8195, 8296, 8533, 8836, 8887, 9216, 9243, 9683, 9878, 9892, 10173, 10293, 10322, 10484, 10492, 10765, 10924, 11179, 11220, 11790, 12113, 12569, 13231, 13749, 14225, 14890, 14954, 15043, 15327, 15392, 15484, 15774, 16242, 16344, 16668, 17169, 17278, 17827, 18002, 18254, 18345, 19493, 19698, 20447, 20559, 21144, 21353, 22531, 22780, 23167, 23429, 23816, 25529, 25606, 25679, 26163, 26494, 26731, 27274, 27810, 28311, 28684, 29316, 29399, 29871, 29872, 29874, 29877, 29880, 29881, 29884, 29888, 29889, 29892, 29893, 29895, 29896, 29899, 29906, 29907, 29915, 29920, 29924, 29929, 29933, 29940, 29941, 29945, 29946, 29947, 29953, 29955, 29973]

# Remote LLM verification endpoint
LLM_SERVER_HOST = "165.132.40.52"
LLM_SERVER_PORT = 5001
LLM_SERVER_URL = f"http://{LLM_SERVER_HOST}:{LLM_SERVER_PORT}/llm_verification"

# Partial hybrid inference: returns full prefix up through the first step phrase as soon as it's complete

def uncertainty_aware_hybrid_inference_partial(prompt: str, max_new_tokens: int = 100, uncertainty_threshold: float = 0.5, verbose: bool = True):
    start = time.time()
    resample = 0
    num_inference = 0
    num_transmission = 0
    # special_token = "<recipe_generation>"
    # prompt = f"{special_token} {prompt} {special_token} 1."
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated = inputs["input_ids"]

    initial_length = len(prompt)

    for i in range(max_new_tokens):
        num_inference = i + 1
        draft_token_id, uncertainty, draft_distribution = slm_inference(generated=generated, allowed_tokens=allowed_tokens)
        
        if uncertainty > uncertainty_threshold:
            num_transmission = num_transmission + 1
            if verbose:
                print(f"High uncertainty ({uncertainty:.2f} > {uncertainty_threshold}); calling remote LLM verification...")
            # Convert draft_distribution to a list if needed.
            if isinstance(draft_distribution, torch.Tensor):
                draft_distribution_list = draft_distribution.tolist()
            else:
                draft_distribution_list = draft_distribution
            
            payload = {
                "draft_distribution": draft_distribution_list,
                "draft_token_id": int(draft_token_id),
                "generated": generated.squeeze(0).tolist(),
                "allowed_tokens": allowed_tokens
            }
            
            server_url = SERVER_URL
            # final_token_id, accepted = llm_verification(draft_distribution_list, draft_token_id, generated, allowed_tokens)
            # if not accepted:
            #     resample = resample + 1
            
            try:
                response = requests.post(server_url, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                final_token_id = data['result_token_id']
                accepted = data['accepted']
                if not accepted:
                    resample += 1
            except Exception as e:
                if verbose:
                    print("Error calling remote llm_verification:", e)
                final_token_id = draft_token_id
            
        else:
            if verbose:
                print(f"Low uncertainty ({uncertainty:.2f} <= {uncertainty_threshold}); using SLM token directly.")
            final_token_id = draft_token_id

        chosen_token = torch.tensor([final_token_id], device=device)
        
        if bool(re.search(r"\d", detokenize(chosen_token))):
            tsr = (1 - resample/num_inference) if num_inference > 0 else 1.0
            tr = num_transmission/num_inference if num_inference > 0 else 1.0
            time_elapsed = time.time() - start
            return detokenize(generated)[initial_length:], tsr, tr, num_inference, time_elapsed

        generated = torch.cat([generated, chosen_token.unsqueeze(0)], dim=1)

        if verbose:
            print(detokenize(generated))

        if detokenize(chosen_token).strip() == "Done":
            break

    tsr = (1 - resample/num_inference) if num_inference > 0 else 1.0
    tr = num_transmission/num_inference if num_inference > 0 else 1.0
    time_elapsed = time.time() - start
    return detokenize(generated)[initial_length:], tsr, tr, num_inference, time_elapsed


# Flask route
@app.route("/hybrid_inference_partial", methods=["POST"])
def process_hybrid_partial():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 100)
    uncertainty_threshold = data.get("uncertainty_threshold", 0.5)
    verbose = data.get("verbose", False)

    generated_text, tsr, tr, num_inf, latency = uncertainty_aware_hybrid_inference_partial(
        prompt, max_new_tokens, uncertainty_threshold, verbose
    )
    return jsonify({
        "partial_text": generated_text,
        "true_skip_ratio": tsr,
        "transmission_rate": tr,
        "num_inference": num_inf,
        "latency_s": latency
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)