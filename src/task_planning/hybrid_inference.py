import torch
import time
import requests  
from slm import slm_inference
from transformers import AutoTokenizer
# from llm import llm_verification
import random
from flask import Flask, request, jsonify
import math
from air_latency import Air_Latency

model_dir = "./models/tiny-llama-mega"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"

LLM_SERVER_HOST = "127.0.0.1"
LLM_SERVER_PORT = 5001
SERVER_URL      = f"http://{LLM_SERVER_HOST}:{LLM_SERVER_PORT}/llm_verification"

allowed_tokens = [262, 263, 265, 269, 273, 274, 276, 280, 286, 290, 293, 295, 298, 301, 304, 306, 310, 311, 314, 315, 316, 317, 319, 329, 335, 341, 343, 345, 347, 349, 350, 355, 360, 365, 367, 371, 374, 377, 379, 380, 381, 385, 402, 407, 411, 412, 438, 446, 447, 454, 468, 488, 496, 505, 508, 521, 524, 528, 542, 549, 598, 600, 603, 612, 617, 624, 625, 672, 678, 679, 719, 728, 763, 831, 837, 853, 907, 923, 932, 964, 968, 1109, 1113, 1133, 1219, 1236, 1281, 1302, 1398, 1532, 1559, 1581, 1610, 1617, 1633, 1648, 1701, 1704, 1725, 1760, 1763, 1773, 1789, 1797, 1815, 1816, 1862, 1878, 1920, 1943, 1999, 2049, 2078, 2139, 2142, 2148, 2163, 2181, 2326, 2442, 2610, 2646, 2696, 2753, 2878, 2911, 3104, 3113, 3118, 3164, 3189, 3445, 3462, 3478, 3712, 3826, 3833, 3848, 3905, 3938, 3973, 4003, 4088, 4094, 4116, 4161, 4227, 4326, 4524, 4628, 4764, 4798, 4805, 4989, 4992, 5342, 5391, 5617, 5642, 6038, 6054, 6235, 6324, 6527, 6556, 6561, 6781, 6803, 6983, 7021, 7053, 7141, 7254, 7347, 7375, 7420, 7537, 7646, 7933, 8142, 8195, 8296, 8533, 8836, 8887, 9216, 9243, 9683, 9878, 9892, 10173, 10293, 10322, 10484, 10492, 10765, 10924, 11179, 11220, 11790, 12113, 12569, 13231, 13749, 14225, 14890, 14954, 15043, 15327, 15392, 15484, 15774, 16242, 16344, 16668, 17169, 17278, 17827, 18002, 18254, 18345, 19493, 19698, 20447, 20559, 21144, 21353, 22531, 22780, 23167, 23429, 23816, 25529, 25606, 25679, 26163, 26494, 26731, 27274, 27810, 28311, 28684, 29316, 29399, 29871, 29872, 29874, 29877, 29880, 29881, 29884, 29888, 29889, 29892, 29893, 29895, 29896, 29899, 29906, 29907, 29915, 29920, 29924, 29929, 29933, 29940, 29941, 29945, 29946, 29947, 29953, 29955, 29973]

def detokenize(token_id):
    if isinstance(token_id, int):
        return tokenizer.decode(token_id)
    else:
        return tokenizer.decode(token_id[0], skip_special_tokens=True)

# Returns the final generated prompt and the true skip ratio.
def uncertainty_aware_hybrid_inference(prompt: str, max_new_tokens: int = 100, uncertainty_threshold: float = 0.5, verbose: bool = True, rand: bool = False):
    start = time.time()
    resample = 0
    num_inference = 0
    num_transmission = 0
    u_calc_skipped = 0
    special_token = "<recipe_generation>"
    prompt = f"{special_token} {prompt} {special_token}"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated = inputs["input_ids"]

    initial_length = len(prompt)

    slm_time_arr = []
    llm_time_arr = []
    comm_time_arr = []
    token_time_arr = []

    tokens = [] 
    transmitted = []
    uncertainty_arr = []
    resampled_arr = []
    u_calc_skipped_arr = []
    payload_bits = 0
    payload_latency = 0
    for i in range(max_new_tokens):
        exceeded = False
        time_now = time.time()
        num_inference = i + 1
        draft_token_id, uncertainty, draft_distribution, skipped = slm_inference(generated=generated, allowed_tokens=allowed_tokens)
        slm_time = time.time() - time_now
        resampled = False
        input_bits = math.ceil(math.log2(len(draft_distribution))) * len(generated.squeeze(0).tolist())
        bits_per_transmission=0



        if skipped == True:
            llm_time = 0
            u_calc_skipped += 1
            final_token_id = draft_token_id
            u_calc_skipped_arr.append(1)

            # print(f"uncertainty: {uncertainty} || {uncertainty <= uncertainty_threshold}")

        else:
            
            u_calc_skipped_arr.append(0)
            if rand:
                probability = 0.23
                if random.random() < probability:
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

                    payload_bits += sum(1 for x in draft_distribution_list if x != 0) * 8 \
                                    + math.ceil(math.log2(len(draft_distribution_list))) \
                                    + input_bits

                    print("\nSLM --> LLM Transmission: ")
                    print(f"Probability list with length {len(draft_distribution_list)}, : {draft_distribution_list[:5]},...")
                    print(f"SLM result: {detokenize(draft_token_id)}")
                    print(f"Input text data: {detokenize(generated)}\n")

                    
                    server_url = SERVER_URL
                    # final_token_id, accepted = llm_verification(draft_distribution_list, draft_token_id, generated, allowed_tokens)
                    # if not accepted:
                    #     resample = resample + 1
                    llm_time = 0
                    
                    try:
                        response = requests.post(server_url, json=payload, timeout=15)
                        response.raise_for_status()
                        # print(response)
                        data = response.json()
                        final_token_id = data['result_token_id']
                        accepted = data['accepted']
                        llm_time = data['llm_time']
                        # print(llm_time)
                        if not accepted:
                            resample += 1
                    except Exception as e:
                        llm_time = 0
                        if verbose:
                            print(f"Low uncertainty ({uncertainty:.2f} <= {uncertainty_threshold}); using SLM token directly.")
                        final_token_id = draft_token_id
                else:
                    llm_time = 0
                    if verbose:
                        print(f"Low uncertainty ({uncertainty:.2f} <= {uncertainty_threshold}); using SLM token directly.")
                    final_token_id = draft_token_id
            else:
                if uncertainty > uncertainty_threshold:
                    # print("Hello")
                    exceeded=True
                    num_transmission = num_transmission + 1
                    if verbose:
                        print(f"High uncertainty ({uncertainty:.2f} > {uncertainty_threshold}); calling remote LLM verification...")
                    # Convert draft_distribution to a list if needed.
                    if isinstance(draft_distribution, torch.Tensor):
                        draft_distribution_list = draft_distribution.tolist()
                    else:
                        draft_distribution_list = draft_distribution
                    
                    payload = {
                        "draft_distribution": draft_distribution_list, #slm이 추측한 distribution 들. 8bit
                        "draft_token_id": int(draft_token_id), #slm이 추측한 단어에 대한 distribution을 알기위한 index
                        "generated": generated.squeeze(0).tolist(), #input prompt.
                        "allowed_tokens": allowed_tokens #쓰는 token들. 안보내도 됩
                    }

                    bits_per_transmission+= sum(1 for x in draft_distribution_list if x != 0)*8 \
                                  + math.ceil(math.log2(len(draft_distribution_list))) \
                                  + input_bits
                    payload_bits += bits_per_transmission
                    payload_latency += Air_Latency(bits_per_transmission)

                    print("\nSLM --> LLM Transmission: ")
                    print(f"Probability list with length {len(draft_distribution_list)}, : {draft_distribution_list[:5]},...")
                    print(f"SLM result: {detokenize(draft_token_id)}")
                    print(f"Input text data: {detokenize(generated)}")
                    print(f"Num of bits transmitted: {bits_per_transmission}")
                    print("Air Latency: {:.5f}ms".format(1000 * Air_Latency(bits_per_transmission)))
                    
                    server_url = SERVER_URL
                    # final_token_id, accepted = llm_verification(draft_distribution_list, draft_token_id, generated, allowed_tokens)
                    # if not accepted:
                    #     resample = resample + 1
                    llm_time = 0
                    slm_time = time.time() - time_now
                    try:
                        response = requests.post(server_url, json=payload, timeout=15)
                        response.raise_for_status()
                        # print(response)
                        data = response.json()
                        final_token_id = data['result_token_id']
                        accepted = data['accepted']
                        llm_time = data['llm_time']
                        # print(llm_time)
                        if not accepted:
                            resample += 1
                            resampled = True
                    except Exception as e:
                        if verbose:
                            print("Error calling remote llm_verification:", e)
                        final_token_id = draft_token_id
                    
                else:
                    llm_time = 0
                    if verbose:
                        print(f"Low uncertainty ({uncertainty:.2f} <= {uncertainty_threshold}); using SLM token directly.")
                    final_token_id = draft_token_id
        token_time = time.time() - time_now
        chosen_token = torch.tensor([final_token_id], device=device)
        generated = torch.cat([generated, chosen_token.unsqueeze(0)], dim=1)
        if verbose:
            print(detokenize(generated))

        if detokenize(chosen_token).strip() == "Done":
            break

        slm_time_arr.append(slm_time)
        llm_time_arr.append(llm_time)
        comm_time_arr.append(token_time - slm_time - llm_time)
        token_time_arr.append(token_time)

        tokens.append(detokenize(chosen_token))
        transmitted.append(exceeded)
        uncertainty_arr.append(uncertainty)
        resampled_arr.append(resampled)
    tsr = (1 - resample/num_inference) if num_inference > 0 else 1.0
    tr = num_transmission/num_inference if num_inference > 0 else 1.0
    u_cal_skip_ratio = u_calc_skipped/num_inference
    time_elapsed = time.time() - start
    return detokenize(generated)[initial_length:], tsr, tr, num_inference, time_elapsed, slm_time_arr, llm_time_arr, comm_time_arr, token_time_arr, tokens, transmitted, uncertainty_arr, resampled_arr, u_cal_skip_ratio, u_calc_skipped_arr, input_bits, payload_bits, payload_latency, num_transmission

app = Flask(__name__)

@app.route('/order', methods=['POST'])
def order():
    data = request.get_json()
    order_text = data.get('order')
    if not order_text:
        return jsonify({'error': 'missing order text'}), 400

    # generate_recipe should take the order string and return a list or dict
    recipe, *rest = uncertainty_aware_hybrid_inference(order_text, uncertainty_threshold=0.15, verbose=True)
    return jsonify({'recipe': recipe}), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

