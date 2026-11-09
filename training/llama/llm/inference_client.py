# inference_client.py
import json
import time
import re
import requests
import csv

# helper to split into steps
def string_to_array(s):
    items = re.split(r'\d+\.\s*', s)
    return [item.strip().rstrip('.') for item in items if item.strip()]

# load test set
with open("mega_coffee_data/test_dataset.json", "r") as f:
    test_data = json.load(f)

# micro‐counters
global_tp = global_fp = global_fn = 0
count = 0
with open(f"evaluation_data/llm_only.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "prompt", "ground_truth", "generated_plan", "precision", "recall", "f1", "true_skip_ratio", "transmission_rate", "num_tokens", "latency_s", "throughput_tok_per_s"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for entry in test_data:
        prompt  = entry["prompt"]
        gt_text = entry["response"]
        special = "<recipe_generation>"
        full_prompt = f"{special} {prompt} {special}"
        initial_length = len(full_prompt)

        # send to server and measure round‐trip
        t0 = time.time()
        resp = requests.post(
            "http://127.0.0.1:5001/llm_verification",
            json={"prompt": full_prompt},
            timeout=30
        ).json()
        t1 = time.time()

        gen_text     = resp["generated_text"][initial_length:]
        gen_time     = resp["gen_time"]
        num_tokens   = resp["num_tokens"]
        total_time   = t1 - t0
        transmission = total_time - gen_time
        throughput   = num_tokens / total_time if total_time > 0 else 0.0

        # step‐level metrics
        gen_steps = string_to_array(gen_text)
        gt_steps  = string_to_array(gt_text)

        rem = gt_steps.copy()
        tp = 0
        for step in gen_steps:
            idx = next((i for i,g in enumerate(rem)
                        if step.lower().strip()==g.lower().strip()), None)
            if idx is not None:
                tp += 1
                rem.pop(idx)

        fp = len(gen_steps) - tp
        fn = len(rem)
        global_tp += tp
        global_fp += fp
        global_fn += fn

        precision = tp/(tp+fp) if tp+fp else 0.0
        recall    = tp/(tp+fn) if tp+fn else 0.0
        f1        = (2*precision*recall/(precision+recall)
                    if precision+recall else 0.0)

        # print per‐example
        print(f"Example prompt: {prompt}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"Gen tokens: {num_tokens}, Gen time: {gen_time:.3f}s")
        print(f"Transmission time: {transmission:.3f}s, Throughput: {throughput:.1f} tok/s")
        print("-"*60)

        writer.writerow({
            "prompt":            prompt,
            "ground_truth":      gt_text,
            "generated_plan":    gen_text,
            "precision":         f"{precision:.3f}",
            "recall":            f"{recall:.3f}",
            "f1":                f"{f1:.3f}",
            "true_skip_ratio":   0,
            "transmission_rate": 0,
            "num_tokens":        num_tokens,
            "latency_s":         f"{total_time:.3f}",
            "throughput_tok_per_s": f"{throughput:.1f}"
        })
        if count > 100:
            break
        count = count + 1

# final micro‐averaged
micro_prec = global_tp/(global_tp+global_fp) if global_tp+global_fp else 0.0
micro_rec  = global_tp/(global_tp+global_fn) if global_tp+global_fn else 0.0
micro_f1   = (2*micro_prec*micro_rec/(micro_prec+micro_rec)
              if micro_prec+micro_rec else 0.0)

print("=== Aggregate Results ===")
print(f"Micro-Precision: {micro_prec:.3f}")
print(f"Micro-Recall:    {micro_rec:.3f}")
print(f"Micro-F1:        {micro_f1:.3f}")
