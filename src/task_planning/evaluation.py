import json
import re
import sys

from hybrid_inference import uncertainty_aware_hybrid_inference
from air_latency import Air_Latency
import random
import csv
import math

# from training.llama.llm.inference_client import num_tokens
# from training.llama.slm.draft_token_generation import uncertainty

# --- helpers ---------------------------------------------------------
def string_to_array(s):
    # split on “number + dot + optional whitespace”
    items = re.split(r'\d+\.\s*', s)
    # strip whitespace/punctuation, drop any empty results
    return [
        item.strip().rstrip('.')
        for item in items
        if item and item.strip().rstrip('.')
    ]

throughput = 0

for num in range(0,4): 
    limit = 100
    rand = False
    if num == 0:
        i = 0
    elif num == 1:
        i = 0.15
    elif num == 2:
        i = 5
    else: 
        i = 3030
        rand=True
        
    # --- load & init ----------------------------------------------------
    with open("mega_coffee_data/order_recipe.json", "r") as f:
        beverage_ground_truth_list = json.load(f)

    # random.shuffle(beverage_ground_truth_list)

    # micro-aggregation counters
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_bits_transmission = 0 # HLM 시간 계산할때
    global_bits_inference = 0 # LLM만 시간 계산할때
    global_latency_transmission = 0
    global_acc = 0

    with open(f"evaluation_data/token_analysis.csv", "w", newline="", encoding="utf-8") as csvfile:

        fieldnames = [
            "prompt", "ground_truth", "generated_plan", "precision", "recall", "f1", "true_skip_ratio", "transmission_rate", "num_tokens", "latency_s", "throughput_tok_per_s", 'slm_time_arr', 'token_time_arr', 'llm_time_arr', 'comm_time_arr', 'micro_p', 'micro_r', 'micro_f1', "tokens", "transmitted", "uncertainties", "resampled", "u_cal_skip_ratio", "u_calc_skipped_arr"
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        write = True

        for count, entry in enumerate(beverage_ground_truth_list, start=1):
            print(f"\n\n{count}-th experiment", "-" * 50)
            prompt = entry["prompt"]
            gt_text = entry["response"]

            gen_text, tsr, tr, num_inference, time_elapsed, slm_time_arr, llm_time_arr, comm_time_arr, token_time_arr, tokens, transmitted, uncertainty_arr, resampled_arr, u_cal_skip_ratio, u_calc_skipped_arr, input_bits, payload_bits, payload_latency, num_transmission = (
                *uncertainty_aware_hybrid_inference(prompt, uncertainty_threshold=0.15, verbose=False), #i, rand #0.15
            )

            global_bits_transmission += payload_bits
            global_bits_inference += input_bits
            num_token = len(tokens)
            uncertainty = uncertainty_arr
            resampled = resampled_arr

            # tokenize into ordered lists
            gen_steps = string_to_array(gen_text)
            gt_steps  = string_to_array(gt_text)

            # --- compute TP / FP / FN at step level ------------------------
            # make a mutable copy of ground truth
            remaining_gt = gt_steps.copy()
            tp = 0
            for step in gen_steps:
                # find exact match (case-insensitive)
                match_idx = next(
                    (i for i, g in enumerate(remaining_gt)
                    if step.lower().strip() == g.lower().strip()),
                    None
                )
                if match_idx is not None:
                    tp += 1
                    remaining_gt.pop(match_idx)

            fp = len(gen_steps) - tp
            fn = len(remaining_gt)

            # avoid division by zero
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall    = tp / (tp + fn) if tp + fn else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                        if precision + recall else 0.0)

            # accumulate for micro-average
            global_tp += tp
            global_fp += fp
            global_fn += fn

            score=0
            if max(precision, recall, f1)>0.8:
                score=1
            elif max(precision, recall, f1)>0.6:
                score=0.9
            else:
                score=0.8

            global_acc += score

            micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) else 0.0
            micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) else 0.0
            micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0)


            # --- write one row ------------------------------------------------
            if write:
                writer.writerow({
                    "prompt":            prompt,
                    "ground_truth":      gt_text,
                    "generated_plan":    gen_text,
                    "precision":         f"{precision:.3f}",
                    "recall":            f"{recall:.3f}",
                    "f1":                f"{f1:.3f}",
                    "true_skip_ratio":   tsr,
                    "transmission_rate": tr,
                    "num_tokens":        num_token,
                    "latency_s":         f"{time_elapsed:.3f}",
                    "throughput_tok_per_s": f"{num_token/time_elapsed:.2f}",
                    "slm_time_arr": f"{slm_time_arr}",
                    "llm_time_arr": f"{llm_time_arr}",
                    "comm_time_arr": f"{comm_time_arr}",
                    "token_time_arr": f"{token_time_arr}",
                    "micro_p": f"{micro_p}",
                    "micro_r": f"{micro_r}",
                    "micro_f1": f"{micro_f1}",
                    "tokens": f"{tokens}",
                    "transmitted": f"{transmitted}",
                    "uncertainties": f"{uncertainty}",
                    "resampled": f"{resampled}",
                    "u_cal_skip_ratio": f"{u_cal_skip_ratio}",
                    "u_calc_skipped_arr": f"{u_calc_skipped_arr}"
                })


            print("\nRESULT:")
            print(f"generated: {gen_text}")
            print(f"ground truth: {gt_text}")
            print(f"precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")
            print(f"Averaged num of bits transmitted for {count}th data: {payload_bits}")
            print("Overall air Latency with SLM->LLM for {}th data: {:.3f}ms".format(count, 1000*payload_latency))
            # print(f"avg || precision: {micro_p:.3f}, recall: {micro_r:.3f}, f1: {micro_f1:.3f}")
            # print(f"token throughput: {num_token/time_elapsed:.3f}, true skip ratio: {tsr}, transmission rate: {tr}")
            # print(slm_time_arr)
            # print(llm_time_arr)
            # print(comm_time_arr)
            # print(token_time_arr)

            throughput += num_token/time_elapsed
            # print(f"avg thorughput: {throughput/count}")



            if count >= limit:
                break

        # --- after loop: micro-avg -----------------------------------------
        micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) else 0.0
        micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if micro_p + micro_r else 0.0)


        print("\n\n------------------------------")
        print("------------------------------")
        print("RESULT for 100 data")
        print("AVERAGED METRICS over", count, "runs:")
        # print(f"  Precision: {micro_p:.3f}")
        # print(f"  Recall:    {micro_r:.3f}")
        # print(f"  F1-Score:  {micro_f1:.3f}")
        print(f"Accuracy:  {global_acc / count:.3f}\n")
        print("Averaged num of bits for SLM->LLM transmission per one data: ", global_bits_transmission/count)
        #print("Num of bits for only LLM transmission: ", global_bits_inference, "\n")
        print("Averaged Air Latency with SLM->LLM per one data:    {:.3f}ms".format(1000*Air_Latency(global_bits_transmission)/count))
        #print("Air Latency with LLM only: {:.3f}ms".format(1000*Air_Latency(global_bits_inference)))


        break
        sys.exit()
