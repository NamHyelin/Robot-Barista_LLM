"""
import os, glob
import pandas as pd

DATA_DIR = 'evaluation_data'  
pattern  = os.path.join(DATA_DIR, 'ood_*.csv')
files    = sorted(glob.glob(pattern),
                  key=lambda f: float(os.path.basename(f)[len('ood_'):-4]))

records = []
for path in files:
    thresh = float(os.path.basename(path)[len('ood_'):-4])
    df     = pd.read_csv(path)
    block  = df.iloc[1:51]  # rows 2–51

    records.append({
        'threshold': thresh,
        'avg_D':     block.iloc[:, 3].mean(),
        'avg_E':     block.iloc[:, 4].mean(),
        'avg_F':     block.iloc[:, 5].mean(),
        'avg_K':     block.iloc[:, 10].mean()
    })

result_df = pd.DataFrame(records).sort_values('threshold').reset_index(drop=True)

# ——————————————
# Save to CSV:
result_df.to_csv('ood_threshold_summary.csv', index=False)
# ——————————————

print("Wrote summary to threshold_summary.csv")
"""
import os
import ast
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import time
import ast
import json
import numpy as np
import pandas as pd
from collections import deque
from sklearn.linear_model import LogisticRegression
import joblib

import pandas as pd
import ast
from transformers import AutoTokenizer
'''
# 1) Load CSV
df = pd.read_csv()


# 2) Parse your stringified token lists
def parse_token_list(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

df["token_list"] = df["tokens"].apply(parse_token_list)

# 3) Load TinyLLaMA‐1.1B tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 4) Convert tokens → ids
def tokens_to_ids(token_list):
    return tokenizer(token_list,
                     is_split_into_words=True,
                     add_special_tokens=False)["input_ids"]

df["token_ids"] = df["token_list"].apply(tokens_to_ids)

# 5) Drop the intermediate column if you don’t need it
# df = df.drop(columns=["token_list"])

# 6) Write out to a new CSV
out_path = r"token_pattern_with_ids.csv"
df.to_csv(out_path, index=False)

print(f"Saved augmented CSV to {out_path}")
'''

import ast
import json
import numpy as np
import pandas as pd
from collections import deque
from sklearn.linear_model import LogisticRegression
import joblib

# ───────── Configuration ─────────
CALIB_CSV      = "token_pattern_with_ids.csv"
CLASSIFIER_OUT = "skip_classifier_new.joblib"
CONFIG_OUT     = "skip_config_new.json"

K              = 10        # history window size
THRESHOLD      = 0.15     # uncertainty threshold θ
DELTA_TARGET   = 0.1     # allowable false‐negative rate
Cu             = 0.162    # average SLM uncertainty compute time (s)
Ce             = 0.71     # worst‐case uplink + remote inference time (s)

# ─── 1) Load calibration data ────────────────────────
# Expect CSV with columns:
#   - "transmitted": str(list[bool])
#   - "uncertainties": str(list[float])
#   - "token_ids": str(list[int])    # numeric IDs for each generated token

raw = pd.read_csv(CALIB_CSV)

# ─── 2) Flatten to token level with history + token_id ─────────
records = []
for _, row in raw.iterrows():
    trans_list = ast.literal_eval(row["transmitted"])
    unc_list   = ast.literal_eval(row["uncertainties"])
    tok_list   = ast.literal_eval(row["token_ids"])
    T = len(unc_list)

    # sliding buffer for last K uncertainties (zero‐pad)
    buf_u = deque([0.0]*K, maxlen=K)

    for t in range(T):
        u_t   = unc_list[t]
        tok_t = tok_list[t]
        # label: whether U_t > THRESHOLD (i.e. we transmitted)
        y_t = int(trans_list[t]) if t < len(trans_list) else int(u_t > THRESHOLD)

        # assemble feature dict
        feat = {}
        # previous K uncertainties
        for i, val in enumerate(buf_u, start=1):
            feat[f"feat_u_{i}"] = val
        # current token ID
        feat["feat_token_id"] = tok_t

        feat["label"] = y_t
        records.append(feat)

        # update buffer
        buf_u.append(u_t)

df = pd.DataFrame.from_records(records)

# feature columns: feat_u_1…feat_u_K plus feat_token_id
feature_cols = [f"feat_u_{i}" for i in range(1, K+1)] + ["feat_token_id"]
X = df[feature_cols].values
y = df["label"].values

# ─── 3) Train the skip classifier ────────────────────
clf = LogisticRegression(max_iter=2000)
clf.fit(X, y)

# ─── 4) Compute thresholds ───────────────────────────
# Bayes‐optimal break‐even
tau_star = Cu / Ce

# “Safe” threshold: ensure false‐negative rate ≤ DELTA_TARGET
proba   = clf.predict_proba(X)[:, 1]
taus    = np.linspace(0.0, 1.0, 10001)
false_neg = lambda τ: ((proba < τ) & (y == 1)).sum() / max(1, y.sum())
safe_tau = next(τ for τ in taus if false_neg(τ) <= DELTA_TARGET)

# ─── 5) Persist model + config ──────────────────────
joblib.dump(clf, CLASSIFIER_OUT)
with open(CONFIG_OUT, "w") as f:
    json.dump({
        "tau_star":       tau_star,
        "tau_safe":       safe_tau,
        "Cu":             Cu,
        "Ce":             Ce,
        "delta_target":   DELTA_TARGET,
        "feature_cols":   feature_cols,
        "history_K":      K
    }, f, indent=2)

print(f"Classifier saved → {CLASSIFIER_OUT}")
print(f"Config saved     → {CONFIG_OUT}")
print(f"• tau_star: {tau_star:.4f}")
print(f"• tau_safe: {safe_tau:.4f}")


'''
import pandas as pd

url = r"evaluation_data/uncertainty_calc_skipping_5.csv"
# url = r"C:\Users\jupar\Downloads\data\token_pattern_0.15.csv"
df = pd.read_csv(url)
import ast

# 1) Define exactly which columns you want:
# cols = ['tokens', 'uncertainties', 'u_calc_skipped_arr']
cols = ['token_time_arr', 'slm_time_arr', 'llm_time_arr', 'comm_time_arr']
# 2) (more efficient) Using .itertuples()
for row in df.itertuples(index=True, name='RowData'):
    tokens = df['tokens'].apply(ast.literal_eval)
    uncertainties = df['uncertainties'].apply(ast.literal_eval)
    u_calc_skipped_arr = df['u_calc_skipped_arr'].apply(ast.literal_eval)
    token_time_arr = df['token_time_arr'].apply(ast.literal_eval)
    slm_time_arr = df['slm_time_arr'].apply(ast.literal_eval)
    llm_time_arr = df['llm_time_arr'].apply(ast.literal_eval)
    comm_time_arr = df['comm_time_arr'].apply(ast.literal_eval)
      
avg_token_time = []
avg_slm_proportion = []
avg_llm_proportion = []
avg_comm_proportion = []

count = 0
sum_slm = 0
sum_llm = 0
sum_comm = 0
sum_total = 0

correct = 0
wrong = 0
count_skip = 0

for i in range(0, 28):
    for j in range(0, len(uncertainties[i])):
        # sum_slm += slm_time_arr[i][j]
        # sum_llm += llm_time_arr[i][j]
        # sum_comm += comm_time_arr[i][j]
        # sum_total += token_time_arr[i][j]
        # print(f"token: {tokens[i][j]} || uncertainties: {uncertainties[i][j]} || u_calc_skipped_arr: {u_calc_skipped_arr[i][j]}")
        
        if (uncertainties[i][j] > 0.15) and (u_calc_skipped_arr[i][j] == 1):
            print("wrong")
            wrong += 1
            count_skip += 1
        elif (uncertainties[i][j] <= 0.15) and (u_calc_skipped_arr[i][j] == 1): 
            print("right")
            correct += 1
            count_skip += 1
        # sum_total += token_time_arr[i][j]
        # print(count)

        # if count == 1000:
        #    break
        
        count = count + 1

        
    
    # time.sleep(0.5)

print(correct)
print(wrong)
print(count_skip)
print(count)
# print(sum_total/count)

# avg_token_time_val = sum(avg_token_time) / len(avg_token_time)
# avg_slm_time_val = sum(avg_slm_proportion) / len(avg_slm_proportion)
# avg_llm_time_val = sum(avg_llm_proportion) / len(avg_llm_proportion)
# avg_comm_time_val = sum(avg_comm_proportion) / len(avg_comm_proportion)
# print(len(avg_comm_proportion))

# print(f"token time: {avg_token_time_val} || slm: {avg_slm_time_val} || llm: {avg_llm_time_val} || comm: {avg_comm_time_val}")
# print(sum_total/count)
# print(f"{url[30:]} total: {sum_total/count} comm: {sum_comm/count} slm: {sum_slm/count} llm: {sum_llm/count} count: {count}")
'''
'''
transmitted = 0.0
num_total   = 0

# iterate row-by-row, pulling out and casting each field
for row in df.itertuples(index=True, name='RowData'):
    # cast to pure Python types
    num  = float(row.num_tokens)
    rate = float(row.transmission_rate)
    
    # update running totals
    num_total   += num
    transmitted += num * rate

print(num_total)
print(transmitted)
print(transmitted/num_total)
'''