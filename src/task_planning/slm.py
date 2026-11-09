from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import json
import time
import joblib
from collections import deque
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH    = "./models/tiny-llama-mega"

_tokenizer = None
_model     = None

def _init_slm():
    global _tokenizer, _model
    if _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False
    )
    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        low_cpu_mem_usage=True,     # optional
        torch_dtype=torch.float16   # optional; speeds things up
    )
    _model = PeftModel.from_pretrained(_model, ADAPTER_PATH)
    print(DEVICE)
    _model.to(DEVICE)
    _model.eval()

import ast
import json
import joblib
from collections import deque

import torch
import numpy as np

# ────────── Load config & classifier ──────────
CONFIG_IN      = "skip_config_new.json"
CLASSIFIER_IN  = "skip_classifier_new.joblib"

with open(CONFIG_IN, "r") as f:
    cfg = json.load(f)

clf         = joblib.load(CLASSIFIER_IN)
feature_cols= cfg["feature_cols"]
K           = cfg["history_K"]
Cu          = cfg["Cu"]
Ce          = cfg["Ce"] * 2
tau_star    = Cu/Ce
# ──── Buffers for history features ────
us_buf = deque([0.0]*K, maxlen=K)

# ──── Utility: assemble feature vector ────
def assemble_features(us_buf, token_id):
    """Return [feat_u_1, …, feat_u_K, feat_token_id]."""
    feats = list(us_buf) + [token_id]
    return np.array(feats).reshape(1, -1)
'''
# ──── Inference function ────
def slm_inference(generated, allowed_tokens, theta_max: float = 2.0, K_loop: int = 20):
    """
    Perform SLM inference with adaptive skip based on history of uncertainties
    and current token ID. Returns:
      draft_token_id,  uncertainty,  draft_dist,  skipped (bool)
    """
    # 1) SLM forward pass
    _init_slm()
    outputs = _model(generated)
    logits  = outputs.logits[0, -1, :]
    
    # 2) Mask disallowed tokens
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[torch.tensor(allowed_tokens, device=logits.device)] = True
    masked_logits = torch.where(mask, logits, torch.tensor(-float("Inf"), device=logits.device))
    
    # 3) Draft distribution & token
    draft_dist     = torch.softmax(masked_logits, dim=-1)
    draft_token_id = draft_dist.argmax().item()
    
    # 4) Decide skip vs full uncertainty loop
    #    build features: [u_{t-1}…u_{t-K}, token_id]
    x_t = assemble_features(us_buf, draft_token_id)
    prob = clf.predict_proba(x_t)[0, 1]

    # threshold = Bayes‐optimal tau_star
    # print(f"tar_star: {tau_star}")
    skipped = True
    if prob < tau_star:
        # skip full K‐perturbation loop
        uncertainty = 0.0
        skipped = True
        # compute true uncertainty via temperature perturbations
    else: 
        skipped = False
        perturbed = []
        temps = torch.empty(K_loop, device=logits.device).uniform_(0.05, theta_max)
        for temp in temps:
            d = torch.softmax(logits / temp, dim=-1)
            perturbed.append(d.multinomial(1).item())
        diff_count  = sum(1 for t in perturbed if t != draft_token_id)
        uncertainty = diff_count / K_loop

    # 5) Update history buffer
    us_buf.append(uncertainty)

    return draft_token_id, uncertainty, draft_dist, skipped
'''
def slm_inference(generated, allowed_tokens, theta_max: float = 2.0, K: int = 20):
    """
    Perform SLM inference with adaptive uncertainty estimation.
    Returns: draft_token_id, uncertainty, draft_distribution, skipped (always False here)
    """

    _init_slm()
    #now = time.time()
    # 1) Forward pass (no grad)
    
    start = time.time()
    with torch.no_grad():
        outputs = _model(generated)
    logits = outputs.logits[0, -1, :]

    # print(time.time() - start)
    start = time.time()
    
        #print("skipped")
        #print(time.time()- now)
        #return id, None, None, True
    
    V = logits.size(0)
    allowed_ids = torch.tensor(allowed_tokens, dtype=torch.long)
    if allowed_ids.min() < 0 or allowed_ids.max() >= V:
        raise ValueError(
            f"allowed_tokens contains invalid id(s): "
            f"[min={int(allowed_ids.min())}, max={int(allowed_ids.max())}], vocab size={V}"
        )
    allowed_ids = allowed_ids.to(logits.device)

    # 3) Mask out disallowed tokens with -inf
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[allowed_ids] = True
    if not mask.any():
        raise RuntimeError("No valid tokens left after masking.")
    masked_logits = logits.masked_fill(~mask, float("-inf"))

    # print(time.time() - start)
    start = time.time()

    # 4) Draft distribution & sample
    draft_distribution = torch.softmax(masked_logits, dim=-1)
    draft_token_id = torch.multinomial(draft_distribution, num_samples=1).item()

    x_t = assemble_features(us_buf, draft_token_id)
    prob = clf.predict_proba(x_t)[0, 1]

    skip = False
    
    if prob < tau_star:
        us_buf.append(0)
        skip = True

    # print(time.time() - start)
    start = time.time()

    diff_count = 0

    for _ in range(K):
        temp = torch.empty((), device=logits.device).uniform_(0.05, theta_max)  # scalar
        perturbed_logits = masked_logits / temp   # [V]
        perturbed_dist = torch.softmax(perturbed_logits, dim=-1)  # [V]
        sampled_id    = torch.multinomial(perturbed_dist, num_samples=1).item()

        if sampled_id != draft_token_id:
            diff_count += 1

    uncertainty = diff_count / K
    us_buf.append(uncertainty)
    return draft_token_id, uncertainty, draft_distribution, False

if __name__ == "__main__":
    _init_slm()
    prompt = "I want iced caramel?"
    special_token = "<recipe_generation>"
    prompt = f"{special_token} {prompt} {special_token} 1. Place Cup 2."
    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    generated = inputs["input_ids"]
    allowed_tokens = [262, 263, 265, 269, 273, 274, 276, 280, 286, 290, 293, 295, 298, 301, 304, 306, 310, 311, 314, 315, 316, 317, 319, 329, 335, 341, 343, 345, 347, 349, 350, 355, 360, 365, 367, 371, 374, 377, 379, 380, 381, 385, 402, 407, 411, 412, 438, 446, 447, 454, 468, 488, 496, 505, 508, 521, 524, 528, 542, 549, 598, 600, 603, 612, 617, 624, 625, 672, 678, 679, 719, 728, 763, 831, 837, 853, 907, 923, 932, 964, 968, 1109, 1113, 1133, 1219, 1236, 1281, 1302, 1398, 1532, 1559, 1581, 1610, 1617, 1633, 1648, 1701, 1704, 1725, 1760, 1763, 1773, 1789, 1797, 1815, 1816, 1862, 1878, 1920, 1943, 1999, 2049, 2078, 2139, 2142, 2148, 2163, 2181, 2326, 2442, 2610, 2646, 2696, 2753, 2878, 2911, 3104, 3113, 3118, 3164, 3189, 3445, 3462, 3478, 3712, 3826, 3833, 3848, 3905, 3938, 3973, 4003, 4088, 4094, 4116, 4161, 4227, 4326, 4524, 4628, 4764, 4798, 4805, 4989, 4992, 5342, 5391, 5617, 5642, 6038, 6054, 6235, 6324, 6527, 6556, 6561, 6781, 6803, 6983, 7021, 7053, 7141, 7254, 7347, 7375, 7420, 7537, 7646, 7933, 8142, 8195, 8296, 8533, 8836, 8887, 9216, 9243, 9683, 9878, 9892, 10173, 10293, 10322, 10484, 10492, 10765, 10924, 11179, 11220, 11790, 12113, 12569, 13231, 13749, 14225, 14890, 14954, 15043, 15327, 15392, 15484, 15774, 16242, 16344, 16668, 17169, 17278, 17827, 18002, 18254, 18345, 19493, 19698, 20447, 20559, 21144, 21353, 22531, 22780, 23167, 23429, 23816, 25529, 25606, 25679, 26163, 26494, 26731, 27274, 27810, 28311, 28684, 29316, 29399, 29871, 29872, 29874, 29877, 29880, 29881, 29884, 29888, 29889, 29892, 29893, 29895, 29896, 29899, 29906, 29907, 29915, 29920, 29924, 29929, 29933, 29940, 29941, 29945, 29946, 29947, 29953, 29955, 29973]
   