import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "./tinyllama-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, ignore_mismatched_sizes=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

special_token = "<recipe_generation>"
prompt_text = "caramel macchiato with extra shot of espresso."
input_text = f"{special_token} {prompt_text} {special_token}"

inputs = tokenizer(input_text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

max_new_tokens = 50       
K = 20                    
theta_max = 2.0           
uncertainty_threshold = 0.5  
uncertainties = []
generated = inputs["input_ids"]

for i in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(generated)
    logits = outputs.logits  
    next_token_logits = logits[0, -1, :]

    draft_distribution = torch.softmax(next_token_logits, dim=-1)
    draft_token = torch.multinomial(draft_distribution, num_samples=1)
    draft_token_id = draft_token.item()
    draft_token_text = tokenizer.decode(draft_token_id).strip()

    perturbed_tokens = []
    perturbed_tokens_text = []
    temperatures = torch.FloatTensor(K).uniform_(0, theta_max)
    for temp in temperatures:
        perturbed_logits = next_token_logits / temp
        perturbed_distribution = torch.softmax(perturbed_logits, dim=-1)
        sampled_token = torch.multinomial(perturbed_distribution, num_samples=1)
        token_id = sampled_token.item()
        perturbed_tokens.append(token_id)
        perturbed_tokens_text.append(tokenizer.decode(token_id).strip())

    diff_count = sum(1 for token in perturbed_tokens if token != draft_token_id)
    uncertainty = diff_count / K

    print(f"Token {i+1:02d}: Draft token ID: {draft_token_id}, Token: '{draft_token_text}', Uncertainty: {uncertainty:.2f}")
    print("Perturbed tokens:")
    uncertainties.append(uncertainty)
    for idx, (token_id, token_text) in enumerate(zip(perturbed_tokens, perturbed_tokens_text), start=1):
        print(f"  {idx:02d}: ID {token_id}, Token: '{token_text}'")
    print("-" * 40)

    chosen_token = draft_token

    generated = torch.cat([generated, chosen_token.unsqueeze(0)], dim=1)

    if draft_token_id == tokenizer.eos_token_id:
        break

output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("Final Output:")
print(output_text)
print(f"Uncertainties: {uncertainties}")

