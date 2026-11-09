import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_recipe")
model = GPT2LMHeadModel.from_pretrained("./gpt2_recipe")
model.eval()

# Define your input prompt.
prompt = "<recipe_generation> Can I have caramel macchiato with extra shot of espresso please? <recipe_generation>"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
'''
# Generate text while returning scores.
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    output_scores=True,
    return_dict_in_generate=True
)

# Extract the generated token IDs.
generated_ids = output.sequences[0]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("Generated Text:", generated_text)

generated = input_ids
past_key_values = None
'''
import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95):
    # Top-k filtering: keep only the top k tokens with the highest logit values.
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        kth_value = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < kth_value, torch.full_like(logits, float('-inf')), logits)
    
    # Top-p (nucleus) filtering: keep tokens with cumulative probability up to top_p.
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        # Remove tokens with cumulative probability above the threshold.
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to include the first token above the threshold.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # Scatter sorted tensors back to original indexing.
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    return logits


generated = input_ids  
past_key_values = None 

for _ in range(100):
    # Forward pass; utilize past_key_values to speed up computation if available.
    outputs = model(generated, past_key_values=past_key_values, return_dict=True)
    
    # Update past_key_values and extract logits for the last generated token.
    logits = outputs.logits
    past_key_values = outputs.past_key_values
    next_token_logits = logits[:, -1, :]
    
    # Apply top-k and top-p filtering to the logits.
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=0.95)
    
    # Convert filtered logits to probabilities.
    probabilities = F.softmax(filtered_logits, dim=-1)
    
    # Sample the next token (modify sampling strategy if needed).
    next_token = torch.multinomial(probabilities, num_samples=1)
    
    # Append the token to the generated sequence.
    generated = torch.cat((generated, next_token), dim=1)
    
    # Decode and print the token immediately.
    token_str = tokenizer.decode(next_token[0])
    print(token_str, end='', flush=True)
    
    # Optionally, break the loop if the EOS token is generated.
    if next_token.item() == model.config.eos_token_id:
        break

print()  # New line after generation.