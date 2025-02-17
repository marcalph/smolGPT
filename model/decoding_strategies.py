import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import plotly.graph_objects as go
import urllib, json

model_name = "gpt2"  # or any other causal LM model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # set model to evaluation mode

prompt = "Once upon a time,"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
steps_info = []
cumulative_prob = 1.0
num_candidates = 5

with torch.no_grad():
    for step in range(30):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=0)
        topk_probs, topk_indices = torch.topk(next_token_probs, num_candidates)
        
        candidate_tokens = [tokenizer.decode([idx]) for idx in topk_indices.tolist()]
        candidate_probs = topk_probs.tolist()
        
        # For visualization, add transitions from prev_token to each candidate
        for token, prob in zip(candidate_tokens, candidate_probs):
            steps_info.append((input_ids, token, prob))
        
        chosen_idx = topk_indices[0].unsqueeze(0)
        chosen_token = tokenizer.decode(chosen_idx)

        input_ids = torch.cat([input_ids, chosen_idx.unsqueeze(0)], dim=-1)
        print(f"Step {step+1}: {tokenizer.decode(input_ids[0])}")
