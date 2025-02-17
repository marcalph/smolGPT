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
num_candidates = 3
with torch.no_grad():
    for step in range(30):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=0)
        topk_probs, topk_indices = torch.topk(next_token_probs, num_candidates)
        # cumulative_prob *= topk_probs.tolist()[0]
        # steps_info.append((step, cumulative_prob, tokenizer.decode([steps_info[0].item()])))
        # input_ids = torch.cat([input_ids, steps_info[0].unsqueeze(0)], dim=1)

print(tokenizer.decode(input_ids[0], skip_special_tokens=True))