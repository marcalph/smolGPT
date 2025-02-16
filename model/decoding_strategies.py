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





# --- Helper function used by non-beam strategies ---
def generate_step(input_ids, strategy, c=5, **kwargs):
    """
    For a given input sequence (input_ids), this function computes the next-token
    probability distribution and extracts the top-c alternatives. Then, according to
    the strategy ('greedy', 'top_k', or 'top_p'), it selects one token.
    
    Returns a dict with:
      - 'alternatives': list of (token_str, probability) for the top c tokens.
      - 'chosen_index': index of the chosen token (0..c-1).
      - 'chosen_token': the chosen token (string).
      - 'chosen_prob': probability of the chosen token.
    """
    with torch.no_grad():
        outputs = model(input_ids)
    # Get logits for the last token only.
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1).squeeze()  # shape: [vocab_size]
    
    # Obtain top-c tokens.
    topk_probs, topk_indices = torch.topk(probs, c)
    topk_probs = topk_probs.tolist()
    topk_indices = topk_indices.tolist()
    topk_tokens = [tokenizer.decode([idx]).strip() for idx in topk_indices]

    # --- Select token based on strategy ---
    if strategy == "greedy":
        chosen_index = 0
    elif strategy == "top_k":
        # Sample from the top-c alternatives
        top_tensor = torch.tensor(topk_probs)
        normalized = top_tensor / top_tensor.sum()
        chosen_index = int(torch.multinomial(normalized, num_samples=1).item())
    elif strategy == "top_p":
        p_threshold = kwargs.get("p", 0.9)
        # Sort full distribution descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_probs = sorted_probs.tolist()
        sorted_indices = sorted_indices.tolist()
        cum_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cum_probs, p_threshold) + 1
        nucleus_probs = sorted_probs[:cutoff]
        nucleus_indices = sorted_indices[:cutoff]
        nucleus_probs = np.array(nucleus_probs)
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        sampled = np.random.choice(len(nucleus_probs), p=nucleus_probs)
        chosen_token_candidate = tokenizer.decode([nucleus_indices[sampled]]).strip()
        # If the sampled token is among our top-c alternatives, use its index; else default to 0.
        if chosen_token_candidate in topk_tokens:
            chosen_index = topk_tokens.index(chosen_token_candidate)
        else:
            chosen_index = 0
    else:
        raise ValueError("Unknown strategy: choose 'greedy', 'top_k', or 'top_p'.")
    
    chosen_token = topk_tokens[chosen_index]
    chosen_prob = topk_probs[chosen_index]
    
    return {
        "alternatives": list(zip(topk_tokens, topk_probs)),
        "chosen_index": chosen_index,
        "chosen_token": chosen_token,
        "chosen_prob": chosen_prob,
    }

# --- General decoding for non-beam strategies ---
def decode_strategy(prompt, strategy, steps=5, c=5, **kwargs):
    """
    Decodes a completion for 'steps' tokens given a prompt and a strategy.
    At each step, the function calls generate_step to obtain the top-c alternatives
    and selects one token as specified by the strategy. It also tracks the cumulative
    probability (product of per-token probabilities) so far.
    
    Returns a list (one per step) with the detailed information for each token decision.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    steps_info = []
    cumulative_prob = 1.0
    for i in range(steps):
        step_info = generate_step(input_ids, strategy, c=c, **kwargs)
        cumulative_prob *= step_info["chosen_prob"]
        step_info["cumulative_prob"] = cumulative_prob
        steps_info.append(step_info)
        # Append chosen token to input_ids.
        chosen_token_ids = tokenizer.encode(step_info["chosen_token"], add_special_tokens=False)
        chosen_token_ids = torch.tensor(chosen_token_ids).unsqueeze(0)
        input_ids = torch.cat([input_ids, chosen_token_ids], dim=1)
    return steps_info

# --- Strategy-specific functions for greedy, top_k, and top_p ---
def greedy_decoding(prompt, steps=5, c=5):
    return decode_strategy(prompt, strategy="greedy", steps=steps, c=c)

def top_k_decoding(prompt, steps=5, c=5):
    return decode_strategy(prompt, strategy="top_k", steps=steps, c=c)

def top_p_decoding(prompt, steps=5, c=5, p=0.9):
    return decode_strategy(prompt, strategy="top_p", steps=steps, c=c, p=p)

# --- Proper Beam Search Implementation ---
def beam_search_decoding(prompt, steps=5, beam_width=5, c=5):
    """
    Implements beam search. At each step, each beam is expanded using the top-c alternatives.
    The beams are sorted by cumulative log probability, and the top beam_width beams are kept.
    For each candidate, we record the step information (like alternatives, chosen token, and
    cumulative probability) so that we can later visualize the chosen path.
    
    Returns the steps_info (a list of dicts, one per step) for the best beam.
    """
    # Each beam is a tuple: (input_ids, cumulative_log_prob, cumulative_prob, steps_info)
    init_ids = tokenizer.encode(prompt, return_tensors="pt")
    beams = [(init_ids, 0.0, 1.0, [])]  # log_prob=0, cum_prob=1
    for step in range(steps):
        candidates = []
        for input_ids, cum_log_prob, cum_prob, steps_info in beams:
            with torch.no_grad():
                outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1).squeeze()  # shape: [vocab_size]
            # Get top-c alternatives
            topk_probs, topk_indices = torch.topk(probs, c)
            topk_probs = topk_probs.tolist()
            topk_indices = topk_indices.tolist()
            topk_tokens = [tokenizer.decode([idx]).strip() for idx in topk_indices]
            # Prepare common alternatives info for this step.
            step_candidates_info = {"alternatives": list(zip(topk_tokens, topk_probs))}
            for i, (token, prob) in enumerate(zip(topk_tokens, topk_probs)):
                new_cum_log_prob = cum_log_prob + np.log(prob)
                new_cum_prob = cum_prob * prob
                # Record the chosen alternative for this step.
                new_step_info = dict(step_candidates_info)
                new_step_info["chosen_index"] = i
                new_step_info["chosen_token"] = token
                new_step_info["chosen_prob"] = prob
                new_step_info["cumulative_prob"] = new_cum_prob
                new_steps_info = steps_info + [new_step_info]
                token_id = tokenizer.encode(token, add_special_tokens=False)
                token_id = torch.tensor(token_id).unsqueeze(0)
                new_input_ids = torch.cat([input_ids, token_id], dim=1)
                candidates.append((new_input_ids, new_cum_log_prob, new_cum_prob, new_steps_info))
        # Keep only the top beam_width beams (by cumulative log probability).
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
    # Select the best beam (highest cumulative log probability)
    best_beam = max(beams, key=lambda x: x[1])
    return best_beam[3]

# --- Plotting Function with Plotly Sankey Diagram ---
def plot_sankey(steps_info, prompt, strategy_name, c=5):
    """
    Creates a Sankey diagram visualizing the token-by-token decoding process.
    - The first column is the prompt.
    - Each subsequent column corresponds to one token decision and shows c nodes
      (one per alternative) with the token and its probability.
    - The chosen alternative at each step (from the steps_info) is highlighted.
    - The links between columns carry information about the token probabilities.
    """
    num_steps = len(steps_info)
    node_labels = []
    node_colors = []
    # Column 0: prompt.
    node_labels.append(f"Prompt\n{prompt}")
    node_colors.append("lightgreen")
    
    # Record the starting node index for each column.
    col_node_start = [0]  # prompt is node 0.
    for step in range(num_steps):
        start_idx = 1 + step * c
        col_node_start.append(start_idx)
        for alt_index, (token, prob) in enumerate(steps_info[step]["alternatives"]):
            if alt_index == steps_info[step]["chosen_index"]:
                label = f"{token}\nP={prob:.3f}\nCum={steps_info[step]['cumulative_prob']:.3f}\n(CHOSEN)"
                color = "cornflowerblue"
            else:
                label = f"{token}\nP={prob:.3f}"
                color = "lightgrey"
            node_labels.append(label)
            node_colors.append(color)
    
    # Build the links between nodes.
    link_sources = []
    link_targets = []
    link_values = []
    link_colors = []
    link_labels = []
    
    # Link from prompt to step 1 alternatives.
    for i, (token, prob) in enumerate(steps_info[0]["alternatives"]):
        source = 0  # prompt
        target = col_node_start[1] + i
        link_sources.append(source)
        link_targets.append(target)
        value = prob * 100  # scale factor for visualization
        link_values.append(value)
        if i == steps_info[0]["chosen_index"]:
            link_colors.append("blue")
            link_labels.append(f"Chosen: {token}\n{prob:.3f}")
        else:
            link_colors.append("grey")
            link_labels.append(f"{token}\n{prob:.3f}")
    
    # For steps 2..s, link from the chosen node of the previous step.
    for step in range(1, num_steps):
        prev_start = col_node_start[step]
        chosen_prev = steps_info[step - 1]["chosen_index"]
        source = prev_start + chosen_prev
        current_start = col_node_start[step + 1]
        for i, (token, prob) in enumerate(steps_info[step]["alternatives"]):
            target = current_start + i
            link_sources.append(source)
            link_targets.append(target)
            value = prob * 100
            link_values.append(value)
            if i == steps_info[step]["chosen_index"]:
                link_colors.append("blue")
                link_labels.append(f"Chosen: {token}\n{prob:.3f}")
            else:
                link_colors.append("grey")
                link_labels.append(f"{token}\n{prob:.3f}")
    
    sankey_data = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values,
            color=link_colors,
            label=link_labels,
        ),
    )
    
    fig = go.Figure(data=[sankey_data])
    fig.update_layout(
        title_text=f"Sankey Diagram for {strategy_name} Decoding (Steps = {num_steps})",
        font_size=10,
    )
    fig.show()

# # --- Example Usage: Generate completions and plot ---
# prompt = "Once upon a time"
# steps = 5       # number of tokens to generate
# c = 5           # number of alternatives per step

# # Greedy Decoding
# greedy_info = greedy_decoding(prompt, steps=steps, c=c)
# plot_sankey(greedy_info, prompt, strategy_name="Greedy", c=c)

# # Top-K Sampling Decoding
# topk_info = top_k_decoding(prompt, steps=steps, c=c)
# plot_sankey(topk_info, prompt, strategy_name="Top-K", c=c)

# # Top-P (Nucleus) Sampling Decoding
# topp_info = top_p_decoding(prompt, steps=steps, c=c, p=0.9)
# plot_sankey(topp_info, prompt, strategy_name="Top-P (Nucleus)", c=c)

# # Beam Search Decoding (using a proper beam search)
# beam_info = beam_search_decoding(prompt, steps=steps, beam_width=5, c=c)
# plot_sankey(beam_info, prompt, strategy_name="Beam Search", c=c)
