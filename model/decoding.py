from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    checkpoint = "HuggingFaceTB/SmolLM2-135M"
    device = "mps" # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    inputs = tokenizer.encode("I have a dream", return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))
