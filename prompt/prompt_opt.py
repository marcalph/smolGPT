import dspy

lm = dspy.LM("huggingface/HuggingFaceTB/SmolLM2-135M-Instruct")
print(lm("Say this is a test!!"))