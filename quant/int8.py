import torch


def absmax_quantize(X):
    scale = 127/torch.max(torch.abs(X))
    X_quant = torch.round(X*scale)
    X_dequant = X_quant/scale
    return X_quant.to(torch.int8), X_dequant




def zeropoint_quantize(X):
    if not (xrange := torch.max(X) - torch.min(X)):
        xrange=1
    
    scale = 255/xrange
    zero_point = (-scale * torch.min(X)-128).round()
    X_quant = torch.clip(torch.round(X*scale + zero_point), -128, 127)
    X_dequant = (X_quant - zero_point)/scale

    return X_quant.to(torch.int8), X_dequant



if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    device = 'cpu'
    model_name = 'gpt2'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Model size: {model.get_memory_footprint()//2**20:,} Mb")
    weights = model.transformer.h[0].attn.c_attn.weight.data
    print("Original weights:")
    print(weights)

    # Quantize layer using absmax quantization
    weights_abs_quant, _ = absmax_quantize(weights)
    print("\nAbsmax quantized weights:")
    print(weights_abs_quant)

    # Quantize layer using absmax quantization
    weights_zp_quant, _ = zeropoint_quantize(weights)
    print("\nZero-point quantized weights:")
    print(weights_zp_quant)
    print(f"Original weights shape: {weights.shape}")
    print(f"Quantized weights shape: {weights_zp_quant.shape}")
    print(f"change: {(weights_abs_quant-weights_zp_quant).float().mean()}")