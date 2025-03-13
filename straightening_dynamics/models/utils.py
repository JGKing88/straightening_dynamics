import math
import torch
import torch.nn as nn
import numpy as np
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class PaperNorm(nn.Module):
    def __init__(self, hidden_size, eps=None):
        super().__init__()
        self.hidden_size = hidden_size
    
    def forward(self, input):
        return math.sqrt(self.hidden_size) * input / torch.norm(input, dim=-1, keepdim=True)
    
def init_qk_weights(param, attention_sigma):
    variance = attention_sigma / param.shape[0]
    nn.init.normal_(param, 0, np.sqrt(variance))

def init_v_weights(param):
    nn.init.normal_(param, 0, np.sqrt(1/param.shape[0]))

def initialize_gpt2_qk_weights(model, attention_sigma):
    """
    Initialize the Q, K parts of GPT-2's attention matrices separately.
    
    Args:
        model: The GPT-2 model
    """
    for name, module in model.named_modules():
        if isinstance(module, GPT2Attention):
            # Get the c_attn weight matrix
            weight = module.c_attn.weight
            # Get the dimensions
            split_size = module.split_size  # This should be equal to hidden_size

            q_weight, k_weight, v_weight = weight.split(split_size, dim=1)
            
            # Initialize each part separately
            init_qk_weights(q_weight, attention_sigma)
            init_qk_weights(k_weight, attention_sigma)
            init_v_weights(v_weight)

def initialize_weights(model, distribution, attention_sigma, mlp_sigma):
    for name, param in model.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
        elif "v_proj" in name or "o_proj" in name or "c_proj" in name:
            nn.init.normal_(param, 0, np.sqrt(1/param.shape[0]))
        elif "layernorm" in name or "ln" in name:
            # Set weight (scale) to identity
            nn.init.ones_(param)  # LayerNorm typically uses ones initialization
        else:
            if "q_proj" in name or "k_proj" in name:
                variance = attention_sigma / param.shape[0]
            else:
                variance = mlp_sigma**2 / param.shape[0]

            if distribution == "uniform":
                width = np.sqrt(variance * 12) / 2
                nn.init.uniform_(param, -width, width)
            elif distribution == "normal":
                nn.init.normal_(param, 0, np.sqrt(variance))
    
    if "GPT2" in model.__class__.__name__:
        initialize_gpt2_qk_weights(model, attention_sigma)