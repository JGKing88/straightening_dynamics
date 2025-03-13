# import numpy as np
# import matplotlib.pyplot as plt
# from transformers import AutoConfig, AutoModel, AutoTokenizer
# from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# from scipy.stats import gaussian_kde
# import math
# import pickle
# from modular_transformers.straightening.straightening_utils import (
#     compute_model_activations,
#     compute_model_curvature,
# )
# from straightening_dynamics.models.utils import initialize_weights
# from straightening_dynamics.models.modeling_llama import LlamaForCausalLM

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# hf_auth_token = "" #REPLACE_WITH_YOUR_HUGGINGFACE_TOKEN


# def generate_input(n, d, correlation):
#     # Generate the "tokens" as described in the paper

#     # Step 1: Generate base vectors
#     # Generate n independent Gaussian vectors of dimension d
#     independent_vectors = np.random.normal(size=(n, d))

#     # Step 2: Create correlation matrix and get Cholesky decomposition
#     correlation_matrix = np.full((n, n), correlation)
#     np.fill_diagonal(correlation_matrix, 1.0)
#     L = np.linalg.cholesky(correlation_matrix)

#     # Step 3: Apply Cholesky factor to correlate vectors
#     # `correlated_vectors` now has the desired correlations across vectors
#     input = L @ independent_vectors  # Shape: (n, d)

#     return input


# def calculate_pq(input):
#     """
#     Calculate the average self dot product (q) and the average cross dot product (p)
#     for a set of token embeddings.

#     Parameters:
#         input (np.ndarray): A 2D array of shape (n_tokens, d) where each row represents
#                             a token embedding.

#     Returns:
#         tuple:
#             p (float): The average off-diagonal dot product, computed as the mean of all
#                        dot products between distinct tokens.
#             q (float): The average diagonal dot product, computed as the mean of the self
#                        dot products (i.e., the squared norms of the tokens).
#     """
#     # Compute dot product matrix
#     dot_product_matrix = input @ input.T

#     # Extract norms (diagonal) and average norm
#     q = np.mean(np.diag(dot_product_matrix))

#     # Extract dot products (off-diagonal) and average dot product
#     p = np.mean(dot_product_matrix[np.triu_indices(len(input), k=1)])

#     return p, q


# def calculate_lyapunov_exponent(p, q, p_prime, q_prime):
#     """
#     Calculate the angle Lyapunov exponent (λₐ) given the token geometry before and
#     after a transformer layer.

#     Parameters:
#         p (float): Average off-diagonal dot product of the input token embeddings.
#         q (float): Average diagonal dot product (self dot product) of the input token embeddings.
#         p_prime (float): Average off-diagonal dot product of the output token embeddings after a transformer layer.
#         q_prime (float): Average diagonal dot product of the output token embeddings after a transformer layer.

#     Returns:
#         float: The computed angle Lyapunov exponent, λₐ, defined as:
#                λₐ = log((1 - (p_prime / q_prime)) / (1 - (p / q)))

#     Description:
#         This function quantifies the rate at which the "angle" (or relative misalignment) between token representations
#         changes as they propagate through a transformer layer. A positive λₐ indicates that small differences between tokens
#         are amplified (a chaotic regime), while a negative λₐ indicates contraction (an ordered regime). The formula
#         compares the normalized deviation from perfect alignment before and after the layer.
#     """
#     numerator = 1 - p_prime / q_prime
#     denominator = 1 - p / q
#     return np.log(numerator / denominator)

# def compute_gradient_exponent(input, last_hidden_state, L):
#     """
#     Given a model and input_tokens (of shape [n, d]), compute the gradient exponent.
#     The gradient exponent is defined as:
#         lambda_g = (1/L) * log( ||∇_{X_0}(X_L⋅R)||^2 )
#     where L is the number of layers.
#     """
#     # Create a random probe tensor R with the same shape as the output.
#     R = torch.randn_like(last_hidden_state)

#     # Compute a scalar by dotting the output with R.
#     scalar = (last_hidden_state * R).sum()

#     # Compute gradient of the scalar with respect to the input tokens.
#     grad = torch.autograd.grad(scalar, input, create_graph=True)[0]
#     grad_norm_sq = grad.pow(2).sum()  # squared Frobenius norm

#     # Compute per-layer gradient exponent.
#     lambda_g = (1.0 / L) * torch.log(grad_norm_sq)

#     return lambda_g.item(), grad_norm_sq.item()

# def calculate_curvature(model, data, device):
#     activations = compute_model_activations(model, data, device)
#     curves = np.array(compute_model_curvature(activations)["all_layer_curve_all"])
#     return curves

# def generate_data(model_name):
#     #prepare data for calculating curvature
#     tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
#     CURVATURE_DATA_PATH = (
#         "/rdma/vast-rdma/vast/evlab/ehoseini/MyData/sent_sampling/analysis/straightening/"
#         "generation/sentences_ud_sentencez_token_filter_v3_textNoPeriod_cntx_3_cont_7.pkl"
#     )
#     with open(CURVATURE_DATA_PATH, "rb") as f:
#         raw_data = pickle.load(f)
#     sentences = tokenizer.batch_encode_plus(
#         raw_data, add_special_tokens=True, padding=False
#     )["input_ids"]
#     # Only keep sentences longer than 14 tokens and truncate to 14 tokens
#     data = [torch.tensor(s) for s in sentences if len(s) > 14]
#     data = torch.stack([s[:14] for s in data])
#     #sample 100 sentences
#     data = data[:100]
#     return data

# if __name__ == "__main__":

#     # Parameters for generating custom tokens:
#     n_tokens = 64  # number of tokens in the sequence
#     correlation = 0.99  # desired off-diagonal correlation (p)

#     # Load GPT-2 model and tokenizer.
#     for model_name in [ "gpt2-llama", "gpt2", "meta-llama/Llama-2-7b-hf", "gpt2-xl"]:
#         config = AutoConfig.from_pretrained(model_name, token=hf_auth_token)
#         activation_function = "tanh" #from paper (silu is llama default)
#         config.hidden_act = activation_function
#         if model_name == "gpt2-llama":
#             config.num_hidden_layers = 12
#             config.hidden_size = 768
#         config.use_rms_norm = False
#         config.use_position_encoding = False
#         embedding_dim = config.hidden_size  # embedding dimension
#         try:
#             num_layers = config.n_layer
#         except:
#             num_layers = config.num_hidden_layers
#         config.n_positions = n_tokens

#         model = AutoModel.from_config(config)

#         print(model)
#         print(model.config)

#         model.to(device)

#         data = generate_data(model_name)

#         # initialize variances
#         mlp_sigmas = np.arange(0, 20, 0.1)
#         attention_sigmas = np.arange(0, 20, 0.1)
#         lyapunov_exponents = np.zeros((len(mlp_sigmas), len(attention_sigmas)))
#         curves = np.zeros((len(mlp_sigmas), len(attention_sigmas)))
#         gradient_exponents = np.zeros((len(mlp_sigmas), len(attention_sigmas)))

#         # Generate custom input tokens (each token is a d-dimensional vector).
#         input = generate_input(n_tokens, embedding_dim, correlation)
#         p, q = calculate_pq(input)
#         input = torch.tensor(input, dtype=torch.float).to(device)
#         input = input.unsqueeze(0)
#         input.requires_grad_()

#         # Create a dummy attention mask (all ones) for consistency.
#         attention_mask = torch.ones(1, n_tokens, dtype=torch.long)

#         for mlp_sigma_idx, mlp_sigma in tqdm(enumerate(mlp_sigmas)):
#             for attention_sigma_idx, attention_sigma in enumerate(attention_sigmas):    
#                 initialize_weights(model, "normal", attention_sigma, mlp_sigma)
#                 outputs = model(inputs_embeds=input, attention_mask=attention_mask)
#                 last_hidden_state = outputs.last_hidden_state
#                 curvature = np.mean(calculate_curvature(model, data, device)) * 180 / np.pi
#                 curves[mlp_sigma_idx, attention_sigma_idx] = curvature

#                 gradient_exponent = compute_gradient_exponent(input, last_hidden_state, num_layers)[0]
#                 gradient_exponents[mlp_sigma_idx, attention_sigma_idx] = gradient_exponent

#                 p_prime, q_prime = calculate_pq(last_hidden_state.detach().cpu().numpy().squeeze())
#                 # Calculate Lyapunov exponent
#                 lyapunov_exponent = calculate_lyapunov_exponent(p, q, p_prime, q_prime)
#                 lyapunov_exponents[mlp_sigma_idx, attention_sigma_idx] = lyapunov_exponent

#         #save results
#         np.save(f"/om2/user/jackking/straightening_dynamics/results/{model_name}_lyapunov_exponents.npy", lyapunov_exponents)
#         np.save(f"/om2/user/jackking/straightening_dynamics/results/{model_name}_curves.npy", curves)
#         np.save(f"/om2/user/jackking/straightening_dynamics/results/{model_name}_gradient_exponents.npy", gradient_exponents)

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from tqdm import tqdm
import torch
import pickle
from modular_transformers.straightening.straightening_utils import (
    compute_model_activations,
    compute_model_curvature,
)
from straightening_dynamics.models.utils import initialize_weights
from straightening_dynamics.models.utils import PaperNorm
from straightening_dynamics.models.modeling_llama import LlamaForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hf_auth_token = "" #REPLACE_WITH_YOUR_HUGGINGFACE_TOKEN
import os


def generate_input(n, d, correlation):
    # Generate the "tokens" as described in the paper

    # Step 1: Generate base vectors
    # Generate n independent Gaussian vectors of dimension d
    independent_vectors = np.random.normal(size=(n, d))

    # Step 2: Create correlation matrix and get Cholesky decomposition
    correlation_matrix = np.full((n, n), correlation)
    np.fill_diagonal(correlation_matrix, 1.0)
    L = np.linalg.cholesky(correlation_matrix)

    # Step 3: Apply Cholesky factor to correlate vectors
    # `correlated_vectors` now has the desired correlations across vectors
    input = L @ independent_vectors  # Shape: (n, d)

    return input


def calculate_pq(input):
    """
    Calculate the average self dot product (q) and the average cross dot product (p)
    for a set of token embeddings.

    Parameters:
        input (np.ndarray): A 2D array of shape (n_tokens, d) where each row represents
                            a token embedding.

    Returns:
        tuple:
            p (float): The average off-diagonal dot product, computed as the mean of all
                       dot products between distinct tokens.
            q (float): The average diagonal dot product, computed as the mean of the self
                       dot products (i.e., the squared norms of the tokens).
    """
    # Compute dot product matrix
    dot_product_matrix = input @ input.T

    # Extract norms (diagonal) and average norm
    q = np.mean(np.diag(dot_product_matrix))

    # Extract dot products (off-diagonal) and average dot product
    p = np.mean(dot_product_matrix[np.triu_indices(len(input), k=1)])

    return p, q


def calculate_lyapunov_exponent(p, q, p_prime, q_prime):
    """
    Calculate the angle Lyapunov exponent (λₐ) given the token geometry before and
    after a transformer layer.

    Parameters:
        p (float): Average off-diagonal dot product of the input token embeddings.
        q (float): Average diagonal dot product (self dot product) of the input token embeddings.
        p_prime (float): Average off-diagonal dot product of the output token embeddings after a transformer layer.
        q_prime (float): Average diagonal dot product of the output token embeddings after a transformer layer.

    Returns:
        float: The computed angle Lyapunov exponent, λₐ, defined as:
               λₐ = log((1 - (p_prime / q_prime)) / (1 - (p / q)))

    Description:
        This function quantifies the rate at which the "angle" (or relative misalignment) between token representations
        changes as they propagate through a transformer layer. A positive λₐ indicates that small differences between tokens
        are amplified (a chaotic regime), while a negative λₐ indicates contraction (an ordered regime). The formula
        compares the normalized deviation from perfect alignment before and after the layer.
    """
    numerator = 1 - p_prime / q_prime
    denominator = 1 - p / q
    return np.log(numerator / denominator)

def compute_gradient_exponent(input, last_hidden_state, L):
    """
    Given a model and input_tokens (of shape [n, d]), compute the gradient exponent.
    The gradient exponent is defined as:
        lambda_g = (1/L) * log( ||∇_{X_0}(X_L⋅R)||^2 )
    where L is the number of layers.
    """
    # Create a random probe tensor R with the same shape as the output.
    R = torch.randn_like(last_hidden_state)

    # Compute a scalar by dotting the output with R.
    scalar = (last_hidden_state * R).sum()

    # Compute gradient of the scalar with respect to the input tokens.
    grad = torch.autograd.grad(scalar, input, create_graph=True)[0]
    grad_norm_sq = grad.pow(2).sum()  # squared Frobenius norm

    # Compute per-layer gradient exponent.
    lambda_g = (1.0 / L) * torch.log(grad_norm_sq)

    return lambda_g.item(), grad_norm_sq.item()

def calculate_curvature(model, data, device):
    activations = compute_model_activations(model, data, device)
    curves = np.array(compute_model_curvature(activations)["all_layer_curve_all"])
    return curves

def generate_data(model_name):
    #prepare data for calculating curvature
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
    CURVATURE_DATA_PATH = (
        "/rdma/vast-rdma/vast/evlab/ehoseini/MyData/sent_sampling/analysis/straightening/"
        "generation/sentences_ud_sentencez_token_filter_v3_textNoPeriod_cntx_3_cont_7.pkl"
    )
    with open(CURVATURE_DATA_PATH, "rb") as f:
        raw_data = pickle.load(f)
    sentences = tokenizer.batch_encode_plus(
        raw_data, add_special_tokens=True, padding=False
    )["input_ids"]
    # Only keep sentences longer than 14 tokens and truncate to 14 tokens
    data = [torch.tensor(s) for s in sentences if len(s) > 14]
    data = torch.stack([s[:14] for s in data])
    #sample 100 sentences
    data = data[:100]
    return data

def main(model, model_name, curvature_data):
    # initialize variances
    mlp_sigmas = np.concatenate([np.arange(0, 0.1, 0.01), np.arange(0.1, 4, 0.05), np.arange(4, 12, 0.25)])
    attention_sigmas = np.arange(0, 30, 0.5)
    lyapunov_exponents = np.zeros((len(mlp_sigmas), len(attention_sigmas)))
    curves = np.zeros((len(mlp_sigmas), len(attention_sigmas), num_layers+1, len(curvature_data[0])-2))
    gradient_exponents = np.zeros((len(mlp_sigmas), len(attention_sigmas)))
    sigmas = np.zeros((len(mlp_sigmas), len(attention_sigmas), 2))

    # Generate custom input tokens (each token is a d-dimensional vector).
    input = generate_input(n_tokens, embedding_dim, correlation)
    p, q = calculate_pq(input)
    input = torch.tensor(input, dtype=torch.float).to(device)
    input = input.unsqueeze(0)
    input.requires_grad_()
    # Create a dummy attention mask (all ones) for consistency.
    attention_mask = torch.ones(1, n_tokens, dtype=torch.long)

    for mlp_sigma_idx, mlp_sigma in tqdm(enumerate(mlp_sigmas)):
        for attention_sigma_idx, attention_sigma in enumerate(attention_sigmas):    
            initialize_weights(model, "normal", attention_sigma, mlp_sigma)
            outputs = model(inputs_embeds=input, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            curvature = np.mean(calculate_curvature(model, curvature_data, device), axis=0) * 180 / np.pi
            curves[mlp_sigma_idx, attention_sigma_idx] = curvature

            gradient_exponent = compute_gradient_exponent(input, last_hidden_state, num_layers)[0]
            gradient_exponents[mlp_sigma_idx, attention_sigma_idx] = gradient_exponent

            p_prime, q_prime = calculate_pq(last_hidden_state.detach().cpu().numpy().squeeze())
            # Calculate Lyapunov exponent
            lyapunov_exponent = calculate_lyapunov_exponent(p, q, p_prime, q_prime)
            lyapunov_exponents[mlp_sigma_idx, attention_sigma_idx] = lyapunov_exponent

            sigmas[mlp_sigma_idx, attention_sigma_idx, 0] = attention_sigma
            sigmas[mlp_sigma_idx, attention_sigma_idx, 1] = mlp_sigma

    return lyapunov_exponents, curves, gradient_exponents, sigmas

if __name__ == "__main__":

    # Parameters for generating custom tokens:
    n_tokens = 64  # number of tokens in the sequence
    correlation = 0.99  # desired off-diagonal correlation (p)
    activation_function = "relu" #tanh is in paper and silu is llama default

    # Load GPT-2 model and tokenizer.
    for use_paper_norm in [False]:
        for activation_function in ["relu"]:
            for use_position_encoding in [True]:
                for model_name in ["gpt2"]:
                    print(use_paper_norm, activation_function, use_position_encoding, model_name)

                    if "llama" in model_name:
                        hf_name = "meta-llama/Llama-2-7b-hf"
                    else:
                        hf_name = model_name
                        
                    config = AutoConfig.from_pretrained(hf_name, token=hf_auth_token)
                    config.hidden_act = activation_function
                    config.use_position_encoding = use_position_encoding
                    config.n_positions = n_tokens

                    #set hidden size and number of layers if not using base config
                    if model_name == "llama-small":
                        config.num_hidden_layers = 12
                        config.hidden_size = 768
                    embedding_dim = config.hidden_size

                    #if using paper norm, set rms norm to false for llama
                    if use_paper_norm and "llama" in model_name:
                        config.use_rms_norm = False
                    else:
                        config.use_rms_norm = True

                    #get number of layers
                    try:
                        num_layers = config.n_layer
                    except:
                        num_layers = config.num_hidden_layers
                    
                    #initialize model
                    if "llama" in model_name:
                        model = LlamaForCausalLM(config)
                    else:
                        model = GPT2LMHeadModel(config)

                    #change positional encoding in weights if not using position encoding and model is gpt2
                    if not config.use_position_encoding and "gpt2" in model_name:
                        #change model.wpe to be all zeroes
                        model.transformer.wpe.weight.data = torch.zeros_like(model.transformer.wpe.weight.data)
                    
                    #change ln1 and ln2 to paper norm if using paper norm and model is gpt2
                    if use_paper_norm and "gpt2" in model_name:
                        for layer in model.transformer.h:
                            layer.ln_1 = PaperNorm(layer.ln_1.weight.data.shape[0])
                            layer.ln_2 = PaperNorm(layer.ln_2.weight.data.shape[0])

                    model.to(device)
                    curvature_data = generate_data(hf_name)

                    lyapunov_exponents, curves, gradient_exponents, sigmas = main(model, model_name, curvature_data)
                    # save results
                    if not os.path.exists(f"/om2/user/jackking/straightening_dynamics/results/{activation_function}"):
                        os.makedirs(f"/om2/user/jackking/straightening_dynamics/results/{activation_function}")
                    np.save(f"/om2/user/jackking/straightening_dynamics/results/{activation_function}/{use_paper_norm}_{use_position_encoding}_{model_name}_lyapunov_exponents.npy", lyapunov_exponents)
                    np.save(f"/om2/user/jackking/straightening_dynamics/results/{activation_function}/{use_paper_norm}_{use_position_encoding}_{model_name}_curves.npy", curves)
                    np.save(f"/om2/user/jackking/straightening_dynamics/results/{activation_function}/{use_paper_norm}_{use_position_encoding}_{model_name}_gradient_exponents.npy", gradient_exponents)
                    np.save(f"/om2/user/jackking/straightening_dynamics/results/{activation_function}/{use_paper_norm}_{use_position_encoding}_{model_name}_sigmas.npy", sigmas)

                