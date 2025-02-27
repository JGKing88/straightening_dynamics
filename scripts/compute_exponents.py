import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


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




def compute_gradient_exponent(model, input_tokens, attention_mask=None):
    """
    Given a model and input_tokens (of shape [n, d]), compute the gradient exponent.
    The gradient exponent is defined as:
        lambda_g = (1/L) * log( ||∇_{X_0}(X_L⋅R)||^2 )
    where L is the number of layers.
    """
    # Ensure input_tokens is a torch tensor and add a batch dimension.
    if not torch.is_tensor(input_tokens):
        input_tokens = torch.tensor(input_tokens, dtype=torch.float)
    input_tokens = input_tokens.unsqueeze(0)  # shape: (1, n, d)

    # Enable gradient tracking on the inputs.
    input_tokens.requires_grad_()

    # Forward pass: using inputs_embeds to bypass the embedding lookup.
    outputs = model(inputs_embeds=input_tokens, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state  # shape: (1, n, hidden_dim)

    # Create a random probe tensor R with the same shape as the output.
    R = torch.randn_like(last_hidden_state)

    # Compute a scalar by dotting the output with R.
    scalar = (last_hidden_state * R).sum()

    # Compute gradient of the scalar with respect to the input tokens.
    grad = torch.autograd.grad(scalar, input_tokens, create_graph=True)[0]
    grad_norm_sq = grad.pow(2).sum()  # squared Frobenius norm

    # Number of layers in the model.
    L = model.config.n_layer

    # Compute per-layer gradient exponent.
    lambda_g = (1.0 / L) * torch.log(grad_norm_sq)

    return lambda_g.item(), grad_norm_sq.item()


# Parameters for generating custom tokens:
n_tokens = 10  # number of tokens in the sequence
embedding_dim = 768  # GPT-2 embedding dimension
correlation = 0.99  # desired off-diagonal correlation (p)

# Generate custom input tokens (each token is a d-dimensional vector).
custom_tokens = generate_input(n_tokens, embedding_dim, correlation)

# Create a dummy attention mask (all ones) for consistency.
attention_mask = torch.ones(1, n_tokens, dtype=torch.long)

# Load GPT-2 model and tokenizer.
model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Evaluate the gradient exponent on these custom tokens.
lambda_g, grad_norm_sq = compute_gradient_exponent(model, custom_tokens, attention_mask=attention_mask)
print("Gradient exponent (lambda_g):", lambda_g)
print("Gradient norm squared:", grad_norm_sq)
