import torch
import numpy as np

def extract_flat_weights(state_dict):
    """Flattens a PyTorch state_dict into a single 1D numpy array."""
    weights = []
    for tensor in state_dict.values():
        weights.append(tensor.cpu().numpy().flatten())
    return np.concatenate(weights)

def unflatten_weights(flat_weights, reference_state_dict):
    """Reconstructs a PyTorch state_dict from a 1D numpy array."""
    reconstructed = {}
    current_idx = 0
    for key, tensor in reference_state_dict.items():
        numel = tensor.numel()
        shape = tensor.shape
        weight_slice = flat_weights[current_idx:current_idx + numel]
        reconstructed[key] = torch.tensor(weight_slice).view(shape)
        current_idx += numel
    return reconstructed

def decompose_weights(flat_weights):
    """
    Decomposes the weight update into its scalar norm and normalized vector.
    Aligns with Methodology Step 2: Privacy-Preserving Verification.
    """
    norm = np.linalg.norm(flat_weights)
    if norm < 1e-8:
        return norm, flat_weights
    normalized_weights = flat_weights / norm
    return norm, normalized_weights

def calculate_cosine_similarity(u_hat, v_hat):
    """Calculates plaintext cosine similarity (dot product of normalized vectors)."""
    return np.dot(u_hat, v_hat)

def update_dynamic_score(current_score, is_aligned, penalty_factor=0.20, reward_factor=0.05):
    """
    Implements the Additive Increase / Additive Decrease (AIAD) logic from Equation 5.
    If similarity >= tau_sim, reward. Else, penalize.
    """
    if is_aligned:
        return min(1.0, current_score + reward_factor)
    else:
        return max(0.0, current_score - penalty_factor)

def calculate_base_score(cpu, max_cpu, ram, max_ram, bw, max_bw):
    """Implements Equation 3: Static Base Score calculation."""
    return (1/3) * ((cpu/max_cpu) + (ram/max_ram) + (bw/max_bw))