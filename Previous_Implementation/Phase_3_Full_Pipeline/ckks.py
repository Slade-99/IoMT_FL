"""
ckks_utils.py

Reusable CKKS utility functions for the FL-IoMT pipeline.

Implements:
- CKKS context creation
- Encryption helpers for vectors and scalars
- ct–pt dot product (matches final design)
- norm encryption + decryption
- scalar multiplication (ct * scalar)
- encrypted aggregation helpers
- size/serialization tools
- error calculation utility

Designed for your final configuration:
    vec_len = 10000
    N = 32768
    coeff_mod_bit_sizes = [60,40,60]
    scale = 2^40
"""

import numpy as np
import pickle
import gc
import time

# =============================
# Import TenSEAL
# =============================
try:
    import tenseal as ts
except ImportError:
    raise ImportError("TenSEAL not installed. Use: pip install tenseal")


# =============================
# Context Creation
# =============================

def create_ckks_context(
    poly_mod_degree=32768,
    coeff_mod_bit_sizes=[60, 40, 60],
    scale_pow=40,
):
    """
    Creates a CKKS context with the required keys.
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_mod_degree,
        -1,
        coeff_mod_bit_sizes
    )

    ctx.global_scale = 2**scale_pow
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


# =============================
# Encryption Helpers
# =============================

def encrypt_vector(ctx, vec):
    """Encrypt a real vector using CKKS."""
    return ts.ckks_vector(ctx, vec)


def encrypt_scalar(ctx, value, vec_len):
    """
    Encrypt scalar by placing it in slot 0
    and padding remaining slots with zero.
    """
    arr = [value] + [0.0]*(vec_len-1)
    return ts.ckks_vector(ctx, arr)


# =============================
# Decryption Helpers
# =============================

def decrypt_first_slot(ct):
    """
    Decrypt ciphertext and return the first value.
    """
    arr = ct.decrypt()
    return float(arr[0])


def decrypt_vector(ct, length=None):
    """
    Returns full decrypted vector.
    If length provided, truncate to that length.
    """
    arr = np.array(ct.decrypt(), dtype=float)
    return arr[:length] if length else arr


# =============================
# ct–pt Dot Product (Final Design)
# =============================

def ct_pt_dot(ct_u, plaintext_vec):
    """
    Computes <Enc(u_hat), v_hat_plaintext> homomorphically.

    Because plaintext multiplication is supported:
        Enc(u_hat * v_hat[i]) for each slot.

    Then rotate-and-sum to accumulate.
    """
    # element-wise multiplication: ct * pt
    prod = ct_u * plaintext_vec

    # rotate-and-sum
    L = len(plaintext_vec)
    step = 1
    acc = prod

    while step < L:
        acc = acc + acc.rotate(step)
        step *= 2

    return acc


# =============================
# Scalar Multiply (ct × plaintext scalar)
# =============================

def ct_scalar_mul(ct, scalar):
    """Encrypted vector × scalar."""
    return ct * scalar


# =============================
# Aggregation
# =============================

def aggregate_ciphertexts(ct_list):
    """
    Encrypted sum of all ciphertexts in the list.
    """
    if len(ct_list) == 0:
        return None
    acc = ct_list[0]
    for ct in ct_list[1:]:
        acc = acc + ct
    return acc


# =============================
# Serialization Size
# =============================

def ciphertext_size(ct):
    """Returns serialized size in bytes."""
    try:
        return len(ct.serialize())
    except Exception:
        return len(pickle.dumps(ct))


# =============================
# Utility for Error Measurement
# =============================

def compute_mae(x, y):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(x - y)))


def compute_max_err(x, y):
    """Maximum absolute error."""
    return float(np.max(np.abs(x - y)))


# =============================
# Timer Wrapper
# =============================

def timed(func, *args, **kwargs):
    """
    Runs func and reports (result, time_in_seconds).
    """
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    return res, t1 - t0
