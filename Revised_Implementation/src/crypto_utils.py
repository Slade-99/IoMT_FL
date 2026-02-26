import tenseal as ts
import numpy as np
import hashlib
import json
import time

# =============================
# CKKS Homomorphic Encryption (From your ckks.py)
# =============================
def create_ckks_context(poly_mod_degree=32768, coeff_mod_bit_sizes=[60, 40, 60], scale_pow=40):
    """Creates a high-precision CKKS context matching Table V specs."""
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = 2**scale_pow
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx

def encrypt_vector(ctx, vec):
    return ts.ckks_vector(ctx, vec)

def ct_pt_dot(ct_u, plaintext_vec):
    """Computes <Enc(u_hat), v_hat_plaintext> homomorphically."""
    # TenSEAL natively supports dot product between ciphertext and plaintext vector
    return ct_u.dot(plaintext_vec) 

def decrypt_scalar(ct):
    """Decrypts a scalar value (like the resulting Cosine Similarity)."""
    return float(ct.decrypt()[0])

# =============================
# Append-Only Immutable Log (For Eq 10)
# =============================
class HashLog:
    """
    Lightweight append-only hash chain. Replaces the blockchain overhead.
    """
    def __init__(self):
        self.log = []
        self.current_hash = hashlib.sha256(b"genesis_block").hexdigest()

    def append_event(self, round_num, sink_id, event_type, details):
        """Appends an event and cryptographically links it to the previous hash."""
        entry = {
            "round": round_num,
            "sink_id": sink_id,
            "event": event_type,
            "details": details,
            "prev_hash": self.current_hash,
            "timestamp": time.time()
        }
        entry_string = json.dumps(entry, sort_keys=True)
        self.current_hash = hashlib.sha256(entry_string.encode('utf-8')).hexdigest()
        self.log.append(entry)
        return self.current_hash

    def get_latest_hash(self):
        return self.current_hash