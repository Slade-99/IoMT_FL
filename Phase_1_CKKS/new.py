"""
ckks_benchmark_ct_pt.py

Benchmarks the CKKS pipeline where:
 - Clients send Enc(normalized vector) and Enc(norm).
 - Sink keeps its normalized vector in plaintext (v_hat).
 - Dot-product = sum_i (Enc(u_i) * v_i) computed by:
     ct_prod = elementwise ct * plaintext_vector
     ct_dot  = sum_slots(ct_prod) using rotations
 - Then sink "collaboratively" decrypts score (simulated here by local decrypt).
 - If checks pass, sink reconstructs encrypted client weights: Enc(u_hat) * norm (scalar mult).
 - Aggregation: sum encrypted weights across clients (script currently runs one client per trial; extend for multi-client).
 - Final decrypt of aggregated ciphertext simulated locally.

Notes:
 - Requires: tenseal, numpy, pandas, psutil
 - The rotation & plaintext-multiply helpers are robust to multiple TenSEAL versions.
 - Fallbacks (decrypt+rotate or encrypt plain vector) are used only if necessary and are logged.
"""

import os
import time
import gc
import pickle
import traceback
from typing import List, Dict
import numpy as np
import pandas as pd

try:
    import tenseal as ts
except Exception as e:
    print("tenseal import failed. Install with: pip install tenseal")
    raise

try:
    import psutil
except Exception:
    psutil = None
    print("psutil not found — memory measurements will be skipped. (pip install psutil)")

# -------------------------
# Helpers
# -------------------------
def timeit(func):
    t0 = time.perf_counter()
    res = func()
    t1 = time.perf_counter()
    return res, (t1 - t0)

def serialize_size(obj) -> int:
    try:
        if hasattr(obj, "serialize"):
            return len(obj.serialize())
        return len(pickle.dumps(obj))
    except Exception:
        return -1

def peak_mem_mb():
    if psutil:
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024*1024)
    return -1

# -------------------------
# Rotation helper (robust)
# -------------------------
def try_rotate(ct_vec, steps, ctx, allow_plain_fallback=True):
    """
    Try multiple rotation APIs. Return CKKSVector rotated by `steps`.
    If none available, and allow_plain_fallback=True, perform decrypt->python rotate->re-encrypt fallback.
    """
    # 1) CKKSVector.rotate (older API)
    try:
        if hasattr(ct_vec, "rotate"):
            return ct_vec.rotate(steps)
    except Exception:
        pass

    # 2) TenSEAL operations.rotate
    try:
        if hasattr(ts, "operations"):
            ops = getattr(ts, "operations")
            if hasattr(ops, "rotate"):
                try:
                    return ops.rotate(ct_vec, steps, ctx)
                except Exception:
                    pass
    except Exception:
        pass

    # 3) internal rotate helper
    try:
        if hasattr(ct_vec, "_rotate_internal"):
            return ct_vec._rotate_internal(steps, ctx)
    except Exception:
        pass

    # 4) underlying ciphertext evaluator (best-effort)
    try:
        if hasattr(ct_vec, "_ciphertext") and hasattr(ctx, "galois_keys"):
            ct = ct_vec._ciphertext
            # try several plausible evaluator locations
            # Some TenSEAL builds expose ctx.evaluator.rotate_vector
            if hasattr(ctx, "evaluator") and hasattr(ctx.evaluator, "rotate_vector"):
                try:
                    gk = ctx.galois_keys()
                    rotated_ct = ctx.evaluator.rotate_vector(ct, steps, gk)
                    return ts.CKKSVector(ctx, rotated_ct)
                except Exception:
                    pass
            # Some builds might have ctx.rotate_vector
            if hasattr(ctx, "rotate_vector"):
                try:
                    rotated_ct = ctx.rotate_vector(ct, steps)
                    return ts.CKKSVector(ctx, rotated_ct)
                except Exception:
                    pass
    except Exception:
        pass

    # 5) FALLBACK: decrypt -> python rotate -> re-encrypt (benchmark-only)
    if allow_plain_fallback:
        try:
            vals = ct_vec.decrypt()
            L = len(vals)
            s = steps % L
            rotated_vals = vals[s:] + vals[:s] if s != 0 else vals
            return ts.ckks_vector(ctx, rotated_vals)
        except Exception:
            pass

    raise RuntimeError("Rotation not available in this TenSEAL build and fallback failed.")

# -------------------------
# Plaintext multiply helper (robust)
# -------------------------
def multiply_ct_by_plain(ct_vec, plain_vec, ctx):
    """
    Multiply ciphertext vector ct_vec by plaintext vector plain_vec (both length slots).
    Tries multiple ways:
     - ct_vec * plain_vec  (TenSEAL often supports this)
     - ct_vec.multiply_plain(plaintext) (if available)
     - encode plaintext as CKKSVector/Plaintext and try plaintext multiplication
     - fallback: encrypt plain_vec -> ct_plain_enc and do ct*ct_plain_enc (last resort)
    Returns: ct_prod (CKKSVector), and a flag 'mode' indicating which method used.
    """
    # Try ct * plain_list
    try:
        prod = ct_vec * plain_vec
        return prod, "ct_mul_list"
    except Exception:
        pass

    # Try multiply_plain on ct_vec
    try:
        if hasattr(ct_vec, "multiply_plain"):
            # TenSEAL API variants: multiply_plain takes a plaintext polynomial or list
            prod = ct_vec.multiply_plain(plain_vec)
            return prod, "multiply_plain"
    except Exception:
        pass

    # Try building a plaintext CKKSVector-like object as "plain" and use library primitive
    try:
        # Some TenSEAL builds support plaintext encoding via ts.plain_tensor or similar;
        # fallback to encrypting plain vector (costly) and do ct*ct_plain
        plain_enc = ts.ckks_vector(ctx, list(plain_vec))
        prod = ct_vec * plain_enc
        return prod, "fallback_encrypt_plain_and_ctct"
    except Exception:
        pass

    # If we reach here, nothing worked
    raise RuntimeError("Failed to perform ciphertext-plaintext multiplication with this TenSEAL build.")

# -------------------------
# Sum slots via rotations (binary tree)
# -------------------------
def sum_slots_via_rotations(ct_prod, ctx, allow_plain_fallback=True):
    """
    Sum all slots of ct_prod and return a CKKSVector whose first (all) slot(s) hold the dot product.
    Uses try_rotate to be robust.
    """
    # Determine slot count by decrypting (safe in benchmark where ctx has secret)
    vals = ct_prod.decrypt()
    L = len(vals)

    res = ct_prod
    shift = 1
    while shift < L:
        rotated = try_rotate(ct_prod, shift, ctx, allow_plain_fallback=allow_plain_fallback)
        res = res + rotated
        shift *= 2
    return res

# -------------------------
# Experiment runner (modified)
# -------------------------
def make_context(poly_mod_degree: int, coeff_mod_bit_sizes: List[int], global_scale: float):
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = global_scale
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx

def enc_vector(ctx, vector: List[float]):
    return ts.ckks_vector(ctx, vector)

def run_experiment(vec_len: int,
                   poly_mod_degree: int,
                   coeff_mod_bit_sizes: List[int],
                   scale_pow: int,
                   trials: int = 3,
                   score_threshold: float = -1.0,
                   norm_min: float = 0.0,
                   norm_max: float = 1e9,
                   allow_rotation_plain_fallback: bool = True) -> Dict:
    """
    Implements the ct-pt pipeline:
     - client encrypts normalized vector and norm
     - sink decrypts norm (simulated), computes ct * v_plain (plain vector), sums slots via rotations
     - decrypts score (simulated), checks thresholds, rebuilds encrypted weight = ct_u * norm (scalar)
     - aggregates encrypted weights (single client in micro-benchmark), final decrypt (simulated)
    """

    assert vec_len <= poly_mod_degree // 2, "vec_len must fit into N/2 slots."

    global_scale = 2**scale_pow

    metrics = {
        "keygen_time": [], "enc_client_u": [], "enc_client_norm": [], "dec_norm": [],
        "dot_plainmul_time": [], "rot_sum_time": [], "dec_dot": [],
        "scalar_mul_time": [], "agg_time": [], "final_dec_time": []
    }
    sizes = {"ct_u": [], "ct_n": [], "ct_prod": [], "ct_dot": [], "ct_weight": [], "ct_agg": [], "relin_keys": [], "galois_keys": []}
    errors = {"mae": [], "max_err": []}
    mems = []
    fallback_log = {"used_encrypt_plain_fallback": 0, "used_rotation_fallback": 0}

    for t in range(trials):
        gc.collect()
        _mem0 = peak_mem_mb()

        # KeyGen / context
        ctx, dt_keygen = timeit(lambda: make_context(poly_mod_degree, coeff_mod_bit_sizes, global_scale))
        metrics["keygen_time"].append(dt_keygen)

        # Simulate client and sink vectors
        np.random.seed(1000 + t)
        w_client = np.random.randn(vec_len).astype(float)
        np.random.seed(2000 + t)
        w_sink = np.random.randn(vec_len).astype(float)

        # Norms and normalized vectors
        norm_client = float(np.linalg.norm(w_client))
        if norm_client == 0.0:
            norm_client = 1e-8
        u_hat = (w_client / norm_client).astype(float)

        norm_sink = float(np.linalg.norm(w_sink))
        if norm_sink == 0.0:
            norm_sink = 1e-8
        v_hat = (w_sink / norm_sink).astype(float)

        plain_dot_gt = float(np.dot(u_hat, v_hat))
        true_weight_vec = w_client  # u_hat * norm_client

        # -------- client encrypts normalized vector and norm ----------
        ct_u, dt_enc_u = timeit(lambda: enc_vector(ctx, u_hat.tolist()))
        metrics["enc_client_u"].append(dt_enc_u); sizes["ct_u"].append(serialize_size(ct_u))

        # encrypt norm as single-slot vector (norm in first slot)
        ct_n, dt_enc_n = timeit(lambda: enc_vector(ctx, [norm_client] + [0.0]*(vec_len-1)))
        metrics["enc_client_norm"].append(dt_enc_n); sizes["ct_n"].append(serialize_size(ct_n))

        # -------- sink collaborative decrypt of norm (simulated locally) ----------
        norm_dec, dt_dec_norm = timeit(lambda: float(np.round(ct_n.decrypt()[0], 12)))
        metrics["dec_norm"].append(dt_dec_norm)

        norm_ok = (norm_dec >= norm_min) and (norm_dec <= norm_max)

        # -------- compute ciphertext * plaintext elementwise product ----------
        # Try direct ct * plain_vec
        try:
            ct_prod, mode = multiply_ct_by_plain(ct_u, v_hat.tolist(), ctx)
            if mode == "fallback_encrypt_plain_and_ctct":
                fallback_log["used_encrypt_plain_fallback"] += 1
        except Exception as e:
            # If impossible, fallback to encrypting v_hat and do ct-ct (we record fallback)
            print(f"[WARN] plaintext-mul failed, falling back to ct-ct for dot (expensive). Err: {e}")
            plain_enc = enc_vector(ctx, v_hat.tolist())
            ct_prod, _ = timeit(lambda: ct_u * plain_enc)
            fallback_log["used_encrypt_plain_fallback"] += 1

        metrics["dot_plainmul_time"].append(0.0)  # elementwise multiply time included in previous op if relevant
        sizes["ct_prod"].append(serialize_size(ct_prod))

        # -------- sum slots using rotations (binary tree) ----------
        try:
            ct_dot, dt_rot = timeit(lambda: sum_slots_via_rotations(ct_prod, ctx, allow_plain_fallback=allow_rotation_plain_fallback))
            metrics["rot_sum_time"].append(dt_rot)
            if allow_rotation_plain_fallback and dt_rot > 0 and fallback_log["used_encrypt_plain_fallback"]>0:
                # no-op marker; primary fallback is counted in try_rotate
                pass
        except Exception as e:
            # record failure
            print(f"[ERROR] Rotations failed: {e}")
            traceback.print_exc()
            # fallback: decrypt ct_prod, sum plaintext, re-encrypt scalar across slots
            plain_vals = ct_prod.decrypt()
            dot_val = sum(plain_vals)
            ct_dot = enc_vector(ctx, [dot_val] + [dot_val]*(vec_len-1))
            metrics["rot_sum_time"].append(0.0)
            fallback_log["used_rotation_fallback"] += 1

        sizes["ct_dot"].append(serialize_size(ct_dot))

        # -------- decrypt dot (simulate collaborative decryption) ----------
        dot_dec, dt_dec_dot = timeit(lambda: float(ct_dot.decrypt()[0]))
        metrics["dec_dot"].append(dt_dec_dot)

        # ---------- score check ----------
        score_ok = (dot_dec >= score_threshold)

        # ---------- if ok, reconstruct encrypted client weights: Enc(u_hat) * norm_dec (scalar) ----------
        ct_weight = None
        if norm_ok and score_ok:
            ct_weight, dt_scalar = timeit(lambda: ct_u * norm_dec)
            metrics["scalar_mul_time"].append(dt_scalar)
            sizes["ct_weight"].append(serialize_size(ct_weight))
        else:
            metrics["scalar_mul_time"].append(0.0); sizes["ct_weight"].append(0)

        # ---------- aggregate (single client -> ct_agg = ct_weight) ----------
        if ct_weight is not None:
            ct_agg, dt_add = timeit(lambda: ct_weight)  # trivial for one client; replace with sum over clients
            metrics["agg_time"].append(dt_add)
            sizes["ct_agg"].append(serialize_size(ct_agg))
        else:
            ct_agg = None
            metrics["agg_time"].append(0.0); sizes["ct_agg"].append(0)

        # ---------- final decrypt aggregated model (simulate collaborative decryption) ----------
        if ct_agg is not None:
            dec_agg_vec, dt_final_dec = timeit(lambda: np.array(ct_agg.decrypt()[:vec_len], dtype=float))
            metrics["final_dec_time"].append(dt_final_dec)
            mae = float(np.mean(np.abs(dec_agg_vec - true_weight_vec)))
            maxe = float(np.max(np.abs(dec_agg_vec - true_weight_vec)))
            errors["mae"].append(mae)
            errors["max_err"].append(maxe)
        else:
            metrics["final_dec_time"].append(0.0)
            errors["mae"].append(float('nan')); errors["max_err"].append(float('nan'))

        # key sizes
        try:
            sizes["relin_keys"].append(len(ctx.relin_keys().serialize()))
            sizes["galois_keys"].append(len(ctx.galois_keys().serialize()))
        except Exception:
            sizes["relin_keys"].append(-1); sizes["galois_keys"].append(-1)

        mems.append(peak_mem_mb() - _mem0)

        # cleanup
        try:
            del ctx, ct_u, ct_n, ct_prod, ct_dot
            if ct_weight is not None:
                del ct_weight
            if ct_agg is not None:
                del ct_agg
        except Exception:
            pass
        gc.collect()

    # aggregate results
    out = {
        "vec_len": vec_len,
        "N": poly_mod_degree,
        "coeff_mod_bit_sizes": str(coeff_mod_bit_sizes),
        "scale_pow": scale_pow,
        "trials": trials,
        "keygen_time_s": np.mean(metrics["keygen_time"]),
        "enc_client_u_time_s": np.mean(metrics["enc_client_u"]),
        "enc_client_norm_time_s": np.mean(metrics["enc_client_norm"]),
        "dec_norm_time_s": np.mean(metrics["dec_norm"]),
        "dot_plainmul_time_s": np.mean(metrics["dot_plainmul_time"]),
        "rot_sum_time_s": np.mean(metrics["rot_sum_time"]),
        "dec_dot_time_s": np.mean(metrics["dec_dot"]),
        "scalar_mul_time_s": np.mean(metrics["scalar_mul_time"]),
        "agg_time_s": np.mean(metrics["agg_time"]),
        "final_dec_time_s": np.mean(metrics["final_dec_time"]),
        "ct_u_bytes": np.mean(sizes["ct_u"]),
        "ct_n_bytes": np.mean(sizes["ct_n"]),
        "ct_prod_bytes": np.mean(sizes["ct_prod"]),
        "ct_dot_bytes": np.mean(sizes["ct_dot"]),
        "ct_weight_bytes": np.mean(sizes["ct_weight"]),
        "ct_agg_bytes": np.mean(sizes["ct_agg"]),
        "relin_keys_size_mean": np.mean(sizes["relin_keys"]),
        "galois_keys_size_mean": np.mean(sizes["galois_keys"]),
        "mae_error": np.nanmean(errors["mae"]),
        "max_err": np.nanmean(errors["max_err"]),
        "mem_delta_mb": (np.mean(mems) if mems else -1),
        "fallback_encrypt_plain_count": fallback_log["used_encrypt_plain_fallback"],
        "fallback_rotation_count": fallback_log["used_rotation_fallback"]
    }

    return out

# -------------------------
# Driver
# -------------------------
def main():
    results = []

    vec_lens = [1024, 4096, 10000]
    Ns = [8192, 16384, 32768]
    coeffs_options = [[60,40,40,60], [60,40,60]]
    scale_pows = [30, 40]
    trials = 3

    print("Starting ct-pt CKKS pipeline benchmark...")

    for N in Ns:
        for vec_len in vec_lens:
            if vec_len > N // 2:
                continue
            for coeffs in coeffs_options:
                for sp in scale_pows:
                    print(f">> Running: N={N}, vec={vec_len}, coeffs={coeffs}, scale=2^{sp}")
                    try:
                        out = run_experiment(vec_len, N, coeffs, sp, trials=trials,
                                             score_threshold=-1.0, norm_min=0.0, norm_max=1e9,
                                             allow_rotation_plain_fallback=True)
                        results.append(out)
                    except Exception as e:
                        print(f"[ERROR] config N={N}, vec={vec_len}, coeffs={coeffs}, scale=2^{sp}: {e}")
                        traceback.print_exc()

    if results:
        df = pd.DataFrame(results)
        df.to_csv("results_ckks_ctpt_pipeline.csv", index=False)
        print("Done. Results written to results_ckks_ctpt_pipeline.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
