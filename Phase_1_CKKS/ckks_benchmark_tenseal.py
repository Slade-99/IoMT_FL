"""
ckks_benchmark_tenseal_pipeline.py

Revised benchmark implementing the pipeline:
- Clients send Enc(normalized weights) and Enc(norm).
- Sink (simulated locally) "collaboratively" decrypts norm and score (here: local decrypt).
- Sink computes dot between its plaintext normalized weights and client's encrypted normalized weights (we encrypt sink vector here to use existing APIs).
- If checks pass: sink computes Enc(original_weights) = Enc(normalized_weights) * norm (scalar mult), sums encrypted weights, and finally "collaboratively" decrypts the final averaged model.

Requirements: tenseal, numpy, pandas, psutil
"""
import time
import os
import gc
import pickle
from typing import List, Dict
import numpy as np
import pandas as pd

try:
    import tenseal as ts
except ImportError:
    print("tenseal import failed. Install with: pip install tenseal")
    raise

try:
    import psutil
except ImportError:
    psutil = None
    print("psutil not found — memory measurements will be skipped. (pip install psutil)")


def timeit(func):
    """Timer returning (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    res = func()
    t1 = time.perf_counter()
    return res, (t1 - t0)

def serialize_size(obj) -> int:
    """Return bytes size of serialized TenSEAL object"""
    try:
        if hasattr(obj, "serialize"):
            b = obj.serialize()
            return len(b)
        else:
            return len(pickle.dumps(obj))
    except Exception:
        return -1

def peak_mem_mb():
    if psutil:
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024*1024)
    return -1

def make_context(poly_mod_degree: int, coeff_mod_bit_sizes: List[int], global_scale: float):
    # TenSEAL API: create CKKS context & keys
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = global_scale
    # generate keys required for operations (Galois for rotations, relin for mults)
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx

def enc_vector(ctx, vector: List[float]):
    # encode + encrypt
    return ts.ckks_vector(ctx, vector)

def run_experiment(vec_len: int,
                   poly_mod_degree: int,
                   coeff_mod_bit_sizes: List[int],
                   scale_pow: int,
                   trials: int = 3,
                   score_threshold: float = -1.0,
                   norm_min: float = 0.0,
                   norm_max: float = 1e9) -> Dict:
    """
    This implements the pipeline per trial:
    - client creates w, norm, u_hat and sends ct_u and ct_n
    - sink decrypts norm (simulated), encrypts its own v_hat, computes dot, decrypts score (simulated)
    - if score & norm pass thresholds, sink forms encrypted client weights = ct_u * norm
    - sink aggregates encrypted weights and finally decrypts averaged model (simulated)
    """

    assert vec_len <= poly_mod_degree // 2, "vec_len must fit into N/2 slots."

    global_scale = 2**scale_pow

    metrics = {
        "keygen": [], "enc_client_u": [], "enc_client_norm": [], "dec_norm": [],
        "enc_sink_v": [], "dot_time": [], "dec_dot": [], "scalar_mul": [],
        "add_time": [], "final_dec": []
    }
    sizes = {"ct_u": [], "ct_n": [], "ct_dot": [], "ct_weight": [], "ct_agg": [], "relin_keys": [], "galois_keys": []}
    errors = {"mae": [], "max_err": []}
    mems = []

    for t in range(trials):
        gc.collect()
        _mem0 = peak_mem_mb()

        # 1) KeyGen / Context
        def do_keygen():
            return make_context(poly_mod_degree, coeff_mod_bit_sizes, global_scale)
        ctx, dt_keygen = timeit(do_keygen)
        # We'll treat keygen as part of setup but not include in per-step metrics here
        # metrics["keygen"].append(dt_keygen)

        # 2) Simulate client and sink model vectors (random)
        np.random.seed(1000 + t)
        w_client = np.random.randn(vec_len).astype(float)
        # Sink's local model vector
        np.random.seed(2000 + t)
        w_sink = np.random.randn(vec_len).astype(float)

        # compute norms and normalized vectors
        norm_client = float(np.linalg.norm(w_client))
        if norm_client == 0.0:
            norm_client = 1e-8
        u_hat = (w_client / norm_client).astype(float)

        norm_sink = float(np.linalg.norm(w_sink))
        if norm_sink == 0.0:
            norm_sink = 1e-8
        v_hat = (w_sink / norm_sink).astype(float)

        # For ground truth checks
        plain_dot_gt = float(np.dot(u_hat, v_hat))
        # original client weights for MAE after final decrypt
        true_weight_vec = w_client  # = u_hat * norm_client

        # ---------- CLIENT: encrypt normalized weights and norm ----------
        def do_enc_client_u():
            return enc_vector(ctx, u_hat.tolist())
        ct_u, dt_enc_u = timeit(do_enc_client_u)
        metrics["enc_client_u"].append(dt_enc_u)
        sizes["ct_u"].append(serialize_size(ct_u))

        def do_enc_client_norm():
            # encrypt scalar as single-slot vector (we still use ckks_vector for simplicity)
            return enc_vector(ctx, [norm_client] + [0.0]*(vec_len-1))
        ct_n, dt_enc_n = timeit(do_enc_client_norm)
        metrics["enc_client_norm"].append(dt_enc_n)
        sizes["ct_n"].append(serialize_size(ct_n))

        # ---------- SINK: decrypt norm (simulate multi-party decryption) ----------
        def do_dec_norm():
            # In practice: collaborative threshold decryption -> here: local decrypt simulation
            plain = ct_n.decrypt()
            # first slot holds the norm
            return float(np.round(plain[0], 12))  # round for numerical stability in sim
        norm_dec, dt_dec_norm = timeit(do_dec_norm)
        metrics["dec_norm"].append(dt_dec_norm)

        # perform norm check
        norm_ok = (norm_dec >= norm_min) and (norm_dec <= norm_max)

        # ---------- SINK: prepare its own plaintext normalized vector (encrypt for homomorphic ops) ----------
        def do_enc_sink_v():
            # sink encrypts its plaintext normalized vector (in real system may use plaintext-mul APIs)
            return enc_vector(ctx, v_hat.tolist())
        ct_v_enc, dt_enc_v = timeit(do_enc_sink_v)
        metrics["enc_sink_v"].append(dt_enc_v)

        # ---------- Homomorphic dot (between ct_u and ct_v_enc) ----------
        def do_dot():
            # TenSEAL supports ct_u.dot(ct_v_enc)
            return ct_u.dot(ct_v_enc)
        ct_dot, dt_dot = timeit(do_dot)
        metrics["dot_time"].append(dt_dot)
        sizes["ct_dot"].append(serialize_size(ct_dot))

        # ---------- decrypt dot (simulate collaborative decryption) ----------
        def do_dec_dot():
            dec = ct_dot.decrypt()
            return float(dec[0])
        dot_dec, dt_dec_dot = timeit(do_dec_dot)
        metrics["dec_dot"].append(dt_dec_dot)

        # Check score threshold
        score_ok = (dot_dec >= score_threshold)

        # ---------- If both checks pass, form encrypted client weights (ct_u * norm_dec) ----------
        ct_weight = None
        dt_scalar_mul = 0.0
        if norm_ok and score_ok:
            def do_scalar_mul():
                # scalar multiply ciphertext by plaintext scalar
                return ct_u * norm_dec
            ct_weight, dt_scalar_mul = timeit(do_scalar_mul)
            metrics["scalar_mul"].append(dt_scalar_mul)
            sizes["ct_weight"].append(serialize_size(ct_weight))
        else:
            # record zeros to keep lists aligned
            metrics["scalar_mul"].append(0.0)
            sizes["ct_weight"].append(0)

        # ---------- Aggregate encrypted weights (only one client in this micro-benchmark) ----------
        # For realistic multi-client aggregation, sum ct_weight across clients
        if ct_weight is not None:
            def do_add():
                # here only one client => aggregate equals the client's encrypted weight
                return ct_weight
            ct_agg, dt_add = timeit(do_add)
            metrics["add_time"].append(dt_add)
            sizes["ct_agg"].append(serialize_size(ct_agg))
        else:
            ct_agg = None
            metrics["add_time"].append(0.0)
            sizes["ct_agg"].append(0)

        # ---------- Final decrypt aggregated model (simulate collaborative decryption) ----------
        if ct_agg is not None:
            def do_final_dec():
                dec = ct_agg.decrypt()
                # first vec_len slots contain the decrypted aggregated vector scaled; decode done by TenSEAL
                return np.array(dec[:vec_len], dtype=float)
            dec_agg_vec, dt_final_dec = timeit(do_final_dec)
            metrics["final_dec"].append(dt_final_dec)
            # convert back to float vector and compute MAE vs true weight vector
            # dec_agg_vec should approximately equal true_weight_vec
            # Note: TenSEAL's decode handles scale; we compare directly
            mae = float(np.mean(np.abs(dec_agg_vec - true_weight_vec)))
            max_err = float(np.max(np.abs(dec_agg_vec - true_weight_vec)))
            errors["mae"].append(mae)
            errors["max_err"].append(max_err)
        else:
            metrics["final_dec"].append(0.0)
            errors["mae"].append(float('nan'))
            errors["max_err"].append(float('nan'))

        # record key sizes
        try:
            sizes["relin_keys"].append(len(ctx.relin_keys().serialize()))
            sizes["galois_keys"].append(len(ctx.galois_keys().serialize()))
        except Exception:
            sizes["relin_keys"].append(-1)
            sizes["galois_keys"].append(-1)

        mems.append(peak_mem_mb() - _mem0)

        # Cleanup current trial objects
        try:
            del ctx, ct_u, ct_n, ct_v_enc, ct_dot
            if ct_weight is not None:
                del ct_weight
            if ct_agg is not None:
                del ct_agg
        except Exception:
            pass
        gc.collect()

    # aggregate outputs
    out = {
        "vec_len": vec_len,
        "N": poly_mod_degree,
        "coeff_mod_bit_sizes": str(coeff_mod_bit_sizes),
        "scale_pow": scale_pow,
        "trials": trials,
        "enc_client_u_time_s": np.mean(metrics["enc_client_u"]),
        "enc_client_norm_time_s": np.mean(metrics["enc_client_norm"]),
        "dec_norm_time_s": np.mean(metrics["dec_norm"]),
        "enc_sink_v_time_s": np.mean(metrics["enc_sink_v"]),
        "dot_time_s": np.mean(metrics["dot_time"]),
        "dec_dot_time_s": np.mean(metrics["dec_dot"]),
        "scalar_mul_time_s": np.mean(metrics["scalar_mul"]),
        "add_time_s": np.mean(metrics["add_time"]),
        "final_dec_time_s": np.mean(metrics["final_dec"]),
        "ct_u_bytes": np.mean(sizes["ct_u"]),
        "ct_n_bytes": np.mean(sizes["ct_n"]),
        "ct_dot_bytes": np.mean(sizes["ct_dot"]),
        "ct_weight_bytes": np.mean(sizes["ct_weight"]),
        "ct_agg_bytes": np.mean(sizes["ct_agg"]),
        "relin_keys_size_mean": np.mean(sizes["relin_keys"]),
        "galois_keys_size_mean": np.mean(sizes["galois_keys"]),
        "mae_error": np.nanmean(errors["mae"]),
        "max_err": np.nanmean(errors["max_err"]),
        "mem_delta_mb": (np.mean(mems) if mems else -1)
    }
    return out


def main():
    results = []

    vec_lens = [4096, 10000]
    Ns = [16384, 32768]
    coeffs_options = [
        [60, 40, 40, 60],
        [60, 40, 60],
    ]
    scale_pows = [30, 40]
    trials = 3

    print("Starting revised CKKS pipeline benchmark...")

    for N in Ns:
        for vec_len in vec_lens:
            if vec_len > N // 2:
                continue
            for coeffs in coeffs_options:
                for sp in scale_pows:
                    print(f">> Running: N={N}, vec={vec_len}, coeffs={coeffs}, scale=2^{sp}")
                    try:
                        out = run_experiment(vec_len, N, coeffs, sp, trials=trials)
                        results.append(out)
                    except Exception as e:
                        print(f"   [Error] config N={N}, vec={vec_len}, coeffs={coeffs}, scale=2^{sp}: {e}")
                        import traceback
                        traceback.print_exc()

    if results:
        df = pd.DataFrame(results)
        df.to_csv("Phase_1_CKKS/results_ckks_tenseal_pipeline.csv", index=False)
        print("Done. Results written to results_ckks_tenseal_pipeline.csv")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
