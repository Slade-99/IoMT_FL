import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv("Phase_1_CKKS/results_ckks_tenseal_pipeline.csv")

os.makedirs("figures_ckks", exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"figures_ckks/{name}.png", dpi=300)
    plt.close()


# =========================================================
# 1 — MAE vs Vector Length
# =========================================================

plt.figure(figsize=(7,5))
for scale in sorted(df["scale_pow"].unique()):
    sub = df[df["scale_pow"] == scale]
    plt.plot(sub["vec_len"], sub["mae_error"], marker="o", label=f"scale={scale}")

plt.xlabel("Vector Length")
plt.ylabel("MAE")
plt.title("MAE vs Vector Length")
plt.yscale("log")
plt.grid(True)
plt.legend()
savefig("mae_vs_vec_len")


# =========================================================
# 2 — MAE vs Scale
# =========================================================

plt.figure(figsize=(7,5))
for N in sorted(df["N"].unique()):
    sub = df[df["N"] == N]
    plt.scatter(sub["scale_pow"], sub["mae_error"], s=70, label=f"N={N}")

plt.xlabel("scale_pow (2^p)")
plt.ylabel("MAE")
plt.title("MAE vs CKKS Scale")
plt.yscale("log")
plt.grid(True)
plt.legend()
savefig("mae_vs_scale")


# =========================================================
# 3 — Dot Time vs Vector Length
# =========================================================

plt.figure(figsize=(7,5))
for N in sorted(df["N"].unique()):
    sub = df[df["N"] == N]
    plt.plot(sub["vec_len"], sub["dot_time_s"], marker="o", label=f"N={N}")

plt.xlabel("Vector Length")
plt.ylabel("Dot Time (s)")
plt.title("Homomorphic Dot-Product Time vs Vector Length")
plt.grid(True)
plt.legend()
savefig("dot_vs_vec_len")


# =========================================================
# 4 — Runtime Breakdown for Each N
# =========================================================

for N in sorted(df["N"].unique()):
    sub = df[df["N"] == N]

    plt.figure(figsize=(8,6))
    plt.plot(sub["vec_len"], sub["enc_client_u_time_s"], marker="o", label="Enc client u_hat")
    plt.plot(sub["vec_len"], sub["enc_client_norm_time_s"], marker="o", label="Enc client norm")
    plt.plot(sub["vec_len"], sub["dec_norm_time_s"], marker="o", label="Dec norm (collaborative)")
    plt.plot(sub["vec_len"], sub["enc_sink_v_time_s"], marker="o", label="Enc sink v_hat")
    plt.plot(sub["vec_len"], sub["dot_time_s"], marker="o", label="Homomorphic dot")
    plt.plot(sub["vec_len"], sub["dec_dot_time_s"], marker="o", label="Dec dot (collaborative)")
    plt.plot(sub["vec_len"], sub["scalar_mul_time_s"], marker="o", label="Scalar multiply (rebuild weights)")
    plt.plot(sub["vec_len"], sub["add_time_s"], marker="o", label="Aggregation (ciphertext add)")
    plt.plot(sub["vec_len"], sub["final_dec_time_s"], marker="o", label="Final decrypt (collaborative)")

    plt.xlabel("Vector Length")
    plt.ylabel("Runtime (s)")
    plt.title(f"Runtime Breakdown (N={N})")
    plt.grid(True)
    plt.legend()
    savefig(f"runtime_breakdown_N{N}")


# =========================================================
# 5 — Ciphertext Size vs N (ct_u_bytes)
# =========================================================

plt.figure(figsize=(7,5))
grouped = df.groupby("N")["ct_u_bytes"].mean() / (1024*1024)

plt.plot(grouped.index, grouped.values, marker="o")
plt.xlabel("Polynomial Degree N")
plt.ylabel("Ciphertext Size (MB)")
plt.title("Encrypted Normalized Vector Size vs N")
plt.grid(True)
savefig("ciphertext_u_vs_N")


# =========================================================
# 6 — Ciphertext Size vs Vector Length
# =========================================================

plt.figure(figsize=(7,5))
for N in sorted(df["N"].unique()):
    sub = df[df["N"] == N]
    plt.plot(sub["vec_len"], sub["ct_u_bytes"]/(1024*1024), marker="o", label=f"N={N}")

plt.xlabel("Vector Length")
plt.ylabel("Ciphertext Size (MB)")
plt.title("Ciphertext Size vs Vector Length")
plt.grid(True)
plt.legend()
savefig("ciphertext_u_vs_vec_len")


# =========================================================
# 7 — Heatmap: Dot Time
# =========================================================

pivot_dot = df.pivot_table(
    index="N",
    columns="scale_pow",
    values="dot_time_s",
    aggfunc="mean"
)

plt.figure(figsize=(7,5))
sns.heatmap(pivot_dot, annot=True, fmt=".3f", cmap="viridis")
plt.title("Dot Product Time (s)")
savefig("heatmap_dot_time")


# =========================================================
# 8 — Heatmap: MAE
# =========================================================

pivot_mae = df.pivot_table(
    index="N",
    columns="scale_pow",
    values="mae_error",
    aggfunc="mean"
)

plt.figure(figsize=(7,5))
sns.heatmap(np.log10(pivot_mae), annot=True, fmt=".2f", cmap="magma")
plt.title("log10(MAE) Heatmap")
savefig("heatmap_mae")


print("All CKKS pipeline plots saved to: figures_ckks/")
