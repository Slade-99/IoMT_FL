import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
ALPHA = 0.05      # Reward for good behavior
BETA = 0.20       # Penalty for minor issues (packet loss, etc)
TAU_SIM = 0.5     # Cosine Similarity Threshold
TAU_NORM = 2.0    # Norm Threshold (relative to Sink's norm)
ROUNDS = 50

# --- Setup Clients ---
# 0: Malicious Attacker (Tries to poison)
# 1: Noisy Client (Good intent, but bad data/connection)
# 2: Good Client (Always reliable)
client_ids = [0, 1, 2]
scores = {i: [0.5] for i in client_ids}
crl_list = set() # Certificate Revocation List (The Ban List)

def run_simulation():
    for t in range(ROUNDS):
        for client in client_ids:
            # 1. Check if Banned
            if client in crl_list:
                scores[client].append(0.0) # Stay dead
                continue

            prev = scores[client][-1]
            
            # 2. Simulate Behavior
            # -- Attacker tries a massive poisoning attack at Round 5 --
            if client == 0:
                if t >= 5: 
                    sim_score = -0.8  # Opposite direction
                    norm_val = 5.0    # Exploding gradient
                else:
                    sim_score = 0.6   # Acting normal initially
                    norm_val = 1.0

            # -- Noisy Client has random network drops --
            elif client == 1:
                norm_val = 1.0
                if t % 10 == 0: sim_score = 0.3 # Packet corruption
                else: sim_score = 0.85
            
            # -- Good Client --
            else:
                sim_score = 0.95
                norm_val = 1.0

            # 3. The Logic (Algorithm 1)
            # Check Norm & Similarity
            if norm_val > TAU_NORM or sim_score < TAU_SIM:
                # --- JURY TRIGGERED ---
                # In simulation: Attacker is guilty, Noisy is innocent (just penalized)
                if client == 0:
                    # Jury finds GUILTY -> BAN
                    crl_list.add(client)
                    new_score = 0.0
                else:
                    # Jury finds INNOCENT (Bad data, not malicious) -> Penalty only
                    new_score = max(0, prev - BETA)
            else:
                # Reward
                new_score = min(1.0, prev + ALPHA)
            
            scores[client].append(new_score)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(scores[0], 'r-x', label='Attacker (Banned at Rd 5)', linewidth=2)
    plt.plot(scores[1], 'y--', label='Noisy (Penalized)', linewidth=2)
    plt.plot(scores[2], 'g-', label='Good Node', linewidth=2)
    
    plt.title('Trust Dynamics with Jury & CRL Banning')
    plt.xlabel('Rounds')
    plt.ylabel('Reliability Score')
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_simulation()