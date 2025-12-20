import hashlib
import matplotlib.pyplot as plt
import random

# --- Constants ---
NUM_NODES = 20
ROUNDS = 1000
ROTATION_PERIOD = 5
POOL_PERCENT = 0.35

# Setup Nodes with variable reliability
# Nodes 0-6 are High Tier (0.9), Nodes 7-19 are Low Tier (0.4)
nodes = []
for i in range(NUM_NODES):
    base = 0.95 if i < 7 else 0.40 
    nodes.append({"id": f"Node_{i}", "score": base})

# Stats
selection_counts = {n["id"]: 0 for n in nodes}
current_sink = None

log_hash = "genesis_hash"

for t in range(1, ROUNDS + 1):
    # Update Scores slightly to simulate real flux
    for n in nodes:
        noise = random.uniform(-0.01, 0.01)
        n["score"] = max(0, min(1, n["score"] + noise))

    # --- ROTATION LOGIC (Every 5 Rounds) ---
    if t == 1 or t % ROTATION_PERIOD == 0:
        # 1. Sort by Reliability
        sorted_nodes = sorted(nodes, key=lambda x: x["score"], reverse=True)
        
        # 2. Define Pool (Top 35%)
        pool_size = int(len(nodes) * POOL_PERCENT) # Top 7 nodes
        candidate_pool = sorted_nodes[:pool_size]
        
        # 3. Entropy Selection (Hash of Log)
        # Simulate Log Hash changing every round
        log_hash = hashlib.sha256(f"{t}{current_sink}".encode()).hexdigest()
        hash_int = int(log_hash, 16)
        
        winner_idx = hash_int % len(candidate_pool)
        current_sink = candidate_pool[winner_idx]["id"]
    
    # Record who is Sink this round
    selection_counts[current_sink] += 1

# --- Plotting ---
names = list(selection_counts.keys())
values = list(selection_counts.values())
colors = ['green' if i < 7 else 'red' for i in range(NUM_NODES)]

plt.figure(figsize=(12, 6))
plt.bar(names, values, color=colors)
plt.title(f'Sink Selection: Top 35% Pool (Green) vs Low Tier (Red)\nRotation Every {ROTATION_PERIOD} Rounds')
plt.ylabel('Rounds Served as Sink')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()