import json
import sys
import matplotlib.pyplot as plt

# --- New Log Structure with Voting & Jury ---
def create_comprehensive_log(round_id, num_clients):
    # 1. The Header (Fixed Size)
    log_entry = {
        "Header": {
            "Round": round_id,
            "Prev_Hash": "a1b2c3d4" * 8, # SHA-256 hex string
            "Timestamp": 1688562341,
            "Sink_ID": "Node_X"
        },
        "Body": {
            # 2. Reliability Updates (Small, just IDs and new scores)
            "Reliability_Table": [ 
                {"ID": f"N_{i}", "R": 0.95} for i in range(num_clients) 
            ],
            
            # 3. Global Model Info
            "Model_Hash": "e3b0c442" * 8,
            
            # 4. THE VOTES (The heavy part - everyone signs)
            "Votes": [],
            
            # 5. Jury Info (Occasional)
            "Jury_Trials": []
        },
        "Signature": "Base64_Sig_" * 10
    }
    
    # Populate Votes
    for i in range(num_clients):
        log_entry["Body"]["Votes"].append({
            "Voter": f"N_{i}",
            "Vote": 1,
            "Sig": "Base64_Short_Sig" # ECDSA signature
        })
        
    # Simulate a Jury Trial appearing occasionally (e.g., 10% of rounds)
    if round_id % 10 == 0:
        log_entry["Body"]["Jury_Trials"].append({
            "Accused": "N_5",
            "Verdict": "GUILTY",
            "Evidence_Hash": "ab12..."
        })
        
    return log_entry

# --- Simulation ---
rounds_sim = [100, 1000, 5000, 10000]
log_sizes_mb = []

num_hospitals = 20 # Fixed network size

for r in rounds_sim:
    # Measure one average block
    dummy_block = create_comprehensive_log(1, num_hospitals)
    block_size_bytes = sys.getsizeof(json.dumps(dummy_block))
    
    # Total Size
    total_size = (block_size_bytes * r) / (1024 * 1024) # MB
    log_sizes_mb.append(total_size)
    print(f"{r} Rounds Size: {total_size:.2f} MB")

# --- Blockchain Comparison (Hyperledger Fabric) ---
# Blockchain stores full certificate chains and heavier metadata per tx
# Approx 3x to 5x larger per 'transaction' (vote)
blockchain_sizes_mb = [x * 4.5 for x in log_sizes_mb] 

# Plot
plt.figure(figsize=(10, 6))
plt.plot(rounds_sim, log_sizes_mb, 'b-o', label='Our Append-Only Log', linewidth=2)
plt.plot(rounds_sim, blockchain_sizes_mb, 'r--', label='Standard Permissioned Blockchain', linewidth=2)
plt.title('Storage Overhead: Log vs Blockchain (20 Nodes)')
plt.xlabel('Total Rounds')
plt.ylabel('Storage Size (MB)')
plt.legend()
plt.grid(True)
plt.show()