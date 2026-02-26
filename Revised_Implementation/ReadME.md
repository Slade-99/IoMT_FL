IoMT_FL_Project/
│
├── data/                     # Store datasets here (ignored by git)
│   ├── CIC_IoMT_2024/
│   └── Edge_IIoTset/         # (New dataset for Reviewer 3)
│
├── src/                      # The core engine
│   ├── __init__.py
│   ├── data_loader.py        # Clean, scale, and partition data (Dirichlet)
│   ├── model.py              # DAE Encoder/Decoder definitions
│   ├── crypto_utils.py       # CKKS encryption and Hash Log logic
│   └── math_utils.py         # Real Cosine Similarity and Norm calculations
│
├── fl_modes/                 # The different simulation environments
│   ├── __init__.py
│   ├── base_client.py        # Shared client training logic
│   ├── vanilla_fl.py         # Standard FedAvg (No security)
│   ├── blockchain_fl.py      # Blockchain baseline (with 2.0s delays)
│   └── proposed_trust_fl.py  # YOUR system (Real math, Peer Review, Sink Rotation)
│
├── experiments/              # Scripts that actually run the tests and save CSVs
│   ├── exp_1_ablation.py     # Turns your features on/off sequentially
│   ├── exp_2_attacks.py      # Runs Gaussian vs. Sign-Flip attacks
│   ├── exp_3_sensitivity.py  # Loops through different tau_sim values
│   └── exp_4_multidata.py    # Runs the best model on the second dataset
│
├── notebooks/                # Jupyter notebooks ONLY for plotting graphs from CSVs
│   ├── plot_robustness.ipynb
│   ├── plot_sensitivity.ipynb
│   └── statistical_tests.ipynb # Run your Wilcoxon/T-tests here
│
├── requirements.txt
└── README.md