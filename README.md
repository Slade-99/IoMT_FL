# Lightweight Encryption and Privacy-Preserving Federated Learning for IoMT

This repository contains the implementation of a federated learning framework designed for **Intrusion Detection in Internet of Medical Things (IoMT) environments**. The framework integrates lightweight encryption mechanisms and privacy-preserving techniques to enable secure collaborative model training across distributed IoMT nodes.

The implementation accompanies our research on **Trust-Aware Federated Intrusion Detection for IoMT environments**.

---

# Repository Structure
IoMT_FL/
│
├── Revised_Implementation/ # Main implementation used in the experiments
├── Previous_Implementation/ # Obsolete implementation (kept for reference)
├── runner.py # Script to run the federated learning workflow



### Important Note
The **`Previous_Implementation`** folder contains an earlier prototype and is **no longer used in the final experiments**.

All experiments described in the research are implemented in the **`Revised_Implementation`** directory.

---

# Description

The proposed framework enables **privacy-preserving intrusion detection in IoMT networks** through federated learning. Instead of sharing raw medical or device data, multiple clients collaboratively train a global model by sharing model updates.

The framework includes:

- Federated learning workflow for distributed model training
- Lightweight encryption mechanisms to protect model updates
- Secure aggregation strategies
- Experimental evaluation using IoMT cybersecurity datasets

---

# Dataset Information

The experiments are conducted using publicly available IoT/IoMT cybersecurity datasets such as:

- CICIoMT2024 dataset
- TON_IoT dataset

These datasets contain both **benign and attack traffic generated from IoMT environments**, enabling evaluation of intrusion detection performance in realistic scenarios.

Datasets must be downloaded separately and placed in the appropriate directories before running the experiments.

---

# Code Implementation

The **core implementation is located in**
Revised_Implementation/


This folder contains the modules responsible for:

- Federated learning training pipeline
- Model aggregation
- Encryption mechanisms
- Data preprocessing
- Experiment orchestration

The repository primarily uses:

- Python
- Jupyter Notebook
- Machine learning libraries

---

# Experiments

All experimental evaluations are implemented as **separate Jupyter Notebook files** inside the repository.

Each notebook corresponds to a specific experiment such as:

- Model training
- Performance evaluation
- Federated learning simulations
- Intrusion detection analysis

This design allows experiments to be executed independently.

---

# Requirements

Typical dependencies include:
Python 3.8+
numpy
pandas
scikit-learn
matplotlib
jupyter

Install dependencies using:
pip install -r requirements.txt


# Installation

Clone the repository:
git clone https://github.com/Slade-99/IoMT_FL.git
cd IoMT_FL


Launch Jupyter Notebook:

Navigate to the relevant notebook in the `Revised_Implementation` folder to run experiments.

---

# Usage

Typical workflow:

1. Download the dataset (TON_IoT or CICIoMT2024)
2. Place the dataset in the appropriate directory
3. Run preprocessing steps if required
4. Execute experiment notebooks
5. Train the federated learning model
6. Evaluate intrusion detection performance

---

# Methodology

The proposed system follows these steps:

1. **Data Collection**  
   IoMT traffic datasets containing normal and malicious activity are used.

2. **Data Preprocessing**  
   Cleaning, normalization, and feature selection are performed.

3. **Federated Training**  
   Multiple simulated clients train local models without sharing raw data.

4. **Secure Aggregation**  
   Encrypted model updates are aggregated at the central server.

5. **Intrusion Detection Evaluation**  
   Performance metrics such as accuracy, precision, recall, and F1-score are computed.

---

# Reproducibility

To reproduce the experiments:

1. Install all required dependencies.
2. Download the required IoMT datasets.
3. Run the experiment notebooks in the `Revised_Implementation` directory.
4. Follow the execution order defined in the notebooks.

---

