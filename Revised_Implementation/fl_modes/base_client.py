import torch
import torch.nn as nn
import torch.optim as optim
from src.model import DAE


class Client:
    def __init__(self, client_id, loader, input_dim, device,
                 lr=1e-4, weight_decay=1e-4, l1_lambda=1e-5):

        self.id = client_id
        self.loader = loader
        self.device = device

        self.model = DAE(input_dim=input_dim, latent_dim=16).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

        self.l1_lambda = l1_lambda

        self.reliability = 0.5
        self.is_malicious = False

        # Needed for Eq.(6)
        self.previous_reconstruction_error = None

    # -------------------------------------------------
    # Initialize trusted baseline before FL begins
    # -------------------------------------------------
    def initialize_reconstruction_baseline(self):
        self.model.eval()

        total, batches = 0.0, 0
        with torch.no_grad():
            for x, _ in self.loader:
                x = x.to(self.device)
                recon, _ = self.model(x)
                total += ((recon - x) ** 2).mean().item()
                batches += 1

        self.previous_reconstruction_error = total / batches

    # -------------------------------------------------
    # Local training
    # -------------------------------------------------
    def local_train(self, global_weights, noise_factor, epochs=1):
        self.model.load_state_dict(global_weights)
        self.model.train()

        for _ in range(epochs):
            for x, _ in self.loader:
                x = x.to(self.device)
                self.optimizer.zero_grad()

                recon, z = self.model(x)
                loss = self.criterion(recon, x) + self.l1_lambda * torch.norm(z, p=1)

                loss.backward()
                self.optimizer.step()

        weights = self.model.state_dict()

        # Byzantine attack (subtle)
        if self.is_malicious:
            weights = {k: v + torch.randn_like(v) * 0.5 for k, v in weights.items()}

        return weights, 0

    # -------------------------------------------------
    # Equation (6) Voting
    # -------------------------------------------------
    def vote_on_global_model(self, proposed_weights, epsilon=0.02):

        baseline = self.previous_reconstruction_error

        self.model.load_state_dict(proposed_weights)
        self.model.eval()

        total, batches = 0.0, 0
        with torch.no_grad():
            for x, _ in self.loader:
                x = x.to(self.device)
                recon, _ = self.model(x)
                total += ((recon - x) ** 2).mean().item()
                batches += 1

        current = total / batches

        if current <= baseline + epsilon:
            self.previous_reconstruction_error = current
            return 1
        else:
            return 0