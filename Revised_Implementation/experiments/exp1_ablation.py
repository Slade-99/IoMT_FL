import torch
import numpy as np
from sklearn.metrics import f1_score
from src.data_loader import load_and_scale_data, partition_data_dirichlet
from src.model import DAE
from fl_modes.base_client import Client
from fl_modes.proposed_trust_fl import ProposedTrustFL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROUNDS = 40
NUM_CLIENTS = 20


def train_threshold_model(model, loader):
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    for _ in range(5):
        for x, _ in loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            loss = ((recon - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


def evaluate(model, loader, threshold):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            loss = ((recon - x) ** 2).mean(dim=1)

            preds.extend((loss > threshold).int().cpu().numpy())
            labels.extend(y.numpy())

    return f1_score(labels, preds)


def run():
    loaders, input_dim = load_and_scale_data("data/CIC_IoMT_2024",32)

    temp_model = DAE(input_dim=input_dim, latent_dim=16).to(DEVICE)
    train_threshold_model(temp_model, loaders['train'])

    val_losses = []
    with torch.no_grad():
        for x, _ in loaders['val']:
            recon, _ = temp_model(x.to(DEVICE))
            val_losses.extend(((recon - x.to(DEVICE)) ** 2).mean(dim=1).cpu().numpy())

    threshold = np.percentile(val_losses, 92)

    variants = {
        "Vanilla": dict(use_filters=False, use_peer_review=False, use_consensus=False, use_rotation=False),
        "Filters_Only": dict(use_filters=True, use_peer_review=False, use_consensus=False, use_rotation=False),
        "Filters_PeerReview": dict(use_filters=True, use_peer_review=True, use_consensus=False, use_rotation=False),
        "No_Rotation": dict(use_filters=True, use_peer_review=True, use_consensus=True, use_rotation=False),
        "Full_System": dict(use_filters=True, use_peer_review=True, use_consensus=True, use_rotation=True),
    }

    for name, config in variants.items():

        print(f"\n=== Running {name} ===")

        c_loaders = partition_data_dirichlet(loaders['train'].dataset, NUM_CLIENTS, alpha=0.5)
        clients = [Client(i, l, input_dim, DEVICE) for i, l in enumerate(c_loaders)]

        for c in clients:
            c.initialize_reconstruction_baseline()

        global_model = DAE(input_dim=input_dim, latent_dim=16).to(DEVICE)
        system = ProposedTrustFL(clients, global_model, **config)

        for r in range(ROUNDS):

            if r == 5:
                for i in range(int(NUM_CLIENTS * 0.3)):
                    clients[i].is_malicious = True

            system.run_round(r, noise_factor=0.03)

            f1 = evaluate(global_model, loaders['test'], threshold)
            print(f"{name} | Round {r:02d} | F1={f1:.4f}")


if __name__ == "__main__":
    run()