import copy
import hashlib
import torch
import numpy as np 
from src.math_utils import (
    extract_flat_weights, decompose_weights,
    calculate_cosine_similarity, update_dynamic_score
)


class ProposedTrustFL:
    def __init__(self, clients, global_model,
                 use_filters=True,
                 use_peer_review=True,
                 use_consensus=True,
                 use_rotation=True,
                 tau_sim=0.5,
                 norm_multiplier=2.5):

        self.clients = clients
        self.global_model = global_model

        self.use_filters = use_filters
        self.use_peer_review = use_peer_review
        self.use_consensus = use_consensus
        self.use_rotation = use_rotation

        self.tau_sim = tau_sim
        self.norm_multiplier = norm_multiplier

        self.crl = set()
        self.current_sink_id = 0

    # -------------------------------------------------
    # Peer Review (relative comparison)
    # -------------------------------------------------
    def trigger_peer_review(self, cid, weights):

        peers = [c for c in self.clients if c.id != cid and c.id not in self.crl]
        peers.sort(key=lambda x: x.reliability, reverse=True)
        jury = peers[:3]

        votes = 0
        for peer in jury:
            baseline = peer.previous_reconstruction_error

            peer.model.load_state_dict(weights)
            peer.model.eval()

            total, batches = 0.0, 0
            with torch.no_grad():
                for x, _ in peer.loader:
                    x = x.to(peer.device)
                    recon, _ = peer.model(x)
                    total += ((recon - x) ** 2).mean().item()
                    batches += 1

            new_error = total / batches

            if new_error > baseline * 1.5:
                votes += 1

        return votes >= 2

    # -------------------------------------------------
    # Sink Rotation
    # -------------------------------------------------
    def rotate_sink(self, round_num):
        valid = [c for c in self.clients if c.id not in self.crl]
        valid.sort(key=lambda x: x.reliability, reverse=True)

        pool = valid[:max(1, int(len(valid) * 0.35))]

        entropy = hashlib.sha256(str(round_num).encode()).hexdigest()
        self.current_sink_id = pool[int(entropy, 16) % len(pool)].id

    # -------------------------------------------------
    # One FL Round
    # -------------------------------------------------
    def run_round(self, round_num, noise_factor):

        active = [c for c in self.clients if c.id not in self.crl]

        updates = []
        for c in active:
            w, _ = c.local_train(self.global_model.state_dict(), noise_factor)
            updates.append((c.id, w))

        ref_flat = extract_flat_weights(self.global_model.state_dict())
        ref_norm, ref_u = decompose_weights(ref_flat)

        valid_updates = []

        for cid, weights in updates:

            aligned = True
            if self.use_filters:
                flat = extract_flat_weights(weights)
                c_norm, c_u = decompose_weights(flat)

                aligned = (
                    c_norm < self.norm_multiplier * ref_norm and
                    calculate_cosine_similarity(c_u, ref_u) >= self.tau_sim
                )

            client = self.clients[cid]
            client.reliability = update_dynamic_score(client.reliability, aligned)

            if aligned:
                valid_updates.append(weights)
            elif self.use_peer_review:
                if self.trigger_peer_review(cid, weights):
                    self.crl.add(cid)

        if not valid_updates:
            return

        avg = copy.deepcopy(valid_updates[0])
        for k in avg:
            for i in range(1, len(valid_updates)):
                avg[k] += valid_updates[i][k]
            avg[k] /= len(valid_updates)

        accepted = True
        if self.use_consensus:
            approvals = sum(c.vote_on_global_model(avg) for c in active)
            accepted = approvals / len(active) >= 0.6

        if accepted:
            self.global_model.load_state_dict(avg)
        else:
            self.crl.add(self.current_sink_id)

        if self.use_rotation and round_num % 5 == 0:
            self.rotate_sink(round_num)