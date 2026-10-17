"""Trainable SMILES LSTM policy network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from rdkit import Chem


class SMILESLSTM(nn.Module):
    """
    Multi-layer LSTM that models P(SMILES) autoregressively.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.last_sample_stats: dict[str, float | int] = {}

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.proj(out)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def log_prob(self, token_ids: torch.Tensor) -> torch.Tensor:
        inputs = token_ids[:, :-1]
        targets = token_ids[:, 1:]
        hidden = self.init_hidden(token_ids.size(0), token_ids.device)
        logits, _ = self.forward(inputs, hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        mask = (targets != 0).float()
        return (gathered * mask).sum(dim=1)

    @torch.no_grad()
    def sample(
        self,
        vocab,
        batch_size: int,
        max_len: int = 72,
        temperature: float = 0.6,
        device: torch.device | None = None,
        greedy: bool = False,
        top_k: int = 20,
    ) -> tuple[list[str], torch.Tensor]:
        """Autoregressive sampling; returns decoded SMILES and token id matrix (padded)."""
        if device is None:
            device = next(self.parameters()).device
        hidden = self.init_hidden(batch_size, device)
        token = torch.full((batch_size, 1), vocab.bos_idx, dtype=torch.long, device=device)
        sequences = [[] for _ in range(batch_size)]
        all_ids = torch.full((batch_size, max_len), vocab.pad_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            logits, hidden = self.forward(token, hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            # Encourage graceful stopping in long trajectories.
            if step > int(max_len * 0.60):
                logits[:, vocab.eos_idx] = logits[:, vocab.eos_idx] + 0.75
            # Top-k truncation removes long-tail token noise.
            if top_k > 0 and top_k < logits.size(-1):
                k = int(top_k)
                vals, idxs = torch.topk(logits, k=k, dim=-1)
                masked = torch.full_like(logits, fill_value=-1e9)
                masked.scatter_(1, idxs, vals)
                logits = masked
            probs = F.softmax(logits, dim=-1)
            if greedy:
                next_token = probs.argmax(dim=-1)
            else:
                next_token = Categorical(probs).sample()

            all_ids[:, step] = next_token
            for i in range(batch_size):
                if not finished[i]:
                    if next_token[i].item() == vocab.eos_idx:
                        finished[i] = True
                    else:
                        sequences[i].append(next_token[i].item())
            token = next_token.unsqueeze(1)
            if finished.all():
                break

        smiles = [vocab.decode(seq) for seq in sequences]
        decoded_total = len(smiles)
        raw_valid = 0
        raw_invalid = 0
        sanitize_failures = 0
        valid_smiles: list[str] = []
        for s in smiles:
            try:
                # lightweight grammar sanity before RDKit parse
                if s.count("(") != s.count(")"):
                    raw_invalid += 1
                    continue
                digits = [ch for ch in s if ch.isdigit()]
                if any(digits.count(d) % 2 != 0 for d in set(digits)):
                    raw_invalid += 1
                    continue
                max_aromatic_run = 0
                run = 0
                for ch in s:
                    if ch in "cnosp":
                        run += 1
                        if run > max_aromatic_run:
                            max_aromatic_run = run
                    else:
                        run = 0
                if max_aromatic_run > 10:
                    raw_invalid += 1
                    continue
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    raw_invalid += 1
                    continue
                raw_valid += 1
                Chem.SanitizeMol(mol)
                valid_smiles.append(s)
            except Exception:
                sanitize_failures += 1
                continue
        raw_valid_fraction = float(raw_valid) / float(max(1, decoded_total))
        self.last_sample_stats = {
            "decoded": int(decoded_total),
            "raw_valid": int(raw_valid),
            "raw_invalid": int(raw_invalid),
            "sanitize_failures": int(sanitize_failures),
            "raw_valid_fraction": float(raw_valid_fraction),
            "sanitized_valid": int(len(valid_smiles)),
        }
        print(f"[PRIOR] decoded={decoded_total}")
        print(f"[PRIOR] raw_valid={raw_valid}")
        print(f"[PRIOR] raw_invalid={raw_invalid}")
        print(f"[PRIOR] sanitize_failures={sanitize_failures}")
        print(f"[PRIOR] raw_valid_fraction={raw_valid_fraction:.4f}")
        smiles = valid_smiles
        return smiles, all_ids
