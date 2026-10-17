"""
Frozen ChEMBL prior: load pretrained SMILES-LSTM, optional download, corpus pre-train.
"""

from __future__ import annotations

import logging
from typing import Any
import random
from pathlib import Path
import os

import requests
import torch

from models.generation.reinvent4.agent import SMILESLSTM
from models.generation.reinvent4.checkpoint_utils import (
    checkpoint_metadata,
    resolve_best_prior_checkpoint,
    state_param_count,
    torch_load_checkpoint,
    unwrap_model_state_dict,
)
from models.generation.reinvent4.config import (
    PRIOR_FILENAME,
    Reinvent4Config,
    resolve_prior_paths,
)
from models.generation.reinvent4.prior_corpus import DRUG_CORPUS_SMILES
from models.generation.reinvent4.vocab import SMILESVocabulary

logger = logging.getLogger(__name__)


def resolve_weights_write_path(config: Reinvent4Config) -> Path:
    primary, _ = resolve_prior_paths()
    primary.mkdir(parents=True, exist_ok=True)
    return primary / PRIOR_FILENAME


def download_prior_weights(prior_path: Path, urls: tuple[str, ...]) -> None:
    for url in urls:
        logger.info("Trying prior mirror: %s...", url[:60])
        try:
            r = requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prior_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded % (8 * 1024 * 1024) == 0:
                        pct = downloaded / total * 100
                        logger.debug("Prior download progress: %.1f%% (%d MB)", pct, downloaded // 1024 // 1024)
            if prior_path.stat().st_size < 1024:
                logger.warning("Downloaded prior file too small, trying next mirror.")
                prior_path.unlink(missing_ok=True)
                continue
            logger.info("Downloaded prior checkpoint (%d MB) successfully.", downloaded // 1024 // 1024)
            return
        except Exception as e:
            logger.warning("Prior mirror failed: %s", e)
            if prior_path.exists():
                prior_path.unlink(missing_ok=True)
            continue
    raise RuntimeError("All download mirrors failed. Place smiles_lstm_prior.pt manually.")


def _torch_load(path: Path, device: torch.device) -> Any:
    logger.info("Resolved REINVENT prior path: %s", path)
    logger.info("Resolved path exists: %s", path.exists())
    logger.debug("Current working directory: %s", os.getcwd())
    return torch_load_checkpoint(path, device)


def load_prior_checkpoint(
    prior: SMILESLSTM,
    prior_path: Path,
    device: torch.device,
) -> bool:
    if not prior_path.exists():
        return False
    try:
        state = _torch_load(prior_path, device)
        sd = unwrap_model_state_dict(state)
        prior.load_state_dict(sd, strict=False)
        params = state_param_count(sd)
        meta = checkpoint_metadata(state)
        logger.info("[PRIOR] checkpoint=%s", prior_path)
        logger.info("[PRIOR] params=%d", params)
        logger.info("[PRIOR] vocab=%d", len(getattr(prior, "embed").weight))
        if meta:
            logger.info("[PRIOR] metadata=%s", meta)
        logger.info("Loaded pretrained REINVENT prior weights from %s", prior_path)
        return True
    except Exception as e:
        logger.warning("Could not load REINVENT prior weights (%s)", e)
        return False


def pretrain_prior_on_corpus(
    prior: SMILESLSTM,
    vocab: SMILESVocabulary,
    device: torch.device,
    config: Reinvent4Config,
    save_path: Path,
) -> None:
    logger.info("Pre-training prior on drug corpus (~30s on CPU)...")
    from rdkit import Chem

    valid_corpus = [s for s in DRUG_CORPUS_SMILES if Chem.MolFromSmiles(s) is not None]
    logger.info("Prior corpus size: %d valid molecules", len(valid_corpus))

    opt = torch.optim.Adam(prior.parameters(), lr=config.pretrain_lr)
    prior.train()

    for step in range(config.pretrain_steps):
        batch = random.choices(valid_corpus, k=config.pretrain_batch_mols)
        batch_loss = torch.tensor(0.0, device=device)
        count = 0
        for smi in batch:
            ids = vocab.encode(smi)
            if len(ids) < 4:
                continue
            t = torch.tensor([ids], dtype=torch.long, device=device)
            lp = prior.log_prob(t)
            batch_loss = batch_loss + (-lp.mean())
            count += 1
        if count == 0:
            continue
        loss = batch_loss / count
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0:
            logger.info("Pre-train step %d/%d loss=%.3f", step + 1, config.pretrain_steps, loss.item())

    prior.eval()
    logger.info("Prior pre-training done.")
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prior.state_dict(), save_path)
        logger.info("Corpus-pretrained prior weights saved to %s", save_path)
    except Exception as e:
        logger.warning("Could not save prior weights: %s", e)


def build_and_load_prior(
    vocab: SMILESVocabulary,
    device: torch.device,
    config: Reinvent4Config,
) -> SMILESLSTM:
    prior = SMILESLSTM(
        len(vocab),
        embed_dim=config.lstm_embed_dim,
        hidden_dim=config.lstm_hidden_dim,
        num_layers=config.lstm_num_layers,
        dropout=config.lstm_dropout,
    ).to(device)

    if config.fine_tuned_prior_path is not None:
        custom_path = Path(config.fine_tuned_prior_path)
        if custom_path.exists() and custom_path.is_file():
            if load_prior_checkpoint(prior, custom_path, device):
                logger.info("Loaded fine-tuned prior weights from %s", custom_path)
                prior.eval()
                return prior
        else:
            logger.warning("Configured fine_tuned_prior_path not found: %s", custom_path)

    write_path = resolve_weights_write_path(config)
    resolved = resolve_best_prior_checkpoint()
    if load_prior_checkpoint(prior, resolved.path, device):
        if resolved.is_bootstrap_fallback:
            logger.warning("[PRIOR] warning: runtime is using bootstrap prior weights.")
        else:
            logger.info("[RL] initialized from healthy prior checkpoint: %s", resolved.path)
        prior.eval()
        return prior

    # Download into primary package weights dir
    try:
        download_prior_weights(write_path, config.weight_urls)
        if load_prior_checkpoint(prior, write_path, device):
            prior.eval()
            return prior
    except Exception as e:
        logger.warning(f"  Download failed ({e}). Using random init — more RL iterations needed.")

    logger.warning("Using randomly initialised prior weights (no pretrained checkpoint found).")
    pretrain_prior_on_corpus(prior, vocab, device, config, write_path)
    prior.eval()
    return prior


@torch.no_grad()
def prior_log_prob(prior: SMILESLSTM, token_ids: torch.Tensor) -> torch.Tensor:
    """Frozen prior log-probability (REINVENT augmented likelihood target)."""
    return prior.log_prob(token_ids)
