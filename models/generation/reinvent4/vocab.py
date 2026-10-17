"""SMILES vocabulary for REINVENT-style LSTM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SMILESVocabulary:
    """
    Character-level vocabulary for SMILES strings.
    Tokens are individual characters plus multi-char tokens (Cl, Br, @@, ##...).
    """

    MULTI_CHAR = ["Cl", "Br", "Si", "Se", "@@", "##", "->"]
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    SPECIAL = [PAD, BOS, EOS, UNK]

    def __init__(self):
        chars = list(
            "CNOSFPIBcnos"
            "()[]=#@+\\/.%"
            "0123456789"
            "HhLrRuv"
        )
        all_tokens = self.SPECIAL + self.MULTI_CHAR + chars
        self.tok2idx: dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.idx2tok: dict[int, str] = {i: t for t, i in self.tok2idx.items()}
        self.pad_idx = self.tok2idx[self.PAD]
        self.bos_idx = self.tok2idx[self.BOS]
        self.eos_idx = self.tok2idx[self.EOS]
        self.unk_idx = self.tok2idx[self.UNK]

    def __len__(self) -> int:
        return len(self.tok2idx)

    def tokenize(self, smiles: str) -> list[str]:
        tokens: list[str] = []
        i = 0
        while i < len(smiles):
            matched = False
            for mc in self.MULTI_CHAR:
                if smiles[i : i + len(mc)] == mc:
                    tokens.append(mc)
                    i += len(mc)
                    matched = True
                    break
            if not matched:
                tokens.append(smiles[i])
                i += 1
        return tokens

    def encode(self, smiles: str) -> list[int]:
        return (
            [self.bos_idx]
            + [self.tok2idx.get(t, self.unk_idx) for t in self.tokenize(smiles)]
            + [self.eos_idx]
        )

    def decode(self, indices: list[int]) -> str:
        tokens: list[str] = []
        for idx in indices:
            tok = self.idx2tok.get(idx, "")
            if tok in (self.PAD, self.BOS, self.EOS, self.UNK):
                if tok == self.EOS:
                    break
                continue
            tokens.append(tok)
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tok2idx": self.tok2idx,
            "version": 1,
            "selfies_ready": False,
        }
        path.write_text(json.dumps(payload, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SMILESVocabulary":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls.__new__(cls)
        vocab.MULTI_CHAR = list(cls.MULTI_CHAR)
        vocab.SPECIAL = list(cls.SPECIAL)
        vocab.PAD = cls.PAD
        vocab.BOS = cls.BOS
        vocab.EOS = cls.EOS
        vocab.UNK = cls.UNK
        vocab.tok2idx = {k: int(v) for k, v in raw["tok2idx"].items()}
        vocab.idx2tok = {i: t for t, i in vocab.tok2idx.items()}
        vocab.pad_idx = vocab.tok2idx[vocab.PAD]
        vocab.bos_idx = vocab.tok2idx[vocab.BOS]
        vocab.eos_idx = vocab.tok2idx[vocab.EOS]
        vocab.unk_idx = vocab.tok2idx[vocab.UNK]
        return vocab
