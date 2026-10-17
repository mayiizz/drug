"""
DeNovoPipeline - protein-only entry point shim.

DEPRECATED ORCHESTRATION: this class is now a thin shim around the unified
:class:`models.runtime.cura_pipeline.CuraPipeline`. It exists to keep the
existing FastAPI ``/denovo`` request/response shape stable while the canonical
runtime is :func:`CuraPipeline.run`.

The full pipeline flow (ESMFold -> pocket -> SAFE filter -> REINVENT -> drug
likeness -> pharmacophore -> DeepPurpose prefilter -> Vina -> ADMET ->
multi-objective ranking) lives in ``cura_pipeline``. Do not add new
orchestration logic here.

TODO: collapse this shim once frontend / external callers migrate to
``CuraPipeline.run`` directly.
"""

from __future__ import annotations

import logging

from models.runtime.errors import CuraPipelineError, DeNovoPipelineError

logger = logging.getLogger(__name__)


class DeNovoPipeline:
    """Thin protein-only shim around :class:`CuraPipeline`.

    Public API matches the legacy de novo entry::

        pipeline = DeNovoPipeline()
        candidates = pipeline.run(
            protein_sequence="MRPSG...",
            target="EGFR",
            num_molecules=10,
            sample_size=500,
        )

    A pre-built :class:`CuraPipeline` instance can be injected to share warm
    models, docking caches, and the REINVENT prior across multiple shims.
    """

    def __init__(self, *, cura: object | None = None) -> None:
        from models.runtime.cura_pipeline import CuraPipeline

        if cura is None:
            logger.info("[DeNovo] Building dedicated CuraPipeline runtime.")
            cura = CuraPipeline()
        self._cura = cura
        # Compatibility metadata expected by existing FastAPI error branches.
        self.generator = getattr(cura, "_reinvent_loop", None)
        self.admet = getattr(cura, "_admet_local", None)
        self._binding_model = getattr(cura, "_binding_predictor", None)

    @property
    def last_benchmark_payload(self) -> dict:
        return getattr(self._cura, "last_benchmark_payload", {})

    def run(
        self,
        protein_sequence: str,
        target: str = "unknown",
        num_molecules: int = 10,
        sample_size: int = 500,
        seed_smiles: str | None = None,
    ) -> list[dict]:
        """Delegate to :func:`CuraPipeline.run`.

        ``seed_smiles`` is optional. When ``None`` the pipeline runs pure
        protein-guided de novo generation; when provided it is forwarded to
        :class:`CuraPipeline` which uses it as a Tanimoto-similarity reward
        bias only (it does **not** condition REINVENT generation).
        """
        mode = "seed-biased" if seed_smiles else "protein-only"
        logger.info(
            "[DeNovoShim] dispatching to CuraPipeline | mode=%s | target=%s | "
            "n=%d | sample=%d",
            mode,
            target,
            num_molecules,
            sample_size,
        )
        try:
            return self._cura.run(  # type: ignore[union-attr]
                protein_sequence=protein_sequence,
                target=target,
                num_molecules=num_molecules,
                sample_size=sample_size,
                seed_smiles=seed_smiles,
            )
        except CuraPipelineError:
            raise
        except Exception:
            raise


__all__ = ["DeNovoPipeline", "DeNovoPipelineError"]
