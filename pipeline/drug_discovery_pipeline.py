"""
DrugDiscoveryPipeline - seed-anchored entry point shim.

DEPRECATED ORCHESTRATION: this class is now a thin shim around the unified
:class:`models.runtime.cura_pipeline.CuraPipeline`. It is kept to preserve
the existing FastAPI ``/discover`` request/response shape during the
migration; CuraPipeline is the single canonical runtime.

Seed conditioning is now optional - the seed only flows into the pipeline as
a *reward bias* (Tanimoto similarity weight) rather than as a generation
control signal. Protein-guided generation is primary in all modes.

TODO: collapse this shim once frontend / external callers migrate to
``CuraPipeline.run`` directly.
"""

from __future__ import annotations

import logging

from models.runtime.errors import CuraPipelineError

logger = logging.getLogger(__name__)


class DrugDiscoveryPipeline:
    """Thin seed-aware shim around :class:`CuraPipeline`.

    Public API matches the legacy seed-anchored entry::

        pipeline = DrugDiscoveryPipeline()
        candidates = pipeline.run(
            seed_smiles="C(#N)c1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
            protein_sequence="MRPSG...",
            target="EGFR",
            num_molecules=10,
        )

    A pre-built :class:`CuraPipeline` instance can be injected to share warm
    models, docking caches, and the REINVENT prior across multiple shims.
    """

    def __init__(self, *, cura: object | None = None) -> None:
        from models.runtime.cura_pipeline import CuraPipeline

        if cura is None:
            logger.info("[Discover] Building dedicated CuraPipeline runtime.")
            cura = CuraPipeline()
        self._cura = cura
        # Compatibility metadata expected by older callers.
        self.generator = getattr(cura, "_reinvent_loop", None)
        self.ranker = getattr(cura, "_ranker", None)
        self._binding_model = getattr(cura, "_binding_predictor", None)
        self.chemberta = getattr(cura, "_chemberta", None)
        self.fragment_grower = None  # legacy expansion path is retired
        self.admet_results: list[dict] = []

    @property
    def last_benchmark_payload(self) -> dict:
        return getattr(self._cura, "last_benchmark_payload", {})

    @property
    def structure_metadata(self) -> dict:
        return getattr(self._cura, "structure_metadata", {})

    def run(
        self,
        seed_smiles: str,
        protein_sequence: str,
        target: str = "unknown",
        num_molecules: int = 10,
    ) -> list[dict]:
        """Delegate to :func:`CuraPipeline.run` with seed-as-bias semantics."""
        try:
            return self._cura.run(  # type: ignore[union-attr]
                protein_sequence=protein_sequence,
                target=target,
                num_molecules=num_molecules,
                sample_size=500,
                seed_smiles=seed_smiles,
            )
        except CuraPipelineError as e:
            # Map structured pipeline errors back to ``ValueError`` for the
            # existing FastAPI ``/discover`` 422 handler in main.py.
            raise ValueError(e.message) from e
        except Exception:
            raise


__all__ = ["DrugDiscoveryPipeline"]
