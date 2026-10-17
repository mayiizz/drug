"""Pipeline error types shared across CURA runtime modules."""

from __future__ import annotations


class CuraPipelineError(ValueError):
    """Structured error raised by ``CuraPipeline`` stages.

    Carries an ``error_code``, ``stage`` label, ``message``, ``hints``,
    ``retriable`` flag, and optional ``dependency_status`` payload so that the
    FastAPI layer can render actionable diagnostics without needing to
    string-match exception messages.
    """

    def __init__(
        self,
        *,
        error_code: str,
        stage: str,
        message: str,
        hints: list[str] | None = None,
        retriable: bool = True,
        dependency_status: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.stage = stage
        self.message = message
        self.hints = hints or []
        self.retriable = retriable
        self.dependency_status = dependency_status or {}

    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code,
            "stage": self.stage,
            "message": self.message,
            "hints": self.hints,
            "retriable": self.retriable,
            "dependency_status": self.dependency_status,
        }


# Backward-compat alias used by legacy DeNovo callers and FastAPI's
# ``hasattr(e, "to_dict")`` branch in ``backend/main.py``.
DeNovoPipelineError = CuraPipelineError


__all__ = ["CuraPipelineError", "DeNovoPipelineError"]
