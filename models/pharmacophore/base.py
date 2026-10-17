from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rdkit import Chem


RuleEvaluator = Callable[[Chem.Mol], tuple[bool, float, str | None]]


@dataclass(slots=True)
class PharmacophoreRule:
    name: str
    evaluator: RuleEvaluator
    hard: bool = False
    weight: float = 1.0

    def evaluate(self, mol: Chem.Mol) -> tuple[bool, float, str | None]:
        return self.evaluator(mol)


@dataclass(slots=True)
class PharmacophoreConstraint:
    rule: PharmacophoreRule
    threshold: float = 0.5
    enabled: bool = True


@dataclass(slots=True)
class PharmacophoreProfile:
    target_name: str
    constraints: list[PharmacophoreConstraint] = field(default_factory=list)
    description: str = ""

    def evaluate(
        self,
        mol: Chem.Mol,
        *,
        enforce_hard_filters: bool,
    ) -> tuple[bool, float, list[str], list[str]]:
        """
        Returns:
          passed, score, failed_rule_names, matched_rule_names
        """
        if mol is None:
            return False, 0.0, ["invalid_smiles"], []

        failed: list[str] = []
        matched: list[str] = []
        weighted_score = 0.0
        total_weight = 0.0

        for c in self.constraints:
            if not c.enabled:
                continue
            ok, raw_score, reason = c.rule.evaluate(mol)
            score = float(raw_score)
            total_weight += max(0.0, c.rule.weight)
            weighted_score += max(0.0, min(1.0, score)) * max(0.0, c.rule.weight)
            if ok and score >= c.threshold:
                matched.append(c.rule.name)
            else:
                failed.append(c.rule.name if reason is None else f"{c.rule.name}:{reason}")
                if c.rule.hard and enforce_hard_filters:
                    return False, 0.0, failed, matched

        final_score = (weighted_score / total_weight) if total_weight > 0 else 0.0
        passed = len([f for f in failed if ":" in f or True]) == 0 if enforce_hard_filters else True
        if enforce_hard_filters and failed:
            # In hard mode, any failed hard rule already returned; non-hard failures are tolerated.
            passed = True
        return passed, float(final_score), failed, matched

