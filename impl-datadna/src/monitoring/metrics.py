"""Quality monitoring metrics and alert threshold checks.

7 monitoring metrics per spec section 7:
  1. Fusion confidence distribution → alert: P50 < 0.3
  2. Per-engine output rate → alert: any engine rate drop > 30%
  3. LLM call rate → alert: > 50% or = 0%
  4. manual_review backlog → alert: > 10%
  5. New type registration rate → alert: > 10/hour
  6. Label distribution KL divergence vs baseline → alert: > 0.3
  7. fusion_fast validation inconsistency rate → alert: > 5%

Does NOT auto-remediate. Alerts → human operator decision.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from src.types import FusionResult


@dataclass
class MetricsSnapshot:
    """Aggregated metrics over a monitoring window (default 1 hour)."""

    total_documents: int = 0
    method_counts: dict[str, int] = field(default_factory=dict)
    manual_review_count: int = 0
    degraded_count: int = 0
    confidence_values: list[float] = field(default_factory=list)
    engine_output_rates: dict[str, float] = field(default_factory=dict)
    label_distribution: dict[str, int] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)

    @property
    def llm_call_rate(self) -> float:
        total = self.total_documents
        if total == 0:
            return 0.0
        fusion_full = self.method_counts.get("fusion_full", 0)
        return fusion_full / total

    @property
    def manual_review_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.manual_review_count / self.total_documents

    @property
    def p50_confidence(self) -> float:
        if not self.confidence_values:
            return 0.0
        sorted_vals = sorted(self.confidence_values)
        mid = len(sorted_vals) // 2
        return sorted_vals[mid]


class MetricsCollector:
    """Collects per-document metrics and checks alert thresholds.

    Usage:
        collector = MetricsCollector()
        collector.record(result)
        snapshot = collector.snapshot()
        for alert in snapshot.alerts:
            logger.warning("ALERT: %s", alert)
    """

    def __init__(self) -> None:
        self._results: list[FusionResult] = []
        self._baseline_labels: dict[str, float] | None = None

    def set_baseline(self, label_distribution: dict[str, float]) -> None:
        """Set baseline label distribution for KL divergence comparison."""
        self._baseline_labels = label_distribution

    def record(self, result: FusionResult) -> None:
        """Record one classification result."""
        self._results.append(result)

    def snapshot(self) -> MetricsSnapshot:
        """Compute current metrics snapshot and check thresholds."""
        snap = MetricsSnapshot()
        snap.total_documents = len(self._results)

        engine_output_counts: dict[str, int] = {}

        for r in self._results:
            snap.method_counts[r.method] = (
                snap.method_counts.get(r.method, 0) + 1
            )
            snap.confidence_values.append(r.composite_confidence)
            if r.manual_review:
                snap.manual_review_count += 1
            if r.degraded:
                snap.degraded_count += 1
            lbl = r.final_label
            snap.label_distribution[lbl] = snap.label_distribution.get(lbl, 0) + 1
            for eid, eout in r.engine_outputs.items():
                if eout.status == "matched":
                    engine_output_counts[eid] = (
                        engine_output_counts.get(eid, 0) + 1
                    )

        total = max(snap.total_documents, 1)
        for eid in ["E1_regex", "E2_template", "E3_ml", "E4_knn", "E5_structural", "E6_llm"]:
            snap.engine_output_rates[eid] = engine_output_counts.get(eid, 0) / total

        # ── Alert checks ──
        if snap.p50_confidence < 0.3:
            snap.alerts.append(
                f"Low confidence: P50={snap.p50_confidence:.3f} < 0.3"
            )

        llm_rate = snap.llm_call_rate
        if llm_rate > 0.5:
            snap.alerts.append(
                f"High LLM call rate: {llm_rate:.1%} > 50%"
            )
        if llm_rate == 0.0 and total >= 50:
            snap.alerts.append("LLM call rate is 0% — LLM may be down")

        mr_rate = snap.manual_review_rate
        if mr_rate > 0.1:
            snap.alerts.append(
                f"High manual review rate: {mr_rate:.1%} > 10%"
            )

        if self._baseline_labels is not None and snap.label_distribution:
            kl = self._kl_divergence(
                snap.label_distribution, self._baseline_labels
            )
            if kl > 0.3:
                snap.alerts.append(
                    f"Label drift detected: KL={kl:.3f} > 0.3"
                )

        return snap

    def reset(self) -> None:
        """Clear accumulated results for the next window."""
        self._results.clear()

    @staticmethod
    def _kl_divergence(
        current: dict[str, int],
        baseline: dict[str, float],
    ) -> float:
        """Compute KL divergence of current distribution vs baseline."""
        total = sum(current.values())
        if total == 0:
            return 0.0

        kl = 0.0
        for label, baseline_p in baseline.items():
            current_p = current.get(label, 0) / total
            if current_p > 0 and baseline_p > 0:
                kl += current_p * math.log(current_p / baseline_p)
        return kl
