"""Fusion voter — weighted voting across 6 engines.

Core algorithm per spec section 2.3:
  1. Group engine outputs by label
  2. score(label) = Σ(engine.weight × confidence × is_available)
  3. final_label = argmax(score)
  4. composite_confidence = max_score / Σ(all engine weights)

LLM gating per spec section 2.4:
  1. Run E1-E5 first
  2. Preliminary fusion → if max_score / Σ(E1..E5 weights) >= 0.85
     → skip E6, method="fusion_fast"
  3. Else → run E6, full 6-engine fusion, method="fusion_full"

All engines unavailable → label="unclassified", manual_review=true.
"""

from __future__ import annotations

from src.types import Document, EngineOutput, FusionResult


# ── Engine weights per spec section 2.3 ──
ENGINE_WEIGHTS: dict[str, float] = {
    "E1_regex": 1.0,
    "E2_template": 1.0,
    "E3_ml": 1.5,
    "E4_knn": 1.0,
    "E5_structural": 0.8,
    "E6_llm": 2.0,
}

# Threshold for skipping LLM (E1-E5 consensus)
PRELIMINARY_CONSENSUS_THRESHOLD = 0.85

# Threshold below which manual review is required
MANUAL_REVIEW_THRESHOLD = 0.4


class FusionVoter:
    """Weighted voting fusion across 6 parallel classification engines.

    Usage:
        voter = FusionVoter(engines=[e1, e2, e3, e4, e5, e6])
        result = voter.classify(document)
        # result.method is "fusion_fast" or "fusion_full"
    """

    def __init__(self, engines: list | None = None) -> None:
        """Initialize with a list of BaseEngine instances.

        Args:
            engines: List of engine instances. If None, no engines registered.
        """
        self._engines: dict[str, object] = {}
        self._fast_engines: list[str] = []  # Engine IDs for E1-E5
        self._all_engines: list[str] = []   # All engine IDs

        if engines:
            for engine in engines:
                self._engines[engine.engine_id] = engine
                self._all_engines.append(engine.engine_id)
                if engine.engine_id != "E6_llm":
                    self._fast_engines.append(engine.engine_id)

    def classify(self, doc: Document) -> FusionResult:
        """Classify a document through the fusion pipeline.

        1. Run E1-E5 in sequence (total < 5ms)
        2. Preliminary fusion → check consensus
        3. If consensus < threshold → run E6
        4. Final fusion with all available engine outputs

        Args:
            doc: Document to classify.

        Returns:
            FusionResult with final_label, composite_confidence, method, etc.
        """
        engine_outputs: dict[str, EngineOutput] = {}
        degraded = False

        # ── Step 1: Run E1-E5 ──────────────────────────────────
        for eid in self._fast_engines:
            engine = self._engines.get(eid)
            if engine is None:
                continue
            try:
                output = engine.analyze(doc)
            except Exception:
                output = EngineOutput(
                    engine_id=eid,
                    status="unavailable",
                )
                degraded = True
            engine_outputs[eid] = output
            if output.status == "unavailable":
                degraded = True

        # ── Step 2: Preliminary fusion (E1-E5 only) ────────────
        fast_scores = self._compute_scores(engine_outputs)
        fast_max = max(fast_scores.values()) if fast_scores else 0.0
        fast_total_weight = sum(
            ENGINE_WEIGHTS.get(eid, 0.0) for eid in self._fast_engines
        )
        prelim_confidence = fast_max / fast_total_weight if fast_total_weight > 0 else 0.0

        # ── Step 3: LLM gating ─────────────────────────────────
        method = "fusion_fast"
        e6_output = None

        if prelim_confidence < PRELIMINARY_CONSENSUS_THRESHOLD:
            # Need LLM — run E6
            e6_engine = self._engines.get("E6_llm")
            if e6_engine is not None:
                try:
                    e6_output = e6_engine.analyze(doc)
                except Exception:
                    e6_output = EngineOutput(
                        engine_id="E6_llm",
                        status="unavailable",
                    )
                    degraded = True
                engine_outputs["E6_llm"] = e6_output
                if e6_output.status == "unavailable":
                    degraded = True
                method = "fusion_full"
            else:
                method = "fusion_fast"
        else:
            # Skip LLM — mark as skipped in output
            engine_outputs["E6_llm"] = EngineOutput(
                engine_id="E6_llm",
                status="skipped",
            )

        # ── Step 4: Full fusion (all available engines) ────────
        final_scores = self._compute_scores(engine_outputs)

        if not final_scores:
            # All engines unavailable → unclassified
            return FusionResult(
                doc_id=doc.doc_id,
                final_label="unclassified",
                composite_confidence=0.0,
                method=method,
                degraded=True,
                manual_review=True,
                engine_outputs=engine_outputs,
                label_scores={},
            )

        max_label = max(final_scores, key=final_scores.get)
        max_score = final_scores[max_label]
        total_weight = sum(
            ENGINE_WEIGHTS.get(eid, 0.0)
            for eid in self._all_engines
            if eid in engine_outputs
            and engine_outputs[eid].status not in ("unavailable", "skipped")
        )
        composite_confidence = (
            max_score / total_weight if total_weight > 0 else 0.0
        )

        manual_review = composite_confidence < MANUAL_REVIEW_THRESHOLD

        return FusionResult(
            doc_id=doc.doc_id,
            final_label=max_label,
            composite_confidence=round(composite_confidence, 4),
            method=method,
            degraded=degraded,
            manual_review=manual_review,
            engine_outputs=engine_outputs,
            label_scores={
                lbl: round(s, 4) for lbl, s in final_scores.items()
            },
        )

    def _compute_scores(
        self, engine_outputs: dict[str, EngineOutput]
    ) -> dict[str, float]:
        """Compute weighted scores for each label.

        score(label) = Σ(weight × confidence) for engines voting for that label.
        Engines with status "unavailable" or "skipped" contribute 0.
        """
        scores: dict[str, float] = {}
        for eid, output in engine_outputs.items():
            if output.status in ("unavailable", "skipped"):
                continue
            if output.label is None:
                continue
            weight = ENGINE_WEIGHTS.get(eid, 0.0)
            scores[output.label] = (
                scores.get(output.label, 0.0) + weight * output.confidence
            )
        return scores
