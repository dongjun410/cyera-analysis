"""Tier 2: Tier2Classifier — cluster-level classification orchestrator.

The central coordinator of Tier 2 — runs the match -> NER -> LLM -> propagate
flow for each cluster. Per spec section 4.3.

Flow for each cluster:
  Step 1: Extract cluster features (TF-IDF, PII distribution summary)
  Step 2: Run KnownTypeMatcher.match()
    - score >= 0.8 -> adopt match label, skip to Step 5
    - score < 0.8 -> continue to Steps 3-4
  Step 3: Run DeBERTa NER on representative docs (up to 3)
  Step 4: LLM classification (only unmatched or mid-confidence clusters)
    - Input: cluster TF-IDF keywords + rep doc text (2000 chars) + NER results
  Step 5: Propagate labels via LabelPropagator
"""

from __future__ import annotations

from src.tier2.matching import KnownTypeMatcher
from src.tier2.propagation import LabelPropagator
from src.ner.deberta import DebertaNER
from src.llm.client import MistralClient
from src.types import ClassificationResult, ClusterInfo, Document


class Tier2Classifier:
    """Orchestrates Tier 2 classification: match -> NER -> LLM -> propagate.

    All external dependencies (matcher, ner, llm, propagator) are injected via
    constructor (DI), enabling full testability with mocks.

    Parameters
    ----------
    matcher : KnownTypeMatcher
        The 3-signal known type matcher.
    ner : DebertaNER
        DeBERTa-v3 NER service for PII detection.
    llm : MistralClient
        Mistral-7B LLM client for classification.
    propagator : LabelPropagator
        Label propagator for assigning cluster labels to member documents.
    config : dict, optional
        - ner_representative_limit : int (default 3) — max rep docs for NER
    """

    def __init__(
        self,
        matcher: KnownTypeMatcher,
        ner: DebertaNER,
        llm: MistralClient,
        propagator: LabelPropagator,
        config: dict | None = None,
    ) -> None:
        self._matcher = matcher
        self._ner = ner
        self._llm = llm
        self._propagator = propagator
        self._config = config or {}

        self._ner_representative_limit: int = self._config.get(
            "ner_representative_limit", 3
        )

    # ── Public API ────────────────────────────────────────────

    def classify_clusters(
        self,
        clusters: list[ClusterInfo],
        documents: list[Document],
    ) -> list[ClassificationResult]:
        """Run the full Tier 2 classification pipeline for a list of clusters.

        For each cluster:
          1. Extract features (TF-IDF keywords, PII distribution, rep docs)
          2. Run KnownTypeMatcher.match()
          3. If not known_match, run DeBERTa NER on representative docs
          4. If not known_match, run LLM classification
          5. Propagate labels to all cluster members

        Parameters
        ----------
        clusters : list[ClusterInfo]
            Clusters from Tier 1 to classify.
        documents : list[Document]
            All documents (used for text lookup and propagation).

        Returns
        -------
        list[ClassificationResult]
            One ClassificationResult per document that received a label.
        """
        if not clusters:
            return []

        # Build doc lookup for text access
        doc_lookup: dict[str, Document] = {d.doc_id: d for d in documents}

        # Collect known type names from the matcher for LLM prompt
        known_type_names = [
            kt.type_name for kt in self._matcher._types.values()
        ]

        all_results: list[ClassificationResult] = []

        for cluster in clusters:
            # ── Step 1: Extract cluster features (already in ClusterInfo) ──
            tfidf_keywords = cluster.tfidf_keywords
            pii_distribution = cluster.pii_distribution

            # Resolve representative doc texts
            rep_doc_ids = cluster.representative_docs[:self._ner_representative_limit]
            if not rep_doc_ids and cluster.doc_ids:
                # Fallback: use first doc_ids if representative_docs is empty
                rep_doc_ids = cluster.doc_ids[:self._ner_representative_limit]

            rep_texts = [
                doc_lookup[doc_id].text
                for doc_id in rep_doc_ids
                if doc_id in doc_lookup
            ]

            # ── Step 2: Match against known types ──
            match_result = self._matcher.match(cluster)

            if match_result.method == "known_match":
                # High-confidence match — adopt the label directly
                label = match_result.matched_type.type_name
                confidence = match_result.score
                method = "known_match"
            else:
                # ── Step 3: DeBERTa NER on representative docs ──
                ner_results = self._ner.predict_batch(rep_texts) if rep_texts else []

                # ── Step 4: LLM classification ──
                # Build document text from representative docs (concatenated)
                combined_text = "\n".join(rep_texts) if rep_texts else ""

                llm_result = self._llm.classify(
                    combined_text,
                    known_type_names,
                    ner_results=ner_results,
                    pii_features=pii_distribution,
                )

                label = llm_result.get("label", "unknown")
                confidence = llm_result.get("confidence", 0.0)
                method = "llm_tier2"

            # ── Step 5: Propagate labels ──
            prop_results, _needs_resplit = self._propagator.propagate(
                cluster, label, confidence, documents
            )

            # Override method in propagation results to reflect true origin
            for pr in prop_results:
                pr.method = method

            all_results.extend(prop_results)

        return all_results

    def cold_start_classify(
        self,
        documents: list[Document],
    ) -> list[ClassificationResult]:
        """Phase -1: zero-shot LLM per document. No clustering. Temporary labels.

        Each document is classified independently by the LLM with an empty
        known types list. This provides a rough initial labeling that can be
        refined once clusters form.

        Parameters
        ----------
        documents : list[Document]
            Documents to classify with zero-shot LLM.

        Returns
        -------
        list[ClassificationResult]
            One ClassificationResult per input document, with method="llm_tier2".
        """
        results: list[ClassificationResult] = []

        for doc in documents:
            llm_result = self._llm.classify(doc.text, [])

            label = llm_result.get("label", "unknown")
            confidence = llm_result.get("confidence", 0.0)
            is_new_type = llm_result.get("is_new_type", False)
            rationale = llm_result.get("rationale", "Cold start LLM classification")

            results.append(
                ClassificationResult(
                    doc_id=doc.doc_id,
                    label=label,
                    confidence=confidence,
                    method="llm_tier2",
                    is_new_type=is_new_type,
                    needs_manual_review=confidence < 0.85,
                    rationale=rationale,
                )
            )

        return results
