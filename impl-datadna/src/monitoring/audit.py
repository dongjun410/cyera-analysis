"""Per-decision JSON audit log writer.

Every document classification produces an AuditRecord with all 6 engine
outputs, the fusion decision, and metadata. Per spec section 8.

Records are written as JSON Lines for append-only streaming.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.types import FusionResult


class AuditLogger:
    """JSON Lines audit log for every classification decision.

    Usage:
        logger = AuditLogger(Path("./output/audit.jsonl"))
        logger.log(result)
        # Each call appends one JSON line to the file.
    """

    def __init__(self, output_path: Path) -> None:
        self._path = output_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def log(self, result: FusionResult) -> None:
        """Append one audit record to the log file.

        Args:
            result: FusionResult from the fusion voter.
        """
        engines = {}
        for eid, output in result.engine_outputs.items():
            engines[eid] = {
                "status": output.status,
                "label": output.label,
                "confidence": output.confidence,
                "metadata": output.metadata,
            }

        record = {
            "doc_id": result.doc_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "final_label": result.final_label,
            "composite_confidence": result.composite_confidence,
            "method": result.method,
            "degraded": result.degraded,
            "manual_review": result.manual_review,
            "label_scores": result.label_scores,
            "engines": engines,
        }

        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._count += 1

    @property
    def count(self) -> int:
        return self._count
