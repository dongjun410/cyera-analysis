# 6-Engine Fusion Implementation Approach

> Based on: `2026-05-23-optimal-architecture.md`
> Status: ready for implementation

## Code Organization

Refactor `impl-datadna/` in-place from serial tiers to parallel engines:

```
src/engines/       NEW — 6 engines + base ABC
src/fusion/        NEW — weighted voting voter
src/knowledge/     NEW — rules, templates, type library (Phase 0)
src/monitoring/    NEW — audit log + quality metrics
src/embeddings/    KEEP (bge_m3.py unchanged)
src/llm/           KEEP (client.py unchanged)
src/distillation/  KEEP (trainer.py unchanged)
src/discovery/     KEEP (adapt to engine outputs)
src/types.py       REVISED — add EngineOutput, FusionResult
main.py            REWRITE — parallel dispatch + fusion
```

Remove: `src/tier0/`, `src/tier1/`, `src/tier2/`, `src/tier3/`, `src/ner/`

## Engine Interface

```python
@dataclass
class EngineOutput:
    engine_id: str       # "E1_regex", ...
    label: str | None    # None = no output
    confidence: float
    status: str          # matched|no_match|unavailable|skipped
    metadata: dict       # engine-specific trace

class BaseEngine(ABC):
    def analyze(self, doc: Document) -> EngineOutput: ...
    weight: float
    is_available: bool
```

## Fusion Flow

1. Run E1-E5 in parallel (< 5ms)
2. Preliminary weighted vote → score per label
3. If max_score ≥ 0.85 → skip E6, output (method=fusion_fast)
4. Else → run E6, full 6-engine fusion (method=fusion_full)
5. Composite confidence < 0.4 → manual_review=true
6. All engines unavailable → label="unclassified"

Weighted score: `Σ(engine.weight × confidence × is_available)`

## Phase 0+1 Implementation Order

1. `types.py` — EngineOutput, FusionResult
2. `engines/base.py` — ABC
3. `knowledge/rules.py` — 50+ regex rules (extend tier0/patterns.py)
4. `knowledge/templates.py` — pre-computed hashes
5. `knowledge/type_library.py` — KnownType registry
6. `engines/e1_regex.py` through `engines/e6_llm.py` — 6 engines
7. `fusion/voter.py` — weighted voting + consensus check
8. `monitoring/audit.py` + `metrics.py`
9. `main.py` — rewrite orchestration
10. Remove old tier/NER modules
11. Tests per engine + fusion + integration
