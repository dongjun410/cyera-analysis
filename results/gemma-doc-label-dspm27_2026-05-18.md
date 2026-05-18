# Benchmark: gemma-doc-label-dspm27

- **Model:** gemma-doc-label (gemma4:e2b) (2000M params, gemma4:e2b)
- **Dataset:** dspm27 (27 samples)
- **Date:** 2026-05-18

## Classification Metrics

| Metric | Value |
|--------|-------|
| L1 Accuracy | 0.6296 |
| L2 Accuracy | 0.5556 |
| L2 Acc (given correct L1) | 0.8824 |
| Macro L1 F1 | 0.4587 |
| Macro L2 F1 | 0.4258 |

## Per-L1 Metrics

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| Customer & Client Records | 0.5000 | 0.5000 | 0.5000 | 2 |
| Executive & Board Documents | 0.0000 | 0.0000 | 0.0000 | 1 |
| Financial & Accounting | 0.6667 | 1.0000 | 0.8000 | 4 |
| Health, Safety & Environment (HSE) | 1.0000 | 1.0000 | 1.0000 | 2 |
| Human Resources & Payroll | 1.0000 | 0.8000 | 0.8889 | 5 |
| IT & Systems | 0.5000 | 1.0000 | 0.6667 | 1 |
| Legal & Compliance | 1.0000 | 0.3333 | 0.5000 | 3 |
| Marketing & Communications | 0.0000 | 0.0000 | 0.0000 | 1 |
| Product & Service Documentation | 0.2500 | 1.0000 | 0.4000 | 1 |
| Projects & Programs | 0.0000 | 0.0000 | 0.0000 | 1 |
| Research & Analysis | 0.5000 | 0.5000 | 0.5000 | 2 |
| Sales & Business Development | 0.3333 | 1.0000 | 0.5000 | 1 |
| Strategy & Corporate Planning | 1.0000 | 0.5000 | 0.6667 | 2 |
| Training & Learning Materials | 0.0000 | 0.0000 | 0.0000 | 1 |

## Performance

| Metric | Value |
|--------|-------|
| Throughput | 4753832650.9 chars/sec |
| Latency P50 | 0.0 ms |
| Latency P95 | 0.0 ms |
| Latency P99 | 0.0 ms |
| GPU Memory Peak | 0.0 GB |
| Total Time | 0.0 sec |
