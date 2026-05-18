# Benchmark: gemma-doc-label-cxh5types

- **Model:** gemma-doc-label (gemma4:e2b) (2000M params, gemma4:e2b)
- **Dataset:** cxh5types (258 samples)
- **Date:** 2026-05-18

## Classification Metrics

| Metric | Value |
|--------|-------|
| L1 Accuracy | 0.9186 |
| L2 Accuracy | 0.8643 |
| L2 Acc (given correct L1) | 0.9409 |
| Macro L1 F1 | 0.9072 |
| Macro L2 F1 | 0.8591 |

## Per-L1 Metrics

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| Financial & Accounting | 0.7246 | 1.0000 | 0.8403 | 50 |
| Human Resources & Payroll | 0.9885 | 0.8190 | 0.8958 | 105 |
| Legal & Compliance | 0.9902 | 0.9806 | 0.9854 | 103 |

## Performance

| Metric | Value |
|--------|-------|
| Throughput | 4519404756.1 chars/sec |
| Latency P50 | 0.0 ms |
| Latency P95 | 0.0 ms |
| Latency P99 | 0.0 ms |
| GPU Memory Peak | 0.0 GB |
| Total Time | 0.0 sec |
