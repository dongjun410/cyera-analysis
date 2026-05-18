# Benchmark: gemma-doc-label-ben25

- **Model:** gemma-doc-label (gemma4:e2b) (2000M params, gemma4:e2b)
- **Dataset:** ben25 (25 samples)
- **Date:** 2026-05-18

## Classification Metrics

| Metric | Value |
|--------|-------|
| L1 Accuracy | 0.8000 |
| L2 Accuracy | 0.6800 |
| L2 Acc (given correct L1) | 0.8500 |
| Macro L1 F1 | 0.5682 |
| Macro L2 F1 | 0.5375 |

## Per-L1 Metrics

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| Customer & Client Records | 0.0000 | 0.0000 | 0.0000 | 1 |
| Financial & Accounting | 0.9000 | 1.0000 | 0.9474 | 9 |
| Health, Safety & Environment (HSE) | 1.0000 | 1.0000 | 1.0000 | 1 |
| Human Resources & Payroll | 0.7500 | 0.6000 | 0.6667 | 5 |
| Legal & Compliance | 0.5000 | 1.0000 | 0.6667 | 1 |
| Marketing & Communications | 0.0000 | 0.0000 | 0.0000 | 1 |
| Product & Service Documentation | 0.7143 | 1.0000 | 0.8333 | 5 |
| Strategy & Corporate Planning | 0.0000 | 0.0000 | 0.0000 | 1 |
| Training & Learning Materials | 1.0000 | 1.0000 | 1.0000 | 1 |

## Performance

| Metric | Value |
|--------|-------|
| Throughput | 10623638513.6 chars/sec |
| Latency P50 | 0.0 ms |
| Latency P95 | 0.0 ms |
| Latency P99 | 0.0 ms |
| GPU Memory Peak | 0.0 GB |
| Total Time | 0.0 sec |
