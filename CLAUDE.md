# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a technology research and analysis repository covering **Cyera's data security platform** (DSPM, Omni DLP, AI Guardian). Includes technical architecture analysis, patent portfolio deep-dive, competitive landscape, and product evolution tracking. The work product is independent analysis reports, not software.

## Key Files

- `overall_analysis_report.md` — Independent analysis of Cyera's technical architecture, classification engine (DataDNA, FLAN-T5/Mistral), data discovery, DataGraph, Omni DLP, AI Guardian, and patent portfolio overview. Uses platform-capability lens (what/why).
- `patents/technical_analysis_report.md` — Per-patent deep-dive with scenario-based unified implementation plans. Uses engineering-implementation lens (how). Authoritative for patent technical details. Cross-validated with `overall_analysis_report.md`.
- `patents/` — All 8 Cyera patent PDFs (5 granted US patents + 3 published applications).
- `CLAUDE.md` — This file.

## Patent Inventory (8 documents, 6 sections)

| # | Patent | Status | Family |
|---|--------|--------|--------|
| 1 | US12026123B2 | Granted 2024-07-02 | Data Discovery |
| 2 | US12499083B2 | Granted 2025-12-16 | Data Discovery (Cont.) |
| 3 | US12566567B2 | Granted 2026-03-03 | Data Discovery (CIP) |
| 4 | US12299167B2 | Granted 2025-05-13 | Data Classification |
| 5 | US12316686B1 | Granted 2025-05-27 | Security Policy (Trail Security) |
| 6 | US20240362301A1 | Pending | Clustering Classification |
| 7 | US20250068701A1 | Pending | Clustering Classification (Cont.) |
| 8 | WO2024224367A1 | PCT National Phase | Clustering Classification (PCT) |

## Research Methodology

When conducting analysis in this repo:

1. **Source grading**: Classify every factual claim by source reliability (A=Cyera official, B=third-party verified like Forrester/Gartner, C=reasonable inference, D=competitor/indirect).
2. **Distinguish facts from inference**: Never present reasonable inference as confirmed fact. Mark speculative claims explicitly.
3. **Verify sources**: Cross-reference claims against primary sources (Cyera blog, USPTO patents, BusinessWire press releases, analyst reports) before citing.
4. **Use WebSearch + WebFetch**: Gather information from multiple angles before writing. Prefer official Cyera sources and third-party analyst reports over competitor content.
5. **Cross-validate between reports**: `overall_analysis_report.md` and `patents/technical_analysis_report.md` describe the same platform from different lenses. When updating one, verify the other remains consistent (patent counts, timelines, technical claims, inventor names). The patent report is authoritative on patent-level detail; the platform report is authoritative on product-level capability.

## Analysis Standards

- Be independently critical — do not cater to prior analyses or preferences
- Note what is MISSING from any reference analysis being compared against
- Include a timeline dimension — Cyera's platform evolves rapidly (major releases every 2-3 months)
- When a reference analysis makes specific technical claims (throughput numbers, model types, etc.), verify each independently
- Cite sources with URLs; group sources by reliability tier at the end of reports
- Competitor claims (patent counts, technical approaches) must carry source grades — do not present them as unqualified facts
