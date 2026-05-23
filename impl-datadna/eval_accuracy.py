#!/usr/bin/env python
"""Accuracy evaluation for 6-engine fusion architecture.

Measures end-to-end Macro F1 against ground truth labels.
Tests two scenarios:
  A) Mixed-format enterprise docs (PII-rich)
  B) Homogeneous plain text (no PII)

Usage:
    python eval_accuracy.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.embeddings.bge_m3 import BgeM3Embedder
from src.engines.e1_regex import E1RegexEngine
from src.engines.e2_template import E2TemplateEngine
from src.engines.e3_ml import E3MLEngine
from src.engines.e4_knn import E4kNNEngine
from src.engines.e5_structural import E5StructuralEngine
from src.engines.e6_llm import E6LLMEngine
from src.fusion.voter import FusionVoter
from src.knowledge.type_library import get_type_library
from src.llm.client import LLMConfig, MistralClient
from src.types import Document


# ── Scenario A: Mixed-format enterprise docs (PII-rich) ──
# Ground truth labels assigned by human review
SCENARIO_A: list[tuple[str, str, str, dict]] = [
    # (doc_id, text, ground_truth_label, metadata)
    ("hr_w2", "Employee SSN: 123-45-6789, Name: John Smith, Start date: 2020-03-15, "
     "Department: Engineering, Salary: $95,000, Payroll ID: PR-001, "
     "W-2 wages: $95,000, Federal tax withheld: $14,250",
     "HR & Payroll", {"file_type": ".docx"}),
    ("hr_offer", "Offer Letter: SSN: 456-78-9012, Name: Sarah Chen, "
     "Position: Senior Manager, Salary: $110,000, Start date: 2024-01-01, "
     "Benefits: Full medical, dental, vision, 401k 6% match",
     "HR & Payroll", {"file_type": ".docx"}),
    ("hr_benefits", "Benefits Enrollment: Employee SSN: 789-01-2345, "
     "Health Plan: PPO Family, Dental: Full, Vision: Basic, "
     "401k Contribution: 8%, Life Insurance: 2x salary",
     "HR & Payroll", {"file_type": ".pdf"}),
    ("hr_termination", "Termination Letter: Employee SSN: 321-65-9870, "
     "Last day: 2024-06-30, Severance: 8 weeks, Final paycheck: $15,384.62, "
     "COBRA offered, Exit interview scheduled",
     "HR & Payroll", {"file_type": ".docx"}),
    ("hr_performance", "Annual Performance Review: Employee: Michael Brown, "
     "Department: Finance, Rating: Exceeds Expectations, "
     "Goals achieved: 4/5, Promotion recommended: Senior Analyst",
     "HR & Payroll", {"file_type": ".docx"}),

    ("fin_quarterly", "Quarterly Revenue Report Q1 2024: Total Revenue: $1,200,000, "
     "Net Income: $340,000, EPS: $0.45, Gross Margin: 62%, "
     "Operating Expenses: $520,000, CFO: Jane Williams",
     "Financial Report", {"file_type": ".pdf"}),
    ("fin_invoice", "Invoice #INV-2024-089: Bill To: Acme Corp, "
     "Amount: $45,230.00, Payment: Mastercard 5500000000000004, "
     "Due: Net 30, Line items: Consulting $30,000, Software $15,230",
     "Financial Report", {"file_type": ".pdf"}),
    ("fin_bank", "Bank Statement: Account #****4321, Period: March 2024, "
     "Beginning Balance: $1,200,000, Deposits: $450,000, "
     "Withdrawals: $200,000, Ending Balance: $1,450,000",
     "Financial Report", {"file_type": ".pdf"}),
    ("fin_tax", "Tax Return 2024: Form 1040, SSN: 987-65-4321, "
     "AGI: $145,000, Schedule A: $28,500, Schedule C: $35,000, "
     "Total Tax: $32,400, Payments: $35,000, Refund: $2,600",
     "Financial Report", {"file_type": ".pdf"}),
    ("fin_expense", "Expense Report March 2024: Employee: CFO, "
     "Travel: $3,450 (flights to NYC), Meals: $890, Hotel: $2,100, "
     "Total: $6,440, Credit card: Visa 4532015112830366",
     "Financial Report", {"file_type": ".pdf"}),

    ("med_record", "Patient ID: MRN: 88421, DOB: 1978-05-15, "
     "Diagnosis: Hypertension Stage 2, NPI: 1234567890, "
     "Prescribed: Lisinopril 10mg daily, BP: 145/92, Follow-up: 3 months",
     "Medical Record", {"file_type": ".pdf"}),
    ("med_lab", "Lab Results: Patient MRN: 55102, DOB: 1985-11-22, "
     "CBC: WBC 7.2, RBC 4.8, HGB 14.5, Glucose: 142 (HIGH), A1C: 7.2%, "
     "Diagnosis: Type 2 Diabetes, NPI: 9876543210",
     "Medical Record", {"file_type": ".pdf"}),
    ("med_prescription", "Prescription: Patient MRN: 77634, NPI: 4567890123, "
     "Medication: Metformin 500mg BID, Quantity: 60, Refills: 3, "
     "Diagnosis: Type 2 Diabetes, Prescriber: Dr. James Wilson",
     "Medical Record", {"file_type": ".pdf"}),
    ("med_insurance", "Insurance Claim: Claim #CL-2024-00421, "
     "Patient MRN: 99201, NPI: 2345678901, Procedure: Pulmonary Function Test, "
     "Amount Billed: $850, Diagnosis: COPD, EOB sent",
     "Medical Record", {"file_type": ".pdf"}),
    ("med_consent", "Informed Consent: Patient MRN: 33567, Procedure: Colonoscopy, "
     "Risks explained: perforation, bleeding, infection, "
     "Patient signature: [signed], Witness: RN Lisa Park, NPI: 3456789012",
     "Medical Record", {"file_type": ".pdf"}),

    ("api_users", '{"timestamp": "2024-01-15T08:23:45Z", "endpoint": "/api/users", '
     '"status": 200, "response_time_ms": 45, "method": "GET", '
     '"ip": "192.168.1.100", "user_agent": "Mozilla/5.0"}',
     "API Log", {"file_type": ".json"}),
    ("api_orders", '{"timestamp": "2024-01-15T09:10:12Z", "endpoint": "/api/orders", '
     '"status": 201, "response_time_ms": 120, "method": "POST", '
     '"payload_size_bytes": 2048, "api_key": "sk-proj-abc123xyz"}',
     "API Log", {"file_type": ".json"}),
    ("api_auth", '{"timestamp": "2024-01-15T10:45:33Z", "endpoint": "/api/auth/login", '
     '"status": 401, "response_time_ms": 15, "error": "Invalid credentials", '
     '"ip": "10.0.0.55", "attempt": 3}',
     "API Log", {"file_type": ".json"}),
    ("api_reports", '{"timestamp": "2024-01-15T11:00:01Z", "endpoint": "/api/reports/generate", '
     '"status": 202, "response_time_ms": 350, "job_id": "job-8a7b3c", '
     '"format": "pdf", "pages": 42}',
     "API Log", {"file_type": ".json"}),
    ("api_errors", '{"timestamp": "2024-01-15T12:30:00Z", "level": "ERROR", '
     '"endpoint": "/api/payments", "status": 500, '
     '"stack_trace": "NullPointerException at PaymentService.java:142", '
     '"ip": "192.168.1.50", "request_id": "req-xyz-789"}',
     "API Log", {"file_type": ".json"}),
]

# ── Scenario B: Homogeneous plain text (no PII, same format) ──
SCENARIO_B: list[tuple[str, str, str, dict]] = [
    ("gen_meeting", "Meeting notes: Discuss Q1 goals and team building activities. "
     "Action items: finalize budget by Friday, schedule all-hands for March. "
     "Attendees: Alice, Bob, Carol, Dave.",
     "Email / Communication", {"file_type": ".txt"}),
    ("gen_roadmap", "Project roadmap 2024: Phase 1 infrastructure upgrade, "
     "Phase 2 feature rollout, Phase 3 performance optimization. "
     "Lead: Engineering team. Timeline: 6 months.",
     "Technical Document", {"file_type": ".txt"}),
    ("gen_agenda", "Quarterly all-hands agenda: Welcome new hires, team updates, "
     "Q&A session. Catering: sandwiches and salad. Room: Conference Hall A. "
     "Time: 2:00 PM - 4:00 PM.",
     "Email / Communication", {"file_type": ".txt"}),
    ("gen_offsite", "Team offsite planning: Location TBD, budget $5,000, "
     "activities: hiking + brainstorming. Date: April 15-16. "
     "RSVP by March 30. Transportation: carpool.",
     "Email / Communication", {"file_type": ".txt"}),
    ("gen_readme", "README: This repository contains the infrastructure automation "
     "scripts. Installation: pip install -r requirements.txt. "
     "Configuration: copy config.example.yaml to config.yaml.",
     "Technical Document", {"file_type": ".txt"}),
    ("gen_spec", "Technical Specification: API Gateway v2.0. Endpoints: "
     "/api/users (GET/POST), /api/orders (GET/POST/PUT). "
     "Authentication: JWT tokens with 24h expiry. Rate limit: 1000 req/min.",
     "Technical Document", {"file_type": ".txt"}),
    ("gen_deploy", "Deployment Guide: Step 1 - build Docker image. "
     "Step 2 - push to registry. Step 3 - update k8s manifests. "
     "Step 4 - apply rolling update. Rollback: kubectl rollout undo.",
     "Technical Document", {"file_type": ".txt"}),
    ("gen_changelog", "Changelog v3.2.1: Fixed memory leak in connection pool, "
     "Added retry with exponential backoff for API calls, "
     "Updated dependencies: requests>=2.31, urllib3>=2.1.",
     "Technical Document", {"file_type": ".txt"}),
    ("gen_legal_memo", "Memorandum: This document outlines the compliance "
     "requirements for GDPR Article 17 (Right to Erasure). All personal data "
     "must be deletable within 30 days. DPO: [name redacted].",
     "Legal Document", {"file_type": ".txt"}),
    ("gen_legal_nda", "Non-Disclosure Agreement: This NDA is between Company A "
     "and Company B. Confidential information includes: source code, "
     "customer lists, financial projections. Term: 5 years.",
     "Legal Document", {"file_type": ".txt"}),
]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute per-class and macro F1, precision, recall."""
    labels = sorted(set(y_true) | set(y_pred))

    per_class: dict[str, dict[str, float]] = {}
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[lbl] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": sum(1 for t in y_true if t == lbl)}

    macro_prec = sum(p["precision"] for p in per_class.values()) / len(per_class) if per_class else 0.0
    macro_rec = sum(p["recall"] for p in per_class.values()) / len(per_class) if per_class else 0.0
    macro_f1 = sum(p["f1"] for p in per_class.values()) / len(per_class) if per_class else 0.0

    return {
        "per_class": per_class,
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "total": len(y_true),
        "correct": sum(1 for t, p in zip(y_true, y_pred) if t == p),
    }


def init_components(config: dict[str, Any]) -> dict[str, Any]:
    components: dict[str, Any] = {}
    type_lib = get_type_library()
    components["e1"] = E1RegexEngine()
    components["e2"] = E2TemplateEngine()
    components["e3"] = E3MLEngine()
    try:
        emb = config.get("embedding", {})
        embedder = BgeM3Embedder(
            model_name=emb.get("model_name", "BAAI/bge-m3"),
            device=emb.get("device", "cuda"),
            batch_size=emb.get("batch_size", 32),
        )
        components["embedder"] = embedder
    except Exception:
        components["embedder"] = None
    knn_cfg = config.get("knn", {})
    components["e4"] = E4kNNEngine(embedder=components["embedder"], type_library=type_lib,
                                   min_types=knn_cfg.get("min_types_for_activation", 5))
    # Bootstrap centroids from keywords for cold start
    if components["embedder"] is not None:
        n = components["e4"].bootstrap_centroids()
        if n > 0:
            print(f"  E4 bootstrapped {n} centroids from keywords")
    components["e5"] = E5StructuralEngine(type_library=type_lib)
    try:
        llm_cfg = config.get("llm", {})
        llm_client = MistralClient(LLMConfig(
            api_base=llm_cfg.get("api_base", "http://localhost:11434/v1"),
            model=llm_cfg.get("model", "mistral:7b"),
            quantization=llm_cfg.get("quantization", "4bit"),
            temperature=llm_cfg.get("temperature", 0.3),
        ))
        components["e6"] = E6LLMEngine(llm_client=llm_client, type_library=type_lib)
    except Exception:
        components["e6"] = E6LLMEngine(llm_client=None, type_library=type_lib)
    all_engines = [components["e1"], components["e2"], components["e3"],
                   components["e4"], components["e5"], components["e6"]]
    components["voter"] = FusionVoter(engines=all_engines)
    return components


def evaluate_scenario(name: str, labeled_docs: list, comp: dict) -> dict:
    """Run classification on a labeled dataset and compute metrics."""
    voter = comp["voter"]
    y_true: list[str] = []
    y_pred: list[str] = []
    details: list[dict] = []
    fast_count = 0
    full_count = 0

    for doc_id, text, label, metadata in labeled_docs:
        doc = Document(doc_id=doc_id, text=text, metadata=metadata)
        result = voter.classify(doc)
        y_true.append(label)
        y_pred.append(result.final_label)
        if result.method == "fusion_fast":
            fast_count += 1
        else:
            full_count += 1
        details.append({
            "doc_id": doc_id,
            "true": label,
            "pred": result.final_label,
            "correct": label == result.final_label,
            "confidence": result.composite_confidence,
            "method": result.method,
        })

    metrics = compute_metrics(y_true, y_pred)
    metrics["method_distribution"] = {"fusion_fast": fast_count, "fusion_full": full_count}
    metrics["llm_call_rate"] = round(full_count / len(labeled_docs), 4) if labeled_docs else 0
    metrics["details"] = details
    return metrics


def main() -> int:
    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    print("=" * 65)
    print("6-Engine Fusion Accuracy Evaluation")
    print("=" * 65)

    print("\nInitializing components...")
    t0 = time.perf_counter()
    comp = init_components(config)
    init_time = time.perf_counter() - t0
    print(f"  Init: {init_time:.1f}s")
    print(f"  E1 Regex:     {'available' if comp['e1'].is_available else 'DOWN'}")
    print(f"  E2 Template:  {'available' if comp['e2'].is_available else 'DOWN'}")
    print(f"  E3 ML:        {'available' if comp['e3'].is_available else 'unavailable (not trained)'}")
    print(f"  E4 kNN:       {'available' if comp['e4'].is_available else 'unavailable (no centroids)'}")
    print(f"  E5 Structural:{'available' if comp['e5'].is_available else 'DOWN'}")
    print(f"  E6 LLM:       {'available' if comp['e6'].is_available else 'DOWN'}")

    # ── Scenario A: Mixed format ──
    print(f"\n{'─' * 65}")
    print(f"Scenario A: Mixed-format enterprise docs (PII-rich)")
    print(f"  {len(SCENARIO_A)} docs, 5 types, formats: .docx/.pdf/.json")
    t0 = time.perf_counter()
    metrics_a = evaluate_scenario("A", SCENARIO_A, comp)
    elapsed_a = time.perf_counter() - t0

    print(f"\n  Accuracy: {metrics_a['correct']}/{metrics_a['total']} = {metrics_a['correct']/metrics_a['total']*100:.1f}%")
    print(f"  Macro F1:  {metrics_a['macro_f1']:.4f}")
    print(f"  Macro Prec: {metrics_a['macro_precision']:.4f}")
    print(f"  Macro Rec:  {metrics_a['macro_recall']:.4f}")
    print(f"  LLM call rate: {metrics_a['llm_call_rate']*100:.0f}%")
    print(f"  Time: {elapsed_a:.1f}s ({elapsed_a/len(SCENARIO_A)*1000:.0f}ms/doc)")

    print(f"\n  Per-class F1:")
    for lbl in sorted(metrics_a["per_class"]):
        p = metrics_a["per_class"][lbl]
        print(f"    {lbl:25s}  F1={p['f1']:.4f}  P={p['precision']:.4f}  R={p['recall']:.4f}  n={p['support']}")

    print(f"\n  Errors:")
    for d in metrics_a["details"]:
        if not d["correct"]:
            print(f"    X {d['doc_id']}: true='{d['true']}' -> pred='{d['pred']}' (conf={d['confidence']:.2f})")

    # ── Scenario B: Homogeneous text ──
    print(f"\n{'─' * 65}")
    print(f"Scenario B: Homogeneous plain text (no PII, all .txt)")
    print(f"  {len(SCENARIO_B)} docs, 3 types")
    t0 = time.perf_counter()
    metrics_b = evaluate_scenario("B", SCENARIO_B, comp)
    elapsed_b = time.perf_counter() - t0

    print(f"\n  Accuracy: {metrics_b['correct']}/{metrics_b['total']} = {metrics_b['correct']/metrics_b['total']*100:.1f}%")
    print(f"  Macro F1:  {metrics_b['macro_f1']:.4f}")
    print(f"  Macro Prec: {metrics_b['macro_precision']:.4f}")
    print(f"  Macro Rec:  {metrics_b['macro_recall']:.4f}")
    print(f"  LLM call rate: {metrics_b['llm_call_rate']*100:.0f}%")
    print(f"  Time: {elapsed_b:.1f}s ({elapsed_b/len(SCENARIO_B)*1000:.0f}ms/doc)")

    print(f"\n  Per-class F1:")
    for lbl in sorted(metrics_b["per_class"]):
        p = metrics_b["per_class"][lbl]
        print(f"    {lbl:25s}  F1={p['f1']:.4f}  P={p['precision']:.4f}  R={p['recall']:.4f}  n={p['support']}")

    print(f"\n  Errors:")
    for d in metrics_b["details"]:
        if not d["correct"]:
            print(f"    X {d['doc_id']}: true='{d['true']}' -> pred='{d['pred']}' (conf={d['confidence']:.2f})")

    # ── R1 check: cross-scenario stability ──
    delta = abs(metrics_a["macro_f1"] - metrics_b["macro_f1"])
    print(f"\n{'═' * 65}")
    print(f"R1 Multi-Scenario Stability Check")
    print(f"  Scenario A Macro F1: {metrics_a['macro_f1']:.4f}")
    print(f"  Scenario B Macro F1: {metrics_b['macro_f1']:.4f}")
    print(f"  |ΔF1| = {delta:.4f}")
    if delta < 0.05:
        print(f"  [PASS] R1: |dF1| = {delta:.4f} < 0.05")
    else:
        print(f"  [FAIL] R1 FAIL: |ΔF1| = {delta:.4f} >= 0.05")

    # ── R2 check: accuracy threshold ──
    print(f"\nR2 Accuracy Check")
    min_f1 = min(metrics_a["macro_f1"], metrics_b["macro_f1"])
    if min_f1 >= 0.90:
        print(f"  [PASS] R2 PASS: min(Macro F1) = {min_f1:.4f} >= 0.90")
    else:
        print(f"  [FAIL] R2 FAIL: min(Macro F1) = {min_f1:.4f} < 0.90")

    # ── R4 check: efficiency ──
    llm_rate = (metrics_a["llm_call_rate"] * len(SCENARIO_A) + metrics_b["llm_call_rate"] * len(SCENARIO_B)) / (len(SCENARIO_A) + len(SCENARIO_B))
    avg_ms = (elapsed_a / len(SCENARIO_A) + elapsed_b / len(SCENARIO_B)) / 2 * 1000
    print(f"\nR4 Efficiency Check (maturity target)")
    print(f"  LLM call rate: {llm_rate*100:.0f}% (target: < 20%)")
    print(f"  Avg latency:   {avg_ms:.0f}ms (target: < 300ms)")
    if llm_rate < 0.20 and avg_ms < 300:
        print(f"  [PASS] R4 PASS")
    else:
        print(f"  [WARN] R4 NOT YET: requires E3 trained + E4 centroids populated")

    # Save report
    report = {
        "scenario_a": {k: v for k, v in metrics_a.items() if k != "details"},
        "scenario_b": {k: v for k, v in metrics_b.items() if k != "details"},
        "r1_delta_f1": round(delta, 4),
        "r1_pass": delta < 0.05,
        "r2_pass": min_f1 >= 0.90,
        "r4_llm_rate": round(llm_rate, 4),
        "r4_avg_ms": round(avg_ms, 0),
        "engine_status": {
            "E1_regex": comp["e1"].is_available,
            "E2_template": comp["e2"].is_available,
            "E3_ml": comp["e3"].is_available,
            "E4_knn": comp["e4"].is_available,
            "E5_structural": comp["e5"].is_available,
            "E6_llm": comp["e6"].is_available,
        },
        "scenario_a_details": metrics_a["details"],
        "scenario_b_details": metrics_b["details"],
    }
    Path("./eval_output").mkdir(exist_ok=True)
    with open("./eval_output/fusion_accuracy.json", "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: eval_output/fusion_accuracy.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
