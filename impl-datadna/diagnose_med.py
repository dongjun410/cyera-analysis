"""Diagnose med_insurance misclassification root cause."""
from src.engines.e1_regex import E1RegexEngine
from src.engines.e2_template import E2TemplateEngine
from src.engines.e5_structural import E5StructuralEngine
from src.engines.e6_llm import E6LLMEngine
from src.fusion.voter import FusionVoter, ENGINE_WEIGHTS
from src.knowledge.type_library import get_type_library
from src.llm.client import LLMConfig, MistralClient
from src.types import Document

lib = get_type_library()
llm = MistralClient(LLMConfig(
    api_base="http://localhost:11434/v1",
    model="mistral:7b",
    quantization="4bit",
    temperature=0.3,
))

e1 = E1RegexEngine()
e6 = E6LLMEngine(llm_client=llm, type_library=lib)

text = ('Insurance Claim: Claim #CL-2024-00421, Patient MRN: 99201, '
        'NPI: 2345678901, Procedure: Pulmonary Function Test, '
        'Amount Billed: $850, Diagnosis: COPD, EOB sent')
doc = Document(doc_id='med_insurance', text=text, metadata={'file_type': '.pdf'})

e1_out = e1.analyze(doc)
print(f'E1 Regex:     label={e1_out.label}, conf={e1_out.confidence:.4f}, '
      f'rule={e1_out.metadata.get("rule_id")}')

e2 = E2TemplateEngine()
e2_out = e2.analyze(doc)
print(f'E2 Template:  status={e2_out.status}, label={e2_out.label}')

e5 = E5StructuralEngine()
e5_out = e5.analyze(doc)
print(f'E5 Structure: status={e5_out.status}, label={e5_out.label}')

e6_out = e6.analyze(doc)
print(f'E6 LLM:       label={e6_out.label}, conf={e6_out.confidence:.4f}')
print(f'  rationale: {e6_out.metadata.get("rationale", "")[:150]}')

print()
print('=== Fusion Score Analysis ===')
voter = FusionVoter(engines=[e1, e2, e5, e6])
result = voter.classify(doc)
print(f'Final label: {result.final_label}, conf={result.composite_confidence:.4f}')
for lbl, score in sorted(result.label_scores.items(), key=lambda x: -x[1]):
    print(f'  score("{lbl}") = {score:.4f}')

print()
print('=== Per-Engine Contribution ===')
for eid, eout in result.engine_outputs.items():
    w = ENGINE_WEIGHTS.get(eid, 0)
    contrib = w * eout.confidence if eout.status == 'matched' else 0.0
    print(f'  {eid}: w={w}, status={eout.status}, label={eout.label}, '
          f'conf={eout.confidence:.2f}, contrib={contrib:.4f}')

e1_contrib = ENGINE_WEIGHTS['E1_regex'] * e1_out.confidence
e6_contrib = ENGINE_WEIGHTS['E6_llm'] * e6_out.confidence
print()
print(f'Root cause: E1(1.0x{e1_out.confidence:.2f}={e1_contrib:.4f}) '
      f'< E6(2.0x{e6_out.confidence:.2f}={e6_contrib:.4f})')
print(f'LLM conf only {e6_out.confidence:.2f} but 2x weight makes it unbeatable')
