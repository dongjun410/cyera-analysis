#!/usr/bin/env python
"""Direct internal-call V2.2 eval (no subprocess)."""
import hashlib, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "src"))
v22_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, v22_dir)

from cyera_bench.datasets.twenty_newsgroups import TwentyNewsgroupsDataset
from cyera_bench.datasets.cxh5types import Cxh5typesDataset
from core.document_processor import DocumentProcessor
from core.embedding_service import EmbeddingService
from core.structure_feature_extractor import StructureFeatureExtractor
from core.pii_preclassifier import PIIPreclassifier
from core.clustering_engine import ClusteringEngine
from core.label_propagator import LabelPropagator

with open(os.path.join(v22_dir, "config.yaml")) as f:
    config = yaml.safe_load(f)

# ── Dataset loading ──
def load_20news(size, seed):
    ds = TwentyNewsgroupsDataset()
    texts, label_dicts = ds.load()
    labels = [ld["l1"] for ld in label_dicts]
    if len(texts) > size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(texts), size=size, replace=False)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    return texts, labels

def load_cxh5():
    ds = Cxh5typesDataset()
    texts, label_dicts = ds.load()
    return texts, [ld["l1"] for ld in label_dicts]

# ── Run V2.2 pipeline ──
def run_v22(texts, labels, name):
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="v22dir_")
    try:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir, exist_ok=True)

        ground_truth = {}
        for i, (text, label) in enumerate(zip(texts, labels)):
            fname = f"doc_{i:05d}.txt"
            fpath = os.path.join(input_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(text)
            doc_id = hashlib.sha256(os.path.abspath(fpath).encode()).hexdigest()[:20]
            ground_truth[doc_id] = label

        # Phase 0: Load
        processor = DocumentProcessor(config.get("document", {}))
        docs = []
        for fp in sorted(Path(input_dir).rglob("*.txt")):
            doc = processor.process_file(str(fp))
            if doc and doc.raw_content.strip():
                docs.append(doc)
        print(f"  [{name}] Loaded {len(docs)} docs")

        # Phase 0.5: PII
        pii = PIIPreclassifier(config["pii_preclassifier"])
        _, unmatched = pii.scan_batch(docs)
        print(f"  [{name}] PII: {len(unmatched)} to cluster")

        if not unmatched:
            print(f"  [{name}] ERROR: all preclassified")
            return {"macro_f1": 0.0, "error": "all_preclassified"}

        # Phase 1: Structure
        ext = StructureFeatureExtractor(config["structure_features"])
        for doc in unmatched:
            doc.structure_features = ext.extract_features(doc)

        # Phase 2: Embeddings
        emb_svc = EmbeddingService(config["embedding"])
        semantic = emb_svc.encode_documents(unmatched)
        structure_vecs = ext.extract_batch(unmatched)
        embeddings = np.hstack([semantic, structure_vecs * 0.3])

        # Phase 3: Clustering
        cluster_engine = ClusteringEngine(config["clustering"])
        labels_arr = cluster_engine.fit(embeddings)
        print(f"  [{name}] Clusters: {len(set(labels_arr))}")

        # Phase 4: LLM Label propagation
        print(f"  [{name}] LLM labeling...")
        propagator = LabelPropagator(config.get("llm", {}))
        cluster_infos = propagator.process_clusters(unmatched, embeddings, labels_arr)

        # Build doc prediction map from cluster_infos
        doc_labels = {}
        for ci in cluster_infos:
            label = ci.llm_label or f"cluster_{ci.cluster_id}"
            # Get doc IDs for this cluster
            cluster_mask = labels_arr == ci.cluster_id
            for i, doc in enumerate(unmatched):
                if cluster_mask[i]:
                    doc_labels[doc.id] = label

        # Evaluate
        common = sorted(set(doc_labels.keys()) & set(ground_truth.keys()))
        y_true = [ground_truth[did] for did in common]
        y_pred = [doc_labels[did] for did in common]

        # Purity
        cluster_map = defaultdict(list)
        for did, lbl in doc_labels.items():
            cluster_map[lbl].append(did)
        purities, sizes = [], []
        for lbl, dids in cluster_map.items():
            tl = [ground_truth[d] for d in dids if d in ground_truth]
            if not tl: continue
            mc = Counter(tl).most_common(1)[0][1]
            purities.append(mc / len(tl))
            sizes.append(len(tl))
        w_purity = sum(p*s for p,s in zip(purities, sizes))/sum(sizes) if sizes else 0

        # ARI/NMI
        c_ids = [doc_labels.get(did, "x") for did in common]
        t_codes, cmap = [], {}
        for s in y_true:
            if s not in cmap: cmap[s] = len(cmap)
            t_codes.append(cmap[s])
        c_codes, cmap2 = [], {}
        for s in c_ids:
            if s not in cmap2: cmap2[s] = len(cmap2)
            c_codes.append(cmap2[s])
        ari = adjusted_rand_score(t_codes, c_codes)
        nmi = normalized_mutual_info_score(t_codes, c_codes)

        # Majority-vote F1
        mapping = defaultdict(Counter)
        for t, p in zip(y_true, y_pred):
            mapping[p][t] += 1
        p2t = {p: c.most_common(1)[0][0] for p, c in mapping.items()}

        all_cls = sorted(set(y_true))
        per_class = {}
        for cls in all_cls:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p2t.get(p) == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p2t.get(p) == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p2t.get(p) != cls)
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            per_class[cls] = {"f1": round(f1,4), "support": sum(1 for t in y_true if t==cls)}

        n = len(all_cls)
        macro_f1 = sum(per_class[c]["f1"] for c in all_cls)/n if n else 0
        correct = sum(1 for t,p in zip(y_true,y_pred) if p2t.get(p)==t)

        print(f"  [{name}] Docs={len(common)}, Clusters={len(cluster_map)}, "
              f"Purity={w_purity:.4f}, ARI={ari:.4f}, MacroF1={macro_f1:.4f}")

        return {
            "num_docs": len(common), "num_clusters": len(cluster_map),
            "purity": round(w_purity,4), "ari": round(ari,4), "nmi": round(nmi,4),
            "macro_f1": round(macro_f1,4),
            "accuracy": round(correct/len(common),4) if common else 0,
            "per_class": per_class, "correct": correct, "total": len(common),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ── Main ──
print("="*60)
print("V2.2 Direct Evaluation")
print("="*60)
print(f"PII enabled: {config['pii_preclassifier']['enabled']}")
print(f"LLM: {config['llm']['model']}")

results = {}
t0 = time.perf_counter()

print("\n--- 20 Newsgroups ---")
texts, labels = load_20news(300, 42)
results["20news"] = run_v22(texts, labels, "20news")

print("\n--- Cxh5types ---")
texts, labels = load_cxh5()
results["cxh5"] = run_v22(texts, labels, "cxh5")

elapsed = time.perf_counter() - t0
print(f"\nTotal time: {elapsed:.0f}s")

for name, r in results.items():
    if "error" not in r:
        print(f"\n{name}: Acc={r['accuracy']*100:.1f}%, MacroF1={r['macro_f1']:.4f}, "
              f"Purity={r['purity']:.4f}, ARI={r['ari']:.4f}, Clusters={r['num_clusters']}")
        for cls in sorted(r.get("per_class", {})):
            print(f"  {cls[:50]:50s} F1={r['per_class'][cls]['f1']:.4f} n={r['per_class'][cls]['support']}")

a = results.get("20news", {})
b = results.get("cxh5", {})
if "error" not in a and "error" not in b:
    delta = abs(a["macro_f1"] - b["macro_f1"])
    print(f"\nR1 |dF1|: {delta:.4f} {'PASS' if delta<0.05 else 'FAIL'}")
    print(f"R2 min(F1): {min(a['macro_f1'],b['macro_f1']):.4f} {'PASS' if min(a['macro_f1'],b['macro_f1'])>=0.9 else 'NOT YET'}")

os.makedirs("./eval_output", exist_ok=True)
with open("./eval_output/v22_direct.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved to eval_output/v22_direct.json")
