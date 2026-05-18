import json
import os
import re
import urllib.request
from typing import Dict, List


GEMMA_API_URL = "http://127.0.0.1:8003/classify_text"
GEMMA_TIMEOUT_SEC = 600


def _to_snake(text: str) -> str:
    t = text.lower()
    t = t.replace("&", "and")
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = t.strip("_")
    t = re.sub(r"_+", "_", t)
    return t


_TAXONOMY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..",
    "ZerosOne", "gemma-doc-label", "config", "file_labels_20_10.json",
)


def _build_taxonomy():
    """Read taxonomy and return (l1_map, l2_to_l1_map, word_index).

    l1_map: {snake: display_name}
    l2_to_l1_map: {snake: (l2_display, l1_display)}
    word_index: {word: [snake_key, ...]}  -- inverted index for word overlap search
    """
    l1_map: Dict[str, str] = {}
    l2_to_l1: Dict[str, tuple] = {}
    word_index: Dict[str, List[str]] = {}

    def _index(snake_key: str) -> None:
        for w in snake_key.split("_"):
            if len(w) > 1:
                word_index.setdefault(w, []).append(snake_key)

    tax_path = os.path.normpath(_TAXONOMY_PATH)
    try:
        with open(tax_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        for cat in taxonomy.get("categories", []):
            l1_display = cat["name"]
            l1_snake = _to_snake(l1_display)
            l1_map[l1_snake] = l1_display
            _index(l1_snake)
            for ft in cat.get("file_types", []):
                l2_snake = _to_snake(ft)
                l2_to_l1[l2_snake] = (ft, l1_display)
                _index(l2_snake)
    except FileNotFoundError:
        pass
    return l1_map, l2_to_l1, word_index


def _word_subset(a: str, b: str) -> bool:
    """True if one word is a close variant of another (substring, differ by ≤1 char)."""
    if a == b:
        return True
    longer = max(len(a), len(b))
    shorter = min(len(a), len(b))
    # Only accept if lengths differ by at most 1 (e.g. "nda"~"ndas", "invoice"~"invoices")
    return (a in b or b in a) and shorter >= longer - 1


def _fuzzy_match(raw_label: str, taxonomy: tuple) -> str:
    """Match a raw label to the closest L1 display name.

    Strategy: exact match → substring containment → word-level IDF overlap.
    """
    if not raw_label:
        return "unknown"

    l1_map, l2_to_l1, word_index = taxonomy
    raw_snake = _to_snake(raw_label)

    # 1) Exact match L1 or L2
    if raw_snake in l1_map:
        return l1_map[raw_snake]
    if raw_snake in l2_to_l1:
        return l2_to_l1[raw_snake][1]

    # 2) Substring containment at full key level (minimum 4 chars to avoid false matches)
    if len(raw_snake) >= 4:
        for l2_snake, (_, l1_display) in l2_to_l1.items():
            if raw_snake in l2_snake or l2_snake in raw_snake:
                return l1_display
        for l1_snake, l1_display in l1_map.items():
            if raw_snake in l1_snake or l1_snake in raw_snake:
                return l1_display

    # 3) Word-level overlap with IDF-weighted scoring + minimum discriminability
    raw_words = [w for w in raw_snake.split("_") if len(w) > 0]
    if not raw_words:
        return "unknown"

    all_snake_keys = set(l1_map.keys()) | set(l2_to_l1.keys())
    total_labels = len(all_snake_keys)
    import math

    # IDF per word (higher = more discriminating). Skip words too short or too common.
    raw_idfs: Dict[str, float] = {}
    for rw in raw_words:
        if len(rw) < 3:  # skip very short tokens (too ambiguous as substrings)
            continue
        n = sum(1 for tax_key in all_snake_keys
                if any(_word_subset(rw, tw) for tw in tax_key.split("_")))
        idf = math.log((total_labels + 1) / (n + 1)) + 1
        if idf >= 2.0:  # require minimum discriminability
            raw_idfs[rw] = idf

    if not raw_idfs:
        return "unknown"

    total_idf = sum(raw_idfs.values())
    best_key, best_score = "", 0.0
    for tax_key in all_snake_keys:
        tax_words = tax_key.split("_")
        matched_idf = 0.0
        for rw, idf_val in raw_idfs.items():
            for tw in tax_words:
                if _word_subset(rw, tw):
                    matched_idf += idf_val
                    break
        if matched_idf > 0:
            score = matched_idf / total_idf
            if score > best_score:
                best_score = score
                best_key = tax_key

    if best_key in l1_map:
        return l1_map[best_key]
    if best_key in l2_to_l1:
        return l2_to_l1[best_key][1]
    return "unknown"


class GemmaDocLabelModel:
    """gemma-doc-label HTTP API wrapper.

    Calls the Gemma4:e2b Ollama classification service at port 8003.
    Requires gemma-doc-label service to be running.
    """

    def __init__(self, variant: str = "gemma4:e2b", device: str = "cpu",
                 quantization: str | None = None,
                 api_url: str = GEMMA_API_URL):
        self._api_url = api_url
        self._variant = variant
        self._taxonomy = _build_taxonomy()

    @property
    def name(self) -> str:
        return f"gemma-doc-label ({self._variant})"

    @property
    def param_count(self) -> int:
        return 2_000_000_000  # ~2B

    def warmup(self, n: int = 1) -> None:
        try:
            self.predict_labels(["Warmup."], [], {})
        except Exception:
            pass

    def predict(self, texts: List[str]) -> List[str]:
        return ["" for _ in texts]

    def predict_labels(
        self,
        texts: List[str],
        l1_options: List[str],
        l2_options: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for text in texts:
            try:
                result = self._classify_one(text)
            except Exception as e:
                result = {"l1": "error", "l2": str(e)[:50]}
            results.append(result)
        return results

    def _classify_one(self, text: str) -> Dict[str, str]:
        data = json.dumps({"text": text, "filename": "benchmark.txt"}).encode("utf-8")
        req = urllib.request.Request(
            self._api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=GEMMA_TIMEOUT_SEC) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        result = raw.get("result", {})

        l1_obj = result.get("l1", {})
        l2_obj = result.get("l2", {})
        l1_label = l1_obj.get("label", "") if isinstance(l1_obj, dict) else ""
        l2_label = l2_obj.get("label", "") if isinstance(l2_obj, dict) else ""

        # If service-side exact matching failed, extract raw label from Ollama output
        if not l1_label:
            l1_label = self._extract_raw_label(result, "l1")
            l1_label = _fuzzy_match(l1_label, self._taxonomy)
        if not l2_label:
            l2_label = self._extract_raw_label(result, "l2")
            l2_label = _fuzzy_match(l2_label, self._taxonomy)

        return {"l1": l1_label or "unknown", "l2": l2_label or "unknown"}

    def _extract_raw_label(self, result: dict, level: str) -> str:
        """Extract raw Gemma label from raw_outputs when service matching failed."""
        raw_outputs = result.get("raw_outputs", {})
        calls = raw_outputs.get(f"{level}_calls", [])
        for call in reversed(calls):  # Last call first (may have retried)
            parsed = call.get("parsed", {})
            if isinstance(parsed, dict) and parsed.get(level, "").strip():
                return parsed[level].strip()
        return ""
