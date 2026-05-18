import json
import re
import urllib.request
from typing import Dict, List


GEMMA_TIMEOUT_SEC = 600
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma4:e2b"

_L1_PROMPT = (
    "You are labeling enterprise documents for a hierarchy classifier.\n\n"
    "Task: Choose EXACTLY ONE L1 label from the list below.\n\n"
    "Allowed L1 labels:\n{l1_lines}\n\n"
    "Document filename:\n{filename}\n\n"
    "Document text:\n{clipped}\n\n"
    "Return STRICT JSON only:\n"
    '{{"l1":"<exact L1 from list>", "confidence":"high|medium|low",'
    ' "rationale":"one short sentence"}}'
)

_L2_PROMPT = (
    "You are labeling enterprise documents for a hierarchy classifier.\n\n"
    "Task: L1 is already fixed. Choose EXACTLY ONE L2 label from this L1's list.\n\n"
    "Fixed L1:\n{l1}\n\n"
    "Allowed L2 labels for this L1:\n{l2_lines}\n\n"
    "Document filename:\n{filename}\n\n"
    "Document text:\n{clipped}\n\n"
    "Return STRICT JSON only:\n"
    '{{"l2":"<exact L2 from list>", "confidence":"high|medium|low",'
    ' "rationale":"one short sentence"}}'
)


def _to_snake(text: str) -> str:
    t = text.lower()
    t = t.replace("&", "and")
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = t.strip("_")
    t = re.sub(r"_+", "_", t)
    return t


class GemmaDocLabelModel:
    """Gemma4:e2b document classification via direct Ollama API calls.

    Uses Ollama /api/generate directly with display-name prompts for L1+L2
    classification, followed by fuzzy label matching on the benchmark side.
    Does NOT use the gemma-doc-label service.
    """

    def __init__(self, variant: str = "gemma4:e2b", device: str = "cpu",
                 quantization: str | None = None,
                 api_url: str = ""):
        self._variant = variant
        # num_gpu: 99=force all layers to GPU, 0=force CPU
        self._num_gpu = 99 if device == "cuda" else 0

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
                result = self._classify_direct(text, l1_options, l2_options)
            except Exception as e:
                result = {"l1": "error", "l2": str(e)[:50]}
            results.append(result)
        return results

    def _classify_direct(self, text: str, l1_options: List[str],
                         l2_options: Dict[str, List[str]]) -> Dict[str, str]:
        """Full classification via direct Ollama calls (bypasses service matching)."""
        # Step 1: L1 classification
        l1_lines = "\n".join(f"- {l}" for l in l1_options)
        prompt = _L1_PROMPT.format(
            l1_lines=l1_lines,
            filename="benchmark.txt",
            clipped=text[:8000],
        )
        raw_l1 = self._call_ollama_json(prompt, key="l1")
        l1 = self._match_l1_label(raw_l1, l1_options)

        # Step 2: L2 classification (only if L1 matched)
        l2 = "unknown"
        if l1 and l1 != "unknown":
            l2_candidates = l2_options.get(l1, [])
            if l2_candidates:
                l2_lines = "\n".join(f"- {c}" for c in l2_candidates)
                prompt = _L2_PROMPT.format(
                    l1=l1,
                    l2_lines=l2_lines,
                    filename="benchmark.txt",
                    clipped=text[:8000],
                )
                raw_l2 = self._call_ollama_json(prompt, key="l2")
                l2 = self._match_l2_label(raw_l2, l2_candidates)

        return {"l1": l1 or "unknown", "l2": l2 or "unknown"}

    def _call_ollama_json(self, prompt: str, key: str) -> str:
        """Call Ollama /api/generate and extract the value for `key` from JSON response."""
        opts = {"temperature": 0.0, "num_predict": 64, "num_gpu": self._num_gpu}
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": opts,
        }
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=GEMMA_TIMEOUT_SEC) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
        answer = raw.get("response", "").strip()
        parsed = self._parse_json(answer)
        return parsed.get(key, "") if isinstance(parsed, dict) else ""

    @staticmethod
    def _match_l1_label(raw_label: str, candidates: List[str]) -> str:
        """Fuzzy match raw L1 label to the closest candidate (display names)."""
        if not raw_label:
            return "unknown"
        raw_snake = _to_snake(raw_label)
        # Exact match
        for c in candidates:
            if _to_snake(c) == raw_snake:
                return c
        # Substring match (raw in candidate or vice versa)
        for c in candidates:
            c_snake = _to_snake(c)
            if (len(raw_snake) >= 3 and raw_snake in c_snake) or c_snake in raw_snake:
                return c
        # Word overlap
        raw_words = set(raw_snake.split("_"))
        for c in candidates:
            c_words = set(_to_snake(c).split("_"))
            if raw_words & c_words:
                return c
        return "unknown"

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON object from text, with fallback."""
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {}

    @staticmethod
    def _match_l2_label(raw_label: str, candidates: List[str]) -> str:
        """Fuzzy match raw L2 label to the closest candidate."""
        if not raw_label:
            return "unknown"
        raw_snake = _to_snake(raw_label)
        # Exact match
        for c in candidates:
            if _to_snake(c) == raw_snake:
                return c
        # Substring match
        for c in candidates:
            c_snake = _to_snake(c)
            if raw_snake in c_snake or c_snake in raw_snake:
                return c
        # Word-level match: any raw word found in candidate words
        raw_words = set(raw_snake.split("_"))
        for c in candidates:
            c_words = set(_to_snake(c).split("_"))
            if raw_words & c_words:
                return c
        return "unknown"
