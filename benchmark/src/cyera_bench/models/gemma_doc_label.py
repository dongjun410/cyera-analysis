import json
import urllib.request
from typing import Dict, List


GEMMA_API_URL = "http://127.0.0.1:8003/classify_text"
GEMMA_TIMEOUT_SEC = 600


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

        l1 = raw.get("l1", {})
        l2 = raw.get("l2", {})
        return {
            "l1": l1.get("label", "unknown") if isinstance(l1, dict) else str(l1),
            "l2": l2.get("label", "unknown") if isinstance(l2, dict) else str(l2),
        }
