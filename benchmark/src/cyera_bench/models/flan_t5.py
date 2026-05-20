from typing import List, Dict, Optional
from transformers import pipeline
from cyera_bench.models.base import BaseModel
from cyera_bench.types import Entity

_MODEL_MAP: Dict[str, Dict[str, str | None]] = {
    "small": {"hf_name": "google/flan-t5-small",  "ner_checkpoint": "agentlans/flan-t5-small-ner"},
    "base":  {"hf_name": "google/flan-t5-base",   "ner_checkpoint": "pepegiallo/flan-t5-base_ner"},
    "large": {"hf_name": "google/flan-t5-large",  "ner_checkpoint": None},
    "xl":    {"hf_name": "google/flan-t5-xl",     "ner_checkpoint": None},
}

_PARAM_COUNTS: Dict[str, int] = {
    "small": 77_000_000,
    "base":  250_000_000,
    "large": 780_000_000,
    "xl":    2_850_000_000,
}


class FlanT5Model(BaseModel):
    def __init__(self, variant: str = "large", device: str = "cuda",
                 quantization: str | None = None, ner_checkpoint: str | None = None):
        if variant not in _MODEL_MAP:
            raise ValueError(f"Unknown variant '{variant}'. Choose: {list(_MODEL_MAP.keys())}")
        self._variant = variant
        self._quantization = quantization
        self._ner_checkpoint = ner_checkpoint
        self._pipe: Optional[object] = None

        # Auto fallback: CPU when CUDA requested but unavailable
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print(f"  [INFO] CUDA not available, falling back to CPU for {variant}")
                    device = "cpu"
            except ImportError:
                print(f"  [INFO] torch not installed, using CPU for {variant}")
                device = "cpu"
        self._device = device

    @property
    def name(self) -> str:
        return _MODEL_MAP[self._variant]["hf_name"]

    @property
    def param_count(self) -> int:
        return _PARAM_COUNTS[self._variant]

    def _load_pipeline(self):
        """Lazily load the HuggingFace pipeline on first use."""
        if self._pipe is not None:
            return

        info = _MODEL_MAP[self._variant]
        checkpoint = self._ner_checkpoint or info["ner_checkpoint"]

        if checkpoint:
            model_kwargs = {}
            if self._quantization == "4bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif self._quantization == "8bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            self._pipe = pipeline(
                "token-classification",
                model=checkpoint,
                aggregation_strategy="simple",
                device=0 if self._device == "cuda" else -1,
                **model_kwargs,
            )
            self._mode = "token-classification"
        else:
            self._pipe = pipeline(
                "text2text-generation",
                model=info["hf_name"],
                device=0 if self._device == "cuda" else -1,
            )
            self._mode = "text2text-generation"

    def predict(self, texts: List[str]) -> List[List[Entity]]:
        if not texts:
            return []

        self._load_pipeline()

        if self._mode == "token-classification":
            return self._predict_token_classification(texts)
        else:
            return self._predict_text2text(texts)

    def _predict_token_classification(self, texts: List[str]) -> List[List[Entity]]:
        results = self._pipe(texts)
        entity_lists: List[List[Entity]] = []

        for per_text_result in results:
            entities = []
            for item in per_text_result:
                entities.append(Entity(
                    type=item["entity_group"],
                    text=item["word"],
                    start=item["start"],
                    end=item["end"],
                    confidence=item["score"],
                ))
            entity_lists.append(entities)

        return entity_lists

    def _predict_text2text(self, texts: List[str]) -> List[List[Entity]]:
        prompts = [f"Extract named entities (person, organization, location, miscellaneous) from the text:\n{t}"
                   for t in texts]
        outputs = self._pipe(prompts, max_new_tokens=128)
        entity_lists: List[List[Entity]] = []

        for output in outputs:
            entities = self._parse_text2text_output(output["generated_text"])
            entity_lists.append(entities)

        return entity_lists

    def _parse_text2text_output(self, text: str) -> List[Entity]:
        entities = []
        tag_map = {"PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}
        for line in text.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                tag, _, value = line.partition(":")
                tag = tag.strip().upper()
                value = value.strip()
                if tag in tag_map and value:
                    entities.append(Entity(type=tag_map[tag], text=value, start=0, end=0, confidence=0.8))
        return entities
