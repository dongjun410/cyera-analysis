from typing import Dict, List, Optional

from cyera_bench.models.flan_t5 import FlanT5Model


DEFAULT_L1_PROMPT = (
    "Classify the following document into exactly one of these categories:\n"
    "{l1_options}\n\n"
    "Document:\n{text}\n\n"
    "Category:"
)

DEFAULT_L2_PROMPT = (
    "The document belongs to \"{l1_pred}\". "
    "Choose the most specific subcategory from:\n"
    "{l2_options}\n\n"
    "Document:\n{text}\n\n"
    "Subcategory:"
)

DEFAULT_SINGLE_PROMPT = (
    "Classify the following document into a primary and sub category.\n\n"
    "Primary categories:\n{l1_options}\n\n"
    "For each primary category, the available subcategories are:\n"
    "{l1_l2_map}\n\n"
    "Document:\n{text}\n\n"
    "Output the result as:\nL1: <category>\nL2: <subcategory>"
)


class FlanT5ClassificationModel(FlanT5Model):
    """FLAN-T5 prompt-based document classification model.

    Uses direct model.generate() for L1/L2 document label prediction.
    """

    def __init__(
        self,
        variant: str = "large",
        device: str = "cuda",
        quantization: str | None = None,
        prompt_style: str = "two_step",
        max_input_chars: int = 8000,
    ):
        super().__init__(
            variant=variant,
            device=device,
            quantization=quantization,
        )
        self._prompt_style = prompt_style
        self._max_input_chars = max_input_chars

    def _load_pipeline(self):
        """Override: load T5 model+tokenizer directly (pipeline not supported
        for T5 in recent transformers text-generation task)."""
        if self._pipe is not None:
            return
        from cyera_bench.models.flan_t5 import _MODEL_MAP
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        info = _MODEL_MAP[self._variant]
        model_name = info["hf_name"]
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if self._device == "cuda":
            self._model = self._model.to("cuda")
        self._pipe = True  # mark as loaded

    def _generate(self, prompts: List[str], max_new_tokens: int) -> List[dict]:
        import torch
        device = self._model.device
        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [{"generated_text": d.strip()} for d in decoded]

    def predict_labels(
        self,
        texts: List[str],
        l1_options: List[str],
        l2_options: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        if not texts:
            return []

        self._load_pipeline()

        if self._prompt_style == "single_step":
            return self._predict_single_step(texts, l1_options, l2_options)
        return self._predict_two_step(texts, l1_options, l2_options)

    def _predict_two_step(
        self,
        texts: List[str],
        l1_options: List[str],
        l2_options: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        l1_str = "\n".join(f"- {l}" for l in l1_options)

        # Step 1: Predict L1
        prompts_l1 = [
            DEFAULT_L1_PROMPT.format(
                l1_options=l1_str,
                text=t[:self._max_input_chars],
            )
            for t in texts
        ]
        l1_outputs = self._generate(prompts_l1, max_new_tokens=32)

        # Step 2: Predict L2 given L1
        results: List[Dict[str, str]] = []
        l2_prompts: List[str] = []
        l2_indices: List[int] = []

        for i, (text, l1_out) in enumerate(zip(texts, l1_outputs)):
            l1_pred = l1_out["generated_text"].strip()
            l1_pred = self._fuzzy_match(l1_pred, l1_options)

            if l1_pred and l1_pred in l2_options:
                l2_list = l2_options[l1_pred]
                l2_str = "\n".join(f"- {l}" for l in l2_list)
                l2_prompts.append(
                    DEFAULT_L2_PROMPT.format(
                        l1_pred=l1_pred,
                        l2_options=l2_str,
                        text=text[:self._max_input_chars],
                    )
                )
                l2_indices.append(i)
            else:
                l1_pred = l1_pred or "unknown"
            results.append({"l1": l1_pred, "l2": ""})

        if l2_prompts:
            l2_outputs = self._generate(l2_prompts, max_new_tokens=64)
            for idx, l2_out in zip(l2_indices, l2_outputs):
                l1_pred = results[idx]["l1"]
                l2_pred = l2_out["generated_text"].strip()
                l2_list = l2_options.get(l1_pred, [])
                results[idx]["l2"] = self._fuzzy_match(l2_pred, l2_list)

        return results

    def _predict_single_step(
        self,
        texts: List[str],
        l1_options: List[str],
        l2_options: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        l1_str = "\n".join(f"- {l}" for l in l1_options)
        l1_l2_lines = []
        for l1_name in l1_options:
            sub_list = l2_options.get(l1_name, [])
            l1_l2_lines.append(
                f"  {l1_name}: {', '.join(sub_list[:5])}"
                + ("..." if len(sub_list) > 5 else "")
            )
        l1_l2_str = "\n".join(l1_l2_lines)

        prompts = [
            DEFAULT_SINGLE_PROMPT.format(
                l1_options=l1_str,
                l1_l2_map=l1_l2_str,
                text=t[:self._max_input_chars],
            )
            for t in texts
        ]
        outputs = self._generate(prompts, max_new_tokens=128)

        results: List[Dict[str, str]] = []
        for out in outputs:
            raw = out["generated_text"].strip()
            l1_pred = ""
            l2_pred = ""
            for line in raw.split("\n"):
                line = line.strip()
                if line.lower().startswith("l1:") or line.lower().startswith("l1"):
                    l1_pred = line.split(":", 1)[-1].split("", 1)[-1].strip()
                elif line.lower().startswith("l2:") or line.lower().startswith("l2"):
                    l2_pred = line.split(":", 1)[-1].split("", 1)[-1].strip()

            l1_pred = self._fuzzy_match(l1_pred, l1_options)
            l2_list = l2_options.get(l1_pred, [])
            l2_pred = self._fuzzy_match(l2_pred, l2_list)
            results.append({"l1": l1_pred or "unknown", "l2": l2_pred or "unknown"})

        return results

    def _fuzzy_match(self, prediction: str, candidates: List[str]) -> str:
        """Match prediction to closest candidate label."""
        if not prediction or not candidates:
            return prediction or ""

        pred_lower = prediction.lower().strip().rstrip(".")

        # Exact match
        for c in candidates:
            if c.lower() == pred_lower:
                return c

        # Substring match (prediction is a substring of candidate or vice versa)
        for c in candidates:
            if pred_lower in c.lower() or c.lower() in pred_lower:
                return c

        # If nothing matches, return the prediction as-is (will be marked wrong)
        return prediction.strip()
