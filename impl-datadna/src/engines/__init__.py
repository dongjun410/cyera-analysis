"""Six parallel classification engines with uniform interface."""

from src.engines.e1_regex import E1RegexEngine
from src.engines.e2_template import E2TemplateEngine
from src.engines.e3_ml import E3MLEngine
from src.engines.e4_knn import E4kNNEngine
from src.engines.e5_structural import E5StructuralEngine
from src.engines.e6_llm import E6LLMEngine

__all__ = [
    "E1RegexEngine",
    "E2TemplateEngine",
    "E3MLEngine",
    "E4kNNEngine",
    "E5StructuralEngine",
    "E6LLMEngine",
]
