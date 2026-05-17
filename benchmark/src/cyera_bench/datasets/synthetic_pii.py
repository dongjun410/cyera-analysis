import random
from typing import List, Dict, Tuple
from datasets import Dataset
from cyera_bench.datasets.base import BaseDataset

_GENERATORS: Dict[str, callable] = {}


def _register(entity_type: str):
    def decorator(fn):
        _GENERATORS[entity_type] = fn
        return fn
    return decorator


@_register("CREDIT_CARD")
def _gen_cc():
    digits = [str(random.randint(0, 9)) for _ in range(16)]
    return "".join(digits)


@_register("SSN")
def _gen_ssn():
    return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"


@_register("EMAIL")
def _gen_email():
    users = ["john.doe", "jane.smith", "admin", "support", "info"]
    domains = ["company.com", "example.org", "acme.co", "corp.net"]
    return f"{random.choice(users)}@{random.choice(domains)}"


@_register("PHONE")
def _gen_phone():
    return f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


@_register("API_KEY")
def _gen_api_key():
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    key = "".join(random.choice(chars) for _ in range(40))
    return f"sk-{key}"


@_register("PASSWORD")
def _gen_password():
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*") for _ in range(12))


@_register("BANK_ACCOUNT")
def _gen_bank():
    return f"{random.randint(10000000, 99999999)}"


@_register("IP_ADDRESS")
def _gen_ip():
    return f"{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


@_register("URL")
def _gen_url():
    paths = ["/api/v1/users", "/admin/login", "/docs/internal", "/health"]
    return f"https://{random.choice(['internal', 'staging', 'prod'])}.company.com{random.choice(paths)}"


@_register("DATE_OF_BIRTH")
def _gen_dob():
    return f"{random.randint(1900, 2010)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"


@_register("DRIVERS_LICENSE")
def _gen_dl():
    return f"DL-{''.join(str(random.randint(0,9)) for _ in range(10))}"


@_register("PASSPORT")
def _gen_passport():
    return f"{chr(random.randint(65, 90))}{random.randint(1000000, 9999999)}"


_TEMPLATES = [
    "The {entity_type} is {value}.",
    "Please update my {entity_type} from {value}.",
    "{value} is the {entity_type} for account #12345.",
    "Record: {entity_type}={value}",
    "Contact info: {value} ({entity_type}).",
    "Customer's {entity_type} changed to {value}.",
    "ID verification: {entity_type} {value}.",
    "This is a test {entity_type}: {value}.",
    "employee record: {value} ({entity_type})",
    "The system detected {entity_type}: {value}",
    "Input field {entity_type} contains: {value}",
    "Legacy data: {entity_type} was {value}",
]

_EDGE_TEMPLATES = [
    ("This is a test {entity_type} {value}.", True),
    ("The sample {entity_type} is {value}.", True),
    ("{value} is a valid {entity_type}.", True),
    ("The number {value} looks like a {entity_type}.", False),
    ("Invalid {entity_type}: {value}.", True),
]


class SyntheticPiiDataset(BaseDataset):
    O_TAG = 0
    B_TAG = 1
    I_TAG = 2

    def __init__(self, size: int = 1000, seed: int = 42, include_edge_cases: bool = False):
        self._size = size
        self._seed = seed
        self._include_edge_cases = include_edge_cases
        random.seed(seed)
        self._cache: Dict[str, Dataset] = {}

    @property
    def entity_types(self) -> List[str]:
        return sorted(_GENERATORS.keys())

    def load(self, split: str = "test") -> Dataset:
        if split not in self._cache:
            data = self._generate_all()
            ds = Dataset.from_list(data)
            split_ds = ds.train_test_split(test_size=0.2, seed=self._seed)
            self._cache["train"] = split_ds["train"]
            self._cache["test"] = split_ds["test"]
        return self._cache[split]

    def _generate_all(self) -> List[dict]:
        samples: List[dict] = []
        random.seed(self._seed)

        for _ in range(self._size):
            entity_type = random.choice(list(_GENERATORS.keys()))
            value = _GENERATORS[entity_type]()
            template = random.choice(_TEMPLATES)
            text = template.format(entity_type=entity_type, value=value)
            tokens, tags = self._bio_annotate(text, value)
            samples.append({"tokens": tokens, "ner_tags": tags})

        if self._include_edge_cases:
            for template, is_pii in _EDGE_TEMPLATES:
                for etype in random.sample(list(_GENERATORS.keys()), min(5, len(_GENERATORS))):
                    value = _GENERATORS[etype]()
                    text = template.format(entity_type=etype, value=value)
                    if is_pii:
                        tokens, tags = self._bio_annotate(text, value)
                    else:
                        tokens = text.split()
                        tags = [self.O_TAG] * len(tokens)
                    samples.append({"tokens": tokens, "ner_tags": tags})

        return samples

    def _bio_annotate(self, text: str, value: str) -> Tuple[List[str], List[int]]:
        words = text.split()
        tags: List[int] = [self.O_TAG] * len(words)
        value_words = value.split()
        for i in range(len(words) - len(value_words) + 1):
            if words[i:i + len(value_words)] == value_words:
                tags[i] = self.B_TAG
                for j in range(1, len(value_words)):
                    tags[i + j] = self.I_TAG
                break
        return words, tags
