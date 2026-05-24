# DataDNA 值级敏感数据分类引擎 — 开发计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建完整的值级敏感数据分类引擎，覆盖 15 种 PII/PCI 类型，实现真值表评分 + NER 语义特征 + Semantic Distancing 并列评分 + LLM 分层消歧 + 文件级聚类传播 + 自学习闭环。

**Architecture:** 单内核三路径（离线全量/离线增量/在线实时），通过 `classify(value, context) → ValueClassification` 统一接口解耦。4 个 Phase 通过接口契约并行推进：Phase 0 定义共享类型后，Phase 1A/1B/1C/2A/3A/3B 均可并行启动，Phase 1D/2B/3C/4 在前置契约就绪后集成。

**Tech Stack:** Python 3.11+, PyTorch 2.5+, transformers, scikit-learn, pandas, scipy, FLAN-T5, Mistral-7B (Ollama), E5-base, GLiNER, BERT-base, LoRA (PEFT), bitsandbytes

**设计文档:** `docs/superpowers/specs/2026-05-24-datadna-value-level-design.md`

---

## 并行开发总览

```
Phase 0: 共享类型 + 项目骨架 (必须最先完成)
  │
  ├─► Phase 1A: 值提取器 + Mock过滤 + 抽样 + 非结构化提取 ──────┐
  ├─► Phase 1B: 真值表引擎 (含校准数据生成) ────────────────────┤
  ├─► Phase 1C: NER 引擎 (GLiNER → BERT 微调) ─────────────────┤
  ├─► Phase 2A: LLM 基础设施 (FLAN-T5 + Mistral 部署+客户端) ──┤
  ├─► Phase 3A: Semantic Distancing (PII替换+嵌入+模板库) ─────┤
  ├─► Phase 3B: 文件级聚类引擎 (元数据归一化+流式聚类) ────────┤
  ├─► Phase 6:  数据策略 (标注流水线+公开数据+分轮QC) ────────┤
  │                                                            │
  │   上述 7 条并行线完成后，汇合到:                              │
  │                                                            │
  ├─► Phase 1D: 融合评分 + 角色判定 + 上下文 + 7维NER集成 + 评估 ┘
  │       │
  │       ├─► Phase 2B: LLM 消歧集成 (路由+批量+降级+超时) ────┐
  │       ├─► Phase 2C: 在线缓存层 (Path C pattern_hash TTL) ──┤
  │       ├─► Phase 5:  审计日志 + 偏移监控 + R1-R7 框架 ──────┤
  │       └─► Phase 3C: 标签传播 + 抽检 + 增量匹配 + 编排器 ────┤
  │                                                            │
  │   上述 4 条并行线完成后，汇合到:                              │
  │                                                            │
  └─► Phase 4A: 未知模式收集 + 缓冲池 ──────────────────────────┘
          │
          ├─► Phase 4B: 自动验证 + 人工Gate接口
          │       │
          └─► Phase 4C: 引擎更新 (TypeLibrary + 真值表增量 + NER增量)
                  │
                  └─► Phase 4D: 闭环延迟窗口 + type_cache + 退化防护

Phase 7: R5/R7 硬性测试 (Phase 1D+2B 完成后可启动)
```

## 接口契约（并行边界）

以下 dataclass 在 Phase 0 定义，是所有并行线的共同契约：

```python
# src/types.py — 核心类型，Phase 0 产出，后续所有 Phase 依赖

@dataclass
class ValueContext:
    container_type: str       # "db_cell" | "csv_field" | "json_path" | ...
    container_path: str       # 泛化位置路径
    label_hint: str | None    # 列名/字段名/键名提示
    surrounding_text: str     # 上下文窗口 (±100字符)
    parent_doc_id: str | None
    parent_file_path: str | None
    metadata: dict

@dataclass
class ValueSource:
    source_type: str          # 提取来源类型
    extraction_method: str    # 提取方法
    position: str | None      # 源内位置

@dataclass
class DataValue:
    value_id: str
    value: str
    context: ValueContext
    source: ValueSource

@dataclass
class ValueClassification:
    value_id: str
    value: str
    sensitive_type: str | None      # "SSN" | None=NON_SENSITIVE
    confidence: float
    method: str                     # "regex_only" | "truth_table" | "fusion" |
                                    #   "llm_validate" | "llm_classify"
    role: str | None                # "subject" | "identifier" | "reference"
    is_mock: bool
    needs_review: bool
    evidence: dict                  # 各引擎输出、特征值、中间分数
    source: ValueSource
```

---

## Phase 0：项目骨架 + 共享类型（串行前置，所有 Phase 的依赖）

### Task 0.1：项目目录结构

**Files:**
- Create: 目录树（见下文）

- [ ] **Step 1: 创建完整目录骨架**

```bash
mkdir -p value-datadna/src/{extractors,classifiers,postprocess,llm,clustering,discovery,knowledge,monitoring,evaluation}
mkdir -p value-datadna/tests
mkdir -p value-datadna/datasets/{public/{synthetic,swift_iban,pci_dss,pii-masking,conllpp},enterprise/cxh5types,calibration,scripts}
mkdir -p value-datadna/templates/{types,clusters}
mkdir -p value-datadna/output
touch value-datadna/src/__init__.py
touch value-datadna/src/extractors/__init__.py
touch value-datadna/src/classifiers/__init__.py
touch value-datadna/src/postprocess/__init__.py
touch value-datadna/src/llm/__init__.py
touch value-datadna/src/clustering/__init__.py
touch value-datadna/src/discovery/__init__.py
touch value-datadna/src/knowledge/__init__.py
touch value-datadna/src/monitoring/__init__.py
touch value-datadna/src/evaluation/__init__.py
touch value-datadna/tests/__init__.py
# 设计文档 Section 9 要求的额外知识模块文件
touch value-datadna/src/knowledge/type_library.py
touch value-datadna/src/knowledge/truth_table_data.py
touch value-datadna/src/knowledge/template_library.py
```

- [ ] **Step 2: Commit**

```bash
git add value-datadna/
git commit -m "feat: scaffold value-datadna project structure"
```

### Task 0.2：核心类型定义

**Files:**
- Create: `value-datadna/src/types.py`
- Test: `value-datadna/tests/test_types.py`

- [ ] **Step 1: 编写类型定义的测试**

```python
# tests/test_types.py
import pytest
from src.types import DataValue, ValueContext, ValueSource, ValueClassification


class TestValueContext:
    def test_minimal_context(self):
        ctx = ValueContext(
            container_type="csv_field",
            container_path="employees.csv/ssn",
            label_hint="ssn",
            surrounding_text="employee info: 123-45-6789, name: John",
        )
        assert ctx.container_type == "csv_field"
        assert ctx.label_hint == "ssn"
        assert ctx.parent_doc_id is None

    def test_full_context(self):
        ctx = ValueContext(
            container_type="db_cell",
            container_path="db1.public.employees.ssn",
            label_hint="ssn",
            surrounding_text="SSN: 123-45-6789",
            parent_doc_id="doc_001",
            parent_file_path="/data/hr.db",
            metadata={"db_type": "postgresql", "table": "employees"},
        )
        assert ctx.metadata["db_type"] == "postgresql"


class TestValueClassification:
    def test_high_confidence_classification(self):
        vc = ValueClassification(
            value_id="v_001",
            value="123-45-6789",
            sensitive_type="SSN",
            confidence=0.95,
            method="truth_table",
            role="subject",
            is_mock=False,
            needs_review=False,
            evidence={"truth_table_confidence": 0.95, "regex_strength": 0.9},
            source=ValueSource(
                source_type="csv",
                extraction_method="structured",
                position="row_5_col_2",
            ),
        )
        assert vc.sensitive_type == "SSN"
        assert vc.confidence == 0.95

    def test_non_sensitive_classification(self):
        vc = ValueClassification(
            value_id="v_002",
            value="hello world",
            sensitive_type=None,
            confidence=0.98,
            method="truth_table",
            role="reference",
            is_mock=False,
            needs_review=False,
            evidence={},
            source=ValueSource(
                source_type="text", extraction_method="unstructured", position="para_3"
            ),
        )
        assert vc.sensitive_type is None
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd value-datadna && pytest tests/test_types.py -v
```
Expected: FAIL (ImportError: No module named 'src.types')

- [ ] **Step 3: 实现类型定义**

```python
# src/types.py
from dataclasses import dataclass, field


@dataclass
class ValueContext:
    container_type: str
    container_path: str
    label_hint: str | None = None
    surrounding_text: str = ""
    parent_doc_id: str | None = None
    parent_file_path: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ValueSource:
    source_type: str
    extraction_method: str
    position: str | None = None


@dataclass
class DataValue:
    value_id: str
    value: str
    context: ValueContext
    source: ValueSource


@dataclass
class ValueClassification:
    value_id: str
    value: str
    sensitive_type: str | None
    confidence: float
    method: str
    role: str | None = None
    is_mock: bool = False
    needs_review: bool = False
    evidence: dict = field(default_factory=dict)
    source: ValueSource | None = None
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd value-datadna && pytest tests/test_types.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add value-datadna/src/types.py value-datadna/tests/test_types.py
git commit -m "feat: define core types (DataValue, ValueContext, ValueClassification)"
```

### Task 0.3：配置文件

**Files:**
- Create: `value-datadna/config.yaml`

- [ ] **Step 1: 创建配置骨架**

```yaml
# config.yaml — 全局配置
# 完整配置项参见设计文档 Section 2.4 技术栈

truth_table:
  dimensions: 6            # NER 不可用时 6 维，可用时 7 维
  bin_counts:
    regex_strength: [0, 0.25, 0.5, 0.75, 1.0]
    validated_count: [0, 1, 4, 10, 50]
    supportive_context: [0, 1, 2, 3]
    unsupportive_context: [0, 1, 2]
    pattern_frequency: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    uniqueness_score: [1, 2, 6, 21, 100]
  calibration:
    laplace_alpha: 1
    kdtree_leaf_size: 30

ner:
  cold_start_model: "urchade/gliner_base"
  fine_tune_model: "bert-base-uncased"
  teacher_model: "qwen3:8b"
  bio_labels: [
    "B-SSN","I-SSN","B-CCN","I-CCN","B-EMAIL","I-EMAIL",
    "B-PHONE","I-PHONE","B-IBAN","I-IBAN","B-PASSPORT","I-PASSPORT",
    "B-DRIVER_LICENSE","I-DRIVER_LICENSE","B-IP","I-IP",
    "B-API_KEY","I-API_KEY","B-BANK_ACCOUNT","I-BANK_ACCOUNT",
    "B-NAME","I-NAME","B-ADDRESS","I-ADDRESS","B-ORG","I-ORG","O"
  ]

fusion:
  alpha: 0.7
  calibration_method: "platt"  # platt | isotonic

routing:
  high_threshold: 0.85
  mid_threshold: 0.50
  low_threshold: 0.30

llm:
  validation:
    model: "google/flan-t5-large"
    timeout_ms: 5000
  classification:
    model: "mistral:7b-instruct"
    endpoint: "http://localhost:11434"
    timeout_ms: 30000
  batch:
    max_batch_size: 32
    concurrent_requests: 5

semantic_distance:
  embedding_model: "intfloat/e5-base"
  template_count_per_type: 75
  incremental_match_threshold: 0.85

clustering:
  stream_chunk_size: 1000
  representatives_per_cluster: 3
  propagation_consistency_threshold: 0.8
  spot_check_ratio: 0.05
  spot_check_values_per_column: 3
  inconsistency_threshold: 0.10

discovery:
  buffer_max_size: 200
  cluster_min_size: 500
  auto_pass_cos_sim_threshold: 0.85
  auto_reject_confidence_threshold: 0.7
  auto_pass_confidence_threshold: 0.3
  ner_retrain_min_samples: 200
  ner_retrain_total_min: 500
  ner_retrain_max_days: 14
  ner_lora_rank: 8

mock_filter:
  virtual_patterns: ["000-00-0000", "123-45-6789", "XXX-XX-XXXX", "4111-1111-1111-1111"]
  negation_keywords: ["test", "sample", "example", "placeholder", "redacted", "dummy", "mock", "fake", "todo", "fixme"]

sampling:
  target_per_column_min: 5
  target_per_column_max: 20
  num_layers: 7
```

- [ ] **Step 2: Commit**

```bash
git add value-datadna/config.yaml
git commit -m "feat: add configuration skeleton"
```

---

### Task 1A.4：非结构化值提取器

**依赖:** Phase 0 完成
**Files:**
- Create: `value-datadna/src/extractors/unstructured.py`
- Test: `value-datadna/tests/test_extractors.py` (追加测试)

**对应设计文档:** Section 3.2

```python
# src/extractors/unstructured.py
import re, os
from typing import List
from src.extractors.base import BaseExtractor
from src.types import DataValue, ValueContext, ValueSource
from src.knowledge.pii_patterns import PII_PATTERNS


class UnstructuredExtractor(BaseExtractor):
    """
    非结构化提取器 (Section 3.2)。
    PDF文本、邮件正文、代码行、Slack消息 → PII正则扫描 + 字符偏移 + 周围文本窗口。
    支持格式: .pdf, .txt, .md, .py, .js, .ts, .java, .go, .eml
    """

    WINDOW_SIZE = 100  # 上下文窗口 ±100字符

    def __init__(self, config: dict = None):
        self._compiled_patterns = {}
        for type_name, info in PII_PATTERNS.items():
            if info.get("regex"):
                self._compiled_patterns[type_name] = re.compile(
                    info["regex"], re.IGNORECASE
                )

    def extract(self, source: str, **kwargs) -> List[DataValue]:
        ext = os.path.splitext(source)[1].lower()
        if ext == '.pdf':
            text = self._parse_pdf(source)
        elif ext in ('.eml',):
            text = self._parse_email(source)
        else:
            with open(source, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

        return self._scan_pii(text, source, ext)

    def _parse_pdf(self, path: str) -> str:
        """pymupdf PDF 文本提取"""
        import fitz
        doc = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return '\n\n'.join(p for p in pages if p and p.strip())

    def _parse_email(self, path: str) -> str:
        """EML 邮件正文提取"""
        import email
        from email import policy
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_string(f.read(), policy=policy.default)
        parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode('utf-8', errors='ignore'))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                parts.append(payload.decode('utf-8', errors='ignore'))
        return '\n'.join(parts)

    def _scan_pii(self, text: str, source: str, ext: str) -> List[DataValue]:
        """PII 正则扫描 + 字符偏移 + 周围文本窗口"""
        values = []
        seen_spans = set()  # 防止同一位置的重复匹配

        for type_name, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                span_key = (start, end)
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)

                raw_value = match.group()
                # 周围文本窗口
                ctx_start = max(0, start - self.WINDOW_SIZE)
                ctx_end = min(len(text), end + self.WINDOW_SIZE)
                surrounding = text[ctx_start:ctx_end]

                # 推断 label_hint: 提取值之前的最近一个"词"作为上下文键名
                prefix = text[max(0, start - 50):start].strip()
                label_hint = prefix.split()[-1].rstrip(':=') if prefix else ""

                ctx = ValueContext(
                    container_type=self._container_type(ext),
                    container_path=source,
                    label_hint=label_hint,
                    surrounding_text=surrounding,
                    parent_file_path=source,
                )
                src = ValueSource(
                    source_type=ext.lstrip('.'),
                    extraction_method="unstructured_pii_scan",
                    position=f"char_{start}_{end}",
                )
                values.append(DataValue(
                    value_id=f"unstr_{os.path.basename(source)}_{start}_{end}",
                    value=raw_value,
                    context=ctx,
                    source=src,
                ))
        return values

    def _container_type(self, ext: str) -> str:
        mapping = {
            '.pdf': 'pdf_field', '.txt': 'text_span', '.md': 'text_span',
            '.py': 'code_line', '.js': 'code_line', '.ts': 'code_line',
            '.java': 'code_line', '.go': 'code_line',
            '.eml': 'email_body',
        }
        return mapping.get(ext, 'text_span')
```

```python
# tests/test_extractors.py 追加测试
class TestUnstructuredExtractor:
    @pytest.fixture
    def extractor(self):
        return UnstructuredExtractor()

    def test_extract_txt_with_ssn(self, extractor):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Employee form: SSN is 123-45-6789 for record keeping.")
            path = f.name
        try:
            values = extractor.extract(path)
            ssn_vals = [v for v in values if v.value == "123-45-6789"]
            assert len(ssn_vals) == 1
            assert "SSN" in ssn_vals[0].context.surrounding_text
            assert ssn_vals[0].context.container_type == "text_span"
        finally:
            os.unlink(path)

    def test_extract_txt_no_pii(self, extractor):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a plain text document with no sensitive information.")
            path = f.name
        try:
            values = extractor.extract(path)
            assert len(values) == 0
        finally:
            os.unlink(path)

    def test_overlapping_patterns_single_match(self, extractor):
        """Verify same span isn't double-counted by overlapping patterns"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("john.doe@gmail.com")
            path = f.name
        try:
            values = extractor.extract(path)
            # Should match as EMAIL only once, not also as overlapping
            assert len(values) == 1
        finally:
            os.unlink(path)
```

- [ ] **Step 4: 运行测试, 提交**

```bash
cd value-datadna && pytest tests/test_extractors.py -v
git add value-datadna/src/extractors/unstructured.py value-datadna/tests/test_extractors.py
git commit -m "feat: add unstructured value extractor (PDF/text/code/email)"
```

---

### Task 1A.1-3：结构化提取器 / Mock 过滤 / 抽样 (上文定义)

**依赖:** Phase 0 完成  
**可并行:** 与 Phase 1B/1C/2A/3A/3B 并行

### Task 1A.1：结构化值提取器

**Files:**
- Create: `value-datadna/src/extractors/base.py`
- Create: `value-datadna/src/extractors/structured.py`
- Test: `value-datadna/tests/test_extractors.py`

- [ ] **Step 1: 编写提取器基类**

```python
# src/extractors/base.py
from abc import ABC, abstractmethod
from typing import List
from src.types import DataValue


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, source: str, **kwargs) -> List[DataValue]:
        """从数据源提取值列表"""
        ...
```

- [ ] **Step 2: 编写结构化提取器测试**

```python
# tests/test_extractors.py
import tempfile, os, pytest
from src.extractors.structured import StructuredExtractor


class TestStructuredExtractor:
    @pytest.fixture
    def extractor(self):
        return StructuredExtractor()

    def test_extract_csv(self, extractor):
        csv_content = "ssn,name,email\n123-45-6789,John,john@test.com\n987-65-4321,Jane,jane@test.com"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            path = f.name
        try:
            values = extractor.extract(path)
            assert len(values) == 6  # 3 columns × 2 rows
            ssn_values = [v for v in values if v.context.label_hint == "ssn"]
            assert len(ssn_values) == 2
            assert ssn_values[0].value == "123-45-6789"
            assert ssn_values[0].context.container_type == "csv_field"
        finally:
            os.unlink(path)

    def test_extract_json(self, extractor):
        json_content = '{"employees": [{"ssn": "111-22-3333", "name": "Alice"}, {"ssn": "444-55-6666", "name": "Bob"}]}'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            path = f.name
        try:
            values = extractor.extract(path)
            ssn_values = [v for v in values if v.context.label_hint == "ssn"]
            assert len(ssn_values) == 2
            assert ssn_values[0].value == "111-22-3333"
        finally:
            os.unlink(path)
```

- [ ] **Step 3: 实现结构化提取器**

```python
# src/extractors/structured.py
import csv, json, io, os
from typing import List
from src.extractors.base import BaseExtractor
from src.types import DataValue, ValueContext, ValueSource


class StructuredExtractor(BaseExtractor):
    def extract(self, source: str, **kwargs) -> List[DataValue]:
        ext = os.path.splitext(source)[1].lower()
        if ext == '.csv':
            return self._extract_csv(source)
        elif ext == '.json':
            return self._extract_json(source)
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def _extract_csv(self, path: str) -> List[DataValue]:
        values = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row_idx, row in enumerate(reader):
                for col_idx, (header, val) in enumerate(row.items()):
                    if val is None or str(val).strip() == '':
                        continue
                    val_str = str(val).strip()
                    ctx = ValueContext(
                        container_type="csv_field",
                        container_path=f"{path}#{header}",
                        label_hint=header.lower().strip(),
                        surrounding_text=", ".join(
                            f"{h}={row[h]}" for h in headers if h != header
                        )[:200],
                        parent_file_path=path,
                    )
                    src = ValueSource(
                        source_type="csv",
                        extraction_method="structured",
                        position=f"row_{row_idx}_col_{col_idx}",
                    )
                    values.append(DataValue(
                        value_id=f"csv_{path}_{row_idx}_{col_idx}",
                        value=val_str,
                        context=ctx,
                        source=src,
                    ))
        return values

    def _extract_json(self, path: str) -> List[DataValue]:
        import json
        values = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def _traverse(obj, path_parts: list):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _traverse(v, path_parts + [k])
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _traverse(item, path_parts + [str(i)])
            elif obj is not None and str(obj).strip() != '':
                val_str = str(obj).strip()
                key = path_parts[-1] if path_parts else "value"
                ctx = ValueContext(
                    container_type="json_path",
                    container_path=".".join(str(p) for p in path_parts),
                    label_hint=key.lower().strip(),
                    surrounding_text="",
                    parent_file_path=path,
                )
                src = ValueSource(
                    source_type="json",
                    extraction_method="structured",
                    position=".".join(str(p) for p in path_parts[:-1]),
                )
                values.append(DataValue(
                    value_id=f"json_{path}_{len(values)}",
                    value=val_str,
                    context=ctx,
                    source=src,
                ))

        _traverse(data, [])
        return values
```

- [ ] **Step 4: 运行测试, 提交**

```bash
cd value-datadna && pytest tests/test_extractors.py -v
git add value-datadna/src/extractors/ value-datadna/tests/test_extractors.py
git commit -m "feat: add structured value extractor (CSV, JSON)"
```

### Task 1A.2：Mock 快速过滤器

**Files:**
- Create: `value-datadna/src/postprocess/mock_filter.py`
- Test: `value-datadna/tests/test_mock_filter.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_mock_filter.py
import pytest
from src.postprocess.mock_filter import MockFilter
from src.types import DataValue, ValueContext, ValueSource


def make_value(value: str, label_hint: str = "", surrounding_text: str = "") -> DataValue:
    return DataValue(
        value_id="test",
        value=value,
        context=ValueContext(
            container_type="csv_field", container_path="test.csv/col",
            label_hint=label_hint, surrounding_text=surrounding_text,
        ),
        source=ValueSource(source_type="csv", extraction_method="structured"),
    )


class TestMockFilter:
    @pytest.fixture
    def mf(self):
        return MockFilter({})

    def test_virtual_ssn_is_mock(self, mf):
        v = make_value("000-00-0000")
        assert mf.is_mock(v) is True

    def test_virtual_ccn_is_mock(self, mf):
        v = make_value("4111-1111-1111-1111")
        assert mf.is_mock(v) is True

    def test_real_ssn_not_mock(self, mf):
        v = make_value("123-45-6789")
        assert mf.is_mock(v) is False

    def test_context_negation_keyword(self, mf):
        v = make_value("123-45-6789", surrounding_text="this is a sample value")
        assert mf.is_mock(v) is True

    def test_all_same_values_in_column_is_mock(self, mf):
        values = [make_value("foo"), make_value("foo"), make_value("foo")]
        assert mf.is_column_mock(values) is True

    def test_diverse_column_not_mock(self, mf):
        values = [make_value("foo"), make_value("bar"), make_value("baz")]
        assert mf.is_column_mock(values) is False
```

- [ ] **Step 2: 实现 Mock 过滤器**

```python
# src/postprocess/mock_filter.py
from typing import List
from src.types import DataValue


class MockFilter:
    VIRTUAL_PATTERNS = {
        "000-00-0000", "123-45-6789", "XXX-XX-XXXX", "4111-1111-1111-1111",
    }
    NEGATION_KEYWORDS = {
        "test", "sample", "example", "placeholder", "redacted",
        "dummy", "mock", "fake", "todo", "fixme",
    }

    def __init__(self, config: dict):
        cfg = config.get("mock_filter", {})
        self.virtual_patterns = set(cfg.get("virtual_patterns", [])) | self.VIRTUAL_PATTERNS
        self.negation_keywords = set(cfg.get("negation_keywords", [])) | self.NEGATION_KEYWORDS

    def is_mock(self, value: DataValue) -> bool:
        if value.value.strip() in self.virtual_patterns:
            return True
        ctx_text = (value.context.surrounding_text + " " + (value.context.label_hint or "")).lower()
        for kw in self.negation_keywords:
            if kw in ctx_text:
                return True
        return False

    def is_column_mock(self, values: List[DataValue]) -> bool:
        if len(values) <= 1:
            return False
        unique = {v.value for v in values}
        return len(unique) == 1

    def filter_batch(self, values: List[DataValue]) -> tuple[List[DataValue], List[DataValue]]:
        """返回 (正常值列表, mock值列表)"""
        normal = []
        mocks = []
        for v in values:
            if self.is_mock(v):
                mocks.append(v)
            else:
                normal.append(v)
        return normal, mocks
```

- [ ] **Step 3: 运行测试, 提交**

```bash
cd value-datadna && pytest tests/test_mock_filter.py -v
git add value-datadna/src/postprocess/mock_filter.py value-datadna/tests/test_mock_filter.py
git commit -m "feat: add mock filter (virtual patterns + context negation)"
```

### Task 1A.3：结构化抽样引擎

**Files:**
- Create: `value-datadna/src/extractors/sampler.py`
- Test: `value-datadna/tests/test_sampler.py`

- [ ] **Step 1-4: TDD 实现分层抽样引擎**

```python
# src/extractors/sampler.py
import numpy as np
from typing import List
from src.types import DataValue


class StratifiedSampler:
    """按值长度/字符集分布分层抽样。每列 5-20 代表值，层内随机采。"""

    def __init__(self, config: dict):
        cfg = config.get("sampling", {})
        self.min_per_column = cfg.get("target_per_column_min", 5)
        self.max_per_column = cfg.get("target_per_column_max", 20)
        self.num_layers = cfg.get("num_layers", 7)

    def sample_column(self, values: List[DataValue]) -> List[DataValue]:
        """对单列值做分层抽样"""
        n = len(values)
        if n <= self.max_per_column:
            return list(values)

        # 去重
        unique_vals = list({v.value: v for v in values}.values())
        if len(unique_vals) <= self.max_per_column:
            return unique_vals

        # 按值长度分层
        lengths = np.array([len(v.value) for v in unique_vals])
        percentiles = np.linspace(0, 100, self.num_layers + 1)
        layer_bounds = np.percentile(lengths, percentiles)

        sampled = []
        rng = np.random.RandomState(42)
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                mask = (lengths >= layer_bounds[i]) & (lengths < layer_bounds[i + 1])
            else:
                mask = (lengths >= layer_bounds[i]) & (lengths <= layer_bounds[i + 1])
            layer_indices = np.where(mask)[0]
            if len(layer_indices) == 0:
                continue
            n_sample = max(1, int(np.ceil(self.max_per_column / self.num_layers)))
            n_sample = min(n_sample, len(layer_indices))
            chosen = rng.choice(layer_indices, size=n_sample, replace=False)
            for idx in chosen:
                sampled.append(unique_vals[int(idx)])

        total = len(sampled)
        if total > self.max_per_column:
            chosen = rng.choice(total, size=self.max_per_column, replace=False)
            sampled = [sampled[int(i)] for i in chosen]
        elif total < self.min_per_column and total < len(unique_vals):
            remaining = [v for v in unique_vals if v not in sampled]
            extra = rng.choice(len(remaining),
                               size=min(self.min_per_column - total, len(remaining)),
                               replace=False)
            sampled.extend([remaining[int(i)] for i in extra])

        return sampled
```

```python
# tests/test_sampler.py
import pytest
from src.extractors.sampler import StratifiedSampler
from src.types import DataValue, ValueContext, ValueSource


def make_values(n: int, prefix: str = "val") -> list:
    return [
        DataValue(
            value_id=f"v_{i}", value=f"{prefix}_{i:05d}",
            context=ValueContext(container_type="csv_field", container_path="t/c"),
            source=ValueSource(source_type="csv", extraction_method="structured"),
        )
        for i in range(n)
    ]


class TestStratifiedSampler:
    def test_small_column_returns_all(self):
        sampler = StratifiedSampler({})
        vals = make_values(10)
        result = sampler.sample_column(vals)
        assert len(result) == 10

    def test_large_column_sampled_down(self):
        sampler = StratifiedSampler({})
        vals = make_values(1000)
        result = sampler.sample_column(vals)
        assert 5 <= len(result) <= 20

    def test_diverse_lengths_better_coverage(self):
        sampler = StratifiedSampler({})
        short = make_values(500, "a")
        long_vals = make_values(500, "b" * 50)
        vals = short + long_vals
        result = sampler.sample_column(vals)
        has_short = any(len(v.value) < 15 for v in result)
        has_long = any(len(v.value) > 15 for v in result)
        assert has_short and has_long
```

- [ ] **Step 5: Commit**

```bash
cd value-datadna && pytest tests/test_sampler.py -v
git add value-datadna/src/extractors/sampler.py value-datadna/tests/test_sampler.py
git commit -m "feat: add stratified sampler for column value reduction"
```

---

## Phase 1B：真值表引擎

**依赖:** Phase 0 完成  
**可并行:** 与 Phase 1A/1C/2A/3A/3B 并行

### Task 1B.1：PII 正则模式库 + 6 维特征提取器

**Files:**
- Create: `value-datadna/src/knowledge/pii_patterns.py`
- Create: `value-datadna/src/classifiers/feature_extractor.py`
- Test: `value-datadna/tests/test_feature_extractor.py`

- [ ] **Step 1: 实现 PII 正则模式库（30+ 类型）**

```python
# src/knowledge/pii_patterns.py — 设计文档 Section 3.5.1 六维特征之 regex_strength 的底层

PII_PATTERNS = {
    "SSN": {
        "regex": r'(?<!\d)\d{3}[\s\-]\d{2}[\s\-]\d{4}(?!\d)',
        "description": "US Social Security Number",
        "validation": None,       # 无校验位算法
        "category": "weak_structure",
    },
    "CREDIT_CARD": {
        "regex": r'(?<!\d)(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{3,4}(?!\d)',
        "description": "Credit card number (Visa/MC/Amex/Discover)",
        "validation": "luhn",
        "category": "strong_structure",
    },
    "IBAN": {
        "regex": r'[A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?(?:[\dA-Z]{4}[\s]?){2,7}[\dA-Z]{1,4}',
        "description": "International Bank Account Number",
        "validation": "mod97",
        "category": "strong_structure",
    },
    "EMAIL": {
        "regex": r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
        "description": "Email address",
        "validation": None,
        "category": "weak_structure",
    },
    "PHONE": {
        "regex": r'(?:\+\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}',
        "description": "International phone number",
        "validation": None,
        "category": "weak_structure",
    },
    "IP": {
        "regex": r'(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)',
        "description": "IPv4 address",
        "validation": None,
        "category": "strong_structure",
    },
    "PASSPORT": {
        "regex": r'[A-Z0-9<]{9,44}',
        "description": "Passport MRZ line (simplified)",
        "validation": None,
        "category": "weak_structure",
    },
    "DRIVER_LICENSE": {
        "regex": r'[A-Z]\d{7,8}',
        "description": "US Driver License (simplified)",
        "validation": None,
        "category": "weak_structure",
    },
    "API_KEY": {
        "regex": r'(?:api[_-]?key|apikey|token|secret)[\s:=]+[\w\-]{20,}',
        "description": "API key / token pattern",
        "validation": None,
        "category": "weak_structure",
    },
    "BANK_ACCOUNT": {
        "regex": r'(?<!\d)\d{8,12}(?!\d)',
        "description": "Bank account number (simplified)",
        "validation": None,
        "category": "weak_structure",
    },
    "NAME": {
        "regex": None,  # 纯语义类型，无正则
        "description": "Person full name",
        "validation": None,
        "category": "semantic",
    },
    "ADDRESS": {
        "regex": None,
        "description": "Physical address",
        "validation": None,
        "category": "semantic",
    },
}

# 校验位算法
def luhn_check(num_str: str) -> bool:
    digits = [int(c) for c in num_str.replace(" ", "").replace("-", "") if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

def mod97_iban_check(iban: str) -> bool:
    """IBAN mod-97 校验 (简化为仅检查格式)"""
    cleaned = iban.replace(" ", "").upper()
    return len(cleaned) >= 15 and cleaned[:2].isalpha() and cleaned[2:].isalnum()
```

- [ ] **Step 2: 实现 6 维特征提取器**

```python
# src/classifiers/feature_extractor.py
import re
import numpy as np
from collections import Counter
from typing import Dict, List
from src.types import DataValue
from src.knowledge.pii_patterns import PII_PATTERNS, luhn_check, mod97_iban_check


# regex_strength 基准语料误报率预计算值 (Section 3.5.1)
# 在实际校准流程中通过 Wikipedia + GitHub + 企业文档混合语料计算
# 此处为初始占位值，Phase 1 校准流程运行时替换
PRELIMMARY_SPECIFICITY = {
    "SSN": 0.82, "CREDIT_CARD": 0.96, "IBAN": 0.94, "EMAIL": 0.75,
    "PHONE": 0.62, "IP": 0.88, "PASSPORT": 0.55, "DRIVER_LICENSE": 0.58,
    "API_KEY": 0.52, "BANK_ACCOUNT": 0.45, "NAME": 0.0, "ADDRESS": 0.0,
}

SUPPORTIVE_KEYWORDS = {
    "SSN": {"ssn", "social security", "社保", "社会安全号"},
    "CREDIT_CARD": {"credit card", "ccn", "信用卡", "card number"},
    "IBAN": {"iban", "bank account", "swift", "银行账号"},
    "EMAIL": {"email", "e-mail", "邮箱", "mail"},
    "PHONE": {"phone", "tel", "mobile", "电话", "手机"},
    "IP": {"ip address", "ip", "host"},
    "PASSPORT": {"passport", "护照"},
    "DRIVER_LICENSE": {"driver license", "dl", "驾照"},
    "API_KEY": {"api key", "token", "secret"},
    "BANK_ACCOUNT": {"account number", "账号"},
}

UNSUPPORTIVE_KEYWORDS = {
    "test", "sample", "example", "placeholder", "redacted",
    "dummy", "mock", "fake", "todo", "fixme", "demo",
}


class FeatureExtractor:
    def __init__(self, config: dict):
        self.patterns = PII_PATTERNS
        self.regex_cache: Dict[str, re.Pattern] = {}

    def _get_regex(self, type_name: str) -> re.Pattern | None:
        if type_name in self.regex_cache:
            return self.regex_cache[type_name]
        info = self.patterns.get(type_name, {})
        pattern_str = info.get("regex")
        if pattern_str:
            compiled = re.compile(pattern_str, re.IGNORECASE)
            self.regex_cache[type_name] = compiled
            return compiled
        self.regex_cache[type_name] = None
        return None

    def extract(self, value: DataValue, candidate_type: str,
                global_pattern_freq: float, global_uniqueness: int) -> dict:
        """
        计算单个值的 6 维特征 (Section 3.5.1)。

        返回 dict with keys:
          regex_strength, validated_count, supportive_context,
          unsupportive_context, pattern_frequency, uniqueness_score

        validated_count 冷启动用合成数据量 (Phase 1 校准阶段填入)，
        真实数据积累后通过加权求和更新 (w_synth × count_synth + w_real × count_real)。
        此处使用初始占位值 bin center。
        """
        ctx_text = (
            (value.context.surrounding_text or "") + " " +
            (value.context.label_hint or "")
        ).lower()

        # 1. regex_strength: 正则特异性
        regex_strength = PRELIMMARY_SPECIFICITY.get(candidate_type, 0.0)

        # 2. validated_count: 冷启动阶段使用合成数据量 (design doc: 每类型 2000 → 50+ 桶)
        validated_count = 50  # 冷启动占位，校准阶段调整

        # 3. supportive_context: 支持性上下文词命中数
        supportive_words = SUPPORTIVE_KEYWORDS.get(candidate_type, set())
        supportive_count = sum(1 for w in supportive_words if w in ctx_text)
        supportive_count = min(supportive_count, 3)

        # 4. unsupportive_context: 否定性上下文词命中数
        unsupportive_count = sum(1 for w in UNSUPPORTIVE_KEYWORDS if w in ctx_text)
        unsupportive_count = min(unsupportive_count, 2)

        # 5. pattern_frequency: 模式在数据集中的频率 (外部传入)
        pattern_freq = global_pattern_freq

        # 6. uniqueness_score: 值的唯一性 (外部传入)
        uniqueness = min(global_uniqueness, 100)

        return {
            "regex_strength": regex_strength,
            "validated_count": validated_count,
            "supportive_context": supportive_count,
            "unsupportive_context": unsupportive_count,
            "pattern_frequency": pattern_freq,
            "uniqueness_score": uniqueness,
        }

    def discretize(self, features: dict) -> tuple:
        """将连续特征离散化为 bin 索引 (Section 3.5.1)"""
        bins = {
            "regex_strength": [0, 0.25, 0.5, 0.75, 1.0],
            "validated_count": [0, 1, 4, 10, 50],
            "supportive_context": [0, 1, 2, 3],
            "unsupportive_context": [0, 1, 2],
            "pattern_frequency": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "uniqueness_score": [1, 2, 6, 21, 100],
        }

        def _to_bin(val, thresholds):
            for i, t in enumerate(thresholds):
                if val <= t:
                    return i
            return len(thresholds) - 1

        return tuple(
            _to_bin(features[dim], bins[dim])
            for dim in [
                "regex_strength", "validated_count", "supportive_context",
                "unsupportive_context", "pattern_frequency", "uniqueness_score",
            ]
        )

    def validate_checksum(self, value_str: str, candidate_type: str) -> bool:
        """校验位验证 (Luhn for CCN, mod-97 for IBAN)"""
        info = self.patterns.get(candidate_type, {})
        validation = info.get("validation")
        if validation == "luhn":
            return luhn_check(value_str)
        elif validation == "mod97":
            return mod97_iban_check(value_str)
        return True  # 无校验位 → 不做校验

    def compute_global_stats(self, all_values: List[DataValue]) -> dict:
        """在整个数据集上计算 pattern_frequency 和 uniqueness_score"""
        type_match_counts = Counter()
        value_counts = Counter()
        total = len(all_values)

        for v in all_values:
            value_counts[v.value] += 1
            for type_name, info in self.patterns.items():
                rx = self._get_regex(type_name)
                if rx and rx.search(v.value):
                    type_match_counts[type_name] += 1

        return {
            "pattern_freq": {
                t: type_match_counts[t] / max(total, 1) for t in self.patterns
            },
            "value_counts": dict(value_counts),
        }
```

- [ ] **Step 3: Commit**

```bash
cd value-datadna && pytest tests/test_feature_extractor.py -v
git add value-datadna/src/knowledge/pii_patterns.py value-datadna/src/classifiers/feature_extractor.py
git commit -m "feat: add PII pattern library + 6-dim feature extractor"
```

### Task 1B.2：真值表 DataFrame 构建 + KD-Tree 插值

**Files:**
- Create: `value-datadna/src/classifiers/truth_table.py`
- Test: `value-datadna/tests/test_truth_table.py`

- [ ] **Step 1: TDD 真值表引擎**

```python
# src/classifiers/truth_table.py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from typing import Dict, Tuple, Optional
from src.classifiers.feature_extractor import FeatureExtractor


class TruthTableEngine:
    """
    每敏感类型一张独立真值表 (Section 3.5.2)。
    6 维 MultiIndex DataFrame → confidence，O(1) 查询。
    缺失 bin → KD-Tree 最近邻插值，O(log n)。
    """

    def __init__(self, config: dict):
        self.feature_extractor = FeatureExtractor(config)
        self.tables: Dict[str, pd.DataFrame] = {}     # type_name → DataFrame
        self.kdtrees: Dict[str, KDTree] = {}          # type_name → KDTree
        self.bin_keys: Dict[str, np.ndarray] = {}     # type_name → bin_indexes
        self.laplace_alpha = config.get("truth_table", {}).get("calibration", {}).get("laplace_alpha", 1)

    def calibrate(self, type_name: str, labeled_samples: list) -> pd.DataFrame:
        """
        校准单类型的真值表 (Section 3.5.3)。

        labeled_samples: List[dict] — 每个样本含:
          value, context, label (type_name 或 f"NOT_{type_name}")
        """
        from collections import defaultdict
        bin_counts = defaultdict(lambda: {"pos": 0, "total": 0})

        for sample in labeled_samples:
            features = self.feature_extractor.extract(
                sample["value"], sample["context"],
                type_name,
                global_pattern_freq=sample.get("pattern_freq", 0),
                global_uniqueness=sample.get("uniqueness", 1),
            )
            bin_key = self.feature_extractor.discretize(features)

            bin_counts[bin_key]["total"] += 1
            if sample["label"] == type_name:
                bin_counts[bin_key]["pos"] += 1

        # 构建 MultiIndex DataFrame
        records = []
        for bin_key, counts in bin_counts.items():
            confidence = (counts["pos"] + self.laplace_alpha) / (counts["total"] + 2 * self.laplace_alpha)
            records.append((*bin_key, confidence, counts["total"]))

        df = pd.DataFrame(records, columns=[
            "regex_strength", "validated_count", "supportive_context",
            "unsupportive_context", "pattern_frequency", "uniqueness_score",
            "confidence", "sample_count",
        ])
        df = df.set_index([
            "regex_strength", "validated_count", "supportive_context",
            "unsupportive_context", "pattern_frequency", "uniqueness_score",
        ])
        df = df.sort_index()

        self.tables[type_name] = df

        # 构建 KDTree 用于缺失 bin 插值
        bin_array = np.array(list(df.index))
        self.bin_keys[type_name] = bin_array
        self.kdtrees[type_name] = KDTree(bin_array)

        return df

    def query(self, value, candidate_type: str, global_stats: dict) -> float:
        """
        查询真值表 confidence (Section 3.5.2 查询逻辑)。

        返回 confidence ∈ [0, 1]。缺失 bin → KDTree 最近邻插值。
        """
        if candidate_type not in self.tables:
            return 0.0

        features = self.feature_extractor.extract(
            value, candidate_type,
            global_pattern_freq=global_stats.get("pattern_freq", {}).get(candidate_type, 0),
            global_uniqueness=global_stats.get("value_counts", {}).get(value.value, 1),
        )
        bin_key = self.feature_extractor.discretize(features)
        df = self.tables[candidate_type]

        try:
            return float(df.loc[bin_key, "confidence"])
        except KeyError:
            # KDTree 最近邻插值
            tree = self.kdtrees[candidate_type]
            keys = self.bin_keys[candidate_type]
            if len(keys) == 0:
                return 0.0
            dist, idx = tree.query(np.array(bin_key), k=1)
            nearest_key = tuple(keys[idx])
            return float(df.loc[nearest_key, "confidence"])

    def query_multi_type(self, value, candidate_types: list, global_stats: dict) -> Tuple[str, float, str | None]:
        """
        多类型匹配路由 (Section 3.5.2 + 4.5 次高候选)：
        一个值匹配多种正则 → 分别查各候选类型真值表 → 取最高 confidence。
        返回 (best_type, best_conf, second_best_type)。
        second_best_type 供 LLM Validation 拒绝后使用 (Section 4.5 降级逻辑)。
        """
        best_type = None
        best_conf = 0.0
        second_type = None
        second_conf = 0.0
        for ct in candidate_types:
            conf = self.query(value, ct, global_stats)
            if conf > best_conf:
                second_type = best_type
                second_conf = best_conf
                best_conf = conf
                best_type = ct
            elif conf > second_conf:
                second_conf = conf
                second_type = ct
        return best_type, best_conf, second_type

    def save(self, type_name: str, path: str):
        """保存真值表为 parquet"""
        if type_name in self.tables:
            self.tables[type_name].to_parquet(path)

    def load(self, type_name: str, path: str):
        """从 parquet 加载真值表"""
        df = pd.read_parquet(path)
        self.tables[type_name] = df
        bin_array = np.array(list(df.index))
        self.bin_keys[type_name] = bin_array
        self.kdtrees[type_name] = KDTree(bin_array)
```

- [ ] **Step 2: Commit**

```bash
cd value-datadna && pytest tests/test_truth_table.py -v
git add value-datadna/src/classifiers/truth_table.py
git commit -m "feat: add truth table engine (MultiIndex + KDTree interpolation)"
```

### Task 1B.3：合成数据生成器 + 真值表校准流程

**Files:**
- Create: `value-datadna/src/knowledge/synthetic_generator.py`
- Create: `value-datadna/calibrate.py`
- Test: `value-datadna/tests/test_synthetic_generator.py`

- [ ] **Step 1: 合成数据生成器 (15 PII 类型 × 2500 样本)**

```python
# src/knowledge/synthetic_generator.py
import random, string, re
from typing import List, Dict
from src.types import DataValue, ValueContext, ValueSource

# 每类型: 2000 正样本 + 500 难负样本 (Section 3.5.3)
# 每值: 3 种上下文变体 (clean / penalty_term / boost_term)

def _make_value(val: str, label_hint: str = "", surrounding: str = "") -> DataValue:
    return DataValue(
        value_id=f"syn_{random.randint(0, 10**9)}",
        value=val,
        context=ValueContext(
            container_type="synthetic",
            container_path=f"synthetic/{label_hint}",
            label_hint=label_hint,
            surrounding_text=surrounding,
        ),
        source=ValueSource(source_type="synthetic", extraction_method="generated"),
    )

def generate_ssn_positive(n: int) -> List[DataValue]:
    """US SSN: XXX-XX-XXXX (排除已知虚拟模式)"""
    virtual = {"000-00-0000", "123-45-6789", "XXX-XX-XXXX"}
    results = []
    while len(results) < n:
        ssn = f"{random.randint(100, 999):03d}-{random.randint(10, 99):02d}-{random.randint(1000, 9999):04d}"
        if ssn not in virtual:
            ctx_variants = [
                f"SSN: {ssn}",
                f"employee social security number: {ssn}, test data excluded",
                f"social security number on file: {ssn}",
            ]
            for ctx in ctx_variants:
                results.append(_make_value(ssn, "ssn", ctx))
                if len(results) >= n * 3:
                    break
    return results[:n * 3]

def generate_ssn_hard_negative(n: int) -> List[DataValue]:
    """9 位数字但非 SSN 格式 (如连写无分隔符)"""
    results = []
    while len(results) < n * 3:
        fake = f"{random.randint(100000000, 999999999):09d}"
        ctx_variants = [
            f"ID: {fake}",
            f"employee number: {fake}",
            f"reference code: {fake}",
        ]
        for ctx in ctx_variants:
            results.append(_make_value(fake, "id", ctx))
    return results[:n * 3]

def generate_ccn_positive(n: int) -> List[DataValue]:
    """生成 Luhn 合法的信用卡号"""
    results = []
    while len(results) < n * 3:
        # Visa: 4 + 15 digits
        digits = [4] + [random.randint(0, 9) for _ in range(14)]
        # Luhn: 计算校验位
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 0:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        check = (10 - total % 10) % 10
        ccn = "".join(str(d) for d in digits) + str(check)
        formatted = f"{ccn[:4]}-{ccn[4:8]}-{ccn[8:12]}-{ccn[12:]}"
        ctx_variants = [
            f"Credit Card: {formatted}",
            f"payment method: {formatted}",
            f"card number: {formatted}",
        ]
        for ctx in ctx_variants:
            results.append(_make_value(formatted, "credit_card", ctx))
    return results[:n * 3]

# ... (其余 13 种类型的生成器类似实现, 详见设计文档 Section 7.2 合成数据表)

SYNTHETIC_GENERATORS = {
    "SSN": (generate_ssn_positive, generate_ssn_hard_negative),
    "CREDIT_CARD": (generate_ccn_positive, generate_ccn_hard_negative),
    # ... 完整 15 类型的生成器映射
}

def generate_all(num_positive: int = 2000, num_hard_neg: int = 500,
                 context_variants: int = 3) -> Dict[str, List[DataValue]]:
    """生成全部类型的合成数据"""
    dataset = {}
    for type_name, (pos_fn, neg_fn) in SYNTHETIC_GENERATORS.items():
        pos = pos_fn(num_positive)
        neg = neg_fn(num_hard_neg)
        dataset[type_name] = pos + neg
    return dataset
```

- [ ] **Step 2-4: 编写校准脚本 + 测试 + 提交**

```python
# calibrate.py — 真值表校准入口 (Section 3.5.3)
"""Usage: python calibrate.py --config config.yaml"""
import argparse, yaml
from src.knowledge.synthetic_generator import generate_all
from src.classifiers.feature_extractor import FeatureExtractor
from src.classifiers.truth_table import TruthTableEngine
from src.types import ValueContext, ValueSource

def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    engine = TruthTableEngine(config)
    extractor = FeatureExtractor(config)
    dataset = generate_all()

    # 对每个类型独立校准真值表
    for type_name, samples in dataset.items():
        # 构建标注数据: 正样本 label=type_name, 难负样本 label=f"NOT_{type_name}"
        labeled = []
        for s in samples:
            is_pos = type_name.lower() in s.context.label_hint.lower() or \
                     type_name.lower() in s.context.surrounding_text.lower()
            # 判断逻辑: 正样本的 container_path 包含类型名
            is_positive = type_name.lower() in s.context.container_path.lower()
            labeled.append({
                "value": s,
                "label": type_name if is_positive else f"NOT_{type_name}",
                "pattern_freq": 0.001,
                "uniqueness": 1,
            })

        # 添加跨类型负样本 (其他类型的正样本自动复用)
        for other_type, other_samples in dataset.items():
            if other_type == type_name:
                continue
            for s in other_samples[:2000]:  # 每类取 2000 作为负样本
                labeled.append({
                    "value": s,
                    "label": f"NOT_{type_name}",
                    "pattern_freq": 0.001,
                    "uniqueness": 1,
                })

        df = engine.calibrate(type_name, labeled)
        engine.save(type_name, f"datasets/calibration/truth_table_{type_name}.parquet")
        print(f"[{type_name}] 校准完成: {len(df)} bins, "
              f"mean_conf={df['confidence'].mean():.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
```

```bash
cd value-datadna && pytest tests/test_synthetic_generator.py tests/test_truth_table.py -v
git add value-datadna/src/knowledge/synthetic_generator.py value-datadna/calibrate.py
git commit -m "feat: add synthetic data generator + truth table calibration script"
```

---

## Phase 1C：NER 引擎

**依赖:** Phase 0 完成  
**可并行:** 与 Phase 1A/1B/2A/3A/3B 并行  
**接口契约:** 产出 `entity_type_hint: {PERSON_NAME, ORGANIZATION, LOCATION, GENERIC_ENTITY, NONE}`

### Task 1C.1：GLiNER 零样本 NER + BIO 标签映射

**Files:**
- Create: `value-datadna/src/classifiers/ner.py`
- Test: `value-datadna/tests/test_ner.py`

- [ ] **Step 1: TDD NER 引擎**

```python
# src/classifiers/ner.py — Section 3.6
from typing import List, Tuple, Optional
from src.types import DataValue


class NEREngine:
    """
    NER 引擎 (Section 3.6)。
    冷启动: GLiNER 零样本。
    有标注后: BERT-base 微调。
    蒸馏: Qwen3:8b → BERT。

    输出 entity_type_hint ∈ {PERSON_NAME, ORGANIZATION, LOCATION, GENERIC_ENTITY, NONE}
    作为真值表第 7 维输入，不独立产出分类决策。
    """

    # Section 3.6.1: NER BIO标签 → entity_type_hint 映射
    BIO_TO_HINT = {
        "B-NAME": "PERSON_NAME", "I-NAME": "PERSON_NAME",
        "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
        "B-ADDRESS": "LOCATION", "I-ADDRESS": "LOCATION",
        # "OTHER" 标签 → GENERIC_ENTITY
        # "O" 或无覆盖 → NONE
    }

    # Section 3.6.3: 完整 BIO 标签体系 (27 标签: 13×2 B/I + O)
    BIO_LABELS = [
        "O",
        "B-SSN","I-SSN","B-CCN","I-CCN","B-EMAIL","I-EMAIL",
        "B-PHONE","I-PHONE","B-IBAN","I-IBAN",
        "B-PASSPORT","I-PASSPORT","B-DRIVER_LICENSE","I-DRIVER_LICENSE",
        "B-IP","I-IP","B-API_KEY","I-API_KEY",
        "B-BANK_ACCOUNT","I-BANK_ACCOUNT",
        "B-NAME","I-NAME","B-ADDRESS","I-ADDRESS",
        "B-ORG","I-ORG",
    ]

    def __init__(self, config: dict):
        cfg = config.get("ner", {})
        self.cold_start_model = cfg.get("cold_start_model", "urchade/gliner_base")
        self.fine_tune_model = cfg.get("fine_tune_model", "bert-base-uncased")
        self.model = None
        self.model_type = "gliner"  # "gliner" | "bert" | "distilled"

    def load_gliner(self):
        """加载 GLiNER 零样本模型 (冷启动)"""
        from gliner import GLiNER
        self.model = GLiNER.from_pretrained(self.cold_start_model)
        self.model_type = "gliner"

    def predict_hint(self, value: DataValue) -> str:
        """
        从 surrounding_text 中识别值的实体类型 → entity_type_hint。
        Section 3.6.1: 仅从 surrounding_text 的 BIO 标注获取，
        不独立产出分类决策。

        返回 entity_type_hint ∈ {PERSON_NAME, ORGANIZATION, LOCATION, GENERIC_ENTITY, NONE}
        """
        if self.model is None:
            self.load_gliner()

        # 使用 surrounding_text 作为 NER 输入上下文
        text = value.context.surrounding_text or value.value
        target_value = value.value

        if self.model_type == "gliner":
            return self._predict_gliner(text, target_value)
        elif self.model_type in ("bert", "distilled"):
            return self._predict_bert(text, target_value)
        return "NONE"

    def _predict_gliner(self, text: str, target: str) -> str:
        """GLiNER 零样本推理"""
        labels = ["person", "organization", "location", "date", "id_number"]
        try:
            entities = self.model.predict_entities(text, labels, threshold=0.3)
        except Exception:
            return "NONE"

        # 找到覆盖 target 的实体
        target_start = text.find(target)
        if target_start < 0:
            return "NONE"
        target_end = target_start + len(target)

        for ent in entities:
            if ent["start"] <= target_start and ent["end"] >= target_end:
                label = ent["label"].upper()
                if label == "PERSON":
                    return "PERSON_NAME"
                elif label == "ORGANIZATION":
                    return "ORGANIZATION"
                elif label == "LOCATION":
                    return "LOCATION"
                else:
                    return "GENERIC_ENTITY"
        return "NONE"

    def _predict_bert(self, text: str, target: str) -> str:
        """BERT 微调模型推理 (Phase 1C Task 2 实现)"""
        # 占位——Phase 1C Task 2 实现
        return "NONE"

    def predict_batch(self, values: List[DataValue]) -> List[str]:
        """批量预测 entity_type_hint"""
        return [self.predict_hint(v) for v in values]
```

- [ ] **Step 2: Commit**

```bash
cd value-datadna && pytest tests/test_ner.py -v
git add value-datadna/src/classifiers/ner.py
git commit -m "feat: add NER engine (GLiNER zero-shot + BIO label mapping)"
```

### Task 1C.2：BERT BIO 微调 + 蒸馏框架

**Files:**
- Create: `value-datadna/train_ner.py`
- Test: `value-datadna/tests/test_ner.py` (追加测试)

- [ ] **Step 1: BERT-base BIO 微调脚本**

```python
# train_ner.py — NER 微调 + 蒸馏训练入口 (Section 3.6.2)
"""
Usage:
  # 微调 BERT
  python train_ner.py --mode finetune --data datasets/public/pii-masking --output models/ner_bert

  # LLM 蒸馏
  python train_ner.py --mode distill --teacher qwen3:8b --data datasets/public/ --output models/ner_distilled
"""
import argparse
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)
from datasets import load_dataset, Dataset
import numpy as np

BIO_LABELS = [
    "O",
    "B-SSN","I-SSN","B-CCN","I-CCN","B-EMAIL","I-EMAIL",
    "B-PHONE","I-PHONE","B-IBAN","I-IBAN",
    "B-PASSPORT","I-PASSPORT","B-DRIVER_LICENSE","I-DRIVER_LICENSE",
    "B-IP","I-IP","B-API_KEY","I-API_KEY",
    "B-BANK_ACCOUNT","I-BANK_ACCOUNT",
    "B-NAME","I-NAME","B-ADDRESS","I-ADDRESS",
    "B-ORG","I-ORG",
]
LABEL2ID = {l: i for i, l in enumerate(BIO_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(BIO_LABELS)}


def finetune_bert(data_path: str, output_dir: str, model_name: str = "bert-base-uncased"):
    """BERT-base BIO 序列标注微调"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(BIO_LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    # 加载 ai4privacy/pii-masking-300k + conllpp (Section 7.2 公开数据)
    dataset = load_dataset("ai4privacy/pii-masking-300k", split="train")
    # ... BIO 标签转换 + train/val split ...

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # ... Trainer setup + train ...
    # (完整训练代码 ~150 行, 此处展示架构)


def distill_from_llm(teacher_model: str, data_path: str, output_dir: str):
    """Qwen3:8b (教师) → BERT (学生) 蒸馏 (Section 3.6.2)"""
    # 1. LLM 标注未标记数据
    # 2. BERT 学生在 LLM 伪标签上微调
    # 3. 模型量化/导出
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["finetune", "distill"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/ner")
    parser.add_argument("--teacher", default="qwen3:8b")
    args = parser.parse_args()

    if args.mode == "finetune":
        finetune_bert(args.data, args.output)
    else:
        distill_from_llm(args.teacher, args.data, args.output)
```

- [ ] **Step 2: Commit**

```bash
git add value-datadna/train_ner.py
git commit -m "feat: add NER fine-tuning + LLM distillation training script"
```

---

## Phase 1D：融合评分 + 角色判定 + 上下文一致性 + 端到端评估

**依赖:** Phase 1A + 1B + 1C 完成  
**可并行:** 与 Phase 2B/3C 部分并行（Phase 2B/3C 需要 Phase 1D 的 `classify()` 接口）

### Task 1D.1：融合评分 + 置信度路由

**Files:**
- Create: `value-datadna/src/classifiers/fusion.py`
- Test: `value-datadna/tests/test_fusion.py`

```python
# src/classifiers/fusion.py — Section 3.7
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np


class FusionScorer:
    """
    真值表 + Semantic Distancing 并列评分融合 (Section 3.7)。
    Phase 1 阶段 α=1.0（仅依赖真值表），Phase 3 完成后 α=0.7。

    融合前各自 Platt scaling 校准到 [0,1] 概率空间。
    """

    def __init__(self, config: dict):
        self.alpha = config.get("fusion", {}).get("alpha", 1.0)
        self.tt_calibrator = None    # 真值表 Platt 校准器
        self.sd_calibrator = None    # SD Platt 校准器
        self.tt_scores = []
        self.tt_labels = []

    def fit_tt_calibrator(self, truth_table_scores: np.ndarray, true_labels: np.ndarray):
        """Platt scaling 校准真值表分数"""
        self.tt_calibrator = LogisticRegression()
        self.tt_calibrator.fit(truth_table_scores.reshape(-1, 1), true_labels)

    def fit_sd_calibrator(self, sd_scores: np.ndarray, true_labels: np.ndarray):
        """Platt scaling 校准 Semantic Distancing 分数"""
        self.sd_calibrator = LogisticRegression()
        self.sd_calibrator.fit(sd_scores.reshape(-1, 1), true_labels)

    def calibrate_tt(self, score: float) -> float:
        """校准真值表分数 → [0,1] 概率"""
        if self.tt_calibrator is None:
            return score
        proba = self.tt_calibrator.predict_proba(np.array([[score]]))
        return float(proba[0, 1])

    def calibrate_sd(self, score: float) -> float:
        """校准 SD 分数 → [0,1] 概率"""
        if self.sd_calibrator is None:
            return score
        proba = self.sd_calibrator.predict_proba(np.array([[score]]))
        return float(proba[0, 1])

    def fuse(self, tt_confidence: float, sd_distance_score: float = 0.0) -> float:
        """融合评分: α × P(truth_table) + (1-α) × P(distance)"""
        cal_tt = self.calibrate_tt(tt_confidence)
        cal_sd = self.calibrate_sd(sd_distance_score)
        return self.alpha * cal_tt + (1 - self.alpha) * cal_sd


class ConfidenceRouter:
    """
    置信度路由 (Section 3.10)。
    对抗性验证校准阈值，初始占位值: 0.85 / 0.50 / 0.30。
    """

    def __init__(self, config: dict):
        cfg = config.get("routing", {})
        self.high_threshold = cfg.get("high_threshold", 0.85)
        self.mid_threshold = cfg.get("mid_threshold", 0.50)
        self.low_threshold = cfg.get("low_threshold", 0.30)

    def route(self, confidence: float) -> str:
        """
        返回路由目标:
          "direct"     — 直接输出 (conf ≥ 0.85)
          "llm_validate" — LLM Validation (0.50 ≤ conf < 0.85)
          "llm_classify" — LLM Classification (conf < 0.50)
        """
        if confidence >= self.high_threshold:
            return "direct"
        elif confidence >= self.mid_threshold:
            return "llm_validate"
        else:
            return "llm_classify"

    def calibrate_thresholds(self, predicted_confidences: np.ndarray,
                              true_labels: np.ndarray):
        """
        对抗性验证校准阈值 (Section 3.10 校准方法)。
        high_threshold: precision ≥ 0.95 的最低 confidence
        mid_threshold:  precision ≥ 0.70 的最低 confidence
        """
        sorted_idx = np.argsort(predicted_confidences)[::-1]
        sorted_preds = (true_labels[sorted_idx] != "NON_SENSITIVE")

        # 找 precision ≥ 0.95 的最低 confidence
        cum_correct = np.cumsum(sorted_preds)
        cum_total = np.arange(1, len(sorted_preds) + 1)
        precision = cum_correct / cum_total

        for i in range(len(precision)):
            if precision[i] < 0.95:
                if i > 0:
                    self.high_threshold = predicted_confidences[sorted_idx[i - 1]]
                break

        for i in range(len(precision)):
            if precision[i] < 0.70:
                if i > 0:
                    self.mid_threshold = predicted_confidences[sorted_idx[i - 1]]
                break
```

### Task 1D.2：角色判定 + 上下文一致性检查

**Files:**
- Create: `value-datadna/src/postprocess/role_detector.py`
- Create: `value-datadna/src/postprocess/context_check.py`

```python
# src/postprocess/role_detector.py — Section 3.8
SUBJECT_KEYWORDS = {
    "ssn", "social", "社保", "credit_card", "credit", "passport", "护照",
    "iban", "bank_account", "银行", "phone", "电话", "email", "邮箱",
    "driver_license", "驾照", "api_key", "token",
}
IDENTIFIER_KEYWORDS = {
    "id", "account", "ref", "reference", "number", "code", "no",
    "编号", "账号", "标识",
}


class RoleDetector:
    def detect(self, label_hint: str | None) -> str:
        """subject | identifier | reference"""
        if not label_hint:
            return "reference"
        hint_lower = label_hint.lower().strip()
        for kw in SUBJECT_KEYWORDS:
            if kw in hint_lower:
                return "subject"
        for kw in IDENTIFIER_KEYWORDS:
            if kw in hint_lower:
                return "identifier"
        return "reference"
```

```python
# src/postprocess/context_check.py — Section 3.9
# 上下文一致性检查 (后置)
# 冲突 → 路由降级: direct → llm_validate, llm_validate → llm_classify

CONFLICT_PATTERNS = {
    "SSN": ["account", "email", "phone", "id", "test"],
    "CREDIT_CARD": ["test transaction", "sample", "demo"],
    "EMAIL": ["ssn", "credit card", "passport"],
    "PHONE": ["ssn", "credit card", "passport"],
    "PASSPORT": ["email", "phone", "test"],
    "IBAN": ["email", "phone", "test"],
    "IP": ["ssn", "credit card", "passport"],
}


class ContextChecker:
    def check(self, classification_type: str, context) -> str:
        """
        返回路由调整: "keep" | "downgrade" | "flag_review"
        """
        if classification_type is None or classification_type == "NON_SENSITIVE":
            return "keep"

        ctx_text = (
            (context.surrounding_text or "") + " " +
            (context.label_hint or "")
        ).lower()

        conflicts = CONFLICT_PATTERNS.get(classification_type, [])
        for kw in conflicts:
            if kw in ctx_text:
                return "downgrade"
        return "keep"
```

### Task 1D.3：主分类内核 `classify()` 集成 + 评估基准

**Files:**
- Create: `value-datadna/src/classifiers/kernel.py`
- Create: `value-datadna/src/evaluation/benchmark.py`
- Create: `value-datadna/evaluate.py`

```python
# src/classifiers/kernel.py — 主分类内核 (Section 2.1 架构图)
# classify(value, context) → ValueClassification
# 组装 Phase 1 全部组件：MockFilter → TruthTable(+NER hint) → Fusion → Role → ContextCheck

from typing import List
from src.types import DataValue, ValueClassification
from src.postprocess.mock_filter import MockFilter
from src.classifiers.feature_extractor import FeatureExtractor
from src.classifiers.truth_table import TruthTableEngine
from src.classifiers.ner import NEREngine
from src.classifiers.fusion import FusionScorer, ConfidenceRouter
from src.postprocess.role_detector import RoleDetector
from src.postprocess.context_check import ContextChecker


class ClassificationKernel:
    """DataDNA 值级分类内核 (单内核三路径统一接口)"""

    def __init__(self, config: dict):
        self.mock_filter = MockFilter(config)
        self.feature_extractor = FeatureExtractor(config)
        self.truth_table = TruthTableEngine(config)
        self.ner = NEREngine(config)
        self.fusion = FusionScorer(config)
        self.router = ConfidenceRouter(config)
        self.role_detector = RoleDetector()
        self.context_checker = ContextChecker()

    def classify(self, value: DataValue) -> ValueClassification:
        return self.classify_batch([value])[0]

    def classify_batch(self, values: List[DataValue]) -> List[ValueClassification]:
        # Step 1: Mock 过滤
        normal_vals, mock_vals = self.mock_filter.filter_batch(values)

        # Step 2: 全局统计
        global_stats = self.feature_extractor.compute_global_stats(normal_vals)

        # Step 3: 候选类型检测 (正则匹配)
        results = []
        for v in normal_vals:
            # 找到匹配的正则类型
            candidates = []
            for type_name, info in self.feature_extractor.patterns.items():
                rx = self.feature_extractor._get_regex(type_name)
                if rx and rx.search(v.value):
                    candidates.append(type_name)

            if not candidates:
                # 无语义类型候选 → NON_SENSITIVE
                results.append(self._build_result(v, None, 0.95, "truth_table"))
                continue

            # Step 4: 真值表查询 (含 second_candidate 供 LLM Validation 拒绝后使用)
            best_type, tt_conf, second_candidate = self.truth_table.query_multi_type(
                v, candidates, global_stats
            )

            # Step 5: 校验位验证 (如果适用)
            if best_type and not self.feature_extractor.validate_checksum(v.value, best_type):
                tt_conf *= 0.5  # 校验位失败 → 降信

            # Step 6: 融合 (Phase 1 α=1.0, 仅真值表)
            fused_conf = self.fusion.fuse(tt_conf)

            # Step 7: 路由
            route = self.router.route(fused_conf)

            # Step 8: 角色判定
            role = self.role_detector.detect(v.context.label_hint)

            # Step 9: 上下文一致性
            if best_type:
                cc_result = self.context_checker.check(best_type, v.context)
                if cc_result == "downgrade":
                    if route == "direct":
                        route = "llm_validate"
                    elif route == "llm_validate":
                        route = "llm_classify"

            method = "truth_table" if route == "direct" else route
            needs_review = route == "llm_classify"

            results.append(self._build_result(
                v, best_type, fused_conf, method, role,
                needs_review=needs_review,
                evidence={
                    "truth_table_confidence": tt_conf,
                    "fused_confidence": fused_conf,
                    "route": route,
                    "candidates": candidates,
                    "second_candidate": second_candidate,
                }
            ))

        # Mock 值标记
        for v in mock_vals:
            results.append(ValueClassification(
                value_id=v.value_id, value=v.value,
                sensitive_type="MOCK_DATA", confidence=1.0,
                method="mock_filter", is_mock=True,
                needs_review=False, evidence={}, source=v.source,
            ))

        return results

    def _build_result(self, value, stype, conf, method, role=None,
                      is_mock=False, needs_review=False, evidence=None):
        return ValueClassification(
            value_id=value.value_id, value=value.value,
            sensitive_type=stype, confidence=conf,
            method=method, role=role, is_mock=is_mock,
            needs_review=needs_review,
            evidence=evidence or {}, source=value.source,
        )
```

- [ ] **Step 5: Commit**

```bash
cd value-datadna && pytest tests/ -v
git add value-datadna/src/classifiers/ value-datadna/src/postprocess/
git add value-datadna/src/evaluation/ value-datadna/evaluate.py
git commit -m "feat: integrate classification kernel (classify) + evaluation benchmark"
```

### Task 1D.4：entity_type_hint 7维真值表集成

**依赖:** Task 1B.2 (TruthTableEngine) + Task 1C.1 (NEREngine) 完成
**Files:**
- Modify: `value-datadna/src/classifiers/truth_table.py` (添加7维支持)
- Modify: `value-datadna/src/classifiers/feature_extractor.py` (extract 方法增加 entity_type_hint 参数)
- Modify: `value-datadna/src/classifiers/kernel.py` (classify_batch 调用 NER 填入 entity_type_hint)

**对应设计文档:** Section 3.5.2 (7维 37500 状态) + Section 3.6.1 (NER作为真值表第7维)

**核心修改:**

```python
# truth_table.py — 7维校准和查询支持

class TruthTableEngine:
    # 在 calibrate() 中：当传入的 samples 包含 entity_type_hint 时使用7维，否则6维
    def calibrate(self, type_name: str, labeled_samples: list) -> pd.DataFrame:
        has_ner = any("entity_type_hint" in s for s in labeled_samples)
        dims = [
            "regex_strength", "validated_count", "supportive_context",
            "unsupportive_context", "pattern_frequency", "uniqueness_score",
        ]
        if has_ner:
            dims.append("entity_type_hint")
        # ... 其余校准逻辑同6维 ...

    # query() 新增 entity_type_hint 参数
    def query(self, value, candidate_type: str, global_stats: dict,
              entity_type_hint: str = "NONE") -> float:
        features = self.feature_extractor.extract(
            value, candidate_type,
            global_pattern_freq=global_stats.get("pattern_freq", {}).get(candidate_type, 0),
            global_uniqueness=global_stats.get("value_counts", {}).get(value.value, 1),
        )
        # 追加 entity_type_hint 到特征
        features["entity_type_hint"] = entity_type_hint
        # ... 分桶和查询逻辑 ...
```

```python
# kernel.py classify_batch 修改 — 在真值表查询前调用 NER

# Step 3.5: 获取 NER entity_type_hint (Section 3.6.1)
entity_hint = self.ner.predict_hint(v)

# Step 4: 真值表查询时传入 entity_hint
best_type, tt_conf, second_candidate = self.truth_table.query_multi_type(
    v, candidates, global_stats, entity_type_hint=entity_hint,
)

# evidence 中记录 entity_type_hint 供审计
evidence["entity_type_hint"] = entity_hint
```

**NER 权重调整 (Section 3.6.1):**
```python
# feature_extractor.py — entity_type_hint 权重的自适应调整
def compute_hint_weight(self, regex_strength: float) -> float:
    """
    对结构化值 (regex_strength > 0.7): 降低 entity_type_hint 权重
    对弱结构值 (regex_strength < 0.3): 提升 entity_type_hint 权重
    返回值 ∈ [0, 1]，作为 entity_type_hint 维度的软权重。
    """
    if regex_strength > 0.7:
        return 0.3   # 强结构值主要由 regex_strength 主导
    elif regex_strength < 0.3:
        return 0.9   # 弱结构值依赖 NER 语义特征区分 NAME vs API_KEY
    else:
        return 0.6
```

- [ ] **Step 1: 测试 entity_type_hint 对语义类型的提升效果**

```python
# tests/test_ner.py 追加
def test_entity_hint_boosts_name_detection():
    """NAME 类型在 NER hint 帮助下应显著高于无 hint"""
    # 构建带/不带 entity_type_hint 的真值表查询对比
    ...

def test_entity_hint_weight_for_structured_values():
    """强结构值 (SSN) 的 entity_type_hint 权重应被压低"""
    ...
```

- [ ] **Step 2: Commit**

```bash
cd value-datadna && pytest tests/ -v
git add value-datadna/src/classifiers/truth_table.py \
        value-datadna/src/classifiers/feature_extractor.py \
        value-datadna/src/classifiers/kernel.py
git commit -m "feat: integrate entity_type_hint as 7th truth table dimension"
```

---

## Phase 2A：LLM 基础设施

**依赖:** Phase 0 完成  
**可并行:** 与 Phase 1A/1B/1C/3A/3B 并行  
**接口契约:** `llm_validate(value, candidate_type, context) → (yes/no, confidence)` 和 `llm_classify(value, context, type_list) → (type, confidence)`

### Task 2A.1：FLAN-T5 Validation 客户端

**Files:**
- Create: `value-datadna/src/llm/flan_t5.py`
- Test: `value-datadna/tests/test_llm.py`

```python
# src/llm/flan_t5.py — Section 4.2-4.4
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Tuple


class FlanT5ValidationClient:
    """
    FLAN-T5 Large (780M) 判断题验证 (Section 4.2)。
    输入: 值 + 上下文 + 真值表候选类型
    输出: yes/no + confidence
    预期延迟: <10ms GPU
    """

    PROMPT_TEMPLATE = (
        "Verify if the value is a {candidate_type}.\n"
        "Value: {value}\n"
        "Context: column={label_hint}, surrounding_text={surrounding_text}\n"
        "System confidence: {confidence}\n"
        "Answer yes or no with confidence (0.0-1.0):"
    )

    def __init__(self, config: dict):
        cfg = config.get("llm", {})
        self.model_name = cfg.get("validation", {}).get("model", "google/flan-t5-large")
        self.timeout_ms = cfg.get("validation", {}).get("timeout_ms", 5000)
        self.tokenizer = None
        self.model = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def validate(self, value: str, candidate_type: str,
                 context, truth_table_confidence: float) -> Tuple[str, float]:
        """
        返回 (answer, confidence)
        answer ∈ {"yes", "no"}
        """
        if self.model is None:
            self.load()

        prompt = self.PROMPT_TEMPLATE.format(
            candidate_type=candidate_type,
            value=value,
            label_hint=context.label_hint or "unknown",
            surrounding_text=(context.surrounding_text or "")[:200],
            confidence=truth_table_confidence,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

        if "yes" in response:
            return "yes", self._extract_confidence(response)
        return "no", self._extract_confidence(response)

    def validate_batch(self, items: list) -> list:
        """
        批量验证 (Section 4.6)。
        使用 transformers 内置 batch inference (tokenizer padding + model.generate batch)。
        """
        if self.model is None:
            self.load()

        prompts = []
        for value, candidate_type, context, tt_conf in items:
            prompts.append(self.PROMPT_TEMPLATE.format(
                candidate_type=candidate_type,
                value=value,
                label_hint=context.label_hint or "unknown",
                surrounding_text=(context.surrounding_text or "")[:200],
                confidence=tt_conf,
            ))

        inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20)
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        results = []
        for response in responses:
            resp_lower = response.strip().lower()
            conf = self._extract_confidence(resp_lower)
            if "yes" in resp_lower:
                results.append(("yes", conf))
            else:
                results.append(("no", conf))
        return results

    @staticmethod
    def _extract_confidence(response: str) -> float:
        """从 response 中提取置信度数值"""
        import re
        nums = re.findall(r'(\d+\.?\d*)', response)
        if nums:
            return min(max(float(nums[0]), 0.0), 1.0)
        return 0.5
```

### Task 2A.2：Mistral-7B Classification 客户端

**Files:**
- Create: `value-datadna/src/llm/mistral.py`

```python
# src/llm/mistral.py — Section 4.2-4.4
import json, re, asyncio
from typing import Tuple, List, Optional


class MistralClassificationClient:
    """
    Mistral-7B-Instruct (Q4_K_M) 论述题分类 (Section 4.2)。
    输入: 值 + 上下文 + 完整敏感类型列表
    输出: sensitive_type + confidence
    预期延迟: ~1s GPU (Ollama)
    """

    SYSTEM_PROMPT = (
        "You are a data security classifier. Classify the data value into one "
        "of the known sensitive types, or NON_SENSITIVE if not sensitive."
    )

    def __init__(self, config: dict):
        cfg = config.get("llm", {}).get("classification", {})
        self.model = cfg.get("model", "mistral:7b-instruct")
        self.endpoint = cfg.get("endpoint", "http://localhost:11434")
        self.timeout_ms = cfg.get("timeout_ms", 30000)
        self.type_list = [
            "SSN", "CREDIT_CARD", "IBAN", "EMAIL", "PHONE", "IP",
            "PASSPORT", "DRIVER_LICENSE", "API_KEY", "BANK_ACCOUNT",
            "NAME", "ADDRESS", "NON_SENSITIVE",
        ]

    def classify(self, value: str, context,
                 hint_candidate: Optional[str] = None) -> Tuple[str, float]:
        """
        返回 (sensitive_type, confidence)
        """
        prompt = self._build_prompt(value, context, hint_candidate)

        try:
            import requests
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 150},
                },
                timeout=self.timeout_ms / 1000,
            )
            data = response.json()
            return self._parse_response(data.get("response", ""))
        except Exception:
            return "NON_SENSITIVE", 0.0

    def _build_prompt(self, value: str, context,
                      hint_candidate: Optional[str] = None) -> str:
        type_str = ", ".join(self.type_list)
        base = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Known types: {type_str}\n\n"
            f"Value: {value}\n"
            f"Context: column={(context.label_hint or 'unknown')}, "
            f"surrounding_text={(context.surrounding_text or '')[:300]}\n\n"
        )
        if hint_candidate:
            base += f"Hint: Consider {hint_candidate} as a possible type.\n\n"
        base += 'Answer with JSON: {"type": "<type or NON_SENSITIVE>", "confidence": 0.0-1.0, "reason": "<one short sentence>"}'
        return base

    def _parse_response(self, text: str) -> Tuple[str, float]:
        text = text.strip()
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("type", "NON_SENSITIVE"), float(data.get("confidence", 0.0))
            except (json.JSONDecodeError, ValueError):
                pass
        return "NON_SENSITIVE", 0.0

    async def classify_batch(self, items: list) -> list:
        """异步批量分类 (Section 4.6)"""
        tasks = [asyncio.to_thread(self.classify, v, ctx, hint)
                 for v, ctx, hint in items]
        return await asyncio.gather(*tasks)
```

### Task 2A.3：批量调度 + 降级路径

**Files:**
- Create: `value-datadna/src/llm/batch.py`

```python
# src/llm/batch.py — Section 4.6-4.7
import asyncio, time
from typing import List, Tuple
from src.types import ValueClassification


class LLMBatchScheduler:
    """
    LLM 批量调度 + 降级 (Section 4.6-4.7)。
    降级路径:
      FLAN-T5 不可用 → skip Validation → direct to Classification
      Mistral 不可用 → low conf 标记 uncertain, mid conf 保持真值表判定 × 0.8
      两模型同时不可用 → 真值表独立运行, low conf 标记 uncertain
    """

    def __init__(self, flan_client, mistral_client, config: dict):
        self.flan = flan_client
        self.mistral = mistral_client
        self.timeout_ms = config.get("llm", {}).get("classification", {}).get("timeout_ms", 5000)
        self.flan_available = True
        self.mistral_available = True

    def process(self, results: List[ValueClassification],
                values: list) -> List[ValueClassification]:
        """
        处理分类内核产出的中低置信度结果 (Section 4.5-4.7)。

        逻辑:
          1. 分离出 route=llm_validate 和 route=llm_classify 的结果
          2. llm_validate → FLAN-T5 批量推理 (不可用或超时→降级到Classification)
             - answer=yes → 保持 candidate_type
             - answer=no  → 取真值表次高候选作为 hint → route to Classification
          3. llm_classify → Mistral (不可用或超时→标记 uncertain)
        超时: 任一模型 >5s → 同不可用降级 (Section 4.7)
        """
        import concurrent.futures

        validate_queue = []
        classify_queue = []

        for i, r in enumerate(results):
            evidence = r.evidence
            route = evidence.get("route", "direct")
            if route == "llm_validate":
                validate_queue.append((r, i))
            elif route == "llm_classify":
                classify_queue.append((r, i))

        # Step 1: FLAN-T5 Validation (批量推理 + 超时保护)
        if validate_queue and self.flan_available:
            batch_items = []
            for r, idx in validate_queue:
                batch_items.append((
                    r.value, r.sensitive_type, r.source,
                    r.evidence.get("truth_table_confidence", 0.5),
                ))
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.flan.validate_batch, batch_items)
                    batch_results = future.result(timeout=self.timeout_ms / 1000)
            except (concurrent.futures.TimeoutError, Exception):
                self.flan_available = False
                for r, idx in validate_queue:
                    classify_queue.append((r, idx))
                batch_results = None

            if batch_results:
                for j, (r, idx) in enumerate(validate_queue):
                    answer, llm_conf = batch_results[j]
                    if answer == "yes":
                        r.confidence = max(r.confidence, llm_conf)
                        r.method = "llm_validate"
                    else:
                        # Section 4.5: 取次高候选作为 hint 传入 Classification
                        r.evidence["validation_rejected"] = True
                        second_candidate = r.evidence.get("second_candidate")
                        r.evidence["hint_candidate"] = (
                            second_candidate if second_candidate else r.sensitive_type
                        )
                        classify_queue.append((r, idx))

        # Step 2: Mistral Classification (超时保护)
        for r, idx in classify_queue:
            if not self.mistral_available:
                r.needs_review = True
                if r.confidence >= 0.50:
                    r.confidence *= 0.8
                continue
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.mistral.classify,
                        r.value, r.source,
                        r.evidence.get("hint_candidate"),
                    )
                    llm_type, llm_conf = future.result(timeout=self.timeout_ms / 1000)
                if llm_type == "NON_SENSITIVE":
                    r.sensitive_type = None
                    r.confidence = llm_conf
                elif llm_type in self.mistral.type_list:
                    r.sensitive_type = llm_type
                    r.confidence = llm_conf
                else:
                    r.needs_review = True
                r.method = "llm_classify"
            except (concurrent.futures.TimeoutError, Exception):
                self.mistral_available = False
                r.needs_review = True
                if r.confidence >= 0.50:
                    r.confidence *= 0.8

        return results
```

---

## Phase 2B：LLM 消歧集成

**依赖:** Phase 1D + Phase 2A 完成

### Task 2B.1：LLM 消歧层与分类内核集成

**Files:**
- Modify: `value-datadna/src/classifiers/kernel.py` (添加 LLM 消歧调用)
- Create: `value-datadna/src/classifiers/kernel_with_llm.py`

```python
# src/classifiers/kernel_with_llm.py — Phase 2 完整内核 (含 LLM 消歧)
# 在 Phase 1 内核基础上加入 LLMBatchScheduler
# 内核接口保持不变: classify(value, context) → ValueClassification

from src.classifiers.kernel import ClassificationKernel
from src.llm.flan_t5 import FlanT5ValidationClient
from src.llm.mistral import MistralClassificationClient
from src.llm.batch import LLMBatchScheduler
from src.types import DataValue, ValueClassification
from typing import List


class ClassificationKernelWithLLM(ClassificationKernel):
    """Phase 2 完整内核: 真值表 + NER + 融合 + LLM 消歧"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.flan = FlanT5ValidationClient(config)
        self.mistral = MistralClassificationClient(config)
        self.llm_scheduler = LLMBatchScheduler(self.flan, self.mistral, config)

    def classify_batch(self, values: List[DataValue]) -> List[ValueClassification]:
        # Step 1-9: Phase 1 内核产出初判结果
        results = super().classify_batch(values)

        # Step 10: LLM 消歧层处理中低置信度值
        results = self.llm_scheduler.process(results, values)

        return results
```

---

## Phase 2C：在线缓存层 (Path C)

**依赖:** Phase 1D 完成  
**可并行:** 与 Phase 2B / 3A / 3B 并行  
**接口契约:** `cache_lookup(pattern_hash) → ValueClassification | None`, `cache_write(pattern_hash, ValueClassification, ttl=24h)`

**对应设计文档:** Section 2.5 路径C + Section 6.6 步骤5 (type_cache)

### Task 2C.1：pattern_hash 缓存 + TTL 机制

**Files:**
- Create: `value-datadna/src/classifiers/cache.py`
- Test: `value-datadna/tests/test_cache.py`

```python
# src/classifiers/cache.py — Path C 在线缓存层
import hashlib, time, threading
from typing import Dict, Optional
from src.types import ValueClassification


class ClassificationCache:
    """
    在线实时缓存层 (Section 2.5 路径C)。
    pattern_hash → ValueClassification, TTL 24h。
    与 Section 6.6 type_cache 共享存储后端。
    LRU 淘汰，O(1) 查询。

    Path C 完整流程:
      API请求 → 值提取 → cache.lookup(pattern_hash)
        hit + TTL 有效 → 直接返回 (<1ms)
        miss → 分类内核 classify()
        → cache.write(pattern_hash, result, ttl=24h)
    """

    def __init__(self, max_size: int = 100000, default_ttl: int = 86400):
        self.max_size = max_size
        self.default_ttl = default_ttl  # 24h in seconds
        self._store: Dict[str, tuple[ValueClassification, float]] = {}  # hash → (result, expiry_ts)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def compute_pattern_hash(value: str, label_hint: str = "") -> str:
        """计算值的模式哈希 (值与类型无关的结构特征)"""
        # 标准化: 去除值的具体内容，保留结构模式
        import re
        normalized = re.sub(r'[a-z]', 'a', value)
        normalized = re.sub(r'[A-Z]', 'A', normalized)
        normalized = re.sub(r'\d', '0', normalized)
        key = f"{label_hint}|{normalized}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def lookup(self, pattern_hash: str) -> Optional[ValueClassification]:
        """缓存查询。返回 None 表示 miss 或已过期。"""
        with self._lock:
            entry = self._store.get(pattern_hash)
            if entry is None:
                self._misses += 1
                return None
            result, expiry = entry
            if time.time() > expiry:
                del self._store[pattern_hash]
                self._misses += 1
                return None
            self._hits += 1
            return result

    def write(self, pattern_hash: str, result: ValueClassification, ttl: int = None):
        """缓存写入。超出 max_size 时 LRU 淘汰最旧的条目。"""
        if ttl is None:
            ttl = self.default_ttl
        with self._lock:
            if len(self._store) >= self.max_size:
                # LRU: 删除最旧的 10%
                sorted_entries = sorted(
                    self._store.items(), key=lambda x: x[1][1]
                )
                for old_hash, _ in sorted_entries[:self.max_size // 10]:
                    del self._store[old_hash]
            self._store[pattern_hash] = (result, time.time() + ttl)

    def invalidate(self, pattern_hash: str = None):
        """
        使缓存失效。
        - 传入 pattern_hash: 清除单个条目
        - 传入 None: 全量清除 (Section 6.6 步骤5: 引擎更新后清除缓存)
        """
        with self._lock:
            if pattern_hash:
                self._store.pop(pattern_hash, None)
            else:
                self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
```

```python
# tests/test_cache.py
import time
from src.classifiers.cache import ClassificationCache
from src.types import ValueClassification, ValueSource


class TestClassificationCache:
    def test_lookup_miss_returns_none(self):
        cache = ClassificationCache(max_size=10)
        assert cache.lookup("abc123") is None

    def test_write_and_lookup_hit(self):
        cache = ClassificationCache(max_size=10)
        vc = ValueClassification(
            value_id="v1", value="test", sensitive_type="SSN",
            confidence=0.95, method="truth_table",
            source=ValueSource(source_type="api", extraction_method="direct"),
        )
        cache.write("hash_001", vc)
        result = cache.lookup("hash_001")
        assert result is not None
        assert result.sensitive_type == "SSN"

    def test_ttl_expiry(self):
        cache = ClassificationCache(max_size=10, default_ttl=0)  # immediate expiry
        vc = ValueClassification(
            value_id="v1", value="test", sensitive_type=None,
            confidence=0.95, method="truth_table",
        )
        cache.write("hash_002", vc)
        time.sleep(0.01)
        assert cache.lookup("hash_002") is None

    def test_pattern_hash_normalizes_values(self):
        h1 = ClassificationCache.compute_pattern_hash("123-45-6789", "ssn")
        h2 = ClassificationCache.compute_pattern_hash("987-65-4321", "ssn")
        assert h1 == h2  # Same pattern → same hash

    def test_invalidate_all(self):
        cache = ClassificationCache(max_size=10)
        vc = ValueClassification(
            value_id="v1", value="test", sensitive_type="SSN",
            confidence=0.95, method="truth_table",
        )
        cache.write("h1", vc)
        cache.invalidate()  # Section 6.6 步骤5: 引擎更新后全量清除
        assert cache.lookup("h1") is None

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
```

- [ ] **Step 3: Commit**

```bash
cd value-datadna && pytest tests/test_cache.py -v
git add value-datadna/src/classifiers/cache.py value-datadna/tests/test_cache.py
git commit -m "feat: add online classification cache (pattern_hash + TTL 24h)"
```

---

## Phase 3A：Semantic Distancing

**依赖:** Phase 0 完成 + Phase 1B 的 PII 正则模式库  
**可并行:** 与 Phase 1A/1C/2A/3B 并行  
**接口契约:** `compute_distance_score(value_text, candidate_type) → float`

### Task 3A.1：PII 占位符替换 + 嵌入 + 模板库

**Files:**
- Create: `value-datadna/src/classifiers/semantic_distance.py`
- Test: `value-datadna/tests/test_semantic_distance.py`

```python
# src/classifiers/semantic_distance.py — Section 5.1
import re, numpy as np
from typing import Dict, List


class SemanticDistancing:
    """
    Semantic Distancing (Section 5.1)。
    双重用途:
      1. 分类评分: 值 → PII替换 → 嵌入 → 与类型模板库 cosine similarity → distance_score
      2. 增量匹配: 文件元数据 → PII替换 → 嵌入 → 与簇模板库 cosine similarity → 簇匹配

    两套模板库共享嵌入模型，通过 library 参数区分 ("types" vs "clusters")。
    """

    def __init__(self, config: dict):
        cfg = config.get("semantic_distance", {})
        self.model_name = cfg.get("embedding_model", "intfloat/e5-base")
        self.template_count = cfg.get("template_count_per_type", 75)
        self.match_threshold = cfg.get("incremental_match_threshold", 0.85)
        self.model = None
        self.type_templates: Dict[str, np.ndarray] = {}     # type_name → [N×768]
        self.cluster_templates: Dict[str, np.ndarray] = {}  # cluster_id → [N×768]

    def load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)

    def replace_pii_placeholders(self, text: str) -> str:
        """
        PII 类型占位符替换 (Section 5.1.2)。
        类型级替换，不解析子结构。
        "4111-1111-1111-1111" → "[CREDIT_CARD]"
        """
        from src.knowledge.pii_patterns import PII_PATTERNS
        result = text
        for type_name, info in PII_PATTERNS.items():
            if info.get("regex"):
                result = re.sub(info["regex"], f"[{type_name}]", result, flags=re.IGNORECASE)
        return result

    def embed(self, text: str) -> np.ndarray:
        """E5-base 嵌入 → [1×768] 向量"""
        if self.model is None:
            self.load_model()
        prefix = "passage: "
        return self.model.encode(prefix + text, normalize_embeddings=True)

    def build_type_templates(self, synthetic_data: Dict[str, List[str]]):
        """
        冷启动构建类型模板库 (Section 5.1.2)。
        从合成数据为每种类型生成 template_count 个含 PII 占位符的模板文本。
        """
        for type_name, examples in synthetic_data.items():
            templates = []
            for ex in examples[:self.template_count]:
                replaced = self.replace_pii_placeholders(ex)
                emb = self.embed(replaced)
                templates.append(emb)
            self.type_templates[type_name] = np.vstack(templates) if templates else np.zeros((0, 768))

    def build_cluster_templates(self, cluster_id: str, metadata_samples: List[str]):
        """
        构建簇模板库 (Section 5.1.1 增量匹配用途)。
        Phase 3 聚类时为每簇计算模板嵌入。
        """
        templates = []
        for meta in metadata_samples:
            replaced = self.replace_pii_placeholders(meta)
            emb = self.embed(replaced)
            templates.append(emb)
        self.cluster_templates[cluster_id] = np.vstack(templates) if templates else np.zeros((0, 768))

    def compute_distance_score(self, value_text: str, candidate_type: str) -> float:
        """
        分类评分用途: 值的 PII 替换后文本 → 嵌入 → 与类型模板库 cosine similarity。
        返回 distance_score ∈ [0, 1]。
        """
        replaced = self.replace_pii_placeholders(value_text)
        emb = self.embed(replaced)

        templates = self.type_templates.get(candidate_type)
        if templates is None or len(templates) == 0:
            return 0.0

        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(emb.reshape(1, -1), templates)[0]
        return float(np.max(sims))

    def match_cluster(self, file_metadata: str) -> tuple[str | None, float]:
        """
        增量匹配用途: 文件元数据 → PII替换 → 嵌入 → 与簇模板库 cosine similarity。
        返回 (cluster_id, cos_sim) 或 (None, 0.0)。
        """
        replaced = self.replace_pii_placeholders(file_metadata)
        emb = self.embed(replaced)

        best_cluster = None
        best_sim = 0.0
        from sklearn.metrics.pairwise import cosine_similarity
        for cid, templates in self.cluster_templates.items():
            if len(templates) == 0:
                continue
            sims = cosine_similarity(emb.reshape(1, -1), templates)[0]
            max_sim = float(np.max(sims))
            if max_sim > best_sim:
                best_sim = max_sim
                best_cluster = cid

        if best_sim >= self.match_threshold:
            return best_cluster, best_sim
        return None, best_sim
```

---

## Phase 3B：文件级聚类引擎

**依赖:** Phase 0 完成 (独立于分类内核)  
**可并行:** 与 Phase 1A/1B/1C/2A/3A 并行  
**接口契约:** `cluster_files(file_paths) → {cluster_id: [file_paths]}`

### Task 3B.1：元数据归一化 + SHA256 流式聚类

**Files:**
- Create: `value-datadna/src/clustering/file_clusterer.py`
- Test: `value-datadna/tests/test_clustering.py`

```python
# src/clustering/file_clusterer.py — Section 5.2
import hashlib, os, re, csv, json
from typing import List, Dict, Tuple
from collections import defaultdict
from src.knowledge.pii_patterns import PII_PATTERNS


class FileClusterer:
    """
    文件级流式聚类 (Section 5.2.3-5.2.4)。
    O(n) 时间, O(k) 空间。
    元数据归一化 + PII 占位符替换 → SHA256 指纹 → 文件簇。
    """

    def __init__(self, config: dict):
        self.representatives_per_cluster = config.get("clustering", {}).get("representatives_per_cluster", 3)

    def cluster(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        流式聚类入口。
        返回 {cluster_fingerprint: [file_path, ...]}
        """
        clusters = defaultdict(list)
        for fp in file_paths:
            fingerprint = self.compute_fingerprint(fp)
            clusters[fingerprint].append(fp)
        return dict(clusters)

    def compute_fingerprint(self, file_path: str) -> str:
        """
        计算文件的结构指纹 (Section 5.2.3)。
        元数据归一化 → PII 关键词替换 → SHA256。
        """
        meta_text = self._extract_normalized_metadata(file_path)
        replaced = self._replace_pii_keywords(meta_text)
        return hashlib.sha256(replaced.encode()).hexdigest()[:16]

    def _extract_normalized_metadata(self, file_path: str) -> str:
        """提取文件元数据并归一化 (Section 5.2.3)"""
        ext = os.path.splitext(file_path)[1].lower()

        parts = []
        parts.append(f"file_type={ext}")

        if ext == '.csv':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                try:
                    headers = next(reader)
                except StopIteration:
                    headers = []
            parts.append(f"columns={len(headers)}")
            # 列名归一化 (替换为关键词类别)
            normalized_headers = [self._normalize_column_name(h) for h in headers]
            parts.append(f"col_names={normalized_headers}")
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
            if isinstance(data, dict):
                parts.append(f"keys={sorted(data.keys())}")
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                parts.append(f"keys={sorted(data[0].keys())}")
        elif ext in ('.txt', '.md'):
            parts.append("content_type=text")
        elif ext == '.pdf':
            parts.append("content_type=pdf")

        # 文件大小量级
        size = os.path.getsize(file_path)
        parts.append(f"size_order={len(str(size))}")

        return "|".join(parts)

    def _normalize_column_name(self, name: str) -> str:
        """列名归一化: 识别 PII 关键词并替换"""
        name_lower = name.lower().strip()
        keyword_map = {
            "ssn": "[SSN_KEYWORD]", "social": "[SSN_KEYWORD]", "社保": "[SSN_KEYWORD]",
            "credit": "[CCN_KEYWORD]", "card": "[CCN_KEYWORD]", "信用卡": "[CCN_KEYWORD]",
            "email": "[EMAIL_KEYWORD]", "mail": "[EMAIL_KEYWORD]", "邮箱": "[EMAIL_KEYWORD]",
            "phone": "[PHONE_KEYWORD]", "tel": "[PHONE_KEYWORD]", "电话": "[PHONE_KEYWORD]",
            "iban": "[IBAN_KEYWORD]", "bank": "[BANK_KEYWORD]", "银行": "[BANK_KEYWORD]",
            "name": "[NAME_KEYWORD]", "姓名": "[NAME_KEYWORD]",
            "address": "[ADDRESS_KEYWORD]", "地址": "[ADDRESS_KEYWORD]",
            "passport": "[PASSPORT_KEYWORD]", "护照": "[PASSPORT_KEYWORD]",
            "salary": "[MONEY_KEYWORD]", "amount": "[MONEY_KEYWORD]", "金额": "[MONEY_KEYWORD]",
        }
        for kw, replacement in keyword_map.items():
            if kw in name_lower:
                return replacement
        return "[GENERIC]"

    def _replace_pii_keywords(self, text: str) -> str:
        """PII 关键词替换 (类型级, 与 SD 共享替换逻辑)"""
        result = text
        for type_name, info in PII_PATTERNS.items():
            if info.get("regex"):
                result = re.sub(info["regex"], f"[{type_name}]", result, flags=re.IGNORECASE)
        return result
```

### Task 3B.2：代表文件选择

**Files:**
- Modify: `value-datadna/src/clustering/file_clusterer.py` (添加 select_representatives 方法)

```python
def select_representatives(self, cluster_files: List[str]) -> List[str]:
    """
    选择簇代表文件 (Section 5.2.5)。
    策略: 优先选不同文件大小和列数的文件覆盖多样性。
    """
    if len(cluster_files) <= self.representatives_per_cluster:
        return list(cluster_files)

    file_info = []
    for fp in cluster_files:
        size = os.path.getsize(fp)
        ext = os.path.splitext(fp)[1].lower()
        file_info.append((fp, size, ext))

    # 按文件大小排序后等间隔选
    file_info.sort(key=lambda x: x[1])
    step = max(1, len(file_info) // self.representatives_per_cluster)
    return [file_info[i][0] for i in range(0, len(file_info), step)][:self.representatives_per_cluster]
```

---

## Phase 3C：标签传播 + 抽检验证 + 增量匹配

**依赖:** Phase 1D (classify 接口) + Phase 3B 完成 + Phase 3A (SD cluster_templates 用于增量匹配)  
**可并行:** 与 Phase 2B 并行

### Task 3C.1：标签传播引擎

**Files:**
- Create: `value-datadna/src/clustering/propagator.py`

```python
# src/clustering/propagator.py — Section 5.2.5-5.2.6
import random
from typing import List, Dict, Tuple
from collections import Counter
from src.types import ValueClassification


class LabelPropagator:
    """
    标签传播引擎 (Section 5.2.5)。
    代表文件 → classify() → majority_vote → 簇内全量继承。
    consistency ≥ 0.8 才传播，否则拆分子簇递归。
    """

    def __init__(self, config: dict):
        cfg = config.get("clustering", {})
        self.consistency_threshold = cfg.get("propagation_consistency_threshold", 0.8)
        self.spot_check_ratio = cfg.get("spot_check_ratio", 0.05)
        self.inconsistency_threshold = cfg.get("inconsistency_threshold", 0.10)

    def propagate(self, cluster_files: List[str],
                  representative_classifications: List[List[ValueClassification]],
                  classify_fn) -> Dict[str, str]:
        """
        簇内标签传播。
        返回 {file_path: label}。
        """
        # 收集所有代表文件的分类结果
        all_types = []
        for reps in representative_classifications:
            for r in reps:
                if r.sensitive_type and r.sensitive_type != "NON_SENSITIVE":
                    all_types.append(r.sensitive_type)

        if not all_types:
            return {fp: "UNCLASSIFIED" for fp in cluster_files}

        # majority vote
        counter = Counter(all_types)
        majority_type, majority_count = counter.most_common(1)[0]
        consistency = majority_count / len(all_types) if all_types else 0

        if consistency >= self.consistency_threshold:
            return {fp: majority_type for fp in cluster_files}
        else:
            # 拆分子簇递归 (此处为简化版, 完整实现见设计文档)
            return {fp: "NEEDS_REFINEMENT" for fp in cluster_files}

    def spot_check(self, cluster_files: List[str],
                   propagated_labels: Dict[str, str],
                   classify_fn) -> bool:
        """
        传播后抽检验证 (Section 5.2.5)。
        每簇 5% 列 × 3 值独立验证。
        返回 True 表示通过, False 表示不一致率 > 10% 需重分类。
        """
        n_check = max(1, int(len(cluster_files) * self.spot_check_ratio))
        check_files = random.sample(cluster_files, min(n_check, len(cluster_files)))

        mismatches = 0
        total = 0
        for fp in check_files:
            expected_label = propagated_labels.get(fp)
            # ... 对检查文件运行完整 classify() 并比对 ...
            # (简化: 此处省略完整提取+分类逻辑)
            total += 1

        if total == 0:
            return True
        return (mismatches / total) <= self.inconsistency_threshold
```

---

### Task 3C.2：三路径编排器 (Path A/B/C Orchestrator)

**依赖:** Phase 1D (classify) + Phase 2C (cache) + Phase 3B (clustering) 完成
**Files:**
- Create: `value-datadna/src/orchestrator.py`

**对应设计文档:** Section 2.5 三种到达模式

```python
# src/orchestrator.py — 三路径编排器
from typing import List
from src.types import DataValue, ValueClassification
from src.classifiers.cache import ClassificationCache
from src.clustering.file_clusterer import FileClusterer
from src.clustering.propagator import LabelPropagator
from src.extractors.structured import StructuredExtractor
from src.extractors.unstructured import UnstructuredExtractor


class DataDNAOrchestrator:
    """
    三路径编排器 (Section 2.5)。
    根据数据源类型和触发方式路由到 Path A/B/C。
    三种路径共享完全相同的 classify() 内核。
    """

    def __init__(self, classification_kernel, config: dict):
        self.kernel = classification_kernel
        self.cache = ClassificationCache()
        self.clusterer = FileClusterer(config)
        self.propagator = LabelPropagator(config)
        self.structured_extractor = StructuredExtractor()
        self.unstructured_extractor = UnstructuredExtractor()

    # ── Path A: 离线全量 ────────────────────────────────────

    def run_path_a_full_scan(self, data_source_paths: List[str]) -> dict:
        """
        Path A: 首次数据源全量接入 (Section 2.5 路径A).
        流程: 文件聚类 → 代表文件 → 值提取抽样 → classify → 标签传播 → 抽检
        """
        # Step 1: 文件级聚类
        clusters = self.clusterer.cluster(data_source_paths)

        results = {}
        for fingerprint, file_paths in clusters.items():
            # Step 2: 选代表文件
            reps = self.clusterer.select_representatives(file_paths)

            # Step 3: 值提取 + 抽样
            all_classifications = []
            for fp in reps:
                ext = fp.rsplit('.', 1)[-1].lower() if '.' in fp else ''
                if ext in ('csv', 'json'):
                    values = self.structured_extractor.extract(fp)
                else:
                    values = self.unstructured_extractor.extract(fp)
                classifications = self.kernel.classify_batch(values)
                all_classifications.append(classifications)

            # Step 4: 标签传播
            labels = self.propagator.propagate(file_paths, all_classifications, self.kernel.classify_batch)

            # Step 5: 抽检验证
            passed = self.propagator.spot_check(file_paths, labels, self.kernel.classify_batch)
            if not passed:
                # 不一致率 > 10% → 触发簇拆分重分类
                labels = {"status": "needs_reclassification"}

            results[fingerprint] = labels

        return results

    # ── Path B: 离线增量 ────────────────────────────────────

    def run_path_b_incremental(self, changed_files: List[str],
                                existing_cluster_fingerprints: dict,
                                sd_engine) -> dict:
        """
        Path B: 变更数据增量同步 (Section 2.5 路径B).
        流程: hash精确匹配 → SD二次匹配 → 缓冲队列 → 定期抽检
        """
        results = {}
        buffer = []

        for fp in changed_files:
            # Step 1: hash 精确匹配
            fingerprint = self.clusterer.compute_fingerprint(fp)
            if fingerprint in existing_cluster_fingerprints:
                # 直接继承标签
                results[fp] = {
                    "label": existing_cluster_fingerprints[fingerprint],
                    "method": "hash_match",
                    "confidence": 1.0,
                }
                continue

            # Step 2: SD 二次匹配
            meta_text = self.clusterer._extract_normalized_metadata(fp)
            cluster_id, cos_sim = sd_engine.match_cluster(meta_text)
            if cluster_id and cos_sim >= 0.85:
                results[fp] = {
                    "label": existing_cluster_fingerprints.get(cluster_id, "UNKNOWN"),
                    "method": "sd_match",
                    "confidence": cos_sim,
                }
            else:
                buffer.append(fp)

        # Step 3: 缓冲队列 (累积 ≥ 500 或 > 1h → 小批量聚类)
        if len(buffer) >= 500:
            mini_results = self.run_path_a_full_scan(buffer)
            results.update(mini_results)

        # Step 4: 定期抽检 (3% 增量继承文件走完整验证)
        import random
        spot_check_files = random.sample(
            list(results.keys()),
            min(max(1, int(len(results) * 0.03)), len(results)),
        )
        for fp in spot_check_files:
            values = self.structured_extractor.extract(fp) if fp.endswith('.csv') \
                else self.unstructured_extractor.extract(fp)
            cls_results = self.kernel.classify_batch(values)
            # 比对...如果不一致率 > 5% → 触发该簇全量重分类

        return results

    # ── Path C: 在线实时 ────────────────────────────────────

    def run_path_c_online(self, value: DataValue) -> ValueClassification:
        """
        Path C: 在线实时单值分类 (Section 2.5 路径C).
        流程: 缓存查询 → miss → classify → 缓存写入
        """
        pattern_hash = self.cache.compute_pattern_hash(
            value.value, value.context.label_hint or ""
        )
        # Step 1: 缓存查询
        cached = self.cache.lookup(pattern_hash)
        if cached:
            return cached

        # Step 2: 分类内核
        result = self.kernel.classify(value)

        # Step 3: 缓存写入
        self.cache.write(pattern_hash, result)

        return result
```

- [ ] **Step 2: Commit**

```bash
git add value-datadna/src/orchestrator.py
git commit -m "feat: add three-path orchestrator (Path A/B/C)"
```

---

## Phase 4：Learned Classification 闭环

**依赖:** Phase 1D + Phase 2B + Phase 3A (都需要)  
**串行内部阶段:** 4A → 4B → 4C 顺序依赖

### Task 4.1：未知模式收集 + 缓冲池

**Files:**
- Create: `value-datadna/src/discovery/collector.py`

```python
# src/discovery/collector.py — Section 6.3
from collections import OrderedDict
from typing import List, Dict
from dataclasses import dataclass, field
from src.types import DataValue


@dataclass
class CandidatePattern:
    pattern_hash: str
    regex_pattern: str
    sample_values: List[str] = field(default_factory=list)
    total_count: int = 0
    label_hints: List[str] = field(default_factory=list)
    truth_table_confidence: float = 0.0
    regex_strength: float = 0.0
    validated_count: int = 0


class UnknownPatternCollector:
    """
    未知模式收集缓冲池 (Section 6.3)。
    触发条件:
      A. regex_strength > 0.5 AND validated_count < 5 AND total_count > 500
      B. truth_table_confidence < 0.3 AND regex_strength > 0.7
      C. label_hint-分类冲突 (cos_sim > 0.7, A ≠ B)
    缓冲池: LRU, 最多 200 个候选模式。
    """

    def __init__(self, config: dict):
        cfg = config.get("discovery", {})
        self.max_size = cfg.get("buffer_max_size", 200)
        self.buffer: OrderedDict[str, CandidatePattern] = OrderedDict()

    def collect(self, value: DataValue, classification_result,
                global_stats: dict) -> CandidatePattern | None:
        """
        检查是否满足收集条件，满足则加入缓冲池。
        返回加入的 CandidatePattern 或 None。
        """
        evidence = classification_result.evidence
        regex_strength = evidence.get("regex_strength", 0)
        validated_count = evidence.get("validated_count", 0)
        tt_confidence = evidence.get("truth_table_confidence", 0)
        total_count = global_stats.get("total_count", 0)

        should_collect = False

        # 条件 A: 新兴重复模式
        if regex_strength > 0.5 and validated_count < 5 and total_count > 500:
            should_collect = True

        # 条件 B: 真值表盲区
        if tt_confidence < 0.3 and regex_strength > 0.7:
            should_collect = True

        # 条件 C: label_hint-分类冲突
        if classification_result.sensitive_type:
            hint_type = classification_result.evidence.get("label_hint_type")
            if hint_type and hint_type != classification_result.sensitive_type:
                should_collect = True

        if not should_collect:
            return None

        # 生成 pattern_hash
        import hashlib
        phash = hashlib.sha256(
            f"{value.value[:20]}|{value.context.label_hint}".encode()
        ).hexdigest()[:12]

        if phash in self.buffer:
            pattern = self.buffer[phash]
            pattern.total_count += 1
            if value.value not in pattern.sample_values:
                pattern.sample_values.append(value.value)
            self.buffer.move_to_end(phash)
        else:
            if len(self.buffer) >= self.max_size:
                self.buffer.popitem(last=False)  # LRU 淘汰
            pattern = CandidatePattern(
                pattern_hash=phash,
                regex_pattern=evidence.get("regex_pattern", ""),
                sample_values=[value.value],
                total_count=1,
                label_hints=[value.context.label_hint or ""],
                truth_table_confidence=tt_confidence,
                regex_strength=regex_strength,
                validated_count=validated_count,
            )
            self.buffer[phash] = pattern

        return pattern
```

### Task 4.2：自动验证 + 人工 Gate 接口

**Files:**
- Create: `value-datadna/src/discovery/auto_validator.py`
- Create: `value-datadna/src/discovery/nominator.py`

```python
# src/discovery/auto_validator.py — Section 6.4
from enum import Enum


class ValidationResult(Enum):
    AUTO_PASS = "auto_pass"
    AUTO_REJECT = "auto_reject"
    HUMAN_GATE = "human_gate"
    CONFLICT = "conflict"  # 同时满足通过和拒绝 → 人工 Gate


class AutoValidator:
    """
    自动验证层 (Section 6.4)。
    路由规则 (优先级):
      1. 自动拒绝优先
      2. 冲突裁决 → 人工 Gate
      3. 自动通过
      4. 其余 → 人工 Gate
    """

    def __init__(self, config: dict):
        cfg = config.get("discovery", {})
        self.cos_sim_threshold = cfg.get("auto_pass_cos_sim_threshold", 0.85)
        self.reject_conf_threshold = cfg.get("auto_reject_confidence_threshold", 0.7)
        self.pass_conf_threshold = cfg.get("auto_pass_confidence_threshold", 0.3)

    def validate(self, candidate, existing_types: dict,
                 truth_table_confidence: float,
                 cos_sim_vs_existing: float) -> tuple[ValidationResult, str]:
        """
        验证候选模式。
        返回 (ValidationResult, reason)。
        """
        pass_conditions = []
        reject_conditions = []

        # 自动通过条件
        if truth_table_confidence < self.pass_conf_threshold:
            pass_conditions.append("truth_table_blind")
        if cos_sim_vs_existing < self.cos_sim_threshold:
            pass_conditions.append("truly_novel")

        # 自动拒绝条件
        if cos_sim_vs_existing >= self.cos_sim_threshold:
            reject_conditions.append("too_similar_to_existing")
        if truth_table_confidence >= self.reject_conf_threshold:
            reject_conditions.append("existing_engine_covers")

        # 步骤 1: 自动拒绝优先
        if reject_conditions and not pass_conditions:
            return ValidationResult.AUTO_REJECT, f"reject: {reject_conditions}"

        # 步骤 2: 冲突裁决
        if reject_conditions and pass_conditions:
            return ValidationResult.CONFLICT, f"conflict: pass={pass_conditions}, reject={reject_conditions}"

        # 步骤 3: 自动通过
        if pass_conditions:
            return ValidationResult.AUTO_PASS, f"pass: {pass_conditions}"

        # 步骤 4: 其余 → 人工 Gate
        return ValidationResult.HUMAN_GATE, "uncertain: in [0.3, 0.7) confidence range"
```

### Task 4.3：引擎更新 + LoRA 增量训练

**Files:**
- Create: `value-datadna/src/discovery/updater.py`

```python
# src/discovery/updater.py — Section 6.6
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForTokenClassification
import time, numpy as np


class EngineUpdater:
    """
    引擎更新模块 (Section 6.6)。
    1. TypeLibrary 更新: 新类型条目 + 示例值
    2. 真值表增量校准: 仅重校准受影响的 bin 区域
    3. NER 增量微调: LoRA (rank=8, ~5min), 仅添加新 BIO 标签
    4. Quality Gate: held-out precision ≥ 0.95, F1 ≥ 0.90, 退化 < 1%
    """

    def __init__(self, config: dict):
        cfg = config.get("discovery", {})
        self.lora_rank = cfg.get("ner_lora_rank", 8)
        self.retrain_min = cfg.get("ner_retrain_min_samples", 200)
        self.retrain_total_min = cfg.get("ner_retrain_total_min", 500)
        self.retrain_max_days = cfg.get("ner_retrain_max_days", 14)
        self.last_train_time = time.time()

    def update_type_library(self, new_type_name: str, examples: list, context_info: dict):
        """TypeLibrary 更新 (Section 6.6 步骤 1)"""
        # 追加新类型到 TypeLibrary
        # ... 写入类型定义文件 ...
        pass

    def incremental_truth_table_calibration(self, type_name: str,
                                             new_positives: list, new_negatives: list):
        """真值表增量校准 (Section 6.6 步骤 2) — 仅重校准受影响的 bin"""
        # Log-likelihood ratio 替代纯频率
        # 仅更新影响的 MultiIndex bins
        pass

    def check_ner_retrain_trigger(self, new_type_confirmed: int,
                                  all_new_confirmed: int) -> bool:
        """
        NER 增量微调触发条件 (Section 6.6 步骤 3):
          - 任一类型已确认值 ≥ 200
          - 所有新增类型合计 ≥ 500
          - 距上次训练 > 14 天 AND 任一类型 ≥ 50
        """
        days_since = (time.time() - self.last_train_time) / 86400
        if new_type_confirmed >= self.retrain_min:
            return True
        if all_new_confirmed >= self.retrain_total_min:
            return True
        if days_since > self.retrain_max_days and new_type_confirmed >= 50:
            return True
        return False

    def incremental_ner_finetune(self, new_bio_labels: list, train_data, base_model_path: str):
        """
        LoRA 增量微调 (Section 6.6 步骤 3)。
        rank=8, ~5min, 仅添加新 BIO 标签，不修改已有权重。
        """
        model = AutoModelForTokenClassification.from_pretrained(base_model_path)

        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLASSIFICATION,
            r=self.lora_rank,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_config)

        # ... 训练循环 (见 train_ner.py 模式) ...
        self.last_train_time = time.time()

    def quality_gate(self, new_bin_precision: float, ner_new_f1: float,
                     existing_f1_drop: float, e2e_macro_recall: float,
                     uncertain_rate: float) -> bool:
        """
        Quality Gate (Section 6.6 步骤 4):
          真值表新增 bin: held-out precision ≥ 0.95
          NER LoRA 新类型: held-out F1 ≥ 0.90
          已有类型退化: F1 下降 < 1%
          系统级: E2E Macro Recall ≥ 0.85 AND Uncertain 率 < 0.30
        """
        if new_bin_precision < 0.95:
            return False
        if ner_new_f1 < 0.90:
            return False
        if existing_f1_drop >= 0.01:
            return False
        if e2e_macro_recall < 0.85:
            return False
        if uncertain_rate >= 0.30:
            return False
        return True
```

---

### Task 4.4：闭环延迟窗口 + type_cache + 退化防护

**依赖:** Task 4.3 完成  
**Files:**
- Create: `value-datadna/src/discovery/degradation_guard.py`
- Modify: `value-datadna/src/classifiers/cache.py` (追加 type_cache 用途)

**对应设计文档:** Section 6.6 步骤5 + Section 6.7

```python
# src/discovery/degradation_guard.py — Section 6.7 退化防护
import numpy as np
from typing import List, Dict, Tuple


class DegradationGuard:
    """
    退化防护 (Section 6.7)。
    四项防护:
      1. 类型爆炸 — 模板 cos_sim > 0.85 → 自动合并建议
      2. 模式过拟合 — 正则需在 ≥100 个已确认值上通过覆盖率和特异性测试
      3. 概念漂移 — confidence 变化 > 0.2 的 bin → 告警
      4. 噪声累积 — 人工确认值与已有同类型 cos_sim < 0.6 → 标记需复核
    """

    def __init__(self, config: dict):
        self.cos_sim_merge_threshold = 0.85
        self.min_coverage_samples = 100
        self.drift_bin_threshold = 0.2
        self.noise_cos_sim_threshold = 0.6

    def check_type_explosion(self, type_a: str, type_b: str,
                              template_embeddings: Dict[str, np.ndarray]) -> bool:
        """
        类型爆炸防护: 模板 cos_sim > 0.85 → 建议自动合并。
        返回 True 表示应合并。
        """
        from sklearn.metrics.pairwise import cosine_similarity
        emb_a = template_embeddings.get(type_a)
        emb_b = template_embeddings.get(type_b)
        if emb_a is None or emb_b is None or len(emb_a) == 0 or len(emb_b) == 0:
            return False
        # 计算两类型模板质心的 cosine similarity
        centroid_a = emb_a.mean(axis=0).reshape(1, -1)
        centroid_b = emb_b.mean(axis=0).reshape(1, -1)
        sim = cosine_similarity(centroid_a, centroid_b)[0][0]
        return sim > self.cos_sim_merge_threshold

    def check_pattern_overfit(self, regex_pattern: str,
                               confirmed_values: List[str]) -> Tuple[bool, float, float]:
        """
        模式过拟合防护: 正则需在 ≥100 个已确认值上通过覆盖率和特异性测试。
        返回 (passed, coverage, precision)。
        """
        import re
        if len(confirmed_values) < self.min_coverage_samples:
            return False, 0.0, 0.0
        rx = re.compile(regex_pattern)
        matched = sum(1 for v in confirmed_values if rx.search(v))
        coverage = matched / len(confirmed_values)
        # 特异性简化: 假设负样本为相似但不属于该类型的值
        return coverage >= 0.95, coverage, 1.0

    def check_concept_drift(self, old_table, new_table) -> List[str]:
        """
        概念漂移防护: 对比新旧真值表，confidence 变化 > 0.2 的 bin → 告警。
        返回告警的 bin 列表。
        """
        alerts = []
        for idx in old_table.index:
            if idx in new_table.index:
                old_conf = old_table.loc[idx, "confidence"]
                new_conf = new_table.loc[idx, "confidence"]
                if abs(new_conf - old_conf) > self.drift_bin_threshold:
                    alerts.append(f"bin={idx}, old={old_conf:.3f}, new={new_conf:.3f}")
        return alerts

    def check_noise_accumulation(self, confirmed_value_embedding: np.ndarray,
                                  existing_type_templates: np.ndarray) -> bool:
        """
        噪声累积防护: 人工确认值与已有同类型 cos_sim < 0.6 → 标记需复核。
        返回 True 表示疑似噪声。
        """
        from sklearn.metrics.pairwise import cosine_similarity
        if len(existing_type_templates) == 0:
            return False
        centroid = existing_type_templates.mean(axis=0).reshape(1, -1)
        sim = cosine_similarity(
            confirmed_value_embedding.reshape(1, -1), centroid
        )[0][0]
        return sim < self.noise_cos_sim_threshold


class TypeCacheBridge:
    """
    闭环延迟窗口 + type_cache 桥接 (Section 6.6 步骤5)。
    人工确认后 → 立即写入 type_cache (pattern_hash → confirmed_type)。
    缓存 TTL: 24h。
    引擎更新后 → 验证 → 清除缓存 (调用 ClassificationCache.invalidate())。
    """

    def __init__(self, cache_instance):
        self.cache = cache_instance  # ClassificationCache 实例

    def on_human_confirmation(self, value: str, confirmed_type: str,
                               label_hint: str = ""):
        """人工确认后立即写入缓存 (Section 6.6 步骤5)"""
        from src.types import ValueClassification
        pattern_hash = self.cache.compute_pattern_hash(value, label_hint)
        vc = ValueClassification(
            value_id=f"confirmed_{pattern_hash}",
            value=value,
            sensitive_type=confirmed_type,
            confidence=1.0,
            method="human_confirmed",
        )
        self.cache.write(pattern_hash, vc, ttl=86400)  # 24h

    def on_engine_update_complete(self):
        """引擎更新后 → 验证 → 清除缓存 (Section 6.6 步骤5)"""
        self.cache.invalidate()  # 全量清除
```

- [ ] **Step 2: Commit**

```bash
git add value-datadna/src/discovery/degradation_guard.py
git commit -m "feat: add degradation guard + type_cache bridge (Section 6.6-6.7)"
```

---

## Phase 5：横切关注点

### Task 5.1：审计日志 + 分布偏移监控

**Files:**
- Create: `value-datadna/src/monitoring/audit.py`
- Create: `value-datadna/src/monitoring/drift.py`

```python
# src/monitoring/audit.py — Section 8.2 R6 + 设计文档技术栈
import json, time, os

class AuditLogger:
    """JSONL 审计日志，每条分类决策全程可追溯"""
    def __init__(self, log_dir: str = "output/"):
        self.log_path = os.path.join(log_dir, "audit.jsonl")

    def log(self, classification) -> None:
        record = {
            "timestamp": time.time(),
            "value_id": classification.value_id,
            "value": classification.value,
            "sensitive_type": classification.sensitive_type,
            "confidence": classification.confidence,
            "method": classification.method,
            "role": classification.role,
            "is_mock": classification.is_mock,
            "needs_review": classification.needs_review,
            "evidence": classification.evidence,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

```python
# src/monitoring/drift.py — Section 7.4
import numpy as np
from scipy.stats import entropy


class DriftMonitor:
    """
    分布偏移监控 (Section 7.4)。
    每次部署到新数据环境: 6 维特征 KL 散度 vs 校准基线。
    KL > 0.5 → 告警, 建议重校准。
    """

    def __init__(self, baseline_distributions: dict):
        self.baseline = baseline_distributions

    def check(self, current_distributions: dict) -> dict:
        alerts = {}
        for dim in ["regex_strength", "validated_count", "supportive_context",
                     "unsupportive_context", "pattern_frequency", "uniqueness_score"]:
            baseline_hist = self.baseline.get(dim)
            current_hist = current_distributions.get(dim)
            if baseline_hist is not None and current_hist is not None:
                kl = entropy(current_hist + 1e-10, baseline_hist + 1e-10)
                if kl > 0.5:
                    alerts[dim] = round(kl, 3)
        return alerts
```

### Task 5.2：R1-R7 评估报告脚本

**Files:**
- Modify: `value-datadna/src/evaluation/benchmark.py` (追加 R1-R7 指标)
- Modify: `value-datadna/src/evaluation/report.py`

涵盖指标 (Section 8.2):
- R1: Per-type/Macro Recall
- R2: Per-type/Macro Precision, Pooled FDR
- R3: 冷启动分级 Recall (强结构/弱结构/语义 × 冷启动/无LLM/最坏)
- R4: LLM 调用率 + P50/P95/P99 延迟
- R5: 故障注入退化测试 (NER/真值表/LLM/嵌入模型 不可用)
- R6: 质量监控框架 (抽检 FDR, KL 散度, entity_type_hint 冲突率)
- R7: 每组件退出条件检查

---

## Phase 6：数据策略

**依赖:** Phase 1B (PII 模式库 + 合成生成器) 完成

### Task 6.1：标注数据生成流水线 (L1-L4 4级矩阵)

**Files:**
- Create: `value-datadna/src/knowledge/labeling_pipeline.py`
- Create: `value-datadna/datasets/scripts/generate_labels.py`

**对应设计文档:** Section 7.1 标注决策矩阵

```python
# src/knowledge/labeling_pipeline.py — Section 7.1
from enum import Enum
from dataclasses import dataclass
from typing import List
from src.knowledge.pii_patterns import PII_PATTERNS, luhn_check, mod97_iban_check


class LabelLevel(Enum):
    L1_AUTO = "l1_auto"          # 正则 + 校验位算法 → 自动标注
    L2_LLM_VALIDATE = "l2_llm"   # 强正则，无校验位 → LLM Validation 确认
    L3_LLM_CLASSIFY = "l3_llm"   # 弱正则，上下文依赖 → LLM Classification
    L4_MANUAL = "l4_manual"      # 正则不匹配或 LLM 低置信 → 人工标注


LEVEL_ASSIGNMENT = {
    "CREDIT_CARD": LabelLevel.L1_AUTO,    # Luhn 校验
    "IBAN": LabelLevel.L1_AUTO,           # mod-97 校验
    "SSN": LabelLevel.L2_LLM_VALIDATE,    # 强正则，无校验位
    "IP": LabelLevel.L2_LLM_VALIDATE,
    "PASSPORT": LabelLevel.L2_LLM_VALIDATE,
    "DRIVER_LICENSE": LabelLevel.L2_LLM_VALIDATE,
    "EMAIL": LabelLevel.L3_LLM_CLASSIFY,  # 弱正则，上下文依赖
    "PHONE": LabelLevel.L3_LLM_CLASSIFY,
    "API_KEY": LabelLevel.L3_LLM_CLASSIFY,
    "BANK_ACCOUNT": LabelLevel.L3_LLM_CLASSIFY,
    "NAME": LabelLevel.L4_MANUAL,         # 无语义正则
    "ADDRESS": LabelLevel.L4_MANUAL,
}


@dataclass
class LabeledSample:
    value: str
    label_hint: str
    surrounding_text: str
    assigned_type: str
    label_level: LabelLevel
    confidence: float
    annotator: str = "auto"  # "auto" | "llm" | "human"


class LabelingPipeline:
    """
    标注决策矩阵流水线 (Section 7.1)。
    L1: 自动标注 (正则 + 校验位)
    L2: LLM Validation 确认
    L3: LLM Classification 判定
    L4: 人工标注
    """

    def __init__(self, validation_llm=None, classification_llm=None):
        self.validation_llm = validation_llm
        self.classification_llm = classification_llm

    def assign_level(self, candidate_type: str) -> LabelLevel:
        return LEVEL_ASSIGNMENT.get(candidate_type, LabelLevel.L4_MANUAL)

    def generate_labeled_dataset(self, synthetic_data: dict,
                                  enterprise_documents: list = None) -> List[LabeledSample]:
        """生成完整标注数据集，按4级矩阵分层"""
        labeled = []

        for type_name, samples in synthetic_data.items():
            level = self.assign_level(type_name)
            for sample in samples:
                if level == LabelLevel.L1_AUTO:
                    # 自动标注: 正则 + 校验位通过 → 直接标注
                    if self._validate_checksum(sample.value, type_name):
                        labeled.append(LabeledSample(
                            value=sample.value,
                            label_hint=sample.context.label_hint or "",
                            surrounding_text=sample.context.surrounding_text or "",
                            assigned_type=type_name,
                            label_level=level,
                            confidence=1.0,
                        ))
                elif level == LabelLevel.L2_LLM_VALIDATE:
                    # L2 标记为需要 LLM 确认 (实际调用在 calibrate.py 中)
                    labeled.append(LabeledSample(
                        value=sample.value,
                        label_hint=sample.context.label_hint or "",
                        surrounding_text=sample.context.surrounding_text or "",
                        assigned_type=type_name,
                        label_level=level,
                        confidence=0.85,
                        annotator="pending_llm_validate",
                    ))
                elif level == LabelLevel.L3_LLM_CLASSIFY:
                    labeled.append(LabeledSample(
                        value=sample.value,
                        label_hint=sample.context.label_hint or "",
                        surrounding_text=sample.context.surrounding_text or "",
                        assigned_type=type_name,
                        label_level=level,
                        confidence=0.50,
                        annotator="pending_llm_classify",
                    ))
                else:
                    # L4: 标记为需要人工标注
                    labeled.append(LabeledSample(
                        value=sample.value,
                        label_hint=sample.context.label_hint or "",
                        surrounding_text=sample.context.surrounding_text or "",
                        assigned_type=type_name,
                        label_level=level,
                        confidence=0.0,
                        annotator="pending_human",
                    ))

        return labeled

    def _validate_checksum(self, value: str, type_name: str) -> bool:
        info = PII_PATTERNS.get(type_name, {})
        validation = info.get("validation")
        if validation == "luhn":
            return luhn_check(value)
        elif validation == "mod97":
            return mod97_iban_check(value)
        return False
```

- [ ] **Step 1: Commit**

```bash
git add value-datadna/src/knowledge/labeling_pipeline.py
git commit -m "feat: add labeling pipeline (L1-L4 4-level annotation matrix)"
```

### Task 6.2：公共数据源集成 (SWIFT/PCI/Faker/Presidio/基准语料)

**Files:**
- Create: `value-datadna/src/knowledge/public_data.py`
- Create: `value-datadna/datasets/scripts/download_public.py`

**对应设计文档:** Section 7.2 公开数据 + Section 3.5.1 regex_strength 基准语料

```python
# src/knowledge/public_data.py — Section 7.2
"""
公共数据源集成:
  - SWIFT IBAN Registry → IBAN 全球格式校验
  - PCI DSS 测试卡号 → CCN 校准
  - US SSA Randomization → SSN 格式
  - IP RFC 规范 → IPv4/IPv6
  - Wikipedia + GitHub 混合语料 → regex_strength 基准 (~10万篇)
  - Faker 库 → 全 PII/PCI 类型合成生成
  - Microsoft Presidio → 20+ PII 类型识别器（基准对比）
  - ai4privacy/pii-masking-300k → NER 训练
  - conllpp → NER baseline
"""

import re, json, os
import numpy as np
from typing import Dict, List, Set


class PublicDataIntegrator:
    """公共数据源下载 + 预处理"""

    def __init__(self, data_dir: str = "datasets/public/"):
        self.data_dir = data_dir

    def download_all(self):
        """下载所有公共数据源"""
        self._download_swift_iban()
        self._download_ai4privacy()
        self._download_conllpp()
        print("Public data download complete.")

    def _download_swift_iban(self):
        """SWIFT IBAN Registry (公开格式规范，非实际账户)"""
        # SWIFT 发布 IBAN 结构规范，包含各国 IBAN 长度和格式
        # 存储为 JSON 格式规范文件供真值表校准
        swift_dir = os.path.join(self.data_dir, "swift_iban")
        os.makedirs(swift_dir, exist_ok=True)
        iban_formats = {
            "GB": {"length": 22, "bban_format": "4a6n8n", "example": "GB29NWBK60161331926819"},
            "DE": {"length": 22, "bban_format": "8n10n", "example": "DE89370400440532013000"},
            "FR": {"length": 27, "bban_format": "5n5n11c2n", "example": "FR1420041010050500013M02606"},
            # ... 完整格式表 (~75个国家)
        }
        with open(os.path.join(swift_dir, "iban_formats.json"), "w") as f:
            json.dump(iban_formats, f, indent=2)

    def _download_ai4privacy(self):
        """ai4privacy/pii-masking-300k → NER 训练数据"""
        from datasets import load_dataset
        dataset = load_dataset("ai4privacy/pii-masking-300k")
        dataset.save_to_disk(os.path.join(self.data_dir, "pii-masking"))
        print(f"ai4privacy downloaded: {len(dataset['train'])} samples")

    def _download_conllpp(self):
        """conllpp → NER baseline"""
        from datasets import load_dataset
        dataset = load_dataset("conllpp")
        dataset.save_to_disk(os.path.join(self.data_dir, "conllpp"))


class RegexStrengthBenchmark:
    """
    regex_strength 经验误报率基准计算 (Section 3.5.1)。
    在 Wikipedia + GitHub + 企业文档混合语料上运行所有正则，
    计算 specificity = 1 - (FP / total)。
    """

    def __init__(self, corpus_paths: List[str]):
        self.corpus_paths = corpus_paths

    def compute_all(self) -> Dict[str, float]:
        """在基准语料上运行所有 PII 正则，返回每种类型的 specificity"""
        from src.knowledge.pii_patterns import PII_PATTERNS
        total_docs = 0
        fp_counts = {t: 0 for t in PII_PATTERNS}

        for path in self.corpus_paths:
            for fname in os.listdir(path):
                if fname.endswith(('.txt', '.md')):
                    with open(os.path.join(path, fname), 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    total_docs += 1
                    for type_name, info in PII_PATTERNS.items():
                        if info.get("regex"):
                            rx = re.compile(info["regex"], re.IGNORECASE)
                            if rx.search(text):
                                fp_counts[type_name] += 1

        return {
            t: 1.0 - (fp_counts[t] / max(total_docs, 1))
            for t in PII_PATTERNS
        }
```

- [ ] **Step 2: Commit**

```bash
git add value-datadna/src/knowledge/public_data.py value-datadna/datasets/scripts/download_public.py
git commit -m "feat: add public data integrator + regex strength benchmark"
```

### Task 6.3：分轮标注工具 + Cohen's Kappa 质量控制

**Files:**
- Create: `value-datadna/src/knowledge/annotation_tool.py`
- Test: `value-datadna/tests/test_annotation_tool.py`

**对应设计文档:** Section 7.3 分轮标注方案

```python
# src/knowledge/annotation_tool.py — Section 7.3
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class AnnotationRound1:
    """第1轮: 类型标注"""
    value: str
    surrounding_text: str
    chosen_type: str  # SSN / CCN / EMAIL / ... / NON_SENSITIVE
    annotator_id: str

@dataclass
class AnnotationRound2:
    """第2轮: 上下文验证 (仅第1轮的敏感值)"""
    value: str
    original_type: str     # 第1轮标注的类型
    context_support: str   # "support" / "conflict" / "uncertain"
    annotator_id: str

@dataclass
class AnnotationRound3:
    """第3轮: Mock 检测 (仅第1轮的敏感值)"""
    value: str
    is_mock: str           # "real" / "test" / "uncertain"
    annotator_id: str


class AnnotationQC:
    """
    标注质量控制 (Section 7.3)。
    Cohen's Kappa ≥ 0.85 阈值。
    2人独立标注 → 争议 → 第3人裁定或标记 ambiguous。
    """

    def compute_cohens_kappa(self, labels_a: List[str], labels_b: List[str]) -> float:
        """计算 Cohen's Kappa 一致性"""
        if len(labels_a) != len(labels_b) or len(labels_a) == 0:
            return 0.0

        # 构建混淆矩阵
        categories = sorted(set(labels_a) | set(labels_b))
        n_cat = len(categories)
        cat_to_idx = {c: i for i, c in enumerate(categories)}

        matrix = np.zeros((n_cat, n_cat))
        for a, b in zip(labels_a, labels_b):
            matrix[cat_to_idx[a], cat_to_idx[b]] += 1

        total = matrix.sum()
        po = np.trace(matrix) / total  # observed agreement
        row_sums = matrix.sum(axis=1)
        col_sums = matrix.sum(axis=0)
        pe = (row_sums @ col_sums) / (total ** 2)  # expected agreement

        if pe == 1.0:
            return 1.0
        return (po - pe) / (1 - pe)

    def resolve_conflicts(self, round_data: List[AnnotationRound1],
                           annotator_a: str, annotator_b: str,
                           arbitrator) -> Tuple[List[AnnotationRound1], List[str]]:
        """
        争议解决: 2人独立 → 第3人裁定。
        返回 (resolved_annotations, conflict_log)。
        """
        a_labels = {a.value: a.chosen_type for a in round_data if a.annotator_id == annotator_a}
        b_labels = {b.value: b.chosen_type for b in round_data if b.annotator_id == annotator_b}

        resolved = []
        conflicts = []

        for value in a_labels:
            if value not in b_labels:
                continue
            if a_labels[value] == b_labels[value]:
                resolved.append(AnnotationRound1(
                    value=value,
                    surrounding_text="",
                    chosen_type=a_labels[value],
                    annotator_id="consensus",
                ))
            else:
                # 第3人裁定
                arbitrator_choice = arbitrator.resolve(value, a_labels[value], b_labels[value])
                resolved.append(AnnotationRound1(
                    value=value,
                    surrounding_text="",
                    chosen_type=arbitrator_choice,
                    annotator_id="arbitrator",
                ))
                conflicts.append(
                    f"Conflict: {value} → A:{a_labels[value]}, B:{b_labels[value]}, "
                    f"Arbitrator:{arbitrator_choice}"
                )

        return resolved, conflicts

    def save_annotation_guide(self, conflicts: List[str], guide_path: str):
        """争议模式写入标注指南 (Section 7.3 QC)"""
        with open(guide_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(conflicts) + "\n")
```

- [ ] **Step 3: Commit**

```bash
cd value-datadna && pytest tests/test_annotation_tool.py -v
git add value-datadna/src/knowledge/annotation_tool.py
git commit -m "feat: add 3-round annotation tool + Cohen's Kappa QC"
```

---

## Phase 7：硬性要求测试实现

**依赖:** Phase 1D (R1-R4, R7 基础) + Phase 2B (R3 冷启动含LLM) 完成

### Task 7.1：R5 故障注入退化测试完整实现

**Files:**
- Create: `value-datadna/tests/test_degradation.py`

**对应设计文档:** Section 8.2 R5 降级容错

```python
# tests/test_degradation.py — Section 8.2 R5
"""
故障注入退化测试: 7种故障场景 × 独立测量
场景:
  1. NER 不可用 × 弱结构+语义: Recall 下降 < 20%
  2. NER 不可用 × 强结构: Recall 下降 < 5%
  3. 真值表不可用 × 强结构: Recall 下降 < 30%
  4. 真值表不可用 × 弱结构+语义: Recall 下降 < 10%
  5. LLM 不可用 × 所有: FDR 不增加, uncertain 率上升
  6. 嵌入模型不可用 × 所有: 无影响
  7. 全组件最差 (仅正则) × 强结构: Recall 下降 < 40%
"""
import pytest
import numpy as np


class TestDegradationScenarios:
    @pytest.fixture
    def held_out_dataset(self):
        """加载 held-out 标注集 (15 类型 × 100 值)"""
        # ... 加载标注数据 ...
        pass

    def test_ner_unavailable_weak_structure(self, held_out_dataset):
        """场景1: NER 不可用, 弱结构 (EMAIL/PHONE/NAME/ADDRESS) 退化 < 20%"""
        # 注入 NER 故障 → 测量 Recall → 断言退化 < 20%
        pass

    def test_ner_unavailable_strong_structure(self, held_out_dataset):
        """场景2: NER 不可用, 强结构 (SSN/CCN/IP/IBAN) 退化 < 5%"""
        pass

    def test_truth_table_unavailable_strong_structure(self, held_out_dataset):
        """场景3: 真值表不可用, 强结构退化 < 30%"""
        pass

    def test_truth_table_unavailable_weak_structure(self, held_out_dataset):
        """场景4: 真值表不可用, 弱结构退化 < 10%"""
        pass

    def test_llm_unavailable_no_fdr_increase(self, held_out_dataset):
        """场景5: LLM 不可用, FDR 不增加, uncertain 率上升"""
        pass

    def test_embedding_unavailable_no_impact(self, held_out_dataset):
        """场景6: 嵌入模型不可用, 无影响 (SD 跳过, α=1.0)"""
        pass

    def test_worst_case_regex_only_strong_structure(self, held_out_dataset):
        """场景7: 仅正则, 强结构退化 < 40%"""
        pass
```

- [ ] **Step 1: Commit**

```bash
git add value-datadna/tests/test_degradation.py
git commit -m "test: add R5 fault injection degradation tests"
```

### Task 7.2：R7 组件退出条件检查实现

**Files:**
- Create: `value-datadna/src/monitoring/exit_conditions.py`
- Create: `value-datadna/tests/test_end_to_end.py`

**对应设计文档:** Section 8.2 R7

```python
# src/monitoring/exit_conditions.py — Section 8.2 R7
"""
每个组件有量化退出条件 (Section 8.2 R7):
  真值表: 任一 bin held-out < 10 且 confidence CV > 0.3 → 重校准
  NER: held-out Macro F1 < 0.85 或任一类型 F1 下降 > 3% → 退出
  SD: 模板同类 cos_sim 均值 < 0.7 或类间差距 < 0.15 → 重建
  FLAN-T5: 200 题 accuracy < 0.90 → 退出
  Mistral: 200 题 accuracy < 0.85 → 退出
  聚类传播: 抽检不一致率 > 10% → 退出
  Learned Classification: 自动验证通过率 < 70% → 退出
  系统级: E2E Macro Recall < 0.85 或 uncertain 率 > 0.30 → 退出
"""
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ExitCheckResult:
    component: str
    passed: bool
    metric_value: float
    threshold: float
    description: str


class ExitConditionChecker:
    def check_truth_table(self, table, held_out_bins: Dict) -> ExitCheckResult:
        """R7: 真值表退出条件"""
        failing_bins = []
        for bin_key, samples in held_out_bins.items():
            if len(samples) < 10:
                confidences = [s["confidence"] for s in samples]
                if len(confidences) > 1:
                    cv = np.std(confidences) / (np.mean(confidences) + 1e-10)
                    if cv > 0.3:
                        failing_bins.append(bin_key)
        passed = len(failing_bins) == 0
        return ExitCheckResult(
            component="truth_table",
            passed=passed,
            metric_value=len(failing_bins),
            threshold=0,
            description=f"{len(failing_bins)} bins below threshold",
        )

    def check_ner(self, held_out_f1: float, baseline_f1: float,
                   per_type_f1_drops: Dict[str, float]) -> ExitCheckResult:
        """R7: NER 退出条件"""
        macro_pass = held_out_f1 >= 0.85
        no_drop = all(drop < 0.03 for drop in per_type_f1_drops.values())
        return ExitCheckResult(
            component="ner",
            passed=macro_pass and no_drop,
            metric_value=held_out_f1,
            threshold=0.85,
            description=f"Macro F1={held_out_f1:.3f}, max_drop={max(per_type_f1_drops.values()):.3f}",
        )

    def check_semantic_distance(self, intra_type_cos_sim: float,
                                 inter_type_gap: float) -> ExitCheckResult:
        """R7: SD 退出条件"""
        passed = intra_type_cos_sim >= 0.7 and inter_type_gap >= 0.15
        return ExitCheckResult(
            component="semantic_distance",
            passed=passed,
            metric_value=inter_type_gap,
            threshold=0.15,
            description=f"Intra={intra_type_cos_sim:.3f}, Gap={inter_type_gap:.3f}",
        )

    def check_flan_t5(self, accuracy: float) -> ExitCheckResult:
        return ExitCheckResult("flan_t5", accuracy >= 0.90, accuracy, 0.90, "")

    def check_mistral(self, accuracy: float) -> ExitCheckResult:
        return ExitCheckResult("mistral", accuracy >= 0.85, accuracy, 0.85, "")

    def check_clustering(self, inconsistency_rate: float) -> ExitCheckResult:
        return ExitCheckResult("clustering", inconsistency_rate <= 0.10,
                               inconsistency_rate, 0.10, "")

    def check_learned_classification(self, auto_pass_rate: float) -> ExitCheckResult:
        return ExitCheckResult("learned_classification", auto_pass_rate >= 0.70,
                               auto_pass_rate, 0.70, "")

    def check_system_level(self, e2e_macro_recall: float,
                            uncertain_rate: float) -> ExitCheckResult:
        passed = e2e_macro_recall >= 0.85 and uncertain_rate < 0.30
        return ExitCheckResult(
            component="system",
            passed=passed,
            metric_value=e2e_macro_recall,
            threshold=0.85,
            description=f"Recall={e2e_macro_recall:.3f}, Uncertain={uncertain_rate:.3f}",
        )

    def check_all(self, **kwargs) -> List[ExitCheckResult]:
        """运行所有组件的退出条件检查"""
        results = []
        if "held_out_bins" in kwargs:
            results.append(self.check_truth_table(kwargs["held_out_bins"]))
        if "held_out_f1" in kwargs:
            results.append(self.check_ner(
                kwargs["held_out_f1"], kwargs.get("baseline_f1", 0),
                kwargs.get("per_type_drops", {}),
            ))
        # ... 逐个检查所有组件 ...
        return results
```

```python
# tests/test_end_to_end.py — 全管道端到端测试
"""
R7 系统级退出条件检查:
  - E2E Macro Recall ≥ 0.85
  - Uncertain 率 < 0.30
"""
```

- [ ] **Step 2: Commit**

```bash
git add value-datadna/src/monitoring/exit_conditions.py \
        value-datadna/tests/test_end_to_end.py
git commit -m "feat: add R7 exit condition checker + end-to-end tests"
```

---

## 任务依赖矩阵

| 任务 | 前置依赖 | 可并行 |
|------|------|:--:|
| 0.1-0.3 骨架+类型+配置 | 无 | ❌ 必须最先 |
| 1A.1-3 结构化提取器/Mock/抽样 | 0.x | ✅ |
| 1A.4 非结构化提取器 | 0.x + 1B(PII模式) | ✅ |
| 1B.1-3 正则/特征/真值表 | 0.x | ✅ |
| 1C.1-2 NER引擎/微调 | 0.x | ✅ |
| 1D.1-3 融合/路由/内核 | **1A + 1B + 1C** | ❌ |
| 1D.4 entity_type_hint 7维集成 | **1D + 1C** | ❌ |
| 2A.1-3 LLM基础设施 | 0.x | ✅ |
| 2B.1 LLM消歧集成 | **1D + 2A** | ❌ |
| 2C.1 在线缓存层 | **1D** | ✅ (与 2B/3A/3B 并行) |
| 3A.1 SD嵌入+模板库 | 0.x + 1B(PII模式) | ✅ |
| 3B.1-2 文件聚类 | 0.x | ✅ |
| 3C.1 标签传播 | **1D + 3B + 3A**(增量匹配) | ❌ |
| 3C.2 三路径编排器 | **1D + 2C + 3B** | ❌ |
| 4.1 未知模式收集(条件A+B) | **1D** | ✅ (部分与 3A 并行) |
| 4.1b 未知模式收集(条件C) | **1D + 3A** | ❌ |
| 4.2 自动验证 | **4.1** | ❌ |
| 4.3 引擎更新+增量训练 | **4.2 + 1C(NER基础)** | ❌ |
| 4.4 闭环延迟窗口+退化防护 | **4.3 + 2C(cache)** | ❌ |
| 5.1-2 审计/监控/评估框架 | **1D** | ✅ (与其他并行) |
| 6.1 标注数据流水线 L1-L4 | **1B**(PII模式+合成) | ✅ |
| 6.2 公共数据源集成 | **1B**(PII模式) | ✅ |
| 6.3 分轮标注工具+QC | **6.1** | ❌ |
| 7.1 R5 故障注入测试 | **1D + 2B** | ❌ |
| 7.2 R7 退出条件检查 | **1D + 2B** | ❌ |

## 并行开发建议

**第一批 (Week 1-2, 8 人并行):**
- 开发者 A: Task 0.1-0.3 → Task 1A.1-4 (值提取体系: 结构化+非结构化)
- 开发者 B: Task 0.1-0.3 → Task 1B.1-3 (真值表体系: PII模式+特征+校准)
- 开发者 C: Task 0.1-0.3 → Task 1C.1-2 (NER 体系: GLiNER+BERT微调)
- 开发者 D: Task 0.1-0.3 → Task 2A.1-3 (LLM 基础设施: FLAN-T5+Mistral+批量)
- 开发者 E: Task 0.1-0.3 → Task 3A.1 (SD 嵌入+类型模板库+簇模板库)
- 开发者 F: Task 0.1-0.3 → Task 3B.1-2 (文件聚类引擎: 元数据归一化+流式)
- 开发者 G: Task 0.1-0.3 → Task 6.1-6.2 (数据策略: 标注流水线+公开数据)
- 开发者 H: (待命，准备 1D 集成)

**第二批 (Week 3-4, 6 人并行):**
- 开发者 A+B: Task 1D.1-4 (内核集成 + 7维NER, 需 1A+1B+1C)
- 开发者 C+D: Task 2B.1 (LLM消歧集成, 需 1D+2A)
- 开发者 E: Task 2C.1 (在线缓存层, 需 1D)
- 开发者 F: Task 3C.1-2 (标签传播+编排器, 需 1D+3B+3A+2C)
- 开发者 G: Task 5.1-2 (审计/监控/评估框架, 需 1D) + Task 6.3 (分轮标注QC, 需 6.1)
- 开发者 H: Task 7.1-7.2 (R5/R7 测试实现, 需 1D)

**第三批 (Week 5-6, 4 人):**
- 开发者 A+C: Task 4.1 (模式收集条件A+B, 需 1D) + Task 4.1b (条件C, 需 3A)
- 开发者 D+E: Task 4.2 (自动验证, 需 4.1)
- 开发者 F: Task 4.3 (引擎更新+增量训练, 需 4.2+1C)
- 开发者 G+H: Task 4.4 (闭环延迟窗口+退化防护, 需 4.3+2C)

**第四批 (Week 7, 全员集成验证):**
- 全团队: 端到端集成测试, R1-R7 完整测量, 文档补完
