# DataDNA 值级敏感数据分类引擎 — 统一设计文档

> **版本**: v1.0  
> **日期**: 2026-05-24  
> **定位**: 对标 Cyera DataDNA 的完整值级敏感数据分类引擎  
> **架构**: 真值表评分 + NER语义特征 + Semantic Distancing并列评分 + LLM分层消歧 + 文件级聚类传播 + 自学习闭环  
> **语言**: 英文优先，中文及多语言次之  
> **状态**: 设计完成，待实施

---

## 一、设计原则

吸取文档级 DataDNA (`2026-05-24-datadna-unified-design.md`) 的经验教训，值级分类引擎采用 9 条原则：

| # | 原则 | 含义 |
|:--:|------|------|
| P1 | **宽进严出** | 入口层最大化检出，出口层高置信才自动判定。不确定的逐级消歧（真值表/NER → LLM Validation → LLM Classification → 人工），不静默丢弃 |
| P2 | **带着知识上线** | 真值表从公开数据预校准，NER 模型预训练，首日即可检测。预校准知识迁移到企业环境存在分布偏移，需持续校准 |
| P3 | **上下文内建于分类** | 真值表的 unsupportive_context 维度、虚假数据过滤器是分类决策的组成部分，不是后处理。上下文判断从第一天就参与置信度计算 |
| P4 | **不确定时保守** | 置信度 < 阈值 → 升级到下一层，最低层走人工。阈值从标注数据对抗性验证校准 |
| P5 | **分层消歧，非分层备份** | 真值表+NER 做初判，LLM Validation 消歧中等难度，LLM Classification 处理高难度。每层不可用只损失该难度区间，清晰样本不受影响 |
| P6 | **结构走真值表，语义信号靠NER增强** | NER 为真值表提供语义特征输入（entity_type_hint），不独立产出分类决策。分界标准是可检测的结构模式（正则、校验位、固定格式），非值是否含字母 |
| P7 | **可证伪可审计** | 每个引擎有量化退出条件，每个分类决策全程可追溯 |
| P8 | **内核接口批量优先** | `classify(values: List[Value]) -> List[Result]`，单值是特例。避免后期集成聚类传播时改接口 |
| P9 | **规模靠减量不靠加速** | PB 级通过文件聚类降维（仅代表值进入分类内核），内核效率指标不与未实现的聚类层耦合 |

---

## 二、架构总览

### 2.1 最终架构（4 个 Phase 完成态）

```
数据源 (Database/CSV/JSON/PDF/Text/Code)
  │
  ▼
[Phase 3] 文件级聚类 ── 元数据替换 → 结构指纹 → 文件簇 → 抽样代表文件
  │
  ▼
值提取器 ── 结构化提取器 (CSV/DB/JSON/XML) + 非结构化提取器 (PDF/Text/Code/Email)
  │
  ▼
[前置] Mock 快速过滤 ── 已知虚拟值、全列重复、否定上下文 → MOCK_DATA, 跳过分类
  │
  ▼
结构化抽样 ── 每列/字段抽代表值（去重+分层抽样）
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 1: 核心分类内核                                     │
│                                                          │
│  候选值 → 6维特征计算 → 真值表引擎 → truth_table_confidence │
│         → Semantic Distancing [Phase 3] → distance_score  │
│                                                          │
│  融合: weighted_confidence = α×truth_table + (1-α)×distance│
│                                                          │
│  NER引擎提供语义特征（作为真值表辅助特征，非独立路径）        │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
角色判定 ── subject / identifier / reference (自动规则)
  │
  ▼
[后置] 上下文一致性检查 ── 值类型与列名/路径是否匹配
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 2: LLM 消歧层                                      │
│                                                          │
│  conf ≥ 0.85  → 直接输出                                  │
│  0.50 ≤ c < 0.85 → FLAN-T5 Validation (判断题)            │
│  conf < 0.50 → Mistral-7B Classification (论述题)          │
│  conf < 0.30 → uncertain, needs_review=True               │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
ValueClassification 输出 (sensitive_type, confidence, evidence, audit trail)
  │
  ▼
[Phase 4] Learned Classification 闭环 ── 发现 → 验证 → 注册 → 训练 → 部署
```

### 2.2 分阶段实施路线

| Phase | 内容 | 可验证目标 |
|------|------|------|
| **Phase 1** | 核心分类内核：6维真值表 + NER引擎 + 融合 + Mock过滤 + 角色判定 + 值提取器 | 15 PII/PCI 类型在公开合成数据上 Macro Recall ≥ 0.85 |
| **Phase 2** | LLM 消歧层：FLAN-T5 Validation + Mistral Classification + 批量优化 | 中低置信值经 LLM 消歧后 Macro Recall ≥ 0.90 |
| **Phase 3** | Semantic Distancing + 文件级聚类传播 | 有效 LLM 调用率对比 Phase 2；10K 文件端到端吞吐 |
| **Phase 4** | Learned Classification 闭环 | 模拟 3 种"新类型"的发现→注册→训练→部署全周期 |

### 2.3 与文档级 DataDNA 的关键差异

| 维度 | 文档级 DataDNA | 值级 DataDNA |
|------|------|------|
| 分类对象 | 整篇文档 → 文档主题 | 单个数据值 → 敏感数据类型 |
| 分类粒度 | 一篇一个标签 | 一文档数十到数千个值级决策 |
| 核心引擎 | 6引擎并行投票融合 | 真值表主路径 + NER语义特征 + Semantic Distancing并列评分 |
| LLM 角色 | 6个投票者之一 (E6, weight=2.0) | 独立消歧层，不参与初判 |
| 效率策略 | 3级门控 (fast/validate/full) | 聚类降维 (文件级) + 2级LLM消歧 |
| 规模机制 | 缺聚类层 (R4始终不达标) | 文件级聚类从Phase 3设计之初就纳入 |
| 自学习 | 文档级类型发现 (~150行未实现) | 值模式发现 + 统计验证 + 人工Gate + 增量训练 |

---

## 三、Phase 1：核心分类内核

### 3.1 值表示模型

```python
@dataclass
class DataValue:
    value_id: str
    value: str                    # 原始值 "4111-1111-1111-1111"
    context: ValueContext
    source: ValueSource

@dataclass
class ValueContext:
    container_type: str           # "db_cell" | "csv_field" | "json_path" | "xml_xpath" |
                                  # "text_span" | "code_line" | "pdf_field" | "email_body"
    container_path: str           # 泛化位置路径
    label_hint: str | None        # 该位置的标签/键名
    surrounding_text: str         # 上下文窗口 (±100字符)
    parent_doc_id: str | None
    parent_file_path: str | None
    metadata: dict                # 容器特定元数据

@dataclass
class ValueSource:
    source_type: str              # 提取来源类型
    extraction_method: str        # 提取方法
    position: str | None          # 源内位置
```

### 3.2 值提取器

两个共接口的提取器插件：

**结构化提取器**：CSV字段、数据库列、JSON路径、XML XPath → 批量提取值，列名/路径作为 label_hint。

**非结构化提取器**：PDF文本、邮件正文、代码行、Slack消息 → PII正则扫描 + 字符偏移 + 周围文本窗口。

### 3.3 Mock 快速过滤（前置）

在分类之前过滤已知非敏感/测试数据，避免浪费算力：

```
硬规则:
  - 已知虚拟模式: "000-00-0000", "123-45-6789", "XXX-XX-XXXX", "4111-1111-1111-1111"
  - 全列/全字段所有值相同 → 测试数据
  - 上下文否定词: test, sample, example, placeholder, redacted, dummy, mock, fake, todo, fixme

命中 → sensitive_type=MOCK_DATA, confidence=1.0, 不进入下游
```

### 3.4 结构化抽样引擎

对标 Cyera 专利 US12299167B2 的抽样组件——1000万行的数据库表不需要逐行分类每列。

```
抽样策略:
  1. 去重: 同一列的所有唯一值去重
  2. 分层: 按值的长度/字符集分布分层 (等频分箱, 5-10 层)
  3. 抽样: 每层随机取 min(ceil(20/n_layers), 层内总数) 个值
  4. 合计: 每列 5-20 个代表值
  5. 输出: 代表值列表 → 进入分类内核

排序偏差防护: 不取"前N个", 每层内随机采样
```

### 3.5 真值表引擎（路径A）

对标 Cyera 专利 US12299167B2 的核心组件。

#### 3.5.1 六维特征

| 维度 | 含义 | 计算方式 | 离散化 |
|------|------|------|------|
| `regex_strength` | 正则特异性 | 匹配模式的 specificity | [0, 0.25, 0.5, 0.75, 1.0] |
| `validated_count` | 同模式已确认实例数 | 该模式在数据集/历史中被确认为真实敏感数据的次数 | {0, 1-3, 4-9, 10-49, 50+} |
| `supportive_context` | 支持性上下文词命中 | 上下文匹配"SSN", "social security"等词数 | {0, 1, 2, 3+} |
| `unsupportive_context` | 否定性上下文词命中 | 上下文匹配"test", "sample", "placeholder"等词数 | {0, 1, 2+} |
| `pattern_frequency` | 模式在数据集中的频率 | 该模式在全量数据中出现次数的分位数 | [0-20%, 20-40%, 40-60%, 60-80%, 80-100%] |
| `uniqueness_score` | 值的唯一性 | 数据集中该值出现次数 | {1, 2-5, 6-20, 21-100, 100+} |
| `entity_type_hint` | NER 语义类型提示 (可选) | NER 对该值的实体判断 | {PERSON_NAME, ORGANIZATION, LOCATION, GENERIC_ENTITY, NONE} |

> `entity_type_hint` 仅当 NER 引擎可用时填充。对不含在敏感类型标注中的实体类型视为 NONE。此维度帮助真值表区分"看起来像名字但实际是 API Key" 等歧义案例。

#### 3.5.2 真值表结构

```
pandas MultiIndex DataFrame: 6维 (NER不可用时) 或 7维 (NER可用时) → confidence
总状态数: 6维 5×5×4×3×5×5 = 7500; 7维增加 ×6 ≈ 45000 组合
实际非零: 6维 ~2000-3000; 7维 ~5000-8000

查询: O(1) via MultiIndex.loc
插值: 缺失键 → KD-Tree 最近邻, O(log n)

多类型匹配路由:
  一个值可能被多种类型的正则同时匹配 (如 "US123456789" 同时匹配
  PASSPORT 和 GENERIC_ID 的 regex)。此时分别查询每个候选类型的真值表，
  取 highest confidence。若所有候选类型的 confidence < 0.3，
  进入 LLM Classification。
```

#### 3.5.3 校准流程

```
输入: 标注数据集 D = {(value, context, label=true/false)}

对每个值:
  1. 计算 6 维特征
  2. 按离散化级别分桶
  3. 每个桶的 confidence = 桶内正样本数 / 桶内总样本数

对稀疏桶: Laplace 平滑
对空桶: KD-Tree 插值
输出: calibration_table.parquet
```

### 3.6 NER 引擎（语义特征源）

NER 不为真值表提供独立的分类路径。其输出作为真值表的 `entity_type_hint` 维度（第 7 维），帮助区分语义歧义案例。

#### 3.6.1 特征计算

```
NER 输入: surrounding_text (值前后 ±100 字符上下文窗口)
NER 输出: BIO 标签序列

对目标值的具体位置:
  - 值被标注为 B-NAME / I-NAME             → entity_type_hint = PERSON_NAME
  - 值被标注为 B-ORG / I-ORG               → entity_type_hint = ORGANIZATION
  - 值被标注为 B-ADDRESS / I-ADDRESS       → entity_type_hint = LOCATION
  - 值被标注为其他实体类型                    → entity_type_hint = GENERIC_ENTITY
  - 值被标注为 O 或无实体覆盖                → entity_type_hint = NONE

对结构化值 (regex_strength > 0.7):
  NER 的 entity_type_hint 权重降低 — 强结构值主要由 regex_strength 主导

对弱结构/无语义值 (regex_strength < 0.3):
  entity_type_hint 成为区分 NAME vs API_KEY vs NON_SENSITIVE 的关键维度
```

#### 3.6.2 模型选型

| 阶段 | 模型 | 用途 |
|------|------|------|
| 冷启动 | GLiNER zero-shot | 零样本实体识别 |
| 有标注后 | BERT-base (英文) / RoBERTa | 微调 BIO 序列标注 |
| 蒸馏 | Qwen3:8b → BERT | 从 LLM 标注蒸馏 |

#### 3.6.3 BIO 标签体系

```
B-SSN, I-SSN, B-CCN, I-CCN, B-EMAIL, I-EMAIL, B-PHONE, I-PHONE,
B-IBAN, I-IBAN, B-PASSPORT, I-PASSPORT, B-DRIVER_LICENSE, I-DRIVER_LICENSE,
B-IP, I-IP, B-API_KEY, I-API_KEY, B-BANK_ACCOUNT, I-BANK_ACCOUNT,
B-NAME, I-NAME, B-ADDRESS, I-ADDRESS, B-ORG, I-ORG, O
```

### 3.7 融合评分

真值表和 Semantic Distancing (Phase 3) 作为并列评分者：

```
融合公式:
  1. 分数校准: truth_table_confidence 和 distance_score 分布不同
     (前者为频率估计, 后者为几何距离)。融合前各自通过 Platt scaling
     或 isotonic regression 映射到 [0,1] 校准概率空间。
  2. weighted_confidence = α × calibrated_truth_table + (1-α) × calibrated_distance

  α 默认值: 0.7 (真值表权重更高，因为 6 维更丰富)
  α 在 Semantic Distancing 未启用时: 1.0 (完全依赖真值表)
  α 在 Phase 3 完成后, 于 held-out 标注集上通过 grid search 校准 (最大化 Macro F1)

NER 语义特征:
  NER 输出作为真值表的 `entity_type_hint` 维度 (第 7 维)，不提供独立 confidence 分数。
  NER 不可用时此维度填 NONE，真值表以 6 维模式运行。
```

### 3.8 角色判定

自动规则，不在人工标注范围：

| 角色 | 判定规则 | 示例 |
|------|------|------|
| subject | 列名含 "ssn", "social", "社保", "credit_card", "passport" 等 | `employee_ssn → subject` |
| identifier | 列名含 "id", "account", "ref", "number", "code" 等 | `customer_id → identifier` |
| reference | 其余情况 | `notes, description, comment` |

### 3.9 上下文一致性检查（后置）

```
分类后验证:
  - 值被分为 SSN 但列名/路径含 "account", "email", "phone" → 冲突降信
  - 值被分为 CREDIT_CARD 但所在文本段含 "test transaction" → 标记审查
  - 冲突 → 路由降一级 (原本直接输出的进入 LLM Validation,
    原本 Validation 的进入 Classification)，同时 flag 标记
  - 折扣量由路由阈值自然决定, 不使用硬编码乘数
```

### 3.10 置信度路由阈值校准

```
校准方法 (对抗性验证):
  1. 标注数据集 N 个值 (value, context, true_label, true_confidence)
  2. 对每个值跑真值表 → predicted_confidence
  3. 按 predicted_confidence 分组，计算每组的 precision
  4. 阈值选择:
     - high_threshold: precision ≥ 0.95 的最低 confidence
     - mid_threshold:  precision ≥ 0.70 的最低 confidence
     - low_threshold:  低于 mid  → uncertain

初始占位值: 0.85 / 0.50 / 0.30 — 第一次真值表校准时同步校准为数据驱动值
```

### 3.11 输出

```python
@dataclass
class ValueClassification:
    value_id: str
    value: str
    sensitive_type: str | None     # "SSN", "CREDIT_CARD", None=NON_SENSITIVE
    confidence: float
    method: str                    # "regex_only" | "truth_table" | "fusion" |
                                    # "llm_validate" | "llm_classify"
    role: str | None               # "subject" | "identifier" | "reference"
    is_mock: bool
    needs_review: bool
    evidence: dict                 # 各引擎输出、特征值、中间分数
    source: ValueSource
```

---

## 四、Phase 2：LLM 消歧层

### 4.1 架构定位

LLM **不在分类内核里**，而是内核之后、输出之前的独立消歧层。LLM 不参与初判——真值表+NER 先对所有值产出初判结果，LLM 只处理中低置信度的值。

### 4.2 两层设计

| | LLM Validation (判断题) | LLM Classification (论述题) |
|------|------|------|
| 对应 Cyera | Layer 3: LLM Validation | Layer 4: LLM-Based Classification |
| 触发条件 | 0.50 ≤ conf < 0.85 | conf < 0.50 |
| 输入 | 值 + 上下文 + 真值表候选类型 | 值 + 上下文 + 完整敏感类型列表 |
| 任务 | "系统判断此值为 {candidate_type}，对吗？" | "此数据属于哪种敏感类型？" |
| 输出 | yes/no + confidence | sensitive_type + confidence |
| 模型 | FLAN-T5 Large | Mistral-7B-Instruct |
| 预期延迟 | <10ms GPU | ~1s GPU |

### 4.3 模型选型

英文优先。

| 层 | 模型 | 参数 | 架构 | 延迟 | 选择理由 |
|------|------|------|------|------|------|
| **Validation** | FLAN-T5 Large | 780M | Encoder-Decoder | <10ms GPU | Cyera 原案；双向注意力完整捕获上下文；Text-to-Text 约束输出降幻觉 |
| **Classification** | Mistral-7B-Instruct | 7B | Decoder-Only | ~1s GPU | Cyera 原案；英文 PII Recall 0.9625（已发表同行评审验证） |

### 4.4 Prompt 设计

**Validation（判断题，FLAN-T5）**：
```
Verify if the value is a {candidate_type}.
Value: {value}
Context: column={label_hint}, surrounding_text={surrounding_text[:200]}
System confidence: {confidence}
Answer yes or no with confidence (0.0-1.0):
```

**Classification（论述题，Mistral）**：
```
[System]: You are a data security classifier. Classify the data value into one
of the known sensitive types, or NON_SENSITIVE if not sensitive.

Known types: {type_list}

Value: {value}
Context: column={label_hint}, surrounding_text={surrounding_text[:300]}

Answer with JSON: {"type": "<type or NON_SENSITIVE>", "confidence": 0.0-1.0,
"reason": "<one short sentence>"}
```

### 4.5 LLM 输出处理

```
Validation 结果处理:
  answer="yes" → 保持 candidate_type, confidence = max(true_table.conf, llm.conf)
  answer="no"  → 降级：
    - 真值表有次高候选 → 取次高, route to Classification
    - 无次高候选 → route to Classification

Classification 结果处理:
  type=NON_SENSITIVE → 输出 NON_SENSITIVE
  type=<known_type>  → 输出该类型, confidence = llm.conf
  type=<unknown>     → 标记 uncertain, needs_review=True
```

### 4.6 批量优化

```
批量策略:
  1. 收集中等置信度值 [v1, v2, ..., vn]
  2. FLAN-T5: transformers 内置 batch inference (tokenizer padding + model.generate batch)
     Mistral: asyncio + Ollama batch API
     两者异步并发
  3. 收集 Validation 不通过的值 [vi, vj, ...]
  4. 并发发送 Classification 请求
  5. 聚合结果
```

### 4.7 降级路径

```
FLAN-T5 不可用:
  → 跳过 Validation 层
  → 中等置信度值直接进入 Classification (Mistral)

Mistral 不可用:
  → Classification 不可用
  → 低置信度值标记 uncertain, needs_review=True
  → 中等置信度值保持原始真值表判定 (× 0.8 折扣)

两模型同时不可用:
  → 真值表+NER 独立运行
  → 低置信度标记 uncertain
  → FDR 不增加 (LLM不推翻真值表初判)

LLM 响应超时 (>5s):
  → 同不可用
```

### 4.8 VRAM 预算

| 组件 | 体积 |
|------|------|
| FLAN-T5 Large (FP16) | ~1.5GB |
| Mistral-7B (Q4_K_M) | ~4GB |
| 嵌入模型 (E5-base / all-mpnet) | ~0.5GB |
| 真值表 + NER 模型 | ~1GB |
| **合计** | **~7GB / 12GB** |

---

## 五、Phase 3：Semantic Distancing + 文件级聚类传播

### 5.1 Semantic Distancing

#### 5.1.1 定位

对标 Cyera 第 2 组件。独立评分路径，与真值表并列融合，不作为真值表的内嵌维度。

#### 5.1.2 方案

```
值 → PII 类型占位符替换 (类型级, 不解析子结构):
  "4111-1111-1111-1111"         → "[CREDIT_CARD]"
  "123-45-6789"                 → "[SSN]"
  "john.doe@gmail.com"          → "[EMAIL]"
  "GB29NWBK60161331926819"      → "[IBAN]"
     ↓
替换后文本 → E5-base / all-mpnet 嵌入 → [1×768] 向量
     ↓
与已知模板库计算 cosine similarity:
  template_library = {sensitive_type: [template_embedding_1, template_embedding_2, ...]}
  distance_score = max_cos_sim(value_embedding, type_templates)
     ↓
  distance_score ∈ [0, 1] → 与真值表 confidence 并列融合
```

#### 5.1.3 融合

```
weighted_confidence = α × calibrated_truth_table + (1-α) × calibrated_distance

α 默认 0.7, Semantic Distancing 不可用时 α=1.0。校准方法见 Section 3.7。
```

### 5.2 文件级聚类传播

#### 5.2.1 核心洞察

对标 Cyera 第 1 组件和专利 US20240362301A1。大部分文件/表不需要逐值分类——相似结构的文件归入同一簇，仅处理代表文件。

#### 5.2.2 两层降维

```
层1: 文件级聚类
  元数据替换 → 结构指纹 → 文件簇
  1000 万文件 → ~5000 簇

层2: 列内抽样 (Phase 1)
  每列 5-20 代表值

总降维比: 99.99%+
```

#### 5.2.3 文件元数据归一化

```
原始元数据:
  "file=employees.csv, columns=[ssn, name, salary, dob, ...], file_type=csv"

PII 替换后:
  "file=[NAME], columns=[[SSN_KEYWORD], [NAME_KEYWORD], [MONEY_KEYWORD], ...], file_type=csv"

Hash → cluster_fingerprint
```

#### 5.2.4 流式聚类算法

```
时间: O(n), n = 文件数
空间: O(k), k = 簇数

for each file in data_sources:
  fingerprint = normalize_and_hash(file.metadata)
  if fingerprint in cluster_map:
    cluster_map[fingerprint].add_file(file)
  else:
    cluster_map[fingerprint] = new_cluster(fingerprint, [file])
```

#### 5.2.5 分类与传播

```
for each cluster:
  1. 选代表文件 (按文件大小/列数多样性选择 3 个)
     哈希指纹聚类无向量空间，代表文件选择策略为:
     - 优先选择不同文件大小的文件 (覆盖大/中/小文件)
     - 优先选择不同列数的文件 (覆盖宽/窄表)
     - 若无多样性差异, 随机选择
  2. 代表文件 → 值提取 → 分类内核 → classifications
  3. majority_type = majority_vote(classifications)
  4. consistency = majority_count / total
  5. if consistency ≥ 0.8:
       propagate(cluster, majority_type, confidence=consistency)
     else:
       refine_and_reclassify(cluster)  # 拆分子簇递归
  6. 传播后随机抽检: 每簇 5% 列 × 3 值独立验证
     不一致率 > 10% → 触发簇拆分重分类
```

#### 5.2.6 列级优化

对结构化数据源（CSV、数据库表），列内所有值通常具有相同敏感类型。同一列的 1000 万行 → 仅分类 10 个代表值。这是文件级聚类之下的加速层，非普遍机制。

---

## 六、Phase 4：Learned Classification 闭环

### 6.1 对标 Cyera

Cyera 第 5 组件：自动识别每个组织独有的专有数据类型——内部客户 ID、产品 SKU、工单编号、仓库编码。这些类型不在预置类型库中，每个组织不同，没有公开标注数据。

### 6.2 四阶段闭环

```
阶段1: 未知模式收集 ── 低置信 + 强结构但未匹配 → 缓冲池
阶段2: 模式聚类与候选提名 ── 缓冲池 ≥ 500 → 聚类 → 候选新类型
阶段2.5: 自动验证 ── 统计一致性 + 模板冲突 + 现有引擎验证
阶段3: 人工 Gate ── 确认/标记非敏感/忽略
阶段4: 引擎更新 ── TypeLibrary + 真值表增量校准 + NER 增量微调 → 部署
```

### 6.3 阶段 1：未知模式收集

```
触发条件 (满足任一):

A. 新兴重复模式:
   regex_strength > 0.5 AND validated_count < 5 AND total_count > 500

B. 真值表盲区:
   truth_table_confidence < 0.3 AND regex_strength > 0.7

C. label_hint-分类冲突 (适用所有数据源):
   真值表分为 type_A AND 值的 label_hint 在模板嵌入中与 type_B 更近 (cos_sim > 0.7, A ≠ B)
   label_hint 对结构化源=列名/字段名，非结构化源=提取时的上下文键名

缓冲池: 最多 200 个候选模式 (LRU淘汰), 去重
```

### 6.4 阶段 2.5：自动验证

在人工 Gate 之前插入，减少人工负担：

```
自动通过条件 (满足任一):
  - 统计一致性: 候选值列名一致性 > 80%
  - 模板冲突: 与已有类型 cos_sim < 0.85 (确实是新类型, 不是已知类型的变体)
  - 现有引擎验证: 真值表 confidence < 0.3 (现有引擎不认识)

自动拒绝条件:
  - 与已有类型 cos_sim ≥ 0.85 (自动合并, 不提名)
  - 现有引擎 confidence ≥ 0.7 且类型正确 (引擎已能覆盖)

仅中等置信度 (0.5-0.8) 的候选 → 进入人工 Gate
```

### 6.5 阶段 3：人工 Gate

唯一必须人工参与的节点。安全团队确认/标记非敏感/忽略。

候选类型的模板嵌入和统计特征用于后续匹配。**不自动推断正则**——安全团队在有领域知识的情况下手动提供正则，或在积累更多示例后补充。

### 6.6 阶段 4：引擎更新

```
1. TypeLibrary 更新: 新类型条目 + 50+ 示例值 + 上下文

2. 真值表增量校准:
   - 正样本: 已确认的示例值
   - 负样本: 人工标记 NON_SENSITIVE 的值 + 跨类型负样本
   - 校准: Log-likelihood ratio (非纯频率)
   - 仅重校准受影响的 bin 区域

3. NER 增量微调:
   触发条件 (满足任一):
     - 任一类型的已确认值 ≥ 200
     - 所有新增类型合计 ≥ 500
     - 距上次训练 > 14 天 AND 任一类型 ≥ 50
   方式: LoRA (rank=8, ~5min)
   仅添加新 BIO 标签, 不修改已有标签权重

4. Quality Gate:
   真值表新增 bin:  held-out precision ≥ 0.95
   NER LoRA 新类型:  held-out F1 ≥ 0.90
   已有类型退化:     F1 下降 < 1%
   系统级:
     - E2E Macro Recall ≥ 0.85
     - Uncertain 率 < 0.30
   Gate 不通过 → 回滚, 标记 uncertain, 等待更多数据

5. 闭环延迟窗口:
   人工确认后 → 立即写入 type_cache (pattern_hash → confirmed_type)
   缓存 TTL: 24h
   引擎更新后 → 验证 → 清除缓存
```

### 6.7 退化防护

| 风险 | 防护 |
|------|------|
| 类型爆炸 (CUST001 vs CLI001 分两类型) | 模板 cos_sim > 0.85 → 自动合并建议 |
| 模式过拟合 (过紧正则) | 正则需在 ≥100 个已确认值上通过覆盖率和特异性测试 |
| 概念漂移 | 每次校准后对比新旧真值表，confidence 变化 > 0.2 的 bin → 告警 |
| 噪声累积 | 人工确认值 template embedding 与已有同类型 cos_sim < 0.6 → 标记需复核 |

---

## 七、数据策略

### 7.1 两条独立流水线

**流水线A：真值表校准数据**

```
输入: 完整数据集 (非标注片段)
  1. 全数据集正则提取 → 统计模式频率、值唯一性
  2. 对每个值计算 6 维特征
  3. 标注每个值的敏感类型 (正则匹配 + LLM验证 + 人工确认)
  4. 构建校准 DataFrame: 6维 bins → annotated confidence
```

**流水线B：NER 训练数据**

```
输入: 标注 span 数据 (ai4privacy, conllpp)
  1. 转换为 BIO 标注
  2. 敏感类型映射到 BIO entity type
  3. 微调 BERT
```

### 7.2 数据源

#### 公开数据

| 数据源 | 覆盖类型 | 用途 |
|------|------|------|
| SWIFT IBAN Registry | IBAN 全球格式 | 真值表校准 |
| PCI DSS 测试卡号 | CCN | 真值表校准 |
| US SSA Randomization | SSN 格式和分配规则 | 真值表校准 |
| IP RFC 规范 | IPv4/IPv6 | 真值表校准 |
| Faker 库 | 全 PII/PCI 类型合成生成 | 真值表校准 + Mock 检测 |
| Microsoft Presidio | 20+ PII 类型识别器 | 基准对比 |
| ai4privacy/pii-masking-300k | 英文 PII span | NER 训练 |
| conllpp | 英文 NER | NER baseline |

#### 合成数据

```python
SYNTHETIC_GENERATORS = {
    "SSN":              random_ssn,
    "CREDIT_CARD":      generate_luhn_valid_ccn,
    "IBAN":             generate_iban_for_country,
    "PASSPORT":         generate_passport_number,
    "EMAIL":            random_email,
    "PHONE":            random_phone_by_country,
    "IP":               random_ip,
    "DRIVER_LICENSE":   random_driver_license,
    # ...
}

每类型: 1000 正样本 + 100 负样本 (相似但不合法)
每值: 3 种上下文变体 (clean / penalty_term / boost_term)
```

#### 企业文档数据

Cxh5types 258 篇文档 → 值提取 → 分轮标注。

### 7.3 分轮标注方案

每轮只做单一决策，避免标注者决策疲劳：

```
第1轮: 类型标注 (500 值/轮/人)
  问题: "这个值属于哪种敏感类型？"
  选项: SSN / CCN / EMAIL / PHONE / IBAN / PASSPORT / DRIVER_LICENSE /
        IP / API_KEY / BANK_ACCOUNT / NAME / ADDRESS / NON_SENSITIVE

第2轮: 上下文验证 (仅第1轮的敏感值)
  问题: "此值的上下文是否支持其类型判定？"
  选项: 支持 / 矛盾 / 不确定

第3轮: Mock 检测 (仅第1轮的敏感值)
  问题: "此值是否为虚假/测试数据？"
  选项: 真实 / 测试 / 不确定

质量控制:
  - 2 人独立标注同一批
  - Cohen's Kappa ≥ 0.85
  - 争议 → 第 3 人裁定或标记 ambiguous
  - 争议模式写入标注指南
```

### 7.4 分布偏移监控

```
每次部署到新数据环境:
  1. 在新数据上计算 6 维特征分布
  2. 与校准数据分布做 KL 散度对比
  3. 任一维度 KL > 0.5 → 告警: 校准表可能不适配
  4. 建议: 在新数据积累标注后重校准
```

### 7.5 数据版本管理

```
datasets/
├── public/
│   ├── swift_iban/
│   ├── pci_dss/
│   ├── pii-masking/
│   ├── conllpp/
│   └── synthetic/
├── enterprise/
│   └── cxh5types/
│       ├── extracted/         # 提取的值 (时间戳版本)
│       ├── annotated/
│       │   ├── v1.0/          # 首轮标注
│       │   └── v1.1/          # 争议解决后
│       └── splits/            # train/val/test
└── calibration/
    ├── truth_table_v1.parquet
    └── truth_table_v2.parquet
```

---

## 八、评估基准与硬性要求

### 8.1 核心评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Per-type Recall | 某一敏感类型的检出率 | 类型级漏洞检测 |
| Per-type Precision | 某一敏感类型的判定准确率 | 类型级误报检测 |
| Macro Recall | 所有类型 Recall 均值 | 防止小众类型被放弃 |
| Macro Precision | 所有类型 Precision 均值 | 防止误报集中在某些类型 |
| Pooled FDR | 全量误报 / 全量正预测 | 用户体验——告警中多少是假的 |
| Per-type Miss Rate | 1 - Per-type Recall | 任一类型放弃上限 |
| LLM 调用率 | 经 LLM 消歧的值占比 | 效率度量（不预设目标值） |
| P50/P95/P99 延迟 | 单值分类延迟分布 | 吞吐规划输入 |

### 8.2 硬性要求 (R1-R7)

#### R1：召回率底线

| 指标 | 目标 | 性质 |
|------|:--:|------|
| Per-type Recall (强结构: SSN/CCN/IP/IBAN) | ≥ 0.98 | 设计目标 |
| Per-type Recall (弱结构: Email/Phone) | ≥ 0.92 | 设计目标 |
| Per-type Recall (语义: Name/Address/API Key) | ≥ 0.80 | 设计目标 |
| Macro Recall (全类型) | ≥ 0.90 | 设计目标 |
| Per-type Miss Rate 上限 | < 0.15 | 合规约束 |

> 注：Cyera 的 Mistral PII Recall 0.9625 为已发表论文数据，作为参考点而非硬门槛。

**当前状态**：❌ 未测量。Phase 1 完成后第一次真值表校准时测定。

#### R2：精度约束

| 指标 | 目标 |
|------|:--:|
| Macro Precision | ≥ 0.90 |
| Pooled FDR | < 0.10 |
| Per-type Precision 下限 | ≥ 0.75 |

> 注：Cyera 95%+ 精度为营销宣称（未经验证），本设计取略保守值。

**当前状态**：❌ 未测量。

#### R3：首日可用

| 条件 | 强结构 (SSN/CCN/IP/IBAN) | 弱结构 (Email/Phone) | 语义 (Name/Address/API Key) |
|------|:--:|:--:|:--:|
| 冷启动 (预校准+NER+LLM) | ≥ 0.92 | ≥ 0.85 | ≥ 0.70 |
| 无 LLM (真值表+NER) | ≥ 0.85 | ≥ 0.75 | ≥ 0.55 |
| 最坏 (仅正则) | ≥ 0.70 | ≥ 0.55 | ≥ 0.30 |

**当前状态**：❌ 未测量。

#### R4：效率

**不预设目标值**。Phase 1 完成后从实测基线出发设定优化目标。

```
测量协议:
  硬件: RTX 5070 12GB
  并发: 单请求串行 + 4并发 (各测一次)
  数据集: 混合合成数据 (10K 值, 15 PII 类型均匀分布)
  GPU 内存: ≤ 10GB (2GB buffer)
  排除: 首次加载模型的冷启动时间

Phase 3 聚类传播完成后:
  - 有效 LLM 调用率对比 (有/无聚类)
  - 端到端吞吐量: 值/秒
```

**当前状态**：⏳ 目标待 Phase 1 实测后校准。

#### R5：降级容错

| 故障场景 | 受影响类型 | 容许退化 | 依据 |
|------|------|:--:|------|
| NER 不可用 | 弱结构 + 语义 | Recall 下降 < 20% | 真值表不覆盖，LLM 接管 |
| NER 不可用 | 强结构 | Recall 下降 < 5% | 强结构不依赖 NER |
| 真值表不可用 | 强结构 | Recall 下降 < 30% | NER+LLM 部分接管，NER 天然弱于格式识别 |
| 真值表不可用 | 弱结构 + 语义 | Recall 下降 < 10% | 这些本来就不走真值表 |
| LLM 不可用 | 所有 | FDR 不增加，uncertain 率上升 | LLM 不推翻初判 |
| 嵌入模型不可用 | 所有 | 无影响 | Semantic Distancing 跳过 |
| 全组件最差 (仅正则) | 强结构 | Recall 下降 < 40% | 正则兜底 |

验证方法：故障注入 + held-out 标注集，每个场景独立测量。

**当前状态**：❌ 未测量。

#### R6：质量监控与告警

| 监控项 | 方法 | 告警阈值 |
|------|------|------|
| Per-type Recall 漂移 | 时间滑动窗口 vs 基线 | 连续 3 窗口下降 > 10% (绝对值) |
| FDR 估算 | 每 1000 值抽 20 **人工**标注 | 抽检 FDR > 0.15 |
| 分布偏移 | 6 维特征 KL 散度 vs 校准基线 | KL > 0.5 |
| entity_type_hint 冲突 | NER entity_type_hint 与最终分类冲突率 | > 3σ 偏离历史均值 |
| LLM 异常 | needs_review 率 | > 3σ 偏离 |
| 缓冲池增长 | 候选新类型发现率 | > 基线 3σ (基线首周后自动计算) |
| 传播质量 | 每簇 5% 列 × 3 值抽检 | 不一致率 > 10% |

**当前状态**：⏳ 框架已定义，阈值需从实测基线校准。

#### R7：可证伪——每个组件有量化退出条件

| 组件 | 退出条件 | 测量方法 |
|------|------|------|
| 真值表 | 任一 bin held-out 样本数 < 10 且 confidence CV > 0.3 | 重校准前 bin 稳定性检验 |
| NER 模型 | held-out Macro F1 < 0.85 或任一类型 F1 下降 > 3% vs 上版本 | 5-fold CV |
| Semantic Distancing | 模板嵌入同类 cos_sim 均值 < 0.7 或类间差距 < 0.15 | 模板库内聚类质量 |
| FLAN-T5 Validation | 200 题标准判断题集 accuracy < 0.90 | 独立判断题集 |
| Mistral Classification | 200 题标准分类题集 accuracy < 0.85 | 独立分类题集 |
| 聚类传播 | 传播后抽检不一致率 > 10% | 每簇 5% 列 × 3 值 |
| Learned Classification | 自动验证层通过率 < 70% | 提议类型总数 / 自动通过数 |
| **系统级** | E2E Macro Recall < 0.85 或 uncertain 率 > 0.30 | held-out 全类别集 |

### 8.3 R 要求状态总览

| # | 核心指标 | Phase 1 目标 | 最终目标 |
|:--:|------|:--:|:--:|
| R1 | Macro Recall | ≥ 0.85 | ≥ 0.90 |
| R2 | Macro Precision | ≥ 0.85 | ≥ 0.90 |
| R3 | 冷启动强结构 Recall | ≥ 0.92 | ≥ 0.92 |
| R4 | LLM率+延迟 | 实测基线 | Phase 3 后定 |
| R5 | 故障退化上限 | < 30% | < 15% (强结构) |
| R6 | 质量监控 | Phase 2 启用 | 自适应基线 |
| R7 | 组件退出条件 | 每组件可测 | 量化条件 |

---

## 九、代码结构

```
value-datadna/
├── src/
│   ├── types.py                  # 核心数据类型 (DataValue, ValueContext, ValueClassification...)
│   ├── extractors/
│   │   ├── base.py               # 值提取器基类
│   │   ├── structured.py         # 结构化提取器 (CSV/DB/JSON/XML)
│   │   └── unstructured.py       # 非结构化提取器 (PDF/Text/Code/Email)
│   ├── classifiers/
│   │   ├── truth_table.py        # 真值表引擎 (6维特征 + calibration)
│   │   ├── ner.py                # NER 引擎 (GLiNER → BERT → 蒸馏)
│   │   ├── semantic_distance.py  # Semantic Distancing (PII替换 → 嵌入 → cos_sim)
│   │   └── fusion.py             # 融合评分
│   ├── postprocess/
│   │   ├── mock_filter.py        # Mock 数据过滤 (前置+后置)
│   │   ├── role_detector.py      # 角色判定
│   │   └── context_check.py      # 上下文一致性检查
│   ├── llm/
│   │   ├── flan_t5.py            # FLAN-T5 Validation 客户端
│   │   ├── mistral.py            # Mistral Classification 客户端
│   │   └── batch.py              # 批量并发调度
│   ├── clustering/
│   │   ├── file_clusterer.py     # 文件级聚类 (元数据归一化 + 流式)
│   │   └── propagator.py         # 标签传播 + 质量监控
│   ├── discovery/
│   │   ├── collector.py          # 未知模式收集 + 缓冲池
│   │   ├── nominator.py          # 模式聚类 + 候选提名
│   │   ├── auto_validator.py     # 自动验证 (统计检验 + 模板冲突)
│   │   └── updater.py            # 引擎更新 (TypeLibrary + 增量校准)
│   ├── knowledge/
│   │   ├── type_library.py       # 敏感类型库
│   │   ├── truth_table_data.py   # 真值表校准数据
│   │   ├── template_library.py   # Semantic Distancing 模板库
│   │   └── pii_patterns.py       # PII 正则模式库 (30+ 类型)
│   ├── monitoring/
│   │   ├── metrics.py            # R6 监控指标收集
│   │   ├── audit.py              # 审计日志
│   │   └── drift.py              # 分布偏移检测 (KL 散度)
│   └── evaluation/
│       ├── benchmark.py          # 评估基准
│       └── report.py             # 评测报告生成
├── tests/
│   ├── test_truth_table.py
│   ├── test_ner.py
│   ├── test_fusion.py
│   ├── test_mock_filter.py
│   ├── test_llm.py
│   ├── test_clustering.py
│   ├── test_discovery.py
│   ├── test_degradation.py       # R5 故障注入测试
│   └── test_end_to_end.py        # 全管道测试
├── config.yaml
├── calibrate.py                  # 真值表校准脚本
├── train_ner.py                  # NER 训练脚本
├── evaluate.py                   # 主评估脚本
└── README.md
```

---

## 十、与文档级 DataDNA 的经验教训对照

| 文档级教训 | 值级设计中的对策 |
|------|------|
| 文档级分类 ≠ 值级分类，问题域错位 | 从设计原则到评估指标全部基于值级敏感数据分类 |
| 6引擎融合过度设计 | 单路径（真值表主路径 + NER语义特征 + SD并列评分） |
| R4 (LLM率<20%, <300ms) 不含聚类层导致永远达不到 | R4 不预设目标值；聚类层从 Phase 3 设计之初就纳入 |
| 门控阈值 0.85 无数据支撑 (改为 0.42/0.32) | 所有阈值对抗性验证校准，初始值仅为占位 |
| SHA256 二值匹配 → 当唯一二值引擎 | Semantic Distancing 作为连续评分，零二值引擎 |
| 3级门控 (fast/validate/full) LLM嵌入引擎中 | LLM 是独立消歧层，与内核解耦，模型可热替换 |
| E3 SetFit OOM 4次 → all-mpnet+LR | 嵌入模型选型从 VRAM 预算出发，FLAN-T5+Mistral+嵌入 ≤ 7GB |
| R5 故障测试只验证"不崩" | R5 每场景有量化退化上限 |
| R7 可证伪只写元描述 | R7 每个组件有具体退出条件和测量方法 |
| 标注方案依赖LLM → 循环验证 | 标注分轮+人工作 Ground Truth |
| PB层分析过但未实现 | 文件级聚类从 Phase 3 实现 |

---
