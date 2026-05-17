# Benchmark 模型与数据集参考

**更新日期:** 2026-05-17

## 一、文档分类数据集

> **图例：** ✓ = 支持 &nbsp;|&nbsp; ✗ = 不支持/不适用

### 1.1 Ben25

| 属性 | 值 |
|---|---|
| Type Key | `ben25` |
| 数据来源 | 本地 JSONL + JSON（`ZerosOne/gemma-doc-label/testdata/ben25/`） |
| 规模 | 25 篇文档 |
| 标签 | L1/L2 两级，GPT 标注，使用 `file_labels_20_10.json` 分类体系 |
| 层级 | 是 |

**适用模型：**

| 模型 | GPU | CPU | 说明 |
|---|---|---|---|
| `FlanT5ClassificationModel` | ✓ 默认 `cuda`，自动回退 | ✓ | prompt 两步推理，25 篇规模适合快速实验 |
| `DocClassifierSklearnModel` | ✗ | ✓ 仅 CPU | TF-IDF + LogisticRegression，预训练标签匹配 |
| `GemmaDocLabelModel` | N/A (HTTP) | N/A (HTTP) | HTTP 调用 Ollama Gemma，服务端标签匹配 |

### 1.2 Cxh5types

| 属性 | 值 |
|---|---|
| Type Key | `cxh5types` |
| 数据来源 | 本地 JSONL + JSON（`ZerosOne/gemma-doc-label/testdata/cxh5types/`） |
| 规模 | 258 篇文档 |
| 标签 | L1/L2 两级，人工标注 |
| 层级 | 是 |

**适用模型：**

| 模型 | GPU | CPU | 说明 |
|---|---|---|---|
| `FlanT5ClassificationModel` | ✓ 默认 `cuda`，自动回退 | ✓ | prompt 两步推理，适合层级标签 |
| `DocClassifierSklearnModel` | ✗ | ✓ 仅 CPU | TF-IDF + LogisticRegression，预训练标签匹配 |
| `GemmaDocLabelModel` | N/A (HTTP) | N/A (HTTP) | HTTP 调用 Ollama Gemma，服务端标签匹配 |

### 1.3 Dspm27

| 属性 | 值 |
|---|---|
| Type Key | `dspm27` |
| 数据来源 | 本地 PDF（`pdfplumber` 提取文本，`ZerosOne/gemma-doc-label/testdata/dspm27/`） |
| 规模 | 27 篇文档 |
| 标签 | L1/L2 两级，GPT 标注 |
| 层级 | 是 |

**适用模型：**

| 模型 | GPU | CPU | 说明 |
|---|---|---|---|
| `FlanT5ClassificationModel` | ✓ 默认 `cuda`，自动回退 | ✓ | prompt 两步推理，适合层级标签 |
| `DocClassifierSklearnModel` | ✗ | ✓ 仅 CPU | TF-IDF + LogisticRegression，预训练标签匹配 |
| `GemmaDocLabelModel` | N/A (HTTP) | N/A (HTTP) | HTTP 调用 Ollama Gemma，服务端标签匹配 |

### 1.4 20 Newsgroups

| 属性 | 值 |
|---|---|
| Type Key | `20newsgroups` |
| 数据来源 | HF `SetFit/20_newsgroups` |
| 规模 | 7,532 篇（test split） |
| 标签 | 20 类新闻组（平层，L2 为空） |
| 层级 | 否 |

**适用模型：**

| 模型 | GPU | CPU | 说明 |
|---|---|---|---|
| `FlanT5ClassificationModel` | ✓ 默认 `cuda`，自动回退 | ✓ | 20 类英文，prompt 长度可控，适合 |
| `DocClassifierSklearnModel` | ✗ | ✗ | 预训练标签集不匹配 20newsgroups |
| `GemmaDocLabelModel` | ✗ | ✗ | 服务端标签体系固定 |

### 1.5 Ledgar

| 属性 | 值 |
|---|---|
| Type Key | `ledgar` |
| 数据来源 | HF `lex_glue/ledgar` |
| 规模 | 10,000 篇（test split） |
| 标签 | 100 类法律合同条款（平层，L2 为空） |
| 层级 | 否 |

**适用模型：**

| 模型 | GPU | CPU | 说明 |
|---|---|---|---|
| `FlanT5ClassificationModel` | ✓ 默认 `cuda`，自动回退 | ✓ | 100 类平层，单 prompt 列出全部候选，FLAN-T5 百级分类精度有限 |
| `DocClassifierSklearnModel` | ✗ | ✗ | 预训练标签集不匹配 Ledgar |
| `GemmaDocLabelModel` | ✗ | ✗ | 服务端标签体系固定 |

### 1.6 German-MultiFin

| 属性 | 值 |
|---|---|
| Type Key | `german-multifin` |
| 数据来源 | HF `anhaltai/german-multifin` |
| 规模 | 2,010 篇（test split） |
| 标签 | 6 L1 × 23 L2，德语财务文档 |
| 层级 | 是 |

**适用模型：**

| 模型 | GPU | CPU | 说明 |
|---|---|---|---|
| `FlanT5ClassificationModel` | ✓ 默认 `cuda`，自动回退 | ✓ | 层级结构适合两步推理，但 FLAN-T5 是英文模型，德语文本影响准确率 |
| `DocClassifierSklearnModel` | ✗ | ✗ | 预训练标签集不匹配 |
| `GemmaDocLabelModel` | ✗ | ✗ | 服务端标签体系固定 |

---

## 二、NER 识别数据集

### 2.1 Conll03

| 属性 | 值 |
|---|---|
| Type Key | `conll03` |
| 数据来源 | HF `conllpp` |
| 规模 | ~3,500 篇（test split） |
| 实体类型 | PER, ORG, LOC, MISC（4 种通用实体） |
| 标注格式 | BIO |

**适用模型：**

| 模型 | GPU | CPU | 说明 |
|---|---|---|---|
| `FlanT5Model` | ✓ 默认 `cuda`，4-bit/8-bit 量化 | ✓ 自动回退 | small/base: token-classification；large: text2text prompt |

---

## 三、PII 识别数据集

> PII 检测是 NER 的子集应用——技术同源，实体类型聚焦于隐私合规领域。

### 3.1 PiiMasking

| 属性 | 值 |
|---|---|
| Type Key | `pii-masking` |
| 数据来源 | HF `ai4privacy/pii-masking-300k` |
| 规模 | ~60K 篇（test split，80/20 划分） |
| 实体类型 | 17 种 PII（见下方列表） |
| 标注格式 | BIO |

**实体类型：** PERSON, EMAIL, PHONE, STREET_ADDRESS, CITY, STATE, ZIP_CODE, DATE_OF_BIRTH, AGE, ID_CARD, PASSPORT, DRIVERS_LICENSE, SSN, CREDIT_CARD, BANK_ACCOUNT, IP_ADDRESS, URL

**适用模型：**

| 模型 | GPU | CPU | 说明 |
|---|---|---|---|
| `FlanT5Model` | ✓ 默认 `cuda`，4-bit/8-bit 量化 | ✓ 自动回退 | small/base: token-classification；large: text2text prompt |

### 3.2 SyntheticPii

| 属性 | 值 |
|---|---|
| Type Key | `synthetic-pii` |
| 数据来源 | 程序化生成（无外部依赖） |
| 规模 | 默认 1,000 篇（80/20 train/test 划分） |
| 实体类型 | 12 种 PII（见下方列表） |
| 标注格式 | BIO |

**实体类型：** API_KEY, BANK_ACCOUNT, CREDIT_CARD, DATE_OF_BIRTH, DRIVERS_LICENSE, EMAIL, IP_ADDRESS, PASSPORT, PASSWORD, PHONE, SSN, URL

**适用模型：**

| 模型 | GPU | CPU | 说明 |
|---|---|---|---|
| `FlanT5Model` | ✓ 默认 `cuda`，4-bit/8-bit 量化 | ✓ 自动回退 | small/base: token-classification；large: text2text prompt |

---

## 四、模型硬件支持总览

| 模型 | Type Key | 任务类型 | GPU | CPU | 说明 |
|---|---|---|---|---|---|
| `FlanT5Model` | `flan-t5` | NER / PII | 默认 `cuda`，4-bit/8-bit 量化 | 自动回退 | HF pipeline，三种规格：small(77M) / base(250M) / large(780M) |
| `FlanT5ClassificationModel` | `flan-t5-classification` | 文档分类 | 同上 | 同上 | 继承 FlanT5Model 硬件逻辑，强制 text2text 模式 |
| `DocClassifierSklearnModel` | `doc-classifier-sklearn` | 文档分类 | ✗ | 仅 CPU | sklearn TF-IDF + LogisticRegression，需 `ZerosOne/doc-classifier` |
| `GemmaDocLabelModel` | `gemma-doc-label` | 文档分类 | N/A | N/A | HTTP → `127.0.0.1:8003/classify_text`，硬件取决于服务端 |

### 硬件详情

- **FlanT5 系列**: 构造函数默认 `device="cuda"`，`torch.cuda.is_available()` 检测，不可用自动回退 CPU。支持 4-bit (`BitsAndBytesConfig(load_in_4bit=True)`) 和 8-bit 量化。
- **sklearn 模型**: 纯 CPU。导入阶段 mock 掉 `torch`/`transformers`（`_FakeModule`），避免加载 GPU 库。
- **Gemma 模型**: 纯 HTTP 客户端，超时 600s。本地无推理计算。

### FlanT5ClassificationModel vs FlanT5Model

`FlanT5ClassificationModel` 不是独立模型权重，是对同一 FLAN-T5 checkpoint 的 prompt 封装。区别在于怎么用：

| | FlanT5Model (NER) | FlanT5ClassificationModel |
|---|---|---|
| **推理模式** | small/base: token-classification；large: text2text | 强制 text2text（所有 variant） |
| **输入** | 原始文本 | 分类指令 + 候选标签 + 文档内容 |
| **输出** | `List[Entity]` | `{"l1": "...", "l2": "..."}` |
| **推理路径** | 单次调用 | 默认两步：L1 → L2（也支持 single_step） |
| **分类灵活性** | 实体类型由 NER checkpoint 写死 | 分类体系通过 prompt 传入，不换模型 |

本质：**同一个模型，不同 prompt**——instruction-tuned 模型的核心优势。

### 本地路径

| Variant | 缓存路径 |
|---|---|
| small | `~/.cache/huggingface/hub/models--google--flan-t5-small/` |
| base | `~/.cache/huggingface/hub/models--google--flan-t5-base/` |
| large | `~/.cache/huggingface/hub/models--google--flan-t5-large/` |

---

## 五、PII、NER 与文档分类的关系

三者从**粒度**和**目的**两个维度区分：

```
文档分类 (Document-Level) → "这份文件是医疗记录"
  │
  ▼
NER / PII检测 (Token-Level) → "患者姓名: John Doe" → PERSON
                               "SSN: 123-45-6789"  → SSN
```

- **NER** 是技术，识别文本中的命名实体（任何类别）
- **PII 检测** 是 NER 的子集，实体类型限定为隐私敏感信息
- **文档分类** 是独立任务，判断文档整体类别

实际部署中的流水线关系：文档分类决定"该找什么"（触发 HIPAA/GDPR/PCI-DSS 规则），PII 检测执行"找到它"（定位具体敏感数据）。

---

## 六、GDPR、HIPAA、PCI-DSS 概述

| | GDPR | HIPAA | PCI-DSS |
|---|---|---|---|
| **地区** | 欧盟（域外适用） | 美国 | 全球 |
| **行业** | 所有行业 | 医疗/健康保险 | 支付卡行业 |
| **保护对象** | 自然人个人数据 | 受保护健康信息 (PHI) | 持卡人数据 + 敏感认证数据 |
| **法律效力** | 法律，罚款最高全球营收 4% | 法律，罚款 + 刑事责任 | 行业合同标准，罚款 + 吊销收单资格 |
| **生效** | 2018-05 | 1996（多次修订） | 2004 v1.0 |

三者**互不替代**，但可同时适用（如欧洲医院接受信用卡 → GDPR + HIPAA + PCI-DSS 叠加）。

---

## 七、NER/PII 数据集 × 合规框架覆盖分析

### 实体类型矩阵

| 实体类型 | Conll03 | PiiMasking | SyntheticPii |
|---|---|---|---|
| PERSON (姓名) | ✓ PER | ✓ | ✗ |
| EMAIL / PHONE | ✗ | ✓ | ✓ |
| ADDRESS (地址相关) | ✗ | ✓ (5种) | ✗ |
| DATE_OF_BIRTH / AGE | ✗ | ✓ | ✓ |
| ID_CARD / PASSPORT / DRIVERS_LICENSE | ✗ | ✓ | ✓ |
| SSN | ✗ | ✓ | ✓ |
| CREDIT_CARD / BANK_ACCOUNT | ✗ | ✓ | ✓ |
| IP_ADDRESS / URL | ✗ | ✓ | ✓ |
| API_KEY / PASSWORD | ✗ | ✗ | ✓ |
| ORG / LOC / MISC | ✓ | ✗ | ✗ |

> Conll03 是通用 NER，不含 PII 实体，对三个合规框架均无直接覆盖。

### 合规覆盖评级

| 框架 | 覆盖率 | 评估 |
|---|---|---|
| GDPR | ~55% | 一般标识符覆盖较好，Art.9 特殊类别（种族、健康、生物特征等）完全缺失 |
| HIPAA | ~50% | 通用标识符覆盖尚可，核心医疗标识符（病历号、医保号）缺失 |
| PCI-DSS | ~29% | 仅覆盖卡号+姓名，支付认证数据（CVV、有效期、PIN）全部缺失 |

### 关键缺口

1. **医疗数据** — 无病历号、医保号、治疗日期 → HIPAA 场景不可用
2. **支付认证** — 无 CVV、有效期、PIN → PCI-DSS 场景无法模拟审计
3. **GDPR 特殊类别** — 无种族、健康、生物特征 → 高风险合规场景无法测试
4. **Conll03** — 通用 NER，对三个框架均无直接价值

---

## 八、实验配置

| 配置组 | 默认 device | 来源 |
|---|---|---|
| `flan-t5-*` | `cuda` | `flan-t5-defaults.yaml` / `flan-t5-classification-defaults.yaml` |
| `doc-classifier-*` | `cpu` | 各实验配置文件 |
| `gemma-doc-label-*` | `cpu` | 各实验配置文件 |

Orchestrator 透传 `device` 参数，不自行决定。throughput 阶段可选 CUDA 同步 + GPU 峰值内存记录。
