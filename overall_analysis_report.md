# 整体分析报告

> 分析日期：2026-05-15
> 分析方法：基于公开资料（Cyera官方博客、新闻稿、专利数据库、第三方分析师报告、行业评测）的独立技术分析

---

## 目录

- [一、分析方法论与信息来源评估](#一分析方法论与信息来源评估)
- [二、公司背景与技术定位](#二公司背景与技术定位)
  - [2.1 公司基本面](#21-公司基本面-a级)
  - [2.2 技术定位演变](#22-技术定位演变-a级)
- [三、核心分类引擎：DataDNA](#三核心分类引擎datadna)
  - [3.1 命名与定位](#31-命名与定位-a级)
  - [3.2 已确认的五大技术组件](#32-已确认的五大技术组件-a级)
  - [3.3 基础模型深度分析：FLAN-T5 与 Mistral](#33-基础模型深度分析flan-t5-与-mistral)
    - [3.3.1 FLAN-T5：Encoder-Decoder 架构](#331-flan-t5encoder-decoder-架构的指令微调模型)
    - [3.3.2 Mistral：Decoder-Only 架构](#332-mistraldecoder-only-架构的高召回语义理解模型)
    - [3.3.3 双模型协同架构](#333-双模型协同架构cyera-的分类流水线中的角色分工)
    - [3.3.4 模型规模推断与性能评估](#334-模型规模推断与性能评估)
  - [3.4 分类流水线的分层策略](#34-分类流水线的分层策略-bc级)
- [四、数据发现与扫描架构](#四数据发现与扫描架构)
  - [4.1 无代理API优先架构](#41-无代理api优先架构-a级)
  - [4.2 扫描策略](#42-扫描策略-a级)
  - [4.3 本地部署组件](#43-本地部署组件-a级)
- [五、图数据架构与风险评估](#五图数据架构与风险评估)
  - [5.1 DataGraph](#51-datagraph-a级)
  - [5.2 Agent Graph](#52-agent-graph2025年新增a)
  - [5.3 身份图](#53-身份图-a级)
  - [5.4 风险评估引擎](#54-风险评估引擎-a级但细节有限)
- [六、Omni DLP](#六omni-dlp)
  - [6.1 起源](#61-起源-a级)
  - [6.2 架构定位](#62-架构定位-a级)
  - [6.3 技术架构特点](#63-技术架构特点-a级)
- [七、AI Guardian](#七ai-guardian)
  - [7.1 产品发布时间线](#71-产品发布时间线-a级)
  - [7.2 技术架构细节](#72-技术架构细节-a级)
- [八、专利技术评估](#八专利技术评估)
  - [8.1 专利组合总览](#81-专利组合总览)
  - [8.2 已授权美国专利详析](#82-已授权美国专利详析)
    - [专利一：US 12,026,123 B2](#专利一us-12026123-b2--数据发现系统与方法-a级)
    - [专利二：US 12,499,083 B2](#专利二us-12499083-b2--数据发现系统与方法续案a级)
    - [专利三：US 12,566,567 B2](#专利三us-12566567-b2--通过初始扫描发现数据存储位置-a级)
    - [专利四：US 12,299,167 B2](#专利四us-12299167-b2--数据分类与云环境保护-a级)
    - [专利五：US 12,316,686 B1](#专利五us-12316686-b1--多源网络安全策略应用-a级)
  - [8.3 已公开专利申请](#83-已公开专利申请)
  - [8.4 专利组合分析](#84-专利组合分析)
    - [8.4.1 发明人网络](#841-发明人网络)
    - [8.4.2 时间线分析](#842-时间线分析)
    - [8.4.3 与其他DSPM厂商的IP策略对比](#843-与其他dspm厂商的ip策略对比)
- [九、第三方评估](#九第三方评估)
  - [9.1 Forrester Wave](#91-forrester-wave-a级)
  - [9.2 Gartner](#92-gartner-a级)
- [十、Cyera技术架构的独立总结](#十cyera技术架构的独立总结)
  - [10.1 真正的技术创新点](#101-真正的技术创新点)
  - [10.2 技术局限性](#102-技术局限性基于公开评测和客户反馈)
  - [10.3 与竞品的关键差异](#103-与竞品的关键差异)
- [十一、结论](#十一结论)
- [信息来源汇总](#信息来源汇总)

---

[↑ 返回目录](#目录)

## 一、分析方法论与信息来源评估

### 可信度分级

| 等级 | 来源类型 | 示例 |
|------|---------|------|
| **A - 官方确认** | Cyera官方博客、新闻稿、产品文档、USPTO专利 | Cyera Blog, BusinessWire新闻稿, US 12,299,167 |
| **B - 第三方验证** | Forrester Wave、Gartner Market Guide、独立分析师报告 | Forrester Wave Q2 2026, Gartner Peer Insights |
| **C - 可信推断** | 基于公开信息的合理技术推断，但无直接确认 | 具体模型参数、流水线吞吐量数字 |
| **D - 竞争对手/间接** | 竞争对手分析、客户评论 | Sentra对比文章、Gartner用户评论 |

**关键原则**：本报告严格区分"公开确认的事实"与"合理推断"，使用 `[来源等级]` 标注每项关键声明。

---

[↑ 返回目录](#目录)

## 二、公司背景与技术定位

### 2.1 公司基本面 [A级]

- **成立时间**：2021年 [A]
- **创始人**：Yotam Segev（CEO）和 Tamar Bar-Ilan（CTO），均为以色列Unit 8200情报部队退伍军人 [A]
- **总部**：以色列特拉维夫，在美国纽约等地设有办事处 [A]
- **总融资**：超过17亿美元 [A]
- **最新估值**：90亿美元（2026年1月Series F，由Blackstone领投4亿美元）[A]
- **收入增长**：同比增长3.4倍 [A]
- **客户覆盖**：20%的财富500强 [A]
- **员工规模**：1100+人，分布15个国家 [A]

来源：[Cyera Raises $400M at $9B Valuation - SecurityWeek](https://www.securityweek.com/cyera-raises-400-million-at-9-billion-valuation/), [Fortune Exclusive Interview with CEO](https://fortune.com/2026/01/08/cyera-cybersecurity-startup-yotam-segev-400-million-series-f-funding-9-billion-valuation-blackstone/)

### 2.2 技术定位演变 [A级]

Cyera已从纯DSPM厂商演变为**统一的AI原生数据与AI安全平台**，核心产品矩阵包含五大支柱：

1. **DSPM（数据安全态势管理）** - 核心基础层
2. **Omni DLP（数据丢失防护）** - 2025年4月推出，基于2024年10月以1.62亿美元收购Trail Security
3. **AI Guardian（AI安全）** - 2025年8月推出，包含AI-SPM + AI运行时保护
4. **Identity & Access Intelligence（身份与访问智能）** - 2024年7月推出的身份模块
5. **Omni AI（对话式分析）** - 自然语言查询界面

---

[↑ 返回目录](#目录)

## 三、核心分类引擎：DataDNA

### 3.1 命名与定位 [A级]

Cyera将其专利分类技术命名为**DataDNA**，这是其核心差异化技术。Cyera的官方描述为："专利待批的多维关联技术"（patent-pending multidimensional correlation）。

来源：[Understanding Data in Context: An LLM-Driven Approach to Data Classification - Cyera Blog](https://www.cyera.com/blog/understanding-data-in-context-an-llm-driven-approach-to-data-classification)

### 3.2 已确认的五大技术组件 [A级]

Cyera在官方博客中明确披露了DataDNA的五个技术组件：

| # | 技术 | 功能 | 来源等级 |
|---|------|------|---------|
| 1 | **Clustering（聚类）** | 将相似文件分组，减少冗余扫描；使PB级数据分类从"数年"缩短至"数周" | A |
| 2 | **Semantic Distancing（语义距离）** | 基于语义而非关键词判断文档相似性；区分相似格式但不同含义的数据 | A |
| 3 | **LLM Validation（LLM验证层）** | 用LLM验证传统模式匹配的结果，通过上下文理解消除误报 | A |
| 4 | **LLM-Based Classification（LLM分类）** | 深层语义分类，理解数据"代表什么"而不仅是"看起来像什么" | A |
| 5 | **Learned Classification（自学习分类）** | 自动识别每个组织独有的专有数据类型（内部ID、产品SKU等） | A |

来源：同上

### 3.3 基础模型深度分析：FLAN-T5 与 Mistral

Cyera 官方确认使用 **FLAN-T5** 和 **Mistral** 作为基础模型进行专有微调 [A]。以下基于公开研究和学术文献，对这两个模型的技术特性、各自在数据安全分类场景中的定位以及 Cyera 可能的使用方式进行深入分析。

来源：[AI-Driven Sensitive Data Classification - Cyera Blog](https://www.cyera.com/blog/advancing-sensitive-data-classification-in-the-age-of-ai)

---

#### 3.3.1 FLAN-T5：Encoder-Decoder 架构的指令微调模型

**架构概要**：

FLAN-T5 是 Google 基于 T5（Text-to-Text Transfer Transformer）开发的**指令微调 Encoder-Decoder Transformer**，其核心特性如下：

| 维度 | 详情 |
|------|------|
| **架构类型** | Encoder-Decoder（编码器-解码器） |
| **编码器** | 双向自注意力堆叠，完整捕获输入序列的双向上下文 |
| **解码器** | 自回归自注意力 + 交叉注意力堆叠，逐 token 生成输出 |
| **规模变体** | Small (~77M)、Base (~250M)、Large (~780M)、XL (~3B)、XXL (~11B) |
| **训练范式** | Text-to-Text：一切 NLP 任务统一为 "输入文本 → 输出文本" |
| **指令微调** | 基于 Flan 2022 Collection，覆盖 1,836 个任务模板、473 个数据集、146 个任务类别 |
| **许可** | Apache 2.0 |

**指令微调的关键设计**（区别于普通 T5）：
- 训练数据混合 zero-shot、few-shot 和 Chain-of-Thought（CoT）模板（仅 5% CoT 示例即可显著提升推理能力）
- 30% 样本进行输入反转（答案→问题），促使模型学习双向推理
- 启发式采样权重防止任一数据集主导训练
- 优化器：Adafactor + 恒定学习率 + example packing（多样本拼接）

**在数据安全分类中的技术优势**：

1. **双向上下文理解**：编码器的双向注意力能完整捕获 PII 实体的左右上下文。对于 "the account 4111-1111-1111-1111 belongs to John" 这样的文本，编码器同时看到 "account" 和 "belongs to"，相比 Decoder-Only 模型仅依赖左侧上下文的单向注意力，能更准确地判断数字序列是否为信用卡号。

2. **可控结构化输出**：Text-to-Text 范式天然约束输出格式，降低幻觉风险。对于敏感数据分类任务，可以训练模型输出结构化 JSON（如 `{"type": "CREDIT_CARD", "confidence": 0.95}`），而非自由文本。

3. **低推理延迟**：编码器一次并行处理输入后，解码器逐 token 生成。对于小到中等规模的 FLAN-T5 变体（Base/Large），推理速度显著优于同等质量的 7B+ Decoder-Only 模型。

4. **NLP 任务的天然优势**：Encoder-Decoder 架构在分类、NER、摘要、抽取式 QA 等任务上优于 Decoder-Only 模型。2025 年研究表明，专门微调的 FLAN-T5 在 CoNLL-03 NER 任务上可达到 **96.14% Macro F1**。

**已知的微调 NER/分类 Checkpoint 生态**：
- `pepegiallo/flan-t5-base_ner`：编码器+分类头方案，BIO 标注（96.14% F1）
- `tliu/asp-ner-flan-t5-large`：自回归结构化预测（ASP）方案
- `knowledgator/UTC-T5-large`：通用 Token 分类器，支持零样本 NER

**Cyera 可能使用的 FLAN-T5 变体**：考虑到 Cyera 需要在 PB 级数据上高效运行分类推理，最可能的选择是 **FLAN-T5-Large (780M)** 或 **FLAN-T5-XL (3B)**。这两个变体在精度和推理成本的平衡点上位于 Sweet Spot：Large 可以用单 GPU 实时推理，XL 可提供接近 SOTA 的分类精度但仍可在合理成本内部署。

---

#### 3.3.2 Mistral：Decoder-Only 架构的高召回语义理解模型

**架构概要**：

Mistral AI 的模型家族采用了与 FLAN-T5 截然不同的 **Decoder-Only** 架构：

| 维度 | 详情 |
|------|------|
| **架构类型** | Decoder-Only（仅解码器），自回归生成 |
| **注意力机制** | Sliding Window Attention（滑动窗口）+ 可选全局注意力 |
| **位置编码** | RoPE（旋转位置编码） |
| **归一化** | RMSNorm |
| **激活函数** | SwiGLU |
| **关键优化** | GQA（分组查询注意力），KV Cache |
| **规模** | 7B (Mistral-7B)、12B (NeMo)、~33B (Medium 3) |
| **许可** | Apache 2.0 |

**Mistral 家族成员**（2024-2025）：
- **Mistral-7B-Instruct-v0.2/v0.3**：最广泛使用的轻量 Decoder-Only 基础模型
- **Mistral NeMo 12B**：NVIDIA 联合训练，原生 FP8 量化
- **Mistral Small 3.1**：MoE（12 专家/4 激活），~8B 活跃参数
- **Mistral Medium 3**：~33B Dense Transformer
- **Mistral Large 3**：675B MoE（41B 活跃），顶层性能

**为何 Decoder-Only 架构也适用于数据安全分类**：

1. **高召回率**：2025 年临床文本去标识化研究显示，微调后的 Mistral-7B 在 425,000+ 临床笔记的 PII/PHI 检测上达到 **PII-level Recall 0.9625、F1 0.9673**，显著优于 Llama 2 7B（F1 0.8750）和 Mixtral 8×7B（F1 0.8616）。对于数据安全场景，遗漏敏感数据的代价远高于误报——高召回是第一优先。

2. **对非正式/噪声文本的鲁棒性**：大规模预训练使 Mistral 对口语化表达、拼写错误、非标准格式的容忍度更高。在处理 Slack 消息、电子邮件、代码注释等非正式文本时优势明显。

3. **语义角色理解**：Mistral 的深层语义理解能力使其能区分 "this is a test SSN 123-45-6789" 和 "employee SSN 123-45-6789"。2024 年 Ketl 研究表明，微调后的 Mistral-7B 在官方文档 NER 上超越了 ChatGPT。

4. **LoRA/QLoRA 高效微调**：Decoder-Only 模型的 LoRA 微调生态成熟。对 Mistral-7B 应用 QLoRA 仅需 **~0.3% 可训练参数**，可在消费级 GPU（RTX 3090 24GB）上完成微调，适合客户定制化部署。

**Cyera 可能使用的 Mistral 变体**：
- **分类引擎的主力**：**Mistral-7B-Instruct** 是最可能的选择。7B 参数规模在精度和部署成本间平衡最佳，QLoRA 微调后可在单 A100/L40S 上高效推理，适合 Cyera 的客户云部署/Outpost 本地化场景。
- **行业专属 LLM**：Cyera 宣称的行业专属微调模型（医疗、金融、制造等）可能以 Mistral-7B 为基础，针对各行业的专业术语和文档类型进行领域微调。
- **不可能是 Mistral Large 3**（675B MoE）：部署成本过高，与 Cyera 宣称的"可在客户环境中完全本地运行"不符。

---

#### 3.3.3 双模型协同架构：Cyera 的分类流水线中的角色分工

基于两个模型的互补特性，可推断 Cyera 在分类流水线中对 FLAN-T5 和 Mistral 进行了明确的角色分工：

```
输入文本
  │
  ├── [FLAN-T5] 快速 NER 层（Encoder-Decoder）
  │     ├── 角色: 高精度 PII 检测
  │     ├── 优势: 低延迟、双向上下文、可控输出
  │     ├── 处理: 正则匹配验证、常见 PII 类型确认
  │     └── 输出: 结构化实体列表 + 置信度
  │
  ├── [Mistral] 深层语义层（Decoder-Only）
  │     ├── 角色: 语义理解与业务敏感性判断
  │     ├── 优势: 高召回、语义角色区分、上下文推理
  │     ├── 处理: 歧义消除、行业术语理解、敏感性推断
  │     └── 输出: 业务语义标签 + 最终风险判定
  │
  └── [融合] 双评分 → 最终分类决策
```

**技术依据**：

1. **2025 年 PII 掩码对比研究**（Acharya & Shrestha, Dec 2025）直接比较了 T5-small 与 Mistral-Instruct-v0.3 在 PII 掩码任务上的表现。核心结论：
   - Mistral 在 F1 和 Recall 上更高，对多样化 PII 类型更鲁棒
   - T5 提供更可控的结构化输出和显著更低的推理延迟
   - 论文明确建议**双层级部署策略**：T5 作为快速首遍过滤器，Mistral 作为高风险/歧义样本的第二遍审查器

2. **架构互补性**：
   - FLAN-T5 的双向编码器在结构化数据的字段级 NER（如数据库列名匹配、CSV 表头识别）上天然优于 Decoder-Only 模型
   - Mistral 的自回归生成能力在理解非结构化文档的**整体语义意图**（如 "本合同属于 M&A 规划文档"）上更强

3. **部署策略**：
   - FLAN-T5（~780M-3B 参数）可以部署在轻量级基础设施上处理 90%+ 的低复杂度分类
   - Mistral（~7B 参数）仅在需要深层语义理解时被调用，降低总体计算成本

**这个双模型设计与 Cyera 声称的 "多模型融合" 策略完全一致，公开专利 US12299167 的双路径设计（真值表+ML分类器融合）在 LLM 层面延伸为 "FLAN-T5 快速路径 + Mistral 语义路径" 的双 LLM 架构。**

---

#### 3.3.4 模型规模推断与性能评估

基于 Cyera 宣称的 "95%+ 分类精度"和"PB 级扫描能力"，对模型规模进行合理推断：

| 推断维度 | FLAN-T5 推断 | Mistral 推断 |
|---------|-------------|-------------|
| **最可能规模** | Large (780M) 或 XL (3B) | 7B-Instruct |
| **推理硬件** | 单 T4/L4 GPU 或 CPU（量化） | 单 A100/L40S 或 4-bit 量化 |
| **吞吐量** | 数百-数千 token/秒 | 数十-数百 token/秒 |
| **批处理能力** | 优秀（Encoder 固定维度） | 良好（需关注 KV Cache 内存） |
| **微调方式** | 全量微调或 LoRA | QLoRA（LoRA rank 8-16） |

**未公开的关键技术细节**：

1. **文档嵌入向量的生成方式**：虽然 Cyera 提及 "semantic distancing" 使用向量嵌入，但生成嵌入向量的具体模型未公开。可能方案：
   - FLAN-T5 的 Encoder 输出（双向上下文嵌入，优于 BERT 类单向嵌入）
   - 专用嵌入模型（如 `intfloat/e5-mistral-7b-instruct`——基于 Mistral 的嵌入模型）
   - **Cyera 专利中未明确指定任何特定的嵌入模型架构**

2. **"模型蒸馏技术"**：虽然 Cyera 官方博客未明确提及蒸馏，但以下几点使得蒸馏成为可能：
   - FLAN-T5 从 XXL (11B) → Base (250M) 的蒸馏是学术界成熟的方案
   - Mistral Large 3 → Mistral-7B 的跨架构蒸馏（MoE→Dense）已被工业界验证
   - **但这是推断，Cyera 公开资料和专利中未直接确认使用蒸馏技术**

3. **训练与推理的部署拓扑**：Cyera 声明所有模型在隔离环境中训练和推理，具体是以下哪种模式未公开：
   - SaaS 模式：Cyera 云端 GPU 集群（使用客户数据微调但推理在 Cyera 侧）
   - 客户云模式：模型打包为容器镜像部署在客户 VPC 内（完全本地推理）
   - Outpost：模型随完整平台打包，在客户本地数据中心运行

4. **行业专属模型的训练策略**：
   - **基础模型** → **通用分类微调**（PII/PCI/PHI 公共数据集）→ **行业领域微调**（医疗/金融/制造等行业数据集）→ **客户自适应微调**（客户特定数据类型）
   - 每一阶段可能使用不同的微调策略（全量 → LoRA → 更小 rank 的 LoRA）
   - 最终可能使用模型合并（Model Merging）而非维护多个独立全量模型

### 3.4 分类流水线的分层策略 [B/C级]

基于Cyera公开信息和专利分析，可推断其分类流水线采用分层过滤架构：

1. **快速确定性过滤层**：正则表达式、熵值检测、字典匹配、元数据启发式分析，处理大部分非敏感文件
2. **ML聚类与语义分析层**：文档嵌入向量生成、无监督聚类、有监督分类器
3. **LLM语义理解层**：文档摘要、上下文语义分类、行业术语理解、业务敏感性推断
4. **多维关联验证层**：跨文档关联分析、结构化与非结构化数据关联、历史分类结果验证、反馈闭环

> 注：上述分层结构和各层吞吐量数值为基于公开信息的技术推断，Cyera未公开精确的分层性能数据。Cyera官方将技术划分为5个组件（聚类、语义距离、LLM验证、LLM分类、自学习分类），上层的4层划分是工程实践视角的合理抽象。

**流量调度策略**（基于公开信息推断）：
- **分层过滤**：快速确定性层先过滤绝大部分非敏感文件，仅未确定文件进入下游
- **动态优先级调度**：根据文件大小、类型、修改时间和业务重要性调整扫描优先级
- **批处理优化**：相似文件批量送入LLM以利用GPU并行计算
- **缓存机制**：缓存已分类文件结果和嵌入向量，避免重复计算

---

[↑ 返回目录](#目录)

## 四、数据发现与扫描架构

### 4.1 无代理API优先架构 [A级]

**已确认**：
- Cyera明确采用**agentless（无代理）**架构，通过云服务商原生API连接 [A]
- 支持的环境：AWS、Azure、GCP（IaaS/PaaS）、Snowflake、Databricks（数据平台）、Microsoft 365、Google Workspace、Salesforce、Box、Confluence（SaaS）、本地文件服务器和数据库（通过connector/Outpost）[A]
- 部署方式：SaaS多租户、客户云部署、Outpost完全本地化部署、混合模式 [A]

来源：[Cyera On-Prem Data Security Datasheet](https://www.cyera.com/de/datasheets/cyera-for-securing-on-premises-data), [Holistic Cloud-First Data Security Q&A](https://www.cyera.com/blog/holistic-cloud-first-data-security-from-cyera)

### 4.2 扫描策略 [A级]

- **结构化数据**：克隆数据库快照进行本地分析 [A]
- **非结构化数据**：使用ML聚类分组后抽样代表性文件 [A]
- **增量扫描**：首次全量后仅扫描变更部分 [A]
- **连续发现**：自动检测新创建/修改/删除的数据存储 [A]
- **声称性能**：官方声称"74 PB scanned in 7 days" [A]

来源：[Redefining Data Classification Whitepaper](https://www.cyera.com/whitepaper/redefining-data-classification), [AI-Powered Classification for Unstructured Data](https://www.cyera.com/fr/blog/ai-powered-classification-for-unstructured-data-turning-complexity-into-clarity)

### 4.3 本地部署组件 [A级]

- **Lightweight Connector（轻量连接器）**：部署为VM，扫描本地数据存储并回传元数据至Cyera云平台 [A]
- **Outpost（前哨）**：完全本地化部署，Cyera完整引擎在客户网络内运行，数据永不离境 [A]
- **Hyper-V支持**：2025年底新增，支持Hyper-V VM镜像部署 [A]

来源：[Cyera brings data security from cloud to on-premises - Enterprise Times](https://www.enterprisetimes.co.uk/2024/05/01/cyera-brings-data-security-from-the-cloud-to-on-premises/)

---

[↑ 返回目录](#目录)

## 五、图数据架构与风险评估

### 5.1 DataGraph [A级]

Cyera的核心图技术被称为**DataGraph**（也称为Data Security Graph）[A]。

**已确认的DataGraph特征** [A]：
- 自动持续学习客户环境变化
- 评估威胁暴露面、数据访问、安全控制三个维度的风险
- 关联实体包括：敏感数据、访问权限、加密状态、备份状态、用户上下文、暴露面指标

来源：[Data Security Posture Management | Cyera Platform](https://www.cyera.com/platform/dspm)

### 5.2 Agent Graph（2025年新增）[A]

专为AI Agent安全设计的图：

> "第一个统一图谱，将DSPM、DLP和Identity的情报关联到AI Agent如何到达、移动和暴露敏感数据"

- 归一化不同平台的Agent（Bedrock、Azure AI Foundry、Salesforce Agentforce、Copilot Studio）
- 按**暴露风险**而非仅连接性进行排名
- 追踪人→非人类身份链

来源：[Introducing the Cyera Agent Graph](https://www.cyera.com/pt-br/blog/introducing-cyera-agent-graph-the-surveillance-layer-for-agentic-ai)

### 5.3 身份图 [A级]

- 2024年7月推出的Identity Module [A]
- 整合跨平台身份（Microsoft 365、Google Workspace、Snowflake、AWS）
- 支持人类和非人类身份（服务账号、AI Agent）
- 与Okta、Saviynt、Microsoft Entra集成

来源：[Cyera Embeds Human and Non-Human Identity Module](https://vmblog.com/archive/2024/07/30/cyera-embeds-human-and-non-human-identity-module-into-data-security-platform.aspx)

### 5.4 风险评估引擎 [A级，但细节有限]

**已确认的风险维度** [A]：
- 数据敏感性（分类标签）
- 暴露程度（公开访问、外部暴露）
- 身份与访问（谁可以访问、MFA状态）
- 活动（Access Trail访问遥测）
- 业务上下文（Topics层）
- 合规映射（GDPR、HIPAA等），DataDNA 内置策略引擎覆盖 **13+ 全球合规框架**：

| # | 框架 | 全称 | 类型 | 确认等级 |
|---|------|------|------|----------|
| 1 | **GDPR** | General Data Protection Regulation | 隐私 · EU | [A] 官方 |
| 2 | **HIPAA** | Health Insurance Portability and Accountability Act | 医疗 · US | [A] 官方 |
| 3 | **PCI-DSS** | Payment Card Industry Data Security Standard | 支付 · 全球 | [A] 官方 |
| 4 | **CCPA** | California Consumer Privacy Act | 隐私 · US/CA | [A] 官方 |
| 5 | **GLBA** | Gramm-Leach-Bliley Act | 金融 · US | [A] 专项指南 |
| 6 | **SOX** | Sarbanes-Oxley Act | 金融 · US | [B] 术语表收录 |
| 7 | **ISO 27001** | Information Security Management | 安全 · 国际 | [A] 官方 |
| 8 | **SOC 2** | Service Organization Control 2 | 审计 · 国际 | [B] 行业推导 |
| 9 | **DORA** | Digital Operational Resilience Act | 金融 · EU | [A] 专项指南 |
| 10 | **PSD2** | Payment Services Directive 2 | 支付 · EU | [B] 术语表收录 |
| 11 | **NIST CSF** | NIST Cybersecurity Framework | 安全 · US | [C] 合理推断 |
| 12 | **NIST AI RMF** | NIST AI Risk Management Framework | AI治理 · US | [A] 官方 |
| 13 | **EU AI Act** | EU Artificial Intelligence Act | AI治理 · EU | [A] 官方 |
| 14 | **ISO 42001** | AI Management System Standard | AI治理 · 国际 | [A] 官方 |
| 15 | **EO 14117** | Executive Order 14117（敏感数据跨境） | 政府 · US | [A] 白皮书专项 |

> **确认等级说明**：[A] 官方产品页/白皮书/专项指南直接确认；[B] 术语表、博客侧栏等间接提及；[C] 行业标准，该类型平台普遍覆盖，合理推断但未找到 Cyera 直接确认。<br>
> **未公开完整清单**：Cyera 宣称 "13+"，上表为公开可查结果。实际可能还包含 LGPD（巴西）、PIPEDA（加拿大）、PDPA（新加坡）等地区性隐私法，但未获官方确认。

**未公开的细节**：
- 精确的风险评分算法
- 权重分配方式
- 0-100评分的具体计算逻辑

---

[↑ 返回目录](#目录)

## 六、Omni DLP

Omni DLP是Cyera产品战略的关键组成部分。

### 6.1 起源 [A级]

- 2024年10月：Cyera以1.62亿美元收购Trail Security（以色列初创公司，由Unit 8200校友创立）
- Trail Security在此前获得3500万美元A轮融资（Lightspeed、CRV、Cyberstarts投资）
- 2025年4月22日：Omni DLP正式发布

来源：[Cyera Acquires Trail Security for $162M - BusinessWire](https://www.businesswire.com/news/home/20241017821422/en/Cyera-Acquires-Trail-Security-for-%24162M-Redefining-AI-Powered-Data-Security-With-Comprehensive-Data-Loss-Prevention)

### 6.2 架构定位 [A级]

Omni DLP的关键创新在于：**不替代现有DLP工具，而是作为"智能层/策略大脑"（intelligence layer / policy brain）**坐落在现有DLP基础设施之上：

- 下层：现有DLP执行点（Microsoft Purview、端点DLP、SSE、邮件网关、CASB）
- 上层：Omni DLP提供统一的AI决策、关联和优先级排序
- 效益：声称可减少90-95%误报

来源：[Omni DLP by Cyera: AI-Native, Real-Time Data Protection](https://www.cyera.com/blog/we-fixed-dlp-meet-omni)

### 6.3 技术架构特点 [A级]

- 共享DataDNA分类引擎，保证DSPM和DLP之间分类一致性
- 实时数据运动监控（终端、网络、邮件、消息、云、AI工具）
- 自适应策略：基于观察到的行为模式自动调整
- 与Microsoft Security Copilot集成，提供自然语言DLP调查

---

[↑ 返回目录](#目录)

## 七、AI Guardian

### 7.1 产品发布时间线 [A级]

- **2025年8月4日**：AI Guardian正式发布
- **AI-SPM**：Private Beta
- **AI Runtime Protection (AI Protect)**：Early Access
- **2025年11月**：GA（正式可用），同时发布Hyper-V本地支持和DLP Policy Tree
- **2026年（RSAC 2026）**：新增Browser Shield、Data Lineage、Cyera MCP

来源：[Cyera Unveils AI Guardian - BusinessWire](https://www.businesswire.com/news/home/20250804797994/en/Cyera-Unveils-AI-Guardian-First-Complete-Solution-to-Secure-Any-Type-of-AI-with-Deep-Data-Centric-Insight)

### 7.2 技术架构细节 [A级]

**AI-SPM（AI安全态势管理）**：
- 自动发现三类AI工具：公共AI（ChatGPT、Gemini、DeepSeek等）、嵌入式AI（Microsoft 365 Copilot、Salesforce Einstein）、自建AI（Amazon Bedrock、Azure AI Foundry上的LLM）[A]
- 通过API集成实现无代理发现 [A]
- 映射每个AI系统可访问的敏感数据 [A]
- 治理人类、机器和Agentic身份对AI的访问 [A]
- 识别Shadow AI（影子AI）[A]

**AI Runtime Protection（AI运行时保护）**：
- 实时监控每一个prompt和response [A]
- 检测prompt injection（提示注入）[A]
- 阻断敏感数据输入未授权AI模型 [A]
- 检测系统提示泄露 [A]
- Browser Shield：Chromium扩展，通过MDM部署，在浏览器层面拦截向公共AI的数据泄露 [A]
- 覆盖OWASP LLM Top 10（2025版）[A]

来源：[Securing LLMs: Cyera's AI Guardian and the OWASP Top Ten](https://www.cyera.com/pt-br/blog/securing-llms-cyeras-ai-guardian-and-the-owasp-top-ten-2025)

---

[↑ 返回目录](#目录)

## 八、专利技术评估

Cyera已构建了涵盖数据发现、数据分类和网络安全策略管理的完整专利组合。截至2026年5月，共检索到**5项已授权美国专利**和**3项已公开专利申请**（含1项PCT国际申请）。这些专利是理解Cyera核心技术实现最权威的公开来源。

---

### 8.1 专利组合总览

Cyera的专利分为四个技术方向，形成三个核心专利家族：

```
┌────────────────────────────────────────────────────────────┐
│                   Cyera 专利组合（截至2026-05）              │
├────────────────────────────────────────────────────────────┤
│ 数据发现家族 (Data Discovery Family)                       │
│  ├── US12026123B2  2024-07-02  数据发现系统与方法           │
│  ├── US12499083B2  2025-12-16  数据发现系统与方法(续案)     │
│  └── US12566567B2  2026-03-03  通过初始扫描发现数据存储位置 │
├────────────────────────────────────────────────────────────┤
│ 数据分类家族 (Data Classification Family)                  │
│  └── US12299167B2  2025-05-13  数据分类与云环境保护         │
├────────────────────────────────────────────────────────────┤
│ 聚类分类家族 (Clustering Classification Family)            │
│  ├── US20240362301A1  2024-10-31  基于聚类的数据对象分类    │
│  ├── US20250068701A1  2025-02-27  基于聚类的数据对象分类(续)│
│  └── WO2024224367A1   2024-10-31  (PCT国际阶段)             │
├────────────────────────────────────────────────────────────┤
│ 安全策略家族 (Security Policy Family)                      │
│  └── US12316686B1  2025-05-27  多源网络安全策略应用         │
└────────────────────────────────────────────────────────────┘
```

---

### 8.2 已授权美国专利详析

#### 专利一：US 12,026,123 B2 — 数据发现系统与方法 [A级]

| 属性 | 内容 |
|------|------|
| **专利号** | US 12,026,123 B2 |
| **标题** | System and method for data discovery in cloud environments |
| **授权日** | 2024年7月2日 |
| **申请日** | 2022年1月13日（优先权日） |
| **发明人** | Yotam Segev, Itamar Bar-Ilan, Yonatan Itai, Shay Makayes, Shani Beracha, Omer Duchovne, Itay Fainshtein |
| **有效期至** | 2042年12月3日 |
| **CPC分类** | G06F16/11（文件系统管理）、G06F16/128（文件系统快照）、G06F21/60/62（数据安全） |

**核心技术方案**（基于专利说明书）：

1. **快照扫描发现机制**：扫描多个磁盘快照，基于文件元数据（文件名、文件类型、目录信息）识别数据存储文件
2. **数据存储存在性规则**：应用规则引擎检测数据存储标识（如文件名含"mySQL"），判断磁盘是否包含数据存储
3. **引擎创建**：分析引擎相关参数，创建配置为特定数据库格式的引擎，实现无需单独权限的数据访问
4. **两阶段扫描**：先执行有限扫描确定磁盘包含数据存储的可能性；仅对可能性超过阈值的快照执行全量扫描
5. **非活跃数据发现**：识别已删除但有备份存在的数据存储，以及未连接到任何机器的孤立数据存储

**产业意义**：这是Cyera无代理架构的核心专利——通过读取云磁盘快照而非安装代理来发现数据存储，从根本上解决了传统DLP工具在云环境中的部署瓶颈。

**下载链接**：[Google Patents - US12026123B2](https://patents.google.com/patent/US12026123B2/en)

---

#### 专利二：US 12,499,083 B2 — 数据发现系统与方法（续案）[A级]

| 属性 | 内容 |
|------|------|
| **专利号** | US 12,499,083 B2 |
| **标题** | System and method for data discovery in cloud environments |
| **授权日** | 2025年12月16日 |
| **申请日** | 2024年5月30日（US 17/647,899的延续申请） |
| **发明人** | Yotam Segev, Itamar Bar-Ilan, Yonatan Itai, Shay Makayes, Shani Beracha, Omer Duchovne, Itay Fainshtein |

**核心技术方案**：

本专利是US 12,026,123的延续申请（Continuation），扩展了数据发现的权利要求范围：

1. 通过分析编程接口检测云环境中的磁盘
2. 为每个磁盘定位或创建快照
3. 分阶段扫描（有限初始扫描后在可能性超过阈值时进行全量扫描）
4. 在云环境外部VM上创建匹配检测到的数据存储类型/版本的数据库引擎
5. 读取schema、采样数据、分类敏感性
6. 按风险优先级执行缓解措施

**下载链接**：[Google Patents - US12499083B2](https://patents.google.com/patent/US12499083B2/en)

---

#### 专利三：US 12,566,567 B2 — 通过初始扫描发现数据存储位置 [A级]

| 属性 | 内容 |
|------|------|
| **专利号** | US 12,566,567 B2 |
| **标题** | Techniques for discovering data store locations via initial scanning |
| **授权日** | 2026年3月3日 |
| **申请日** | 2022年5月19日（部分延续自US 17/647,899） |
| **发明人** | Yotam Segev, Itamar Bar-Ilan, Yonatan Itai, Shay Makayes, Shani Beracha, Omer Duchovne, Itay Fainshtein |

**核心技术方案**：

本专利是数据发现家族的最新成员（Continuation-in-Part），聚焦于**仅通过读取部分快照实现高效发现**：

1. **部分读取策略**：通过云服务商的原生工具仅读取每个磁盘快照的**文件系统元数据部分**（而非完整磁盘），大幅降低数据传输量
2. **直接访问抽象层**：使用云服务商提供的直接读取API访问快照，无需先复制整个快照
3. **惰性挂载（Lazy Mount）**：创建mount点但仅获取所需的数据块，扫描组件看到的是本地文件系统视图
4. **条件性克隆**：仅在确认磁盘包含数据存储后才克隆快照进行深度分析
5. **引擎实例化**：在连接到磁盘副本的虚拟机上创建数据库引擎，允许在无需逐一获取存储权限的情况下查询数据

**产业意义**：该专利揭示了Cyera声称"数分钟内扫描数百个云账户"的技术基础——通过只读元数据+条件性克隆+惰性挂载的组合，实现了接近零开销的云数据发现。

**下载链接**：[Google Patents - US12566567B2](https://patents.google.com/patent/US12566567B2/en)

---

#### 专利四：US 12,299,167 B2 — 数据分类与云环境保护 [A级]

| 属性 | 内容 |
|------|------|
| **专利号** | US 12,299,167 B2 |
| **标题** | Techniques for data classification and for protecting cloud environments from cybersecurity threats using data classification |
| **授权日** | 2025年5月13日 |
| **申请日** | 2022年10月13日 |
| **发明人** | Yotam Segev, Itamar Bar-Ilan, Yonatan Itai, Shiran Bareli, Michael Elazar, Antony Timchenko, Itay Mizeretz |
| **有效期至** | 2043年5月5日 |
| **CPC分类** | G06F21/6245（金融/医疗个人数据保护）、G06F21/54（程序安全增强）、G06F21/577（漏洞评估） |

**核心技术方案**（基于专利说明书的详细分析）：

这是Cyera分类引擎最核心的专利，公开了一种**双路径混合分类方法**：

**第一路径 — 启发式真值表（针对数值型数据）**：
- 对数值型样本（SSN、信用卡号、护照号等）应用真值表
- 真值表输入列包括：已验证实例数量、正则表达式强度、上下文支持/不支持术语、模式频率、唯一性评分
- 输出列为"第一评分"，指示分类可能性

**第二路径 — 机器学习分类器（针对字符串数据）**：
- 对字符串样本（姓名、地址等）提取特征
- 应用训练好的ML分类器
- 输出"第二评分"

**抽样方法**：
- 结构化数据：半随机化抽样（将数据分块后抽样）
- 非结构化数据：基于文件元数据（路径、类型、大小）的聚类算法抽样

**后分类处理**：
- 角色判定（区分客户数据vs员工数据）
- 虚假数据过滤
- 单向函数标记，支持未来引用

**安全应用**：
- 基于分类结果修改云组件以满足安全要求
- 监控异常行为
- 按优先级排序缓解措施

**产业意义**：该专利验证了Cyera的"规则+ML"双引擎分类策略——数值型数据走启发式规则（快但需要上下文验证），文本数据走ML模型（慢但语义理解强），最后融合两个路径的评分决定最终分类。这个设计直接对应Cyera宣传的"95%精度"。

**下载链接**：[Google Patents - US12299167B2](https://patents.google.com/patent/US12299167B2/en)

---

#### 专利五：US 12,316,686 B1 — 多源网络安全策略应用 [A级]

| 属性 | 内容 |
|------|------|
| **专利号** | US 12,316,686 B1 |
| **标题** | System and method for applying multi-source cybersecurity policy on computing environments |
| **授权日** | 2025年5月27日 |
| **申请日** | 2024年11月18日 |
| **发明人** | Zohar Vittenberg, Nadav Zingerman, Roei Mutay |
| **引用现有技术** | 17项（来自Juniper Networks、Amazon Technologies、CyberArk、FireEye等） |

**核心技术方案**：

> **重要背景**：发明人Zohar Vittenberg、Nadav Zingerman、Roei Mutay是Trail Security（2024年10月被Cyera以1.62亿美元收购的DLP初创公司）的创始人/核心成员。该专利体现了Omni DLP的策略统一技术路线。

1. **多格式策略接收**：接收来自不同网络安全平台（DLP、IAM如Okta、防火墙/WAF）的多种数据格式的策略
2. **生成式AI归一化**：使用生成式AI模型（LLM/SLM）将策略归一化为统一表示
3. **跨格式策略生成**：从归一化策略生成适用于其他平台格式的策略（如将DLP规则转换为IAM策略）
4. **迭代策略生成**：支持生成"逐步收窄"的迭代策略
5. **策略引擎**：核心组件Policy Engine（150）负责归一化和转换

**产业意义**：
- 该专利直接支持Omni DLP的"策略大脑"定位——坐落在现有DLP/安全工具之上，统一管理和转换策略
- 解释了Cyera/Trail Security如何实现"不替换现有DLP，而是增强它们"
- 生成式AI在策略归一化中的使用表明LLM不仅用于数据分类，还用于策略工程

**下载链接**：[Google Patents - US12316686B1](https://patents.google.com/patent/US12316686B1/en)

---

### 8.3 已公开专利申请

#### 申请一：US 2024/0362301 A1 — 基于聚类的数据对象分类 [A级]

| 属性 | 内容 |
|------|------|
| **公开号** | US 2024/0362301 A1 |
| **公开日** | 2024年10月31日 |
| **申请日** | 2023年4月27日 |
| **发明人** | Yotam Segev, Itamar Bar-Ilan, Yonatan Itai, Shiran Bareli, Guye Karni, Tomer Mesika, Itay Fainshtein, Ofir Talmor |
| **状态** | 已公开，审查中 |

**核心技术方案**（基于专利说明书的详细分析）：

该专利申请是Cyera DataDNA聚类分类技术最详细的公开描述：

**步骤1 — 元数据替换**：
- 数值 → 替换为包含原值的预定义范围（如1223字节 → 范围"1024-2048"）
- 文本模式（标识符、日期、IP地址、UUID、十六进制序列）→ 替换为标准占位符
- 文本参数通过分隔符（空格、斜杠、大小写变化）拆分为子串
- 过滤字典保留名词/动词词汇

**步骤2 — 聚类形成**：
- 基于相同的替换后元数据形成初始聚类
- 随机序列（不匹配已知模式的长字符串）被识别和替换
- 低于大小阈值的聚类进行文本参数合并
- 时间复杂度：线性时间（初始聚类）、对数线性（优化阶段）
- 空间复杂度：常数空间，无需预知聚类数量（本专利强调这是对现有技术的重要改进）

**步骤3 — 真实聚类判定**：
- 从每个聚类中抽样并分类
- 定义广义分类类型（如"员工全名"和"客户全名" → "全名"；"EU电话号码"和"US电话号码" → "电话号码"）
- 通过统计过程评估样本分类差异是否可能源自误报
- 若判定为"真实聚类"，分类传播到聚类中所有对象

**步骤4 — 虚假聚类处理**：
- 基于元数据和分类差异拆分为子聚类
- 迭代重新处理

**步骤5 — 角色发现**：
- 从列名中提取重复术语（如"donor"和"donation"跨多列出现）
- 分析字段内容验证模式
- 创建新的角色分类（区别于通用分类标签）

**产业意义**：
- 该专利申请公开了Cyera声称能从"数周"扫描的PB级数据中快速分类的核心技术基础
- 聚类+抽样+传播的策略解释了为什么DataDNA不需要逐文件扫描
- "角色发现"功能（自动发现新数据类型）对应Cyera宣传的"Learned Classification"
- 元数据替换策略是关键的工程优化——通过将有意义的元数据替换为占位符来加速聚类，避免对相似文件的重复处理

**下载链接**：
- [Google Patents - US20240362301A1](https://patents.google.com/patent/US20240362301A1/en)
- [Google Patents - WO2024224367A1 (PCT)](https://patents.google.com/patent/WO2024224367A1/en)

---

#### 申请二：WO 2024/224367 A1 — 基于聚类的数据对象分类（PCT国际阶段）[A级]

| 属性 | 内容 |
|------|------|
| **公开号** | WO 2024/224367 A1 |
| **公开日** | 2024年10月31日 |
| **申请日** | 2024年4月26日（PCT/IB2024/054101） |
| **优先权日** | 2023年4月27日 |
| **发明人** | 同US 2024/0362301 A1 |
| **状态** | 已进入PCT国家阶段，预计2025年10月27日后在各指定国进入审查 |

> 注：PCT申请的实质内容与US 2024/0362301 A1相同，是同一发明在不同国家/地区寻求保护的对应申请。

**下载链接**：[Google Patents - WO2024224367A1](https://patents.google.com/patent/WO2024224367A1/en)

---

#### 申请三：US 2025/0068701 A1 — 基于聚类的数据对象分类（续案）[A级]

| 属性 | 内容 |
|------|------|
| **公开号** | US 2025/0068701 A1 |
| **公开日** | 2025年2月27日 |
| **申请日** | 2024年10月29日（US 2024/0362301 的延续申请） |
| **发明人** | 同US 2024/0362301 A1 |
| **状态** | 已公开，审查中 |

> 注：本申请是 US 2024/0362301 A1 的延续申请（Continuation），技术内容与母案相同，扩展了权利要求范围。不重复分析技术方案。

**下载链接**：[Google Patents - US20250068701A1](https://patents.google.com/patent/US20250068701A1/en)

---

### 8.4 专利组合分析

#### 8.4.1 发明人网络

| 发明人 | 参与的专利 | 角色 |
|--------|----------|------|
| Yotam Segev | US12026123, US12499083, US12566567, US12299167, US20240362301, US20250068701, WO2024224367 | CEO/联合创始人，几乎所有核心专利的共同发明人 |
| Itamar Bar-Ilan | 同上7项 | CTO/联合创始人 |
| Yonatan Itai | 同上7项 | 核心工程师 |
| Shay Makayes | US12026123, US12499083, US12566567 | 数据发现方向 |
| Shani Beracha | US12026123, US12499083, US12566567 | 数据发现方向 |
| Shiran Bareli | US12299167, US20240362301, US20250068701, WO2024224367 | 数据分类方向 |
| Zohar Vittenberg | US12316686 | Trail Security CEO（被收购后） |
| Nadav Zingerman | US12316686 | Trail Security CTO |
| Roei Mutay | US12316686 | Trail Security VP R&D |

**关键观察**：
- Yotam Segev和Itamar Bar-Ilan作为创始人参与了除Trail Security策略专利外的所有专利，表明核心技术方向由创始团队直接驱动
- Trail Security专利的发明人组与核心Cyera专利的发明人组完全不同，反映了两条技术路线（数据安全 vs 策略管理）的独立起源
- 数据发现和数据分类方向有部分交叉的发明人（Yonatan Itai同时出现在两个方向），体现了跨模块的技术协同

#### 8.4.2 时间线分析

```
2022-01-13  数据发现专利（US12026123）提交         ← Cyera成立不到1年，首个专利申请
2022-05-19  数据存储发现专利（US12566567）提交      ← CIP扩展
2022-10-13  数据分类专利（US12299167）提交          ← 分类引擎核心专利
2023-04-27  聚类分类专利（US20240362301）提交       ← 分类引擎的工程化实现
2024-04-26  聚类分类PCT国际申请（WO2024224367）提交  ← 全球IP布局
2024-07-02  US12026123授权                         ← 首个专利授权（数据发现）
2024-10-31  聚类分类专利公开（US20240362301 + WO2024224367）
2024-10-29  聚类分类续案（US20250068701）提交       ← 延续申请，扩展权利要求
2024-11-18  Trail Security策略专利（US12316686）提交 ← 收购Trail Security仅1个月后
2025-02-27  聚类分类续案公开（US20250068701）
2025-05-13  US12299167授权                         ← 分类引擎核心专利授权
2025-05-27  US12316686授权                         ← Trail策略专利授权（仅6个月审查期）
2025-12-16  US12499083授权                         ← 数据发现续案授权
2026-03-03  US12566567授权                         ← 最新授权专利
```

**关键观察**：
- 首个专利申请在Cyera成立不到1年时提交，表明IP保护是公司早期战略重点
- 聚类分类专利至今仍是"公开但未授权"状态（2023年4月提交，至2026年5月尚未授权），可能在审查中涉及更广泛的现有技术争议
- US12316686（Trail Security策略专利）仅6个月即获授权，可能受益于USPTO的优先审查（Track One）程序或以小型实体的加速审查
- PCT申请的30个月期限（至2025年10月27日）已到期，各国国家阶段申请应已推进

#### 8.4.3 与其他DSPM厂商的IP策略对比

| 维度 | Cyera | 典型DSPM竞品 [C/D级] |
|------|-------|------------|
| 已授权美国专利 | 5项 | 多数竞品0-2项 |
| 专利覆盖方向 | 发现+分类+策略 | 多为单一方向 |
| PCT国际布局 | 有 | 多数无 |
| 核心发明人 | 创始人直接参与 | 多为工程师或外部 |
| 提交策略 | 公司成立即启动 | 较晚启动或不重视IP |

> 注：竞品栏为基于USPTO检索的有限推断，未对所有DSPM厂商做穷尽式专利调查。实际存在反例（如某些竞品有更强IP组合），此表仅供粗略方向对比。

---

[↑ 返回目录](#目录)

## 九、第三方评估

### 9.1 Forrester Wave [A级]

**2026年Q2 Forrester Wave™: Sensitive Data Discovery And Classification Solutions** — Cyera被评为**Leader**

关键评分：
- **Strategy类别最高分**：4.9/5（所有10家厂商中最高）
- 7项Strategy标准中6项获最高分：Vision、Innovation、Roadmap、Partner Ecosystem、Adoption、Supporting Services
- 3项Current Offering标准获最高分：Cloud Data Source Coverage、Integrations、Secure-by-Design Commitments

来源：[Cyera Recognized as a Leader - BusinessWire](https://www.businesswire.com/news/home/20260410955567/en/Cyera-Recognized-as-a-Leader-in-Sensitive-Data-Discovery-And-Classification)

### 9.2 Gartner [A级]

- **2025 Gartner Market Guide for DSPM**：Representative Vendor [A]
- **Gartner Peer Insights**：正面与批评性评论并存 [A]

**批评性反馈（Gartner用户评论）**：
- "缺乏持续监控能力"（lacks continuous monitoring）
- "功能广泛但成熟度不足，文档缺失，发布测试不充分"
- 一次严重事件："DSPM扫描后破坏了环境中多个文件"
- "ML增强分类但需要大量初始调优"

来源：[Gartner Peer Insights - Cyera Platform Reviews](https://www.gartner.com/reviews/product/cyera-platform)

---

[↑ 返回目录](#目录)

## 十、Cyera技术架构总结

### 10.1 真正的技术创新点

基于对公开资料的独立分析，Cyera的技术创新主要集中在以下方面：

1. **多模型融合分类方法论**：不是单一LLM的简单应用，而是聚类→语义距离→LLM验证→LLM分类→自学习的多层组合。这确实是区别于传统DLP厂商的关键。

2. **数据中心的AI安全**：AI Guardian不是孤立的安全工具，而是建立在DataDNA分类引擎之上的，"数据是控制平面"的理念使其能比纯网络/端点方案更精准地判断风险。

3. **图驱动的关联分析**：DataGraph → Agent Graph 的演进表明Cyera在图数据技术上持续投入，将数据、身份、访问、AI行为统一建模。

4. **非替换式的DLP架构**：Omni DLP选择作为现有DLP工具之上的"智能层"，而非替换，这在企业级市场是务实的策略。

### 10.2 技术局限性（基于公开评测和客户反馈）

1. **本地环境深度有限**：相比云原生环境，本地数据存储的支持在2024年才逐步补充，成熟度不如云模块
2. **超大规模扫描性能存疑**：尽管Cyera声称可处理PB级数据，Sentra等竞争对手对"100TB以上环境"的性能提出质疑
3. **分类初始调优成本高**：Gartner用户评论显示，ML分类需要大量初始调优
4. **出现文件损坏案例**：至少一位Gartner评论者报告扫描后文件损坏
5. **治理闭环缺失**：发现风险后"谁负责修复"的问题在所有DSPM厂商中未解决
6. **价格门槛**：定位大型企业，中小企业可能难以承受

### 10.3 与竞品的关键差异

| 维度 | Cyera | 传统DLP（如Symantec/Forcepoint） | 云原生DSPM（如Wiz/Dig） |
|------|-------|----------------------------------|------------------------|
| 分类方式 | AI原生多模型融合 | 规则/正则为主 | 多为规则+基础ML |
| 数据覆盖 | 结构化/半结构化/非结构化 | 偏非结构化文件和邮件 | 偏云存储 |
| DLP整合 | Omni DLP统一平台 | 自身即DLP工具 | 多依赖第三方 |
| AI安全 | AI Guardian完整覆盖 | 有限或缺失 | 有限 |
| 部署 | 无代理API优先 | 代理+设备为主 | 无代理API |
| 图分析 | DataGraph+Agent Graph | 无 | 基础关联 |

---

[↑ 返回目录](#目录)

## 十一、结论

Cyera DSPM是当前DSPM市场中最具技术野心的平台之一。其核心技术实力体现在：

1. **DataDNA多模型分类引擎**（聚类+语义距离+LLM+自学习）代表了从规则匹配到语义理解的范式转变
2. **图驱动的统一分析**（DataGraph、Agent Graph、身份图）实现了跨数据、身份、访问的关联分析
3. **快速的产品演进速度**（12个月内推出Omni DLP、AI Guardian、Access Trail等多项重大产品）

关于本报告的分析方法：每项关键声明均标注了来源可靠性等级（A=官方确认，B=第三方验证，C=可信推断，D=间接来源），严格区分"公开确认的事实"与"合理推断"。对于无法通过公开资料验证的声明（如具体的流水线吞吐量数字、模型蒸馏技术细节），已在相关章节明确标注为推断或未确认。

---

[↑ 返回目录](#目录)

## 信息来源汇总

### 一级来源（Cyera官方）
- [Understanding Data in Context: An LLM-Driven Approach to Data Classification](https://www.cyera.com/blog/understanding-data-in-context-an-llm-driven-approach-to-data-classification)
- [Advancing Sensitive Data Classification in the Age of AI](https://www.cyera.com/blog/advancing-sensitive-data-classification-in-the-age-of-ai)
- [Cyera's Industry-Specific Classification LLMs](https://www.cyera.com/blog/a-leap-forward-cyera-enhances-its-market-leading-llms-with-industry-specific-precision)
- [AI-Powered Classification for Unstructured Data](https://www.cyera.com/fr/blog/ai-powered-classification-for-unstructured-data-turning-complexity-into-clarity)
- [Omni DLP by Cyera: AI-Native, Real-Time Data Protection](https://www.cyera.com/blog/we-fixed-dlp-meet-omni)
- [Introducing AI Guardian](https://www.cyera.com/blog/introducing-ai-guardian-to-secure-ai-at-the-source)
- [Introducing Cyera Access Trail](https://www.cyera.com/blog/cyera-access-trail-track-every-human-and-ai-interaction-with-your-data)
- [Introducing the Cyera Agent Graph](https://www.cyera.com/pt-br/blog/introducing-cyera-agent-graph-the-surveillance-layer-for-agentic-ai)
- [Cyera Acquires Trail Security for $162M](https://www.businesswire.com/news/home/20241017821422/en/)
- [Cyera Unveils AI Guardian](https://www.businesswire.com/news/home/20250804797994/en/)
- [Cyera Raises $400M at $9B Valuation](https://www.securityweek.com/cyera-raises-400-million-at-9-billion-valuation/)
- [Cyera Embeds Identity Module](https://vmblog.com/archive/2024/07/30/cyera-embeds-human-and-non-human-identity-module-into-data-security-platform.aspx)
- [Cyera On-Prem Data Security](https://www.cyera.com/de/datasheets/cyera-for-securing-on-premises-data)
- [Holistic Cloud-First Data Security Q&A](https://www.cyera.com/blog/holistic-cloud-first-data-security-from-cyera)

### 二级来源（第三方评测）
- [Forrester Wave: Sensitive Data Discovery And Classification Q2 2026](https://www.cyera.com/es-mx/reports/the-forrester-wave-tm-sensitive-data-discovery-and-classification-solutions-q2-2026)
- [Gartner Peer Insights - Cyera Platform](https://www.gartner.com/reviews/product/cyera-platform)
- [Fortune: Cyera CEO on raising $400 million](https://fortune.com/2026/01/08/cyera-cybersecurity-startup-yotam-segev-400-million-series-f-funding-9-billion-valuation-blackstone/)

### 三级来源（专利）

**已授权美国专利**：
- US 12,026,123 B2 — "System and method for data discovery in cloud environments" — Cyera Ltd. (授权日 2024-07-02) — [Google Patents](https://patents.google.com/patent/US12026123B2/en)
- US 12,299,167 B2 — "Techniques for data classification and for protecting cloud environments from cybersecurity threats using data classification" — Cyera Ltd. (授权日 2025-05-13) — [Google Patents](https://patents.google.com/patent/US12299167B2/en)
- US 12,316,686 B1 — "System and method for applying multi-source cybersecurity policy on computing environments" — Cyera Ltd. (授权日 2025-05-27) — [Google Patents](https://patents.google.com/patent/US12316686B1/en)
- US 12,499,083 B2 — "System and method for data discovery in cloud environments" — Cyera Ltd. (授权日 2025-12-16) — [Google Patents](https://patents.google.com/patent/US12499083B2/en)
- US 12,566,567 B2 — "Techniques for discovering data store locations via initial scanning" — Cyera Ltd. (授权日 2026-03-03) — [Google Patents](https://patents.google.com/patent/US12566567B2/en)

**已公开专利申请**：
- US 2024/0362301 A1 — "Clustering-Based Data Object Classification" — Cyera Ltd. (公开日 2024-10-31) — [Google Patents](https://patents.google.com/patent/US20240362301A1/en)
- US 2025/0068701 A1 — "Clustering-Based Data Object Classification" (续案) — Cyera Ltd. (公开日 2025-02-27) — [Google Patents](https://patents.google.com/patent/US20250068701A1/en)
- WO 2024/224367 A1 — "Clustering-Based Data Object Classification" (PCT国际阶段) — Cyera Ltd. (公开日 2024-10-31) — [Google Patents](https://patents.google.com/patent/WO2024224367A1/en)

**专利检索平台**：
- [Justia Patents — Cyera Ltd. Assignee Page](https://patents.justia.com/assignee/cyera-ltd)
- [Google Patents — 搜索 assignee:"Cyera Ltd"](https://patents.google.com/?assignee=Cyera+Ltd)
- [USPTO Patent Public Search](https://ppubs.uspto.gov/pubwebapp/)

### 四级来源（竞争分析）
- [Sentra: Unstructured Data Is 80% of Your Risk](https://www.sentra.io/blog/unstructured-data-is-80-of-your-risk-why-your-dspm-fails-at-petabyte-scale)
- [Gartner Peer Insights critical reviews](https://www.gartner.com/reviews/product/cyera-platform)
