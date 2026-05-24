# DataDNA 文档智能分类引擎 — 统一设计文档

> **版本**: v2.0  
> **日期**: 2026-05-24  
> **定位**: 企业安全产品 (DSPM/DLP) 的文档智能分类引擎  
> **架构**: 6引擎并行融合 + 加权投票  
> **状态**: Phase 0-4 代码完成，E3/E4 待数据积累激活  

---

## 一、设计原则

| # | 原则 | 含义 |
|:--:|------|------|
| P1 | **带着知识上线** | 出厂内置规则和模板，首日即可分类，无需训练 |
| P2 | **不确定时保守** | 综合置信度 < 0.4 → manual_review，不臆断 |
| P3 | **可观测可审计** | 每个决策可追溯到每个引擎的完整输出 |
| P4 | **稳定优于智能** | 分类行为不自动漂移，变更需证据和运维确认 |
| P5 | **渐进退化** | 任一引擎故障不导致系统不可用，其余引擎自动接管 |

---

## 二、硬性要求 (R1-R7)

| # | 要求 | 量化标准 | 当前状态 |
|:--:|------|---------|:--:|
| R1 | 多场景稳定 | 两场景 Macro F1 差值 < 5% | ⚠️ 冷启动未达 |
| R2 | 准确率 | 所有引擎可用时 Macro F1 > 90% | ⚠️ E3待训练，降级模式下不适用 |
| R3 | 首次部署可用 | 首篇文档即可分类 | ✅ qwen2.5:7b零样本，cxh5types 95% |
| R4 | 成熟期效率 | LLM调用率 < 20%，延迟 < 300ms | ⚠️ 需E3+E4成熟 |
| R5 | 不依赖数据假设 | 任一引擎失效自动重分配权重 | ✅ 8个故障注入测试通过 |
| R6 | 异常检测告警 | 质量下降>5%告警，低置信→review | ✅ 7项监控+告警已实现 |
| R7 | 可证伪 | 每模块有量化失败条件 | ✅ 每个引擎独立可测 |

---

## 三、架构演进历程

### 3.1 第一版：分层递进 (Tier 0→1→2→3，已废弃)

```
文档 → Tier0(PII检测) → Tier1(双阶段聚类) → Tier2(簇分类) → Tier3(LLM质量门) → 标签
```

**废弃原因**: 
- 串行门控的单点故障风险（LLM不可用则系统报废）
- 信号耦合（三个分类信号共享一个PII检测模块）
- 门控阈值无数据支撑（预设0.85无依据）

详见 `2026-05-22-datadna-optimized-design.md`。

### 3.2 第二版：6引擎并行融合 (当前)

```
文档 → [E1 正则, E2 模板, E3 ML, E4 kNN, E5 结构, E6 LLM] 并行
    → 加权投票融合 → 标签 + 综合置信度
```

**核心变化**: 取消串行门控，6引擎独立运行，无单点故障。详见 `2026-05-23-optimal-architecture.md`。

### 3.3 与 Cyera 的关系

Cyera 专利 (US12026123B2等8项) 覆盖的是数据发现和元数据替换方案，本架构在以下方面借鉴但不构成侵权：
- **元数据替换 (E2模板引擎)**: 借鉴思想，实现方式不同 (PII正则 + SHA256)
- **多引擎融合**: Cyera 使用真值表融合，本方案使用加权投票，本质不同
- **LLM使用**: Cyera 专利不含 LLM，本方案的 E6 是独立创新

详见 `2026-05-22-patent-infringement-analysis.md`。

---

## 四、6引擎详细设计

### 4.1 引擎总览

| 引擎 | 类型 | 延迟 | 权重 | 依赖 | 冷启动状态 |
|------|------|:--:|:--:|------|:--:|
| E1 正则规则 | 确定性 | <1ms | 1.0 | 55条规则库 | ✅ 就绪 |
| E2 模板Hash | 确定性 | <1ms | 1.0 | PII检测+模板库 | ✅ 就绪 |
| E3 ML (SetFit) | 统计 | ~2ms | 1.5 | 训练数据≥50/类 | ⚠️ 待训练 |
| E4 语义kNN | 统计 | ~2ms | 1.0 | BGE-M3+质心 | ⚠️ 关键词质心 |
| E5 结构签名 | 确定性 | <1ms | 0.8 | 文件元数据 | ✅ 就绪 |
| E6 LLM | 生成式 | ~1.5s | 2.0 | Ollama服务 | ✅ 就绪 |

**引擎独立性**: 每个引擎有独立的输入来源和依赖。PII检测仅在E2中使用，E1的自包含规则不受影响。E3未训练仅影响E3自身，E4的BGE-M3故障仅影响E4。

### 4.2 引擎接口

```python
class EngineOutput:
    engine_id: str       # "E1_regex", ...
    label: str | None    # None = 无输出
    confidence: float    # 0.0-1.0, 引擎自评置信度
    status: str          # matched | no_match | unavailable | skipped
    metadata: dict       # 引擎特定追踪信息

class BaseEngine(ABC):
    def analyze(self, doc: Document) -> EngineOutput: ...
    weight: float    # 融合权重
    is_available: bool   # 运行时健康检查
```

### 4.3 各引擎置信度定义

- **E1**: `base_confidence + PII boost`（规则匹配分 + 关联PII类型命中加成）
- **E2**: `1.0`（精确Hash匹配）或 `0.5`（前缀匹配）
- **E3**: `SetFit.predict_proba` 最大值
- **E4**: `1 - cosine_distance` 最近质心的余弦相似度
- **E5**: `1.0`（精确签名匹配）或 `0.0`（无匹配）
- **E6**: LLM 输出的 `confidence` 字段

---

## 五、融合机制

### 5.1 加权投票算法

```
score(label) = Σ (engine.weight × confidence × is_available)
final_label  = argmax(score)
composite_confidence = max_score / Σ(所有可用引擎的weight)
```

### 5.2 LLM调用优化

```
1. 运行 E1-E5（总延迟 <5ms）
2. 融合 E1-E5 结果 → preliminary_confidence
3. 若 preliminary_confidence ≥ 0.85 → 跳过 E6 (method="fusion_fast")
4. 否则 → 调用 E6，6引擎完整融合 (method="fusion_full")
5. composite_confidence < 0.4 → manual_review=true
6. 所有引擎均无输出 → "unclassified", manual_review=true
```

### 5.3 降级路径

| 故障引擎 | 影响 | 降级行为 |
|---------|------|---------|
| E1 正则 | 微小 | 其他引擎覆盖类似类型 |
| E2 模板 | 小 | E1/E3/E4/E6可接管 |
| E3 ML | 小-中 | 效率下降，准确率不受影响 |
| E4 kNN | 中 | 覆盖面最广的引擎失效 |
| E5 结构 | 微小 | 权重最低 |
| E6 LLM | 中-大 | E1-E5继续融合，置信度偏低 |
| 全部故障 | 极端 | E1+E5确定性引擎仍可用，其余标记unclassified |

---

## 六、代码结构

```
impl-datadna/
├── src/
│   ├── engines/          # 6引擎 + BaseEngine ABC
│   │   ├── base.py           # 引擎基类
│   │   ├── e1_regex.py       # 正则规则引擎 (55条规则)
│   │   ├── e2_template.py    # 模板Hash引擎
│   │   ├── e3_ml.py          # SetFit ML引擎
│   │   ├── e4_knn.py         # 语义kNN引擎
│   │   ├── e5_structural.py  # 结构签名引擎
│   │   └── e6_llm.py         # LLM分类引擎
│   ├── fusion/           # 加权投票融合
│   │   └── voter.py
│   ├── knowledge/        # 预置知识 (55规则+模板库+13类型)
│   │   ├── rules.py
│   │   ├── templates.py
│   │   └── type_library.py
│   ├── monitoring/       # 审计日志 + 7项监控指标
│   │   ├── audit.py
│   │   └── metrics.py
│   ├── distillation/     # SetFit训练+管理
│   │   ├── trainer.py
│   │   └── manager.py
│   ├── discovery/        # 类型发现循环
│   │   └── loop.py
│   ├── embeddings/       # BGE-M3嵌入服务 (复用)
│   ├── llm/              # LLM客户端 (复用)
│   └── types.py          # 核心数据类型
├── tests/                # 73个测试
├── main.py               # 主入口
├── incremental.py        # 增量处理入口
├── benchmark.py          # 延迟基准测试
├── eval_cross.py         # 跨框架评估
└── config.yaml           # 全局配置
```

---

## 七、测试基准

### 7.1 Cxh5types (258篇企业文档，3类，人工标注)

| 引擎配置 | LLM模型 | Accuracy | Macro F1 | LLM率 |
|---------|---------|:--:|:--:|:--:|
| E1+E2+E5+E6 (E3/E4不可用) | mistral:7b | 52.7% | 0.6057 | 100% |
| E1+E2+E5+E6 (E3/E4不可用) | qwen2.5:7b | 95.0% | 0.9487 | 100% |
| E1-E6全引擎 (E3已训练+E4真实质心) | qwen2.5:7b | 98.0%* | 0.9760* | 100% |

> *E3 SetFit在训练的207条文档上达到MacroF1=0.976，但因GPU显存限制未能部署到评估管道。实际评估的是冷启动基线（E3不可用）。

| 类别 | cold-start F1 | 样本数 |
|------|:--:|:--:|
| Financial & Accounting | 0.9259 | 50 |
| Human Resources & Payroll | 0.9394 | 105 |
| Legal & Compliance | 0.9808 | 103 |

### 7.2 20 Newsgroups (100篇同构文本，20类)

| 引擎配置 | LLM模型 | Accuracy | Macro F1 |
|---------|---------|:--:|:--:|
| E1-E6 (冷启动) | mistral:7b | 29.1% | 0.3381 |
| E1-E6 (冷启动) | qwen2.5:7b | 50.0% | 0.4009 |

### 7.3 降级测试

8个故障注入测试全部通过。逐一关停每个引擎后系统仍可分类，全部引擎故障时返回 `unclassified`。

### 7.4 与 V2.2 对比 (同LLM: qwen2.5:7b)

| 数据集 | V2.2 (聚类+LLM命名) | DataDNA (逐文档分类) | 差异 |
|------|:--:|:--:|:--:|
| Cxh5types | MacroF1=1.0000 | MacroF1=0.9487 | V2.2 +0.05 |
| 20newsgroups(300) | MacroF1=0.1663 | MacroF1=0.4009 | DataDNA +0.23 |

**关键发现**: V2.2在企业文档上略优（聚类天然分离3类），但在高基数同构文本上远不如逐文档分类。两种范式各有所长，不是替代关系。

---

## 八、待解决事项

| 问题 | 优先级 | 方案 |
|------|:--:|------|
| E3 SetFit无法在12GB GPU训练 | 中 | 换小骨架 (all-MiniLM-L6-v2, 80MB) 或停Ollama后训练 |
| R1跨场景稳定性未达标 | 中 | 需要E3+E4成熟以减少对LLM的单一依赖 |
| R4效率 (LLM率<20%) | 中 | E3+E4激活后大部分文档走fusion_fast |
| LLM模型选型 | 低 | 测试 qwen3:8b (已开始) 替代 qwen2.5:7b |

---

## 九、与V2.2方案的本质差异

| 维度 | V2.2 | DataDNA |
|------|------|------|
| 分类范式 | 聚类→簇命名 | 逐文档多引擎融合分类 |
| LLM角色 | 为簇生成标签名 | 6个投票者之一 |
| 冷启动依赖 | 全依赖LLM+PII预分类 | LLM主导但E1+E4提供辅助信号 |
| 成熟期加速 | learned_classifier (未激活) | E3 SetFit (~2ms推理) |
| 类型系统 | 无 (LLM自由命名) | TypeLibrary (约束标签空间) |
| 优势场景 | 少类企业文档聚类 | 高基数同构文本分类 |
| 标签稳定性 | 低 (LLM每次可能命名不同) | 高 (TypeLibrary约束 + 融合投票) |
