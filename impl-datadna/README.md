# DataDNA 分类引擎 — 工程实现

基于分层递进架构（Tier 0→1→2→3）的企业文档智能分类系统。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型
# BGE-M3
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3').save('./models/bge-m3')"
# Mistral-7B (via Ollama)
ollama pull mistral:7b

# 3. 运行
python main.py --input /path/to/documents --output ./output/
```

## 架构

```
Tier 0: 确定性规则引擎 (PII特征提取)
Tier 1: 双阶段聚类 (Stage A: 结构哈希, Stage B: FAISS语义精炼)
Tier 2: 簇级分类 (已知类型匹配 + DeBERTa NER + Mistral-7B 4-bit)
Tier 3: LLM质量门 (Mistral-7B INT8 — 仅~2%文档触发)
类型发现: 离群缓冲区 → 周期性重聚类 → 新类型注册
知识蒸馏: LLM → SetFit, ~2ms/文档
```

## 硬件需求

- GPU: 12GB VRAM (开发/测试)
  - BGE-M3 ~4GB + DeBERTa-v3 ~1GB + Mistral-7B INT8 ~9GB
  - 模型顺序加载，峰值约9GB
- RAM: 32GB
- 磁盘: 20GB+ (含模型权重)

## 项目结构

```
impl-datadna/
├── src/
│   ├── tier0/          # Tier 0: 确定性规则引擎
│   ├── tier1/          # Tier 1: 双阶段聚类
│   ├── tier2/          # Tier 2: 簇级分类
│   ├── tier3/          # Tier 3: LLM质量门
│   ├── discovery/      # 类型发现循环
│   ├── distillation/   # 知识蒸馏
│   ├── embeddings/     # BGE-M3嵌入服务
│   ├── ner/            # DeBERTa-v3 NER服务
│   └── llm/            # Mistral-7B客户端
├── tests/              # 测试
├── output/             # 运行输出
├── main.py             # 主入口
├── incremental.py      # 增量文档处理
└── config.yaml         # 全局配置
```

## 与 impl-v2.2 的关系

本方案和 `../impl-v2.2/`（V2.2）是同一问题的两种实现路径。详见 `../docs/superpowers/specs/2026-05-22-final-comparison.md`。
