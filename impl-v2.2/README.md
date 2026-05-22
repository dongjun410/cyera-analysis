# 企业文档智能聚类系统 V2.2

基于双通道嵌入架构 + 簇级分类的企业文档自动化分类系统。

## 快速开始

```bash
# 1. 启动 Elasticsearch + Kibana
docker-compose up -d

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3').save('/opt/models/bge-m3')"
ollama pull qwen2.5:7b

# 4. 运行
python main.py --input /path/to/documents --output ./output/
```

## 架构

```
Phase 0: 文档解析 & PII预分类
Phase 1: 结构特征提取
Phase 2: 双通道向量化 (BGE-M3 语义 + 结构特征)
Phase 3: 两阶段聚类 (KMeans粗聚类 + 层次细分裂)
Phase 4: 簇级分类 & LLM语义命名
Phase 5: 质量评估 & 知识蒸馏
```

## 硬件需求

- GPU: 12GB VRAM (BGE-M3 ~4GB + Qwen2.5-7B 4-bit ~5GB)
- RAM: 16GB+
- 磁盘: 10GB+ (含模型权重)

## 项目结构

```
impl-v2.2/
├── core/           # 10个核心模块
├── models/         # Pydantic数据模型
├── classifiers/    # 蒸馏后的轻量分类器
├── tests/          # 测试
├── output/         # 运行输出
├── main.py         # 主入口
├── incremental.py  # 增量更新
├── distill.py      # 知识蒸馏训练
├── config.yaml     # 全局配置
└── docker-compose.yml
```

## 与 impl-datadna 的关系

本方案（V2.2）和 `../impl-datadna/`（DataDNA优化方案）是同一问题的两种实现路径。V2.2 侧重可执行性和快速部署，DataDNA方案侧重架构可扩展性和方法论严谨性。详见 `../docs/superpowers/specs/2026-05-22-final-comparison.md`。
