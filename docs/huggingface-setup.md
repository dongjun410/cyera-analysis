# Hugging Face 模型与数据集参考

**更新日期:** 2026-05-17

## 镜像配置

中国大陆使用 hf-mirror.com 镜像访问 Hugging Face：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

当前环境已配置该镜像，确认方式：

```bash
hf env | grep ENDPOINT
# ENDPOINT: https://hf-mirror.com
```

备用镜像（如主镜像不稳定可尝试）：

- `https://hf-mirror.com` (主镜像，已配置)
- `https://hf.xethub.hf.co` (Hugging Face 官方 Xet 存储后端，直接访问较慢)

## 下载命令

HF CLI v2 (`hf`) 替代了已废弃的 `huggingface-cli`：

```bash
# 模型
hf download <repo_id>

# 数据集
hf download <repo_id> --repo-type dataset
```

## 缓存位置

所有文件存储在 HF 默认缓存目录 `~/.cache/huggingface/hub/`。代码通过 `transformers` / `datasets` 库调用时会自动从该缓存加载，无需手动指定路径。

## 已下载模型

| # | 模型 ID | 用途 | 大小 |
|---|---------|------|------|
| 1 | `agentlans/flan-t5-small-ner` | FLAN-T5 Small NER 微调 (替代 pepegiallo) | ~297M |
| 2 | `pepegiallo/flan-t5-base_ner` | FLAN-T5 Base NER 微调 (token-classification) | ~422M |
| 3 | `google/flan-t5-small` | FLAN-T5 Small 基础模型 (text2text fallback) | ~760M |
| 4 | `google/flan-t5-base` | FLAN-T5 Base 基础模型 (text2text fallback) | ~1.0G |
| 5 | `google/flan-t5-large` | FLAN-T5 Large 基础模型 (text2text 模式) | ~3.1G |

缓存路径映射：

```
~/.cache/huggingface/hub/
├── models--agentlans--flan-t5-small-ner/
├── models--pepegiallo--flan-t5-base_ner/
├── models--google--flan-t5-small/
├── models--google--flan-t5-base/
└── models--google--flan-t5-large/
```

## 已下载数据集

| # | 数据集 ID | 用途 | 大小 |
|---|-----------|------|------|
| 1 | `conllpp` | CoNLL-03 NER 基准 (PER/ORG/LOC/MISC) | ~33K |
| 2 | `ai4privacy/pii-masking-300k` | 真实 PII 检测基准 (17种实体) | ~767M |

缓存路径映射：

```
~/.cache/huggingface/hub/
├── datasets--conllpp/
└── datasets--ai4privacy--pii-masking-300k/
```

## 模型变体映射

benchmark 代码 `benchmark/src/cyera_bench/models/flan_t5.py` 中的映射关系：

| variant | 基础模型 (hf_name) | NER 微调 (ner_checkpoint) |
|---------|-------------------|--------------------------|
| small | `google/flan-t5-small` | `agentlans/flan-t5-small-ner` |
| base | `google/flan-t5-base` | `pepegiallo/flan-t5-base_ner` |
| large | `google/flan-t5-large` | None (text2text 模式) |

## 注意事项

- `pepegiallo/flan-t5-small_ner` 在 HF 上不存在，已替换为 `agentlans/flan-t5-small-ner`
- `google/flan-t5-xl` 因网络和磁盘限制已放弃下载，代码/config/docs 中已移除 xl 引用
- 国内镜像偶有断连，重试即可；`hf download` 支持断点续传
