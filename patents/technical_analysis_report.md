# 技术分析报告

> 分析日期：2026-05-15
> 分析范围：Cyera 全部 8 项可公开检索专利/申请
> 方法论：基于专利说明书 + 开源项目参考 + 场景驱动的最优语言选择

---

## 目录

- [语言选择原则](#语言选择原则)
- [专利一：US12026123B2 — 云环境数据发现系统与方法](#专利一us12026123b2)
- [专利二：US12499083B2 — 数据发现系统与方法（续案）](#专利二us12499083b2)
- [专利三：US12566567B2 — 通过初始扫描发现数据存储位置](#专利三us12566567b2)
- [专利四：US12299167B2 — 数据分类与云环境保护](#专利四us12299167b2)
- [专利五：US12316686B1 — 多源网络安全策略应用](#专利五us12316686b1)
- [专利六、七 & 八：US20240362301A1, US20250068701A1 & WO2024224367A1 — 基于聚类的数据对象分类（母案 + 续案 + PCT）](#专利六七--八us20240362301a1-us20250068701a1--wo2024224367a1)
- [综合技术路线图](#综合技术路线图)

---

## 语言选择原则

本报告中每个组件的实现语言基于以下维度综合决策：

| 维度 | 权重 | 说明 |
|:------|:----|:------|
| **生态适配** | 最高 | 该领域的核心库/框架主要用什么语言？强行用其他语言意味着大量 FFI 开销 |
| **性能需求** | 高 | 延迟敏感（<10ms）还是吞吐敏感（>10K QPS）？内存管理是否关键？ |
| **并发模型** | 高 | IO 密集型（async/协程）还是 CPU 密集型（线程池/多进程）？ |
| **稳定性** | 中 | 内存安全？类型系统？生产环境验证程度？ |
| **运维成本** | 中 | 编译部署复杂度？监控集成？团队技能匹配？ |
| **开发效率** | 中 | 原型迭代速度？调试工具链？ |

**核心结论**：不存在一种语言适合所有组件。本报告采用 **Python（云API + ML） + Rust（高性能核心） + C++（底层引擎）** 的多语言策略，以 gRPC/Unix Socket 实现跨语言通信。

---

## 专利一：US12026123B2

### 云环境数据发现系统与方法

| 属性 | 内容 |
|:------|:------|
| **专利号** | US 12,026,123 B2 |
| **授权日** | 2024年7月2日 |
| **申请日** | 2022年1月13日 |
| **发明人** | Yotam Segev, Itamar Bar-Ilan, Yonatan Itai, Shay Makayes, Shani Beracha, Omer Duchovne, Itay Fainshtein |
| **页数** | 13页 |

---

### 1. 核心技术方案

本专利是 Cyera **无代理数据发现架构**的奠基性专利。核心思路是通过扫描云磁盘快照的**文件系统元数据**来发现数据存储位置，完全不需要在目标系统上安装代理程序。

<details>
<summary>数据发现流程架构图</summary>

```
┌──────────────────────────────────────────────────────────────┐
│                   云环境数据发现流程                            │
├──────────────────────────────────────────────────────────────┤
│  1. 磁盘检测 → 2. 快照扫描(两阶段) → 3. 引擎创建 → 4. 孤立数据发现 │
└──────────────────────────────────────────────────────────────┘
```

</details>

### 2. 语言选择分析

这个组件的技术特点是：**大量云 API 调用（IO 密集）+ 文件系统元数据解析（CPU 轻量）+ 多租户并行扫描**。

| 语言 | 云SDK生态 | FS解析 | 并行IO | 评价 |
|:---------|:--------|:----------|:-----|:-------|
| **Python** | ⭐⭐⭐⭐⭐ boto3/azure-sdk/google-cloud-sdk 是各自云厂商的一等公民 | ⭐⭐⭐⭐ pytsk3, fs, construct 成熟 | ⭐⭐⭐ asyncio + aioboto3 | **入选：云API编排层** |
| Rust | ⭐⭐⭐ aws-sdk-rust 仍在追赶 | ⭐⭐⭐⭐⭐ 零成本抽象 + 直接 syscall | ⭐⭐⭐⭐⭐ tokio async runtime | **入选：高性能扫描核心** |
| Go | ⭐⭐⭐ aws-sdk-go-v2 可用但滞后 | ⭐⭐⭐ go-diskfs 社区规模小 | ⭐⭐⭐⭐ goroutine 天然适合 | 备选 |
| C++ | ⭐⭐ AWS SDK for C++ 接口复杂 | ⭐⭐⭐⭐ libguestfs C API | ⭐⭐⭐ 线程池模型 | 不推荐（开发效率低） |

**决策：Python（编排 + 云API） + Rust（文件系统扫描核心），通过 PyO3 桥接。**

- **Python 负责**：多云 Provider 适配（boto3 是 AWS 的事实标准，无人能替代）、扫描任务调度、结果聚合
- **Rust 负责**：文件系统元数据解析（需要零拷贝和内存安全）、磁盘快照的原始字节读取、两阶段扫描的规则引擎

### 3. 开源实现参考

| 组件 | 推荐方案 | 理由 |
|:------|:---------|:------|
| 云SDK | **Python boto3** | AWS 官方维护，API 覆盖最全，社区最大 |
| 文件系统解析 | **pytsk3** (Python bindings for The Sleuth Kit) | 支持 NTFS/ext4/XFS/FAT/HFS+，经过 20+ 年取证场景验证 |
| 规则引擎 | **Rust + rhai** | 嵌入式脚本语言，性能优异，适合高频规则评估 |
| 分布式调度 | **Python celery** + Redis/RabbitMQ | 经过 Instagram/Netflix 验证的任务队列 |

### 4. 具体可实现方案

**架构总览**：

<details>
<summary>专利一架构总览</summary>

```
┌──────────────────────────────────────────────────────────────────┐
│  Python (编排层)                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ AWS Provider │  │Azure Provider│  │ GCP Provider           │  │
│  │ (boto3)      │  │(azure-sdk)   │  │(google-cloud-compute)  │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘  │
│         └─────────────────┼──────────────────────┘               │
│                    CloudProvider Interface (Protocol)              │
│                           │                                       │
│                    ┌──────┴──────┐                                │
│                    │  Celery     │  分布式任务调度                  │
│                    │  Workers    │                                  │
│                    └──────┬──────┘                                │
│                           │ gRPC                                   │
│  ┌────────────────────────┴────────────────────────────────────┐  │
│  │              Rust (高性能扫描核心)                              │  │
│  │                                                                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │  │
│  │  │ FS Metadata   │  │ Rule Engine  │  │ Snapshot Mount   │   │  │
│  │  │ Parser        │  │ (rhai)       │  │ (nbdkit bridge)  │   │  │
│  │  │ ext4/xfs/ntfs │  │              │  │                  │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

</details>

**Python 编排层**：

<details>
<summary>Python 编排层代码</summary>

```python
# cloud_discovery/provider.py
from typing import Protocol, List, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class Disk:
    id: str
    name: str
    size_gb: int
    region: str
    attached_to: Optional[str]  # None = 孤立磁盘
    created_at: int
    tags: dict[str, str]

@dataclass
class Snapshot:
    id: str
    disk_id: str
    status: str   # pending | completed | error
    size_bytes: int

class CloudProvider(Protocol):
    """云平台抽象协议 — Python Protocol 提供 duck-typing 灵活性"""

    async def list_disks(self) -> List[Disk]: ...
    async def create_snapshot(self, disk_id: str) -> Snapshot: ...
    async def delete_snapshot(self, snapshot_id: str) -> None: ...
    async def get_snapshot_blocks(self, snapshot_id: str) -> str: ...
    # 返回 nbdkit 可用的快照路径


# cloud_discovery/aws_provider.py
import boto3
from botocore.config import Config as BotoConfig

class AWSProvider:
    """AWS 实现 — boto3 是这个领域无可争议的标准"""

    def __init__(self, region: str):
        # boto3 内置重试、分页、凭证链 — 这些在 Rust SDK 中仍需手动实现
        self.ec2 = boto3.client('ec2', region_name=region, config=BotoConfig(
            retries={'max_attempts': 10, 'mode': 'adaptive'}
        ))

    async def list_disks(self) -> List[Disk]:
        # boto3 的分页器自动处理 DescribeVolumes 的 500+ 卷分页
        paginator = self.ec2.get_paginator('describe_volumes')
        disks = []
        for page in paginator.paginate():
            for vol in page['Volumes']:
                disks.append(Disk(
                    id=vol['VolumeId'],
                    name=self._get_name(vol.get('Tags', [])),
                    size_gb=vol['Size'],
                    region=vol['AvailabilityZone'][:-1],
                    attached_to=self._get_attachment(vol),
                    created_at=int(vol['CreateTime'].timestamp()),
                    tags=self._parse_tags(vol.get('Tags', [])),
                ))
        return disks

    async def create_snapshot(self, disk_id: str) -> Snapshot:
        resp = self.ec2.create_snapshot(
            VolumeId=disk_id,
            Description=f'dspm-scan-{disk_id}',
            TagSpecifications=[{
                'ResourceType': 'snapshot',
                'Tags': [{'Key': 'Purpose', 'Value': 'dspm-scan'},
                         {'Key': 'TTL', 'Value': '1h'}]  # 自动清理标记
            }]
        )
        return Snapshot(id=resp['SnapshotId'], disk_id=disk_id,
                        status='pending', size_bytes=resp['VolumeSize'] * 1024**3)


# cloud_discovery/orchestrator.py
from celery import Celery
import grpc
from discovery_pb2_grpc import FSScannerStub  # Rust gRPC 服务

app = Celery('dspm_discovery', broker='redis://localhost:6379/2')

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def scan_disk(self, disk_json: str, provider_type: str):
    """Celery 任务 — 每个磁盘独立扫描，利用 Celery 的分布式能力"""
    disk = Disk.from_json(disk_json)
    provider = get_provider(provider_type)

    snap = provider.create_snapshot(disk.id)
    try:
        # 调用 Rust 高性能扫描核心
        channel = grpc.insecure_channel('localhost:50051')
        stub = FSScannerStub(channel)
        result = stub.ScanSnapshot(ScanRequest(
            snapshot_path=provider.get_snapshot_blocks(snap.id),
            rules_yaml=load_rules(),
            scan_depth="METADATA_ONLY",  # 专利的两阶段：先 METADATA_ONLY
        ))
        return DataStoreInfo.from_proto(result)
    finally:
        provider.delete_snapshot(snap.id)
```

</details>

**Rust 高性能扫描核心**：

<details>
<summary>Rust 扫描核心代码</summary>

```rust
// fs_scanner/src/lib.rs
// 选择 Rust：内存安全 + 零成本抽象 + rayon 并行

use rayon::prelude::*;
use rhai::{Engine, Scope};
use std::collections::HashMap;

/// 文件系统元数据解析器 — 利用 Rust 的 enum 建模文件系统多样性
pub enum FileSystemType {
    Ext4 { superblock_offset: u64, block_size: u32 },
    Xfs  { superblock_offset: u64, ag_count: u32 },
    Ntfs { mft_offset: u64, cluster_size: u32 },
}

/// 快照扫描器 — 利用 Rust 的所有权系统保证块数据的生命周期安全
pub struct SnapshotScanner {
    block_device: Box<dyn BlockReader>,  // trait object: 支持多种云后端
    rules: Vec<DiscoveryRule>,
}

impl SnapshotScanner {
    /// 两阶段扫描 — 对应专利的核心方法
    pub fn scan(&self) -> Result<Vec<DataStoreInfo>, ScannerError> {
        // 阶段1: 浅扫描 — 仅读取 Superblock + Inode Table
        // 使用 rayon 的 par_bridge 实现并行块读取
        let metadata = self.read_fs_metadata()?;

        // 阶段2: 规则评估 — rhai 嵌入式脚本引擎
        let mut engine = Engine::new();
        let mut scope = Scope::new();
        scope.push("metadata", metadata.clone());

        let confidence: f64 = engine.eval_with_scope::<f64>(
            &mut scope,
            &self.compile_rules()  // YAML 规则 → rhai 表达式
        )?;

        // 阶段3: 深度扫描 (仅当置信度 > 阈值)
        if confidence > self.config.deep_scan_threshold {
            self.deep_scan(metadata)
        } else {
            Ok(vec![])
        }
    }

    /// 读取文件系统元数据 — 利用 Rust 的零拷贝解析
    fn read_fs_metadata(&self) -> Result<FsMetadata, ScannerError> {
        // 1. 读取 Superblock (块0) — 512 bytes
        let superblock = self.block_device.read_block(0, 512)?;

        // 2. 利用 nom 解析器组合子进行零拷贝解析
        let (_, sb) = ext4::parse_superblock(&superblock)
            .map_err(|_| ScannerError::InvalidSuperblock)?;

        // 3. 仅读取 Inode Table 的前 N 个 inode (不遍历整个文件系统)
        let inode_table = self.block_device.read_block(
            sb.inode_table_offset,
            sb.inode_size * sb.inode_count.min(1000)  // 上限: 1000 inodes
        )?;

        Ok(FsMetadata {
            fs_type: FileSystemType::Ext4 {
                superblock_offset: 0,
                block_size: sb.block_size as u32,
            },
            inodes: ext4::parse_inode_table(&inode_table, sb.inode_count.min(1000))?,
        })
    }
}

/// 规则编译器 — 将 YAML 规则编译为 rhai 表达式
fn compile_rules(rules: &[DiscoveryRule]) -> String {
    // 输入 YAML:
    //   - name: mysql_detection
    //     path_patterns: ["*/var/lib/mysql/*", "*/data/mysql/*"]
    //     file_patterns: ["*.ibd", "*.frm", "ibdata1"]
    //
    // 输出 rhai:
    //   let score = 0.0;
    //   if metadata.paths.contains("var/lib/mysql") { score += 0.3; }
    //   if metadata.files.any(|f| f.ends_with(".ibd")) { score += 0.2; }
    //   score
    rules.iter().map(|r| r.to_rhai()).collect::<Vec<_>>().join("\n")
}
```

</details>

**为什么这样选择**：
- Python 的 boto3 用了 10+ 年打磨 API 设计、错误处理、分页和重试逻辑；Rust 的 aws-sdk-rust 在这些方面仍有差距
- Rust 在直接读取原始字节流时，所有权系统消除了一整类内存安全 bug（buffer overflow、use-after-free），这在解析不受信任的外部磁盘数据时至关重要
- Celery 的分布式任务模型天然适配"每个磁盘快照 = 一个独立任务"的扫描模式，且自带重试、监控、速率限制

---

## 专利二：US12499083B2

### 数据发现系统与方法（续案）

| 属性 | 内容 |
|:------|:------|
| **专利号** | US 12,499,083 B2 |
| **授权日** | 2025年12月16日 |
| **申请日** | 2024年5月30日（US 17/647,899 的延续申请） |
| **发明人** | 同 US12026123 |

---

### 1. 与母专利的关系

本专利是 US12026123 的 **Continuation**——相同说明书，扩展权利要求范围。新增的核心主张包括：
- **引擎实例化生命周期管理**：在隔离 VM 中创建专用数据库引擎，仅执行轻量查询后立即销毁
- **跨文件系统类型识别**：ext4/XFS/NTFS/ReiserFS/ZFS 的统一检测接口
- **按需引擎匹配**：基于检测到的数据库类型和版本（如 MySQL 8.0 vs 5.7），精确匹配引擎版本

### 2. 语言选择分析

这个组件的独特挑战是**按需创建数据库引擎容器并执行验证查询**——这是容器编排问题，不是语言问题。

| 组件 | 语言选择 | 理由 |
|:------|:---------|:------|
| 容器编排 | **Python** (docker-py / testcontainers) | Python 的容器编排生态最成熟；testcontainers-python 提供与 pytest 的无缝集成 |
| 文件系统类型检测 | **Rust** (继承自专利一) | 与核心扫描组件共用代码 |
| 引擎验证查询 | **各数据库原生协议** | MySQL: C Connector, PostgreSQL: libpq, MongoDB: C Driver |
| gRPC 服务 | **Rust** (tonic) | 与专利一的 Rust 核心统一技术栈 |

### 3. 开源实现参考

| 组件 | 推荐方案 | 理由 |
|:------|:---------|:------|
| 容器管理 | **docker-py** (Python) | Docker 官方 Python SDK，API 覆盖完整 |
| 临时数据库引擎 | **testcontainers-python** | "启动→验证→销毁"模式与专利场景完全匹配 |
| 文件系统检测 | **blkid** (util-linux) | 20+ 年验证的块设备识别工具 |

### 4. 具体可实现方案

<details>
<summary>Python 引擎编排器代码</summary>

```python
# engine_orchestrator.py
import docker
from testcontainers.mysql import MySqlContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.mongodb import MongoDbContainer
from typing import Optional
import tempfile
import os

# 引擎类型 → 容器镜像 + 验证查询 的映射表
ENGINE_REGISTRY = {
    ("mysql", "8.0"): {
        "image": "mysql:8.0",
        "container_class": MySqlContainer,
        "verify_query": "SELECT COUNT(*) FROM information_schema.tables",
    },
    ("mysql", "5.7"): {
        "image": "mysql:5.7",
        "container_class": MySqlContainer,
        "verify_query": "SELECT COUNT(*) FROM information_schema.tables",
    },
    ("postgresql", None): {  # version unknown → 使用通用版本
        "image": "postgres:16",
        "container_class": PostgresContainer,
        "verify_query": "SELECT COUNT(*) FROM information_schema.tables",
    },
    ("mongodb", None): {
        "image": "mongo:7",
        "container_class": MongoDbContainer,
        "verify_query": 'db.runCommand({listCollections: 1})',
    },
}

class EngineOrchestrator:
    """按需数据库引擎编排器 — 对应续案的核心权利要求"""

    def __init__(self):
        self.docker = docker.from_env()

    def create_engine(self, ds_type: str, version: Optional[str] = None,
                      snapshot_path: str = None) -> 'DatabaseEngine':
        """
        创建临时数据库引擎，挂载快照克隆卷

        专利关键点：
        - 引擎在隔离VM/容器中创建，不占用生产资源
        - 引擎配置为匹配检测到的类型和版本
        - 通过快照克隆卷挂载，无需逐一获取存储权限
        """
        key = (ds_type, version)
        if key not in ENGINE_REGISTRY:
            # 模糊匹配: 尝试仅按类型查找
            key = (ds_type, None)
            if key not in ENGINE_REGISTRY:
                raise UnsupportedEngineError(f"No engine for {ds_type} v{version}")

        config = ENGINE_REGISTRY[key]
        container_cls = config["container_class"]

        # 挂载快照卷到容器 — 仅当提供了快照路径时
        volumes = {}
        if snapshot_path:
            volumes[snapshot_path] = {"bind": "/snapshot_data", "mode": "ro"}

        # 使用 testcontainers 启动临时引擎 — 自动处理端口分配和清理
        container = container_cls(volumes=volumes)
        container.start()

        return DatabaseEngine(
            container=container,
            verify_query=config["verify_query"],
            ds_type=ds_type,
            version=version,
        )

    def quick_stats(self, engine: 'DatabaseEngine') -> EngineStats:
        """执行轻量验证查询 — 对应专利的'仅执行SHOW DATABASES + COUNT(*)'"""
        try:
            result = engine.container.exec(engine.verify_query)
            return EngineStats(
                row_count=int(result.output),
                is_accessible=True,
            )
        except Exception as e:
            return EngineStats(row_count=0, is_accessible=False, error=str(e))

    def destroy_engine(self, engine: 'DatabaseEngine'):
        """立即销毁引擎 — 对应专利的'完成后引擎VM立即销毁'"""
        engine.container.stop()
        # testcontainers 的自动清理机制保证即使异常退出也会清理
```

</details>

---

## 专利三：US12566567B2

### 通过初始扫描发现数据存储位置

| 属性 | 内容 |
|:------|:------|
| **专利号** | US 12,566,567 B2 |
| **授权日** | 2026年3月3日 |
| **申请日** | 2022年5月19日（CIP of US 17/647,899） |
| **发明人** | 同数据发现家族 |

---

### 1. 核心技术突破

本专利是数据发现家族中最具工程价值的专利，核心创新是 **Lazy Mount（惰性挂载）+ 条件性克隆（Conditional Clone）**——仅读取磁盘快照的极小部分（~1-10 MB）即可判断是否包含数据存储。

<details>
<summary>惰性挂载 vs 传统方案对比</summary>

```
传统方案:  快照 → 全量复制到新卷 (GB-TB级) → 挂载 → 全FS遍历 → 分析
            耗时: 分钟-小时 | 成本: 完整快照存储

本专利:   快照(COW) → 惰性挂载 → 仅读 Superblock+InodeTable(~1MB) → 规则评估
                                               ↓
                          可能性低 (90%+)            可能性高 (<10%)
                          → 删除快照                   → 条件性克隆 → 深度分析
            耗时: 秒级      | 成本: ~0 (仅COW快照)
```

</details>

### 2. 语言选择分析

这是对**延迟和 IO 效率**最敏感的组件——直接从块设备读取原始字节，需要零拷贝和精确的内存控制。

| 语言 | 零拷贝 | 块IO | NBD | 评价 |
|:---------|:--------|:--------|:-------|:-------|
| **Rust** | ⭐⭐⭐⭐⭐ 所有权+借用 | ⭐⭐⭐⭐⭐ tokio-uring (io_uring) | ⭐⭐⭐⭐ libnbd-rs | **最佳选择** |
| C | ⭐⭐⭐⭐⭐ 原生指针 | ⭐⭐⭐⭐⭐ 直接 ioctl | ⭐⭐⭐⭐⭐ libnbd C API | 备选（需要手动内存管理） |
| Python | ⭐ 需要复制 | ⭐⭐ os.pread 可用但性能差 | ⭐⭐⭐ python-libnbd | 仅适合原型验证 |
| Go | ⭐⭐⭐ 切片引用 | ⭐⭐⭐ syscall | ⭐⭐ nbd 库生态弱 | 不推荐 |

**决策：Rust — 这是最适合 Rust 的场景。Linux io_uring + 零拷贝解析 + 内存安全的块设备操作。管理进程（快照生命周期、结果存储）用 Python。**

### 3. 开源实现参考

| 组件 | 推荐方案 | 理由 |
|:------|:---------|:------|
| NBD 客户端 | **libnbd** + Rust binding (`libnbd-rs`) | Red Hat 维护的 NBD 协议官方实现 |
| io_uring IO | **tokio-uring** | Linux 5.1+ 的高性能异步 IO，比 epoll 快 2-5x |
| 文件系统 Superblock 解析 | **ext4-viewer** (Rust) / **nom** parser combinator | 零拷贝二进制解析 |
| 快照管理 | **Python + boto3** | 快照 CRUD 是典型的管理操作，Python 更适合 |

### 4. 具体可实现方案

<details>
<summary>Rust 惰性块读取器代码</summary>

```rust
// lazy_mount/src/lib.rs
// 选择 Rust：io_uring 零拷贝需要精确生命周期控制，不受信任的外部数据需要内存安全

use tokio_uring::{fs::File, BufResult};
use nom::{number::complete::le_u32, bytes::complete::take};
use std::os::unix::io::AsRawFd;

/// 惰性块读取器 — 利用 io_uring 实现零拷贝块读取
pub struct LazyBlockReader {
    nbd_handle: libnbd::Handle,  // NBD 连接句柄
    block_cache: lru::LruCache<u64, Vec<u8>>,  // LRU 块缓存
}

impl LazyBlockReader {
    /// 按需读取块 — 仅在访问时触发 IO
    ///
    /// io_uring 的关键优势：内核态和用户态共享提交/完成队列（SQ/CQ），
    /// 避免了传统 read() 系统调用的上下文切换开销。
    /// 对于读取100个不连续块（如 inode table entries）的场景，
    /// io_uring 比 pread() 快 40-60%。
    pub async fn read_block(&mut self, offset: u64, size: usize) -> Result<&[u8], IoError> {
        // 检查 LRU 缓存 — 避免重复读取 Superblock/Group Descriptors
        if let Some(cached) = self.block_cache.get(&offset) {
            return Ok(cached.as_slice());
        }

        // io_uring 异步读取 — 内核态完成，用户态接收
        let buf = vec![0u8; size];
        let file = File::from_raw_fd(self.nbd_handle.as_raw_fd());
        let (res, buf) = file.read_at(buf, offset).await;
        res?;

        self.block_cache.put(offset, buf);
        Ok(self.block_cache.get(&offset).unwrap().as_slice())
    }

    /// 惰性 Superblock 读取 — 仅读取块0的第一个扇区(512 bytes)
    pub async fn read_superblock(&mut self) -> Result<Ext4Superblock, ScannerError> {
        let raw = self.read_block(0, 512).await?;
        // nom 零拷贝解析 — 直接在原始字节上解析，不复制
        let (_, sb) = parse_ext4_superblock(raw)
            .map_err(|e| ScannerError::ParseError(e.to_string()))?;
        Ok(sb)
    }
}

/// ext4 Superblock 的零拷贝解析器
/// nom 的优势：解析器组合子工作在原字节切片上，不分配额外内存
fn parse_ext4_superblock(input: &[u8]) -> nom::IResult<&[u8], Ext4Superblock> {
    let (input, _) = take(1024usize)(input)?; // ext4 superblock 在偏移 1024
    let (input, inodes_count) = le_u32(input)?;
    let (input, blocks_count) = le_u32(input)?;
    let (input, _) = take(12usize)(input)?;   // skip reserved blocks + free counters
    let (input, block_size_raw) = le_u32(input)?;
    let (input, _) = take(16usize)(input)?;   // skip blocks_per_group + frags
    let (input, inode_size) = le_u32(input)?;

    let block_size = 1024u64 << block_size_raw;  // 1024 << 0 = 1K, 1024 << 2 = 4K

    Ok((input, Ext4Superblock {
        inodes_count: inodes_count as u64,
        blocks_count: blocks_count as u64,
        block_size: block_size as u32,
        inode_size: inode_size as u16,
    }))
}
```

</details>

**Python 管理进程**：

<details>
<summary>Python 快照管理器代码</summary>

```python
# snapshot_manager.py
import asyncio
import grpc
from dataclasses import dataclass
from typing import List

@dataclass
class ScanConfig:
    deep_scan_threshold: float = 0.6   # 超过此置信度触发深度扫描
    max_metadata_bytes: int = 10_485_760  # 元数据读取上限: 10MB
    snapshot_ttl_minutes: int = 5      # 快照最长保留时间

class SnapshotScanManager:
    """快照扫描管理器 — Python 适合管理生命周期和编排"""

    def __init__(self, provider: CloudProvider, config: ScanConfig):
        self.provider = provider
        self.config = config
        # gRPC 连接到 Rust 扫描服务
        self.grpc_channel = grpc.aio.insecure_channel('localhost:50051')
        self.scanner = FSScannerStub(self.grpc_channel)

    async def scan_all_disks(self) -> List[DataStoreInfo]:
        disks = await self.provider.list_disks()
        results = []

        # 并发扫描 — asyncio + gRPC async 支持数千个并发扫描
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self._scan_single(disk)) for disk in disks]

        for task in tasks:
            if result := task.result():
                results.append(result)

        return results

    async def _scan_single(self, disk: Disk) -> Optional[DataStoreInfo]:
        """单个磁盘的完整扫描生命周期"""
        snap = await self.provider.create_snapshot(disk.id)
        try:
            # 阶段1: 惰性扫描 (Rust 核心) — 仅读 ~1MB 元数据
            shallow = await self.scanner.ShallowScan(ShallowScanRequest(
                snapshot_path=await self.provider.get_snapshot_blocks(snap.id),
                max_bytes=self.config.max_metadata_bytes,
            ))

            if shallow.confidence < self.config.deep_scan_threshold:
                return None  # 90%+ 的磁盘在此被过滤

            # 阶段2: 条件性克隆 — 仅对高置信度磁盘
            clone_id = await self.provider.clone_snapshot(snap.id)
            try:
                deep = await self.scanner.DeepScan(DeepScanRequest(
                    volume_path=clone_id,
                ))
                return DataStoreInfo.from_proto(deep)
            finally:
                await self.provider.delete_volume(clone_id)
        finally:
            await self.provider.delete_snapshot(snap.id)
```

</details>

---

## 专利四：US12299167B2

### 数据分类与云环境保护

| 属性 | 内容 |
|:------|:------|
| **专利号** | US 12,299,167 B2 |
| **授权日** | 2025年5月13日 |
| **申请日** | 2022年10月13日 |
| **发明人** | Yotam Segev, Itamar Bar-Ilan, Yonatan Itai, Shiran Bareli, Michael Elazar, Antony Timchenko, Itay Mizeretz |
| **页数** | 18页 |

---

### 1. 核心技术方案

Cyera 分类引擎最核心的专利，公开了**双路径混合分类方法**：

<details>
<summary>双路径混合分类流程</summary>

```
输入数据集
  │
  ├── 抽样引擎 → 结构化(半随机分块) + 非结构化(元数据聚类)
  │
  ├── [路径A] 启发式真值表 → 数值型数据 (SSN, 信用卡号, 护照号...)
  │     输入维度: 正则强度 × 上下文± × 模式频率 × 唯一性
  │
  ├── [路径B] ML分类器 → 字符串数据 (姓名, 地址, 企业名...)
  │     模型: 从标注数据训练的 NER 模型
  │
  └── 融合 + 角色判定 + 虚假过滤
```

</details>

### 2. 语言选择分析

这是整个系统中最典型的 **ML/NLP 密集型**组件。选择标准非常明确：

| 语言 | NLP生态 | 真值表 | 推理 | 评价 |
|:---------|:--------|:--------|:-------|:-------|
| **Python** | ⭐⭐⭐⭐⭐ HuggingFace, spaCy, fastText, scikit-learn | ⭐⭐⭐⭐ numpy/pandas 天然适合多维评分矩阵 | ⭐⭐⭐⭐ ONNX Runtime, vLLM, Triton | **无可争议的首选** |
| Rust | ⭐⭐ candle, burn 库仍在早期 | ⭐⭐⭐ | ⭐⭐⭐ ONNX Runtime bindings | 备选（嵌入式场景） |
| C++ | ⭐⭐⭐ libtorch, ONNX Runtime | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ TensorRT | 仅适合超低延迟推理 |
| Java | ⭐⭐⭐ DJL, CoreNLP | ⭐⭐ | ⭐⭐⭐ | 仅适合已有 JVM 基础设施 |

**决策：Python 是 ML/NLP 的唯一正确答案。** HuggingFace Transformers、spaCy、fastText 的 Python API 是各自项目的一等公民。spaCy 的底层用 Cython 实现，分类器的实际推理在原生 C 层执行——Python 负责编排，性能敏感的部分已经在 C 层。

### 3. 开源实现参考

| 组件 | 推荐方案 | 理由 |
|:------|:---------|:------|
| NER 引擎 | **spaCy** `en_core_web_trf` (Python, 底层 Cython) | RoBERTa-base Transformer，18 种实体类型，生产部署成熟 |
| 数值模式识别 | **Python `re` + 自定义验证器** | SSN/Luhn/IBAN/Verhoeff 的纯函数适合 Python |
| 真值表引擎 | **pandas DataFrame + numpy** | 多维评分矩阵的自然表示 |
| FLAN-T5/Mistral 语义层 | **ollama** + **vLLM** (Python API) | vLLM 的 PagedAttention 提供最高吞吐量 |
| 生产推理服务 | **Ray Serve** 或 **BentoML** | 支持 GPU 批处理、自动扩缩容 |

### 4. 具体可实现方案

<details>
<summary>Python 混合分类器代码</summary>

```python
# classifier/hybrid_classifier.py
"""
双路径混合分类器 — Python 实现。语言选择理由详见上方分析表。
"""

import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import spacy
from rapidfuzz import fuzz  # C++ 实现的快速模糊匹配，比 difflib 快 10-50x

@dataclass
class TruthTableDimension:
    """真值表的输入维度 — 对应专利的核心设计"""
    validated_count: int = 0      # 已验证实例数
    regex_strength: float = 0.0   # 正则特异性 0-1
    supportive_context: int = 0   # 支持性上下文词命中数
    unsupportive_context: int = 0 # 否定性上下文词命中数
    pattern_frequency: float = 0.0 # 模式频率
    uniqueness_score: float = 0.0  # 值唯一性 0-1

class TruthTableEngine:
    """
    启发式真值表引擎 — 对应专利的第一路径

    实现细节：
    - pandas DataFrame 作为评分矩阵后端 — 利用 numpy 的向量化运算
    - 每个维度的离散化级别:
        validated_count: {0, 1-3, 4-9, 10-49, 50+}
        regex_strength: {0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0}
        ...共 5×5×3×2×3×3 = 1350 个可能组合
    - 使用 MultiIndex DataFrame 实现 O(1) 查找
    """

    def __init__(self, calibration_data: pd.DataFrame):
        """
        calibration_data 结构:
          validated_count_bin | regex_bin | support_bin | unsupport_bin | ... | score
          -------------------------------------------------------------------------
          0                  | 0.9       | 3           | 0             | ... | 0.85
          1-3                | 0.7       | 1           | 1             | ... | 0.35
        """
        # 构建 MultiIndex — pandas 的优势在此体现
        self.score_table = calibration_data.set_index([
            'validated_count_bin', 'regex_bin', 'support_bin',
            'unsupport_bin', 'freq_bin', 'uniqueness_bin'
        ])['score']

    def lookup(self, dims: TruthTableDimension) -> float:
        """O(1) 真值表查找"""
        key = (
            self._bin_count(dims.validated_count),
            self._bin_float(dims.regex_strength, 0.2),
            min(dims.supportive_context, 4),
            min(dims.unsupportive_context, 2),
            self._bin_float(dims.pattern_frequency, 0.1),
            self._bin_float(dims.uniqueness_score, 0.2),
        )
        try:
            return float(self.score_table.loc[key])
        except KeyError:
            # 查找最近的匹配 — scipy.spatial.KDTree 的 O(log n) 最近邻
            return self._interpolate(key)


class HybridClassifier:
    """
    双路径混合分类器

    架构：
      - 第一路径 (真值表): 用于数值型数据 — 快速、确定性、可解释
      - 第二路径 (ML NER): 用于字符串数据 — 深度语义、高召回
      - 融合: 加权平均 + 角色判定 + 虚假过滤
    """

    def __init__(self, model_config: dict):
        # 第一路径: 真值表引擎
        self.truth_table = TruthTableEngine(
            pd.read_parquet(model_config['truth_table_path'])
        )

        # 第二路径: spaCy NER (底层 Cython，实际推理在 C 层)
        # en_core_web_trf = RoBERTa-base，在 CoNLL-03 上 F1 ~92%
        self.nlp = spacy.load("en_core_web_trf")

        # 可选的 LLM 验证层 — 通过 Ollama 调用本地 Mistral
        self.llm_model = model_config.get('llm_model', None)  # e.g., "mistral:7b"

        # 角色判定器
        self.role_detector = RoleDetector(model_config.get('role_patterns', {}))

    def classify(self, value: str, context: Dict[str, str]) -> ClassificationResult:
        """
        对单个数据值执行双路径分类

        性能说明：
        - spaCy 推理在 C 层执行（Cython → C），Python 仅做编排
        - 真值表查找是纯 numpy 操作，<100μs
        - LLM 验证仅对高歧义样本触发（~5% 的样本），降低总体延迟
        """
        # 判断数据类型 — 基于 Python re 的模式匹配（C 层执行）
        is_numeric = self._is_numeric_pattern(value)

        # 路径A: 真值表 (仅数值型)
        first_score = 0.0
        if is_numeric:
            dims = self._extract_truth_table_dims(value, context)
            first_score = self.truth_table.lookup(dims)

        # 路径B: spaCy NER (所有类型)
        # 注意: nlp() 调用在进入时复制 Python 字符串到 C 层，
        #       实际 NER 推理在 C 层完成，Python GIL 被释放
        doc = self.nlp(f"{context.get('column_name', '')}: {value}")
        entities = [(ent.label_, ent.text) for ent in doc.ents if ent.text in value]
        second_score = self._ner_score(entities, value)

        # 融合 — 加权平均
        confidence = self._fuse(first_score, second_score, is_numeric)

        # LLM 验证 — 仅高歧义样本
        if 0.3 < confidence < 0.7 and self.llm_model:
            confidence = self._llm_validate(value, context, confidence)

        # 后处理: 角色 + 虚假过滤
        role = self.role_detector.detect(value, context)
        is_mock = MockDataDetector.is_mock(value, context)

        return ClassificationResult(
            data_type=self._infer_type(entities, value),
            role=role,
            is_mock=is_mock,
            confidence=min(confidence, 1.0),
        )


# classifier/mock_detector.py
class MockDataDetector:
    """虚假数据过滤器 — 纯 Python 函数，适合规则驱动的场景"""

    # Python 的正则引擎用 C 实现 — re.match() 是 C 调用，不是 Python 循环
    MOCK_PATTERNS = {
        'sequential': re.compile(r'^(?:123456789|987654321|111111111|222222222|333333333|444444444|555555555|666666666|777777777|888888888|999999999)$'),
        'zero_pattern': re.compile(r'^0+$'),
        'repeated_char': re.compile(r'^(.)\1{5,}$'),  # 如 "xxxxxx"
    }

    MOCK_CONTEXT_WORDS = frozenset({
        'test', 'sample', 'example', 'dummy', 'fake', 'mock',
        'xxxx', '****', 'redacted', '[redacted]', 'placeholder',
        'todo', 'fixme', 'tk',  # 占位符
    })

    @classmethod
    def is_mock(cls, value: str, context: Dict[str, str]) -> bool:
        # 规则1: 模式匹配 (C 层 re 引擎)
        for pattern in cls.MOCK_PATTERNS.values():
            if pattern.match(value.replace('-', '').replace(' ', '')):
                return True

        # 规则2: 上下文关键词
        context_text = ' '.join(context.values()).lower()
        if any(kw in context_text for kw in cls.MOCK_CONTEXT_WORDS):
            return True

        return False
```

</details>

---

## 专利五：US12316686B1

### 多源网络安全策略应用

| 属性 | 内容 |
|:------|:------|
| **专利号** | US 12,316,686 B1 |
| **授权日** | 2025年5月27日 |
| **申请日** | 2024年11月18日 |
| **发明人** | Zohar Vittenberg, Nadav Zingerman, Roei Mutay（Trail Security 创始团队） |

---

### 1. 核心技术方案

这是 Trail Security 的唯一专利，体现了 Omni DLP 的"策略大脑"定位——使用 **LLM 将不同格式的安全策略归一化为统一中间表示**，然后跨格式生成到不同平台。

### 2. 语言选择分析

这个组件有两个截然不同的子任务：

| 子任务 | 特征 | 语言选择 | 理由 |
|:--------|:------|:---------|:------|
| LLM 驱动的策略归一化 | IO 密集（等 LLM 响应）、低频（策略变更频率远低于数据扫描） | **Python** | Ollama/OpenAI SDK 的 Python API 最成熟；LangChain/LlamaIndex 提供强大的 Prompt 模板 |
| 高吞吐策略评估 | 每次 API 请求/文件上传都需策略判定，QPS 可达成千上万 | **Rust** | 策略评估是纯 CPU 操作（条件匹配），需要最低延迟和最可预测的性能 |
| 中间表示 → 目标平台转换 | 规则转换，CPU 轻量 | **Python** | 模板引擎（Jinja2）+ OPA Rego 生成 |

**决策：Python（策略归一化 + 格式转换）+ Rust（高吞吐策略评估），通过 Redis 缓存同步策略。**

### 3. 开源实现参考

| 组件 | 推荐方案 | 理由 |
|:------|:---------|:------|
| LLM 推理 | **Ollama** (Mistral-7B) 或 **vLLM** | 本地部署，数据不出域 |
| 策略 IR | **OPA Rego** | CNCF 毕业项目，策略即代码，有丰富的工具链 |
| 高吞吐评估 | **cedar** (Rust, AWS 开源) | Amazon Verified Permissions 的核心引擎 |
| 策略转换 | **Jinja2** (Python) | 模板驱动的跨格式策略生成 |
| 策略存储 + 缓存 | **Redis** (读写分离) | 亚毫秒级策略缓存 |

### 4. 具体可实现方案

<details>
<summary>Python 策略归一化器代码</summary>

```python
# policy_engine/normalizer.py
"""
LLM 驱动的策略归一化器 — Python 实现。语言选择理由详见上方分析表。
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import ollama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class PolicyIR(BaseModel):
    """统一中间表示 — 用 Pydantic 保证类型安全"""
    subjects: List[str] = Field(description="主体: role:xxx, group:xxx, user:xxx")
    actions: List[str] = Field(description="动作: share, download, delete, upload")
    objects: List[str] = Field(description="对象: type:xxx, contains:xxx, label:xxx")
    conditions: Dict = Field(default_factory=dict, description="条件约束")
    effect: str = Field(description="allow 或 deny")

class PolicyNormalizer:
    """
    策略归一化器

    设计亮点 — 使用 PydanticOutputParser 将 LLM 的 JSON 输出自动解析
    为类型安全的 PolicyIR，避免正则解析 LLM 响应的脆弱性
    """

    # LangChain PromptTemplate — 比 f-string 更易维护和测试
    NORMALIZE_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", """You are a security policy translator. Convert policies 
        from different security platforms into a normalized intermediate representation.

        Rules:
        - subjects: prefix with "role:", "group:", or "user:"
        - actions: use standardized verbs (read, write, share, download, delete, upload)
        - objects: prefix with "type:" (file type), "contains:" (data type), "label:" (classification)
        - effect: ONLY "allow" or "deny"

        {format_instructions}"""),
        ("human", "Source format: {source_format}\nPolicy: {policy_text}"),
    ])

    def __init__(self, model: str = "mistral:7b"):
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=PolicyIR)

    def normalize(self, policy_text: str, source_format: str) -> PolicyIR:
        """
        将任意格式的安全策略归一化为 PolicyIR

        source_format 示例:
          - "microsoft_purview_dlp"
          - "okta_policy"
          - "aws_iam"
          - "palo_alto_firewall"
          - "modsecurity_waf"
        """
        prompt = self.NORMALIZE_TEMPLATE.format_messages(
            source_format=source_format,
            policy_text=policy_text,
            format_instructions=self.parser.get_format_instructions(),
        )

        # Ollama 调用 — 本地 Mistral-7B，数据不出域
        response = ollama.chat(model=self.model, messages=[
            {"role": m.type, "content": m.content} for m in prompt
        ])

        # PydanticOutputParser 自动验证并解析 JSON → PolicyIR
        # 如果 LLM 输出格式错误，自动重试（最多3次）
        return self.parser.parse(response['message']['content'])
```

</details>

<details>
<summary>Rust 策略评估器代码</summary>

```rust
// policy_engine/evaluator/src/lib.rs
/*
高吞吐策略评估器 — Rust 实现。必须 <1ms P99，Python GIL 无法保证此延迟。
架构: Python (归一化) → Redis (策略存储) → Rust (评估)
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use rustc_hash::FxHashMap;  // 比标准 HashMap 快 2-3x，无 HashDOS 风险

/// 编译后的策略 — 加载时一次性编译为优化结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPolicy {
    pub id: u64,
    pub name: String,
    pub severity_threshold: Severity,  // 检测结果需达到的最低严格性
    pub required_entities: HashSet<String>,  // 必须存在的实体类型
    pub file_type_filter: Option<String>,    // 可选的文件类型过滤
    pub action: Action,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    None = 0, Low = 1, Medium = 2,
    High = 3, VeryHigh = 4, Critical = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action { Allow, Alert, Block }

/// 策略评估器 — 本 should_evaluate_file 是系统中调用最频繁的函数
pub struct PolicyEvaluator {
    /// 用 FxHashMap（rustc-hash）替代标准 HashMap
    /// FxHashMap 使用 FX Hasher（基于移位和乘法），对于整数 key
    /// 比 SipHash（标准 HashMap）快 2-3x。网络安全场景不需要 HashDOS 防护。
    file_rules: FxHashMap<u64, CompiledPolicy>,
    prompt_rules: FxHashMap<u64, CompiledPolicy>,
}

impl PolicyEvaluator {
    /// 从 Redis 加载策略并编译 — Python 的归一化器写入 Redis 后触发重载
    pub fn reload_from_redis(&mut self, redis_conn: &mut redis::Connection) -> Result<()> {
        // Redis GET "policies:file_rules" → Vec<CompiledPolicy>
        // 编译: HashSet 构建 + 排序
        let rules: Vec<CompiledPolicy> = redis_conn.get("policies:file_rules")?;
        self.file_rules = rules.into_iter().map(|r| (r.id, r)).collect();
        Ok(())
    }

    /// 文件策略评估 — P99 < 50μs (仅条件匹配，无 IO)
    pub fn evaluate_file(
        &self,
        severity: Severity,
        entities: &HashSet<String>,
        labels: &[String],
        file_type: &str,
    ) -> EvaluationResult {
        // 机密标签优先判定 — O(1) 检查
        if labels.iter().any(|l| l == "Company Confidential") {
            return EvaluationResult::block("confidential_label");
        }

        // 遍历规则 — 使用 FxHashMap 的 values() 迭代器
        // 规则数量通常 < 100，线性扫描足够
        for rule in self.file_rules.values() {
            // 严格性检查
            if severity < rule.severity_threshold { continue; }

            // 实体类型检查 — HashSet::is_subset O(min(|A|, |B|))
            if !rule.required_entities.is_subset(entities) { continue; }

            // 文件类型过滤
            if let Some(ref ft) = rule.file_type_filter {
                if !file_type.to_lowercase().contains(&ft.to_lowercase()) { continue; }
            }

            // 所有条件满足 → 执行动作
            match rule.action {
                Action::Block => return EvaluationResult::block(&rule.name),
                Action::Alert => return EvaluationResult::alert(&rule.name),
                Action::Allow => continue,
            }
        }

        EvaluationResult::allow()
    }
}
```

</details>

---

## 专利六、七 & 八：US20240362301A1, US20250068701A1 & WO2024224367A1

### 基于聚类的数据对象分类（母案 + 续案 + PCT）

| 属性 | US20240362301A1 | US20250068701A1 | WO2024224367A1 |
|:------|:----------------|:-----------------|:----------------|
| **公开日** | 2024-10-31 | 2025-02-27 | 2024-10-31 |
| **申请日** | 2023-04-27 | 2024-10-29 (延续申请) | 优先权日: 2023-04-27 |
| **发明人** | Yotam Segev + 7人 | 同左 | 同左 |
| **页数** | 20页 | 20页 | 50页 |
| **状态** | 审查中 | 审查中 | PCT 国际阶段 |

> WO2024224367A1 技术内容与 US20240362301A1 相同，不重复分析。

---

### 1. 核心技术创新

这两份专利描述了 Cyera DataDNA 聚类分类技术的完整流程：

**元数据替换 → O(n) 聚类 → 抽样分类 → 统计传播**

这是 Cyera 声称"数周内完成 PB 级数据分类"的技术基础。

### 2. 语言选择分析

聚类分类引擎有三个关键的性能约束：

| 约束 | 描述 | 影响 |
|:------|:------|:------|
| **线性时间复杂度** | 聚类必须 O(n)，处理数十亿对象 | 需要无锁哈希表 + SIMD |
| **常数额外空间** | 不能将整个数据集加载到内存 | 需要流式处理 + 外存算法 |
| **高并发抽样分类** | 每个聚类抽样后进行独立 ML 推理 | 需要 GPU 批处理或多核并行 |

| 语言 | 无锁哈希 | SIMD | GPU推理 | 评价 |
|:---------|:--------|:----|:-------|:-------|
| **Rust** (聚类引擎) | ⭐⭐⭐⭐⭐ dashmap + evmap | ⭐⭐⭐⭐ std::simd (nightly) | ⭐⭐⭐ ONNX Runtime | **聚类核心** |
| **C++** (向量索引) | ⭐⭐⭐⭐⭐ tbb::concurrent_hash_map | ⭐⭐⭐⭐⭐ AVX-512 | ⭐⭐⭐⭐⭐ CUDA/TensorRT | **FAISS 索引** |
| **Python** (ML推理) | N/A | N/A | ⭐⭐⭐⭐⭐ HuggingFace + vLLM | **抽样分类** |

**决策：Rust（聚类引擎 + 元数据归一化）+ C++/FAISS（十亿级向量索引）+ Python（抽样 ML 推理）。** 组件间通过 Apache Arrow Flight 传递数据，避免序列化开销。

### 3. 开源实现参考

| 组件 | 推荐方案 | 语言 | 理由 |
|:------|:---------|:------|:------|
| 十亿级聚类 | **FAISS** (GPU IVF+PQ) | C++ / Python API | Meta 维护，十亿级向量搜索的工业标准 |
| 流式元数据归一化 | **Rust + dashmap** | Rust | 无锁并发哈希表，O(n) 聚类 |
| 跨语言数据传输 | **Apache Arrow Flight** | C++/Rust/Python | 零拷贝列式数据交换，gRPC 传输 |
| 抽样 ML 推理 | **Ray Serve + vLLM** | Python | 支持动态批处理 + GPU 共享 |

### 4. 具体可实现方案

<details>
<summary>Rust 聚类引擎代码</summary>

```rust
// cluster_engine/src/normalizer.rs
/*
元数据归一化器 — Rust 实现。这是 O(n) 内循环，Python 中数十亿次 GIL 获取/释放不可接受。
Rust regex (DFA) 比 Python re (回溯) 快 10-50x，dashmap 提供无锁并发。
*/

use dashmap::DashMap;
use rayon::prelude::*;
use regex::RegexSet;
use std::sync::Arc;

pub struct MetadataNormalizer {
    // RegexSet 允许同时匹配多个正则，一次扫描完成
    patterns: RegexSet,
    replacements: Vec<&'static str>,
}

impl MetadataNormalizer {
    pub fn new() -> Self {
        Self {
            patterns: RegexSet::new([
                r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",      // IPv4
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", // UUID
                r"\d{4}-\d{2}-\d{2}",                           // Date
                r"0x[0-9A-Fa-f]+",                              // Hex
                r"\b\d+(\.\d+)?\b",                             // Numeric
            ]).unwrap(),
            replacements: vec!["<IPV4>", "<UUID>", "<DATE>", "<HEX>", "<NUM>"],
        }
    }

    /// 归一化 — 内循环，被调用数十亿次
    /// 关键优化: RegexSet 一次扫描匹配所有模式，O(n) 遍历输入
    pub fn normalize(&self, metadata: &str) -> String {
        let matches: Vec<_> = self.patterns.matches(metadata).into_iter().collect();
        if matches.is_empty() {
            return metadata.to_string();
        }
        let mut result = metadata.to_string();
        for idx in matches.into_iter().rev() {
            result = self.patterns
                .replace(&result, idx, self.replacements[idx]);
        }
        result
    }
}

/// 流式聚类器 — O(n) 时间，O(1) 额外空间（与数据集大小无关）
pub struct StreamingClusterer {
    clusters: DashMap<String, Vec<u64>>,  // normalized_key → [object_ids]
    // DashMap: 无锁分片哈希表，16 个分片
    // 写入: ~50ns (lock-free CAS)
    // 读取: ~20ns
}

impl StreamingClusterer {
    pub fn cluster_parallel(&self, objects: &[DataObject]) -> DashMap<String, Vec<u64>> {
        // rayon par_iter: 自动将工作分配到所有 CPU 核心
        // DashMap: 每个分片独立加锁 → 几乎线性的并行加速
        objects.par_iter().for_each(|obj| {
            let key = self.normalizer.normalize(&obj.metadata_str);
            self.clusters.entry(key).or_default().push(obj.id);
        });
        // 10 亿对象在 32 核上的预期耗时: ~60 秒（仅聚类，不含分类）
        self.clusters.clone()
    }
}
```

</details>

<details>
<summary>Python 抽样分类器代码</summary>

```python
# cluster_engine/sample_classifier.py
"""
抽样分类器 — Python 实现。ML 推理是 Python 无可争议的强项，vLLM 提供最高 GPU 利用率。
"""

import pyarrow as pa
import pyarrow.flight as flight
import numpy as np
from vllm import LLM, SamplingParams

class SampleClassifier:
    """
    对聚类抽样执行 ML 分类

    性能关键:
    - vLLM 的 continuous batching 自动合并多个请求 → 最大化 GPU 利用率
    - PagedAttention 管理 KV Cache → 避免 GPU 内存碎片
    """

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        # vLLM 初始化 — 自动检测 GPU 数量，Tensor Parallelism
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=4,   # 4 × A100
            max_num_batched_tokens=8192,
            gpu_memory_utilization=0.90,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,           # 确定性输出
            max_tokens=32,             # 仅输出分类标签
            stop=["\n", "</s>"],
        )

    def classify_samples(self, samples: List[Sample]) -> List[Classification]:
        """批量分类 — vLLM 自动合并为最优 batch size"""
        prompts = [self._build_prompt(s) for s in samples]
        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for output in outputs:
            label = output.outputs[0].text.strip()
            results.append(Classification(
                label=label,
                confidence=self._extract_confidence(output),
            ))
        return results

    def evaluate_cluster_consistency(
        self, classifications: List[Classification]
    ) -> Tuple[bool, Optional[Classification]]:
        """
        统计判定 — 对应专利的"真实聚类 vs 虚假聚类"判定

        使用 scipy.stats 进行统计检验 — scipy 底层 C/Fortran，性能无问题
        """
        from scipy import stats
        from collections import Counter

        # 广义化分类标签
        generalized = [generalize_label(c.label) for c in classifications]
        counts = Counter(generalized)
        majority = counts.most_common(1)[0]
        consistency = majority[1] / len(classifications)

        # 二项检验: null hypothesis = 分类结果随机（p=0.5）
        # 如果 p < 0.01，拒绝零假设 → 判定为"真实聚类"
        p_value = stats.binomtest(majority[1], len(classifications), p=0.5).pvalue

        if p_value < 0.01 and consistency >= 0.7:
            # 真实聚类 → 传播多数分类到聚类中所有对象
            return True, Classification(label=majority[0], confidence=consistency)
        else:
            # 虚假聚类 → 需要拆分重新处理
            return False, None
```

</details>

---

## 综合技术路线图

基于 8 件专利的分析，多语言分工如下（详细架构见下方「三场景统一技术方案全景」）：

| 语言 | 占比 | 负责领域 | 核心理由 |
|:------|:----|:---------|:---------|
| **Python** | ~40% | 云API编排、ML/NLP推理、LLM策略归一化 | boto3/spaCy/vLLM生态无可替代 |
| **Rust** | ~35% | 块设备IO、文件系统解析、聚类引擎、策略评估 | 零拷贝+内存安全+io_uring |
| **C++** | ~10% | 十亿级向量索引 | FAISS CUDA/AVX-512 |
| **Cython / SQL** | ~15% | spaCy底层C实现、存储查询 | 已存在于依赖中 / 标准SQL |

跨语言通信：Python ↔ gRPC ↔ Rust ↔ Arrow Flight ↔ C++，策略缓存走 Redis。

---

## 专利场景归并与统一技术方案

8 件专利按功能场景可归入**三个独立的技术方案簇**，每个簇内的专利具有高度互补性，可以合并为单一可落地的工程方案。

### 三方案划分的合理性评估

将 8 件专利划分为"数据发现"、"数据分类"、"策略管理"三个独立方案，而非合并为一个统一方案，其合理性需要从工程维度严格审视。

**核心判断标准**：两个组件是否应合并到同一方案，取决于它们在以下五个维度上的相似度。如果差异大到需要不同的扩展策略、不同的部署节奏或不同的故障域，则独立为优。

| 维度 | 场景一：数据发现 | 场景二：数据分类 | 场景三：策略管理 |
|:------|:---|:---|:---|
| **解决的问题** | "数据在哪里？" | "数据是什么？" | "数据能做什么？" |
| **调用频率** | 每天/每周（扫描周期） | 每次扫描时（批量） | 每次API请求/文件上传（实时） |
| **延迟要求** | 分钟-小时级 | 分钟级（批量） | **毫秒级（<1ms P99）** |
| **核心资源瓶颈** | 云API限速 + 磁盘IO带宽 | GPU显存 + CPU核心数 | CPU缓存 + 内存带宽 |
| **水平扩展方式** | 按云账户数扩展 | 按数据量扩展 | 按QPS扩展 |
| **一致性要求** | 最终一致 | 批量一致 | 强一致 |
| **故障影响** | 延迟发现问题（低风险） | 分类结果滞后（中风险） | 实时拦截失效（高风险） |
| **部署独立性** | 可独立启停，不依赖其他场景 | 依赖场景一的 DataAsset 作为输入 | 依赖场景二的 ClassificationResult 作为输入 |

上表揭示了一个关键事实：**三个方案在延迟要求上存在数量级差异**。场景三的策略评估是实时路径上的内循环——每次用户请求、每个文件上传都触发——必须在亚毫秒级完成。场景一是离线批量任务，云API调用可能耗时数秒，磁盘IO可能持续数分钟。如果强行合并到同一进程，会导致两个后果：

1. **离线慢任务可能阻塞实时快路径**：如果数据发现和策略评估共享线程池或事件循环，一次慢API调用可能使策略评估延迟飙升到秒级
2. **实时引擎被迫携带不需要的依赖**：策略评估器不需要 boto3、不需要 nbdkit、不需要 testcontainers——合并后却必须携带这些依赖，增加攻击面、内存占用和部署复杂度

**场景二与场景三的边界情况**：场景二的分类结果（`ClassificationResult`）是场景三策略评估（`PolicyEvaluator`）的输入。但这不是合并的理由——这恰是接口契约的典型场景。`ClassificationResult` 的数据结构一旦确定，两侧可以独立演进：场景二可以换用更快的分类模型，场景三可以增加更多策略规则，互不影响。

**结论**：三方案独立划分合理。场景一和场景二可以共享部分基础设施（云API适配层、文件系统解析库），但作为方案应保持独立。场景三必须完全独立部署，以保证实时路径的延迟确定性。交叉点通过接口契约（gRPC/Arrow Flight）解耦。

**一个替代划分值得讨论但最终不推荐**：将场景一和场景二合并为"数据智能层"。理由是两个场景都是批量任务（不需要实时延迟），共享数据流水线的上下游关系。但反对的理由更强——场景一的核心瓶颈是云API和磁盘IO，场景二的核心瓶颈是GPU和CPU。在云环境中，这两个场景的硬件需求截然不同（场景一需要IO优化实例，场景二需要GPU实例），合并后无法最优利用资源。

---

### 场景一：数据发现与扫描（3件专利 → 1个方案）

**涉及专利**：
| 专利 | 贡献 |
|:------|:------|
| US12026123B2 | 基础发现方法：快照扫描 + 两阶段过滤 + 引擎创建 |
| US12499083B2 | 引擎生命周期管理 + 多文件系统统一检测 + 按需版本匹配 |
| US12566567B2 | 惰性挂载 + 条件性克隆 + 直接块读取（性能核心） |

**可合并性分析**：

三件专利是同一发明人团队的同一专利家族（US12026123 → Continuation US12499083 → CIP US12566567），**天然设计为互补关系**：

- US12026123 定义了"做什么"——扫描快照发现数据存储
- US12566567 定义了"怎么做才快"——惰性挂载仅读1MB元数据
- US12499083 定义了"发现后做什么"——按需启动验证引擎

三者合并后的完整数据发现流水线：

```
                   ┌── US12566567 ──┐
                   │  惰性挂载       │
                   │  条件性克隆     │
                   │  直接块读取     │
                   └───────┬────────┘
                           │ 快速判定 (>90%磁盘在此被过滤)
                   ┌───────┴────────┐
                   │  US12026123    │
                   │  两阶段扫描     │
                   │  规则引擎评估   │
                   └───────┬────────┘
                           │ 高置信度命中
                   ┌───────┴────────┐
                   │  US12499083    │
                   │  临时引擎容器   │
                   │  版本精确匹配   │
                   │  轻量验证查询   │
                   └───────┬────────┘
                           │
                    最终数据资产清单
                    (type, version, size, row_count, is_active)
```

**合并后的统一架构**：

<details>
<summary>场景一合并架构</summary>

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Data Discovery Engine                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Python (编排层)                                            │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────────┐  │   │
│  │  │  AWS    │  │  Azure  │  │  GCP    │  │  On-Prem   │  │   │
│  │  │Provider │  │Provider │  │Provider │  │Provider    │  │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └─────┬──────┘  │   │
│  │       └─────────────┼─────────────┼────────────┘          │   │
│  │              CloudProvider Protocol                        │   │
│  │                     │                                      │   │
│  │              Celery Task Queue                             │   │
│  └─────────────────────┼──────────────────────────────────────┘   │
│                        │ gRPC                                     │
│  ┌─────────────────────┼──────────────────────────────────────┐   │
│  │ Rust (扫描核心)      │                                       │   │
│  │                     ▼                                       │   │
│  │  ┌──────────────────────────────────────────────────┐     │   │
│  │  │ Phase 1: LazyBlockReader (US12566567)              │     │   │
│  │  │  ├─ nbdkit bridge → 远程快照映射为本地块设备       │     │   │
│  │  │  ├─ io_uring 零拷贝读取 Superblock + InodeTable    │     │   │
│  │  │  ├─ 读取上限: 10MB / 磁盘                          │     │   │
│  │  │  └─ 结果: FsMetadata { fs_type, inodes, paths }   │     │   │
│  │  └──────────────────────┬───────────────────────────┘     │   │
│  │                         │                                  │   │
│  │  ┌──────────────────────┴───────────────────────────┐     │   │
│  │  │ Phase 2: RuleEngine (US12026123)                   │     │   │
│  │  │  ├─ rhai 嵌入式脚本 → 可配置检测规则                │     │   │
│  │  │  ├─ 多文件系统检测: ext4/XFS/NTFS/ReiserFS/ZFS    │     │   │
│  │  │  ├─ 已知数据库签名: MySQL/PostgreSQL/Mongo/Redis   │     │   │
│  │  │  └─ 输出: Confidence + DataStoreTypeGuess          │     │   │
│  │  └──────────────────────┬───────────────────────────┘     │   │
│  │                         │ confidence > 0.6                  │   │
│  │  ┌──────────────────────┴───────────────────────────┐     │   │
│  │  │ Phase 3: ConditionalClone (US12566567)            │     │   │
│  │  │  └─ 仅对高置信度磁盘克隆快照 → 挂载完整卷          │     │   │
│  │  └──────────────────────┬───────────────────────────┘     │   │
│  │                         │                                  │   │
│  │  ┌──────────────────────┴───────────────────────────┐     │   │
│  │  │ Phase 4: EngineValidator (US12499083)              │     │   │
│  │  │  ├─ testcontainers 启动临时数据库容器               │     │   │
│  │  │  ├─ 版本匹配: MySQL 8.0/5.7, PG 16/15/14, ...    │     │   │
│  │  │  ├─ 轻量查询: SHOW DATABASES + COUNT(*)            │     │   │
│  │  │  └─ 完成后立即销毁容器                              │     │   │
│  │  └──────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  输出: DataAsset { type, version, size_gb, table_count,          │
│                    row_count_estimate, is_active, orphaned }       │
└─────────────────────────────────────────────────────────────────┘
```

</details>

**合并后的核心接口**：

<details>
<summary>统一发现引擎接口代码</summary>

```python
# unified_discovery/engine.py
"""
场景一统一方案：合并 US12026123 + US12499083 + US12566567

三件专利的合并策略：
  US12566567 (惰性挂载) → 作为 Phase 1 的块读取后端
  US12026123 (规则引擎)  → 作为 Phase 2 的判定核心
  US12499083 (引擎验证)  → 作为 Phase 4 的可选深度确认

合并后的 API 只暴露一个入口: discover_all()
调用者不需要知道内部是 4 阶段流水线
"""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol
from enum import Enum

class DiscoveryDepth(Enum):
    """扫描深度 — 对应三件专利的不同覆盖范围"""
    # US12566567 覆盖: 仅元数据判定
    METADATA_ONLY = "metadata_only"
    # US12026123 覆盖: 元数据 + 规则深度分析
    RULE_DEEP = "rule_deep"
    # US12499083 覆盖: 元数据 + 规则 + 引擎验证
    ENGINE_VERIFY = "engine_verify"

@dataclass
class DiscoveredAsset:
    """统一输出 — 合并三件专利的所有输出字段"""
    # US12566567 贡献: 元数据
    disk_id: str
    fs_type: str                    # ext4 / xfs / ntfs
    size_gb: int

    # US12026123 贡献: 规则判定
    data_store_type: str            # mysql / postgres / mongodb / unknown
    version_hint: Optional[str]     # 8.0 / 5.7 / 16 / ...
    confidence: float               # 0.0 - 1.0
    is_active: bool                 # 是否挂载到活跃实例
    is_orphaned: bool               # 是否孤立（无挂载实例）

    # US12499083 贡献: 引擎验证（仅当 depth=ENGINE_VERIFY）
    verified: bool = False
    table_count: int = 0
    row_count_estimate: int = 0
    engine_version: Optional[str] = None

@dataclass
class DiscoveryConfig:
    """合并配置 — 控制三件专利各自的参数"""
    depth: DiscoveryDepth = DiscoveryDepth.RULE_DEEP

    # US12566567 参数
    max_metadata_bytes: int = 10_485_760       # 10MB
    deep_scan_threshold: float = 0.6

    # US12026123 参数
    rule_set: str = "default"                   # YAML 规则集

    # US12499083 参数
    engine_timeout_seconds: int = 30
    engine_max_concurrency: int = 10

class UnifiedDiscoveryEngine:
    """
    统一数据发现引擎

    合并三件专利后的单一入口：
      engine = UnifiedDiscoveryEngine(config)
      assets = await engine.discover_all(provider, depth=DiscoveryDepth.ENGINE_VERIFY)

    内部自动编排 Phase 1→2→3→4 的流水线
    """

    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.block_reader = LazyBlockReader()        # US12566567
        self.rule_engine = RuleEngine(config.rule_set) # US12026123
        self.engine_validator = EngineValidator(       # US12499083
            timeout=config.engine_timeout_seconds,
            max_concurrency=config.engine_max_concurrency,
        )

    async def discover_all(
        self, provider: CloudProvider
    ) -> List[DiscoveredAsset]:
        """
        完整发现流水线 — 合并三件专利

        阶段对应:
          Phase 1 (US12566567): 惰性挂载 + 元数据读取
          Phase 2 (US12026123): 规则评估 + 置信度判定
          Phase 3 (US12566567): 条件性克隆 (仅高置信度)
          Phase 4 (US12499083): 引擎验证 (仅 depth=ENGINE_VERIFY)
        """
        disks = await provider.list_disks()

        # 并发处理所有磁盘
        tasks = [self._discover_single(provider, disk) for disk in disks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if isinstance(r, DiscoveredAsset)]

    async def _discover_single(
        self, provider: CloudProvider, disk: Disk
    ) -> Optional[DiscoveredAsset]:
        """单个磁盘的 4 阶段流水线"""
        snap = await provider.create_snapshot(disk.id)
        try:
            # ── Phase 1: US12566567 ──
            metadata = await self.block_reader.read_metadata(
                provider.get_snapshot_blocks(snap.id),
                max_bytes=self.config.max_metadata_bytes,
            )
            if not metadata:
                return None

            # ── Phase 2: US12026123 ──
            result = self.rule_engine.evaluate(metadata)
            if result.confidence < self.config.deep_scan_threshold:
                return DiscoveredAsset(
                    disk_id=disk.id,
                    fs_type=metadata.fs_type,
                    size_gb=disk.size_gb,
                    data_store_type="unknown",
                    confidence=result.confidence,
                    is_active=disk.attached_to is not None,
                    is_orphaned=disk.attached_to is None,
                )

            # ── Phase 3: US12566567 ──
            clone_path = None
            if self.config.depth in (DiscoveryDepth.RULE_DEEP, DiscoveryDepth.ENGINE_VERIFY):
                clone_path = await provider.clone_snapshot(snap.id)

            # ── Phase 4: US12499083 ──
            verified = False
            table_count = 0
            row_count = 0
            engine_version = None

            if self.config.depth == DiscoveryDepth.ENGINE_VERIFY and result.data_store_type != "unknown":
                engine = self.engine_validator.create_engine(
                    result.data_store_type,
                    result.version_hint,
                    clone_path,
                )
                try:
                    stats = self.engine_validator.quick_stats(engine)
                    verified = stats.is_accessible
                    table_count = stats.table_count
                    row_count = stats.row_count
                    engine_version = stats.actual_version
                finally:
                    self.engine_validator.destroy_engine(engine)

            return DiscoveredAsset(
                disk_id=disk.id,
                fs_type=metadata.fs_type,
                size_gb=disk.size_gb,
                data_store_type=result.data_store_type,
                version_hint=result.version_hint,
                confidence=result.confidence,
                is_active=disk.attached_to is not None,
                is_orphaned=disk.attached_to is None,
                verified=verified,
                table_count=table_count,
                row_count_estimate=row_count,
                engine_version=engine_version,
            )
        finally:
            await provider.delete_snapshot(snap.id)
```

</details>

---

### 场景二：数据分类（4件专利 → 1个方案）

**涉及专利**：
| 专利 | 贡献 |
|:------|:------|
| US12299167B2 | 双路径混合分类 + 角色判定 + 虚假数据过滤 |
| US20240362301A1 | 聚类抽样 + 元数据替换 + 统计传播（母案） |
| US20250068701A1 | 聚类分类延续 + 多级替换 + 独立角色发现 |
| WO2024224367A1 | 同上 PCT 国际阶段（技术内容相同） |

**可合并性分析**：

四件专利覆盖了分类流水线的**两个正交维度**——US12299167 关注"如何分类一个数据值"（微观），聚类三件关注"如何在 PB 级数据上高效执行分类"（宏观）。

两者合并后形成完整的分类流水线：

```
输入: PB级数据集 (数十亿数据对象)
  │
  ├── [聚类层 — US20240362301 / US20250068701]
  │     元数据替换 → O(n)聚类 → 统计传播
  │     效果: 将数十亿对象压缩为数万聚类
  │
  ├── [抽样层 — US20240362301]
  │     从每个聚类抽样 5-10% 的代表性对象
  │
  ├── [分类层 — US12299167]
  │     对每个样本执行双路径分类:
  │       路径A: 真值表 (数值型)
  │       路径B: ML NER  (字符串)
  │     融合 + 角色判定 + 虚假过滤
  │
  └── [传播层 — US20240362301 / US20250068701]
        统计检验 → 真实聚类判定 → 分类传播到全量对象
        角色发现 → 自动创建新数据类型标签
```

**合并后的统一架构**：

<details>
<summary>场景二合并架构</summary>

```
┌─────────────────────────────────────────────────────────────────┐
│                  Unified Data Classification Engine                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  输入: 数据对象流 (结构化列值 + 非结构化文本片段)                  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Phase A: 流式聚类 (US20240362301 + US20250068701)          │   │
│  │                                                            │   │
│  │  ┌────────────────┐    ┌──────────────────┐              │   │
│  │  │ Rust           │    │ Rust (rayon)      │              │   │
│  │  │ Normalizer     │───▶│ StreamingClusterer│              │   │
│  │  │                │    │                   │              │   │
│  │  │ IP/UUID/Date   │    │ dashmap 无锁并发   │              │   │
│  │  │ → 占位符       │    │ O(n) time, O(1)   │              │   │
│  │  │ 数值→范围      │    │ extra space       │              │   │
│  │  └────────────────┘    └────────┬─────────┘              │   │
│  │                                  │                         │   │
│  │  输出: ~数万聚类 (key → [object_ids])                      │   │
│  └──────────────────────────────────┼───────────────────────┘   │
│                                     │                            │
│  ┌──────────────────────────────────┼───────────────────────┐   │
│  │ Phase B: 抽样 + 分类 (US12299167)│                        │   │
│  │                                  ▼                         │   │
│  │  对每个聚类:                                                │   │
│  │    sample = random.sample(cluster, ceil(|C| * 0.1))        │   │
│  │                                                             │   │
│  │  ┌───────────────────────────────────────────────────┐    │   │
│  │  │         HybridClassifier (Python)                   │    │   │
│  │  │                                                    │    │   │
│  │  │   ┌─────────────┐        ┌─────────────┐          │    │   │
│  │  │   │ Path A       │        │ Path B       │          │    │   │
│  │  │   │ TruthTable   │        │ ML NER       │          │    │   │
│  │  │   │ (pandas)     │        │ (spaCy trf)  │          │    │   │
│  │  │   │              │        │              │          │    │   │
│  │  │   │ 数值型数据    │        │ 字符串数据    │          │    │   │
│  │  │   │ SSN/CC/IBAN  │        │ 姓名/地址/   │          │    │   │
│  │  │   │              │        │ 企业名       │          │    │   │
│  │  │   └──────┬───────┘        └──────┬──────┘          │    │   │
│  │  │          └───────────┬───────────┘                 │    │   │
│  │  │                      ▼                              │    │   │
│  │  │              融合评分 (加权平均)                      │    │   │
│  │  │                      │                              │    │   │
│  │  │         ┌────────────┼────────────┐                │    │   │
│  │  │    ┌────┴────┐ ┌─────┴─────┐ ┌───┴──────┐         │    │   │
│  │  │    │角色判定  │ │虚假过滤   │ │LLM验证   │         │    │   │
│  │  │    │customer/ │ │test/mock/ │ │(歧义样本) │         │    │   │
│  │  │    │employee  │ │sequential │ │Mistral-7B│         │    │   │
│  │  │    └────┬────┘ └─────┬─────┘ └───┬──────┘         │    │   │
│  │  │         └────────────┼────────────┘                │    │   │
│  │  │                      ▼                              │    │   │
│  │  │          ClassificationResult                        │    │   │
│  │  │          { type, role, is_mock, confidence }        │    │   │
│  │  └───────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────┬───────────────────────┘   │
│                                     │                            │
│  ┌──────────────────────────────────┼───────────────────────┐   │
│  │ Phase C: 传播 + 角色发现          │                        │   │
│  │  (US20240362301 + US20250068701)  │                         │   │
│  │                                   ▼                         │   │
│  │  对每个聚类:                                                │   │
│  │    1. 统计检验 (scipy.stats.binomtest)                      │   │
│  │       p < 0.01 + consistency >= 0.7 → "真实聚类"            │   │
│  │    2. 传播: 真实聚类 → 多数分类应用于全量对象               │   │
│  │       虚假聚类 → 拆分子聚类 → 递归处理                       │   │
│  │    3. 角色发现: 跨列术语提取 → 候选角色 → 内容验证          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  输出: 全量分类结果 (每个数据对象的 type + role + confidence)     │
└─────────────────────────────────────────────────────────────────┘
```

</details>

**合并策略分析**：

<details>
<summary>聚类与分类协同分析</summary>

```
合并的关键洞察:

US20240362301 + US20250068701 (聚类) 解决 "WHERE to classify":
  → 哪些数据对象需要分类？不重复分类相似对象。
  → 聚类 → 抽样 → 传播：仅对 ~5% 的对象执行完整分类。
  → 10亿对象 × 5% 抽样 = 5000万次分类 (vs 10亿次)

US12299167 (分类) 解决 "HOW to classify":
  → 每个样本如何获得准确的分类结果？
  → 双路径 (真值表 + ML) → 角色判定 → 虚假过滤

两者互补，不是替代:
  - 没有聚类，US12299167 无法在 PB 级数据上运行
    (10亿次 ML 推理 × 10ms = 115 天)
  - 没有 US12299167，聚类后的分类精度无法达到 95%+
    (仅靠正则匹配 = 大量误报)

合并流水线的时间估算 (10亿对象):
  Phase A 聚类:         60s   (Rust + rayon, 32核)
  Phase B 抽样分类:     500s  (Python + vLLM, 4×A100, 5000万样本)
  Phase C 传播:         5s    (Python + scipy, 仅统计计算)
  ─────────────────────────
  总计:                ~10分钟  (vs 天级单独分类)
```

</details>

**合并后的核心接口**：

<details>
<summary>统一分类引擎接口代码</summary>

```python
# unified_classification/pipeline.py
"""
场景二统一方案：合并 US12299167 + US20240362301 + US20250068701

四件专利的合并策略:
  - US20240362301 (母案) → Phase A 聚类 + Phase C 传播
  - US20250068701 (续案) → Phase A 多级替换 + Phase C 独立角色发现
  - US12299167           → Phase B 双路径分类 + 角色判定 + 虚假过滤
  - WO2024224367          → 与 US20240362301 技术内容相同，不重复

合并后的 API 只暴露一个入口: classify_all()
内部自动编排 聚类→抽样→分类→传播 的完整流水线
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator
from enum import Enum
import numpy as np
from scipy import stats

class ClassificationMethod(Enum):
    TRUTH_TABLE    = "truth_table"       # US12299167 路径A
    ML_NER         = "ml_ner"            # US12299167 路径B
    CLUSTER_PROPAGATION = "cluster_propagation"  # US20240362301
    INDIVIDUAL     = "individual"         # 虚假聚类，逐个分类

@dataclass
class UnifiedClassificationResult:
    """统一分类输出 — 合并四件专利的所有字段"""
    # US12299167 贡献
    data_type: str                  # SSN / EMAIL / CREDIT_CARD / ...
    role: str                       # customer / employee / supplier / unknown
    is_mock: bool                   # 是否为测试/虚假数据
    confidence: float               # 0.0 - 1.0
    first_score: float              # 真值表评分
    second_score: float             # ML评分

    # US20240362301 贡献
    method: ClassificationMethod    # 分类方式
    cluster_id: Optional[str] = None
    cluster_consistency: Optional[float] = None  # 聚类一致性

    # US20250068701 贡献
    discovered_role: bool = False   # 是否为自动发现的新角色

@dataclass
class ClassificationConfig:
    """合并配置 — 控制各专利的参数"""
    # US20240362301 参数
    cluster_min_size: int = 5
    cluster_confidence_threshold: float = 0.7

    # US12299167 参数
    truth_table_path: str = "config/truth_table.parquet"
    llm_model: Optional[str] = "mistral:7b"   # LLM 验证 (歧义样本)
    llm_ambiguity_range: tuple = (0.3, 0.7)   # 触发 LLM 验证的置信度区间

    # US20250068701 参数
    multi_level_normalize: bool = True  # 多级替换


class UnifiedClassificationPipeline:
    """
    统一数据分类流水线

    四件专利合并后的单一入口:
      pipeline = UnifiedClassificationPipeline(config)
      results = pipeline.classify_all(data_objects)
    """

    def __init__(self, config: ClassificationConfig):
        self.config = config

        # Phase A 组件 — US20240362301 + US20250068701
        self.normalizer = MultiLevelNormalizer() if config.multi_level_normalize \
                         else MetadataNormalizer()
        self.clusterer = StreamingClusterer(
            normalizer=self.normalizer,
            min_size=config.cluster_min_size,
        )

        # Phase B 组件 — US12299167
        self.classifier = HybridClassifier({
            'truth_table_path': config.truth_table_path,
            'llm_model': config.llm_model,
        })

        # Phase C 组件 — US20240362301 + US20250068701
        self.role_discoverer = RoleDiscoverer()

    def classify_all(
        self, objects: Iterator[DataObject]
    ) -> List[UnifiedClassificationResult]:
        """
        完整分类流水线 — 合并四件专利

        顺序:
          Phase A (聚类):  US20240362301 + US20250068701 → Rust 流式聚类
          Phase B (分类):  US12299167                     → Python 双路径分类
          Phase C (传播):  US20240362301 + US20250068701 → 统计传播 + 角色发现
        """
        # Phase A: 聚类 — US20240362301 + US20250068701
        clusters = self.clusterer.cluster_parallel(objects)
        print(f"Phase A: {len(clusters)} clusters formed")

        # Phase B + C: 对每个聚类抽样分类并传播
        results = []
        new_roles = set()

        for cluster_key, obj_ids in clusters.items():
            cluster_objs = [self._resolve_object(oid) for oid in obj_ids]

            # ── Phase B: US12299167 ──
            sample_size = max(int(len(cluster_objs) * 0.1), 5)
            samples = cluster_objs[:sample_size]

            sample_results = []
            for obj in samples:
                result = self.classifier.classify(obj.value, obj.context)
                sample_results.append(result)

            # ── Phase C: US20240362301 ──
            generalized = [self._generalize_label(r.data_type) for r in sample_results]
            majority = max(set(generalized), key=generalized.count)
            consistency = generalized.count(majority) / len(generalized)

            # 统计检验
            p_value = stats.binomtest(
                generalized.count(majority), len(generalized), p=0.5
            ).pvalue

            if p_value < 0.01 and consistency >= self.config.cluster_confidence_threshold:
                # 真实聚类 → 传播
                representative = sample_results[generalized.index(majority)]
                for obj in cluster_objs:
                    results.append(UnifiedClassificationResult(
                        data_type=representative.data_type,
                        role=representative.role,
                        is_mock=representative.is_mock,
                        confidence=representative.confidence * consistency,
                        first_score=representative.first_score,
                        second_score=representative.second_score,
                        method=ClassificationMethod.CLUSTER_PROPAGATION,
                        cluster_id=cluster_key,
                        cluster_consistency=consistency,
                    ))
            else:
                # 虚假聚类 → 逐个分类
                for obj in cluster_objs:
                    r = self.classifier.classify(obj.value, obj.context)
                    results.append(UnifiedClassificationResult(
                        data_type=r.data_type, role=r.role,
                        is_mock=r.is_mock, confidence=r.confidence,
                        first_score=r.first_score, second_score=r.second_score,
                        method=ClassificationMethod.INDIVIDUAL,
                    ))

            # ── Phase C: US20250068701 ──
            discovered = self.role_discoverer.discover_from_cluster(cluster_objs)
            new_roles.update(discovered)

        # 后处理: 将发现的新角色应用于相关对象
        if new_roles:
            results = self._apply_discovered_roles(results, new_roles)

        print(f"Phase C: {len(results)} classified, {len(new_roles)} new roles discovered")
        return results
```

</details>

---

### 场景三：安全策略管理（1件专利 → 独立方案，无需合并）

**涉及专利**：
| 专利 | 贡献 |
|:------|:------|
| US12316686B1 | LLM 策略归一化 + 跨格式转换 + 迭代策略生成 |

本专利是 Trail Security 团队的独立发明，与其他 7 件 Cyera 核心专利**无重合**——它解决的是策略管理问题，而非数据发现/分类问题。技术方案已在专利五的章节中完整展开，此处不重复。

**但在 Omni DLP 产品中，场景三的策略引擎需要消费场景一和场景二的输出**：

```
场景一 (数据发现)     场景二 (数据分类)
       │                     │
       └─────────┬───────────┘
                 │
        DataAsset + ClassificationResult
                 │
                 ▼
        场景三 (策略引擎 — US12316686)
            │
            ├── 归一化: DLP策略 ←→ IAM策略 ←→ FW策略
            ├── 评估:   检测结果 vs 策略规则 → Block/Alert/Allow
            └── 同步:   转换后策略 → 各执行点 (Okta, Purview, AWS IAM)
```

---

### 三场景统一技术方案全景

<details>
<summary>三场景统一架构全景图</summary>

```
┌─────────────────────────────────────────────────────────────────────┐
│              Cyera DSPM — 三场景统一技术架构                           │
│                (8件专利 → 3个场景 → 1个平台)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 场景一: 数据发现 (3件专利)                                      │  │
│  │                                                                │  │
│  │ Cloud APIs ──▶ LazyBlockReader ──▶ RuleEngine ──▶ Validator  │  │
│  │ (Python/boto3)  (Rust/io_uring)   (Rust/rhai)   (Python/docker)│  │
│  │                                                                │  │
│  │ 输出: DataAsset[] { type, version, size, row_count, active }  │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                      │
│                               ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 场景二: 数据分类 (4件专利)                                      │  │
│  │                                                                │  │
│  │ DataObjects ──▶ Clusterer ──▶ Sampler ──▶ HybridClassifier   │  │
│  │   (stream)      (Rust)        (Python)     (Python/spaCy)      │  │
│  │                     │                                          │  │
│  │                     └──▶ Propagator ──▶ RoleDiscoverer        │  │
│  │                          (Python)         (Python)             │  │
│  │                                                                │  │
│  │ 输出: ClassificationResult[] { type, role, is_mock, conf }    │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                      │
│                               ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 场景三: 策略管理 (1件专利)                                      │  │
│  │                                                                │  │
│  │ 分类结果 ──▶ PolicyEvaluator ──▶ Action (Block/Alert/Allow)   │  │
│  │                 (Rust)                                         │  │
│  │                                                                 │  │
│  │ 多源策略 ──▶ PolicyNormalizer ──▶ PolicyIR ──▶ Translator     │  │
│  │               (Python/Ollama)                  (Python/Jinja2) │  │
│  │                                                                 │  │
│  │ 输出: Enforcement { action, reason, translated_policies[] }    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 跨场景共享基础设施                                              │  │
│  │                                                                │  │
│  │  Redis ──── 文件DNA缓存 + 策略热加载 + 特征标志                │  │
│  │  Arrow ──── 场景间零拷贝数据传输                                │  │
│  │  gRPC  ──── Python ↔ Rust ↔ C++ 跨语言通信                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

</details>

**合并收益总结**：

| 维度 | 合并前 (独立实现) | 合并后 (统一方案) |
|:------|:-----------------|:-----------------|
| 代码库 | 3 个独立服务，API 不一致 | 1 个 Discovery Pipeline + 1 个 Classification Pipeline |
| 数据传递 | 每个专利有自己的数据模型，需要手动映射 | Arrow Flight 零拷贝传递，统一 DataAsset/ClassificationResult |
| 配置管理 | 3 套配置文件和参数 | 1 个 UnifiedConfig，但保留 per-patent 的参数分组 |
| 语言比例 | 各语言碎片化 | Python ~40% / Rust ~35% / C++ ~10% / 其他 ~15%（Cython + SQL 等），清晰分工 |
| 部署单元 | 多个微服务 | 3 个场景服务 + 1 个共享基础设施层 |
| 可测试性 | 每个专利独立测试 | 场景内集成测试 + 场景间端到端测试 |
| 专利覆盖 | 3 个独立权利要求集 | 合并后不改变专利覆盖范围（技术实现与专利权利要求独立） |

---

## 补充分析：Cyera 平台能力与专利方案的缺口

将专利分析报告与 [Cyera 平台独立分析报告](../cyera_independent_analysis.md) 并置审视，可以发现**当前三方案仅覆盖了有专利支撑的平台能力，Cyera 的实际产品矩阵中存在两个专利未覆盖但工程上独立的能力层**。

### 平台能力 × 专利覆盖矩阵

| Cyera 平台能力 | 平台报告章节 | 对应专利 | 当前方案覆盖 |
|:---|:---|:---|:---|
| 数据发现与扫描 | 四 | US12026123 + US12499083 + US12566567 | ✅ 场景一 |
| DataDNA 分类引擎 | 三 | US12299167 + US20240362301 + US20250068701 + WO2024224367 | ✅ 场景二 |
| Omni DLP 策略归一化 | 六 | US12316686B1 | ✅ 场景三 |
| **DataGraph + 风险评估引擎** | **五** | **无** | ❌ |
| **Agent Graph + 身份图** | **五.2/五.3** | **无** | ❌ |
| **AI Guardian (AI-SPM + Runtime)** | **七** | **无** | ❌ |
| Omni DLP 实时数据运动监控 | 六.3 | **无** | ❌ |

### 可纳入的方案候选

#### 候选一：场景四 — 上下文图谱与风险评估（推荐纳入）

**解决的问题**："谁能访问这些数据？暴露面有多大？综合风险有多高？"

这是 Cyera 区别于传统 DLP 的核心能力层。DataDNA 分类告诉你"数据是什么"（场景二），但 DataGraph 回答的是后续关键问题——数据在谁手里、通过什么路径暴露、加密了吗、有备份吗、访问者身份可信吗。

**技术特征**：

| 维度 | 特征 |
|:------|:------|
| 核心数据结构 | 属性图（节点=数据资产/身份/资源，边=访问/所有权/血缘/依赖） |
| 查询模式 | 图遍历（"从这个S3桶出发，找到所有能访问它的外部身份"） |
| 评分算法 | 多维加权（数据敏感性 × 暴露程度 × 身份风险 × 合规要求 → 0-100） |
| 更新模式 | 持续增量（新数据资产发现 → 增量更新图 → 重新计算受影响节点的风险分） |
| 延迟要求 | 近实时（分钟级），非毫秒级 |
| 核心瓶颈 | 图数据库写入吞吐 + 评分计算的 CPU 密度 |

**与现有三方案的本质差异**：

- 场景一/二解决的是**实体属性**问题（"这个磁盘包含MySQL"、"这个值是SSN"）
- 场景四解决的是**实体关系**问题（"这个SSN被37个身份访问，其中3个已离职"）
- 技术栈完全不同：场景一二是流水线（Pipeline），场景四是图数据库+推理引擎（Graph+Inference）
- 可以独立部署为一个"风险评分服务"：消费场景一/二的输出（DataAsset + ClassificationResult），生产风险评分（RiskScore），供场景三的策略引擎消费
- **依赖链变化**：引入场景四后，场景三的直接输入从场景二的原始 `ClassificationResult` 变为经场景四增强后的 `RiskScore + ExposureContext`，不再直接依赖场景二

**为什么无专利**：图数据架构和风险评分算法可能属于 Cyera 认为更适合作为**商业秘密**保护的技术——算法参数和权重是核心竞争力，一旦在专利中公开，竞争对手可以直接复制。这与"可反向工程的"聚类分类方法（场景二专利）形成对比——后者一旦产品上市就可能被逆向，因此必须专利保护。

**开源参考**：

| 组件 | 方案 | 理由 |
|:------|:------|:------|
| 图数据库 | **Neo4j** / **JanusGraph** / **Amazon Neptune** | 属性图模型天然映射 Cyera 的实体-关系模型 |
| 图计算 | **Apache Spark GraphX** + **GraphFrames** | 大规模图的批量风险传播计算 |
| 评分引擎 | **Python + numpy/scipy** | 多维加权评分的快速原型和调参 |
| 图可视化 | **Cytoscape.js** / **G6 (AntV)** | 安全分析师的交互式风险图谱 |

---

#### 候选二：场景五 — AI 安全态势与运行时保护（暂不推荐纳入，但值得跟踪）

**解决的问题**："企业使用了哪些 AI 工具？它们能访问什么数据？Prompt 中是否泄露了敏感信息？"

**为什么暂不推荐纳入**：

1. **无专利支撑**：AI Guardian 发布仅 9 个月（2025年8月），相关专利申请极可能仍在 18 个月保密期内。没有专利说明书的技术细节无法做深度工程分析
2. **AI-SPM 与场景一高度相似**：AI 工具发现本质上是"数据发现"的特殊化——把"发现数据存储"换成"发现 AI 工具"，核心方法（无代理 API 扫描）相同。可能只是场景一的一个 Provider 扩展，而非独立方案
3. **Runtime Protection 与场景三高度相似**：实时 Prompt 检测→策略评估→Block/Alert 的链路与场景三的策略执行完全一致，只是检测对象从"文件上传"变为"Prompt 提交"。差异在检测器的类型（注入检测 vs PII 检测），不在架构

```
AI Guardian 的能力分解:

  AI-SPM (AI工具发现)        → 本质是场景一的特化: 扫描AI服务而非数据存储
  Runtime Protection (拦截)   → 本质是场景三的特化: Prompt检测 + 策略评估

  新增部分:
    Prompt Injection 检测      ← 新的检测器类型 (场景三的检测器扩展)
    Browser Shield              ← 新的执行点 (场景三的执行点扩展)
    数据血缘 (Data Lineage)     ← 跨方案的新能力 (可能属于场景四)
```

**建议**：在场景三的方案设计中预留检测器接口和执行点接口的扩展点，使 AI Guardian 的能力可以作为插件接入。不单独成方案。

---

#### 候选三：Omni DLP 实时数据运动监控（不单独成方案）

Omni DLP 中未被 US12316686 专利覆盖的部分——实时数据运动监控（endpoint/network/email/cloud/AI tools 的统一监控）——本质上是**场景三策略引擎的执行点矩阵**。

专利覆盖了"策略如何归一化和转换"（策略归一化器），未覆盖的是"策略如何在多个执行点上统一执行"。这个执行层与策略定义层的关系，类似于场景三内部的"PolicyEvaluator"（Rust 高吞吐评估）和"PolicyNormalizer"（Python LLM 归一化）的关系——同一个方案内的两个子组件。

**不单独成方案的原因**：策略执行是策略管理的自然延伸，不构成独立的技术问题域。

---

### 更新后的方案全景

```
                            ┌─────────────────────────┐
                            │   场景一: 数据发现        │
                            │   (3件专利)              │
                            │   "数据在哪里?"          │
                            │   批量/离线              │
                            └───────────┬─────────────┘
                                        │ DataAsset
                                        ▼
                            ┌─────────────────────────┐
                            │   场景二: 数据分类        │
                            │   (4件专利)              │
                            │   "数据是什么?"          │
                            │   批量/GPU               │
                            └───────────┬─────────────┘
                                        │ ClassificationResult
                                        ▼
┌───────────────────────────────────────────────────────────────┐
│                    场景四: 上下文图谱与风险评估  [NEW — 无专利] │
│                                                                │
│  "谁能访问? 暴露面多大? 风险多高?"                              │
│  近实时 / 图数据库+评分引擎                                     │
│                                                                │
│  DataGraph ──▶ 实体关系图谱                                     │
│  RiskEngine ──▶ 多维加权评分 (0-100)                            │
│  IdentityGraph ──▶ 人类/非人类身份映射                           │
│  Access Trail ──▶ 访问活动时间线                                 │
└───────────────────────────┬───────────────────────────────────┘
                            │ RiskScore + ExposureContext
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    场景三: 策略管理                             │
│                    (1件专利)                                    │
│                                                                │
│  "应该采取什么行动?"                                            │
│  实时 / 毫秒级                                                  │
│                                                                │
│  PolicyNormalizer ──▶ LLM驱动的多源策略归一化                    │
│  PolicyEvaluator  ──▶ 高吞吐策略评估 (Rust, <50μs)              │
│  PolicyTranslator ──▶ 跨平台策略转换                             │
│                                                                │
│  扩展点:                                                        │
│    ├── 检测器接口 → Prompt Injection 检测器 (AI Guardian)       │
│    └── 执行点接口 → Browser Shield (AI Guardian)               │
└───────────────────────────────────────────────────────────────┘
```

### 纳入建议总结

| 候选方案 | 是否纳入 | 理由 |
|:---|:---|:---|
| **场景四：上下文图谱与风险评估** | ✅ **推荐纳入** | 独立问题域（关系推理 vs 属性分类）、独立技术栈（图数据库 vs 流水线）、Cyera 的核心差异化能力、公开资料充分可做工程分析 |
| 场景五：AI 安全态势与运行时保护 | ❌ 暂不纳入 | 无专利支撑（可能在保密期）、AI-SPM 是场景一的特化、Runtime 是场景三的特化、独立价值不足以成单独方案 |
| Omni DLP 执行层 | ❌ 不单独成方案 | 是场景三的内部子组件，不构成独立技术问题域 |

---

## 对标策略总纲

### 核心判断：不二选一，两层次各司其职

"以平台能力对标"和"以专利方案对标"解决的是不同层次的问题。错误的做法是只取其一。

| | 以平台能力对标 | 以专利方案对标 |
|:---|:---|:---|
| **信息来源** | Cyera 官网、白皮书、Gartner/Forrester | USPTO/WIPO 专利说明书 |
| **信息可靠性** | 含营销成分，能力描述可能被夸大 | 法律文件，方法必须可实施才能授权 |
| **工程指导性** | 低——知道目标，不知道怎么实现 | 高——方法步骤、数据结构、参数范围全部公开 |
| **时效性** | 紧跟产品发布（0-3个月滞后） | 滞后 18 个月（专利申请到公开的保密期） |
| **覆盖范围** | 完整产品矩阵（含无专利的能力） | 仅覆盖有专利的能力 |
| **适用场景** | 产品规划、市场需求分析、竞品调研 | 架构设计、技术选型、模块边界定义 |

### 推荐操作：两层三阶段

```
        ┌──────────────────────────────────────────────┐
        │  第一层: 平台能力对标 (What / Why)              │
        │  面向: 产品规划 + 市场定位 + 客户沟通             │
        │  输入: Cyera 官网 + Gartner/Forrester + 财报    │
        │  周期: 季度/半年度更新                           │
        │  产出: 产品路线图、竞品对比表、市场需求文档        │
        └──────────────────┬───────────────────────────┘
                           │ 定义需求边界
                           ▼
        ┌──────────────────────────────────────────────┐
        │  第二层: 专利方案对标 (How)                     │
        │  面向: 架构设计 + 工程实现 + 技术选型             │
        │  输入: USPTO/WIPO 专利说明书 + 开源参考           │
        │  周期: 每个专利的深度分析融入技术架构              │
        │  产出: 模块接口定义、数据模型设计、性能基线        │
        └──────────────────┬───────────────────────────┘
                           │ 识别空白地带
                           ▼
        ┌──────────────────────────────────────────────┐
        │  第三层: 空白地带自主设计 (Differentiate)        │
        │  Cyera 无专利 / 商业秘密保护的能力               │
        │  场景四(风险评估引擎)等                          │
        │  产出: 原创算法设计、差异化技术护城河             │
        └──────────────────────────────────────────────┘
```

### 四方案 × 平台能力映射表

以专利方案的四个场景为技术架构基线，包装平台能力的市场语言：

| 技术方案（内部架构） | 市场对标语言 | 来源 |
|:---|:---|:---|
| 场景一 + 场景二的合并输出 | **AI 驱动的敏感数据发现与分类** | Cyera DSPM / DataDNA |
| 场景四 | **多维上下文关联与风险评分** | Cyera DataGraph + Risk Engine |
| 场景三（核心） | **统一数据安全策略编排** | Cyera Omni DLP |
| 场景三的扩展点（Prompt Injection 检测器、Browser Shield 执行点） | **AI 数据泄露实时防护** | Cyera AI Guardian |

**关键效果**：技术上专利方案提供了明确的实现路径和可验证的**不侵权边界**——你的实现若落在专利权利要求的字面范围之外即安全。市场上平台能力对标让你的产品描述与 Gartner DSPM Market Guide、Forrester Wave 评估标准对齐——客户和评估机构的语言系统不变。

### 更新节奏建议

1. **每季度**：跟踪 Cyera 产品发布（官网、BusinessWire），更新平台能力对标表
2. **每半年**：检索 Cyera 新公开的专利（USPTO 18个月保密期后集中公开），更新专利方案
3. **每 12-18 个月**：重新评估 AI Guardian 相关专利是否已公开——如果公开了完整的 AI-SPM 或 Runtime Protection 方法，则启动场景五的纳入评估

---

> **声明**：以上合并方案为技术架构层面的统一设计，不改变各专利的独立法律地位。每件专利的权利要求保护范围由其权利要求书界定，不受工程实现方式的影响。
