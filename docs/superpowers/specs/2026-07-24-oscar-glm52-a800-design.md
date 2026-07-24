# OSCAR × GLM‑5.2 × A800 集成设计

## 1. 文档目的

本文给出一套可分阶段验证、可回滚、可复现的实现方案，用于完成以下目标：

1. 在 vLLM 中集成 FutureMLS-Lab/OSCAR；
2. 将 OSCAR 从 full-attention K/V cache 扩展到 GLM‑5.2 的 DSA/MLA 架构；
3. 让剪枝版 GLM‑5.2 FP8 在单机 8×NVIDIA A800 80GB 上通过原生
   DSA/sparse MLA 路径运行；
4. 在功能与显存压缩验证通过后，使用固定精度套件对比原生 KV baseline；
5. 精度不满足硬阈值时，按预先定义的顺序优化，而不是放宽验收标准。

本文是设计与实施路线，不包含尚未实际执行的实验结果。后续实验数据必须在每个阶段
完成后及时写入中文实验报告，不得将计划值写成实测值。

## 2. 已确认的范围和决策

### 2.1 首阶段硬范围

- 硬件：单机 8×NVIDIA A800 80GB；
- 当前适配模型：
  `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-staticgate-e154-H001-nfs`；
- 模型特征：GLM‑5.2 FP8，专家数量减少 40%，当前 config 含 154 个 routed
  experts；
- 并行方式：TP=8；
- 服务形态：vLLM OpenAI-compatible API；
- attention：必须保持 GLM‑5.2 原生 DSA 与 sparse MLA；
- 必过上下文：`max_model_len=32768`，本文简称 32K；
- 扩展上下文：32K 全部通过后验证 `max_model_len=131072`，本文简称 128K；
- 执行模式：首阶段使用 eager execution；
- vLLM 基线：已验证 GLM‑5.2/A800 的定制 vLLM v0.19.0；
- OSCAR 压缩对象：MLA 的共享 latent KV，即 `compressed_kv`；
- 保持原精度的对象：RoPE key cache 和 DSA index/cache；
- 精度数据集：`/nfs/AE/txc/vllm_turbo_baseline_acc`；
- calibration 数据：必须与正式精度套件相互独立。

用户提供的当前模型在 GSM8K-full 上已知 accuracy 为 12.43%。该数值反映当前
40% 专家剪枝模型自身的较大精度损失。本阶段接受它作为工程适配模型，用于先打通
OSCAR × DSA/MLA × A800；不把该模型的低绝对精度归因于 OSCAR。

2026-07-24 已在当前环境只读确认上述目录可访问，且 `config.json`、权重索引、
tokenizer 与 141 个 safetensors 分片齐全。阶段 1 启动前仍须再次检查路径可读性和
轻量级文件指纹，防止 NFS 内容在设计完成后发生替换。

后续用户提供更高精度的剪枝模型时，将其视为新 checkpoint。即使 attention 几何
保持不变，也必须重新冻结原生 KV baseline，并重新生成或至少重新验证 calibration
artifact；禁止直接沿用当前模型的精度结论。

### 2.2 首阶段非目标

以下能力不进入首个完成定义：

- 现有 6 机 2P1D 拓扑；
- NIXL、PD disaggregation 和跨机 KV 传输；
- speculative decoding；
- KV offload；
- prefix cache；
- CUDA graph；
- 对 DSA index cache 或 RoPE key cache 量化；
- 生产级 202K 上下文；
- upstream vLLM 合入质量。

这些能力只能在单机 32K 的正确性、精度和压缩率全部通过后逐项恢复，不能与
DSA/MLA 首次适配同时推进。

## 3. 现有资产与关键事实

### 3.1 `oscar_vllm`

`oscar_vllm/vllm` 基于 vLLM v0.25.0，已包含较完整的 OSCAR full-attention
实现，涉及：

- OSCAR 配置和 rotation 加载；
- prefix/recent/history 三池管理；
- INT2 store、demotion、prefill/decode Triton kernel；
- KV cache spec、scheduler ownership 和请求生命周期；
- CPU/CUDA、端到端、GSM8K 和多请求验证。

已有报告显示该实现主要在 Qwen3 full attention 和 B200 上验证。它是本项目的算法、
三池设计和测试模式参考，但不能直接作为 GLM‑5.2/A800 的运行基线。

### 3.2 `glm52_speed_up_v2_stable_8th`

该目录实际是 GLM‑5.2 的 6 机 2P1D 部署包，不是完整源码仓库。现有清单记录：

- vLLM 版本线：定制 v0.19.0；
- 源码 commit：`bfd727e11b0e501bab0a4a943d92ba5ea3b2f980`；
- 镜像内源码：`/opt/vllm_glm52_v1`；
- 已验证镜像：
  `192.168.14.129:80/ae/vllm_openai_glm52:v0.19.0-v2-stable-2p1d-usagefix-20260625_114616`；
- 镜像 ID：
  `sha256:d6faf4d3a5f7f3800a745f8aea15884c881ea20b1100993d83ca8c4f985bd7a5`；
- 现有运行路径强制使用 sparse MLA/DSA，不允许 dense fallback；
- 现有部署包含大量 GLM 模型、FP8 MoE、DSA/MLA、PP 和性能定制。

因此，本项目不能把两个目录做普通 Git 合并，也不应把 GLM‑5.2 整体升级到 vLLM
v0.25.0。

### 3.3 OSCAR 上游

OSCAR 固定参考仓库为
[FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR)。本地固定
`main` commit 为：

```text
41ebcdba3db5f0ce1339c3727caea80df575d437
```

OSCAR `main` 的 INT2 路径面向 full-attention 模型。其 README 曾记录实验性
MLA 分支，但当前远端没有可直接依赖的公开 `glm-mla` head。因此，GLM‑5.2 的共享
latent rotation、cache 布局和 sparse MLA kernel 必须按本文设计实现并验证，不能把
未固定的实验分支作为交付依赖。

### 3.4 当前剪枝模型

首个适配 checkpoint 固定为：

```text
/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-staticgate-e154-H001-nfs
```

用户提供的已知结果是 GSM8K-full accuracy 12.43%。该结果只作为模型现状记录；
正式 baseline 仍须在本文冻结的 vLLM、prompt/template、生成参数和评测 runner 下
重跑。若重跑结果与 12.43% 不一致，应保留两者并分析配置差异，不能用口头已知结果
覆盖正式实验。

2026-07-24 的只读元数据核验结果如下：

| 项目 | 已核验值 |
| --- | --- |
| Architecture | `GlmMoeDsaForCausalLM` |
| Model type | `glm_moe_dsa` |
| Hidden size | 6144 |
| Transformer layers | 78 |
| Attention heads / KV heads | 64 / 64 |
| Q/K/V 几何 | `qk_nope=192`、`qk_rope=64`、`v=256` |
| Q/Shared-KV latent rank | 2048 / 512 |
| DSA index | 32 heads、head dim 128、top-k 2048、frequency 4 |
| Routed/shared experts | 154 / 1，单 token 选 8 个 routed experts |
| 权重量化 | FP8 E4M3、dynamic activation、128×128 block |
| 模型最大位置 | 1,048,576 tokens |
| 权重分片 | 141 个 safetensors |
| 索引声明/分片实际总字节 | 462,831,219,288 |

轻量级文件指纹：

| 文件 | SHA256 |
| --- | --- |
| `config.json` | `21a509ab82dad35a8584b724f41aa25c775a3c8f0c7604f7b10d21ee53e5f7fc` |
| `model.safetensors.index.json` | `e9767570b7b56aa97759c11aec7a81fbe7b84f56d751977788b0195c3a35d983` |
| `tokenizer_config.json` | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| `tokenizer.json` | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |

上述核验没有对约 463GB 权重逐分片计算内容 hash。阶段 1 应保存分片文件名、大小、
mtime 和索引 hash 作为快速指纹；若需要长期归档或跨存储复制验收，再异步计算完整
分片 hash manifest。

## 4. 路线选择

### 4.1 采用路线：GLM 基线优先，分层回移 OSCAR

以已验证的 GLM v0.19.0 代码线为主线，按以下三层回移 OSCAR：

1. OSCAR 配置、rotation artifact 和三池 allocator；
2. 共享 latent rotation 的 calibration/reference correctness；
3. 面向 SM80 的 Triton store、demotion 和 mixed sparse MLA kernel。

该路线最大限度保留现有 GLM‑5.2、FP8 MoE、DSA/MLA 和 A800 能力。v0.25 的
KV cache 接口只能作为设计参考，适配层应按 v0.19 的 scheduler/worker 契约重写。

### 4.2 不采用的路线

**OSCAR v0.25 主线前移 GLM 改动**会同时迁移模型、MoE、DSA/MLA、调度器和
分布式路径，导致现有 A800 验证失效，风险过高。

**镜像内旁路注入 kernel**无法可靠接管 KV allocation、ownership、preemption 和
回收，容易形成“INT2 kernel 被调用、完整 BF16 cache 仍然存在”的伪压缩，只能作为
一次性实验，不能作为最终交付。

## 5. 目标架构

建议创建独立工作代码线：

```text
/nfs/AE/txc/oscar-glm/glm52_oscar_vllm
```

现有 `oscar_vllm` 和 `glm52_speed_up_v2_stable_8th` 保持为只读参考。新代码线分为
四个边界清晰的单元。

### 5.1 GLM baseline 单元

职责：

- 恢复并冻结完整 GLM v0.19.0 源码；
- 构建可复现 Docker 镜像；
- 加载剪枝版 GLM‑5.2 FP8；
- 在 A800 TP=8 上证明原生 DSA/sparse MLA baseline 可用；
- 产出后续所有比较使用的唯一 baseline。

该单元不依赖 OSCAR。

### 5.2 Calibration 单元

职责：

- 采集共享 latent、吸收后的 query、DSA 选择结果及 attention 相关统计；
- 构建兼顾 K-score 和 V-output 的共享 covariance；
- 计算每层共享 rotation 和 clipping 参数；
- 导出带完整指纹的 calibration artifact；
- 提供 PyTorch reference/oracle。

该单元只负责离线统计与 artifact，不负责 serving cache 分配。

### 5.3 OSCAR KV 管理单元

职责：

- 计算三池容量；
- 管理请求级 prefix/recent ownership；
- 管理 paged INT2 history；
- 将物理地址和 block metadata 传给 worker；
- 处理 finish、abort、preemption、reuse 和容量失败回滚；
- 证明不保留完整 BF16 latent 副本。

### 5.4 SM80 attention/kernel 单元

职责：

- latent rotation、clipping 和 INT2 pack；
- BF16 prefix/recent store；
- recent-to-history demotion；
- INT2 history dequant；
- mixed sparse MLA prefill/decode；
- 全局 online softmax；
- history value accumulator 的 inverse rotation；
- A800/SM80 的 cold compile、launch 和 profiling。

该单元不能依赖 H100/B200 专属 FA3 或 SM90 指令。

## 6. 共享 latent OSCAR 数学设计

### 6.1 为什么不能复用两套 K/V rotation

full-attention 为每个 token 分别保存 K 和 V，因此 OSCAR 可以分别训练
`R_K` 和 `R_V`。MLA 只保存一份 `compressed_kv`，它同时参与 K-score 和
V-output。如果为 K、V 各存一份旋转后的 latent，会破坏 MLA 的共享缓存优势并增加
显存。因此，每层只能使用一份共享 rotation。

### 6.2 等价变换

对每层共享 latent `c` 和正交矩阵 `R`：

```text
c_rot = c · R
q_rot = q_abs · R
score = q_rot · c_rotᵀ = q_abs · cᵀ

z_rot = softmax(score) · c_rot
z = z_rot · Rᵀ
output = z · W_UV
```

在没有量化误差时，该路径必须与原生 MLA 数学等价。

reference path 先显式执行 query rotation 和 output inverse rotation。完成数值验证
后，才允许评估把 `R` 吸收到 query/key up-projection 与 value up-projection。

GLM‑5.2 权重为 FP8。若采用 weight absorption，必须生成新的派生权重 artifact，
记录原始 checkpoint、rotation 和重编码参数的指纹，并单独验证。禁止原地修改用户
checkpoint。

### 6.3 共享 calibration 目标

每层构造两类 latent-space covariance：

- score-sensitive covariance：衡量 latent 方向对 `q_abs · cᵀ` 的影响；
- value-sensitive covariance：衡量 latent 方向对
  `softmax(score) · c` 和最终 value output 的影响。

两者先做尺度归一化，再以固定权重合成为共享目标矩阵：

```text
Σ_shared = α · normalize(Σ_score) + (1 - α) · normalize(Σ_value)
```

对 `Σ_shared` 做特征分解，并沿用 OSCAR 的正交 eigenbasis、Hadamard 和 PBR
组合思想。`α`、clipping 和窗口只能在独立 calibration train/holdout 上确定。
首轮固定搜索 `α ∈ {0.25, 0.50, 0.75}`，clip ratio 使用
`{0.92, 0.94, 0.96, 0.98, 0.99}`。只有当最优点位于搜索边界时才扩展网格，且扩展
决策仍不得使用 official_v4。

### 6.4 初始量化配置

- group size：128；
- data：INT2；
- quantization：非对称；
- metadata：每组 FP32 `scale + zero_point`；
- prefix：64 tokens，BF16；
- recent：256 tokens，BF16；
- history：共享 rotation + clipping + INT2；
- clipping：每层一套共享参数；
- 初始 clip 搜索范围：0.92–0.99；
- DSA index cache：保持原精度；
- RoPE key cache：保持原精度。

当前模型的 `kv_lora_rank=512`，因此 group size 128 恰好把每个共享 latent
划分为 4 个完整 quant groups，不需要尾组 padding。该结论仍需与实际 vLLM TP=8
下的 latent tensor 布局核对；不能仅凭 Hugging Face config 推断 runtime shard。

窗口、group size 和 per-layer clip 是显式配置，但正式精度评测使用的候选配置必须先
在 calibration holdout 上冻结。

### 6.5 Artifact 合约

每份 rotation artifact 至少记录：

- 模型 config hash；
- checkpoint/shard hash；
- 剪枝专家映射 hash；
- calibration 代码 commit；
- calibration manifest SHA256；
- 随机种子；
- 层号与总层数；
- latent rank；
- group size；
- rotation format version；
- covariance 合成权重；
- clipping 参数；
- prefix/recent 配置。

层缺失、shape 不符、模型指纹不符或 format version 不支持时，服务必须拒绝启动。

## 7. KV cache 布局与数据流

### 7.1 每层物理布局

每层保留原生辅助缓存：

1. RoPE key cache：原精度；
2. DSA index/cache：原精度。

`compressed_kv` 使用三个独立物理池：

1. BF16 prefix pool：每请求固定 64 tokens；
2. BF16 recent ring：每请求固定 256 tokens；
3. paged INT2 history pool：按 block table 增量分配。

三池必须独立计费。不能通过给每个 INT2 page 添加固定大 padding 的方式隐藏
prefix/recent 成本，也不能同时保存完整 BF16 history。

### 7.2 Prefill/write

1. 原生模型生成 `compressed_kv`、RoPE key 和 DSA index；
2. RoPE key、DSA index 按原逻辑写入；
3. token 位于 prefix 时写 BF16 prefix；
4. 其余最新 token 写 BF16 recent ring；
5. recent token 变旧时，在 Triton kernel 中执行
   `rotation → clipping → quantize → pack`；
6. packed latent 写入 history page；
7. recent slot 被后续 token 覆盖，不保留 BF16 副本。

chunked prefill 的最终 token 分区必须与非 chunked prefill 一致。

### 7.3 Decode

1. DSA indexer 使用原精度 index cache 生成 top-k token IDs；
2. 按 token 逻辑位置将选中项映射到 prefix、recent、history；
3. BF16 token 使用原始 `q_abs`；
4. history token 使用 `q_abs · R`；
5. mixed sparse MLA kernel 直接读取三池；
6. history INT2 在寄存器中反量化；
7. 各 pool/split 计算局部 max、sum 和 accumulator；
8. 用统一 LSE 合并为全局 online softmax；
9. history accumulator 乘 `Rᵀ` 恢复到原 latent 空间；
10. 与 BF16 accumulator 合并后进入原生 value up-projection。

允许多 stage Triton 实现，但 serving 路径不允许创建与完整序列长度成比例的 BF16
临时 cache。PyTorch 全量反量化只用于测试 oracle。

### 7.4 Scheduler 与生命周期

scheduler 为每个请求显式维护：

- prefix row/slot range；
- recent row/ring state；
- history block IDs；
- 当前 logical sequence length；
- demotion 进度；
- cache generation/version。

worker 只消费 scheduler 下发的 ownership，不能用易变化的 batch row 作为持久地址。
三池与原生辅助缓存必须在 finish、abort、preemption 和 reuse 时保持生命周期一致。

## 8. 配置与失败策略

新增独立 cache dtype：

```text
oscar_mla_int2
```

启用后必须验证：

- 模型属于已支持的 GLM‑5.2 MLA 几何；
- TP shard 与 rotation shard 一致；
- native DSA/sparse MLA 已启用；
- rotation artifact 完整匹配；
- A800/SM80 kernel 可用；
- cache 容量足够；
- 不兼容能力已关闭。

出现不兼容时 fail closed。禁止自动回退到 BF16、dense MLA 或 full attention。
需要 baseline 时，用户必须显式选择原生 KV cache 配置。

启动日志和 metrics 至少提供：

- DSA/sparse MLA backend；
- rotation artifact hash；
- prefix/recent/group/clip 配置；
- 三池实际 bytes 和 slots；
- RoPE/index cache bytes；
- INT2 store、demotion 和 read 的调用计数；
- theoretical/padded/allocated compression ratio；
- 当前是否存在任何 BF16 history 副本。

## 9. 分阶段实施方案

每个阶段只有在出口条件全部满足后才能进入下一阶段。

### 阶段 0：恢复并冻结源码

步骤：

1. 优先读取现有外部完整源码路径；
2. 若外部源码不可用，从已验证镜像 `/opt/vllm_glm52_v1` 提取源码；
3. 核对 commit、镜像 ID、关键文件 SHA256 和原生 `.so`；
4. 建立独立 Git 仓库；
5. 记录所有运行时 patch、native extension 和环境变量；
6. 创建可从固定基础镜像复现的 Dockerfile；
7. 在不改代码的条件下重建 baseline 镜像。

出口条件：

- 完整源码可追溯；
- 关键文件与已验证镜像一致；
- baseline 镜像可重复构建；
- 依赖版本和原生扩展清单已冻结。

若完整源码无法恢复，停止开发，不基于零散 patch 猜测重建。

### 阶段 1：GLM‑5.2/A800 原生 baseline

步骤：

1. 挂载并读取
   `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-staticgate-e154-H001-nfs`；
2. 核对 attention、latent、RoPE、DSA、原生 KV dtype 和 expert pruning 几何；
3. 验证 FP8 scale、tokenizer 和 generation config；
4. 在用户授权的 A800 GPU 上运行 TP=8；
5. 依次验证短请求、超过 320 tokens 请求、连续 decode 和 32K 请求；
6. 记录显存、KV capacity、启动时间和请求结果；
7. 冻结 official_v4 与 WikiText‑2 baseline。

出口条件：

- native DSA/sparse MLA 明确生效；
- 32K 服务稳定；
- 无请求失败；
- baseline 精度、PPL、环境和命令完整保存。

这里的 baseline “通过”指服务、请求和评测流程有效且结果可复现，不要求当前剪枝
模型达到某个绝对 accuracy。已知的 GSM8K-full 12.43% 不阻塞适配；如果 baseline
运行存在请求失败、配置不明或结果不可复现，则不进入 OSCAR 适配。

### 阶段 2：Calibration 与 PyTorch reference

步骤：

1. 加入只读 activation capture hook；
2. 构建与 official_v4 不重叠的 calibration manifest；
3. 开发阶段先使用 30K–100K tokens 验证流程；
4. 最终 rotation 至少使用约 1M calibration tokens；
5. 计算 score/value covariance 和共享 rotation；
6. 在独立 holdout 搜索 covariance 权重和 clip；
7. 导出带完整指纹的 artifact；
8. 实现共享 latent PyTorch reference。

出口条件：

- `RᵀR≈I`；
- 未量化 rotation path 与原生 MLA 等价；
- INT2 pack/unpack 误差在预定义数值界限内；
- mixed prefix/recent/history 与 reference 一致；
- DSA index 输入未被修改。

### 阶段 3：三池 CacheSpec 与 CPU allocator

步骤：

1. 根据实际模型几何建立 bytes/token 公式；
2. 实现纯 CPU capacity planner；
3. 实现请求级 prefix/recent allocator；
4. 实现 paged history allocator；
5. 接入 v0.19 scheduler metadata；
6. 接入 worker tensor allocation；
7. 添加容量守恒和回滚检查。

出口条件：

- 单请求和多请求 ownership 正确；
- finish、abort、preemption、reuse 全部容量守恒；
- OOM 分配失败不泄漏 slots/blocks；
- 三池 tensor 边界与 planner 计费完全一致。

### 阶段 4：A800/SM80 Triton kernels

按以下顺序开发：

1. rotation + clipping + INT2 pack；
2. BF16 prefix/recent store；
3. recent demotion；
4. history dequant oracle；
5. mixed sparse decode；
6. mixed sparse prefill；
7. global LSE/accumulator merge；
8. inverse rotation；
9. TP=8 shard 支持。

每个 kernel 必须先通过：

- 全新 Triton cache 的 SM80 cold compile；
- A800 实际 launch；
- 与 PyTorch oracle 的 shape、finite 和数值误差比较；
- block、group、split 和边界长度测试。

SM90/B200 编译结果不能代替 SM80/A800 验证。

### 阶段 5：vLLM 接入与 32K 端到端

步骤：

1. 注册 `oscar_mla_int2`；
2. 接入 CLI/config/env；
3. 接入 GLM attention layer；
4. 接入 cache allocation 和 scheduler metadata；
5. 先单请求，再多请求；
6. 验证大于 320 tokens 时触发 demotion；
7. 验证 DSA top-k 与 mixed read 同时生效；
8. 运行接近 32K 的长请求；
9. 对比原生 KV 与 OSCAR 的 cache blocks 和显存。

出口条件：

- 所有请求 HTTP 200；
- 日志证明 DSA、三池 write/demotion/read；
- 无 dense/full-attention fallback；
- 无完整 BF16 latent history；
- 理论与实测容量满足第 10.3 节。

### 阶段 6：冻结候选镜像

所有定向测试和 32K 端到端通过后，构建不可变候选镜像，记录：

- image tag、ID 和 digest；
- Git commit；
- Dockerfile/base image；
- Python、PyTorch、Triton、vLLM、Transformers 和 CUDA 版本；
- native extension SHA256；
- rotation artifact SHA256。

后续精度与性能评测只使用该镜像。代码发生变化时生成新 tag，不覆盖旧镜像。

### 阶段 7：完整精度与 PPL

1. 重新验证 baseline artifact 与冻结结果一致；
2. 使用同一模型、prompt/template、生成参数、runner 和数据；
3. 运行 official_v4 四项 accuracy；
4. 运行 WikiText‑2 perplexity；
5. 合并并校验所有预测；
6. 生成总体、分 benchmark 和样本级 diff；
7. 按第 10.4 节判定。

### 阶段 8：精度优化

仅当阶段 7 不通过时执行第 11 节的固定优化顺序。每轮生成新配置 manifest、新镜像
或新 rotation artifact，不覆盖先前结果。

### 阶段 9：性能和 128K 扩展

精度通过后：

1. 运行固定性能矩阵；
2. 对超过阈值的性能回退做 profiling；
3. 完成必要 kernel 优化；
4. 重新执行数值与精度回归；
5. 在容量允许时验证 128K；
6. 输出 128K 可行或不可行的实际证据。

## 10. 验收设计

### 10.1 CPU/reference 测试

- artifact 正常、缺层、错层、错 shape、错模型指纹；
- rotation 正交性；
- 未量化等价变换；
- INT2 clip/pack/unpack；
- constant、窄分布、随机和 outlier 输入；
- `seq_len=63/64/319/320/321`；
- block boundary 和 partial block；
- chunked/non-chunked 最终分区一致；
- 单请求、多请求 ownership；
- finish、abort、preemption、reuse；
- capacity failure rollback。

### 10.2 CUDA/分布式测试

- 实际 latent rank 和 group size；
- 非连续 rotation stride；
- group/block/split 边界；
- mixed logits、LSE 和 output；
- single-token decode；
- multi-token prefill；
- batch 1/4/8；
- TP=8 rotation/cache shard；
- SM80 cold compile；
- A800 actual launch；
- 输出 finite；
- 与 PyTorch oracle 的最大和平均误差。

具体数值容差必须在阶段 2 根据 BF16/FP32 reference 和模型几何固定，不能在看到正式
精度结果后临时放宽。

### 10.3 显存和压缩率

分别报告：

1. 纯 BF16 latent bytes/token；
2. INT2 data bytes/token；
3. FP32 scale/zero metadata；
4. prefix/recent reserve；
5. page padding 和不可用 slots；
6. RoPE key cache；
7. DSA index/cache；
8. 整体 KV bytes/token。

硬阈值：

- 实际分配与理论计算误差不超过 5%；
- 实际 KV capacity/显存提升不低于 padding 后理论提升的 95%；
- 必须同时报告 latent-only 和 overall ratio；
- 不允许只用 history INT2 的 2-bit 数字代替整体压缩率。

### 10.4 精度和 PPL

固定套件：

```text
/nfs/AE/txc/vllm_turbo_baseline_acc/
  accuracy_suites/model_agnostic_accuracy_official_v4
```

accuracy 包括：

| Benchmark | 样本数 |
| --- | ---: |
| GSM8K | 1319 |
| IFEval | 541 |
| LiveCodeBench v6 | 175 |
| MultiPL‑E | 325 |
| 合计 | 2360 |

另运行 WikiText‑2 perplexity。

硬阈值：

- accuracy `2360/2360` 全部完成评分；
- request failure 为 0；
- overall accuracy 相对原生 baseline 下降不超过 1.5 个百分点；
- 任一单项下降不超过 3 个百分点；
- WikiText‑2 perplexity 相对上升不超过 3%。

当前工程适配模型已知 GSM8K-full accuracy 为 12.43%，但仍以阶段 1 在完全相同
环境下重跑得到的原生 KV 结果作为比较分母。OSCAR 没有最低绝对 accuracy 要求，
其硬门槛是相对同 checkpoint baseline 的退化幅度。该规则用于隔离“剪枝损失”和
“KV 量化损失”，不表示 12.43% 满足最终业务模型的质量要求。

必须输出：

- Correct/Total 和 accuracy；
- baseline delta；
- 共同正确；
- 仅 baseline 正确；
- 仅 OSCAR 正确；
- 共同错误；
- request/status 分类；
- 完整 predictions SHA256。

### 10.5 A800 功能与性能

A800 功能硬验收：

- 单机 8 卡、TP=8；
- native DSA/sparse MLA；
- 32K 服务；
- SM80 kernel 实际 launch；
- 无 illegal instruction、CUDA error、OOM 或 fallback；
- 服务可稳定完成连续请求。

精度通过后运行性能矩阵：

- 输入长度：1K、8K、32K；
- batch：1、4、8；
- 指标：TTFT、TPOT、throughput、peak memory、kernel time；
- baseline/OSCAR 使用相同模型、并行配置和请求集；
- 每组先 warm-up，再执行多轮稳定测量。

性能提升不是首个功能里程碑的硬门槛，但任何超过 20% 的回退必须完成 profiling、
根因归类和书面说明后才能交付。

## 11. 精度不通过时的固定优化顺序

### 11.1 路径和实现审计

先排除实现错误：

- rotation 加载或 layer mapping 错误；
- `R`/`Rᵀ` 方向错误；
- clipping index 或 scale 下限不一致；
- prefix/recent/history 边界错误；
- softmax 不同 pool 的 LSE 合并错误；
- DSA token ID 到 physical slot 的映射错误；
- 隐式 BF16/dense fallback；
- TP shard 不一致。

### 11.2 Calibration 目标

在 calibration holdout 上调整：

- score/value covariance 权重；
- normalization 方法；
- calibration token 数；
- calibration 数据组成；
- per-layer clipping。

### 11.3 Mixed window

依次评估扩大 recent 或 prefix。每次必须重新计算：

- overall bytes/token；
- 32K/128K 容量；
- 精度；
- 性能。

### 11.4 量化粒度

比较 group size 128 与 64。group size 变小会增加 FP32 metadata，必须同时报告精度
收益和整体压缩率损失。

### 11.5 敏感层高精度

只允许基于 calibration 敏感度选择层：

- 保留 BF16；
- 使用 INT4；
- 使用更保守 clip。

不能基于 official_v4 的具体错题选择层。

### 11.6 Rotation absorption 与 kernel 融合

只有 reference path 精度通过后才评估：

- 把 query rotation 吸收到 projection；
- 把 inverse rotation 吸收到 value up-projection；
- 重新编码派生 FP8 权重；
- 融合 dequant、score、softmax 和 accumulator；
- 消除与 top-k 或序列长度成比例的临时张量。

每项性能优化后必须重跑数值、32K smoke 和精度回归。

## 12. 独立 Calibration 数据原则

正式精度套件不得用于：

- rotation 拟合；
- clip 搜索；
- covariance 权重搜索；
- mixed window 选择；
- group size 选择；
- 敏感层选择。

若没有现成 calibration 数据，建立固定的通用文本、数学和代码混合集：

- 开发 manifest：30K–100K tokens；
- 最终 manifest：至少约 1M tokens；
- train/holdout 固定拆分；
- 固定随机种子；
- 保存每个源文件、样本 ID 和 manifest SHA256；
- 排除与 official_v4 的已知重复。

official_v4 只在冻结候选里程碑上运行。精度失败可以做样本级诊断，但后续参数选择
必须回到 calibration holdout 或独立开发集完成。

## 13. 实验安全与可复现要求

执行 GPU 实验前：

1. 由用户明确授权可用 GPU 编号；
2. 在允许范围内检查进程和显存；
3. 只有连续两次检查为空闲的 GPU 才可分配；
4. 每次性能/profiling 固定卡数和卡号；
5. 长实验至少每 10 分钟记录进度；
6. 性能测试先 warm-up；
7. 所有实验在固定 Docker 容器内运行。

每个实验记录：

- Docker image/tag/digest；
- Git commit；
- Python 版本；
- PyTorch、Triton、vLLM、Transformers 等关键包版本；
- CUDA/driver；
- `CUDA_VISIBLE_DEVICES`；
- 完整 `nvidia-smi`；
- 模型路径和 checkpoint hash；
- rotation/calibration hash；
- 数据 manifest hash；
- 完整命令；
- stdout/stderr；
- 输出文件 SHA256。

## 14. Git 开发与同步规范

### 14.1 仓库职责

各仓库的职责固定如下：

| 仓库或目录 | 职责 | 稳定分支 |
| --- | --- | --- |
| `XiancaiTian/oscar-glm52-a800` | 设计文档、部署和评测脚本、实验结论、交付清单及 submodule 指针 | `main` |
| `XiancaiTian/vllm` | 已有 OSCAR-vLLM full-attention 实现和适配参考 | `oscar-vllm-v0.25.0` |
| `FutureMLS-Lab/OSCAR` | OSCAR 官方参考实现，只读固定版本 | 固定 commit |
| `glm52_oscar_vllm` | 阶段 0 恢复后建立的 GLM‑5.2 × OSCAR 实际开发代码线 | `main` |

`oscar_vllm/vllm` 和 `oscar_vllm/oscar_reference` 继续由主仓库以 submodule
固定到准确 commit。阶段 0 创建 `glm52_oscar_vllm` 独立仓库时，必须先确认其
GitHub 远端和稳定分支，再将其作为 submodule 纳入主仓库；不得在主仓库中复制一份
无法追踪来源的源码快照。

主仓库中的 submodule 指针只允许指向已经成功推送到对应远端的 commit。禁止提交
只能在本机访问的子仓库 commit，否则其他开发者和自动化环境无法复现。

### 14.2 分支策略

稳定分支只保存已验证、可复现的检查点。功能开发和实验必须使用独立分支：

```text
feat/glm52-model-load
feat/glm52-mla-kv-cache
feat/glm52-oscar-decode
feat/glm52-oscar-prefill
fix/a800-kernel-compat
exp/oscar-accuracy-tuning
```

若一次任务同时修改源码和主仓库文档，两个仓库使用相同的分支名，便于对应。一个
分支只承载一个明确主题；模型加载、allocator、kernel 和精度优化不得混在同一开发
分支中。

`main`、`oscar-vllm-v0.25.0` 以及后续 `glm52_oscar_vllm` 的稳定分支不得直接
承载大规模试验性开发。`exp/*` 分支允许保留未采用方案及其真实实验记录，但不得在
未通过验收时合入稳定分支。

### 14.3 Commit 时机和粒度

出现以下任一情况时必须 commit：

1. 完成一个独立的小功能或明确缺陷修复；
2. 新增或更新与改动对应的测试；
3. 形成一个可以运行的实验配置；
4. 准备开始长时间 GPU 实验；
5. 准备切换机器、切换任务或结束当日工作；
6. 即将执行 rebase、大规模重构或其他高风险操作。

每个 commit 只解决一个主题，并满足：

- 已检查 `git diff` 和 `git status`；
- 至少通过与改动直接相关的静态检查、单元测试或 smoke test；
- 不混入模型、镜像、日志、密钥和临时产物；
- 文档数据来自真实执行结果；
- commit message 能准确说明本次变更。

推荐使用以下 commit 类型：

```text
feat: add GLM-5.2 MLA cache specification
fix: correct MLA latent cache head dimension
test: add GLM-5.2 OSCAR decode coverage
perf: fuse OSCAR history dequant and attention
docs: record A800 smoke test results
chore: update GLM-5.2 submodule revision
```

提交时应明确列出文件，例如：

```bash
git add vllm/v1/kv_cache_interface.py tests/v1/core/test_oscar_kv_cache.py
git commit -m "feat: add GLM-5.2 MLA cache specification"
```

除非已经逐项检查全部未跟踪文件，否则不得使用无差别的 `git add -A`。

### 14.4 Push 时机

以下时间点必须将当前有效 commit 推送到远端功能分支：

1. 每完成一个可验证的开发检查点；
2. 每次开始长时间 A800 实验之前；
3. 每次获得需要保留的重要实验结果之后；
4. 每天工作结束之前；
5. 需要其他开发者或 AI agent 接手之前；
6. 准备创建 PR 或请求审查时。

尚未完成但确有远端备份需要的代码，只能推送到 `feat/*` 或 `exp/*` 分支，并使用
明确的 `wip:` commit。WIP commit 不得直接进入稳定分支；合并前应整理为可审查的
原子提交或在 PR 中 squash。

### 14.5 Submodule 固定同步顺序

修改子仓库时，必须遵循“先子仓库、后主仓库”的顺序：

1. 在子仓库功能分支完成代码、测试和 commit；
2. 将子仓库 commit 推送到其 GitHub 远端；
3. 确认远端能够解析该 commit；
4. 回到主仓库，提交新的 submodule 指针和对应文档；
5. 推送主仓库分支。

以现有 OSCAR-vLLM 子仓库为例：

```bash
git -C oscar_vllm/vllm add \
  vllm/v1/attention/backends/oscar_attn.py \
  tests/quantization/test_oscar.py
git -C oscar_vllm/vllm commit -m "feat: extend OSCAR attention backend"
git -C oscar_vllm/vllm push

git add oscar_vllm/vllm oscar_vllm/progress.md
git commit -m "chore: update OSCAR-vLLM submodule"
git push
```

禁止先提交主仓库指针、后补推子仓库；禁止让主仓库稳定分支指向子仓库的 WIP 或
本地-only commit。

### 14.6 实验版本冻结

每次正式 GPU 实验开始前必须：

1. 提交并推送本轮源码和配置；
2. 确认主仓库与实际开发子仓库没有未提交修改；
3. 记录主仓库 commit SHA；
4. 记录所有 submodule commit SHA；
5. 记录镜像 digest、模型和 calibration artifact 指纹；
6. 将这些版本信息写入实验目录或实验 manifest。

实验运行中发现必须修改代码时，应停止当前结果归档，创建新 commit 和新实验编号
后重新执行。禁止用修改后的未提交工作区继续覆盖原实验输出。

实验完成后，原始日志和大型产物保留在本地受控目录；关键指标、命令、环境、输出
SHA256 和结论写入中文报告，使用独立 `docs:` commit 提交并推送。代码 commit 和
实验报告 commit 应分开，保证源码变化与结果记录都可单独审计。

### 14.7 PR、合并和里程碑

功能分支只有满足对应阶段出口条件后才能创建合并 PR。合并前至少确认：

- 子仓库 commit 已推送；
- 相关测试和 A800 验证已通过，或明确记录未通过项；
- 主仓库 submodule 指针正确；
- 设计文档、进度和实验报告已同步；
- 没有模型、镜像、日志或凭据进入 Git；
- PR 描述列出验证命令、结果和已知限制。

涉及子仓库的合并顺序固定为：

```text
子仓库功能分支 → 子仓库稳定分支
              → 更新主仓库 submodule 指针
              → 主仓库功能分支 → main
```

阶段出口全部通过并冻结候选镜像后，应创建带注释的里程碑 tag。tag 至少关联阶段、
模型版本和候选序号，例如：

```text
phase1-a800-baseline-v1
phase5-oscar-mla-32k-v1
phase7-accuracy-pass-v1
```

tag 只能指向报告、submodule 指针和候选镜像信息均完整的主仓库 commit。

### 14.8 日常同步与安全规则

每次开始工作时先同步主仓库和 submodule：

```bash
git switch main
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

进入开发分支前，应先从最新稳定分支创建或 rebase。共享稳定分支禁止
`git push --force`。个人功能分支确需改写历史时，只允许使用
`git push --force-with-lease`，且必须确认没有其他开发者基于旧历史工作。

以下内容不得提交到任何 GitHub 仓库：

- 模型权重、rotation 大文件和 calibration 原始数据；
- Docker image tar、构建缓存和虚拟环境；
- 原始运行日志、完整 predictions 和临时 profiling 产物；
- `.env`、密码、token、私钥、VPN 配置和内部访问凭据；
- 未经确认允许公开的机器访问配置。

`.gitignore` 只是最后一道保护。每次 commit 前仍必须执行 `git status`、检查暂存
diff，并对新增文件做敏感信息和大文件扫描。

## 15. 交付物

最终交付至少包括：

1. `glm52_oscar_vllm` 独立源码仓库和 commit；
2. 可复现 Dockerfile；
3. 不可变 Docker image/tag/digest；
4. 源码恢复与镜像一致性审计；
5. shared-latent calibration/capture 工具；
6. rotation/clip artifact 和 manifest；
7. `oscar_mla_int2` cache dtype；
8. 三池 planner、allocator 和生命周期管理；
9. SM80 Triton kernels；
10. A800 TP=8 baseline/OSCAR 启动脚本；
11. CPU、CUDA、TP=8 和端到端测试；
12. 32K 显存/压缩率报告；
13. official_v4 与 WikiText‑2 报告；
14. 性能报告；
15. 精度失败时每轮优化记录；
16. 128K 扩展验证或容量阻塞证据。

所有报告使用中文。每完成一个阶段立即同步对应结果，不等待整个项目结束。

## 16. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| 完整 GLM 源码无法恢复 | 阶段 0 fail closed，不用零散 patch 猜测重建 |
| NFS 模型路径或内容发生变化 | 阶段 1 前复核目录、分片清单、大小和轻量级文件指纹 |
| 剪枝模型改变 attention 几何 | checkpoint/config/pruning 指纹不匹配即拒绝加载 |
| 后续替换更高精度剪枝模型 | 作为新 checkpoint 重跑 baseline、artifact 校验和精度 |
| shared rotation 精度不足 | calibration、clip、window、group 和敏感层按固定顺序优化 |
| FP8 weight absorption 引入额外误差 | 首先保留显式 rotation；派生权重单独生成和验证 |
| SM80 kernel 不支持 | PyTorch 只作 oracle；最终 serving 不接受全量反量化替代 |
| RoPE/index cache 稀释总压缩率 | 同时报告 latent-only 和 overall ratio，不夸大收益 |
| DSA top-k 与 cache 地址不一致 | token-ID-to-slot 定向测试和运行时断言 |
| 性能回退 | 精度通过后按固定矩阵 profiling，再做融合和访存优化 |
| 多项变量同时变化导致无法定位 | 每阶段冻结镜像、artifact、配置和结果 |

## 17. 完成定义

首个版本只有同时满足以下条件才算完成：

1. 完整 GLM v0.19.0 源码基线可追溯且镜像可复现；
2. 剪枝版 GLM‑5.2 FP8 在单机 8×A800、TP=8 上运行；
3. native DSA/sparse MLA 明确生效；
4. `max_model_len=32K` 稳定完成请求；
5. `oscar_mla_int2` 三池路径真实启用；
6. INT2 write、demotion、mixed read 和 shared rotation 有日志/指标证据；
7. 不存在完整 BF16 latent history 副本；
8. 理论与实测显存/容量满足压缩率阈值；
9. CPU、CUDA、TP=8、生命周期和端到端测试全部通过；
10. official_v4 与 WikiText‑2 满足精度/PPL 硬阈值；
11. 候选镜像、命令、环境、artifact 和报告可复现；
12. 超过 20% 的性能回退已完成 profiling 和归因。

128K 是首个版本完成后的扩展闸门，不阻塞上述 32K 完成定义。多机 2P1D、NIXL、
prefix cache、CUDA graph 和 DSA index cache 量化均应在后续独立设计中推进。

当用户替换为更高精度剪枝模型后，完成状态不会自动继承。新模型至少重新执行阶段 1
的 baseline、阶段 2 的 calibration/artifact、阶段 6 的候选镜像冻结和阶段 7 的完整
精度评测；若 config、latent rank、层数、TP shard 或剪枝映射发生变化，还必须回归
阶段 3–5 的 allocator、kernel 和端到端验证。
