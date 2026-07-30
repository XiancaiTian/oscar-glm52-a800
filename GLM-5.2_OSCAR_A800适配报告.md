# GLM-5.2 支持 OSCAR 并运行于 NVIDIA 苹果800 的适配报告

> 状态截点：2026-07-31
> 当前主仓库分支：`feat/glm52-model-load`  
> Stage 9 BF16 运行提交：`0918f3a4ee3ae17713ecf43679ec557d77e5fc39`
> 当前 OSCAR-vLLM 源码提交：`a94b1f640fe504be3d741a1070e43f806eaad894`

## 1. 报告范围与结论

### 1.1 报告范围

测试模型 checkpoint：
`/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001`。

本文记录两类事实：

1. 已实现并通过验证的适配；
2. 截至状态截点仍未完成的验收项。

报告中的性能、容量、精度和 PPL 数字均来自仓库已有实验报告或实际运行记录。

### 1.2 总体结论

为了让 GLM-5.2 支持 OSCAR 并在单机 8×苹果800-80GB 上运行，项目没有把面向 full-attention、基于较新 vLLM 的 OSCAR 实现整体合并进 GLM 代码线，而是
选择“GLM 基线优先、分层回移 OSCAR”的路线：

1. 保留已经验证的 GLM-5.2、FP8 MoE、DSA/MLA、TP=8 和 苹果800 运行路径；
2. 把 OSCAR 改造成适用于共享 latent KV 的单 rotation 方案；
3. 实现 BF16 prefix、BF16 recent、INT2 history 三段式 KV cache；
4. 将三段式所有权、页分配和生命周期接入 vLLM scheduler/worker；
5. 为 苹果800 的 SM80 架构实现并验证 Triton store、demotion、mixed sparse
   decode/prefill 和 inverse rotation；
6. 增加 fail-closed 运行时合约。

截至状态截点，已经证明：

- 当前模型可以在 苹果800 上以原生 `TRITON_MLA_SPARSE`、TP=8、32K 启动；
- OSCAR 三段式路径可以在同一模型上完成 TP=8 服务启动、31,996+64 tokens 请求和
  8 请求并发；
- 78 层实际执行 store、recent-to-INT2 demotion 和 DSA-selected mixed read；
- 没有完整 BF16 latent history，也没有 dense/full-attention fallback；
- 苹果800/SM80 kernel cold compile、实际 launch、数值 oracle 和 8 卡 rank-local
  回归已经通过；
- OSCAR 固定 256 题测试完成 256/256 scored、107 题正确，accuracy 为
  `0.41796875`，request failure 为 0。
- Stage 9 BF16 固定性能矩阵已完成 9/9 格，全部格点均为 3 轮正式测量、
  0 request failure，并通过逐 rank profiler 证据校验。

尚不能声称“最终适配全部完成”，原因是：

- Stage 9 OSCAR 完整同负载矩阵尚未完成；当前 decode 快路径的
  1K/batch1 定向结果仍比 BF16 回退，TTFT/TPOT 分别为
  `+1,322.8%/+31.9%`；
- 128K 候选扩展验证尚未完成。

## 2. 为什么不能直接复用原始 OSCAR

### 2.1 vLLM 与模型架构不一致

仓库中的参考 OSCAR-vLLM 基于较新的 vLLM，并主要服务 full-attention 模型；当前
GLM-5.2 运行基线则包含定制 vLLM、FP8 MoE、DSA 和 sparse MLA。直接整体升级或
普通 Git 合并会同时改变模型、调度器、分布式和 attention 路径，使已有 苹果800
验证失效。

因此，项目以 GLM 代码线为主，只分层迁移：

- 配置、artifact 和三段式 allocator；
- 共享 latent calibration/reference；
- SM80 Triton kernels；
- 必要的 scheduler、worker 和 attention 接线。

### 2.2 full-attention 的双 rotation 不适用于 MLA

full-attention 为每个 token 分别保存 K 和 V，原始 OSCAR 可以分别处理两套空间。
GLM-5.2 MLA 缓存的是共享 `compressed_kv`；同一 latent 同时参与 score 和 value
路径。若直接保存两套旋转后的副本，会抵消 KV 压缩收益。

本项目因此实现一份共享 rotation：

- history query 使用旋转后的 latent 空间计算；
- history value accumulator 通过 `Rᵀ` 回到原 latent 空间；
- score、value 和共享 latent covariance 联合参与 rotation 拟合；
- 每层只保存一份 `512×512` rotation。

### 2.3 单 rotation 为什么能保持 MLA 数学等价

设某层原始共享 latent 为 `c`，吸收投影后的 query 为 `q_abs`，拟合得到的正交
矩阵为 `R`，满足 `R·Rᵀ=I`。OSCAR history 先把 latent 旋转到新的正交基：

```text
c_rot = c · R
q_rot = q_abs · R
```

attention score 不变：

```text
q_rot · c_rotᵀ
= q_abs · R · (c · R)ᵀ
= q_abs · R · Rᵀ · cᵀ
= q_abs · cᵀ
```

因此，仅做正交旋转且不量化时，history token 的注意力分数与原生 MLA 完全一致。
value 聚合先在旋转空间得到：

```text
z_rot = softmax(score) · c_rot
```

再乘逆变换：

```text
z = z_rot · Rᵀ
  = softmax(score) · c · R · Rᵀ
  = softmax(score) · c
```

恢复到原始 latent 空间后，继续进入 GLM-5.2 原生 value up-projection。也就是说，
rotation 本身只是换了一组正交坐标，不改变未量化 MLA 的计算结果；近似误差来自
后续对 history 的 clipping 和 INT2 量化，而不是 rotation 等价变换。

### 2.4 如何把 INT2 history 接回 DSA/MLA

GLM-5.2 的 DSA indexer 仍使用原精度 index/cache 选择 top-k token ID，OSCAR
不改 DSA 的选择逻辑。适配发生在“选中 token 如何取 KV 并完成 MLA attention”：

1. scheduler 根据 token 的逻辑位置，把 DSA 选中的 ID 映射到 prefix、recent
   或 history；
2. prefix 和 recent 保存原始 BF16 latent，使用 `q_abs` 直接计算；
3. history 保存 `c·R` 的 INT2 表示，kernel 按 group 的 scale/zero-point 在
   寄存器中反量化，并使用 `q_abs·R` 计算；
4. RoPE key cache 保持原精度，三个分段都叠加同一条原生 RoPE score 路径；
5. 三个分段分别计算局部最大值、指数和 accumulator，再通过统一 LSE 合并成一个
   全局 softmax，而不是分别做三个独立 attention；
6. history accumulator 乘 `Rᵀ` 回到原始 latent 空间，再与 prefix/recent 的
   BF16 accumulator 相加；
7. 合并结果继续进入 GLM-5.2 原生 value up-projection。

这一设计同时保留了：

- MLA 只缓存一份共享 latent 的显存优势；
- DSA 的原始稀疏 token 选择语义；
- RoPE 与原生辅助缓存的精度；
- prefix/recent 的高精度边界信息；
- history 的 INT2 压缩收益。

### 2.5 苹果800 需要真实 SM80 验证

苹果800 的 compute capability 为 8.0。SM90/B200 上能编译或运行不能证明 苹果800
可用，因此本项目把以下项目设为独立硬门禁：

- 全新 Triton cache 上 cold compile；
- 苹果800 实际 kernel launch；
- 非连续 rotation stride；
- 跨页 INT2 store/dequant；
- mixed prefix/recent/history decode 与 prefill；
- 多请求 ownership 隔离；
- 8 张卡分别进行 rank-local cold compile。

## 3. 原生 GLM-5.2/苹果800 基线适配

在引入 OSCAR 前，先冻结同一模型 checkpoint 的原生 KV 对照，避免把模型加载、
DSA sparse MLA 或基础运行时问题误判为 OSCAR 问题。

### 3.1 固定环境

| 项目 | 实际值 |
|---|---|
| GPU | 8×苹果800-80GB |
| Driver / CUDA | 575.51.03 / 12.9 |
| 并行方式 | TP=8、PP=1 |
| 执行方式 | eager，关闭 CUDA graph |
| Attention backend | `TRITON_MLA_SPARSE` |
| Python | 3.12.13 |
| PyTorch / Triton | 2.11.0+cu129 / 3.6.0 |
| Transformers / Tokenizers | 5.8.1 / 0.22.2 |
| 权重 | 141 个 safetensors，463,045,186,640 字节 |

preflight 对 4,711 个 runtime 源码文件、7 个原生扩展、141 个权重分片、模型几何、
expert mapping 和评测输入进行了身份检查。

### 3.2 原生功能基线

141/141 个 checkpoint shard 全部加载。原生 KV cache 可用内存为 15.45GiB，
容量为 179,008 tokens。

| 用例 | prompt + completion | 结果 |
|---|---:|---|
| 短请求 | 21 + 64 | HTTP 200 |
| 超过 320-token 边界 | 506 + 5 | HTTP 200 |
| 连续 decode | 26 + 384 | HTTP 200 |
| 近 32K | 31,996 + 64 | HTTP 200 |

## 4. 共享 latent Calibration 与 Rotation Artifact

### 4.1 Calibration 数据采集（Capture）适配

这里的 Capture 指：在离线 calibration 阶段运行当前模型，并从各层内部只读采集
用于拟合 OSCAR rotation 的统计数据。采集对象不是模型最终输出，也不是线上服务
需要持续执行的功能；rotation artifact 生成后，正常推理只加载最终 artifact，
不再运行 Capture。

之所以需要专门适配，是因为原始 OSCAR 面向 full-attention，可以分别采集 K 和 V
并拟合两套 rotation；GLM-5.2 MLA 只缓存共享 `compressed_kv`，同一 latent 同时
参与 score 和 value 路径。因此，本项目需要从每层采集并按 TP=8 语义合并：

- 共享 latent 的 activation、covariance 和 reservoir 样本；
- DSA query/score 相关的 covariance 与样本；
- value 路径的 covariance 与样本；
- 用于独立选择 rotation 和 clipping 参数的 holdout 数据。

这条只读采集路径还要求：

- 仅在显式环境开关下启用；
- 逐层限额，避免无界保存 activation；
- 正确合并 TP=8 各 rank 的 score/value covariance；
- 共享 latent covariance 只保存一份；
- calibration 数据与正式 accuracy suite 独立；
- checkpoint、expert mapping、模型几何和 split 身份均参与校验。

当前模型的正式 Capture 结果：

| Split | 请求 | Prompt tokens | Capture 文件 | 大小 |
|---|---:|---:|---:|---:|
| Train | 256/256 HTTP 200 | 900,000/900,000 | 624/624 | 2,783,307,114 bytes |
| Holdout | 36/36 HTTP 200 | 100,000/100,000 | 624/624 | 13,272,777,066 bytes |

每个 split 均为 8 个 TP rank × 78 层。Holdout 的 624 个文件均包含 4,096 行
latent/query/value reservoir 和 `512×2048` DSA 样本。

### 4.2 Shared rotation 搜索

项目实现共享 covariance 合并、正交 basis、Hadamard/PBR 组合、clip 搜索和独立
holdout 选择。三个 alpha 的实测结果为：

| Alpha | 归一化 holdout loss |
|---:|---:|
| 0.25 | 0.026186010882512642 |
| 0.5 | 0.02872817461999513 |
| 0.75 | 0.032144202654983196 |

最终选择 `alpha=0.25`。78 层中，clip ratio 0.92 有 61 层，0.94 有 17 层。

## 5. 三段式 KV Cache 与 Scheduler 适配

### 5.1 三段式物理布局

每层 KV cache 被拆为：

1. BF16 prefix pool：每请求固定 64 tokens；
2. BF16 recent ring：每请求固定 256 tokens；
3. paged INT2 history pool：以 16-token page 增量分配。

当前 latent rank 为 512、group size 为 128。每层每个 history token 的物理开销为：

- packed INT2 data：128 bytes；
- scale/zero metadata：32 bytes；
- 合计：160 bytes。

history 相对 BF16 latent 的理论压缩率为 6.4×。该数字只描述 history data；
整体容量还必须计入 BF16 prefix/recent、BF16 RoPE 和原生 DSA/index cache。

### 5.2 写入与降级

写路径按以下顺序执行：

1. prefix token 写入 BF16 prefix；
2. 最新 token 写入 BF16 recent ring；
3. recent token 变旧时执行
   `rotation → clipping → INT2 quantize → pack`；
4. packed latent 和量化 metadata 写入 history page；
5. recent slot 允许覆盖，不保留完整 BF16 history 副本。

运行时特别保证先 demote 即将被覆盖的 recent token，再写入新的 recent 数据，
避免 ring overwrite 破坏 history。

### 5.3 读取与 attention

DSA 为每个请求产生 token IDs。OSCAR metadata 把选中 token 映射到 prefix、
recent 和 history：

- prefix/recent 直接读取 BF16 latent；
- history query 使用 `q_abs·R`；
- INT2 history 在 kernel 内反量化；
- 三个分区分别计算局部输出和 LSE；
- 使用全局 log-sum-exp 合并；
- history value accumulator 乘 `Rᵀ` 返回原 latent 空间；
- RoPE 维度保持 BF16，并使用独立 block table。

### 5.4 Scheduler、ownership 与生命周期

项目新增请求级：

- prefix row/slot range；
- recent ring state；
- history block IDs/page table；
- demotion positions、page IDs 和 offsets；
- stable request index 与 HP row；
- exact sequence lengths。

finish、abort、preemption、reuse 和分配失败回滚都必须同时维护三段式与原生辅助
cache。OSCAR history OOM 时回滚 length、page 和 version，避免 scheduler 状态
部分提交。

### 5.5 容量规划验证

14GiB、78 层、`max_num_seqs=16` 的 CPU planner 确定性结果：

| 指标 | 实际值 |
|---|---:|
| 总预算 | 15,032,385,536 bytes |
| usable blocks | 36,215 |
| logical token slots | 579,440 |
| 总分配 / 未使用 | 15,032,096,256 / 289,280 bytes |
| native logical token slots | 162,256 |
| 理论容量比 | 3.5711468297012128× |

`总分配 + 未使用 = 总预算` 精确成立。该表是 CPU planner 结果，不是 GPU 峰值
容量实测。

定向 CPU 回归为 145 passed、0 failed、26 CUDA skipped；26 个 skip 后续均在
苹果800 阶段显式执行。

## 6. 苹果800/SM80 Triton Kernel 适配

### 6.1 实现内容

新增/适配的 kernel 包括：

- shared latent rotation、clip、INT2 pack/store；
- BF16 prefix/recent store；
- recent-to-history demotion；
- history INT2 dequant；
- mixed sparse MLA decode；
- mixed sparse MLA prefill；
- DSA selected IDs 与 padding 屏蔽；
- RoPE 分数合并；
- global LSE merge；
- history accumulator inverse rotation。

### 6.2 GPU 0 cold-cache 结果

在独立空 Triton cache 上：

- 完整套件 114/114 passed；
- 26/26 CUDA 门禁实际执行；
- 0 failed、0 skipped；
- 耗时 77.63 秒；
- 独立复跑再次得到 114/114 passed。

### 6.3 八卡 rank-local 结果

GPU 0–7 各自绑定一个进程和独立空 cache，每卡执行 28 个节点：

- 总计 224/224 passed；
- 其中 208 次是实际 CUDA kernel 测试；
- 每卡耗时 54.18–55.08 秒；
- 每卡生成 316 个 Triton cache 文件，字节数一致。

### 6.4 真实 artifact oracle

正式 loader 读取当前 REAP artifact 的 layer 0，在另一份空 cache 上执行跨两个
16-token pages 的：

`rotation → INT2 store → dequant`

rotation oracle 与 dequant oracle 的最大绝对误差均为
`3.0994415283203125e-06`。

## 7. vLLM 运行时接入

### 7.1 显式 cache dtype

项目注册 `oscar_mla_int2` 作为显式 KV cache dtype。配置激活后，
`MLAAttention.get_kv_cache_spec` 根据模型几何创建三段式
`OscarMLAAttentionSpec`，而不是沿用标准单一 KV tensor。

### 7.2 Fail-closed 约束

首版只支持已验证组合。以下条件不满足时在服务启动或 cache 分配前拒绝：

- backend 必须为 `TRITON_MLA_SPARSE`；
- 必须是 eager；
- TP=8、PP=1；
- 关闭 vLLM prefix caching；
- 关闭 CUDA graph；
- 关闭 speculative decoding；
- 关闭 asynchronous scheduling；
- 不使用 context parallelism、dual batch overlap；
- 不使用 KV transfer 或 KV offloading；
- rotation artifact 和 runtime expectation 必须完整匹配；
- 不允许回退到 BF16 history、dense MLA 或 full attention。

这些限制是当前已验证边界，不代表相关能力永远不能支持。

### 7.3 TP=8/32K 端到端结果

当前 REAP checkpoint 的 OSCAR 服务完成 141/141 分片加载，每个 rank 的实际
三段式计划一致：

| 指标 | 每 rank 实际值 |
|---|---:|
| logical capacity | 637,632 tokens |
| BF16 prefix | 1,024 slots / 81,788,928 bytes |
| BF16 recent | 4,096 slots / 327,155,712 bytes |
| INT2 history | 637,696 slots / 9,964 pages / 7,958,446,080 bytes |
| BF16 RoPE | 6,366,756,864 bytes |
| native index/cache | 1,767,693,312 bytes |
| allocated capacity ratio | 3.5812365205× |

功能请求：

| 用例 | prompt + completion | 结果 |
|---|---:|---|
| 短请求 | 21 + 14 | HTTP 200 |
| 超过 320-token 边界 | 506 + 5 | HTTP 200 |
| 连续 decode | 26 + 384 | HTTP 200 |
| 近 32K | 31,996 + 64 | HTTP 200 |
| 并发 | 8 请求 | 8/8 HTTP 200 |

运行时日志明确记录：

- `BF16 history=absent`；
- three-pool write active；
- recent-to-INT2 demotion active；
- DSA-selected mixed read active；
- 78 层 store `42,666` 次，每层 547；
- 78 层 demotion `18,330` 次，每层 235；
- 78 层 read `42,666` 次，每层 547；
- 未发现 ERROR、Traceback、RuntimeError 或 ValueError。

这些证据排除了“INT2 kernel 被调用，但完整 BF16 history 仍保留”的伪压缩。

### 7.4 OSCAR 256 题测试结果

OSCAR 使用 8K、`reasoning_effort=high`、concurrency 16 完成固定 256 道
GSM8K 测试。结果来自运行
`20260730T0438Z_candidate_fast256_c16_docker_v3` 的原始
`summary.json` 和 `validation.json`：

| 指标 | OSCAR |
|---|---:|
| scored / total | 256 / 256 |
| 正确数 | 107 |
| accuracy | 0.41796875 |
| request failure | 0 |
| 截断数 / 截断率 | 128 / 0.5 |
| 平均 completion tokens | 4,194.87890625 |
| 总 completion tokens | 1,073,889 |
| 活跃耗时 | 21,964.45053267479 秒 |
| 吞吐 | 41.95870953516493 requests/hour |

候选单轮 validation 状态为 `passed`。结果文件 SHA256：

- `summary.json`：
  `7ee4372b83c24223aa62915dd082b9c756998412d137a4fb1001fad7997a3ba2`；
- `validation.json`：
  `ef85c6aba47645d5efff22c48d8bf8641a654940d96156f0c161b966cf3bbb30`；
- `predictions.jsonl`：
  `b2b1bba41edb6ee69b82961ae926b31c545a7644c9463f666750e3529bc0b16b`。

作为汇总参考，原生 BF16 同为 256 题时为 105 题正确、accuracy
`0.41015625`；OSCAR 汇总值多 2 题，即 `+0.0078125`（`+0.78125` 个百分点）。
但原生与 OSCAR 保存的协议指纹分别为
`183a499b39cc05e8c03c9cda5df0c908b2a441769822728f9d4dbee952bd0db0`
和
`5bc5f1a00a7c48e86baf8a4e1e2b52b17ebbf2f0321b319a6e27b8ca0a404718`，
且当前缺少原生逐题 predictions。因此，只能确认 OSCAR 单轮 256 题结果已经完成，
不能把净多 2 题解释为由 OSCAR 带来的可归因精度提升，也不能完成逐题翻转统计。

### 7.5 Stage 9 BF16 固定性能基线

BF16 正式轮次
`20260730T1342Z_stage9_baseline_v4` 使用固定控制镜像、TP=8、eager、
async scheduling 关闭、`max_model_len=131072`、
`max_num_batched_tokens=2048`、GPU memory utilization 0.92。固定矩阵为
1K/8K/32K 输入 × batch 1/4/8，每格输出 128 tokens、1 次 warm-up 后执行
3 轮正式测量，再单独采集 8 个 TP worker trace、8 张 CUDA table 和 1 个
frontend trace。下表为三轮 `mean_ttft_ms`、`mean_tpot_ms` 和 request
throughput 的中位数：

| 输入 | Batch | TTFT（ms） | TPOT（ms） | 吞吐（req/s） |
|---:|---:|---:|---:|---:|
| 1K | 1 | 352.445 | 156.705 | 0.04934 |
| 1K | 4 | 792.733 | 181.125 | 0.16749 |
| 1K | 8 | 1,292.619 | 185.086 | 0.32102 |
| 8K | 1 | 2,751.019 | 179.002 | 0.03925 |
| 8K | 4 | 4,968.413 | 217.451 | 0.12136 |
| 8K | 8 | 7,119.445 | 272.702 | 0.18775 |
| 32K | 1 | 12,528.026 | 178.832 | 0.02838 |
| 32K | 4 | 21,838.365 | 400.676 | 0.05392 |
| 32K | 8 | 80,081.107 | 419.976 | 0.05559 |

9/9 格的 cell status 与总 summary status 均为 `passed`，总矩阵时长为
`13495.650912761688` 秒，summary SHA256 为
`c0e312299bb6aba034bf01fd848197605c3763ac87e9cb367b58bf71abf4e2f5`。
32K/batch8 的服务端证据显示最多只有 4 个请求同时驻留，最多 7 个请求等待，
KV cache usage 峰值为 82.24%，因此其高 TTFT 包含 BF16 KV 容量导致的排队。
该现象将在 OSCAR 同负载轮次中按同一客户端并发配置直接比较。

矩阵和 profiler 全部完成后，容器退出清理阶段因 Bash `EXIT` trap 在局部
`wrapper_pid` 离开作用域后再次读取该变量而返回退出码 1。该错误发生在
`summary.json` 完整写入之后；容器已删除、GPU 0–7 均为 0 MiB，未污染上述
TTFT/TPOT。清理入口已增加未定义变量保护并在正常清理后撤销 trap。由于正式
比较器要求 BF16 与 OSCAR 的主仓库 commit 完全一致，OSCAR 配对轮次仍使用
已发布的 BF16 运行提交，清理修复将在配对实验完成后合入主功能分支。

### 7.6 OSCAR 首轮性能诊断

OSCAR 首轮正式性能运行
`20260730T1741Z_stage9_candidate_v1` 使用与 BF16 相同的主仓库提交
`0918f3a4ee3ae17713ecf43679ec557d77e5fc39`、源码提交
`065af88a010dc5746029198088ba01edc4a61516`、模型、控制镜像和服务/客户端
参数。运行前静态、发布身份及间隔 60 秒的两次 8/8 GPU 空闲检查均通过。

首个 `1K/batch1` 格点完成 3/3 正式轮次、0 request failure，并通过 8 个
rank table、8 个 worker trace 和 1 个 frontend trace 的完整证据校验。
cell summary SHA256 为
`306349dfde90733c010d7c69a4dfcbe3bc2cc2e50619c20c00463ffd272d3dc4`：

| 指标 | BF16 | OSCAR | 相对变化 |
|---|---:|---:|---:|
| TTFT（ms） | 352.445 | 5,049.520 | +1,332.7% |
| TPOT（ms） | 156.705 | 243.127 | +55.1% |
| 请求吞吐（req/s） | 0.04934 | 0.02783 | -43.6% |
| profiler critical-rank CUDA time（ms） | 26,939 | 33,048 | +22.7% |

上述三项主要服务指标均超过固定 20% 回退门限。同期
`1K/batch4` 已落盘的前两轮也分别观测到 TTFT
`13,534.416/11,807.347 ms`、TPOT `367.949/356.435 ms`，且均为
12/12 请求成功；但第三轮和该格 profiler 未完成，因此这些 batch4 数值只作为
方向一致的部分证据，不冒充完整格点结果。

首格逐 rank profiler 已把回退定位到 OSCAR 热路径，而不是请求失败、排队或
显存不足：

- BF16 `_sparse_mla_kernel_final_static` 平均为 `238.654 µs/层调用`；
  OSCAR `_mixed_sparse_decode_stage1` 为 `629.748 µs/层调用`，约慢
  `163.9%`；
- OSCAR `unified_mla_kv_cache_update` 的 CPU total 为
  `1.639 ms/层调用`。其中 `aten::nonzero` 共 29,952 次、CPU total
  `7.137 秒`，`aten::index` 共 110,004 次、CPU total `8.968 秒`；
  这些张量索引在 78 层逐层重复；
- OSCAR `_rotate_latent_kernel` 共 29,952 次，CUDA total
  `845.825 ms`；
- rank 0 的 NCCL all-reduce 平均从 BF16 `491.400 µs` 增至 OSCAR
  `984.264 µs`。它更符合各 rank 热路径变慢后同步等待被放大的结果，当前证据
  不支持把 NCCL 本身作为首要根因。

由于首个完整格点已明确超过门限且 profiler 足以进入优化，继续执行其余 8 个
未优化格点只会重复已确认的回退。该轮因此在 batch4 第三轮期间主动停止；没有
生成全矩阵 summary，也不作为最终候选结果。容器停止后 8 张 GPU 均为
0 MiB。下一步先消除 decode 时不可能命中的 current-history 布尔索引链，
把跨层不变的 OSCAR 元数据移出 78 层热路径，再对 mixed decode kernel 的
split 配置做苹果800 实测选择；优化后以新提交和新 run ID 重跑完整 9 格。

### 7.7 Decode KV update 第一轮优化

首轮优化提交
`98ddd3f4ef645bddec76d96cd86a11d17232aaa2` 已推送到
`feat/glm52-oscar-integration`。该提交只处理 profiler 已证明存在的 decode
重复索引，不提前修改 mixed decode kernel。

对纯 decode batch，每个请求本轮只产生一个新 token，其逻辑位置恒为
`seq_len - 1`。只要 recent window 非空，该位置不可能属于当前 history；
需要离开 recent 的旧 token 已由同一轮 demotion metadata 单独处理。实现据此：

- 直接从 `seq_lens` 得到 query position 和 final sequence length；
- 按请求顺序直接读取 HP row，避免逐层构造 token-to-request 索引；
- 跳过不可能命中的 current-history 布尔 mask、`nonzero` 和空张量索引链；
- 保留 prefill/chunked prefill 的通用 current-history 写入路径；
- 保留 recent-to-INT2 demotion、RoPE store 和 BF16 recent store 的原有顺序。

两请求 decode 回归同时覆盖不同 sequence length、不同 HP row 和仍有 demotion
的情形，并断言 history store 不得被调用。容器内 CPU 验证结果为：

- `test_runtime_cache_path.py`：10 passed；
- 完整 `tests/oscar_mla`：89 passed、26 CUDA skipped、0 failed；
- 26 个 skip 均由显式 GPU 授权门禁产生，本节不把它们冒充 GPU 通过；
- Ruff check、mypy、typos、SPDX、Python compile 和相关 pre-commit 门禁通过。

该源码已被重新封装为候选
`glm52-oscar-a800-phase6-98ddd3f4e-0275043c`。实际 OCI/Docker 身份为：

- image/config digest：
  `sha256:e90f4a84cd0aa958b86236d4d982a735f8d6a903c82168fee1093ff357bdcbf7`；
- manifest digest：
  `sha256:96455a6e4d0eaccde087957f56061c3442db15a9a9c7a8a60395b9c7764a69f9`；
- candidate layer digest：
  `sha256:badb252b2e916f4d84d23253b28d72d90e74c52c7d009f73bdcbb0ca17f4f538`。

验收实际核对 33 层、4,744 个源码文件、4 份 rotation artifact 和 7 个基础层
原生扩展；源码 Git tree 精确匹配
`536e0b9d05ea9d12e9fe805650d401f1ec37e8f9`，候选层不含原生扩展或
whiteout。Docker runtime import 复核 Python/PyTorch/Triton 为
`3.12.13/2.11.0+cu129/3.6.0`、rotation tensors 为 78，并确认
`reasoning_effort=max` 可解析、`cuda_initialized=false`。该构建/导入阶段没有
分配 GPU。

基于上述候选构建的新 Stage 9 控制镜像为
`oscar-glm-stage9-runtime:98ddd3f4e`，image ID 为
`sha256:f25d8d5ff9f5f3aee5f4b4f869e60c1242804a412e5839bcace74bc8ab8d40f8`。
Phase 1/5/7/9 的 source、OCI、overlay 和 Docker identity 已同步到该候选；
JSON、shell 语法以及 Stage 9 工具测试 15/15 通过。

containerized preflight
`20260730T1946Z_stage9_candidate_fastpath_preflight_v1` 在间隔 60 秒的两次
8/8 GPU 空闲检查后执行，64/64 静态检查通过。实际解析参数为
TP=8、`max_model_len=131072`、`max_num_batched_tokens=2048`、
`oscar_mla_int2`、eager、async scheduling 关闭和 torch profiler；
`cuda_initialized=false`。`static_preflight.json` SHA256 为
`9a5c11125c535772aa63954fb14528b6012ee2ad4c7d6ba818ae44d4eed0c920`。
preflight 容器退出后 8 张 GPU 均为 0 MiB、0%，没有 compute app。

本节只证明语义回归、源码发布和运行前门禁完成；实际 GPU 性能结果记录在 7.8。

### 7.8 Decode 快路径定向性能结果

定向轮次
`20260730T2000Z_stage9_candidate_fastpath_probe_1k_b1_v1` 使用主仓库提交
`a00c997350c13cd2c12c81ea4ea28a015741bd1d`、源码提交
`98ddd3f4ef645bddec76d96cd86a11d17232aaa2` 和 7.7 中冻结的候选/控制镜像。
它只选择固定矩阵的 1K/batch1 格点，但没有缩减单格协议：仍为 128 输出 token、
1 次 warm-up、3 轮正式测量及一轮 profiler。summary 显式标记
`scope=single_cell_probe`，未运行 128K，不能冒充完整矩阵。

运行前外层和容器内均完成间隔至少 60 秒的两次 8/8 GPU 空闲检查。141/141
权重分片加载完成后，3/3 正式轮次均为 0 request failure；服务端最多运行
1 个请求、等待为 0、preemption 为 0，因此该格点没有容量排队。8 个 rank
table、8 个 worker trace 和 1 个 frontend trace 均通过完整性与 SHA256 校验。
退出后容器删除，8 张 GPU 均回到 0 MiB、0%。总 summary 和 cell summary
SHA256 分别为：

- `a2c2a0fffe26fcb649840de3564d87cfb63582ea4d7fa764cb4f6ced3909c17a`；
- `96c1312a78a74c70ae53d5e6e8c80ac2ed4cf5c6abf032ddd3125d6b9fc4fba4`。

三轮 `mean` 指标的中位数对比如下：

| 指标 | BF16 | 未优化 OSCAR | Decode 快路径 | 快路径相对未优化 | 快路径相对 BF16 |
|---|---:|---:|---:|---:|---:|
| TTFT（ms） | 352.445 | 5,049.520 | 5,014.582 | -0.7% | +1,322.8% |
| TPOT（ms） | 156.705 | 243.127 | 206.735 | -15.0% | +31.9% |
| 请求吞吐（req/s） | 0.04934 | 0.02783 | 0.03196 | +14.9% | -35.2% |

逐 rank profiler 中位数证明优化命中了预期路径：

| profiler 项 | 未优化 | Decode 快路径 | 变化 |
|---|---:|---:|---:|
| `unified_mla_kv_cache_update` CPU total | 16.107 s | 9.293 s | -42.3% |
| `unified_mla_kv_cache_update` CUDA total | 1.080 s | 0.439 s | -59.3% |
| `aten::nonzero` 调用数 | 29,952 | 234 | -99.2% |
| `aten::index` 调用数 | 110,004 | 10,944 | -90.1% |
| `_mixed_sparse_decode_stage1` CUDA total | 6.270 s | 6.264 s | -0.1% |
| 1K prefill/首 token CUDA time | 5.034 s | 5.051 s | +0.3% |
| `_rotate_latent_kernel` CUDA total | 0.846 s | 0.843 s | -0.4% |

因此，decode 索引快路径已经由 GPU 实测证明有效：它显著减少逐层 KV update 的
索引工作，并把 TPOT 降低约 15%。但它没有改变 mixed decode kernel，也没有改善
1K prefill/首 token，所以 TTFT 基本不变，TPOT 仍超过 BF16 31.9%。split
参数实测结果记录在 7.9；之后仍需分别处理 mixed kernel 的结构性开销与
prefill/首 token：

1. 若固定 split 已无调优空间，继续分析 mixed kernel 的访存、反量化和 merge；
2. 单独剖析 1K prefill 的三段式写入/attention 路径，不能用 decode 优化结果
   解释 TTFT。

### 7.9 Mixed attention split 单卡实测

split sweep 的有效轮次为
`20260730T2051Z_oscar_mixed_split_sweep_1k_b1_v2`，使用已发布主仓库提交
`00dc17c11155143cf14794458737a94d0625acac`、源码提交
`98ddd3f4ef645bddec76d96cd86a11d17232aaa2` 和控制镜像
`sha256:f25d8d5ff9f5f3aee5f4b4f869e60c1242804a412e5839bcace74bc8ab8d40f8`。
实验固定使用 GPU 0 一张卡；运行前两次检查分别在
`2026-07-30T20:51:46Z` 和 `20:52:56Z` 完成，8 张卡均为 0 MiB、0% 且没有
compute process。退出后 GPU 0 回到 0 MiB。

实验形状对应 TP=8 下的 1K/batch1 decode：每 rank 8 heads、2,048 个 top-k
槽位、1,024 个有效 token，prefix/history/recent 为 `64/704/256`，
latent/RoPE 维度为 `512/64`。输入使用正式 layer 0 rotation；每个 split 先
warm-up 20 次，再交替正反顺序执行 7 组、每组 100 次完整 OSCAR mixed
attention 调用。CUDA event 计时包含 query rotation、mixed stage1、split
merge、inverse rotation 和输出合并。

有效结果如下：

| `num_splits` | CUDA 中位数（ms/调用） | 墙钟中位数（ms/调用） | 峰值 allocated（MiB） |
|---:|---:|---:|---:|
| 4 | 0.674417 | 0.674722 | 1.8330 |
| 8 | 0.375931 | 0.376241 | 1.9580 |
| 16 | **0.322437** | **0.322737** | 2.2080 |
| 32 | 0.322836 | 0.323158 | 2.7085 |

四种 split 的 output/LSE 均通过以 split 16 为基准的
`atol=rtol=0.002` 检查；所有输出有限。跨 split 的 output 最大绝对差不超过
`2.980232238769531e-07`，LSE 最大绝对差不超过
`4.76837158203125e-07`。有效 summary SHA256 为
`e96df05eed5af6d89fd7b6d47e7f0eacc0c8d856c4e5cf9041cab1fde7b774b5`。

首轮同名 v1 sweep 虽完成计时，但自审计发现误用了系统 Python 和 PyTorch
`2.10.0+cu129`，与正式服务的 PyTorch `2.11.0+cu129` 不同，因此整轮作废，
不参与上表或参数选择。v2 已 fail-closed 校验解释器为
`/opt/fp8_speed_up_v4_venv/bin/python`、PyTorch `2.11.0+cu129`、CUDA
runtime `12.9`。

结论是：当前默认 split 16 已是该固定形状的实测最优值。split 32 中位 CUDA
时间反而慢约 `0.12%`，并多占约 `0.50 MiB`；split 4/8 则明显更慢。因此本轮
不修改 split，也不能把参数扫描冒充为性能优化。后续应转向 prefill/首 token
和 mixed kernel 本身的结构性开销。

### 7.10 Prefill/首 token 的 8-rank trace 归因

为避免把 128 个 decode step 的累计时间误算到 TTFT，本轮没有重新占用 GPU，
而是流式解析 7.8 有效轮次已经冻结的 8 份 worker trace。分析工具提交
`4830508` 按每个 rank 的首个
`execute_context_1(1024)_generation_0(0)` 时间窗归集 kernel，并逐文件重新
记录 bytes 和 SHA256。有效分析目录为
`20260730T2112Z_fastpath_prefill_trace_analysis_v2`，summary SHA256 为
`5a32a7ca96dc46aec7214324df5c16a8e262aa3bd8a22ed9bc62ce6c2bd8a056`。

8 个 rank 均包含 129 个 `execute_context`。首个 1,024-token prefill 和窗口内
CUDA kernel 统计如下：

| 指标 | Min（ms） | Median（ms） | Max（ms） |
|---|---:|---:|---:|
| Prefill execute context | 4,997.066 | 4,997.096 | 4,997.264 |
| 窗口内 CUDA kernel 合计 | 4,948.995 | 4,952.601 | 4,957.278 |
| `_mixed_sparse_decode_stage1` | 4,574.163 | 4,576.783 | 4,604.965 |

mixed stage1 在每个 rank 都精确调用 78 次，即每层一次；其中位时间占 prefill
墙钟 `91.59%`，占窗口内 CUDA kernel 合计 `92.41%`。其余中位 kernel 总量明显
更小：

| Kernel/类别 | Calls 中位数 | CUDA 中位数（ms） |
|---|---:|---:|
| `_rotate_latent_kernel` | 233 | 105.583 |
| MoE 主 Marlin kernel | 148 | 80.307 |
| NCCL BF16 all-reduce | 155 | 66.869 |
| 通用 Marlin kernel | 330 | 28.711 |
| `_merge_mixed_splits_kernel` | 77 | 25.933 |

这证明 TTFT 回退不是 rank 7 偶发等待，也不是 rotation、MoE 或通信主导；根因
跨 8 个 rank 一致地集中在 OSCAR mixed stage1。代码路径同时表明，当前 prefill
仍沿用单 query decode 的默认 split 16，并对每个 query 固定扫描 2,048 个
top-k 槽位。对 1,024 个 query、每 rank 8 heads，这会启动
`1024×8×16` 个 stage1 programs；而 prefill 已经具有充足的 query/head
并行度。

因此，下一项实验必须把 prefill 与 decode 分开：保持 7.9 已证明最优的 decode
split 16，同时实测 prefill 较小 split，并在 `max_seq_len<2048` 时裁去不可能
有效的 top-k 尾部。只有 output/LSE 与完整 TP=8 TTFT 都通过后，才能把该方向
认定为优化。单卡参数选择结果见 7.11。

### 7.11 Prefill split 与有效 top-k 宽度单卡实测

有效轮次
`20260730T2116Z_oscar_prefill_sweep_1k_b1_v1` 使用已发布主仓库提交
`5276b60e4598f7c9959be1d3dba8b85978b386a9`、源码提交
`98ddd3f4ef645bddec76d96cd86a11d17232aaa2` 和 7.7 的固定控制镜像。
实验前在 `2026-07-30T21:16:50Z` 与 `21:17:56Z` 两次检查，8 张 GPU 均为
0 MiB、0% 且没有 compute app；实验固定只使用 GPU 0。正式解释器为
`/opt/fp8_speed_up_v4_venv/bin/python`，PyTorch/CUDA runtime 为
`2.11.0+cu129/12.9`。容器退出后 8 张 GPU 均回到 0 MiB。

输入复现 TP=8 下 1K/batch1 的 prefill 几何：1,024 个 query、每 rank 8 heads、
2,048 个 DSA top-k 槽位，prefix/history/recent 为 `64/704/256`，
latent/RoPE 为 `512/64`。每个配置 warm-up 2 次，再以交替正反顺序执行 5 轮
CUDA event 与墙钟测量。`cropped` 只裁去槽位 1,024–2,047 中确定为无效的
`-1` 尾部，不改变前 1,024 个 causal token ID。

| 配置 | CUDA 中位数（ms/调用） | 墙钟中位数（ms/调用） | 临时峰值 allocated delta（MiB） | 相对现状加速 |
|---|---:|---:|---:|---:|
| full top-k / split 16（现状） | 62.883839 | 62.911265 | 592.53125 | 1.000× |
| full top-k / split 1 | 57.757694 | 57.786386 | 112.06250 | 1.089× |
| cropped top-k / split 16 | 51.245056 | 51.272084 | 592.53125 | 1.227× |
| cropped top-k / split 8 | 48.530434 | 48.559195 | 336.28125 | 1.296× |
| cropped top-k / split 4 | 47.248383 | 47.280767 | 208.15625 | 1.331× |
| cropped top-k / split 2 | 46.680065 | 46.713796 | 144.09375 | 1.347× |
| cropped top-k / split 1 | **46.382080** | **46.410246** | **112.06250** | **1.356×** |

7 组 output/LSE 均通过以 full top-k/split16 为基准的
`atol=rtol=0.002` 检查，所有值均有限。跨配置的 output 最大绝对差为
`1.1920928955078125e-06`，LSE 最大绝对差为
`9.5367431640625e-07`。summary 与 runner log SHA256 分别为：

- `25355d535bef1daaf4099d7b06aee7ee123a81fbbaffcf3b4d6ffc88113025d4`；
- `9ce24b25e97e2f3f8bc4150eb14d19c6be39c991528a7086761c390c8010de36`。

结果表明，prefill 不应沿用 decode 的 split 16。单独改为 split1 只加速约
8.9%，单独裁剪无效 top-k 尾部加速约 22.7%；组合后 CUDA 时间下降
`26.24%`，即加速 `1.356×`，临时峰值 allocated delta 同时下降约
480.47 MiB。按 78 层线性外推，这一项约可减少 1.287 秒，但仍不足以把
OSCAR 的 1K TTFT 降到 BF16 水平。因此，下一步实现必须仅在 prefill 路径使用
有效 top-k 宽度和 split1，保持 7.9 的 decode split16；完成 TP=8 TTFT 实测后
还需继续 profile 剩余 mixed stage1。

### 7.12 Prefill 快路径实现与候选冻结

7.11 选出的最小改动已落地到 OSCAR-vLLM 源码提交
`a94b1f640fe504be3d741a1070e43f806eaad894`，对应 Git tree 为
`7b5650fef2986e783c4ab4ad1cbd434a4f64b252`。实现只在 OSCAR attention 调用点
区分两类执行形状：

- 纯 decode 继续使用完整 top-k view 和 7.9 已证明最优的 split16；
- prefill 或 mixed batch 把 top-k view 宽度裁到
  `min(topk_tokens, max_seq_len)`，并使用 split1。

该逻辑没有改变 DSA 选中 token 的前缀、三段式 cache 内容、output/LSE 合并或
decode 参数。新增回归分别覆盖纯 decode split16、prefill/mixed split1 和
top-k view 裁剪。源码验证实际结果为：

- 定向回归 11/11 passed；
- 完整 `tests/oscar_mla` 为 90 passed、26 个显式 CUDA skip、0 failed；
- Ruff check/format、typos、SPDX、forbidden imports、增量 mypy、Python
  compile、Git diff check 和相关 pre-commit 门禁通过。

26 个 skip 仍只表示本轮源码级 CPU 验证没有分配 GPU，不能当作苹果800 性能或
CUDA 回归结果。主仓库提交
`dde5301b1ced344333ab71555ae59f89893e4a45` 已冻结 Phase 1/5/7/9 配置、正式
wrapper、source tree、候选 evidence hash 和控制镜像身份。

新候选目录为
`artifacts/phase6/20260730T2132Z_candidate_a94b1f640_prefill_fastpath`。
构建与递归验收均通过，实际不可变身份为：

- 候选 tag：`glm52-oscar-a800-phase6-a94b1f640-0275043c`；
- image/config：
  `sha256:e0f6b4066011732ce16a62ca2e154b8c825829590b4b7e6455d86ea8817e5635`；
- manifest：
  `sha256:41a70b2ae775482ddc52c2fca34bd60993874559fca1582dfb0f58200696c826`；
- candidate layer：
  `sha256:34a5e717e36393120dab1c6810aa5dae78791d1e1dd3c860fc44f652b49a45e6`；
- `build_result.json` / `verification.json` SHA256：
  `e007ba672465afeff1a4141f0a8860ce0ecd516431c9e18a9a4c4c8cbfaa181b` /
  `c0bd1ce74397af94c9070e9a7aa30df87fcedf301d86102d9f8ac75c0f1e4b17`。

验收重新核对 4,744 个源码文件、4 份 rotation、7 个基础层原生扩展和 33 层
身份。正式 venv 的 runtime import 确认为
Python/PyTorch/Triton `3.12.13/2.11.0+cu129/3.6.0`，并保持
`cuda_initialized=false`。OCI 导入 Docker 后的 image ID 与上述
image/config digest 精确一致。新控制镜像
`oscar-glm-stage9-runtime:a94b1f640` 的 image ID 为
`sha256:a7482d1c709e02720e9bad7e442f558af4ebc27315904763e1feac0744179ed9`；
其中 Phase 9 工具测试 19/19、修正挂载后的 Phase 7 工具测试 20/20 通过。

本节只证明 prefill 快路径已经实现、验证并封装成不可变候选。正式
containerized preflight 和 TP=8 1K/batch1 TTFT/TPOT 探针尚未执行，因此
不能用 7.11 的单卡单层外推值代替端到端性能结果。

## 8. 当前完成度与待办

| 工作项 | 状态 | 证据边界 |
|---|---|---|
| REAP 原生 TP=8/32K sparse MLA | 已完成 | 141/141 分片与功能请求通过 |
| REAP calibration/rotation | 已完成 | 1M tokens、78 层 artifact、33/33 tests |
| 三段式 planner/allocator/scheduler | 已完成回归 | 145 passed；理论容量单独标注 |
| 苹果800/SM80 kernels | 已完成 | GPU 0 114/114；8 卡 224/224 |
| OSCAR TP=8/32K 功能 | 已完成 | 31,996+64、8 并发、78 层调用证据 |
| OSCAR 固定 256 题测试 | 已完成 | 256/256、107 正确、accuracy 0.41796875、0 request failure |
| BF16 固定性能矩阵与 profiling | 已完成 | 9/9 格 passed；每格 3 轮与 8+8+1 profiler 证据 |
| OSCAR 固定性能矩阵与比较 | 优化中 | decode 快路径 1K/b1 TTFT +1,322.8%、TPOT +31.9%；prefill 快路径已冻结为新候选，尚待 TP=8 验证 |
| 128K 扩展 | 未完成 | 将随 OSCAR 候选轮次验证 |
