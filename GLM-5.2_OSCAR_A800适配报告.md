# GLM-5.2 支持 OSCAR 并运行于 NVIDIA 苹果800 的适配报告

> 状态截点：2026-07-30  
> 当前主仓库分支：`feat/glm52-model-load`  
> Stage 9 BF16 运行提交：`0918f3a4ee3ae17713ecf43679ec557d77e5fc39`
> 当前 OSCAR-vLLM 源码提交：`065af88a010dc5746029198088ba01edc4a61516`

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

截至 2026-07-30，已经证明：

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

- Stage 9 OSCAR 同负载矩阵尚未完成，因而还不能计算 BF16/OSCAR 的 TTFT、
  TPOT 回退比例；
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
| OSCAR 固定性能矩阵与比较 | 进行中 | 等待同提交、同负载候选轮次 |
| 128K 扩展 | 未完成 | 将随 OSCAR 候选轮次验证 |
