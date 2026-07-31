# GLM-5.2 支持 OSCAR 并运行于 NVIDIA 苹果800 的适配报告

> 状态截点：2026-07-31
> 当前主仓库分支：`feat/glm52-model-load`  
> Stage 9 BF16 运行提交：`0918f3a4ee3ae17713ecf43679ec557d77e5fc39`
> 当前 OSCAR-vLLM 源码提交：`24938975f70bbf6d502b3556bdb44de0a5c7bde7`

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

- Stage 9 OSCAR 完整同负载矩阵尚未完成；当前 grouped prefill 候选的
  1K/batch1 定向结果为 TTFT `1317.120 ms`、TPOT `202.668 ms`。它相对
  上一版 OSCAR 分别改善 `64.54%/1.57%`，但相对现有旧 BF16 reference
  仍回退 `273.71%/29.33%`；且两个结果不是同一最终源码提交，不能替代最终
  同提交严格比较；
- grouped prefill 单卡单层实验已把 cropped top-k/split1 从
  `46.382 ms` 降至 `13.284 ms`，并通过完整冷 cache CUDA 套件
  124/124；首个新候选 OCI 虽通过字节与 runtime 验收，但后续审计发现其
  Dockerfile 默认源码身份滞后，已拒绝进入正式 preflight。修正身份后的第二次
  构建又发现 GNU tar 的 PAX 扩展头路径含构建进程 PID，导致内容相同的
  candidate layer digest 不同，因此同样被拒绝。PAX 路径固定后，两个新目录
  的独立完整构建和递归验收均通过，image/config、manifest、candidate layer
  和 diff-ID 完全一致；v3 也已成功导入 Docker，daemon 身份与 labels 精确
  匹配，driver-injected runtime import 也已通过且没有初始化 CUDA；新控制
  镜像已从该 v3 构建并完成 CPU-only 环境检查。Phase 1/5/7/9 的正式配置与
  wrapper 也已切换到该候选，并通过静态身份、语法、派生哈希及 Phase 7/9
  工具测试；正式 containerized preflight 也已通过 64/64，且没有初始化
  CUDA。TP=8 定向探针已完成并证明 grouped prefill 可转化为端到端收益，但
  TTFT/TPOT 仍未关闭性能门限。同口径 8-rank trace 进一步证明 OSCAR 的
  prefill/generation worker 窗口相对 BF16 分别慢 `445.12%/27.09%`；
  8-rank table 将 decode 最大可控 CPU 差距定位到逐层 KV update，按 78 层
  折算约多 `43.05 ms/token`。针对该差距的首项源码优化已经把跨层不变的
  decode/demotion 索引移到 worker metadata，并复用每层 demotion scratch；
  CPU 套件为 96 passed、29 个 CUDA 显式 skip，随后完整苹果800 cold-cache
  CUDA 套件为 125/125 passed；新候选 OCI 已在两个独立目录完成确定性构建
  和递归验收，并已导入 Docker、通过 daemon identity/label 审计与
  driver-injected runtime import；新控制镜像也已构建并通过 CPU-only
  身份/环境审计。Phase 1/5/7/9 配置和 wrapper 已迁移到该候选，工具测试
  39/39、正确容器挂载命名空间中的递归静态 verifier 64/64 通过；
  driver-injected 完整 preflight 也已通过且没有初始化 CUDA。按 Shawn
  2026-07-31 指定的 32K/batch1 负载，首次探针三轮诊断中位数为 TTFT
  `106,660.424 ms`、TPOT `200.303 ms`；相对现有 BF16 同格点分别为
  `+751.37%/+12.01%`。不过该轮在 profiler 完成后因新增未跟踪优化记录触发
  仓库洁净门禁，没有生成单格 summary，因此不能标记为正式通过，也不能用来
  单独归因 metadata/scratch 收益。多 chunk 分析器已确认两边 32K prefill
  均为 16 个 2,048-token 窗口；OSCAR/BF16 的 8-rank prefill wall 中位数为
  `105,753.449/10,086.470 ms`，其中 OSCAR grouped prefill stage1 单项为
  `93,913.327 ms`、占其 prefill wall `88.80%`。因此下一步先优化该 kernel，
  再以新 run ID 重跑同一格点；
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

正式 containerized preflight
`20260730T2152Z_stage9_candidate_prefill_fastpath_preflight_v1` 随后通过
64/64 静态检查。它重新验证上述 source/OCI/native/rotation/baseline 身份，
并实际解析出 TP=8、`max_model_len=131072`、
`max_num_batched_tokens=2048`、`oscar_mla_int2`、eager、async scheduling
关闭和 torch profiler。固定环境与参数解析均确认
`cuda_initialized=false`；`static_preflight.json` SHA256 为
`4c3086c63479868c15931bde9e5ca16e4e7b5e0a94813a7b99c1d6687edfb324`。
preflight 容器退出后 8 张 GPU 均为 0 MiB、0%，没有 compute app。

本节只证明 prefill 快路径已经实现、验证、封装并通过运行前门禁；在该阶段
不能用 7.11 的单卡单层外推值代替端到端性能结果。随后完成的 TP=8
1K/batch1 TTFT/TPOT 结果见 7.13。

### 7.13 Prefill 快路径 TP=8 结果与剩余瓶颈

定向轮次
`20260730T2156Z_stage9_candidate_prefill_fastpath_probe_1k_b1_v1` 使用已发布
主仓库提交 `aed7fdf4d9983d3e29fa13cbe4cfcd7bb5d3daa5`、源码提交
`a94b1f640fe504be3d741a1070e43f806eaad894` 和 7.12 的固定候选/控制镜像。
它仍只选择固定矩阵的 1K/batch1 格点，不包含 128K，也不冒充完整矩阵。

运行前外层两次 GPU 检查分别在 `2026-07-30T21:55:08Z` 和 `21:56:14Z`
完成，间隔 66 秒；容器内再次完成间隔 60 秒的两次 8/8 空闲检查。141/141
模型分片加载后服务 ready。探针执行 1 次 warm-up、3 轮正式测量、128 输出
token 和完整 profiler；10 分钟运行及 profiler 心跳均实际打印。三轮全部
0 request failure，服务端最多运行 1 个请求、等待为 0、preemption 为 0，
KV cache usage 峰值约 `0.203%`，不存在容量排队。

8 张 CUDA table、8 份 worker trace 和 1 份 frontend trace 均通过数量、
rank、bytes 与 SHA256 校验；cell 和总 summary status 均为 `passed`。退出后
容器删除，8 张 GPU 均为 0 MiB、0%，没有 compute app。总 summary 与 cell
summary SHA256 分别为：

- `f14b0088b144b9982a078014d860adefb1da370d0be75b245aaf08ea9bd4bc59`；
- `b010278a9d9cb8da77ccc0b25112731b6e257d69d6866ace52d7c7a70a192f73`。

三轮 `mean` 指标的中位数对比如下：

| 指标 | BF16 | Decode 快路径 | Prefill+decode 快路径 | 相对 decode 快路径 | 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| TTFT（ms） | 352.445 | 5,014.582 | 3,714.821 | -25.9% | +954.0% |
| TPOT（ms） | 156.705 | 206.735 | 205.908 | -0.4% | +31.4% |
| 请求吞吐（req/s） | 0.04934 | 0.03196 | 0.03346 | +4.7% | -32.2% |

TTFT 实际减少 `1,299.761 ms`，与 7.11 按单层 sweep 外推的约 1.287 秒一致；
TPOT 基本不变也符合实现只改变非纯 decode 分支的预期。这证明 prefill top-k
裁剪和 split1 是有效的端到端优化，但仍未关闭相对 BF16 的差距。

同轮 8-rank trace 使用固定 Python `3.12.13`、`ijson 3.4.0.post0` 重新解析。
有效分析目录为
`20260730T2225Z_prefill_fastpath_probe_trace_analysis_v2`，summary SHA256
为 `ff69be060cc7d9d91738af3697362f66850af27bba2ccfef1537735cf64d9922`。
8 个 rank 均包含 129 个 execute context，prefill 仍为每层一次、共 78 次
mixed stage1：

| Prefill 指标 | Decode 快路径 | Prefill+decode 快路径 | 变化 |
|---|---:|---:|---:|
| execute context 中位（ms） | 4,997.096 | 3,697.620 | -26.0% |
| 窗口内 kernel 合计中位（ms） | 4,952.601 | 3,650.752 | -26.3% |
| mixed stage1 中位（ms） | 4,576.783 | 3,300.032 | -27.9% |
| mixed stage1 占 prefill | 91.59% | 89.25% | -2.34 个百分点 |
| split merge 中位（ms） | 25.933 | 3.169 | -87.8% |

首次分析 v1 因命令使用绝对解释器而实际记录系统 `ijson 3.5.0`，未满足预期的
固定分析环境，因此不作为上表证据；v2 修正后与 v1 数值完全一致。

结论是：本轮优化已经命中并按预期减少约 1.3 秒 TTFT，但 mixed stage1 仍占
prefill 约 89.25%，单项累计 3.300 秒；TPOT 也仍超过 BF16 31.4%。下一轮应
继续优化 mixed stage1 的结构性访存、INT2 反量化和三段式分支开销，而不是直接
消耗 8 卡运行剩余完整矩阵。

### 7.14 Prefill 跨 head 共享 KV 单卡实测

代码与 7.13 的 trace 共同表明：同一 query row 的 8 个本地 attention head
共享 DSA selected token、BF16 prefix/recent、INT2 history 和 RoPE KV，但旧
stage1 按 head 分别启动 program，因而把同一批 INT2 unpack、scale/zero、
BF16 KV 和 RoPE 读取重复执行 8 次。新 prefill 专用路径把最多 16 个 head
放入同一 program，复用上述 KV 数据；纯 decode 仍走旧 stage1/split16，
没有改动 7.9 的 decode 参数。

实现过程保留了固定正确性门限，并记录了两次被拒绝的中间版本：

1. 提交 `c3728be9f` 的 grouped kernel 使用 `num_stages=2`，轮次
   `20260730T2250Z_oscar_prefill_headgroup_1k_b1_v1` 在 SM80 编译时需要
   184,320 bytes shared memory，超过 166,912-byte 硬件上限；未进入正确性
   或计时；
2. 提交 `a0171ed6a` 将 `num_stages` 降为 1 后成功编译，但轮次
   `20260730T2252Z_oscar_prefill_headgroup_1k_b1_v2` 的 BF16 tensor-core
   版本产生 output/LSE 最大绝对误差 `0.009153/0.002593`，超过既有
   `0.002/0.002` 门限，因此没有放宽门限，也没有进入计时；
3. 提交 `35ab1846447fc86b4b2177e76c5939503cc3701b` 保留跨 head 数据复用，
   将 score/value dot 改为 FP32 `input_precision=ieee`。轮次
   `20260730T2255Z_oscar_prefill_headgroup_1k_b1_v3` 的 7 组配置全部通过
   output/LSE 门限后，才执行正式计时。

源码侧先完成 TDD：新增 helper 测试在实现前因符号缺失而 collection 失败；
实现后 helper 与 Triton interpreter 6/6 通过，完整 CPU 套件为
95 passed、29 个 CUDA 显式 skip。Ruff、format、mypy、SPDX、typos、
forbidden imports、Python compile 和全部提交门禁通过；上述三个源码提交均已
推送到 `feat/glm52-oscar-integration`。

三轮 GPU 实验前分别重新执行间隔至少 60 秒的两次 8/8 空闲检查。最终 v3
固定使用 GPU 0、Python `3.12.13`、PyTorch `2.11.0+cu129`、CUDA runtime
`12.9`、正式 layer 0 rotation，并复现 1,024 query、每 rank 8 heads、
latent/RoPE `512/64` 的 1K prefill 几何。每个配置 warm-up 2 次，再交替
正反顺序执行 5 轮：

| 配置 | CUDA 中位数（ms/调用） | 墙钟中位数（ms/调用） | 临时峰值 allocated delta（MiB） | 相对 full/split16 |
|---|---:|---:|---:|---:|
| 旧 stage1：full top-k / split16 | 62.896130 | 62.925726 | 592.53125 | 1.000× |
| grouped：full top-k / split1 | 15.216640 | 15.245755 | 112.06250 | 4.133× |
| grouped：cropped top-k / split1 | **13.284352** | **13.315970** | **112.06250** | **4.735×** |

最终 winner 相对 7.11 的旧 cropped/split1 `46.382080 ms` 进一步加速
`3.491×`，CUDA 时间下降 `71.36%`。相对 full/split16 的完整优化为
`4.734602649×`。grouped winner 相对旧 full/split16 的 output/LSE 最大绝对
差分别为 `3.874301910400391e-06` 和
`1.430511474609375e-06`，远低于固定 `0.002/0.002` 门限；所有值有限。

有效 `summary.json` 和 `runner.log` SHA256 分别为：

- `f87f3624072431d7e9d6219091c01ff1a9ab348424287f036edc57e1c60ef2c0`；
- `c62cbf5061d200275e6268be23cd2c496017b68979fa1808cb9f012497fc7699`。

最终单卡容器退出后无 compute app，8 张 GPU 显存均为 0 MiB。随后对同一源码
提交执行完整冷 Triton cache CUDA 回归。首轮
`20260730T2259Z_oscar_headgroup_full_cuda_v1` 在 pytest collection 阶段因
源码中的 4 个原生扩展 symlink 指向未挂载的 phase0 rootfs 宿主绝对路径而退出；
该轮执行 0 个测试、0 个 kernel，不计为 CUDA 结果。第二轮只补回该既有绝对路径
的只读挂载，未修改源码、测试或运行时。

有效轮次
`20260730T2302Z_oscar_headgroup_full_cuda_v2` 前的两次 8/8 GPU 空闲检查分别
在 `2026-07-30T23:00:38Z` 和 `23:01:44Z` 完成。结果为 124 passed、
0 skipped、0 failed、17 warnings，耗时 80.50 秒；新增的 split1 prefill
组合均实际执行。冷 cache 共生成 380 个文件、29,220,161 bytes，pytest 日志
SHA256 为
`3f8006e38ef6f49fb3f0832c3d003e5997a79c9a054b57db5f470c6c38f6f1f7`。
容器退出后 8 张 GPU 显存均为 0 MiB，且没有 compute app。

同一源码随后冻结为新候选
`glm52-oscar-a800-phase6-35ab18464-0275043c`，目录为
`artifacts/phase6/20260730T2315Z_candidate_35ab18464_headgroup`。构建与独立
递归验收均通过：

- image/config：
  `sha256:d06a82948c342de84bcd2400ca51ab9ada67ad45a20fa7723629a3d0487367df`；
- manifest：
  `sha256:28a8f1daec3c52420073eaad93e984982b9d1647521ddd99d00a3fac8b88c9d0`；
- candidate layer：
  `sha256:a599892d73b9b54723a06700ed82b8022c10a9bfe1302a7d829e84b38e192da0`；
- `build_result.json` / `verification.json` SHA256：
  `618191fd25e75e3f351193f87b498e1359063c11e62be75f74e901ae81683a0b` /
  `7adad3e1479947aae7563f61d1dc9b077902edd0f29a15763eb4e8eec39f747d`。

验收精确核对源码提交/tree、4,744 个源码文件、4 份 rotation、7 个基础层原生
扩展和 33 层身份；基础层完全匹配，candidate layer 不含原生扩展或 whiteout。
该构建/验收为 CPU-only，没有分配 GPU。

OCI 由一次性 Ubuntu 22.04 工具容器中的 `skopeo 1.4.1` 从只读 layout 导入
Docker daemon；daemon image ID 精确等于上述 image/config digest。正式 runtime
import 前的两次 8/8 GPU 空闲检查分别在
`2026-07-30T23:17:30Z` 和 `23:18:40Z` 完成。探针只注入 NVIDIA 驱动并执行
只读 import/identity 检查，实际确认：

- Python/PyTorch/Triton 为 `3.12.13/2.11.0+cu129/3.6.0`；
- vLLM source 与 `vllm._C` 均来自候选 `/opt/vllm_glm52_v1`；
- 78 个 rotation、manifest/rotation SHA256 和 runtime expectation 匹配；
- `reasoning_effort=max` 可解析；
- `cuda_initialized=false`。

`runtime_import.json` / log SHA256 分别为
`0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a` /
`f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`。
探针退出后在 `23:20:15Z` 复查，8 张 GPU 均为 0 MiB、0%，无 compute app。
首次未注入 NVIDIA runtime 的探针因缺少 `libcuda.so.1` 在原生扩展 import
阶段退出，CUDA 未初始化；该失败不计作 runtime import 通过结果。

构建控制镜像后的全配置审计进一步发现：上述 OCI 实际打包的 source
commit/tree 确实为 `35ab1846…/22b1c44e…`，但候选 label 固定的 Dockerfile
SHA256 对应文件仍把默认 `SOURCE_COMMIT/SOURCE_TREE` 写成旧
`a94b1f640…/7b5650fe…`。该问题不改变已经验收的候选字节，却使 Dockerfile
复现元数据不自洽。因此，这一 v1 OCI、runtime import 和临时控制镜像只保留为
被审计拒绝的证据，不进入正式 preflight 或 TP=8 性能测试。

Dockerfile 默认身份随后已修正为源码提交/tree
`35ab1846…/22b1c44e…`，修正后的文件 SHA256 为
`cb8a62ccc041bf2ae7e84740c3fd58f47f1d44f3ec426ebd9072c01ddbc49f23`。
第二次构建使用新目录
`artifacts/phase6/20260730T2325Z_candidate_35ab18464_headgroup_v2`，生成：

- image/config：
  `sha256:362c3d1f7062b2c7f0a3777560a42d6fe28a02666bde06fdb548e2b8269880fd`；
- manifest：
  `sha256:56fcc9b8052adee6c39c53b84fd03a2cbb368c932443cd4dcd97e11bc0a1decd`；
- candidate layer：
  `sha256:829ceb0e5436b902006d7389d9356867765b1903e43fc6a68a3e083aac9510e0`。

源码、rotation、runtime expectation 与 v1 均未改变，但 v2 candidate layer
与 v1 的 `sha256:a599892d…92da0` 不同，因此在独立验收和 import 前暂停。
逐 member 对比确认两层各有 5,298 个条目，其路径、权限、时间、解压内容和
逐文件 SHA256 全部一致；原始 tar 的首个差异来自自动 PAX 扩展头目录
`PaxHeaders.852152` 与 `PaxHeaders.966541`。后缀数字是 GNU tar 默认
`exthdr.name=%d/PaxHeaders.%p/%f` 中的构建进程 PID，它会污染 layer
diff-ID 和 gzip digest。该 v2 因而也只保留为被拒绝证据，不进入验收、
runtime import、控制镜像或正式 preflight。

构建器随后把扩展头显式固定为不含 `%p` 的
`exthdr.name=%d/PaxHeaders/%f`，并增加强制触发 PAX header 的长路径回归。
该回归连续启动两个独立 tar 子进程，要求未压缩 tar、gzip layer、compressed
digest 和 diff-ID 全部相同；固定 CPython 3.12.3 下为 1/1 passed，Ruff
check/format 与 Python compile 同时通过。修复已由主仓库提交
`d14be61a216381a22b902a4fd0591aa286068490` 发布。

随后使用完全相同输入在两个新目录执行完整构建与独立递归验收：

- v3：
  `artifacts/phase6/20260730T2333Z_candidate_35ab18464_headgroup_v3`；
- v4 重建：
  `artifacts/phase6/20260730T2334Z_candidate_35ab18464_headgroup_v4_rebuild`。

两轮共同得到：

- image/config：
  `sha256:6b5aeb4b1b26c8012062163d510fc5a58255c85a71a602d83f9affc386a7bb59`；
- manifest：
  `sha256:a629a99ec78a90562ff506709c176e98c324f1c373378ac038a3ce0c06de9105`；
- candidate layer：
  `sha256:2ec5ec198ccbdfb92712143087b0e0c57d6269e59b88d1724496f32986e206bd`；
- diff-ID：
  `sha256:f3f1d91ce4b321d648f5b0af2181f6e4cc558ebe76180d613feeeba872a90ee9`；
- candidate layer size/member：
  `109,147,025 bytes` / `5,298`；
- `index.json` SHA256：
  `0a1bf2a41fb28718cf8114da70c3d19e36244f7bb5f0bcf6ce80f57722dd676d`。

两份 manifest/config/layer blob 均逐字节相同。两次验收状态均为 `passed`，
分别重新核对 4,744 个源码文件、4 份 rotation、7 个基础层原生扩展、33 层
身份、精确 Git tree、无原生扩展覆盖和无 whiteout。v3 的
`build_result.json` / `verification.json` SHA256 为
`ada8128b15e51b4239d77c2ab13d72d94d739d6bb57132dcc3c30f97fae804da` /
`6577845fb29cee8504c8f5f3e3979237a2bf16f92f54eb859e5769b26a6c18ee`；
v4 对应为
`9226f02782d429338365c4d8bb05441c86fb3032c31b474b5d8c6435220a3e5e` /
`6c038b8ca8f46db6ea85d0ceac08a1a1113f97ee4f13a8091b243ba71d00b152`。
两组报告哈希不同只来自各自记录的输出/解压目录路径，不影响上述完全一致的 OCI
不可变身份。本阶段为 CPU-only，没有分配 GPU；v3 被选为后续运行的正式候选。

v3 随后由一次性 Ubuntu 22.04 工具容器中的 `skopeo 1.4.1` 从只读 OCI
layout 导入 Docker daemon。工具容器安装时无关 deadsnakes PPA 出现 TLS
握手警告，但 Ubuntu 主仓库中的 `skopeo` 已成功安装，33 层复制和 manifest
写入完整结束；工具容器随后自动删除。daemon image ID 精确为
`sha256:6b5aeb4b1b26c8012062163d510fc5a58255c85a71a602d83f9affc386a7bb59`，
层数为 33；以下 labels 均与 v3 验收值精确匹配：

- source commit/tree；
- candidate layer digest；
- Dockerfile SHA256；
- rotation manifest SHA256；
- runtime expectation SHA256；
- base manifest digest。

导入阶段仍为 CPU-only，没有注入 NVIDIA runtime 或初始化 CUDA。需要驱动库的
正式 runtime import 在两次 8/8 GPU 空闲检查后执行；检查时间为
`2026-07-30T23:42:03Z` 和 `23:43:11Z`，间隔 68 秒，均为 0 MiB、0%
且没有 compute app。探针只注入驱动库并执行只读 import/identity 检查，结果为
`passed`：

- Python/PyTorch/Triton 为 `3.12.13/2.11.0+cu129/3.6.0`；
- vLLM source 与 `vllm._C` 均来自候选 `/opt/vllm_glm52_v1`；
- 78 个 rotation、manifest/rotation SHA256 和 runtime expectation 匹配；
- `reasoning_effort=max` 可解析；
- `cuda_initialized=false`。

主探针最初对 `sys.executable` 调用了 `resolve()`，因此显示底层
`/usr/bin/python3.12`；随后 CPU-only 原值探针确认实际
`sys.executable=/opt/fp8_speed_up_v4_venv/bin/python`。最终落盘的是未经
realpath 展开的实测原值。`runtime_import.json` / log SHA256 分别为：

- `0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`；
- `f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`。

该 payload 与 v1 相同，因此两份 runtime evidence 字节也相同；这不改变 v1
因 Dockerfile 元数据不自洽而被拒绝的结论。探针容器自动删除，退出后的
`23:44:16Z` 和 `23:44:53Z` 两次复查均为 8 张 GPU 0 MiB、无 compute app。

控制镜像构建前又检查了 Phase 9 Dockerfile，发现其默认 base 仍为旧 a94
候选。虽然命令行 `--build-arg` 可以覆盖该值，但会使复现入口与实际构建身份
不一致，因此先把默认 base 最小切换到 v3 tag，并以主仓库提交
`b62d459d28e30a282c99d39f2815d9927911f391` 发布。

随后构建的新控制镜像为
`oscar-glm-stage9-runtime:35ab18464`，实际 image ID 为
`sha256:bef0320ddb28d2591fb59758911c2333c39ae3e77b4b90ff87815e068b20bb7c`。
该镜像为 34 层；其 base v3 为 33 层、image ID
`sha256:6b5aeb4b…7bb59`。CPU-only 验证确认：

- source revision 与 candidate layer labels 精确匹配 v3；
- Git 为 `2.34.1`；
- iproute2 为 `5.15.0`；
- Python 为 `3.12.13`；
- `/opt/phase9-control-packages.txt` 记录的 Git/iproute2 包版本与实际一致。

构建和验证容器均已删除，本阶段没有分配 GPU。随后已把 Phase 1/5/7/9 的正式
配置与 wrapper 统一切换到源码提交 `35ab1846…`、v3 overlay、manifest
`a629a99e…`、image/config `6b5aeb4b…`、candidate layer
`2ec5ec19…` 和上述 control image `bef0320d…`。正式
`configs/scripts/docker` 范围内已无旧 a94 候选的 source、path、digest 或
control image 引用。

本轮配置文件 SHA256 为：

- Phase 1：`46cc283cd83329275d044c8e548bd8f23d22537dfad25739c49d6d6156808a43`；
- Phase 5：`cb432402c6537012024362aedfaa036d67a8d59b3cb3a932a5d4b2f1d27f5933`；
- Phase 7：`1eb373a717768c58bc8656eddf346a47fa236a2e73dfcfca96818e480133ff9b`；
- Phase 9：`e76e2556a558d2c2e589d5c5103186eac28adc26df57ab933f795803a3278ed6`。

Phase 5 中记录的 Phase 1 manifest hash、Phase 7 中记录的 Phase 5 manifest
hash 均与文件实算值精确相等。4 个 JSON 解析、9 个 shell `bash -n`、
Python compile 和 `git diff --check` 全部通过；固定控制镜像内的 Phase 9
工具测试为 16/16、Phase 7 工具测试为 20/20。该阶段为 CPU-only，没有启动
benchmark，也没有产生新的 TTFT/TPOT。在这一静态检查截点，正式
containerized preflight 仍待配置与本报告提交发布后执行。

上述配置、wrapper、测试与本阶段报告随后以主仓库提交
`36c1b8a420b09280481da04f68d50076946498e7` 发布，远端分支与本地提交精确
一致。正式 candidate preflight
`20260731T0001Z_stage9_candidate_headgroup_preflight_v1` 前，外层在
`2026-07-30T23:59:34Z` 和 `2026-07-31T00:00:35Z` 两次检查 8 张 GPU，
间隔 61 秒；两次均为 0 MiB、0% 且没有 compute app。

preflight 实际退出码为 0，`static_preflight.json` 状态为 `passed`，
64/64 检查通过，SHA256 为
`f2f1948b0d2aac710f6199fea993fdc9b154746b1ab1d31ceda2637b7188a532`。
它重新验证 source、OCI 三项 digest、4,744 个源码文件、6 个 lower native
symlink、rotation/runtime expectation、BF16 baseline evidence 和性能配置，
并解析出：

- TP=8、PP=1；
- `TRITON_MLA_SPARSE` 与 `oscar_mla_int2`；
- `max_model_len=131072`、`max_num_batched_tokens=2048`；
- eager、async scheduling 关闭；
- torch profiler。

固定环境 import 与服务参数解析均记录 `cuda_initialized=false`，两份 JSON
SHA256 分别为
`5434bc30966d13b323f08df49dbe419cc5f7bab9752fceee0dce6760f96de195`
和
`91139680b0a2819230274eddaf637bfc4857571003874f68d5cf02e332880388`。
preflight 容器退出后的 `00:02:06Z` 复查为 8 张 GPU 0 MiB、0%，无
compute app。

preflight 阶段报告以主仓库提交
`efec5ef2bb1b9502762f7c17196a1574f82d5461` 发布后，正式运行 TP=8
1K/batch1 定向探针
`20260731T0005Z_stage9_candidate_headgroup_probe_1k_b1_v1`。新 GPU
分配前的外层空闲检查为 `00:04:01Z/00:05:02Z`，间隔 61 秒；容器内又完成
两次 8/8 idle。141/141 分片全部加载，模型加载耗时
`202.236202 秒`、每卡模型内存 `56.0 GiB`、可用 KV cache
`13.74 GiB`。服务、完整运行和 profiler 分别实际打印 10 分钟心跳。

探针只选择固定矩阵的 1K/batch1，不包含 128K，也不冒充完整矩阵。固定单格协议
仍为 1 次 warm-up、3 轮正式测量、每轮 3 个请求、128 输出 token 和完整
profiler。三轮均为 3/3 completed、0 failed：

| 轮次 | TTFT（ms） | TPOT（ms） | 吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 1,308.837 | 202.355 | 0.037026 |
| 2 | 1,317.120 | 202.668 | 0.036960 |
| 3 | 1,336.220 | 203.521 | 0.036787 |

三轮 `mean` 指标中位数对比如下：

| 指标 | 现有旧 BF16 reference | 上一版 OSCAR | Grouped prefill | 相对上一版 OSCAR | 相对旧 BF16 |
|---|---:|---:|---:|---:|---:|
| TTFT（ms） | 352.445 | 3,714.821 | 1,317.120 | -64.54% | +273.71% |
| TPOT（ms） | 156.705 | 205.908 | 202.668 | -1.57% | +29.33% |
| 请求吞吐（req/s） | 0.04934 | 0.03346 | 0.03696 | +10.47% | -25.09% |

这里的旧 BF16 与 grouped 候选不是同一最终源码/主仓库提交，只用于决定是否继续
优化；最终结论必须在候选冻结后以同一提交重跑 BF16 与 OSCAR。当前
TTFT/TPOT 对旧 BF16 均超过 20% 门限，因此不能直接进入完整 9 格。

单格与总 summary、profiler 均为 `passed`。8 个 rank table、8 份 worker
trace 和 1 份 frontend trace 全部通过数量、rank、bytes 和 SHA256 校验。
profile 耗时 `645.7748026847839 秒`，critical rank 为 6、kernel total 为
`33,962 ms`。该 rank 的当前主要 OSCAR 项包括：

- grouped `_mixed_sparse_prefill_stage1`：
  `912.239 ms / 78`，即 `11.695 ms/层`；
- `_mixed_sparse_decode_stage1`：
  `1.688 s / 9,906`，即 `170.405 µs/调用`；
- `_rotate_latent_kernel`：
  `844.125 ms / 29,952`；
- `unified_mla_kv_cache_update`：
  CPU/CUDA total `5.593 s / 440.567 ms`。

上一版 trace 的 prefill mixed stage1 为 `3,300.032 ms`，本轮下降约
`72.36%`，与单卡方向及 TTFT 改善一致。critical rank 的 NCCL all-reduce
累计为 `27.857 s`，但它包含 rank 间等待，不能在逐 rank trace 归因前直接认定
为首要根因。服务端最多运行 1 个请求、等待为 0、preemption 为 0，KV usage
峰值为 `0.2028%`，排除了容量排队。

总 summary 与 cell summary SHA256 分别为：

- `4d2945bfc3c6687058916994acb617d8792aca51af01578f4c66c7d093992628`；
- `59aee3156e54eaf5422444bbe7a461d8f350028449fc74041f92613dfabbf3e5`。

探针容器已删除，`00:32:07Z/00:33:27Z` 两次退出复查均为 8 张 GPU
0 MiB、0%，没有 compute app。跨 head 复用已经由单层性能、正确性、完整
苹果800 CUDA 回归和 TP=8 端到端结果共同证明有效；当前下一步是按同轮
profiler/trace 分析剩余 TTFT 与 TPOT，继续最小优化，而不是直接运行完整矩阵。

### 7.15 Grouped 候选与 BF16 的同口径 trace 归因

本阶段没有重新分配 GPU，而是用同一分析器分别流式解析 7.5 的 BF16 v4 和
7.14 的 grouped 候选首个 1K/batch1 profiler 所冻结的 8 份 worker trace。
两轮均固定使用 Python `3.12.13`、`ijson 3.4.0.post0`、4 workers 和分析器
SHA256
`0fa4ebf5efa393e62d3979202a3575b8b04b6ceed80b21b3c3f1ad9c0f4294cf`。
有效输出为：

- BF16：
  `/dev/shm/oscar-glm-stage9/analysis/20260731T0040Z_bf16_prefill_trace_v1/summary.json`，
  SHA256
  `a8c16dbd341e290c67945ac29977fa04b28999a7d1ee0b479efad9a07c768f34`；
- grouped OSCAR：
  `/dev/shm/oscar-glm-stage9/analysis/20260731T0035Z_headgroup_prefill_trace_v1/summary.json`，
  SHA256
  `5ee0887c5828f44e578450074694b7e74b8f0730f55a4263ff1dfd0e0266e9a1`。

8 个 rank 均包含 129 个 execute context。同口径中位数如下：

| Trace 指标 | BF16 | Grouped OSCAR | OSCAR 相对 BF16 |
|---|---:|---:|---:|
| Prefill execute context（ms） | 248.300301 | 1,353.539450 | +445.12% |
| Prefill kernel 合计（ms） | 238.743450 | 1,271.603995 | +432.62% |
| 各 rank generation 中位数再取中位（ms） | 216.485253 | 275.123262 | +27.09% |

OSCAR grouped prefill stage1 为 78 次、`911.005226 ms`，占 prefill wall
`67.31%`；BF16 原生 `_sparse_mla_kernel_final_static` 为 53 次、
`77.446137 ms`。因此，grouped 优化虽然已显著降低 OSCAR TTFT，但剩余 TTFT
主差距仍来自 OSCAR prefill attention 本身，不是调度等待或 TTFT 统计口径。
generation 窗口的 `+27.09%` 又与端到端 TPOT 的 `+29.33%` 接近，证明
decode 回退同样存在于 worker 热路径内。

为避免 BF16/OSCAR critical rank 分别为 1/6 造成偏差，本阶段还聚合了两轮各
8 份 profiler table。逐层调用的 8-rank 中位数如下：

| Profiler 项 | BF16（ms/层） | Grouped OSCAR（ms/层） | 按 78 层折算的 OSCAR 增量 |
|---|---:|---:|---:|
| KV update CPU time avg | 0.012496 | 0.564347 | 43.05 ms/token |
| KV update CUDA time avg | 0.002759 | 0.043536 | 3.18 ms/token |
| Attention wrapper CPU time avg | 0.399557 | 0.746773 | 27.08 ms/token |
| Attention wrapper CUDA time avg | 0.256852 | 0.347315 | 7.06 ms/token |

OSCAR 纯 decode stage1、三次 rotation 和 split merge 的 CUDA 中位数分别为
`0.170181/0.084600/0.005300 ms/层`。NCCL all-reduce 每次中位数由 BF16
`1.0345 ms` 增至 OSCAR `1.2820 ms`，但各 rank 范围差异较大且该值包含等待；
当前证据更支持把它解释为其他 rank 热路径变慢后的同步放大，而不是首要根因。

源码调用链与 profiler 一致：纯 decode 虽已跳过 7.7 的 prefill-only
current-history 索引，但每层仍重复解析 context/layer/metadata，计算
`seq_lens - 1` 和 demotion HP row，并为 recent demotion 每次新建 BF16
gather 与 FP32 rotation 临时 Tensor，再依次执行 gather、rotation 和
INT2 store。worker metadata builder 已掌握相同 batch 的 request→HP row、
最终长度和 demotion page/offset；下一轮最小优化将先把这些跨 78 层不变的
索引一次性物化，并复用 layer demotion scratch。只有该路径实测不足时，才考虑
融合 gather→rotation→INT2 store kernel。

### 7.16 Decode metadata 与 demotion scratch 优化（CPU/CUDA/OCI 门禁）

7.15 确定的首项最小优化已落地到 OSCAR-vLLM 源码提交
`14c768b406b3e39a2d4d5be77a9046ac7ccc26d1`，Git tree 为
`4ad8be8a10fb07321d4ac9c81d31d009e854bde9`。该提交已推送到
`feat/glm52-oscar-integration`，本地与远端分支精确一致；主仓库 submodule
也已指向该提交。

实现只修改 5 个直接相关文件，共 124 行新增、18 行删除：

- worker 在每个 batch 只物化一次 decode position、final sequence length
  和 demotion HP row；
- 纯 decode 的 78 层 attention 调用直接复用上述 metadata，不再逐层计算
  `seq_lens - 1` 或执行 demotion HP row 的高级索引；
- 每个 layer 缓存 BF16 gather 与 FP32 rotated demotion scratch；只有
  device、dtype、latent rank 改变或所需行数增大时才重新分配；
- demotion helper 接受可选的 gather/rotation 输出 Tensor，未传入时保持原有
  行为；
- prefill 分支、INT2 量化 kernel、page/offset、demotion 顺序和 BF16/INT2
  cache 语义均未改变。

TDD 首先在固定控制容器中得到预期红灯：两个定向节点分别因
`decode_positions` 和 `_get_oscar_demotion_scratch` 尚不存在而失败。实现后
三个定向节点转为 3/3 passed；随后又故意把主 `seq_lens` 扰动为错误值，验证
runtime 仍只消费预计算字段。最终源码状态下，固定控制容器中的完整
`tests/oscar_mla` 结果为：

- 96 passed；
- 29 个 CUDA 显式 skip；
- 0 failed；
- 17 warnings；
- 30.15 秒。

Ruff check/format、typos、增量 mypy、SPDX、forbidden imports、Python/diff
及其他适用提交门禁均通过。`check-torch-cuda-call` 指向的
`torch.cuda.empty_cache()` 经 `git blame` 证明来自旧提交 `53d8be94f`，
不在本次 diff；attention backend 文档 hook 只会改写既有 OSCAR 能力表。
因此提交时只精确跳过这两个已审计旧项，没有扩大跳过范围。

完成上述 CPU/静态阶段并由主仓库提交
`4d93b0df417e251290ddc7498af53c8c49005aeb` 发布后，苹果800 CUDA 回归轮次
`20260731T0116Z_decode_metadata_full_cuda_v1` 使用同一源码提交/tree 和固定
控制镜像
`sha256:bef0320ddb28d2591fb59758911c2333c39ae3e77b4b90ff87815e068b20bb7c`。
源码与 phase0 native rootfs 均只读挂载，GPU 0 使用独立空 Triton cache。

GPU 分配前两次 8/8 空闲检查为 `2026-07-31T01:14:47Z` 和
`01:15:54Z`，间隔 67 秒；两次均为 0 MiB、0% 且没有 compute app。有效
pytest 从 `01:17:05Z` 运行至 `01:18:36Z`，结果为：

- 125 passed；
- 0 skipped、0 failed；
- 19 warnings；
- 80.88 秒；
- cold Triton cache 为 380 个文件、29,222,093 bytes。

pytest 日志 SHA256 为
`a923d118983186600cc06e6a372d0671f0da8f0c6bfb32f22f2b3d086eeab02e`。
相对 CPU 套件显式跳过的 29 个 CUDA 节点，本轮全部实际执行。容器自动删除；
`01:18:50Z` 复查 8 张 GPU 均为 0 MiB、0%，没有 compute app。

该结果证明 metadata/scratch 改动通过当前完整苹果800 CUDA 正确性回归，但仍
没有产生新的 TTFT/TPOT。

Phase 6 输入和 Dockerfile 随后由主仓库提交
`47769e047b37a5539acd259d6da55fa49a029373` 发布，只把 output tag、
source commit/tree 和 Dockerfile 默认身份切换到当前源码；base manifest、
rotation artifact、runtime expectation 和确定性 PAX 构建逻辑均未改变。
固定 Python 3.12 下的 PAX 确定性回归为 1/1 passed。

同一已发布输入在两个独立目录完成完整构建与递归验收：

- v1：
  `artifacts/phase6/20260731T0125Z_candidate_14c768b40_decode_metadata_v1`；
- v2 重建：
  `artifacts/phase6/20260731T0128Z_candidate_14c768b40_decode_metadata_v2_rebuild`。

两轮共同得到：

- 候选 tag：`glm52-oscar-a800-phase6-14c768b40-0275043c`；
- image/config：
  `sha256:dbd78a779001300cca5f3802a4e69def3de5a82eb96746c614a5e6d7e14f0a99`；
- manifest：
  `sha256:52a74b155567c24ee9875f695f5be96be87397e74217bf04fa6c6582544468e4`；
- candidate layer：
  `sha256:4b9070484b4441fbfd011b68a4d91f102b55d762fb3c0d15a46c500f68a83312`；
- diff-ID：
  `sha256:619ae46042e6909be60dca188029ab275db6837c96c44dce6001f767d0377d9f`；
- candidate layer size/member：`109,147,574 bytes` / `5,298`；
- `index.json` SHA256：
  `3ac25034d1580116a8b139d9b021a8072561ad2ef34e73f08378658a4977d303`。

两份 `index.json`、config、manifest 和 candidate layer 均逐字节相同。两次
验收状态均为 `passed`，分别重新核对 4,744 个源码文件、4 份 rotation、
7 个基础层原生扩展、33 层身份、精确 Git tree、无原生扩展覆盖和无
whiteout。v1 的 `build_result.json` / `verification.json` SHA256 为
`b18c8744c0fa05ff8b44816eb2965327d7c1f2f8945c73fec3595c467ad6f1c0` /
`10ad7ce5db5dc33316e583583143688e09823e3cd811fb5fc0a6853b1adf19ff`；
v2 对应为
`28119b952fbe2615c4261f5c1ef1d2ea0c7baecd245bff77051ad03b0fc0cac9` /
`afafd564cf660f1f9de4e9828c4b84d433dbb4d5b234040ff3c2924c5e6a54d4`。
报告哈希不同只来自各自记录的输出/解压目录路径，不影响完全一致的 OCI 身份。
该构建/验收阶段为 CPU-only，8 张 GPU 均保持 0 MiB。

v1 被选为后续运行的正式候选。它随后由一次性 Ubuntu 22.04 工具容器中的
`skopeo 1.4.1` 从只读 OCI layout 导入 Docker daemon。33 层复制、config 和
manifest 写入完整结束，工具容器自动删除。daemon tag 为
`glm52-oscar-a800-phase6-14c768b40-0275043c:latest`，image ID 精确等于
上述 image/config digest，层数为 33；source commit/tree、candidate layer、
Dockerfile、rotation manifest、runtime expectation 和 base manifest labels
均与 v1 验收值精确匹配。导入日志与 daemon inspect JSON 的 SHA256 分别为：

- `70c3dc1c4b93c1f5ec1261d7e841206ca6d7f9917be6a31e3b25a40dec10b5b9`；
- `3089a6a73058a17770bf2ea9dc0023331259400956568dcb3b4987921a1f8b5f`。

导入和 daemon 审计为 CPU-only，没有注入 NVIDIA runtime；8 张 GPU 均保持
0 MiB。

driver-injected runtime import 前在 `2026-07-31T01:48:44Z` 和
`01:49:50Z` 完成两次 8/8 空闲检查，间隔 66 秒；两次均为 0 MiB、0%
且没有 compute process。有效探针固定使用 GPU 0，只注入驱动库并执行只读
import/身份断言，实际结果为 `passed`：

- Python/PyTorch/Triton 为 `3.12.13/2.11.0+cu129/3.6.0`；
- vLLM source 与 `vllm._C` 均来自候选 `/opt/vllm_glm52_v1`；
- 78 层 rotation、manifest/rotation SHA256 和 runtime expectation 匹配；
- `reasoning_effort=max` 可解析；
- `cuda_initialized=false`。

`runtime_import.json` / log SHA256 分别为：

- `0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`；
- `f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`。

有效探针前有两次被拒绝的脚本轮次：首次 `docker run` 缺少 `-i`，导致
`python -` 未收到 stdin；第二次误用旧
`vllm.entrypoints.openai.protocol` 路径。两次都没有产生通过 JSON，也没有
初始化 CUDA。有效探针容器自动删除，`01:50:38Z` 复查 8 张 GPU 均为
0 MiB、0%，没有 compute process。

Stage 9 控制镜像 Dockerfile 随后只把默认 base 切换到上述新候选，
文件 SHA256 为
`6b6f4d1d509d744c628535e5d33b1f75f18111acf61dff29a0fa6eba1cee2d3e`；
该复现入口先由主仓库提交
`be9abfd1dfe57198e2e8a60ba9ce89de207c1476` 发布，再执行 CPU-only
构建。有效目录为
`artifacts/phase9-control/20260731T015740Z_runtime_14c768b40_v1`，
控制镜像 tag/image ID 为：

- `oscar-glm-stage9-runtime:14c768b40`；
- `sha256:84c48782f440d2080a81347bfa31ec1e3bbf77bc6151629ef43d879bbe90989f`。

控制镜像共 34 层，前 33 层与新候选逐层相同，全部 inherited labels 也完全
相等。CPU-only runtime 检查实际确认 Git `2.34.1`、iproute2 `5.15.0`、
Python/glibc `3.12.13/2.35` 与 `/opt/phase9-control-packages.txt`
清单匹配，`cuda_initialized=false`。build log、daemon inspect 与 runtime
check JSON SHA256 分别为：

- `691e629a01685dec92a1690bf68bdc552203e67cdfe522b84b7c3d708f8fad50`；
- `de86beaed7811f798af1e339c5464a3cf790e168363a037a0f35ff0196b39489`；
- `5ac65b5da3ffc9bcf642bd3beb8a989b177710d38c51a9457b5198ffd18c1f20`。

构建和验证均未分配 GPU，8 张 GPU 始终为 0 MiB。随后从新 candidate layer
机械生成正式 runtime overlay；程序化门禁确认其包含 4,744 个普通文件和
6 个 native symlink，symlink 相对路径、目标及 6 个目标文件 SHA256 均与
冻结的正式 runtime 对照精确一致。

Phase 1/5/7/9 配置、对应 wrapper、Phase 7 候选 identity 及 Phase 9 固定
control image 测试值均已迁移到源码提交 `14c768b…` 和上述新镜像。依赖顺序
重算后的四份配置 SHA256 为：

- Phase 1：
  `d588e627ee8ea663e31f97bc711eac73b0823c966a4fda529071c7e9b286c820`；
- Phase 5：
  `b7a58d100876266cd49c3363d4b1896205de13bc5ce1e7a5bc530a28e1484d66`；
- Phase 7：
  `680708fcef388db304668ff8c4bc746851cbb3f3b94abe0c573bed5e00c3b15f`；
- Phase 9：
  `1bdfabb916cb4a0ed6b629693201a49a5d8e7c15e8ccba1df093748ad9c2b379`。

4 个 JSON 解析、9 个 shell 语法检查、Python compile、旧候选身份清零扫描和
`git diff --check` 均通过。新控制镜像中，Phase 7/Phase 9 工具测试分别为
20/20 和 19/19 passed，合计 39/39、0 failed；分阶段日志 SHA256 分别为
`6203ef0610bc28347c8a0a29b1c70370a0ae748ee0753cb7d6d57ec57545f324`
和
`df2842e355c056aef7a9b106ec0fa63dfe48ce73e843edcd4a13df901abb4c89`。
唯一 warning 是只读项目挂载无法写 pytest cache，不影响测试结果。

首次在宿主机直接运行递归 verifier 时，Phase 5 报告
`source.rootfs_runtime_tree_match` 失败。展开证据表明 NFS 把普通文件 mode
统一呈现为 `777`，无法与 Git `100644` mode 比较；内容没有因此被判定为候选
身份错误。按正式控制容器的 bind-mount namespace 重跑后，Phase 9 递归
verifier 的 64/64 检查全部通过，静态结果 JSON SHA256 为
`69f217948c61cb526e408d4cd4be4570731e6be4d528a4161f16ccd588d8519d`。
该无 driver 容器随后在 fixed-environment import 阶段因缺少
`libcuda.so.1` 退出，因此这里只证明静态递归门禁通过，不能冒充完整
containerized preflight。

上述配置、wrapper、静态门禁与报告由主仓库提交
`04c96567abd77723645146c06c80db2135afcb71` 发布；恢复进度再由提交
`bf80eef75a3b3d5af3a123f35af4a3adbd144a2e` 发布。正式 preflight
`20260731T0220Z_stage9_candidate_decode_metadata_preflight_v1` 前，外层在
`2026-07-31T02:19:57Z` 和 `02:21:02Z` 两次检查 8 张 GPU，间隔 65 秒；
两次均为 0 MiB、0% 且没有 compute process。

driver-injected preflight 实际退出码为 0。`static_preflight.json` 状态为
`passed`，64/64 检查通过，SHA256 为
`7e74d36288ceed0647390d13913ff9175b1c0ab2f18ec1877da0fde6bcdd9351`。
它重新核对源码/OCI/native/rotation/BF16 evidence、冻结评测器与 Phase 9
性能配置。固定环境 import 和服务参数解析均记录
`cuda_initialized=false`，对应 JSON SHA256 为：

- fixed environment：
  `25ee886b5c2365b5ed3327484752db7f11955c082e2355cf50fd292913a8e01d`；
- parsed server args：
  `a801418f4bc96688498adb1c80837f319c62a2fd22d57bbe85a3da7e9529aa6b`。

实际解析值包括 TP=8、PP=1、`TRITON_MLA_SPARSE`、
`oscar_mla_int2`、`max_model_len=131072`、
`max_num_batched_tokens=2048`、eager、async scheduling 关闭和 torch
profiler。preflight 容器自动删除；`02:22:27Z` 退出复查为 8 张 GPU
0 MiB、0%，没有 compute process。

preflight 报告由主仓库提交
`151e1c6a91d6aaa3d53d9a36ab913c272fab85e9` 发布后，正式启动 TP=8
32K/batch1 定向探针
`20260731T0225Z_stage9_candidate_decode_metadata_probe_32k_b1_v1`。
外层空闲检查为 `02:25:27Z/02:26:32Z`，间隔 65 秒，8 张 GPU 均为
0 MiB、0% 且没有 compute process；容器内又完成两次空闲检查。服务于
`02:35:20Z` ready，负载固定为 32,768 输入 token、128 输出 token、并发 1、
1 次 warm-up、3 轮正式测量和 8+8+1 profiler，并按要求持续输出 10 分钟
进度。

三轮均为 3/3 completed、0 failed，结果为：

| 轮次 | TTFT（ms） | TPOT（ms） | 吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 106,666.970 | 200.547 | 0.007568 |
| 2 | 106,606.493 | 200.303 | 0.007573 |
| 3 | 106,660.424 | 199.203 | 0.007578 |

三轮中位数为 TTFT `106,660.424 ms`、TPOT `200.303 ms`、吞吐
`0.007573 req/s`。相对现有 BF16 v4 同格点的
`12,528.026 ms/178.832 ms/0.02838 req/s`，分别为
`+751.37%/+12.01%/-73.32%`。因此当前诊断值中的 TPOT 已落入 20% 差距
以内，但 TTFT 仍约为 BF16 的 `8.51×`；由于没有改动前同口径的 32K/batch1
OSCAR 结果，不能把 TPOT 数字直接归因于 metadata/scratch 改动。

profile 命令实际生成 8 份 worker trace、8 份 CUDA table 和 1 份 frontend
trace。rank 0 trace 有 144 个 execute context：前 16 个均为 2,048-token
prefill chunk，合计 32,768 token；首个 chunk 约 `4,088 ms`，后续 chunk
逐步增长，最后一个约 `6,894 ms`；之后 127 个 generation 窗口大多约
`266–271 ms`。这说明约 106.7 秒 TTFT 主要由 16 个 chunked prefill 窗口
累计形成。现有 trace 分析器按单 prefill 窗口设计，遇到第二个窗口时按预期
fail closed；下一步需最小扩展其多 chunk 聚合能力，再对 8 个 rank 做正式
归因。

本轮最终不能标记为通过：profile 命令结束后，runner 检测到主仓库新增了当时
尚未跟踪的 `OSCAR精度与性能优化记录.md`，触发
`RuntimeError: repository became dirty`，外层退出码为 1。门禁发生在
profiler bundle 校验和单格/总 summary 生成之前，因此上述三轮测量与 trace
只能作为中间诊断证据，不是完整单格验收结果。容器已删除，`03:21:35Z`
复查 8 张 GPU 均为 0 MiB、无 compute process。下一步先把实时优化记录纳入
Git、发布多 chunk trace 分析器及其测试，再使用新 run ID 重跑同一格点。

多 chunk 分析器随后已完成最小扩展。新版本聚合 generation 前全部正 token
prefill 窗口，同时保留单窗口既有字段，并新增 chunk 数、总 token 和逐 chunk
时长；它还会拒绝 generation 后的 prefill 与重叠窗口。双 chunk 测试先复现
旧实现预期失败，改动后与原单窗口测试共同 2/2 passed；固定控制镜像中的
Phase 9 三个工具测试文件合计 20/20 passed。分析器 SHA256 为
`8b6b2393f93be3783f47ccbe3ecb26020cc2b65526c99fcb464c4768ce4330f7`。

新分析器没有分配 GPU，只流式重放已冻结的 OSCAR 与 BF16 32K/batch1 各 8
份 worker trace。两边每个 rank 都有 144 个 execute context、16 个 prefill
chunk 和精确 32,768 个 prefill token。同口径中位数如下：

| Trace 指标 | BF16 | OSCAR | OSCAR 相对 BF16 |
|---|---:|---:|---:|
| Prefill wall（ms） | 10,086.470 | 105,753.449 | +948.47% |
| Prefill kernel 合计（ms） | 9,533.580 | 105,609.233 | +1,007.76% |
| 各 rank generation 中位数再取中位（ms） | 223.325 | 268.625 | +20.28% |

OSCAR prefill kernel 覆盖率中位数为 `99.8628%`。其
`_mixed_sparse_prefill_stage1` 精确执行 `1,248=16×78` 次，累计中位数
`93,913.327 ms`，平均约 `75.251 ms/层/chunk`，占 OSCAR prefill wall
`88.80%`；BF16 原生 `_sparse_mla_kernel_final_static` 累计中位数为
`3,384.374 ms`。这排除了约 95.7 秒差距主要来自调度、Python 或 chunk 间
空隙的解释，并把下一优化对象收敛到 grouped prefill stage1 本身。

有效输出为：

- OSCAR：
  `/dev/shm/oscar-glm-stage9-analysis-20260731T0322Z_decode_metadata_32k_prefill_trace_v1/summary.json`，
  SHA256
  `cf4887875df1577837576eb606d99e72161a5ae586d14d9c72ff7762d113ffa7`；
- BF16：
  `/dev/shm/oscar-glm-stage9-analysis-20260731T0340Z_bf16_32k_b1_prefill_trace_v1/summary.json`，
  SHA256
  `06eecce0b6b3fad99b43885bb1e83355ac6f0a518a978c3db8be268cbc8a158d`。

下一步先用单卡、单层的 2,048-query/2,048-top-k 形状验证 kernel 精度与性能
方案；仍以 output/LSE 最大绝对误差 `0.002/0.002` 为硬门限，不以性能为由
放宽精度。候选通过后再构建正式 OCI，并以新 run ID 重跑 32K/batch1。

为使该单层实验与 32K chunk 几何精确一致，Phase 9 prefill benchmark 已新增
`--seq-len`。入口要求 sequence length 大于 prefix+recent、不超过固定 top-k
2,048，且 history token 数按 16-token cache block 对齐。在
`seq_len=2,048` 时，full/cropped top-k 同为 2,048，因此只保留 IEEE
split16 参考和 grouped split1 候选两个语义唯一配置。旧实现先在新增测试中
出现 3 个预期失败；实现后固定控制镜像中的 Phase 9 三个工具测试文件为
21/21 passed。该阶段没有分配 GPU，也没有产生新的性能或精度数字；脚本与
记录发布后才会启动单卡测量。

固定入口由主仓库提交
`49c9a1e` 发布后，单卡轮次
`20260731T0345Z_prefill_2k_ieee_v1` 使用 GPU 0 和独立 Triton cache。
分配前两次 8/8 空闲检查为 `03:44:15Z/03:45:26Z`，间隔 71 秒；两次均为
0 MiB、0% 且没有 compute process。固定 5 次 warm-up、7 次正式测量、每次
1 iteration 的结果为：

| 配置 | CUDA 中位数 | 峰值增量显存 | 相对 split16 |
|---|---:|---:|---:|
| IEEE split16 | 195.772 ms | 1,185.063 MiB | 1.000× |
| Grouped IEEE split1 | **47.158 ms** | **224.125 MiB** | **4.151×** |

grouped split1 相对 split16 的 output/LSE 最大绝对差为
`3.337860107421875e-06/9.5367431640625e-07`，低于
`0.002/0.002` 门限，轮次状态为 `passed`。结果 JSON/log SHA256 为
`f98578db5b6326ff19cf4e62a284c47ca95869f66eb19413f00507570c071f8a` /
`cf0b5f7da96cc4b491b1022552b4b07fe1b16a8d5b1b498f6ece4dd992b24616`。
容器自动删除，退出后 8 张 GPU 均为 0 MiB、无 compute process。下一步以
该 IEEE split16 为固定严格参考，测试 grouped TF32；只有误差继续满足门限且
单层时间实际下降，才进入源码候选。

grouped TF32 候选随后以源码提交
`24938975f70bbf6d502b3556bdb44de0a5c7bde7` 落地并推送。改动仅把
`_mixed_sparse_prefill_stage1` 中 5 个 score/value `tl.dot` 的
`input_precision` 从 `ieee` 切换为 `tf32`；decode stage1、split16 IEEE
参考、三段式/INT2 语义、softmax/global LSE、inverse rotation、cache
write/demotion 和调度均未改变。

固定控制镜像中的 Triton interpreter 与 prefill head-block 定向测试为
6/6 passed；源码提交时全部适用 pre-commit hooks 通过，包括 ruff
check/format、typos、mypy、SPDX、forbidden imports、CUDA API 与 attention
backend 文档门禁。主仓库提交
`b0370c3d7b168233fd7ed3e568fb04df343f0c51` 随后绑定该源码、submodule
和实验前记录并完成推送。

单卡筛选轮次 `20260731T0354Z_prefill_2k_tf32_v1` 前，两次 8/8 GPU
空闲检查为 `03:52:49Z/03:53:54Z`，间隔 65 秒，均为 0 MiB、0% 且没有
compute process。该轮没有进入正确性检查、warm-up 或计时：Triton 编译
grouped TF32 stage1 时需要 `169,984 bytes` shared memory，超过苹果800
单 block 上限 `166,912 bytes`，超出 `3,072 bytes`。容器退出码为 1，
没有生成 `result.json`；独立 cache 留下 53 个编译文件。日志与当轮 kernel
源码 SHA256 分别为
`3024ebc944d713a68b498d3e51703370bc7f388e5f109faeb2b15c2e4f0f19a1` /
`d74c04b0fc227209415cca4ef2c8b6a9ee1f5b9c86ce12045b58dbcdd56040ea`。
退出后 8 张 GPU 均为 0 MiB、0%，没有 compute process。

因此，源码 `24938975f…` 的当前 TF32 形态被 shared-memory 门禁拒绝，
没有任何 TF32 精度或性能结果，也不能宣称满足 `0.002/0.002`。下一步先最小
降低 grouped kernel 的 shared-memory 占用，再重新发布并按相同协议筛选。

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
| OSCAR 固定性能矩阵与比较 | 优化中 | grouped prefill TP=8 1K/b1 为 1,317.120/202.668 ms；同口径 trace 为 prefill +445.12%、generation +27.09%，KV update CPU 增量约 43.05 ms/token；源码 `14c768b…` 的 metadata/scratch 优化为 CPU 96 passed/29 CUDA skip、苹果800 CUDA 125/125 passed，新 OCI 两次确定性构建/验收、Docker daemon identity、runtime import、新控制镜像审计、工具测试 39/39、容器内递归静态 verifier 64/64 及 driver-injected preflight 均通过；32K/b1 三轮诊断中位数为 106,660.424/200.303 ms，相对 BF16 为 +751.37%/+12.01%，但整轮因新增未跟踪文档触发仓库洁净门禁，未生成单格 summary，不能标记为通过；多 chunk trace 进一步量化 OSCAR/BF16 prefill wall 为 105,753.449/10,086.470 ms，OSCAR grouped prefill stage1 占 88.80%；2,048×2,048 单层 grouped IEEE split1 为 47.158 ms、相对 IEEE split16 加速 4.151×且严格误差通过；TF32 源码 `24938975f…` 通过 CPU/静态门禁，但首次 GPU 编译需 169,984-byte shared memory、超过 166,912-byte 上限，未进入精度或性能测量；需降低资源占用后重筛，再以新 run ID 重跑；之后仍需跑同提交完整矩阵 |
| 128K 扩展 | 未完成 | 将随 OSCAR 候选轮次验证 |
