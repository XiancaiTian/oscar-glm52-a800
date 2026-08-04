# GLM-5.2 OSCAR 苹果800适配与性能优化报告

> 报告日期：2026-08-04
> 模型：`/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001`
> 主要环境：单机 8×苹果800-80GB、TP=8、PP=1、`TRITON_MLA_SPARSE`、eager
> 数据来源：项目内已落盘的 `artifacts`、运行摘要、验证结果与 Git 记录

所有数字均来自已经落盘的运行摘要、验证结果或性能记录。

## 第一部分：GLM-5.2 OSCAR 苹果800适配、原理与精度

### 1.1 目标、约束与技术路线

目标是在不破坏既有 GLM-5.2 推理能力的前提下，将 OSCAR 的低比特 KV history 引入 GLM 5.2 苹果800 定制 vLLM（以下称 GLM52-苹果800-vLLM），使模型可以在 8×苹果800、TP=8 上运行 DSA/sparse MLA，并完成长上下文、并发、精度和性能验证。

参考 OSCAR 实现面向较新的 vLLM 和 full-attention 模型；GLM52-苹果800-vLLM 则包含定制 vLLM、GLM-5.2、FP8 MoE、DSA、sparse MLA 和既有 TP=8 路径。若整体升级或直接合并，会同时改变模型、调度、分布式和 attention，无法判断问题来自 OSCAR 还是基础运行时。

因此采用“GLM52-苹果800-vLLM 优先、分层回移 OSCAR”的路线：

1. 保留已验证的 GLM-5.2、FP8 MoE、DSA/MLA、TP=8 和苹果800运行路径；
2. 迁移 OSCAR 配置、artifact、三段式 allocator 和必要运行时接线；
3. 把 full-attention 的双 rotation 改造成共享 latent 的单 rotation；
4. 实现 BF16 prefix、BF16 recent、INT2 history 三段式 KV cache；
5. 将 cache ownership、页分配、抢占和释放接入 scheduler/worker；
6. 为苹果800实现并验证 SM80 Triton store、demotion、mixed attention 和 inverse rotation；
7. 如果配置不符合要求就直接停止运行，防止系统没有真正使用 OSCAR，却仍然显示运行成功。

### 1.2 为什么 MLA 不能直接照搬原始 OSCAR

full-attention 分别缓存 K 和 V，因此原始 OSCAR 可以为 K、V 使用两套 rotation。GLM-5.2 的 MLA 缓存共享 `compressed_kv`；同一 latent 同时参与 attention score 和 value 聚合。如果仍保留两套旋转副本，会重复存储共享 latent，抵消 MLA 和 OSCAR 的压缩收益。

从模型定义看，每个历史 token 都需要先由 latent 生成 key 和 value：

```text
k_i = c_i · W_UK
v_i = c_i · W_UV
```

但实际 MLA 推理利用矩阵结合律调整了计算顺序。非 RoPE key 的 up-projection 被吸收到当前 query 一侧：

```text
q_abs = q · W_UKᵀ
score_i = q_abs · c_iᵀ
```

因此，key up-projection 并没有被省略，而是从每个历史 token 转移为每个当前 query 只计算一次。RoPE 对应的 key 仍单独保存和计算，不参与这种吸收。

value 路径也利用线性运算的结合律，把 up-projection 移到 attention 聚合之后：

```text
Σ p_i · (c_i · W_UV) = (Σ p_i · c_i) · W_UV
```

也就是说，系统先在低维 latent 空间完成加权聚合，最后只对聚合结果执行一次 value up-projection，从而避免对每个选中的历史 token 分别生成高维 value。

本项目改为每层只保存一个 `512×512` 正交矩阵 `R`，让 score 和 value 共用同一旋转空间：

```text
c_rot = c · R
q_rot = q_abs · R
```

其中 `c` 是原始共享 latent，`q_abs` 是吸收投影后的 query。因为 `R·Rᵀ=I`：

```text
q_rot · c_rotᵀ
= q_abs · R · (c · R)ᵀ
= q_abs · R · Rᵀ · cᵀ
= q_abs · cᵀ
```

所以仅做正交旋转、不量化时，attention score 与原生 MLA 等价。value 在旋转空间聚合后再逆旋：

```text
z_rot = softmax(score) · c_rot
z = z_rot · Rᵀ = softmax(score) · c
```

结合 DSA、RoPE 和三段式 cache 后，实际执行顺序如下：

1. 当前 query 在 query 侧完成非 RoPE key 投影吸收，得到 `q_abs`；RoPE query 仍走原生路径；
2. DSA 选择需要参与计算的历史 token，并将它们映射到 BF16 prefix、BF16 recent 或 INT2 history；
3. prefix/recent 使用 `q_abs` 与原始 latent 计算 score；history 使用 `q_abs·R` 与反量化后的 `c·R` 计算 score；
4. 各分段叠加原生 RoPE score，再通过统一的 log-sum-exp 得到全局 softmax 权重；
5. prefix/recent 在原始 latent 空间聚合，history 在旋转后的 latent 空间聚合；
6. history 聚合结果乘 `Rᵀ` 恢复到原始 latent 空间，再与 prefix/recent 聚合结果合并；
7. 最后对合并后的 latent 执行一次 GLM-5.2 原生 value up-projection，生成 attention 输出。

因此，key up-projection 实际被吸收到 query 侧，value up-projection 实际被移动到 latent 聚合之后。rotation 只是可逆的正交换基，本身不会造成近似误差；误差主要来自 history 的 clipping 和 INT2 量化。

### 1.3 Shared latent calibration 与 rotation artifact

参考实现分别采集 K、V；MLA 适配后需要从每层采集并按 TP=8 合并：

- 共享 latent activation、covariance 和 reservoir；
- DSA query/score covariance 与样本；
- value covariance 与样本；
- 独立选择 rotation 和 clipping 参数所需的 holdout 数据。

Capture 只在离线 calibration 阶段显式开启，逐层限额；正式推理只加载最终 artifact，不持续采集 activation。模型几何、checkpoint、expert mapping、TP rank 和 split 身份均参与校验。

正式采集结果如下：

| Split | 请求结果 | Prompt tokens | Capture 文件 |
|---|---:|---:|---:|
| Train | 256/256 HTTP 200 | 900,000/900,000 | 624/624 |
| Holdout | 36/36 HTTP 200 | 100,000/100,000 | 624/624 |

三个 alpha 的 holdout loss 为：

| Alpha | 归一化 holdout loss |
|---:|---:|
| 0.25 | 0.026186010882512642 |
| 0.5 | 0.02872817461999513 |
| 0.75 | 0.032144202654983196 |

最终选择 `alpha=0.25`。78 层中，61 层使用 clip ratio 0.92，17 层使用 0.94。

### 1.4 三段式 KV cache 设计

每层 cache 被拆为三段：

| 分段 | 精度与布局 | 作用 |
|---|---|---|
| Prefix | 每请求固定 64 tokens，BF16 | 保留请求起始边界的高精度信息 |
| Recent | 每请求固定 256 tokens，BF16 ring | 保留最新上下文，并承担后续 demotion |
| History | 16-token page 增量分配，INT2 | 压缩长历史 latent |

当前 latent rank 为 512，group size 为 128。每层每个 history token 包含 128 bytes packed INT2 data 和 32 bytes scale/zero metadata，合计 160 bytes；相对 BF16 latent 的理论压缩率为 6.4×。该比值只针对 history data，不能代表包含 prefix、recent、RoPE、DSA/index cache 后的整机显存压缩率。

写入和降级顺序为：prefix 写 BF16 prefix；新 token 写 BF16 recent；recent token 变旧时执行 `rotation → clipping → INT2 quantize → pack`；再写入 history page。运行时必须先 demote 即将被覆盖的 recent token，再复用 ring slot，且不保留完整 BF16 history 副本。

### 1.5 DSA/MLA mixed attention 数据流

OSCAR 不修改 DSA 的 top-k 选择语义。DSA 仍用原精度 index/cache 产生 token ID；适配只改变选中 token 的存储位置解析和 MLA 计算：

1. metadata 将选中 ID 映射到 prefix、recent 或 history；
2. prefix/recent 直接读取 BF16 latent，并使用 `q_abs` 计算；
3. history 在 kernel 内按 group 反量化 INT2 latent，并使用 `q_abs·R` 计算；
4. RoPE key cache 保持原精度，三个分段叠加同一条原生 RoPE score 路径；
5. 三个分段分别计算局部 max、exp 和 accumulator；
6. 通过全局 log-sum-exp 合并为一个 softmax，不能把三段当成三个独立 attention；
7. history accumulator 乘 `Rᵀ` 回到原 latent 空间，再与 prefix/recent accumulator 相加；
8. 合并结果进入原生 value up-projection。

这一设计同时保留了共享 latent 的显存优势、DSA 稀疏选择语义、RoPE/辅助 cache 精度、prefix/recent 高精度边界，以及 history INT2 压缩。

### 1.6 Scheduler、ownership 与生命周期

每个请求需要维护 prefix row/slot、recent ring、history block/page table、demotion position/page/offset、stable request index 和精确 sequence length。

finish、abort、preemption、request reuse 和分配失败回滚均同时维护三段式 cache 与原生辅助 cache。history OOM 时回滚 length、page 和 version，避免 scheduler 只提交一部分状态；请求复用时必须清理旧 ownership，防止跨请求读取。

14 GiB、78 层、`max_num_seqs=16` 的 CPU planner 得到 579,440 个 logical token slots；native planner 为 162,256，理论容量比为 `3.5711468297012128×`。这是确定性 planner 结果，不是 GPU 峰值容量实测。

### 1.7 苹果800 Triton kernel 适配与数值门禁

为苹果800实现或适配的 kernel 包括 shared latent rotation、clip、INT2 pack/store、BF16 prefix/recent store、recent-to-history demotion、history dequant、mixed sparse MLA decode/prefill、DSA selected ID/padding mask、RoPE score 合并、global LSE merge 和 inverse rotation。

由于其他架构上的可编译或可运行结果不能证明苹果800可用，项目设置了独立硬门禁：全新 Triton cache cold compile、真实 kernel launch、非连续 rotation stride、跨页 store/dequant、mixed decode/prefill、多请求 ownership 隔离和 8 卡 rank-local cold compile。

实测结果：

| 门禁 | 结果 |
|---|---:|
| GPU 0 cold-cache 完整套件 | 114/114 passed |
| 8 卡 rank-local | 224/224 passed，其中 208 次实际 CUDA kernel |
| 真实 artifact rotation oracle 最大绝对误差 | 3.0994415283203125e-06 |
| 真实 artifact dequant oracle 最大绝对误差 | 3.0994415283203125e-06 |

### 1.8 vLLM 运行时接入与 fail-closed 边界

项目注册 `oscar_mla_int2` 为显式 KV cache dtype。激活后，`MLAAttention.get_kv_cache_spec` 创建三段式 `OscarMLAAttentionSpec`，而不是标准单 KV tensor。

当前只允许已验证组合：`TRITON_MLA_SPARSE`、eager、TP=8、PP=1，并关闭 vLLM prefix caching、CUDA graph、speculative decoding、asynchronous scheduling、context parallelism、dual batch overlap、KV transfer 和 KV offloading。artifact 与运行时期望必须完全匹配；不允许静默回退到 BF16 history、dense MLA 或 full attention。

这些限制定义当前验证边界，不表示相关能力以后不能支持。

### 1.9 端到端功能与容量验证

OSCAR 服务完成 141/141 权重分片加载。每个 rank 的实际计划为：

| 指标 | 每 rank 实际值 |
|---|---:|
| logical capacity | 637,632 tokens |
| BF16 prefix | 1,024 slots / 81,788,928 bytes |
| BF16 recent | 4,096 slots / 327,155,712 bytes |
| INT2 history | 637,696 slots / 9,964 pages / 7,958,446,080 bytes |
| BF16 RoPE | 6,366,756,864 bytes |
| native index/cache | 1,767,693,312 bytes |
| allocated capacity ratio | 3.5812365205× |

短请求、506+5、26+384、31,996+64 和 8 请求并发均返回 HTTP 200。运行日志确认 78 层均实际执行 store、recent-to-INT2 demotion 和 DSA-selected mixed read；没有完整 BF16 history，也没有 dense/full-attention fallback。

### 1.10 适配后的精度对比

适配阶段使用固定 256 道 GSM8K、8K、`reasoning_effort=high`、concurrency 16。OSCAR 运行 `20260730T0438Z_candidate_fast256_c16_docker_v3` 的结果为：

| 指标 | BF16 baseline | OSCAR | 差值 |
|---|---:|---:|---:|
| 题目数 | 256 | 256 | 0 |
| 正确数 | 105 | 107 | +2 |
| Accuracy | 0.41015625 | 0.41796875 | +0.0078125 |
| Request failure | 未保存逐题明细 | 0 | 不作比较 |

OSCAR 完成 256/256 scored，截断 128 题，平均 completion tokens 为 4,194.87890625，validation 状态为 `passed`。

这组结果证明适配后的 OSCAR 没有出现总体精度坍塌，单轮结果与 BF16 baseline 接近。但两边保存的协议指纹不同，且缺少 BF16 逐题 predictions，无法做逐题翻转统计。因此不能把多 2 题解释为 OSCAR 带来的可归因精度提升；准确结论是：在现有单轮证据下，OSCAR 为 107/256，BF16 汇总为 105/256，两者处于同一水平。

后续性能优化分支继续采用固定 256 题作为晋升门禁。历史上曾实测 K=1,536、
K=1,024、K=768 和 split-K，但这些运行改变了与 BF16 baseline 对比所需的 top-k 身份。
按最新冻结口径，prefill 与 decode top-k 都必须与 BF16 baseline 的 2,048 一致，
top-k 变化不得视为优化项。历史结果只作为无效对比边界保留，详见 2.15。

## 第二部分：TTFT 与 TPOT 性能优化、效果及当前结论

### 2.1 指标、负载与晋升原则

- TTFT：从发起请求到首 token 返回的时间，32K/batch1 下主要反映长 prefill、调度和首 token 路径。
- TPOT：首 token 之后每个输出 token 的平均时间，主要反映 decode、跨 rank 同步和调度稳定性。
- 当前正式优化负载：32K/batch1；固定 256 题 GSM8K 用于精度筛选。
- 候选必须依次通过静态/CPU 合同、苹果800 CUDA correctness、固定 256 题精度门禁和 32K/batch1 端到端复测。单卡 kernel 更快不能直接晋升。

本文只使用最终同源码 32K/batch1 BF16 结果作为性能 baseline：

| 负载 | BF16 TTFT | BF16 TPOT | BF16 top-k |
|---|---:|---:|---:|
| 32K/batch1 | 12,507.854171 ms | 151.236017 ms | Prefill/Decode 均为 2,048 |

早期 1K/batch1 只用于定位优化方向，不再引用另一组 BF16 结果，也不与 32K baseline 跨负载比较。该阶段的初始 OSCAR 为 TTFT/TPOT `5049.520/243.127 ms`；主要瓶颈为 78 层重复 mask/nonzero/index、prefill 固定 2,048 top-k 与 split16、8 个本地 head 重复读取/反量化 KV，以及逐层 metadata/scratch 分配。

### 2.2 Decode 索引与 metadata 快路径

改动内容：把跨层不变的 token partition、mask、`nonzero` 和 index 从每层 attention 热路径移到 worker metadata；复用 demotion scratch，减少逐层 CPU 构造、同步和临时分配。

KV update CPU `16.107→9.293 s`，下降 42.3%；CUDA `1.080→0.439 s`，下降 59.3%；其中 `nonzero` 下降 99.2%，index 下降 90.1%。1K/batch1 端到端结果为：

| 指标 | 上一版本 | 当前版本 | 当前相对上一版 |
|---|---:|---:|---:|
| TTFT | 5,049.520 ms | 5,014.582 ms | -0.692% |
| TPOT | 243.127 ms | 206.735 ms | -14.968% |

该优化主要作用于 decode，因此 TPOT 收益明显，TTFT 变化很小。

后续 metadata/demotion scratch 复用通过 CPU 96 项和苹果800 CUDA 125 项门禁。首次 32K 运行因仓库洁净门禁未生成 summary，不能为这一子项单独给出端到端收益。

### 2.3 Prefill 有效 top-k 裁剪与 split 收敛

初始实现无论实际序列长度都按固定 2,048 top-k 工作，并用 split16 产生额外中间结果和 LSE merge。改动只在序列实际可见 token 少于 2,048 时裁掉越界槽位，配置的 prefill/decode top-k 仍保持 2,048；同时将可行场景收敛到 split1，减少无效 score/value 计算与归并。

单层从 `62.884→46.382 ms`，下降 26.24%。1K/batch1 端到端结果为：

| 指标 | 上一版本 | 当前版本 | 当前相对上一版 |
|---|---:|---:|---:|
| TTFT | 5,014.582 ms | 3,714.821 ms | -25.920% |
| TPOT | 206.735 ms | 205.908 ms | -0.400% |

该优化明确针对 prefill，因此主要降低 TTFT。

### 2.4 Grouped prefill：跨 head 共享 KV 读取与反量化

原实现让 8 个本地 query head 分别读取、反量化同一组 KV。grouped kernel 在同一 program 中共享 KV tile、scale/zero 和中间值，再为多个 head 计算 score/value，消除重复全局内存读取和反量化。

单层从 `46.382→13.284 ms`，加速 3.491×；相对最初 `62.884 ms` 加速 4.735×。1K/batch1 端到端结果为：

| 指标 | 上一版本 | 当前版本 | 当前相对上一版 |
|---|---:|---:|---:|
| TTFT | 3,714.821 ms | 1,317.120 ms | -64.544% |
| TPOT | 205.908 ms | 202.668 ms | -1.574% |

吞吐相对上一版本提高 10.47%。

精度方面，BF16 tensor-core 快版的 output/LSE 最大误差分别为 `0.009153/0.002593`，超过冻结的 0.002 门限，因此被拒绝。最终使用 FP32 IEEE dot，output/LSE 最大误差为 `3.874e-6/1.431e-6`，并通过 cold-cache CUDA 124/124。这里体现了性能优化的基本原则：不以超出门限的数值误差换取速度。

### 2.5 32K/batch1 优化阶段的起点

切换到正式 32K/batch1 负载后，首次诊断值为 TTFT/TPOT `106,660.424/200.303 ms`。该轮因仓库洁净门禁未生成正式 summary，只能作为后续优化的诊断起点，不能作为正式通过版本。

以下所有 32K 表格统一使用 2.1 的最终 BF16 baseline。2.6–2.17 的早期候选与该 baseline 并非同源码，因此“相对 baseline”只用于统一观察差距；2.18 才是严格的同源码比较。

### 2.6 Value 精度恢复

改动内容：修正 grouped prefill 的 value 计算路径，在满足冻结数值门限的前提下降低 stage1 开销，避免使用误差超标的 BF16 tensor-core 快版。

| 指标 | 上一版本（诊断） | 当前版本 | 当前相对上一版 | BF16 baseline | 当前相对 baseline |
|---|---:|---:|---:|---:|---:|
| TTFT | 106,660.424 ms | 47,143.207 ms | -55.801% | 12,507.854 ms | +276.909% |
| TPOT | 200.303 ms | 199.458 ms | -0.422% | 151.236 ms | +31.885% |

TTFT 大幅下降，但相对最终 BF16 baseline 仍慢 276.909%。

### 2.7 Grouped prefill 改为 8 warps

改动内容：把 grouped prefill 的启动参数从 4 warps 调整为 8 warps，提高并行度，并将实际 cubin 从 255 registers、656-byte stack 改善为 247 registers、0-byte stack。

| 指标 | 上一版本 | 当前版本 | 当前相对上一版 | BF16 baseline | 当前相对 baseline |
|---|---:|---:|---:|---:|---:|
| TTFT | 47,143.207 ms | 41,618.560 ms | -11.719% | 12,507.854 ms | +232.739% |
| TPOT | 199.458 ms | 201.347 ms | +0.947% | 151.236 ms | +33.134% |

TTFT 继续改善，但 TPOT 小幅回退。

### 2.8 Grouped prefill 扩展到 8-head

改动内容：让一个 grouped program 同时覆盖更多本地 query heads，进一步复用同一批 KV tile、scale 和 zero-point，减少重复读取与反量化。

| 指标 | 上一版本 | 当前版本 | 当前相对上一版 | BF16 baseline | 当前相对 baseline |
|---|---:|---:|---:|---:|---:|
| TTFT | 41,618.560 ms | 36,245.415 ms | -12.910% | 12,507.854 ms | +189.781% |
| TPOT | 201.347 ms | 199.205 ms | -1.064% | 151.236 ms | +31.718% |

本项同时改善 TTFT 和 TPOT。

### 2.9 Causal loop 精简

改动内容：根据 causal 边界减少无效循环、越界 tile 和不必要的边界处理，使长 prefill 只计算当前 token 实际可见的历史范围。

| 指标 | 上一版本 | 当前版本 | 当前相对上一版 | BF16 baseline | 当前相对 baseline |
|---|---:|---:|---:|---:|---:|
| TTFT | 36,245.415 ms | 35,683.893 ms | -1.549% | 12,507.854 ms | +185.292% |
| TPOT | 199.205 ms | 197.826 ms | -0.692% | 151.236 ms | +30.806% |

### 2.10 BF16 tile gate

改动内容：先判断一个 tile 是否只包含 BF16 prefix/recent token；满足条件时绕过不需要的 INT2 history 反量化路径，同时保留数值正确性门禁。

| 指标 | 上一版本 | 当前版本 | 当前相对上一版 | BF16 baseline | 当前相对 baseline |
|---|---:|---:|---:|---:|---:|
| TTFT | 35,683.893 ms | 32,683.066 ms | -8.409% | 12,507.854 ms | +161.300% |
| TPOT | 197.826 ms | 200.037 ms | +1.118% | 151.236 ms | +32.268% |

TTFT 收益明显，但 TPOT 小幅回退。

### 2.11 Top-k 索引排序

改动内容：对 prefill 选出的 top-k token ID 排序，改善后续 KV 读取的地址局部性，减少跨页和随机访存。

| 指标 | 上一版本 | 当前版本 | 当前相对上一版 | BF16 baseline | 当前相对 baseline |
|---|---:|---:|---:|---:|---:|
| TTFT | 32,683.066 ms | 32,449.245 ms | -0.715% | 12,507.854 ms | +159.431% |
| TPOT | 200.037 ms | 199.155 ms | -0.441% | 151.236 ms | +31.685% |

### 2.12 Contiguous inverse rotation

改动内容：把 inverse rotation 所需输入整理为连续布局，减少非连续 stride 带来的访存和地址计算开销。

| 指标 | 上一版本 | 当前版本 | 当前相对上一版 | BF16 baseline | 当前相对 baseline |
|---|---:|---:|---:|---:|---:|
| TTFT | 32,449.245 ms | 30,539.197 ms | -5.886% | 12,507.854 ms | +144.160% |
| TPOT | 199.155 ms | 202.514 ms | +1.687% | 151.236 ms | +33.906% |

该项降低了 TTFT，但 TPOT 出现回退，说明同一改动对 prefill 和 decode 的影响可能相反。

### 2.13 History kernel 重构候选

改动内容：分别尝试手工 value reduction、把 score/LSE/value 拆成三个 kernel，以及使用 `h4+maxnreg128` 调整并行度和寄存器上限，希望降低 history stage 的执行时间。

| 候选 | 相对上一单卡参考 | 32K TTFT/TPOT | 相对 BF16 baseline | 结论 |
|---|---:|---:|---:|---|
| 手工 value reduction | 慢 18.152 ms | 未运行 | 无法计算 | 单卡门禁淘汰 |
| score/LSE/value 三 kernel | 慢 32.902 ms | 未运行 | 无法计算 | 单卡门禁淘汰 |
| `h4+maxnreg128` | 慢 10.073 ms | 未运行 | 无法计算 | 单卡门禁淘汰 |

这些候选在单卡筛选阶段已经慢于上一版本，因此没有占用 8 卡运行端到端测试，也不存在可报告的 baseline 对比。

### 2.14 History compact-load

改动内容：把 history 的 packed data、scale 和 zero-point 改成更紧凑的批量加载，减少单个 stage1 kernel 的访存指令。单卡微基准从 `20.4319→16.8192 ms`，改善 17.68%。

| 指标 | 上一版本 | compact-load | compact-load 相对上一版 | BF16 baseline | compact-load 相对 baseline |
|---|---:|---:|---:|---:|---:|
| TTFT | 30,539.197 ms | 32,843.679 ms | +7.546% | 12,507.854 ms | +162.584% |
| TPOT | 202.514 ms | 195.271 ms | -3.577% | 151.236 ms | +29.117% |

虽然 TPOT 改善，但 TTFT 反而回退 7.546%，因此该候选被淘汰。回滚后恢复到 TTFT/TPOT `30,519.625/197.274 ms`；相对 compact-load 分别为 `-7.076%/+1.026%`，相对最终 BF16 baseline 分别为 `+144.004%/+30.441%`。这证明单卡 kernel 更快不等于 32K 端到端更快。

### 2.15 Top-k 对比身份纠正（非优化项）

BF16 固定 256 题 baseline 的冻结配置为 `model.geometry.index_topk=2048`；13 份 Phase 1
static preflight 均实测 `actual=expected=2048`，实际 kernel dispatch 的 indices 宽度与
`topk` 也均为 2,048。该时代源码没有独立 prefill top-k override，prefill 直接使用
模型 `topk_tokens`；因此 baseline 的 prefill/decode top-k 均为 2,048。

历史上曾运行 K=1,536、K=1,024、K=768 与 prefill K=768/decode K=1,024 的 split-K。
这些数据保留为实验历史，但因 top-k 与 BF16 baseline 不同，不得继续视为性能优化项，
也不得用于“同样负载”的最终 TTFT/TPOT 对比：

| 历史身份 | 已落盘结果 | 当前分类 |
|---|---:|---|
| K=1,536 | 256题106正确；TTFT/TPOT 25,682.410/198.952 ms | top-k 不同，非同负载对比 |
| K=1,024 | 256题108正确；TTFT/TPOT 21,032.014/197.719 ms | top-k 不同，非同负载对比 |
| K=768 | 256题97正确；32K未运行 | top-k 不同且精度门禁失败 |
| Prefill K=768 / Decode K=1,024 | TTFT/TPOT 17,880.081/199.606 ms | top-k 不同且存在 stride 回归 |

2026-08-04 启动的 `ea8-prefill1024-fast256-v2` 实际身份为 prefill/decode K=1,024。40 分钟
固定节点为24/256完成、15题正确、7题截断；发现与baseline身份不符后已人工停止。该结果
不是正式精度结果、不用于优化结论。停止后容器/tmux消失，GPU0–7均为0 MiB/0%且
compute为空；人工中止路径未生成`wrapper.exit`，不得写为自然终局。

独立审计为22/22 checks passed，manifest从证据目录复算8/8全部`OK`。证据位于
`artifacts/phase9-control/20260804T0140Z_topk_baseline_identity_audit_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `baseline_identity.json` | 812 bytes | `655d983855ad5b90e112f15f9da7e1930564a043fc4d5194d2d2560b205ffb8e` |
| `invalid_candidate_identity.json` | 479 bytes | `a654eb8f9b699cbc9e6e96266f868b452324b721767a5b97e632e4fc2435e524` |
| `post_stop_gpu.csv` | 64 bytes | `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f` |
| `validation.json` | 3,766 bytes | `5aaa90a033ab806610ea83c3a9c3dfa89cf5d122055c5bf88cb74db3305c3a2f` |
| `evidence_manifest.sha256` | 721 bytes | `6df2dbe068415a5468899c0faa02d086536ebba44891f646888a2082346aaf3d` |
| `manifest_check.txt` | 239 bytes | `afad0e186f09be9e74e9bab2ae553411b323ba23550e80da88ef42719da06c30` |

回退实现采用TDD：首先只将`test_candidate_runtime_environment_is_wired`和
`test_candidate_index_topk_override_is_wired`的期望改为2,048，并要求配置、容器wrapper、
candidate wrapper与verifier不再包含1,024/768的旧top-k字面量。在固定
`oscar-glm-stage9-runtime:ea8ae6b77`、network none且无GPU的control容器中，两个
目标测试均按预期失败：实际prefill为1,024而期望2,048，实际`index_topk`为
1,024而期望2,048。该红灯已进入目标断言，production四文件尚未修改，因此为有效合同红灯。

绿灯阶段只修改4个production/config文件，将prefill top-k和`index_topk`从1,024恢复为
2,048；加上红灯阶段已发布的合同测试，本次完整变更范围共5个文件。decode backend仍为
`legacy`、prefill排序开关仍为1，source commit仍为`ea8ae6b7758ae2b4db7cae44d638ae5de80148ac`，
其余运行负载身份未改变。固定control容器内目标测试2/2、Phase 9工具测试25/25、Phase 9
递归测试90/90以及两个wrapper的shell语法和performance JSON语法全部通过。

结构化证据生成器首轮为23/24：唯一失败是生成器要求当前diff至少出现9处新的2,048
字面量，但合同测试已在红灯提交中发布，本轮4个production/config文件的合法diff实际只有
8处。保留该失败结果并将生成器期望修正为8后，最终validation为24/24、manifest为9/9。
证据位于`artifacts/phase9-control/20260804T0200Z_ea8_topk2048_contract_tdd_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `validation_attempt1.json` | 4,836 bytes | `324c2db6e7ab479a8e7985c56b3cf383718cdcfc1b8026719a3ceb895d6fa842` |
| `target_test.log` | 775 bytes | `3a6aa23a9dfffeb58142194d62029281882475b9b9faf1cb31c83c262ba5655c` |
| `phase9_tools_test.log` | 643 bytes | `1ae3ed14e938e5aebdff67b18b6373d0abd2f1bfd261a840ae3db712e10dda80` |
| `phase9_recursive_test.log` | 4,209 bytes | `1e9cfb43deec8ec50de50ac20b4ab9a4f76732635fd8ae363859fd795f14e7f8` |
| `shell_syntax.log` | 157 bytes | `1fa5541b0d14002ab68a1fab055167360b99cfb523b464459b322145a94355ab` |
| `candidate_change.diff` | 5,153 bytes | `41ecfe1db5d18c76ea5a360beaed2d7bf3aeebd391a776e3629411f2b90aa4f7` |
| `performance_matrix.json` | 3,615 bytes | `fa5d0bf78ccce903ba7a387daedeab542c2e6649d076d73ab04a493ba5c54ea1` |
| `validation.json` | 4,835 bytes | `4c217da10a8eedc832b7d2960b68ad0959409afdec6eddb1bc47047c35bc9db0` |
| `evidence_manifest.sha256` | 780 bytes | `e371ac9de6478501426af9390b34f5c9bc29cc1a0f0667f94021e0ce2ffed0a9` |
| `evidence_manifest_check.txt` | 236 bytes | `cf2995f4a2eb29afb7b0fe33d9791abd4783023e0559ebd1799e78e2c4c16a47` |

至此所有活动配置与fail-closed消费者均已恢复为prefill/decode top-k 2,048。下一步重新执行
固定256题精度门禁；top-k从此作为冻结负载身份，不再作为优化方向。

### 2.16 Inverse rotation 与最终加法融合

改动内容：把 `history_merged` 的 inverse rotation 和后续 FP32 add 融合成一个 kernel，直接写最终 output，减少中间 tensor、显存读写和一次独立 kernel launch。

| 几何 | 旧路径 | 融合路径 | 融合相对旧路径 | 临时分配 |
|---:|---:|---:|---:|---:|
| 8×512 | 0.056816640 ms/call | 0.027653120 ms/call | -51.329% | 16,384→0 bytes |
| 16,384×512 | 0.910182381 ms/call | 0.853913593 ms/call | -6.182% | 33,554,432→0 bytes |

| 指标 | 历史非同口径 split-K | 融合版本 | 融合相对历史版 | BF16 baseline | 融合相对 baseline |
|---|---:|---:|---:|---:|---:|
| 32K TTFT | 17,880.081 ms | 未运行 | 无法计算 | 12,507.854 ms | 无法计算 |
| 32K TPOT | 199.606 ms | 未运行 | 无法计算 | 151.236 ms | 无法计算 |

融合 helper 的 BF16/FP32 latent 微基准均保持 bitwise equal，但固定 256 题得到 0/256、256 题答案提取失败、233 题截断。精度门禁失败后禁止运行 32K，因此微基准收益没有转化为端到端收益，候选已回退。

### 2.17 Top-k 输出连续化（stride 修复）

改动内容：历史 split-K 路径的基础输出 buffer 宽度为 1,024，prefill 取 `[:, :768]` 后逻辑宽度虽为 768，行 stride 仍为 1,024；native CUDA producer 按连续 768 写入，下游却按 stride 1,024 读取。修复方案是在目标非连续时创建显式连续临时 tensor，native top-k 写完后再 copy-back。这是输出布局 correctness 修复，不改变 top-k 的配置值；后续 K=2,048 路径仍保留该连续性保护。

GPU0 native 最小复现已确认根因：非连续 view 得到错误结果，连续临时 tensor 加 copy-back 与 reference 完全一致，且不会覆盖原 buffer 尾列。pre-fusion 回退控制的最终结果为 0/256、236 题截断，进一步证明仅回退 inverse-rotation 融合不能恢复精度。

| 指标 | 历史非同口径 split-K | stride 修复版本 | 修复相对历史版 | BF16 baseline | 修复相对 baseline |
|---|---:|---:|---:|---:|---:|
| 固定256题 | 0/256 | 待运行 | 无法计算 | 105/256 | 无法计算 |
| 32K TTFT | 17,880.081 ms | 待精度通过后运行 | 无法计算 | 12,507.854 ms | 无法计算 |
| 32K TPOT | 199.606 ms | 待精度通过后运行 | 无法计算 | 151.236 ms | 无法计算 |

修复已发布为 source `ea8ae6b...48ac`，并完成 runtime、静态检查和正式 preflight。但之后的精度运行使用了与baseline不同的 top-k，不能闭合本项的正式精度或 TTFT/TPOT。下一轮必须在 prefill/decode K=2,048 下重跑；在固定256题达到至少105正确前，不能启动正式32K/batch1。

### 2.18 当前性能结论与后续优先级

当前可确认的结论是：

1. 稳定适配版本已完成长上下文、并发、苹果800 kernel 和精度验证，OSCAR 为107/256，BF16汇总为105/256；两轮协议指纹不同，只能判定总体精度处于同一水平。
2. prefill/decode top-k 从此冻结为2,048，与baseline完全一致；降低 top-k、统一 K 或 split-K 都不再属于可接受的优化方向。
3. 历史 split-K 相对 BF16 的 TTFT/TPOT 为`+42.951%/+31.983%`，但其 top-k 与baseline不同且存在stride回归，只可用于定位方向，不是“同样负载”的最终性能对比。
4. 当前还没有一份同时满足 top-k=2,048/2,048、固定256题至少105正确和32K/batch1的OSCAR正式TTFT/TPOT；因此完整目标尚未达成。
5. 已落盘trace表明历史TPOT差距不能由decode backend专属kernel单独解释；合法候选精度通过后，应优先检查跨rank上游负载和到达不均衡。
6. 下一步先将所有活动配置和fail-closed消费者回退到prefill/decode K=2,048，执行静态/correctness门禁后重跑固定256题；达到105/256后，才能在完全相同的32K/batch1负载下重测BF16与OSCAR。
7. 后续候选继续执行“correctness → 256题精度 → 32K/batch1端到端”的顺序，禁止用微基准收益代替可交付性能结果。
