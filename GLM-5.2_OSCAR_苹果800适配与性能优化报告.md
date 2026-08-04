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

以下所有 32K 表格统一使用 2.1 的最终 BF16 baseline。2.6–2.17 的早期候选与该 baseline 并非同源码，因此“相对 baseline”只用于统一观察差距；2.19 才是严格的同源码比较。

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

正式精度运行前已为`ea8-topk2048-fast256-v1`建立全新输出根
`/dev/shm/oscar-glm-ea8-topk2048-fast256-v1`。2026-08-04 10:08:19 CST与10:09:33 CST
两次GPU0–7空闲采样间隔74秒；两轮均为0 MiB、0%且compute进程为空。门禁同时确认主仓
`fd2ab808f07c7944e651b5ae0f5fac3f6916fa76`与source
`ea8ae6b7758ae2b4db7cae44d638ae5de80148ac`均clean/upstream，候选prefill/decode
top-k均为2,048，固定镜像身份、输出根0777权限、容器内root写探针和目标run目录未创建
均符合预期。最终validation为25/25、manifest为17/17，且`formal_run_started=false`；因此
这里只完成GPU门禁，没有启动模型或生成任何精度结果。证据位于
`artifacts/phase9-control/20260804T0207Z_ea8_topk2048_fast256_gpu_gate_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `gpu_idle_sample_first.txt` | 85 bytes | `c860eb49a4139aff7a8e685c7ad7a5331509ffcc2253caae8ebd28436c9ec281` |
| `gpu_idle_sample_second.txt` | 85 bytes | `1b10fcbd2e384f4dab68e2e4f9694ff1ff83727e51a409d7c76ca92b42dd108a` |
| `main_head.txt` | 41 bytes | `cd30c376d461360bbc02c1282db636548085beb7d74cda8358dd23bdda43a0b1` |
| `source_head.txt` | 41 bytes | `1a15b628cb10b4536f0526b15b3e925c03d3dcd404e0407e70f6612afadc9cb4` |
| `output_root_identity.txt` | 106 bytes | `a725b1df789db68bc6a0150bf885a5b4d657a88bf7b6f2801cdf5a27aae1b7a7` |
| `validation.json` | 5,605 bytes | `50fc3b8dc604a2d99fb4f5ad0432bcd1eaf6908e43f40fb3ecf45839f1009967` |
| `evidence_manifest.sha256` | 1,458 bytes | `e47ef03a104dd16df66b926b88a833faf26c9a0596a45fcd4d5f2471b3a4096e` |
| `manifest_check.txt` | 420 bytes | `e9c60e4c99a5973183473f42fb218f7489ab91e0d6e77732ffd49c5dd6846526` |

门禁报告由主仓`fada474641300026d81666ce26f421aa0a60c013`发布后，正式run于
2026-08-04 10:12:29 CST启动。外层即时GPU门禁通过，容器内又完成两轮8/8空闲检查；
实际服务参数为TP=8、max model len 8,192、max sequences 16、max batched tokens 2,048、
`hf_overrides={"index_topk":2048}`，runtime环境为prefill top-k 2,048、legacy decode与
prefill排序1。44/44 static preflight通过，EngineCore确认world size 8，8个TP worker均已
分配GPU并开始加载模型，日志没有Traceback、RuntimeError或OOM。

启动快照validation为33/33、manifest为17/17；该快照明确记录
`formal_run_started=true`和`accuracy_runner_started=false`，因此只证明K=2,048/2,048
身份下的服务启动与模型加载已经开始，不能提前报告精度。证据位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1/startup_snapshot`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `launch_state.txt` | 370 bytes | `2f9843627ed468446811ba9bb139c40db00efc58b57cd90573f971a5ca61aefe` |
| `parsed_server_args.json` | 575 bytes | `385a6defaf5347e38d33e46e5a2887f57569f982e7633a9514290a8ac3564401` |
| `runtime_environment.txt` | 3,585 bytes | `1f32450e36b9c460443cf460bd295ce7894e2345cdde3ea57ec441c9a1dd9ebe` |
| `static_preflight.json` | 12,549 bytes | `833ab9a56c60b3e4f4db5b035780ed28b4861eca0235af677a7f3aaae955eba7` |
| `startup_gpu.csv` | 94 bytes | `a075e012167eef1cb9d8d595ec09fa4e225998b9e7d85bc8400d533d5b9381f6` |
| `startup_compute.csv` | 592 bytes | `0e7bdf748bebb49e95e541c943bf6128cb0315372837b739077165382c4b1b45` |
| `validation.json` | 5,087 bytes | `0e96a5c1aafc0b5bf430a77953ee151accd71decf88b5bc3216b91e384504b33` |
| `evidence_manifest.sha256` | 1,464 bytes | `45bb4c7a7531acadf8fc23922bda59b169b9126ac142006da6c37bb02b26647c` |
| `manifest_check.txt` | 426 bytes | `fff0797eed53a915183727e66a048a41c6683fc444523046fbb5cdd445a45f43` |

启动后600秒固定截止为2026-08-04 10:22:29 CST：完成8/256、正确6题，当前完成集精度
75.000000%，全量精度2.343750%，0 invalid、0 truncated。模型服务约在10:19完成加载并
启动accuracy runner，因此该节点实际只包含约3分钟答题时间；样本过少，不能外推终局，也
不能据此判定是否达到105/256门槛。截止逐题复算与monitor一致，运行时仍为index/prefill
top-k 2,048/2,048，容器、tmux、c16 runner、EngineCore和8个TP worker均存活；23/23
validation与9/9 manifest通过。

诊断期间一次容器内`curl /health`返回connection refused，但同一时段服务日志持续记录16个
运行请求、生成吞吐和HTTP 200，EngineCore/TP worker/runner均存活，因此没有把该次curl
用于健康通过结论。宿主Python直接读取root-owned checkpoint也得到PermissionError；正式
截止复算改为在目标容器内只读文件，并由结构化validation核对。证据位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_10min.log` | 145 bytes | `3dde9042571d15445d529ed37d840c746a5456bc8aac8b8ca6535f955528bb8a` |
| `checkpoint_10min_cutoff_rows.json` | 19,133 bytes | `cf22f06133c9d107ea7a2d9444862085f1cc83db06b290fd757f91ab48204140` |
| `checkpoint_10min_gpu.csv` | 96 bytes | `4b70a618679b220bdd299edb7c9191326b122a4656c16177c97723d5468e8d8c` |
| `checkpoint_10min_compute.csv` | 600 bytes | `7773a96773fa0cdd60709d3596b416ac2420a0986ede2500748f84e5c084035c` |
| `checkpoint_10min_runtime_state.txt` | 3,946 bytes | `32d9ecfdc09243c2db7c2aec4ddf3f09b7f42b276f1c3a4a7fff035bf166a18d` |
| `checkpoint_10min_validation.json` | 3,960 bytes | `b85abb761ce4abf2d8fe562e29447b62f1af4b85b8a8e22a96f9a0cab3cd40cf` |
| `checkpoint_10min_manifest.sha256` | 860 bytes | `430bb9de19869401acc918a5604160248e69469f0888cb5ea102f77d065f5c3d` |
| `checkpoint_10min_manifest_check.txt` | 316 bytes | `206a4b0f6dfb76c1c83b6072c5f3b0d07df2f50870fd7aecccb211979abab6e7` |

启动后1,200秒固定截止为2026-08-04 10:32:29 CST：完成16/256、正确11题，当前完成集
精度68.750000%，全量精度4.296875%，0 invalid、0 truncated。该节点较10分钟节点新增8个
完成样本、其中5题正确；样本仍不足以外推终局，也不能据此判定是否达到105/256门槛。
截止逐题复算与monitor一致，运行时仍为index/prefill top-k 2,048/2,048，容器、tmux、
c16 runner、EngineCore和8个TP worker均存活，服务日志未出现Traceback、RuntimeError或
OOM；23/23 validation与9/9 manifest通过。证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_20min.log` | 148 bytes | `4263243abd02ff65bb8c61a76a3e4db12476a277fa127d16e48b98503687a52b` |
| `checkpoint_20min_cutoff_rows.json` | 45,728 bytes | `533ed41e7852f10c53721d6bf8cd84ff2280dd1de44bf3fccf5d89b2bb2b162b` |
| `checkpoint_20min_gpu.csv` | 107 bytes | `a75222bd932109c69ab400485ff4ec1fcbf1222b5e889622a35a5d4c5dd71fbe` |
| `checkpoint_20min_compute.csv` | 600 bytes | `1781d6b94f6648f75ea04b65723aa5210204e5881e8d67c01c79053a13d80574` |
| `checkpoint_20min_runtime_state.txt` | 3,946 bytes | `8a03956191172cf26f7e9fd6d7844ceda09c7b194f980d6682dde344397d9660` |
| `checkpoint_20min_validation.json` | 3,972 bytes | `86c651e3fa4a405c6d642c5489fca0f719238ebe56e247c2af723d73539ef812` |
| `checkpoint_20min_manifest.sha256` | 860 bytes | `a9a510a82cbb23576e63616cc918b0e2bef551ab7aea16926f000c88f3ed1efc` |
| `checkpoint_20min_manifest_check.txt` | 316 bytes | `d596dff63d2e511e13870d6e77871ab7b864a6bee70830fb1fb9aeadb44a11ee` |

启动后1,800秒固定截止为2026-08-04 10:42:29 CST：仍为16/256完成、11题正确，当前
完成集精度68.750000%，全量精度4.296875%，0 invalid、0 truncated；相对20分钟节点没有
新增完成题。该节点8卡利用率均为98%，容器、tmux、c16 runner、EngineCore和8个TP worker
均存活，服务日志无Traceback、RuntimeError或OOM，因此判断为长输出批次仍在生成，不是
实验停滞。截止逐题复算与monitor一致，23/23 validation与9/9 manifest通过。证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_30min.log` | 148 bytes | `0898e83857aaa2419417139e8d5da1b7565cadf4f4760e3b6804cf8d7beafdb2` |
| `checkpoint_30min_cutoff_rows.json` | 45,728 bytes | `533ed41e7852f10c53721d6bf8cd84ff2280dd1de44bf3fccf5d89b2bb2b162b` |
| `checkpoint_30min_gpu.csv` | 107 bytes | `f8073fdcc5675e2bb3ccc61a4a176dd8465285cf3ed6f87e2ff1ee153ec8eeb8` |
| `checkpoint_30min_compute.csv` | 600 bytes | `5670309d5b0be6a20182f83b3e4fc4aa48afa4d43bd6f504f1c922d886053a6e` |
| `checkpoint_30min_runtime_state.txt` | 3,946 bytes | `6f09f5ea274cf8be6a374e5a99408beefdfc1e8bc3ff341f5fe5fbb30c378077` |
| `checkpoint_30min_validation.json` | 3,972 bytes | `206c122e16a4d8de6b58b921671594ccbbc4577c9ad0eef7068236d1a373a86d` |
| `checkpoint_30min_manifest.sha256` | 860 bytes | `d93d0f31d0e5e5064e64dbf060ca75337b406ec3ae23bdcfb61ba8b6e5a38224` |
| `checkpoint_30min_manifest_check.txt` | 316 bytes | `bc4a26814ce8ebe5134f44a2b80b9f47fcd7fb0ebb2bdbdf419581fc61e24987` |

启动后2,400秒固定截止为2026-08-04 10:52:29 CST：仍为16/256完成、11题正确，当前
完成集精度68.750000%，全量精度4.296875%，0 invalid、0 truncated；连续两个固定节点
没有新增完成题。节点前10:49现场采样8卡利用率为97%–100%，容器、tmux、c16 runner、
EngineCore和8个TP worker均存活，服务日志无Traceback、RuntimeError或OOM，因此仍判定为
长输出批次在生成，不重启。截止逐题复算与monitor一致，23/23 validation与9/9 manifest
通过。证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_40min.log` | 148 bytes | `a4a43cb766a71cb7f2618384bbdd28a8303c026580a6498d58d737532c1778ac` |
| `checkpoint_40min_cutoff_rows.json` | 45,728 bytes | `533ed41e7852f10c53721d6bf8cd84ff2280dd1de44bf3fccf5d89b2bb2b162b` |
| `checkpoint_40min_gpu.csv` | 105 bytes | `4dc24ca40be55090d91352469a7f9e95cf3d17ca3caa1625b9de20df95b801ce` |
| `checkpoint_40min_compute.csv` | 600 bytes | `bbbb4e3399d501ee9e94f3a6de94aa27d95fb1136d4ec2e768319f4a10c89936` |
| `checkpoint_40min_runtime_state.txt` | 3,946 bytes | `78d4b1eef63af31bdf5b7649ea305d2f6609c189b8e1cd40256abfc90f91baaa` |
| `checkpoint_40min_validation.json` | 3,972 bytes | `ede86c4648eb92de805abec92bd410094440a47a592910b10689ee37edf5d5ed` |
| `checkpoint_40min_manifest.sha256` | 860 bytes | `9379ce3a1be4e3566f9d44bd4f33dcb44123df047aabbc74d81b6c3451f7ba7d` |
| `checkpoint_40min_manifest_check.txt` | 316 bytes | `31c28600b3179cbda90d2fef468b46573879ee2b3830b7e2a938c176d45a9968` |

启动后3,000秒固定截止为2026-08-04 11:02:29 CST：完成34/256、正确17题，当前
完成集精度50.000000%，全量精度6.640625%，0 invalid、13 truncated。较40分钟
节点新增18个完成样本、其中6题正确；该中间值仍不能外推终局，也不能据此
判定是否达到105/256门槛。截止逐题复算与monitor一致；现场8卡利用率为
87%–98%，容器、tmux、c16 runner、EngineCore和8个TP worker均存活，服务无
Traceback、RuntimeError或OOM，实验继续运行。

结构化证据生成器首轮为22/23：唯一失败是沿用10分钟合同仍期望0 truncated，
而50分钟截止数据实际为13。保留首轮validation后，只将该节点的期望值修正
为13，最终23/23 validation与9/9 manifest通过。证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_50min.log` | 149 bytes | `f4ffa5117efee64f863cb0871235e50fc085d495b23396ac48e0dd9e946363c4` |
| `checkpoint_50min_cutoff_rows.json` | 373,179 bytes | `a07766d3818269a7119e95710dd8f15d832e26ee65f7499b431d96d9f1a8f5b3` |
| `checkpoint_50min_gpu.csv` | 96 bytes | `6c5b15d67343024fe0c17b25c8534847fc53a69f1a60090cba98ac734efd638d` |
| `checkpoint_50min_compute.csv` | 600 bytes | `be964dfabeffe666be1d2337020fb858b127973d54ecd179c1b82fb698d80856` |
| `checkpoint_50min_runtime_state.txt` | 3,946 bytes | `506fbae2acd49bf32a6b8529a6774563bae412774ae2dbfde0eee9d32cb47a8e` |
| `checkpoint_50min_validation_attempt1.json` | 3,975 bytes | `23c7c6d1d50c99ebec0e0d3dd36150059a33c9f0bbbc43698b764a25196f564a` |
| `checkpoint_50min_validation.json` | 3,976 bytes | `85db10f2e58c67a34dee6af7f80ecb655248512b72d1853d7249bb81b9579899` |
| `checkpoint_50min_manifest.sha256` | 860 bytes | `1405106522187d54c6773f81c7b0be981d1113921106944aed314e8efa1d8dc7` |
| `checkpoint_50min_manifest_check.txt` | 316 bytes | `b17dfe5896085d66cc1568ae8c50e8566fc55f245b74d6d780dfbeca079bd0aa` |

启动后3,600秒固定截止为2026-08-04 11:12:29 CST：完成38/256、正确18题，当前
完成集精度47.368421%，全量精度7.031250%，0 invalid、16 truncated。较50分钟
节点新增4个完成样本、其中1题正确；该中间值仍不能外推终局，也不能据此
判定是否达到105/256门槛。截止逐题复算与monitor一致；现场8卡利用率为
97%–100%，容器、tmux、c16 runner、EngineCore和8个TP worker均存活，服务无
Traceback、RuntimeError或OOM，实验继续运行。23/23 validation与9/9 manifest
一次通过。证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_60min.log` | 149 bytes | `6e04a638d9e311d087484fe83258bb3ae29042f6fe42fcc923c609e8c3eb5d7c` |
| `checkpoint_60min_cutoff_rows.json` | 456,868 bytes | `46b31a8496b17854c69bf0a3333be6b7ab706ec0259a41f7850c637fb32472f3` |
| `checkpoint_60min_gpu.csv` | 105 bytes | `6affaa83fe9996c531ec16e2edca19738845c94d7e43376bf44c99e722752e3f` |
| `checkpoint_60min_compute.csv` | 600 bytes | `be964dfabeffe666be1d2337020fb858b127973d54ecd179c1b82fb698d80856` |
| `checkpoint_60min_runtime_state.txt` | 3,949 bytes | `e8a98c327c37655f24c7a21f5f6f32399770b88861ce24615b4e75ab6bb23f7e` |
| `checkpoint_60min_validation.json` | 3,976 bytes | `84e3676b36e47dac495e41f696f935027f1b69543e8eddf5a99a2931eb8b30ce` |
| `checkpoint_60min_manifest.sha256` | 860 bytes | `b0ed6830baa181f135414b06b3daf7b34f62ff8d325c4ea4d4cdd342285aafc4` |
| `checkpoint_60min_manifest_check.txt` | 316 bytes | `1d0002a33928343079e0c23c5ecc915cdcc46c23d20d747ce4a2891cc421309c` |

启动后4,200秒固定截止为2026-08-04 11:22:29 CST：仍为38/256完成、正确18题，当前
完成集精度47.368421%，全量精度7.031250%，0 invalid、16 truncated；相对60分钟节点
没有新增完成题。截止逐题复算与monitor一致；现场8卡利用率为98%–99%，容器、tmux、
c16 runner、EngineCore和8个TP worker均存活，服务无Traceback、RuntimeError或OOM，
因此仍判定为长输出批次在生成，不重启。23/23 validation与9/9 manifest一次通过。
证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_70min.log` | 149 bytes | `f5a9c04cf2a01d7c703497e39bbf5cdf671a07cea871ae9915f42b6f457c8f85` |
| `checkpoint_70min_cutoff_rows.json` | 456,868 bytes | `46b31a8496b17854c69bf0a3333be6b7ab706ec0259a41f7850c637fb32472f3` |
| `checkpoint_70min_gpu.csv` | 104 bytes | `eb3db1b53b90b4a4d91aec315689d6fd875246ddbdd0f6d5c5d6b1754782ebf5` |
| `checkpoint_70min_compute.csv` | 600 bytes | `be964dfabeffe666be1d2337020fb858b127973d54ecd179c1b82fb698d80856` |
| `checkpoint_70min_runtime_state.txt` | 3,949 bytes | `5fa34494c1516d6e0904fd4698124c23b60f150576f1581a9c319ace74114a1a` |
| `checkpoint_70min_validation.json` | 3,976 bytes | `812f1d12752eb3b44f27477f19effb099a28a163ea07ae3772069cd4370b789e` |
| `checkpoint_70min_manifest.sha256` | 860 bytes | `88708512ec71a0bd5fd9ea79a4b840d4268efe7370d7cb744745d99feb66e08f` |
| `checkpoint_70min_manifest_check.txt` | 316 bytes | `f9f7badd24a34df31a0f10bf345b8644186bde235ff04295c7beb4ef9407e1cc` |

启动后4,800秒固定截止为2026-08-04 11:32:29 CST：完成44/256、正确20题，当前
完成集精度45.454545%，全量精度7.812500%，0 invalid、21 truncated。较70分钟节点
新增6个完成样本、其中2题正确；中间值仍不能判断终局是否达到105/256门槛。截止逐题
复算与monitor一致，运行进程保持存活；23/23 validation与9/9 manifest一次通过。
证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_80min.log` | 149 bytes | `966a6ed36babe1101addc59bfc2aa1d6dc30bc74187e07941ca1ca50cebd9ae6` |
| `checkpoint_80min_cutoff_rows.json` | 602,988 bytes | `00db053b157716c33c0a77472a13615014bd67b89ffe1b66e54360d43b9d9c9a` |
| `checkpoint_80min_gpu.csv` | 104 bytes | `18f577a6a68c4440605c0f48d3d847523c9122175fb7f366146ac1bf7bcc190e` |
| `checkpoint_80min_compute.csv` | 600 bytes | `be964dfabeffe666be1d2337020fb858b127973d54ecd179c1b82fb698d80856` |
| `checkpoint_80min_runtime_state.txt` | 3,949 bytes | `9d3975b59574cef9f02f253690881f827f287e8125426e2f7cec9c2c26d4f343` |
| `checkpoint_80min_validation.json` | 3,976 bytes | `98a58705cf15dbac4c7aab5534c96cefae908ef5b7f7696d78955271012617e9` |
| `checkpoint_80min_manifest.sha256` | 860 bytes | `c58dc072085cfc6be89f3bc214186ab2fafc8b2f385dc3325c8e4e36c89c216d` |
| `checkpoint_80min_manifest_check.txt` | 316 bytes | `7d32178295efa8fdc421e562572b176d0efe1ba0f6f5d115fd7890e217a9d276` |

启动后5,400秒固定截止为2026-08-04 11:42:29 CST：完成71/256、正确32题，当前
完成集精度45.070423%，全量精度12.500000%，0 invalid、32 truncated。较80分钟节点
新增27个完成样本、其中12题正确；中间值仍不能判断终局是否达到105/256门槛。截止逐题
复算与monitor一致，运行进程保持存活；23/23 validation与9/9 manifest一次通过。
证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_90min.log` | 150 bytes | `13284ba8bcf650272b79f462b1081754a324ba8df180d83494d44ea0e829bc9f` |
| `checkpoint_90min_cutoff_rows.json` | 892,431 bytes | `99661d84fd39a422cc2b4e94a94359b41360a0e03e63ee7f8d0d78de326f33e9` |
| `checkpoint_90min_gpu.csv` | 104 bytes | `3d3a671f46ba264cd944f3bcd638cdc64c36b228c12ccc787f5a066f6432da28` |
| `checkpoint_90min_compute.csv` | 600 bytes | `331d77ae5a947a49855120af5d07dc05075d4847d5a5481f17dc8cc6a74109b3` |
| `checkpoint_90min_runtime_state.txt` | 3,943 bytes | `661ae440031b345a0b9632cc7715403a02960a3cb42e7a2030e67fbe1770982a` |
| `checkpoint_90min_validation.json` | 3,978 bytes | `597d77a0a3a28c8ca79a9a42db191b0ff3fcf5167fc46784da453d852a034b98` |
| `checkpoint_90min_manifest.sha256` | 860 bytes | `f147df004c513738eac04a8cea987226f2180773aa1a7a1af8ee3b8bb2c81111` |
| `checkpoint_90min_manifest_check.txt` | 316 bytes | `9c37222c3b4cc1823e2cb3f6980537b4233f3b8fdb82393ca99cb1641c3a2a1d` |

启动后6,000秒固定截止为2026-08-04 11:52:29 CST：仍为71/256完成、正确32题，当前
完成集精度45.070423%，全量精度12.500000%，0 invalid、32 truncated；相对90分钟节点
没有新增完成题。截止逐题复算与monitor一致，运行进程保持存活，因此继续判定为长输出
批次执行中，不重启。23/23 validation与9/9 manifest一次通过。证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_100min.log` | 150 bytes | `08865ca5dc5a5d40d812e0372e389d39567961265acb71425808e271d2affb53` |
| `checkpoint_100min_cutoff_rows.json` | 892,431 bytes | `99661d84fd39a422cc2b4e94a94359b41360a0e03e63ee7f8d0d78de326f33e9` |
| `checkpoint_100min_gpu.csv` | 105 bytes | `8e2cb7f413483815d90a998b9a3d5472fea0afd0bf571d96ae3e3410efbc4c4f` |
| `checkpoint_100min_compute.csv` | 600 bytes | `331d77ae5a947a49855120af5d07dc05075d4847d5a5481f17dc8cc6a74109b3` |
| `checkpoint_100min_runtime_state.txt` | 3,943 bytes | `67a7b60b6d3b74bbff1a60a1814c6dcdfbefa1c086115896ae4690b625f0885a` |
| `checkpoint_100min_validation.json` | 3,979 bytes | `7f99790f43151fcca464c136e598ec3c6a594006a9fef5c9d38375a51e271d1d` |
| `checkpoint_100min_manifest.sha256` | 869 bytes | `7bfb582f47bace92bc12dcc029509a23bbd97f0b3cfb906d4d55e199723043e1` |
| `checkpoint_100min_manifest_check.txt` | 325 bytes | `039c69b1e493d6b2f79b8ab15bc735e4ed6d6cc0ec1787a6343064bc62daf359` |

启动后6,600秒固定截止为2026-08-04 12:02:29 CST：仍为71/256完成、正确32题，当前
完成集精度45.070423%，全量精度12.500000%，0 invalid、32 truncated；连续两个固定节点
没有新增完成题。截止逐题复算与monitor一致；现场8卡利用率为98%–100%，容器、tmux、
c16 runner、EngineCore和8个TP worker均存活，继续判定为长输出批次执行中，不重启。
23/23 validation与9/9 manifest一次通过。证据仍位于
`artifacts/phase9-control/20260804T0212Z_ea8_topk2048_fast256_launch_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `checkpoint_110min.log` | 150 bytes | `676b020cf0d86de8fad5b524b25c20fc6f1d5e8b4a7b15648b18bfe7d16620d9` |
| `checkpoint_110min_cutoff_rows.json` | 892,431 bytes | `99661d84fd39a422cc2b4e94a94359b41360a0e03e63ee7f8d0d78de326f33e9` |
| `checkpoint_110min_gpu.csv` | 105 bytes | `92849c6c06e9f3880edbbe5970f90375d15336001640b8cc862c27d2b0fedb7f` |
| `checkpoint_110min_compute.csv` | 600 bytes | `331d77ae5a947a49855120af5d07dc05075d4847d5a5481f17dc8cc6a74109b3` |
| `checkpoint_110min_runtime_state.txt` | 3,943 bytes | `a2f14fca609860adac766e235d426fce11058f461be820b39c419e5b590860e5` |
| `checkpoint_110min_validation.json` | 3,979 bytes | `bfd529990070d678b05da4ec45ad60b626fc48a4d4f24cfed37b929ca3aae5f7` |
| `checkpoint_110min_manifest.sha256` | 869 bytes | `99086aaf8a0497bcab6fdd21c36887a7500b7f5311c2468e5fa6ff13456aee8c` |
| `checkpoint_110min_manifest_check.txt` | 325 bytes | `741e79ead8a02c29cfac7f818a59104f844f2f0aeca84200fa2a72ff8caa1915` |

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

### 2.18 隐藏层特征相似度快速筛选（代理门禁，待标定）

固定256题耗时较长，因此新增CPU-only离线比较器
`scripts/phase9/compare_hidden_captures.py`，用于比较BF16与OSCAR在相同token轨迹上的
隐藏层特征。它按hook、layer、capture counter、TP rank和PP rank建立语义identity；
aux-runner没有显式`layer_idx`时从hook严格派生。两侧shape、dtype和positions必须完全一致，
identity缺失、重复、集合不一致、非有限值或零范数都直接失败，避免把错请求或错token配成
一对。输出包括token级cosine、relative L2、最大绝对误差、MSE以及输入文件SHA256。

该工具没有相似度阈值参数，输出固定标记
`classification=hidden_state_similarity_proxy_unthresholded`和`promotion_gate=false`。
原因是当前报告已有的rotation/dequant、output/LSE最大误差属于kernel数值oracle，不是已证明
与GSM8K准确率相关的隐藏层代理；TurboQuant单token round-trip的0.95/0.85 cosine阈值也不能
外推到端到端任务。自由生成在首个不同token后会失去位置语义，因此正式标定必须使用相同
prompt、相同BF16 teacher-forced token轨迹、单请求固定顺序，并至少覆盖进入INT2 history后的
位置。标定完成前，本工具只允许快速淘汰；通过后仍需32/64题任务级复筛和最终256题晋升门禁。

TDD首先在production文件不存在时得到预期FileNotFoundError红灯；实现基础配对后，又用
aux-runner真实payload缺少`layer_idx`的合同得到预期ValueError红灯，再加入hook派生。最终
目标测试5/5、Phase 9递归95/95、ruff 0.14.0全部通过。合成`.pt` CLI smoke覆盖1对、2个
token，得到cosine mean `0.9850712500726659`、relative L2 mean
`0.14142135307629597`，只证明计算与落盘链有效，不代表任何模型精度阈值。

结构化validation首轮24/25：唯一失败是验证器用通用`threshold`子串误伤合法分类名
`unthresholded`；保留失败证据后改为精确禁止`--threshold` CLI参数，最终25/25、manifest
15/15通过。证据位于
`artifacts/phase9-control/20260804T0240Z_hidden_similarity_proxy_tdd_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `scripts/phase9/compare_hidden_captures.py` | 10,069 bytes | `8920435f5f3c69d72263e74f45c747f5f5193cc14beb774bfe10e7e8a18108a5` |
| `scripts/phase9/test_compare_hidden_captures.py` | 3,844 bytes | `f9de6916443c7e86f95f6de9ac5c7282b8b066d9a9e177d8bad68a086489bb00` |
| `red_missing_production.json` | 297 bytes | `3c18aee5483933882b4ef906066a786805f947b74496d697100106be84d0dad5` |
| `red_aux_layer_derivation.json` | 305 bytes | `4118a0114493a7728a84578f12db5eab7f8f0995c9c5f262f1d88254abce3055` |
| `validation_attempt1.json` | 3,207 bytes | `c4ebcd4f573919d2adc992567ad0f0fab57515351c64c2163552f8d20f7f530e` |
| `target_test.log` | 147 bytes | `ad035c769a576336901d65775660f12a33aa36d00e34c7a56f2a658e11d6a849` |
| `phase9_recursive_test.log` | 3,720 bytes | `9936ed07485c534ef8af9892829b17ed075346e75e83a90e57472fddb62243b3` |
| `ruff.log` | 63 bytes | `a1951a5d0687eea67e0e7dfb4ea86c86811c8bb134661f15731d2042bdf2bfdc` |
| `smoke/summary.json` | 1,836 bytes | `3a790e600d9149e4c07347e345563ceec0ab37500d421ecfaa18824b98ea0413` |
| `validation.json` | 3,210 bytes | `2254dcd65f9dc4e505212688e90d4984896d1a89be50717a8ee07071cce9fa0a` |
| `evidence_manifest.sha256` | 1,323 bytes | `b1d1b7581975d9c329eb22a4ab433fefa059ed57d62e3e007f0d54eb6355bc7b` |
| `manifest_check.txt` | 409 bytes | `91831d4424a8616891a349a92dfd29d4a13ee8c90c01c0ebee72fc7e694166aa` |

OSCAR三段式缓存的prefix为64 token、recent为256 token，只有position达到320后才开始
受到INT2 history影响。若对完整序列直接求均值，前320个仍由BF16 prefix/recent保存的高
相似度可能掩盖history误差。因此比较器新增非负`--min-position`参数；正式代理采样将使用
`--min-position 320`，只聚合position大于等于320的隐藏层特征，样本没有合格位置时直接
失败。输出同时记录`min_position`，但仍固定`promotion_gate=false`，不引入经验阈值。

该变更先新增两个合同测试：一项用position 319的大误差确认它被过滤，另一项确认全部位置
小于320时fail-closed。production未实现前7项测试中原有5项通过、新增2项均以
`TypeError: unexpected keyword argument 'min_position'`失败；最小实现后目标测试7/7、
Phase 9递归97/97、ruff 0.14.0通过，结构化validation 24/24、manifest 8/8通过。该结果
只闭合“如何计算代理”的实现合同，尚未产生BF16/OSCAR真实模型相似度，也未标定其与256题
准确率的相关性。证据位于
`artifacts/phase9-control/20260804T0320Z_hidden_similarity_min_position_tdd_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `red_min_position.json` | 785 bytes | `e257a8cd1ed3a1ffc96117215bf62d40f70c7a113482b226fa5ff03c96b51885` |
| `target_test.log` | 149 bytes | `58fbc417cb8536f9df2ff5f34961decf4bf54b357b5209c1faa8af3f165d4dd0` |
| `phase9_recursive_test.log` | 3,722 bytes | `9ad962d2f2f8e658f702ab790392f1869118c1fe499c17370e45a6559a1e38bd` |
| `ruff.log` | 63 bytes | `a1951a5d0687eea67e0e7dfb4ea86c86811c8bb134661f15731d2042bdf2bfdc` |
| `source_identity.json` | 359 bytes | `fcccd93ff34a1de383f38fe36f646c90a6b772837802ee3ce2aaa1ba91b47d37` |
| `validation.json` | 3,098 bytes | `dcbe8886eda58c6a4d3b8edc33bc11a5cb54ee222c7acf22ee8a55936f7862d2` |
| `evidence_manifest.sha256` | 671 bytes | `b36a9dbdb4d4138ec5f5568a1d8b9386a88d007fc091d45b8798f845f8d5fbbc` |
| `manifest_check.txt` | 189 bytes | `d862aebf9093509a6656d63415698b54eabf82dfb06ffdedb9c3df207a13a7a8` |

对固定256子集进一步执行离线长度筛选：使用模型自身`chat_template.jinja`，把每题user
prompt与assistant标准推理答案按`reasoning_effort=high`、`enable_thinking=true`渲染，
再用同一`tokenizer.json`编码。实测仅9/256达到320 token，长度依次为418、389、385、
385、376、358、353、341、325；对应样本ID为`gsm8k:001086`、`gsm8k:000882`、
`gsm8k:000144`、`gsm8k:001209`、`gsm8k:001029`、`gsm8k:000831`、
`gsm8k:001199`、`gsm8k:001122`、`gsm8k:000690`。因此第一版真实模型代理固定使用这9条
长样本，不能用大量短题稀释INT2 history误差。

离线选择9/9 validation、5/5 manifest通过，但当前control镜像的transformers 4.57.6
不支持模型声明的v5 `TokenizersBackend`，本轮使用Jinja模板与tokenizers 0.22.2直接编码。
正式capture前必须逐条与服务端`/tokenize`返回的token IDs复核；因此选择产物明确标记
`capture_ready=false`、`server_token_ids_validation_pending=true`和
`promotion_gate=false`，不能把本阶段写成真实模型相似度结果。证据位于
`artifacts/phase9-control/20260804T0335Z_hidden_similarity_replay_selection_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `selector.log` | 29 bytes | `80e92a5d1ea0998087d2dd581e8a20c7e7d072fef2da63c59141d1a9ec7f9bfb` |
| `selection.json` | 47,711 bytes | `1ce6446de5dec5486ee23372397b81f19d26fe446629cc20e00a48d6532d0a47` |
| `validation.json` | 1,987 bytes | `8d557161616c25832e89a4b6bcecf84e7432790a719c65d583eb485f62e3f8c5` |
| `evidence_manifest.sha256` | 410 bytes | `f080844e1905f0f25cc192fc576a750a488b358efccf4d6884e201b5985e6866` |
| `manifest_check.txt` | 114 bytes | `aa5cd2e710ea0aebdc52edd1b9b0b00bb8255dd022a56f52f7c2f431bb20d07b` |

随后在当前OSCAR服务的隔离网络命名空间内调用CPU `/tokenize`接口，对上述9条样本逐条
复核。服务端返回的token数仍为418、389、385、385、376、358、353、341、325，9/9
完整token ID序列及逐样本SHA256均与离线选择产物完全一致，9/9均报告
`max_model_len=8192`；6/6结构化检查和2/2 manifest通过。该操作没有调用生成接口，未向
GPU提交新推理请求，也未改变正在运行的fixed256实验。

因此回放输入的token身份门禁现已闭合，验证产物标记`capture_ready=true`；下一步可以在
当前实验释放GPU后，使用完全相同的9组token IDs顺序采集BF16和OSCAR隐藏层。但
`promotion_gate=false`仍保持不变，因为真实hidden capture、相似度结果以及与256题准确率的
相关性尚未产生。证据位于
`artifacts/phase9-control/20260804T0338Z_hidden_similarity_server_tokens_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `validate_server_tokens.py` | 6,411 bytes | `0b0b9dc985d6bbccee570c6c0b8c05f47c4060470f93eef4107bedbd6370edd6` |
| `validation.json` | 3,945 bytes | `0789900c7c88f1bef9cb65a16d9a31a1977e4565a5841ca904eeb5b525122911` |
| `evidence_manifest.sha256` | 174 bytes | `222f008e01906b0645a2b40333f757ea9a1693acc6e7a045402de74bd0f3cdb1` |
| `manifest_check.txt` | 64 bytes | `0ad9b995071be0851f49ee6136c9b8dfa68f8b60a14c46f2fce0ee9ef59de2a0` |

在token身份闭合后，新增正式回放runner
`scripts/phase9/run_hidden_similarity_replay.py`。runner只接受上述9条且必须同时满足服务端
validation的`capture_ready=true`、选择文件SHA一致、逐样本token数量/hash完全一致；随后按
固定顺序逐条向本机隔离loopback completion接口发送token IDs，冻结`max_tokens=1`、
`temperature=0.0`、`seed=42`。capture环境冻结为aux-runner、layer 36
post-attention-layernorm、TP rank 0；完成后必须恰好得到counter 0–8九个文件，并逐项验证
positions为`0..token_count-1`。数量、counter、hook、rank或positions任一不一致都直接失败。

TDD先在runner不存在时得到预期FileNotFoundError红灯；最小实现后目标测试4/4、Phase 9
递归101/101、ruff 0.14.0通过，结构化validation 21/21、manifest 8/8通过。该阶段只验证
CPU合同和fail-closed行为，`gpu_capture_started=false`、`promotion_gate=false`；尚未启动
BF16或OSCAR正式capture。证据位于
`artifacts/phase9-control/20260804T0350Z_hidden_similarity_replay_runner_tdd_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `scripts/phase9/run_hidden_similarity_replay.py` | 12,207 bytes | `bf262e2eaf0ad1e2076b73b6c0d0a003c625c7d47dc3841f9f0963a02bc69111` |
| `scripts/phase9/test_run_hidden_similarity_replay.py` | 4,570 bytes | `8d8307c049f87585625377a1750b02222b7b26ba8107276ab10dbedd76ce4d1e` |
| `red_missing_runner.json` | 378 bytes | `09b0ad133bcb8dbeacb4d4afdf04360383c7f24a26fd75f528901e6905ff3601` |
| `target_test.log` | 146 bytes | `fe985cd1c9ca69679969098926f1b81bc25a47ddb714d3da663ef2a7beaea556` |
| `phase9_recursive_test.log` | 3,727 bytes | `f3559d0f54b4ea1ddda35cd5a2454b496699f5e4054e79098bd38d6e7ede637c` |
| `ruff.log` | 63 bytes | `a1951a5d0687eea67e0e7dfb4ea86c86811c8bb134661f15731d2042bdf2bfdc` |
| `validation.json` | 2,663 bytes | `b90c2f8b9e7865763b84212835bf4eb15793ce271032ead14e9c8230c544780a` |
| `evidence_manifest.sha256` | 673 bytes | `67f2b2106e8d6ceaeb850146bcf0ea84f1fad0bf2cd758d2d8d0f31cd52c6130` |
| `manifest_check.txt` | 191 bytes | `2ef941fd2d2897cd3c85de06ec43c60c426ed40c2a3ad12317ee74dffdcb6b04` |

为使BF16与OSCAR正式capture沿用同源码、同模型、同top-k和8K协议，进一步在Stage 9
容器编排中新增`hidden-capture-baseline`与`hidden-capture-candidate`入口。两个入口均冻结
index/prefill top-k为2,048、`MAX_MODEL_LEN=8192`、layer 36、TP rank 0，并在
`--network none`容器内启动各自既有server wrapper；服务ready后才调用上述9样本runner，
退出时检查8卡全部释放。原32K/128K wrapper的默认`MAX_MODEL_LEN=131072`保持不变，只有
hidden capture显式覆盖为8,192。

TDD先新增编排合同，目标suite 26项中仅新测试因缺`hidden-capture-baseline`得到1个
AssertionError红灯；最小接线后目标26/26、Phase 9递归102/102、bash syntax与ruff均通过，
结构化validation 25/25、manifest 9/9通过。正式入口仍强制主/source仓clean且已发布，
当前HTTPS认证未恢复且fixed256占用GPU，因此`formal_capture_started=false`，没有越过发布或
空闲门禁。证据位于
`artifacts/phase9-control/20260804T0355Z_hidden_similarity_capture_orchestration_tdd_v1`：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `scripts/phase9/run_containerized_performance.sh` | 19,794 bytes | `ad091fed8ed278f5fe16dc1644fa5bb0efa31de18d483cef98c9fe845b7a1805` |
| `scripts/phase9/run_native_tp8.sh` | 3,037 bytes | `26e91a1e77b98d60c279227cc2195e6b08fab4e94fb0d7d1f1735d7379a33415` |
| `scripts/phase9/run_candidate_tp8.sh` | 4,879 bytes | `a72ca682c47b8de5ebe00c6afb266ef78069fa5fd49451f5a564bd5369f3da4a` |
| `scripts/phase9/test_phase9_tools.py` | 33,366 bytes | `c77496b9353b09c036059d35ec06ad3923fafdda620ad0975cdf4ee62f3ba086` |
| `red_missing_orchestration.json` | 312 bytes | `559c8e22dee8597668c5e60da2a7ddd51cece0ed46fb6408fbd303a371d786d3` |
| `target_test.log` | 169 bytes | `8bbb59c43b91928c9fbb7a0e9602503ff1e7434dacc09e5a59770d3d27e11c18` |
| `phase9_recursive_test.log` | 3,728 bytes | `e0752ec47dac663b289db02cc7775b9259db3e202c6587ee7d7ad13e7e237069` |
| `bash_syntax.log` | 44 bytes | `c8cd67031ad49938245661967d5f2c3bc5b8b68f1d5eeb8970ead76a42e52a36` |
| `ruff.log` | 63 bytes | `a1951a5d0687eea67e0e7dfb4ea86c86811c8bb134661f15731d2042bdf2bfdc` |
| `validation.json` | 3,328 bytes | `4af3d37ddc614abf0d84eb101a1ed4dbd021b62be1128debc72e0df3adc97e9e` |
| `evidence_manifest.sha256` | 762 bytes | `989b652102d8a63e1bd416fef4e735c9df3c5fa61752f945b588c41dde926d4f` |
| `manifest_check.txt` | 218 bytes | `189079c56ec7c782d619805c778f5fc63b754519e5232663eab01a91f037088c` |

### 2.19 当前性能结论与后续优先级

当前可确认的结论是：

1. 稳定适配版本已完成长上下文、并发、苹果800 kernel 和精度验证，OSCAR 为107/256，BF16汇总为105/256；两轮协议指纹不同，只能判定总体精度处于同一水平。
2. prefill/decode top-k 从此冻结为2,048，与baseline完全一致；降低 top-k、统一 K 或 split-K 都不再属于可接受的优化方向。
3. 历史 split-K 相对 BF16 的 TTFT/TPOT 为`+42.951%/+31.983%`，但其 top-k 与baseline不同且存在stride回归，只可用于定位方向，不是“同样负载”的最终性能对比。
4. 当前还没有一份同时满足 top-k=2,048/2,048、固定256题至少105正确和32K/batch1的OSCAR正式TTFT/TPOT；因此完整目标尚未达成。
5. 已落盘trace表明历史TPOT差距不能由decode backend专属kernel单独解释；合法候选精度通过后，应优先检查跨rank上游负载和到达不均衡。
6. 活动配置与fail-closed消费者已回退到prefill/decode K=2,048，CPU合同和GPU双空闲门禁均已通过；下一步运行固定256题，达到105/256后，才能在完全相同的32K/batch1负载下重测BF16与OSCAR。
7. 后续候选继续执行“correctness → 256题精度 → 32K/batch1端到端”的顺序，禁止用微基准收益代替可交付性能结果。
8. 隐藏层代理比较器已经完成CPU门禁，并可只统计进入INT2 history后的position；但尚无同top-k、同协议的真实模型相似度及相关性阈值，当前只能用于收集标定数据，不能停止或替代正在运行的256题正式门禁。
