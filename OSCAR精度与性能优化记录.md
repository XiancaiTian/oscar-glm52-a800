# OSCAR 精度与性能优化记录

> 状态截点：2026-07-31
> 本文只记录已经落地的代码改动、测试结果和实验数据。详细实验身份与证据哈希见
> [GLM-5.2_OSCAR_A800适配报告.md](./GLM-5.2_OSCAR_A800适配报告.md)。

## 1. 精度优化改动

### 1.1 为 MLA 改造共享 rotation

原始 OSCAR 面向 full-attention，可分别处理 K、V；GLM-5.2 MLA 只缓存一份共享
`compressed_kv`。项目因此改为每层一份 `512×512` 正交 rotation：

- history latent 使用 `c_rot = c·R`；
- history query 使用 `q_rot = q_abs·R`；
- history value accumulator 最后乘 `Rᵀ`，返回原始 latent 空间；
- score、value 和共享 latent covariance 共同参与 rotation 拟合。

该设计保证未量化情况下 rotation 只是正交换基，不改变 MLA score 和 value；
INT2 clipping/量化才是近似误差来源。

### 1.2 使用独立 train/holdout 数据选择 rotation 与 clipping

为避免用正式精度测试集调参，calibration 与 accuracy suite 完全分离。实际采集：

| Split | 请求完成情况 | Prompt tokens | 文件数 |
|---|---:|---:|---:|
| Train | 256/256 HTTP 200 | 900,000 | 624 |
| Holdout | 36/36 HTTP 200 | 100,000 | 624 |

项目实现共享 covariance 合并、正交 basis、Hadamard/PBR 组合、alpha 搜索和逐层
clip 搜索。三个 alpha 的 holdout loss 为：

| Alpha | 归一化 holdout loss |
|---:|---:|
| 0.25 | **0.026186010882512642** |
| 0.5 | 0.02872817461999513 |
| 0.75 | 0.032144202654983196 |

最终选择 `alpha=0.25`；78 层中 61 层使用 `clip_ratio=0.92`，17 层使用
`clip_ratio=0.94`。

### 1.3 对精度敏感的数据保留 BF16

三段式 KV cache 没有把所有信息统一量化：

- 每请求前 64 个 prefix token 保持 BF16；
- 最近 256 个 token 保持 BF16；
- 只有更早的 history token 降级为 INT2；
- RoPE key cache 保持 BF16；
- DSA index/cache 保持原精度，token 选择逻辑不变；
- prefix、recent、history 通过统一 global LSE 合并，而不是分别做独立
  softmax。

这使边界 token、位置编码和稀疏选择路径不受 INT2 量化影响。

### 1.4 拒绝超出误差门限的 grouped prefill 实现

grouped prefill 的首个可编译 BF16 tensor-core 版本虽然更快，但 output/LSE
最大绝对误差分别为 `0.009153/0.002593`，超过既定
`0.002/0.002` 门限，因此未放宽门限，也未接受该版本。

最终实现保留跨 head 的 KV 复用，但把 score/value dot 改为 FP32
`input_precision=ieee`。相对旧实现的 output/LSE 最大绝对差降至：

- output：`3.874301910400391e-06`；
- LSE：`1.430511474609375e-06`。

该实现随后完成苹果800 cold-cache CUDA 回归：124 passed、0 skipped、
0 failed。

### 1.5 当前精度结果与证据边界

固定 256 道 GSM8K 测试中：

| 指标 | BF16 | OSCAR |
|---|---:|---:|
| 正确数 | 105 | 107 |
| Accuracy | 0.41015625 | 0.41796875 |
| Request failure | 0 | 0 |

OSCAR 汇总值比 BF16 多 2 题，即高 `0.0078125`。但两轮保存的协议指纹不同，
且当前缺少 BF16 逐题 predictions，因此不能把这 2 题差异归因于 OSCAR，也不能
据此声称 OSCAR 提升了模型精度。

## 2. 性能优化改动

### 2.1 初始性能与瓶颈

在 1K/batch1、128 输出 token 的固定探针中，初始结果为：

| 指标 | BF16 | 初始 OSCAR | OSCAR 相对 BF16 |
|---|---:|---:|---:|
| TTFT | 352.445 ms | 5,049.520 ms | +1,332.7% |
| TPOT | 156.705 ms | 243.127 ms | +55.1% |
| 请求吞吐 | 0.04934 req/s | 0.02783 req/s | -43.6% |

Profiler 将主要问题定位为：

- decode 中 78 层重复执行布尔 mask、`nonzero` 和张量索引；
- prefill 沿用 decode 的 split16，并扫描固定 2,048 个 top-k 槽位；
- prefill 的 8 个本地 head 重复读取和反量化同一批 KV；
- decode 每层重复解析 metadata，并反复分配 demotion 临时 Tensor。

### 2.2 Decode 索引快路径

源码提交：`98ddd3f4ef645bddec76d96cd86a11d17232aaa2`。

纯 decode 每请求只产生一个新 token，当前位置必为 `seq_len-1`，不可能属于
current history。实现据此：

- 直接计算 query position 和 final sequence length；
- 按请求顺序读取 HP row；
- 跳过 current-history mask、`nonzero` 和空张量索引；
- 保留 prefill 通用路径和 recent-to-history demotion 语义。

实测效果：

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| KV update CPU total | 16.107 s | 9.293 s | -42.3% |
| KV update CUDA total | 1.080 s | 0.439 s | -59.3% |
| `aten::nonzero` 调用数 | 29,952 | 234 | -99.2% |
| `aten::index` 调用数 | 110,004 | 10,944 | -90.1% |
| TPOT | 243.127 ms | 206.735 ms | -15.0% |

该改动不处理 prefill，因此 TTFT 只从 `5,049.520 ms` 变为
`5,014.582 ms`，变化约 `-0.7%`。

### 2.3 Prefill top-k 裁剪与 split1

源码提交：`a94b1f640fe504be3d741a1070e43f806eaad894`。

实现把 prefill/mixed batch 的 top-k view 裁到
`min(topk_tokens, max_seq_len)`，去掉确定无效的 `-1` 尾部，并把 prefill
从 split16 改为 split1；纯 decode 继续使用实测最优的 split16。

单卡单层 CUDA 时间从 `62.883839 ms` 降至 `46.382080 ms`，下降
`26.24%`。TP=8 端到端结果：

| 指标 | Decode 快路径 | 增加 prefill 快路径 | 变化 |
|---|---:|---:|---:|
| TTFT | 5,014.582 ms | 3,714.821 ms | -25.9% |
| TPOT | 206.735 ms | 205.908 ms | -0.4% |
| 请求吞吐 | 0.03196 req/s | 0.03346 req/s | +4.7% |

### 2.4 Grouped prefill：跨 head 共享 KV

源码提交：`35ab1846447fc86b4b2177e76c5939503cc3701b`。

同一 query row 的 8 个本地 head 使用相同的 DSA selected token、BF16
prefix/recent、INT2 history 和 RoPE KV。旧 kernel 为每个 head 独立读取、
解包和反量化；新 kernel 在一个 Triton program 中同时处理最多 16 个 head，
复用上述 KV 数据。为适配苹果800：

- 使用 `num_stages=1`，避免 shared memory 超限；
- score/value dot 使用 FP32 IEEE 精度，满足既定数值门限；
- prefill 保持 cropped top-k/split1；
- decode 路径保持 split16。

单卡单层结果：

| 实现 | CUDA 时间 |
|---|---:|
| 旧 full top-k/split16 | 62.896130 ms |
| grouped 前 cropped top-k/split1 | 46.382080 ms |
| grouped 后 cropped top-k/split1 | **13.284352 ms** |

grouped 后相对前一版 cropped/split1 加速 `3.491×`，相对最初
full/split16 加速 `4.735×`。TP=8 端到端结果：

| 指标 | Grouped 前 | Grouped 后 | 变化 |
|---|---:|---:|---:|
| TTFT | 3,714.821 ms | 1,317.120 ms | -64.54% |
| TPOT | 205.908 ms | 202.668 ms | -1.57% |
| 请求吞吐 | 0.03346 req/s | 0.03696 req/s | +10.47% |
| Prefill mixed stage1 | 3,300.032 ms | 约 911–912 ms | 约 -72.36% |

### 2.5 Decode metadata 与 demotion scratch 复用

源码提交：`14c768b406b3e39a2d4d5be77a9046ac7ccc26d1`。

针对 grouped 后剩余的 decode CPU 开销，最新实现：

- 每个 batch 只物化一次 decode position、final sequence length 和
  demotion HP row；
- 78 层 attention 直接复用这些 metadata；
- 每层缓存 BF16 gather 与 FP32 rotated demotion scratch；
- 仅在 device、dtype、latent rank 改变或容量不足时重新分配 scratch。

验证结果为 CPU 96 passed、29 个 CUDA 显式 skip；苹果800完整的 CUDA 回归
125 passed、0 skipped、0 failed。候选 OCI 的两次独立构建和递归验收、
runtime import、控制镜像及正式 preflight 均已通过。

该改动的首次 32K/batch1 定向探针已得到三轮完整测量值，三轮均为 3/3
completed、0 failed；但整轮在 profiler 完成后被仓库洁净门禁拒绝，没有生成
单格/总 summary。因此这些数据只能作为诊断证据，不能标记为正式通过结果，也
不能单独用于归因 metadata/scratch 改动的收益。具体数据和失败边界见 2.7。

### 2.6 当前剩余差距

在已有完整 1K/batch1 grouped 结果中：

| 指标 | BF16 | OSCAR | OSCAR 相对 BF16 |
|---|---:|---:|---:|
| TTFT | 352.445 ms | 1,317.120 ms | +273.71% |
| TPOT | 156.705 ms | 202.668 ms | +29.33% |
| 请求吞吐 | 0.04934 req/s | 0.03696 req/s | -25.09% |

8-rank trace 显示 grouped OSCAR 的 prefill/generation worker 窗口相对
BF16 分别慢 `445.12%/27.09%`。按 78 层折算，KV update CPU 路径约多
`43.05 ms/token`，仍是后续 decode 优化的首要可控项；prefill attention
本身仍是 TTFT 的主要差距。

### 2.7 32K/batch1 首轮诊断结果与证据边界

首次 32K/batch1 定向探针
`20260731T0225Z_stage9_candidate_decode_metadata_probe_32k_b1_v1`
固定使用 32,768 输入 token、128 输出 token、batch/并发 1、1 次 warm-up
和 3 轮正式测量。三轮均为 3/3 completed、0 failed：

| 轮次 | TTFT（ms） | TPOT（ms） | 吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 106,666.970 | 200.547 | 0.007568 |
| 2 | 106,606.493 | 200.303 | 0.007573 |
| 3 | 106,660.424 | 199.203 | 0.007578 |

三轮中位数与现有 BF16 v4 同格点对比如下：

| 指标 | BF16 | OSCAR 诊断值 | OSCAR 相对 BF16 |
|---|---:|---:|---:|
| TTFT | 12,528.026 ms | 106,660.424 ms | +751.37% |
| TPOT | 178.832 ms | 200.303 ms | +12.01% |
| 请求吞吐 | 0.02838 req/s | 0.007573 req/s | -73.32% |

这组数据表明当前候选的 TPOT 已落入相对 BF16 的 20% 差距以内，但 TTFT
仍约为 BF16 的 `8.51×`。由于没有旧 metadata/scratch 前、同口径的
32K/batch1 OSCAR 结果，不能把当前 TPOT 差距直接归因于 2.5 的改动。

profile 命令实际生成了 8 份 worker trace、8 份 CUDA table 和 1 份
frontend trace。rank 0 trace 中共有 144 个 execute context，其中前 16 个
均为 2,048-token prefill chunk，合计正好 32,768 token；单 chunk wall time
从约 `4,088 ms` 增至约 `6,894 ms`，后续 127 个 generation 窗口大多约
`266–271 ms`。这直接说明约 106.7 秒 TTFT 主要由 16 个 chunked prefill
窗口累计形成。

整轮最终没有通过：profile 完成后，正式 runner 检测到主仓库新增了当时尚未
跟踪的本文档，触发
`RuntimeError: repository became dirty` 并以退出码 1 结束。容器已删除，
退出后 8 张 GPU 均为 0 MiB、没有 compute process。由于门禁发生在
profiler bundle 校验和单格 summary 生成之前，本轮只能保留上述三轮测量与
trace 作为中间诊断证据。下一步先把本文档纳入 Git、扩展 trace 分析器以支持
多 prefill chunk 并完成归因，再以新 run ID 重跑同一正式格点。

### 2.8 32K 多 chunk trace 归因

`scripts/phase9/analyze_prefill_trace.py` 已从“只允许一个 prefill 窗口”最小
扩展为：

- 聚合 generation 前全部正 token prefill chunk；
- 保留单窗口既有字段的兼容语义；
- 新增 chunk 数、每 chunk token/时长和总 token；
- 只累计落在任一 prefill 窗口内的 kernel 与嵌套 annotation；
- 拒绝 generation 后出现 prefill 或 prefill 窗口重叠。

双 chunk 测试先复现旧实现的预期失败，改动后与原单窗口测试共同 2/2 通过；
固定控制镜像中的 Phase 9 三个工具测试文件合计 20/20 passed。

同一分析器随后流式解析 OSCAR 与 BF16 32K/batch1 的各 8 份 worker trace。
两边每个 rank 均为 144 个 execute context、16 个 prefill chunk 和精确
32,768 个 prefill token。同口径 8-rank 中位数为：

| Trace 指标 | BF16 | OSCAR | OSCAR 相对 BF16 |
|---|---:|---:|---:|
| Prefill wall（ms） | 10,086.470 | 105,753.449 | +948.47% |
| Prefill kernel 合计（ms） | 9,533.580 | 105,609.233 | +1,007.76% |
| 各 rank generation 中位数再取中位（ms） | 223.325 | 268.625 | +20.28% |

OSCAR prefill kernel 覆盖率中位数为 `99.8628%`，排除了约 95.7 秒差距主要
来自 Python、调度或 chunk 间空隙的解释。其
`_mixed_sparse_prefill_stage1` 为精确 `1,248=16×78` 次，累计中位数
`93,913.327 ms`，平均约 `75.251 ms/层/chunk`，占 OSCAR prefill wall
`88.80%`。BF16 原生 `_sparse_mla_kernel_final_static` 同轮累计中位数为
`3,384.374 ms`。

两份聚合结果 SHA256 为：

- OSCAR：`cf4887875df1577837576eb606d99e72161a5ae586d14d9c72ff7762d113ffa7`；
- BF16：`06eecce0b6b3fad99b43885bb1e83355ac6f0a518a978c3db8be268cbc8a158d`。

该阶段没有分配新 GPU，只分析已冻结 trace。结论已经收敛：32K TTFT 的下一
优化对象应是 grouped prefill stage1 本身，而不是继续优化 decode metadata
或调度。后续先用单卡、单层的 2,048-query/2,048-top-k 形状验证 kernel
精度/性能方案；任何超过既定 output/LSE `0.002/0.002` 门限的方案继续拒绝。

### 2.9 固化 2,048×2,048 单层优化负载

`scripts/phase9/benchmark_oscar_prefill.py` 已新增 `--seq-len`，用于复现
32K 请求中每个 chunk 的实际 2,048-query/2,048-top-k 单层形状。入口会拒绝：

- sequence length 不大于 prefix+recent 的配置；
- sequence length 超过固定 top-k 2,048 的配置；
- history token 数没有按 16-token cache block 对齐的配置。

当 `seq_len=2,048` 时，full top-k 与 cropped top-k 宽度相同，脚本只保留
语义唯一的两个配置：IEEE split16 参考与 grouped split1 候选，避免重复测量
同一配置。TDD 中旧实现先出现 3 个预期失败；实现后固定控制镜像中的 Phase 9
三个工具测试文件为 21/21 passed。该阶段未分配 GPU，也尚未产生新的 kernel
性能或精度结果；下一步发布该固定入口后再执行单卡测量。

### 2.10 2,048×2,048 IEEE 单层基线

固定入口由主仓库提交
`49c9a1e` 发布后，单卡轮次
`20260731T0345Z_prefill_2k_ieee_v1` 使用 GPU 0 和独立 Triton cache。
GPU 分配前两次 8/8 空闲检查为 `03:44:15Z/03:45:26Z`，间隔 71 秒；
两次均为 0 MiB、0% 且没有 compute process。

轮次固定为 5 次 warm-up、7 次正式测量、每次 1 iteration。结果为：

| 配置 | CUDA 中位数 | 峰值增量显存 | 相对 split16 |
|---|---:|---:|---:|
| IEEE split16 | 195.772 ms | 1,185.063 MiB | 1.000× |
| Grouped IEEE split1 | **47.158 ms** | **224.125 MiB** | **4.151×** |

grouped split1 相对 split16 的 output/LSE 最大绝对差为
`3.337860107421875e-06/9.5367431640625e-07`，均远低于
`0.002/0.002` 门限，轮次状态为 `passed`。结果 JSON 与日志 SHA256 为：

- result：`f98578db5b6326ff19cf4e62a284c47ca95869f66eb19413f00507570c071f8a`；
- log：`cf0b5f7da96cc4b491b1022552b4b07fe1b16a8d5b1b498f6ece4dd992b24616`。

容器自动删除，退出后 8 张 GPU 均为 0 MiB、无 compute process。这一结果
冻结了与 32K chunk 几何一致的当前 IEEE 基线；下一步可以只替换 grouped
prefill dot precision，继续用同轮 IEEE split16 作严格数值参考。

### 2.11 Grouped prefill TF32 与 hybrid 候选

源码提交：`24938975f70bbf6d502b3556bdb44de0a5c7bde7`。

该候选只把 `_mixed_sparse_prefill_stage1` 中 5 个 score/value
`tl.dot` 的 `input_precision` 从 `ieee` 切换为 `tf32`。以下路径没有改变：

- decode stage1 与 2,048 单层 benchmark 的 split16 IEEE 参考；
- token 选择、prefix/recent/history 分段和 INT2 反量化；
- softmax、global LSE、inverse rotation 与输出合并；
- cache 写入、demotion 和调度。

固定控制镜像中，Triton interpreter 与 prefill head-block 定向测试为
6/6 passed。源码提交时 ruff check/format、typos、mypy、SPDX、forbidden
imports、CUDA API 和 attention backend 文档等全部适用 hooks 通过；提交已
推送，源码仓库本地与远端一致。

主仓库提交 `b0370c3d7b168233fd7ed3e568fb04df343f0c51` 绑定该源码、
submodule 和实验前记录并完成推送。GPU 筛选轮次
`20260731T0354Z_prefill_2k_tf32_v1` 前，两次 8/8 空闲检查为
`03:52:49Z/03:53:54Z`，间隔 65 秒；两次均为 0 MiB、0%，且没有
compute process。

该轮未进入正确性检查、warm-up 或计时。Triton 在编译 grouped TF32 stage1
时报告：

- 所需 shared memory：`169,984 bytes`；
- 苹果800 单 block 上限：`166,912 bytes`；
- 超出：`3,072 bytes`。

容器以退出码 1 结束，没有生成 `result.json`；独立 cache 留下 53 个编译文件。
日志 SHA256 为
`3024ebc944d713a68b498d3e51703370bc7f388e5f109faeb2b15c2e4f0f19a1`，
当轮 kernel 源码 SHA256 为
`d74c04b0fc227209415cca4ef2c8b6a9ee1f5b9c86ce12045b58dbcdd56040ea`。
容器退出后 8 张 GPU 均回到 0 MiB、0%，没有 compute process。

因此，源码 `24938975f…` 的当前 TF32 形态被 GPU shared-memory 门禁拒绝；
它没有产生精度或性能结果，不能宣称更快或满足 `0.002/0.002`。下一步只能先
最小降低 grouped kernel 的 shared-memory 占用，并重新经过源码发布、双空闲
检查和相同协议筛选。

随后执行 CPU-only 的 SM80 离线资源 sweep
`20260731T0400Z_tf32_offline_resource_sweep_v1`。它复用 GPU 轮次实际生成的
TTIR，以固定 `num_warps=4/num_stages=1` 重编译；没有注入 GPU，运行前后
8 张 GPU 均为 0 MiB、0%。结果为：

| Dot 组合 | Shared memory |
|---|---:|
| 全 IEEE | 139,264 bytes |
| 全 TF32 | 169,984 bytes |
| 仅 RoPE score 改回 IEEE | 172,032 bytes |
| 仅 BF16 score 改回 IEEE | 202,752 bytes |
| 仅 history score 改回 IEEE | 169,984 bytes |
| 两个 value dot 改回 IEEE | 202,752 bytes |
| 三个 score dot 改回 IEEE | 204,800 bytes |
| 原生 BF16 pool 使用 BF16、history 使用 TF32 | **135,168 bytes** |

这说明简单混用 IEEE 会引入更大的 lowering 临时区，不能解决超限。最后一项只
让输入本来就是 BF16 的 query、prefix/recent 和 RoPE 路径使用 BF16
tensor core；旋转/反量化后的 history score 与 history value 继续使用 TF32。
它比硬件上限低 `31,744 bytes`，是当前唯一通过离线资源门禁的优化方向。

离线 summary 状态为 `passed`，SHA256 为
`1065b8416b2424a9c0fa48dce8c6a3b134f0cb1f7d6bc54a535481394b6136db`。
该 sweep 只证明 SM80 编译资源预算，不是 GPU kernel launch，也没有产生
output/LSE 或性能结果；仍必须落地源码并通过相同 GPU 严格筛选。

资源 sweep 选出的 hybrid 已由源码提交
`b9626ce9fdd627da23fd29629ceb09df83e1458b` 落地并推送。它只修改
`_mixed_sparse_prefill_stage1` 一个文件，共 7 行新增、8 行删除：

- 输入本来就是 BF16 的 query、prefix/recent latent 与 RoPE 使用 BF16
  tensor core；
- 对应 BF16 value 路径只在 dot 输入处把 probability 转为 BF16，accumulator
  仍为 FP32；
- rotation/INT2 反量化后的 history score 与 history value 继续使用 TF32；
- softmax/global LSE、最终输出、decode、cache 与调度均未改变。

固定控制环境中的 Triton interpreter/head-block 定向测试为 6/6 passed。
源码提交时 ruff check/format、typos、mypy、SPDX、forbidden imports、
CUDA API、attention backend 文档和 sign-off 等全部适用 hooks 通过；源码
本地与远端一致。前述离线 TTIR 对应资源预算为 `135,168 bytes`，但新源码
尚未经过 GPU launch，因此仍没有 hybrid output/LSE 或性能数字。
