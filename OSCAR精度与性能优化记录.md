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
`torch.allclose(atol=0.002, rtol=0.002)` 协议的允许范围，因此未放宽门限，
也未接受该版本。

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
精度/性能方案；任何不满足冻结 allclose 门限的方案继续拒绝。
这里的冻结门限精确定义为 output/LSE 分别执行
`torch.allclose(atol=0.002, rtol=0.002)`，同时报告 max_abs/max_rel；不是单独
要求 `max_abs<=0.002`。

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

主仓库 `96cd9c2133e96561ac4fb7db1b2b4bef8fa4e7ed` 发布该 submodule 与记录后，
GPU 筛选轮次 `20260731T0408Z_prefill_2k_hybrid_v1` 前两次空闲检查为
`04:06:50Z/04:07:56Z`，间隔 66 秒；8 张 GPU 均为 0 MiB、0%，没有
compute process。实际编译/launch 的 grouped stage1 shared memory 为
`135,168 bytes`，与离线结果一致。

该轮在正式正确性门禁被拒绝：

- output 最大绝对误差：`0.004933357238769531`；
- LSE 最大绝对误差：`0.0020360946655273438`；
- 冻结门限：output/LSE 分别执行
  `torch.allclose(atol=0.002, rtol=0.002)`；其中 output 未通过。

因此没有进入 warm-up 或计时，也没有生成 `result.json`；不能得到或推断 hybrid
性能。独立 cache 为 61 个文件，日志 SHA256 为
`2bc3e050dba4b3336b4384b140605f5cb7477a802c3b596b79d526f4958afd95`。
容器退出后 8 张 GPU 均为 0 MiB、0%，没有 compute process。源码
`b9626ce9f…` 被精度门禁拒绝，后续不能放宽门限；下一候选必须恢复对误差敏感
的 BF16 value probability 精度，再重新验证资源、精度和性能。

随后执行 CPU-only 的 SM80 离线资源轮次
`20260731T0412Z_hybrid_value_resource_sweep_v1`。它直接复用上述 hybrid
GPU 轮次生成的 TTIR，仅把 BF16 value 累加从 BF16 probability tensor-core
dot 恢复为 FP32 probability/TF32 dot；BF16 score、RoPE score 和 history
TF32 路径保持不变。有效编译结果为：

- shared memory：`135,168 bytes`；
- 苹果800 单 block 上限：`166,912 bytes`；
- 余量：`31,744 bytes`。

第一次容器命令因假设的 `/opt/vllm/bin/python` 不存在而没有启动编译；
查明固定镜像的实际解释器为
`/opt/fp8_speed_up_v4_venv/bin/python` 后，第二次执行状态为 `passed`。
summary SHA256 为
`7c0f2b560ea288b90be0626bacaf53f8cea9621ea5441a820a4761b39c6b09d9`，
候选 TTIR SHA256 为
`acc277e5605b5a012eefaa343b0b8430ebb6b28e65ed8145f3d960962f0e114c`。
全程没有分配 GPU。该结果只证明恢复 value probability 精度不会重新触发
shared-memory 超限，尚不代表 GPU output/LSE 或性能通过；下一步才落地最小
源码候选并执行 CPU/静态门禁。

该最小改动已由源码提交
`b247211c91cd787149123f0373945e8a0c6c9937` 落地并推送，Git tree 为
`619ea47d74296e77e1357858a53d3aaf11e349d6`。它只修改 grouped prefill
kernel 的 BF16 value 累加：

- softmax probability 保持 FP32，不再截断为 BF16；
- BF16 prefix/recent value 在 dot 输入处扩展为 FP32，并使用 TF32；
- BF16 score、RoPE score、history score/value、softmax/LSE、decode 和
  三段式 cache 语义均不变。

实际源码 diff 为 1 个文件、3 行新增、2 行删除。固定 CPU 容器中的 5 个
prefill head-block 参数节点与 Triton interpreter smoke 合计 6/6 passed，
耗时 10.27 秒；ruff check/format 和提交时全部适用 hooks 均通过。kernel
源码 SHA256 为
`978b260511a8a1aa6dd822f0eff8196a519982e454f694dba18335c9b09e78a0`，
源码仓库本地与远端一致。本阶段没有分配 GPU；`135,168 bytes` 仍只是对应
TTIR 的离线预算，GPU 精度与性能需要在主仓库发布该 submodule 后重新筛选。

主仓库 `329962cc3c35e9db0cd1cf11a32c603b66d0c6ce` 发布源码与记录后，
GPU 轮次 `20260731T0422Z_prefill_2k_value_precision_v1` 前两次 8/8
空闲检查为 `04:19:54Z/04:21:07Z`，间隔 73 秒；固定只使用 GPU 0。
实际 grouped stage1 shared memory 为 `135,168 bytes`，独立 cache 为
61 个文件。

该轮按自 benchmark 首次提交 `60acb2e8` 起冻结的
`torch.allclose(atol=0.002, rtol=0.002)` 协议状态为 `passed`，完成每个
配置 5 次 warm-up 和 7 次正式测量。相对同轮 IEEE split16：

| 配置 | CUDA 中位数 | 峰值增量显存 | 相对 split16 |
|---|---:|---:|---:|
| IEEE split16 | 195.836 ms | 1,185.063 MiB | 1.000× |
| Value 精度恢复 grouped split1 | **26.906 ms** | **224.125 MiB** | **7.279×** |

grouped split1 的 output/LSE max_abs 为
`0.004912614822387695/0.0020360946655273438`，max_rel 为
`201.39968872070312/0.0002782414376270026`；这些诊断值均完整保留，不能把
allclose 通过误写成 `max_abs<=0.002`。结果与日志 SHA256 为：

- result：`4749ee1262bd7f47fb5b1415a1706f6f08eb6aec179c39abb73afb5b48dd82b1`；
- log：`38b853106d3e9591ab55827da498b2a2cac55bb71d7f2c04dbd95d229436d406`。

`04:21:59Z` 复查 8 张 GPU 均为 0 MiB、0%，没有 compute process。该轮只证明
单卡单层协议、资源和性能筛选通过，不能直接用微基准替代 32K/batch1 端到端
结果。

完整 cold-cache CUDA 回归第一次启动
`20260731T0429Z_value_precision_full_cuda_v1` 时，控制镜像已带
`/bin/bash` entrypoint，命令却再次传入 `/bin/bash`，因此在进入 Python 前
退出；该轮执行 0 个测试、生成 0 个 Triton cache 文件，日志 SHA256 为
`66b1df48d7e195bc89c09c288e56602009988d13fd61eb57b089af85ebffc041`，
不计为 CUDA 结果。

修正命令后重新执行双空闲检查：`04:29:28Z/04:30:42Z` 间隔 74 秒，两次
8/8 GPU 均为 0 MiB、0%，没有 compute process。有效轮次
`20260731T0431Z_value_precision_full_cuda_v2` 固定只使用 GPU 0，绑定：

- 主仓库 `bb63852288139b505d14cbebfdb3777199d6666a`；
- 源码 `b247211c91cd787149123f0373945e8a0c6c9937`；
- 源码 tree `619ea47d74296e77e1357858a53d3aaf11e349d6`；
- 控制镜像
  `sha256:84c48782f440d2080a81347bfa31ec1e3bbf77bc6151629ef43d879bbe90989f`。

源码和 native rootfs 只读挂载，并使用独立空 Triton cache。有效结果为：

- 125 passed、0 skipped、0 failed；
- 19 warnings、80.32 秒；
- cold cache 380 个文件、文件内容合计 26,557,655 bytes；
- pytest 日志 SHA256：
  `392cccbedb4832c48d575268a4f239d75676320d6350ed90e89b370286843fbf`。

容器自动删除，`04:32:34Z` 复查 8 张 GPU 均为 0 MiB、0%，没有 compute
process。该结果证明 value 精度恢复源码通过当前完整苹果800 CUDA 正确性回归；
它仍未产生新的 32K/batch1 TTFT/TPOT。

Phase 6 候选输入已最小切换到源码
`b247211c91cd787149123f0373945e8a0c6c9937`、tree
`619ea47d74296e77e1357858a53d3aaf11e349d6` 和 tag
`glm52-oscar-a800-phase6-b247211c9-0275043c`。Dockerfile 默认身份已同步，
文件 SHA256 为
`a073e9432d8b21601c3736c4bac69b8902e1e0a4543123ddecb8e586ca9bc8a3`。
固定控制镜像的 Python 3.12.13 中，强制触发 PAX header 的两次独立 tar
确定性回归为 1/1 passed；JSON、Dockerfile hash、源码 commit/tree 与 diff
检查也全部通过。

输入由主仓库提交 `3d6c9765502bdcf246330f46a4d8ec263be41708` 发布后，
在两个独立目录完成完整构建与递归验收：

- v1：
  `artifacts/phase6/20260731T0440Z_candidate_b247211c9_value_precision_v1`；
- v2 重建：
  `artifacts/phase6/20260731T0443Z_candidate_b247211c9_value_precision_v2_rebuild`。

两轮共同得到：

- image/config：
  `sha256:8053b791ca3de5a7f2f47ac79ab35f981931b6a2b99848e0c7123c30d79a9e46`；
- manifest：
  `sha256:c9230c5fa3a499baa40bbb908e7ef81528f35510427494b4b4f4c4b7f715cb94`；
- candidate layer：
  `sha256:94ee660d577a3bd5e5f84753f9eb09b4fcc2211ca8451eeda209bd0957f9f3e6`；
- diff-ID：
  `sha256:1af1788b4225bb6d1f28072100a130624419fb3de01765828817a18f416f8679`；
- candidate layer size/member：109,147,537 bytes / 5,298；
- `index.json` SHA256：
  `5d866599528d8f9c381a43d76bf23b236a49e9b183b61fb2b591e2489fbd7108`。

两份 index/config/manifest/candidate layer 均逐字节相同。两次验收状态均为
`passed`，各自重新核对 4,744 个源码文件、4 份 rotation、7 个基础层原生扩展、
33 层身份和精确 Git tree，且 candidate layer 不含原生扩展或 whiteout。v1
的 build/verification SHA256 为
`7683b1ac2a0e544e5e548ada75eac98fc4420b4bcf366e0ca7a17f76e49f442e` /
`f379031bb573472e2186c7e7428751cf7ad7e2a011df3d81477eaf9719a8346a`；
v2 对应为
`96291d80f7a211ddf8bd693d29103cc3045584ecde75394df49846410eba6b51` /
`064b4a87a7f68a88934a9d3af492ac497b3ea2872102ff1e3d989475f1183863`。
两组报告哈希不同只来自记录的输出/解压目录路径，不影响完全一致的 OCI
不可变身份。本阶段为 CPU-only，没有分配 GPU；v1 作为后续导入候选。

v1 随后由一次性 Ubuntu 22.04 工具容器中的 `skopeo 1.4.1` 从只读 OCI
layout 导入 Docker daemon。33 层复制、config 和 manifest 写入完整结束，
工具容器自动删除。daemon tag 为
`glm52-oscar-a800-phase6-b247211c9-0275043c:latest`，image ID 精确等于
上述 image/config digest；source commit/tree、candidate layer、
Dockerfile、rotation manifest、runtime expectation 和 base manifest 共
8 项关键 label 均与 v1 验收值精确匹配。导入日志与 daemon inspect JSON
SHA256 分别为：

- `33bcbe794882548e79432aca1eabc881910c26465e2af230996a389cb1a6d5c3`；
- `3a4747ef3e29e8973cb8aa78fbead6e76363e55588c8915dc21bcd485a4c1c0c`。

导入和 daemon 审计没有注入 NVIDIA runtime，8 张 GPU 全程为 0 MiB、0%，
没有 compute process。

driver-injected runtime import 前的第一组空闲检查为
`04:51:55Z/04:53:19Z`，间隔 84 秒；两次 8 张 GPU 均为 0 MiB、0%，没有
compute process。首版探针已成功导入正式 Python/PyTorch/Triton、候选 vLLM
Python/原生扩展并读取 rotation artifact，但错误地对 artifact 顶层字典执行
`len(payload)==78`；实际顶层只有 `format_version/rotations` 两个键，78 层
张量位于 `payload["rotations"]`。该轮在输出 JSON 前退出，空 JSON 与日志
SHA256 分别为：

- `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`；
- `5a7571914ab553060e0167d52358dfd9854f16df2195b44f366da2ee390dc082`。

容器删除后重新完成 `04:54:06Z/04:55:34Z` 双空闲检查，间隔 88 秒。v2 已改为
读取内层 78 项，但探针额外执行了 `import flashinfer` 和
`import flashinfer.jit`，最后被 `cuda_initialized=false` 断言拒绝；此前冻结
且已通过的 runtime 协议只用 `importlib.metadata` 读取
`flashinfer-python/flashinfer-jit-cache` 版本，不导入这两个模块。因此 v2
不能证明候选镜像主动初始化 CUDA，也不能记作 runtime import 通过结果。v2
同样没有输出 JSON，空 JSON 与日志 SHA256 分别为：

- `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`；
- `3c40b97aa353aeed7873df66d94882ed38219680ea5d9b82e8e7f5fa5b3d67bd`。

`04:57:33Z` 退出复查为 8 张 GPU 0 MiB、0%，没有 compute process。两份失败
证据均已保留且未覆盖。

失败记录由主仓库提交
`5a636fef9e3353795465b795711eda29cbe81637` 发布后，
`05:02:03Z/05:03:13Z` 再次完成间隔 70 秒的双空闲检查。有效 v3 精确复用
既有协议，只通过 `importlib.metadata` 读取 FlashInfer 包版本；GPU 0 仅注入
驱动库。实际结果为 `passed`：

- Python/PyTorch/Triton：
  `3.12.13/2.11.0+cu129/3.6.0`；
- Transformers/Tokenizers：
  `5.8.1/0.22.2`；
- FlashInfer Python/JIT cache：
  `0.6.6/0.6.6+cu129`；
- vLLM Python/原生扩展均来自 `/opt/vllm_glm52_v1`；
- 78 层 rotation、manifest/rotations/runtime expectation 三项 SHA256
  全部匹配；
- `reasoning_effort=max` 可解析；
- `cuda_initialized=false`。

`runtime_import_v3.json` / log SHA256 分别为：

- `0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`；
- `f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`。

这两个哈希与此前同协议的有效候选完全一致。容器自动删除，`05:03:54Z` 复查
8 张 GPU 均为 0 MiB、0%，没有 compute process。driver-injected runtime
import 门禁现已完成。

Stage 9 控制镜像 Dockerfile 随后只把默认 base 切换到当前候选，文件 SHA256
为 `6e894ed39cfec2ef55386cb22187e86a775b32b33a39d3d7ba7b7f09e4454845`；
该复现入口由主仓库提交
`eeaf56c81b7c65116a84bf59d3467e4185c98aa5` 发布后再构建。CPU-only
有效目录为
`artifacts/phase9-control/20260731T0508Z_runtime_b247211c9_v1`，控制镜像：

- tag：`oscar-glm-stage9-runtime:b247211c9`；
- image ID：
  `sha256:edbbc87d609963abd05b3f55c81d0ecad7016f7f31a860f6a62fa74f2bb7b1b8`。

控制镜像共 34 层，前 33 层与候选 image
`sha256:8053b791…9e46` 逐层相同，全部 inherited labels 也完全匹配。
CPU runtime 检查确认 Git `2.34.1`、iproute2 `5.15.0`、Python/glibc
`3.12.13/2.35` 与固定包清单一致，且 `cuda_initialized=false`。

首次身份审计脚本把构建日志中的 12 位短 ID 扩写成猜测的完整 ID，因该错误断言
退出；它没有发现镜像内容错误，也没有分配 GPU。v2 改为从 daemon 读取完整 ID
后，34/33 层、基础层逐层继承、labels 和 entrypoint 全部通过。build log、
daemon inspect、runtime check 和有效 identity audit log SHA256 分别为：

- `5b3a06c5ce1d304b35a240b705121938bfc672a9dfb64453ca46658e2f3533ab`；
- `065210b59ff879e5b2eb90a66bebe4029d12aabc3a6c673ed9baf6fb8cded807`；
- `5ac65b5da3ffc9bcf642bd3beb8a989b177710d38c51a9457b5198ffd18c1f20`；
- `4d467e47fd9b9f943a7ba0e500db6322368eaa9f0ff3a95fa12c771fe98395a0`。

该阶段没有注入 NVIDIA runtime，`05:08:02Z` 复查 8 张 GPU 均为 0 MiB、0%，
没有 compute process。控制镜像门禁现已完成；配置迁移、工具测试、递归
verifier、preflight 和 32K/batch1 端到端仍未执行。

### 2.12 新候选正式链路迁移准备

为把已经通过单层精度/性能筛选和完整 CUDA 回归的源码
`b247211c91cd787149123f0373945e8a0c6c9937` 接入正式 32K/batch1
链路，已从候选 v1 的已验收 `extracted-layer` 机械派生新的 runtime
overlay。该操作只生成实验 artifact，没有修改源码、模型或冻结的 phase0
原生扩展。

新 overlay 的实际门禁结果为：

- 候选层与 overlay 均为 4,749 个普通文件，其中 4,744 个为源码文件、5 个为
  rotation/runtime artifact；
- 两边按相对路径与文件内容生成的递归清单 SHA256 均为
  `0568662ba0737a161ba20638cd926941be9ac6b633410ea51273ec3f144f360d`；
- overlay 另含 6 个原生扩展符号链接，其相对路径和绝对目标与上一份正式
  overlay 完全一致；
- 6 个目标均存在，SHA256 依次保持为 `_C` `1812bd98…ec70`、
  stable libtorch `e79f6ea4…6cea`、MoE `c59dc1aa…9f49`、
  cumem `a73a69ea…483`、FA2 `f8926ed5…9fb4`、FA3
  `170b2341…823c`。

第一次检查发生在 NFS 复制尚未退出时，只看到 3,216 个源码文件和 4 个链接；
复查确认原 `cp` 仍处于活动 I/O 状态，因此该中间计数没有被接受，也没有重复
启动复制。原进程完成后文件数达到 4,744，再补建最后 2 个链接并通过上述完整
门禁。

有效 overlay 位于
`artifacts/phase6/20260731T0440Z_candidate_b247211c9_value_precision_v1/overlay_rootfs`。
随后已按 Phase 1→Phase 5→Phase 7→Phase 9 的依赖顺序切换正式配置和
wrapper，并逐级使用上一份配置的实际 SHA256。四份配置的新 SHA256 为：

- Phase 1：`e6e5b5994d03a64dc4dfdeaab9a8adb3ecbe34ee1e19051c029f5de7278c2d25`；
- Phase 5：`40083bf3fd6887880283c5029c96cf29b6e551578fb45f9561893f3ad60c801b`；
- Phase 7：`871feea023d7c101dee1b45698e42b7249c8d1f1f1b61b34fe75c238827ed50a`；
- Phase 9：`e2c764d75e370a8c7784f1807faaf9ab674c3fc0e9b5ed8c17188c1d73349114`。

4 个 JSON 解析、9 个 shell 语法、Phase 9 Python compile、Git diff 和正式
配置/脚本范围的旧候选身份清零检查全部通过。

新控制镜像中的第一轮 Phase 7 工具测试得到 19 passed、1 failed；失败节点的
冻结 evaluator Python 启动器指向
`/dev/shm/oscar-glm-recovery-tools`，但本轮容器漏挂载该目录，子进程以
127 退出。失败日志 SHA256 为
`9d059fe3d5081ec9ea3d764fb260e42003bb29686b1a497a86c86264b6711879`；
它不是配置或测试断言失败。

补上与既有有效协议一致的只读工具挂载后，实际结果为：

| 测试组 | 结果 | 日志 SHA256 |
|---|---:|---|
| Phase 7 | 20 passed、0 failed | `53c86a8a3169d7ff8e56b7ec4073c3218aa86506c1c7d45682bbb9a8d85fbd62` |
| Phase 9 | 21 passed、0 failed | `92f4f2ced88076edab81ca744fa2c4118bf19a369bb8fa02e6758666d53caf5c` |

两轮唯一 warning 是只读项目目录无法写 pytest cache，不影响测试或实验产物。
随后在新控制容器的 bind-mount 命名空间内执行递归 verifier，64/64 checks
全部通过，状态为 `passed`；有效 JSON SHA256 为
`bfad6c621e58008ac215a2b1b223538efb786400c04f136ff3be174d509232c7`。
该门禁覆盖 Phase 1/5/7/9 派生身份、OCI descriptor、4,744 个 Git 文件、
6 个 lower-layer 原生扩展链接、rotation/runtime artifact、冻结 evaluator
及 32K/128K 配置约束。

本轮没有注入 NVIDIA driver；递归静态检查结束后，固定环境在导入
`vllm._C` 时按预期因缺少 `libcuda.so.1` 退出，整体 dry-run 退出码为 1，
固定环境 JSON 为空。完整日志与空 JSON SHA256 分别为
`70b50fa1f9a7ee9c1d407a92204160627e57e9bf093ac6cf0b1270b16199efed` /
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
这不会否定 64/64 静态结果，但也不能冒充完整 preflight。

上述配置、记录和 planning 已由主仓库提交
`696bd8f762438fe56d9bd5c934773b69ce5c78fd` 发布，本地与远端分支精确
一致。正式 driver-injected preflight
`20260731T0528Z_stage9_candidate_b247211c9_preflight_v1` 前，外层在
`2026-07-31T05:26:12Z` 和 `05:27:29Z` 两次检查 8 张 GPU，间隔 77 秒；
两次均为 0 MiB、0% 且没有 compute process。

preflight 实际退出码为 0。`static_preflight.json` 状态为 `passed`，
64/64 检查通过；固定环境 import 和服务参数解析均记录
`cuda_initialized=false`。实际解析值包括 TP=8、PP=1、
`TRITON_MLA_SPARSE`、`oscar_mla_int2`、
`max_model_len=131072`、`max_num_batched_tokens=2048`、eager、
async scheduling 关闭和 torch profiler。preflight log、静态检查、
固定环境和服务参数 JSON 的 SHA256 分别为：

- `e773a9d29c147325d5259cfcce6b839b049765a674b07b062a3d4960b159a362`；
- `5cf51c4be324b474eedcacf3f7fb604e881db3f36a3bb6bb06956cc36c21bd6c`；
- `1805df1cacf586e999ade577b811985b84d2646f0e8fc5092f9e598cb9e52464`；
- `22b4af3b6cfc95266334a20151b74c255ddffb925cdf29b92a0b87f986f95aa5`。

preflight 容器已自动删除，退出后 8 张 GPU 均为 0 MiB、0%，没有 compute
process。至此新候选的正式运行前门禁已经完成；随后执行的 32K/batch1
正式结果见 2.13。

### 2.13 Value 精度恢复候选的 32K/batch1 正式结果

正式单格轮次
`20260731T0536Z_stage9_candidate_b247211c9_32k_b1_v1` 绑定：

- 主仓库提交
  `1d32d26cfba5239886cba6fa8781acd7f1461f48`；
- 源码提交
  `b247211c91cd787149123f0373945e8a0c6c9937`；
- Phase 9 配置 SHA256
  `e2c764d75e370a8c7784f1807faaf9ab674c3fc0e9b5ed8c17188c1d73349114`；
- 32,768 输入 token、128 输出 token、batch/并发 1、1 次 warm-up、
  3 轮正式测量和 8+8+1 profiler。

新 GPU 分配前的外层空闲检查为 `05:34:03Z/05:35:12Z`，间隔 69 秒；
容器内检查为 `05:37:14Z/05:38:18Z`，间隔 64 秒。四次检查均为 8 张 GPU
0 MiB、0% 且没有 compute process。141/141 权重分片全部加载，模型加载耗时
`272.713535 秒`、每卡模型内存 `56.0 GiB`、可用 KV cache `13.74 GiB`。
启动、服务和 profiler 的长阶段均实际输出了 10 分钟进度。

三轮均为 3/3 completed、0 failed：

| 轮次 | TTFT（ms） | TPOT（ms） | 吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 47,170.434 | 199.458 | 0.013793 |
| 2 | 47,099.465 | 199.931 | 0.013795 |
| 3 | 47,143.207 | 198.385 | 0.013824 |

三轮 `mean` 指标中位数对比如下：

| 指标 | BF16 | 上一 OSCAR 诊断值 | 当前正式 OSCAR | 相对上一 OSCAR | 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| TTFT（ms） | 12,528.026 | 106,660.424 | 47,143.207 | -55.80% | +276.30% |
| TPOT（ms） | 178.832 | 200.303 | 199.458 | -0.42% | +11.53% |
| 请求吞吐（req/s） | 0.02838 | 0.007573 | 0.013795 | +82.16% | -51.39% |

因此 value 精度恢复候选已经把 32K TTFT 相对上一候选降低一半以上，TPOT
也继续处于相对 BF16 的 20% 门限内；但 TTFT 仍为 BF16 的约 `3.76×`，
性能优化尚未完成。

单格和总 summary 状态均为 `passed`。Profiler 耗时
`787.793621301651 秒`，8 份 worker trace、8 份 CUDA table 和 1 份
frontend trace 全部通过数量、rank、bytes 与 SHA256 校验；critical rank
为 3，kernel total 为 `78,446 ms`。8-rank profiler table 中，
`_mixed_sparse_prefill_stage1` 均为 1,248 次，CUDA total 中位数为
`34,398 ms`，即 `27.563 ms/层/chunk`；相对上一候选的
`93,913.327 ms` 下降 `63.37%`，但仍为 BF16 同项
`3,384.374 ms` 的约 `10.16×`。服务端最多运行 1 个请求、等待为 0、
preemption 为 0，KV usage 峰值为 `5.7902%`，排除了容量排队。

总 summary、单格 summary、profile result 和 profile runner log SHA256
分别为：

- `ae1ffb5c29f12d93b94f389904057b9cd7e25f25dfa36029a7ba255688f53418`；
- `139c2d8ca22e85bcd729113d6f5a1b3d5985202be3853998339d8e4fcec8801b`；
- `46ab91fad9e1c40986b32b0dc7db00502a1784b74943fafcd3b2dff1b698c80b`；
- `8812ac76efa75096636ebc97e37170392d493e7bc2a2d895cebeddc893a4e7c9`。

正式 runner 退出码为 0，容器自动删除；`06:17:50Z` 复查 8 张 GPU 均为
0 MiB、0%，没有 compute process。随后完成的多 chunk trace 归因见 2.14。

### 2.14 新候选 32K 多 chunk trace 归因

本阶段没有分配 GPU，只在固定控制镜像
`oscar-glm-stage9-runtime:b247211c9` 中流式解析 2.13 已冻结的 8 份 worker
trace。分析器 SHA256 为
`8b6b2393f93be3783f47ccbe3ecb26020cc2b65526c99fcb464c4768ce4330f7`，
固定环境为 Python `3.12.13`、`ijson 3.4.0.post0`、4 workers、top 40。

有效轮次为
`20260731T0618Z_value_precision_32k_prefill_trace_v1`。8 个 rank 均包含
144 个 execute context、16 个 prefill chunk 和精确 32,768 个 prefill
token。有效 summary 状态为 `passed`，耗时
`114.88569264579564 秒`，SHA256 为
`599c035a36a65135885aee53f9a9964e4db181f25e62185a0182ecfcb4fc5a58`。

启动有效分析前有两次环境错误，均没有读取 trace、生成有效结果或分配 GPU：

- 宿主机尝试在 root 所有的 `/dev/shm/oscar-glm-stage9/analysis` 下创建结果
  目录，被权限门禁拒绝；
- 首次容器命令没有传入 `uv run --no-project`，误触发完整 vLLM 项目依赖解析，
  因包 shadow 冲突在分析前退出。

改用容器内 root 写结果目录并显式增加 `--no-project` 后，得到以下同口径
8-rank 中位数：

| Trace 指标 | BF16 | 上一 OSCAR | 当前 OSCAR | 当前相对上一 OSCAR | 当前相对 BF16 |
|---|---:|---:|---:|---:|---:|
| Prefill wall（ms） | 10,086.470 | 105,753.449 | 46,934.364 | -55.62% | +365.32% |
| Prefill kernel 合计（ms） | 9,533.580 | 105,609.233 | 46,100.251 | -56.35% | +383.56% |
| 各 rank generation 中位数再取中位（ms） | 223.325 | 268.625 | 269.297 | +0.25% | +20.59% |

当前 prefill kernel 覆盖率中位数为 `98.2234%`。
`_mixed_sparse_prefill_stage1` 仍精确执行 `1,248=16×78` 次，CUDA total
中位数为 `34,398.099 ms`，平均 `27.5625 ms/层/chunk`，占当前 prefill wall
`73.29%`。它相对上一候选的 `93,913.327 ms` 下降 `63.37%`，但其相对
BF16 原生 prefill attention 的超额时间仍占当前 OSCAR 与 BF16 总 prefill
wall 差距的 `84.17%`。

去掉 stage1 后的剩余 prefill wall，当前候选为 `12,536.264 ms`，上一候选为
`11,840.122 ms`，反而增加 `5.88%`；因此上一轮约 58.8 秒的 TTFT 改善几乎
全部来自 stage1，而不是调度、Python 或其他 kernel。当前其他较大的 kernel
包括 rotation `3,390.661 ms`、MoE 主 Marlin `1,985.188 ms`、NCCL BF16
all-reduce `1,207.154 ms`、GEMM `915.758 ms` 和 FP8 indexer
`882.489 ms`，单项均显著小于 stage1。

结论是下一轮仍应只针对 grouped prefill stage1 做最小优化。现有
2,048×2,048 单层基准的 `26.906 ms` 与端到端 trace 的
`27.5625 ms/层/chunk` 接近，说明该微基准可继续作为候选筛选入口；不能把
优化方向改到已经由证据排除的调度间隙，也不能用 TPOT 已过门限替代 TTFT
收敛。

### 2.15 Grouped prefill 8-warps 候选

源码提交：`b87a401daf55b557b0b052f302fd35be222d1ff1`，Git tree：
`7df314f222234b3744794d59736e8bba36f8f8ae`。

2.14 对应的实际 SM80 编译产物显示，当前 grouped stage1 使用 4 warps、
128 threads，达到每线程 `255` registers，并产生 `656 bytes` stack；
shared memory 为 `135,168 bytes`。由于两个 `16×512` FP32 accumulator
贯穿整个 top-k 循环，当前最小候选只把 grouped prefill/split1 的
`num_warps` 从 4 改为 8，让同一 CTA 使用更多线程分摊 accumulator；纯 decode
和非 grouped 路径继续使用 4 warps。三段式 cache、dot precision、softmax/LSE、
输出、调度和所有张量内容均未修改，实际源码 diff 为 1 个文件、1 行新增、
1 行删除。

固定控制镜像中的 prefill head-block 参数节点与 Triton interpreter smoke
合计 6/6 passed、11.39 秒；ruff check/format 和提交时全部适用 hooks 均通过。
源码文件 SHA256 为
`28a70612e8f7efd30e58442de255757d966e7c74a4215721f928c019a935a32b`，
提交已推送，源码仓库本地与远端一致。

CPU 验证启动过程中，固定解释器最初没有安装 pytest；第一次 `uv run` 又因命令
显式调用原解释器而绕过 uv 临时环境，第二次改用 uv 的 `python` 后暴露缺少
`tblib`。显式加入 `pytest/tblib` 后才得到上述有效 6/6。首次 ruff 因源码只读
挂载无法创建 `.ruff_cache` 而退出；把 cache 指向容器 `/tmp` 后有效检查通过。
这些均为测试环境启动错误，不是源码断言失败，也没有分配 GPU。

主仓库发布该 submodule 与实验前记录后，单卡轮次
`20260731T0645Z_prefill_2k_8warps_v1` 前在 `06:43:45Z/06:44:51Z`
完成两次 8/8 GPU 空闲检查，间隔 66 秒；两次均为 0 MiB、0% 且没有
compute process。实验固定只使用 GPU 0，继续使用 2,048 query、2,048
top-k、每 rank 8 heads、latent 512、prefix/history/recent
`64/1728/256`、5 次 warm-up、7 次正式测量和冻结
`torch.allclose(atol=0.002, rtol=0.002)` 协议。

有效结果为：

| 配置 | CUDA 中位数（ms） | 墙钟中位数（ms） | 峰值增量显存（MiB） |
|---|---:|---:|---:|
| 同轮 split16 参考 | 195.790 | 195.829 | 1,185.063 |
| 8-warps grouped split1 | **24.090** | **24.145** | **224.125** |

8-warps grouped split1 相对同轮 split16 加速 `8.128×`；相对旧 4-warps
grouped split1 的 `26.906 ms` 再降低 `10.47%`。output/LSE 均通过冻结
allclose，诊断 max_abs 仍为 `0.004912614822387695` /
`0.0020360946655273438`，max_rel 为
`201.39968872070312` / `0.0002782414376270026`，与 b247 候选一致。

实际 SM80 产物的 shared memory 保持 `135,168 bytes`，每线程 registers
从 255 降至 247，stack 从 656 bytes 降至 0；有效 cubin、PTX 和 metadata
JSON SHA256 分别为：

- `8b1ada686adf05914127789fd9b106e39dfdfa914f855655e65749e9864654c4`；
- `9de74886c5c8d27d3debba052cb38aaa6c15035dbcd09e16465b474918ac68e3`；
- `49a78f6362794a8ababcb9394c4101c4ab0a739cdcf484038a32f576ecbcbd9c`。

有效 result/log SHA256 分别为
`5fed778844380192f642b5ba43ddae1394534d27cd664407641a7e05795791fb` /
`b2e43fd7a4d0142f3c3bfeef11661cc50ce014828e6298874d54813f82380f39`。
独立 Triton cache 共 61 个文件；容器退出后 `06:45:45Z` 复查 8 张 GPU
均为 0 MiB、0%，没有 compute process。

单卡阶段记录由主仓库提交 `9b5d81d` 发布后，完整 CUDA v1
`20260731T0655Z_8warps_full_cuda_v1` 因错误假设控制镜像中的 uv 位于
`/opt/uv/bin/uv`，在 pytest 启动前以 127 退出；该轮为 0 测试、0
Triton cache，不计为 CUDA 结果。失败 pytest/exit-code 文件 SHA256 为：

- `c84ec79ecb39af5d0a5a4fd484f2c901751f931f898d628ac68c79c9367b6517`；
- `743c7850cccfba5e53a9002663ec1ddd1079315a98bdbfdde10e6044f56abefe`。

CPU-only 探针确认实际 uv 路径为 `/usr/local/bin/uv`。失败记录由主仓库提交
`4971f21` 发布后，`06:54:58Z/06:56:21Z` 两次 8/8 GPU 空闲检查间隔
83 秒；两次均为 0 MiB、0% 且没有 compute process。有效轮次
`20260731T0657Z_8warps_full_cuda_v2` 固定只使用 GPU 0，源码与 phase0
native rootfs 只读挂载，并使用独立空 Triton cache。结果为：

- 125 passed、0 skipped、0 failed；
- 18 warnings、78.19 秒；
- cold cache 380 个文件、文件内容合计 25,886,842 bytes；
- pytest/exit-code SHA256：
  `bd3b6d624a79c5867279f28dcb924c56d8a9a812df5ad8ae4cc5538c1a0f63a1` /
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

容器自动删除，`06:58:25Z` 复查 8 张 GPU 均为 0 MiB、0%，没有 compute
process。因此 8-warps 已通过单层资源/精度/性能筛选和完整 cold-cache CUDA
正确性回归；下一步先发布本阶段记录，再构建新候选 OCI。当前仍没有新的
32K/batch1 端到端结果。

### 2.16 8-warps 候选 OCI 构建、daemon 与 runtime 验收

为把已经通过单层筛选和完整 CUDA 回归的源码接入正式 32K/batch1 链路，
Phase 6 构建输入已最小切换为：

- 源码 commit：
  `b87a401daf55b557b0b052f302fd35be222d1ff1`；
- Git tree：
  `7df314f222234b3744794d59736e8bba36f8f8ae`；
- 候选 tag：
  `glm52-oscar-a800-phase6-b87a401da-0275043c`；
- Dockerfile SHA256：
  `7c21383b7964210044d0820b2ff0587b3c20601b79bb066e4b03bfe5a46130d0`。

本次只修改 `configs/phase6/candidate_inputs.json` 中的源码 commit/tree、
output tag 和 Dockerfile hash，以及 `docker/Dockerfile.phase6-oscar` 的
默认源码 commit/tree。base manifest、rotation artifact、
runtime expectation、native extension 合约和确定性 PAX 构建逻辑均未改变。

固定控制镜像 Python 3.12.13 中，强制生成 PAX header 的两次独立 tar 回归为
1/1 passed。JSON 解析、源码 commit/tree、Dockerfile 实算 SHA256、
Phase 6 Python compile 和 `git diff --check` 均通过；旧 b247 身份在
Phase 6 配置和 Dockerfile 范围内为 0。

上述输入由主仓库提交
`b15cf3020a4b0af795db84c65cab629df3a06824` 发布后，在两个独立目录完成
CPU-only 构建与递归验收：

- v1：
  `artifacts/phase6/20260731T0707Z_candidate_b87a401da_8warps_v1`；
- v2 重建：
  `artifacts/phase6/20260731T0710Z_candidate_b87a401da_8warps_v2_rebuild`。

两轮验收状态均为 `passed`，共同得到：

- image/config：
  `sha256:eef27939ae476dd0d727a49ffeaacfcea7c7dfba3311de3146a6cd8feca06eab`；
- manifest：
  `sha256:57c03fca6636b9d0ebd97d5d8aecf94b82e872995c76057938738f3e469e484a`；
- candidate layer：
  `sha256:2ac4b80a9c50ba87db18b41646e5fbf843220e61d8206069b719a7dcf8a2d108`；
- diff-ID：
  `sha256:ab049b45296d89f143fe3e3caa9715a2e9f06404974ba9290ee12ac5fae1c02a`；
- candidate layer size/member：109,147,643 bytes / 5,298；
- `index.json` SHA256：
  `beea07784088897fbcf3339bae8f00faacb41e11d0adb969112c27348c54af88`。

两轮的 `index.json`、config blob、manifest blob 和 candidate layer blob
均逐字节相同。两次递归验收各自重新核对 4,744 个源码文件、4 份 rotation、
7 个基础层原生扩展、33 层身份和精确 Git tree；基础层逐层完全匹配，
candidate layer 不含原生扩展或 whiteout。v1 的 build/verification/log
SHA256 分别为：

- `63eef67f2b74c9ae260589b487c45a13e772816f6930ff8df09d2c434e87eab9`；
- `b5fceb3aba1820de39654887d559401d7b58ebbd36f372cf819c54cbd22642ae`；
- `5ed02a8b68f6d3fdf7b2b08dd6042e0c463c10f53c1bab25841444c75f6dff9a`。

v2 对应为：

- `b83ab95a515ab5d654d2584a85e39db2c2eee50df19d82f8bc4e0a6858d294e4`；
- `1fbf342e8bb67d3e2ba0e6a9339c1e667c07b5733f0fd17c05e40894a22683f9`；
- `686eda75572e422440f58a0b3d3cfc7aba0ade69e4c2519873171a964b3c24e0`。

这些记录文件包含各自的输出/解压目录，因此哈希不同；不可变 OCI 四项的逐字节
比对结果不受影响。本阶段没有分配 GPU，结束后 8 张 GPU 均为
0 MiB、0%，没有 compute process。

双构建与实时记录由主仓库提交 `699ca34` 发布后，v1 由一次性 Ubuntu 22.04
工具容器中的 `skopeo 1.4.1` 从只读 OCI layout 导入 Docker daemon。
导入退出码为 0，33 层、config 和 manifest 写入完整结束，工具容器自动删除。
daemon tag 为
`glm52-oscar-a800-phase6-b87a401da-0275043c:latest`，image ID 精确等于
上述 image/config digest；最后一层 diff-ID 也精确等于
`sha256:ab049b45296d89f143fe3e3caa9715a2e9f06404974ba9290ee12ac5fae1c02a`。

独立 daemon 身份审计状态为 `passed`，source commit/tree、candidate layer、
Dockerfile、rotation manifest/rotations、runtime expectation 和 base
manifest 共 8 项关键 label 全部与 v1 验收值匹配。导入日志、daemon inspect
和身份审计 JSON 的 SHA256 分别为：

- `52b5e778b182c7e1056eba83806612ac24398e18018f440fd24fe87f61319940`；
- `f5106ea2c06a62c7c47e25335431bb4ababcb5ff750518fd9e3f1c6a7fb8c85c`；
- `0435fca1244896b07f0e698ee4cd21db0ee99e2407f18a2b30607dce9588c4ee`。

APT 更新时非必需 deadsnakes PPA 出现一次 TLS warning，但已有索引随后成功
安装固定 `skopeo 1.4.1`，导入和上述身份审计均通过。整个导入阶段没有传入
`--gpus` 或注入 NVIDIA runtime，前后 8 张 GPU 均为 0 MiB、0%，没有
compute process。

导入阶段记录由主仓库提交 `763e226` 发布后，driver-injected runtime import
前在 `07:26:23Z/07:27:34Z` 完成两次 8/8 GPU 空闲检查，间隔 71 秒；
两次均为 0 MiB、0% 且没有 compute process。有效轮次固定只向容器注入
GPU 0 的驱动可见性，精确复用此前已通过的冻结协议；FlashInfer 只通过
`importlib.metadata` 读取包版本，没有导入 `flashinfer` 或
`flashinfer.jit`。

runtime import 退出码为 0，状态为 `passed`：

- Python/PyTorch/Triton：
  `3.12.13/2.11.0+cu129/3.6.0`；
- Transformers/Tokenizers：
  `5.8.1/0.22.2`；
- FlashInfer Python/JIT cache：
  `0.6.6/0.6.6+cu129`；
- vLLM Python 与 `_C` 均来自 `/opt/vllm_glm52_v1`；
- 78 层 rotation 和 manifest/rotations/runtime expectation 三项 SHA256
  全部匹配；
- `reasoning_effort=max` 可解析；
- `cuda_initialized=false`。

双空闲检查日志、有效 JSON 和日志 SHA256 分别为：

- `04358a5b8ffde60e99e91efd3b713e40a8ee28e62e9bcbeda2753515a8481982`；
- `0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`；
- `f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`。

JSON/log 与此前同协议的有效证据逐字节一致。容器自动删除，`07:28:38Z`
复查 8 张 GPU 均为 0 MiB、0%，没有 compute process。runtime import
门禁现已完成；控制镜像、正式配置、preflight 和新的 32K/batch1 端到端仍未
执行。

### 2.17 Stage 9 控制镜像输入切换

runtime 阶段记录由主仓库提交 `4e78917` 发布后，Stage 9 控制镜像
Dockerfile 只把默认 base 从
`glm52-oscar-a800-phase6-b247211c9-0275043c:latest` 切换为
`glm52-oscar-a800-phase6-b87a401da-0275043c:latest`。Git diff 只有这一行，
其余 apt 源、`git/iproute2` 安装和 entrypoint 均未修改。

新 Dockerfile SHA256 为
`580ae65d4e89094d08ceb17cd158cc28217fbc01fc235e822c90e99cdfdbbd9a`。
daemon 中对应 base 已重新核对为：

- image ID：
  `sha256:eef27939ae476dd0d727a49ffeaacfcea7c7dfba3311de3146a6cd8feca06eab`；
- 层数：33；
- source commit/tree：
  `b87a401daf55b557b0b052f302fd35be222d1ff1` /
  `7df314f222234b3744794d59736e8bba36f8f8ae`；
- candidate layer：
  `sha256:2ac4b80a9c50ba87db18b41646e5fbf843220e61d8206069b719a7dcf8a2d108`。

上述构建入口由主仓库提交 `b28f825` 发布后，已完成 CPU-only 控制镜像构建。
有效目录为
`artifacts/phase9-control/20260731T0736Z_runtime_b87a401da_v1`，
控制镜像为：

- tag：`oscar-glm-stage9-runtime:b87a401da`；
- image ID：
  `sha256:f38a75eca80d8c460a331d9046832152d949317f7bb5d9286809f71d991400b4`。

控制镜像共 34 层，基础候选为 33 层，前 33 层逐层完全匹配；继承 labels、
环境变量和 entrypoint 均与基础候选一致。CPU runtime 检查状态为 `passed`，
确认 Git `2.34.1`、iproute2 `5.15.0`、Python `3.12.13`、glibc `2.35`
及固定安装包版本，并记录 `cuda_initialized=false`。

build log、build exit code、daemon inspect、identity audit 和 runtime check
SHA256 分别为：

- `fc7ca5c5d8f169a73e62d9bea9f312334e1af8b3e414babaa1c0572b1146c26d`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `1cad083166e0dbc0e61c0c5dd0ffe281518a9365387e25ab86d57b81c9200ef9`；
- `21a9b01d7da558ad172e33b0716fb6b822758fa9f51856570f645497c367570b`；
- `5ac65b5da3ffc9bcf642bd3beb8a989b177710d38c51a9457b5198ffd18c1f20`。

构建、身份审计与 runtime 检查均未注入 NVIDIA runtime，也没有分配 GPU；
`07:38:15Z` 复查 8 张 GPU 均为 0 MiB、0%，没有 compute process。控制镜像
门禁现已完成；随后完成的正式 overlay、配置与静态门禁见 2.18。

### 2.18 8-warps 正式链路静态迁移

控制镜像阶段记录由主仓库提交 `31fac2c` 发布后，已从 2.16 的 v1
`extracted-layer` 机械派生正式 runtime overlay：

`artifacts/phase6/20260731T0707Z_candidate_b87a401da_8warps_v1/overlay_rootfs`。

候选层与 overlay 均为 4,749 个普通文件，其中 4,744 个为源码文件、5 个为
rotation/runtime artifact；两边按相对路径和文件内容生成的递归清单 SHA256
同为
`20ee2d142bf14c51be2b453b798358db41591209dff3a2303519a5e60628cdc1`。
overlay 另含 6 个 lower-layer 原生扩展符号链接，目标均存在；`_C`、
stable libtorch、MoE、cumem、FA2 和 FA3 的 SHA256 分别为：

- `1812bd980b0c50681bc853d922f5d1a70a572bcb53e599963cc05621e86aec70`；
- `e79f6ea4b1e89658ad8a74747f9af1551ca93b97d089361277134245e9bd6cea`；
- `c59dc1aaba3b60ebc42a4523fe66ecc439863878ebdd7c5530accd9c75879f49`；
- `a73a69ea63fe10a8ffe5d805e71c69453aea042cb6bea384b65706bba8628483`；
- `f8926ed5fa3a80bfdf19a2ccb2cbc1d886bd2bac2a82237eb330a761a7c19fb4`；
- `170b2341b508feaff514478cf8c2fca5a5ff6fed5d4b748c9470b1aebc8a823c`。

随后按 Phase 1→Phase 5→Phase 7→Phase 9 的依赖顺序切换源码、OCI、
control image、overlay 和 wrapper 身份，并逐级使用上一份配置的实算
SHA256。四份配置的新 SHA256 为：

- Phase 1：`3258f706435400bf1e450f200a0f870a87cbda6d1afe72dddba4978cd2760b05`；
- Phase 5：`2c945b1d3a4d5e431c40c9292b50b4f7176d88779ec3a3df48100c2a11a252ce`；
- Phase 7：`78590b2077122ea0f697f123796a8a81c79b41b94cadfce243f4120ceefa2483`；
- Phase 9：`f15100e4871344bc82e9e9b7cd891502066f025573e652ecf57e7650b54a05f0`。

4 个 JSON 解析、9 个 shell 语法、Phase 9 Python compile、源码 commit/tree、
正式配置/脚本范围的旧候选身份清零和 `git diff --check` 均通过。新控制镜像
中的有效工具测试结果为：

| 测试组 | 结果 | 日志 SHA256 |
|---|---:|---|
| Phase 7 | 20 passed、0 failed | `c36356c3e397e8b9ad3f00fe3078273eb00891d9ee0c4b9633d2b41f135a723f` |
| Phase 9 | 21 passed、0 failed | `3ed122efec705e534f8e2ecfe71aa12d72cb802c823a6edd3dd99c60884f6f10` |

两轮唯一 warning 是只读项目目录无法写 pytest cache，不影响测试结果。Phase 7
首次启动命令没有覆盖镜像的 `/bin/bash` entrypoint，导致 bash 把 uv
二进制当作脚本解释，并在 pytest collection 前以 126 退出；失败日志 SHA256
为 `d99f9b7564b472dc1aa4df417ba89288a02a392401f15dd19249b75fa42a0a30`。
显式使用 `/usr/local/bin/uv` 作为 entrypoint 后得到上述有效结果。

递归 verifier 的 v1 漏挂载模型目录，在读取模型 `config.json` 前退出且没有
生成结果 JSON。v2 补上模型后，64 项中只有 Phase 0 runtime source tree
失败；展开检查确认 NFS 把普通文件执行位映射为可执行，与历史已知的 mode
假失败一致，其他身份、OCI、overlay、模型、rotation、冻结 evaluator 和
32K/128K 约束均通过。v3 再按正式协议把冻结 Docker source volume 覆盖到
Phase 0 source 路径后，64/64 checks 全部通过，状态为 `passed`；有效
JSON SHA256 为
`27926ec6ac9decac5e6ebb0a27c18c5e438e527f093d849e8cafbbb1f6e7c72b`。

本阶段没有注入 NVIDIA runtime，也没有分配 GPU；`07:52:09Z` 复查 8 张 GPU
均为 0 MiB、0%，没有 compute process。正式静态链路门禁现已完成；
driver-injected preflight 和新的 32K/batch1 端到端仍未执行。
