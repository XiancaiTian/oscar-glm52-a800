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
门禁现已完成；后续控制镜像、正式配置和 preflight 分别见
2.17、2.18 和 2.19，新的 32K/batch1 端到端仍未执行。

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
随后完成的 driver-injected preflight 见 2.19。

### 2.19 8-warps 正式 driver-injected preflight

2.18 的配置、wrapper、记录和 planning 已由主仓库提交 `e24f754` 发布，本地
与远端分支精确一致。正式轮次
`20260731T0758Z_stage9_candidate_b87a401da_preflight_v1` 前，外层在
`07:56:24Z/07:57:24Z` 两次检查 8 张 GPU，间隔 60 秒；两次均为
0 MiB、0% 且没有 compute process，因此不需要终止任何外部 GPU 进程。

preflight 实际退出码为 0。`static_preflight.json` 状态为 `passed`，
64/64 checks 全部通过。固定环境导入确认：

- Python/PyTorch/Triton：
  `3.12.13/2.11.0+cu129/3.6.0`；
- Transformers/Tokenizers：
  `5.8.1/0.22.2`；
- FlashInfer Python/JIT cache：
  `0.6.6/0.6.6+cu129`；
- vLLM Python 与 `_C` 均来自 2.18 的 b87 overlay；
- `cuda_initialized=false`。

服务参数解析也记录 `cuda_initialized=false`，实际值包括 TP=8、PP=1、
`TRITON_MLA_SPARSE`、`oscar_mla_int2`、
`max_model_len=131072`、`max_num_batched_tokens=2048`、eager、
chunked prefill 开启、prefix caching 和 async scheduling 关闭，以及
torch profiler。

双空闲检查、preflight log、exit code、静态检查、固定环境和服务参数 JSON
SHA256 分别为：

- `331fdd6bcdbb75e0b5217c85c1c3740c10b66416d3e9cd16246a312e9f59f6b1`；
- `9d3bfa9aba3b5c9bbb0b04a6fdbb05f4b15005e59af1a7e37673ee1f3b349d70`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `4c39b130024014b0285937b4014ac43b76aedd1767527bf04b91776783b22fde`；
- `31b3fecc3edc877325995f719b91365ffd4b12bb6b3dfe8f0948051afe6593c3`；
- `54d3dc89dc2544346b2b2c1f7b2352dee12956dda8ecbb3733c63d86c0bdd24f`。

preflight 容器已自动删除，`07:59:10Z` 复查 8 张 GPU 均为 0 MiB、0%，
没有 compute process。正式运行前门禁现已完成；新的 32K/batch1 端到端
结果见 2.20。

### 2.20 8-warps 候选的 32K/batch1 正式结果

正式单格轮次
`20260731T0805Z_stage9_candidate_b87a401da_32k_b1_v1` 绑定：

- 主仓库提交
  `085f0b1ecbfed37198cff0d98517983032438a25`；
- 源码提交
  `b87a401daf55b557b0b052f302fd35be222d1ff1`；
- Phase 9 配置 SHA256
  `f15100e4871344bc82e9e9b7cd891502066f025573e652ecf57e7650b54a05f0`；
- 32,768 输入 token、128 输出 token、batch/并发 1、1 次 warm-up、
  3 轮正式测量和 8+8+1 profiler。

新 GPU 分配前的外层空闲检查为 `08:03:28Z/08:04:28Z`，间隔 60 秒；
两次均为 8 张 GPU 0 MiB、0% 且没有 compute process。141/141 权重分片
全部加载；服务启动、三轮测量和 profiler 的长阶段均实际输出了 10 分钟进度。

三轮均为 3/3 completed、0 failed：

| 轮次 | TTFT（ms） | TPOT（ms） | 吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 41,618.560 | 201.766 | 0.014871 |
| 2 | 41,596.471 | 201.347 | 0.014888 |
| 3 | 41,622.931 | 201.265 | 0.014885 |

三轮 `mean` 指标中位数与 2.13 的 4-warps 正式结果、同格 BF16 对比如下：

| 指标 | BF16 | 4-warps OSCAR | 8-warps OSCAR | 8-warps 相对 4-warps | 8-warps 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| TTFT（ms） | 12,528.026 | 47,143.207 | 41,618.560 | **-11.72%** | +232.20% |
| TPOT（ms） | 178.832 | 199.458 | 201.347 | +0.95% | +12.59% |
| 请求吞吐（req/s） | 0.02838 | 0.013795 | 0.014885 | **+7.90%** | -47.55% |

三轮相对极差分别为 TTFT `0.0636%`、TPOT `0.2490%`、请求吞吐
`0.1121%`，结果稳定。8-warps 已将 32K TTFT 相对 4-warps 再降低
`5,524.647 ms`，请求吞吐提高 `7.90%`；但 TPOT 反而回退 `0.95%`，
且 TTFT 仍为 BF16 的约 `3.32×`，不能据此宣称整体性能已经追平 BF16。

单格和总 summary 状态均为 `passed`。Profiler 耗时
`838.0433747768402 秒`，生成并校验 8 份 worker trace、8 份 CUDA table
和 1 份 frontend trace；critical rank 为 6，kernel total 为
`73,788 ms`。8-rank profiler table 中，
`_mixed_sparse_prefill_stage1` 均精确执行 `1,248=16×78` 次；按表格的
毫秒级显示精度，其中位 CUDA total 为 `29,014 ms`，即约
`23.248 ms/层/chunk`。相对 2.13 的 `34,398.099 ms` 降低约
`15.65%`，方向与端到端 TTFT 改善一致。本正式轮次结束时，精确的多 chunk
trace 归因尚未执行，因此本节不把 table 中的阶段改善进一步外推为完整 TTFT
因果分解；后续归因结果见 2.21。

服务端最多运行 1 个请求、等待为 0、preemption 为 0，KV usage 峰值为
`5.7902%`。三轮测量峰值显存为每卡 `80,679 MiB`，profile 峰值为每卡
`80,691 MiB`，没有容量排队或抢占。

总 summary、单格 summary、profile result、profile runner log、正式外层日志
和双空闲检查日志 SHA256 分别为：

- `71417678ac1fdffde43d72674b38c75216e8727c566bcf31c34093b1bc7e130f`；
- `2340d04d04246aa230f44d902b662a5e546215c007111ccc4f08485de8a49f4d`；
- `acf24a7783d9bc520d159e30e8bdd6cecfa070314a51088ff0fd5bb93aa3ea73`；
- `f93dcfe07e5da4715e4d35ddeb5cdeb6bee4a14a6907d63a07c009ff7dc2c8e6`；
- `cc2a6208c115a1633685b9f5a9090ef32bd71b03eb780a822480d24363c6ec12`；
- `1e5e920f5d2821e88e24e1e3cc018b50c31123df73756861ce06a19d611156b1`。

正式 runner 退出码为 0，实验容器自动删除；退出后 8 张 GPU 均为
0 MiB、没有 compute process。当前仅有一个不占 GPU 的外部下载容器，未对其
执行终止操作。随后完成的 CPU-only 多 chunk 归因见 2.21。

### 2.21 8-warps 32K 多 chunk trace 归因

本阶段没有分配 GPU，只在固定控制镜像
`oscar-glm-stage9-runtime:b87a401da` 中流式解析 2.20 已冻结的 8 份
worker trace。分析器 SHA256 为
`8b6b2393f93be3783f47ccbe3ecb26020cc2b65526c99fcb464c4768ce4330f7`，
固定环境为 Python `3.12.13`、`ijson 3.4.0.post0`、4 workers、top 40。

有效轮次为
`20260731T0857Z_8warps_32k_prefill_trace_v1`。8 个 rank 均包含
144 个 execute context、16 个 prefill chunk 和精确 32,768 个 prefill
token。分析一次通过，summary 状态为 `passed`，耗时
`114.80616178922355 秒`。同口径 8-rank 中位数为：

| Trace 指标 | BF16 | 4-warps OSCAR | 8-warps OSCAR | 8-warps 相对 4-warps | 8-warps 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| Prefill wall（ms） | 10,086.470 | 46,934.364 | 41,516.570 | **-11.54%** | +311.61% |
| Prefill kernel 合计（ms） | 9,533.580 | 46,100.251 | 40,627.542 | **-11.87%** | +326.15% |
| 各 rank generation 中位数再取中位（ms） | 223.325 | 269.297 | 269.464 | +0.06% | +20.66% |

8-warps prefill kernel 覆盖率中位数为 `97.8594%`。
`_mixed_sparse_prefill_stage1` 仍精确执行 `1,248=16×78` 次，CUDA total
中位数为 `29,014.135 ms`，平均 `23.2485 ms/层/chunk`，占当前 prefill
wall 的 `69.89%`。它相对 4-warps 的 `34,398.099 ms` 降低 `15.65%`。

4-warps 到 8-warps 的 prefill wall 共减少 `5,417.793 ms`，其中 stage1
减少 `5,383.964 ms`，占 wall 改善的 `99.38%`。去掉 stage1 后的剩余
prefill wall 从 `12,536.264 ms` 变为 `12,502.436 ms`，只变化
`-0.27%`；generation 也只变化 `+0.06%`。因此 2.20 的 TTFT 改善可以
归因于 stage1，而不是 decode、调度间隙或其他 kernel。

相对 BF16，stage1 超出 BF16 原生
`_sparse_mla_kernel_final_static` 的时间仍占当前 OSCAR 与 BF16 prefill
wall 总差距的 `81.55%`。其余主要 kernel 与 4-warps 基本同量级：
rotation `3,390.088 ms`、MoE 主 Marlin `1,983.714 ms`、NCCL BF16
all-reduce `1,119.261 ms`、GEMM `915.478 ms` 和 FP8 indexer
`882.432 ms`，单项均显著小于 stage1。

有效 summary 与运行日志 SHA256 分别为：

- `a1a8e418fc0115efac9b63ba10c2cecc77968ed75a24a6004d52e24d0e731080`；
- `86347967aba42fdbd3557a68f9140a00ff6c346ed3aa30a8f223d25b5ff0a951`。

本轮只复用冻结 trace，没有启动服务、没有注入 NVIDIA runtime，也没有修改
源码。结论继续指向 grouped prefill stage1：下一步应先检查其当前循环、
访存和 accumulator 布局，再提出只影响该 kernel 的最小候选；不能把优化方向
转到已由数据排除的 generation 或调度路径。

### 2.22 Grouped prefill head block 离线资源筛选

对 2.21 指向的 stage1 做只读源码检查后，确认当前固定 TP=8 负载中每个
rank 实际只有 8 个 local heads，但
`_prefill_head_block_size(num_heads=8)` 返回 `block_h=16`。因此每个
program 的两个 `16×512` FP32 accumulator 中有一半 head 行最终被
`head_mask` 屏蔽。该检查没有修改源码。

随后在固定控制镜像
`oscar-glm-stage9-runtime:b87a401da` 中执行 CPU-only 的 SM80 AST
离线编译轮次
`20260731T0914Z_prefill_block_shape_offline_v3`。轮次绑定源码
`b87a401daf55b557b0b052f302fd35be222d1ff1`，固定
`num_warps=8`、`num_stages=1`、8 heads、latent 512、top-k 2,048
及正式 32K chunk 的 stride；`CUDA_VISIBLE_DEVICES` 为空，没有传入
NVIDIA runtime。三组编译结果为：

| 几何 | Shared memory | 相对当前 | 相对 166,912 B 上限 | 离线 cubin registers / stack |
|---|---:|---:|---:|---:|
| 当前 `block_h=16, block_t=16` | 135,168 B | 0 | 余 31,744 B | 255 / 0 B |
| 候选 `block_h=8, block_t=16` | **109,568 B** | **-25,600 B（-18.94%）** | **余 57,344 B** | 255 / 24 B |
| 对照 `block_h=16, block_t=32` | 219,136 B | +83,968 B | **超 52,224 B（+31.29%）** | 255 / 840 B |

`block_h=8` 精确匹配当前 8 个 local heads，并把 shared memory 占硬件上限
的比例降至 `65.64%`，因此保留为下一步的最小源码候选。
`block_t=32` 已明确超过苹果800单 block 上限，直接拒绝，不进入 GPU
验证。

离线 AOT cubin 的 registers/stack 与 2.15 中实际 runtime cold compile
产物的 `247/0 B` 并不相同，因此本轮只把 shared-memory 编译结果用于
资源筛选；候选出现的 24 B stack 也作为风险保留。该轮没有执行 kernel、
没有产生 output/LSE，也没有产生 CUDA 时间，不能据此声称精度通过或性能
改善。下一步必须先以最小源码改动落地并发布，再经过双 GPU 空闲检查和同一
2,048×2,048 单层冻结协议实测。

有效 summary、运行日志和 `cuobjdump` 资源日志 SHA256 分别为：

- `15fd8e83e7f5d4f434f537db2adf13086c76303c65f096ef07e3fd83943885fd`；
- `7507cdb4a452577deba052e802149b27bbcf855e49680433088f69b3fa8d27ea`；
- `3121f423044fab46601028c6cd559e065059caf058e83eaf7e52f2717d40ab4f`。

正式 v3 前有两次 fail-closed 启动错误：v1 把 78 个 kernel 参数误断言为
77 个，在编译前退出；v2 编译完第一组后把 Triton metadata 误当 dataclass，
在序列化时退出。两轮都没有 GPU 可见性，也没有形成完整三组结果；失败日志
SHA256 分别为
`c18c28cdd2fededcd64be2ebd8a6817bf14d3897090eebe77ead0d3e44508077` /
`1fe22bf3c9b6a0a26a73a5b8d909f188b248e75f05e8a1ca6e13beac3cc37a66`。
v3 退出码和资源审计退出码均为 0，轮次后 GPU compute process 查询为空。

### 2.23 8-head block 最小源码候选

2.22 筛选出的最小改动已由源码提交
`a2fe0205577b7f4707e9d31213cb5a80eda1f7d4` 落地并推送，Git tree 为
`b73806b6067b4533e94bf936610a0bf62f1a506d`。改动只把 grouped prefill
head-block bucket 调整为：

- 1–8 heads 使用 `block_h=8`；
- 9–16 heads 使用 `block_h=16`；
- 超过 16 heads 使用 `block_h=32`。

固定 TP=8、每 rank 8 heads 的 32K/batch1 负载因此从 16 行 block 精确收窄
为 8 行。kernel 数学、dot precision、softmax/LSE、三段式 cache、decode、
调度和张量内容均未修改。实际源码 diff 为 2 个文件、3 行新增、1 行删除：
生产 helper 新增 2 行，参数测试只修改 1 行期望值。

TDD 红灯阶段先只修改测试，旧实现得到 2 failed、3 passed；失败节点精确为
`1→8` 和当前固定负载的 `8→8`。落地生产逻辑后，正式 CPU-only 固化轮次
`20260731T0925Z_headblock_cpu_validation_v1` 在
`oscar-glm-stage9-runtime:b87a401da` 中运行 5 个 head-block 参数节点和
Triton interpreter smoke，结果为：

- 6 passed、0 failed；
- 3 个环境 warning；
- 10.72 秒；
- `CUDA_VISIBLE_DEVICES` 为空。

Ruff check/format、Python compile 和 `git diff --check` 均通过。提交时
typos、mypy、SPDX、root lazy imports、filename、Dockerfile graph、
forbidden imports、CUDA API、配置、attention backend 文档、boolean
context、suggestion 与 sign-off 等全部适用 hooks 通过。源码文件与测试文件
SHA256 分别为：

- `98ad2982a6d235ff71c68f668ce6b9e90959334fd96620267a89bce7a01b52ac`；
- `c3aed8b9a59906a3b7227f92160299c06497892ae030fb5b3f798423c480b03e`。

正式 CPU 测试日志和退出码文件 SHA256 分别为：

- `2f5c526e930244ff582105f91fa5de49252c43eece198c6991183be3d4b5323d`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

源码分支本地、远端均指向上述提交；本阶段没有分配 GPU，结束后的 compute
process 查询为空。2.22 的 `109,568 bytes` 仍是离线编译筛选值，本节也没有
产生 output/LSE 或 CUDA 时间；下一步必须先发布主仓库 submodule 与本记录，
再执行双空闲检查和同一 2,048×2,048 单卡协议。

### 2.24 8-head block 单卡精度与性能筛选

主仓库提交 `f5d51d097c0cbda0c518ef462619998702a94f54` 发布 2.23 的
submodule 与记录后，单卡轮次
`20260731T0931Z_prefill_2k_headblock_v1` 绑定：

- 源码提交/tree：
  `a2fe0205577b7f4707e9d31213cb5a80eda1f7d4` /
  `b73806b6067b4533e94bf936610a0bf62f1a506d`；
- 固定控制镜像：`oscar-glm-stage9-runtime:b87a401da`；
- 2,048 query、2,048 top-k、8 heads、latent 512；
- prefix/history/recent：`64/1728/256`；
- 5 次 warm-up、7 次正式测量、每次 1 iteration、seed 42；
- 固定只使用 GPU 0 和独立 cold Triton cache。

GPU 分配前在 `09:30:41Z/09:31:41Z` 完成两次 8/8 空闲检查，间隔
60 秒；两次均为 0 MiB、0% 且没有 compute process。唯一外部下载容器不占
GPU，因此没有执行终止操作。

轮次状态为 `passed`。同轮 split16 参考与 8-head grouped split1 结果为：

| 配置 | CUDA 中位数（ms） | 墙钟中位数（ms） | 峰值增量显存（MiB） |
|---|---:|---:|---:|
| 同轮 split16 参考 | 195.760 | 195.789 | 1,185.063 |
| 8-head grouped split1 | **19.077** | **19.106** | **224.125** |

8-head grouped split1 相对同轮 split16 加速 `10.262×`；相对 2.15 的旧
8-warps `24.090 ms` 再降低 `20.81%`，即加速 `1.263×`。7 个 CUDA 样本
范围为 `19.047–19.161 ms`，相对极差 `0.596%`。

output/LSE 均通过冻结
`torch.allclose(atol=0.002, rtol=0.002)`。诊断 max_abs 仍为
`0.004912614822387695/0.0020360946655273438`，max_rel 为
`201.39968872070312/0.0002782414376270026`，与 2.15 完全一致；因此仍不能
把 allclose 通过误写为 `max_abs<=0.002`。

实际 runtime cold compile 复现 shared memory `109,568 bytes`，与 2.22
离线结果一致；实际 cubin 为每线程 255 registers、0-byte stack。离线候选的
24-byte stack 没有在 runtime 产物中复现，但 registers 相对 2.15 的 247
增加到 255。独立 Triton cache 共 61 个文件。有效 metadata/cubin/PTX
SHA256 分别为：

- `c07c160b7dfb2c7c971213251403475da5c908c41c3009c2b023a9a7002877fa`；
- `663819a2e856e1518c60a9b04cecf56e6e74146c75e83275554cc1a871faaf1c`；
- `7bd5b008ccd5315d242a37e38716207e02d417935a8b3ecd84bb48f324380dbf`。

结果、运行日志、双空闲检查和资源日志 SHA256 分别为：

- `1861b7cb10c0bbebb5f0e984db5dc9d5de867c5852214430b343f0b6dd5806f4`；
- `97f3b0668050d35a75e24b00f9a1474758928b8d284220a7ff4057d3e163e122`；
- `2a26da124195edd79ef56288dc90854aa66aae2902424209d9a0118b90ae5b03`；
- `c4b61cceb4235a6d12e2428a324e0196811b9aea6b96f30f23b94fd318674000`。

benchmark 与资源审计退出码均为 0，实验容器自动删除；退出后 8 张 GPU
均为 0 MiB、0% 且没有 compute process。该结果只证明单卡单层筛选通过，
尚不等同于完整 CUDA 回归或 32K/batch1 端到端改善。下一步先发布本阶段记录，
再执行完整 cold-cache CUDA 回归。

### 2.25 8-head block 完整 cold-cache CUDA 回归

2.24 的单卡记录由主仓库提交
`b0f8644cb0dc1de291e66728ae34c6d156291f89` 发布后，完整回归轮次
`20260731T0936Z_headblock_full_cuda_v1` 绑定：

- 源码提交/tree：
  `a2fe0205577b7f4707e9d31213cb5a80eda1f7d4` /
  `b73806b6067b4533e94bf936610a0bf62f1a506d`；
- 固定控制镜像：`oscar-glm-stage9-runtime:b87a401da`；
- 源码与 phase0 native rootfs 只读挂载；
- 固定只使用 GPU 0 和独立空 Triton cache；
- `VLLM_OSCAR_RUN_CUDA_TESTS=1`，完整执行 `tests/oscar_mla`。

新的 GPU 分配前在 `09:36:51Z/09:37:51Z` 完成两次 8/8 空闲检查，
间隔 60 秒；两次均为 0 MiB、0% 且没有 compute process。

有效结果为：

- 125 passed、0 skipped、0 failed；
- 19 warnings、79.04 秒；
- cold cache 380 个文件；
- cache 文件内容合计 24,937,748 bytes；
- Docker 退出码 0。

pytest 日志、退出码和双空闲检查日志 SHA256 分别为：

- `a0dc4b047f218c255128d7b88abd4ebb95adc7f49dfaae9df58241563983420d`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `220768cee3f09eabb1a23b99ba4f2fb3ddc4f1391adff1b88ac21922b304d0b8`。

实验容器自动删除；退出后 8 张 GPU 均为 0 MiB、0% 且没有 compute
process。至此 8-head block 已通过离线资源、CPU/interpreter、单卡精度/
性能和完整 cold-cache CUDA 正确性门禁；但仍没有新的 32K/batch1 端到端
TTFT/TPOT。下一步先发布本阶段记录，再迁移候选 OCI 与正式 Stage 9 链路。

### 2.26 8-head block 候选 OCI 双构建与递归验收

2.25 的完整 CUDA 记录由主仓库提交
`0037f4e90a08c1bc39bf5eed69e5fa5514ae40b1` 发布后，Phase 6 构建输入
已最小切换为：

- 源码 commit：
  `a2fe0205577b7f4707e9d31213cb5a80eda1f7d4`；
- Git tree：
  `b73806b6067b4533e94bf936610a0bf62f1a506d`；
- 候选 tag：
  `glm52-oscar-a800-phase6-a2fe02055-0275043c`；
- Dockerfile SHA256：
  `13c687ed6b394ee095cbbccce38292ab96af6677516acb00f4a2a870984e9809`。

本次只修改 `configs/phase6/candidate_inputs.json` 中的源码 commit/tree、
output tag 和 Dockerfile hash，以及 `docker/Dockerfile.phase6-oscar` 的
默认源码 commit/tree。base manifest、rotation artifact、
runtime expectation、native extension 合约和确定性 PAX 构建逻辑均未改变。

首次控制容器命令虽启动了 `uv`，但没有指定临时 pytest 环境，在测试
collection 前因找不到 `pytest` 退出；该轮没有形成测试结果，也没有生成 OCI。
改用固定控制镜像 Python 3.12.13、`pytest==8.4.1` 和清华 PyPI 镜像后，
强制生成 PAX header 的确定性回归结果为：

- 1 passed、0 failed；
- 1 个只读 pytest cache warning；
- 0.22 秒。

JSON 解析、源码 commit/tree 与远端分支一致性、Dockerfile 实算 SHA256、
Phase 6 Python compile、旧 b87 Phase 6 身份清零和 `git diff --check`
均通过。

上述输入由主仓库提交
`504c8223554baeefc79b33629b3d47d0ffe5ac7e` 发布后，在两个独立目录完成
CPU-only 构建与递归验收：

- v1：
  `artifacts/phase6/20260731T0958Z_candidate_a2fe02055_headblock_v1`；
- v2 重建：
  `artifacts/phase6/20260731T1000Z_candidate_a2fe02055_headblock_v2_rebuild`。

两轮 build 状态均为 `built`，verification 状态均为 `passed`，共同得到：

- image/config：
  `sha256:51cd8c879b48f8556bc77a2feb8437838c4126ed1f2949e402a130999f4f68e4`；
- manifest：
  `sha256:4a8cec04b92ebafbc9f04ca1b4faf225c9ebdb044a40a3035e8f480682476334`；
- candidate layer：
  `sha256:37d706674b17841eea114ad352fd12f1071fc4c70cb4644ef32d4ec36fffbbd9`；
- diff-ID：
  `sha256:4b51d9dc80f82195fe2e7b4b3a8b8ea3b6f0b7204ba201f6152273ac4cb6caf8`；
- candidate layer size/member：109,147,568 bytes / 5,298；
- `index.json` SHA256：
  `bc20fcb62620ccaa5909925cdd3d57f91d2a52a140e0afda6b53ef535aee4872`。

两轮的 `index.json`、config blob、manifest blob 和 candidate layer blob
逐字节完全相同。两次递归验收各自核对 4,744 个源码文件、4 份 rotation、
7 个基础层原生扩展、33 层身份和精确 Git tree；基础层逐层完全匹配，
candidate layer 不含原生扩展或 whiteout。

v1 的 build/verification JSON SHA256 分别为：

- `1c6d8fe285bc02393d568f61de845fc09fd0d1ab176ee796a9f7d8a81c20d29d`；
- `10c363c4c704eba16d13f21436ee9d624943bb4e8b9c3df12cfcb78d00fa2840`。

v2 对应为：

- `ecf8d4101b10b954d82572fe7fd3df465da90a06182fbf1be3745eca17404cc6`；
- `135b875492a0355d2b10b1c9eac764db8cebe4ba05dfc07c22f277175040fa4f`。

两组 JSON 记录各自的输出/解压目录，因此哈希不同，不影响四项不可变 OCI
内容完全一致。本阶段没有注入 NVIDIA runtime，也没有分配 GPU；结束后
8 张 GPU 均为 0 MiB、0% 且没有 compute process。

双构建记录由主仓库提交 `06a0c181348d738fd5f5929afae43741bd8acbbb`
发布后，v1 使用一次性 Ubuntu 22.04 工具容器中的 `skopeo 1.4.1` 从只读
OCI layout 导入 Docker daemon。导入前目标 tag 不存在；`skopeo` 实际完成
33 个 blob、config 和 manifest 的写入并执行到 `Storing signatures`，
工具容器随后自动删除。

宿主侧原计划把组合日志写回 v1 artifact 目录，但该目录由构建容器的 root
所有，`tee` 在工具容器启动后因 permission denied 失败，使组合 shell 最终
返回 1。该错误只影响宿主日志落盘，不代表 `skopeo` 失败。为避免重复导入，
后续直接读取 daemon 状态并执行独立身份审计；审计状态为 `passed`：

- daemon image ID：
  `sha256:51cd8c879b48f8556bc77a2feb8437838c4126ed1f2949e402a130999f4f68e4`；
- 层数：33；
- 最后一层 diff-ID：
  `sha256:4b51d9dc80f82195fe2e7b4b3a8b8ea3b6f0b7204ba201f6152273ac4cb6caf8`；
- tag：
  `glm52-oscar-a800-phase6-a2fe02055-0275043c:latest`；
- source commit/tree、candidate layer、Dockerfile、rotation manifest、
  rotations、runtime expectation 和 base manifest 共 8 项 labels 全部匹配。

daemon inspect 与身份审计 JSON SHA256 分别为：

- `287a4af2aaeb8acdc9fec56a9daabe9626f6c2160958983ad56b52e5b82fa05c`；
- `c9250c6fd0e935cfb98a9f11421aaaf460ea3df0e2716cf048e1cfdb827b026c`。

导入和审计均未传入 `--gpus` 或注入 NVIDIA runtime；结束后 8 张 GPU
均为 0 MiB、0% 且没有 compute process。daemon 导入门禁现已完成；
该记录由主仓库提交 `9366900136694aea47168f11c4b1a90e266278e3`
发布后，driver-injected runtime import 在轮次
`20260731T1002Z_headblock_runtime_import_v1` 中执行。

新的 GPU 分配前在 `10:02:41Z/10:03:42Z` 完成两次 8/8 空闲检查，
间隔 61 秒；两次均为 0 MiB、0% 且没有 compute process。随后固定只向
GPU 0 注入 NVIDIA driver，运行不触发 kernel 的只读 import 探针。探针
一次通过、退出码为 0，确认：

- Python `3.12.13`；
- PyTorch `2.11.0+cu129`、Triton `3.6.0`；
- Transformers `5.8.1`、Tokenizers `0.22.2`；
- FlashInfer Python/JIT cache 为 `0.6.6` / `0.6.6+cu129`；
- vLLM Python 与 `_C` 分别来自
  `/opt/vllm_glm52_v1/vllm/__init__.py` 和
  `/opt/vllm_glm52_v1/vllm/_C.abi3.so`；
- rotation tensor 数为 78，rotation manifest、rotations 和 runtime
  expectation 三项 artifact 身份全部匹配；
- `reasoning_effort` 最大值为 `max`；
- 探针结束时 `cuda_initialized=false`。

双空闲检查、runtime import JSON 和运行日志 SHA256 分别为：

- `82cfb7f0c910b7b557e461181ccafe2a04e924f668585d68f962836985b60234`；
- `0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`；
- `f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`。

本轮 JSON/日志与历史同协议候选的证据逐字节一致。实验容器自动删除，
退出后 8 张 GPU 均为 0 MiB、0% 且没有 compute process。该门禁只证明
候选镜像运行时依赖和 artifact 身份正确，尚未测得新的 32K/batch1
端到端 TTFT/TPOT。下一步先发布本阶段记录，再把 Stage 9 控制镜像切换到
该候选并执行 CPU-only 身份审计。

### 2.27 Stage 9 控制镜像输入切换

2.26 的 runtime import 记录由主仓库提交
`062c910` 发布后，Stage 9 控制镜像 Dockerfile 只把默认 base 从
`glm52-oscar-a800-phase6-b87a401da-0275043c:latest` 切换为
`glm52-oscar-a800-phase6-a2fe02055-0275043c:latest`。Git diff 只有这一行；
其余 apt 源、`git/iproute2` 安装和 entrypoint 均未修改。

新 Dockerfile SHA256 为
`a9b9e22bfc004305b6b71167e40d43d5373eaf46c5c3b9b4d84e8ba69742e97c`。
daemon 中对应 base 已重新核对为：

- image ID：
  `sha256:51cd8c879b48f8556bc77a2feb8437838c4126ed1f2949e402a130999f4f68e4`；
- 层数：33；
- source commit/tree：
  `a2fe0205577b7f4707e9d31213cb5a80eda1f7d4` /
  `b73806b6067b4533e94bf936610a0bf62f1a506d`；
- candidate layer：
  `sha256:37d706674b17841eea114ad352fd12f1071fc4c70cb4644ef32d4ec36fffbbd9`；
- 最后一层 diff-ID：
  `sha256:4b51d9dc80f82195fe2e7b4b3a8b8ea3b6f0b7204ba201f6152273ac4cb6caf8`。

上述复现入口由主仓库提交 `7126356` 发布后，已完成 CPU-only 控制镜像
构建。有效目录为
`artifacts/phase9-control/20260731T1011Z_runtime_a2fe02055_v1`，
控制镜像为：

- tag：`oscar-glm-stage9-runtime:a2fe02055`；
- image ID：
  `sha256:0e13b724a2b89f3d698a3a130f13f27d8f8ef1c3acf96fbf38920306f79d50d5`。

控制镜像共 34 层，基础候选为 33 层，前 33 层逐层完全匹配；继承 labels
和 entrypoint 均与基础候选一致。CPU runtime 检查状态为 `passed`，确认
Git `2.34.1`、iproute2 `5.15.0`、Python `3.12.13`、glibc `2.35`
及固定安装包版本，并记录 `cuda_initialized=false`。

build log、build exit code、daemon inspect、identity audit 和 runtime check
SHA256 分别为：

- `7779e25026ec8373472f82193667155a5816a471540a66aec8ff84fda903cb2b`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `36d9ea8ea917744e8b2da936a032d1c1744fd54473d044ad9e476365d143be04`；
- `723515b7776893ba8481c51de3041a023db88117b720bfee2015fb431ca313f3`；
- `5ac65b5da3ffc9bcf642bd3beb8a989b177710d38c51a9457b5198ffd18c1f20`。

构建、身份审计与 runtime 检查均未注入 NVIDIA runtime，也没有分配 GPU；
前后 8 张 GPU 均为 0 MiB、0%，退出后没有 compute process。控制镜像门禁
现已完成；下一步先发布本阶段记录，再派生正式 overlay、迁移配置并执行
CPU-only 静态门禁。

### 2.28 8-head block 正式链路静态迁移

2.27 的控制镜像记录由主仓库提交 `e3176da` 发布后，已从 2.26 的 v1
`extracted-layer` 机械派生正式 runtime overlay：

`artifacts/phase6/20260731T0958Z_candidate_a2fe02055_headblock_v1/overlay_rootfs`。

候选层与 overlay 均为 4,749 个普通文件，其中 4,744 个为源码文件、5 个为
rotation/runtime artifact；两边按相对路径和文件内容生成的递归清单 SHA256
同为
`5cf59c75a0d60f85bc12d949a8630061f566bd1cb41c54a6f920716b28d6f51a`。
overlay 另含 6 个 lower-layer 原生扩展符号链接，目标均存在；`_C`、
stable libtorch、MoE、cumem、FA2 和 FA3 的 SHA256 分别为：

- `1812bd980b0c50681bc853d922f5d1a70a572bcb53e599963cc05621e86aec70`；
- `e79f6ea4b1e89658ad8a74747f9af1551ca93b97d089361277134245e9bd6cea`；
- `c59dc1aaba3b60ebc42a4523fe66ecc439863878ebdd7c5530accd9c75879f49`；
- `a73a69ea63fe10a8ffe5d805e71c69453aea042cb6bea384b65706bba8628483`；
- `f8926ed5fa3a80bfdf19a2ccb2cbc1d886bd2bac2a82237eb330a761a7c19fb4`；
- `170b2341b508feaff514478cf8c2fca5a5ff6fed5d4b748c9470b1aebc8a823c`。

首次链接审计使用了宿主 Python 3.8 不支持的 `Path.readlink()`，在只读打印
阶段退出；此前 4,749 个普通文件的清单比对已经通过，overlay 未被修改。
改用 `os.readlink()` 后，6 个链接的相对路径、绝对目标、目标存在性和上述
哈希全部通过。2.26 的有效 runtime import JSON 也已只读复制到 v1
artifact，SHA256 保持
`0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`。

随后按 Phase 1→Phase 5→Phase 7→Phase 9 的依赖顺序切换源码、OCI、
control image、overlay 和 wrapper 身份，并逐级使用上一份配置的实算
SHA256。四份配置的新 SHA256 为：

- Phase 1：`ff6c853fc46dacb184873f1d5fcf68489336fd3f97c4df923fe5138e37770ee6`；
- Phase 5：`de58d99aaef8e06244e21209e0b34b9e7b81f64592c99df52b85845eada2ba0c`；
- Phase 7：`4d66f3c6173e4d61e61d9b520727eb359a5e1e3cca964b7b1584c13a5027d006`；
- Phase 9：`f9d939580757ad5cb0673be5f55dca23f6d42c441efd47396f50c78e7a1bb520`。

4 个 JSON 解析、9 个 shell 语法、Phase 9 Python compile、源码
commit/tree、正式配置/脚本范围的旧 b87 身份清零和 `git diff --check`
均通过。新控制镜像中的工具测试结果为：

| 测试组 | 结果 | 日志 SHA256 |
|---|---:|---|
| Phase 7 | 20 passed、0 failed | `4f9d33d0a33d4f1608f0ab974091537eaef8ea22cfbe3177247fe3a1da92cc69` |
| Phase 9 | 21 passed、0 failed | `f28fce5f14dfd6f0caf05139b5344b5a83b71870f5670ec5181fde352796de9e` |

两组 warnings 仅为只读项目目录无法写 pytest cache，不影响测试结果。随后在
正式 phase0 source volume 覆盖 NFS mode 映射的容器挂载命名空间内执行递归
verifier，64/64 checks 全部通过，状态为 `passed`；有效 JSON SHA256 为
`99d90ff60315b9949e121c539239a2e7e13599b020cd2aab269c5d2765d4390f`。
该门禁覆盖 Phase 1/5/7/9 派生身份、OCI descriptor、4,744 个 Git 文件、
6 个 lower-layer 原生扩展链接、rotation/runtime artifact、冻结 evaluator
及 32K/128K 配置约束。

本阶段的两组工具测试和递归 verifier 三项退出码均为 0；全程没有注入
NVIDIA runtime，也没有分配 GPU，结束后 8 张 GPU 均为 0 MiB、0% 且没有
compute process。正式静态链路门禁现已完成；下一步先发布本阶段记录，再执行
driver-injected preflight，preflight 通过前不运行新的 32K/batch1。

### 2.29 8-head block 正式 driver-injected preflight

2.28 的正式链路、记录和 planning 已由主仓库提交 `ea88b55` 发布，本地与
远端分支精确一致。正式轮次
`20260731T1026Z_stage9_candidate_a2fe02055_preflight_v1` 前，外层在
`10:25:11Z/10:26:12Z` 两次检查 8 张 GPU，间隔 61 秒；两次均为
0 MiB、0% 且没有 compute process，因此不需要终止任何外部 GPU 进程。

preflight 实际退出码为 0。`static_preflight.json` 状态为 `passed`，
64/64 checks 全部通过。固定环境导入确认 Python/PyTorch/Triton 为
`3.12.13/2.11.0+cu129/3.6.0`，Transformers/Tokenizers 为
`5.8.1/0.22.2`，FlashInfer Python/JIT cache 为
`0.6.6/0.6.6+cu129`；vLLM Python 与 `_C` 均来自 2.28 的 a2fe
overlay，并记录 `cuda_initialized=false`。

服务参数解析同样记录 `cuda_initialized=false`，实际值包括 TP=8、PP=1、
`TRITON_MLA_SPARSE`、`oscar_mla_int2`、
`max_model_len=131072`、`max_num_batched_tokens=2048`、eager、
chunked prefill 开启、prefix caching 和 async scheduling 关闭，以及
torch profiler。

双空闲检查、preflight log、exit code、静态检查、固定环境和服务参数 JSON
SHA256 分别为：

- `cd1df9cacd7e0a7b064ebcbddf42f052c1231b0b865f7f133f46b01bb0ee7ee2`；
- `7406ae3b1997fed5b40f3759c8b5e1b87c036b8c041e0b1541c2f93b8fe5471c`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `18870284961afb12dbe985b5f1bcd1317013c1e62afef9f5bea0148945cafb78`；
- `28a394155b410830f3901ef3668ad6ba2f23071015cac0b354a4bd8e99766c3e`；
- `6395c4cb1039323850f5c683f5a3f6d0c082a53871d77b7414cb317fc329f19c`。

preflight 容器已自动删除，退出后 8 张 GPU 均为 0 MiB、0%，没有 compute
process。唯一仍运行的外部下载容器不占 GPU，因此没有执行终止操作。正式
32K/batch1 运行前门禁现已完成；下一步先发布本阶段记录，再执行新的双空闲
检查并启动端到端正式轮次。

### 2.30 8-head block 候选的 32K/batch1 正式结果

2.29 的 preflight 记录由主仓库提交
`c7cd7ed04dbcadeb31c53460c2a3b33d86f26955` 发布后，正式单格轮次
`20260731T1032Z_stage9_candidate_a2fe02055_32k_b1_v1` 绑定：

- 源码提交
  `a2fe0205577b7f4707e9d31213cb5a80eda1f7d4`；
- Phase 9 配置 SHA256
  `f9d939580757ad5cb0673be5f55dca23f6d42c441efd47396f50c78e7a1bb520`；
- 32,768 输入 token、128 输出 token、batch/并发 1、1 次 warm-up、
  3 轮正式测量和 8+8+1 profiler。

新 GPU 分配前的外层空闲检查为 `10:32:33Z/10:33:34Z`，间隔 61 秒；
容器内检查为 `10:34:51Z/10:35:55Z`，间隔 64 秒。四次检查均为 8 张 GPU
0 MiB、0% 且没有 compute process。141/141 权重分片全部加载，模型加载耗时
`316.964274 秒`、每卡模型内存 `56.0 GiB`、可用 KV cache `13.74 GiB`。
启动、服务和 profiler 长阶段均实际输出了 10 分钟进度。

三轮均为 3/3 completed、0 failed：

| 轮次 | TTFT（ms） | TPOT（ms） | 吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 36,245.415 | 200.278 | 0.016212 |
| 2 | 36,248.404 | 199.205 | 0.016248 |
| 3 | 36,240.998 | 198.588 | 0.016270 |

三轮 `mean` 指标中位数与 2.20 的 8-warps 正式结果、同格 BF16 对比如下：

| 指标 | BF16 | 8-warps OSCAR | 8-head OSCAR | 8-head 相对 8-warps | 8-head 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| TTFT（ms） | 12,528.026 | 41,618.560 | 36,245.415 | **-12.91%** | +189.31% |
| TPOT（ms） | 178.832 | 201.347 | 199.205 | **-1.06%** | +11.39% |
| 请求吞吐（req/s） | 0.02838 | 0.014885 | 0.016248 | **+9.15%** | -42.75% |

三轮相对极差分别为 TTFT `0.0204%`、TPOT `0.8484%`、请求吞吐
`0.3557%`，结果稳定。8-head block 已将 32K TTFT 相对 8-warps 再降低
`5,373.145 ms`，TPOT 也没有回退；但 TTFT 仍为 BF16 的约 `2.89×`，
请求吞吐仍低 `42.75%`，因此性能优化尚未完成。

单格和总 summary 状态均为 `passed`。Profiler 耗时
`769.6217455863953 秒`，8 份 worker trace、8 份 CUDA table 和 1 份
frontend trace 全部通过数量、rank、bytes 与 SHA256 校验；critical rank
为 7，kernel total 为 `68,199 ms`。8-rank profiler table 中，
`_mixed_sparse_prefill_stage1` 均精确执行 `1,248=16×78` 次；按表格显示
精度，其中位 CUDA total 为 `23,688.5 ms`，即约
`18.981 ms/层/chunk`，相对 2.21 的 `29,014.135 ms` 降低约
`18.36%`。该方向与端到端 TTFT 改善一致；精确的多 chunk trace 归因尚未
执行，因此本节不进一步外推为完整 TTFT 因果分解。

服务端最多运行 1 个请求、等待为 0、preemption 为 0，KV usage 峰值为
`5.7902%`。三轮测量峰值显存为每卡 `80,679 MiB`，profile 峰值为每卡
`80,691 MiB`，没有容量排队或抢占。

总 summary、单格 summary、profile result、profile runner log、正式外层日志
和双空闲检查日志 SHA256 分别为：

- `3ba959018e7d6944d06e266e42738b78fe9084e6eb8e5cd74831782b8b380cb9`；
- `5c18de9a6838f7873b45b90c6ec2cafb7e3709369b02ba4b289285f785f8033a`；
- `2d5d94bf5c288dcb7fe5f41b1ddb1e9d47049e0f25ed436b0967627baf5b8e02`；
- `80484f47449fe535b80bec802a78b89ad2982cf7e7aab6581919983a325b7c34`；
- `04444101d9cf07ffb7d57b56a2e247814932ff09baab3e9a2c4b0fdcbe6e31be`；
- `84cd6697d6182f4651be91c8366397f6fc0fda780e26a9518d65e185331c247d`。

关键小型证据已逐字节复制到
`artifacts/phase9-control/20260731T1011Z_runtime_a2fe02055_v1/formal_32k_a2fe02055`，
复制前后哈希一致。正式 runner 退出码为 0，实验容器自动删除；退出后
8 张 GPU 均为 0 MiB、0%，没有 compute process。唯一仍运行的外部下载
容器不占 GPU，因此没有执行终止操作。下一步先发布本阶段记录，再对本轮冻结
trace 做 CPU-only 多 chunk 归因。

### 2.31 8-head block 32K 多 chunk trace 归因

2.30 的正式结果已由主仓库提交 `a358e15` 发布。归因仅对该轮冻结的 8 份
worker trace 做 CPU-only 解析，未重新运行模型。首次尝试沿用了历史
root-owned 的 `/dev/shm/oscar-glm-stage9/analysis` 父目录，在 `mkdir`
阶段即因权限不足退出；容器、分析器和 trace 读取均未启动，也没有分配 GPU。
有效轮次改用当前用户独立目录，analysis ID 为
`20260731T1119Z_headblock_32k_prefill_trace_v2`。

有效轮次使用控制镜像 `oscar-glm-stage9-runtime:a2fe02055`、固定 Python
`3.12.13`、`ijson==3.4.0.post0`、4 个 CPU worker 和 top-40 汇总；显式
设置空的 `CUDA_VISIBLE_DEVICES`。分析器为
`scripts/phase9/analyze_prefill_trace.py`，SHA256 为
`8b6b2393f93be3783f47ccbe3ecb26020cc2b65526c99fcb464c4768ce4330f7`。
轮次耗时 `115.57293074764311 秒`，状态为 `passed`；8/8 ranks 均解析到
144 个 execute context、16 个 prefill chunk 和 32,768 个输入 token。

与 2.21 的 8-warps trace 及同格 BF16 32K trace 的精确中位数对比如下：

| 指标 | BF16 | 8-warps OSCAR | 8-head OSCAR | 8-head 相对 8-warps | 8-head 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| prefill wall（ms） | 10,086.470 | 41,516.570 | 36,257.407 | **-12.67%** | +259.47% |
| prefill kernel total（ms） | 9,533.580 | 40,627.542 | 35,316.438 | **-13.07%** | +270.44% |
| generation worker window（ms） | 223.325 | 269.464 | 269.448 | **-0.01%** | +20.65% |

8-head `_mixed_sparse_prefill_stage1` 的 8-rank 中位 CUDA total 为
`23,688.690 ms`，共 `1,248=16×78` 次，即 `18.981322 ms/次`，占
prefill wall 的 `65.33%`；相对 8-warps 的 `29,014.135 ms` 减少
`18.35%`。8-warps 到 8-head 的 prefill wall 共改善 `5,259.164 ms`，
其中 stage1 减少 `5,325.445 ms`，解释改善的 `101.26%`；去掉 stage1
后的 wall 反而增加 `0.53%`，generation worker window 基本不变。这说明
2.30 的 TTFT 改善确实来自 8-head block 对 stage1 的加速，而不是 decode
或其他 kernel 的共同改善。

与 BF16 相比，当前 stage1 超出 BF16 原生 attention 的部分仍解释
prefill wall 差距的 `77.58%`。当前其余较大的 kernel 中位 CUDA total 为
rotation `3,391.580 ms`、MoE `1,984.988 ms`、NCCL `1,128.943 ms`、
GEMM `915.454 ms`、FP8 indexer `882.513 ms`，均显著小于 stage1。因此
下一项最小性能候选仍应只针对 grouped prefill stage1；现有证据不支持把
优化范围扩到 generation 或其他 kernel。

首次结构化对比脚本误读了旧 1K BF16 summary，并因旧 schema 缺少多 chunk
字段只读退出；a2fe、8-warps 和 4-warps 已打印的数据以及有效 a2fe summary
均未被修改。复算使用的正确 BF16 32K summary 为
`/dev/shm/oscar-glm-stage9-analysis-20260731T0340Z_bf16_32k_b1_prefill_trace_v1/summary.json`，
SHA256 为
`06eecce0b6b3fad99b43885bb1e83355ac6f0a518a978c3db8be268cbc8a158d`。

有效轮次的 summary、run log、exit code 和 input manifest SHA256 分别为：

- `0a306ff31db2271d99e8ecc0857f0d3cddd9b3e7954289483b37b3e326f30e65`；
- `5996cd0e7568a09d0f83b2a0a296ae28146ba6e024cce0b074b7068a54017428`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `aef290cdeecd3bad55d32958c283c969fa239ae19b276e98b217e3a585a8a3dd`。

完整小型证据已逐字节复制到
`artifacts/phase9-control/20260731T1011Z_runtime_a2fe02055_v1/formal_32k_a2fe02055/trace_analysis`，
复制前后哈希一致。整个归因阶段没有注入 NVIDIA runtime 或分配 GPU，结束后
8 张 GPU 均为 0 MiB、0%，没有 compute process。下一步先发布本阶段记录，
再只围绕 grouped prefill stage1 筛选下一项最小优化。

### 2.32 Grouped prefill tile/warps 离线淘汰筛选

2.31 的归因记录已由主仓库提交
`bd17f5164f27f769512f332ce572513ee626a853` 发布。当前 grouped prefill
stage1 固定使用 `block_h=8`、`block_t=16`、`block_d=512`、8 warps 和
1 stage；每个 program 跨 8 heads 复用 value/rope 载入，同时为 BF16 与
history 两种 value basis 分别保留 `8×512` FP32 accumulator。

本阶段没有修改生产源码，而是在 a2fe 控制镜像内用 Triton
`ASTSource` 和 `GPUTarget("cuda", 80, 32)` 执行 CPU-only SM80 离线编译。
首次轮次 `20260731T1137Z_prefill_tile_offline_v1` 为避免 root-owned 证据，
使用了宿主 UID；该 UID 不在镜像 passwd 中，PyTorch 在 import 期调用
`getpass.getuser()` 时触发 `KeyError`。该轮没有进入任何 Triton 编译，
没有生成结果 JSON，也没有注入 GPU。

有效轮次 `20260731T1138Z_prefill_tile_offline_v2` 继续使用同一非 root UID，
并显式传入 `USER/LOGNAME`。轮次绑定源码
`a2fe0205577b7f4707e9d31213cb5a80eda1f7d4`、控制镜像
`oscar-glm-stage9-runtime:a2fe02055`、Python `3.12.13`、PyTorch
`2.11.0+cu129`、Triton `3.6.0`；`CUDA_VISIBLE_DEVICES` 显式为空。
summary 状态为 `passed`，12 组筛选耗时 `17.987540774047375 秒`。

`block_t=16` 的有效资源结果如下；registers/stack 均来自离线 cubin 的
`cuobjdump`：

| 几何 | Shared memory | 相对当前 | Registers / stack | 资源结论 |
|---|---:|---:|---:|---|
| h8/t16/w8（当前） | 109,568 B | 0 | 255 / 24 B | 精确复现当前 shared |
| h8/t16/w4 | 109,568 B | 0 | 255 / 1,240 B | shared 不降且 stack 显著增加 |
| h4/t16/w8 | 96,768 B | -12,800 B | 255 / 8 B | 仍高于双 block 阈值 |
| h2/t16/w8 | 90,368 B | -19,200 B | 255 / 8 B | 仍高于双 block 阈值 |
| h1/t16/w8 | 87,168 B | -22,400 B | 255 / 0 B | 仍高于双 block 阈值 |

苹果800 每个 SM 的 shared-memory 上限为 `166,912 B`，仅按 shared 计算的
双 block 阈值为 `83,456 B`；h1 仍超过 `3,712 B`。8-warps 版本还达到
每线程 255 registers，单个 256-thread block 已接近 SM register 文件容量，
因此更小 head block 也没有得到双 block 驻留条件。4-warps 的 h4/h2/h1
虽然每 block 线程更少，但 shared 分别仍为 `96,768/90,368/87,168 B`，
同样不能双驻留，且 stack 分别增至 `1,192/1,136/1,104 B`。

同时，固定 8 heads 下，h4/h2/h1 会把每 query 的 program 数从 1 增至
2/4/8，重复载入同一批 value/rope。由于没有增加驻留率，现有资源证据不支持
用这些重复访存换取 GPU 实测。

token tile 两侧也已闭合：

- h8/h4 的 `block_t=8` 均因 `tl.dot` 要求 K 维至少为 16 而在编译期拒绝；
- h8/h4 的 `block_t=32` 虽生成 cubin，但 shared 分别为
  `193,536/180,736 B`，均超过 `166,912 B` 单 block 上限。

因此当前简单的 head tile、token tile 和 warps 搜索空间全部淘汰，不进入
GPU 精度或性能测量。这一阶段只有编译资源结论，不能外推 output/LSE 或 CUDA
时间；下一候选需要改变 stage1 的实际计算/访存量，而不是继续调 launch tile。

有效 v2 的 compile script、summary、run log、exit code 和 resource log
SHA256 分别为：

- `1a1cd766dcc1e8a4b23f4a2a9a8a44dd9a3e32894677f6768f5e5460bed77ce7`；
- `8d719a0087c11253f7bbdaa9bb5bfd739365b1fe429bad6fbefdb8a491ea7987`；
- `effef1eebf621b4ced1a05df0e286a846db6e7b799c9f04e8c9fa5869f7da480`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `499c9f370f69765071045abd3a9b9bcb5a59cb22d6cb67421c6df739385d41cb`。

小型证据已逐字节复制到
`artifacts/phase9-control/20260731T1011Z_runtime_a2fe02055_v1/formal_32k_a2fe02055/tile_offline_v2`，
复制前后哈希一致。整个阶段没有注入 NVIDIA runtime 或分配 GPU，结束后
8 张 GPU 均为 0 MiB、0%，没有 compute process；唯一外部下载容器不占
GPU。下一步先发布本阶段记录，再检查能够减少 stage1 实际无效工作的最小
算法候选。

### 2.33 Grouped prefill causal 有效前缀循环候选

2.32 的离线淘汰记录已由主仓库提交
`68a5127baf5ea228c8f45c6bc02d8308624dd9f3` 发布。随后对正式 prefill 的
selected-token 生产链路做了只读语义核对：CUDA indexer 调用原生
`top_k_per_row_prefill`，其 `topKPerRowJob` 在行长不超过 top-k 时，明确把
前 `rowLen` 个槽写为有效 request-local 索引，并把其余槽写为 `-1`；行长
超过 top-k 时则输出恰好 top-k 个有效索引。

对请求最终长度 `L`、本批 query 数 `m` 和请求内第 `i` 个 query（从 0
开始），indexer 的有效行长为 `L-m+i+1`。OSCAR prefill 的
`query_position+1` 由独立 metadata 公式计算后也是 `L-m+i+1`。因此现有
stage1 已计算的 `causal_seq_len`，正好等于每个 query 的有效 selected-token
前缀长度；无需扫描 `-1`，也不会漏掉有效 top-k token。

最小源码候选已由提交
`fd281f5f974207998a95666d4015c441c5db49ab` 落地并推送，Git tree 为
`86185b214eb3d6f25108076a0a2c2c8dabb3d122`。生产改动仅位于 grouped
prefill stage1：

- 计算 `effective_topk=min(topk, causal_seq_len)`；
- 把固定 `range(0, topk, block_t)` 改为运行时
  `tl.range(0, effective_topk, block_t)`；
- 保留原有 selected、request、causal、HP row、prefix/recent/history mask，
  不修改 dot precision、softmax/LSE、accumulator、launch 几何、decode 或
  三段式 cache。

该改动让 2,048-query chunk 中较早 query 不再执行确定无效的尾部 tile；
最后一个 query 仍执行完整 2,048 个槽。TDD 先增加 source-invariant 测试，
旧实现按预期得到 1 failed；改动后定向测试为 1 passed。固定控制容器中的
完整 `test_triton_decode.py` CPU/interpreter 适用范围为：

- 7 passed、19 个 CUDA 显式 skip、0 failed；
- 3 个既有 Swig/vLLM version import warning；
- 15.96 秒。

其中 interpreter smoke 覆盖单请求多 query 和 multi-request 数值路径。
Ruff 0.14.0 check/format、固定 Python 3.12.13 compile、`git diff --check`
及提交时全部适用 hooks 均通过。

CPU-only SM80 离线轮次
`20260731T1208Z_causal_loop_offline_v1` 在
`oscar-glm-stage9-runtime:a2fe02055` 中完成，显式设置空的
`CUDA_VISIBLE_DEVICES`。12 组矩阵状态为 `passed`，耗时
`15.74477749876678 秒`；正式 h8/t16/w8 候选生成 201,648-byte cubin，
shared memory 保持 `109,568 bytes`。宿主 CUDA 12.9 `cuobjdump` 得到
255 registers、32-byte stack；相对 2.32 的静态循环离线产物只增加
8-byte stack，shared 和 registers 不变，没有触发 CPU-only 资源淘汰条件。

该轮复用了 2.32 的 compile script，因此 summary 内
`source_commit=a2fe...` 只表示控制 base，不能代表当前工作树身份；实际候选
由 `input.diff` 和两份 source SHA256 单独冻结。持久证据目录为
`artifacts/phase9-control/20260731T1011Z_runtime_a2fe02055_v1/formal_32k_a2fe02055/causal_loop_offline_v1`，
关键 SHA256 为：

- input diff：
  `fbd4d53c34f4482bcaa588817576dd69cc43cbb31c464d1e27dc8bf11d17512b`；
- source hashes：
  `5f8114e0589c9a5e749aa3e0988b97eececef8d2ef057fbf0f4e842ba51ec4b3`；
- summary：
  `bfad847f3db78271783a617c36359bcaaa6f51da608d21ff5b41ea33fe6ea5e3`；
- run log：
  `2dad6fe8b84582323e89af52919b25e17e6a9c05742cbc802a5d3e69f136fc59`；
- resource log：
  `9133653c8dd34b1538101a0ed8cb31df2de0390fa9fb066e6dedc22dcd8aab40`；
- compile/resource exit code 均为 0，对应文件 SHA256 均为
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

上述 CPU-only 阶段没有注入 NVIDIA runtime、执行 kernel 或分配 GPU；结束后
8 张 GPU 均为 0 MiB、0%，没有 compute process。源码、主仓库 submodule 与
本节第一部分随后已由主仓库提交 `26ebefd` 发布，满足进入 GPU 筛选的发布
门禁。

单卡冻结轮次 `20260731T1223Z_causal_loop_2k_gpu_v1` 前，在
`12:10:45Z/12:12:07Z` 完成两次 8/8 空闲检查，间隔 82 秒；两次均为
0 MiB、0% 且没有 compute process。唯一运行的外部下载容器不占 GPU，
因此没有执行终止操作。轮次固定只使用 GPU 0，绑定提交
`fd281f5f974207998a95666d4015c441c5db49ab`，使用独立冷 Triton cache；
冻结协议为 2,048 query、2,048 final sequence/top-k、8 个本地 head、
latent rank 512、prefix/history/recent 为 64/1,728/256、seed 42、每配置
5 次 warm-up、7 个正式样本和每样本 1 次迭代。

同轮 split16 参考与 causal-loop grouped split1 的结果为：

| 配置 | CUDA 中位数（ms） | Wall 中位数（ms） | 峰值增量显存（MiB） |
|---|---:|---:|---:|
| full-top-k split16 | 195.772415 | 195.803821 | 1,185.0625 |
| causal-loop grouped split1 | **12.858368** | **12.892746** | **224.1250** |

split1 相对同轮 split16 的 CUDA 加速为 `15.2253×`。与 2.24 中相同
2,048×2,048 协议下 a2fe 静态循环的 `19.077120 ms` 相比，本候选降低
`32.597960%`，即加速 `1.483635×`。7 个 split1 CUDA 样本全部完成，范围为
`12.843008–12.954624 ms`。

正确性状态为 `passed`。相对冻结参考，output/LSE 的最大绝对误差分别为
`0.0049126148/0.0020360947`，最大相对误差分别为
`201.3996887/0.0002782414`；判定使用既定的 `atol=0.002`、`rtol=0.002`
组合 allclose，而不是只按最大绝对误差判定。上述诊断值与 2.24 的 a2fe
候选相同，没有观察到 causal 有效前缀循环带来的新增数值漂移。

实际苹果800 cubin 的 `_mixed_sparse_prefill_stage1` 为 247 registers、
0-byte stack，独立 cache 共生成 61 个文件；未出现离线产物的 32-byte stack。
实验与资源审计退出码均为 0，容器自动删除；退出后 8 张 GPU 均为
0 MiB、0%，没有 compute process。

关键小型证据已逐字节复制到
`artifacts/phase9-control/20260731T1011Z_runtime_a2fe02055_v1/formal_32k_a2fe02055/causal_loop_2k_gpu_v1`，
SHA256 为：

- result：
  `4a8a7b6c6d966c2be0c81697e4a4fdb69fa7f73c10af0744fe61c75e970d5d52`；
- run log：
  `bcdc0d7da76a88892cb0a29386543d1afb09abf35997c10a8f4d43fc160cc5f0`；
- 双空闲检查：
  `60fe976bc046107267bfa3335b4a164828782387228971c879bd521cc2c749d0`；
- resource log：
  `2597b01f29dac3a366e0f12f2e13613cdf20358fc8a926bf98577f26a088104a`；
- benchmark/resource exit code 文件均为
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

因此该候选已经通过单层苹果800正确性和性能筛选；随后完成的完整 CUDA
回归见 2.34。此处单层结果仍不能替代 32K/batch1 端到端 TTFT/TPOT。

### 2.34 Causal 有效前缀循环完整 cold-cache CUDA 回归

2.33 的单卡结果与 planning 已由主仓库提交 `4d9a44b` 发布，发布确认由
`149245733fdfef80212881cf347219d8f858a0d9` 推送后，两仓均保持 clean。
完整回归轮次 `20260731T1223Z_causal_loop_full_cuda_v1` 绑定：

- 源码提交/tree：
  `fd281f5f974207998a95666d4015c441c5db49ab` /
  `86185b214eb3d6f25108076a0a2c2c8dabb3d122`；
- 固定控制镜像：`oscar-glm-stage9-runtime:a2fe02055`；
- 源码与 phase0 native rootfs 只读挂载；
- 固定只使用 GPU 0 和独立空 Triton cache；
- `VLLM_OSCAR_RUN_CUDA_TESTS=1`，完整执行 `tests/oscar_mla`。

新的 GPU 分配前在 `12:23:08Z/12:24:28Z` 完成两次 8/8 空闲检查，间隔
80 秒；两次均为 0 MiB、0% 且没有 compute process。唯一运行的外部下载
容器不占 GPU，因此没有执行终止操作。

有效结果为：

- 126 passed、0 skipped、0 failed；
- 19 warnings、86.77 秒；
- cold Triton cache 为 380 个文件；
- cache 目录的 `du -sb` 表观大小为 24,964,627 bytes；
- Docker 退出码为 0。

相对 2.25 的 125 项，新增的 1 项是本候选在 2.33 增加的 causal-loop
source-invariant 回归；不是跳过旧用例或缩小测试范围。19 条 warning 由既有
Swig/vLLM version、14 条 PyTorch JIT deprecation 和只读源码目录的两条
pytest cache warning 组成，不影响测试结论。

pytest 日志、退出码、双空闲检查、退出后 GPU 状态和 cache summary 的
SHA256 分别为：

- `0f85ebb7118e491df6d0465e1eb1051b7d208d0647e5219f3dcec78c429283e4`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `8e124fa9f2bb380548838b3f438cec61b036b6b3b69aa004e255d8c40de5d33d`；
- `8e808aba31dca047042b7c06b34b6711dbb43892dc950eef7d95019b0c696683`；
- `a2745e0e20cfbca9d5740b6c0c9063ab3985d62677b2c3c7e768f0cebd7a0ace`。

上述五份小型证据已逐字节复制到
`artifacts/phase9-control/20260731T1011Z_runtime_a2fe02055_v1/formal_32k_a2fe02055/causal_loop_full_cuda_v1`，
复制前后哈希一致。实验容器自动删除，`12:27:03Z` 复查 8 张 GPU 均为
0 MiB、0% 且没有 compute process。

至此 causal 有效前缀循环已经通过源码语义、CPU/interpreter、离线资源、
单卡苹果800正确性/性能和完整 cold-cache CUDA 正确性门禁；当前仍没有新的
32K/batch1 端到端 TTFT/TPOT。下一步先发布本阶段实时记录，再迁移候选 OCI
与正式 Stage 9 链路。

### 2.35 Causal 有效前缀循环候选 OCI 输入迁移

2.34 的完整 CUDA 结果已由主仓库提交 `9bfd77d` 发布。Phase 6 构建输入
随后只做候选身份迁移：

- source commit/tree 更新为
  `fd281f5f974207998a95666d4015c441c5db49ab` /
  `86185b214eb3d6f25108076a0a2c2c8dabb3d122`；
- output tag 更新为
  `glm52-oscar-a800-phase6-fd281f5f9-0275043c`；
- `docker/Dockerfile.phase6-oscar` 的默认 source commit/tree 同步更新；
- `configs/phase6/candidate_inputs.json` 记录新的 Dockerfile 实算哈希。

base manifest、rotation artifact、runtime expectation、7 个原生扩展合约、
确定性 PAX 构建逻辑和容器运行参数均未修改。新 Dockerfile SHA256 为
`2c97b4ef6397850b2ac095f8120d3e4f08206e829cdb8adcfc71281d85894b87`，
Phase 6 config SHA256 为
`9618cd4fc0fe53a0624e2bc7cab79b2be5d3f34cc05a9eaf4b9c50134e6a62d4`。

静态门禁确认：

- config 的 status、tag、source commit/tree 和 Dockerfile hash 共 5 项
  结构化断言通过；
- source 本地/远端提交与 tree 精确匹配；
- Phase 6 配置和 Dockerfile 范围内旧 a2fe commit/tree/tag 为 0；
- `git diff --check` 通过；
- 固定 Python 3.12.13 对 build/verify 两个脚本的 compile 通过；
- 固定 pytest 8.4.1 的 PAX header 确定性回归为 1 passed、0 failed，
  耗时 0.11 秒。

PAX 定向测试第一次启动时写错 unittest 类名，pytest 在 collection 后报告
node 不存在，0 个测试执行，后续 compile 也未运行；读取真实类名
`BuildCandidateOciTest` 后以上述同一源码和环境重跑通过。两轮都没有注入
NVIDIA runtime 或分配 GPU，结束后 8 张 GPU 均为 0 MiB、0%，没有
compute process。

上述输入和本节第一部分已由主仓库提交
`8b414a85a9cdfeb9b5a55e7da2ff368c7e57a2f5` 发布。随后在两个独立目录
并行完成 CPU-only 构建和递归验收：

- v1：
  `artifacts/phase6/20260731T1235Z_candidate_fd281f5f9_causal_loop_v1`；
- v2 重建：
  `artifacts/phase6/20260731T1235Z_candidate_fd281f5f9_causal_loop_v2_rebuild`。

两轮均于 `12:36:05Z` 启动；v2/v1 分别于 `12:40:56Z/12:42:40Z`
退出，退出码均为 0。两轮 build 状态均为 `built`，递归 verification
状态均为 `passed`，共同得到：

- image/config：
  `sha256:2369d967545750e55e0cb1544243725364bac57f89d22cf484083ef7e0dcd692`；
- manifest：
  `sha256:e18b2252cac0127b32b8de15e9185389bfdd43c411dcb345e2f18ca7a71663ee`；
- candidate layer：
  `sha256:9b6a02c438d6cc24dd013ca584f22663b8685ad33220c0407843271bd408d02e`；
- diff-ID：
  `sha256:f1b88c829fdcce24ff2f51906c9cfd4c31c3f9462c8e46950b22d22f92e335a5`；
- candidate layer size/member：109,147,808 bytes / 5,298；
- `index.json` SHA256：
  `20d0e846ba494ea1b3c41e669c80d4aa7e246b62e5df3d4dd4611b4f470b181d`。

两轮的 `index.json`、config blob、manifest blob 和 candidate layer blob
均已用 `cmp` 复核为逐字节完全相同。两次递归验收各自核对 4,744 个源码
文件、4 份 rotation artifact、7 个基础层原生扩展和精确 Git tree；33 层
中的前 32 层与 base 逐层完全匹配，candidate layer 不含原生扩展或
whiteout。

v1 的 build/verification JSON 和组合日志 SHA256 分别为：

- `f09720c652fffa6d6a83affe83439c0284d05af7c6264ea6277a93c3bb00d9fb`；
- `f18194ac52082161aefe90fbb48fb99c2ae7c594af8b5b56efc49775cd7f0d81`；
- `ecac59bae7b1f5561db427b7452decb1378e58a398d1a8471edbcc3e2255b507`。

v2 对应为：

- `ce109dd05185b18fb6e30898243901acb0d30d8955869c1e904fe2c1d660b417`；
- `0800804a5b58de7ab427f0ec38ee38635d6ee19f3d4fad7e7a8d943ba7ec5edd`；
- `d917c84404e227e164ffe779117e23f9841d737cbec4177cbeacfaa02d8b0cb4`。

两轮 JSON 会记录各自输出/解压目录，因此文件哈希不同，不影响四项不可变
OCI 内容完全一致。构建与验收全程未注入 NVIDIA runtime、未分配 GPU；
8 张 GPU 始终为 0 MiB、0%，没有 compute process，唯一外部下载容器不占
GPU。双构建结果与本节对应记录已由主仓库提交 `e1078ec` 发布。

随后执行 v1 的 Docker daemon 导入。正式导入前有两轮 fail-closed 错误：

- 第一轮把阿里 Ubuntu 镜像切为 HTTPS，但最小 Ubuntu 22.04 工具镜像尚无
  CA 证书；`apt-get update` 无法建立证书链，命令在安装 skopeo 前退出 100；
- 第二轮改用阿里 HTTP 镜像并成功安装 skopeo 1.4.1，但手写错 OCI source
  ref name；skopeo 在 descriptor 读取阶段退出 1，尚未复制任何 blob。

两轮退出后均复核目标 tag 不存在，没有覆盖 daemon 中的已有镜像。对应失败
日志/退出码文件 SHA256 分别为：

- HTTPS/CA 轮次：
  `6d21d0db919e7679028bca040b5e9c2745d72b07b6c19c00e185862dc04642d9` /
  `eea8254c7500ba3de996aa8ad6af399183f04e17d4a8102fde539dbc93a90012`；
- 错误 ref 轮次：
  `b53a743cee1140343f224674841b152d831d404f16d96dc8d7138dc431d89a81` /
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`。

有效轮次从 v1 `index.json` 原样读取
`glm52-oscar-a800-phase6-fd281f5f9-0275043c`，使用同一只读 OCI layout
和 skopeo 1.4.1 复制到对应 `:latest` daemon tag。日志完整执行到
`Storing signatures`，退出码为 0；工具容器自动删除。独立 daemon 身份
审计状态为 `passed`：

- image ID：
  `sha256:2369d967545750e55e0cb1544243725364bac57f89d22cf484083ef7e0dcd692`；
- 层数：33；
- 最后一层 diff-ID：
  `sha256:f1b88c829fdcce24ff2f51906c9cfd4c31c3f9462c8e46950b22d22f92e335a5`；
- tag：
  `glm52-oscar-a800-phase6-fd281f5f9-0275043c:latest`；
- source commit/tree、candidate layer、Dockerfile、rotation manifest、
  rotations、runtime expectation 和 base manifest 共 8 项 labels 全部匹配。

有效 import 日志、退出码、daemon inspect 和身份审计 JSON SHA256 分别为：

- `777795b9c822edd9c83e7aa469453fb834666ae2d551e91a202940c6ce15d29f`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `722785723b77f2561576df4ac195e333ad518dc3c5ceaa5ac8f257335ce9304a`；
- `c411dd4a8c359dfb9263b05c217058b70453a2846fd71707ac5197924ef75173`。

整个导入和身份审计阶段没有传入 `--gpus`、没有注入 NVIDIA runtime；结束后
8 张 GPU 均为 0 MiB、0%，没有 compute process。至此 causal-loop 候选的
双 OCI 构建与 daemon identity 门禁均已完成，并已通过主仓提交 `152a26c`
实时发布。

随后执行 driver-injected runtime import。第一轮命令遗漏 `docker run -i`，
容器内 Python 从空标准输入正常退出，因此 JSON 和日志均为空文件，不能计为
一次有效验证；该轮空 JSON、空日志的 SHA256 均为
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`，
退出码文件 SHA256 为
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

失败后重新执行固定 GPU 分配门禁：`2026-07-31 13:06:39Z` 与
`13:07:48Z` 两次检查间隔 69 秒，8 张 GPU 均为 0 MiB、0%，无 compute
process。有效轮次固定使用 GPU 0，并显式增加 `docker run -i`；仅注入 NVIDIA
driver，不触发 CUDA kernel。runtime import JSON 状态为 `passed`，实测身份为：

- Python `3.12.13`；
- Torch `2.11.0+cu129`；
- Triton `3.6.0`；
- Transformers `5.8.1`；
- Tokenizers `0.22.2`；
- FlashInfer Python/JIT cache 均为 `0.6.6` / `0.6.6+cu129`，仅通过
  `importlib.metadata` 读取；
- vLLM Python 入口为 `/opt/vllm_glm52_v1/vllm/__init__.py`，原生扩展为
  `/opt/vllm_glm52_v1/vllm/_C.abi3.so`；
- rotation 数量为 78；rotation manifest、rotations 与 runtime expectation
  的 SHA256 均与镜像 labels 一致；
- `reasoning_effort=max`，`cuda_initialized=false`。

本轮 runtime import JSON 与上一候选 a2fe 的冻结 runtime JSON 逐字节一致。
重试前双空闲检查日志、有效 JSON、有效日志、退出码与结束后 GPU 快照的
SHA256 依次为：

- `6c0f255cca7823ba2671744beeb43e8e214734d76eab9567e51e83796afeb99d`；
- `0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`；
- `f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `013b857a9253b9254e678ad24377d608f585d519425f9b2af4a2e0401917e524`。

有效轮次结束于 `2026-07-31 13:08:33Z`，容器已自动删除；8 张 GPU 再次为
0 MiB、0%，没有 compute process。至此 causal-loop 候选的双 OCI 构建、
daemon identity 与 driver-injected runtime import 门禁均已完成；对应记录
已由主仓库提交 `4af2aca` 实时发布。

Stage 9 控制镜像 Dockerfile 随后只把默认 base 从
`glm52-oscar-a800-phase6-a2fe02055-0275043c:latest` 切换为
`glm52-oscar-a800-phase6-fd281f5f9-0275043c:latest`；其余 apt 源、
`git/iproute2` 安装和 entrypoint 均未修改，旧 a2fe tag 在该文件中已清零。
新 Dockerfile SHA256 为
`93111035802a79bcb31564111a542e33660d167f4b4175addbdd89e5f65d8bbb`。

daemon 中对应 base 已重新只读核对为：image ID
`sha256:2369d967545750e55e0cb1544243725364bac57f89d22cf484083ef7e0dcd692`、
33 层、最后 diff-ID
`sha256:f1b88c829fdcce24ff2f51906c9cfd4c31c3f9462c8e46950b22d22f92e335a5`；
source revision/tree 分别为
`fd281f5f974207998a95666d4015c441c5db49ab` /
`86185b214eb3d6f25108076a0a2c2c8dabb3d122`，candidate layer 为
`sha256:9b6a02c438d6cc24dd013ca584f22663b8685ad33220c0407843271bd408d02e`。

上述复现入口与实时记录已由主仓库提交 `efba906` 发布。随后完成 CPU-only
控制镜像构建，有效目录为
`artifacts/phase9-control/20260731T1319Z_runtime_fd281f5f9_v1`；新镜像为：

- tag：`oscar-glm-stage9-runtime:fd281f5f9`；
- image ID：
  `sha256:9be0cbb72088f9fe4b48680814254be9306e0cd9b1a13c4fb49fa95911db321b`；
- 层数：34，前 33 层与上述 base 逐层完全一致；
- inherited labels 与 entrypoint 均和 base 匹配。

首次身份审计额外加入了冻结协议未要求的 base/control `Cmd` 相等断言。实际
base `Cmd` 为 `['-lc', 'sleep infinity']`，control 为 `null`，因此该轮在
正式输出前 fail-closed；空 audit log 与退出码文件 SHA256 分别为
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` /
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`。
失败证据已单独保留，CPU runtime 尚未在该轮执行。

按上一 a2fe 有效协议重跑后，34/33 层、前 33 层逐层继承、labels 和
entrypoint 全部通过。无 NVIDIA runtime 的 CPU runtime check 状态为
`passed`，确认 Git `2.34.1`、iproute2 `5.15.0`、Python `3.12.13`、
glibc `2.35` 和固定安装包版本，且 `cuda_initialized=false`；其 JSON 与
上一 a2fe 有效控制镜像逐字节一致。

build log、build exit code、daemon inspect、有效 identity audit、有效
runtime check 和 GPU 快照的 SHA256 分别为：

- `af6dbd291706b0783be6e0f11f4e984182af441f5ce9d1f29876e390abfb71b5`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `6de8292712570ac63c5855d3b1f07367ad9c1fccdd1239e416be71690f743577`；
- `0e5686ab1e46764c88685808fc396ba17cc64c24b87551b3a360c5c3f844f88c`；
- `5ac65b5da3ffc9bcf642bd3beb8a989b177710d38c51a9457b5198ffd18c1f20`；
- `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f`。

三项有效退出码均为 0。整个构建和审计阶段未传入 `--gpus`、未注入 NVIDIA
runtime；前后 8 张 GPU 均为 0 MiB、0%，compute process 为空。控制镜像
门禁现已完成；尚未迁移 Phase 1/5/7/9 正式配置或执行新的 32K/batch1
端到端测试。下一步先发布本阶段实时记录，再派生正式 overlay、迁移配置并
执行 CPU-only 静态门禁。

### 2.36 Causal 有效前缀循环正式链路静态迁移

2.35 的控制镜像结果已由主仓库提交 `a64dbc0` 发布。随后从已通过
递归验收的 Phase 6 v1 `extracted-layer` 机械派生正式 runtime
overlay：

`artifacts/phase6/20260731T1235Z_candidate_fd281f5f9_causal_loop_v1/overlay_rootfs`。

候选层与 overlay 均为 4,749 个普通文件，其中 4,744 个为源码文件、
5 个为 rotation/runtime artifact。两边按相对路径和文件内容生成的递归
清单 SHA256 同为
`bec9e45c5e0bd8a6b6a9e0fabb1a5f615d6877845a909cc5316b58e1780835d7`，
`cmp` 为逐字节一致。overlay 另含 6 个 lower-layer 原生扩展符号链接；
其相对路径和绝对目标与上一份正式 overlay 逐字节一致，链接清单
SHA256 均为
`f17949949ff89f8a6e2624c9999f4276cfbe9ab4da42cc586029b0bf373276b2`。
6 个目标均存在，`_C`、stable libtorch、MoE、cumem、FA2 和 FA3 的
SHA256 分别为：

- `1812bd980b0c50681bc853d922f5d1a70a572bcb53e599963cc05621e86aec70`；
- `e79f6ea4b1e89658ad8a74747f9af1551ca93b97d089361277134245e9bd6cea`；
- `c59dc1aaba3b60ebc42a4523fe66ecc439863878ebdd7c5530accd9c75879f49`；
- `a73a69ea63fe10a8ffe5d805e71c69453aea042cb6bea384b65706bba8628483`；
- `f8926ed5fa3a80bfdf19a2ccb2cbc1d886bd2bac2a82237eb330a761a7c19fb4`；
- `170b2341b508feaff514478cf8c2fca5a5ff6fed5d4b748c9470b1aebc8a823c`。

随后按 Phase 1→Phase 5→Phase 7→Phase 9 的依赖顺序迁移 source、
OCI、control image、overlay 和 wrapper 身份，并逐级使用上一份配置的
实算 SHA256。四份配置的新 SHA256 为：

- Phase 1：`853b337ae9c3719f97e2beaadfb8b5d9568304dcf2626b1437fd1b5252484e6a`；
- Phase 5：`df14f75b294b982a6112cbffc68b5269764b9d694115803826a97626b4dc839a`；
- Phase 7：`1032ee0b8693c52394c88b5ca9d88cdbb1ae949a05a596cc5412172aaa5a7549`；
- Phase 9：`cf61a2b8a130439ec27735775b83fe4003c36f1286a44e4ca38d37a8b30b8e12`。

Phase 9 正式身份绑定 source commit
`fd281f5f974207998a95666d4015c441c5db49ab`、candidate image/config
`sha256:2369d967545750e55e0cb1544243725364bac57f89d22cf484083ef7e0dcd692`、
manifest `sha256:e18b2252cac0127b32b8de15e9185389bfdd43c411dcb345e2f18ca7a71663ee`、
candidate layer
`sha256:9b6a02c438d6cc24dd013ca584f22663b8685ad33220c0407843271bd408d02e`
与控制镜像
`sha256:9be0cbb72088f9fe4b48680814254be9306e0cd9b1a13c4fb49fa95911db321b`。
4 个 JSON 解析、9 个正式 shell 语法、实际存在的 11 个 Phase 9 Python
文件 compile、旧 a2fe 身份清零和 `git diff --check` 均通过。

新控制镜像内的工具测试结果为：

| 测试组 | 结果 | 耗时 | 日志 SHA256 |
|---|---:|---:|---|
| Phase 7 | 20 passed、0 failed、1 warning | 26.53 秒 | `b244eec0b972e586f38bd915e9bb02a1ba10881818601d7da5fb301e1fe61d16` |
| Phase 9 | 21 passed、0 failed、2 warnings | 1.84 秒 | `b308eec8666bd32ea8abf6b692aa1f84c5a9a99b17ef0eb4e0ba68347571ab0d` |

warnings 仅为只读项目目录无法写 pytest cache，不影响测试结果。两组退出
码均为 0。工具测试准备期间曾猜测了不存在的
`/dev/shm/oscar-glm-recovery-tools/.venv/bin/python`，在 pytest 启动前即
fail-closed；读取冻结 launcher 后改用实际的 Python 3.12 路径，才得到上述
有效结果。静态 compile 也曾猜测不存在的 Phase 9 文件名；改为只对
`rg --files` 实际列出的 11 个入口执行后通过。两个错误都没有分配
GPU。

最后在正式 phase0 source Docker volume 覆盖 NFS mode 映射的控制容器
挂载命名空间内执行递归 verifier。`candidate_static_recursive_preflight.json`
状态为 `passed`，64/64 checks 全部通过、0 failed；其覆盖 Phase 1/5/7/9
派生身份、OCI descriptor、4,744 个 Git 文件、6 个 lower-layer 原生扩展链接、
rotation/runtime artifact、基线证据、冻结 evaluator 与 32K/128K 配置约束。
JSON 与 stdout 日志逐字节一致，SHA256 均为
`604f1fde53b7db2025fa8ac819acbec08890a5e402f428665af520ea5f326cc7`，
退出码为 0。首个 launcher 命令因包含预清理 `rm -f` 而在容器创建前被安全
策略拒绝；该轮没有删除文件、没有启动 verifier。改为 fail-closed
断言输出目标不存在后，上述有效轮次一次通过。

整个 overlay、配置、工具测试和递归 verifier 阶段都没有注入 NVIDIA
runtime、没有分配 GPU。本节只证明正式静态链路已完成；driver-injected
preflight 与 causal-loop 候选的新 32K/batch1 端到端 TTFT/TPOT 尚未
执行。下一步先发布本阶段配置与实时记录，再做正式 preflight 前的
两次 8/8 GPU 空闲检查。

### 2.37 Causal 有效前缀循环正式 driver-injected preflight

2.36 的正式链路、wrapper 和实时记录已由主仓库提交
`df51df6` 发布；后续 planning 状态已发布至
`078e84cb44c3ad2c282f7bf00650be6749f4df4e`。正式 preflight 绑定该主
仓库提交、源码提交
`fd281f5f974207998a95666d4015c441c5db49ab` 与 2.36 的四级配置。

新 GPU 分配前的外层空闲检查为
`2026-07-31T13:51:51Z/13:53:00Z`，间隔 69 秒；两次都是 8/8 张
苹果800 `0 MiB/0%`，没有 compute process。正式启动前在
`13:53:58Z` 做第三次即时复查，结果仍全部空闲。唯一运行的外部
下载容器不占 GPU，因此没有执行终止操作。

正式轮次为
`20260731T1354Z_stage9_candidate_fd281f5f9_preflight_v1`，退出码为
0。`static_preflight.json` 状态为 `passed`，64/64 checks 全部通过、
0 failed；其再次覆盖 source/tree、OCI、overlay/native links、rotation、
基线证据、冻结 evaluator 与 32K/128K 约束。

固定环境导入确认：

- Python/PyTorch/Triton 为 `3.12.13/2.11.0+cu129/3.6.0`；
- Transformers/Tokenizers 为 `5.8.1/0.22.2`；
- FlashInfer Python/JIT cache 为 `0.6.6/0.6.6+cu129`；
- vLLM Python 与 `_C` 均来自 2.36 的 fd281 overlay；
- 固定环境结束时 `cuda_initialized=false`。

导入时出现一条既有 RuntimeWarning：候选源码包没有生成版
`vllm._version`，因此 reported version 为 `dev`。候选 Python 与原生扩展路径、
源码 commit/tree 和递归身份已由独立门禁绑定；该 warning 与历史有效
preflight 一致，没有导致门禁放宽或失败。

服务参数解析同样记录 `cuda_initialized=false`，实际值为 TP=8、
PP=1、`TRITON_MLA_SPARSE`、`oscar_mla_int2`、
`max_model_len=131072`、`max_num_batched_tokens=2048`、
`max_num_seqs=16`、`gpu_memory_utilization=0.92`、eager、chunked prefill 开启、
prefix caching 与 async scheduling 关闭、seed 42 和 torch profiler。

外层空闲检查、preflight 日志、退出码、静态 JSON、固定环境 JSON、
服务参数 JSON 和退出后 GPU 快照的 SHA256 分别为：

- `c16861267ab8d4f8ed9e2ffbe76922bf957ee23adcdb4e32631a3f6f51c340c9`；
- `2ad73636f5fc5e701ed3909e6be12bb18e7c890da01a068c07b12be088c3b89a`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `8ddf37fde510f9a1de2cf887e7571ee405fae171c741e34945c3eeeeb7056981`；
- `85994ced55276751a9f9c4bc569b1e61f52d0bee172733aa49a6a4b5a41d87ca`；
- `ba4b022eef58d405bbe253b73570ca9d1cadbbe596f16f7606954029131f7c61`；
- `c7d9e9be1d8c3acbfa5cd6e409448fe05dba78de83a81f8c621c10c8891feef2`。

preflight 容器已自动删除，`13:56:42Z` 复查 8 张 GPU 均为
0 MiB/0%、没有 compute process。该轮只注入 NVIDIA driver 用户态库，
没有加载模型或执行 kernel；静态、固定环境和参数门禁现已全部关闭。
当前仍没有 causal-loop 候选的新 32K/batch1 TTFT、TPOT 或吞吐结果。
下一步先发布本节实时记录，再为正式 32K/batch1 轮次重新完成两次
8/8 GPU 空闲检查；发布前不启动端到端性能实验。

### 2.38 Causal 有效前缀循环的 32K/batch1 正式结果

2.37 的 preflight 记录已由主仓库提交 `3d51cc1` 发布；启动前
planning 状态也已发布，两个仓库均保持 clean/published。正式单格轮次
`20260731T1403Z_stage9_candidate_fd281f5f9_32k_b1_v1` 绑定：

- 主仓库提交
  `aa51f9cfcb77c1a7939e18cad67b58b5046934ee`；
- 源码提交/tree：
  `fd281f5f974207998a95666d4015c441c5db49ab` /
  `86185b214eb3d6f25108076a0a2c2c8dabb3d122`；
- Phase 9 配置 SHA256：
  `cf61a2b8a130439ec27735775b83fe4003c36f1286a44e4ca38d37a8b30b8e12`；
- 32,768 输入 token、128 输出 token、batch/并发 1、1 次 warm-up、
  3 轮正式测量和 8+8+1 profiler。

新 GPU 分配前的外层空闲检查为
`14:01:00Z/14:02:08Z`，间隔 68 秒；启动前 `14:03:55Z` 的第三次
即时复查也为 8/8 张卡 `0 MiB/0%`、无 compute process。容器内部
`14:05:02Z/14:06:06Z` 的两次检查同样为 8/8 空闲。唯一运行的
外部下载容器不占 GPU，因此没有执行终止操作。

141/141 个权重分片全部加载；服务于 `14:17:45Z` ready，记录的
startup 为 660 秒。运行器在 10、20、30 分钟分别输出服务心跳，
profiler 汇总超过 10 分钟时也额外输出
`label=input_32768_batch_1/profile elapsed_seconds=600`，满足长实验
进度记录要求。

三轮均为 3/3 completed、0 failed：

| 轮次 | TTFT（ms） | TPOT（ms） | 吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 35,685.089 | 198.756 | 0.016413 |
| 2 | 35,683.893 | 197.826 | 0.016445 |
| 3 | 35,672.591 | 196.706 | 0.016487 |

三轮 `mean` 指标中位数与 2.30 的 a2fe 正式结果、同格 BF16 对比
如下：

| 指标 | BF16 | a2fe OSCAR | Causal-loop OSCAR | 相对 a2fe | 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| TTFT（ms） | 12,528.026 | 36,245.415 | 35,683.893 | **-1.55%** | +184.83% |
| TPOT（ms） | 178.832 | 199.205 | 197.826 | **-0.69%** | +10.62% |
| 请求吞吐（req/s） | 0.028377 | 0.016248 | 0.016445 | **+1.22%** | -42.05% |

三轮的 TTFT、TPOT 和吞吐相对极差分别为
`0.0350%/1.0362%/0.4490%`。相对 a2fe，当前 TTFT 只减少
`561.522 ms`；相对 2.33 的单层 `-32.597960%` 降幅，端到端收益
明显更小。这表明单层微基准不能直接外推为整个 32K TTFT 收益。

TPOT 仍低于 BF16 `+20%` 上限 `214.598 ms`，但 TTFT 为 BF16
的约 `2.85×`，高于上限 `15,033.631 ms`；请求吞吐也仍低
`42.05%`。因此本轮虽然比 a2fe 进一步改善，但没有关闭 TTFT 的 20%
门限，性能优化尚未完成。

单格和总 summary 状态均为 `passed`。三轮测量峰值显存为每卡
`80,679 MiB`，profile 峰值为每卡 `80,691 MiB`；最多运行 1 个请求、
等待为 0、preemption 为 0，KV usage 峰值为 `5.7902%`，没有容量
排队或抢占。

Profiler 校验状态为 `passed`，耗时 `750.0379951000214 秒`；
8 份 worker trace、8 份 CUDA table 和 1 份 frontend trace 均通过
rank、bytes 与 SHA256 检查。critical rank 为 6，kernel total 为
`65,795 ms`。8-rank table 中 `_mixed_sparse_prefill_stage1` 均精确执行
`1,248=16×78` 次，CUDA total 中位数为 `23,134.5 ms`，即约
`18.537 ms/层/chunk`；相对 a2fe table 的 `23,688.5 ms` 只下降约
`2.34%`。该 table 趋势与端到端改善同向，但冻结 trace 的精确
prefill wall/kernel 归因尚未执行，本节不将 table 进一步外推为
完整 TTFT 因果分解。

总 summary、单格 summary、外层正式日志、外层空闲检查、外层
退出码和小型证据清单的 SHA256 分别为：

- `d2bb22c74e443ae7db8abf6d0c46c7d2cfc85160deefdfa13e7d975511fc3c5b`；
- `cf9d8653c15f7a2e95ab3545ef88251b2f70b82311e7fb9d9bc29f7a5140be56`；
- `74550a93ce8bff819758c9553c5bbcd36e45ec262f163f6580bffbfa566875aa`；
- `eba1d33c306bffc0b7f1e8bfc5d55cc38c0cf24b97c72602b25a09cfcef4128b`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `aefe596a288825959251bc379dcfca760deb309fab7f1de80a023fcabc4cb96d`。

小型证据已从 `/dev/shm` 逐文件复制到
`artifacts/phase9-control/20260731T1319Z_runtime_fd281f5f9_v1/formal_32k_b1_results`，
40 个文件、共 1,114,300 bytes，复制前后 SHA256 全部一致。大型
trace 没有重复复制，但 profile validation/summary 已逐 rank 冻结原始
trace 的路径、bytes 和 SHA256。

正式运行器退出码为 0，实验容器已自动删除；退出后 8 张 GPU
均无 compute process。下一步先发布本轮实时记录，再对已冻结的
8-rank trace 执行 CPU-only 多 chunk 归因，据此选择下一项最小优化。

### 2.39 Causal 有效前缀循环 32K 多 chunk trace 归因

2.38 的正式结果已由主仓库提交 `7001b05` 发布，发布状态由后续提交
`f33b184` 固化。本阶段只在固定控制镜像
`oscar-glm-stage9-runtime:fd281f5f9` 中解析 2.38 已冻结的 8 份 worker
trace，没有重新运行模型。有效 analysis ID 为
`20260731T1457Z_causal_loop_32k_prefill_trace_v1`；固定环境为 Python
`3.12.13`、`ijson 3.4.0.post0`、4 个 CPU worker 和 top-40 汇总，显式
设置空的 `CUDA_VISIBLE_DEVICES`。分析器 SHA256 为
`8b6b2393f93be3783f47ccbe3ecb26020cc2b65526c99fcb464c4768ce4330f7`。

轮次退出码为 0，summary 状态为 `passed`，耗时
`110.78364903014153 秒`；8/8 ranks 均解析到 144 个 execute context、
16 个 prefill chunk 和精确 32,768 个输入 token。同口径 8-rank 中位数
对比如下：

| Trace 指标 | BF16 | a2fe OSCAR | Causal-loop OSCAR | 相对 a2fe | 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| Prefill wall（ms） | 10,086.470 | 36,257.407 | 35,731.482 | **-1.45%** | +254.25% |
| Prefill kernel 合计（ms） | 9,533.580 | 35,316.438 | 34,779.955 | **-1.52%** | +264.82% |
| 各 rank generation 中位数再取中位（ms） | 223.325 | 269.448 | 266.230 | **-1.19%** | +19.21% |

当前 `_mixed_sparse_prefill_stage1` 的 8-rank 中位 CUDA total 为
`23,134.871 ms`，共 `1,248=16×78` 次，占 prefill wall 的
`64.75%`；相对 a2fe 的 `23,688.690 ms` 减少 `553.819 ms`，即
`-2.34%`。a2fe 到当前候选的 prefill wall 共减少 `525.925 ms`，stage1
减少量解释 wall 改善的 `105.30%`。去掉 stage1 后的剩余 wall 从
`12,568.716 ms` 增至 `12,596.610 ms`，即约 `+0.22%`；rotation、MoE、
NCCL、GEMM 和 FP8 indexer 分别为 `3,391.413/1,984.514/1,140.270/`
`915.641/882.474 ms`，与 a2fe 的
`3,391.580/1,984.988/1,128.943/915.454/882.513 ms` 基本同量级。
因此 2.38 的改善全部可以由 stage1 解释，其他 prefill 工作没有同步加速。

单层结果与端到端收益差距的原因也已闭合。32K 输入被拆为 16 个 2,048-token
chunk；`effective_topk=min(2048, causal_seq_len)` 只会缩短第一个 chunk
中 query 的无效尾部。第 2–16 个 chunk 的最小 `causal_seq_len` 已不低于
2,049，因此所有 query 的 `effective_topk` 仍为 2,048。也就是说，只有
第一个 chunk 的 78 次 stage1 调用受益，占全部 1,248 次调用的 `6.25%`；
2.33 的 2,048×2,048 单层微基准恰好只代表这个首 chunk。把该微基准的
`32.597960%` 降幅粗略除以 16，得到约 `2.04%` 的全 32K stage1 预期，
与 trace 实测 `2.34%` 同量级。这解释了为何端到端 TTFT 最终只改善
`1.55%`，而不是接近单层的 `32.60%`。

相对 BF16，当前 stage1 超出 BF16 原生 prefill attention 的时间仍解释
总 prefill wall 差距的 `77.01%`。因此下一候选仍应聚焦 grouped prefill
stage1，但必须减少所有 16 个 chunk 都会执行的有效 top-k 计算或访存，不能
继续只跳过首 chunk 的 causal 无效尾部，也不能把优化范围转移到已由数据排除
的 generation 或其他 kernel。

首次只读结构化对比脚本只匹配 OSCAR 的 stage1 符号，读取 BF16 原生
attention 时触发 `StopIteration`；已改为同时识别 OSCAR 与 BF16 符号后
重新计算，冻结输入和有效分析结果均未被修改。有效 summary、run log、
exit code 和 trace input manifest SHA256 分别为：

- `16a97002c441d5324d187424c39f9dca19d48c158cb1714c888031f7ce7988e1`；
- `fd333686b064b26158eb3c80d78fb436d7d4c0e8d5ab67a0a88a33f1d231ab1c`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `57a3860d20872530825b7030239623b2e6feee38b0cd229643106c8dccf9af53`。

5 份小型证据、共 343,360 bytes，已逐字节复制到
`artifacts/phase9-control/20260731T1319Z_runtime_fd281f5f9_v1/formal_32k_b1_trace_analysis`，
复制前后哈希一致；证据清单 SHA256 为
`326754811f7a0e440cfbb8a69fcd5e24732f811a0dce96897ab478737e17199b`。
本阶段没有注入 NVIDIA runtime 或分配 GPU，结束后 8 张 GPU 均无 compute
process。下一步先发布本节实时记录，再依据上述约束筛选能够覆盖全部 16 个
chunk 的最小 stage1 优化。

### 2.40 Grouped prefill 无 BF16 token tile 计算门禁候选

根据 2.39 的归因，下一候选必须减少全部 16 个 prefill chunk 都会执行的
stage1 工作。只读源码审查发现：grouped prefill stage1 对每个 token tile
先用 mask 将非 prefix/recent token 的 `bf16_values` 置零，之后仍无条件执行
BF16 score dot 和 BF16 value contribution dot。对于只含 history token 的
tile，这两个 dot 的数学贡献均严格为零；online softmax 仍只需保留
`bf16_acc * previous_scale` 的既有缩放。

本阶段据此实现了最小 tile 级门禁：以 `is_bf16=is_prefix|is_recent` 计算
标量 `has_bf16`，仅当 tile 至少含一个 BF16 token 时执行上述两个 dot；
无 BF16 token 时 score 从零开始、value contribution 取零，但每个 tile
仍照常更新 accumulator 的 `previous_scale`。源码提交为
`ca4a404e913ce55237ca60383cc86e221fbfea26`，tree 为
`079815219a02add3f37318ed434924e80f80a35d`，已经推送到源码远端。
生产源码与定向测试文件 SHA256 分别为：

- `23b08ffae200cfefa3e7a2190c436c0f517bfc509e8479bb22245230b10bc70c`；
- `92a8340ed5a528232a1685a4deb09288efc86792e0002d6289b00877e8d2da0b`。

该候选没有修改 selected index 顺序、top-k/indexer、history/RoPE、数值精度、
launch 几何、decode 路径或三段式 cache。它也没有启用既有的 index sort
环境变量，因此后续性能变化可以归因于 stage1 内两个零贡献 BF16 dot 的
运行时跳过，而不会混入排序开销。

实现采用先红后绿的定向门禁。新增 source-invariant 用例在修改生产源码前
按预期为 1 failed。第一次生产 patch 的通用上下文误命中 decode；diff 审计
在重跑绿灯前发现该问题，随后恢复 decode 原逻辑，并以 prefill 独有张量布局
重新放置 gate。有效 CPU 轮次
`20260731T1530Z_bf16_tile_gate_cpu_v1` 在固定 Python 3.12.13 环境中得到
8 passed、19 个 CUDA 用例显式 skipped、3 warnings、0 failed，耗时
16.15 秒，退出码为 0。Ruff 0.14.0 check/format、固定 Python compile、
diff check 和全部适用的源码提交 hooks 也均通过。该轮显式设置空的
`CUDA_VISIBLE_DEVICES`，没有注入 NVIDIA runtime。

CPU-only SM80 离线编译轮次
`20260731T1521Z_bf16_tile_gate_offline_v1` 状态为 `passed`。固定
h8/t16/w8、1 stage 几何成功生成 206,640-byte cubin，cubin SHA256 为
`2a9e402ddbdff825cf655daa728c7b36d12948c8cc7fb2a122e9a36eb2540282`；
编译元数据中的动态 shared memory 为 109,568 bytes，低于苹果800
166,912-byte 上限，且与 fd281f5f9 候选相同。离线 `cuobjdump` 记录为
255 registers、0-byte stack。离线 summary、run log 和资源日志 SHA256
分别为：

- `a092f126add6fd638b343ecc0d0dbcd0acbb987fedf1e77d881094459b8a8ee2`；
- `54712c605c21aa8e37963d96885ba9b57016b332a67693d1aa07d46eee4c0e8d`；
- `c4b61cceb4235a6d12e2428a324e0196811b9aea6b96f30f23b94fd318674000`。

CPU pytest 日志 SHA256 为
`e55abd1e6654dfaed142c3397d88935e98b38fe0df87d6b1cf207f02b5cbfcca`。
10 份证据文件及其清单已复制到
`artifacts/phase9-control/20260731T1319Z_runtime_fd281f5f9_v1/bf16_tile_gate_cpu_offline_v1`，
目录总计 14,353 bytes；清单 SHA256 为
`50535adca593513a0eb226614ce5413fdaa081645f48f4b11329a2e9dd361e95`。

本节只关闭源码语义、CPU/interpreter 回归和离线编译资源门禁。当前尚未获得
苹果800上的 output/LSE 正确性、kernel CUDA 时间或 32K/batch1
TTFT/TPOT/吞吐结果，因此不能宣称性能改善。下一步先发布本节、主仓库
submodule 指针和 planning 状态；两仓 clean/published 后，再按既有协议
执行两次至少间隔 60 秒的 GPU 空闲检查，使用同一冻结几何筛选该候选。

### 2.41 32K 后续 chunk 单层筛选协议

2.40 发布后，主仓库状态由提交 `1ade59d` 固化，两仓均为
clean/published。GPU 分配前对 2.33 沿用的单层入口做只读复核，发现既有
`--seq-len 2048` 同时控制 query 数和最终序列长度：它只生成 positions
`[0,2048)`、64/1,728/256 的 prefix/history/recent，与首个 2K chunk
一致。2.39 已证明首 chunk 不足以代表后 15 个 chunk，因此不能直接复用该
几何判断 2.40 的 tile gate 收益。

本阶段只扩展性能工具，没有修改 OSCAR 生产源码。新增独立参数
`--final-seq-len`，默认值仍等于 `--seq-len`，所以原有 1K/2K 调用语义不变。
筛选 32K 末段时固定使用：

- `--seq-len 2048 --final-seq-len 32768`；
- query positions 为 `[30720,32768)`；
- final sequence/cache metadata 为 32,768 token；
- top-k width 仍为 2,048，只保留 split16 参考与 grouped split1 候选；
- prefix/recent 仍为 64/256，history 为 32,448 token；
- seed 42，后续 GPU 轮次仍使用 5 次 warm-up、7 次正式测量和每次
  1 iteration。

selected index 生成器使用最终 32K 范围的固定随机 permutation；每个 query
只保留 causal 前缀中的前 2,048 个 index。结果 JSON 同时新增 tile coverage，
显式记录有效/BF16 selected token 数、含/不含 BF16 的 tile 数和全 history
tile 数，避免只凭负载名称推断 gate 覆盖率。脚本与测试 SHA256 分别为：

- `27f9d5e7a084c8b23c4bb80aead140b58790531ff67fa51d99aadaa7c2d4ca90`；
- `857c17410b0d5965b8324ab40386191acbb04d744f96967f5395b03ba6110d47`。

TDD 首先新增 later-chunk 用例，旧入口因不接受 `final_seq_len` 按预期得到
1 error。实现后定向 unittest 为 4/4 passed；固定控制容器内
`test_benchmark_oscar_prefill.py`、`test_phase9_tools.py` 和
`test_analyze_prefill_trace.py` 合计 22/22 passed。Ruff 0.14.0 check、
format、固定 Python 3.12.13 compile、CLI help、非法
`final_seq_len<seq_len` 边界和 `git diff --check` 均通过。首轮完整工具
门禁已得到 22/22 passed 和 Ruff check 通过，但 format check 发现主脚本
需要机械格式化，因此该轮整体退出码为 1；用同一 Ruff 只格式化主脚本后才得到
上述有效绿色结果。

有效 CPU-only coverage 轮次为
`20260731T1622Z_later_chunk_coverage_cpu_v2`。实际生成 2,048 行、每行
2,048 个 selected index，共 4,194,304 个；逐行检查确认全部有效、唯一且
不超过对应 query position。16-token tile 覆盖统计为：

| Coverage 指标 | 实际值 |
|---|---:|
| Selected token 总数 | 4,194,304 |
| 其中 BF16 selected token | 4,696 |
| Tile 总数 | 262,144 |
| 至少含一个 BF16 token 的 tile | 4,518 |
| 全 history、无 BF16 token 的 tile | 257,626（98.2765%） |

首轮 coverage 命令遗漏 `docker run -i`，容器内 `python -` 从空 stdin
正常退出，日志为空，不能记为通过；v2 增加 stdin 透传并强制断言日志非空、
`status=passed` 后得到上述有效结果。整个工具和 coverage 阶段均显式设置
空的 `CUDA_VISIBLE_DEVICES`，没有注入 NVIDIA runtime 或分配 GPU。

8 份红灯/绿灯/工具/coverage 证据及其清单已复制到
`artifacts/phase9-control/20260731T1319Z_runtime_fd281f5f9_v1/later_chunk_benchmark_cpu_v1`，
目录共 2,614 bytes。有效 coverage log、工具日志和证据清单 SHA256 分别为：

- `46dcd18df3d1da69520bae2c2e68df2a029974b51e3e3f628b0e6ccd9c3f93ed`；
- `050b9acd687255704d450c412c767859f81fce0cf81c215d31acc18995152c72`；
- `9cea5c83d4adb30cc85e4823b3ebcd81bffef3b558bbd5f3bb3ad6aa96b5d7f7`。

`98.2765%` 只描述 seed 42 的合成随机 selected 分布，不等于正式 DSA top-k
分布，也不能外推为 kernel 或端到端加速。本节尚无苹果800 output/LSE、
CUDA 时间或 32K/batch1 TTFT/TPOT。下一步先发布该可复现筛选入口；主仓库
恢复 clean/published 后，再执行新的双 GPU 空闲检查，并分别在 fd281f5f9
与 ca4a404e9 源码上运行完全相同的末段负载。

### 2.42 BF16 tile gate 32K 后续 chunk 单层筛选

2.41 的筛选入口由主仓库提交 `6469cbf` 发布，状态由
`bd2a67f` 固化。GPU 对照前的外层空闲检查时间为
`2026-07-31T15:43:02Z` 和 `2026-07-31T15:44:36Z`，间隔 94 秒；
两次均为 8/8 张苹果800 `0 MiB/0%`，且没有 compute process。
唯一运行的项目外下载容器没有 GPU DeviceRequests，因此未终止该容器。
空闲状态由主仓库提交 `8c4eb2a` 发布；启动前的即时复查仍为
8/8 张卡空闲。

有效 GPU 轮次为
`20260731T1637Z_later_chunk_tile_gate_compare_v1`。它在同一个只分配
GPU 0 的 Docker 容器中，先后运行以下两份源码，并为两者使用相互独立的
cold Triton cache：

- 控制：`fd281f5f974207998a95666d4015c441c5db49ab`，tree
  `86185b214eb3d6f25108076a0a2c2c8dabb3d122`，kernel SHA256
  `e8b1baabc43b080e0992dab8901ff4de347fb68c905793f906934effdacb11ca`；
- 候选：`ca4a404e913ce55237ca60383cc86e221fbfea26`，tree
  `079815219a02add3f37318ed434924e80f80a35d`，kernel SHA256
  `23b08ffae200cfefa3e7a2190c436c0f517bfc509e8479bb22245230b10bc70c`。

两边均使用 2.41 冻结的 2,048-query、32K final sequence、positions
`[30720,32768)`、2,048 top-k、seed 42、5 次 warm-up 加 7 次正式测量
协议，只比较 grouped split1 和 split16 参考。CPU coverage 在此负载上实测为
257,626/262,144 个 tile 全为 history，占 `98.2765%`；但 selected index
仍是确定性随机分布，不是正式 DSA 的抓取结果。

两份源码的 result 都为 `passed`，容器总退出码为 0。实测中位数如下：

| 指标 | fd281f5f9 控制 | ca4a404e9 候选 | 候选相对控制 |
|---|---:|---:|---:|
| grouped split1 CUDA | 22.618113 ms | 20.226048 ms | -10.575883%（1.118267×） |
| grouped split1 wall | 22.670865 ms | 20.256273 ms | -10.650641%（1.119202×） |
| split16 CUDA | 338.324493 ms | 338.262024 ms | -0.018464% |

控制的 7 个 split1 CUDA 样本中有一个 `25.821184 ms` 高值，其余样本约为
`22.54–22.68 ms`，因此中位数未被该高值直接决定。候选的 7 个样本为
`20.185087–20.427776 ms`。这些样本支持“末段合成单层负载的中位数改善”，
但不构成统计显著性或端到端收益结论。

每份源码内的 grouped split1 均与同源 split16 参考执行冻结的
`torch.allclose(atol=0.002, rtol=0.002)`，两边都通过。两份诊断字典逐字段
完全相同：output max_abs/max_rel 为
`0.0033702850341796875/132.1691436767578`，LSE max_abs/max_rel 为
`0.0022249221801757812/0.00023601796419825405`。output max_abs 大于
`0.002` 不与 allclose 通过矛盾，因为判定同时包含逐元素的相对容差。
需要明确的边界是：这里的参考是各自源码内的 split16，不是直接对比
fd281f5f9 与 ca4a404e9 的输出 tensor；跨源码的完整 CUDA 回归仍需单独执行。

runtime cubin 审计显示，控制到候选的 registers 从 247 降为 242，
stack 均为 0 byte，dynamic shared memory 均为 109,568 bytes。控制/候选
cubin 大小为 137,264/142,000 bytes，SHA256 分别为
`2659bf421be3c256ed876d4636723ac5d539f4cc5f9934f357496612259f6e0a`/
`17a5c9ed155224d5e4ca1880fa0a90e825306ff363d75277ddc9b1fc3a38d5f5`。

17 份 GPU/结果/资源证据及其清单已复制到
`artifacts/phase9-control/20260731T1319Z_runtime_fd281f5f9_v1/later_chunk_tile_gate_gpu_v1`，
目录共 306,881 bytes。comparison summary 与证据清单 SHA256 分别为：

- `95cf66b18ac8e4bc7f00f44ceee99b4985feb333487ebefc97f3e9306181cc53`；
- `b5fa7e93c2046ee704da6a199cc91e649b92b5301edd9a02f20f5d408ea56783`。

控制/候选 result JSON SHA256 分别为
`ae7152bf91b6209e2c38e5518a1c2d54b6f065b3a0e11eff070cda7e5bca3523`/
`60a3b67fa0d9012a604d2a62ef982a542a31a3c0177dde5cd67ad89f491f7d0c`。
轮次结束后容器已删除，退出后证据检查为 8/8 张 GPU 空闲、无
compute process。

本阶段只证明 BF16 tile gate 在 32K 后续 chunk 的合成单层路径上有
`10.575883%` 的 CUDA 中位数改善。它尚未通过完整 cold-cache CUDA
回归，也没有新的 32K/batch1 TTFT、TPOT 或吞吐结果，因此不能宣称
端到端已改善。下一步先发布本节实时记录，再重新执行两次间隔至少
60 秒的 GPU 空闲检查，以独立 cold Triton cache 运行候选的完整 CUDA 回归。

### 2.43 BF16 tile gate 完整 cold-cache CUDA 回归

2.42 与 planning 由主仓库提交 `0a8a559` 发布，状态由
`7dbb455` 固化，两仓恢复 clean/published。完整 CUDA 回归前重新执行
空闲检查：`2026-07-31T16:00:27Z` 和 `16:01:33Z`，间隔 66 秒；
两次都是 8/8 张苹果800 `0 MiB/0%`，没有 compute process。唯一
运行的项目外下载容器的 GPU DeviceRequests 为 null，所以无需终止。
该空闲状态由主仓库提交 `14a55ba` 发布后才进入正式轮次。

有效轮次为
`20260731T1604Z_bf16_tile_gate_full_cuda_v1`，绑定：

- 主仓库提交：`14a55ba822709dc07afd0a536ff1638e4f5c5a4a`；
- 源码提交/tree：
  `ca4a404e913ce55237ca60383cc86e221fbfea26` /
  `079815219a02add3f37318ed434924e80f80a35d`；
- 固定控制镜像：`oscar-glm-stage9-runtime:a2fe02055`，image ID
  `sha256:0e13b724a2b89f3d698a3a130f13f27d8f8ef1c3acf96fbf38920306f79d50d5`；
- 源码和 phase0 native rootfs 只读挂载，只分配 GPU 0；
- 显式设置 `VLLM_OSCAR_RUN_CUDA_TESTS=1`，使用独立空 Triton cache，
  完整执行 `tests/oscar_mla`。

启动前 `16:04:55Z` 第三次即时检查仍为 8/8 张卡空闲。容器内
使用固定 Python `3.12.13`、`pytest==8.3.5` 和 `tblib==3.1.0`。
有效结果为：

- 127 passed、0 skipped、0 failed；
- 19 warnings、87.91 秒；
- Docker 退出码 0；
- cold Triton cache 为 380 个文件，文件内容合计 25,038,227 bytes。

相对 2.34 的 fd281f5f9 完整回归 126 passed，本轮多出的 1 项是
ca4a404e9 在 2.40 新增的
`test_grouped_prefill_skips_zero_bf16_tile_dots` source-invariant 用例。Git diff
的 `tests/` 范围只在 `tests/oscar_mla/test_triton_decode.py` 新增该用例的
6 行；旧用例没有
被跳过或删除。19 条 warning 由 2 条既有 Swig deprecation、1 条源码树
`vllm._version` RuntimeWarning、14 条 PyTorch JIT deprecation 和 2 条只读源码目录
pytest cache warning 组成，不改变 127 项通过结论。

容器于 `16:06:44Z` 自动删除；退出后复查为 8/8 张 GPU
`0 MiB/0%`，没有 compute process。两次外层空闲检查、启动前检查、
pytest、退出码、cache summary、380 行 cache 文件哈希和退出后 GPU 状态
共 8 份证据，连同清单已复制到：

`artifacts/phase9-control/20260731T1319Z_runtime_fd281f5f9_v1/bf16_tile_gate_full_cuda_v1`。

证据目录共 94,565 bytes。主要 SHA256 为：

- pytest log：
  `3e57bee7830b8c06f3c3e186df780e9a4e85f63e48032cda920308f88171cbfd`；
- 退出码：
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- cache summary：
  `0dc79e89747cd12f0599c92501d8fa827532a1d7f8fd5db11801a4dd04c9a615`；
- 380 行 cache 文件哈希：
  `58d9daffd5446c82b592f8c2a496f4df1ce04a9753a39e1a0dfa2d427f832f48`；
- 证据清单：
  `d55a7235ac758731bc8a7b6fae7a1682c6bca78d69a491873c61d35f226ea0c3`。

双空闲检查、启动前 GPU 和退出后 GPU 日志 SHA256 依次为：

- `89d85415543e0d39530027c93cdeb2c6124492131e05e7bbdbac0323372ab3de`；
- `06f3dcda40918156224ed4a2ca7e697d265552984290278696cad3f9c036d859`；
- `89dcf57eb77b40474faa3b5c58882dfda1b8728ff7e50d464afa733acdb66913`；
- `0c21a5a3a8b185d9f26c44d1fff80cff5801b63a3af0dcd35e92612a726e53f8`。

至此 BF16 tile gate 已通过源码语义、CPU/interpreter、离线资源、32K 后续
chunk 单层苹果800正确性/性能和完整 cold-cache CUDA 正确性门禁。
本轮不是 32K/batch1 端到端实验，因此仍没有新的 TTFT、TPOT 或吞吐
结果，也不能把 2.42 的单层收益写成端到端收益。下一步先发布本节
实时记录，再进行 Phase 6 候选输入迁移与确定性 OCI 双构建；发布前不进入
后续阶段。

### 2.44 BF16 tile gate Phase 6 候选输入迁移

2.43 的完整 CUDA 回归与实时记录已由主仓库提交 `b6407f0` 发布，
发布状态由 `15a3076` 固化；迁移开始前主仓库和源码仓库均为
clean/published。本阶段只把 Phase 6 候选输入从 fd281f5f9 切换到已经通过
2.40–2.43 门禁的 BF16 tile gate 源码：

- source commit：
  `ca4a404e913ce55237ca60383cc86e221fbfea26`；
- source tree：
  `079815219a02add3f37318ed434924e80f80a35d`；
- output tag：
  `glm52-oscar-a800-phase6-ca4a404e9-0275043c`；
- Phase 6 Dockerfile SHA256：
  `51ed571f615559a0008631c17244d88878ef0c8437c3c3161a61d1e4c022e6f4`；
- `candidate_inputs.json` SHA256：
  `5f3fb384f5591011db8a4c8f511cb7ea79d2c14b1d7c8186e32d3151bcb8b802`。

迁移只修改 source commit/tree、output tag 和由此实算得到的 Dockerfile
哈希。base manifest、rotation artifact 及其 4 项哈希、runtime expectation、
7 个预期原生扩展和 Phase 6 构建/验收脚本均未修改。配置与 Dockerfile 中
旧 fd281f5f9 commit、tree 和 tag 的计数均为 0；源码本地 HEAD、源码远端
分支和配置中的 commit 三者一致，Git tree 也与配置一致。

首轮 CPU-only 输入门禁目录为
`20260731T1614Z_phase6_ca4a404e9_input_validation_v1`。静态身份检查通过，
PAX 确定性测试也得到 1 passed、1 个只读 pytest cache warning、0.54 秒；
但随后 Python compile 尝试向只读源码目录
`scripts/phase6/__pycache__` 写入字节码并收到 `Errno 30`。因此该组合命令
没有完整通过，不能作为有效阶段结果。该轮 static log 与组合日志 SHA256
分别为：

- `a11a10876f81bf73070404eb4fa32e85ec8a15096efd540a6278899bc4bb6dfc`；
- `4a61458a2042d36384d16116ee6cd8ff921fcfa1da66b7f2e93e6cc8906c9bcd`。

有效重试目录为
`20260731T1614Z_phase6_ca4a404e9_input_validation_v2`。该轮保持源码只读，
使用固定 Python `3.12.13`，并将 `PYTHONPYCACHEPREFIX` 显式重定向到
`/tmp/pycache`。结果为：

- JSON 解析、Dockerfile/config 哈希、source commit/tree、output tag、
  旧身份清零和源码本地/远端一致性检查全部通过；
- PAX 确定性测试为 1 passed、1 个只读 pytest cache warning、0.12 秒；
- `build_candidate_oci.py`、`verify_candidate_oci.py` 和
  `test_build_candidate_oci.py` 三份 Python 文件 compile 全部通过；
- `git diff --check` 通过，组合退出码为 0。

有效 static log、PAX+compile log 与退出码文件 SHA256 分别为：

- `a11a10876f81bf73070404eb4fa32e85ec8a15096efd540a6278899bc4bb6dfc`；
- `7e00675914d074cd2ddf8fed6091582737a58349a974fdc442dd8e1e44fc2803`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

两轮均为 CPU-only，没有注入 NVIDIA runtime 或分配 GPU；检查期间 8 张
GPU 均为 0 MiB/0%，没有 compute process。本节只证明 Phase 6 的新输入和
确定性构建入口已通过静态门禁；尚未执行新的 OCI 构建，因此当前没有新的
OCI descriptor、candidate layer、daemon image，也没有 BF16 tile gate 的
32K/batch1 端到端 TTFT、TPOT 或吞吐结果。下一步先发布本节、配置和
Dockerfile；主仓库恢复 clean/published 后，再执行两轮相互独立的
CPU-only 确定性 OCI 构建与递归验收。

### 2.45 BF16 tile gate 候选 OCI 双构建与递归验收

2.44、Phase 6 输入与 Dockerfile 已由主仓库提交 `323671c` 发布，发布状态
由 `a1720a0` 固化；双构建启动记录随后由 `5eb077f` 发布。首次并行轮次
使用两个独立目录：

- `20260731T1628Z_candidate_ca4a404e9_bf16_tile_gate_v1`；
- `20260731T1628Z_candidate_ca4a404e9_bf16_tile_gate_v2_rebuild`。

两轮都在 build 的首个 `git status` 处 fail-closed。项目与源码 worktree
实际属于 UID 0，固定容器按宿主 UID 22633 运行，而容器内没有宿主的
`safe.directory` 配置，因此 Git 以 dubious ownership 退出 128。并行组合
运行器在 31 秒结束，两轮组合退出码均为 1，均未创建 OCI layout。两份失败
日志 SHA256 相同，为：

`b52115e991a6bfafbd0ac1f3afa1ebabf267a6713f4760389c060a07513e3f6e`。

两份退出码文件 SHA256 也相同，为：

`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`。

失败证据由主仓库提交 `0051472` 发布后，以进程级 `GIT_CONFIG_*` 只为
`/workspace` 和 `/workspace/glm52_oscar_vllm` 声明精确
safe.directory；没有写全局 Git 配置，也没有修改仓库所有权。只读预检确认
两仓在固定容器内均为 clean 且 HEAD 等于 upstream。有效重试使用两个新的
独立目录：

- v3：
  `artifacts/phase6/20260731T1628Z_candidate_ca4a404e9_bf16_tile_gate_v3`；
- v4 重建：
  `artifacts/phase6/20260731T1628Z_candidate_ca4a404e9_bf16_tile_gate_v4_rebuild`。

两轮绑定主仓库提交
`00514720643346092066cf29acdbece87651500c`、源码提交/tree
`ca4a404e913ce55237ca60383cc86e221fbfea26` /
`079815219a02add3f37318ed434924e80f80a35d` 和 2.44 的输入配置。并行组合
运行器在 1,171 秒结束，v3/v4 退出码均为 0；第 600 秒实际打印心跳，记录
两份日志大小为 2,544/2,552 bytes。两轮 build 状态均为 `built`，递归
verification 状态均为 `passed`，共同得到：

- image/config：
  `sha256:7c85cdd01bdc18d286aaabccd442967be660e59fc964f334f3fc30b1cd5a4eb8`；
- manifest：
  `sha256:fb8e914ca146adddaa1eca23134cc55f72243a1abe3eb79a64faa38b71c44a23`；
- candidate layer：
  `sha256:3f03376d01935fc9e057a34a01b6e701737a5384b19281a8bc204cb7ddbae203`；
- diff-ID：
  `sha256:5f8875b9e7e5c465e2c935f1518355782a7736e74e2d1dc71624900d08a67a14`；
- candidate layer size/member：109,147,892 bytes / 5,298；
- `index.json` SHA256：
  `183104650b765f8270909c7aa6df51f671b08b01850cc265dbd7d8454a8e4d90`。

v3/v4 的 `index.json`、config blob、manifest blob 和 candidate layer
blob 已逐字节比较为完全相同。对应字节数依次为 284、32,352、5,662 和
109,147,892 bytes。两次递归验收各自核对 4,744 个源码文件、4 份 rotation
artifact、7 个基础层原生扩展和精确 Git tree；33 层中的前 32 层与 base
逐层完全匹配，candidate layer 不含原生扩展或 whiteout。

v3 的 build/verification JSON、组合日志和退出码文件 SHA256 分别为：

- `c885f1b12f2e3e929a54e7b4d5a16f82e25eb6a93e1695b8f07ae80b874b3dbe`；
- `472f09c8f55bb6259469e694e2dc75531882cbd6f0a06d530397e251ecbb90fd`；
- `fa9d00f1418960b8c8891ebee2456682fe987b62e699d71c84b43b504ca63f6b`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

v4 对应为：

- `4f26045b981e6297674f06d9ab8f2a00400bb2a7940db9c110b1aed5d55b7b71`；
- `30d3bdd33447d221180fa3eb855d705a1e01c9cbf719eff22eb9dd469c91a41d`；
- `609cb966db3f6e85d5bf30541f78834254c3cf388aec748eb29fe18900eaecb7`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

两轮 JSON 和日志会记录各自输出/解压目录，因此文件哈希不同，不影响四项
不可变 OCI 内容完全一致。完成后的首次只读汇总因宿主没有 `jq` 提前退出；
固定 Python 的第一次比较又因手工错误补全 `0051472` 的完整哈希而触发
断言。两次都没有修改构建产物；最终直接从 Git 和 build report 读取完整
提交后，上述确定性比较全部通过。

有效构建与验收全程使用固定 Python 3.12.13、禁用网络、显式清空
`CUDA_VISIBLE_DEVICES`，没有传入 `--gpus` 或注入 NVIDIA runtime。结束后
8 张 GPU 均为 0 MiB/0%，没有 compute process；唯一外部下载容器的
DeviceRequests 为 null。当前 daemon 中也不存在
`glm52-oscar-a800-phase6-ca4a404e9-0275043c:latest`。因此本节只关闭
可复现 OCI 双构建和递归身份门禁，尚未完成 daemon import、runtime import、
正式链路迁移或新的 32K/batch1 TTFT、TPOT、吞吐验证。下一步先发布本节
实时记录；发布前不执行 daemon import。

### 2.46 BF16 tile gate 候选 daemon 导入与身份审计

2.45 与 planning 已由主仓库提交 `9fbba33` 发布，状态由
`11b8ccd` 固化；导入前主仓库与源码仓库均为 clean/published。
本阶段只从已通过确定性双构建和递归验收的 v3 只读 OCI layout
导入 Docker daemon：

`artifacts/phase6/20260731T1628Z_candidate_ca4a404e9_bf16_tile_gate_v3`。

导入入口从 v3 `index.json` 原样解析 ref，并确认与 config 记录
一致：

`glm52-oscar-a800-phase6-ca4a404e9-0275043c`。

一次性 Ubuntu 22.04 工具容器使用 skopeo 1.4.1、阿里云 HTTP
软件源、只读 OCI layout 与 Docker socket 完成复制；没有传入
`--gpus`、没有注入 NVIDIA runtime，也没有执行 CUDA kernel。

首次外层客户端在工具容器仍运行时提前返回，当时预期的退出码
文件尚未生成。本阶段没有重复启动第二次导入，而是对同一容器执行
`docker logs -f` 和 `docker wait` 接管。该容器最终真实退出码为
0，日志完整运行到 `Storing signatures`，工具容器随后自动删除。
原日志与接管日志均为 14,998 bytes，逐字节完全一致，SHA256 均为：

`75b98c9d9140e9c2d217d3660f868d7e2cd740dbdae66cda1bb02719d7094551`。

原退出码文件与接管退出码文件内容均为 `0`，SHA256 均为：

`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

独立 daemon 身份审计状态为 `passed`，实测身份为：

- image ID：
  `sha256:7c85cdd01bdc18d286aaabccd442967be660e59fc964f334f3fc30b1cd5a4eb8`；
- 层数：33；
- 最后一层 diff-ID：
  `sha256:5f8875b9e7e5c465e2c935f1518355782a7736e74e2d1dc71624900d08a67a14`；
- tag：`glm52-oscar-a800-phase6-ca4a404e9-0275043c:latest`；
- source commit/tree：
  `ca4a404e913ce55237ca60383cc86e221fbfea26` /
  `079815219a02add3f37318ed434924e80f80a35d`。

审计同时确认 source revision/tree、candidate layer、Dockerfile、
rotation manifest、rotations、runtime expectation 和 base manifest 共
8 项 labels 全部与 v3 build report 一致。其中 candidate layer 为
`sha256:3f03376d01935fc9e057a34a01b6e701737a5384b19281a8bc204cb7ddbae203`，
Dockerfile SHA256 为
`51ed571f615559a0008631c17244d88878ef0c8437c3c3161a61d1e4c022e6f4`。

daemon inspect、身份审计 JSON 和导入后 GPU 快照的 SHA256 依次为：

- `d9a47f165c4d8c17b9c80908b8dbf7ffb88ce030a7ad503ea5300a8529fa76bd`；
- `15cd9abd07705eae36c6a71402c91eab127c6848b45a65bb70e8c7df42d37c5b`；
- `e3d6d9dc02a2547485121197643573ab891c39da13806e49e571dd6ce6d4ad40`。

导入后 `2026-07-31T17:05:25Z` 复查 8 张苹果800 均为
`0 MiB/0%`，没有 compute process；唯一项目外下载容器的
DeviceRequests 为 null，不占用 GPU，因此未终止。至此 BF16 tile gate
候选已关闭可复现 OCI 双构建、递归验收与 daemon identity 门禁。
本节尚未执行 driver-injected runtime import、正式链路迁移或新的
32K/batch1 端到端测试，因此没有新的 TTFT、TPOT 或吞吐结果。
下一步先发布本节实时记录；发布前不执行 runtime import。

### 2.47 BF16 tile gate 候选 driver-injected runtime import

2.46 与 planning 已由主仓库提交 `5498fee` 发布，发布状态由
`07a2fa3` 固化。runtime import 前新双空闲检查由提交
`1770f2b` 发布后才进入有效轮次。

双空闲检查时间为
`2026-07-31T17:21:52Z/17:23:08Z`，间隔 76 秒；两次均为
8/8 张苹果800 `0 MiB/0%`，没有 compute process。唯一运行的
项目外下载容器 DeviceRequests 为 null，不占用 GPU，因此未终止。
启动前 `17:24:55Z` 的即时复查仍为 8/8 卡全部空闲。

有效 runtime import 只将 GPU 0 映射给候选镜像，只注入 NVIDIA
driver 用户态库；没有加载模型、分配模型显存或执行 CUDA kernel。
探针精确复用 2.35 已通过的冻结协议，FlashInfer 只通过
`importlib.metadata` 读取包版本。本轮一次通过，退出码为 0，
`runtime_import.json` 状态为 `passed`，实测身份为：

- Python/PyTorch/Triton：`3.12.13/2.11.0+cu129/3.6.0`；
- Transformers/Tokenizers：`5.8.1/0.22.2`；
- FlashInfer Python/JIT cache：`0.6.6/0.6.6+cu129`；
- vLLM Python：`/opt/vllm_glm52_v1/vllm/__init__.py`；
- vLLM 原生扩展：`/opt/vllm_glm52_v1/vllm/_C.abi3.so`；
- rotation 数量为 78，rotation manifest、rotations 与 runtime
  expectation 三项 SHA256 全部匹配；
- `reasoning_effort=max`，`cuda_initialized=false`。

导入 vLLM Python 时出现一条与历史有效轮次一致的 RuntimeWarning：
候选源码包没有生成版 `vllm._version`，因此 reported version 为
`dev`。候选 Python/原生扩展路径和 source commit/tree 已由独立
OCI/daemon 身份门禁绑定，该 warning 没有导致断言放宽或失败。

有效 JSON 与日志均与 2.35 的 fd281f5f9 冻结证据逐字节完全一致。
双空闲检查、启动前复查、有效 JSON、有效日志、退出码和退出后 GPU
快照的 SHA256 依次为：

- `65a6f94ae124b26a8f3b3452eb1c82bc3980828d27728af7858f54c5fa1df733`；
- `bad6a83f0a53db9986860812b7e24136e8ec3e0df7fb2b17459146a485ca2e51`；
- `0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`；
- `f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `3749a54e044b619d2751464c0229d39f8c14c859044819c9a522017a075335e9`。

有效容器已自动删除，`17:25:04Z` 复查 8 张 GPU 均为
`0 MiB/0%`，没有 compute process。至此 BF16 tile gate 候选已关闭
OCI 双构建、递归验收、daemon identity 与 driver-injected runtime import
门禁。本轮不是 32K/batch1 端到端测试，因此没有新的 TTFT、
TPOT 或吞吐结果。下一步先发布本节实时记录；发布前不切换
Stage 9 控制镜像。

### 2.48 BF16 tile gate 的 Stage 9 控制入口迁移

2.47 与 planning 已由主仓库提交 `53c55b0` 发布，发布状态由
`40e0624` 固化。进入本阶段前，主仓库 HEAD 与 upstream 均为
`40e0624dae2cbf30616e057719cf75899d65c342`；源码仓库 HEAD 与 upstream
均为 `ca4a404e913ce55237ca60383cc86e221fbfea26`，源码仓库工作区干净。

本阶段只修改 `docker/Dockerfile.phase9-runtime` 的第一行，将 Stage 9
控制镜像的默认基础镜像从 fd281f5f9 候选切换为已经通过 2.43–2.47
完整门禁的 BF16 tile gate 候选：

`glm52-oscar-a800-phase6-ca4a404e9-0275043c:latest`。

Dockerfile 其余安装步骤和入口均未修改。修改前后 SHA256 分别为：

- 修改前：
  `93111035802a79bcb31564111a542e33660d167f4b4175addbdd89e5f65d8bbb`；
- 修改后：
  `f832ebb19cf7ffe28e19b26a3978aa80b59a5fd4b2f07dff40d04b6aeb342737`。

修改后的 Dockerfile 中旧 fd281f5f9 身份计数为 0。CPU-only 输入门禁目录为：

`artifacts/phase9-control/20260731T1735Z_runtime_ca4a404e9_input_v1`。

该门禁直接读取 daemon image、Phase 6 配置和冻结 build report，状态为
`passed`，实际核对结果为：

- 默认基础镜像 tag 与 `candidate_inputs.json` 的 output tag 一致；
- daemon image ID 为
  `sha256:7c85cdd01bdc18d286aaabccd442967be660e59fc964f334f3fc30b1cd5a4eb8`；
- 基础镜像为 33 层，入口为 `/bin/bash`；
- source commit/tree 为
  `ca4a404e913ce55237ca60383cc86e221fbfea26` /
  `079815219a02add3f37318ed434924e80f80a35d`；
- candidate layer digest 为
  `sha256:3f03376d01935fc9e057a34a01b6e701737a5384b19281a8bc204cb7ddbae203`；
- 上述身份均与 2.45 的 v3 build report 一致，源码仓库保持
  clean/published。

有效 JSON、日志、GPU 快照和证据清单的 SHA256 依次为：

- `d068ccfea15b01d5b3ceb75620f316c8071494182071cff77b756ee0ec66975d`；
- `d068ccfea15b01d5b3ceb75620f316c8071494182071cff77b756ee0ec66975d`；
- `bc78eb1a6ba834da56d59e852e9ae653fdfec6381c413a7a11bdb9968e1e2b4a`；
- `410ee592646fb7a9f20aa6f157bcb8db2655843a2e34f6d31a2b1e0dad984ba3`。

四份文件合计 2,179 bytes。本阶段没有构建新的控制镜像，也没有注入
NVIDIA runtime 或分配 GPU；检查时 8 张苹果800 均为 `0 MiB/0%`，没有
compute process。目标 tag `oscar-glm-stage9-runtime:ca4a404e9` 当前仍不
存在。因此本节只证明 Stage 9 控制入口已经精确绑定到新候选，尚未产生新的
32K/batch1 TTFT、TPOT 或吞吐结果。下一步先发布本节、Dockerfile 与
planning；主仓库恢复 clean/published 后，再执行 CPU-only 控制镜像构建、
34/33 层继承审计与不注入 GPU 的 runtime 检查。

### 2.49 BF16 tile gate 的 Stage 9 控制镜像构建与 CPU runtime 验收

2.48、控制 Dockerfile 与 planning 已由主仓库提交 `a8cb54b` 推送到
`origin/feat/glm52-model-load`，构建开始前本地 HEAD 与 upstream 均为
`a8cb54b9b96ed2607ce9547db376bf054a269533`。本阶段沿用 2.48 已发布的
Dockerfile，以 `docker/` 作为 4.608 kB 的最小构建上下文，CPU-only 构建：

`oscar-glm-stage9-runtime:ca4a404e9`。

有效证据目录为：

`artifacts/phase9-control/20260731T1740Z_runtime_ca4a404e9_v1`。

构建退出码为 0，新控制镜像的完整 image ID 为：

`sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`。

基础镜像仍为 2.48 已验收的
`sha256:7c85cdd01bdc18d286aaabccd442967be660e59fc964f334f3fc30b1cd5a4eb8`。
独立身份审计状态为 `passed`，结果为：

- 控制镜像/基础镜像层数为 34/33；
- 控制镜像前 33 层与基础镜像逐层完全一致；
- 两者 labels 完全一致，source commit/tree 仍为
  `ca4a404e913ce55237ca60383cc86e221fbfea26` /
  `079815219a02add3f37318ed434924e80f80a35d`；
- 两者 entrypoint 均为 `/bin/bash`；
- 身份审计退出码为 0。

随后在不传入 `--gpus`、不注入 NVIDIA runtime、禁用网络的一次性容器中，
使用固定 Python `/opt/fp8_speed_up_v4_venv/bin/python` 执行 CPU runtime
检查。结果状态为 `passed`：Python `3.12.13`、glibc `2.35`、Git
`2.34.1`、iproute2 `5.15.0`，两个固定系统包版本与 fd281f5f9 控制镜像
一致，且 `cuda_initialized=false`。本轮 `runtime_check.json` 与
fd281f5f9 历史控制镜像对应 JSON 逐字节完全一致，SHA256 为：

`5ac65b5da3ffc9bcf642bd3beb8a989b177710d38c51a9457b5198ffd18c1f20`。

构建日志、基础镜像 inspect、控制镜像 inspect、身份审计 JSON/日志、
runtime JSON 和证据清单的 SHA256 依次为：

- `1c17f55a5992f4e73107248d69c6037ec6ae4d13a01c4eaa3999f17f9668d7ef`；
- `d9a47f165c4d8c17b9c80908b8dbf7ffb88ce030a7ad503ea5300a8529fa76bd`；
- `5a266feb26ac8f0793ac7ef5a8510431b501f70f2edc880d1d826ba70ed29c31`；
- `0652ae1fd113d164faa330f3c92173ae879b04626dc2e829a486b7ab52b52b20`；
- `28bbd3f4c5db4efdd258ac187472fb7bf9eb778e0c2e4449f24fe7e626f4ef79`；
- `5ac65b5da3ffc9bcf642bd3beb8a989b177710d38c51a9457b5198ffd18c1f20`；
- `c33aa9848b74dff87abe950d59f93a50826bf4d770c136961f821f3d5753cf6a`。

目录内 13 份文件合计 42,387 bytes；构建、身份审计和 runtime 三个退出码
均为 0。构建前后 8 张苹果800 均为 `0 MiB/0%`，结束时没有 compute
process；该阶段没有使用 GPU。至此新控制镜像的构建、34/33 层继承和
CPU runtime 门禁已经关闭，但 Phase 1/5/7/9 正式配置与 overlay 尚未迁移，
也尚未执行新的 driver-injected preflight 或 32K/batch1 端到端测试，因而
没有新的 TTFT、TPOT 或吞吐结论。下一步先发布本节与 planning；主仓库恢复
clean/published 后，再最小迁移正式配置并执行 CPU-only 递归门禁。

### 2.50 BF16 tile gate 的 Stage 9 正式静态链路迁移

2.49 与 planning 已由主仓库提交 `c28333d` 推送到
`origin/feat/glm52-model-load`。本阶段首先从 2.45 已验收的 v3
candidate layer 单份机械派生正式 overlay：

`artifacts/phase6/20260731T1628Z_candidate_ca4a404e9_bf16_tile_gate_v3/overlay_rootfs`。

只读 `extracted-layer` 和新 overlay 各有 4,749 个普通文件，其中 4,744 个
为源码文件，另外 5 个为 rotation/runtime artifact。两边按相同相对路径
生成的递归 SHA256 清单逐字节完全一致，清单 SHA256 均为：

`45eb2d62d0f02daa3e123c28681be31ab133e574fd399b1dc29b57979e70eeb6`。

overlay 另外建立 6 个指向冻结 Phase 0 rootfs 的原生扩展 symlink；其相对
路径和绝对目标与 fd281f5f9 正式 overlay 逐字节完全一致。链接清单与目标
哈希清单 SHA256 分别为：

- `f17949949ff89f8a6e2624c9999f4276cfbe9ab4da42cc586029b0bf373276b2`；
- `c2a5c7c2953265878d11f0b0e54d7a8c362c6aad68ce433a20a441e713149468`。

随后按 Phase 1→5→7→9 的依赖顺序最小迁移正式配置。Phase 1 更新 source
commit/tree；Phase 5 绑定实算后的 Phase 1 哈希；Phase 7 更新 Phase 5 哈希、
v3 OCI/build/verification/runtime、overlay 和 source；Phase 9 更新 candidate
与控制镜像身份。四份配置 SHA256 依次为：

- `configs/phase1/native_baseline.json`：
  `84163a1975698e324c6b00dd760763615f7bc33004932a4f9838bf94af0ac04e`；
- `configs/phase5/oscar_tp8.json`：
  `aea9d51be4124044a943d199e16cdbb1da05c8ccec310e3a26c37dd5e67a5613`；
- `configs/phase7/oscar_evaluation.json`：
  `2f725eeef48b49fe47a4071f61bfd7c7db16dd7a96b2b65c6fa491db3ddb9678`；
- `configs/phase9/performance_matrix.json`：
  `14d3f71e8a36b429d31e78ff8a7c0a9810048a308440693aef08b2ae9aea4760`。

9 个正式 shell wrapper 和 1 个 Phase 9 工具测试期望值同步到
ca4a404e9。4 个 JSON 均可解析，9 个 shell 的 `bash -n`、固定 Python 对
11 个 Phase 9 Python 文件的 compile、正式范围旧 fd281f5f9 身份清零和
`git diff --check` 均通过。

CPU-only 工具测试使用新控制镜像、固定 Python 和只读项目挂载。Phase 7
首轮为 19 passed、1 failed：唯一失败是控制容器未挂载 frozen evaluator
launcher 指向的 `/dev/shm` 固定解释器；对应日志 SHA256 为
`d20bbef0ac06d75deb22eb1c5afff259670edcf18e74cb10c7cecadf79249d02`。
补入同绝对路径只读挂载后，第二轮已正确启动子进程，但同一 resume 用例在
固定 30 秒处冷启动超时，日志 SHA256 为
`e558aa1b595c535ad9103b0554988207c04389c6366bbfd15d5268e2b23cc803`。
两轮均保留且不计入通过结果；没有修改实现、测试或超时。

缓存预热后的有效 Phase 7 v3 为 20 passed、2 个只读 pytest cache warning、
4.94 秒，退出码 0；Phase 9 为 22 passed、3 个同类 warning、2.46 秒，
退出码 0。两份有效日志及 11/11 compile 日志 SHA256 依次为：

- `09a1c836a0825ff1c3390751250da1aabb32691976365a49477e1e2b18633404`；
- `0fd7cf60971a29b7b45aee0bf42991ad18365cf1c25d6d4c2276ab94897e994d`；
- `526e053b0aa1b775ed9081fc0c783039a4014410eb851d5e550d7756ad13e9d5`。

递归门禁首个组合命令已打印 64/64 静态检查通过，但随后继续进入需要 driver
的固定环境 import；由于本阶段刻意不注入 GPU，最终因缺 `libcuda.so.1`
退出，组合日志 SHA256 为
`3ed93cfcc53e69879aa5c76cad21c6e1499b80d130003dce90b3527f2939f895`。
该轮不计为绿色。随后只运行递归 verifier 的 v2 命令遗漏模型目录只读挂载，
在读取模型 `config.json` 前退出；日志 SHA256 为
`87476f877a648079b1f80fff301a7208a28eacde6fd6932551bb84af7fedd689`。

有效 v3 只补正式协议已有的模型目录只读挂载，继续不注入 GPU。结果为
64/64 checks、0 failed、exit=0，覆盖 source/tree、OCI descriptor、6 个
原生链接、rotation、runtime expectation、baseline、frozen evaluator 与
Phase 9 32K/128K 参数身份。v3 JSON 与 stdout 日志逐字节完全一致，SHA256
均为：

`67b040dc202160c93e28ac0f994bfd4e1ceeabb6f52ca6beb35b61402b3fac9c`。

静态汇总 JSON/日志 SHA256 均为
`d940e52ce916fc17bc146973bc94fc2a621ed1ac3ba64adb47698778ffa3d838`，
证据 manifest SHA256 为
`ec772942b1d52c413e2f538b95a30f6f75eb4caf1dc45cefdb81e8cb74024a5a`。
有效与失败边界共 25 份文件、102,643 bytes。递归门禁前后 8 张苹果800
均为 `0 MiB/0%`，没有 compute process；本阶段全程未分配 GPU。

至此 ca4a404e9 的正式 overlay、Phase 1/5/7/9 配置、工具测试和递归静态
身份门禁已经关闭。本节仍不是 driver-injected preflight 或 32K/batch1
端到端测试，因此没有新的 TTFT、TPOT 或吞吐结论。下一步先发布本节、配置、
wrapper 与 planning；两仓恢复 clean/published 后，再执行新的双 GPU 空闲
检查和正式 driver-injected preflight。

### 2.51 BF16 tile gate 的 Stage 9 正式 driver-injected preflight

2.50、正式配置、wrapper 与 planning 已由主仓库提交 `59e8d1a` 推送到
`origin/feat/glm52-model-load`。正式 preflight 的双空闲状态随后由提交
`77b5aa5` 发布；进入有效轮次时，主仓库本地 HEAD 与 upstream 均为
`77b5aa51d42c65d6753adf0e3a465a3c21c490ae`，源码仓库本地 HEAD 与
upstream 均为 `ca4a404e913ce55237ca60383cc86e221fbfea26`。

有效证据目录为：

`artifacts/phase9-control/20260731T1813Z_stage9_candidate_ca4a404e9_preflight_v1`。

外层双空闲检查时间为 `2026-07-31T18:13:35Z/18:14:47Z`，间隔 72 秒；
两次均为 8/8 张苹果800 `0 MiB/0%`，没有 compute process。双检日志
SHA256 为：

`28991d3d9417e1ce1d2c4f5c065dd901ed6b29b622b9f96461713be4e6d06c5c`。

双检发布后，启动前即时复查仍为 8/8 卡空闲。正式 run ID 为：

`20260731T1816Z_candidate_ca4a404e9_preflight_v1`。

该轮只通过 `--gpus all` 注入 NVIDIA driver 和可见设备，不加载模型，执行
递归静态身份、固定环境 import 与服务参数解析。preflight 退出码为 0，
容器随后自动删除。三项结果为：

- 静态递归门禁状态 `passed`，64/64 checks、0 failed；
- 固定环境 import 状态通过，Python/PyTorch/Triton 为
  `3.12.13/2.11.0+cu129/3.6.0`，Transformers/Tokenizers 为
  `5.8.1/0.22.2`，FlashInfer Python/JIT cache 为
  `0.6.6/0.6.6+cu129`，`cuda_initialized=false`；
- 参数解析状态通过，tensor/pipeline parallel 为 8/1，attention backend
  为 `TRITON_MLA_SPARSE`，KV cache dtype 为 `oscar_mla_int2`，
  max model len/max seqs/max batched tokens 为 `131072/16/2048`，
  eager、chunked prefill、torch profiler 生效，prefix cache、speculative、
  async scheduling 均关闭，`cuda_initialized=false`。

静态、固定环境和解析参数 JSON 的 SHA256 依次为：

- `d8db6b233c6249bed47b81de211a533e5aabe7cfe18025b9b688f5f0e91f126c`；
- `56f92356322376f98d88bbbe4786eed46e4a590a666fc57143ba96326c250574`；
- `6365691f6e7d357f3123e8fadb26d61c8447da76bbe8f908a19e664addae9f2c`。

完整 preflight 日志、退出码、启动前和退出后 GPU 快照的 SHA256 依次为：

- `34c09065dfce3a54f56b63a3292085eeaa5cf6d9dde19ee0a0750cb2b75d7a15`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `bc78eb1a6ba834da56d59e852e9ae653fdfec6381c413a7a11bdb9968e1e2b4a`；
- `bc78eb1a6ba834da56d59e852e9ae653fdfec6381c413a7a11bdb9968e1e2b4a`。

退出后的 compute-process 快照为空文件，SHA256 为
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`；
8 张 GPU 均已恢复到 `0 MiB/0%`。

preflight 本身通过后的首版汇总脚本手工补全了错误的 `77b5aa5` 完整哈希，
只在 Git identity 断言处失败，留下 0-byte validation log，未生成 JSON；
该空日志 SHA256 为上述空文件哈希，首版证据清单 SHA256 为
`25934ff1002a4e325bb11d895307200395af943e0fd00bec6083d197d69780c4`。
这不改变 preflight 结果。有效 v2 直接从 Git 读取完整哈希，validation
JSON/日志逐字节完全一致，SHA256 均为：

`f6ad93acacb38544142c1965fcde360f648179674bcf08f969b8eba2887c7572`。

有效 v2 证据 manifest SHA256 为：

`28dd8109239e2be1e3a1c7afadb117d8d309fe32393f953e133ae9c74b252665`。

包含失败汇总边界在内共 14 份文件、45,826 bytes。至此 ca4a404e9 已关闭
正式 driver-injected preflight 门禁；但本轮没有加载模型或执行
32K/batch1 请求，因此仍没有新的 TTFT、TPOT 或吞吐结论。下一步先发布本节
与 planning；两仓恢复 clean/published 后，再为正式 32K/batch1 OSCAR
轮次重新执行两次 GPU 空闲检查。

### 2.52 BF16 tile gate 的 32K/batch1 正式结果

2.51 与 planning 已由主仓库提交 `cedeb16` 发布。正式运行前的外层双空闲
检查由提交 `5dc8155` 发布；本轮冻结身份为：

- 主仓库提交：
  `5dc8155c44e8baea55f8fce265763dcd848be7a7`；
- 源码提交/tree：
  `ca4a404e913ce55237ca60383cc86e221fbfea26` /
  `079815219a02add3f37318ed434924e80f80a35d`；
- Phase 9 配置 SHA256：
  `14d3f71e8a36b429d31e78ff8a7c0a9810048a308440693aef08b2ae9aea4760`；
- run ID：
  `20260731T1826Z_candidate_ca4a404e9_32k_b1_v1`；
- 负载：32,768 输入 token、128 输出 token、batch/并发 1、1 次 warm-up、
  每轮 3 个 measured 请求、3 轮正式测量和 8+8+1 profiler。

外层空闲检查为 `18:24:29Z/18:25:45Z`，间隔 76 秒；容器内检查为
`18:28:16Z/18:29:19Z`，间隔 63 秒。四次均为 8/8 张苹果800
`0 MiB/0%`、没有 compute process；唯一项目外下载容器不占 GPU，未执行
终止操作。

141/141 个模型 shard 全部加载；权重加载耗时 `744.85 秒`，模型加载共用
`804.392436 秒`、每卡模型内存 `56.0 GiB`，可用 KV cache 为
`13.74 GiB`。服务于 `18:52:53Z` ready，正式启动记录为 `1,321 秒`。
启动、服务和 profiler 阶段均实际输出 10 分钟进度；profiler 子任务超过
10 分钟时也输出
`label=input_32768_batch_1/profile elapsed_seconds=600`。

三轮均为 3/3 completed、0 failed，验证状态均为 `passed`：

| 轮次 | TTFT（ms） | TPOT（ms） | 吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 32,657.086 | 200.037 | 0.017223 |
| 2 | 32,683.066 | 200.404 | 0.017201 |
| 3 | 32,705.530 | 199.531 | 0.017228 |

三轮 `mean` 指标中位数与 2.38 的 causal-loop 正式结果、同格 BF16 对比
如下：

| 指标 | BF16 | Causal-loop OSCAR | BF16 tile gate OSCAR | 相对 causal-loop | 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| TTFT（ms） | 12,528.026 | 35,683.893 | 32,683.066 | **-8.41%** | +160.88% |
| TPOT（ms） | 178.832 | 197.826 | 200.037 | +1.12% | +11.86% |
| 请求吞吐（req/s） | 0.028377 | 0.016445 | 0.017223 | **+4.73%** | -39.31% |

本候选相对 causal-loop 将 TTFT 降低 `3,000.827 ms`，请求吞吐提升
`4.729%`，但 TPOT 回退 `2.211 ms`。三轮 TTFT、TPOT 和吞吐相对极差
分别为 `0.1482%/0.4365%/0.1521%`，结果稳定。TPOT 仍低于 BF16
`+20%` 上限 `214.598 ms`；TTFT 上限为 `15,033.631 ms`，当前仍高出
`17,649.435 ms`，即为 BF16 的 `2.609×`。因此 tile gate 已产生明确的
端到端 TTFT 收益，但性能优化尚未关闭 TTFT 门限。

单格和总 summary 均为 `passed`。三轮测量峰值显存为每卡
`80,679 MiB`，profile 峰值为每卡 `80,691 MiB`；最多运行 1 个请求、
等待为 0、preemption 为 0，KV usage 峰值为 `5.7902%`，没有容量排队
或抢占。

Profiler 状态为 `passed`，耗时 `829.0547113418579 秒`；8 份 worker
trace、8 份 CUDA table 和 1 份 frontend trace 均通过 rank、bytes 与
SHA256 校验，critical rank 为 2，kernel total 为 `65,104 ms`。8-rank
table 中 `_mixed_sparse_prefill_stage1` 均执行 `1,248=16×78` 次，CUDA
total 中位数从 causal-loop 的 `23,134.5 ms` 降到 `20,128.0 ms`，减少
`3,006.5 ms`（`-13.00%`）；该变化与端到端 TTFT 的 `-3,000.827 ms`
高度一致。精确的多 chunk trace 归因尚未执行，因此本节只记录 table 趋势，
不提前把它写成完整因果分解。

总 summary、单格 summary、profile result、profile runner log、外层正式
日志、双空闲检查、退出码和小型证据清单的 SHA256 依次为：

- `158ef17ef38101b4f61d58ddef44e590b6f7bb26b68c21f6840be34fb85c5fde`；
- `114176f5c0a3b2b69e203e3d6790468914a49049562af98f085adfab1eb8dd12`；
- `95850582c23837f521257a4e1da189dcea843668b69e40320ef7193890a08ba5`；
- `be3eb5e2db3b673213457b32b6a9c50ac853cec95939dd24d090ff4ed2904004`；
- `dc3cffd9f25169cf6e7165aa09cb0fba8393c2816ae270af09dfe163a38ceb8b`；
- `c5abd67960b949ba11fb362360dbc5bb804edaa04ac25b89f75b739069cca811`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `21fdbe75a2aab8027620b0225f24f2883a4d9164113308dc4a63e8c6c88b1fee`。

小型证据已复制到
`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/formal_32k_b1_results`。
证据清单包含 40 项并已 40/40 通过复算；目录连同清单共 41 个文件、
1,164,335 bytes。约 1.2 GiB 的原始 trace 保留在 `/dev/shm`，没有重复
复制；profile validation/summary 已冻结每个 trace 的路径、bytes 和
SHA256。

正式 runner 退出码为 0，实验容器已自动删除；结束后 8 张 GPU 均为
`0 MiB/0%`、没有 compute process。下一步先发布本节实时记录，再对本轮
冻结的 8-rank trace 做 CPU-only 多 chunk 归因；归因完成并更新报告前，
不选择或实施下一项性能源码改动。

### 2.53 BF16 tile gate 32K 多 chunk trace 归因

2.52 的正式结果已由主仓库提交 `c1562a0` 发布。本阶段只流式
解析 2.52 冻结的 8 份 worker trace，没有重新启动模型或分配 GPU。
有效 analysis ID 为：

`20260731T1948Z_ca4a404e9_32k_prefill_trace_v2`。

有效轮次固定使用控制镜像
`oscar-glm-stage9-runtime:ca4a404e9`、Python `3.12.13`、
`ijson 3.4.0.post0`、4 个 CPU worker 和 top-40 汇总。Docker 显式使用
`runc`、`--network none`、4 CPUs，并清空 `CUDA_VISIBLE_DEVICES`，
`NVIDIA_VISIBLE_DEVICES=void`；没有注入 NVIDIA runtime。分析器 SHA256 为：

`8b6b2393f93be3783f47ccbe3ecb26020cc2b65526c99fcb464c4768ce4330f7`。

预备轮次曾直接使用控制镜像预装的 `ijson 3.5.0` 完成一次解析；
事后 summary 的环境字段与冻结协议不符，因此该轮已判为无效，
没有混入下述结论。随后一次 offline `uv run --with` 因无法从 cache
重建临时环境退出；首次持久 venv 创建也因默认 `/.cache/uv` 对宿主
UID 不可写而退出。这两次都在读取 trace 前结束。有效轮次改为先用
uv 和清华 PyPI 源在任务专属 `/dev/shm` 环境锁定安装
`ijson==3.4.0.post0`，再在断网分析容器中直接使用该环境。

有效轮次退出码为 0，summary 状态为 `passed`，耗时
`114.4914088351652 秒`。8/8 ranks 均解析到 144 个 execute context、
16 个 prefill chunk 和精确 32,768 个输入 token；8 份 trace 的 bytes 和
SHA256 也均与 summary 逐份一致。与 2.39 的 causal-loop trace 及同格
BF16 trace 的中位数对比如下：

| Trace 指标 | BF16 | Causal-loop OSCAR | BF16 tile gate OSCAR | Tile gate 相对 causal-loop | Tile gate 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| Prefill wall（ms） | 10,086.470 | 35,731.482 | 32,756.592 | **-8.3257%** | +224.7577% |
| Prefill kernel 合计（ms） | 9,533.580 | 34,779.955 | 31,755.930 | **-8.6947%** | +233.0955% |
| 各 rank generation 中位数再取中位（ms） | 223.325 | 266.230 | 269.372 | +3.142 ms | +46.047 ms |
| 主 attention kernel（ms） | 3,384.374 | 23,134.871 | 20,128.143 | **-12.9965%** | +16,743.769 ms |

BF16 的主 attention kernel 为 `_sparse_mla_kernel_final_static`，中位调用数
为 933；两个 OSCAR 候选的对应 kernel 为
`_mixed_sparse_prefill_stage1`，均为 `1,248=16×78` 次。当前 stage1 中位
CUDA total 为 `20,128.143242 ms`，占 prefill wall 的 `61.447612%`。

causal-loop 到 tile gate 的 prefill wall 共减少
`2,974.890437 ms`，其中 stage1 减少 `3,006.728218 ms`，解释 wall
改善的 `101.070217%`。去掉 stage1 后的剩余 wall 从
`12,596.610478 ms` 变为 `12,628.448260 ms`，反而增加
`31.837781 ms`（`+0.252749%`）。这证明 2.52 的 TTFT 收益的因果主体确实是
tile gate 减少的 stage1 工作，而不是其他 kernel、generation 或调度噪声。

性能仍未收敛。当前 OSCAR prefill wall 比 BF16 多
`22,670.121734 ms`；stage1 比 BF16 原生 attention 多
`16,743.769268 ms`，仍解释两者 prefill wall 差距的
`73.858312%`。因此下一项最小优化仍应聚焦 grouped prefill stage1，
并减少全部 16 个 chunk 都会执行的有效计算或访存；当前数据不支持把
优化重心转向 generation 或其他小 kernel。

有效 summary、comparison、validation、run log、run identity、trace input
清单和退出后 GPU 快照的 SHA256 依次为：

- `54b8feebbafc425691f7f5c6cb52abf9ea42dc5248740c95a59b2acfad9b27b9`；
- `9633a0de1c7da53ebd731d99ed1bce4ac2919aee1b84fa2d0d8b811333f1570b`；
- `025b2b7bb79835f1bf1233465aa0d30939059a94d4cb60e82b5e7c553d851875`；
- `b527b0ddc522e5afa3368d4ff1583da7bc4be37ef3e24828de0e363f81310983`；
- `1c9897d0d2c77115e473c820c8521b6add2e5a31d0e6070e5027c5c6ecfc0169`；
- `ff4699856f92ef48397a641094c06f6bdf5de28831aec78d2d0eef34c3f64a34`；
- `4de51a8c40e0c48cd31544134a950dbfc8958ec2b4f0c35fe38aeb6fb808abab`。

有效证据已复制到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/formal_32k_b1_trace_analysis`。

证据 manifest 包含 8 项并已 8/8 通过复算；目录连同 manifest 共 9 份文件、
346,895 bytes，manifest SHA256 为
`bf94602fc922654482cfbf649d5411eea32d63a3d1aa81baf0552abca6eb5b73`。
原始 trace 仍仅保留在 `/dev/shm`，没有复制到仓库。`2026-07-31T19:51:40Z`
退出复查显示 8 张苹果800 均为 `0 MiB/0%`，没有 compute process。
下一步先发布本节与 planning；主仓库恢复 clean/published 前，不实施下一项
性能源码改动。

### 2.54 Prefill selected index 排序候选的 CPU-only 覆盖率筛选

2.53 与 planning 已由主仓库提交 `a66db02` 发布，发布状态由后续提交
`83cf7b1` 固化；主仓库和源码仓库均保持 clean/published。2.53 证明当前
stage1 相对 BF16 原生 attention 的超额仍解释 prefill wall 差距的
`73.858312%`，因此本阶段继续只筛选能覆盖全部 16 个 chunk 的最小
stage1 优化，没有修改生产源码或正式配置。

只读源码检查发现，原生 `csrc/sampler.cu` 已支持环境变量：

`VLLM_TOPK_PREFILL_SORT_INDICES=1`。

该路径的语义为：

- 当 `rowLen<=topK` 时，原生 kernel 在 index sort 前直接返回连续有效
  index 和 `-1` 尾部，因此首个 2K chunk 不受排序影响；
- 当 `rowLen>topK` 时，对已选中的同一 top-k index 集合使用 CUB
  `BlockRadixSort<int>` 按 token index 重排，不改变 selected token 集合；
- 重排后 prefix、history 和 recent token 按 index 聚集，ca4a404e9 已落地的
  两个 `has_bf16` gate 因此可以在更多全 history tile 上跳过两个
  数学贡献为零的 BF16 dot。

当前 `configs/phase9/performance_matrix.json` 中没有启用该变量。
sampler、stage1 与 Phase 9 配置的 SHA256 依次为：

- `6a815b61e110a8a5815507a229a8a1a5295a8ed1611c9b85e7cc092db7ccf37b`；
- `23b08ffae200cfefa3e7a2190c436c0f517bfc509e8479bb22245230b10bc70c`；
- `14d3f71e8a36b429d31e78ff8a7c0a9810048a308440693aef08b2ae9aea4760`。

随后执行的 CPU-only 合成覆盖率审计 ID 为：

`20260731T2002Z_prefill_sort_coverage_32k_v1`。

轮次固定使用控制镜像
`oscar-glm-stage9-runtime:ca4a404e9`、Python `3.12.13`、
Torch `2.11.0+cu129`、runc、network none 和 4 CPUs；显式清空
`CUDA_VISIBLE_DEVICES`、设置 `NVIDIA_VISIBLE_DEVICES=void`，未注入
NVIDIA runtime。合成协议为 seed 42、每 chunk 2,048 个 query、
top-k 2,048、tile width 16、prefix/recent 为 64/256，并依次处理
final sequence length 2,048–32,768 的 16 个 chunk。

有效轮次退出码为 0、summary 状态为 `passed`，耗时
`809.2952793529257 秒`。轮次超过 10 分钟；在约 10 分钟时已输出
12/16 chunks 进度，之后继续到 16/16 完成。覆盖结果为：

| 范围 | 排序前含 BF16 的 tile | 排序后含 BF16 的 tile | 减少量 | 减少比例 |
|---|---:|---:|---:|---:|
| 全部 16 chunks | 457,470 | 131,621 | 325,849 | **71.228496%** |
| 后 15 chunks | 372,103 | 46,254 | 325,849 | **87.569571%** |
| 第 16 chunk | 4,518 | 2,326 | 2,192 | 48.517043% |

全部合成输入共含 65,012,736 个有效 selected token，其中
581,379 个属于 BF16 prefix/recent。排序后全 history tile 从
3,605,435 增到 3,931,284，增量同样是 325,849。首 chunk 的
BF16 tile 数在排序前后均为 85,367，与 `rowLen<=topK` 短行快捷路径
的源码预期一致。

该结果只支持将“启用 prefill selected index 排序”保留为下一筛选候选，
候选状态为 `screening_supported_runtime_unmeasured`。必须保留的边界是：

- selected index 来自 seed 42 的确定性随机生成，不是正式 DSA 输出；
- 本轮只统计了需要执行 BF16 dot 的 tile 数，没有测量 native top-k
  额外排序成本或 stage1 CUDA 时间；
- 本轮没有 output/LSE、TTFT、TPOT 或吞吐结果，不能宣称性能已改善。

小型证据已复制到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/prefill_sort_coverage_cpu_v1`。

证据 manifest 包含 9 项并已 9/9 通过复算；目录连同 manifest 共 10 份文件、
25,005 bytes。summary、candidate assessment、validation、run log、覆盖逻辑脚本、
source evidence、run identity 和退出后 GPU 快照的 SHA256 依次为：

- `871b03f49b16450e31c8ca6aa0390e76b2e4697e28f13d2c95bba8a90e63dace`；
- `3c4921b19daf4ffdd4fb56f7f91654c7eb1779b526510a5c554a78979f6d67c3`；
- `46f2cb3683dfd01e26a2cdfb6e45daea0708162ffb3bc0f32ab008e668752c59`；
- `0f7b1e7f9b4d5be83175dccd67d8fe6418a2d127bebcb6ccd1136d4567a90113`；
- `c9a69ee156b6373618ed657d09ce53b592c1c759f6fc5a89e598bddda5022890`；
- `f58f40079127726c6898ec539d41ccf966885059bd2c1fdfcc599dd6feea7639`；
- `4d9a9f950dbee345861be0cadd7fc6c23a7d3a5fa12b8dd500e37b0fd77ff1b9`；
- `ce84c9c98bef946537f1dd5f4a418cd7f1eaa2f0d6334de67123f0e66f644bb5`。

退出码文件 SHA256 为
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，证据 manifest
SHA256 为
`f2949866f90de61b963fa3c68a0cf2171effb294cd0b82cd452c4ecaacb1dbb0`。
`2026-07-31T20:18:21Z` 复查 8 张苹果800 均为 `0 MiB/0%`，没有
compute process。下一步先发布本节与 planning；发布前不修改性能工具
或正式配置。发布后再为冻结单层工具增加“对同一 selected 集合只重排”
的筛选口径，分别验证 output/LSE 和 stage1 CUDA 时间；该门禁通过前，
不进入正式 32K/batch1 端到端实验。

### 2.55 Prefill selected index 排序候选的工具化 CPU 门禁

2.54 与 planning 已由主仓库提交 `8e01df5` 发布；本阶段只扩展冻结的
`scripts/phase9/benchmark_oscar_prefill.py` 和定向测试，没有修改
OSCAR 生产源码、源码 submodule 或 `configs/phase9/performance_matrix.json`。
新增显式参数：

`--include-sorted-selected-indices`。

默认不传参数时，配置列表仍逐字保持为 `full_topk_split16` 与
`full_topk_split1` 两项。显式启用时，只增加第三项
`full_topk_split1_sorted_indices`；它与原始 split1 使用同一 selected token
集合、相同 top-k width 和相同 split 数，只把长行中的 index 按 token index
排序。因此后续能在同一轮中比较原始/排序 split1，并继续以原始 split16
作为 output/LSE 参考。

工具在输入构造阶段一次性生成 `selected_tokens_sorted`，计时循环只选择已经
生成的 tensor，不在 `call_attention()` 中执行 `torch.sort`。排序 helper
同时复现 2.54 的原生边界：当 `query_position+1<=topK` 时保持有效前缀和
`-1` 尾部原样；只有长行才排序。这一设计使下一次单层 GPU 筛选只测量
stage1 因 index 顺序变化产生的时间差，不把工具侧 PyTorch sort 混入 CUDA
计时。

TDD 的第一次容器命令直接调用候选 venv，但该 venv 没有 pytest，因此在
collection 前退出，不能计作有效红灯。把既有 uv 安装的
`pytest==8.3.5` target 只读挂载到固定 ca4a404e9 控制镜像后，有效红灯为
`4 failed, 4 passed`，四个失败分别覆盖 CLI、排序 config、短行快捷路径和
长行排序。实现后的定向结果为 `8/8 passed`；Phase 9 三个工具测试文件合计
`26/26 passed`、0 failed。

Ruff `0.14.0` 使用 uv 和清华 PyPI 源安装到任务专属 `/dev/shm` target。
验证过程保留了三个 fail-closed 边界：默认 Ruff cache 无法写只读仓库；改用
`/tmp/ruff-cache` 后，format check 实际发现主脚本需要格式化；格式化后，
Python compile 又因只读 `__pycache__` 失败。最终只机械格式化该主脚本，并
设置 `PYTHONPYCACHEPREFIX=/tmp/pycache`；Ruff check/format、固定 Python
compile、`26/26` pytest、CLI help 和 `git diff --check` 的有效组合退出码为
0。最终脚本与测试 SHA256 分别为：

- `f7d7b4836d9ace3000a0924085eda479849672eeffce9f583d29d5010be73f1e`；
- `0bac8d11af088e73f6647b597623d3eeef621911eadd5e6666f19c68f27be8c5`。

随后在同一控制镜像中执行 CPU-only 的 32K 末段语义审计，固定 Python
`3.12.13`、Torch `2.11.0+cu129`、seed 42、2,048 个 query、positions
`[30720,32768)` 和每行 2,048 个 selected index。轮次状态为 `passed`，
耗时 `89.43562247697264 秒`；共处理 4,194,304 个 index，并逐行验证排序
前后的 selected 集合完全相同。实际 tile 覆盖为：

| Coverage 指标 | 原始顺序 | 按 token index 排序 | 变化 |
|---|---:|---:|---:|
| 有效 selected token | 4,194,304 | 4,194,304 | 0 |
| BF16 selected token | 4,696 | 4,696 | 0 |
| 含 BF16 token 的 tile | 4,518 | 2,326 | -2,192（-48.517043%） |
| 全 history tile | 257,626 | 259,818 | +2,192 |

这些数值与 2.54 的第 16 chunk 完全一致，证明新工具复现了对应合成排序语义；
它仍不是正式 DSA selected 分布。CPU 阶段使用 `runc`、network none、4 CPUs、
空 `CUDA_VISIBLE_DEVICES` 和 `NVIDIA_VISIBLE_DEVICES=void`，没有注入 NVIDIA
runtime。结束后 8 张苹果800 均为 `0 MiB/0%`，没有 compute process。

小型证据位于：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/prefill_sort_benchmark_tool_cpu_v1`。

`cpu_32k_semantics.json`、`run_identity.json`、`tdd_validation.json` 和退出后
GPU 快照的 SHA256 依次为：

- `412e63fd11f59a0469dac32c70e3b49ef62480ac5dc5fbe67c98c529f56f5ff6`；
- `8d7c45a95b815411b5bf33261558802f66396864c06e1ebc531621ff5df55057`；
- `a9ce84895b2d13f2688e55bc2d47993614dc1d633c57122111d36f658b394183`；
- `bc163a6022190725bf0cd0b7bbc3a759c024b18eb07e9b4edddf0b039137d1fd`。

4 项证据已 4/4 通过 manifest 复算；目录连同 manifest 共 5 个文件、
3,955 bytes，manifest SHA256 为：

`441738faf490bf30b6fbcf043774e1801ef5d6dad8cda08501c9d1648b706365`。

本节没有 output/LSE、stage1 CUDA 时间、原生 CUB sort 成本、TTFT、TPOT
或吞吐结果。下一步先发布工具、测试、本节与 planning；两仓恢复
clean/published 后，再重新执行两次至少间隔 60 秒的 GPU 空闲检查，固定只用
GPU 0，在 2,048-query/32K-final、5 次 warm-up、7 个正式样本的同一单层协议
下比较原始/排序 split1 的 output/LSE 与 stage1 CUDA 时间。即使该筛选通过，
也必须在后续端到端候选中启用原生
`VLLM_TOPK_PREFILL_SORT_INDICES=1`，把 CUB sort 开销一起计入正式
32K/batch1 TTFT、TPOT 与吞吐；不能用本节或单层微基准替代端到端结果。

### 2.56 Prefill selected index 排序候选的单卡 stage1 筛选

2.55 的工具、测试、CPU 语义门禁与实时记录已由主仓库提交 `c0c7b44`
发布，发布状态由 `9ccad9d` 固化。GPU 分配前的双空闲状态由提交
`05f0bef4318b6a11528a697123045602840b6ea2` 发布；进入有效轮次时，主仓库
本地 HEAD 与 upstream 均为该提交，源码仓库本地 HEAD 与 upstream 均为
`ca4a404e913ce55237ca60383cc86e221fbfea26`。

外层空闲检查时间为 `2026-07-31T20:41:52Z/20:43:14Z`，间隔 82 秒；
两次均为 8/8 张苹果800 `0 MiB/0%`，没有 compute process。唯一运行的
项目外下载容器没有 GPU DeviceRequests，因此未终止。有效轮次为：

`20260731T2044Z_prefill_sort_2k_gpu_v1`。

轮次固定只使用 GPU 0，控制镜像为
`oscar-glm-stage9-runtime:ca4a404e9`，完整 image ID 为
`sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`；
Docker 使用 NVIDIA runtime、network none、4 CPUs 和独立空 Triton cache。
冻结负载为 2,048 个 query、final sequence 32,768、positions
`[30720,32768)`、top-k 2,048、8 个本地 head、seed 42，每项 5 次
warm-up、7 个正式样本、每样本 1 次迭代。排序前后使用完全相同的 selected
token 集合；selected tensor 在计时前只生成一次。

轮次一次通过，Docker 退出码为 0。实测中位数如下：

| 配置 | CUDA 中位数（ms） | Wall 中位数（ms） | BF16 tile 数 |
|---|---:|---:|---:|
| 同轮 split16 参考 | 339.247101 | — | — |
| 原始顺序 split1 | 20.206593 | 20.233033 | 4,518 |
| 按 token index 排序的 split1 | **19.581951** | **19.609085** | **2,326** |

排序 split1 相对原始 split1 的 CUDA 中位数减少
`0.624641 ms`（`-3.091275%`，加速 `1.031899×`），wall 中位数减少
`0.623948 ms`（`-3.083808%`）。原始 CUDA 样本范围为
`20.166656–20.531200 ms`，排序后为 `19.561472–19.602432 ms`。
BF16 tile 数减少 `2,192`（`-48.517043%`），但 stage1 时间只减少
`3.091275%`；因此 BF16 tile coverage 只是工作量代理指标，不能按比例外推
CUDA 或端到端收益。

原始与排序 split1 都相对同轮 split16 参考通过冻结的
`torch.allclose(atol=0.002, rtol=0.002)`。原始 split1 的 output/LSE
max_abs 为 `0.0033702850341796875/0.0022249221801757812`，max_rel 为
`132.1691436767578/0.00023601796419825405`；排序 split1 对应为
`0.0034520626068115234/0.0022249221801757812` 和
`150.6269073486328/0.00023601796419825405`。两组 combined allclose
均为 `passed`；output max_abs 大于 `0.002` 不与通过矛盾，不能将本结果
误写为单项 `max_abs<=0.002`，也不能声称逐 bit 一致。

实际 runtime cold compile 的 `_mixed_sparse_prefill_stage1` 使用 8 warps、
1 stage，dynamic shared memory 为 109,568 bytes、每线程 242 registers、
0-byte stack；cubin 为 142,000 bytes。cubin 与 metadata SHA256 分别为：

- `e27aba860f4e5203640341916eb1b65fb7df5b266c06240d4881cba1d89377b2`；
- `c02d98e3d720bf9185b9e51e4f0e6626d224df3caea135cfb3d8a73296dd3d76`。

独立 cold cache 共 61 个文件，`du -sb` 为 2,951,353 bytes；cache 文件
清单 SHA256 为
`3114b4f5e7edb44eb2484248e2704d4138e13672f98c17b8ef562a24a5f4dba3`。
实验容器已自动删除，退出后 8 张 GPU 均为 `0 MiB/0%`，没有 compute
process。

小型证据位于：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/prefill_sort_2k_gpu_v1`。

目录内 17 项证据已 17/17 通过 manifest 复算；连同 manifest 共 18 个文件、
189,482 bytes。result、run log、comparison、resource summary、run identity、
container inspect 和 manifest 的 SHA256 依次为：

- `29c54689dc1e3361152e6b97bf889e547bcde931ef902da025c11f623a007d24`；
- `8a8711e920dca481c31e23bc5a124bfe8be1a51e2711d5190de0f9c925cc2b9a`；
- `66e5fa0c6182ee84358c5348790832abfe536aa81f8ff21773dc1e8ed0919c82`；
- `99ae6a7d7609d204e5d1db8b764bb4a6e9ae3897e4fc83fc298a266e02900d70`；
- `fa9f1333cff4a0afdce2b9e5006023ec4de58013f19b8c3981742ad1517b7245`；
- `f5e27c67aa9a202b9077dc54a536a431bdef871711eb2796898cbbd980236be4`；
- `62084dacb21de5cfc236731366a18d60df7a25c6f153563b6038a9f692e2efe2`。

必须保留的结论边界是：本轮在计时前预先完成 selected index 排序，只测得
index 顺序变化对 stage1 的影响；没有计入生产路径
`topKPerRowPrefill` 的 CUB 排序成本，也没有产生新的 TTFT、TPOT 或吞吐
结果。候选状态因此为
`stage1_screen_passed_native_sort_cost_unmeasured`，不能宣称已经取得净性能
收益。下一步先发布本节与 planning；主仓库恢复 clean/published 后，先在
精确 32K prefill 几何下单独测量原生 sorted/unsorted top-k 成本。只有组合
成本仍支持净收益时，才最小启用生产环境变量并进入正式 32K/batch1 验证。

### 2.57 原生 prefill top-k 排序成本工具的 CPU/TDD 门禁

2.56 与 planning 已由主仓库提交
`810593beb59cd2af49f13497759752e25609b922` 发布，发布状态由提交
`b412ef0` 固化。进入本阶段时，主仓库本地 HEAD 与 upstream 均为
`b412ef0`；源码仓库本地 HEAD 与 upstream 均为
`ca4a404e913ce55237ca60383cc86e221fbfea26`，tree 为
`079815219a02add3f37318ed434924e80f80a35d`。

本阶段新增独立工具
`scripts/phase9/benchmark_topk_prefill_sort.py` 及其定向测试，没有修改
OSCAR 生产源码、源码 submodule 或
`configs/phase9/performance_matrix.json`。只读源码检查确认，原生绑定
`torch.ops._C.top_k_per_row_prefill` 位于 `csrc/torch_bindings.cpp`，CUDA
实现位于 `csrc/sampler.cu`；当 `VLLM_TOPK_ENV_CACHE=0` 时，每次调用读取
`VLLM_TOPK_PREFILL_SORT_INDICES`。当前正式 Phase 9 配置没有设置这两个变量。

工具固定复现 32K 最后一个 2K prefill chunk 的几何：2,048 个 query、final
sequence 32,768、2,048×32,768 的 FP32 logits、全部 row start 为 0、row end
为 `[30721,32768]`、top-k width 2,048、seed 42。默认计时协议为每项 5 次
warm-up、7 个正式样本、每个样本连续调用 20 次后折算单次时延。工具在导入
`vllm._C` 前强制设置 `VLLM_TOPK_ENV_CACHE=0`，随后对同一份 logits 依次设置
排序变量为 0 和 1；输出同时记录 CUDA event 与 wall timing。

语义门禁把 unsorted 输出逐行排序后与 sorted 输出做精确相等比较，并检查
sorted 输出逐行单调、final chunk 中负 index 数为 0。因此未来 GPU 微基准若
通过，能够证明两种模式保留同一 selected index 集合；它仍不会证明正式 DSA
logits 下的 output/LSE 或端到端精度。

TDD 有效红灯发生在目标脚本尚不存在时，pytest collection 为 1 error；这是真实
缺失实现，不是环境故障。完成最小实现后，新增定向测试为 `6/6 passed`。第一次
完整组合在 Ruff 阶段发现 import、typing 和行长问题并 fail-closed；宿主首次
机械修复没有显式加载源码仓库的严格 `pyproject.toml`，控制容器复查仍发现
3 项 import/typing 问题。随后显式使用
`glm52_oscar_vllm/pyproject.toml` 完成机械修复，固定 ca4a404e9 控制镜像内的
最终组合结果为：

- Ruff check 与 format check 通过；
- 固定 Python compile 与 CLI help 通过；
- Phase 9 四个工具测试文件合计 `32/32 passed`、0 failed；
- 仅出现 2 条只读 pytest cache warning，不影响测试结论。

最终工具与定向测试 SHA256 分别为：

- `3ca80e410911542bec7e2bc38dda67d5bac44315bcc7d73c35b87023e9c581fc`；
- `3b936e997b8b9384df0c829a1f0687802dadb8f6c45df0a94f5f9f46b9408839`。

上述验证均为 CPU-only 工具门禁：没有注入 NVIDIA runtime，没有为本阶段分配
GPU，也没有执行 `top_k_per_row_prefill` 原生 CUDA 算子。因此本节没有
sorted/unsorted CUDA 时延、stage1 净收益、TTFT、TPOT 或吞吐结果，不能据此
判定排序候选有效。下一步先发布工具、测试、本节与 planning；两仓恢复
clean/published 后，重新执行两次至少间隔 60 秒的 GPU 空闲检查，再固定只用
GPU 0 按上述协议实测原生排序成本。只有原生排序额外成本小于 2.56 测得的
单层 stage1 收益 `0.624641 ms`，并且结果语义门禁通过，才考虑最小启用生产
环境变量；最终性能结论仍必须来自后续正式 32K/batch1 端到端轮次。

### 2.58 原生 prefill top-k GPU 微基准首轮启动失败

2.57 的工具、测试、报告与 planning 已由主仓库提交
`b3f2159a3771a40949dfab7aaf75be6e9e0dc654` 发布；发布状态、双空闲门禁和
启动前即时状态随后分别由 `9af381b`、`e012df3` 和 `04a303b` 发布。GPU
双空闲检查时间为 `2026-07-31T21:07:42Z/21:08:55Z`，间隔 73 秒；两次均为
8/8 张苹果800 `0 MiB/0%`、无 compute process。唯一运行的项目外下载容器
`deepseek_v4_hf_downloader_vllm0230` 的 DeviceRequests 为 null，不占 GPU，
因此未终止。`21:09:34Z` 启动前即时检查 GPU 0 仍为空闲。

首轮 run ID 为：

`20260731T2110Z_topk_prefill_sort_2k_gpu_v1`。

轮次固定只向容器暴露 GPU 0，控制镜像为
`oscar-glm-stage9-runtime:ca4a404e9`，完整 image ID 为
`sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`；
原计划保持 2,048×32,768 logits、top-k 2,048、5 次 warm-up、7 个样本和
每样本 20 次调用的冻结协议。该轮从 `21:10:22Z` 到 `21:10:26Z`，Docker
退出码为 1。

失败发生在导入 `vllm._C` 的 Torch Dynamo 初始化阶段。容器命令显式使用宿主
UID 22633，但镜像 `/etc/passwd` 中没有该 UID；`getpass.getuser()` 调用
`pwd.getpwuid(os.getuid())` 时抛出：

`KeyError: 'getpwuid(): uid not found: 22633'`。

因此该轮没有进入 `top_k_per_row_prefill` 原生 CUDA 算子，没有执行 warm-up
或正式计时，也没有生成 `result.json`。它不包含 sorted/unsorted 时延、语义
门禁、stage1 净收益、TTFT、TPOT 或吞吐数据，不能纳入性能对比。容器已由
`--rm` 删除；`21:10:51Z` 复查 8 张 GPU 全部为 `0 MiB/0%`，没有 compute
process。

失败证据已复制到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/topk_prefill_sort_native_gpu_v1_failed`。

目录中的 control image ID、起止时间、退出码、退出后 GPU/compute 快照和
run log 共 7 项，已 7/7 通过 manifest 复算；连同 manifest 共 8 个文件，
`du -sb` 为 4,247 bytes。run log 与 manifest SHA256 分别为：

- `0464c950d6b632e949b7353b2890d3a3104558515ff0768625c3b1e784d38e70`；
- `0a21133d5220bbfe83c5666107710f9f928d04628eb3bf8b123b37886297cd84`。

manifest 首次从仓库根目录校验时，相对文件名被解析到错误目录，7 项均报
`FAILED open or read`；这不是证据内容哈希失败。随后进入证据目录执行同一
manifest 校验，7/7 全部通过。下一轮不重复错误的用户映射：使用镜像默认 root，
同时保持 GPU、镜像、负载、warm-up 和样本协议不变，并使用新 run ID。按照
实时记录门禁，必须先发布本节与 planning，才能启动修正后的轮次。

### 2.59 原生 prefill top-k 排序成本的单卡 GPU 门禁

2.58 与 planning 已由主仓库提交
`3f2682cc91079f83f4b892b0dcaedee4820efb07` 发布；修正轮次的新双空闲状态由
提交 `8fbf4af4a856f82c7cd0077a5a16a9f68d1b9532` 发布。新双检使用 v1 退出后的
`2026-07-31T21:10:51Z` 复查与 `21:13:40Z` 新检查，间隔 169 秒；两次均为
8/8 张苹果800 `0 MiB/0%`、无 compute process，主仓库与源码仓库均为
clean/published。

有效轮次为：

`20260731T2114Z_topk_prefill_sort_2k_gpu_v2`。

该轮只移除 2.58 中导致失败的宿主 UID 映射，改用镜像默认 root；其余协议保持
不变。轮次固定只使用 GPU 0，控制镜像仍为
`oscar-glm-stage9-runtime:ca4a404e9`，image ID 为
`sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`，
network none、4 CPUs、8 GiB shared memory。冻结负载为 2,048×32,768 FP32
logits、final sequence 32,768、positions `[30720,32768)`、row end
`[30721,32768]`、top-k 2,048、seed 42；unsorted 和 sorted 分别执行 5 次
warm-up、7 个正式样本、每个样本 20 次调用并折算单次时延。

轮次从 `21:14:24Z` 到 `21:14:28Z`，Docker exit=0、status=passed。原生
`top_k_per_row_prefill` 的实测结果为：

| 模式 | CUDA 中位数（ms） | CUDA 样本范围（ms） | Wall 中位数（ms） |
|---|---:|---:|---:|
| unsorted | 0.368998408 | 0.368537593–0.370329595 | 0.370009430 |
| sorted | 0.542361593 | 0.542105579–0.542668819 | 0.543393847 |

sorted 相对 unsorted 的 CUDA 中位数增加 `0.173363185 ms`
（`+46.982096%`），wall 中位数增加 `0.173384417 ms`
（`+46.859459%`）。百分比看似较大，是因为原始 top-k 本身不足 0.4 ms；决定
候选是否值得继续的绝对排序成本是 `0.173363185 ms`。

语义门禁全部通过：把 unsorted 输出逐行排序后与 sorted 输出精确相等；sorted
输出逐行单调；invalid index count 为 0。该结论只证明同一份合成 logits 上的
selected index 集合不变，不是正式 DSA logits 或完整 output/LSE 精度结果。

将本轮原生 top-k 成本与 2.56 的独立 stage1 微基准做算术组合，可得到筛选级
估算：

| 分离微基准组合 | 原始路径（ms） | 排序路径（ms） | 变化 |
|---|---:|---:|---:|
| stage1 + native top-k | 20.575590968 | 20.124312735 | -0.451278234（-2.193270%） |

其中 2.56 的 stage1 节省为 `0.624641418 ms`，本轮原生排序成本消耗其
`27.754033%`，剩余估算净收益 `0.451278234 ms`。这两个数来自不同的独立合成
微基准，只能用于候选筛选，不能视为同一执行流中的融合测量，也不能按 1,248 次
stage1 调用直接线性外推 TTFT。

小型证据已复制到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/topk_prefill_sort_native_gpu_v2`。

目录中的 result、comparison、run log、镜像/Git 身份、起止/退出码、启动前和
退出后的 GPU/compute 快照共 13 项，已 13/13 通过 manifest 复算；连同
manifest 共 14 个文件，`du -sb` 为 8,822 bytes。result、comparison、run log
和 manifest SHA256 分别为：

- `a5c447b2dab8531afb5fbd12afa495c5286de4d7e394e898bf5fe26fdb0975d9`；
- `ead985dffb732c9ee254c0add29c75978b472cbe09899d2bbc692b472f3f951c`；
- `201c078c4d9c354678a6506c1b3d57ef8c60877aed96591762a316c17f77f152`；
- `ca6ba33919cc5e5ed135aa9a075f11cd9c6f43c4b9921269c27340f3e232d2da`。

实验容器已删除；`21:14:55Z` 复查 8 张 GPU 均为 `0 MiB/0%`，没有 compute
process。本轮候选状态为 `native_sort_cost_gate_passed_formal_unmeasured`：
原生排序成本小于单层 stage1 收益，支持进入最小生产配置候选；但本轮没有正式
DSA selected 分布、TTFT、TPOT 或吞吐结果。下一步先发布本节与 planning，
再只在正式 Phase 9 环境中启用 `VLLM_TOPK_PREFILL_SORT_INDICES=1`，完成静态/
CPU 门禁、提交推送和新的 GPU 双空闲检查后，复跑同一 32K/batch1 正式负载。

### 2.60 Prefill top-k 排序的 candidate-only 正式配置门禁

2.59 与 planning 已由主仓库提交
`87960878d58bb00d6bad248afb22564937c8a01d` 发布；源码仓库继续保持
`ca4a404e913ce55237ca60383cc86e221fbfea26` clean/published。本阶段没有修改
CUDA/C++ 生产源码、源码 submodule、控制镜像或 BF16 baseline wrapper，只修改
正式 Phase 9 candidate 的配置传播与验证链路。

`configs/phase9/performance_matrix.json` 新增唯一的 candidate-only 映射：

```json
"candidate_runtime_environment": {
  "VLLM_TOPK_PREFILL_SORT_INDICES": "1"
}
```

配置 SHA256 从
`14d3f71e8a36b429d31e78ff8a7c0a9810048a308440693aef08b2ae9aea4760`
变为
`22ecca75b358aff3e04a3280d69a288239925f586f938a1d7e03dcf59e37b219`。
`scripts/phase9/run_candidate_tp8.sh` 在进入 Phase 7 基础 wrapper 前从该 JSON
读取映射，要求它与上述单键字典精确相等后才 export；配置缺失、键多余或值不是
字符串 `"1"` 都会 fail-closed。`scripts/phase9/run_native_tp8.sh` 没有该变量，
因此 BF16 baseline 不受影响。

Phase 1 基础 serve 路径原本已经 export `VLLM_TOPK_ENV_CACHE=1`。candidate
wrapper 在调用它之前设置排序变量，所以原生扩展 load-time cache 会固定读取
排序开关。该传播方式与 2.59 为在同一进程切换两种模式而强制 cache=0 的
微基准不同，但两者执行的是同一原生 CUDA 排序路径；2.59 的绝对排序成本可作
候选筛选，正式性能仍只能由后续端到端轮次给出。

`scripts/phase9/verify_candidate_performance.py` 新增两层 fail-closed 检查：先
要求配置中的 candidate runtime environment 精确等于上述单键映射，再逐键比较
实际 `os.environ`。既有 Phase 1 wrapper 会把全部 `VLLM_*` 写入
`runtime_environment.txt`，并把该文件 SHA256 记录在 `runtime_manifest.json`，
因此无需修改 manifest 格式即可保留正式证据。baseline verifier 与 wrapper
均保持不变。

TDD 有效红灯为 1 failed：测试在正式配置缺少
`candidate_runtime_environment` 时精确触发 `KeyError`。完成 config→candidate
wrapper export→candidate verifier 三点最小实现后，同一定向测试为
`1/1 passed`。

首轮完整 CPU 门禁在 Ruff 首步 fail-closed；4 项诊断均为既有代码：两个文件的
历史 import block I001，以及测试文件中两条 127 字符 profiler 表头 E501，均
不在本次 diff。遵守最小修改约束，本阶段没有重排整文件或改写历史测试数据。
第二轮在显式忽略这 4 项已确认的历史规则后，Ruff check 通过，但 format check
发现本次新增测试块需要格式化并停止。Ruff formatter 随后只改写新增测试块；
最终 CPU-only 组合结果为：

- Ruff 其余规则与 format check 通过；
- JSON parse、candidate shell syntax、固定 Python compile 和
  `git diff --check` 通过；
- Phase 9 四个工具测试文件合计 `33/33 passed`、0 failed，耗时 4.35 秒。

最终 config、candidate wrapper、candidate verifier 和测试 SHA256 分别为：

- `22ecca75b358aff3e04a3280d69a288239925f586f938a1d7e03dcf59e37b219`；
- `ec849660e7e0505980b2f43edaf1bf64a4901cc2b391db33b6bb14fe71589e05`；
- `eb89bb1f5f8d834839b395de2281d4e6b1682d9cb779737b0e3050acff7e6104`；
- `29ddd91cde1ac210f2de9eb79509ca8e9abdaae76847322724ced053271fc992`。

上述组合使用固定 ca4a404e9 控制镜像、network none、4 CPUs，未注入 NVIDIA
runtime，也没有分配 GPU。本节只证明 candidate-only 配置传播在静态/CPU
层面可审计，不证明服务能够启动、runtime environment 已实际写入，亦没有新的
TTFT、TPOT、吞吐或精度结果。下一步先发布配置、wrapper、verifier、测试、
本节与 planning；两仓恢复 clean/published 后重新执行双空闲检查，再运行
driver-injected candidate preflight。preflight 必须同时通过既有 64/64 静态
门禁、固定环境/服务参数解析和新增 runtime environment 检查，之后才能启动
正式 32K/batch1。

### 2.61 Prefill top-k 排序候选的正式 driver-injected preflight

2.60 的 candidate-only 配置链路、测试、报告与 planning 已由主仓库提交
`8b347e1dadf2f89dad370b9cecdb4f01af2dd5cf` 发布，发布状态由
`b6e1cf744ceeaf18cec83a6367e8bf2ba3c9d526` 固化。preflight 双空闲状态由
提交 `c9322e9e889b1ea19a714a99622837af859e2d80` 发布；两次有效检查时间为
`2026-07-31T21:25:00Z/21:26:18Z`，间隔 78 秒，8/8 张苹果800 均为
`0 MiB/0%`、无 compute process。外部下载容器 DeviceRequests=null。

有效 preflight run ID 为：

`20260731T2127Z_candidate_topk_sort_preflight_v1`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:ca4a404e9`，image ID 为
`sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`；
driver-injected 容器固定暴露 8 张 GPU，但不加载模型。轮次从
`21:27:10Z` 运行到 `21:29:09Z`，耗时 119 秒，Docker exit=0。

静态 verifier 状态为 passed，共 `66/66` checks passed：在既有 64 项
source/candidate/server/matrix/frozen evaluator 检查之上，新增两项均通过：

- `performance.candidate_runtime_environment` 的实际值和期望值均为
  `{"VLLM_TOPK_PREFILL_SORT_INDICES":"1"}`；
- `runtime_environment.VLLM_TOPK_PREFILL_SORT_INDICES` 的实际值和期望值均为
  字符串 `"1"`。

固定环境导入通过，Python/Torch/Triton 分别为
`3.12.13/2.11.0+cu129/3.6.0`，候选 `vllm._C` 来自冻结 overlay；
`cuda_initialized=false`。服务参数解析也通过：TP=8、PP=1、
`max_model_len=131072`、`max_num_batched_tokens=2048`、
`kv_cache_dtype=oscar_mla_int2`、`TRITON_MLA_SPARSE`、eager、chunked prefill
与 async scheduling=false 均符合正式配置，且同样
`cuda_initialized=false`。

监控过程中曾在静态 JSON 打印 passed 后过早判断整轮已经完成；当时
`fixed_environment_import.json` 仍为 0 bytes，尚无 exit/end 文件，Docker
仍在运行。该判断随即被更正。之后 fixed import 增至 804 bytes 并证明
CUDA=false，parsed args 先为 0 bytes、随后增至 644 bytes；只有确认
exit=0、end 时间和容器删除后，才把整轮标记为 passed。该过程没有终止或重跑
有效 preflight。

小型证据已复制到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/topk_sort_candidate_preflight_v1`。

目录中的 static/fixed/args、validation、run log、镜像/Git/配置身份、起止时间、
退出码和前后 GPU/compute 快照共 14 项，已 14/14 通过 manifest 复算；连同
manifest 共 15 个文件，`du -sb` 为 43,485 bytes。static、fixed、args、
validation、run log 与 manifest SHA256 分别为：

- `93d4507fc5d85c625a853ac8752de8f6286c031b63d10867521647b0ecc690a6`；
- `56f92356322376f98d88bbbe4786eed46e4a590a666fc57143ba96326c250574`；
- `227f21f6725c7a0a966f4bedc074867dd231c153dd2e284d9149703167e53236`；
- `ee3a64dcd9916f2fe609dc5da629ca50a03f3f494a19e37f87f6dd3b60b1d57a`；
- `916b72c9eecc2596f20a25be751dc88e54430dd27a4b2653650c231d139314d3`；
- `a93f4dc3ac7be1122404a3fa29303d303d27f32a734298f1a19abf323c52ad3b`。

容器已删除；`21:29:57Z` 复查 8 张 GPU 均为 `0 MiB/0%`，没有 compute
process。preflight 全程不足 10 分钟，没有触发 10 分钟进度打印门槛。本节只
证明正式 candidate 环境传播、固定依赖和服务参数可在 CUDA 未初始化前通过
门禁，没有加载模型，也没有新的精度、TTFT、TPOT 或吞吐结果。下一步先发布
本节与 planning；两仓恢复 clean/published 后，为正式 32K/batch1 新轮次重新
执行双空闲检查，再按 2.52 的同一三轮加 8+8+1 profiler 协议运行并与 BF16、
ca4a404e9 未排序结果比较。

### 2.62 Prefill top-k 排序候选的 32K/batch1 正式结果

2.61 已由主仓库提交
`139d5254d359b382a7a79bffe0be48fc18a357aa` 发布。正式轮次启动前的状态提交为
`cf1ad69b29f9a63e8276ec794530952dd1fd9b50`；两次有效 GPU 空闲检查时间为
`2026-07-31T21:32:45Z/21:34:05Z`，间隔 80 秒，8/8 张苹果800 均为
`0 MiB/0%`、无 compute process。源码仓库保持
`ca4a404e913ce55237ca60383cc86e221fbfea26` clean/published。

正式 run ID 为：

`20260731T2135Z_candidate_topk_sort_32k_b1_v1`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:ca4a404e9`，image ID 为
`sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`；
固定使用 8 张 GPU，负载为 input length 32,768、batch size 1、output length
128，按与 2.52 相同的三轮加 8 tables、8 worker traces、1 frontend trace
profiler 协议执行。`runtime_environment.txt` 实际记录
`VLLM_TOPK_PREFILL_SORT_INDICES=1` 和 `VLLM_TOPK_ENV_CACHE=1`，排序开关已在
正式服务进程加载前生效。轮次从 `21:34:50Z` 运行至 `22:13:27Z`，Docker
exit=0，summary/cell/profile 均为 passed；matrix 部分耗时
`1675.581609249115 s`。

三轮均为 3/3 completed、0 failed，实际结果为：

| 轮次 | mean TTFT（ms） | mean TPOT（ms） | 请求吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 32434.97844568143 | 199.15507386714768 | 0.017322629236321652 |
| 2 | 32449.24456657221 | 198.4476374286249 | 0.01734534049545483 |
| 3 | 32456.20973625531 | 199.49260165333123 | 0.01730342414633551 |

三轮中位汇总为：mean TTFT `32449.24456657221 ms`、mean TPOT
`199.15507386714768 ms`、请求吞吐 `0.017322629236321652 req/s`；同时
median TTFT/TPOT 为 `32449.20253008604/198.86147414928112 ms`，output/total
token throughput 为 `2.2172965422491715/569.845211358037 token/s`。

与 2.52 的未排序 ca4a404e9 正式结果以及同一 32K/batch1 BF16 baseline 比较：

| 对照 | mean TTFT 变化 | mean TPOT 变化 | 请求吞吐变化 |
|---|---:|---:|---:|
| 未排序 ca4a404e9 | -0.715420841% | -0.440883283% | +0.578794357% |
| BF16 baseline | +159.013232667% | +11.364501492% | -38.955186240% |

因此，排序候选在正式负载上带来了方向一致但很小的改善：TTFT/TPOT 分别下降
约 `0.715%/0.441%`，吞吐提高约 `0.579%`。该收益明显小于 2.59 中两个分离
微基准算术组合得到的 `-2.193270%`，说明合成单层收益不能直接外推到完整服务。
相对 BF16 的 TPOT 仍在 `+20%` 门限内，但 TTFT 仍高 `159.013%`，远未通过
门限；当前瓶颈不能仅靠 selected index 排序解决。

profiler 状态为 passed，实际生成 8 张算子表、8 个 worker trace 和 1 个
frontend trace；critical rank=4，critical-rank kernel total=`63657 ms`，
profile 耗时 `764.1544263362885 s`。正式三轮峰值显存为 `80679 MiB/GPU`，
profile 峰值为 `80691 MiB/GPU`。summary、comparison 与 outer log SHA256
分别为：

- `c16c9596e9e9ae5d77b4cfc2205cf4f212cf35fa0899f54c56678b3ed46856c6`；
- `ca50cba28ba530023e9c0788c6107559902ee145922d27f675cb15455c5c8bde`；
- `384f7e9250cd942789df2af507d74911db84a14ecfb1dc1ef00fa656bfd2fca8`。

长实验进度按 10 分钟门限输出：启动监控在 `21:44:56Z` 打印 601 秒；服务监控
在 `21:47:23Z/21:57:24Z/22:07:24Z` 分别打印 600/1201/1801 秒；profile
在 `22:10:31Z` 打印 600 秒。实验容器已删除，`22:13:56Z` 复查 8 张 GPU
均为 `0 MiB/0%`、无 compute process。

小型证据已复制到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/formal_topk_sort_32k_b1_results`。

目录封存 52 项证据，已从仓库根目录 52/52 通过 manifest 复算；连同
`evidence_manifest.sha256` 共 53 个文件，`du -sb` 为 1,156,489 bytes。
manifest 与 `formal_validation.json` SHA256 分别为：

- `770166b54bd3a3a2a430533afc3ad1451f7750c53ad53494891773b20de6a198`；
- `467dfa2911f36f3e4e16c13a184b7875cbf4e8d2e82df6628f2a333348b85220`。

约 1.2 GiB 的原始 trace 仍保留在 `/dev/shm`，没有复制进仓库；正式 summary
已重新哈希并验证全部 8+8+1 profiler 输出。本轮没有修改模型、数据集或精度
配置，也没有产生新的 GSM8K 精度结果；性能比较只适用于上述 32K/batch1
单请求负载。下一步先发布本节与 planning，再使用既有 CPU-only trace 工具对
排序与未排序正式 trace 做阶段归因，确认实际 stage1/top-k 分布后再决定下一项
性能改动。

### 2.63 Prefill top-k 排序候选的 32K 多 chunk trace 归因

2.62 的正式结果与 planning 已由主仓库提交
`d38dfde4ea5d1a4af1d575cae48fa56e38528d7d` 发布，发布状态由
`8411b80ba18d130e9b667d1bf49c9c68515560be` 固化。本阶段没有修改源码、模型、
数据集、正式配置或控制镜像，也没有分配 GPU；只对 2.62 的 8 份正式 worker
trace 与 2.53 的未排序 ca4a404e9 有效 trace summary 做 CPU-only 对比。

有效 analysis ID 为：

`20260731T2228Z_topk_sort_32k_prefill_trace_v1`。

分析固定使用 `oscar-glm-stage9-runtime:ca4a404e9`、runc、network none、4 个
CPU worker，Python/ijson 精确为 `3.12.13/3.4.0.post0`；analyzer SHA256 为
`8b6b2393f93be3783f47ccbe3ecb26020cc2b65526c99fcb464c4768ce4330f7`。
有效分析从 `22:28:17Z` 到 `22:30:09Z`，analyzer 实测耗时
`111.15939520858228 s`，exit=0、summary/validation 均为 passed。8/8 ranks
的 trace 大小与 SHA256 均和 2.62 正式 summary 精确一致；每个 rank 均包含
144 个 execute context、16 个 prefill chunk 和 32,768 个 prefill token。

旧持久 venv 已变成指向 `/usr/bin/python3.12` 的断链，不能复用。新的任务专用
venv 由固定控制容器内的 uv 创建；首次 offline 安装因缓存不能解析固定 ijson
版本而退出，发生在读取 trace 之前。随后使用清华 PyPI 安装
`ijson==3.4.0.post0`，并在独立 network-none 容器中验证精确版本后才启动有效
分析。无效环境尝试没有生成或混入 summary。

排序与未排序的 profile-to-profile 中位结果为：

| 指标 | 未排序 ca4a404e9 | 排序候选 | 变化 |
|---|---:|---:|---:|
| prefill wall（ms） | 32756.591502 | 32478.967757 | -277.623745（-0.847536%） |
| prefill kernel（ms） | 31755.930453 | 31481.247675 | -274.682777（-0.864981%） |
| `_mixed_sparse_prefill_stage1`（ms） | 20128.143242 | 19849.393880 | -278.749362（-1.384874%） |
| `topKPerRowPrefill`（ms） | 221.593157 | 251.608034 | +30.014877（+13.545038%） |
| stage1 + top-k（ms） | 20349.736399 | 20101.001914 | -248.734485（-1.222298%） |

stage1 调用数保持 1,248，top-k 调用数保持 1,344；top-k 模板从未排序的
`<512, false, false>` 变为排序的 `<512, false, true>`。因此排序没有让 top-k
本身变快，反而增加 `30.014877 ms`；收益来自排序后的 selected index 改善后续
stage1 路径，stage1 减少 `278.749362 ms`。stage1 节省解释 profile prefill
wall 改善的 `100.405447%`，top-k 额外成本消耗 stage1 节省的
`10.767694%`；两者合计净省 `248.734485 ms`，解释 wall 改善的
`89.594096%`。去掉 stage1 与 top-k 后，剩余 wall 仍减少 `28.889261 ms`
（`-0.232849%`）。

逐 rank 方向也一致：8/8 rank 的 prefill wall 和 stage1 都下降，8/8 rank 的
top-k 都上升。wall delta 范围为 `-279.001862–-275.514826 ms`，stage1 delta
范围为 `-286.008914–-235.569612 ms`，top-k delta 范围为
`+29.019011–+30.693986 ms`。这排除了“只由单个 critical rank 偶然改善”这一
解释。

该 profile-to-profile wall 改善为 `277.623745 ms`，与 2.62 三轮正式 mean
TTFT 改善 `233.821466 ms`（`-0.715421%`）方向一致，但两者不是同一统计量，
不能要求数值相等。排序后 stage1 仍为 `19849.393880 ms`，占 prefill wall
`61.114608%`，仍是绝对主瓶颈。因此当前结论是：保留 candidate-only 排序开关
有实测依据，但它只关闭了很小一部分 TTFT 差距；下一优化仍应针对全部 16 个
chunk 的 stage1 有效工作，而不是继续压缩仅占 wall `0.774680%` 的 top-k。

小型证据已复制到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/formal_topk_sort_trace_analysis_v1`。

目录中的 summary、comparison、validation、run log/identity、起止/退出码、
trace inputs hash 和容器状态共 10 项，已 10/10 通过 manifest 复算；连同
manifest 共 11 个文件，`du -sb` 为 371,268 bytes。summary、comparison、
validation 与 manifest SHA256 分别为：

- `96c1a755c54775c2da1ccb4db7a7a7f989126a180957bb3a9ae4addfaaa8a9be`；
- `23b43324a390f5ea8d9cec5ee8e25b8309027033c161e61b6ce3e7a4b1021ed1`；
- `ac04e1b5a05a7e0580dc43d866d64472f80c18b9cd3fd695fb686df1cc483630`；
- `378404730e3553077121fca07019ea18e802a482a7782bfa034f582e9e053aff`。

原始约 1.2 GiB trace 没有复制进仓库，仍保留在 `/dev/shm`，并由正式 summary
与本轮 `trace_inputs.sha256` 双重关联。分析后仅有外部下载容器在运行，其
DeviceRequests=null。本阶段没有新的 GSM8K 精度结果。下一步先发布本节与
planning，再只读检查 stage1 的 16-chunk 工作分布和现有源码路径；形成下一个
最小候选前不修改源码、不启动 GPU 实验。

### 2.64 32K 逐 chunk kernel 归因工具的 CPU/TDD 门禁

2.63 与 planning 已由主仓库提交
`2df5f5727f7eeb0335451166c29103cb0f9cca74` 发布，发布状态由
`afc66313ece07f271573dedf7faf4311b4ba6c3c` 固化。两仓在本阶段开始前均为
clean/published。本阶段只修改主仓库的 trace analyzer 与单元测试，没有修改
OSCAR 运行时源码、模型、正式配置或控制镜像，也没有分配 GPU。

2.63 已证明排序后 stage1 仍占 prefill wall `61.114608%`，但 format version 2
的 `analyze_prefill_trace.py` 在每个 chunk 上只记录 name、tokens 与 wall
duration；stage1、top-k 和其他 kernel 只按 16 个 chunk 整段聚合。因此，现有
证据无法判断首块约 1.340 秒到末块约 2.196 秒的增长由哪个 kernel 贡献，不能
据此直接选择下一个运行时候选。

本阶段把 analyzer format version 从 2 升至 3，并在每个
`prefill.chunks[*]` 中新增：

- `kernel_total_ms`：该 chunk 内全部 kernel 的累计 CUDA 时间；
- `kernels`：按完整 kernel name 记录 `calls` 与 `total_ms`。

整段既有 `prefill.kernels` 聚合、wall、token、rank 和 trace identity 字段均
保持不变。实现按每个 prefill window 的时间范围把 kernel 分配到唯一 chunk；
window 外和 generation 内的 kernel 仍不计入 prefill。该 schema 足以在同一
工具中比较排序/未排序 trace 的逐 chunk stage1 与 top-k 分布，不需要修改或
重新运行模型服务。

TDD 有效红灯为 1 error、1 pass：新增测试在读取
`chunks[0].kernel_total_ms` 时精确触发 `KeyError`，既有单窗口隔离测试仍通过。
首次实现补丁在新增字典中多留一个提前闭合大括号，静态查看时发现，尚未执行
绿色测试；修正后 compile 与同一测试均通过，最终为 `2/2 passed`。

更广 CPU-only 门禁使用固定 ca4a404e9 控制镜像、runc、network none、4 CPUs，
未注入 NVIDIA runtime。固定镜像 Python 不含 pytest、PATH 也无 Ruff，因此
首次依赖探测在测试前退出，没有把未运行测试计作通过。随后复用宿主已验收的
Ruff 0.14.0，并直接运行四个基于 unittest 的 Phase 9 测试文件。首轮 Ruff
check 通过，format check 要求机械格式化新增 comprehension 后停止；formatter
只重排新增块及相邻长条件。最终结果为：

- Ruff check 与 format check 通过；
- analyzer compile、`git diff --check` 通过；
- analyzer、OSCAR prefill benchmark、原生 top-k benchmark 与 Phase 9 tools
  四个测试文件合计 `33/33 passed`、0 failed。

最终 analyzer 与测试 SHA256 分别为：

- `724aeb5e45f8a9322b7e52d096fb38670ec768f89cb9844d1d49ab213cddbf43`；
- `f57b985ab2fb72258eb9212a5f662203e0c639a9174af7b38a2f21709690a48c`。

本节只证明逐 chunk kernel 归因工具的 schema、窗口隔离和回归兼容性，没有
产生新的 TTFT、TPOT、吞吐或 GSM8K 精度结果。下一步先发布工具、测试、本节
与 planning；恢复 clean/published 后，使用固定 Python
3.12.13/ijson 3.4.0.post0、4 CPU worker、network none 分别重跑 2.53 未排序与
2.63 排序的 8-rank trace，再依据逐 chunk 实测选择最小运行时候选。

### 2.65 排序候选的逐 chunk 实测与 history 覆盖率工具门禁

2.64 的 analyzer v3、测试、报告与 planning 已由主仓库提交
`1ae08c23ffe771999462796c32e407574fbb0e87` 发布，发布状态由
`e8ba81145dc1c38aca6f2b289f29eb125b117e4e` 固化。本阶段没有修改 OSCAR
运行时源码、模型、正式配置或控制镜像，也没有分配 GPU；只在固定控制镜像中
重新分析既有 2.53 未排序与 2.63 排序的正式 trace，并扩展 CPU-only tile
覆盖率工具。

两组 format version 3 analysis ID 分别为：

- 排序：`20260731T2245Z_topk_sort_32k_chunk_trace_v1`；
- 未排序：`20260731T2247Z_ca4a404e9_32k_chunk_trace_v1`。

分析固定使用 `oscar-glm-stage9-runtime:ca4a404e9`、runc、network none、
4 CPUs，以及 Python/ijson `3.12.13/3.4.0.post0`。两组均为 exit=0、8/8 ranks、
每 rank 16/16 chunks；每 rank 的逐 chunk kernel total 求和与整段 prefill
kernel 聚合在 `1e-6 ms` 内一致，stage1/top-k 调用数分别为 1,248/1,344。
排序与未排序 summary SHA256 分别为：

- `a3c58c84073b7801ae8a2f439666b7ddbf6a4d8411fc0cb620f75e82ef67f0e7`；
- `b90f54cfae34cea7e4f98a5d5de9fbd3d802bf6658c8a9ef7196ed023177ab06`。

8-rank 中位数的逐 chunk 结果如下。负值表示排序候选更快；stage1/top-k
每个 chunk 的调用数均固定为 78/84。

| chunk 结束长度 | prefill wall：未排序→排序（ms） | wall 变化（ms） | stage1：未排序→排序（ms） | stage1 变化（ms） | top-k：未排序→排序（ms） | top-k 变化（ms） |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 1371.473105→1339.631304 | -31.841801 | 676.798623→676.913921 | +0.115299 | 0.396254→0.387492 | -0.008762 |
| 4,096 | 1989.758663→1965.317632 | -24.441031 | 1296.583235→1273.340653 | -23.242583 | 5.841000→8.931533 | +3.090533 |
| 6,144 | 2006.118946→1976.755295 | -29.363650 | 1297.006163→1273.683491 | -23.322672 | 9.992573→12.725435 | +2.732862 |
| 8,192 | 2017.976015→1989.397959 | -28.578055 | 1296.953107→1274.672630 | -22.280477 | 12.322107→14.965149 | +2.643041 |
| 10,240 | 2025.641572→2006.852767 | -18.788805 | 1293.930009→1274.044798 | -19.885211 | 13.002766→15.454736 | +2.451970 |
| 12,288 | 2042.257414→2026.392618 | -15.864797 | 1297.748460→1276.804448 | -20.944011 | 13.514496→15.789948 | +2.275452 |
| 14,336 | 2052.576299→2042.206195 | -10.370104 | 1295.220143→1276.243792 | -18.976351 | 14.083409→16.058472 | +1.975063 |
| 16,384 | 2067.893408→2055.499068 | -12.394340 | 1294.038632→1276.594673 | -17.443959 | 14.385036→16.479381 | +2.094345 |
| 18,432 | 2088.285722→2074.966763 | -13.318959 | 1294.841905→1277.908170 | -16.933735 | 15.467616→17.175722 | +1.708107 |
| 20,480 | 2107.802375→2091.701730 | -16.100646 | 1299.711217→1280.649087 | -19.062130 | 15.677926→17.497825 | +1.819900 |
| 22,528 | 2118.373793→2109.242792 | -9.131001 | 1294.474777→1279.381987 | -15.092789 | 16.586913→18.199426 | +1.612513 |
| 24,576 | 2138.864715→2125.569674 | -13.295042 | 1294.094281→1279.992195 | -14.102086 | 17.210763→18.516844 | +1.306081 |
| 26,624 | 2152.136153→2138.998028 | -13.138125 | 1294.695473→1280.822491 | -13.872982 | 17.301544→18.964154 | +1.662610 |
| 28,672 | 2175.864135→2163.195056 | -12.669079 | 1295.937901→1281.706790 | -14.231112 | 18.201285→19.614723 | +1.413438 |
| 30,720 | 2188.958320→2176.890090 | -12.068230 | 1296.231854→1281.880162 | -14.351692 | 18.763394→20.267990 | +1.504596 |
| 32,768 | 2212.833548→2196.264003 | -16.569545 | 1303.666977→1284.593046 | -19.073931 | 19.007975→20.637608 | +1.629633 |

首个 chunk 的 stage1 没有改善，符合 2.54 中 `rowLen<=topK` 不排序的快捷
路径；第 2–16 个 chunk 的 stage1 每块稳定下降 `14.102086–23.322672 ms`，
而 top-k 每块增加 `1.306081–3.090533 ms`。排序版 stage1 从第 2 块到末块
基本维持在 `1273.340653–1284.593046 ms`，因此排序版 wall 从
`1965.317632 ms` 增至 `2196.264003 ms` 主要来自其他 kernel，而不是 stage1
随上下文长度继续线性增长。这与 2.63 的整段归因一致：排序收益来自后续
stage1，而 top-k 本身更慢。

源码复核进一步确认，stage1 已用 `has_bf16` 对两次 BF16 dot 做动态门禁；
但 history page/data/scale/zero 的 masked load、history score dot 和 history
value contribution dot 仍没有对称的 `has_history` 分支。为了避免直接把该观察
误写成性能结论，本阶段先扩展 `summarize_selected_tiles`，新增：

- `active_tiles`、`tiles_with_history`、`tiles_without_history`；
- `history_only_tiles`、`mixed_precision_tiles`、`all_bf16_tiles`。

其中 `tiles_without_history` 的口径是“至少一个 valid 槽且没有 history 槽”，
因此包含最后一个部分有效 tile，但不会把完全 invalid 的尾部 tile 计作优化机会；
`all_bf16_tiles` 则严格要求 16 个槽全部属于 BF16。TDD 红灯在访问缺失的
`active_tiles` 时得到 `KeyError`，为 0 passed/1 error。最小实现后 compile、
定向测试 1/1、Ruff 0.14.0 check/format 和四个 Phase 9 unittest 文件合计
34/34 均通过。最终工具与测试 SHA256 分别为：

- `61c505ff6cc92bd17d4b5fb9446ace9964402ee9ca4bb78806c693bdd5e5fbb4`；
- `26690342999f9b96207ac278f62dbbe43b0acc2fe3c81dc9274e5cf2064cd8da`。

首轮绿色组合因 `py_compile` 尝试在只读 bind mount 写 `__pycache__` 而以
`Errno 30` 退出，测试未启动；改为任务专用 `/tmp` pycache 后通过。宿主默认
`PATH` 没有 Ruff，后续复用已验收的 Ruff 0.14.0 绝对路径完成门禁。这些失败
均发生在覆盖率实验前，没有生成或混入覆盖率结果。

本节尚未实测排序后 `tiles_without_history` 的数量，也没有修改运行时或产生
新的 TTFT、TPOT、吞吐、output/LSE 或 GSM8K 结果。因此，“增加
`has_history` 可提速”目前仍是未验证假设，不能视为优化结论。下一步先发布
工具、测试、本节与 planning；恢复 clean/published 后，再以 2.54 的同一
seed 42、16 个 2,048-token chunk、top-k 2,048 协议执行 CPU-only 覆盖率
实测，并在任何运行时改动前先实时记录该结果。

### 2.66 排序后的 no-history tile 覆盖率实测与候选淘汰

2.65 的工具、测试、报告与 planning 已由主仓库提交
`d80d4fe08381f32b4034195667c538062b358d5a` 发布，发布状态由
`a4298a4` 固化；主仓库与源码仓库随后均为 clean/published。本阶段没有修改
运行时源码、模型、数据集、正式配置或控制镜像，也没有注入 NVIDIA runtime。

有效 CPU-only analysis ID 为：

`20260731T2301Z_topk_sort_history_coverage_cpu_v1`。

实验精确复用 2.54 的 seed 42、16 个 2,048-token chunk、top-k 2,048、
prefix 64、recent 256 与 16-token tile 协议；固定使用
`oscar-glm-stage9-runtime:ca4a404e9`、runc、network none、4 CPUs，Python/
PyTorch 为 `3.12.13/2.11.0+cu129`，`cuda_visible=false`。有效轮次 exit=0，
16/16 chunks 完成，summary 实测耗时 `828.5257903169841 s`。轮次超过
10 分钟；脚本在
`600.0 s` 独立打印 `13/16 chunks` 进度，满足长实验进度门禁。

逐 chunk 的直接 history 跳过机会如下；`no-history` 表示 active tile 中至少
有一个 valid 槽、且没有任何 history 槽，mixed 表示同一 tile 同时含 BF16 与
history token。

| chunk 结束长度 | no-history：未排序→排序 | mixed：未排序→排序 |
|---:|---:|---:|
| 2,048 | 199→199 | 85,168→85,168 |
| 4,096 | 0→6,512 | 97,664→1,947 |
| 6,144 | 0→2,587 | 42,701→1,626 |
| 8,192 | 0→2,508 | 36,912→2,281 |
| 10,240 | 0→555 | 30,726→2,102 |
| 12,288 | 0→101 | 21,617→2,299 |
| 14,336 | 0→185 | 30,469→2,271 |
| 16,384 | 0→110 | 21,030→2,289 |
| 18,432 | 0→144 | 8,319→2,282 |
| 20,480 | 0→201 | 12,932→2,286 |
| 22,528 | 0→132 | 17,215→2,228 |
| 24,576 | 0→69 | 6,514→2,278 |
| 26,624 | 0→98 | 12,978→2,252 |
| 28,672 | 0→0 | 19,350→2,264 |
| 30,720 | 0→48 | 9,158→2,273 |
| 32,768 | 0→32 | 4,518→2,294 |

全部 16 chunks 共有 4,064,256 个 active tile。排序前后聚合结果为：

| 指标 | 未排序 | 排序 | 变化 |
|---|---:|---:|---:|
| 含 BF16 的 tile | 457,470 | 131,621 | -325,849 |
| 含 history 的 tile | 4,064,057 | 4,050,775 | -13,282（-0.326816%） |
| no-history active tile | 199 | 13,481 | +13,282 |
| no-history / active | 0.004896% | 0.331697% | +0.326800 个百分点 |
| mixed tile | 457,271 | 118,140 | -339,131（-74.164117%） |
| full-BF16 tile | 109 | 13,391 | +13,282 |
| full-history tile | 3,605,435 | 3,931,284 | +325,849 |

含 BF16 的 tile 与 full-history tile 精确复现 2.54 的既有结果；有效 selected
token 与 BF16 selected token 也分别精确复现 65,012,736 与 581,379。新增
分区校验进一步证明，对排序/未排序两套数据，active tile 均可无重叠地分解为
history-only、mixed 与 no-history；active/history/BF16 分区各 32/32、selected
集合不变量 16/16、首块不排序 shortcut 和 2.54 六项对账全部通过。

首次生成的 raw summary 使用“对所有整数值求和”的通用聚合，错误地把每个
chunk 的常量 `tile_width=16` 累加为 256，并生成无意义的常量 delta。该错误
没有改变逐 chunk rows，但 raw aggregate 不作为正式结论。随后没有重跑昂贵
输入生成，而是只从已落盘、未修改的 16 个 rows 重建显式字段 aggregate，生成
validated summary 并执行上述一致性门禁。raw summary、validated summary 与
validation SHA256 分别为：

- `ae009fa6bf2c74d01b73a728617ee19957c68a007f7098b7c4df21d75dc3cdea`；
- `5612b29ce88d51ab9a2ba1b50f5c494dadf055b9402e41dc80e503a97901d9d2`；
- `c51f917f2b2589ad5204fd46ff9805889ffd12fb7713c2a7cef3bf5ef578c6d9`。

小型证据已封存到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/formal_topk_sort_history_coverage_cpu_v1`。

目录含 raw/validated summary、validation 与两份脚本共 5 项；连同 manifest
共 6 个文件、59,291 bytes，5/5 manifest 复算通过。manifest SHA256 为：

`e8a134219161667e59f0b5f4f5157d7545dca8193e0322cc1a89fedd7a57c5d4`。

结论是淘汰对称 `has_history` 候选：排序虽然减少了 325,849 个含 BF16 的 tile，
但只新增 13,282 个可以完全跳过 history 路径的 tile，前者是后者的
`24.533128×`；排序后 no-history 也只占 active tile 的 `0.331697%`，且第 14
chunk 为 0。即便动态分支没有任何开销，可跳过的 history dot 调用覆盖率也过低；
现有证据不支持为此修改生产 kernel，更不能宣称 TTFT 会改善。本阶段没有新的
TTFT、TPOT、吞吐、output/LSE 或 GSM8K 精度结果。下一步先发布本节与 planning，
再从 2.65 的逐 chunk 其他 kernel 分布中选择覆盖面更大的候选；发布前不修改
运行时源码或启动 GPU 实验。

### 2.67 Rotation IEEE/TF32 独立筛选工具的 CPU/TDD 门禁

2.66 的 coverage 结果、候选淘汰结论、报告与 planning 已由主仓库提交
`21d60b75295a6439606cad65a436b0a3470531dc` 发布，发布状态由
`d7621fe` 固化；下一候选的只读定位由 `67b6b38` 固化。主仓库与源码仓库在
本阶段开始前均为 clean/published。本阶段只新增主仓库 CPU 可验证的筛选工具
及测试，没有修改 OSCAR 生产源码、源码 submodule、模型、数据集、正式配置或
控制镜像，也没有检查、注入或分配 GPU。

2.65 的排序 trace 中，除 stage1 外最大的稳定 OSCAR 项为
`_rotate_latent_kernel`：按“每个 chunk 先取 8-rank 中位数，再跨 16 chunks
求和”的只读候选排序口径为 `3391.069 ms`，首末 chunk 分别约
`209.712/212.105 ms`。该口径适合比较覆盖面，不等同于“整段各 rank 总和再取
中位数”，因此不能直接当作新的正式 TTFT 归因值。

源码复核确认，该 kernel 在每层 current-history 写入时执行
`latent @ rotation`；78 层 rotation 不同，且每个 chunk 都有新的 history row，
不能跨层或跨 chunk 缓存。caller 已复用 gathered 与 FP32 rotated scratch，
排除了“只消除 output 分配”的候选。当前生产 kernel 的固定参数为：

- rows 对应当前 history 行数，正式后续 chunk 的筛选几何固定为 2,048；
- latent rank 512；block M/N/K 为 `16/64/32`；
- 4 warps、2 stages、FP32 accumulator；
- latent 与 rotation 显式转 FP32，dot 使用 `input_precision="ieee"`。

rotation 后立即进入 INT2 clip、量化与 pack；既有 CUDA correctness 对 rotated
输出使用 `atol=0.35, rtol=0.02`。因此本阶段只把 TF32 作为待筛选候选，不改
生产 kernel；新增 `scripts/phase9/benchmark_oscar_rotation.py`，其门禁设计为：

1. benchmark-local IEEE kernel 与生产 kernel 保持相同 block、warps、stages
   和 dot 顺序，要求输出 `atol=rtol=0`，避免错误的本地参考；
2. 只以 constexpr 切换 IEEE/TF32，计时固定 2,048×512 rotation；
3. 精度使用真实 rotation artifact 的层 0/25/51/77 与 seed 42 合成 BF16
   latent，TF32 rotation 必须通过 `0.35/0.02`；
4. 两套 rotated 输出继续调用生产 INT2 quantize/dequantize，恢复结果也必须
   通过同一门限，并记录 packed byte 相同比例、scale/zero 最大误差；
5. 每种 precision 先 warm-up 20 次，再做 7 组、每组 20 次 CUDA Event 与
   wall 计时；输出使用原子 JSON 写入并记录脚本、rotation artifact 与环境身份。

TDD 红灯先新增测试而不创建工具；固定 ca4a404e9 控制镜像、4 CPUs、network
none 中，在导入目标脚本时得到预期 `FileNotFoundError`，0 项测试执行。实现
最小工具后，定向 compile 与 7/7 unittest passed。随后新增生产源码静态契约，
确认生产 block `16×64×32`、4 warps、2 stages 与 IEEE precision 仍存在，且
benchmark 源码恰有一个 IEEE 和一个 TF32 分支。

首轮 Ruff check 和 `git diff --check` 通过，但 Ruff format check 要求机械
格式化两个新增文件并停止；此时广回归尚未运行。机械格式化后，最终 CPU-only
门禁为：

- Ruff 0.14.0 check 与 format check passed；
- 固定控制镜像内 compile 与生产 IEEE 静态契约 passed；
- analyzer、OSCAR prefill、原生 top-k、rotation benchmark 与 Phase 9 tools
  五个 unittest 文件合计 `42/42 passed`、0 failed；
- `git diff --check` passed。

负向参数测试打印的 argparse usage/error 是预期 stderr，不是回归失败。最终
工具与测试分别为 619/125 行，SHA256 为：

- `28a132e39e615500709115d0daba5448be24a85e5d9cfec6548955c7c07658b8`；
- `cc937b7d5d4020fb3d36186f4125cb697ce75434f2c7865efdfc145d3c66d80a`。

本节只证明筛选工具的参考契约、精度检查路径、计时统计与 CPU 回归，没有产生
IEEE/TF32 CUDA 时间、INT2 实测误差、TTFT、TPOT、吞吐或 GSM8K 精度结果。
四个代表层也不能替代 78 层完整回归。下一步先发布工具、测试、本节与 planning；
恢复 clean/published 后，按 GPU 双空闲门禁在固定单卡上运行该筛选。只有
benchmark-local IEEE 与生产逐值一致、四层 rotation/INT2 恢复精度通过且 TF32
有稳定实测收益时，才考虑修改生产源码。

### 2.68 Rotation TF32 单卡筛选的 INT2 精度失败

2.67 的 rotation 工具、测试、报告与 planning 已由主仓库提交
`25077d00bee732251c96f3b63a005016357aff66` 发布，发布状态由
`ec3ea6d` 固化。GPU 双空闲状态由 `90887f9` 发布；主仓库与源码仓库在启动前
均为 clean/published，生产源码仍固定
`ca4a404e913ce55237ca60383cc86e221fbfea26`。

GPU 空闲检查在 `23:32:39Z/23:33:46Z` 执行，间隔 67 秒；两次 8/8 张苹果800
均为 0 MiB/0%、无 compute process，外部下载容器 DeviceRequests=null。启动前
即时复查 GPU 0 仍为 0 MiB/0%，本轮固定只使用 GPU 0。有效 run ID 为：

`20260731T2334Z_rotation_tf32_screen_v1`。

实验使用固定 `oscar-glm-stage9-runtime:ca4a404e9` 控制镜像、network none、
4 CPUs、32 GiB 内存与 1 张 GPU；工具 SHA256 为
`28a132e39e615500709115d0daba5448be24a85e5d9cfec6548955c7c07658b8`。
筛选按 2.67 的顺序先比较 benchmark-local IEEE 与生产 IEEE，再比较 TF32
rotation，最后比较两者经过生产 INT2 quantize/dequantize 的恢复结果；只有精度
全部通过后才进入 warm-up 与 timing。

本轮在 `tf32_int2_restored` 精度门禁失败：

- `atol=0.35, rtol=0.02`；
- 超限值 128 个；
- 最大绝对误差 `1.6203639507293701`；
- 进程 exit code 1。

从 traceback 的执行位置可以确定，发生失败的该层在此前已经通过
benchmark-local IEEE=生产 IEEE 和 TF32 rotation 两个检查；但工具只在全部
四层检查完成后才写 JSON，异常消息没有携带 layer，也没有在异常前输出各层
数值。因此不能断言失败发生在层 0/25/51/77 中的哪一层，也不能报告前序检查
的具体最大误差。该信息缺口不影响候选判定：任一代表层的 INT2 恢复结果超过
冻结门限，都足以淘汰全局 TF32 production 改动。

实验没有进入 warm-up 或性能 timing，没有生成 `result.json`，因此没有任何
IEEE/TF32 CUDA 时间或 speedup 可报告。日志中的 `vllm._version` RuntimeWarning
没有终止进程；真正的退出原因是上述 AssertionError。容器随后自动删除，
`23:35:00Z` 复查 8/8 GPU 均为 0 MiB/0%、无 compute process。

失败小型证据已封存到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/formal_rotation_tf32_screen_failure_v1`。

目录包含 exit code、结构化 failure、run identity 与原始 stderr 共 4 项；连同
manifest 共 5 个文件、3,126 bytes，4/4 manifest 复算通过。exit、failure、
identity、stderr 与 manifest SHA256 分别为：

- `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`；
- `4ece62802fe7bd6bc4b27210436aaa14e4a22ddb335c1e26e271c87a597ce822`；
- `7a91f8875f7eec8f795f83932e05d689bf8b924b739ca123b26f76fc58c6a5ce`；
- `e5ac37b5d06f7dc990513af1b399b3dd22bc7b1f304b0ca49b26af7c09c9f344`；
- `f558b23603692be50dccca13bc83fe9aa257500c3faf969776e06dd3ff203712`。

结论是淘汰“只把 rotation dot 从 IEEE 改为 TF32”的候选。TF32 rotation 本身
虽未在冻结 rotation 门限处退出，但细小差异跨越了后续 INT2 clip/量化边界，
使恢复结果最大误差达到 `1.620364`，不能为未知性能收益牺牲精度。本轮没有修改
生产源码，也没有新的 TTFT、TPOT、吞吐或 GSM8K 结果。下一步先发布本节与
planning，再只筛选保持 IEEE precision 和 K 维累加顺序不变的 block M/N/warps
参数；发布前不修改生产 kernel、不启动下一 GPU 实验。

### 2.69 Rotation IEEE tile/warps sweep 工具的 CPU/TDD 门禁

2.68 的 TF32 精度淘汰结果、报告与 planning 已由主仓库提交
`603e098731b2a60c23a082b65d1d13b1043998a3` 发布，发布状态由
`2c7aa4c` 固化。生产源码继续固定 ca4a404e9。本阶段只扩展主仓库的 rotation
筛选工具与 CPU 测试，没有修改源码 submodule、模型、数据集、正式配置或控制
镜像，也没有重新检查、注入或分配 GPU。

2.68 证明降低 dot precision 会在后续 INT2 量化边界放大误差，因此本阶段新增
`--mode ieee-sweep`，冻结以下不变量：

- `input_precision="ieee"`；
- FP32 accumulator；
- block K=32，K 维循环与累加顺序不变；
- num stages=2；
- 输入、rotation、输出 dtype 与 2.67 相同。

只比较 block M/N 与 warps，矩阵为：

| 配置 | block M | block N | block K | warps | stages | 角色 |
|---|---:|---:|---:|---:|---:|---|
| `m16_n64_w4` | 16 | 64 | 32 | 4 | 2 | 生产 baseline |
| `m16_n64_w8` | 16 | 64 | 32 | 8 | 2 | 仅增加 warps |
| `m32_n64_w4` | 32 | 64 | 32 | 4 | 2 | 增大 M |
| `m32_n64_w8` | 32 | 64 | 32 | 8 | 2 | 增大 M 与 warps |
| `m16_n128_w4` | 16 | 128 | 32 | 4 | 2 | 增大 N |
| `m16_n128_w8` | 16 | 128 | 32 | 8 | 2 | 增大 N 与 warps |

每个配置先对真实 rotation artifact 的层 0/25/51/77 与 seed 42 合成 BF16
latent 运行；输出必须与生产 `oscar_mla_rotate` 在 `atol=rtol=0` 下逐值一致，
才进入 20 次 warm-up、7 组×20 次 CUDA Event/wall 计时。bitwise equality 已
覆盖完整 rotation 输出，因此通过配置的后续 INT2 输入也完全相同，不需要为
该 sweep 放宽 2.68 的任何精度门限。候选编译或运行失败会被单项记录；生产
baseline 失败则整轮失败，避免在错误参考下选择“最快”配置。

TDD 红灯在扩展测试后为 7 passed/3 errors，精确缺少 `mode`、
`build_ieee_sweep_configs` 和 `select_best_ieee_config`。最小实现 mode、六配置、
逐配置 bitwise 检查、失败隔离、相对 baseline 统计与 best selection 后，定向
compile 和 rotation unittest 为 10/10 passed。

首轮 Ruff check 与 `git diff --check` 通过，但 format check 要求机械格式化
工具后停止，广回归尚未运行。格式化后最终 CPU-only 门禁为：

- Ruff 0.14.0 check/format passed；
- 固定 ca4a404e9 控制镜像、4 CPUs、network none 中 compile passed；
- analyzer、OSCAR prefill、原生 top-k、rotation benchmark 与 Phase 9 tools
  五个 unittest 文件合计 `44/44 passed`、0 failed；
- `git diff --check` passed。

扩展后工具与测试分别为 819/172 行，SHA256 为：

- `a677340e053b21d628096530e3a12c2252701fab9da89e964e493b614f01cabb`；
- `298d4a966759a1553701f1ec9d3807d3d44a589e0f781e59e5558018c2b5e58a`。

本节只证明 IEEE sweep 的配置边界、bitwise 门禁、失败隔离和 CPU 回归，没有
产生六配置的编译结果、CUDA 时间、TTFT、TPOT、吞吐或 GSM8K 精度结果。下一步
先发布工具、测试、本节与 planning；恢复 clean/published 后重新执行 GPU 双
空闲门禁，并在固定单卡运行 2,048×512 sweep。只有 bitwise 通过且相对生产
baseline 有稳定实测收益的配置才进入 production 源码候选。

### 2.70 Rotation IEEE tile/warps 的固定单卡 GPU sweep 结果

2.69 的工具、测试、报告与 planning 已由主仓库提交
`ad7595fbb3772f389940a50c87c9b07028f015a4` 发布，发布状态由 `b5eee23`
固化。GPU 空闲状态由 `021b7c7` 发布；启动时主仓库为
`021b7c746b5058d76546f584248bbf8985e4738e`，源码仓库继续固定
`ca4a404e913ce55237ca60383cc86e221fbfea26`，两仓均为 clean/published。
本阶段没有修改 production kernel。

GPU 双空闲检查在 `23:44:19Z/23:45:28Z` 执行，间隔 69 秒；两次 8/8 张
苹果800均为 0 MiB/0%、无 compute process，外部下载容器
DeviceRequests=null。启动前即时复查 GPU 0 仍为空闲；正式 run 固定只使用
GPU 0，结束后 `23:46:46Z` 再次确认 8/8 张卡均为 0 MiB/0%、无 compute
process。有效 run ID 为：

`20260731T2346Z_rotation_ieee_sweep_v1`。

实验使用固定 `oscar-glm-stage9-runtime:ca4a404e9` 控制镜像（image ID
`sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`）、
network none、4 CPUs、32 GiB 内存与 1 张 GPU。实际环境为 Python 3.12.13、
PyTorch 2.11.0+cu129、CUDA runtime 12.9、NVIDIA 苹果800-SXM4-80GB；工具与
rotation artifact SHA256 分别为：

- `a677340e053b21d628096530e3a12c2252701fab9da89e964e493b614f01cabb`；
- `256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d`。

固定负载为 2,048×512 BF16 latent、FP32 rotation/output、seed 42；所有配置
保持 IEEE precision、FP32 accumulator、block K=32 与 2 stages。每个配置先在
真实 rotation 层 0/25/51/77 上与 production 输出做 `atol=rtol=0` 比较；每层
比较 1,048,576 个值。只有逐值一致才执行 20 次 warm-up 与 7 组×20 次 CUDA
Event/wall 计时。本轮进程 exit code 0，6/6 配置、4/4 层全部通过 bitwise 门禁，
每项 mismatched values、最大绝对误差和最大相对误差均为 0。

CUDA 中位结果如下；“耗时变化”以 production `m16_n64_w4` 为基准，负值表示
更快：

| 配置 | CUDA 中位耗时（ms） | 相对 baseline 耗时变化 | speedup |
|---|---:|---:|---:|
| `m16_n64_w4` | 0.1016319990158081 | 0% | 1.0× |
| `m16_n64_w8` | 0.14187519550323485 | +39.5969742572585% | 0.7163479046165672× |
| `m32_n64_w4` | 0.09359359741210938 | -7.909321553783877% | 1.0858862339514979× |
| `m32_n64_w8` | 0.1004032015800476 | -1.209065498720896% | 1.012238628016068× |
| `m16_n128_w4` | 0.09784319996833801 | -3.7279587965998506% | 1.0387231718575858× |
| `m16_n128_w8` | 0.10460159778594971 | +2.9219131758686734% | 0.9716103880533602× |

最佳配置为 `m32_n64_w4`：CUDA 中位耗时由 `0.1016319990158081 ms` 降至
`0.09359359741210938 ms`，减少 `0.00803840160369873 ms`，即
`7.909321553783877%`；wall 中位耗时由 `0.10287309996783733 ms` 降至
`0.09484267793595791 ms`，减少 `7.806143719193925%`。8-warps 配置没有优于
对应 4-warps 配置；增大 block N 到 128 也不及只把 block M 增至 32。

小型证据封存到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/formal_rotation_ieee_sweep_v1`。

目录含 exit code、退出后 GPU 状态、原始 `result.json` 与 run identity 共 4 项；
连同 manifest 共 5 个文件、18,988 bytes，4/4 manifest 复算通过。上述四项与
manifest SHA256 分别为：

- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `ffbc22947f64ab166bd8fd74ad5a81d9b88e329bfe69ff97db12a880731a8ab7`；
- `a6e68a844d398f93eb04414201d8fb100f1a38f8ec762bfd64aedd63a5c1c1b4`；
- `8e75a06d7e94a17f49e79b3de3e7be8da6eb6abe74b22161b67bc0bf2d886472`；
- `295d3f64d1ada2fdca902963569b7b06aa08cdf3a159d3a0e3cf939c9d62e7fb`。

当前只能确认 `m32_n64_w4` 在 2,048-row 单 kernel 几何上 bitwise 等价且更快，
不能把 `7.909%` 直接外推为 2.65 trace 中约 `3391.069 ms` rotation 总量或 32K
端到端收益。只读源码检查已发现 `_rotate_latent_kernel` 除 current-history store
外还由 query rotation 共用；query 的扁平行数可能大于 2,048，因此本轮尚未覆盖
全部真实调用几何。本阶段没有产生新的 TTFT、TPOT、吞吐或 GSM8K 精度结果，
也不据此修改 production。下一步先发布本节与 planning，再用 CPU/trace 核对
同名 kernel 的调用数和两类真实行数；确认候选同时覆盖实际几何后，才进入生产
源码修改与正式回归。

### 2.71 Rotation 真实调用几何归因与 trace-layout 工具 CPU/TDD 门禁

2.70 的单卡 sweep 结果、报告与 planning 已由主仓库提交
`c06d45486fa298b85c1a31dad0e4f113bcc174b4` 发布，发布状态由 `33a282c`
固化。开始本阶段时主仓库和源码仓库均为 clean/published，production source
继续固定 `ca4a404e913ce55237ca60383cc86e221fbfea26`。本阶段只读分析已有 trace，
并扩展主仓库 benchmark 与 CPU 测试；没有检查、注入或分配 GPU，也没有修改
production kernel、模型、数据集、正式配置或控制镜像。

首先对 2.65 的 sorted/unsorted v3 summary 逐 rank、逐 chunk 复核同名
`_rotate_latent_kernel`。两组 trace 的 8/8 ranks 调用数完全一致：

- chunk 1 每 rank 233 calls；
- chunk 2–16 每 rank 均为 311 calls；
- 每 rank 合计 `233+15×311=4898` calls；
- sorted 每 rank 总耗时范围为 `3387.745537–3399.351272 ms`；
- unsorted 每 rank 总耗时范围为 `3387.972586–3394.243401 ms`。

这证明 top-k selected-index 排序不改变 rotation 工作量，也否定了“每层每块仅
一次 current-history rotation”的旧隐含假设。源码调用链显示，同一 kernel 同时
服务四类计算：recent demotion、current-history store、query 正向 rotation 和
attention history 输出的逆向 rotation。正式模型 `config.json` 实际为 78 layers、
64 attention heads、KV LoRA rank 512；TP=8 时每 rank 有 8 heads，2,048 个 query
row 会展平为 `2048×8=16384` rows。

为避免只靠配置推导，进一步在固定 ca4a404e9 镜像、network none、4 CPUs、
32 GiB、无 GPU 的环境中，用 Python 3.12.13 与 ijson 3.4.0.post0/yajl2_c 流式
读取 sorted rank0 的原始压缩 trace。该文件为：

`/dev/shm/oscar-glm-stage9-topk-sort-formal/profiles/20260731T2135Z_candidate_topk_sort_32k_b1_v1/dp0_pp0_tp0_dcp0_ep0_rank0.1785535426403869390.pt.trace.json.gz`

压缩后大小为 147,257,334 bytes。raw trace 中每个 prefill chunk 有内外两层同名
execute annotation，共 32 个 span；既有 v3 summary 使用每对第一个内层 span。
沿用该口径得到 16 个 span、4,898 calls、`3390.564987 ms`，逐 chunk calls/time
与 v3 summary 完全一致，排除了重复统计外层 annotation 的风险。

production block M/N 为 16/64，latent rank 512，因此 grid 第一维满足
`ceil(rows/16)×8`。rank0 内层 span 的 launch signature 可直接反解为：

| 路径/布局 | grid | rows | calls | 总耗时（ms） | rotation 总耗时占比 |
|---|---:|---:|---:|---:|---:|
| recent demotion，连续矩阵 | 128 | 256 | 1,170 | 28.359872 | 0.836434993% |
| 首块 current-history store，连续矩阵 | 864 | 1,728 | 78 | 5.213370 | 0.153761099% |
| 后续 current-history store，连续矩阵 | 896 | 1,792 | 1,170 | 85.427658 | 2.519569993% |
| query 正向，连续矩阵 | 8,192 | 16,384 | 1,248 | 714.301325 | 21.067324406% |
| history 逆向，非连续 `rotation.T` | 8,192 | 16,384 | 1,232 | 2557.262762 | 75.422909509% |

三类 store 合计仅 `119.000900 ms/3.509766085%`。16,384-row 正向单次中位为
`572.354 us`；逆向 `rotation.T` 单次中位为 `2075.656 us`，约为正向的
3.6265 倍，并独占 rotation 总时间的 75.423%。两者 grid/block 相同，但正向
signature 为 48 registers/thread、11,264 bytes shared memory，逆向为 64
registers/thread、10,240 bytes shared memory；直接证据表明，非连续转置矩阵访问
才是当前主要 rotation 瓶颈。2.70 的 2,048-row 正向 sweep 没有命中该主路径。

据此给 `scripts/phase9/benchmark_oscar_rotation.py` 增加 `trace-layout` mode，
强制 `--rows 16384`，固定 IEEE precision、FP32 accumulator、block K=32 与
2 stages。新 mode 把数学方向、矩阵物理布局和 M tile 分开，比较五个 case：

| case | 数学方向 | 矩阵布局 | block M | 角色 |
|---|---|---|---:|---|
| `forward_m16` | 正向 | contiguous | 16 | 正向 production baseline |
| `forward_m32` | 正向 | contiguous | 32 | 正向 tile candidate |
| `inverse_strided_m16` | 逆向 | strided transpose view | 16 | 逆向 production baseline |
| `inverse_contiguous_m16` | 逆向 | contiguous transpose | 16 | 只改变布局 |
| `inverse_contiguous_m32` | 逆向 | contiguous transpose | 32 | 改变布局与 M tile |

每个 case 都必须在真实 rotation 层 0/25/51/77 上，相对同一数学方向的 production
输出通过 `atol=rtol=0` 后才允许计时；forward 和 inverse 分别使用自己的 baseline，
不直接比较不同数学结果。工具还记录若为 78 层各预存一份 512×512 FP32 contiguous
transpose，静态额外显存为 `78×512×512×4=81,788,928 bytes=78 MiB/GPU`。该值
只是待评估代价，本阶段没有创建 production buffer。

TDD 红灯先增加3项 CPU 测试；固定控制镜像、4 CPUs、network none 中为
`10 passed/3 errors`，精确缺少新 CLI mode、case builder 和 storage helper。
最小实现后，首次 compile 命令因 py_compile 尝试写只读 `/workspace` 的
`__pycache__` 而以 `Errno 30` 退出，测试未执行；把任务专用 pycache 改到容器
`/tmp` 后，定向 compile 与 `13/13 passed`。

最终 CPU-only 门禁为：

- Ruff 0.14.0 check/format passed，两文件无需 formatter 改写；
- 固定控制镜像内 compile passed；
- analyzer、OSCAR prefill、原生 top-k、rotation benchmark 与 Phase 9 tools
  五个 unittest 文件合计 `47/47 passed`、0 failed；
- 五 case、16,384 rows、bitwise 与 78-layer storage 静态契约 passed；
- `git diff --check` passed；源码 submodule clean/published。

扩展后工具与测试分别为 1,078/229 行，SHA256 为：

- `2fab0327e279b6bbcd9fbe40ff2d704e25400408778d940f5dd292d86af0bb36`；
- `012ad5be597491a4e657fc56795a3269876742355f2bdc28103f6173f4b093a0`。

本节产生了实际 trace 几何/耗时归因和 CPU 工具门禁，但尚未产生五 case 的 CUDA
编译结果或计时，也没有新的 TTFT、TPOT、吞吐或 GSM8K 精度结果。下一步先发布
工具、测试、本节与 planning；恢复 clean/published 后重新执行 GPU 双空闲门禁，
再固定单卡运行 trace-layout 筛选。只有四层 bitwise 全通过且 contiguous inverse
收益足以覆盖 78 MiB/GPU 代价，才考虑修改 production rotation 生命周期。

### 2.72 Rotation trace-layout 的固定单卡 GPU 筛选结果

2.71 的 trace-layout 工具、测试、报告与 planning 已由主仓库提交
`6a7d9f3d96568391ca6bd9a55401d0c1fb8ff8ca` 发布，发布状态由 `d06fe54`
固化；GPU 空闲状态由 `7a13db8` 发布。正式启动提交为
`54a6c557ddcb97a6df691ff73e8bb79e59a5e215`，源码仓库继续固定
`ca4a404e913ce55237ca60383cc86e221fbfea26`；启动时两仓均为
clean/published，production source 未改。

GPU 双空闲检查在 `00:10:42Z/00:11:50Z` 执行，间隔 68 秒；两次 8/8 张
苹果800均为 0 MiB/0%、无 compute process，外部下载容器
DeviceRequests=null。启动前 `00:15:48Z` 即时复查 GPU 0 仍为 0 MiB/0%、
无 compute process。本轮固定只使用 GPU 0，有效 run ID 为：

`20260801T0017Z_rotation_trace_layout_v1`。

实验使用固定 `oscar-glm-stage9-runtime:ca4a404e9` 控制镜像（image ID
`sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`）、
network none、4 CPUs、32 GiB 内存与 1 张 GPU；镜像内 rotation artifact 与
工具 SHA256 分别为：

- `256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d`；
- `2fab0327e279b6bbcd9fbe40ff2d704e25400408778d940f5dd292d86af0bb36`。

实际环境为 Python 3.12.13、PyTorch 2.11.0+cu129、CUDA runtime 12.9、NVIDIA
苹果800-SXM4-80GB。固定负载为 16,384×512 BF16 latent、FP32
rotation/output、IEEE precision、block K=32、2 stages、seed 42；每个 case
先在真实层 0/25/51/77 上逐值校验，再做 20 次 warm-up、7 组×20 次 CUDA
Event/wall 计时。

进程在 `00:15:56Z` exit=0。5/5 cases、4/4 层全部通过 bitwise 门禁；每个层/case
比较 8,388,608 个值，mismatched values、最大绝对误差和最大相对误差全部为 0。
CUDA 中位结果如下；正向相对 `forward_m16`，逆向相对
`inverse_strided_m16`，负值表示更快：

| case | CUDA 中位耗时（ms） | 相对同方向 baseline 耗时变化 | speedup |
|---|---:|---:|---:|
| `forward_m16` | 0.7081984043121338 | 0% | 1.0× |
| `forward_m32` | 0.5045760154724122 | -28.75216713281048% | 1.4035514622094731× |
| `inverse_strided_m16` | 2.1363199234008787 | 0% | 1.0× |
| `inverse_contiguous_m16` | 0.573798418045044 | -73.14080106824098% | 3.7231192283161274× |
| `inverse_contiguous_m32` | 0.5047296047210693 | -76.37387550467753% | 4.232602770708252× |

对应 wall 中位数中，`inverse_strided_m16` 为 `2.1375562995672226 ms`，
`inverse_contiguous_m16` 为 `0.5750032607465982 ms`，改善
`73.09997117441934%`；`inverse_contiguous_m32` 为
`0.5059401504695415 ms`，改善 `76.33090877784244%`。仅把 transpose 从
strided view 改成 contiguous，block M 保持 16，就已消除大部分逆向成本；其
`0.573798 ms` 也接近 2.71 raw trace 的正向单次中位 `0.572354 ms`，验证非连续
stride 是主因，而不是逆向数学本身。

需要保留一个计时边界：`forward_m16` 的前五个 CUDA 样本约为 0.708 ms，后两项
降到 `0.6767104148864747/0.5740032196044922 ms`，显示固定执行顺序下存在 GPU
频率或状态漂移；随后 `forward_m32` 七项稳定在约 0.5045–0.5049 ms。因此本轮
`-28.752%` 是实际顺序测量值，但不能单独作为最终 production m32 收益。相比之下，
inverse strided 七项稳定在约 2.1362–2.1366 ms，contiguous m16 七项稳定在约
0.5737–0.5738 ms，73.141% 的布局收益远大于该漂移。

若只把本轮 inverse m16 的单 kernel 比率 `0.2685919893175902` 乘到 2.71 rank0
trace 的逆向项，估算 inverse 可由 `2557.262762 ms` 降到
`686.8602924533752 ms`，节省 `1870.4024695466246 ms`；rotation 总量可由
`3390.564987 ms` 降到 `1520.1625174533756 ms`，估算减少
`55.16492020409767%`。这是基于两次既有实测的投影，不是新的端到端 TTFT 实测，
不能写成约 1.87 秒的正式 TTFT 收益。

78 层各预存一份 512×512 FP32 contiguous transpose 的静态额外显存为
81,788,928 bytes，即 78 MiB/GPU，约占单张 80 GiB 卡的
`0.09521484375%`。比例虽小，仍必须在 production 阶段通过模型加载和显存容量
门禁，不能先假设对 KV capacity 没有影响。

容器退出后的第一次采样中，GPU 0 已为 0 MiB、无 compute process，但利用率
仍显示 14%；9 秒后 `00:16:05Z` 复查 8/8 张卡均为 0 MiB/0%、无 compute
process，判定为退出采样滞后。日志中的 `vllm._version` RuntimeWarning 没有终止
进程，不影响 result JSON。

小型证据封存到：

`artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/formal_rotation_trace_layout_v1`。

目录包含原始 result/log、起止时间、退出码、主/源码提交、运行身份和三组 GPU
状态等 12 项数据；连同 manifest 共 13 个文件、35,067 bytes，12/12 manifest
复算通过。result 与 manifest SHA256 分别为：

- `a54502cfc9fc4445e2beeb56a7dfd70af192a25ed013c667017a52637aa5db66`；
- `5af7db4f3aee8b281c24cade8e120c494412194ca43ec3b5a822b9675d7294ea`。

本轮结论是：contiguous inverse 已同时通过真实主几何、四层 bitwise 和单卡性能
门禁，值得进入 production 候选；m32 虽在连续正/逆向都更快，但正向 baseline
有顺序漂移，且全局 block M 改动还需覆盖 256/1,728/1,792-row store 几何。
因此下一阶段优先做最小改动：预存并传递 contiguous inverse，先保持 production
block M=16；通过源码 TDD、78层 tensor identity、模型加载/显存和 CUDA correctness
后，再决定是否独立推进 m32。本轮没有新的 TTFT、TPOT、吞吐或 GSM8K 精度结果。

### 2.73 Contiguous inverse 的最小 production 源码候选

2.72 的单卡筛选记录已由主仓库提交
`971f0c4f3a57bbd73e89f7cf7dff63c928e9aff2` 发布，发布状态由 `e204e7b`
固化。随后只推进 2.72 选出的 contiguous inverse 布局改动，保持 production
rotation kernel 的 block M/N/K、warps、stages 与 IEEE precision 不变，也不修改
三段式 cache 的 store/demotion、token selection、softmax 或 global LSE 路径。

源码最小调用链为：

- `MLAAttention` 在加载每层 512×512 FP32 rotation 时，同时注册
  `rotation.T.contiguous()` 为非持久 `_oscar_inverse_rotation` buffer；
- `process_weights_after_loading` 把 forward/inverse 两个 buffer 迁移到与模型权重
  相同的 device；
- `TritonMLASparseImpl.forward_mqa` 显式按 keyword 传递 layer 上的 contiguous
  inverse，production 路径不再临时使用 strided transpose view；
- sparse decode/prefill 公共接口新增 keyword-only `inverse_rotation=None`，保留
  现有直接调用的兼容 fallback；传入显式 tensor 时会检查 512×512 几何和 device；
- history accumulator 回到原 latent 空间时使用显式 inverse；query 正向 rotation
  和全部 cache 写入仍使用原 forward rotation。

按 78 层、每层 512×512 FP32 计算，该候选理论静态增量仍为
`78×512×512×4=81,788,928 bytes=78 MiB/GPU`。这是源码形状推导值，不是新的
模型加载显存实测；是否影响 KV capacity 仍要由新镜像的模型加载/容量门禁确认。

源码 TDD 先增加 production backend identity 传递与两个公共接口 signature 契约。
固定 ca4a404e9 控制镜像、network none、4 CPUs、正式 Python 3.12.13 加只读
pytest 8.3.5 target 中，有效红灯为 `3 failed`：backend 缺少
`inverse_rotation` keyword，decode/prefill signature 均缺少该参数。最小实现后，
相同三个节点为 `3/3 passed`。

扩大 CPU/解释器验证使用空 `CUDA_VISIBLE_DEVICES`，让无 GPU 容器按 vLLM 已支持
的分布式初始化路径保留真实 Triton JITFunction；否则 0 个 active driver 会令
`@triton.jit` 退化为 placeholder，两个既有 `.fn` 源码断言不能成立。有效结果为：

| 验证组 | 结果 |
|---|---:|
| runtime activation + cache path | 24 passed、0 failed |
| decode 文件（不含独立 interpreter smoke） | 7 passed、19 skipped、1 deselected、0 failed |
| 独立 Triton interpreter smoke | 1 passed、0 failed |
| 合计 | 32 passed、19 skipped、0 failed |

19 个 skip 全部是本轮没有分配 GPU 时的 CUDA 用例，不能替代苹果800数值正确性。
其中一条既有 CUDA prefill oracle 已改为显式传入 contiguous inverse，待后续完整
CUDA 回归实际执行。

最终 Ruff 0.14.0 check/format、Python compile、mypy、SPDX、typos、forbidden
imports、root lazy imports、配置检查、boolean context、suggestion、sign-off 与
`git diff --check` 均通过。pre-commit 的 `check-torch-cuda-call` 只命中
`53d8be94f` 已有的 `torch.cuda.empty_cache()`，不在本次 diff；attention backend
文档 hook 会重写与本候选无关的既有 capability 表。提交时只精确跳过这两个已审计
项目，其余适用 hooks 全部执行并通过。

候选源码已由提交
`67a0e47ff72f10a322de17b81c4134984e017bd6`（tree
`60d5e606ce522dd78fecd890509372b727802f43`）通过 HTTPS 推送。最终 diff 为 6 个
文件、60 insertions/4 deletions；三个 production 文件 SHA256 为：

- `vllm/model_executor/layers/attention/mla_attention.py`：
  `988c922a1e2009b41bd495c564256d9603cd6aadef9e8246cc7749a6c8c075c6`；
- `vllm/v1/attention/backends/mla/triton_mla_sparse.py`：
  `121a9b779308a729107a90eecc090864b977beb6cfc8acec8c076336b0a7bada`；
- `vllm/v1/attention/ops/triton_oscar_mla_decode.py`：
  `13953366bb1e6a81fa3b858379f9abc61505284f1b911e7d216fa8099551942f`。

本阶段没有分配 GPU，没有新建候选镜像，也没有新的模型加载、苹果800 CUDA
correctness、32K/batch1 TTFT/TPOT/吞吐或 GSM8K 精度结果。因此当前性能对比仍是
2.62 的同负载正式值：BF16 TTFT/TPOT 分别为 `12528.025781735778 ms` /
`178.8317383000544 ms`，OSCAR 分别为 `32449.24456657221 ms` /
`199.15507386714768 ms`。下一步先发布本节与 submodule pointer，再基于
`67a0e47ff` 构建不可变候选镜像，完成模型加载/78层 contiguous tensor/显存和
完整 cold-cache CUDA 正确性门禁；这些门禁通过后，才运行同一 32K/batch1/
output128/TP8 端到端性能对比。

### 2.74 Contiguous inverse 的 Phase 6 候选输入迁移

2.73 的源码候选、submodule pointer、报告与 planning 已由主仓库提交
`83a1df0ea1ca6eb1d56403fbcf0aca4f121da115` 发布；主仓库和源码仓库均与
各自远端一致。随后只迁移 Phase 6 候选构建输入，Stage 9 Dockerfile、wrapper、
performance matrix 和正式 overlay 配置均未提前修改。

`configs/phase6/candidate_inputs.json` 的最小变化为：

- output tag 切换为 `glm52-oscar-a800-phase6-67a0e47ff-0275043c`；
- source commit/tree 切换为
  `67a0e47ff72f10a322de17b81c4134984e017bd6` /
  `60d5e606ce522dd78fecd890509372b727802f43`；
- Dockerfile SHA256 切换为
  `42b772b0f322b884d426e0b4c67b65b2aa83bbe2b1f6632b274cf11c68d9bf26`。

`docker/Dockerfile.phase6-oscar` 只同步修改默认 `SOURCE_COMMIT` 和
`SOURCE_TREE`。Phase 0 base manifest、rotation artifact、runtime expectation、
native extension manifest、复制路径、环境变量和 OCI label 结构均保持不变。
两个输入文件当前 SHA256 分别为：

- candidate inputs：
  `13141613fbf76f805a9c23c7b75633bce3ff4b96c62d6472e1a26b15e5a6ce09`；
- Phase 6 Dockerfile：
  `42b772b0f322b884d426e0b4c67b65b2aa83bbe2b1f6632b274cf11c68d9bf26`。

固定 ca4a404e9 控制镜像、network none、4 CPUs、无 GPU 中，配置与构建器门禁
结果为：

- candidate inputs JSON 解析通过；
- build/verify/test 三个 Phase 6 Python 文件 compile 通过；
- PAX header 确定性 unittest `1/1 passed`、0 failed；
- manifest source commit/tree 与已发布源码 Git identity 精确一致；
- manifest 内 Dockerfile SHA256 与文件实算值一致；
- `git diff --check` 通过。

本节只完成可复现构建输入迁移，没有执行 OCI 构建、daemon 导入或 runtime import，
也没有分配 GPU，因此没有新的模型加载、显存、CUDA correctness、32K/batch1
性能或 GSM8K 精度结果。下一步先发布这两个输入文件和本节，再在 clean/published
状态下以两个独立输出目录构建并递归验收 candidate OCI；只有确定性内容与递归
身份全部通过后，才迁移 Stage 9 控制镜像。

### 2.75 Contiguous inverse 候选 OCI 的首次 CPU-only 构建与递归验收

2.74 的 Phase 6 输入与报告由主仓库提交
`abf870d237f24a831652dce0136a8c9bd71912b2` 发布，发布状态由
`51c6e310438193440e08487185534dab82b1ab55` 固化；构建启动时主/源码仓库均为
clean/published。首次输出目录为：

`artifacts/phase6/20260801T004550Z_candidate_67a0e47ff_contiguous_inverse_v1`。

CPU-only builder 状态为 `built`，递归 verifier 状态为 `passed`。候选内容身份为：

- image/config：
  `sha256:22c2539e42b27a6e3740a9add92de45dea3520920b35b2175e15377271c39c66`；
- manifest：
  `sha256:f700ee725986a14dae34509282522bc2831072e8edb33b5d3ccc30b06419a537`；
- candidate layer：
  `sha256:37e119e5f697c933dcd4cbb76d121dd8c05cafdea00e8ba0be8c090b6529a2b2`；
- diff-ID：
  `sha256:a11fef0c12e27361b887f7e0f28861e3e327bd1437bf11f9fa8fcdfd91a991e7`；
- 层数：base 32、candidate 33；
- candidate layer：109,147,697 bytes、5,298 members，无原生扩展、无 whiteout。

递归验收确认 base 32 层精确继承，4,744 个源码文件与
`67a0e47ff72f10a322de17b81c4134984e017bd6` 的 Git tree 精确一致；7 个 lower
native extension 均由 base 提供且 hash 匹配，candidate layer 没有覆盖；4 份
rotation artifact、runtime expectation 和三个 runtime environment 变量全部匹配。

build/verification report SHA256 分别为：

- `a2e1c9d1731f43aa7c6d2a653dfffb185eb46051e453a949589a36ef3ef7bb54`；
- `285ed9978bd39e8b7b6fec71796ebedbcaa3b926f1fb75a5f5508895f808185a`。

读取证据时出现一次时序错误：build report 已可见后，首次读取 verification report
得到 `FileNotFoundError`；目录复查显示原 verifier 仍在完成递归检查，最终文件
mtime 比 build report 晚约 38 秒并为 `passed`。本轮没有重跑、覆盖或删除该现场；
后续长组合会等待目标文件和进程终态后再读取。

本阶段没有注入 NVIDIA runtime，也没有分配 GPU；没有新的模型加载、显存、CUDA
correctness、32K/batch1 性能或 GSM8K 精度结果。下一步先发布本节，再使用独立
新目录执行第二次 CPU-only 重建与递归验收；只有两轮 OCI 不可变内容逐字节一致，
才接受该候选并迁移 Stage 9。

### 2.76 Contiguous inverse 候选 OCI 的独立重建确定性

2.75 的首次构建记录已由主仓库提交
`db6303b1fedf3dfcb62d400084c59e605f3a54e3` 发布。随后在独立新目录执行第二次
CPU-only build 与递归 verifier：

`artifacts/phase6/20260801T004846Z_candidate_67a0e47ff_contiguous_inverse_v2_rebuild`。

v2 的 build/verification 状态同样为 `built/passed`。显式等待 verification report
出现后再读取，确认 4,744 个源码文件、7 个 base native extension、4 份 rotation
artifact、runtime expectation、base 32 层与 runtime environment 门禁全部通过。

v1/v2 的以下四项文件已直接逐字节比较并完全一致：

| OCI 内容 | bytes | SHA256 |
|---|---:|---|
| `index.json` | 284 | `d440789212be1e3ba34e90afde7f9cd184f58427ebff5f196a6f623a8a8a9290` |
| candidate config | 32,352 | `22c2539e42b27a6e3740a9add92de45dea3520920b35b2175e15377271c39c66` |
| candidate manifest | 5,662 | `f700ee725986a14dae34509282522bc2831072e8edb33b5d3ccc30b06419a537` |
| candidate layer | 109,147,697 | `37e119e5f697c933dcd4cbb76d121dd8c05cafdea00e8ba0be8c090b6529a2b2` |

两轮 diff-ID 也同为
`sha256:a11fef0c12e27361b887f7e0f28861e3e327bd1437bf11f9fa8fcdfd91a991e7`。
v2 build/verification report SHA256 分别为：

- `7935c471e637eb5dcaa5efa84b94fe8d656735f25282cf18757c385e8490c7bc`；
- `df5171373a8d565201a326526749a71dbd797e2073b7cf3ec0fabc22816a2597`。

两轮 report 本身包含不同输出路径和各自构建时的主仓库提交，因此 report hash
不同是预期现象；不可变 OCI 内容的四项逐字节一致才是确定性验收依据。

本阶段仍没有注入 NVIDIA runtime 或分配 GPU，没有新的模型加载、显存、CUDA
correctness、32K/batch1 性能或 GSM8K 精度结果。双构建与递归验收已经通过；
下一步先发布本节，再把 v1 OCI layout 导入 Docker daemon并审计 image/layer/label
身份，导入成功前不迁移 Stage 9。

### 2.77 Contiguous inverse 候选 OCI 的 daemon 导入与身份审计

2.76 的双构建确定性记录已由主仓库提交
`a8792e778677284fc6eff4e0815d18ea2e7c4376` 发布。导入前确认目标 tag
`glm52-oscar-a800-phase6-67a0e47ff-0275043c:latest` 不存在，并从 v1
`index.json` 原样读取 OCI ref name，未手工改写 source ref。

一次性 Ubuntu 22.04 工具容器使用阿里 HTTP 源安装 `skopeo 1.4.1`，从只读 v1
OCI layout 复制到 Docker daemon；工具容器没有 GPU DeviceRequests，最终 exit=0
并自动删除。导入后 daemon 身份审计状态为 `passed`：

- image ID：
  `sha256:22c2539e42b27a6e3740a9add92de45dea3520920b35b2175e15377271c39c66`；
- 层数：33；
- 最后一层 diff-ID：
  `sha256:a11fef0c12e27361b887f7e0f28861e3e327bd1437bf11f9fa8fcdfd91a991e7`；
- source commit/tree：
  `67a0e47ff72f10a322de17b81c4134984e017bd6` /
  `60d5e606ce522dd78fecd890509372b727802f43`；
- Dockerfile、rotation manifest、rotations 与 runtime expectation 四项 label
  均与 Phase 6 manifest 精确一致。

daemon inspect JSON SHA256 为
`8724ac2d8418f8b88377fa31322f13735f6cf62316c9911297e454ea74a3d969`。
唯一外部下载容器继续为 `DeviceRequests=null`，没有执行终止操作。

导入前检查中再次误用 `docker ps` 不支持的 `.HostConfig.DeviceRequests` 模板并
退出；该错误发生在工具容器启动前，没有改变 daemon。随后改为逐个 `docker inspect`
完成相同检查，未重复错误模板。

本阶段没有注入 NVIDIA runtime 或分配 GPU，也没有新的模型加载、显存、CUDA
correctness、32K/batch1 性能或 GSM8K 精度结果。下一步先发布本节，再把 Stage 9
控制镜像 base 切换到该已审计 daemon tag并完成CPU-only构建/身份/runtime检查；
控制镜像通过前不运行driver-injected模型加载。

### 2.78 Contiguous inverse 的 Stage 9 控制镜像输入切换

2.77 的 daemon 导入记录已由主仓库提交
`222c0aac11687ba913a56ed6faba2939759119da` 发布。随后只把
`docker/Dockerfile.phase9-runtime` 的默认 base 从旧 ca4a404e9 候选切换为：

`glm52-oscar-a800-phase6-67a0e47ff-0275043c:latest`。

该文件 diff 只有一行；apt 源、`git/iproute2` 安装、package audit、USER、
entrypoint 和其余构建逻辑均未修改。新 Dockerfile SHA256 为
`65f1ed38599af68d7c836a1676225d56897eb26c585c49585630fbc6cc9b79fc`。

本阶段刻意没有同步修改 `configs/phase9/performance_matrix.json`、
`scripts/phase9/run_containerized_performance.sh`、Phase 1/5/7 配置或正式 overlay；
它们继续绑定 ca4a404e9，等新控制镜像完成 CPU runtime 身份验收后再按依赖顺序迁移。

本节没有构建控制镜像，没有注入 NVIDIA runtime 或分配 GPU，也没有新的模型加载、
显存、CUDA correctness、32K/batch1 性能或 GSM8K 精度结果。下一步先发布这一行
输入变更，再构建 `oscar-glm-stage9-runtime:67a0e47ff`，核对前 33 层继承、labels、
entrypoint、固定软件版本和 `cuda_initialized=false`。

### 2.79 Contiguous inverse 的 Stage 9 控制镜像 CPU 验收

2.78 的控制输入由主仓库提交
`25ace9787555cd9cfdcb15ccb6bba38325021b75` 发布。首次 build 错把项目根目录
作为 context；运行 10 分钟仍未产生 image，FD 诊断证明 Docker 正在遍历 NFS 上的
`artifacts/phase0-candidate-bundle/rootfs`。正式 wrapper 实际使用
`${PROJECT_ROOT}/docker`。因此向本轮 build PID 发送 SIGTERM，进程正常退出，目标
tag 仍不存在；没有把该轮记作构建成功。

第二轮严格复用正式 wrapper 口径：显式 base build-arg、同一 Dockerfile、
`docker/` context。实际 context 只有 4.608 kB，构建完成并生成：

- tag：`oscar-glm-stage9-runtime:67a0e47ff`；
- image ID：
  `sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`；
- 层数：34；
- 控制层 diff-ID：
  `sha256:126c2fb0b8f8767d7598ecab8dd56fc4538d50eebd4f77c50d7290b7671f1343`。

候选 base 为 33 层，新控制镜像前 33 层逐层完全一致；全部继承 labels 和
`["/bin/bash"]` entrypoint 与 base 精确一致。network-none、4 CPUs、无 GPU 的
runtime 检查确认：

- Git `2.34.1`、iproute2 `5.15.0`；
- PyTorch `2.11.0+cu129`；
- public sparse prefill 存在 keyword-only `inverse_rotation=None`；
- 打包后的 `MLAAttention.__init__` 包含
  `runtime_parameters.rotation.T.contiguous()` buffer 构造；
- `cuda_initialized=false`。

无 NVIDIA runtime 时出现缺 `libcuda.so.1`、0 active Triton driver 和
`vllm._version` warning，均未终止 Python import；这符合 CPU-only 控制验收边界，
不能替代后续 driver-injected 检查。

本阶段没有分配 GPU，也没有新的完整 CUDA correctness、模型加载显存、
32K/batch1 性能或 GSM8K 精度结果。控制镜像 CPU 门禁已经通过；下一步先发布本节，
再按 Phase 1→5→7→9 依赖顺序迁移正式 overlay/config/wrapper并执行递归静态验收，
静态链路发布前不申请 GPU。

### 2.80 Contiguous inverse 候选 driver-injected runtime import

2.79 的控制镜像 CPU 验收已由主仓库提交
`26de4085978899db953e6761b828548178986b9c` 发布。Phase 7 配置必须冻结新候选的
`runtime_import.json` 路径与 SHA256，因此在正式 overlay/config 迁移前，先完成该
只读运行时身份门禁；本节没有改变 2.79 已发布的控制镜像或候选源码。

分配前两次空闲检查时间为
`2026-08-01T01:11:53Z/01:14:08Z`，间隔 135 秒；两次均为 8/8 张苹果800
`0 MiB/0%`，没有 compute process。唯一运行的项目外下载容器
`deepseek_v4_hf_downloader_vllm0230` 的 DeviceRequests 为 null，未占用 GPU，
因此没有终止该容器。启动前 `01:16:10Z` 的即时复查仍为 8/8 卡全部空闲。

有效探针固定只把 GPU 0 映射给
`oscar-glm-stage9-runtime:67a0e47ff`，使用正式 Python
`/opt/fp8_speed_up_v4_venv/bin/python`，只注入 NVIDIA driver 用户态库。
探针导入候选 vLLM Python 与原生扩展、读取 rotation/runtime expectation，并在
退出前断言 `torch.cuda.is_initialized()` 仍为 false；没有加载模型、分配模型显存
或执行 CUDA kernel。本轮一次通过，退出码为 0，`runtime_import.json` 状态为
`passed`，实测身份为：

- Python/PyTorch/Triton：`3.12.13/2.11.0+cu129/3.6.0`；
- Transformers/Tokenizers：`5.8.1/0.22.2`；
- FlashInfer Python/JIT cache：`0.6.6/0.6.6+cu129`；
- vLLM Python：`/opt/vllm_glm52_v1/vllm/__init__.py`；
- vLLM 原生扩展：`/opt/vllm_glm52_v1/vllm/_C.abi3.so`；
- vLLM dist info/reported version：`0.11.2.dev278+gdbc3d9991/dev`；
- rotation 数量为 78，rotation manifest、rotations 与 runtime expectation
  三项 SHA256 全部匹配；
- `reasoning_effort=max`，`cuda_initialized=false`。

导入 vLLM Python 时只出现与既有有效轮次一致的 RuntimeWarning：候选源码包没有
生成版 `vllm._version`，所以 reported version 为 `dev`。候选 Python/原生扩展路径
已经由 OCI、daemon 与控制镜像身份门禁绑定；该 warning 没有导致断言放宽或失败。
有效 JSON 与上一版冻结 runtime import 协议逐字节一致。

双空闲检查、启动前复查、有效 JSON、有效日志、退出码、退出后快照和最终复查的
SHA256 依次为：

- `0da4353a74c2561370af56063f0bfcfd83efbe067916121752a54838c39bd315`；
- `24c14d00cd79a7aa97d0b8c787a43c0c2cb539a5d0a5f47fc68e2cbae9db1f51`；
- `0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`；
- `f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `6128a07272d6a8a6a8d209566a7a4043a2c1add852ebeb34d9d49ab3a5905896`；
- `dc0a052cafb5ddef1cd3e22b88ef2a8aaf6c88ec7d5fd47b3b5f72ecf83b28aa`。

有效容器已自动删除；`01:16:16Z` 的退出快照和 `01:17:54Z` 的最终复查均为
8 张 GPU `0 MiB/0%`、没有 compute process。一次探查 artifact schema 的宿主命令
误用了缺少 `pathlib` 的默认 `python`，在读取任何 JSON 前退出；随后明确改用
`python3` 完成只读检查。该错误没有启动容器、修改候选或进入上述有效探针。

本节不是模型加载、CUDA correctness、32K/batch1 性能或 GSM8K 精度测试，因此
没有新的 TTFT、TPOT、吞吐或精度结果。下一步先发布本节，再按
Phase 1→5→7→9 依赖顺序迁移正式 overlay/config/wrapper并执行工具测试与递归静态
验收；静态链路发布前不申请 GPU。

### 2.81 Contiguous inverse 正式链路静态迁移

2.80 的 runtime import 记录已由主仓库提交 `7bfee80` 发布。随后从验收通过的
`20260801T004550Z_candidate_67a0e47ff_contiguous_inverse_v1/extracted-layer`
机械派生正式 overlay：4,749 个普通文件与候选层逐文件一致，另为候选层按设计不含的
6 个 native extension 建立指向已验收 Phase 0 base 的只读绝对 symlink。

候选层与 overlay 的普通文件递归 SHA256 清单逐字节一致，两个清单 SHA256 均为
`797e7c2e66276b83fb731e4416a3ea6a1fd6a9db1c93ea04a12ee0d4f9dbee83`。
6 个链接清单与链接目标哈希清单的 SHA256 分别为：

- `f17949949ff89f8a6e2624c9999f4276cfbe9ab4da42cc586029b0bf373276b2`；
- `c2a5c7c2953265878d11f0b0e54d7a8c362c6aad68ce433a20a441e713149468`。

两份链接清单与上一正式链路逐字节一致，说明本轮只替换候选 Python/source 和
OSCAR artifact，未替换冻结的 lower native binary。首次 overlay 脚本错误假设
候选层已包含 6 个 native 文件，在检查第一个目标不存在时退出；4,749 个普通文件
已经复制完成，但没有文件被删除。复核候选层结构后，只为上述 6 个精确路径创建
symlink，没有重复制或覆盖候选普通文件。

正式身份按 Phase 1→5→7→9 的依赖顺序迁移。Phase 1/5 绑定源码
`67a0e47ff72f10a322de17b81c4134984e017bd6` / tree
`60d5e606ce522dd78fecd890509372b727802f43`；Phase 7 绑定新 OCI、overlay、
build/verification/runtime import；Phase 9 绑定候选 image、控制镜像
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`
和相同源码。4 份配置的 SHA256 依次为：

- Phase 1：`1d33af7fa9f3f4114d03d97de1e7733ceea94296237eaecd950db44f90c29ee1`；
- Phase 5：`a9508b0d1ceca3c9971354165d58f2350efcd395705a5ad63c35a129ce17ed15`；
- Phase 7：`0c3c97cfda931468abc9fe39d711d7dc6a442dca479e609bdaf7fa0610f56783`；
- Phase 9：`a478b72d2e13c0f5794b7ca4a8749e97471bba1c1076ec612cf5f9d8438d3f4b`。

9 个正式 shell 入口同步到相同身份，旧 ca4a404e9 commit/tree、OCI 路径和控制
镜像身份在 `configs/`、`scripts/` 中均清零。第一次静态输入验证中，手工从缩写
补全的 build/verification report 哈希错误；12 项中其余 10 项通过，shell 与测试
尚未启动。直接从落盘文件重新实算并修正 Phase 7 配置后，v2 为 12/12 passed，
shell 语法为 9/9 passed。

固定控制镜像、network-none、无 GPU 的工具结果为：

- Phase 7 工具测试：20/20 passed；
- Phase 9 工具测试：47/47 passed；
- Phase 9 全部 15 个 Python 文件：15/15 compile passed。

递归 verifier 首轮为 62/66 passed。全部候选 OCI/source/overlay/artifact/config/
evaluator 检查均已通过；唯一失败是 Phase 1/5 的 4 个汇总状态。定向 Stage 5
诊断确认底层原因是 NFS 将 Phase 0 source 的 Git `100644` mode 映射为
`100755`，不是内容差异。v2 在同一控制容器命名空间内把既有只读
`oscar-glm-phase0-source-fd3e0b3` 卷挂载到精确 base source 路径，不修改文件、
配置、verifier 或阈值；结果为 66/66 passed、退出码 0。本轮代码新增检查后，
实际总数为 66，不能沿用旧候选的 64 项口径。

最终静态汇总状态为 `passed`：输入 12/12、shell 9/9、Phase 7 20/20、
Phase 9 47/47、compile 15/15、recursive 66/66。清单覆盖 25 份证据，共
114,644 bytes；静态汇总、递归 JSON/log 和证据清单 SHA256 分别为：

- `ce56900f46f3382bd874cacbeb3347e96208364146264499f6287055752c673c`；
- `5e652f0c316b991bbb4126ed0f156867d39863e08fcb107dce366c52c2c2c449`；
- `3e4caaa4b08fa0ee2347fc83f911d945b628737433f3968eb5d3ac5f0bdcc5ba`。

CPU-only 阶段前后 `2026-08-01T01:26:31Z/01:31:01Z` 的 8 张 GPU 均为
`0 MiB/0%`，没有 compute process。本节没有执行 driver-injected preflight、
模型加载、CUDA correctness、32K/batch1 性能或 GSM8K 精度，因此没有新的
TTFT、TPOT、吞吐或精度结果。下一步先发布本节和正式配置；发布后重新执行两次
GPU 空闲检查，再运行 driver-injected preflight。

### 2.82 Contiguous inverse 的 driver-injected preflight

2.81 的正式静态链路已由主仓库提交 `4dddc09` 发布；随后 planning 状态由
`d1b7089` 发布，主仓库与源码仓库在进入本轮前均为 clean/published。

preflight 前双空闲检查为
`2026-08-01T01:33:27Z/01:34:40Z`，间隔 73 秒；两次均为 8/8 张苹果800
`0 MiB/0%`，没有 compute process。唯一项目外下载容器的 DeviceRequests 为
null，没有占用 GPU，因此没有终止。启动前 `01:35:12Z` 的即时复查仍为 8/8 卡
全部空闲。

正式 run ID 为
`20260801T013320Z_stage9_candidate_67a0e47ff_preflight_v1`。控制镜像为
`oscar-glm-stage9-runtime:67a0e47ff`，image ID 为
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`；
容器注入 8 张 GPU 的 driver/runtime，但只执行正式 wrapper 的
`preflight-candidate`，没有启动服务或加载模型。

preflight 退出码为 0，结果包括：

- 静态递归身份：66/66 passed；
- 固定 Python 环境导入：passed，`cuda_initialized=false`；
- 服务参数解析：passed，`cuda_initialized=false`；
- 解析到 tensor/pipeline parallel 为 `8/1`、`max_model_len=131072`、
  `max_num_seqs=16`、`max_num_batched_tokens=2048`；
- attention backend/KV cache dtype 为
  `TRITON_MLA_SPARSE/oscar_mla_int2`；
- eager、chunked prefill 与 torch profiler 参数均与正式配置一致。

固定环境继续得到 Python/PyTorch/Triton
`3.12.13/2.11.0+cu129/3.6.0`，vLLM Python 和原生扩展均来自新 overlay。
`vllm._version` 缺失产生与既有候选一致的 RuntimeWarning，但没有改变退出状态或
断言。两处 CUDA 均未初始化，说明本轮只验证 driver 可见条件下的 import 和参数链，
不能替代真实模型加载或性能测试。

证据清单覆盖 10 份文件，共 41,458 bytes。双空闲检查、启动快照、完整 preflight
日志、退出码、静态 JSON、固定环境 JSON、解析参数 JSON、退出快照、最终快照、
验证 JSON 与证据清单 SHA256 依次为：

- `f1d5f6f34aab24f14c25aab9f7de01e23ed7e48c9797f94edec89e6fcb080609`；
- `d341723328148d9e22c2cb538394481e3c421cc05477f545d45881cdafe0f151`；
- `a376b58bb6f4456566b5d445199a541b1ac82715906031f70ab713d05458ccfb`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `7fa78bac303dc4604eca1642389bb64032e69cb84edb360bae20f58d435f2a61`；
- `30e969900b6925c31b79223375fd88636efe7151bdad1dee2b0d9e8276e33e4b`；
- `b0a67b21f963c6763d19aff0ce73f12af780335f90b3b6907348a97e6151ec8f`；
- `710428841c77b7ad1f99c8c989c08b42a36d509b0f91514e969358caa6f9d43a`；
- `c0478c5686cf8f8b12dab540bfed03886b25eb5a65e861134d6779de5f000eb2`；
- `5a3760c8e99ad632985f912bb7adafe4866c1e546073bef9153efa3a1db23628`；
- `df26f02857474362e314c56529abff6c1c38f964578ba3c9df9df3cbb20ddc83`。

有效容器已自动删除；`01:36:15Z` 退出快照和 `01:36:47Z` 最终复查均为
8 张 GPU `0 MiB/0%`、没有 compute process。本节没有模型加载、完整 CUDA
correctness、32K/batch1 性能或 GSM8K 精度，因此没有新的 TTFT、TPOT、吞吐或
精度结果。下一步先发布本节；发布后重新执行双空闲检查，再以固定
`32K/batch1/output128/TP8` 单格正式运行验证 contiguous inverse 的端到端收益。

### 2.83 Contiguous inverse 的 32K/batch1 正式性能结果

2.82 已由主仓库提交 `e02fd5f4e9fc598474ee88832512048565648122` 发布，
随后 planning 状态由 `9d287f4466a556dccab3516009f5d1b8fe5570cd` 发布；正式轮次
启动前主仓库与源码仓库均为 clean/published。两次有效 GPU 空闲检查时间为
`2026-08-01T01:39:26Z/01:40:41Z`，间隔 75 秒；8/8 张苹果800 均为
`0 MiB/0%`、无 compute process。项目外下载容器
`deepseek_v4_hf_downloader_vllm0230` 的 DeviceRequests 为 null，未占用 GPU，
因此没有终止。

正式 run ID 为：

`20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`，image ID 为
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`；
候选源码提交为 `67a0e47ff72f10a322de17b81c4134984e017bd6`，性能配置 SHA256
为 `a478b72d2e13c0f5794b7ca4a8749e97471bba1c1076ec612cf5f9d8438d3f4b`。
固定负载为 input length 32,768、batch size 1、output length 128、TP=8，按与
2.62 相同的三轮正式测量加 8 tables、8 worker traces、1 frontend trace profiler
协议执行。服务在 `01:48:29Z` ready，启动耗时 300 秒；Docker 最终退出码为 0，
top-level summary、cell summary、三轮 validation 与 profile validation 均为
`passed`，matrix 总耗时 `1609.7440812587738 s`。

三轮均为 3/3 completed、0 failed，实际结果为：

| 轮次 | mean TTFT（ms） | mean TPOT（ms） | 请求吞吐（req/s） |
|---:|---:|---:|---:|
| 1 | 30539.197439017396 | 203.5127659658278 | 0.017735015695573196 |
| 2 | 30541.20112862438 | 202.51436330123838 | 0.0177743652305229 |
| 3 | 30535.28801413874 | 200.28568437357194 | 0.017866137433002558 |

三轮中位汇总为 mean TTFT `30539.197439017396 ms`、mean TPOT
`202.51436330123838 ms`、请求吞吐 `0.0177743652305229 req/s`；同时 median
TTFT/TPOT 为 `30537.7194872126/201.4937229806513 ms`，output/total token
throughput 为 `2.275118749506931/584.7055186232813 token/s`。mean TTFT 三轮
相对极差仅 `0.019362%`，mean TPOT 相对极差为 `1.593508%`；服务侧没有
preemption、waiting request 或容量限制，三轮峰值显存均为 `80757 MiB/GPU`。

与 2.62 的上一版已发布 OSCAR 结果及同一 32K/batch1 BF16 baseline 比较：

| 对照 | mean TTFT 变化 | mean TPOT 变化 | 请求吞吐变化 |
|---|---:|---:|---:|
| 上一版 OSCAR（2.62） | -5.886260691% | +1.686770700% | +2.607779616% |
| BF16 baseline | +143.767038567% | +13.242965274% | 未在本轮重测 |

因此 contiguous inverse 在完整正式负载上把 TTFT 从
`32449.24456657221 ms` 降到 `30539.197439017396 ms`，节省
`1910.047127554814 ms`，并把请求吞吐提高 `2.608%`；但 TPOT 从
`199.15507386714768 ms` 回退到 `202.51436330123838 ms`。这是明确的 TTFT
改善，不是全面性能胜出。相对 BF16，TPOT 仍在 `+20%` 上限
`214.59808596006527 ms` 内；TTFT 上限为 `15033.630938082934 ms`，当前仍高
`143.767%`，性能优化尚未关闭 TTFT 门限。

profiler 状态为 passed，耗时 `743.1806800365448 s`，实际生成 8 张 CUDA
算子表、8 个 worker trace 和 1 个 frontend trace；worker trace 合计
`1213749498 bytes`，frontend trace 为 `1092 bytes`。critical rank=6，
critical-rank kernel total=`63359 ms`；profile 峰值显存为 `80769 MiB/GPU`。
停止 profiler 后，8 个 worker 在 CPU 侧解析与压缩约 1.2 GiB trace，因此 GPU
利用率一度为 0%，但进程持续高 CPU，随后 8/8 CUDA 表全部生成并正常退出；没有
OOM、CUDA error 或超时。

长实验进度按 10 分钟门限输出：服务监控在
`01:53:29Z/02:03:29Z/02:13:30Z` 分别打印 600/1200/1801 秒；profile 客户端在
`02:13:00Z` 打印 600 秒。容器已自动删除；`02:15:32Z` 的退出快照无 compute
process，紧接着独立复查 8 张 GPU 均为 `0 MiB/0%`。

小型正式证据已复制到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_results`。

目录封存 41 项证据，已 41/41 通过 manifest 复算；连同
`evidence_manifest.sha256` 共 42 个文件，`du -sb` 为 1,103,416 bytes。
top-level summary、cell summary、profile validation、manifest、comparison 与
outer log SHA256 分别为：

- `62b112567621f94275c7f50c5f0234ba008fa2128b528d467defad586180599d`；
- `098c28d2221cde03e4b85cd1b18e38c352e3917dd21319f9238b9bb9a06db85b`；
- `3be4c69336c3d60df4a22cf06701b05a15486a3deafbe35826332ad7337ad218`；
- `c9aa31f78ec0269d95d33f6b7387fba36ef83ae05c8aa52a97bfe2488bc39a3d`；
- `fd44144ab22b543f3040113f377508fbf4b77f0fb8812180de07c8037e520a21`；
- `2146591af403563d910b3fcbbbe748956f1ac52ad5a62fa774cc9def71e0550b`。

约 1.2 GiB 原始 trace 保留在 `/dev/shm`，没有复制进仓库；正式 summary 已记录
全部 trace 的路径、字节数与 SHA256。本轮没有修改模型、数据集或精度配置，也没有
产生新的 GSM8K 精度结果；性能结论只适用于上述 32K/batch1 单请求负载。下一步先
发布本节与 planning，再对本轮冻结 trace 做 CPU-only 归因，定位约 15.5 秒的剩余
TTFT 门限差距后再选择下一项最小候选。

### 2.84 Contiguous inverse 的 32K trace CPU-only 归因

2.83 的正式结果已由主仓库提交
`fcdcac0ebdfe2ae1173638ba8be4a884bfa43fbe` 发布，发布状态由
`162e77a` 固化；主仓库与源码仓库在本阶段开始前均为 clean/published。本阶段只
读取 2.62 与 2.83 已冻结的 profiler trace，没有修改生产源码、模型、正式配置或
镜像，也没有向容器分配 GPU。

候选和参考的 analysis ID 分别为：

- `20260801T0222Z_contiguous_inverse_32k_prefill_trace_v1`；
- `20260801T0225Z_topk_sort_format3_reference_v1`。

两轮均固定使用 `oscar-glm-stage9-runtime:67a0e47ff`、runc、network none、
4 CPUs、空 `CUDA_VISIBLE_DEVICES`、`NVIDIA_VISIBLE_DEVICES=void`，以及
Python/ijson `3.12.13/3.4.0.post0`。analyzer 为已发布的 format version 3，
SHA256 为 `724aeb5e45f8a9322b7e52d096fb38670ec768f89cb9844d1d49ab213cddbf43`。
候选轮次从 `02:22:04Z` 到 `02:24:06Z`，analyzer 内部耗时
`121.28143209964 s`；参考轮次从 `02:25:27Z` 到 `02:27:26Z`，内部耗时
`118.892893427052 s`。两轮均 exit=0、summary 状态为 passed，8/8 ranks 均为
144 个 execute context、16 个 prefill chunk、精确 32,768 tokens。

为消除工具版本差异，2.62 的排序参考 trace 也用当前同一 format-v3 analyzer
重新解析；其 aggregate 与 2.63 的 format-v2 冻结结果逐字段一致。两组
profile-to-profile 中位结果如下：

| 指标 | 上一版 OSCAR（2.62） | Contiguous inverse | 变化 |
|---|---:|---:|---:|
| prefill wall（ms） | 32478.9677565 | 30570.2517695 | -1908.715987（-5.876775%） |
| prefill kernel（ms） | 31481.2476750 | 29553.8873795 | -1927.360295（-6.122249%） |
| `_rotate_latent_kernel`（ms） | 3390.9417820 | 1486.4346605 | -1904.507122（-56.164548%） |
| `_mixed_sparse_prefill_stage1`（ms） | 19849.3938800 | 19846.5877630 | -2.806117（-0.014137%） |
| `topKPerRowPrefill`（ms） | 251.6080345 | 251.3525445 | -0.255490（-0.101543%） |

rotation 调用数保持 4,898，stage1/top-k 调用数保持 1,248/1,344。rotation
减少的 `1904.5071215 ms` 解释 prefill wall 改善的 `99.779492%`；去掉
rotation 后的 residual wall 只从 `29088.0259745 ms` 变为
`29083.8171090 ms`，下降 `4.2088655 ms`。因此 2.83 的 TTFT 改善不能归因于
stage1、top-k 或调度偶然波动，因果主体就是预存 contiguous inverse 后的
inverse rotation 访存改善。

逐 rank 与逐 chunk 方向也一致：8/8 rank 的 prefill wall、kernel 和 rotation
均下降；rank wall 变化范围为 `-1910.088545–-1907.435962 ms`，rotation 变化
范围为 `-1909.725942–-1903.141927 ms`。16/16 个 2,048-token chunk 的
rotation 都减少约 119 ms，范围为 `-119.080431–-119.012720 ms`；chunk wall
变化范围为 `-124.178443–-110.597555 ms`。端到端正式 mean TTFT 改善为
`1910.047127554814 ms`，profile wall 改善为 `1908.715987 ms`，后者解释前者的
`99.930308%`，两种独立口径高度吻合。

该优化关闭了 rotation 主瓶颈，但没有关闭整体 TTFT 门限。当前 stage1 仍为
`19846.587763000007 ms`，占当前 prefill wall 的 `64.921244%`；当前 prefill
wall 比 2.53 冻结的 BF16 trace 高 `20483.782002 ms`，stage1 比 BF16 原生
attention 的 `3384.373974500002 ms` 高 `16462.2137885 ms`，仍解释 wall 差距的
`80.367062%`。因此下一项最小优化仍应针对全部 16 个 chunk 的 grouped prefill
stage1 有效计算或访存，而不是继续优化当前仅占 wall `4.862357%` 的 rotation。

首次预检把输出目录放在 `/dev/shm/oscar-glm-stage9/analysis` 下，该目录由 root
创建，宿主用户在 analyzer 启动前收到 `Permission denied`；同一组合 shell 未启用
fail-fast，随后仍完成了只读 Python/ijson/analyzer/8-trace 身份预检，但没有启动
分析或写入 summary。正式输出改到任务专属可写 `/dev/shm` 根目录。候选与参考
目录标签最初又分别写成未来分钟 `0225Z/0227Z`；完成后按落盘 start UTC 只做目录
重命名，修正为上述 `0222Z/0225Z`，没有重跑或修改 summary 内容。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_trace_analysis_v1`。

目录包含两份 format-v3 summary、comparison builder/output、validation、两组
trace input hash、运行日志/身份/起止/退出码与退出状态，共 18 项证据；已 18/18
通过 manifest 复算，连同 manifest 共 19 个文件，`du -sb` 为 8,087,709 bytes。
候选 summary、参考 summary、comparison、validation、run identity、退出状态与
manifest SHA256 分别为：

- `359ef056ef75f933420e6cac7b4a4b5295b5779bcc2831a8678abd617b5eaeab`；
- `8c87e447f1a85a3908f33ca0a0ca9d513a8eb7fa5ef6f2acc1ad63b4f975d0d8`；
- `9a25177ae252f338b37775ea785beacb25dba919454557fe78c6e394111bc802`；
- `68f3e57f2d065a39c301ab4a4a799ba332c5e1d1854ece5bd75e4b1ec8e43367`；
- `6b354eed8e75de9cb8efe291d189c8818945c3bfc3228174ebd79e1367465e40`；
- `366a681587be1bf75981b42933c59cd04ad885caa882c3bbed0ce3c8eb3e35dd`；
- `02a77de22baac82817afe0a62690e4ef164994ded3948f1c01f879a248b4c855`。

两组原始 worker trace 共 `2411923332 bytes`（约 `2.246 GiB`），继续只保留在
`/dev/shm`，没有复制进仓库；
两份 trace input hash 和 summary 已逐文件绑定 bytes/SHA256。退出复查没有残留
分析容器，8 张苹果800均为 `0 MiB/0%`、无 compute process。本阶段没有新的
GSM8K 精度、TTFT、TPOT 或吞吐测量；2.83 的正式性能结论不变。下一步先发布本节
与 planning，再对当前 stage1 的逐 chunk kernel/源码路径做只读筛选，形成下一项
最小候选前不修改 production 或启动 GPU 实验。

### 2.85 Stage1 cache-type 拆分的 CPU-only SM80 资源筛选

2.84 的 trace 归因与 planning 已由主仓库提交 `db9e027` 发布，后续发布状态由
`ac8abd0aabf9475d26986fb860c5140c3dc343e8` 固化；源码仓库继续固定在已发布的
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 OSCAR production
源码、模型、数据集、正式配置或控制镜像，只新增主仓库的离线编译筛选工具和测试。

2.84 证明当前 `_mixed_sparse_prefill_stage1` 为 `19846.587763 ms`，占
prefill wall 的 `64.921244%`；它相对 BF16 原生 attention 的超额仍解释两者
prefill wall 差距的 `80.367062%`。源码复核进一步确认，当前 mixed kernel 同时
保留 history 与 BF16 两套 512 维 FP32 accumulator。将两种 cache 独立计算在
数学上可行，但两条路径必须分别输出 LSE，并通过 log-sum-exp 重新合并；因此不能
只删除一套 accumulator 而保持现有单 kernel 输出不变，还要承担额外 launch、
重复 query/index load 与归约顺序变化。本阶段据此只做资源筛选，不提前实现正式
调度或宣称性能收益。

新增工具为 `scripts/phase9/compile_oscar_prefill_cache_split.py`，它保留当前 mixed
kernel 作为资源基线，并以实验性 standalone kernel 分别编译 history-only 与
BF16-only accumulator 路径。TDD 红灯在工具尚不存在时得到
`FileNotFoundError`、exit=1；最小实现与组合门禁补齐后，定向测试为 `4/4 passed`。
最终 Ruff 0.14.0 check/format、固定 Python compile、该测试与既有 prefill
benchmark 测试合计 `13/13 passed`，`git diff --check` 也通过。最终工具与测试
SHA256 分别为：

- `8e09ca7e793d016d711b20ed6af6b1869663789140215edf9b7bf49259b6f1fc`；
- `23226bd372e1ec6b402e6d20d57adaa9309a0c33f04a78d1e118b360080e64bf`。

最终有效离线轮次为：

`/dev/shm/oscar-glm-20260801T025156Z_prefill_cache_split_offline_v4`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）、
runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void`。环境为 Python 3.12.13、PyTorch
2.11.0+cu129、Triton 3.6.0，离线目标为 SM80，宿主 `cuobjdump` 为 CUDA
12.9。轮次没有初始化 CUDA，`cuda_initialized=false`；15/15 个 variant 编译
成功、0 rejected，内部耗时 `13.654013657011092 s`。当前 mixed h8/t16/w8
基线在同一离线工具链中精确复现 109,568-byte dynamic shared memory；该轮
`cuobjdump` 为 255 registers/thread、0-byte stack。离线寄存器数不能替代 2.42
等真实 runtime cubin 的资源记录，后续判断只在本轮各 variant 的同口径内比较。

关键资源结果如下。`dual-block` 仅表示按 166,912-byte shared memory、65,536
registers/SM 与 threads/block 做资源算术；`strict` 还要求 stack=0，不能把算术
可行直接写成真实 GPU 双驻留或性能结论。

| 路径/配置 | Shared memory | Registers/thread | Stack/thread | 离线门禁结论 |
|---|---:|---:|---:|---|
| mixed h8/t16/w8 | 109,568 B | 255 | 0 B | 资源基线，不支持双 block |
| history h8/t16/w8 | 84,992 B | 199 | 0 B | shared 超过双 block 线 1,536 B，register 也不通过 |
| history h4/t16/w8 | 76,288 B | 206 | 0 B | shared 通过，register 不通过 |
| history h4/t16/w4 | 76,288 B | 255 | 176 B | 资源算术可双 block，但有 stack spill |
| history h2/t16/w4 | 71,936 B | 255 | 184 B | 资源算术可双 block，但有 stack spill |
| history h1/t16/w4 | 69,760 B | 255 | 176 B | 资源算术可双 block，但有 stack spill |
| BF16 h8/t16/w8 | 42,496 B | 189 | 0 B | shared 通过，register 不通过 |
| BF16 h8/t16/w4 | 42,496 B | 255 | 0 B | strict 资源门禁通过 |
| BF16 h4/t16/w4 | 37,632 B | 255 | 0 B | strict 资源门禁通过 |

history 的 h8/t32/w8 与 h4/t32/w8 分别需要 152,576/143,872 bytes shared
memory，并出现 8/16-byte stack，因此也不构成候选。组合汇总中，history
h4/h2/h1 的 w4 与 BF16 w4 使
`cache_split_dual_block_feasible=true`；但所有 history 算术候选都有
176–184-byte stack，history strict candidate 为空，所以最终
`cache_split_strict_promotion_feasible=false`。也就是说，拆分确实显著降低了
shared/register 压力，但当前 history 路径尚未同时满足零 spill 与双 block 资源
线，不能直接进入 production 或 GPU 性能实验。

准备过程中保留了三个 fail-closed 边界。v1 的 `tee` 先在输出目录创建
`run.log`，触发工具的空目录契约，在任何 Triton 编译前退出；v3 已完成 15/15
编译，但随后 Ruff format 改变了工具哈希，因此不作为最终证据；封存 v4 时一次
`cp` 同时显式指定 summary 且又由 `*.json` 命中，产生 source specified more
than once warning，目标 summary 只写入一次，最终 manifest 仍全部通过。以上
错误均未分配 GPU，也没有修改 production 候选。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_cache_split_offline_v1`。

目录内 38 项证据已 38/38 通过 manifest 复算；连同 manifest 共 39 个文件、
91,112 bytes。summary、validation、manifest、run identity 与退出后 GPU 状态的
SHA256 依次为：

- `7387725be90ba817a46a3fa7268bea41b84ac70e9710c80724cddd980bec624e`；
- `442e320079bfd33ed0b69cd408522d98ed4d2d8936b69fde345adddeb967e4d6`；
- `c7100e0b458b8b26b30fda38ef5db351e3a3ab78ab1d2c67170ec2f8dbd0e82f`；
- `209144beda71c61e0ca3bafb4d47a3097956aaad1d5174020a6f0ed46244fc43`；
- `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f`。

退出状态中 8 张苹果800均为 `0 MiB/0%`，没有 compute process。本阶段没有
模型加载、output/LSE CUDA correctness、TTFT、TPOT、吞吐或 GSM8K 精度结果；
2.83 的正式性能对比不变。下一步先发布本节、工具、测试与 planning；恢复
clean/published 后，只继续缩减 history 路径的资源或消除 w4 stack spill。严格
资源门禁通过前，不修改 production kernel，也不启动新的 GPU 性能实验。

### 2.86 History value reload 的 CPU-only SM80 淘汰结果

2.85 的离线工具、测试、报告与 planning 已由主仓库提交
`615a95f73f720d1b0fe7f9ea8527a63c23a31788` 发布，后续发布状态由
`2c34476307758794437c9de9799227ab9644f671` 固化；源码仓库继续固定在已发布的
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 OSCAR production
源码、模型、数据集、正式配置或控制镜像，只扩展 2.85 的离线编译工具和测试。

2.85 中 history h4/t16/w4 已满足 shared/register 的双 block 资源算术，但仍有
176-byte/thread stack spill。本阶段筛选的假设是：history score dot 结束后再重新
load 和 dequantize value，而不让原 value 中间量跨 score/softmax 继续存活，可能
缩短 live range 并消除 spill；代价是重复读取相同 history data/scale/zero。该方案
若不能改变离线 cubin 或资源，则不进入 GPU，更不会修改 production。

工具新增 `reload_history_for_value` compile-time 分支，并为以下 5 个 history
几何各增加一个 reload variant：h8/w8、h4/w8、h4/w4、h2/w4、h1/w4。TDD 首个
有效红灯为 4 passed/1 error，精确因 `Variant` 尚无该属性而触发
`AttributeError`；最小实现后定向测试为 5/5 passed。为把结论固化为结构化结果，
随后增加 pair comparison；对应红灯为 5 passed/1 error，精确缺少
`summarize_reload_comparison`。完成最小实现和机械格式化后，最终 Ruff 0.14.0
check/format、固定 Python compile、组合 unittest `15/15 passed` 与
`git diff --check` 全部通过。最终工具与测试 SHA256 分别为：

- `38d2aacf18fc7ae2a355b566aea08d29428e7aecd52a656690c590a318848ce3`；
- `1e94e08c44130cc2cfd4cf9e50d51b269f3a16e55d1894cb9e1322462b66cc54`。

最终有效离线轮次为：

`/dev/shm/oscar-glm-20260801T031013Z_prefill_history_reload_offline_v3`。

轮次继续使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）、
runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void`。环境仍为 Python 3.12.13、PyTorch
2.11.0+cu129、Triton 3.6.0，离线目标为 SM80；轮次没有初始化 CUDA，
`cuda_initialized=false`。format version 3 的结果为 20/20 variants 编译成功、
0 rejected，内部耗时 `18.302869768813252 s`。

5 组直接配对结果如下；每一行的 base/reload 不仅资源数字相同，cubin SHA256
也逐字节相同：

| 几何 | Base shared/register/stack | Reload shared/register/stack | Cubin/资源比较 |
|---|---:|---:|---|
| history h8/t16/w8 | 84,992 B / 199 / 0 B | 84,992 B / 199 / 0 B | binary/resource identical |
| history h4/t16/w8 | 76,288 B / 206 / 0 B | 76,288 B / 206 / 0 B | binary/resource identical |
| history h4/t16/w4 | 76,288 B / 255 / 176 B | 76,288 B / 255 / 176 B | binary/resource identical |
| history h2/t16/w4 | 71,936 B / 255 / 184 B | 71,936 B / 255 / 184 B | binary/resource identical |
| history h1/t16/w4 | 69,760 B / 255 / 176 B | 69,760 B / 255 / 176 B | binary/resource identical |

结构化汇总中 5 组的 shared、register 和 stack delta 全部为 0，
`all_pairs_binary_and_resource_identical=true`，且
`reload_changed_any_candidate=false`。这表明 Triton 对同一 kernel 内人为重复的
history value load/dequantize 做了公共子表达式消除；源码表面上的 reload 没有形成
编译器可见的阶段边界，也没有缩短最终 cubin 的 live range。h4/w4 的
176-byte/thread spill 原样保留，history strict candidate 继续为空，组合
`cache_split_strict_promotion_feasible=false`。因此该方案正式淘汰，不申请 GPU、
不进入 production，也不把重复访存的理论代价误写成实际 runtime 开销。

v1/v2 是工具 schema 与审计日志补齐过程中的中间轮次；在加入完整结构化 pair
comparison 后，以 v3 作为最终证据。准备过程中，Ruff 两次先后发现 import 顺序和
format 问题，均在启动对应最终离线轮次前 fail-closed；这些边界没有分配 GPU，
也没有修改 production 候选。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_reload_offline_v1`。

目录内 47 项证据已 47/47 通过 manifest 复算；连同 manifest 共 48 个文件、
118,786 bytes。summary、run log、manifest、run identity 与退出后 GPU 状态的
SHA256 依次为：

- `f48652e588310e9f4e3860fd7da403d53308bb216cc238b4186d2b3a88bbbaa6`；
- `7a6a0ec44a84569be3e3299cf1cb024761bdde314f195b4e1fb1f1c911092284`；
- `3f9b7d77cd067c638325683ab22341cf7e8fab40023a87b8fb79400802d7b415`；
- `a7ad4a7fcc7f89f639776ba66009c044fea1b18f803b05402942cf88e45bef92`；
- `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f`。

退出状态中 8 张苹果800均为 `0 MiB/0%`，没有 compute process。本阶段没有
模型加载、output/LSE CUDA correctness、TTFT、TPOT、吞吐或 GSM8K 精度结果；
2.83 的正式性能对比不变。下一步先发布本节、工具、测试与 planning；恢复
clean/published 后，只筛选具有编译器可见阶段边界的 history 路径结构。同一
kernel 内的等价 reload 已由本轮证据排除，不再重复该方向。

### 2.87 History narrow-token tile 的 CPU-only SM80 编译淘汰

2.86 的离线工具、测试与实时记录已由主仓库提交
`77f4a2dadd49cb21d3424e5954df7d58dbe3d155` 发布，发布状态由后续提交
`37de9a41d96e0c41eb39e6d2f01b69f85db59cb1` 固化；源码仓库继续固定在已发布的
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 OSCAR production
源码、模型、数据集、正式配置或控制镜像，只在 2.85/2.86 的 standalone history
离线工具中增加更窄的 token tile。

筛选假设是：把 history h4/h2/h1、w4 的 `block_t` 从 16 缩小到 8，可能减少
同一 program 内反量化中间量的存活范围，从而消除 2.85 记录的
176/184/176-byte thread stack spill。成功门禁要求候选首先完成 SM80 离线编译，
随后同时满足 shared/register 双 block 资源算术和零 stack；任一条件失败就淘汰，
不进入 GPU 或 production。

需要明确的是，2.32 已经在当时的 mixed grouped-prefill kernel 中证明 h8/h4 的
t8 会因 `tl.dot` 的 K 维限制而编译拒绝。本轮不是重复声称发现一个新的 Triton
限制，而是补测 cache-type 拆分后的 standalone history h4/h2/h1 路径，确认拆分
并没有让 t8 在该专用 kernel 中变为合法几何。

TDD 有效红灯为 6 passed/1 failed：测试要求 h4/h2/h1 三个 t8/w4 variant 时，
工具返回空列表。最小增加这三个显式 variant 并把 summary format version 从 3
升至 4 后，Ruff 0.14.0 check/format、固定 Python compile 与定向 unittest
`7/7 passed`。最终工具与测试 SHA256 分别为：

- `56a9f14f41c75fedc5a2f151852aedae08c7fa039811c7c2a8eec113b9d32f9e`；
- `1b50ca98acef7ba59c4b1161cd70402e3fd38ad135ad8e3dd67559e84e413292`。

最终有效离线轮次为：

`/dev/shm/oscar-glm-20260801T032045Z_prefill_history_t8_offline_v1`。

轮次继续使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）、
runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void`。环境仍为 Python 3.12.13、PyTorch
2.11.0+cu129、Triton 3.6.0，离线目标为 SM80；轮次没有初始化 CUDA，
`cuda_initialized=false`。format version 4 共包含 23 个 variant，其中 20 个
既有 t16/t32 variant 编译成功，新增的 3 个 t8 variant 全部被编译拒绝，内部耗时
`18.34389810077846 s`。

三个新增结果如下：

| 几何 | 编译状态 | 失败位置 | Triton 约束 |
|---|---|---|---|
| history h4/t8/w4 | `compile_rejected` | history value `tl.dot` | `K >= 16` |
| history h2/t8/w4 | `compile_rejected` | history value `tl.dot` | `K >= 16` |
| history h1/t8/w4 | `compile_rejected` | history value `tl.dot` | `K >= 16` |

三项错误文本均为
`Input shapes should have M >= 1, N >= 1 and K >= 16`。由于失败发生在 TTIR
构造阶段，三个 t8 候选没有 cubin，也没有 shared/register/stack 资源数字；因此
不能声称窄 tile 降低了资源或改善性能。20 个既有 variant 则继续复现 2.85/2.86
的结论：history 严格候选仍为空，reload 配对仍为 binary/resource identical，
`cache_split_strict_promotion_feasible=false`。

首次封存误把完整 Triton cache/cubin 一并复制，得到 230 个文件、
19,092,392 bytes；虽然清单复算通过，但不符合小型证据原则。该目录没有删除，已
可恢复地移动到：

`/dev/shm/oscar-glm-20260801T032045Z_prefill_history_t8_oversized_package_v1`。

正式小型证据重新封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_t8_offline_v1`。

最终目录共 50 个文件、163,039 bytes；manifest 内 49 项已 49/49 通过复算。
summary、run log、manifest 与退出后 GPU 状态的 SHA256 依次为：

- `33d2088c0d3530d5f7b563bd12048960c7e37ed2dec9f876c7fa598820ca0f6a`；
- `975a25eb378c8651d39ace4b2103bf25fd6cf93abf53d6d0ac5d3255a68a1d39`；
- `6f2a710c4d9f98e97fee5bfad62b19ddc2841b06a26e77b5e9f69a6d18222852`；
- `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f`。

退出状态中 8 张苹果800均为 `0 MiB/0%`，没有 compute process。本阶段没有
模型加载、output/LSE CUDA correctness、TTFT、TPOT、吞吐或 GSM8K 精度结果；
2.83 的正式性能对比不变。结论是淘汰 standalone history 的简单 t8 tile：现有
dot 结构要求 K 至少为 16，不能靠继续缩小 token tile 消除 w4 spill。下一步先
发布本节、工具、测试与 planning；恢复 clean/published 后，只考虑改变 value
计算结构或建立编译器可见阶段边界的候选，不再重复简单 t8 tile 搜索。

### 2.88 History t8 手工 value 归约的 CPU-only SM80 资源门禁

2.87 的工具、测试与实时记录已由主仓库提交
`d96faa69ec11c9bac0fb4cc4f3892af5916f9515` 发布，发布状态由后续提交
`3e6e3081e9af5936fb697cf76e3e770c5294980f` 固化；源码仓库继续固定在已发布的
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 OSCAR production
源码、模型、数据集、正式配置或控制镜像，也没有申请 GPU；改动只位于 standalone
history 的 CPU-only 离线编译工具与对应测试。

2.87 已证明简单 `tl.dot(K=8)` 不合法。本轮筛选不同的 value 计算结构：保持 t8
加载、score 和 softmax 口径不变，把原来的
`probabilities @ history_values.T` 改写为三维 elementwise 乘积，再沿 token 维
执行 `tl.sum(axis=2)`。该写法可能绕过 `tl.dot` 的最小 K 限制，但也可能增加
中间量、改变寄存器分配或产生 spill，因此仍以实际 SM80 cubin 资源为门禁，不能
根据源码形态预设收益。

成功标准保持为：候选必须完成 SM80 离线编译，同时满足 shared/register 双 block
资源算术与零 stack；任一项失败即淘汰。TDD 红灯为 `7 passed/1 error`，精确因
`Variant` 尚无 `manual_history_value_reduce` 属性触发 `AttributeError`。最小实现
只增加 h4/h2/h1、t8、w4 三个显式手工归约 variant、一个 compile-time 分支，并把
summary format version 从 4 升至 5。首次绿灯已通过固定容器 compile 与定向
`8/8` unittest，但 Ruff 0.14.0 随后对新增测试中的 93 字符行报 E501，流程按
fail-closed 停止；机械格式化后，最终 Ruff check/format、固定容器 compile、
cache-split 与 benchmark 合并回归 `17/17 passed`、`git diff --check` 全部通过。

最终有效离线轮次为：

`/dev/shm/oscar-glm-20260801T033149Z_history_manual_value_offline_v1`。

轮次继续使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）、
runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void`。环境为 Python 3.12.13、PyTorch
2.11.0+cu129、Triton 3.6.0，离线目标为 SM80；轮次没有初始化 CUDA，
`cuda_initialized=false`。format version 5 共包含 26 个 variant，23 个编译
成功、3 个拒绝，内部耗时 `21.680984777398407 s`，summary 状态为 `passed`。

三个手工归约结果如下：

| 几何 | shared | registers/thread | stack/thread | cubin bytes | 严格资源门禁 |
|---|---:|---:|---:|---:|---|
| history manual value h4/t8/w4 | 26,112 B | 215 | 0 B | 187,056 | 通过 |
| history manual value h2/t8/w4 | 21,760 B | 190 | 0 B | 146,096 | 通过 |
| history manual value h1/t8/w4 | 19,584 B | 168 | 0 B | 126,640 | 通过 |

三项均为 128 threads/block，registers/block 分别为 27,520、24,320、21,504；
shared、register 与零 stack 三个条件同时满足，`strict_promotion_candidate=true`。
对应 cubin SHA256 分别为：

- h4：`10004c89859b84aec7c5992420633ed71c43e499c89e519420c84a5237275319`；
- h2：`a2b7259579d293794cb2e15cbba18469544f98525a8b269233a1c7353b749331`；
- h1：`16a15b8b614227ca2e390353fb4498fca748355730025b5acdab26c25f3b11c2`。

resource log SHA256 则依次为
`274055b25857d1d3021b58282a4080946964b3c560f65c9123a82e604d1bd1a8`、
`cb9a87a278403d3ddced296e6c7b93773f7eea9f800226fec4f3b7f6c5419ff8`、
`a3258138362062a811fa6a26dbd3df980f4da3083262c5414927160bf5056567`。
三个 2.87 的简单 dot t8 variant 仍全部因 `K >= 16` 编译拒绝；本轮通过的是手工
归约的新结构，不是原编译约束消失。

结构化 summary 中，history strict candidates 首次出现上述三个手工归约 variant；
结合既有 BF16 h8/h4、t16、w4 的严格候选，
`cache_split_strict_promotion_feasible=true`。这里的 true 只表示离线
shared/register/stack 算术门禁第一次形成组合候选，不等于硬件上已经实现双 block
驻留，也不等于候选数值正确或性能更快。

尤其需要保留四项边界。第一，elementwise 加 `tl.sum` 改变了浮点归约顺序，尚未
与冻结 reference 比较 output/LSE。第二，t8 相对 t16 会让 32K 序列上的循环次数
翻倍。第三，h4/h2/h1 相对当前 h8 会把每个 query 的 program 数分别增加到
2/4/8，并可能重复 query、rope 与索引加载。第四，本轮没有模型加载、CUDA kernel
计时、实际 occupancy、TTFT、TPOT、吞吐或 GSM8K 精度测量。因此不能根据资源表
宣称 h1 最优，也不能把本节写成精度或性能已经提升；2.83 的正式 32K/batch1 性能
对比保持不变。

正式小型证据封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_manual_value_offline_v1`。

目录共 60 个文件，按普通文件大小求和为 177,221 bytes；manifest 内 59 项已
59/59 通过复算。目录没有复制 cubin 本体，cubin SHA256 只记录在对应 JSON 中。
summary、run log、manifest 与退出后 GPU 状态的 SHA256 依次为：

- `89c557bc700c6dcc7df99dc0713a0ca880362645e45ad1ec38d6c17ff869041f`；
- `f81ea88fadaedf4ae1d8f75f94628762dde4079a94987bf81fd5340d25d9d42a`；
- `1f43363e3fb91658a55d0394ad7dbc1675628055da520089837ff379612e710b`；
- `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f`。

离线工具与测试 SHA256 分别为
`9b4bb46184037513cb5fbfeb7b16f71144d572cea6b245a387318c95bcbc862c`、
`be5b56437b845bfcf4b4613986810da07199662b64a10a4774130b5409140c80`。
退出状态中 8 张苹果800均为 `0 MiB/0%`，没有 compute process。下一步先发布本节、
离线工具、测试与 planning；恢复 clean/published 后，优先建立三项手工归约候选的
output/LSE correctness 与 CUDA 性能筛选路径。h4 的 program 重复最少，可作为首个
实测对象，但最终选择必须由冻结 reference 和实际 GPU 数据决定，不能只按离线资源
大小排序。

### 2.89 History 手工 value 归约的 32K 末段单卡筛选入口

2.88 的离线工具、测试与实时记录已由主仓库提交 `389bb26` 发布，发布状态由
`f4decaa0946c4d2fa882902de5459c58f0fdd65b` 固化；源码仓库继续固定在已发布的
`67a0e47ff72f10a322de17b81c4134984e017bd6`。2.88 的结果只通过 CPU-only SM80
资源门禁，没有验证手工归约改变浮点顺序后的 output/LSE correctness，也没有证明
实际 CUDA 时间更快。因此本阶段先建立独立筛选入口，不修改 OSCAR production
源码、模型、数据集、正式配置或控制镜像；本节记录的是静态/TDD 准备，不是新的
GPU 实验结果。

新入口为 `scripts/phase9/benchmark_oscar_history_manual_value.py`，只直接调用已发布
离线工具中的 standalone history kernel。冻结 reference 为 h8/t16/w8 dot，候选为
h4/t8/w4 manual；筛选 shape 固定为 batch1、final sequence length 32,768、末段
2,048 个 query token、topk 2,048、8 个本地 attention head、latent rank 512、
rope head size 64、prefix 64、recent 256，seed 为 42。query position 覆盖
`[30720, 32768)`，对应 32K 最后一个 prefill chunk。

输入是为隔离 history 计算路径构造的确定性 synthetic 数据，不是正式模型 trace。
`query_rotated`、query rope、rope cache 与量化 history cache 均由固定 seed 随机生成；
selected token 从最早 query 已可见的 history 区间中无放回抽取同一行 2,048 个索引，
再复制到全部 query，因而所有被选 token 对全部 2,048 个 query 都合法且全部落在
history 路径。这个负载有意排除了 prefix/recent/BF16 混合、真实 DSA 选择分布、完整
stage1 合并、模型层间状态与端到端调度；其结果只能回答两个 standalone history
实现的数值与 kernel 时间，不能替代 2.83 的正式 TTFT/TPOT 对比。

门禁顺序为 fail-closed。脚本先各运行一次 reference/candidate，要求候选 output 与
LSE 均为有限值，并分别以 `atol=0.002`、`rtol=0.002` 对 reference 通过 allclose；
correctness 失败时只原子写出 `correctness_failed` JSON 后退出，不进入计时。通过后
每个 variant 默认预热 2 次，再执行 5 个 repeat、每个 repeat 1 次 iteration；奇偶
repeat 反转 variant 顺序以降低固定先后顺序偏差，使用 CUDA event 统计每次 kernel
时间并取中位数。只有 correctness 通过且 candidate median CUDA time 严格小于
reference，`promotion_eligible` 才为 true。脚本要求恰好 1 张可见 GPU，校验固定
source commit 与 Python/PyTorch/CUDA runtime identity，并在运行超过 10 分钟时打印
一次进度。

TDD 红灯首先在固定 67a 控制镜像、runc、断网、2 CPUs、空 CUDA 可见集下因目标
脚本尚不存在得到 `FileNotFoundError`。最小实现后的首轮合并测试为
`17 passed/1 error`：新测试的动态导入没有把 `scripts/phase9` 加入 `sys.path`，
内部 helper 导入因 `ModuleNotFoundError` 失败；只补测试加载路径后达到 `22/22`
通过。随后 Ruff fail-closed 报告 2 项 I001、3 项 E501 与 1 项 SIM117；机械修复
I001/SIM117并拆分超长行后，最终 Ruff check/format、固定镜像 `py_compile`、
合并 `22/22` unittest（1.452 秒）与 `git diff --check` 全部通过。唯一 warning
仍是控制镜像内既有的 `vllm._version` 缺失，不是本阶段新增错误。

benchmark 与测试文件 SHA256 分别为：

- `7941004cb1b820a1488b8b5748a1e1ca0becba5f0db2fd252224fb2cd8422a32`；
- `414cfe85810050607f18813bb9a45b31fe0fffe1c5ce9c24315f9701c84f2e41`。

截至本节，没有申请 GPU、没有加载模型，也没有产生 correctness、CUDA time、TTFT、
TPOT、吞吐或 GSM8K 精度新结果；2.83 的正式 32K/batch1 性能结论保持不变。下一步
先发布本脚本、测试、报告与 planning，确认主仓库和源码仓库 clean/published；之后
对固定 GPU 做两次间隔至少 60 秒的空闲检查，才可在固定 67a 控制镜像中运行单卡
筛选。若 correctness 失败，或候选 median CUDA time 不严格优于 reference，候选即
淘汰；只有该小型门禁通过，才讨论下一轮 production 或完整 stage1 验证。

### 2.90 History 手工 value 归约的 32K 末段单卡淘汰结果

2.89 的 benchmark、测试、报告与 planning 已由主仓库提交 `bcc0577` 发布，
发布状态由 `2ad6af520f889a98ef5056e50c7891c9b61e0610` 固化；源码仓库继续固定在
已发布的 `67a0e47ff72f10a322de17b81c4134984e017bd6`。正式启动前两仓均为
clean/published。本阶段没有修改 production 源码、模型、数据集、正式配置或
控制镜像，只运行 2.89 已发布的 standalone history 单卡筛选。

GPU 空闲检查时间为 `2026-08-01T04:00:48Z/04:02:00Z`，间隔 72 秒；两次均为
8/8 张苹果800 `0 MiB/0%`，没有 compute process。唯一运行的项目外下载容器
`deepseek_v4_hf_downloader_vllm0230` 的 DeviceRequests 为 null，不占用 GPU，
因此没有终止。有效轮次固定只使用物理 GPU 0，运行目录为：

`/dev/shm/oscar-glm-20260801T040200Z_history_manual_value_cuda_v1`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`，image ID 为
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`；
实际环境为 Python 3.12.13、PyTorch 2.11.0+cu129、CUDA runtime 12.9、Triton
3.6.0、SM80。负载继续使用 2.89 冻结的 synthetic standalone history shape：
batch1、final sequence 32,768、末段 2,048 个 query、top-k 2,048、8 个本地
head、latent rank 512、seed 42；reference 为 h8/t16/w8 dot，候选为
h4/t8/w4 manual。每项预热 2 次，再交替顺序执行 5 个 repeat、每个 repeat 1 次
iteration。

正确性门禁通过。候选与 reference 的 output/LSE 均为有限值，且均通过
`torch.allclose(atol=0.002, rtol=0.002)`；实际误差为：

| 输出 | max_abs | max_rel | 门禁 |
|---|---:|---:|---|
| output | 0.0006555914878845215 | 0.00081673200475052 | passed |
| LSE | 0.0000019073486328125 | 0.00000023754792266572622 | passed |

这说明手工 elementwise 加 `tl.sum` 的归约顺序在该冻结 synthetic 输入上没有超过
既定容差；它不表示与 reference 逐 bit 一致，也不能替代正式模型精度回归。

CUDA event 实测结果如下：

| Variant | 5 个 CUDA 样本范围（ms） | CUDA 中位数（ms） | Wall 中位数（ms） |
|---|---:|---:|---:|
| h8/t16/w8 dot reference | 20.393983841–22.794240952 | 20.436992645 | 20.463432185 |
| h4/t8/w4 manual candidate | 38.565887451–38.593536377 | 38.589439392 | 38.617588580 |

候选相对 reference 的 CUDA 中位数增加 `18.152446747 ms`，即慢
`88.821516267%`；结构化结果中的 `candidate_is_faster=false`、
`promotion_eligible=false`。result 的顶层 `status=passed` 只表示 benchmark
按协议完成且 correctness 通过，不表示候选通过性能晋升。根据 2.89 预先冻结的
“correctness 通过且 candidate median CUDA time 严格更小”标准，h4/t8/w4
manual 候选正式淘汰，不进入 production 或完整 stage1。

该结果也否定了“只要离线 shared/register/stack 更低就会更快”的推断。2.88 中
h4/t8/w4 的 26,112-byte shared、215 registers/thread、零 stack 只证明资源算术
可行；本轮实际净时间反而接近 reference 的 `1.888215×`。这与 t8 让 token 循环
次数翻倍、h4 让每个 query 的 program 数翻倍这一既有风险方向一致，但本轮没有
单独 profile 两项开销，不能进一步把 `18.152447 ms` 精确拆分到循环、重复加载、
归约指令或调度中的某一项。

必须保留的边界是：该轮只测 synthetic all-history selected token、standalone
history kernel、单卡和 5 个单 iteration repeat；没有真实 DSA selected 分布、
prefix/recent/BF16 混合、独立 LSE 合并、完整 stage1、模型加载、TP8 服务或
端到端请求。因此不能把 `20.436993/38.589439 ms` 当作 2.83 的 TTFT/TPOT，
也不能据此生成新的 GSM8K 精度结论。2.83 的正式 32K/batch1/output128/TP8
性能对比保持不变。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_manual_value_cuda_v1`。

目录共 6 个文件、按普通文件大小求和为 7,501 bytes；manifest 内 5 项已 5/5
通过复算。result、run log、manifest、双空闲检查和退出后 GPU 状态的 SHA256
依次为：

- `6a8971c0d94458b1091ae0ad7ea6dbfc4aeec38c675b517bdc8b3a70e6173df2`；
- `b3de8168454ff48b64e6d0295a17df697af5d11574b2d25d8ae842fe45da4dfc`；
- `ddc7a1e104d038408aa8769c1234b2c06d4ce5cf17e73e04eff9c22fffe1200e`；
- `a843b23be770e962cd9c193c9e534daad0e481ce93a550f71a75ffb1270049f8`；
- `d229170501c24d62c862126e1197f6b5081cd0be87d7f1628f478f8025079764`。

有效容器已自动删除；`04:03:17Z` 封存的退出状态显示 8 张 GPU 全部为
`0 MiB/0%`，没有 compute process。h2/h1 手工归约仍只有 2.88 的离线资源数字，
没有本轮 CUDA 实测，不能把 h4 的时间伪装成它们的结果；但也不能根据更小资源数字
自动晋升。下一步先发布本节、证据与 planning；恢复 clean/published 前不运行
h2/h1 或其他候选，后续候选仍须以实际 correctness 与 CUDA/端到端数据决定。

### 2.91 History t16 手工 value 归约的 CPU-only SM80 淘汰结果

2.90 的单卡淘汰结果与 planning 已由主仓库提交 `9aaa906` 发布，发布状态由
`22e166e6f2407a39bf4d9817d3de29abc361ed19` 固化；源码仓库继续固定在已发布的
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 OSCAR production
源码、模型、数据集、正式配置或控制镜像，也没有申请 GPU；只扩展 standalone
history 的 CPU-only SM80 离线矩阵。

2.90 已证明 h4/t8/w4 manual 虽然零 stack，但实际 CUDA 时间比 h8/t16/w8 dot
reference 慢 `88.821516%`。为区分“手工归约结构本身”与“t8 让 token 循环次数
翻倍”的影响，本阶段补齐此前缺失的交叉项：保持 manual elementwise 加 `tl.sum`
不变，只把 token tile 恢复为 16，并分别检查 h4/h2/h1、w4。该阶段仍只做编译资源
门禁；只有 shared/register 双 block 算术和 `stack=0` 同时满足，才可能进入后续
correctness/CUDA 筛选。

TDD 红灯先要求 manual variant 列表增加三项 t16；旧工具得到
`7 passed/1 failed`，实际列表只有三项 t8。最小实现只增加
`history_manual_value_h4/h2/h1_t16_w4` 三个显式 variant，并把 summary format
version 从 5 升至 6；既有 kernel 的 manual 计算分支没有修改。随后任务专属
Ruff 0.14.0 check/format、固定镜像 `py_compile`、cache-split/prefill/history
benchmark 合并 `22/22` unittest（1.647 秒）与 `git diff --check` 全部通过。

最终有效离线轮次为：

`/dev/shm/oscar-glm-20260801T041000Z_history_manual_t16_offline_v1`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）、
runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void`。环境为 Python 3.12.13、PyTorch
2.11.0+cu129、Triton 3.6.0，离线目标为 SM80；轮次没有初始化 CUDA，
`cuda_initialized=false`。format version 6 共包含 29 个 variant，26 个编译
成功、3 个既有简单 t8 dot variant 因 `K >= 16` 拒绝，内部耗时
`29.52599819097668 s`，summary 状态为 `passed`。

三项 t16 manual 与同几何普通 dot 结果如下：

| 几何 | 计算结构 | Shared | Registers/thread | Stack/thread | Cubin bytes | Strict |
|---|---|---:|---:|---:|---:|---|
| h4/t16/w4 | dot | 76,288 B | 255 | 176 B | 176,944 | false |
| h4/t16/w4 | manual | 43,520 B | 255 | 576 B | 362,672 | false |
| h2/t16/w4 | dot | 71,936 B | 255 | 184 B | 172,848 | false |
| h2/t16/w4 | manual | 39,168 B | 255 | 104 B | 257,840 | false |
| h1/t16/w4 | dot | 69,760 B | 255 | 176 B | 168,368 | false |
| h1/t16/w4 | manual | 36,992 B | 255 | 96 B | 206,896 | false |

三项 manual 的 shared 与 register 算术均允许双 block，但 stack 全部非零。h4 的
stack 反而从 176 增至 576 bytes/thread；h2/h1 虽分别降到 104/96 bytes/thread，
仍未满足预先冻结的零 spill 门禁。因此三项
`strict_promotion_candidate=false`，没有任何新的 t16 history strict candidate。
summary 顶层 `cache_split_strict_promotion_feasible=true` 只因为 2.88 已存在的
三项 t8 manual 仍在同一矩阵中，不能误写成 t16 manual 通过。

三个 manual t16 cubin SHA256 依次为：

- h4：`480909d0cb5b1d247d6df59ccbad2bd528d96dec32cbeaa4dbc6dfc9c6e2336d`；
- h2：`5756ef29faad5f4c22cb2e590cc2200a41b7130ffdbe950dd3dc73783fffbdb8`；
- h1：`55358fe10dced7a89d77e2fc2b632b379fc131a0549b561095cde98ef306ea7f`。

对应 resource log SHA256 分别为
`0aa47d98c49bebad11fd79b0c80c865b0028c397fb67797a3e707c011001d1b1`、
`6e487c0df45e1b5af2b7fa98de3bf3d639cd9c71ee8461059dda64d6e40e23a4`、
`8a5a196bfe39f71485c96dddda3098a1cfc143cb35eda35009a68e23172b4923`。
这些数据证明恢复 t16 虽避免了 2.90 的 t8 循环翻倍，但 manual 三维中间量在
当前编译形态下重新产生 spill；不能根据 shared 数字更低而忽略实际 stack。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_manual_value_t16_offline_v1`。

目录共 62 个文件、按普通文件大小求和为 189,284 bytes；manifest 内 61 项已
61/61 通过复算。目录没有复制 cubin 本体，cubin SHA256 记录在对应 JSON 中。
summary、run log、manifest 与退出后 GPU 状态的 SHA256 依次为：

- `ca30e181bba6f4af38ee2efc0958eb4926244f2a782ade90e1008591164a8414`；
- `feecdf47f19d1d0a24ad4eebce1dd3e6655f5a7b61c6d1e05a9cdfbb9b286b58`；
- `96fce5ec8a29597627d4de6adebb236e66b34159aeb0d657ba6a80a67c8faa72`；
- `b68827a85e0ff87b58c6d3a5a5240023076ddc1756668dff0078e4b6c5344134`。

离线工具与测试 SHA256 分别为
`c0baf77605b7e457d26b9b59fe5d010f982a3bd709e7b2d3b60f55a933fedd5d`、
`8c507e560e26d6f48ae578c78e620de3cb31a56aeeddfebdd40314d4952cc72a`。
退出状态中 8 张苹果800均为 `0 MiB/0%`，没有 compute process。本阶段没有
模型加载、output/LSE CUDA correctness、kernel CUDA 时间、TTFT、TPOT、吞吐或
GSM8K 精度结果；2.83 的正式性能对比不变。结论是淘汰 t16 manual 交叉项，不为
三项申请 GPU。下一步先发布本节、工具、测试与 planning；恢复 clean/published
前不继续候选实验，后续结构必须同时避免 t8 的实际回退和 t16 manual 的 stack
spill，不能只在 shared 数字上选择候选。

### 2.92 History score/LSE/value 三段式的 CPU-only SM80 资源筛选

2.91 的离线工具、测试、报告与 planning 已由主仓库提交 `2cda53b` 发布；本阶段
开始前主仓库 HEAD/upstream 均为
`2cda53b4ad6186c38dbd59c7f966024738719eb6`，源码仓库继续固定在已发布的
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 OSCAR production
源码、模型、数据集、正式配置或控制镜像，也没有申请 GPU；改动仅为 standalone
CPU-only SM80 离线编译工具、测试与本记录。

2.85–2.91 已形成当前单 kernel history 路径的资源边界：w8 几何没有 stack
spill，但 h8/h4/h2/h1 仍需 199/206/199/206 registers/thread，256-thread block
不能满足双 block 寄存器算术；w4 虽减少线程数，却产生 192/176/184/176-byte
stack。2.90 又实测零 stack 的 h4/t8/w4 manual 候选比 h8/t16/w8 reference 慢
`88.821516%`，所以本轮不继续排列同构 tile，而建立编译器可见的三段式边界：

1. score kernel 只计算 history 与 rope score，物化 FP32 raw score，并输出每个
   token tile 的 LSE；
2. LSE kernel 独立合并 128 个 tile LSE，形成每个 query/head 的 final LSE；
3. value kernel 读取 raw score 与 final LSE，以 128 维 value tile 反量化并累加
   history value。

该结构会增加两个 kernel launch 和显存读写，不预设会更快。冻结的 32K 末段 shape
为 2,048 query tokens、8 个本地 heads、2,048 top-k；每个 TP rank 的 scratch
精确为：FP32 score `134,217,728 bytes`、tile LSE `8,388,608 bytes`、final LSE
`65,536 bytes`，合计 `142,671,872 bytes`，约 `136.06 MiB`。scratch 可按层复用，
不是 78 层同时各分配一份；但约 128 MiB score 的写回与重读仍是后续实际性能门禁
必须裁决的风险。

新增工具为 `scripts/phase9/compile_oscar_history_score_pipeline.py`，实现 score、
LSE merge 与 value 三个真实 Triton kernel，并使用 AST compile 加 `cuobjdump`
提取 SM80 资源。测试先冻结 5 个 variant、上述 scratch 精确字节、源码中的三段式
边界以及“三段均有 strict candidate”门禁；实现文件不存在时，固定 67a 只读容器
按预期得到 `FileNotFoundError`。最小实现后，任务专属 Ruff 0.14.0 check/format、
固定镜像 8 个目标文件 `py_compile`、cache-split/prefill/history benchmark/本工具
合并 `26/26` unittest（0.707 秒）与 `git diff --check` 全部通过。唯一 warning
仍为控制镜像既有的 `vllm._version` 缺失。

第一次离线轮次目录为
`/dev/shm/oscar-glm-20260801T042900Z_history_score_pipeline_offline_v1`。文件入口的
模块搜索路径无法解析 `from scripts.phase9`，在任何 Triton variant 编译前因
`ModuleNotFoundError` 退出；没有 summary、CUDA 初始化或 GPU 分配。修正为同目录
本地导入后，以新目录完成有效轮次：

`/dev/shm/oscar-glm-20260801T043200Z_history_score_pipeline_offline_v2`。

有效轮次使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）、
runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void`。环境为 Python 3.12.13、PyTorch
2.11.0+cu129、Triton 3.6.0，离线目标为 SM80，`cuobjdump` 来自 CUDA 12.9；
轮次没有初始化 CUDA，`cuda_initialized=false`。format version 1 共 5/5
variants 编译成功、0 rejected，内部耗时 `2.3760272739455104 s`。

实际资源结果如下；strict 要求 shared/register 双 block 算术同时通过且
stack/thread 为 0：

| 阶段/配置 | Shared | Registers/thread | Stack/thread | Cubin bytes | Strict |
|---|---:|---:|---:|---:|---|
| score h8/t16/w8 | 52,224 B | 164 | 0 B | 112,176 | false |
| score h4/t16/w8 | 43,520 B | 162 | 0 B | 110,256 | false |
| LSE tiles128/w4 | 16 B | 17 | 0 B | 11,952 | true |
| value h2/d128/t16/w4 | 8,320 B | 112 | 0 B | 57,056 | true |
| value h1/d128/t16/w4 | 8,256 B | 114 | 0 B | 55,648 | true |

相对现有 history h8/t16/w8 的 84,992-byte shared、199 registers/thread、0-byte
stack，三段式已经显著缩短各阶段 live range。LSE 与两个 value variant 均通过
strict 门禁；但两个 score variant 仍为 256 threads/block，每 block 分别需要
41,984/41,472 registers，两个 block 会超过每 SM 65,536-register 上限。因此
score strict candidate 为空，结构化结果为
`missing_strict_stages=["score"]`、
`pipeline_strict_promotion_feasible=false`。这不是完整三段式资源门禁通过，不能
进入 production 或 GPU correctness/性能实验。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_score_pipeline_offline_v2`。

目录包含 v1 失败日志、v2 summary、5 份 variant JSON/resource、工具、测试、镜像/
仓库身份与退出后 GPU 状态，共 19 个文件、按普通文件大小求和为 70,064 bytes；
manifest 内 18 项已 18/18 通过复算。summary、v2 run log、v1 失败日志、manifest、
工具、测试与退出后 GPU 状态的 SHA256 依次为：

- `72e0478f740df364e1ebc8b81b4431eafe62d5a2a3d3b7c877ee96950101c5ed`；
- `9884cf212bb33ef4a4222be4b4b44b73a3f4e554ab68bb9d3e8ebdb25baec5ee`；
- `93f78661f8a03ce0cdb0c258fb299356d8b02c4961d72a820a8643c3bf2f761e`；
- `21a08a5e6be4643cc330417b6f99f18bf3f15f72271f85b49748b028dacce3ac`；
- `f7fb251c3e5bb7bd433184cf7a5e7807d0ffa3e28758ed84371590ed0c067cc6`；
- `9fe01b49293d75ea8dd6e359c7a9364c8162bb47af2b2305f5976bc97436c15e`；
- `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f`。

退出状态中 8 张苹果800均为 `0 MiB/0%`，没有 compute process。本阶段没有模型
加载、output/LSE CUDA correctness、kernel CUDA 时间、TTFT、TPOT、吞吐或
GSM8K 精度结果；2.83 的正式 32K/batch1/output128/TP8 性能对比保持不变。下一步
先发布本节、工具、测试与 planning；恢复 clean/published 后，只在 score 阶段补测
w4 离线资源，检验 128-thread block 能否在零 stack 条件下通过寄存器门禁。该补测
仍不申请 GPU，且在 score strict candidate 出现前不实现 production 三段式。

### 2.93 History score w4 的 CPU-only SM80 资源门禁结果

2.92 的三段式工具、测试、报告与 planning 已由主仓库提交 `03d1b78` 发布，
发布状态由后续提交 `e80a7895118418ccf82d914d92ca79ea8cc49120` 固化；源码仓库
继续固定在已发布的 `67a0e47ff72f10a322de17b81c4134984e017bd6`。两仓在本阶段
开始前均为 clean/published。本阶段没有修改任何 Triton kernel 语义、OSCAR
production 源码、模型、数据集、正式配置或控制镜像，也没有申请 GPU；只给 2.92
的 standalone 离线矩阵增加两个 score 编译配置。

2.92 的 score h8/h4、t16、w8 分别为 164/162 registers/thread，但每 block 有
256 threads，两个 block 需要 83,968/82,944 registers，超过每 SM 65,536 上限。
本阶段固定的假设是：只把 score 的 `num_warps` 从 8 降到 4，使 block 线程数从
256 变为 128，寄存器双 block 上限等价放宽到 256 registers/thread；成功门禁仍
要求 SM80 编译通过、shared/register 双 block 算术通过且 stack/thread 为 0。
如果出现 spill，仍按失败处理，不能只看线程数减少。

TDD 先要求 format version 从 1 升为 2，并把 `score_h8_t16_w4`、
`score_h4_t16_w4` 加入显式矩阵；旧工具得到 5 tests/2 failures，精确失败于旧
format 和缺少两个 variant。最小实现只修改 `FORMAT_VERSION` 与 `VARIANTS`，没有
改三个 kernel 的计算。实现后任务专属 Ruff 0.14.0 check/format、固定镜像 8 个
目标文件 `py_compile`、cache-split/prefill/history benchmark/本工具合并
`27/27` unittest（0.628 秒）与 `git diff --check` 全部通过。唯一 warning 仍为
控制镜像既有的 `vllm._version` 缺失。

有效 CPU-only 轮次为：

`/dev/shm/oscar-glm-20260801T0444Z_history_score_w4_offline_v3`。

轮次继续使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）、
runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void`。环境为 Python 3.12.13、PyTorch
2.11.0+cu129、Triton 3.6.0，离线目标为 SM80，`cuobjdump` 来自 CUDA 12.9；
轮次没有初始化 CUDA，`cuda_initialized=false`。format version 2 共 7/7
variants 编译成功、0 rejected，内部耗时 `3.924135982990265 s`。

两个新增配置的实际资源为：

| 配置 | Shared | Registers/thread | Registers/block | Stack/thread | Cubin bytes | Strict |
|---|---:|---:|---:|---:|---:|---|
| score h8/t16/w4 | 52,224 B | 255 | 32,640 | 40 B | 148,144 | false |
| score h4/t16/w4 | 43,520 B | 254 | 32,512 | 0 B | 143,152 | true |

h8/w4 的 shared/register 算术允许双 block，但出现 40-byte/thread stack spill，
因此按预设门禁淘汰。h4/w4 的两个 block 共需 65,024 registers，只比每 SM 上限
少 512 registers；shared 为两个 block 共 87,040 bytes，且 stack 为 0，因此
`strict_promotion_candidate=true`。它是 score 阶段第一个严格资源候选，但
寄存器余量很窄，实际 GPU occupancy 仍不能只靠离线算术宣称。

h8/w4 与 h4/w4 的 cubin SHA256 分别为：

- `17f720397fbb075bfa664888edb76d4c69629225387c1a7b2473dba9dc97afc5`；
- `68a4f6c8231217ef1c9f846f8f521e85bf1599027ab1d5fd45ec75e61ca34a6d`。

对应 resource log SHA256 分别为
`00508f7b0faef2c2958258539b02043556a49752c610b8fd8dfe321504b8bc41`、
`2ab6f65bce5f26e368cb87aeb9434f8ba46316beec042cfc35d84b0923feeb7f`。
结合 2.92 已通过的 LSE 与 value 候选，结构化结果首次变为
`missing_strict_stages=[]`、`pipeline_strict_promotion_feasible=true`；score/LSE/
value 的严格候选分别为 h4/t16/w4、tiles128/w4、h2/h1-d128/t16/w4。

该 true 只表示三个 standalone kernel 都有满足离线 shared/register/stack 算术的
配置。三段式会物化约 `136.06 MiB` scratch，增加 score 写回/重读与两个 launch，
并改变 LSE/value 的浮点归约边界；本轮没有检查 output/LSE，也没有 CUDA kernel
计时、实际 occupancy、真实 DSA selected 分布、完整 mixed stage1、模型加载或
端到端请求。因此不能把资源门禁写成数值正确、比当前 kernel 更快或 TTFT 已改善，
也不能据此直接修改 production。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_score_pipeline_w4_offline_v3`。

目录共 23 个文件、按普通文件大小求和为 76,967 bytes；manifest 内 22 项已
22/22 通过复算。summary、run log、manifest、工具、测试与退出后 GPU 状态的
SHA256 依次为：

- `1ae44c6433f7d6d55b718d2f4297334145078c251bb303889fa6fdfc6daaee93`；
- `7d72e5b99650f7f1dc96f041204d1c4a9ecac66733890a553370a34e947c0b1b`；
- `b2d78b50b3f1d54b383c55bb48054d20ef10b0016adeb4572158e0e35c7f778b`；
- `c95a1d75e1a21ca4932a83418a76fa389cd8e1dd1f50b2c65f8a58d3cb4dba35`；
- `8d32f8f88d49fa681b3ff7e3569ed138c6822b9ab7bb1911c85135052b84ab7e`；
- `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f`。

启动命令最初把 `/dev/shm` 目录标签误写为未来的 `051800Z`；summary 实际完成时间
为 `2026-08-01T04:44:08Z`。封存前只把目录重命名为上述 `0444Z`，没有修改
summary、run log 或证据包内容与哈希。退出状态中 8 张苹果800均为 `0 MiB/0%`，
没有 compute process。

本阶段没有新的 GSM8K 精度、TTFT、TPOT 或吞吐结果；2.83 的正式
32K/batch1/output128/TP8 性能对比保持不变。下一步先发布本节、工具、测试与
planning；恢复 clean/published 后，再建立 standalone 三段式相对 2.90 冻结
h8/t16/w8 reference 的 output/LSE correctness 与单卡 CUDA 时间入口。只有数值
门禁通过且三段式总 CUDA 时间严格优于 reference，才讨论 production 或完整
stage1 集成。

### 2.94 History 三段式的 32K 末段单卡筛选入口

2.93 的 score-w4 工具、测试、报告与 planning 已由主仓库提交 `cdd820a` 发布；
源码仓库继续固定在已发布的
`67a0e47ff72f10a322de17b81c4134984e017bd6`。2.93 只证明三段式三个 kernel 都有
通过 CPU-only SM80 资源算术的配置，没有验证分段归约后的 output/LSE，也没有
测量约 136.06 MiB scratch 的写回/重读和三个 launch 的总时间。因此本阶段先新增
独立筛选入口，不修改 OSCAR production 源码、模型、数据集、正式配置或控制镜像；
本节记录的是静态/TDD 准备，不是新的 GPU 实验结果。

新入口为 `scripts/phase9/benchmark_oscar_history_score_pipeline.py`。输入直接复用
2.89/2.90 已冻结的确定性 synthetic all-history builder：batch1、final sequence
length 32,768、末段 2,048 个 query、top-k 2,048、8 个本地 attention head、
latent rank 512、rope head size 64、prefix 64、recent 256、seed 42。selected token
从最早 query 已可见的 history 区间中无放回抽取同一行 2,048 个索引，再复制到
全部 query；因此全部 token 对所有 query 合法且只走 history 路径。

冻结 reference 继续是 2.90 实测过的 standalone history h8/t16/w8 dot。三段式
candidate 固定使用 2.93 的严格 score h4/t16/w4、LSE tiles128/w4，以及 2.92 中
program 数较少的 value h2/d128/t16/w4。候选一次调用顺序执行 score、LSE、value
三个 kernel；CUDA event 包围整个调用，统计的是三个 launch 的合计时间，不是只测
其中最快的单 kernel。candidate scratch 精确为 score `134,217,728 bytes`、tile
LSE `8,388,608 bytes`、final LSE `65,536 bytes`，总计
`142,671,872 bytes`。

筛选按 fail-closed 顺序执行。脚本先各 launch 一次 reference/candidate，要求候选
output 与 LSE 全部有限，并分别通过冻结的
`torch.allclose(atol=0.002, rtol=0.002)`；失败时只原子写出
`correctness_failed` JSON 后退出，不进入计时。correctness 通过后，每个 variant
默认预热 2 次，再执行 5 个 repeat、每个 repeat 1 次 iteration；奇偶 repeat 反转
顺序，使用同一对 CUDA event 覆盖 candidate 的三个 launch 并取中位数。只有
correctness 通过且 candidate 总 CUDA 中位数严格小于 reference，
`promotion_eligible` 才为 true。脚本要求恰好 1 张可见 GPU，校验固定 Python/
PyTorch/CUDA runtime 与 source commit，并保留超过 10 分钟时的进度输出。

必须保留的范围边界与 2.89 相同：输入没有真实 DSA selected 分布、BF16
prefix/recent 混合、global 三段 cache 合并、完整 stage1、模型层间状态或 TP8
调度。即使该入口通过，也只说明 standalone all-history 路径的数值与合计 kernel
时间，不等于 production 已实现，更不能把结果直接替代 2.83 的端到端 TTFT/TPOT。

TDD 红灯首先在固定 67a 控制镜像、runc、断网、2 CPUs、空 CUDA 可见集下因目标
脚本尚不存在得到 `FileNotFoundError`。最小实现后，Ruff check 通过但 format-check
要求机械重排，流程按 fail-closed 停止，固定容器 compile/tests 尚未执行；机械
format 后，任务专属 Ruff 0.14.0 check/format、固定镜像 10 个目标文件
`py_compile`、cache-split/prefill/manual-history/score-pipeline 相关合并
`32/32` unittest（0.896 秒）与 `git diff --check` 全部通过。唯一 warning 仍为
控制镜像既有的 `vllm._version` 缺失。

benchmark 与测试文件 SHA256 分别为：

- `9f8301d671aa42906e2b91aedc9ab869689e80a4763d9dd8bd36e2bea6078713`；
- `f0a1baf30c90383ac20f9c6b6adebf7884fe78c10510ce4e4f95545e91e9117e`。

截至本节，没有申请 GPU、没有加载模型，也没有产生三段式 correctness、CUDA
时间、TTFT、TPOT、吞吐或 GSM8K 精度新结果；2.83 的正式 32K/batch1 性能结论
保持不变。下一步先发布本脚本、测试、报告与 planning，确认主仓库和源码仓库
clean/published；之后对固定 GPU 做两次间隔至少 60 秒的空闲检查，才可在固定
67a 控制镜像中运行单卡筛选。若 output/LSE 不通过，或 candidate 三 kernel 合计
CUDA 中位数不严格优于 reference，三段式即淘汰，不进入 production 或完整
stage1。

### 2.95 History 三段式的 32K 末段单卡淘汰结果

2.94 的 benchmark、测试、报告与 planning 已由主仓库提交 `35708d0` 发布，
发布状态由后续提交 `13e2a57` 固化；源码仓库继续固定在已发布的
`67a0e47ff72f10a322de17b81c4134984e017bd6`。正式启动前主仓库、源码仓库与各自
upstream 一致且工作树 clean。本阶段没有修改 production 源码、模型、数据集、
正式配置或控制镜像，只运行 2.94 已发布的 standalone 三段式单卡筛选。

两次 GPU 空闲检查时间为 `2026-08-01T04:56:13Z/04:57:32Z`，间隔 79 秒；
两次均为 8/8 张苹果800 `0 MiB/0%`，没有 compute process。启动前即时复查仍为
8/8 卡全部空闲。有效轮次固定只映射物理 GPU 0，运行目录为：

`/dev/shm/oscar-glm-20260801T045613Z_history_score_pipeline_cuda_v1`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`，image ID 为
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`；
实际环境为 Python 3.12.13、PyTorch 2.11.0+cu129、CUDA runtime 12.9、Triton
3.6.0、SM80。负载继续使用 2.94 冻结的 synthetic standalone all-history shape：
batch1、final sequence 32,768、末段 2,048 个 query、top-k 2,048、8 个本地
head、latent rank 512、seed 42。reference 为 h8/t16/w8 single-kernel dot，
candidate 为 score h4/t16/w4、LSE tiles128/w4、value h2/d128/t16/w4 三个顺序
launch；每项预热 2 次，再交替顺序执行 5 个 repeat、每个 repeat 1 次 iteration。

正确性门禁通过。候选与 reference 的 output/LSE 均为有限值，且均通过
`torch.allclose(atol=0.002, rtol=0.002)`；实际误差为：

| 输出 | max_abs | max_rel | 门禁 |
|---|---:|---:|---|
| output | 0.00008726119995117188 | 0.000136669506900944 | passed |
| LSE | 0.0000019073486328125 | 0.000000237577324924132 | passed |

这说明 score/LSE/value 分段归约在该冻结 synthetic 输入上没有超过既定容差；它
不表示与 reference 逐 bit 一致，也不能替代真实 mixed stage1 或模型精度回归。

CUDA event 实测结果如下；candidate 时间包含三个 launch 的合计：

| Variant | 5 个 CUDA 样本范围（ms） | CUDA 中位数（ms） | Wall 中位数（ms） |
|---|---:|---:|---:|
| h8/t16/w8 single-kernel reference | 20.403200149–20.452352524 | 20.418560028 | 20.451338030 |
| score/LSE/value 三段式 candidate | 53.301246643–53.323776245 | 53.320705414 | 53.349165246 |

候选相对 reference 的 CUDA 中位数增加 `32.902145386 ms`，即慢
`161.138421811%`，只达到 reference 速度的 `0.382938670×`；结构化结果中的
`candidate_is_faster=false`、`promotion_eligible=false`。顶层 `status=passed`
只表示 benchmark 按协议完成且 correctness 通过，不表示候选通过性能晋升。
根据 2.94 预先冻结的“correctness 通过且 candidate 三 kernel 合计 CUDA 中位数
严格更小”标准，三段式正式淘汰，不进入 production 或完整 stage1。

这一结果再次否定“离线 shared/register/stack 门禁通过就会更快”的推断。2.93
证明三个阶段各自存在 strict candidate，但本轮实际总时间为 reference 的
`2.611384218×`。约 136.06 MiB scratch 写回/重读、两个额外 launch、score 与 value
的分离遍历以及 h4/h2 增加的 program 数都与回退方向一致；本轮没有对子 kernel
分别 profile，因此不能把 `32.902145 ms` 精确分摊给其中任何一项。

必须保留的边界是：该轮只测 synthetic all-history selected token、standalone
kernel、单卡和 5 个 single-iteration repeat；没有真实 DSA selected 分布、BF16
prefix/recent 混合、完整 global LSE 合并、生产 stage1、模型加载、TP8 服务或
端到端请求。因此不能把 `20.418560/53.320705 ms` 当作 2.83 的 TTFT/TPOT，也没有
新的 GSM8K 精度结论；但它已足以按预设 standalone 门禁关闭当前三段式结构。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_score_pipeline_cuda_v1`。

目录共 13 个文件、按普通文件大小求和为 74,950 bytes；manifest 内 12 项已
12/12 通过复算。result、run log、manifest、两次空闲检查与退出后 GPU 状态的
SHA256 依次为：

- `a556d72dac103eb62664b19036bf37623d72b0a5c6662f596a37d334c8ee8543`；
- `6dd5d53175887b3cc4b599ac31f7b514c4cbd09b9412fafad7c728759f305985`；
- `b37d1578a90f2bd0a7379ec1cbf0f3686e54bb8d791bda89de6dfe94fddc3103`；
- `caf5207bd701619442c50ee3da669a2965497914be0bc5ae027f3e1906c9e95f`；
- `c4aa6c61ac45f7bd445003ff198ea1d26759368164d66e25cfa068aa37065d5b`；
- `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f`。

有效容器已自动删除；退出状态显示 8 张 GPU 全部为 `0 MiB/0%`，没有 compute
process。2.83 的正式 32K/batch1/output128/TP8 性能对比保持不变。下一步先发布
本节与 planning；恢复 clean/published 前不运行其他候选。后续优化必须避免当前
三段式的大规模 score 物化与重复遍历，仍需以实际 correctness 和 CUDA/端到端
数据裁决，不能只根据离线资源表选择。

### 2.96 History h4/w8 maxnreg 的 CPU-only SM80 资源结果

2.95 的三段式单卡淘汰结果与 planning 已由主仓库提交 `e050f11` 发布，发布状态由
后续提交 `e211817c1963b2eb8c561f3fff8e3d289f46f7c3` 固化；源码仓库继续固定在
已发布的 `67a0e47ff72f10a322de17b81c4134984e017bd6`。两仓在本阶段开始前均为
clean/published。本阶段没有修改 OSCAR production 源码、模型、数据集、正式配置
或控制镜像，也没有申请 GPU；只扩展 standalone history 的 CPU-only SM80 离线
编译矩阵。

2.85 的 history h4/t16/w8 需要 76,288-byte shared、206 registers/thread、0-byte
stack。shared 已允许每 SM 两个 block，但 256-thread block 的寄存器总量不允许
双驻留。固定 Triton 3.6 的 `CUDAOptions` 源码与签名确认支持 `maxnreg`，会生成
PTX `.maxnreg`，限制每线程 32-bit register 数。本阶段因此保持 h4/t16/w8 的 kernel
语义、token tile 和 8 warps 不变，只补测 `maxnreg=128/120/112/96`，观察以 spill
换取双 block 寄存器算术的实际 cubin 资源；不预设该权衡会更快。

TDD 先要求 summary format version 从 6 升为 7、`Variant` 增加可选 `maxnreg`、
矩阵显式包含四个 cap，并要求 compile options 仅向这些 variant 传递对应值。旧工具
得到 10 tests、1 failure/1 error，精确失败于旧 format 和缺少字段；其余 8 项通过。
最小实现只增加字段、四个 variant 与 compile-options helper，没有改任何 Triton
kernel 计算。实现后任务专属 Ruff 0.14.0 check/format、固定镜像 10 个目标文件
`py_compile`、cache-split/prefill/manual-history/score-pipeline 相关合并
`34/34` unittest（0.769 秒）与 `git diff --check` 全部通过。唯一 warning 仍为
控制镜像既有的 `vllm._version` 缺失。

有效 CPU-only 轮次为：

`/dev/shm/oscar-glm-20260801T050549Z_history_maxnreg_offline_v1`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）、
runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void`。环境为 Python 3.12.13、PyTorch
2.11.0+cu129、Triton 3.6.0，离线目标为 SM80；轮次没有初始化 CUDA，
`cuda_initialized=false`。format version 7 共 33 个 variant，30 个编译成功，
3 个既有简单 t8 dot variant 继续因 `K >= 16` 拒绝，内部耗时
`32.5338608063757 s`。

四个 cap 与两项同几何参考的实际资源为：

| 配置 | Shared | Registers/thread | Registers/block | Stack/thread | Cubin bytes | 双 block 算术 | Strict |
|---|---:|---:|---:|---:|---:|---|---|
| h4/t16/w8，无 cap | 76,288 B | 206 | 52,736 | 0 B | 129,712 | false | false |
| h4/t16/w8，maxnreg128 | 76,288 B | 128 | 32,768 | 192 B | 133,680 | true | false |
| h4/t16/w8，maxnreg120 | 76,288 B | 120 | 30,720 | 232 B | 134,192 | true | false |
| h4/t16/w8，maxnreg112 | 76,288 B | 112 | 28,672 | 256 B | 135,728 | true | false |
| h4/t16/w8，maxnreg96 | 76,288 B | 96 | 24,576 | 360 B | 137,136 | true | false |
| h4/t16/w4，无 cap | 76,288 B | 255 | 32,640 | 176 B | 177,072 | true | false |

结果证明 `.maxnreg` 对四项均实际生效，且 shared/register 算术均允许双 block；
但 cap 越低，stack 从 192 单调增加到 360 bytes/thread。最温和的 maxnreg128 也比
h4/w4 的既有 176-byte stack 更高，四项均为 `stack_free=false`、
`strict_promotion_candidate=false`。无 cap h4/w8 保持零 stack，却仍因 206
registers/thread 只能单 block。也就是说，该方向没有同时保留零 spill 与双 block
资源算术的新候选。

四个 cap 的 cubin SHA256 依次为：

- maxnreg128：`6b1797abf72ed7143c75bd541984abe0f5c5e6700d9e6456cc64aba906688656`；
- maxnreg120：`1f12d24d933de0aa789b68b1f32c6766e68b84d3e87d0880f1dd4c00b3104de5`；
- maxnreg112：`3b3c490b9ef4556c51ea7075ab3d2151bc9fb2b28d6d60ae5f214312486a49ee`；
- maxnreg96：`d6b66178c158575e48c82460e23bae5b578965b4a0ac64c6a718c53848780ecf`。

对应 resource log SHA256 依次为
`8b036299880e708fb66d1c101a523029e4010bb473969d07231f594d634ade72`、
`8c2d99501c09e481f3e5d9adb82575157762efdcf47ba168eefb63e585858ee0`、
`7a9534a7b2935988648b4f5c6d4ac8d29cd8fa868111336abb66bcc42cadd18c`、
`e35aa2f07385716553d9364721d62ca60f98321aae9b8183fcc130d0c6fee69a`。
summary 顶层 `cache_split_strict_promotion_feasible=true` 仍只来自 2.88 已知的
t8 manual history 严格候选与 BF16 w4 候选，不能误写成任一 maxnreg variant
通过 strict 门禁。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_maxnreg_offline_v1`。

目录封存四个 cap、h4/w8 与 h4/w4 两项参考的 JSON/resource，以及 summary、日志、
工具、测试、镜像/仓库身份与退出 GPU 状态，共 21 个文件、按普通文件大小求和为
188,448 bytes；manifest 内 20 项已 20/20 通过复算。summary、run log、manifest、
工具、测试与退出 GPU 状态的 SHA256 依次为：

- `a8e53e121d7789dee4b45e5f566520b68dbfb7c73e5b24dffc6e95159dfa0d56`；
- `f36652abaf713723289551570b364114346a079803b363cc12c2f3d5cad18843`；
- `6174675ef21e05f5794ad3e9dbfdf1497a9961137669e5b79842a7feb4866670`；
- `1364b5389c3f97aeb353d36d2fafb51ef3263638c839b990a1271122ceb26ac0`；
- `7ffde20c4489459e2f82c49ea5b7a1e7173af5d6f4cbfd61395a02f8afe68f21`；
- `d58e14c76372ae3e8a5b4492f7a47ee9b033ff0ad5f5f30f947d76350fa40e9f`。

退出状态中 8 张苹果800均为 `0 MiB/0%`，没有 compute process。本阶段没有
output/LSE CUDA correctness、kernel CUDA 时间、模型加载、TTFT、TPOT、吞吐或
GSM8K 精度新结果；2.83 的正式 32K/batch1/output128/TP8 性能对比保持不变。
下一步先发布本节、工具、测试与 planning；恢复 clean/published 前不运行 GPU。
当前证据不支持仅凭 maxnreg128 的双 block 算术直接晋升，后续优先寻找不依赖大
spill 的 single-kernel live-range 改写，或先用现有 trace/源码证明更具体的候选。

### 2.97 History maxnreg128 的单卡裁决入口

2.96 的 CPU-only 工具、测试、报告与 planning 已由主仓库提交 `c709e10` 通过
HTTPS 推送；本阶段开始时主仓库 HEAD 与 upstream 一致，源码仓库继续固定在
`67a0e47ff72f10a322de17b81c4134984e017bd6` 且 clean。本阶段没有修改 OSCAR
production 源码、模型、数据集、正式配置或控制镜像，也没有申请 GPU；只建立
standalone history 的 maxnreg128 单卡裁决入口。

2.96 的 strict 门禁失败表示 maxnreg128 不能仅凭离线资源直接晋升 production，
但不能回答“由双 block 算术带来的收益是否覆盖 192-byte/thread spill”。为隔离
变量，本轮固定同一份 32K 末段 synthetic all-history 输入，并同时保留三项：

| 配置 | block_h | block_t | warps | maxnreg | 作用 |
|---|---:|---:|---:|---:|---|
| h8/t16/w8 reference | 8 | 16 | 8 | 无 | 冻结的 standalone history 参考 |
| h4/t16/w8 uncapped control | 4 | 16 | 8 | 无 | 隔离 head-group 从 8 改为 4 的影响 |
| h4/t16/w8 maxnreg128 candidate | 4 | 16 | 8 | 128 | 只相对上一项增加寄存器上限 |

三项都保持普通 dot value reduction、相同 token tile 和相同输入。候选只有在
output/LSE correctness 通过，且 CUDA 中位数同时严格小于 h8 reference 与 h4
uncapped control 时，`promotion_eligible` 才为 true；只快于其中一项不能晋升。
这样不会把 h4 增加 program 数与 maxnreg128 的实际净效应混成一个变量。

只读源码复核还关闭了一个看似更直接但并不同构的选项：当前 grouped prefill 只有
在 `group_prefill_heads=true` 且 `num_splits=1` 时才启动；任何
`num_splits>1` 都会退回逐 head 的通用 stage1。历史同形状结果已证明 grouped
split1 显著优于通用 split16。新增 grouped split2 则需要新内核、partial buffer
和 merge，且 split 不会在编译期缩小 accumulator 几何；因此本轮不把直接 split2
冒充保持 head 复用的低风险优化。

实现先扩展既有 standalone history benchmark 的公共 runner，并增加
`launch_options`：variant 未显式设置 `maxnreg` 时不向 Triton 传该参数，只有
candidate 传 `maxnreg=128`。新入口只声明上述三项矩阵、scope、reference、control
和 candidate，继续复用冻结输入、固定 runtime 身份、单可见 GPU 检查、原子 JSON、
2 次 warm-up、5 个交替顺序 repeat、CUDA event/wall 计时及每 10 分钟进度输出。
该重用没有改变旧 manual-value 默认 variant，也没有改变任何 Triton kernel 计算。

TDD 红灯首先在固定 67a 控制镜像、断网、空 CUDA 可见集下因目标 benchmark 文件
尚不存在得到 `FileNotFoundError`，没有初始化 CUDA。最小实现后，任务专属 Ruff
0.14.0 check/format、固定镜像 11 个相关文件 `py_compile`、cache-split、prefill、
manual-history、score-pipeline 与新 maxnreg 入口的合并 `37/37` unittest
（0.095 秒）以及 `git diff --check` 全部通过。唯一 warning 仍为控制镜像既有的
`vllm._version` 缺失。宿主 PATH 与固定镜像 Python 均未直接提供 Ruff 命令，最终
使用任务此前固定的本机 Ruff 0.14.0 可执行文件完成相同 check/format；这不是源码
或测试失败。

被复用的 history benchmark、新 maxnreg benchmark 与新测试文件 SHA256 依次为：

- `4cd236491d4f31d02abd290dc0df4c552cf7586a34010a753a3cb529218e4f78`；
- `0016dbdc381ccefbe4e46d80b1f89ca00c027d443ef73613a77964b513a4330e`；
- `2d920f329a04aa0dcffee801f4a4c423d81071c30c674b3ce75c63f85592a059`。

截至本节，没有 GPU correctness、CUDA 时间、模型加载、TTFT、TPOT、吞吐或
GSM8K 精度新结果；2.83 的正式 32K/batch1/output128/TP8 性能对比保持不变。
下一步先提交并推送本入口、测试、报告与 planning，确认两仓 clean/published；
之后对固定 GPU 做两次间隔至少 60 秒的空闲检查，才可在固定 67a 控制镜像中运行
单卡筛选。无论结果正负，都必须先更新下一节记录再进入后续优化。

### 2.98 History maxnreg128 的 32K 末段单卡淘汰结果

2.97 的 benchmark、测试、报告与 planning 已由主仓库提交 `125d693` 发布，发布
状态由后续提交 `ef7ee2c2da03198b8d3d2804689e8be7f2e459e7` 固化；正式启动前
主仓库、源码仓库与各自 upstream 一致且工作树 clean，源码仓库仍为
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 production
源码、模型、数据集、正式配置或控制镜像，只运行 2.97 已发布的 standalone
history maxnreg128 单卡筛选。

两次 GPU 空闲检查为 `2026-08-01T05:21:38Z/05:22:43Z`，间隔 65 秒；两次均为
8/8 张苹果800 `0 MiB/0%`，没有 compute process。`05:23:12Z` 启动前即时复查
仍为 8/8 卡全部空闲。有效轮次固定只映射物理 GPU 0，运行目录为：

`/dev/shm/oscar-glm-20260801T0522Z_history_maxnreg128_cuda_v1`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`，image ID 为
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`，
runtime 为 runc、network none；实际环境为 Python 3.12.13、PyTorch
2.11.0+cu129、CUDA runtime 12.9、Triton 3.6.0、SM80。输入继续使用 2.97
冻结的 synthetic standalone all-history shape：batch1、final sequence 32,768、
末段 2,048 个 query、top-k 2,048、8 个本地 head、latent rank 512、seed 42。
三项各预热 2 次，再交替顺序执行 5 个 repeat、每个 repeat 1 次 iteration。

候选相对 h8 reference 的 correctness 门禁通过。candidate output 与 LSE 均全部
有限，并通过 `torch.allclose(atol=0.002, rtol=0.002)`；结构化结果中 output/LSE
的 `max_abs` 与 `max_rel` 四项均为 `0`。这只证明该冻结 synthetic 输入上的候选
输出与参考一致，不代替真实 mixed cache、模型级精度或 GSM8K 回归。

CUDA event 实测如下：

| Variant | 5 个 CUDA 样本范围（ms） | CUDA 中位数（ms） | Wall 中位数（ms） |
|---|---:|---:|---:|
| h8/t16/w8 reference | 20.402176–20.453377 | **20.432896** | 20.477039 |
| h4/t16/w8 uncapped control | 40.364033–40.543232 | 40.378368 | 40.405517 |
| h4/t16/w8 maxnreg128 candidate | 30.481407–30.517248 | 30.505983 | 30.532127 |

maxnreg128 相对同几何 h4 无 cap 控制减少 `9.872385 ms`，即 CUDA 中位数降低
`24.449687844%`、加速 `1.323621268×`。因此 2.96 的“双 block 算术可能覆盖一部分
spill 成本”假设确有实测支持；不能把 192-byte/thread stack 直接等同于必然更慢。
但 h4 无 cap 本身因每个 query 的 program 数相对 h8 翻倍而明显回退，maxnreg128
只收回了其中一部分成本。

相对最终冻结的 h8 reference，candidate 仍增加 `10.073088 ms`，即慢
`49.298385602%`，速度只有 reference 的 `0.669799607×`。结构化结果为
`candidate_is_faster=false`、`promotion_eligible=false`。根据 2.97 预先冻结的
“同时严格快于 h8 reference 与 h4 uncapped control”标准，maxnreg128 正式淘汰，
不修改 production，也不运行完整 stage1 或端到端服务。

这组数据把两个变量分开了：把 head group 从 8 降到 4 使中位数从
`20.432896 ms` 回退到 `40.378368 ms`；随后只增加 maxnreg128 又降到
`30.505983 ms`。因此 register cap 的方向性收益是真实的，但不足以抵消 h4 的
program 数与 spill/访存总成本。本轮没有采集逐指令或 L1/L2 counter，不能把剩余
`10.073088 ms` 精确分摊给 spill、调度或其他微架构因素。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_maxnreg128_cuda_v1`。

目录共 12 个文件、按普通文件大小求和为 53,237 bytes；manifest 内 11 项已
11/11 通过复算。result、run log、manifest、两次空闲检查与退出 GPU 状态的
SHA256 依次为：

- `88a144c183b7d90ccfaddce32837846d6488daa95b46ba72d9c6bd7bccf59b5b`；
- `305beaea84167a162f9a7c075a080cafcc0b660c1ed4a8bbed6c4bb567067258`；
- `ceb6f3c9002b9736ad1e47aea389fba1b7734fc94ce1b2984f26df1485e9f224`；
- `fc1a031772d03b4df21e5028a20d43c6fcdf26c28799d95d0596d7052a07eae0`；
- `cef2c1212cea9f5b24b0bc859c092826c1271e69e1abf1976afbffd7d4bcfbf0`。

有效容器已自动删除；`05:23:53Z` 的退出状态显示 8 张 GPU 全部为
`0 MiB/0%`，没有 compute process。本阶段没有模型加载、端到端 TTFT/TPOT/吞吐
或 GSM8K 精度新结果；2.83 的正式 32K/batch1/output128/TP8 性能对比保持不变。
下一步先发布本节、证据与 planning；恢复 clean/published 前不运行下一候选。

### 2.99 History 唯一 packed/group load 的 CPU-only SM80 结果

2.98 的 maxnreg128 单卡淘汰结果与 planning 已由主仓库提交 `ef4b115` 通过
HTTPS 推送；本阶段开始时主仓库 HEAD 与 upstream 一致，源码仓库继续固定为
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 OSCAR
production 源码、模型、数据集、正式配置或控制镜像，也没有申请 GPU；只扩展
standalone history 的 CPU-only SM80 离线编译工具。

固定几何的 latent rank 为 512、`group_size=128`。因此每个 history token 实际
只有 128 个 2-bit packed byte、4 个 scale 和 4 个 zero。旧表达式却以 512 个 dim
构造 `byte_offsets=dims//4` 与 `groups=dims//128`；只读检查现存 h8/t16/w8 PTX
得到 165 条静态 `ld.global`，LLVM 循环体也仍展开了多组 data/scale/zero load。
这不能直接换算动态显存事务，但足以证明编译结果没有把所有重复地址归并成“唯一
元素只 load 一次再广播”。

本阶段新增一个且仅一个 h8/t16/w8 candidate：先按 128×16 加载唯一 packed byte，
展开四个 2-bit 值并 reshape 为 512×16；scale/zero 先按 4×16 加载，再沿每组 128
个 dim 广播并 reshape 为 512×16。reference 与 candidate 的 head group、token
tile、warps、score/value dot、softmax、输出和 program 数完全相同。离线晋升门禁
预先固定为：candidate 必须成功编译、cubin 与 reference 不同、PTX 静态
`ld.global` 数更少且 stack 不增加；否则直接关闭，不申请 GPU。

TDD 首轮在固定 67a 控制镜像、空 CUDA 可见集下得到 12 tests、1 failure/2 errors，
精确失败于 format 仍为 7、Variant 缺少 compact 标志以及比较 helper 不存在；其余
9 项通过。最小实现把 format 升为 8，增加一个 constexpr 标志、唯一 candidate、
PTX load 计数与结构化比较，没有改 production kernel。既有 standalone benchmark
显式向旧 variant 传 `compact_history_loads=false`，保持历史入口兼容。

CPU-only 首次启动目录为：

`/dev/shm/oscar-glm-20260801T0531Z_history_compact_loads_offline_v1`。

该轮在进入 Python 前因 `cuobjdump` 不在固定镜像 PATH 而退出码 1；`run.log` 为
0 bytes、结果目录为空，没有编译任何 variant，也没有初始化 CUDA。镜像内实际工具
随后只读核验为 `/usr/local/cuda-12.9/bin/cuobjdump`。有效 v2 使用该绝对路径：

`/dev/shm/oscar-glm-20260801T0532Z_history_compact_loads_offline_v2`。

v2 使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）、
runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void`。环境为 Python 3.12.13、PyTorch
2.11.0+cu129、Triton 3.6.0，离线目标为 SM80；`cuda_initialized=false`。
format version 8 共 34 个 variant，31 个成功、3 个既有普通 t8 dot 继续因
`K >= 16` 被拒绝，总耗时 `33.3843373442069 s`。

reference 与 compact candidate 的实际结果为：

| 配置 | PTX `ld.global` | Shared | Registers/thread | Stack/thread | Cubin bytes |
|---|---:|---:|---:|---:|---:|
| history h8/t16/w8 reference | 165 | 84,992 B | 199 | 0 B | 135,856 |
| compact-load h8/t16/w8 candidate | **71** | 84,992 B | 230 | 0 B | **106,800** |

candidate 的 PTX 静态 global-load 指令减少 94 条，即 `56.969696970%`；cubin 减少
29,056 bytes，即 `21.387351313%`。shared 与 stack 均无变化，cubin SHA256 不同，
结构化 `offline_promotion_candidate=true`。代价是 registers/thread 增加 31，
即 `15.577889447%`。两项都仍因 84,992-byte shared 超过双 block 的每块上限
83,456 bytes 而只能按单 block 资源形态运行；本候选的目的不是提升 occupancy，而是
减少重复解量化 load 指令。

必须强调：PTX 静态指令数不等于动态 DRAM transaction、L1/L2 命中或实际时间；
重复地址可能被 warp 合并或 cache 吸收，新增 broadcast/reshape 与更高寄存器数也
可能抵消收益。因此本节只证明候选形成了不同二进制、显著减少静态 load 且没有
spill，不宣称 CUDA 加速或 production 收益。

任务专属 Ruff 0.14.0 check/format、固定镜像 12 个相关文件 `py_compile`、
cache-split、prefill、manual-history、maxnreg 与 score-pipeline 合并 `39/39`
unittest（0.092 秒）及 `git diff --check` 全部通过。唯一 warning 仍为固定镜像既有
的 `vllm._version` 缺失。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_compact_loads_offline_v2`。

目录包含 reference/candidate 的 cubin、PTX、JSON/resource、summary、日志、工具、
测试、镜像/仓库身份、v1 失败记录与退出 GPU 状态，共 20 个文件、按普通文件大小
求和为 685,084 bytes；manifest 内 19 项已 19/19 通过复算。summary、run log、
manifest、工具、测试与退出 GPU 状态 SHA256 依次为：

- `0a8ea367caf28707d4acf6377a01746dc31cc511ebdec1dcd0df6cbec5aaa280`；
- `3c5ab61c7eb8157a90be50f9474a3caf83bbc4bea31905a44347641c388aac6b`；
- `2a59fd7839a7731108027c7b143060ee8f43995cbc2c83a52c73a6de7fd13658`；
- `a1add08e2708ff28fc2082e4f94066644aca10feef5bd28144598e2d6b605372`；
- `43657383026d71a7a8d6082d2986d7cab22889e223fb299de4f3d83f41eec3c2`；
- `fa02d70d10544231204e23790c29d9b8f3737604598a7413b06eee6cfd91b3b5`。

退出状态显示 8 张苹果800均为 `0 MiB/0%`，没有 compute process。本阶段没有
output/LSE correctness、CUDA 时间、模型加载、TTFT、TPOT、吞吐或 GSM8K 精度
新结果；2.83 的正式 32K/batch1/output128/TP8 性能对比保持不变。下一步先发布
本节、工具、测试与 planning；恢复 clean/published 后，才可建立同一 h8 reference
与 compact candidate 的 standalone correctness/单卡时间入口。

### 2.100 History compact-load 的单卡裁决入口

2.99 的 CPU-only 工具、测试、报告与 planning 已由主仓库提交
`432ee5f176df319a80ba2a67db666e916bf8d5b0` 通过 HTTPS 推送；本阶段开始时
主仓库 HEAD 与 upstream 一致，源码仓库继续固定为
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 production
源码、模型、数据集、正式配置或控制镜像，也没有申请 GPU；只建立已通过 2.99
离线门禁的 compact-load 单卡裁决入口。

入口固定为同一份 32K 末段 synthetic standalone all-history 输入：batch1、final
sequence 32,768、末段 2,048 个 query、top-k 2,048、8 个本地 head、latent rank
512、seed 42。矩阵只包含两项：

| 配置 | block_h | block_t | warps | compact loads |
|---|---:|---:|---:|---|
| h8/t16/w8 reference | 8 | 16 | 8 | false |
| h8/t16/w8 candidate | 8 | 16 | 8 | true |

两项的 program 数、head grouping、token tile、warps、输入、score/value dot、softmax
和输出缓冲完全一致；唯一开关是 2.99 的 `compact_history_loads`。因此该轮能直接
裁决“减少静态 load 指令但增加 31 registers/thread”的实际净效应，不再混入 h4
program 数翻倍或 maxnreg spill。

筛选复用 2.89 后持续使用的冻结 runner。脚本先各 launch 一次 reference/candidate，
要求 candidate output 与 LSE 全部有限，并分别通过
`torch.allclose(atol=0.002, rtol=0.002)`；correctness 失败时原子写出失败 JSON，
不进入计时。通过后每项默认预热 2 次，再执行 5 个 repeat、每个 repeat 1 次
iteration，奇偶轮反转顺序并同时记录 CUDA event 与 wall time。只有 correctness
通过且 candidate CUDA 中位数严格小于 reference，`promotion_eligible` 才为 true。
脚本继续要求恰好 1 张可见 GPU、固定 runtime/source identity，并保留超过 10 分钟
时的进度输出。

TDD 红灯首先在固定 67a 控制镜像、断网、空 CUDA 可见集下因目标 benchmark 文件
尚不存在得到 `FileNotFoundError`，没有初始化 CUDA。最小实现只新增两项 variant
和对公共 runner 的参数声明；定向 `3/3` 测试通过。随后任务专属 Ruff 0.14.0
check/format、固定镜像 14 个相关文件 `py_compile`、cache-split、prefill、
manual-history、maxnreg、score-pipeline 与 compact-load 合并 `42/42` unittest
（0.092 秒）及 `git diff --check` 全部通过。唯一 warning 仍为固定镜像既有的
`vllm._version` 缺失。

compact-load benchmark、测试、被复用的公共 benchmark 与离线编译工具 SHA256
依次为：

- `afa4efc43b50719567d10e3c69ec879fe892c9a9d116655f37bb3b94b3110eb8`；
- `05897d6d16e1708f3bb4abf770d4de692e32a7f2dbfb80a16e213eef046b6c36`；
- `d56be7a01322294a8aed5ea2b2bfc5009c182f8db8fcfd290438b72da642e9f8`；
- `a1add08e2708ff28fc2082e4f94066644aca10feef5bd28144598e2d6b605372`。

必须保留 2.89 的范围边界：输入没有真实 DSA selected 分布、BF16 prefix/recent
混合、global 三段 cache 合并、完整 stage1、模型层间状态或 TP8 调度。即使候选
通过，也只说明 standalone all-history history kernel 的数值和时间，不等于
production 已实现，更不能直接替代 2.83 的端到端 TTFT/TPOT。

截至本节，没有 GPU correctness、CUDA 时间、模型加载、TTFT、TPOT、吞吐或
GSM8K 精度新结果；2.83 的正式 32K/batch1/output128/TP8 性能对比保持不变。
下一步先提交并推送本入口、测试、报告与 planning，确认两仓 clean/published；
之后对固定 GPU 做两次间隔至少 60 秒的空闲检查，才可在固定 67a 控制镜像中运行
单卡筛选。无论结果正负，都必须先更新下一节记录再进入后续优化。

### 2.101 History compact-load 的 32K 末段单卡通过结果

2.100 的 benchmark、测试、报告与 planning 已由主仓库提交 `ce2c4fa` 发布，发布
状态由后续提交 `182e97089fd0007a963325b8d5c7ab34007f9c3d` 固化；正式启动前
主仓库、源码仓库与各自 upstream 一致且工作树 clean，源码仓库仍为
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段没有修改 production
源码、模型、数据集、正式配置或控制镜像，只运行 2.100 已发布的 standalone
history compact-load 单卡筛选。

两次 GPU 空闲检查为 `2026-08-01T05:40:42Z/05:41:47Z`，间隔 65 秒；两次均为
8/8 张苹果800 `0 MiB/0%`，没有 compute process。`05:42:06Z` 启动前即时复查
仍为 8/8 卡全部空闲。有效轮次固定只映射物理 GPU 0，运行目录为：

`/dev/shm/oscar-glm-20260801T0540Z_history_compact_loads_cuda_v1`。

轮次使用固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`，image ID 为
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`，
runtime 为 runc、network none；实际环境为 Python 3.12.13、PyTorch
2.11.0+cu129、CUDA runtime 12.9、Triton 3.6.0、SM80。输入继续使用 2.100
冻结的 synthetic standalone all-history shape：batch1、final sequence 32,768、
末段 2,048 个 query、top-k 2,048、8 个本地 head、latent rank 512、seed 42。
reference/candidate 各预热 2 次，再交替顺序执行 5 个 repeat、每个 repeat 1 次
iteration。

correctness 门禁通过。candidate output 与 LSE 均全部有限，并通过
`torch.allclose(atol=0.002, rtol=0.002)`；结构化结果中 output/LSE 的
`max_abs` 与 `max_rel` 四项均为 `0`。这说明显式唯一 load、广播与 reshape 在该
冻结输入上没有改变输出，但不代替真实 mixed cache、模型级精度或 GSM8K 回归。

CUDA event 实测结果如下：

| Variant | 5 个 CUDA 样本范围（ms） | CUDA 中位数（ms） | Wall 中位数（ms） |
|---|---:|---:|---:|
| h8/t16/w8 reference | 20.399103–21.846016 | 20.431871 | 20.458233 |
| h8/t16/w8 compact-load candidate | 16.813057–16.832512 | **16.819201** | **16.846409** |

reference 首个样本为 `21.846016 ms`，其余四个样本为
`20.399103–20.448256 ms`；预设统计量是中位数，因此没有删除该高值。candidate
五个样本范围仅 `0.019455 ms`，结果稳定。candidate 相对 reference 的 CUDA
中位数减少 `3.612671 ms`，即降低 `17.681546762%`、加速
`1.214794448×`。结构化结果为 `candidate_is_faster=true`、
`promotion_eligible=true`，通过 2.100 预设的 standalone correctness/性能门禁。

该结果也说明 2.99 的 PTX load 减少不只是静态表面变化：在 program 数、h8 几何、
tile 与 warps 相同的条件下，它覆盖了 registers/thread 从 199 增到 230 的代价并
形成实际 CUDA 收益。不过本轮没有硬件 counter，不能把 `3.612671 ms` 全部精确
归因于 DRAM transaction、cache、指令发射或其中任一项。

必须保留范围边界：该轮只测 synthetic all-history selected token、standalone
history kernel、单卡和 5 个 single-iteration repeat；没有 BF16 prefix/recent
混合、global LSE merge、production mixed stage1、模型加载、78 层集成、TP8 服务
或端到端请求。因此 `16.819201 ms` 不能当作新的 TTFT，也不能直接按 17.68% 外推
2.83 的端到端收益。候选只获得进入 production 集成与更完整 correctness/性能
门禁的资格，尚未落地 production。

小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_compact_loads_cuda_v1`。

目录共 12 个文件、按普通文件大小求和为 51,111 bytes；manifest 内 11 项已
11/11 通过复算。result、run log、manifest、两次空闲检查与退出 GPU 状态的
SHA256 依次为：

- `43ada259bc2e2a8d455cb3991d35b3ada2399e082ee74fa0acd2018ff8a315dc`；
- `5ea86abc278ace63229bb17237fda725acc6699101039e2f04f6a9023fd683e8`；
- `f1887b5f2bf079b87331e77bfe2713c07c37378b7400c4b5bbbcfd294fb44c5d`；
- `828cc4c58e313dabdb2ee4cdb8a192beaf55c382619230dc0fa8072899df6855`；
- `caa2aa8a9a3811a89c63482db1e9bd1a02e8b6fdde0fea09a068ff305c2547fd`。

有效容器已自动删除；`05:42:32Z` 的退出状态显示 8 张 GPU 全部为
`0 MiB/0%`，没有 compute process。本阶段没有模型加载、端到端 TTFT/TPOT/吞吐
或 GSM8K 精度新结果；2.83 的正式 32K/batch1/output128/TP8 性能对比保持不变。
下一步先发布本节、证据与 planning；恢复 clean/published 前不修改 production
或运行下一实验。

### 2.102 History compact-load 的 production 集成与 CPU-only 门禁

2.101 的 standalone 通过结果、证据与 planning 已由主仓库提交
`256a98665a60f5d1ba2d2bb09270ee5117c54358` 发布；本阶段开始时主仓库与源码仓库
均为 clean/published，源码仓库基线为
`67a0e47ff72f10a322de17b81c4134984e017bd6`。本阶段把 2.101 已验证的唯一
packed/group load 表达式最小集成到 production
`_mixed_sparse_prefill_stage1`，没有修改模型、数据集、正式性能配置或控制镜像，
也没有申请 GPU。

production launch 已把 `latent_rank` 与 `block_d` 作为编译期常量传入。最小实现
只在 `latent_rank == block_d` 的满宽几何中加载唯一 packed byte、scale 与 zero，
再广播并 reshape 到完整 latent 维；非满宽几何继续保留原来的 `dim_mask`、
`byte_offsets=dims//4` 与 `groups=dims//group_size` 路径。这样避免 compact reshape
把 padding 维误当成有效值，同时不改变 score/value dot、softmax、program 数或
kernel launch 接口。

静态 TDD 的有效红灯为 `1 failed`，精确失败于 production 源码中尚无
`if latent_rank == block_d` 分支；最小实现后定向测试为 `1/1 passed`。最终固定
Python 的整文件 CPU/interpreter 测试为 `9 passed, 19 skipped, 0 failed`，耗时
`19.21 s`；19 项 skip 均为需要授权 GPU 的 CUDA 测试。Ruff 0.14.0 check/format、
`py_compile`、`git diff --check` 与源码提交的全部适用 pre-commit hooks 均通过。
准备期间第一次 Docker 命令遗漏测试挂载，第二次又挂载到含旧
`typing_extensions` 的错误 pytest 环境；两次均在有效测试开始前失败。改用固定
`/dev/shm/oscar-glm-stage9-pytest-py312` 后才得到上述有效红灯与绿灯。

production 源码已提交为
`c0bcbbbdfb5ab1d2cafd9096bd3d6556a6ec3264`，提交信息为
`perf(oscar): compact grouped prefill history loads`，并已通过 HTTPS 推送到
`origin/feat/glm52-oscar-integration`；源码仓库 HEAD/upstream 一致且 clean。
前两次推送沿用了失效的 VS Code Git IPC socket，均在认证前失败；显式使用当前
有效 socket 后推送成功。这些失败没有改变远端分支或实验结果。

固定控制镜像 `oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`）
中，以 runc、network none、2 CPUs、空 `CUDA_VISIBLE_DEVICES`、
`NVIDIA_VISIBLE_DEVICES=void` 和 SM80 target 编译实际 production kernel。
轮次没有初始化 CUDA，`cuda_initialized=false`；format version 8 共 31 个
variant 编译成功，3 个既有 `block_t=8` dot-shape variant 被拒绝，内部耗时
`33.440725268796086 s`。production 满宽 mixed kernel 与上一版同口径实际结果为：

| 指标 | 上一版 production | compact-load production | 变化 |
|---|---:|---:|---:|
| PTX 静态 `ld.global` | 245 | 167 | -78（-31.836734694%） |
| Cubin bytes | 206,640 | 187,056 | -19,584（-9.477351916%） |
| Shared memory | 109,568 B | 109,568 B | 0 |
| Registers/thread | 255 | 255 | 0 |
| Stack/thread | 0 B | **136 B** | +136 B |

新 production cubin SHA256 为
`e949d9711233254797f2084bdda132ecd36e51b51c6b96934f727157b3fcd04c`，
resource log SHA256 为
`f769a3c50adc7a80adae8a3c3d8a91a1eb939e45bc75b2e7deab2c25fbb90ed9`。
静态 load 与 cubin 大小均下降，证明 compact 表达式已经进入真实 mixed 二进制；
但新增的 136-byte/thread stack spill 是 standalone kernel 中没有出现的新风险。
因此本轮只获得 production GPU correctness/性能裁决资格，不能根据离线结果宣称
性能改善。

非满宽 fallback 也用 `latent_rank=384, block_d=512` 单独编译成功，保持 245 个
PTX 静态 `ld.global`、109,568-byte shared、255 registers/thread 与 0-byte
stack；其 cubin 为 209,584 bytes。该结果证明编译期 `else` 仍保留旧加载路径；
由于 latent 几何不同，不能把该 cubin 大小与满宽结果直接作性能比较。

CPU-only 小型证据已封存到：

`artifacts/phase9-control/20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1/formal_32k_b1_stage1_history_compact_loads_production_offline_v1`。

目录共 13 个文件、按普通文件大小求和为 558,070 bytes；manifest 内 12 项已
12/12 通过复算。summary、manifest、production 源码、测试、run identity 与
validation 的 SHA256 依次为：

- `b7751f6e0e53954dbd679a331054d7f2bffd67f9a29e795ccc7ddf7bff3314a4`；
- `a1abf2fb60f4eef980911bfc9d1bb065ddff4554d039178867653ec2521ee619`；
- `6138a842150460e90b6423bef308be51948bdd1542582d42f9f22945533e062a`；
- `c622f8e9cb96816a0b272c3b424a33054deb3d4128055074ba10444a256b551b`；
- `61942712bb8b4fd0d7fbb38365a4eea0e42f27a0466fed4a1837f8577552279d`；
- `9e492df474d0a73be3952f635867db13d65fcbe46bb0beb62272cfa1909dbb9a`。

初建证据 identity 时曾从短提交哈希错误扩写出不存在的 40 位值；随后用
`git rev-parse HEAD` 发现并在生成 manifest、validation 与本节前修正为上述实际
提交，没有错误值进入正式证据或报告。

本阶段没有 production CUDA output/LSE correctness、模型加载、端到端 TTFT、
TPOT、吞吐或 GSM8K 精度新结果；2.83 的正式
32K/batch1/output128/TP8 对比仍保持不变。下一步先发布本节、源码 gitlink 与
planning；两仓恢复 clean/published 后，才构建绑定 `c0bcbbb` 的候选运行时并按
正式 GPU 空闲检查、correctness 和性能门禁裁决新增 spill 的实际净效应。

### 2.103 History compact-load 的 Phase 6 候选输入迁移

2.102、源码 gitlink 与 planning 已由主仓库提交 `ca33b4f` 发布，随后发布状态由
`dcf480a4c41fc693f388d0634dea8b5fbabb4710` 固化；本阶段开始时主仓库与源码仓库
均为 clean/published。源码仓库 HEAD/upstream 为
`c0bcbbbdfb5ab1d2cafd9096bd3d6556a6ec3264`，源码 tree 为
`061c294d38eaad48e697095a8047955ca228dcb2`。本阶段只迁移 Phase 6 候选构建输入，
没有启动 OCI 构建或申请 GPU。

最小输入变更如下：

- candidate tag 从上一版源码身份切换为
  `glm52-oscar-a800-phase6-c0bcbbbdf-0275043c`；
- `source.commit` 与 Dockerfile 默认 `SOURCE_COMMIT` 切换为
  `c0bcbbbdfb5ab1d2cafd9096bd3d6556a6ec3264`；
- `source.tree` 与 Dockerfile 默认 `SOURCE_TREE` 切换为
  `061c294d38eaad48e697095a8047955ca228dcb2`；
- 更新后的 `docker/Dockerfile.phase6-oscar` SHA256 为
  `17ef020a3f23a94eac3e16b18308fccf3f02dd5a0f453a4136fe81493d2fbb69`，
  `configs/phase6/candidate_inputs.json` SHA256 为
  `086505cbf0c2e6708aee592c6185e542577252ae0a7c10b46d6d1c6a6ba58f78`。

基础镜像 manifest、rotation artifact、runtime expectation、native extension
contract、构建脚本与 PAX 逻辑均未改变。静态门禁使用固定控制镜像
`oscar-glm-stage9-runtime:67a0e47ff`（image ID
`sha256:2d0e9f1ea034eeb24b5557cb71ce2a6d45b178c3ef548b6264df3dc957026f74`），
运行条件为 runc、network none、4 CPUs、空 `CUDA_VISIBLE_DEVICES`、
`NVIDIA_VISIBLE_DEVICES=void`；容器内 Python 为 3.12.13。JSON 解析、
`build_candidate_oci.py`、`verify_candidate_oci.py`、
`test_build_candidate_oci.py` 的 `py_compile` 均通过；PAX 确定性单元测试
1/1 通过，耗时 0.014 s。源码 commit/tree 与配置逐字节一致，上一版
`67a0e47ff`、`60d5e606` 和旧 candidate tag 已从这两个 Phase 6 输入文件清零，
`git diff --check` 通过。

本阶段没有生成新的 OCI layout、manifest/config/layer digest 或验证报告，也没有
Docker daemon、runtime import、GPU、模型加载、production CUDA correctness、
TTFT、TPOT、吞吐或 GSM8K 精度新结果；因此不能把本节视为候选运行时已经构建或
性能已经改善。2.83 的正式 32K/batch1/output128/TP8 对比仍保持不变。下一步先
发布本节、Phase 6 输入与 planning；两仓恢复 clean/published 后，才在两个独立
输出目录构建候选 OCI，并对递归文件集合与内容摘要执行独立验证。

### 2.104 History compact-load 候选 OCI 第一次构建与递归验收

2.103 的 Phase 6 输入和实时记录已由主仓库提交 `fcef35e` 发布，发布状态由
`0284368118d07b9169bc43a03b630e66ee3e4bf4` 固化；有效构建开始前两仓均为
clean/published。第一次直接使用宿主 `python3` 时，脚本在读取输入后的时间戳
准备阶段因宿主 Python 不支持 `datetime.UTC` 立即退出；输出目录当时为空，尚未
生成 OCI 或 build report。随后在 2.103 已冻结的 Python 3.12.13 控制容器中，
以 runc、network none、4 CPUs、宿主 UID/GID、空 `CUDA_VISIBLE_DEVICES` 和
`NVIDIA_VISIBLE_DEVICES=void` 重新执行同一输入；该轮才计为有效构建。

有效 v1 目录为：

`artifacts/phase6/20260801T062220Z_candidate_c0bcbbbdf_compact_loads_v1`。

build 状态为 `built`，递归 verification 状态为 `passed`。候选不可变身份为：

- tag：`glm52-oscar-a800-phase6-c0bcbbbdf-0275043c`；
- image/config：
  `sha256:08d8ea6ffdd1e28bd53b17c963571931561f42daa76b9fd71ea2b3cb26cd360f`；
- manifest：
  `sha256:320e011ef89a5ba60487248a9a40bf2903931a643a6131b11b633f3a40ba9006`；
- candidate layer：
  `sha256:c8f6d0075ddc8f5a405cc835e3a197ba9592c2520cb109c35ed2f5dda25811fe`；
- diff-ID：
  `sha256:c2c7cd6fea116a1756867ec67d7cfbd196a7dcd7fd015c23dcd4abe7c1d737f3`；
- 层数：33；candidate layer 为 109,148,106 bytes、5,298 个 member，
  不含原生扩展或 whiteout；确定性 created 字段为 `2026-08-01T06:03:04Z`。

递归验收确认前 32 个基础层逐层完全匹配；源码 commit/tree 为
`c0bcbbbdfb5ab1d2cafd9096bd3d6556a6ec3264` /
`061c294d38eaad48e697095a8047955ca228dcb2`，4,744 个源码文件与 Git tree
精确匹配。4 份 rotation artifact 与 runtime expectation 的 SHA256 均和冻结
输入一致；7 个原生扩展继续来自基础层、内容哈希匹配且没有被 candidate layer
覆盖。解压层共 4,749 个普通文件，OCI layout 共 41 个普通文件。

build report、verification report 与 `index.json` 的 SHA256 分别为：

- `9a13862709f2875664de943fe9c2855e3e6ac457fe7502ab7b71d5f6775e92a4`；
- `c9c48c9abcdaa67a37f0787ebb93fe98a81e564b7d5d418e5850e44b15a8437b`；
- `a7e7f2e2bbfa7a8b249026955c9cdf62b8a19610aad66b19835ce6e6dc009960`。

本阶段没有 Docker daemon 导入、runtime import、GPU、模型加载、production
CUDA correctness、TTFT、TPOT、吞吐或 GSM8K 精度新结果；2.83 的正式
32K/batch1/output128/TP8 对比仍保持不变。v1 只证明第一份 OCI 可以按冻结输入
构建并通过递归身份验收，尚不能证明构建确定性。下一步先发布本节与 planning；
恢复 clean/published 后，再在第二个独立目录重建，并逐字节比较 index、config、
manifest 与 candidate layer。

### 2.105 History compact-load 候选 OCI 独立重建确定性结果

2.104 的 v1 结果与 planning 已由主仓库提交 `4992fcb` 发布，发布状态由
`5b0ae31f9ae10c23248da4818445b79277ad2293` 固化；v2 启动前两仓再次为
clean/published。第二次构建继续使用与 2.104 相同的固定 Python 3.12.13
控制容器、runc、network none、4 CPUs、宿主 UID/GID 和无 GPU 环境，并写入
独立目录：

`artifacts/phase6/20260801T062729Z_candidate_c0bcbbbdf_compact_loads_v2_rebuild`。

v2 build 状态为 `built`，递归 verification 状态为 `passed`；再次核对
4,744 个源码文件、4 份 rotation artifact、runtime expectation、7 个基础层
原生扩展、33 层身份与精确 Git tree。基础层逐层完全匹配，candidate layer
仍不含原生扩展或 whiteout。v2 得到的 image/config、manifest、candidate layer、
diff-ID、layer size/member 与 2.104 v1 完全相同，分别为：

- image/config：
  `sha256:08d8ea6ffdd1e28bd53b17c963571931561f42daa76b9fd71ea2b3cb26cd360f`；
- manifest：
  `sha256:320e011ef89a5ba60487248a9a40bf2903931a643a6131b11b633f3a40ba9006`；
- candidate layer：
  `sha256:c8f6d0075ddc8f5a405cc835e3a197ba9592c2520cb109c35ed2f5dda25811fe`；
- diff-ID：
  `sha256:c2c7cd6fea116a1756867ec67d7cfbd196a7dcd7fd015c23dcd4abe7c1d737f3`；
- candidate layer：109,148,106 bytes、5,298 个 member。

除比较结构化 digest 外，还直接读取两份 layout descriptor，并用逐字节比较复核
以下四个普通文件：

| 内容 | SHA256 | 逐字节结果 |
|---|---|---|
| `index.json` | `a7e7f2e2bbfa7a8b249026955c9cdf62b8a19610aad66b19835ce6e6dc009960` | 相同 |
| manifest blob | `320e011ef89a5ba60487248a9a40bf2903931a643a6131b11b633f3a40ba9006` | 相同 |
| config blob | `08d8ea6ffdd1e28bd53b17c963571931561f42daa76b9fd71ea2b3cb26cd360f` | 相同 |
| candidate layer blob | `c8f6d0075ddc8f5a405cc835e3a197ba9592c2520cb109c35ed2f5dda25811fe` | 相同 |

v2 build/verification report SHA256 分别为：

- `0aa5ebf962eb32cfbc6c8d2dc4c2e205a1007f485144e002734ffab72a6d06d8`；
- `0b265abbeb57a98401ab529f700edc0d9ed3b74ec05cdb4f2362e2d206590d47`。

它们与 v1 报告哈希不同，是因为报告记录了不同的主仓库发布提交以及各自的
layout/extract 输出路径；这不影响上述四项不可变 OCI 内容逐字节相同。两次独立
构建与递归验收共同关闭了本候选的确定性门禁。

本阶段仍没有 Docker daemon 导入、runtime import、GPU、模型加载、production
CUDA correctness、TTFT、TPOT、吞吐或 GSM8K 精度新结果；2.83 的正式
32K/batch1/output128/TP8 对比保持不变。下一步先发布本节与 planning；恢复
clean/published 后，以已验收的 v1 layout 作为 daemon 导入候选，并独立核对 image
ID、层数、最后 diff-ID 和关键 labels。

### 2.106 History compact-load 候选的 daemon 导入与身份审计

2.105 的双构建确定性结果与 planning 已由主仓库提交 `31b2cc3` 发布，发布状态
由 `e13165e` 固化；导入前两仓为 clean/published，目标 daemon tag 不存在。
本阶段以 2.104 已验收的 v1 OCI layout 为唯一输入，在一次性 Ubuntu 22.04
工具容器中安装并使用 `skopeo 1.4.1`，将
`glm52-oscar-a800-phase6-c0bcbbbdf-0275043c` 导入 Docker daemon。工具日志完整
执行到 `Storing signatures`，退出码为 0，工具容器随后自动删除。

独立 daemon 身份审计状态为 `passed`：

- daemon tag：`glm52-oscar-a800-phase6-c0bcbbbdf-0275043c:latest`；
- image ID：
  `sha256:08d8ea6ffdd1e28bd53b17c963571931561f42daa76b9fd71ea2b3cb26cd360f`，
  精确等于 2.104/2.105 的 image/config digest；
- 层数为 33；最后 diff-ID 为
  `sha256:c2c7cd6fea116a1756867ec67d7cfbd196a7dcd7fd015c23dcd4abe7c1d737f3`；
- source commit/tree、candidate layer、Dockerfile、rotation manifest、
  rotations、runtime expectation 和 base manifest 共 8 项关键 label 全部与
  v1 build report 匹配。

导入日志、退出码、daemon inspect 与身份审计 JSON 的 SHA256 分别为：

- `5a75a93eff561cae9d9f20e510879fd663967d182c961bc667ab082c62ab61b9`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `ce5ac99e8a961fab1033dc08d4606a47a427c1c15a8ce19872c76b756c100331`；
- `f14d8292a708c2bcd7b92c0a79f4d5f26b01175e5c9149787154f08d817cc1d5`。

导入与审计命令没有传入 `--gpus`，也没有注入 NVIDIA runtime。本阶段没有
runtime import、模型加载、production CUDA correctness、TTFT、TPOT、吞吐或
GSM8K 精度新结果；2.83 的正式 32K/batch1/output128/TP8 对比保持不变。
下一步先发布本节与 planning；恢复 clean/published 后，重新执行两次至少间隔
60 秒的 8 卡空闲检查，再做只注入驱动、不运行 kernel 的 runtime import 门禁。

### 2.107 History compact-load 候选的 driver-injected runtime import

2.106 的 daemon 身份结果与 planning 已由主仓库提交 `f444517` 发布，发布状态
由 `1ea7303` 固化；runtime import 前两仓为 clean/published。外层在
`2026-08-01T06:35:26Z` 和 `06:36:39Z` 两次检查 8 张 GPU，间隔 73 秒；
两次均为 `0 MiB/0%` 且没有 compute process，因此无需终止任何进程。

有效探针固定只向候选容器注入 GPU 0 的驱动可见性，不运行 CUDA kernel。探针
一次通过、退出码为 0，实测身份为：

- Python/PyTorch/Triton：`3.12.13/2.11.0+cu129/3.6.0`；
- Transformers/Tokenizers：`5.8.1/0.22.2`；
- FlashInfer Python/JIT cache：`0.6.6/0.6.6+cu129`，只通过
  `importlib.metadata` 读取版本，没有导入 `flashinfer` 或 `flashinfer.jit`；
- vLLM Python：`/opt/vllm_glm52_v1/vllm/__init__.py`；
- vLLM 原生扩展：`/opt/vllm_glm52_v1/vllm/_C.abi3.so`；
- rotation 数量为 78；rotation manifest、rotations 与 runtime expectation
  SHA256 均与镜像身份一致；
- `reasoning_effort=max` 可解析；探针结束时 `cuda_initialized=false`。

导入候选源码包时出现一条既有 RuntimeWarning：没有生成版
`vllm._version`，因此无法读取提交哈希；候选 Python/原生扩展路径、源码
commit/tree 和 daemon/OCI 身份已经由前置独立门禁绑定，该 warning 没有被忽略
为错误，也没有触发 CUDA 初始化。容器自动删除；`06:38:29Z` 退出复查显示
8 张 GPU 再次全部为 `0 MiB/0%`，没有 compute process。

双空闲检查、有效 JSON、运行日志、退出码与退出后 GPU 状态的 SHA256 分别为：

- `8215665b67426e0dc3c33abb4caa930b8fdf5ad44d54c35d2094a962dcbec979`；
- `9bdfc8ca5cfc2a65e69c6db4ee270755e90fe604c5c1ed6f7cfc4ea06d3f3b20`；
- `f2e60043b027c55fa6b5401d9c890dddc5ae71cfdda30ff1344022566c25189a`；
- `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`；
- `d94713b276dce85e5290d67ae5c4fee56137975efc5f7789fd35c182ca3683b4`。

本阶段没有模型加载、production CUDA correctness、TTFT、TPOT、吞吐或 GSM8K
精度新结果；2.83 的正式 32K/batch1/output128/TP8 对比保持不变。下一步先发布
本节与 planning；恢复 clean/published 后，才把 Stage 9 控制镜像的默认 base
最小切换到本候选，并执行 CPU-only 构建与继承身份审计。

### 2.108 History compact-load 的 Stage 9 控制镜像输入切换

2.107 的 runtime import 结果与 planning 已由主仓库提交 `6081d66` 发布，发布
状态由 `ba0b4e9eeaf613fdcfc91602cb822a2316d92006` 固化；本阶段开始时主仓库与
源码仓库均为 clean/published。Stage 9 控制镜像 Dockerfile 只把默认 base 从
`glm52-oscar-a800-phase6-67a0e47ff-0275043c:latest` 切换为
`glm52-oscar-a800-phase6-c0bcbbbdf-0275043c:latest`；其余 apt 源、
`git/iproute2` 安装和 entrypoint 均未修改，实际 Git diff 只有这一行。

新 `docker/Dockerfile.phase9-runtime` SHA256 为
`99932fd21937a449d86f3520230da2679b5b5ce56325ee2d6e0dfa2ad251fd10`。
daemon 中对应 base 已只读核对为：

- image ID：
  `sha256:08d8ea6ffdd1e28bd53b17c963571931561f42daa76b9fd71ea2b3cb26cd360f`；
- 层数：33；
- source commit/tree：
  `c0bcbbbdfb5ab1d2cafd9096bd3d6556a6ec3264` /
  `061c294d38eaad48e697095a8047955ca228dcb2`；
- candidate layer：
  `sha256:c8f6d0075ddc8f5a405cc835e3a197ba9592c2520cb109c35ed2f5dda25811fe`。

静态门禁确认 Dockerfile 只有一个 `ARG BASE_IMAGE`，其值与上述已验收 daemon
tag 完全一致；旧 67a candidate tag 已从该文件清零，`git diff --check` 通过。
目标控制镜像 tag `oscar-glm-stage9-runtime:c0bcbbbdf` 在本阶段尚不存在，因此
没有覆盖历史镜像，也没有把未实际构建的 control image ID 写入本节。

本阶段没有启动 Docker build、注入 NVIDIA runtime、申请 GPU、加载模型、执行
production CUDA correctness，亦没有 TTFT、TPOT、吞吐或 GSM8K 精度新结果；
2.83 的正式 32K/batch1/output128/TP8 对比保持不变。下一步先发布本节、
Dockerfile 与 planning；恢复 clean/published 后，才构建
`oscar-glm-stage9-runtime:c0bcbbbdf` 并审计 34/33 层继承、labels、entrypoint
和固定 CPU runtime。

### 2.109 History compact-load 的 Stage 9 控制镜像构建与审计

2.108 的控制镜像入口、报告与 planning 已由主仓库提交 `8319820` 发布；构建
协议由 `14393c9`/`6be3e3b` 固化，正式有效构建前两仓为 clean/published。
有效 CPU-only 目录为：

`artifacts/phase9-control/20260801T064436Z_runtime_c0bcbbbdf_v1`。

首个构建命令错误地把包含大型实验 artifact 的仓库根作为 Docker build context；
43 秒内尚未进入 Dockerfile，也没有产生 build 输出。该命令被主动中断后，目标
control tag 仍不存在，空日志与失败说明均单独保留。由于 Dockerfile 不执行任何
`COPY`，有效 v2 改用 `mktemp -d` 创建的任务专属空 context，并继续使用 2.108
已经发布的同一 Dockerfile；发送 context 仅 2.095 kB，随后正常执行固定 apt、
git/iproute2 安装与 entrypoint 步骤，退出码为 0。

新控制镜像为：

- tag：`oscar-glm-stage9-runtime:c0bcbbbdf`；
- image ID：
  `sha256:b478512379f67337608137ba2e5c5591be81eae1962a886a9150d8d356435088`；
- 层数：34；基础候选为 33 层，前 33 层逐层完全匹配；
- inherited labels 与 `/bin/bash` entrypoint 均和基础候选一致；
- source commit/tree 仍为
  `c0bcbbbdfb5ab1d2cafd9096bd3d6556a6ec3264` /
  `061c294d38eaad48e697095a8047955ca228dcb2`。

独立 identity audit 状态为 `passed`。第一次 CPU runtime 探针遗漏
`docker run -i`，容器内 Python 从空 stdin 正常退出，形成空 JSON 和退出码 0；
该轮不能记为 runtime 通过。v2 使用新文件名、增加 `-i` 并强制 JSON 非空后，
在 runc、network none、2 CPUs、空 `CUDA_VISIBLE_DEVICES` 与
`NVIDIA_VISIBLE_DEVICES=void` 下通过：Git `2.34.1`、iproute2 `5.15.0`、
Python `3.12.13`、glibc `2.35` 及两项冻结包版本均匹配，且
`cuda_initialized=false`。

有效 build log、daemon inspect、identity audit、runtime check 与证据 manifest
的 SHA256 分别为：

- `0fd69104ba991cce24876b89228e4a3e8322dad0a0941fc45252829d654ea6d5`；
- `3adc363299ebcc6ef4c36b627faf408a0eedbcc411366d1dba2a3484c0cf793e`；
- `fc6640fc8c4aaec217584449a11dea6a10413f2a06b09aed5c10114abcaa2f35`；
- `5ac65b5da3ffc9bcf642bd3beb8a989b177710d38c51a9457b5198ffd18c1f20`；
- `f86b6f0497b4a9955ac71465c584ebd282bcc9ea87a806ed6732ffddf3ff8cea`。

证据目录共 21 个普通文件、42,031 bytes；manifest 内 20 项已 20/20 通过复算，
其中也保留两次启动错误。构建、身份审计和有效 runtime 检查均未传入 `--gpus`
或注入 NVIDIA runtime；前后 compute process 查询为空。

本阶段没有模型加载、production CUDA correctness、TTFT、TPOT、吞吐或 GSM8K
精度新结果；2.83 的正式 32K/batch1/output128/TP8 对比保持不变。下一步先发布
本节与 planning；恢复 clean/published 后，从 2.104 的已验收 candidate layer
机械派生新的正式 overlay，再迁移 Phase 1/5/7/9 配置并执行 CPU-only 静态门禁。
