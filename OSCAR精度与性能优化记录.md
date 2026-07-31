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
