# OSCAR-vLLM 三 Pool 优化实验报告

## 1. 目标与边界

本任务将当前单请求 OSCAR-vLLM 原型重构为 first-class 三 pool KV backend。
第一阶段支持 Qwen3 decoder-only full attention、TP=1、PP=1、DCP=1、PCP=1
和 eager execution；CUDA graph、speculative decoding、KV connector、offload、
disaggregation、MLA、sliding-window、attention sink 与混合 attention 架构不在本轮范围。

目标物理布局为独立的 INT2 history、共享 BF16 prefix 和请求独占 BF16 recent
三个 pool。主容量指标继续使用同框架 `OSCAR tokens / BF16 tokens`，总 KV tensor
预算同时包含 INT2 packed K/V、FP32 scale/zero、BF16 prefix/recent K/V、page
padding 和不可用 slots。

## 2. 已确认的硬性验收

| 项目 | 标准 |
| --- | --- |
| 单请求物理压缩率 | 10 GiB 下不低于 `6.2x` |
| allocator 效率 | `max_num_seqs=1/8/48` 距理论最优不超过 2% |
| recent 语义 | chunked/非 chunked 最终精确保留相同的最后 256 tokens |
| prefix cache | 共享 BF16 prefix 与 INT2 history，排除末尾 recent |
| 生命周期 | finish、abort、preemption、eviction、reuse 全部容量守恒 |
| GSM8K | 不低于 `1134/1319`，相对当前 `1157/1319` 最多减少 2 条 |
| 性能 | 完成计划中的 M2 门槛后才宣告任务完成 |

## 3. 阶段 0：冻结基线

冻结时间为 UTC `2026-07-20 09:26:53`。当前实际状态如下：

| 项目 | 实际值 |
| --- | --- |
| vLLM 分支 | `oscar-vllm-v0.25.0` |
| vLLM HEAD | `702f4814fe54fabff350d43cb753ae3e47c0c276` |
| staged 变更 | 20 个文件，新增 2579 行，删除 0 行 |
| unstaged 变更 | 无 |
| staged tree | `65e4204535366a8765f5da19bc238727aae95b80` |
| staged diff SHA256 | `6b1f885bf5dd9e3419c50721c460e62a9c4ed6b8475c0bad0b2ad68fc2ac6e31` |
| 冻结 Docker 镜像 | `oscar-vllm:v0.25.0-dev` |
| 镜像 ID | `sha256:0356873d691c97f72ccc08eda4ca7aecb9b297206763fab8ea3ad01b3b4f1e7e` |

该镜像和已有 `oscar_vllm_report.md`、`oscar_kv_compression_comparison.md` 作为优化前
baseline，不覆盖同名镜像。后续里程碑镜像使用独立 tag。

本阶段没有新增 GPU 实验结果。此前 baseline 的 GPU、Python、包版本、
`CUDA_VISIBLE_DEVICES` 和完整 `nvidia-smi` 已记录在上述两份既有报告及其 artifacts。

## 4. 阶段状态

| 阶段 | 状态 |
| --- | --- |
| 冻结源码、镜像和已有结果 | 已完成 |
| 三 pool spec、容量模型和 CPU reference allocator | 已完成 |
| 单请求新布局 store/decode | 进行中（GSM8K 绝对门槛通过，相对门槛差 1 条） |
| 多请求与完整生命周期 | 已完成（batch 1/8/48、finish/reuse、abort、preemption、eviction 全部通过） |
| 滚动 recent 与 fused mixed prefill | 已完成 |
| Prefix cache | 已完成（shared-hit ownership 修复、CUDA/core 与新镜像真实服务通过） |
| 容量与性能优化 | 进行中（容量、allocator、单请求与 batch M2 通过；prefix TTFT 首轮失败，workspace 待测） |
| 长上下文、GSM8K 与三方终验 | 待开始 |

## 5. 阶段 1：Spec、容量模型与 CPU Reference Allocator

本阶段新增 first-class `OscarKVCacheSpec`，分别公开 INT2 history page 和 BF16
prefix/recent page 的字节大小。阶段 1 结束时平台层尚未切换到该 spec；切换已在
阶段 2 的三段式配置与 worker tensor 契约同时就绪后完成。

容量模型使用 Qwen3-8B 的实际配置：36 层、8 个 KV heads、K/V head size 均为
128、INT2 group size 128、block size 16、每请求 BF16 prefix 64 tokens、BF16
recent 256 tokens。模型计算出的全层单 token 大小为：

| Pool | 单 token 字节数 | 单 page 字节数 |
| --- | ---: | ---: |
| INT2 history（含 K/V data、FP32 scale/zero） | 23040 | 368640 |
| BF16 prefix/recent K/V | 147456 | 2359296 |

在总 KV tensor 预算严格为 10 GiB、无额外 prefix cache 预留下，实测规划结果如下：

| `max_num_seqs` | BF16 prefix slots | BF16 recent slots | INT2 物理 slots | 最坏保证 slots | 物理压缩率 | 最坏碎片占比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 256 | 463984 | 463969 | 6.372006152x | <2% |
| 8 | 512 | 2048 | 449648 | 449528 | 6.175126346x | <2% |
| 48 | 3072 | 12288 | 367728 | 367008 | 5.050098879x | <2% |

这里的“最坏保证 slots”扣除了每个活动请求最多 15 个未填满的 INT2 page slots。
`max_num_seqs=8/48` 的总压缩率降低来自为每个并发请求固定预留 320 个 BF16
tokens，并非 allocator 浪费；同等 HP 预留条件下的 allocator 碎片仍低于 2%。

CPU reference allocator 覆盖请求开始、history 追加、partial page 转完整 page、
容量失败回滚、请求结束回收和三段式守恒。Docker 镜像
`oscar-vllm:v0.25.0-dev` 中的最终验证为 `11 passed`，Ruff lint 与 format check
均通过；该阶段未运行 GPU 实验。

## 6. 阶段 2：单请求三张量 Store/Decode

### 6.1 Engine-Core 与 Worker 布局

平台的 OSCAR attention layer 已改为生成 `OscarKVCacheSpec`，不再用
`TQFullAttentionSpec.page_size_padded` 把 BF16 arena 均摊进每个 INT2 page。
engine-core 按本地全部 attention layers 一次性规划总预算，`num_blocks` 仅表示
INT2 history pages；每层实际分配 bytes 同时包含固定 BF16 prefix/recent pools。

worker 将每层同一份已计费 backing storage 无遗漏地解释为三个 tensor：

- INT2 history：`[num_blocks, 16, num_kv_heads, 80]`，`uint8`；
- BF16 prefix：`[prefix_slots, num_kv_heads, 2, 128]`，`bfloat16`；
- BF16 recent：`[recent_slots, num_kv_heads, 2, 128]`，`bfloat16`。

CPU 合约测试验证了 10 GiB 配置的总 bytes、INT2 block 数、三 tensor 边界和共享
backing storage。定向配置测试为 `13 passed`；合并已有 OSCAR CPU 测试后结果为
`20 passed, 1 skipped, 12 subtests passed`，Ruff lint 通过。

### 6.2 CUDA Kernel Oracle

GPU 0 经两次检查连续空闲后用于本实验，实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:v0.25.0-dev` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU | NVIDIA B200，183359 MiB |
| Driver | 595.71.05 |
| `nvidia-smi` CUDA | 13.2 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA | 13.0 |
| Triton | 3.6.0 |
| vLLM | 0.25.0 |

实验前 `nvidia-smi` 显示 GPU 0 为 `0 MiB / 183359 MiB`、利用率 0%、无运行进程。
Triton store 已直接写独立 BF16 prefix/recent tensor；demote 从 recent ring 读取
BF16 并写入 INT2 page；mixed decode 在一次 online softmax 中读取 BF16 与 INT2。

CUDA/Triton 定向测试实际结果为 `6 passed in 46.43s`，覆盖：

- INT2 store/dequant，head dim 64 与 128；
- INT2 decode 与 PyTorch dequant reference，含 GQA 与 MHA；
- 独立 BF16 prefix/recent store、recent demote 和三段式 mixed decode reference。

本结果证明三张量 kernel ABI 与定向数值 oracle 一致。

### 6.3 Qwen3 单请求端到端 Smoke

使用 `/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507`、10 GiB KV tensor 预算、
`max_model_len=8192`、`max_num_seqs=1`、eager 模式启动真实 OpenAI-compatible
服务。chunked prefill 与 prefix caching 在本阶段继续关闭。

首次运行的三条 1373-token 请求均为 HTTP 200，答案为 `5`、`Paris`、`10`。
但容量日志显示 `371187` tokens；审计发现 `OscarKVCacheSpec.max_memory_usage_bytes`
把已经由全局 planner 固定计费的 BF16 reserve 又计入每请求 scheduler block demand。
实际 tensor 分配未错，但该语义会低估并发，因此修正为每请求只计算 INT2 blocks。

修正后的 CPU 定向测试为 `13 passed`、Ruff 通过。重新启动服务后，实际三段式日志为：

```text
OSCAR KV pools: INT2 history=463984 tokens (9.96),
BF16 prefix=64 tokens (0.01), BF16 recent=256 tokens (0.04), unused=0.0
GPU KV cache size: 463,984 tokens
Maximum concurrency for 8,192 tokens per request: 56.64x
```

随后重发相同三条请求，结果如下：

| 请求 | Prompt tokens | Completion tokens | HTTP | 输出 |
| --- | ---: | ---: | ---: | --- |
| 1 | 1373 | 2 | 200 | `5` |
| 2 | 1373 | 2 | 200 | `Paris` |
| 3 | 1373 | 3 | 200 | `10` |

所有请求都超过 320-token BF16 prefix/recent 总窗口，服务日志实际出现 OSCAR
Triton KV write 和 mixed attention read；未出现 traceback、runtime error 或 CUDA
error。服务最终 `exit=0`、`OOMKilled=false`，停止后 GPU 0 为 0 MiB。

原始结果位于 `artifacts/oscar_vllm_optimization/20260720/single_request_smoke/`。
阶段 2 的 kernel oracle 与端到端 smoke 已通过；完整 GSM8K 已完成，但相对冻结
精度门槛差 1 条，详见第 6.5 节，因此本阶段仍未完成。

### 6.4 GSM8K 无效运行记录

为执行完整精度回归，构建了不可变镜像
`oscar-vllm:three-pool-single-20260720`，镜像 ID 为
`sha256:91f23026e633c1c8919bed2d62a376416f465807f4ac1692ac72371bb73876c3`。
GPU 0-7 在 UTC `10:10:15` 和 `10:11:52` 两次检查中均为 0 MiB、0% 利用率、
无 compute process。随后每卡启动一个单请求服务，并运行冻结的 8 个 GSM8K shards。

该轮最终只有 3 条 `scored`，其余 1316 条均为 `request_failed`，因此是无效实验，
不得用于精度结论。实际根因是首个需要 recent demote 的长生成触发 Triton 冷编译时，
kernel 对 Python `int` 类型的 `recent_idx` 调用了 `.to(tl.int64)`，抛出：

```text
AttributeError("'int' object has no attribute 'to'")
```

8 个 engine 均出现相同错误并停止，Docker 状态均非 OOM。无效 shard、failed cases、
server logs 和 runner logs 保存在
`artifacts/oscar_vllm_optimization/20260720/gsm8k_single/`。修正 scalar cast 并重新
通过 CUDA cold-compile 测试前，不进行精度合并。

修正后使用新的 `TRITON_CACHE_DIR` 强制 cold compile，mixed
store/demote/decode oracle 实际结果为 `1 passed in 26.07s`。随后构建新镜像
`oscar-vllm:three-pool-single-fix1-20260720`，镜像 ID 为
`sha256:3b38882546c67bf0555092c89251a02a47ec06cad16920bf40372fdfec2be7ce`；
未覆盖产生无效实验的旧镜像。

精度重跑分配 GPU 前的第一次检查显示 0-7 空闲，但第二次检查时外部作业 PID
3284803-3284810 已占用全部 8 卡，每卡约 95.4 GiB。依据实验规范，本任务未启动
重跑容器，改为每分钟检查；只有所需 GPU 连续两次空闲后才继续。

### 6.5 GSM8K 有效重跑

外部作业结束后，GPU 0-7 在 UTC `10:32:49` 与 `10:35:13` 连续两次确认空闲。
本实验固定使用物理 GPU 0-7，每卡运行一个 `max_num_seqs=1` 服务；容器内均设置
`CUDA_VISIBLE_DEVICES=0`，并分别运行一个冻结 GSM8K shard，runner concurrency 为 1。
实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:three-pool-single-fix1-20260720` |
| 镜像 ID | `sha256:3b38882546c67bf0555092c89251a02a47ec06cad16920bf40372fdfec2be7ce` |
| GPU | 8 x NVIDIA B200，单卡 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / Triton | 0.25.0 / 2.11.0+cu130 / 3.6.0 |
| Transformers | 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | 每个服务容器内均为 `0` |

启动前完整 `nvidia-smi` 显示八张卡均为 `0 MiB`、利用率 0%、无运行进程，原始
输出保存在 `gsm8k_single_fix1/nvidia_smi_before.txt`。八个 runner 均退出码 0、
`OOMKilled=false`；七个 shard 为 165/165 scored，一个为 164/164 scored，合计
1319/1319 scored，无 request failure、unsupported 或 extraction failure。八个服务
正常退出码 0、`OOMKilled=false`，停止后八张卡均为 0 MiB。

按 official manifest 原顺序合并后的实际结果为：

| 配置 | Correct / Total | Accuracy | 相对冻结 OSCAR-vLLM |
| --- | ---: | ---: | ---: |
| 冻结 OSCAR-vLLM | 1157 / 1319 | 0.8771796816 | - |
| 三 pool 单请求 fix1 | 1154 / 1319 | 0.8749052312 | -3 条（-0.227445 个百分点） |

该结果比绝对门槛 1134 多 20 条，但相对冻结结果少 3 条；硬性标准要求最多减少
2 条，因此相对精度门槛差 1 条，阶段 2 暂不能标记完成。评测结果本身有效，失败的
是验收门槛，不是请求执行。

合并墙钟时长为 `1864.9968598` 秒。`predictions.jsonl` 共 1319 行、1319 个唯一
ID、无空行，与 official GSM8K ID 及顺序完全一致；SHA256 为
`c4413b137683013d9ff2697bf75da6910a0626298464eed60756baa0f2b7e110`。合并结果、
逐 shard 原始结果、完整服务/runner 日志、容器 inspect、Python/包版本和前后
`nvidia-smi` 均保存在
`artifacts/oscar_vllm_optimization/20260720/gsm8k_single_fix1/`。

## 7. 阶段 3：多请求 Ownership 与生命周期

GPU 被外部作业占用期间，先推进不依赖 GPU 的 engine-core 合约。审计确认 worker
的 persistent batch 会在请求移除后压缩并交换行号，因此 batch row 不能充当
prefix/recent pool 的持久地址。当前实现新增请求级稳定 HP row：同一个 row 同时
确定该请求在 BF16 prefix 和 BF16 recent pool 中的连续范围，并与 quant blocks
一起在 engine-core manager 中分配和释放。

第一组 Docker CPU 定向测试实际结果为 `15 passed in 3.98s`，新增覆盖：

- 两个并发请求获得不同且连续的 prefix/recent ranges；
- HP rows 耗尽时第三个请求明确失败；
- 请求结束后 row 及对应 ranges 可复用；
- 专用 OSCAR manager 的 quant block 与 HP row 生命周期一致。

该轮测试后的前两次 Ruff 调用分别因临时 venv 未暴露 executable、`uv` 自动尝试
editable-build 无 `.git` 的 bind mount 而未执行，均不是 lint 结论。改用
`uv run --no-project --with ruff` 后，修正一个测试 import 排序问题，最终结果为
`15 passed in 3.22s`、Ruff lint 全通过、4 个文件 format check 全通过。

UTC `10:28:51` 检查时 GPU 0-7 仍由外部作业占用，每卡约 166 GiB；UTC
`10:32:49` 第一次检查到 0-7 全部为空闲。依据连续两次检查规则，此时尚未分配 GPU。

第二个 CPU 子步把 ownership 接入正式调度链路：专用 manager 提供本步所有请求的
HP row，scheduler 以 `request_id → hp_row` 写入 `SchedulerOutput`；model runner
再按当前 batch 中的请求 ID 重排到预分配 GPU buffer，并经
`CommonAttentionMetadata` 传入 OSCAR backend。worker 不分配 row，只消费
engine-core 的显式 ownership。该子步实际结果为 `16 passed in 4.21s`、Ruff lint
全通过、10 个文件 format check 全通过。

UTC `10:35:13` 第二次检查确认 GPU 0-7 继续全部空闲，满足分配规则。固定使用
GPU 0-7，于 `10:35:52` 启动 8 个修复镜像服务，`10:37:47` 全部 health ready；
随后用冻结的 8 个互斥 manifest 启动完整 GSM8K 重跑，结果写入独立目录
`artifacts/oscar_vllm_optimization/20260720/gsm8k_single_fix1/`。该实验已完成，
最终有效结果与结论见第 6.5 节。

第三个 CPU 子步完成 batched kernel ABI：INT2/BF16 store 在 kernel 内通过
token→request、query starts 和 seq lens 计算每个 token 的逻辑位置，decode/demote
直接加载 scheduler-owned HP row；batch 8/48 不需要按请求串行 launch。新增
batch=2、请求与 HP row 顺序相反的 CUDA full-dequant oracle，用于捕获错误依赖
batch row 的寻址。其前置静态验证实际为 Python 编译通过、CPU
`17 passed in 8.49s`、Ruff lint 与 format check 全通过。

真实单请求 smoke 日志显示 Qwen3 默认采用 V2 Model Runner，因此又补齐了实际主
路径：V2 根据本步排序后的请求 ID 从同一 `SchedulerOutput` 字典取 row，写入
`InputBatch` 的预分配 GPU buffer，再由 default model state 传给通用 attention
metadata。该路径的 Python 编译、CPU `17 passed in 9.63s` 以及 4 个相关文件的
Ruff lint/format check 均通过；CUDA 端到端结果见第 7.3 节。

### 7.1 Batched CUDA Cold-Compile Oracle

GSM8K 容器回收后，GPU 0 在两次相隔超过一分钟的检查中均为 0 MiB、利用率 0%。
本实验固定使用物理 GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`，并设置全新的
`TRITON_CACHE_DIR=/tmp/oscar-batched-triton-cache`，确保 batched kernels 经过
冷编译。实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:three-pool-single-fix1-20260720` |
| GPU / Driver / `nvidia-smi` CUDA | NVIDIA B200 / 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / pytest | 3.6.0 / 5.13.0 / 9.1.1 |
| `CUDA_VISIBLE_DEVICES` | `0` |

测试将当前工作区 Python 源覆盖到镜像内
`/usr/local/lib/python3.12/dist-packages/vllm`，保留镜像中的已编译扩展。实际结果为
`7 passed in 37.72s`，容器退出码 0、`OOMKilled=false`；测试结束后 GPU 0 为
0 MiB。除既有 6 项 oracle 外，新增用例同时执行两个 385-token 请求，将 scheduler
HP rows 设为 `[1, 0]`，并验证 batched INT2 store、BF16 store、recent demote 和
mixed decode 与 full-dequant reference 一致。

完整环境、pytest 日志、容器 inspect 与实验前后 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/cuda_batched_oracle/`。

### 7.2 多请求启动限制与 Core 回归

batched CUDA oracle 通过后，删除了 `arg_utils.py` 中强制
`max_num_seqs == 1` 的 OSCAR 拒绝逻辑。eager、关闭 chunked prefill、关闭 prefix
caching、关闭 speculative decoding 等 fail-fast 均保持不变。该文件的
`py_compile`、Ruff lint 和 format check 全部通过。

扩展 core 回归最初在无 GPU 容器中得到两次相同的 `72 passed, 14 failed`；14 项
均在 `DeviceConfig` 构造阶段因 CUDA build 无法检测 platform 失败，未进入测试断言。
随后透传 GPU 的首次重跑又因 bind-mounted 源码遮蔽镜像内
`_C_stable_libtorch` 而在收集前退出。最终使用与 CUDA oracle 相同的
installed-package overlay 方式，保留编译扩展后重跑。

有效回归固定使用物理 GPU 0，仅用于初始化 CUDA platform，测试内容为 engine-core
CPU 逻辑。实际环境为 Docker 镜像 `oscar-vllm:v0.25.0-dev`、容器内
`CUDA_VISIBLE_DEVICES=0`、Python 3.12.13、vLLM 0.25.0、PyTorch 2.11.0+cu130、
pytest 9.1.1，检测到的 platform 为 `cuda`。最终结果为
`86 passed, 18 warnings in 98.83s`，容器退出码 0、`OOMKilled=false`；GPU 0
随后为 0 MiB。

完整 pytest 日志、环境、容器 inspect 和实验前后 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/core_regression_cuda_platform/`。

### 7.3 `max_num_seqs=8` 端到端 Smoke

首次多请求镜像 `oscar-vllm:three-pool-batched-20260720` 在模型加载后的
flashinfer autotune dummy attention 中退出：`InputBatch.make_dummy()` 没有 OSCAR
HP rows。补充 dummy 临时唯一 rows 后，CPU 定向测试为 `18 passed in 3.73s`。
fix1 镜像继续通过 flashinfer autotune，但 vLLM 后续正式 kernel warmup 构造的
`_warmup_*` 合成 `SchedulerOutput` 同样绕过 engine-core manager，因此仍缺少 rows。
两次服务均退出码 1、`OOMKilled=false`，均未进入真实请求阶段。

最终实现仅为上述 dummy/warmup 合成请求附加临时唯一 rows，真实请求仍执行严格的
scheduler ownership 完整性校验。相关 CPU 定向测试增至 `19 passed in 3.99s`，
Ruff lint/format 全通过。随后构建不可变镜像
`oscar-vllm:three-pool-batched-fix2-20260720`，镜像 ID 为
`sha256:6dc7d5e92d33e591e0052c7ddf3eb49242ff689f6d9c47ec0dbb1887b2f6a31a`。

GPU 0 在每次重跑前均满足连续两次空闲检查。有效实验固定使用物理 GPU 0，实际
环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:three-pool-batched-fix2-20260720` |
| GPU / Driver / `nvidia-smi` CUDA | NVIDIA B200 / 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / Triton | 0.25.0 / 2.11.0+cu130 / 3.6.0 |
| Transformers | 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | `0` |
| `TRITON_CACHE_DIR` | `/tmp/oscar-batched-smoke-fix2-triton-cache` |

服务以 `max_model_len=8192`、`max_num_seqs=8`、eager、关闭 chunked prefill 和
prefix caching 启动成功，使用 V2 Model Runner 与 OSCAR backend。9.35 GiB available
KV memory 下实际三段式规划为 INT2 history 419408 tokens、BF16 prefix 512 tokens、
BF16 recent 2048 tokens，8192-token 最大并发为 51.20x。

真实请求结果如下：

| 子实验 | 请求数 | Prompt tokens | Completion tokens | HTTP 200 | 并发证据 |
| --- | ---: | ---: | ---: | ---: | --- |
| 第一轮长请求 | 8 | 383-698/请求 | 300/请求 | 8/8 | 服务日志 `Running: 8 reqs` |
| 第二轮 row reuse | 8 | 350-560/请求 | 128/请求 | 8/8 | metrics 峰值 8 |

两轮之间第一轮请求均已完成，因此第二轮要求释放并重新分配全部 8 个 HP rows。
随后启动一个 `max_tokens=1000` 的流式请求，收到首个 SSE 后在 0.373 秒主动断开；
metrics 在 0.227 秒内由 running=1 降到 0，补发请求为 HTTP 200、完成 64 tokens。
这验证了真实 finish/reuse 与 abort/reuse 路径。

有效服务最终退出码 0、`OOMKilled=false`，日志无 traceback、runtime error、CUDA
error 或 OOM；停止后 GPU 0 为 0 MiB。完整结果、metrics、服务/runner 日志、镜像
与容器 inspect、环境和前后 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/multi_request_smoke_fix2/`。preemption 与
eviction 尚未在本实验中触发，因此阶段 3 仍保持进行中。

### 7.4 Scheduler Preemption 与 HP Row 复用

审计 scheduler 释放路径时确认：正常 finish、abort 和 preemption 最终都调用
OSCAR manager 的 `pop_blocks_for_free()`，该调用同时移除 quant block bookkeeping
并释放请求独占 HP row。若启用 consumer KV connector 与 overlapping batches，vLLM
可能延迟 quant blocks 的实际回收，而当前 HP row 会立即释放，存在提前复用风险。
因此本轮在参数层新增明确 fail-fast：OSCAR 拒绝已启用的 KV transfer connector 和
`--kv-offloading-size`；这与第 1 节冻结的支持边界一致，Ruff lint/format 均通过。

生命周期测试复用了 vLLM 现有的真实压力抢占条件：block size 为 16，共 11 个
blocks，其中 1 个为 null block；两个 80-token 请求各占 5 个可用 blocks。两个请求
先后获得 HP rows 0 和 1；处理第一个请求输出后再次调度，因为没有新 block 可分配，
scheduler 将第二个请求置为 `PREEMPTED`，同时 ownership 表只保留第一个请求的 row。
终止第一个请求后，被抢占请求重新调度并获得 row 0，最终 allocator 守恒检查通过。

本测试固定使用物理 GPU 0，仅用于让 CUDA build 正确初始化 platform；被测逻辑为
engine-core scheduler/allocator。GPU 0 在 UTC `11:52:38` 与 `11:54:08` 两次检查中
均为 0 MiB、利用率 0%、无计算进程。实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:three-pool-batched-fix2-20260720` |
| 镜像 ID | `sha256:6dc7d5e92d33e591e0052c7ddf3eb49242ff689f6d9c47ec0dbb1887b2f6a31a` |
| GPU / Driver / `nvidia-smi` CUDA | NVIDIA B200 / 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Transformers / pytest | 5.13.0 / 9.1.1 |
| `CUDA_VISIBLE_DEVICES` | `0` |

首次无 GPU 容器结果为 `19 passed, 1 failed`，唯一失败发生在 `DeviceConfig` 的
platform 推断阶段。首次 GPU 重跑又因新增测试漏导入 `KVCacheConfig` 而得到
`19 passed, 1 failed`；补导入后第二次重跑因测试 manager 的 `log_stats` 与 scheduler
不一致再次得到 `19 passed, 1 failed`。修正测试夹具后，最终有效结果为
`20 passed, 17 warnings in 44.40s`，Ruff lint 全通过、2 个文件 format check 全通过。
测试后 UTC `12:01:17` GPU 0 为 0 MiB、无计算进程。

完整两次 GPU 检查、容器内 `nvidia-smi`、Python/包版本、镜像 inspect、各轮日志与
最终日志 SHA256 位于
`artifacts/oscar_vllm_optimization/20260720/preemption_lifecycle/`。本结果验证的是
quant block 压力下的 preemption/free/reuse；prefix-cache LRU eviction 尚未实现，
不能用本结果替代，阶段 3 因此仍保持进行中。

### 7.5 `max_num_seqs=48` 端到端 Smoke

GPU 0 在 UTC `12:03:33` 与 `12:05:00` 两次检查中均为 0 MiB、利用率 0%、
无计算进程。本实验固定使用物理 GPU 0，服务继续使用 eager、关闭 chunked prefill
与 prefix caching，并将 `max_num_seqs` 提高到 48。实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:three-pool-batched-fix2-20260720` |
| 镜像 ID | `sha256:6dc7d5e92d33e591e0052c7ddf3eb49242ff689f6d9c47ec0dbb1887b2f6a31a` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / pytest | 3.6.0 / 5.13.0 / 9.1.1 |
| `CUDA_VISIBLE_DEVICES` | `0` |

首次启动命令重复传入了镜像 entrypoint 已包含的 `serve` 子命令，服务在参数解析阶段
退出码 2、`OOMKilled=false`，GPU 仍为 0 MiB；该轮未进入模型或 OSCAR 实现，作为
无效编排记录保留。去掉重复子命令后服务启动成功，日志确认 V2 Model Runner 与
OSCAR backend。9.35 GiB available KV memory 下实际规划为 INT2 history 337488
tokens、BF16 prefix 3072 tokens、BF16 recent 12288 tokens；8192-token 请求的容量
日志并发值为 41.20x。该 41.20x 只表示全部请求都达到 8192 tokens 时的容量，
不限制本实验的 48 个短请求同时运行。

48 条请求通过 barrier 同时发出，prompt 长度为 90-106 tokens，每条固定生成 256
tokens。实际结果为 48/48 HTTP 200、0 errors、所有 finish reason 均为 `length`，
墙钟 `10.372958660125732` 秒。`/metrics` 共采集 88 个样本，其中 57 个样本记录
`num_requests_running=48`，因此并发证据不依赖客户端线程数推断。

服务最终退出码 0、`OOMKilled=false`，日志未出现 traceback、runtime error、CUDA
error、OutOfMemory 或 OOM；UTC `12:12:00` GPU 0 为 0 MiB、无计算进程。完整请求
结果、metrics、服务/runner inspect、环境、前后 `nvidia-smi`、失败启动日志和 SHA256
位于 `artifacts/oscar_vllm_optimization/20260720/maxseq48_smoke/`。至此 batch 1/8/48
均有实际端到端或完整精度执行证据；阶段 3 仍只因 prefix-cache LRU eviction 未实现
而保持进行中。

## 8. 阶段 4：滚动 Recent 与 Fused Mixed Prefill

### 8.1 多 Token Recent Demotion

现有 decode 每步只增加一个 token，因此旧 kernel 固定 demote
`final_seq_len - 1 - recent_tokens` 一个位置。chunked prefill 一步可能增加多个 tokens；
应从旧 recent ring 推出的实际区间为：

```text
[max(prefix_tokens, cached_len - recent_tokens),
 min(cached_len, final_seq_len - recent_tokens))
```

新 kernel 使用 request、chunk 内 demote offset 和 KV head 三维 grid，一次 launch
处理 batch 内所有请求的完整推出区间；未提供 query starts 时保留原单 token decode
语义。Python 编译、Ruff lint 与 format check 均通过。

GPU 0 在 UTC `12:18:21` 与 `12:19:48` 连续两次确认为 0 MiB、利用率 0%、无计算
进程。本实验固定使用物理 GPU 0，并用全新 Triton cache 冷编译。实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:three-pool-batched-fix2-20260720` |
| 镜像 ID | `sha256:6dc7d5e92d33e591e0052c7ddf3eb49242ff689f6d9c47ec0dbb1887b2f6a31a` |
| GPU / Driver / `nvidia-smi` CUDA | NVIDIA B200 / 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / pytest | 3.6.0 / 5.13.0 / 9.1.1 |
| `CUDA_VISIBLE_DEVICES` | `0` |

新增 oracle 构造 cached length 384、chunk length 17、prefix 64、recent 256，要求
positions 128-144 一步全部转为 INT2、position 145 保持未写。首轮 PyTorch 数值
reference 在 value 侧出现 2/4352 个 INT2 舍入临界差异，完整结果为
`7 passed, 1 failed in 40.66s`；这不能可靠区分区间错误与浮点舍入顺序差异，因此
该轮不作为有效正确性结论。

修正后的严格 oracle 将一次 17-token demote 产生的完整 packed cache，与既有单
token demote 连续执行 17 次的结果逐字节比较，同时逐项检查 128-144 已写、145 未写。
最终完整 CUDA 文件结果为 `8 passed, 16 warnings in 48.77s`，退出码 0；原有单 token、
batch2 反序 HP rows、store/decode oracle 同时通过。UTC `12:24:53` GPU 0 已回收为
0 MiB、无计算进程。

完整环境、两轮 CUDA 日志、静态检查、前后 `nvidia-smi`、镜像 inspect 和最终日志
SHA256 位于 `artifacts/oscar_vllm_optimization/20260720/chunked_prefill/`。本子步只
证明 rolling recent 的多 token demotion 正确；cached mixed attention 与当前 BF16
chunk 的 LSE 合并尚未实现，启动参数仍拒绝 chunked prefill，因此阶段 4 保持进行中。

### 8.2 Cached Split-KV 的 Query-to-Request 映射与 LSE

chunked prefill 中，一个 request 的多个 query token 需要共享同一份 block table、
BF16 prefix/recent row 和 scheduler-owned HP row，但每个 query 仍有独立的 cached
length 与输出。cached split-KV kernel 因此新增显式 `query_to_req_indices`：cache
寻址按映射后的 request index，`seq_lens`、输出和 LSE workspace 继续按 query index。
未提供映射时仍使用 `query index == request index`，保持现有单 token decode ABI。

wrapper 同时新增可选 LSE 返回；V inverse rotation 在返回前继续作用于 output，LSE
不受旋转影响。Python 编译、Ruff lint 与 format check 均通过。首次静态容器命令因
login shell 重置 PATH 而在检查前退出码 127；改用镜像实际的 `/usr/bin/python3`
后完成有效静态检查。首次 CUDA pytest 又因仓库根优先于 installed package、遮蔽
`_C_stable_libtorch` 而在收集前退出码 4，未执行任何测试；有效重跑将单个测试文件
复制到 `/tmp`，并从已覆盖当前源码的 installed package 导入，保留镜像编译扩展。

有效重跑前 GPU 0 在 UTC `12:33:39` 与 `12:35:37` 两次检查中均为 0 MiB、利用率
0%、无计算进程。本实验固定使用物理 GPU 0，实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:three-pool-batched-fix2-20260720` |
| 镜像 ID | `sha256:6dc7d5e92d33e591e0052c7ddf3eb49242ff689f6d9c47ec0dbb1887b2f6a31a` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / pytest | 3.6.0 / 9.1.1 |
| `CUDA_VISIBLE_DEVICES` | `0` |

新增 oracle 将 5 个 query 映射到 2 个 request，映射为 `[0, 0, 0, 1, 1]`，并把
一次 batched mapped launch 的 output 与 LSE 分别同 5 次既有单 query launch 做
逐元素精确比较。完整 CUDA 文件实际结果为 `8 passed in 47.50s`，退出码 0；原有
单 token、batch2 反序 HP rows、store/decode 和 multi-token demote oracle 同时通过。
UTC `12:37:15` 实验后 GPU 0 为 0 MiB、无计算进程。

完整两轮 GPU 检查、无效收集日志、有效 pytest 日志、环境与前后 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/chunked_prefill_mapped_query/`。本子步
证明 cached 分支可在一次 batched launch 中服务同一请求的多个 query 并返回 LSE；
当前 BF16 chunk 的 FlashAttention LSE 与两分支稳定合并仍待实现，阶段 4 保持进行中。

### 8.3 Cached/Current Attention LSE 合并与更新时序

continuation prefill 现在分成两个不构造全历史 BF16 K/V 的 attention state：cached
分支由第 8.2 节的 mapped split-KV kernel 读取旧三段式 cache，当前 chunk 分支由
FlashAttention varlen 对 raw BF16 K/V 执行 causal attention。两边分别返回 output
和 LSE，再复用 vLLM 已有的 `merge_attn_states` 做稳定归一化合并。cached 分支当前
固定使用 1 个 KV split；以 Qwen3 的 32 query heads、head dim 128 计算，8192 query
tokens 的中间 output/LSE workspace 约 129 MiB，低于 10 GiB KV budget 的 5%。

cache update 时序同时改为：decode 先 demote/store、再读取包含当前 token 的 cache；
prefill 先读取旧 cache 并与当前 raw chunk 合并、再批量 demote/store。mixed batch
先拆出 decode 与 prefill metadata，再分别执行上述顺序。首 chunk 仍直接走原有
FlashAttention fast path，无 cached workspace；FlashAttention 不可用时仅允许无缓存
prefill，continuation 会明确报错。

Python 编译、Ruff lint 与 format check 均通过。GPU 0 在 UTC `12:45:07` 与
`12:46:36` 两次检查中均为 0 MiB、利用率 0%、无计算进程；本实验固定使用物理
GPU 0，实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:three-pool-batched-fix2-20260720` |
| 镜像 ID | `sha256:6dc7d5e92d33e591e0052c7ddf3eb49242ff689f6d9c47ec0dbb1887b2f6a31a` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / pytest | 3.6.0 / 9.1.1 |
| `CUDA_VISIBLE_DEVICES` | `0` |

新增 forward-level oracle 在同一 batch 中构造两个请求：request 0 为 cached length
384、chunk length 17，request 1 为无缓存、chunk length 13。reference 从实际 INT2
dequant history、BF16 prefix/recent 和当前 raw BF16 chunk 逐 query 计算完整 causal
attention；实际 backend 输出与 reference 在 `atol=rtol=1.5e-2` 下通过。该用例还
检查 attention 后 positions 128-144 已全部 demote 为 INT2；如果当前 chunk 在
attention 前覆写 recent ring，这 17 个位置的 cached attention 输入会错误，数值
oracle 无法通过。

完整 CUDA 文件实际结果为 `9 passed, 61 warnings in 79.06s`，退出码 0；原有单
token decode、batch2 反序 HP rows、mapped output/LSE 和 multi-token demote oracle
同时通过。UTC `12:49:08` 实验后 GPU 0 为 0 MiB、无计算进程。完整环境、前后
`nvidia-smi`、pytest 日志和退出码位于
`artifacts/oscar_vllm_optimization/20260720/chunked_prefill_lse_merge/`。

本子步证明 fused mixed prefill 的核心数值路径和读后写时序正确；启动参数仍拒绝
chunked prefill，且 chunk 128/256/257/2048/8192 的端到端最终 tier 对照尚未执行，
因此阶段 4 保持进行中。

### 8.4 Chunk Size 最终 Tier 一致性矩阵

在第 8.3 节数值路径通过后，删除了参数层唯一的 OSCAR chunked-prefill 拒绝逻辑；
eager、prefix caching、speculative decoding、KV connector 和 offload 等其他
fail-fast 均保持不变。解除限制及新增测试的 Python 编译、Ruff lint 与 format check
全部通过。

最终 tier oracle 使用同一条 8449-token BF16 K/V 和同一 physical block table：
reference 一次性按最终 sequence length 写入三张量；被测路径分别按 chunk size
`128/256/257/2048/8192` 重复执行当前 chunk INT2 store、旧 recent 批量 demote 与
BF16 prefix/recent store。每个 chunk size 最终都对完整 INT2 tensor、BF16 prefix
tensor 和 BF16 recent tensor做逐字节比较，因此不仅检查 tier 数量，也覆盖逻辑位置、
recent ring wrap 和最后 256 tokens 的 BF16 内容。

GPU 0 在 UTC `12:53:08` 与 `12:54:28` 两次检查中均为 0 MiB、利用率 0%、无计算
进程。本实验固定使用物理 GPU 0，实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:three-pool-batched-fix2-20260720` |
| 镜像 ID | `sha256:6dc7d5e92d33e591e0052c7ddf3eb49242ff689f6d9c47ec0dbb1887b2f6a31a` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / pytest | 3.6.0 / 9.1.1 |
| `CUDA_VISIBLE_DEVICES` | `0` |

5 个参数化 tier 用例和原有 9 项 CUDA 回归全部通过，完整结果为
`14 passed, 61 warnings in 80.81s`，退出码 0。UTC `12:56:51` 实验后 GPU 0 为
0 MiB、无计算进程。完整环境、前后 `nvidia-smi`、pytest 日志和退出码位于
`artifacts/oscar_vllm_optimization/20260720/chunked_prefill_tier_matrix/`。

本结果满足指定 chunk size 的 kernel/tensor 级最终 tier 一致性，但尚未证明 vLLM
scheduler 会按这些 budget 正确生成 continuation metadata，也未覆盖真实模型服务的
HTTP 请求和 decode continuation。因此阶段 4 保持进行中，下一子步使用新镜像运行
真实 chunked-prefill 服务。

### 8.5 真实 Scheduler 与服务端到端矩阵

构建不可变里程碑镜像 `oscar-vllm:chunked-prefill-20260720`，镜像 ID 为
`sha256:50d636d920be2f541147a929c952a295789af7dd77614bfbf6e0741e794d3568`。
GPU 0-4 在 UTC `13:00:27` 与 `13:02:10` 两次检查中均为 0 MiB、利用率 0%、
无计算进程。本实验固定分配物理 GPU 0-4，分别运行 chunk
`128/256/257/2048/8192`，每个容器内 `CUDA_VISIBLE_DEVICES=0`，端口彼此独立。

实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:chunked-prefill-20260720` |
| 镜像 ID | `sha256:50d636d920be2f541147a929c952a295789af7dd77614bfbf6e0741e794d3568` |
| 模型 | `/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507` |
| GPU | 5 x NVIDIA B200，单卡 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton | 3.6.0 |
| `CUDA_VISIBLE_DEVICES` | 每个服务容器内均为 `0` |
| 公共配置 | eager、`max_model_len=8192`、`max_num_seqs=8`、prefix caching 关闭 |

5 个服务日志均明确记录 `enable_chunked_prefill=True` 和各自 budget，使用 V2 Model
Runner 与 OSCAR backend；UTC `13:05:46` 全部 health ready。每个点同时发送两条
temperature 0 请求，prompt 分别为 2101 和 4201 tokens，各生成 64 tokens。前四个
budget 均小于至少一条 prompt，因而必然经过 scheduler continuation；8192 点的两条
prompt 合计 6302 tokens，可作为同镜像单步 prefill 对照。

实际结果如下；时长由落盘的整秒 UTC 开始/结束时间重建，只用于说明运行完成，不能
作为阶段 6 的性能数据：

| Chunk | HTTP 200 | Prompt tokens | Completion tokens | Finish | 粗粒度时长 | Peak running | Peak waiting |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 128 | 2/2 | 2101 / 4201 | 64 / 64 | length / length | 21 s | 2 | 1 |
| 256 | 2/2 | 2101 / 4201 | 64 / 64 | length / length | 26 s | 2 | 1 |
| 257 | 2/2 | 2101 / 4201 | 64 / 64 | length / length | 19 s | 2 | 0 |
| 2048 | 2/2 | 2101 / 4201 | 64 / 64 | length / length | 16 s | 2 | 0 |
| 8192 | 2/2 | 2101 / 4201 | 64 / 64 | length / length | 5 s | 2 | 0 |

首次 workload 汇总在 10 条响应均已落盘后，因 jq 变量名 `$end` 与关键字冲突而使
5 个 runner 退出码均为 3；该错误只影响 summary 生成。10 个原始 HTTP code 均为
200，response JSON 均有完整 usage/choice。修复变量名后直接从现有响应与时间戳重建
summary，没有重复模型请求，也没有把 runner 退出码误写为服务失败。

5 个服务日志均出现 OSCAR Triton mixed read/write，且 traceback、RuntimeError、
CUDA error 和 OOM 关键字计数均为 0。服务最终全部退出码 0、`OOMKilled=false`；
UTC `13:09:38` GPU 0-7 均为 0 MiB、无计算进程。完整 GPU 检查、镜像/容器 inspect、
health 轮询、metrics、原始响应、重建 summary、服务日志和回收后 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/chunked_prefill_e2e_matrix/`。

结合第 8.1-8.4 节的 multi-token demote、mapped output/LSE、full-attention 数值
oracle 和三张量 byte-exact tier 矩阵，阶段 4 的正确性与真实服务验收已完成。性能和
总 workspace 峰值仍按计划在阶段 6 用 warm-up 后的独立 benchmark/profiling 验证。

## 9. 阶段 5：Prefix Cache

### 9.1 Scheduler Ownership 与可共享命中边界

vLLM 的普通 prefix cache 只管理 scheduler block 的内容哈希、引用计数和 LRU；
OSCAR 原实现则把 BF16 prefix 与 recent 一起按请求 HP row 寻址。若只删除启动参数的
prefix-caching fail-fast，scheduler 会复用命中的 INT2 block ID，但新请求会获得另一
个 HP row，kernel 因而从未初始化的 BF16 prefix 地址读取。这不是可用的 prefix cache。

本子步把绝对位置 `[0, 64)` 的 4 个 blocks 分别绑定到共享 BF16 prefix pages；
recent 仍只由活动请求的 HP row 持有。64 tokens 之后的 block 只有在离开 256-token
recent window、实际执行 demotion 后才标记为 INT2-ready。cache-hit 查找沿内容哈希链
逐 block 检查 `BF16 prefix page` 或 `INT2-ready`，在首个仍属于 recent 的 block 停止。
因此命中边界由实际 tier 状态决定，而不是把完整哈希链错误地视为可共享。

prefix pool 默认仍只预留 `max_num_seqs * 4` pages；已结束请求的 hashed prefix pages
按对应 quant block 的 LRU 顺序保留或驱逐。新活动请求缺 page 时，manager 先驱逐
`ref_cnt=0` 的最旧 quant block hash，再回收其 BF16 page；活动请求引用的 page 不会
进入 free queue。`VLLM_OSCAR_PREFIX_CACHE_EXTRA_TOKENS` 已计费的额外 pages 将用于
增加可保留的 cached prefix，而不改变默认并发保障。

新增 CPU 合约先在旧实现上得到有效失败：两项均失败，其中 400-token source 被错误
报告为 400-token 命中，且 manager 不存在 prefix page ownership。实现后，定向结果为
`2 passed, 20 deselected in 0.84s`。实际状态推进如下：

| Source 已计算 tokens | 可共享命中 tokens | 原因 |
| ---: | ---: | --- |
| 400 | 144 | 前 64 为共享 BF16，随后 80 已 demote 为 INT2，末尾 256 recent 排除 |
| 512 | 256 | source 继续计算后，更多旧 recent blocks 已 demote，命中边界单调推进 |

同一测试还验证默认 4-page pool 在旧请求释放后保持 cached pages；新请求到来时按
quant-block LRU 驱逐旧 hash，page IDs 完整复用为 `(0, 1, 2, 3)`，最终 page accounting
守恒。完整 OSCAR core 文件结果为 `21 passed, 1 failed`；唯一失败发生在既有
scheduler preemption 用例构造 `DeviceConfig` 时，无 GPU 容器无法推断 CUDA platform，
早于测试断言。这与第 7.4 节已记录的环境限制相同，将在本阶段 GPU 双检后复跑。

本子步使用 Docker 镜像 `oscar-vllm:chunked-prefill-20260720` 的已安装 vLLM 包，并
只读覆盖当前 3 个 core 源文件；Python 为 3.12.13。Python 编译、Ruff lint 和 4 个
文件的 format check 全部通过。本结果只完成 scheduler-side ownership、命中裁剪和
LRU page reuse；prefix page table 尚未传入 worker/kernel，启动参数仍保持 fail-fast，
因此阶段 5 继续进行中。

### 9.2 Prefix Page Table Metadata 链路

scheduler 现在对本步每个 OSCAR 请求同时输出 `request_id → hp_row` 和
`request_id → prefix_page_ids`。page table 当前固定为 4 entries，对应 64-token
BF16 prefix 的 4 个 16-token pages。两套字典必须与本步请求 ID 集完全相等；任一
缺失都会在 worker 准备输入时明确失败，不能回退到 batch row 或隐式连续地址。

V1 与 V2 Model Runner 都只按当前 batch 的 `request_id → index` 重排这两组
scheduler-owned metadata，并复制到预分配 int32 GPU buffer。二维 prefix page table
随后经 `InputBatch`、`CommonAttentionMetadata` 和 `OscarMetadata` 进入 attention
backend；mixed decode/prefill 拆分与 unpadded metadata 都按请求维同步切片。worker
不分配、不释放 page，也不维护第二套 LRU 状态。

dummy attention 与正式 kernel warmup 不经过 engine-core allocator，因此为合成请求
生成互不重叠的临时 page table；真实请求仍执行严格 ID 集校验。buffer 的第二维在
KV cache 初始化时从 `prefix_tokens / block_size` 计算，默认测试值为 4，不把 64 写入
kernel ABI。

本子步使用 Docker 镜像 `oscar-vllm:chunked-prefill-20260720`，将当前工作区源码复制
进镜像已安装 package，保留原有编译扩展；Python 为 3.12.13。14 个相关文件的 Python
编译、Ruff lint 和 format check 全部通过。无 GPU core 回归排除已知依赖 CUDA
platform 的 scheduler preemption 用例后，实际结果为
`21 passed, 1 deselected, 16 warnings in 4.26s`，容器随后正常停止并删除。

本结果完成了 page-table 控制面传递，但 store/decode kernel 仍按旧 HP row 读取
BF16 prefix；因此尚不能开启 prefix caching，阶段 5 继续进行中。

### 9.3 Page-Table Kernel 寻址与物化屏障

store 与 decode kernel 已改为按 `prefix_page_ids[request, absolute_block]` 读取
BF16 prefix page，再用 `page_id * block_size + block_offset` 计算物理 slot；请求独占
recent 仍按 HP row 和 256-token ring 寻址。生产 backend 在普通 decode、continuation
prefill 的 cached 分支以及 cache update 三条路径都显式传入 page table。store/decode
wrapper 都校验 page table 必须为二维，且第二维乘 block size 必须恰好覆盖 prefix window；
不合法 metadata 会在 Triton launch 前失败。

调度审计进一步发现，vLLM 会在 worker 执行前把本步完整 blocks 加入内容哈希表。
普通 BF16 attention 先为全 batch 写 cache 再读取，可以容忍同一步相同内容共享；
OSCAR continuation prefill 为保护旧 recent ring，必须先 attention、后 demote/store。
若只检查 page ownership 或 INT2 tier，同一步后到请求会命中尚未物化的数据。

新增失败测试在修复前实际得到 `1 failed, 24 deselected, 16 warnings in 0.71s`：
64-token source 在 `cache_blocks()` 后、worker 尚未执行时，branch 错误命中 3 个完整
blocks。修复后的状态机为：`cache_blocks()` 只把符合 tier 条件的 block 记为 pending；
同步 worker step 完成后，下一次 `new_step_starts()` 才提升为 `prefix_ready` 或
`int2_ready`。block reuse、LRU eviction 和 reset 都会同步清除 pending/ready，避免
复用 block ID 继承旧内容状态。

由于 vLLM 0.25 默认启用 async scheduling，而 async 的下一 scheduler step 不保证上一
GPU step 已完成，OSCAR prefix caching 现在将默认 `async_scheduling=None` 收紧为
`False`；用户若显式请求 async 与 OSCAR prefix cache 的组合，启动参数会明确拒绝。
关闭 prefix caching 时不改变既有 OSCAR async 行为。

修复后的 prefix 定向合约为 `5 passed, 20 deselected, 16 warnings in 1.36s`；新增
pending eviction/reuse 合约后，完整无 GPU core 回归为
`25 passed, 1 deselected, 16 warnings in 5.08s`。涉及 barrier 的 5 个文件均通过
Python 编译、Ruff lint 与 format check。该阶段使用 Docker 镜像
`oscar-vllm:chunked-prefill-20260720`、Python 3.12.13；未分配 GPU。

当前源码已构建为候选镜像 `oscar-vllm:prefix-cache-candidate-20260720`，镜像 ID 为
`sha256:e0545221538afa2a06f039240b80016449cabcd74e72a8d2b9512a8ac1d95e79`。
直接从该镜像运行同一无 GPU core 回归的实际结果为
`25 passed, 1 deselected, 16 warnings in 5.91s`。镜像环境为 Python 3.12.13、
vLLM 0.25.0、PyTorch 2.11.0+cu130、PyTorch CUDA 13.0、Triton 3.6.0。该 tag 仅为
待验证 candidate，不是 CUDA 已通过的阶段里程碑。

UTC `13:36:35`、`13:37:41`、`13:40:47`、`13:46:15`、`13:54:20`、`13:56:30`
的检查均显示
GPU 0-7 被同一外部 SGLang 作业占用，单卡约 181.7-182.4 GiB；最后一次检查利用率为
100%。因此 page-table CUDA oracle 尚未执行，本节不声称 kernel 数值验证或真实
prefix-cache 服务已经通过，阶段 5 继续进行中。

### 9.4 Prefix Page-Table CUDA Oracle

外部 SGLang 作业释放后，UTC `14:08:58` 首次确认 GPU 0-3 为 0 MiB、无计算进程；
UTC `14:11:16` 的完整 `nvidia-smi` 再次确认 GPU 0-7 全部为 0 MiB、利用率 0%、
无运行进程。本实验固定使用物理 GPU 0，实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:prefix-cache-candidate-20260720` |
| 镜像 ID | `sha256:e0545221538afa2a06f039240b80016449cabcd74e72a8d2b9512a8ac1d95e79` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU | NVIDIA B200，183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM | 0.25.0 |
| PyTorch / PyTorch CUDA | 2.11.0+cu130 / 13.0 |
| Triton | 3.6.0 |
| Transformers | 5.13.0 |
| pytest | 9.1.1 |
| Triton cache | `/tmp/oscar-prefix-page-table-triton`（实验前为空） |

直接运行候选镜像内完整 `tests/quantization/test_oscar.py`，实际结果为
`15 passed, 75 warnings in 87.27s`，进程退出码 0。新增 batched page-table oracle
刻意让两个请求的 HP rows 为 `[1, 0]`，而 BF16 prefix pages 分别为 `[0,1,2,3]`
和 `[4,5,6,7]`，证明 prefix 寻址不再隐式依赖请求独占 HP row。测试同时覆盖非法
page-table 宽度在 launch 前被拒绝，并完整回归既有 store/decode、mapped-query、
multi-token demote、mixed cached prefill 和 chunked-prefill oracle。

UTC `14:14:44` 实验后完整 `nvidia-smi` 显示 GPU 0-7 均为 0 MiB、利用率 0%、
无运行进程。由此 page-table 数据面 CUDA 数值验证已通过；真实 prefix-cache 服务的
共享命中、冷并发物化屏障和 LRU eviction 仍需下一子阶段验证，因此阶段 5 尚未完成。

### 9.5 GPU Core Preemption 回归

为补齐第 9.1 节无 GPU 环境中唯一排除的 scheduler preemption 用例，GPU 0 在 UTC
`14:14:44` 与 `14:17:27` 两次间隔超过一分钟的检查中持续为 0 MiB、无计算进程。
本实验继续固定物理 GPU 0，使用与第 9.4 节相同的候选镜像及软件环境，容器内
`CUDA_VISIBLE_DEVICES=0`。

首次完整 core 文件结果为 `25 passed, 1 failed, 17 warnings in 45.01s`。唯一失败
发生在 preemption 后恢复请求的 prefix page 断言：测试硬编码期望 `(0,1,2,3)`，
allocator 实际从 free-list 返回 `(3,2,1,0)`。四个物理 page 无遗漏、无重复，且
page table 本来就负责把逻辑 prefix block 映射到任意物理 page ID；因此 tuple 顺序
不是 allocator 对外保证，失败不能解释为容量泄漏或错误寻址。

测试已收紧到真正契约：恢复请求必须取得 4 个互异 page、集合必须完整复用
`{0,1,2,3}`，并同时执行 HP row 与 prefix page accounting 一致性检查。项目配置下
Ruff lint 与 format check 均通过；该修改只涉及测试断言，不改变候选镜像运行时代码。

GPU 0 在 UTC `14:20:50` 与 `14:21:59` 再次连续确认为 0 MiB、无计算进程。使用同一
候选镜像的已安装 vLLM 包，只读挂载修正版测试目录后，完整结果为
`26 passed, 17 warnings in 44.60s`，退出码 0。UTC `14:23:52` 实验结束后完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、利用率 0%、无运行进程。首次失败与有效重跑
均保留在 artifacts 中；本结果补齐 prefix page 生命周期的 GPU platform 回归。

### 9.6 默认 Prefix Pool 真实服务

GPU 0 在 UTC `14:23:52` 与 `14:25:24` 连续空闲后固定分配给真实服务。服务使用
候选镜像、Qwen3-4B-Instruct-2507、eager、chunked prefill、prefix caching、
`max_num_seqs=2`、`max_num_batched_tokens=8192` 和 0.1 GPU memory utilization；
容器内 `CUDA_VISIBLE_DEVICES=0`。其余软件、GPU 和 driver 环境与第 9.4 节一致。

服务日志明确记录 `enable_prefix_caching=True`、`enable_chunked_prefill=True`、
`Asynchronous scheduling is disabled`，并实际规划 INT2 history 431696 tokens、
BF16 prefix 128 tokens、BF16 recent 512 tokens。这里默认 prefix pool 恰为
`max_num_seqs × 64 = 128` tokens，即 8 个 16-token pages。UTC `14:28:24` health
ready 后执行 11 条真实 OpenAI completion 请求，全部 HTTP 200、completion 1 token。

| 场景 | Source cached | Repeat/branch cached | 结果 |
| --- | ---: | ---: | --- |
| 相同 alpha | 0 / 2100 | 1840 / 2100 | 命中 87.62% |
| alpha 分支 | - | 1840 / 2100 | 命中 87.62% |
| 冷并发 gamma | 0、1840 / 2100 | 1840 / 2100 | 三条生成文本一致 |
| LRU lambda | 0 / 2100 | 1840 / 2100 | 最新项命中 87.62% |
| LRU theta 驱逐后 | 0 / 2100 | 0 / 2100 | 最旧项已驱逐 |

alpha 场景 metrics delta 为 queries 6300、hits 3680；冷并发场景相对前一快照为
queries 6300、hits 3680；LRU 场景为 queries 10500、hits 1840。所有 reset endpoint
均 HTTP 200，日志未出现 traceback、RuntimeError、CUDA error、assertion error 或 OOM。

工作负载首次退出码为 1，但失败只来自测试假设：脚本要求两个并发 curl 都必须报告
0 cached tokens。并发到达不保证同一个 scheduler step；本次第一条冷 miss 后，第二条
在后续同步 step 合法命中 1840 tokens。物化屏障的“同一步不得命中”已由第 9.3 节的
失败/修复 CPU 合约精确控制。服务验收已改为至少一条冷 miss，其他并发请求只能为
0 或不低于 70% 的完整命中，并要求三条确定性输出一致。修正断言只对已落盘
summary 重新执行，结果为 `true`、退出码 0；没有重复模型推理，也没有把首次 runner
退出码改写为 0。

服务于 UTC `14:31:21` 正常停止，容器 `exit=0`、`OOMKilled=false`；完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、利用率 0%、无运行进程。默认 prefix pool 的
共享、分支、并发响应一致性和 LRU eviction 已完成真实服务验证，也补齐阶段 3 最后
一项 eviction 验收。阶段 5 仍需验证额外 prefix pages 能延长缓存保留后再结束。

### 9.7 额外 Prefix Pages 保留验证

为验证 `VLLM_OSCAR_PREFIX_CACHE_EXTRA_TOKENS` 不只是容量日志参数，GPU 0 在 UTC
`14:31:21` 与 `14:33:08` 连续空闲后固定分配给第二个真实服务。配置与第 9.6 节
相同，仅新增 `VLLM_OSCAR_PREFIX_CACHE_EXTRA_TOKENS=64`。日志中的 BF16 prefix
从默认 128 tokens 增至 192 tokens，即从 8 pages 增至 12 pages；BF16 recent 保持
512 tokens，INT2 history 因同一显存预算相应从 431696 降至 431296 tokens。

UTC `14:36:12` health ready 后，依次发送 theta、kappa、lambda 三个 2100-token
不同前缀，再重复最新 lambda 和最旧 theta。5/5 请求均 HTTP 200，工作负载直接
退出码 0、断言为 `true`：

| 请求 | Cached tokens | 结果 |
| --- | ---: | --- |
| theta source | 0 / 2100 | 冷 miss |
| kappa source | 0 / 2100 | 冷 miss |
| lambda source | 0 / 2100 | 冷 miss |
| lambda repeat | 1840 / 2100 | 命中 87.62% |
| theta repeat | 1840 / 2100 | 命中 87.62% |

metrics delta 为 queries 10500、hits 3680、local compute 6820、local cache hit 3680。
与第 9.6 节默认 8-page pool 中 theta 被驱逐的结果对照，新增 4 pages 真实保留了第三
个完整 BF16 prefix，使最旧 theta 仍可复用；这证明额外 pool 的预算、allocator LRU、
scheduler metadata 和 kernel page table 已端到端连通。

服务于 UTC `14:37:22` 正常停止，容器 `exit=0`、`OOMKilled=false`，日志无
traceback、RuntimeError、CUDA error、assertion error 或 OOM；完整 `nvidia-smi`
显示 GPU 0-7 均为 0 MiB、利用率 0%、无运行进程。已验证候选镜像建立里程碑 tag
`oscar-vllm:prefix-cache-20260720`，镜像 ID 保持
`sha256:e0545221538afa2a06f039240b80016449cabcd74e72a8d2b9512a8ac1d95e79`。

结合第 9.1-9.7 节的 CPU 状态机、CUDA page-table oracle、GPU core preemption、
默认/额外 pool 真实服务及 A/B eviction 结果，当时阶段 5 的既定功能验收已完成；但第
10.6 节后续发现 prefix-hit recent ownership 数值缺陷，因此阶段 5 已重新打开。修复进展
见第 10.7 节，重建镜像的真实服务复测通过前不能再次标记完成。

## 10. 阶段 6：容量与性能优化

### 10.1 精确 10 GiB 容量与 Allocator 效率

使用已验证里程碑镜像 `oscar-vllm:prefix-cache-20260720` 直接调用当前三 pool
planner，固定 Qwen3-4B 的 36 层、8 KV heads、head size 128、block size 16、
prefix 64、recent 256 和精确 `10737418240` bytes。该子步为 Docker 内 CPU 计算，
未分配 GPU；量化 token 为 23040 bytes，BF16 token 为 147456 bytes。

| `max_num_seqs` | BF16 slots | Prefix / recent slots | OSCAR 物理 slots | 最坏保证 slots | 同 HP 理论最优 | 物理倍率 | 最坏保证倍率 | Allocator waste |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 72816 | 64 / 256 | 463984 | 463969 | 463985 | 6.372006152x | 6.371800154x | 0.003448% |
| 8 | 72816 | 512 / 2048 | 449648 | 449528 | 449649 | 6.175126346x | 6.173478356x | 0.026910% |
| 48 | 72816 | 3072 / 12288 | 367728 | 367008 | 367729 | 5.050098879x | 5.040210943x | 0.196068% |

三点实际分配均为 10737377280 bytes，只留下 40960 bytes。最坏保证值额外扣除每个
活动请求最多 15 个 partial-page slots；即使采用该保守口径，三点相对同 HP 预留下
的理论最优损失仍远低于 2% 门槛。`max_num_seqs` 增大时总体物理倍率下降来自每个
请求固定 320-token BF16 reserve，不是 allocator 碎片。

单请求最坏保证倍率 `6.371800154x` 高于 `6.2x` 硬门槛。相对旧 padded-page
OSCAR-vLLM 的 `4.571522742x`，当前物理倍率提高 39.384763%；相对既有同预算官方
OSCAR-SGLang 单请求物理结果 `6.286860031x`，当前物理倍率高 1.354351%，最坏保证
倍率仍高 1.351074%。该跨框架数值使用各自已落地的默认 HP pool 策略，不能解释为
算法语义差异；当前提升来自三 pool 独立计费后不再为每个 INT2 block 捆绑 BF16 arena。

原始结构化结果位于
`artifacts/oscar_vllm_optimization/20260720/capacity_10g/capacity.json`。容量与
allocator 两项门槛已通过；阶段 6 继续执行 warm-up 后的 prefill、decode、batch、
prefix-hit TTFT 和 workspace 验证。

### 10.2 冻结原型单请求性能基线

为建立 M2 的同机对照，固定使用物理 GPU 0 运行阶段 0 冻结镜像，服务配置为
Qwen3-4B-Instruct-2507、BF16、OSCAR INT2、精确 10 GiB KV tensor 预算、
`max_model_len=8192`、`max_num_seqs=1`、eager，并关闭 chunked prefill 与 prefix
caching。服务端实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:v0.25.0-dev` |
| 镜像 ID | `sha256:0356873d691c97f72ccc08eda4ca7aecb9b297206763fab8ea3ad01b3b4f1e7e` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU | NVIDIA B200，183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

GPU 0 在服务分配前两次间隔超过一分钟的检查中均为空闲。服务于 UTC
`14:45:59` health ready；benchmark client 使用里程碑镜像中的官方
`vllm bench serve`，不分配 GPU，通过 host network 访问服务。每个正式点先执行 2 次
warm-up，再以 concurrency 1、固定长度 random dataset、`random_range_ratio=0`、
`ignore_eos` 和 seed 42 测量 5 次。client 未显式传 `temperature`，因此由同一服务端
generation config 决定；后续当前实现对照必须保持这一设置不变。

| 工作负载 | 成功 / 失败 | 关键结果 | 波动 |
| --- | ---: | --- | --- |
| Prefill 8000 input / 1 output | 5 / 0 | mean TTFT `130.479 ms` | std `7.567 ms`，CV `5.80%` |
| Decode 512 input / 512 output | 5 / 0 | mean TPOT `22.631 ms`，output `44.032 tok/s` | TPOT std `0.061 ms`，CV `0.268%` |

首轮 prefill CV 超过 3% 门槛，因此以 seed 43、无额外 warm-up 连续补充 10 个相同
样本。合并 15 个样本后全部成功，累计处理 120000 input tokens 和 15 output tokens；
mean TTFT 为 `135.026 ms`、population std 为 `8.293 ms`、CV 仍为 `6.142%`。该点已按
计划继续采样，剩余波动作为实际服务噪声保留，不用首轮较低均值替代合并基线。
decode 的 5 个正式样本累计处理 2560 input tokens 和 2560 output tokens，mean TTFT
为 `63.245 ms`、mean ITL 为 `22.675 ms`、mean E2EL 为 `11627.632 ms`。

服务于 UTC `14:50:59` 正常停止，容器 `exit=0`、`OOMKilled=false`；停止后
`nvidia-smi` 显示 GPU 0 为 `0 MiB / 183359 MiB`、利用率 0%，无计算进程。原始逐请求
结果、15 样本合并统计、benchmark/server 日志、容器 inspect、镜像与软件版本、服务
前后完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/performance_m2/baseline/`。本节只冻结旧版
性能基线；是否满足“不回退超过 10%”需待相同配置的当前里程碑实测后判断。

### 10.3 当前里程碑单请求性能对照

冻结服务停止后的 UTC `14:51:07` 与 `14:53:01` 两次完整 `nvidia-smi` 均显示
GPU 0-7 为 0 MiB、利用率 0%、无计算进程，因此继续固定使用物理 GPU 0。当前服务
使用里程碑镜像 `oscar-vllm:prefix-cache-20260720`，镜像 ID 为
`sha256:e0545221538afa2a06f039240b80016449cabcd74e72a8d2b9512a8ac1d95e79`；
容器内 `CUDA_VISIBLE_DEVICES=0`。GPU、driver、Python、vLLM、PyTorch、CUDA、
Triton 和 Transformers 版本均与第 10.2 节相同。

服务参数与冻结原型保持一致：Qwen3-4B-Instruct-2507、BF16、OSCAR INT2、精确
10 GiB、`max_model_len=8192`、`max_num_seqs=1`、eager、关闭 chunked prefill 和
prefix caching。服务 UTC `14:55:54` health ready；日志确认 OSCAR backend 和
asynchronous scheduling 均已启用。benchmark client 继续使用相同镜像、host network、
固定长度 random dataset、concurrency 1、`random_range_ratio=0`、`ignore_eos`、相同
seed/warm-up 及服务端默认 temperature 行为。

| 指标 | 冻结原型 | 当前里程碑 | 当前相对变化 | 10% 门槛 |
| --- | ---: | ---: | ---: | --- |
| Prefill 8000/1 mean TTFT（15 样本） | 135.026 ms | 135.602 ms | +0.426% | 通过 |
| Decode 512/512 mean TPOT（5 样本） | 22.631 ms | 23.386 ms | +3.338% | 通过 |
| Decode output throughput | 44.032 tok/s | 42.593 tok/s | -3.269% | 参考值 |

当前 prefill 首轮 5/5 成功，mean TTFT 为 `136.341 ms`、CV 为 `7.390%`；按相同规则
追加 10 次后，合并 15 个样本全部成功，mean 为 `135.602 ms`、population std 为
`6.825 ms`、CV 为 `5.033%`。冻结与当前两侧都已因 CV 超过 3% 继续采样，表中比较
统一使用各自 15 样本合并均值。

当前 decode 5/5 成功，累计处理 2560 input tokens 与 2560 output tokens；mean TTFT
为 `70.106 ms`、mean TPOT 为 `23.386 ms`、TPOT std 为 `0.071 ms`、TPOT CV 为
`0.303%`，mean ITL 为 `23.432 ms`。本轮 client 结果文件未包含 E2EL percentile 字段，
因此不补写该值；M2 decode 门槛使用实际存在且预先确定的 TPOT。两项主要性能回退均
低于 10%，单请求 M2 门槛通过。

服务于 UTC `15:00:13` 正常停止，容器 `exit=0`、`OOMKilled=false`，日志未检出
traceback、RuntimeError、CUDA error、assertion error 或 OOM；停止后 GPU 0 为
`0 MiB / 183359 MiB`、利用率 0%，无计算进程。当前逐请求结果、合并统计、结构化
新旧对比、benchmark/server 日志、容器 inspect、软件版本和完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/performance_m2/`。阶段 6 继续执行
batch 8/48、prefix-hit TTFT 和 mixed-prefill workspace 验证。

### 10.4 Batch 8/48 Decode 吞吐扩展

为排除不同 pool reserve 对吞吐对比的影响，三个 concurrency 点共用同一个
`max_num_seqs=48` 服务。配置为当前里程碑镜像、Qwen3-4B-Instruct-2507、BF16、
OSCAR INT2、精确 10 GiB、`max_model_len=8192`、eager，并关闭 chunked prefill 与
prefix caching。实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefix-cache-20260720` / `sha256:e0545221...1d95e79` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

GPU 0 在 UTC `15:00:21` 与 `15:02:15` 两次检查中均为 0 MiB、利用率 0%、无计算
进程。服务 UTC `15:04:39` health ready；日志实际规划 INT2 history 367728 tokens、
BF16 prefix 3072 tokens、BF16 recent 12288 tokens，并启用 async scheduling。

每个请求固定 512 input / 512 output、`ignore_eos`、seed 44。concurrency 1/8/48
分别先 warm-up 2/8/48 个请求，正式阶段发送 5/40/240 个请求，即每个并发 lane 处理
5 个正式请求。官方 client 的 aggregate 结果如下：

| Concurrency | 成功 / 失败 | Duration | Output throughput | 相对 c1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5 / 0 | 60.918 s | 42.023 tok/s | 1.000x |
| 8 | 40 / 0 | 63.568 s | 322.175 tok/s | 7.667x |
| 48 | 240 / 0 | 66.242 s | 1855.024 tok/s | 44.143x |

为落实每点至少 5 次的波动要求，利用详细结果中的 request start、TTFT 和逐 token
ITL，将每点正式请求按 client 发出顺序划为 5 个等大波次，并以各波最早 start 到
最晚最后 token 的区间计算 output throughput。五波均值与 population CV 为：

| Concurrency | 五波 mean output throughput | Std | CV | 相对 c1 mean |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 42.028 tok/s | 0.373 | 0.888% | 1.000x |
| 8 | 322.113 tok/s | 2.264 | 0.703% | 7.664x |
| 48 | 1836.699 tok/s | 17.689 | 0.963% | 43.702x |

三个点 CV 均低于 3%，无需追加采样。结合第 7.1 节已验证的 batched store/demote/
decode 单次 kernel ABI、请求顺序与 HP row 解耦 oracle，以及本次 c8/c48 吞吐分别达到
c1 的 7.66x/43.70x，batch 路径没有表现出逐请求串行 launch，M2 batch 门槛通过。

服务于 UTC `15:11:14` 正常停止，容器 `exit=0`、`OOMKilled=false`，日志未检出
traceback、RuntimeError、CUDA error、assertion error 或 OOM；停止后 GPU 0 为
0 MiB、利用率 0%，无计算进程。原始详细结果、五波派生统计、服务/benchmark 日志、
容器 inspect、版本和完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/performance_m2/batch_scaling/`。阶段 6
还需完成 prefix-hit TTFT 与 mixed-prefill workspace 验证。

### 10.5 Prefix-Hit TTFT 首轮失败

本轮使用当前里程碑镜像、Qwen3-4B-Instruct-2507、BF16、OSCAR INT2、精确
10 GiB、`max_num_seqs=2`、`max_model_len=8192`、eager，同时启用 prefix caching
与 chunked prefill。实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefix-cache-20260720` / `sha256:e0545221...1d95e79` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

GPU 0 在 UTC `15:11:25` 与 `15:14:12` 两次检查中均为空闲。服务 UTC
`15:17:16` health ready；OSCAR prefix cache 组合按设计关闭 async scheduling，日志
实际规划 INT2 history 461936 tokens、BF16 prefix 128 tokens、BF16 recent 512 tokens。

实验编排保留了三次无效尝试，但不将其混入性能结论：第一次因未启用 dev mode，cache
reset endpoint 返回 HTTP 404；第二次因脚本把 SSE 终止标记 `[DONE]` 交给 `jq` 解析而
退出；第三次虽然 5/5 请求成功，但 `curl time_starttransfer` 测到的是 SSE 响应头，不是
首个生成 token。随后改用镜像内 Python streaming client，以单调时钟记录第一个非空
completion token SSE event，并从最终 usage 读取 `cached_tokens`；该 client 已通过
`py_compile`、Ruff lint 与 format check。

正式数据中第五组 `upsilon` 经 tokenizer 实测为 4200 tokens，因此从等长比较中排除。
其余四组均为 2100-token prompt，每组先冷请求，再发相同 prefix-hit 请求：

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 4 | 4 |
| Mean TTFT | 52.587 ms | 189.157 ms |
| Population std | 2.892 ms | 0.758 ms |
| CV | 5.500% | 0.401% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

Prefix hit 将本地 prefill token 数减少 `87.619%`，但 mean TTFT 反而增加
`259.706%`，即 hit 为 cold 的约 `3.597x`。因此本轮 prefix-hit TTFT 门槛明确失败，
不能把 KV token 减少等同于端到端延迟改善。cold 侧 CV 超过 3%，且等长样本只有 4 组；
但 hit 的四个样本稳定在 188--190 ms，失败幅度远大于波动。后续应先修复实现，再用至少
5 组等长样本重测，而不是对当前失败路径追加样本来弱化结论。

静态审计给出两个必须继续验证的实现问题。第一，cached prefill 当前把每个新 query
token 当作独立 decode query，使用 `BLOCK_KV=4`、`num_warps=1` 循环 1840-token
history；260 个 query token 无法复用 history tile，强烈吻合本轮反向加速。第二，kernel
以 `cached_len - recent_tokens` 推导 recent tier 起点，但 prefix-hit 新请求在命中边界前
没有 request-owned recent row；这可能错误读取新请求尚未写入的 BF16 recent 空间。
第二点在本轮服务结束时仍只是代码审计发现的正确性风险，不能仅凭一次输出认定不存在；
第 10.6 节随后以 poisoned recent row 的 CUDA oracle 建立了失败证据。修复仍须补齐共享
命中长度 metadata 与 tiled mixed-prefill kernel，并遵守“不把完整 history 解压成 BF16
K/V workspace”的边界。

服务于 UTC `15:25:25` 正常停止，容器 `exit=0`、`OOMKilled=false`，日志未检出服务
错误；UTC `15:25:35` GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。原始请求、有效
2100-token 汇总、Prometheus 前后计数、client/server 日志、容器 inspect、版本和完整
`nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/performance_m2/prefix_ttft/`。阶段 6 保持
进行中，prefix TTFT 当前未通过，mixed-prefill workspace 尚待验证。

### 10.6 Prefix-Hit Recent Ownership 失败 Oracle

为验证第 10.5 节的正确性风险，新增单请求 CUDA oracle：先物化 384-token 源请求的
INT2 history 与 64-token 共享 BF16 prefix page，再让新请求复用这些 cache block/page，
但分配不同的 HP row。新请求尚未计算任何自有 token，因此其 recent row 不应参与
attention；测试将该 row 全部填为 `1000`，并以“共享 prefix 使用 BF16、其余共享
history 使用已物化 INT2”的 full-dequant reference 对照。

本轮继续固定 GPU 0，环境与第 10.5 节相同。GPU 0 在 UTC `15:29:47` 与
`15:31:16` 两次检查中均为 0 MiB、利用率 0%、无计算进程。测试前目标文件通过
`py_compile`、Ruff lint 与 format check。第一次 CUDA 运行在 41.31 秒后失败，但仅因
测试把 FP32 output 与 FP16 reference 交给 `assert_close` 的 dtype 检查，不能作为语义
证据；将 reference 显式转为 FP32 后，在相同 GPU 上执行有效重跑。

有效重跑在 41.41 秒后得到 `1 failed, 15 deselected`。1024/1024 个输出元素全部不一致，
最大绝对误差为 `666.666748`，远高于 `atol=rtol=0.006`；最大相对误差为
`22550024`。该误差量级与 poisoned value row 被 256/384 个位置读取完全一致，证明当前
mapped cached-prefill kernel 确实把共享 history 尾部误判为新请求自有 BF16 recent。
因此第 9 阶段已经验证的 prefix cache 命中/生命周期功能并不等于数值语义完整，修复前
不能继续把阶段 5 的正确性视为完成状态。

所需边界为：对每个新请求持久记录初始共享命中长度 `shared_hit_tokens`，cached attention
读取 request recent 的起点必须是
`max(prefix_tokens, shared_hit_tokens, cached_len - recent_tokens)`。修复还必须覆盖后续
chunked/decode 推进：当 `cached_len` 超过初始共享边界后，只有本请求已经实际写入 recent
ring 的位置才允许回到 BF16 tier。

UTC `15:33:47` 检查确认 GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。测试源码、
静态检查、两次 CUDA 输出与 GPU 检查记录位于
`artifacts/oscar_vllm_optimization/20260720/prefix_recent_oracle/`。下一步先修复 metadata
链路和数值语义，再实现 query/history tiled kernel 并重测 TTFT。

### 10.7 Shared-Hit Metadata 与 Recent Ownership 修复

修复不修改通用 request 语义，而由 OSCAR allocator 在首次分配请求 HP row 时记录
`first_new_block * block_size`。该值恰好是当前请求已经附着的本地共享 cache hit；冷请求
为 0，preemption 后重新分配 row 时按重新命中的 blocks 重算。请求结束、分配失败回滚均
同步清除该状态。scheduler 以 `request_id -> shared_hit_tokens` 输出，V1/V2 runner、
dummy/warmup、`CommonAttentionMetadata` 与 `OscarMetadata` 全链路透传。

mixed attention 与 recent demotion 现在共用相同的 request-owned recent 起点：
`max(prefix_tokens, shared_hit_tokens, cached_len - recent_tokens)`。因此初次 prefix-hit
prefill 不会从新请求空 recent row 读取共享 history 尾部，也不会在 chunk 结束时把该空
row 再量化并覆盖已物化 INT2 history；只有请求自身实际计算的 token 离开 recent window
后才会 demote。

修复前新增的五项 CPU metadata 合约实际为 `4 failed, 1 passed`：allocator getter、
dummy batch、synthetic warmup 和 unpadded common metadata 均明确缺少共享长度。修复后
相同定向集合为 `5 passed, 21 deselected in 1.22s`。16 个相关 Python 文件随后全部通过
`py_compile`、Ruff lint 与 format check。

本轮继续使用第 10.5 节的软件/GPU 环境和里程碑镜像，并将当前 Python 源覆盖到镜像内
site-packages，保留已编译扩展。GPU 0 在 UTC `15:39:47` 与 `15:41:31` 两次检查中均
为空闲，固定 `CUDA_VISIBLE_DEVICES=0`。首次 GPU 命令因工作目录 `/src` 遮蔽镜像内
`_C_stable_libtorch` 而在收集阶段退出，不计为测试结果；改从 `/tmp` 导入 installed
package 后得到以下有效结果：

| 验证 | 实际结果 |
| --- | --- |
| poisoned recent read + demote oracle | `1 passed, 15 deselected in 43.23s` |
| 完整 OSCAR CUDA 文件 | `16 passed, 75 warnings in 72.74s` |
| 完整 OSCAR core 生命周期文件 | `26 passed, 17 warnings in 51.98s` |

poisoned oracle 除原有 attention reference 外，还在 384-token 共享命中后模拟 17-token
chunk，并断言 demotion 前后整个 INT2 cache byte-exact 不变。该断言通过，证明空 recent
row 既未参与 attention，也未写回 quant history。完整 CUDA/core 回归同时覆盖既有
store/decode、mapped query、chunked tier、page table、preemption 与 reuse，未出现回退。

UTC `15:46:59` GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。静态检查、CPU 前后
对照、CUDA/core 日志和 GPU 检查位于
`artifacts/oscar_vllm_optimization/20260720/prefix_recent_fix/`。本节证明源码级数值修复
通过；尚未重建服务镜像，也尚未完成真实 prefix-hit 响应复测，因此阶段 5 保持进行中。

### 10.8 Recent Ownership 修复后的真实服务复测

基于当前源码构建不可变候选镜像 `oscar-vllm:prefix-recent-fix-20260720`，镜像 ID 为
`sha256:712088047e9381b9c06a3c28aa30d82fa1bf841212730a4fdf9a56a65d33388d`。服务继续
使用 Qwen3-4B-Instruct-2507、BF16、OSCAR INT2、精确 10 GiB、maxseq2、eager，
同时启用 prefix caching、chunked prefill 和 prompt token details。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefix-recent-fix-20260720` / `sha256:71208804...5d33388d` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

GPU 0 在 UTC `15:46:59` 与 `15:50:08` 两次检查中均为空闲。第一次服务启动遗漏
rotation 目录和 `VLLM_OSCAR_{K,V}_ROTATION_PATH`，在 engine 创建前的配置校验中退出码
1、非 OOM，没有发送请求；该运行仅作为无效编排保留。补齐与既有服务相同的 rotation
挂载后，新容器 UTC `15:54:46` health ready。日志确认 async scheduling 关闭，实际规划
INT2 history 461936、BF16 prefix 128、BF16 recent 512 tokens。

首先以 streaming client 发送一个 sigma 2100-token cold/repeat pair，结果均 HTTP 200，
cold 为 0 cached tokens，repeat 为 1840 cached tokens，命中率 `87.619%`。该 pair 同时
触发新 Triton 签名的首次编译，TTFT 分别为 2665.856/7172.831 ms，只用于功能验证，
不得作为稳态性能结果。

随后发送一个独立 rho 2100-token cold/repeat pair，每次生成 16 tokens、temperature 0：

| 请求 | Cached tokens | Completion | 结果 |
| --- | ---: | ---: | --- |
| rho cold | 0 / 2100 | 16 | HTTP 200 |
| rho repeat | 1856 / 2100 | 16 | HTTP 200 |

两次完整生成文本逐字一致。repeat 本地 prefill 仅 244 tokens，cache hit 为
`88.381%`；结合第 10.7 节 poisoned numerical oracle，这证明真实 scheduler 产生的共享
长度已通过新镜像到达 worker，并且修复后的服务仍能共享 prefix/INT2 history。这里不把
离散 token 文本相等单独当作数值证明，数值正确性仍由 CUDA full-dequant oracle 支撑。

服务于 UTC `15:56:17` 停止，容器最终 `exit=0`、`OOMKilled=false`，日志未检出
traceback、RuntimeError、CUDA error、assertion error 或 OOM；UTC `15:56:25` 与
`15:56:49` 检查均显示 GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。镜像构建日志、
请求/usage、完整服务日志、容器 inspect、Python/包版本和完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/prefix_recent_service/`。阶段 5 的修复后
源码、CUDA/core 和真实服务证据均已闭环，重新标记完成；阶段 6 的 prefix TTFT 性能
仍未通过，下一步继续实现 tiled cached-prefill kernel。

### 10.9 Tiled Cached-Prefill CUDA Oracle

原 cached-prefill 路径对每个 query token、每个 query head 独立运行 decode-oriented
kernel，并以 `BLOCK_KV=4` 重复遍历共享 history。新实现仅替换 cached-prefill：Triton
grid 按 request、query head 和 16-token query tile 组织，每个 tile 以 16-token K/V tile
循环 history，INT2 K/V 在片上解量化后由 `tl.dot` 完成 score 与 value accumulation。
decode 继续使用既有 split-KV kernel。

该 kernel 直接读取 scheduler block table、共享 BF16 prefix pages、请求 recent row 和
`shared_hit_tokens`，用 online softmax 输出当前 query tile 的 cached attention 与 LSE，
再与当前 BF16 suffix FlashAttention 结果合并。其 workspace 只有 query-sized FP32 output、
LSE 和旋转 query，不包含 history-sized BF16 K/V，仍满足禁止完整历史解压的硬边界。

三个相关文件通过 `py_compile`、Ruff lint 与 format check。GPU 0 在 UTC
`15:56:49` 与 `16:03:10` 两次检查中均为空闲；实验固定 `CUDA_VISIBLE_DEVICES=0`，
Docker、GPU、driver、Python 与包版本同第 10.8 节。先运行 poisoned prefix-hit 与
mixed cached-prefill 两项定向 oracle，结果为 `2 passed, 14 deselected in 49.50s`；随后
完整 OSCAR CUDA 文件为 `16 passed, 61 warnings in 80.41s`。

新增 poisoned 用例同时直接调用 tiled kernel，并与“共享 prefix BF16 + 共享 history
INT2”的 full-dequant reference 比较，容差为 `atol=rtol=0.015`；既有 mixed backend
用例继续覆盖 cached request 与 cold request 同 batch、cached/current LSE 合并和最终
tier 更新。完整回归通过说明新 kernel 的数值、GQA/page-table 寻址和其他 store/decode
路径没有出现已知回退。

UTC `16:06:30` GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。源码静态检查、定向与
完整 CUDA 日志、GPU 检查位于
`artifacts/oscar_vllm_optimization/20260720/tiled_prefill/`。本节只证明 CUDA 正确性；
真实 TTFT 是否改善必须由新镜像 warm-up 后的至少 5 组服务样本判定。

### 10.10 16x16 Tiled Prefix-Hit TTFT 首轮

基于第 10.9 节源码构建镜像 `oscar-vllm:tiled-prefill-20260720`，镜像 ID 为
`sha256:aef03b66621dbc3631e6410998cf65bfc28a09800ded0e41c28a62180e9aeae9`。
服务配置保持 Qwen3-4B-Instruct-2507、BF16、OSCAR INT2、精确 10 GiB、maxseq2、
eager，并同时启用 prefix caching、chunked prefill 和 prompt token details。GPU 0 在
UTC `16:06:30` 与 `16:08:25` 两次检查中均为空闲，本轮固定使用
`CUDA_VISIBLE_DEVICES=0`。完整环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:tiled-prefill-20260720` / `sha256:aef03b66...e9aeae9` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

服务于 UTC `16:11:05` health ready。日志确认 prefix/chunked 均启用，实际规划 INT2
history 461936、BF16 prefix 128、BF16 recent 512 tokens；OSCAR prefix-cache 契约下
未启用 async scheduling。性能客户端先以 epsilon 2100 tokens 和 dynamo 4200 tokens
完成两次 warm-up，其中 epsilon 已覆盖正式样本的精确长度。正式 prompt 使用 vector、
matrix、tensor、scalar、kernel 五个不同前导词，均在实验前经同一 tokenizer 验证为
2100 tokens，避免跨 pair 的缓存污染和长度混杂。

10 个正式请求均为 HTTP 200。每个 cold 请求均为 0 cached tokens，对应 repeat 均命中
1840/2100 tokens，本地实际 prefill 从 2100 降至 260 tokens，减少 `87.619%`：

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 5 | 5 |
| Mean TTFT | 48.129 ms | 70.506 ms |
| Population std | 1.429 ms | 0.328 ms |
| CV | 2.968% | 0.465% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

相对第 10.5 节 decode-oriented 命中路径的 189.157 ms，16x16 tiled kernel 将 hit mean
TTFT 降低 `62.726%`，说明 query/history tiling 的优化方向有效。但本轮 hit 仍比同轮
cold 慢 `46.496%`，因此“共享前缀命中率超过 75% 时 TTFT 有统计改善”的 M2 门槛仍然
失败。两侧 CV 均不超过 3%，无需因波动追加样本；也不能只引用相对旧失败实现的改善而
宣称性能门槛通过。

服务于 UTC `16:12:07` 正常停止，容器最终 `exit=0`、`OOMKilled=false`；UTC
`16:12:15` GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。逐请求结果、汇总统计、
client/server 日志、容器 inspect、GPU 检查与完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/tiled_prefill_ttft/`。阶段 6 保持进行中，
下一步扩大 query/KV tile 并用相同 workload 重测，而不是接受当前失败结果。

### 10.11 32x32 Tile 定向 CUDA 验证

为单独评估 tile 尺寸，将 cached-prefill 的 query/KV tile 从 16x16 扩大为 32x32，
`num_warps=4`、`num_stages=2` 及其余 kernel 逻辑保持不变。该修改不增加 history-sized
workspace，输出、LSE 和旋转 query 仍只随本次 query 数增长。目标文件通过容器内
`py_compile`、Ruff lint 与 format check。

本轮继续使用 `oscar-vllm:tiled-prefill-20260720` 作为已编译扩展底座，并将当前 Python
源码覆盖至容器 site-packages。环境与第 10.10 节相同：固定
`CUDA_VISIBLE_DEVICES=0`，NVIDIA B200 183359 MiB、Driver 595.71.05、
`nvidia-smi` CUDA 13.2、Python 3.12.13、vLLM 0.25.0、PyTorch 2.11.0+cu130、
PyTorch CUDA 13.0、Triton 3.6.0、Transformers 5.13.0。GPU 0 在 UTC
`16:16:17` 与 `16:17:39` 两次检查中均为 0 MiB、利用率 0%、无计算进程。

poisoned prefix-hit 与 mixed cached-prefill 两项定向 oracle 实际结果为
`2 passed, 14 deselected, 61 warnings in 64.34s`。前者继续验证 32x32 kernel 不读取
新请求未拥有的 recent row，并与 full-dequant reference 比较；后者继续覆盖
cached/current attention 的 LSE 合并与 tier 更新。pytest 独立退出码为 0，说明该 tile
尺寸可在 B200 上编译执行且定向数值语义未出现已知回退。

测试载体使用 `sleep infinity` 保持容器供源码覆盖和 `docker exec` 使用，测试结束后由
`docker stop` 主动终止，因此容器最终 `exit=143`、`OOMKilled=false`；该停止码不是
pytest 结果。UTC `16:20:04` GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。静态检查、
两次 GPU 检查、pytest 日志/退出码和容器 inspect 位于
`artifacts/oscar_vllm_optimization/20260720/tiled_prefill_32x32/`。本节只允许进入真实
服务性能验证，不代表 32x32 已通过完整 CUDA 回归或 M2 TTFT 门槛。

### 10.12 32x32 Tiled Prefix-Hit TTFT

当前源码构建为 `oscar-vllm:tiled-prefill-32x32-20260720`，当前 tag 的完整镜像 ID 为
`sha256:c03b8f720155489fb799763eaee055bafd093e29c8447aa887c24a7a2836704d`。
服务继续使用与第 10.10 节相同的 Qwen3-4B-Instruct-2507、BF16、OSCAR INT2、精确
10 GiB、maxseq2、eager、prefix caching、chunked prefill 和 prompt token details。
环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:tiled-prefill-32x32-20260720` / `sha256:c03b8f72...2836704d` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

GPU 0 在 UTC `16:20:04` 与 `16:22:29` 两次检查中均为空闲，固定
`CUDA_VISIBLE_DEVICES=0`。服务于 UTC `16:25:25` health ready；日志确认
prefix/chunked 均启用，OSCAR 三段式仍为 INT2 history 461936、BF16 prefix 128、BF16
recent 512 tokens。首轮完全复用第 10.10 节客户端和参数：epsilon 2100 tokens、dynamo
4200 tokens 两次 warm-up 后，运行 5 组 2100-token cold/hit pair。所有请求均 HTTP
200，cold 均为 0 cached tokens，hit 均为 1840/2100。

首轮 cold/hit mean TTFT 为 45.983/60.457 ms，hit 慢 `31.478%`；hit CV 仅
0.464%，但 cold 首个 52.364 ms 样本使其 CV 为 7.010%。因此按门槛保留首轮原值并继续
采样。经同一 tokenizer 实测确认后，追加 memory、attention、model、data、compute、
graph、layer、hidden、query、value 十个互异 2100-token prompt；服务已经预热，追加
客户端不再另设 warm-up。追加 10 组仍全部 HTTP 200、cold=0、hit=1840 cached tokens。

15 组合并结果如下，不删除两次客户端批次的首个高 cold 样本：

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 15 | 15 |
| Mean TTFT | 47.101 ms | 61.248 ms |
| Population std | 5.151 ms | 1.062 ms |
| CV | 10.936% | 1.734% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

cold 波动集中在两个客户端批次的首个请求，分别为 52.364 和 64.536 ms，其余样本主要
位于 44--49 ms；追加采样后 CV 仍超过 3%，因此不能声称 cold 均值已经稳定。配对统计
仍给出清晰的失败边界：15 组中 14 组 hit 更慢，`hit - cold` mean 为 +14.147 ms、
population std 为 4.551 ms；唯一 hit 更快的 pair 正是 cold=64.536 ms 的批次首请求，
差值仅 -0.860 ms。合并均值下 hit 比 cold 慢 `30.035%`，并非 M2 要求的统计改善。

32x32 hit mean 相对 16x16 的 70.506 ms 继续降低 `13.132%`，相对第 10.5 节旧路径的
189.157 ms 降低 `67.621%`，说明扩大 tile 继续有效；但该相对改善不能覆盖 M2 仍失败
的事实。服务于 UTC `16:29:15` 正常退出 `0`、`OOMKilled=false`，GPU 0 同时回收为
0 MiB、利用率 0%、无计算进程。最初宽泛日志扫描仅命中 vLLM 配置提示中的
`If OOM'ed` 文本；收紧为 traceback、运行时/CUDA/assertion/实际 OOM 错误后无命中。

构建日志、镜像 ID、两轮原始请求、合并与配对统计、client/server 日志、容器 inspect、
版本、GPU 检查和完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/tiled_prefill_32x32_ttft/`，定向 CUDA 证据
位于相邻 `tiled_prefill_32x32/`。阶段 6 继续进行；32x32 尚不值得执行完整 CUDA 回归，
下一步应先继续降低 cached-prefill 的稳定命中成本。

### 10.13 64x64 Tile 定向 CUDA 验证

根据 32x32 的稳定 hit 成本，将 cached-prefill query/KV tile 同时扩大至 64x64；
`num_warps=4`、`num_stages=2` 和其他逻辑不变。对本轮 260-query、1840-history 的
命中形状，每个 query head 的 program 数由 9 降至 5，history 循环次数由 58 降至 29。
这是执行形状推导，不作为实际加速结果；更大 accumulator 可能增加寄存器压力，仍需
真实服务验证。

目标文件通过容器内 `py_compile`、Ruff lint 与 format check。GPU 0 在 UTC
`16:29:15` 与 `16:32:27` 两次检查中均为 0 MiB、利用率 0%、无计算进程。实验固定
`CUDA_VISIBLE_DEVICES=0`，继续使用第 10.12 节镜像作为编译扩展底座并覆盖当前 Python
源码；GPU、driver、Python 和包版本均与第 10.12 节相同。

poisoned prefix-hit 与 mixed cached-prefill 定向 oracle 为
`2 passed, 14 deselected, 61 warnings in 62.47s`，pytest 退出码 0。64x64 tile 可在
B200 上编译执行，且当前 full-dequant poisoned reference、shared-hit ownership、
cached/current LSE 合并与 tier 更新未出现已知数值回退。测试载体由 `docker stop`
主动终止，因此容器 `exit=143`、`OOMKilled=false`，不代表 pytest 失败。

UTC `16:34:49` GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。静态检查、GPU 检查、
pytest 日志/退出码和容器 inspect 位于
`artifacts/oscar_vllm_optimization/20260720/tiled_prefill_64x64/`。本节同样只允许该
候选进入服务 TTFT 测试，不代表完整 CUDA 回归或 M2 已通过。

### 10.14 64x64 Tiled Prefix-Hit TTFT

候选镜像 `oscar-vllm:tiled-prefill-64x64-20260720` 的完整 ID 为
`sha256:aef83df5270685afaf8645873813d24723bcef57150fb70baeb7fb037162cf35`。
服务、模型和软件环境与第 10.12 节完全相同，固定 `CUDA_VISIBLE_DEVICES=0`；GPU 0
在 UTC `16:34:49` 与 `16:36:37` 两次检查中均为空闲。服务 UTC `16:39:39` health
ready，日志确认 prefix/chunked 启用，三段式仍为 INT2 history 461936、BF16 prefix 128、
BF16 recent 512 tokens。

先按相同的两次 warm-up 和五组正式 pair 运行。所有请求均 HTTP 200，cold 均为 0、
hit 均为 1840/2100 cached tokens；首轮 cold/hit mean 为 46.526/61.790 ms，但 CV
分别为 3.288%/4.101%，均超过 3%。因此继续以第 10.12 节相同的十个 tokenizer 验证
prompt 追加采样，并保留首轮全部原值。15 组合并如下：

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 15 | 15 |
| Mean TTFT | 47.449 ms | 62.358 ms |
| Population std | 3.868 ms | 2.079 ms |
| CV | 8.152% | 3.335% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

追加后 cold 与 hit CV 仍分别高于 3%，不能声称两侧均值完全稳定；但 15/15 组 pair 的
hit 都更慢，`hit - cold` mean 为 +14.908 ms、population std 为 3.921 ms。合并均值下
hit 比 cold 慢 `31.420%`，M2 明确失败。更重要的是，64x64 hit mean 相对 32x32 的
61.248 ms 回退 `1.812%`。因此同时扩大 query 与 history tile 没有带来收益，实际结果
支持“大 query accumulator 的寄存器/occupancy 代价抵消循环减少”这一解释，但没有
profiling 证据时不把该解释写成已证明根因。

服务 UTC `16:41:22` 正常退出 0、`OOMKilled=false`，日志未检出 traceback、运行时/
CUDA/assertion 或实际 OOM 错误；UTC `16:41:23` GPU 0 已回收为 0 MiB、利用率 0%、
无计算进程。构建、原始/合并/配对数据、日志、inspect、GPU 检查与完整 `nvidia-smi`
位于 `artifacts/oscar_vllm_optimization/20260720/tiled_prefill_64x64_ttft/`，定向 CUDA
位于相邻 `tiled_prefill_64x64/`。64x64 候选淘汰；下一步改为 32x64，只扩大 history
tile，保留 32-query accumulator。

### 10.15 32x64 Tile 定向 CUDA 验证

为隔离 64x64 的两个变量，本候选把 query tile 恢复为 32，仅保留 64-token history
tile；`num_warps=4`、`num_stages=2` 与其余逻辑不变。对当前命中形状，program/head
保持 9，history loop 从 32x32 的 58 降至 29，同时避免 64-query accumulator。

目标文件通过容器内 `py_compile`、Ruff lint 与 format check。GPU 0 在 UTC
`16:41:23` 与 `16:43:25` 两次检查中均为空闲，实验固定
`CUDA_VISIBLE_DEVICES=0`；环境与第 10.14 节相同。poisoned prefix-hit 与 mixed
cached-prefill 定向 oracle 为 `2 passed, 14 deselected, 61 warnings in 67.62s`，
pytest 退出码 0。该总时长包含导入、FlashAttention 初始化和 Triton 冷编译，不能用来
排序 tile 性能；有效结论仅是 32x64 可编译执行且定向数值语义未出现已知回退。

测试载体由 `docker stop` 主动终止，容器 `exit=143`、`OOMKilled=false`；UTC
`16:45:06` GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。静态检查、GPU 检查、
pytest 与 inspect 位于
`artifacts/oscar_vllm_optimization/20260720/tiled_prefill_32x64/`。下一步构建独立
镜像并使用相同 TTFT workload；本节不代表完整 CUDA 回归或 M2 通过。

### 10.16 32x64 Tiled Prefix-Hit TTFT

候选镜像 `oscar-vllm:tiled-prefill-32x64-20260720` 的完整 ID 为
`sha256:7b8068c5ea2fa09900bd82d873a04502d814fe4a251ec39297b1a6dcd2aa9529`。
服务与第 10.14 节配置、模型和软件环境相同，固定 `CUDA_VISIBLE_DEVICES=0`；GPU 0
在 UTC `16:45:06` 与 `16:47:19` 两次检查中均为空闲。服务 UTC `16:50:38` health
ready，日志确认 prefix/chunked 启用，三段式仍为 461936/128/512 tokens。

两次 warm-up 后的 5 组 cold/hit pair 全部 HTTP 200，cold 均为 0、hit 均为
1840/2100 cached tokens。结果如下：

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 5 | 5 |
| Mean TTFT | 49.086 ms | 65.546 ms |
| Population std | 0.434 ms | 0.392 ms |
| CV | 0.885% | 0.599% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

两侧 CV 均低于 3%，无需追加采样。hit 比 cold 慢 `33.534%`，M2 失败；相对 32x32
合并 hit mean 61.248 ms 又回退 `7.017%`。因此单独扩大 history tile 不但没有兑现
循环减半收益，反而稳定变慢；32x64 候选淘汰，32x32 仍是当前最佳已测 tile。该结果
本身不能区分内存访问、`tl.dot` 形状或 occupancy 的具体影响，后续若需根因应使用
kernel 级 profiling，而不是继续从循环次数推断。

服务 UTC `16:51:37` 正常退出 0、`OOMKilled=false`，日志未检出 traceback、运行时/
CUDA/assertion 或实际 OOM 错误；UTC `16:51:38` GPU 0 已回收为 0 MiB、利用率 0%、
无计算进程。构建、原始数据、日志、inspect、GPU 检查与完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/tiled_prefill_32x64_ttft/`，定向 CUDA
位于相邻 `tiled_prefill_32x64/`。下一步测试 64x32，只减少 query program 数并保留
当前表现更好的 32-token history tile。

### 10.17 64x32 Tile 定向 CUDA 验证

本候选仅将 query tile 从 32 扩至 64，history tile 保持当前实测更好的 32；对
260-query 形状，program/head 从 9 降至 5，history loop 仍为 58。目标文件通过容器内
`py_compile`、Ruff lint 与 format check。

GPU 0 在 UTC `16:51:38` 与 `16:53:26` 两次检查中均为空闲，固定
`CUDA_VISIBLE_DEVICES=0`；环境与第 10.16 节相同。poisoned prefix-hit 与 mixed
cached-prefill 定向 oracle 为 `2 passed, 14 deselected, 61 warnings in 50.63s`，
pytest 退出码 0。与其他 tile 一样，该墙钟包含冷编译，不作为性能数据；有效结论仅是
64x32 可编译执行，且定向 shared-hit 数值、LSE 合并与 tier 更新未出现已知回退。

测试载体由 `docker stop` 主动终止，容器 `exit=143`、`OOMKilled=false`；UTC
`16:55:04` GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。静态检查、GPU 检查、
pytest 与 inspect 位于
`artifacts/oscar_vllm_optimization/20260720/tiled_prefill_64x32/`。下一步用相同服务
TTFT workload 决定是否保留；本节不代表完整 CUDA 回归或 M2 通过。

### 10.18 64x32 Tiled Prefix-Hit TTFT

候选镜像 `oscar-vllm:tiled-prefill-64x32-20260720` 的完整 ID 为
`sha256:31f4204fb3a2fb339a804a16c68b21312eada7a1527bf49efbaee9698eceea56`。
服务与第 10.16 节配置、模型和软件环境相同，固定 `CUDA_VISIBLE_DEVICES=0`；GPU 0
在 UTC `16:55:04` 与 `16:57:09` 两次检查中均为空闲。服务 UTC `17:00:16` health
ready，日志确认 prefix/chunked 启用，三段式仍为 461936/128/512 tokens。

两次 warm-up 后的 5 组 cold/hit pair 全部 HTTP 200，cold 均为 0、hit 均为
1840/2100 cached tokens：

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 5 | 5 |
| Mean TTFT | 46.889 ms | 63.689 ms |
| Population std | 0.654 ms | 0.668 ms |
| CV | 1.395% | 1.049% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

两侧 CV 均低于 3%，无需追加。hit 比 cold 慢 `35.831%`，相对 32x32 的 61.248 ms
回退 `3.986%`。因此单独把 query tile 扩至 64 同样没有收益；64x32 淘汰，32x32 是
16x16、32x32、64x64、32x64、64x32 五个已测组合中的最佳值，但其 hit 仍比 cold 慢
30.035%，不能通过 M2。下一步不再继续枚举 tile，而应恢复 32x32 并采集 kernel 级证据，
再决定结构性优化。

服务 UTC `17:01:30` 正常退出 0、`OOMKilled=false`，日志未检出 traceback、运行时/
CUDA/assertion 或实际 OOM 错误；UTC `17:01:31` GPU 0 已回收为 0 MiB、利用率 0%、
无计算进程。构建、原始数据、日志、inspect、GPU 检查与完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/tiled_prefill_64x32_ttft/`，定向 CUDA
位于相邻 `tiled_prefill_64x32/`。

### 10.19 V 逆旋转组件 Microbenchmark

恢复 32x32 后对 cached-prefill 周边路径做静态分解。现有 V 逆旋转以
`(num_tokens, num_heads)` 为 grid；实际 260-query、32-head 形状会产生 8320 个
program，每个 program 加载 128x128 rotation 并以逐元素乘加/归约计算一个 head row，
之后结果又被转换为 BF16 参与 attention state 合并。为量化该项，新增独立 CUDA-event
microbenchmark，精确使用 260x32x128 输入；每条路径预热 50 次，再运行 5 批、每批
200 次，报告每次操作均值。

本轮使用 32x32 镜像 `oscar-vllm:tiled-prefill-32x32-20260720`，镜像 ID
`sha256:c03b8f720155489fb799763eaee055bafd093e29c8447aa887c24a7a2836704d`。
环境如下：

| 项目 | 实际值 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

GPU 0 在 UTC `17:01:31` 与 `17:05:45` 两次检查中均为空闲。三条路径均预分配输出，
并把实际后续需要的 BF16 转换计入计时：

| 路径 | Mean | Population std | CV | 相对现有加速 |
| --- | ---: | ---: | ---: | ---: |
| 现有 Triton inverse + FP32-to-BF16 | 0.048875 ms | 0.000163 ms | 0.334% | 1.000x |
| FP32 `mm` + FP32-to-BF16 | 0.016381 ms | 0.000188 ms | 1.150% | 2.984x |
| input cast + BF16 `mm` | 0.016564 ms | 0.000301 ms | 1.816% | 2.951x |

三个 CV 均低于 3%，结果说明现有逐 row inverse 确实低效，FP32 GEMM 是最小且最快的
替代候选。但单层差值只有 0.032494 ms；按 Qwen3-4B 的 36 层线性外推约 1.170 ms，
远小于第 10.12 节 32x32 的 +14.147 ms 平均配对差。该 1.170 ms 只是组件测量的线性
估算，不是服务 TTFT 实测收益；因此逆旋转应优化，但主 cached-attention 和其他周边
路径仍需继续分解。

microbenchmark 在 `py_compile` 后有效完成、Docker exit 0。前两次 Ruff 检查因项目
要求第三方与本地 import 紧邻而报 I001，第二次 shell 又未以 `set -e` 阻止 benchmark；
Ruff 精确修正仅删除空行，不改变已运行代码语义，修正后的脚本已通过 py_compile、lint
和 format check，故未重复 GPU 测量。UTC `17:07:22` GPU 0 已回收，UTC `17:09:42`
完整 `nvidia-smi` 显示所有 GPU 0 MiB、无进程。脚本、原始 5 批数据、日志、退出码、
GPU 检查和完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/inverse_rotation_microbench/`。

### 10.20 Cached-Prefill FP32 Inverse MM CUDA Oracle

cached-prefill 的 V 逆旋转改为对扁平化 `[num_tokens * num_heads, head_dim]` 输出执行
预分配 FP32 `torch.mm`；decode 继续使用原逐 row Triton kernel，cache layout、主
attention kernel 和外部 ABI 均不变。poisoned prefix-hit oracle 同时扩展为随机非单位
正交 V rotation，并以主 attention reference 右乘该 rotation 的 FP32 结果对照，避免
identity rotation 掩盖逆旋转错误。

两个修改文件通过容器内 `py_compile`、Ruff lint 与 format check。GPU 0 在 UTC
`17:07:22` 与 `17:11:21` 两次检查中均为空闲，固定 `CUDA_VISIBLE_DEVICES=0`；镜像、
GPU、driver、Python 与包版本同第 10.19 节。poisoned prefix-hit 与 mixed backend 两项
定向 CUDA 实际为 `2 passed, 14 deselected, 61 warnings in 69.59s`，pytest 退出码 0。
这同时覆盖非单位 V inverse、shared-hit recent ownership、cached/current LSE 合并与
tier 更新，未出现已知数值回退。

测试载体由 `docker stop` 主动终止，容器 `exit=143`、`OOMKilled=false`，不代表 pytest
失败；UTC `17:13:11` GPU 0 已回收为 0 MiB、利用率 0%、无计算进程。源码静态检查、
GPU 检查、pytest 日志/退出码和 inspect 位于
`artifacts/oscar_vllm_optimization/20260720/prefill_inverse_mm/`。本节只闭环实现级
正确性；第 10.19 节的组件收益是否进入真实 TTFT，仍须由新镜像服务实测。

### 10.21 FP32 Inverse MM Prefix-Hit TTFT

候选镜像 `oscar-vllm:prefill-inverse-mm-20260720` 的完整 ID 为
`sha256:2903c7c4b69a2d2555321886004af39bdd9862f14201983485d556aff5bf694b`。
服务配置、模型和软件环境与第 10.12 节相同，固定 `CUDA_VISIBLE_DEVICES=0`；GPU 0
在 UTC `17:13:11` 与 `17:14:52` 两次检查中均为空闲。服务 UTC `17:17:47` health
ready，prefix/chunked 启用，三段式仍为 461936/128/512 tokens。

首轮 5 组全部 HTTP 200、cold=0、hit=1840/2100 cached tokens；cold/hit mean 为
43.899/58.126 ms，但 CV 为 7.573%/3.502%，均超过 3%。因此追加第 10.12 节相同的
10 个 tokenizer 验证 prompt，并保留全部原值。15 组合并结果为：

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 15 | 15 |
| Mean TTFT | 46.427 ms | 61.478 ms |
| Population std | 5.408 ms | 4.769 ms |
| CV | 11.650% | 7.758% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

追加后两侧 CV 仍高，不能从跨服务均值确认约 1.17 ms 的组件线性估算是否落地。15/15
组 hit 都更慢，配对差 mean 为 +15.052 ms、population std 为 5.938 ms；hit 比 cold
慢 `32.420%`，M2 失败。相对原 32x32 服务的 61.248 ms，当前 hit mean 仅高
`0.376%`，该差异远小于本轮波动，不能解释为确定回退，也不能宣称优化成功。

FP32 inverse mm 保留在源码：组件 microbenchmark 有稳定 2.984x 证据、非单位 rotation
CUDA oracle 通过，且真实服务没有可确认回退；但它不作为 M2 进展计分。服务 UTC
`17:19:15` 正常退出 0、`OOMKilled=false`，日志未检出 traceback、运行时/CUDA/
assertion 或实际 OOM 错误；UTC `17:19:16` GPU 0 已回收为 0 MiB、利用率 0%、无计算
进程。构建、原始/合并/配对数据、日志、inspect、GPU 检查与完整 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/prefill_inverse_mm_ttft/`。下一步分解主
cached-attention、Q rotation 与 merge 的单层 GPU 时间。

### 10.22 Prefix-Hit Prefill 组件级耗时分解

为避免继续依据服务 TTFT 猜测瓶颈，本轮直接复现第 10.21 节的单层有效形状：
260 个 query tokens、1840 个 cached tokens、32 个 query heads、8 个 KV heads、
head dim 128、block size 16、prefix 64 和 recent 256。INT2、prefix 与 recent cache
均按生产 tensor shape 分配；block table、prefix page table 和 shared-hit metadata 也使用
真实命中边界。每项先预热 20 次，再执行 5 批、每批 100 次 CUDA Event 测量。

实验固定使用物理 GPU 0。UTC `17:24:03` 与 `17:25:23` 两次检查均显示 GPU 0 为
0 MiB、利用率 0%、无计算进程；实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:prefill-inverse-mm-20260720` |
| 镜像 ID | `sha256:2903c7c4b69a2d2555321886004af39bdd9862f14201983485d556aff5bf694b` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

实测结果如下：

| 独立测量路径 | Mean | Population std | CV |
| --- | ---: | ---: | ---: |
| Q rotation | 0.022043 ms | 0.007084 ms | 32.139% |
| Cached attention，不含 V inverse | 0.833402 ms | 0.007275 ms | 0.873% |
| Cached attention，含 V inverse | 0.835954 ms | 0.008562 ms | 1.024% |
| Output/LSE postprocess 与 merge | 0.059563 ms | 0.000236 ms | 0.396% |
| 完整 cached path | 0.860729 ms | 0.000421 ms | 0.049% |

Q rotation 首批为 0.036023 ms，后四批均值为 0.018548 ms，因此其五批均值波动过高，
不能作为稳定优化收益依据。其余主要测量 CV 均低于 3%。不含 inverse 的主 attention
为 0.833402 ms，数值上相当于完整路径均值的 96.825%，明确是下一步的首要优化对象；
postprocess/merge 单独只有 0.059563 ms。含/不含 inverse 的跨批均值只差
0.002552 ms，小于两组各自的批间标准差，不能用本轮结果进一步量化 inverse 收益；
第 10.19 节专门 microbenchmark 仍是该组件更可靠的证据。

这些项目是分别运行的独立测量，受 allocator reuse、cache 和 GPU 时钟状态影响，不能
把表中均值直接相加当作完整路径。完整路径自身的稳定实测为每层 0.860729 ms，按
Qwen3-4B 的 36 层线性外推为 30.986249 ms。该值不是服务 TTFT，也不能直接等同于
第 10.21 节 +15.052 ms 的 cold/hit 配对差；cold 路径同时有自身 attention 工作，服务
还包含调度、模型其他算子和网络开销。但二者处于同一量级，足以否定“只优化约微秒级
外围操作即可通过 M2”的假设，并把后续工作收敛到 cached-attention 主 kernel。

benchmark 进程退出码 0；UTC `17:27:08` GPU 0 已回收为 0 MiB、无计算进程。脚本在
GPU 运行前通过 `py_compile`、Ruff lint 与 format check；首次 format check 仅要求四处
机械换行，修正后完整门禁通过。脚本、5 批原始数据、环境、前后 GPU 检查和日志位于
`artifacts/oscar_vllm_optimization/20260720/prefill_component_microbench/`。阶段 6 的
prefix TTFT 仍未通过；下一步针对 32x32 主 kernel 做配置级测量与结构性优化。

### 10.23 32x32 Kernel Launch 配置筛选

本轮不修改生产源码，直接调用同一个 32x32 cached-attention Triton kernel，交叉测试
`num_warps={2,4,8}` 与 `num_stages={1,2,3}`。shape、cache tensors、block/page tables
和 shared-hit metadata 与第 10.22 节相同。9 个配置各自先完成编译和 20 次预热，随后
执行 5 批、每批 100 次 CUDA Event 测量；每批轮换配置执行顺序，降低固定顺序与 GPU
状态漂移的混淆。所有配置的 output 与 LSE 均和当前 `4 warps / 2 stages` 在
`atol=rtol=5e-4` 下通过。

GPU 0 在 UTC `17:31:44` 与 `17:33:01` 两次检查中均为 0 MiB、利用率 0%、无计算
进程。本实验继续固定 `CUDA_VISIBLE_DEVICES=0`，环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:prefill-inverse-mm-20260720` |
| 镜像 ID | `sha256:2903c7c4b69a2d2555321886004af39bdd9862f14201983485d556aff5bf694b` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

| Warps | Stages=1 mean / CV | Stages=2 mean / CV | Stages=3 mean / CV |
| ---: | ---: | ---: | ---: |
| 2 | 0.644202 ms / 0.110% | **0.643783 ms / 0.022%** | 0.643882 ms / 0.018% |
| 4 | 0.818863 ms / 0.004% | 0.818955 ms / 0.010% | 0.818956 ms / 0.012% |
| 8 | 1.068355 ms / 0.005% | 1.068396 ms / 0.006% | 1.068446 ms / 0.005% |

最佳 `2 warps / 2 stages` 相对生产配置 `4 / 2` 加速 `1.272098x`，绝对减少
0.175172 ms/层；两者 CV 分别为 0.022% 和 0.010%。同一 warps 下三个 stages 的差距
最多约 0.00042 ms，而 warps 从 2 增至 4、8 时稳定变慢，因此本 shape 的收益明确来自
较少 warps，不应解释为 pipeline stages 优化。这里仍是单层 kernel 结果，不是服务
TTFT 收益；按 36 层线性外推的理论差约 6.306 ms，只有生产配置定向回归与真实服务
复测后才能判断落地程度。

benchmark 退出码 0；UTC `17:35:24` GPU 0 已回收为 0 MiB、无计算进程。脚本在 GPU
实验前通过 py_compile、Ruff lint 与 format check；首次 format check 只要求一处生成式
机械排版，修正后完整门禁通过。脚本、9 个配置的 45 组原始值、正确性标记、环境与
GPU 检查位于
`artifacts/oscar_vllm_optimization/20260720/prefill_kernel_config_microbench/`。
下一步仅把生产 kernel 的 `num_warps` 从 4 改为 2，保留 `num_stages=2`，再执行定向
poisoned/mixed CUDA oracle 和相同服务 TTFT workload。

### 10.24 `num_warps=2` 生产路径 CUDA Oracle

生产 wrapper 仅将 32x32 kernel launch 的 `num_warps` 从 4 改为 2，`num_stages=2`、
tile、kernel 语义和 ABI 均不变；目标源码与扩展后的 non-identity rotation 测试通过
py_compile、Ruff lint 与 format check。

GPU 0 在 UTC `17:37:31` 与 `17:38:50` 两次检查中均为空闲，本轮固定
`CUDA_VISIBLE_DEVICES=0`。使用第 10.23 节同一 Docker 镜像作为编译扩展底座，将当前
Python 源码覆盖进镜像已安装 package；GPU、driver、Python、vLLM、PyTorch、CUDA、
Triton 和 Transformers 版本均与第 10.23 节一致。使用全新
`TRITON_CACHE_DIR=/tmp/oscar-prefill-warps2-triton`，确保修改后的生产 launch 完成
冷编译。

poisoned prefix-hit 与 mixed cached/current backend 两项定向 CUDA 结果为
`2 passed, 14 deselected, 61 warnings in 57.59s`，pytest 退出码 0。前者覆盖新请求
不得读取未拥有 recent row、随机非单位 V inverse 与 full-dequant reference；后者覆盖
cached/current attention、LSE merge 和读后写 tier 更新。该结果证明 `2 warps` 生产
配置未引入当前 oracle 可见的数值或状态回退。

测试容器由 `docker stop` 主动终止，停止状态不是 pytest 结果；UTC `17:41:11` GPU 0
已回收为 0 MiB、无计算进程。源码静态检查、GPU 检查、pytest 日志/退出码和容器
inspect 位于 `artifacts/oscar_vllm_optimization/20260720/prefill_warps2/`。本节只完成
定向正确性门禁；下一步构建包含当前源码的独立镜像，并用相同 2100-token cold/hit
workload 验证第 10.23 节的 kernel 收益是否进入服务 TTFT。

### 10.25 `num_warps=2` Prefix-Hit TTFT 反证

当前源码构建为独立镜像 `oscar-vllm:prefill-warps2-20260720`，完整镜像 ID 为
`sha256:2028a0468889399fb66e540d3ebb35effde98ee45aa3113b2fb44d4f504dddc7`。
服务继续使用 Qwen3-4B-Instruct-2507、BF16、OSCAR INT2、精确 10 GiB、maxseq2、
eager、prefix caching、chunked prefill 和相同 K/V rotation；实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

GPU 0 在 UTC `17:43:39` 与 `17:44:56` 两次检查中均为空闲，服务于 UTC
`17:48:18` health ready。日志确认 prefix/chunked 启用、async scheduling 关闭，三段式
仍为 INT2 history 461936、BF16 prefix 128、BF16 recent 512 tokens。复用第 10.21 节
首轮完全相同的两组 warm-up、vector/matrix/tensor/scalar/kernel 五组正式 prompt 和
streaming client，所有请求均 HTTP 200，cold=0、hit=1840/2100 cached tokens。

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 5 | 5 |
| Mean TTFT | 47.408 ms | 62.627 ms |
| Population std | 0.850 ms | 0.501 ms |
| CV | 1.794% | 0.799% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

两侧 CV 均低于 3%，因此按既定规则不追加采样。hit 比 cold 慢 `32.104%`，五组配对
差均为正，`hit - cold` mean 为 +15.220 ms、population std 为 0.722 ms；M2 仍失败。
相对第 10.21 节同 prompt 首轮，当前 cold/hit 均值分别整体上移 7.991%/7.745%，幅度
接近；跨服务对比不能据此证明 `2 warps` 本身回退，但可以确定没有观察到第 10.23 节
线性估算的约 6.3 ms 服务收益。

回收前直接检查镜像内源码，实际记录为 `num_warps=2`，排除构建未包含修改。服务于
UTC `17:49:37` 正常退出 0、`OOMKilled=false`，日志未检出 traceback、运行时/CUDA/
assertion 或实际 OOM；UTC `17:49:48` 完整 `nvidia-smi` 显示所有 GPU 0 MiB、无进程。
构建日志、镜像/源码核对、逐请求结果、server/client 日志、inspect、版本和 GPU 检查
位于 `artifacts/oscar_vllm_optimization/20260720/prefill_warps2_ttft/`。

本结果是对零 cache 配置 microbenchmark 外推的直接反证：定向 kernel 基准与真实服务
之间仍有未解释变量，不能因为 microbenchmark 加速就保留生产配置。下一步使用合法
非零 INT2 scale/zero/data 和非零 prefix 数据复测 2/4 warps；若收益不再存在则恢复
4 warps，若仍存在则需要服务内 layer-level tracing 定位收益被何处抵消。

### 10.26 非零量化 Cache 的 Launch 配置复核

为检验第 10.23 节全零 cache 是否导致错误外推，配置脚本新增可选非零模式，默认零
模式保持不变。非零模式先用生产 `oscar_store` 将随机 BF16 K/V 量化为合法 INT2
data、FP32 scale/zero 和 packed layout，再填入非零 BF16 prefix/recent；其余 shape、
9 个 launch 配置、20 次预热、5 批 x 100 次交错测量及 output/LSE 对照均与第 10.23
节相同。扩展后脚本通过 pycompile、Ruff lint 与 format check。

GPU 0 在 UTC `17:52:50` 与 `17:54:02` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，固定 `CUDA_VISIBLE_DEVICES=0`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:prefill-warps2-20260720` |
| 镜像 ID | `sha256:2028a0468889399fb66e540d3ebb35effde98ee45aa3113b2fb44d4f504dddc7` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

9 个配置的 output/LSE 仍全部通过。与生产决策直接相关的两点为：

| 配置 | Mean | Population std | CV |
| --- | ---: | ---: | ---: |
| 2 warps / 2 stages | 0.643991 ms | 0.000310 ms | 0.048% |
| 4 warps / 2 stages | 0.822931 ms | 0.007632 ms | 0.927% |

2/2 相对 4/2 加速 `1.277860x`，绝对减少 0.178939 ms/层；两点 CV 均低于 3%。
4/2 的首批为 0.838193 ms，后四批位于 0.818987--0.819210 ms，即使排除首批也不会
改变 2 warps 明显更快的排序。其他配置仍呈现约 0.644/0.819/1.069 ms 的 2/4/8
warps 分层。因此全零 cache 不是第 10.23 节收益的原因，恢复 4 warps 没有 kernel
数据依据；当前保留 2 warps。

benchmark 退出码 0；UTC `17:56:15` GPU 0 已回收为 0 MiB、无计算进程。脚本、合法
非零 cache 的 45 组原始值、环境和 GPU 检查位于
`artifacts/oscar_vllm_optimization/20260720/prefill_kernel_nonzero_microbench/`。
结合第 10.25 节，现有证据只支持“kernel 独立加速真实存在，但服务关键路径未体现”，
不支持继续枚举 launch 配置。下一步需要在真实 prefix-hit 请求中采集 layer/kernel
时间线，确认实际调用次数、kernel duration 与未被覆盖的同步/调度开销。

### 10.27 同进程 Cold/Hit Torch Profiler 对照

本轮启用 vLLM 内置 Torch profiler endpoint，`ignore_frontend=true`，关闭 stack、shape
和 memory 采集。服务先用 vector cold/hit 完成 Triton 热身，再用 matrix cold 物化缓存；
第一个采集窗口只包含 matrix 的 1840/2100 hit。随后在同一已加载服务、同一 GPU 上
重启 profiler，第二个窗口只包含 tensor 的 0/2100 cold。两个请求均为 2100-token
prompt、1-token completion、HTTP 200；因此 trace 不包含模型加载或 Triton 冷编译。

GPU 0 在 UTC `18:00:03` 与 `18:01:19` 两次检查中均为空闲，固定
`CUDA_VISIBLE_DEVICES=0`。服务环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefill-warps2-20260720` / `sha256:2028a046...0dddc7` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| Profiler | Torch CPU+CUDA，worker only，no stack/shape/memory |

Profiler 顶层 `execute_context` 的 CUDA total 包含嵌套事件，百分比超过 100%，不能作为
普通 kernel 总量；下表使用 profiler 的 `Self CPU/CUDA time total` 和同一层级的
key-averages 行。Profiler 本身会增加 CPU 开销，因此这些数值用于同进程归因，不替代
第 10.25 节未开启 profiler 的 TTFT。

| 指标 | Cold 2100 | Hit 260+1840 cached | Hit - Cold |
| --- | ---: | ---: | ---: |
| Worker `execute_context` CPU total | 43.143 ms | 65.534 ms | +22.391 ms |
| 普通 GPU kernel Self CUDA total | 19.675 ms | 30.897 ms | +11.222 ms |
| 36 层 attention CPU total | 19.358 ms | 41.640 ms | +22.282 ms |
| 36 层 attention CUDA total | 5.766 ms | 26.832 ms | +21.066 ms |
| `_oscar_cached_prefill_kernel` | 0 | 23.311 ms | +23.311 ms |

Hit trace 中 `_oscar_cached_prefill_kernel` 恰好调用 36 次，单次均值 0.647527 ms，
累计 23.311 ms；这与第 10.26 节非零 microbenchmark 的 0.643991 ms 一致，证明
`num_warps=2` 已在真实服务执行，且 kernel 加速没有丢失在构建或调用选择中。Hit 的
普通 GPU kernel 总量为 30.897 ms，其中该 kernel 占 75.45%。

Cold 的 MLP/投影等非 attention GPU 工作因 2100-token shape 更重；Hit 虽减少这些
工作，但新增 cached attention 后，普通 GPU kernel 总量仍多 11.222 ms。若其他 GPU
工作不变，要使 Hit 总量不高于 Cold，cached kernel 累计需从 23.311 ms 降至约
`19.675 - (30.897 - 23.311) = 12.089 ms`，即还需约 1.929x；这只是 trace 导出的局部
预算，不是最终 TTFT 预测。

CPU 侧的 +22.391 ms 几乎全部落在 36 层 attention 的 +22.282 ms。Hit trace 还记录
144 次 `aten::where`，CPU total 4.094 ms、CUDA total 0.584 ms；`aten::copy_` 相对 cold
多约 1.647 ms CPU total。当前 cached kernel 在 `cached_len=0` 时本来就输出 zero 和
`-inf` LSE，因此 backend 再通过 token-to-request index、comparison、两次 where、cast
和 LSE transpose/contiguous 做相同 masking 存在可消除的逐层开销。下一实现先让 kernel
直接输出 merge 所需的 `[heads, tokens]` LSE，并删除该冗余 postprocess；这不会单独
满足 12.089 ms kernel 预算，但可降低 eager CPU dispatch 和小 kernel 数。

两个 profiler trace、两份 key-averages、四个 cold/hit 响应、配置、日志与容器 inspect
位于 `artifacts/oscar_vllm_optimization/20260720/torch_profile_prefix_hit_warps2/`。
服务 UTC `18:08:20` 正常退出 0、`OOMKilled=false`，无运行错误；UTC `18:08:30`
所有 GPU 为 0 MiB、无进程。阶段 6 的 prefix TTFT 仍未通过。

### 10.28 Direct-LSE 与冗余后处理移除 CUDA Oracle

根据第 10.27 节的 trace，本轮做两项局部修改。第一，cached-prefill Triton kernel
不再先写 `[tokens, heads]` LSE，而是按实际消费者 `merge_attn_states` 所需的
`[heads, tokens]` strides 直接写出；backend 相应删除 token-to-request gather、cached
长度比较、两次显式 `where` 和 LSE transpose/contiguous，仅保留输出 dtype cast。
当请求 `cached_len=0` 时，kernel 的空循环本身产生 zero output 和 `-inf` LSE。第二，
每层 prefill 对 cached context 的判断从 attention 与 cache update 各一次合并为一次，
同一布尔值同时传给两个调用点。poisoned oracle 新增 LSE shape `(num_query_heads, 1)`
及值有限的断言。

当前三个改动文件通过 pycompile、Ruff lint 和 Ruff format check。首轮静态检查仅发现
删除旧索引后遗留的局部变量 `N`，删除该孤立变量后完整重跑通过。本轮 GPU 环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 底座镜像 | `oscar-vllm:prefill-warps2-20260720` |
| 镜像 ID | `sha256:2028a0468889399fb66e540d3ebb35effde98ee45aa3113b2fb44d4f504dddc7` |
| 源码使用方式 | 当前 workspace Python 源码覆盖镜像内已安装 package，保留镜像编译扩展 |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

物理 GPU 0 在 UTC `18:13:54` 与 `18:15:19` 两次检查中均为 0 MiB、利用率 0%、
无计算进程，间隔 85 秒；测试固定使用该卡和全新
`TRITON_CACHE_DIR=/tmp/triton-cache-direct-lse`。poisoned prefix-hit 与 mixed
cached/cold backend 两项定向 CUDA 结果为
`2 passed, 14 deselected, 61 warnings in 55.40s`，pytest 退出码 0。前者覆盖随机
非单位 V rotation、未拥有 recent row、direct-LSE ABI 和 full-dequant reference；
后者在同一 batch 中包含一个 384-token cached 请求与一个 `cached_len=0` 请求，覆盖
删除显式 mask 后的 zero/`-inf` 自然值、LSE merge 和 cache 更新时序。

因此，当前 oracle 未发现 direct-LSE、后处理删除或 cached-context 判断合并引入数值、
混合请求隔离或状态回退。该结果只完成正确性门禁，尚不能证明 CPU/GPU 开销或服务
TTFT 已改善。UTC `18:17:37` GPU 0 已回收为 0 MiB、利用率 0%；完整 `nvidia-smi`、
两次分配检查、镜像 inspect、pytest 日志/退出码和回收检查位于
`artifacts/oscar_vllm_optimization/20260720/prefill_direct_lse_cuda/`。下一步构建包含
当前修改的独立镜像，并复用相同 cold/hit workload 与同进程 profiler 验证实际收益。

### 10.29 Direct-LSE Prefix-Hit TTFT

当前源码构建为独立镜像 `oscar-vllm:prefill-direct-lse-20260720`，完整镜像 ID 为
`sha256:18dd8545e9f89d15712b7af6ff2a0d0f8743747bea856760858be777f99bec97`。
构建退出码 0。服务保持 Qwen3-4B-Instruct-2507、BF16、OSCAR INT2、精确 10 GiB、
maxseq2、eager、prefix caching、chunked prefill 和相同 K/V rotations；环境如下：

| 项目 | 实际值 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

GPU 0 在 UTC `18:19:31` 与 `18:21:10` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 99 秒。服务于 UTC `18:21:39` 启动，UTC `18:24:27` 首次观测到 health
200；日志确认三段式仍为 INT2 history 461936、BF16 prefix 128、BF16 recent 512 tokens，
prefix/chunked 启用且 async scheduling 关闭。容器内源码也确认 direct-LSE strides、
后处理删除和单次 cached-context 判断均已进入镜像。

client 使用 tokenizer 得到的单 token 词元构造每个 2100-token prompt：cold/hit 前
1840 tokens 完全相同，后 260 tokens 使用不同词元。先运行两对不计入统计的 warm-up，
再依次运行 vector、matrix、tensor、scalar、kernel 五对。首轮 cold/hit CV 分别为
8.696%/3.644%，均超过 3%，因此按规则继续追加十对并保留全部样本。client 在首个非空
SSE token 到达时使用单调时钟记录 TTFT；脚本在运行前通过 pycompile、Ruff lint 和
format check。

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 15 | 15 |
| Mean TTFT | 35.376 ms | 45.196 ms |
| Population std | 1.991 ms | 1.188 ms |
| CV | 5.627% | 2.629% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

所有 30 个正式请求均 HTTP 200、prompt tokens=2100；15 个 cold 全为 miss，15 个 hit
全部精确命中 1840 tokens。合并后 cold CV 仍超过 3%，原始样本完整保留；该波动不会
改变本轮门槛判定，因为 15/15 配对的 `hit - cold` 均为正，配对差 mean 为
`+9.820 ms`、population std 为 1.517 ms。Hit 比 cold 慢 `27.759%`，因此 M2 继续
明确失败。

本轮 client 使用确定的 token-ID prompt，而第 10.25 节记录的是此前落盘的 prompt
构造；两节绝对均值不能直接归因为 direct-LSE 的跨服务加速或回退。可靠结论仅来自
本节同服务配对：删除冗余后处理仍不足以抵消 cached-prefill 主 kernel 成本。当前服务
未配置 worker profiler，故不把 endpoint 调用伪装为有效 trace；容器于 UTC
`18:31:05` 正常退出 0、`OOMKilled=false`、无运行错误，同刻 GPU 0 回收为 0 MiB、
利用率 0%、无进程。构建日志、两次 GPU 检查、client、原始结果、派生统计、容器
inspect、版本/源码核对和服务日志位于
`artifacts/oscar_vllm_optimization/20260720/prefill_direct_lse_ttft/`。下一步启动显式
配置 worker-only Torch profiler 的独立服务，不能复用本容器。

### 10.30 Direct-LSE 同 Payload Torch Profiler 对照

为与第 10.27 节直接比较，本轮重新启动显式配置 worker-only Torch profiler 的
direct-LSE 服务，完整复用该节的 vector、matrix、tensor 三个 14700 字符 payload 和
调用顺序：vector cold/hit 完成 Triton 热身，matrix cold 物化缓存，第一个 profile
窗口仅含 matrix hit；第二个窗口仅含 tensor cold。matrix hit 为 1840/2100 cached，
tensor cold 为 0/2100，两个推理请求和四个 profiler endpoint 均 HTTP 200。

GPU 0 在 UTC `18:32:11` 与 `18:33:51` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 100 秒。实验固定 `CUDA_VISIBLE_DEVICES=0`，镜像、模型、OSCAR 配置与
第 10.29 节完全相同；Profiler 为 Torch CPU+CUDA、worker only，并关闭 stack、shape、
memory 和 gzip。软件环境仍为 Python 3.12.13、vLLM 0.25.0、PyTorch
2.11.0+cu130/PyTorch CUDA 13.0、Triton 3.6.0、Transformers 5.13.0，完整
`nvidia-smi` 为 NVIDIA B200 183359 MiB、Driver 595.71.05、CUDA 13.2。

与第 10.27 节一样，下表使用 profiler 的 worker `execute_context` CPU total、普通 GPU
kernel `Self CUDA time total` 和同层级 attention 行；顶层嵌套 CUDA total 不参与比较。

| 指标 | 旧 Cold | 旧 Hit | 旧差值 | Direct-LSE Cold | Direct-LSE Hit | 新差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Worker CPU total | 43.143 ms | 65.534 ms | +22.391 ms | 38.076 ms | 52.254 ms | +14.178 ms |
| 普通 GPU kernel Self CUDA total | 19.675 ms | 30.897 ms | +11.222 ms | 19.624 ms | 30.114 ms | +10.490 ms |
| 36 层 attention CPU total | 19.358 ms | 41.640 ms | +22.282 ms | 18.178 ms | 30.698 ms | +12.520 ms |
| 36 层 attention CUDA total | 5.766 ms | 26.832 ms | +21.066 ms | 5.758 ms | 26.011 ms | +20.253 ms |
| Cached-prefill kernel | 0 | 23.311 ms | +23.311 ms | 0 | 23.229 ms | +23.229 ms |

在完全相同 payload/profile 配置下，worker CPU 的 hit-cold 差缩小 8.213 ms，即
36.680%；attention CPU 差缩小 9.762 ms，即 43.810%。Raw trace 的 operator 对照与
源码预期一致：hit 的 `aten::where` 从 144 次降为 0，`aten::contiguous` 从 36 次降为
0，`aten::any` 从 72 次降为 36 次；`aten::copy_` 从 344 次、4.023 ms CPU total 降至
236 次、3.099 ms。上述 operator total 存在嵌套，不能彼此求和；且每侧只有一个
profile 窗口，因此跨服务 CPU 差只作为与实现一致的观测证据，不冒充统计显著性结论。

GPU 主瓶颈没有改变。Direct-LSE hit 中 `_oscar_cached_prefill_kernel` 仍调用 36 次，
累计 23.229 ms、单次 0.645246 ms，仅比旧 trace 的 23.311 ms 少 0.352%，可视为同一
水平；它占 hit 普通 GPU kernel 的 77.14%。Direct-LSE 后其他 hit GPU 工作约为
`30.114 - 23.229 = 6.885 ms`。若其他工作不变，要使 hit 普通 GPU kernel 总量不高于
cold 19.624 ms，该 kernel 累计需降至约 `19.624 - 6.885 = 12.739 ms`，即仍需
`1.824x`，约 0.354 ms/层。这仍是 trace 局部预算，不是最终 TTFT 预测。

结论是：direct-LSE 与冗余后处理删除是有实证的 CPU/小算子优化，应保留；但第 10.29
节 M2 失败的主因仍是 cached-prefill kernel，不应继续从后处理寻找数量级收益。下一项
结构优化应利用 Qwen3 的 GQA 比例，让同一 KV head 对应的 query heads 共享 K/V
解量化与读取，再以 CUDA oracle 和 kernel microbenchmark 决定是否进入服务。

两份 direct-LSE trace、两份 profiler table、同 payload 副本、HTTP 结果、旧新 operator
派生表、环境、容器 inspect 和日志位于
`artifacts/oscar_vllm_optimization/20260720/torch_profile_direct_lse/`。服务于 UTC
`18:40:33` 正常退出 0、`OOMKilled=false`、无运行错误；同刻 GPU 0 回收为 0 MiB、
利用率 0%、无进程。辅助 operator JSON 首次因 jq 保留字变量 `label` 解析失败，随后
改用 `run_label` 从完整 trace 重建；GPU 实验未重跑，原始结果不受影响。

### 10.31 GQA 2-Head K/V 复用 CUDA Oracle

首个主 kernel 结构候选利用 Qwen3 的 4:1 GQA：每个 Triton program 不再只处理一个
query head，而是将同一 KV head 下两个 query heads 的 32-token query tile 展平为
64 rows，共享每个 32-token history tile 的 block-table 查找、K/V 读取和 INT2
解量化。program 数由每个 query tile 32 个减为 16 个；若模型的 KV group 为奇数，
wrapper 自动退回 1 head/program，避免跨 KV head 共享。生产 tile 仍为 32x32，launch
仍为 2 warps/2 stages，history-sized BF16 workspace 仍未引入。

修改后的 kernel 与现有 poisoned oracle 通过 pycompile、Ruff lint 和 format check。
首轮 format check 只要求一处 accumulator 声明的机械排版，修正后完整静态门禁通过。
GPU 0 在 UTC `18:45:14` 与 `18:46:47` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 93 秒。本轮固定 `CUDA_VISIBLE_DEVICES=0`，使用
`oscar-vllm:prefill-direct-lse-20260720` 作为编译扩展底座并覆盖当前 Python 源码，
使用全新 `TRITON_CACHE_DIR=/tmp/triton-cache-gqa2`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 底座镜像 ID | `sha256:18dd8545e9f89d15712b7af6ff2a0d0f8743747bea856760858be777f99bec97` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

poisoned prefix-hit 与 mixed cached/cold backend 两项定向 CUDA 结果为
`2 passed, 14 deselected, 61 warnings in 54.10s`，pytest 退出码 0。前者覆盖随机
非单位 V inverse、未拥有 recent row 和 direct-LSE；后者覆盖 GQA2 展平寻址、同批
cached_len=384/0、LSE merge 和 cache 更新时序。该结果说明 64x128 FP32 accumulator
在 B200/2 warps 下能够编译执行，当前 oracle 未发现 query-head/KV-head 映射、数值或
状态回退。

UTC `18:49:30` GPU 0 已回收为 0 MiB、利用率 0%、无进程。静态日志、两次 GPU
检查、完整 `nvidia-smi`、镜像 inspect、pytest 日志/退出码和回收检查位于
`artifacts/oscar_vllm_optimization/20260720/prefill_gqa2_cuda/`。本节只完成正确性和
可编译性门禁；program 数减半不是性能证据。下一步必须使用合法非零 cache、相同输入
和交错顺序直接比较 1-head 与 2-head kernel，再决定是否保留 GQA2。

### 10.32 GQA1/GQA2 非零 Cache Microbenchmark

本轮使用生产 `oscar_store` 从随机 BF16 K/V 生成合法非零 INT2 packed data、scale 和
zero，并填充非零 BF16 prefix/recent；shape 精确复现服务 hit：260 query tokens、1840
cached tokens、32 query heads、8 KV heads、head dim 128、32x32 tile。交错比较
GQA1/GQA2 与 2/4 warps 共四个配置，每点 20 次预热、5 批 x 100 次 CUDA Event 测量；
所有配置先与 GQA1/2-warps 的 output/LSE 做相同输入数值对照。

脚本在实验前通过 pycompile、Ruff lint 与 format check；首次 format check 只要求两处
机械排版，修正后完整门禁通过。GPU 0 在 UTC `18:52:52` 与 `18:54:21` 两次检查中
均为 0 MiB、利用率 0%、无计算进程，间隔 89 秒；固定 `CUDA_VISIBLE_DEVICES=0`，
使用第 10.31 节相同底座镜像、当前 GQA2 源码和全新
`TRITON_CACHE_DIR=/tmp/triton-cache-gqa2-microbench`。GPU、driver、Python、vLLM、
PyTorch、CUDA、Triton 和 Transformers 版本均与第 10.31 节一致。

四个配置的 output/LSE 对照全部通过，测量结果为：

| Query heads/program | Warps | Programs/query tile | Mean | Population std | CV |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 32 | 0.650279 ms | 0.000351 ms | 0.054% |
| 1 | 4 | 32 | 0.825287 ms | 0.000045 ms | 0.006% |
| 2 | 2 | 16 | 0.612758 ms | 0.000903 ms | 0.147% |
| 2 | 4 | 16 | 0.486029 ms | 0.000029 ms | 0.006% |

GQA2/2-warps 相对 GQA1/2-warps 只加速 `1.061x`；更宽的 64x128 accumulator 在
4 warps 下才有效利用，GQA2/4-warps 相对 GQA1/2-warps 加速 `1.338x`，相对
GQA2/2-warps 加速约 `1.261x`。GQA1/4-warps 反而最慢，说明 warps 的最佳值依赖
program 内 query rows，不能把第 10.23 节单 head 的 2-warps 结论机械套用到 GQA2。

按 36 层线性计算，GQA1/2-warps 与 GQA2/4-warps 分别约 23.410 ms 和 17.497 ms，
局部减少约 5.913 ms；后者仍高于第 10.30 节 12.739 ms 的 trace 预算，距离 0.354
ms/层还需约 `1.373x`。这是 microbenchmark 推导，不替代服务测量，但表明只做 GQA2
仍不足以从局部预算证明过线。生产候选应改为 GQA2/4-warps，同时继续测试同一 KV head
四个 query heads 全共享的 GQA4，若寄存器压力抵消收益则回到 GQA2。

benchmark 退出码 0；UTC `18:58:16` GPU 0 已回收为 0 MiB、利用率 0%、无进程。
脚本、合法非零 cache 的全部批次原始值、正确性标记、环境、两次 GPU 检查和回收检查
位于 `artifacts/oscar_vllm_optimization/20260720/prefill_gqa2_microbench/`。

### 10.33 GQA4 首轮无效实验

为保持第 10.32 节 artifact 不变，GQA4 使用独立 runner 复用其数据构造与测量函数，
计划比较 GQA1/2-warps、GQA2/4-warps、GQA4/4-warps 和 GQA4/8-warps。runner 在
GPU 实验前通过 pycompile、Ruff lint 与 format check；两轮静态检查只涉及 import
空行和路径表达式机械排版。

GPU 0 在 UTC `19:02:20` 与 `19:03:54` 两次检查中均为空闲，间隔 94 秒；环境与第
10.32 节一致，使用全新 `TRITON_CACHE_DIR=/tmp/triton-cache-gqa4-microbench`。
容器退出码为 1，最终异常为 `KeyError: 'gqa_2_warps_2'`：被复用脚本的结果汇总固定
读取 GQA2/2-warps，而 runner 的配置矩阵遗漏该点。`results.json` 未生成，因此本轮
没有可报告的 GQA4 正确性或性能结果，不能从已经执行过的内部循环推测数值。

UTC `19:05:59` GPU 0 已回收为 0 MiB、利用率 0%、无进程。失败日志、runner、静态
检查、两次 GPU 检查和回收检查保留在
`artifacts/oscar_vllm_optimization/20260720/prefill_gqa4_microbench/`。修正方法只是在
配置矩阵补回 GQA2/2-warps，随后使用全新结果文件和新一轮 GPU 分配检查重跑。

### 10.34 GQA4 修正版 Microbenchmark

修正版 runner 仅补回汇总逻辑需要的 GQA2/2-warps，随后重新通过 pycompile、Ruff
lint 与 format check。GPU 0 在 UTC `19:08:12` 与 `19:10:17` 两次检查中均为 0 MiB、
利用率 0%、无计算进程，间隔 125 秒；本轮使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-gqa4-microbench-run2`，环境、合法非零 cache、
shape、预热和 5 批 x 100 次交错测量方法均与第 10.32 节一致。

五个配置的 output/LSE 全部通过 GQA1/2-warps 对照，修正版有效结果为：

| Query heads/program | Warps | Programs/query tile | Mean | Population std | CV |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 32 | 0.649523 ms | 0.000323 ms | 0.050% |
| 2 | 2 | 16 | 0.611988 ms | 0.000617 ms | 0.101% |
| 2 | 4 | 16 | 0.486021 ms | 0.000047 ms | 0.010% |
| 4 | 4 | 8 | 0.522671 ms | 0.000032 ms | 0.006% |
| 4 | 8 | 8 | 0.540910 ms | 0.000057 ms | 0.010% |

GQA2/4-warps 再次排名第一，并复现第 10.32 节的 0.486029 ms；两轮差仅约 0.002%。
GQA4/4-warps 虽把 program 数进一步减半，却比 GQA2/4-warps 慢约 7.541%；GQA4/8
更慢约 11.294%。因此额外 K/V 复用已被 128x128 FP32 accumulator 的寄存器/执行资源
压力抵消；停止扩大 query heads/program，生产候选确定为 GQA2/4-warps。

benchmark 退出码 0，`results.json` 已生成；UTC `19:12:22` GPU 0 已回收为 0 MiB、
利用率 0%、无进程。修正版静态日志、全新 GPU 检查、完整五配置原始值、正确性标记、
环境和回收检查均位于
`artifacts/oscar_vllm_optimization/20260720/prefill_gqa4_microbench/`，并与第 10.33 节
首轮失败日志并存且文件名区分。下一步把生产 GQA2 launch 从 2 warps 改为 4，重跑
定向 CUDA 后构建独立服务镜像验证 TTFT。

### 10.35 GQA2/4-Warps 生产路径 CUDA Oracle

生产 wrapper 仅把偶数 KV group 的 GQA2 launch 从 2 warps 改为第 10.32/10.34 节
稳定胜出的 4 warps；奇数 KV group 的 GQA1 fallback 仍为 2 warps，tile、stages、
direct-LSE 和 kernel 语义均不变。目标源码与 poisoned oracle 通过 pycompile、Ruff
lint 和 format check。

GPU 0 在 UTC `19:14:28` 与 `19:16:01` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 93 秒。本轮固定 `CUDA_VISIBLE_DEVICES=0`，继续使用 direct-LSE 镜像
作为编译扩展底座、覆盖当前源码，并使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-gqa2-warps4-production`。环境版本与第 10.31 节
一致。

poisoned prefix-hit 与 mixed cached/cold backend 两项定向 CUDA 结果为
`2 passed, 14 deselected, 61 warnings in 66.74s`，pytest 退出码 0。当前 oracle 未发现
GQA2/4-warps 引入 query-head 映射、随机非单位 V inverse、direct-LSE、mixed merge 或
cache 更新时序回退。UTC `19:18:03` GPU 0 已回收为 0 MiB、利用率 0%、无进程。

两次 GPU 检查、完整 `nvidia-smi`、镜像 inspect、pytest 日志/退出码与回收检查位于
`artifacts/oscar_vllm_optimization/20260720/prefill_gqa2_warps4_cuda/`。本节完成生产
路径正确性门禁；下一步构建独立镜像并用第 10.29 节同一 token-ID client 验证服务
TTFT，不能用 0.486021 ms microbenchmark 代替端到端结论。

### 10.36 GQA2/4-Warps Prefix-Hit TTFT

当前生产源码构建为独立镜像 `oscar-vllm:prefill-gqa2-warps4-20260720`，完整镜像 ID
为 `sha256:cc08fd6981c66b6a07bf92f0aa0cecfab058c5b7edc103dae358e51f75e7b36f`，构建
退出码 0。镜像内源码确认偶数 KV group 选择 2 query heads/program 和 4 warps，奇数
group 回退 1 head/program 和 2 warps。服务保持第 10.29 节的 Qwen3-4B-Instruct-2507、
BF16、OSCAR INT2、精确 10 GiB、maxseq2、eager、prefix caching、chunked prefill 和
相同 K/V rotations，环境如下：

| 项目 | 实际值 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` |
| 固定物理 GPU / 显存 | GPU 0，NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

UTC `19:19:24` 与 `19:20:58` 的两次完整 `nvidia-smi` 检查中，GPU 0-7 均为
0 MiB、利用率 0%，无计算进程，间隔 94 秒。本轮固定分配物理 GPU 0；服务于 UTC
`19:23:07` 启动，UTC `19:25:56` 首次观测 health 200。正式采样 client 与第 10.29
节文件完全相同，两份 SHA256 均为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`，因此 prompt
结构、warm-up、SSE TTFT 计时和追加样本规则一致。

两对不计入统计的 warm-up 后，首五对 cold/hit mean 为 40.359/51.553 ms，CV 分别
为 12.813%/3.219%，超过 3% 门槛，因此继续追加十对并保留全部 15 对。最终实际结果
如下：

| 指标 | Cold | Prefix hit |
| --- | ---: | ---: |
| 有效样本数 | 15 | 15 |
| Mean TTFT | 39.079 ms | 49.456 ms |
| Population std | 4.828 ms | 4.170 ms |
| CV | 12.355% | 8.432% |
| `cached_tokens` | 0 | 1840 / 2100 |
| 本地实际 prefill tokens | 2100 | 260 |

所有 30 个正式请求均 HTTP 200、prompt tokens=2100；全部 cold 为 0 cached tokens，
全部 hit 精确命中 1840 tokens。15/15 配对的 `hit - cold` 均为正，配对差 mean 为
`+10.377 ms`、population std 为 3.458 ms；hit 比 cold 慢 `26.554%`。虽然追加后
两侧 CV 仍高，但不存在符号不确定性，因此本轮 M2 仍明确失败：GQA2/4-warps 的
microbenchmark 收益没有转化为 prefix-hit TTFT 优于 cold。

与第 10.29 节的同 client direct-LSE 服务相比，本轮 cold 从 35.376 ms 上移到
39.079 ms（`+10.466%`），hit 从 45.196 ms 上移到 49.456 ms（`+9.424%`），两侧
近似共同平移；配对差从 `+9.820 ms` 变为 `+10.377 ms`，增加 0.557 ms。两轮均为
独立服务且当前两侧 CV 高，服务日志还在正式采样期间记录了一次
`_oscar_cached_prefill_kernel` JIT compilation latency 提示。因此这些跨服务差值不足以
证明 GQA2 有端到端收益或回退，不能把 0.557 ms 当作算法差异；可靠结论仍是两轮各自
同服务内的 prefix hit 均慢于 cold。

容器于 UTC `19:29:04` 请求停止，最终退出码 0、`OOMKilled=false`，严格扫描未发现
traceback、CUDA error、runtime error 或实际 OOM。UTC `19:29:09` 的完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、利用率 0%，无进程。镜像/源码核对、两次分配
检查、完整 `nvidia-smi`、client、原始 30 请求、配对统计、服务日志、环境、容器
inspect 和回收检查位于
`artifacts/oscar_vllm_optimization/20260720/prefill_gqa2_warps4_ttft/`。下一步使用与
第 10.30 节完全相同的 profiler payload 和调用顺序，直接检查 GQA2 是否降低了 36 次
cached-prefill kernel 累计时间；在该证据落地前，不根据 endpoint 总均值保留或撤销
GQA2。

### 10.37 GQA2 同 Payload Torch Profiler 对照

本轮使用第 10.36 节完整相同的 GQA2/4-warps 镜像，并逐文件复制第 10.30 节的
vector、matrix、tensor 三个 payload；三组新旧 SHA256 分别完全一致。服务参数和
调用顺序也保持不变：vector cold/hit 完成 Triton 热身，matrix cold 物化缓存，第一个
profile 窗口只包含 matrix hit，第二个窗口只包含 tensor cold。matrix hit 精确命中
1840/2100 tokens，tensor cold 为 0/2100；五个推理请求与四个 profiler endpoint 均
HTTP 200。

GPU 0 在 UTC `19:32:49` 与 `19:34:02` 两次检查中均为 0 MiB、利用率 0%，无计算
进程，间隔 73 秒；GPU 1-7 也均为空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`。服务实际容器于 UTC `19:34:20` 启动，UTC `19:36:36`
首次观测 health 200。实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefill-gqa2-warps4-20260720` / `sha256:cc08fd69...e7b36f` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| Profiler | Torch CPU+CUDA，worker only，no stack/shape/memory/gzip |

下表与第 10.30 节使用相同 key-averages 口径。每侧仍只有一个 profile 窗口，CPU
跨服务差只作为观测，不作统计显著性结论；普通 GPU kernel Self CUDA total 和目标
kernel 的 raw trace duration 用于核对结构优化是否进入真实服务。

| 指标 | Direct-LSE Cold | Direct-LSE Hit | 旧差值 | GQA2 Cold | GQA2 Hit | 新差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Worker CPU total | 38.076 ms | 52.254 ms | +14.178 ms | 42.281 ms | 54.410 ms | +12.129 ms |
| 普通 GPU kernel Self CUDA total | 19.624 ms | 30.114 ms | +10.490 ms | 19.640 ms | 24.712 ms | +5.072 ms |
| 36 层 attention CPU total | 18.178 ms | 30.698 ms | +12.520 ms | 18.891 ms | 29.978 ms | +11.087 ms |
| 36 层 attention CUDA total | 5.758 ms | 26.011 ms | +20.253 ms | 5.763 ms | 20.640 ms | +14.877 ms |
| Cached-prefill kernel | 0 | 23.229 ms | +23.229 ms | 0 | 17.856 ms | +17.856 ms |

GQA2 hit trace 中 `_oscar_cached_prefill_kernel` 恰好调用 36 次，raw duration 累计
17.855682 ms、单次 0.495991 ms；cold trace 为 0 次。相对 direct-LSE 同 payload 的
23.228842 ms、0.645246 ms/层，累计减少 5.373160 ms，即 23.131%，实际加速
`1.301x`。这与第 10.32/10.34 节非零 cache microbenchmark 的 0.486 ms 方向和量级
一致，证明 GQA2 K/V 复用与 4-warps 选择已经进入真实服务关键路径，不是仅存在于
独立 benchmark 的收益。因此保留 GQA2 实现有 profiler 证据支持。

但它仍未跨过局部 GPU 门槛。GQA2 hit 的其他普通 GPU 工作约为
`24.712 - 17.856 = 6.856 ms`；若其他工作不变，要让 hit 普通 GPU kernel 总量不高于
cold 19.640 ms，cached kernel 累计需降至约
`19.640 - 6.856 = 12.784 ms`，即当前仍需 `1.397x`，约 0.355 ms/层。实测 Self CUDA
hit-cold 差虽从 10.490 ms 降到 5.072 ms、缩小约 51.65%，仍为正；这与第 10.36 节
TTFT 未过线并不矛盾。下一项优化仍必须作用于 cached-prefill 主 kernel 或移除等量的
其他 hit GPU/CPU 工作，不能把当前 5.373 ms kernel 收益等同于 M2 已通过。

两个 profile 窗口分别为 UTC `19:37:21`--`19:37:23` 和
`19:37:43`--`19:37:47`，均在 warm-up 后采集。容器于 UTC `19:39:14` 请求停止，
最终退出码 0、`OOMKilled=false`，严格日志扫描无运行错误；UTC `19:39:22` 完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、利用率 0%，无计算进程。原始 trace、profiler
table、逐请求响应、payload SHA256、派生 summary、环境、源码核对、完整
`nvidia-smi`、日志与容器 inspect 位于
`artifacts/oscar_vllm_optimization/20260720/torch_profile_gqa2_warps4/`。

### 10.38 Materialize Prefix-KV 首轮组件基准

第 10.37 节仍需 1.397x 后，重新审计本地论文官方 SGLang 配套代码。其 FA3 INT2
prefill 路径实际先用 Triton 将混合 HP/INT2 prefix 分别物化为连续 BF16 K/V，再逐请求
与当前 extend K/V `torch.cat`，调用 `flash_attn_varlen_func`，最后做 V inverse；不是
融合 INT2 attention。基于该已落地路径，本轮只新增独立 benchmark，未修改生产
backend，并额外测试直接把 cached K/V 写入 unified workspace、避免 prefix K/V 与
unified K/V 同时存活的候选。

脚本先用生产 `oscar_store` 生成合法非零 INT2 cache，并写入对应 BF16 prefix；shape
为 260 query、1840 cached、32 query heads、8 KV heads、D=128，与前述 TTFT/profile
一致。当前 split 路径执行 suffix FA4、GQA2/4-warps cached kernel 和 LSE merge；候选
路径把 cached 与 current K/V 物化为单个 2100-token BF16 K/V 后执行一次 causal FA4。
两条路径使用相同输入，候选输出通过当前 split 输出对照，最大绝对差为
`0.001953125`，`atol=rtol=0.03`。

GPU 0 在 UTC `19:49:18` 与 `19:50:58` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 100 秒；GPU 1-7 也均为空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，每点 20 次 warm-up、5 批 x 100 次 CUDA Event 测量。
环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefill-gqa2-warps4-20260720` / `sha256:cc08fd69...e7b36f` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / FA | 3.6.0 / 5.13.0 / FA4 |

首轮主结果如下：

| 路径 | Mean | Population std | CV | 36 层线性值 |
| --- | ---: | ---: | ---: | ---: |
| 当前 split+merge | 0.502143 ms | 0.000150 ms | 0.030% | 18.077 ms |
| Direct unified workspace | 0.105739 ms | 0.000312 ms | 0.295% | 3.807 ms |
| 官方式 prefix+`cat` 分配 | 0.108051 ms | 0.000352 ms | 0.326% | 3.890 ms |
| Full FA4 only | 0.054877 ms | 0.002613 ms | 4.761% | 1.976 ms |

Direct unified 相对当前 split 稳定加速 `4.749x`，0.105739 ms/层显著低于第 10.37
节 0.355 ms/层局部预算。最佳 mixed dequant 配置为 8 tokens/program、4 warps，mean
0.020995 ms、CV 1.572%。Direct unified K+V workspace 为 8,601,600 bytes，即
8.20 MiB、10 GiB 的 `0.080109%`；官方式 prefix+unified 主张量合计 16,138,240 bytes，
即约 15.39 MiB、`0.150299%`。两者都低于 5% workspace 门槛，direct unified 的主张量
峰值更小。

本轮主决策点的 CV 均低于 3%，数值和数量级已支持继续推进 direct unified 候选；但
辅助 `Full FA4 only` CV 为 4.761%，淘汰的 1 token/program、1-warp dequant 配置 CV
为 44.078%。依照采样规则，这两个辅助点需要追加样本后才能形成完整稳定性记录，因此
本节仍标为首轮，不把组件结果直接宣告为生产 M2 通过。benchmark UTC
`19:51:18`--`19:52:45`、容器退出码 0；UTC `19:53:09` 完整 `nvidia-smi` 显示
GPU 0-7 均为 0 MiB、利用率 0%，无计算进程。脚本、静态检查、完整批次值、输出对照、
workspace 计算、环境、镜像 inspect 和 `nvidia-smi` 位于
`artifacts/oscar_vllm_optimization/20260720/prefill_materialize_microbench/`。

### 10.39 Materialize Prefix-KV 追加采样复核

追加复核保持第 10.38 节的镜像、shape、合法非零 OSCAR cache、20 次 warm-up
和每批 100 次 CUDA Event 测量不变。脚本先测 5 批；仅当首批 CV 超过 3%
时自动追加 10 批，并同时保留初始 CV、最终批数与全部原始值。GPU 0 在 UTC
`19:56:02` 与 `19:57:36` 两次检查中均为 0 MiB、利用率 0%、无计算进程，
间隔 94 秒；GPU 1-7 也均空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`。环境与第 10.38 节完全相同：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefill-gqa2-warps4-20260720` / `sha256:cc08fd69...e7b36f` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / FA | 3.6.0 / 5.13.0 / FA4 |

追加采样的主结果如下：

| 路径 | Mean | Population std | CV | 批数 | 36 层线性值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 当前 split+merge | 0.502096 ms | 0.000106 ms | 0.021% | 5 | 18.075 ms |
| Direct unified workspace | 0.109149 ms | 0.001493 ms | 1.368% | 5 | 3.929 ms |
| 官方式 prefix+`cat` 分配 | 0.109605 ms | 0.000428 ms | 0.391% | 5 | 3.946 ms |
| Full FA4 only | 0.054046 ms | 0.001514 ms | 2.802% | 15 | 1.946 ms |

Direct unified 相对当前 split 加速 `4.600x`，再次显著低于第 10.37 节给出的
0.355 ms/层局部预算。与首轮相比，split 只变化 `-0.009%`，direct unified
变化 `+3.225%`，官方式路径变化 `+1.439%`；两轮的输出最大绝对差均为
`0.001953125`，且都通过 `atol=rtol=0.03` 对照。追加到 15 批后 Full FA4 only
的 CV 从初始 3.731% 降至 2.802%，弥补了第 10.38 节的辅助稳定性缺口。

四个 dequant 配置的均值仍集中在 0.0211--0.0212 ms；其中 1 token/1-warp 受
首批首个 0.053754 ms 离群值影响，追加后 CV 仍为 34.127%，4 tokens/2-warps
追加后 CV 为 3.557%。这两个点不影响主路径决策；稳定的 1 token/2-warps 与
8 tokens/4-warps 分别为 0.021202 ms/CV 1.077% 和 0.021229 ms/CV 1.154%，性能差
仅约 0.125%。因此不根据本轮极小的均值差宣称某一 dequant launch 具有稳定
优势；生产集成优先保持两轮都稳定的配置，再用 backend oracle 与真实服务决定。

实验过程中有一次可追溯的操作失败：首次容器命令使用了镜像中不存在的
`python` entrypoint，OCI 在进入 benchmark 前 exit127，未产生 GPU 计算。确认全卡仍为
0 MiB/0% 后，改用镜像实际的 `/usr/bin/python3` 3.12.13 重启；正式 run2 在
UTC `19:59:58`--`20:01:21` 完成、exit0。UTC `20:01:37` 的完整检查显示
GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。失败日志、正式日志、完整
批次值、SHA256、两次分配检查、`nvidia-smi` 和镜像 inspect 位于第 10.38 节
同一 artifact 目录。

组件证据现已满足继续集成 direct unified materialize 候选的条件：两轮均正确，
主路径 CV 低于 3%，且耗时低于局部预算。这仍不等于生产 M2 通过：下一步需在
vLLM backend 中实现多请求 packed 布局、设置受限 workspace 及超限 fallback，通过
prefix-hit 和 mixed cached/cold CUDA oracle 后，再用真实服务 TTFT 验证 M2。

### 10.40 受限 Packed Materialize 生产实现与静态门禁

基于第 10.38--10.39 节的两轮组件证据，已在生产 prefill 路径实施首版受限
materialize，改动仅涉及 `oscar_attn.py`、`triton_oscar_prefill.py` 和既有
`test_oscar.py` CUDA oracle，未改 allocator 或 scheduler ownership 合约。实现要点
如下：

- Metadata builder 为每批请求生成 final sequence 的 packed `seq_start_loc`。
- 新 Triton kernel 按 request/head/token tile 直接把 BF16 prefix、INT2 history、BF16
  recent 和当前 K/V 写入一组 packed K/V workspace，同批 cached/cold 请求不需要
  Python 逐请求循环。
- Backend 对 Q/K/V 旋转后执行一次 causal FA4，再做 V 逆旋转；当前 K/V
  的旋转结果同时复用于后续 cache store，不重复做两次 K/V 矩阵乘。
- Workspace 复用 vLLM 全局 `WorkspaceManager`，各层串行共享，不为 36 层各自预留。
  只在显式设置 `kv_cache_memory_bytes` 时启用，容量硬限为该 budget 的 5%；
  未显式设置、metadata 不完整或该批 final tokens 超限时，回退第 10.35 节
  已验证的 GQA2/4-warps fused cached-prefill。

对当前 10 GiB budget、8 KV heads、D=128、BF16 shape，每个 final token 的 K+V
workspace 为 4096 bytes，因此上限精确为 131072 tokens、536870912 bytes，即
512 MiB/5%。这是全局可复用缓冲区的上限，不是每层上限。同时必须保留约束
声明：当 fast path 启用时，它仍物化该批次的完整 active BF16 K/V；硬上限防止
显存无界增长，但不会使其满足原计划“不构造全历史 BF16 K/V”的字面门槛。

CUDA oracle 已改为使用非恒等 K/V 正交旋转，并在同批中同时包含一个 384-token
cached 请求与一个 cold 请求；它还会显式验证 workspace 上限比需求少一个
token 时选择 fused fallback。目前只完成静态门禁：首轮 `py_compile` 和 Ruff
lint 通过，Ruff formatter 要求对两个生产文件机械排版；使用相同镜像内
Ruff 排版后，三个文件的 pycompile、lint 和 format check 全部通过。完整日志位于
`artifacts/oscar_vllm_optimization/20260720/materialized_prefill_integration/`。

本节不构成可保留结论：新 Triton kernel 尚未在 B200 冷编译，packed 寻址、旋转精度、
mixed cached/cold 语义和超限 fallback 仍需 CUDA 证据。下一步在两次空闲检查后
固定物理 GPU 0，用全新 Triton cache 运行 poisoned prefix-hit 与 mixed oracle。

### 10.41 Packed Materialize 首轮 CUDA Oracle 的 Reference 失败

GPU 0 在 UTC `20:13:39` 与 `20:14:40` 两次检查中均为 0 MiB、利用率
0%、无计算进程，间隔 61 秒；GPU 1-7 也均空闲。本轮固定物理 GPU 0，
容器内 `CUDA_VISIBLE_DEVICES=0`，使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-materialized-prefill-run1`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefill-gqa2-warps4-20260720` / `sha256:cc08fd69...e7b36f` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

定向结果为 `1 passed, 1 failed, 14 deselected, 2 warnings in 58.00s`。Poisoned
prefix-hit 既有 kernel oracle 通过；新 mixed materialize oracle 在进入 backend 和新 Triton
kernel **之前** 失败。具体是 `oscar_full_dequant_kv` 返回 FP16 reference，测试直接将它
与 FP32 正交旋转矩阵做 `torch.matmul`，PyTorch 报
`RuntimeError: expected scalar type Half but found Float`。这是新增测试 reference 的
dtype 错误，不是 materialize 数值反例；但因为生产路径尚未执行，本轮不能用于
证明新实现正确。

容器 UTC `20:15:15`--`20:16:20`、exit1，非 OOM；UTC `20:16:34` 完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整
pytest 日志、exit code、分配/回收检查与 SHA256 位于第 10.40 节同一 artifact
目录。下一步只将两个 full-dequant reference 显式转为 FP32，重跑三文件静态门禁，
然后重新执行两次 GPU 空闲检查和全新 Triton cache CUDA oracle。

### 10.42 Packed Materialize 修正后 CUDA Oracle

仅将 full-dequant K/V reference 显式转为 FP32 后，三个改动文件的 pycompile、
Ruff lint 与 format check 再次全部通过。GPU 0 在 UTC `20:18:52` 与
`20:19:53` 两次检查中均为 0 MiB、利用率 0%、无计算进程，间隔 61 秒；
GPU 1-7 也均空闲。本轮仍固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-materialized-prefill-run2`。镜像、GPU、Driver、
CUDA、Python、vLLM、PyTorch、Triton 和 Transformers 版本与第 10.41 节完全相同。

修正后定向结果为 `2 passed, 14 deselected, 61 warnings in 70.81s`，pytest
exit0。证据覆盖两条独立路径：

- Poisoned prefix-hit oracle 继续通过，保证 GQA2 fused fallback 的 shared-hit/recent
  ownership 和直接 LSE ABI 未回归。
- Mixed oracle 先把 materialize 上限设为 413 tokens，对 414-token 需求显式断言
  `_materialize_tokens(...) == 0`；再将上限设为 414，显式断言返回 414 并进入
  materialize fast path。该批同时包含 384 cached + 17 current 的请求和 13-token
  cold 请求，使用非恒等 K/V 正交旋转。Packed mixed-KV Triton 物化、单次 causal
  FA4、V 逆旋转以及后续 cache store 的整体输出通过 FP32 reference，
  `atol=rtol=0.015`。

因此新 kernel 已在 B200 全新 Triton cache 下完成冷编译，多请求 packed 寻址、
cached/cold 分支、非恒等旋转与上限选路都有定向 CUDA 证据。这足以保留当前
实现并进入独立服务镜像验证，但仍不代表 M2 已通过：512 MiB 预留对启动与
allocator 的影响、36 层实际 materialize + rotation + FA4 时间和 endpoint TTFT 均尚未实测。

容器 UTC `20:20:19`--`20:21:41`、exit0；UTC `20:21:54` 完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整
日志、分配/回收记录、exit code 与三个源文件的 SHA256 位于第 10.40 节同一
artifact 目录。下一步构建独立镜像，先复用第 10.37 节完全相同的 profiler
payload 定位生产总成本，过局部预算后再运行 TTFT M2。

### 10.43 Materialize 独立镜像与完整 CUDA 回归

首次构建组合命令在 vLLM 子仓库 workdir 中使用根目录报告的相对路径，
文档验证因 `FileNotFoundError` 立即退出，Docker build 未启动。该错误与前述
已记录的 workdir 混用属同一类；修正后所有命令固定在 workspace 根目录，
Dockerfile 和 build context 显式使用 `vllm/docker/Dockerfile.oscar-vllm` 和 `vllm/`。

修正后构建于 UTC `20:24:42`--`20:24:47` 完成、exit0。独立镜像为
`oscar-vllm:prefill-materialized-bounded-20260720`，ID
`sha256:8dd642da7e53798601f0a5ee2a67d561af50930ba9862be3164122bdabef3b65`，
entrypoint 仍为 `["vllm", "serve"]`，workdir 为 `/workspace`。

为验证镜像中实际安装的源码，本轮没有再 bind mount 或覆盖生产文件，而是直接
运行镜像内完整 `tests/quantization/test_oscar.py`。GPU 0 在 UTC `20:25:01`
和 `20:26:02` 两次检查中均为 0 MiB、利用率 0%、无计算进程，间隔
61 秒；GPU 1-7 也均空闲。固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-materialized-prefill-full`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefill-materialized-bounded-20260720` / `sha256:8dd642da...bef3b65` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

完整回归结果为 `16 passed, 75 warnings in 75.28s`，pytest exit0。这不仅覆盖
新 materialize oracle，还包含 OSCAR config、INT2 store/dequant、batched mapped decode、
prefix ownership、recent demote、chunked prefill 和 cache update 时序的全文件回归。警告均来自
PyTorch JIT/CUTLASS/FA4 弃用提示，无断言、CUDA runtime 或数值失败。

容器 UTC `20:26:36`--`20:28:27`、exit0；UTC `20:28:37` 完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。构建
日志、image inspect、两次分配检查、完整 pytest 日志、SHA256 与回收记录位于
`artifacts/oscar_vllm_optimization/20260720/materialized_prefill_image/` 和
`artifacts/oscar_vllm_optimization/20260720/materialized_prefill_full_cuda/`。镜像与完整
CUDA 门禁已通过，下一步进入同 payload 生产 profiler；M2 状态仍为未通过。

### 10.44 受限 Materialize 同 Payload 生产 Profiler

本轮使用第 10.37 节 GQA2 profiler 完全相同的三份 token-ID payload 和调用顺序，
其 SHA256 分别为 `0bfa9352...6077ecd`、`0c33cf27...8c42cf3` 和
`3e762929...2b92f`。GPU 0 在 UTC `20:31:10` 和 `20:32:11` 两次检查中均为
0 MiB、利用率 0%、无计算进程，间隔 61 秒；GPU 1-7 也均空闲。本轮固定物理
GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`，服务参数与 GQA2 基线相同，仅端口改为
8442，并启用同配置的 PyTorch worker profiler。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefill-materialized-bounded-20260720` / `sha256:8dd642da...bef3b65` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / FA | 3.6.0 / 5.13.0 / FA4 |
| `CUDA_VISIBLE_DEVICES` | `0` |

服务 UTC `20:33:00` 启动，`20:35:28` 首次观测 health 200。启动日志仍报告
INT2 history 461936 tokens（9.91 GiB）、BF16 prefix 128 tokens（0.02 GiB）、
BF16 recent 512 tokens（0.07 GiB）、unused 0，规划容量仍为 461936 tokens；停止前
进程总显存为 20510 MiB。因此 512 MiB 全局共享 workspace 没有降低显式 10 GiB
OSCAR KV allocator 的规划容量。Vector cold/hit、matrix cold、被 profile 的 matrix
hit 和被 profile 的 tensor cold 均为 HTTP 200；cold 的 cached tokens 为 0，hit 为
1840，prompt tokens 均为 2100。Hit 日志明确出现
`OSCAR bounded materialized prefill active`，证明生产 fast path 实际生效。

同进程 trace 汇总如下；CPU 值包含 profiler 开销，只用于同配置归因，不能替代
未 profile 的 endpoint TTFT：

| 指标 | GQA2 cold | GQA2 hit | Materialize cold | Materialize hit |
| --- | ---: | ---: | ---: | ---: |
| Worker CPU total | 42.281 ms | 54.410 ms | 40.743 ms | 53.025 ms |
| 普通 kernel Self CUDA | 19.640 ms | 24.712 ms | 19.644 ms | 7.996 ms |
| Attention CPU total | 18.891 ms | 29.978 ms | 19.491 ms | 32.619 ms |
| Attention CUDA total | 5.763 ms | 20.640 ms | 5.776 ms | 3.888 ms |

Materialize kernel 在 hit trace 中正好调用 36 次，累计 0.491421 ms、平均
0.013651 ms/层；hit 的 `aten::mm` 为 289 次、CUDA 累计 4.074 ms。相对 GQA2，
hit attention CUDA 从 20.640 ms 降到 3.888 ms，减少 16.752 ms/81.163%，加速
5.309x；普通 kernel Self CUDA 从 24.712 ms 降到 7.996 ms，减少 16.716 ms。
更关键的是 hit-cold attention CUDA 差由 `+14.877 ms` 变为 `-1.888 ms`，普通
kernel Self CUDA 差由 `+5.072 ms` 变为 `-11.648 ms`。因此第 10.37 节定义的 GPU
局部预算已经通过，当前 materialize 结构在 GPU 执行层面足以跨过 fused GQA2 的瓶颈。

但 worker CPU 的 hit-cold 差仍为 `+12.282 ms`，与 GQA2 的 `+12.129 ms` 接近，
attention CPU 差还从 `+11.087 ms` 增至 `+13.128 ms`。这可能包含 profiler 对额外
框架算子的放大，也可能反映真实 host 调度开销；仅靠本轮不能区分。因此 M2 仍不判
通过，下一步必须用第 10.36 节相同未 profile TTFT client、相同 warm-up 和 CV 追加
规则进行真实服务复测。若 hit 仍慢，再从 trace 的 CPU operator 差异入手消除逐层
metadata/workspace/Python 开销，而不是继续优化已过预算的 materialize GPU kernel。

Profiler 预热 UTC `20:36:30`--`20:36:39`；hit profile 于 `20:36:39`--`20:36:46`
完成，cold profile 于 `20:36:46` 完成。服务 UTC `20:38:31` 请求停止，容器最终
exit0、非 OOM，严格错误扫描无命中；UTC `20:38:40` 完整 `nvidia-smi` 显示 GPU
0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整请求响应、trace、表格、SHA256、
服务日志、两次分配检查和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/torch_profile_materialized_bounded/`。

### 10.45 受限 Materialize 真实服务 TTFT M2

本轮复用第 10.36 节原始 `benchmark.py`，两份文件 SHA256 均为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`。
负载仍为每对 1840-token 公共前缀加 260-token 不同后缀、2 对 warm-up、首轮
5 对；任一侧 CV 超过 3% 时自动追加到 15 对。为满足实验 Python 容器化要求，
client 使用相同 materialized 镜像的独立无 GPU 容器，通过 host network 访问服务；
脚本只读挂载且未修改。服务参数与第 10.36 节相同。

GPU 0 在 UTC `20:43:55` 与 `20:45:11` 两次检查中均为 0 MiB、利用率 0%、
无计算进程，间隔 76 秒；GPU 1-7 也均空闲。本轮固定物理 GPU 0，服务容器内
`CUDA_VISIBLE_DEVICES=0`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefill-materialized-bounded-20260720` / `sha256:8dd642da...bef3b65` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | `0` |

服务 UTC `20:45:52` 启动，`20:48:13` 首次观测 health 200；正式 client 于
`20:48:37`--`20:48:49` 完成、exit0。首轮 CV 超过 3%，因此脚本按既定规则自动
追加到 15 对。结果如下：

| 路径 | Mean TTFT | Population std | CV | 请求数 |
| --- | ---: | ---: | ---: | ---: |
| Cold / 0 cached tokens | 35.931 ms | 2.428 ms | 6.758% | 15 |
| Hit / 1840 cached tokens | 47.036 ms | 1.796 ms | 3.819% | 15 |

30 个正式请求均为 HTTP 200、prompt tokens 均为 2100；所有 cold 请求均报告
0 cached tokens，所有 hit 请求均报告 1840 cached tokens。15/15 对都是 hit 更慢，
配对差 `hit-cold` 均值为 `+11.105 ms`、population std 为 `3.007 ms`；hit 均值
比 cold 慢 `30.906%`。前 5 对也已失败：cold/hit 为 35.057/47.129 ms，配对差
`+12.072 ms`，因此追加样本没有改变方向。服务日志明确出现 bounded materialized
fast path 提示，不能把结果解释为误走 GQA2 fallback。

结论是 **M2 仍失败**。第 10.44 节已经证明 hit 的 GPU attention 成本相对 GQA2
下降 81.163%，但这没有转化为 endpoint 命中快于 cold；与 profiler 中 worker CPU
仍有约 12.282 ms hit-cold 差的现象一致。GQA2 服务的 cold/hit 为 39.079/49.456 ms，
本轮两侧绝对值分别低 3.148/2.420 ms，但独立服务轮次和 client 容器边界不同，不能
把跨服务差值直接归因为 materialize 收益；可靠的同进程结论仍是 hit 慢 11.105 ms。
下一步应比较 materialize cold/hit trace 的 CPU operator 次数和耗时，先消除每层
metadata 判定、workspace 视图及同步相关 host 开销，再重跑同一 M2，而不是继续优化
已经过局部预算的 GPU kernel。

辅助 paired JSON 首次用 jq 汇总时因 `length` 作用域误指根对象，打印了不可能的
18.508 ms；原始 `stream_result.json` 未受影响。该派生文件已从 15 个逐对差重算并
覆盖为上述 11.105 ms，错误过程在 artifact 中保留说明。服务 UTC `20:50:31`
请求停止，最终 UTC `20:50:39` exit0、非 OOM，严格错误扫描为空；UTC `20:50:56`
完整 `nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，服务容器已删除。
完整 client、原始响应、汇总、服务日志、环境、两次分配检查与回收记录位于
`artifacts/oscar_vllm_optimization/20260720/prefill_materialized_bounded_ttft/`。

### 10.46 Materialize Cold/Hit CPU Operator 差分

从第 10.44 节已经落盘的同进程 key-averages 表做差，不重新运行 GPU。Materialize
hit 相对 cold 的关键 operator 变化如下：

| Operator | Cold calls | Hit calls | 增量 | Cold CPU total | Hit CPU total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `aten::mm` | 217 | 289 | +72 | 6.035 ms | 7.269 ms |
| `aten::to` | 267 | 411 | +144 | 2.130 ms | 4.088 ms |
| `aten::_to_copy` | 115 | 259 | +144 | 1.991 ms | 3.840 ms |
| `aten::copy_` | 200 | 344 | +144 | 2.373 ms | 3.598 ms |
| `aten::sub` | 37 | 109 | +72 | 0.413 ms | 1.403 ms |
| `cudaLaunchKernel` | 302 | 518 | +216 | 1.961 ms | 3.117 ms |
| `cuLaunchKernelEx` | 301 | 445 | +144 | 1.972 ms | 2.710 ms |

这些 CPU total 存在父子嵌套，不能相加成总瓶颈；调用次数则能与代码逐层对应。
每层 Q 旋转和 V 逆旋转当前都执行 `BF16 -> FP32`、FP32 matmul、`FP32 -> BF16`，
36 层正好额外产生 72 次 mm 与 144 次 dtype copy。Materialize kernel 前每层又重新
计算 `query_lens` 和 `cached_lens`，正好产生 72 次 `sub`。WorkspaceManager 每层
生成两个 view，对应额外 slice/view，但这部分 CPU total 较小，且直接缓存 view 会影响
ubatch workspace ownership，当前不先改。

官方 OSCAR-SGLang 代码还提供把 V 旋转吸收到 QKV 权重的可选实现；该方向会跨越
模型 load-weight、量化权重布局和 checkpoint 兼容边界，不能在尚未验证更小改动时直接
移植。当前先实施两个可由现有 trace 和 oracle 闭环的最小候选：metadata 一次性生成
`cached_lens`；每层首次加载 rotation 时同时缓存 BF16 的 K-query rotation 与 V inverse
矩阵，让这两个热路径 matmul 输入/矩阵/输出都保持 BF16。预期消除 hit 每层 2 次
`sub` 和 4 次 dtype copy；能否满足 `atol=rtol=0.015` 及是否改善 TTFT，必须由非恒等
rotation CUDA oracle 和独立服务实测决定，本节不提前宣称收益。

### 10.47 Cached-Lens 与 BF16 Rotation 候选静态门禁

已实施第 10.46 节的两个局部候选。`OscarMetadataBuilder` 新增一个
`max_num_seqs` 长度的 INT32 `cached_lens` 缓冲，每批由 final sequence lengths 与
query start locations 一次性计算；纯 prefill、mixed decode+prefill 和 GQA2 fallback
均透传同一 metadata。Materialize Q 旋转和 V 逆旋转改用 layer 首次加载时按模型
dtype 缓存的 rotation 副本，FP32 原矩阵仍保留给 cache store、decode 和 fused fallback，
因此没有扩大数值变更范围。非恒等 rotation mixed oracle 已显式传入 BF16 fast 矩阵
与 `[384, 0]` cached lengths，后续 CUDA 测试会实际覆盖新路径。

无 GPU 静态检查使用 `oscar-vllm:prefill-materialized-bounded-20260720` 容器。首轮
pycompile 与 Ruff lint 通过，format check 要求对 `oscar_attn.py` 和 `test_oscar.py`
机械排版；Ruff formatter 处理后，两个文件的 pycompile、lint 和 format check 全部
通过。首轮包装脚本错误读取 pipeline 中 `tee` 的退出位，导致 artifact `.exit` 写成
0，不能用该文件代表 format check 状态；实际日志明确记录两个文件需重排。修正后的
run2 使用 `PIPESTATUS[0]`，真实 exit0。完整日志位于
`artifacts/oscar_vllm_optimization/20260720/materialized_host_fastpath/`。

本节只证明语法、lint 和格式成立。BF16 rotation 相对 FP32 reference 的误差、
cached-lens metadata 的 packed 寻址以及 fallback 回归尚未通过 GPU 验证，当前候选
不能保留；下一步按两次空闲检查后固定物理 GPU 0，使用全新 Triton cache 运行
poisoned fused fallback 与 mixed materialize CUDA oracle。

### 10.48 Cached-Lens 与 BF16 Rotation CUDA Oracle

GPU 0 在 UTC `20:58:36` 与 `20:59:52` 两次检查中均为 0 MiB、利用率 0%、
无计算进程，间隔 76 秒；GPU 1-7 也均空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-materialized-host-fastpath`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:prefill-materialized-bounded-20260720` / `sha256:8dd642da...bef3b65` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

定向结果为 `2 passed, 14 deselected, 61 warnings in 60.79s`，pytest exit0。
Poisoned prefix-hit oracle 覆盖 GQA2 fused fallback；mixed oracle 同时覆盖 384-token
cached + 17-token current 请求与 13-token cold 请求、非恒等 K/V 正交旋转、显式
`cached_lens=[384, 0]`、BF16 Q/V fast rotation、packed materialize、单次 causal FA4、
cache store 和 FP32 reference。整体输出继续通过 `atol=rtol=0.015`，说明 BF16
rotation 的额外舍入在当前定向 shape 和随机正交矩阵下仍在既定误差界内；这不等于
完整模型精度回归，后续阶段 7 仍需 GSM8K 终验。

容器 UTC `21:00:41`--`21:01:52`、exit0、非 OOM；UTC `21:02:07` 完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。当前候选
通过实现级正确性门禁，可以构建独立镜像做 operator trace 和 M2；尚无性能证据，
不能根据预期减少的调用次数宣称 endpoint 收益。完整日志、两次分配检查、容器状态、
源码 SHA256 与回收记录位于
`artifacts/oscar_vllm_optimization/20260720/materialized_host_fastpath/`。

### 10.49 Host-Fastpath 独立镜像

文档 10.1--10.48 顺序唯一、交叉引用未越界、冲突标记和行尾空白检查通过，vLLM
子仓库 `git diff --check` 也通过。随后使用当前工作树和
`vllm/docker/Dockerfile.oscar-vllm` 构建独立镜像，UTC `21:03:15`--`21:03:21`
完成、exit0。镜像为 `oscar-vllm:materialized-host-fastpath-20260720`，完整 ID
`sha256:31784757c2eef578e8e5ae42c0481e2062ff1ebe908795b59f5043e83c37d7e2`；
entrypoint 仍为 `['vllm', 'serve']`，workdir 为 `/workspace`。

本轮构建未分配 GPU。完整 build log 和 image inspect 位于
`artifacts/oscar_vllm_optimization/20260720/materialized_host_fastpath_image/`。
镜像只证明源码已固化，尚未验证镜像内完整 OSCAR CUDA 回归；下一步重新执行两次
GPU 空闲检查后，直接运行镜像内 `tests/quantization/test_oscar.py`，不 bind mount
或覆盖源码。

### 10.50 Host-Fastpath 独立镜像完整 CUDA 回归

完整回归的首次前置组合命令在任何 GPU 检查前被文档门禁拒绝：第 10.49 节的补丁
锚点命中了第 10.47 节相同 artifact 路径，使标题暂时成为 10.47、10.49、10.48。
该命令没有写出 `gpu_check_1_time.txt`，未分配 GPU。重新读取实际段落后只移动整节，
恢复 10.1--10.49 严格连续，并再次通过冲突标记、行尾空白和 `git diff --check`。

GPU 0 在 UTC `21:05:54` 与 `21:07:05` 两次检查中均为 0 MiB、利用率 0%、
无计算进程，间隔 71 秒；GPU 1-7 也均空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-materialized-host-fastpath-full`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-host-fastpath-20260720` / `sha256:31784757...37d7e2` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |

直接运行镜像内完整 `tests/quantization/test_oscar.py`，未 bind mount 或覆盖源码。
结果为 `16 passed, 75 warnings in 75.44s`，pytest exit0。警告仍为 JIT、CUTLASS、
FA4 弃用提示，无断言、数值或 CUDA runtime 失败。这覆盖 config、INT2 store/dequant、
batched mapped decode、prefix ownership、recent demote、chunked prefill、cache update、
GQA2 fallback 和新增 materialize fast path。

容器 UTC `21:07:38`--`21:09:30`、exit0、非 OOM；UTC `21:09:44` 完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整日志、
两次分配检查、容器状态和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/materialized_host_fastpath_full_cuda/`。
正确性回归已闭环，下一步用同 SHA256 client 复测 TTFT M2；若仍失败，再启动同 payload
profiler 确认预期 operator 消除及剩余 host 成本。

### 10.51 Host-Fastpath Prefix-Hit TTFT M2 复测

本轮三份 `benchmark.py` 的 SHA256 均为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`，负载、
2 对 warm-up、首轮 5 对及 CV 超过 3% 自动追加到 15 对的规则均未修改。Client
继续使用同镜像独立无 GPU 容器和 host network。GPU 0 在 UTC `21:10:42` 与
`21:11:55` 两次检查中均为 0 MiB、利用率 0%、无计算进程，间隔 73 秒；GPU 1-7
也均空闲。本轮固定物理 GPU 0，服务容器内 `CUDA_VISIBLE_DEVICES=0`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-host-fastpath-20260720` / `sha256:31784757...37d7e2` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | `0` |

服务 UTC `21:12:20` 启动，`21:14:29` 首次观测 health 200；正式 client 于
`21:14:41`--`21:14:59` 完成、exit0。首轮 CV 超过 3%，自动追加到 15 对：

| 路径 | Mean TTFT | Population std | CV | 请求数 |
| --- | ---: | ---: | ---: | ---: |
| Cold / 0 cached tokens | 33.203 ms | 2.211 ms | 6.660% | 15 |
| Hit / 1840 cached tokens | 40.088 ms | 2.119 ms | 5.285% | 15 |

30 个请求均 HTTP 200、prompt tokens 均为 2100；cold 全部 0 cached tokens，hit
全部 1840 cached tokens，日志明确出现 bounded materialized fast path。15/15 对
仍是 hit 更慢，配对差均值 `+6.884 ms`、population std `1.976 ms`，hit 均值比
cold 慢 `20.734%`。前 5 对 cold/hit 为 32.557/38.765 ms，配对差 `+6.208 ms`，
追加样本没有改变方向。因此 **M2 仍失败**。

与第 10.45 节相同 client 方法相比，配对 gap 从 11.105 ms 缩到 6.884 ms，名义缩小
38.006%；但两轮为独立服务，cold/hit 绝对值也同时下降，且两轮 CV 都超过 3%，不能
只凭跨轮差值证明全部 4.221 ms 来自本次实现。现有证据足以说明候选没有恶化且值得
继续归因，但不足以宣告统计收益。下一步使用第 10.44 节相同 payload/profile 顺序采集
operator trace：若 144 次 dtype copy 与 72 次 `sub` 确实消失，则保留实现并针对剩余
workspace view、materialize launch 或 rotation weight absorption 评估下一步；否则先修正
fast path 未命中的实现问题。

服务 UTC `21:16:13` 请求停止，最终 `21:16:22` exit0、非 OOM，严格错误扫描为空；
UTC `21:16:37` 完整 `nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，
容器已删除。完整原始响应、汇总、paired 数据、服务日志、环境、分配检查和回收记录
位于 `artifacts/oscar_vllm_optimization/20260720/prefill_materialized_host_fastpath_ttft/`。

### 10.52 Host-Fastpath 同 Payload Operator Profiler

三份 payload SHA256 与第 10.44 节完全相同，调用顺序仍为 vector cold/hit、matrix
cold、被 profile 的 matrix hit、被 profile 的 tensor cold。GPU 0 在 UTC `21:18:06`
与 `21:19:23` 两次检查中均为 0 MiB、利用率 0%、无计算进程，间隔 77 秒；GPU
1-7 也均空闲。本轮固定物理 GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`，环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-host-fastpath-20260720` / `sha256:31784757...37d7e2` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / FA | 3.6.0 / 5.13.0 / FA4 |
| `CUDA_VISIBLE_DEVICES` | `0` |

服务 UTC `21:20:03` 启动，`21:22:07` 首次 health 200；所有五个推理请求和四个
profiler endpoint 均 HTTP 200，cold/hit cached tokens 严格为 0/1840。Hit profile
于 `21:23:01`--`21:23:07` 完成，cold profile 于 `21:23:07` 完成。生产日志确认
bounded materialized fast path 生效。

收集脚本末尾错误假定每轮都会新增 trace 和独立 profiler text 两个文件。实际
`profiler_out_0.txt` 在 cold 轮覆盖 hit 轮的同名文本，因此 cold 轮只新增第二份 trace，
文件数断言 exit1；两份 trace、全部 HTTP 响应和 cold text 都已完整落盘，未重复请求。
以下结果直接由两份 Chrome trace 的原始 `cpu_op`、`kernel` 和 `user_annotation`
事件汇总，不依赖被覆盖的 hit text。

| 指标 | Cold | Hit | Hit-Cold |
| --- | ---: | ---: | ---: |
| 主 worker CPU | 41.537 ms | 49.790 ms | +8.253 ms |
| Attention CPU total | 19.639 ms | 28.922 ms | +9.282 ms |
| 全部 GPU kernel | 19.388 ms | 6.815 ms | -12.574 ms |
| Materialize kernel | 0 次 | 36 次 / 0.487615 ms | +0.487615 ms |

相对第 10.44 节，worker CPU gap 从 12.282 ms 降至 8.253 ms，减少 4.029 ms/
32.805%；attention CPU gap 从 13.128 ms 降至 9.282 ms，减少 3.846 ms。更重要的
调用次数验证如下：旧 hit/cold 的 `to`、`_to_copy`、`copy_`、`sub` 分别为
411/267、259/115、344/200、109/37；本轮全部变成 267/267、115/115、200/200、
39/39。即原有 hit 额外的 144 次 dtype conversion/copy 和 72 次逐层 cached-lens
`sub` 已全部消失，证明第 10.46--10.48 节候选实际进入生产路径，应予保留。

剩余结构也很明确：hit/cold `mm` 为 289/217，仍多 72 次，正好对应每层一次 Q
rotation 与一次 V inverse；同时 `view` 为 834/690，多 144 次。72 次 matmul 的
CPU total 增量为 1.293 ms，其父级 `matmul` 增量为 1.509 ms；此外每层 materialize、
rotation dispatcher/driver launch 和 Python 调用都包含在 attention 的 9.282 ms gap 中。
GPU kernel hit 已比 cold 少 12.574 ms，所以继续微调 materialize GPU kernel 不会解决
M2。下一结构候选必须减少这 72 次 rotation matmul/launch：Q rotation 发生在 RoPE
之后，不能错误地吸收到 q/k projection 权重；可评估将 Q rotation 融入 materialize
kernel。V 路径没有 RoPE，可审计将 V rotation 与 inverse 成对吸收到 v/o projection
权重的 TP=1 dense 实现，但必须先证明模型权重布局和完整精度。

服务 UTC `21:26:56` 请求停止，最终 `21:27:05` exit0、非 OOM，严格错误扫描为空；
UTC `21:27:20` GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整 trace、
派生 operator JSON、请求响应、服务日志、环境、分配检查和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/torch_profile_materialized_host_fastpath/`。

### 10.53 Materialize 单次 Triton 旋转候选与静态门禁

第 10.52 节剩余的 72 次 `mm` 对应 36 层各一次 Q rotation 与 V inverse。Q/K
projection 之后仍有 Q/K RMSNorm 与 RoPE，且 Q rotation 必须作用于 RoPE 后的 Q；
一般旋转矩阵不与这两个算子交换。因此把 Q/K rotation 直接吸收到 projection 权重会
改变算法语义，本轮明确不采用。V/O 权重的成对吸收虽然代数上可能成立，但会跨越
模型加载、量化权重布局与输出投影边界，也不与本轮低风险候选同时实施。

本轮复用既有 `_inverse_v_rotation_kernel`，新增 BF16 输入/输出的
`oscar_rotate_bf16` wrapper，并仅把受限 materialize 路径的 Q rotation 与 V inverse
从 `torch.matmul` 切换为该 wrapper。每处旋转由一个二维 Triton grid 完成，目标是将
每层两次高层 matmul 及其底层 launch/view 开销压缩为两次直接 kernel launch。Decode
路径、GQA2 fused fallback、cache allocator、workspace 上限和旋转矩阵内容均未修改。

静态检查在无 GPU 容器中使用
`oscar-vllm:materialized-host-fastpath-20260720`，依次对 backend、decode Triton 模块和
OSCAR 测试运行 pycompile、Ruff lint 与 Ruff format check；结果为 `All checks passed!`、
`3 files already formatted`、exit0。原始日志位于
`artifacts/oscar_vllm_optimization/20260720/materialized_triton_rotation/`。

当前证据只证明代码可导入且满足项目静态规范，**尚不证明数值正确或性能改善**。
下一步必须用新 Triton cache 运行 poisoned GQA2 fallback 与 mixed cached/cold、非恒等
rotation CUDA oracle；通过后才允许构建独立镜像和复测 TTFT/operator trace。若 oracle
不满足既有 `atol=rtol=0.015`，将只撤回本节的 wrapper 切换。

### 10.54 单次 Triton 旋转 CUDA Oracle

GPU 0 在 UTC `21:34:07` 与 `21:35:22` 两次检查中均为 0 MiB、利用率 0%、
无计算进程，间隔 75 秒；GPU 1-7 也均空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-materialized-triton-rotation`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-host-fastpath-20260720` / `sha256:31784757...37d7e2` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | `0` |

为避免源码目录遮蔽镜像内已编译扩展，本轮只把当前 backend、decode Triton 模块复制到
镜像的 installed-package 路径，并把测试复制到 `/tmp` 后执行。定向结果为
`2 passed, 14 deselected, 61 warnings in 62.41s`，pytest exit0。Poisoned
prefix-hit oracle 覆盖 GQA2 fused fallback；mixed oracle 覆盖 384-token cached +
17-token current 请求与 13-token cold 请求、非恒等 K/V 正交旋转、显式
`cached_lens=[384, 0]` 和受限 materialize FP32 reference。因此本节新增的 Q rotation
与 V inverse 单次 Triton launch 在既有 `atol=rtol=0.015` 门槛下通过，且没有改变
fallback 语义。

容器 UTC `21:36:25`--`21:37:40`、exit0、非 OOM；UTC `21:37:55` 完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整日志、
两次分配检查、镜像与容器 inspect、完整 `nvidia-smi` 和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/materialized_triton_rotation/`。

正确性候选保留，但性能仍未验证。下一步构建不依赖源码覆盖的独立镜像，并运行完整
OSCAR CUDA 回归；全部通过后才复用相同 SHA256 TTFT client 与 operator payload，确认
72 次 `mm` 是否真实消失以及 M2 配对差是否改善。

### 10.55 单次 Triton 旋转独立镜像

在 workspace 根目录以当前 vLLM tree 为 build context，使用
`vllm/docker/Dockerfile.oscar-vllm` 构建独立镜像
`oscar-vllm:materialized-triton-rotation-20260720`。构建 UTC `21:40:01`--
`21:40:07`、exit0；不可变 image ID 为
`sha256:c03219a504e9a010c306f86aef20891f1d6896226c71c634215a63db4ae93a97`。
该阶段未分配 GPU。构建日志与 image inspect 位于
`artifacts/oscar_vllm_optimization/20260720/materialized_triton_rotation_image/`。

下一步不再覆盖任何镜像内源码，直接在该 image ID 上运行完整 `test_oscar.py` CUDA
回归。这样可以同时验证 Docker build context 确实包含新 wrapper，并排除定向测试遗漏
其他 decode、store、prefix-cache 或 chunked-prefill 路径回归的风险。

### 10.56 单次 Triton 旋转独立镜像完整 CUDA 回归

GPU 0 在 UTC `21:41:02` 与 `21:42:10` 两次检查中均为 0 MiB、利用率 0%、
无计算进程，间隔 68 秒；GPU 1-7 也均空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-materialized-triton-rotation-full`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-triton-rotation-20260720` / `sha256:c03219a5...ae93a97` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | `0` |

本轮不挂载或覆盖源码，直接运行镜像内完整 `tests/quantization/test_oscar.py`。结果为
`16 passed, 75 warnings in 75.35s`、pytest exit0。容器 UTC `21:42:36`--
`21:44:18`、exit0、非 OOM；UTC `21:44:30` 完整 `nvidia-smi` 显示 GPU 0-7
均为 0 MiB、0%、无计算进程，容器已删除。完整 pytest 日志、两次分配检查、容器
inspect、完整 `nvidia-smi` 和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/materialized_triton_rotation_full_cuda/`。

至此静态检查、非恒等定向 oracle 和独立镜像完整 CUDA 回归均通过。候选进入性能验证，
但 **M2 状态仍为失败**；下一步必须复用第 10.51 节完全相同 SHA256 的 TTFT client，
在同一服务内比较 cold/hit 配对结果，不能用 CUDA 单测时长推断服务性能。

### 10.57 单次 Triton 旋转 Prefix-Hit TTFT M2 复测

本轮 `benchmark.py` SHA256 为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`，与第
10.51 节逐字节一致；负载、2 对 warm-up、首轮 5 对及 CV 超过 3% 自动追加到 15 对
的规则均未修改。Client 使用同镜像独立无 GPU 容器和 host network。GPU 0 在 UTC
`21:46:08` 与 `21:47:25` 两次检查中均为 0 MiB、利用率 0%、无计算进程，间隔
77 秒；GPU 1-7 也均空闲。本轮固定物理 GPU 0，服务容器内
`CUDA_VISIBLE_DEVICES=0`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-triton-rotation-20260720` / `sha256:c03219a5...ae93a97` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | `0` |

服务 UTC `21:48:03` 启动，`21:50:09` 首次 health 200；正式 client 于
`21:50:32`--`21:50:44` 完成、exit0。首轮 cold/hit CV 为 4.664%/4.318%，因此
按规则自动追加到 15 对：

| 路径 | Mean TTFT | Population std | CV | 请求数 |
| --- | ---: | ---: | ---: | ---: |
| Cold / 0 cached tokens | 35.817 ms | 2.902 ms | 8.102% | 15 |
| Hit / 1840 cached tokens | 44.724 ms | 2.966 ms | 6.632% | 15 |

30 个请求均 HTTP 200、prompt tokens 均为 2100；cold 全部 0 cached tokens，hit
全部 1840 cached tokens，生产日志明确出现 bounded materialized prefill。15/15 对
均为 hit 更慢，配对差均值 `+8.907 ms`、population std `2.034 ms`，范围
`+6.770`--`+15.141 ms`；hit 均值比 cold 慢 `24.867%`。因此 **M2 仍失败**。

与第 10.51 节独立服务相比，配对 gap 从 6.884 ms 名义增加到 8.907 ms，增加
2.022 ms/29.375%；但两轮 cold/hit 绝对均值都变化，且 CV 均超过 3%，不能把跨服务
差值直接归因为 wrapper 回退。可靠结论仅是本轮同服务内 hit 明确慢于 cold，候选没有
带来 endpoint 级 M2 改善。下一步必须用第 10.52 节相同 payload/profile 顺序确认
72 次 `mm` 是否消失，并量化新 Triton rotation 的 CPU launch 与 GPU 时间；如果
operator trace 没有净收益，应撤回本候选而保留第 10.52 节 host-fastpath 基线。

服务 UTC `21:51:26` 请求停止，最终 `21:51:35` exit0、非 OOM，停止前和最终严格
错误扫描均为 0 条；同刻 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整
请求、逐对统计、服务日志、环境、分配检查与回收记录位于
`artifacts/oscar_vllm_optimization/20260720/prefill_materialized_triton_rotation_ttft/`。

### 10.58 单次 Triton 旋转同 Payload Operator Profiler 与否决

三份 payload SHA256 与第 10.52 节完全相同：vector/matrix/tensor 分别为
`0bfa9352...6077ecd`、`0c33cf27...b8c42cf3`、`3e762929...a2b92f`；调用顺序仍为
vector cold/hit、matrix cold、被 profile 的 matrix hit、被 profile 的 tensor cold。
GPU 0 在 UTC `21:54:16` 与 `21:55:29` 两次检查中均为 0 MiB、利用率 0%、
无计算进程，间隔 73 秒；GPU 1-7 也均空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-triton-rotation-20260720` / `sha256:c03219a5...ae93a97` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / FA | 3.6.0 / 5.13.0 / FA4 |
| `CUDA_VISIBLE_DEVICES` | `0` |

服务 UTC `21:56:12` 启动，`21:58:18` 首次 health 200。五个推理请求与四个
profiler endpoint 均 HTTP 200；vector cold/hit、matrix cold/hit、tensor cold 的
cached tokens 严格为 0/1840、0/1840、0。Hit trace 于 UTC `21:59:28`--`21:59:35`
采集，cold trace 于 `21:59:35`--`21:59:37` 采集。两份 trace 均完整落盘。

| 指标 | Cold | Hit | Hit-Cold |
| --- | ---: | ---: | ---: |
| 主 worker CPU | 40.510 ms | 52.812 ms | +12.302 ms |
| Attention CPU total | 19.924 ms | 29.251 ms | +9.327 ms |
| 全部 GPU kernel | 19.416 ms | 10.131 ms | -9.285 ms |
| Materialize kernel | 0 次 | 36 次 / 0.486169 ms | +0.486169 ms |
| 新 Triton rotation | 0 次 | 72 次 / 3.569770 ms | +3.569770 ms |

高层 operator 的确按预期变化：hit/cold 的 `mm` 与 `matmul` 均从 host-fastpath 的
289/217 降为 217/217；`view` 从 834/690 降为 762/690，`reshape` 从 291/147
降为 219/147，`_unsafe_view` 从 144/72 降为 72/72。但这不等于减少实际 launch：
hit/cold GPU kernel 数仍为 822/642；`cuLaunchKernelEx` 仍为 445/301，
`cudaLaunchKernel` 仍为 304/304，`cudaLaunchKernelExC` 仍为 73/37，与第 10.52 节
host-fastpath trace 完全相同。原先“一个 Triton wrapper 会减少底层 launch”的假设
被实测否定；它只改变了调用栈和 kernel 实现。

更关键的是，host-fastpath 的 72 次 BF16 rotation GEMM 累计仅 0.241569 ms，而新
Triton rotation 72 次累计 3.569770 ms，慢 `14.777x`。相同 payload 下 hit GPU
总时间从 6.814657 ms 增至 10.130711 ms，增加 3.316054 ms；worker hit-cold gap
从 8.252913 ms 增至 12.301763 ms，attention gap 从 9.282290 ms 增至
9.326954 ms。该方向也与第 10.57 节 endpoint M2 失败一致。因此本候选**明确否决**：
撤回 `oscar_rotate_bf16` wrapper 及 materialize 两处调用切换，恢复第 10.52 节的
BF16 `torch.matmul`；此前 cached lengths 预计算与 BF16 rotation 缓存继续保留。

服务 UTC `22:02:05` 请求停止，最终 `22:02:14` exit0、非 OOM，停止前和最终
严格错误扫描均为 0 条；同刻 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。
完整 trace、派生 operator JSON、请求响应、服务日志、环境、分配检查与回收记录位于
`artifacts/oscar_vllm_optimization/20260720/torch_profile_materialized_triton_rotation/`。

### 10.59 单次 Triton 旋转候选撤回闭环

按第 10.58 节决定，仅删除 `oscar_rotate_bf16` wrapper 及其 import，并把受限
materialize 路径的 Q rotation、V inverse 恢复为缓存 BF16 rotation 上的
`torch.matmul`。Decode、GQA2 fallback、cached lengths 预计算、BF16 rotation 缓存、
allocator、workspace 与测试均未修改。

无 GPU 容器内对 backend、decode Triton 模块和 OSCAR 测试运行 pycompile、Ruff
lint 与 Ruff format check，结果为 `All checks passed!`、`3 files already formatted`、
exit0。首次 SHA256 对照发现 decode 已匹配 host-fastpath 镜像，但 backend 仅因 import
被压成单行而不同；只读导出镜像文件确认唯一 diff 后，恢复原 trailing-comma 多行
import 并重跑全部静态门禁。最终当前 backend/decode SHA256 分别为
`4243f203...606479`、`e502e44d...030ec7`，与
`oscar-vllm:materialized-host-fastpath-20260720` 镜像内文件逐字节一致。

该阶段未分配 GPU。由于恢复后两个生产文件与第 10.50 节已完成 16 项 CUDA 回归、
第 10.51--10.52 节已完成服务 TTFT/profiler 的镜像内容字节相同，不重复运行相同 GPU
实验。撤回日志、静态结果、镜像文件导出、diff 与 SHA256 对照位于
`artifacts/oscar_vllm_optimization/20260720/materialized_triton_rotation_revert/`。
当前有效基线回到第 10.52 节 host-fastpath；M2 仍失败。

### 10.60 官方 V Rotation Absorption 语义审计

本轮只读审计本地论文官方 SGLang 配套代码，未分配 GPU。官方 README 的推荐服务命令
显式设置 `SGLANG_OSCAR_ABSORB_V_ROTATION=1`；对应实现位于
`sglang/srt/models/utils.py::maybe_absorb_oscar_v_rotation_into_qkv`。Dense 路径按每个
KV head 把 `R_v.T` 左乘到 fused QKV 权重的 V slice，并同步旋转 V bias；Q/K 权重不变。
它要求在模型权重加载后、量化权重重排前执行，对不支持的权重布局明确 fail-fast。

需要纠正一个容易误解的点：官方实现**没有修改 O projection，也没有删除 attention
输出的 inverse**。`prepare_qkv_for_quantized_prefill` 在 absorption 打开时跳过 runtime
V rotation，但仍设置 `need_v_inverse=True`；FA/Triton backend 随后继续执行
`result @ R_v.T`。因此官方 absorption 的作用是让 V projection 直接生成已旋转的 V，
省掉写 cache 前的 V rotation，而不是把整个 V/O 旋转对都消掉。

映射到当前 vLLM 路径，语义要求如下：

- Cold prefill：投影后的 V 已旋转，FA 输出后必须新增一次 inverse；cache store 跳过
  V rotation，因此一增一减，launch 数大致不变。
- Materialized prefix hit：当前已经在 FA 后 inverse；只需让 current V 直接进入
  materialize/store，能够实减一次 V rotation launch/层。
- Fused cached-prefill fallback：suffix 与 cached attention 必须都先保持在旋转域，
  merge 完成后再统一 inverse，不能把旋转域和原始域的 output 直接合并。
- Decode：cache 中 V 保持旋转域，既有 output inverse 继续保留。

下一候选严格复现官方 qkv-only absorption，不扩展到未经官方验证的 O projection 折叠。
新增环境开关默认关闭；首阶段仅支持当前目标范围 TP=1、dense BF16/FP16/FP32 Qwen3
权重，LoRA、量化权重或不一致布局 fail-fast。验证顺序为 CPU 权重代数、backend
absorbed/non-absorbed oracle、独立镜像完整 CUDA、同 SHA TTFT 与同 payload trace。

### 10.61 V Rotation Absorption 失败用例

本阶段先新增两组约束，未修改生产实现。CPU 测试要求
`VLLM_OSCAR_ABSORB_V_ROTATION` 能严格解析 true/false 并拒绝非法值，同时验证 dense
fused-QKV 仅修改 V slice 和 V bias，且折叠后的 V projection 等价于原始输出右乘
`R_v`；不支持的权重布局必须 fail-fast。CUDA mixed cached/cold oracle 则增加
absorbed/non-absorbed 参数，absorbed 输入显式模拟已经输出 `V @ R_v` 的模型投影，
最终结果仍须匹配原始域 FP32 reference。

使用稳定镜像 `oscar-vllm:materialized-host-fastpath-20260720` 的无 GPU 容器运行新增
CPU 测试。测试在收集阶段按预期失败，错误为无法从镜像内
`rotation.py` 导入尚未实现的 `absorb_v_rotation_into_qkv`；pytest exit 非零。该结果
证明 red 用例确实覆盖新能力，不是既有实现上的假阳性。此阶段没有分配 GPU，也没有
产生性能或精度结论。原始日志位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_red_cpu/`。

下一步实现默认关闭的配置开关、官方 dense QKV 折叠代数和 backend 旋转域处理；先让
CPU 测试与静态门禁通过，再按 GPU 分配规范运行 absorbed/non-absorbed CUDA oracle。

### 10.62 V Rotation Absorption CPU 实现与静态门禁

生产实现新增默认关闭的 `VLLM_OSCAR_ABSORB_V_ROTATION`。开关只在 `oscar_int2`
Qwen3 上生效，并限制为 TP=1、无 LoRA、无权重量化的 dense BF16/FP16/FP32 布局；
模型权重加载后逐层把 `R_v.T` 左乘到 fused QKV 的 V slice，并旋转 V bias，然后在
attention layer 写入 absorbed 标记。不支持的配置或权重几何直接报错，不静默跳过。

Backend 同时校验配置开关与 layer 标记一致。Absorbed 路径写 cache 时直接使用模型投影
产生的 rotated V；cold prefill 在 FA/SDPA 后 inverse；materialized hit 保留已有统一
inverse；fused fallback 让 suffix/cached output 都在旋转域合并，再做一次 inverse；
decode 保持原有 inverse。O projection、K/Q rotation、allocator、workspace 和 cache
格式均未修改。

无 GPU 容器仍使用稳定镜像 `oscar-vllm:materialized-host-fastpath-20260720`，把当前
config/rotation 覆盖到 installed package 后执行。首轮 `py_compile` 通过，Ruff lint
只报告新增 CPU 测试两行 E501；拆行后第二轮 pycompile/lint 通过，format check 要求
三个生产文件机械排版，pytest 尚未运行。按 formatter 精确 diff 排版后，第三轮结果为：

- 6 个相关文件 pycompile exit0；
- Ruff lint `All checks passed!`；
- Ruff format `6 files already formatted`；
- CPU pytest `10 passed, 1 skipped, 12 subtests passed in 1.93s`，exit0。

该阶段未分配 GPU。结果证明配置解析、dense 权重/偏置折叠代数和静态接口成立，但不
证明 CUDA backend 各旋转域路径正确，也不证明真实 Qwen3 权重加载成功或性能改善。
完整三轮日志位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_cpu_static/`。下一步检查并补足
fused fallback oracle，再运行 GPU 定向测试。

### 10.63 V Rotation Absorption CUDA 旋转域 Oracle

在第 10.62 节参数化 mixed cached/cold 用例中补充一次真实低 cap `forward`：先用
克隆 cache 执行 GQA2 fused fallback，再用原 cache 执行 bounded materialized。两条
路径共用 384-token cached + 17-token current 请求与 13-token cold 请求、非恒等 K/V
正交旋转和原始域 FP32 reference。结合 absorbed/non-absorbed 参数，实际覆盖四种
组合，而不再只是断言 fallback cap 为 0。补测后的 pycompile、Ruff lint/format 与
CPU pytest 全部再次通过。

GPU 0 在 UTC `22:22:36` 与 `22:24:01` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 85 秒；GPU 1-7 同样空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-v-absorption`。环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-host-fastpath-20260720` / `sha256:31784757...37d7e2` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | `0` |

只覆盖当前 config、rotation 和 backend 到镜像 installed package 后运行定向测试，
结果为 `2 passed, 15 deselected, 120 warnings in 55.50s`、pytest exit0。两个参数实例
均分别执行 fused 与 materialized，因此四个旋转域组合都在既有
`atol=rtol=0.015` 下匹配 FP32 reference。该结果同时证明 absorbed fused 路径没有
把旋转域 suffix 与原始域 cached output 直接合并。

容器 UTC `22:24:39`--`22:25:42`、exit0、非 OOM；UTC `22:26:06` 完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整日志、两次
GPU 检查、容器/镜像 inspect 和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_cuda_oracle/`。

该阶段证明 backend 数值语义，但仍通过源码覆盖，尚未证明 Qwen3 的真实权重加载 hook
会折叠全部 36 层。下一步先用真实模型启动验证标记与权重加载，再构建不覆盖源码的
独立镜像完成全量 CUDA 回归和服务性能测量。

### 10.64 V Rotation Absorption 独立镜像

在 vLLM 子仓库根目录以当前 tree 为 build context，使用
`docker/Dockerfile.oscar-vllm` 构建独立镜像
`oscar-vllm:materialized-v-absorption-20260720`。构建 exit0；不可变 image ID 为
`sha256:7cbc8db3ba107467c075f56e0af06230225ba7dd5505ed0a28795a8db590797e`，entrypoint
保持 `vllm serve`，工作目录为 `/workspace`。

对 config、rotation、Qwen3 model、OSCAR backend 和两份测试逐文件比较镜像内/本地
SHA256，六项全部一致，排除 Docker build context 遗漏当前改动。该阶段未分配 GPU。
构建日志、image inspect 与两侧 SHA256 位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_image/`。

下一步按新一轮 GPU 连续空闲检查固定单卡，直接运行镜像内完整 OSCAR CUDA 回归，
不挂载或覆盖源码；通过后再用相同镜像启动真实 Qwen3 服务，验证 36 层权重折叠。

### 10.65 V Rotation Absorption 独立镜像完整 CUDA 回归

GPU 0 在 UTC `22:28:41` 与 `22:30:06` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 85 秒；GPU 1-7 同样空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用全新
`TRITON_CACHE_DIR=/tmp/triton-cache-v-absorption-full`。环境与第 10.63 节相同，
但镜像切换为 `oscar-vllm:materialized-v-absorption-20260720` /
`sha256:7cbc8db3...590797e`。

本轮不挂载或覆盖任何源码，直接运行镜像内完整
`tests/quantization/test_oscar.py`。结果为
`17 passed, 134 warnings in 90.03s`、pytest exit0；相对旧基线的 16 项，多出的 1 项
来自 absorbed 参数实例。回归覆盖 store/demote/decode、scheduler BF16 ownership、
prefix hit、chunked tier 一致性、poisoned shared-hit、fused/materialized mixed prefill
及 absorbed/non-absorbed 语义。

容器 UTC `22:30:38`--`22:32:29`、exit0、非 OOM；UTC `22:32:40` 后的完整
`nvidia-smi` 显示 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整 pytest、
两次分配检查、container inspect 和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_full_cuda/`。

独立镜像 backend 正确性闭环，但测试仍手工构造 absorbed layer 标记。下一步必须用
同一镜像启动真实 Qwen3 服务并启用环境开关，确认权重加载日志显示全部 36 层折叠、
首个 cold/hit 请求成功，再进入 TTFT M2 与 operator trace。

### 10.66 V Rotation Absorption 真实加载与 Prefix-Hit TTFT

本轮复用第 10.51 节字节相同的 TTFT client，SHA256 为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`；负载仍为
1840-token 公共 prefix + 260-token suffix、2 对 warm-up、首轮 5 对，任一 CV 超过
3% 时自动追加到 15 对。服务参数保持 10 GiB KV budget、8192 上下文、TP=1 eager、
prefix caching、chunked prefill 和 `max_num_seqs=2`，仅新增
`VLLM_OSCAR_ABSORB_V_ROTATION=1`。

GPU 0 在 UTC `22:34:57` 与 `22:36:24` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 87 秒；GPU 1-7 同样空闲。本轮固定物理 GPU 0，使用第 10.64 节独立镜像，
环境版本与第 10.63 节相同。服务 UTC `22:37:08` 启动，真实权重加载日志显示
V rotation checkpoint 共 36 层，并明确输出
`Absorbed OSCAR V rotation into 36 Qwen3 QKV layers`；权重加载 9.41 秒。10 GiB cache
仍规划 INT2 history 461936 tokens、BF16 prefix 128、BF16 recent 512，说明 absorption
没有改变物理 cache 容量。Health 于 UTC `22:39:36` 首次返回 200。

正式 client UTC `22:39:49`--`22:40:05`、exit0。首轮 CV 触发追加，最终结果为：

| 路径 | Mean TTFT | Population std | CV | 请求数 |
| --- | ---: | ---: | ---: | ---: |
| Cold / 0 cached tokens | 36.472 ms | 1.079 ms | 2.958% | 15 |
| Hit / 1840 cached tokens | 43.818 ms | 1.964 ms | 4.483% | 15 |

30 个请求均 HTTP 200、prompt tokens 均为 2100；cold 全部 0 cached tokens，hit 全部
1840 cached tokens，生产日志确认 bounded materialized prefill 生效。15/15 对均为
hit 更慢，配对差均值 `+7.346 ms`、population std `2.351 ms`，范围
`+0.919`--`+12.014 ms`；hit 均值比 cold 慢 `20.142%`。因此 **M2 仍失败**。

相对第 10.51 节 host-fastpath 的 `+6.884 ms`，本轮配对 gap 名义增加约 0.462 ms；
但本轮 hit CV 为 4.483%，且两者是独立服务，不能把该小差异归因于 absorption。可靠
结论仅是本轮同服务内 prefix hit 仍显著慢于 cold，官方 absorption 没有带来 endpoint
级门槛改善。下一步必须用同 payload operator trace 检查预期减少的逐层 matmul。

启动日志还出现 `Unknown vLLM environment variable detected:
VLLM_OSCAR_ABSORB_V_ROTATION`。开关实际被 config 读取并完成 36 层折叠，但该告警证明
环境变量注册尚不完整；在下一镜像前必须补入 `vllm.envs`。服务最终 exit0、非 OOM，
严格错误扫描为空；UTC `22:40:54` 后 GPU 0-7 均为 0 MiB、0%、无计算进程，容器已
删除。完整 client、统计、服务日志、inspect 和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/prefill_materialized_v_absorption_ttft/`。

### 10.67 V Rotation Absorption 环境变量注册修正

在 `vllm.envs` 的类型声明与 lazy `environment_variables` 注册表中新增
`VLLM_OSCAR_ABSORB_V_ROTATION`，接受 `1/true/yes/on` 为 true，默认 false。Config
原有严格解析仍负责拒绝非法值；本修正只让 vLLM 环境变量审计识别该开关，不改变
absorption、attention、allocator 或性能路径。

无 GPU 容器使用第 10.64 节镜像并只覆盖当前 envs/config/rotation。7 个相关文件的
pycompile exit0，Ruff lint 为 `All checks passed!`，Ruff format 为
`7 files already formatted`；显式断言 `dir(vllm.envs)` 包含新开关且值为 true 通过。
`validate_environ` 输出只剩基础镜像已有的 4 个 build metadata 告警，不再报告
absorption 开关。CPU pytest 为
`10 passed, 1 skipped, 12 subtests passed in 1.51s`、exit0。

该阶段未分配 GPU。日志位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_env_registration/`。下一步
重建修正版独立镜像；由于唯一生产差异是 env 注册，不重复数值 CUDA 回归，但 profiler
服务必须直接使用新镜像并确认启动日志不再出现 absorption unknown 告警。

### 10.68 环境变量注册修正版独立镜像

以当前 tree 构建 `oscar-vllm:materialized-v-absorption-env-20260720`，build exit0；
不可变 image ID 为
`sha256:8dcc4ece9cbebfb611a342e666d522eaa33094b270a2057dd392881297ac8bb3`。
镜像内 `vllm/envs.py` SHA256 为 `89b4b531...45310`，与本地逐字节一致。无 GPU
镜像自检中，新开关解析为 true，`validate_environ` 只报告基础镜像 4 个 build
metadata 变量，不再报告 absorption unknown。

需要记录一个编排问题：构建组合命令先在 vLLM 子仓库工作目录读取 workspace 根目录
报告，文档门禁因路径错误失败；该 shell 未设置 `set -e`，因此 build 仍继续并成功。
这不影响镜像内容，但该次文档检查无效。随后已回到 workspace 根目录独立重跑，报告
章节 10.1--10.67 严格连续、无冲突标记或行尾空白，实际通过。后续不再把根目录文档
检查与子仓库 build 放在同一 workdir 命令中。

该阶段未分配 GPU。构建、image inspect、envs SHA256 和镜像自检日志位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_env_image/`。下一步用本镜像
和第 10.52 节相同 payload/profile 顺序运行 operator profiler，验证 absorption 是否
按预期减少每层一次 hit matmul，并量化其 CPU/GPU 净收益。

### 10.69 V Rotation Absorption 同 Payload Profiler 与 Stride 缺口

三份 payload 及调用顺序与第 10.52 节完全相同；matrix、tensor、vector payload 的
SHA256 分别为 `0c33cf27...b8c42cf3`、`3e762929...a2b92f`、
`0bfa9352...6077ecd`。GPU 0 在 UTC `22:47:42` 与 `22:49:07` 两次检查中均为
0 MiB、利用率 0%、无计算进程，间隔 85 秒；GPU 1-7 同样空闲。本轮固定物理 GPU 0，
容器内 `CUDA_VISIBLE_DEVICES=0`，环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-v-absorption-env-20260720` / `sha256:8dcc4ece...ac8bb3` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / FA | 3.6.0 / 5.13.0 / FA4 |
| `CUDA_VISIBLE_DEVICES` | `0` |

服务 UTC `22:49:58` 启动，`22:52:19` 首次 health 200；日志确认 36 层 V
absorption、461936-token INT2 history 容量和 bounded materialized path 均生效，且
不再出现 absorption unknown 告警。五个推理请求与四个 profiler endpoint 全部
HTTP 200，prompt tokens 均为 2100，cached tokens 严格为
`0/1840/0/1840/0`。Hit profile 位于 UTC `22:53:15`--`22:53:20`，cold profile 位于
`22:53:40`--`22:53:46`。两份 Chrome trace 的实测汇总为：

| 指标 | Cold | Hit | Hit-Cold |
| --- | ---: | ---: | ---: |
| 主 worker CPU | 44.736 ms | 48.202 ms | +3.465 ms |
| Attention CPU total | 22.693 ms | 28.129 ms | +5.437 ms |
| 全部 GPU kernel | 19.755 ms | 6.839 ms | -12.916 ms |
| Materialize kernel | 0 次 | 36 次 / 0.481567 ms | +0.481567 ms |

Absorption 的预期结构变化确实发生：相对第 10.52 节，hit `mm/matmul` 从 289 次降为
253 次，cold 仍为 217 次，即 hit 每层一次 V rotation 共 36 次已消失。与此同时，
hit/cold 的 `copy_` 均从 200 次升到 272 次，GPU `cudaLaunchKernel` 均从 304 次升到
376 次，kernel 数从 822/642 升到 858/714。精确 trace 对照显示新增的是每条路径
72 次 `clone/empty/empty_like/copy_` 及对应 BF16 copy kernel；因此当前 absorption
虽减少了 hit 的 36 次 matmul，却又在 cold 与 hit 各引入 72 次额外拷贝 launch。

源码回溯发现这不是单纯性能问题，而是一个必须先修的正确性缺口。真实 Qwen3 fused
QKV 输出经 split 后，absorption 路径跳过 runtime V matmul，因此 current V 保持为
**非连续切片**；K 经过 rotation matmul 后是连续 tensor。现有
`oscar_materialize_prefill_kv` 只接收 current K 的 token/head stride，并把同一组
stride 错用于 current V，所以真实 materialized hit 会按错误地址读取 current V。
第 10.63/10.65 节 absorbed CUDA oracle 先用独立 matmul 构造 `V @ R_v`，得到的是
连续 tensor，因而没有覆盖这一真实布局。此前 oracle 对旋转域合并的结论仍成立，但
不能作为非连续 fused-QKV V 布局的正确性证据；第 10.66 节 TTFT 与本节 profiler 也
只能作为路径和开销诊断，不能证明 absorption materialized 输出正确或 M2 改善。

额外 72 次拷贝来自 `oscar_store` 与 `oscar_store_hp` 分别对同一个非连续 V 调用
`.contiguous()`。下一步不改变 cache 格式或算法语义，只让 store、HP store 与
materialize 三个 Triton kernel 分别接收 K/V 的 token/head/dim stride，直接读取
真实非连续视图，并新增显式非连续 fused-QKV V oracle。修复必须先通过该失败用例、
完整 CUDA 回归与真实服务正确性检查，之后才允许重新测量 TTFT/profiler。

服务 UTC `22:57:48` 请求停止，`22:57:53` exit0、非 OOM；严格错误扫描只有基础镜像
4 个 build metadata unknown 告警，没有 absorption unknown 或运行时错误。GPU 0-7
最终均为 0 MiB、0%、无计算进程，容器已删除。完整 trace、派生 JSON、请求响应、
日志、inspect、两次分配检查和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/torch_profile_materialized_v_absorption_env/`。

### 10.70 非连续 Fused-QKV V 失败 Oracle

本阶段只修改测试，不修改生产 kernel。Absorbed 参数先计算正确的 rotated V，再把它
写入宽度为 `(num_query_heads + 2 * num_kv_heads) * head_dim` 的 fused-QKV backing
tensor，最后通过 V slice 构造 `[tokens, kv_heads, head_dim]` view。测试显式断言该
view 非连续，且 token stride 与连续 K 不同。无 GPU 容器中的 pycompile、Ruff lint
和 Ruff format check 均通过。

GPU 0 在 UTC `23:00:59` 与 `23:02:55` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 116 秒；GPU 1-7 同样空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用第 10.68 节镜像和全新
`TRITON_CACHE_DIR=/tmp/triton-cache-v-absorption-stride-red`；其余 GPU、Driver、
Python、vLLM、PyTorch、CUDA、Triton 和 Transformers 版本均与第 10.69 节相同。

只运行 absorbed 参数。Fused GQA2 fallback 先通过最终 FP32 reference 断言；随后
materialized 输出按预期失败，`23436 / 30720` 个元素不匹配，占 76.3%，并出现 NaN。
pytest 结果为 `1 failed, 120 warnings in 50.22s`、exit1。该定位说明失败不是 V
absorption 的权重代数或 fused 旋转域合并，而是 materialize 对当前非连续 V 的地址
计算错误，直接验证了第 10.69 节的源码判断。

容器 UTC `23:03:07`--`23:04:09`、exit1 符合 red 预期、非 OOM；容器已删除，最终
GPU 0-7 均为 0 MiB、0%、无计算进程。完整测试、静态门禁、两次 GPU 检查、镜像与
容器 inspect、回收记录位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_stride_red/`。下一步修改三个
Triton 入口，使 K/V 各自使用真实 token/head/dim stride；同一用例转绿前不进入
服务或性能复测。

### 10.71 独立 K/V Stride 实现与静态门禁

本轮只修改三个 Triton 输入入口及对应 CUDA 测试，不改变 cache 物理布局、INT2
量化、rotation、attention、allocator 或 workspace。`oscar_store` 不再 reshape、
contiguous 并预转 FP32，而是把 K/V 各自的 token/head/dim stride 传给 kernel，读取后
再转 FP32 量化；`oscar_store_hp` 同样删除两次 `.contiguous()`，按独立 stride 直接
读取并转 BF16；materialize kernel 则拆分原先共用的 current stride，分别计算 K/V
地址。三个入口的 token/head base 均使用 int64。

测试除第 10.70 节真实 fused-QKV V view 外，还把 INT2 roundtrip 与 mixed HP
store/decode 的 K/V 改为具有不同 padding 和不同 stride 的非连续 view，使 store、
HP store 和 materialize 都有数值覆盖。首次无 GPU 静态检查中 pycompile、Ruff lint
通过，format check 仅要求机械重排 store 和 HP store 两个文件；应用 formatter 后，
又把 HP source base 显式提升为 int64并修正旧注释。最终四个文件结果为：

- pycompile exit0；
- Ruff lint `All checks passed!`；
- Ruff format `4 files already formatted`；
- 子仓库 `git diff --check` exit0。

静态命令实际由 `printf | sudo docker | tee` 三段组成，因此 round1 的
`PIPESTATUS[1]=1` 正确记录了 Docker format failure。Round2/round3 曾改读
`PIPESTATUS[0]`，这实际对应 password `printf`，虽与完整通过日志一致，却不是可靠的
Docker 退出码。复核后新增无 pipeline 的 round4，直接捕获 Docker exit0，输出仍为
`All checks passed!` 和 `4 files already formatted`；最终静态结论以 round4 为准。

该阶段未分配 GPU。最终四文件 SHA256、四轮日志和机械 formatter 记录位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_stride_fix_static/`。下一步
按新一轮连续空闲检查固定 GPU，用同一 absorbed 非连续参数转绿，并同时运行两个
非连续 store 数值用例；成功前不扩大到完整 CUDA 回归。

### 10.72 独立 K/V Stride 定向 CUDA Green

GPU 0 在 UTC `23:09:37` 与 `23:11:36` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 119 秒；GPU 1-7 同样空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用第 10.68 节镜像并只读覆盖三个当前 Triton 文件和
测试文件；全新 `TRITON_CACHE_DIR` 为
`/tmp/triton-cache-v-absorption-stride-green`。镜像、GPU、Driver、Python、vLLM、
PyTorch、CUDA、Triton 和 Transformers 版本均与第 10.70 节相同。

本轮明确运行 5 个实例：head_dim 64/128 的非连续 INT2 store/dequant roundtrip、
非连续 mixed store/demote/decode，以及 absorbed/non-absorbed 两个 mixed prefill
参数；后两项各自仍同时执行低 cap fused fallback 和高 cap materialized。结果为
`5 passed, 120 warnings in 80.96s`、pytest exit0。原先第 10.70 节失败的 absorbed
非连续 fused-QKV V materialized 路径现已在相同 `atol=rtol=0.015` 下匹配原始域
FP32 reference；不同 K/V stride 的 INT2 与 BF16 HP store 也通过各自数值 oracle。

该定向结果证明三个 stride 入口的数值修复成立，但当前仍通过只读源码覆盖，不能代替
独立镜像完整回归，也尚未用 operator trace 证明 72 次 copy launch 已消失。容器 UTC
`23:11:50`--`23:13:24`、exit0、非 OOM；容器已删除，UTC `23:13:28` GPU 0-7 均为
0 MiB、0%、无计算进程。完整 pytest、两次 GPU 检查、mount/容器 inspect 和回收记录
位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_stride_fix_cuda_targeted/`。
下一步构建独立镜像并核对三个生产文件与测试的 SHA256，再运行无源码覆盖的完整 OSCAR
CUDA 回归。

### 10.73 独立 K/V Stride 修复镜像

在 vLLM 子仓库根目录以当前 tree 为 build context，UTC `23:14:45`--`23:14:51`
构建 `oscar-vllm:materialized-v-absorption-stride-20260720`，build exit0。不可变
image ID 为
`sha256:578c5a9bf1e6f21b935e1a64254b4cc030494cb252676a54db86fecb111251b5`，entrypoint
保持 `vllm serve`，工作目录为 `/workspace`。

对 INT2 store、BF16 HP store、materialize 和完整 CUDA 测试逐文件比较镜像内/本地
SHA256，四项分别为 `91f21802...9c418`、`ecdc6fd0...c4bb25`、
`e898fc3a...c6d686`、`b37f863a...8ac25`，全部逐字节一致。该阶段未分配 GPU。
构建日志、image inspect、两侧 SHA256 和汇总位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_stride_fix_image/`。下一步
按新一轮 GPU 连续空闲检查固定物理 GPU，直接运行镜像内完整 OSCAR CUDA 回归，不
挂载或覆盖源码。

### 10.74 独立 K/V Stride 镜像完整 CUDA 回归

GPU 0 在 UTC `23:15:55` 与 `23:17:36` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 101 秒；GPU 1-7 同样空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用第 10.73 节独立镜像和全新
`TRITON_CACHE_DIR=/tmp/triton-cache-v-absorption-stride-full`；其余环境版本与
第 10.72 节相同。

本轮不挂载或覆盖任何源码，直接运行镜像内完整
`tests/quantization/test_oscar.py`。结果为
`17 passed, 134 warnings in 104.95s`、pytest exit0。回归覆盖非连续 store/dequant、
mixed store/demote/decode、scheduler BF16 ownership、prefix hit、chunked tier、
poisoned shared-hit、fused/materialized mixed prefill，以及 absorbed/non-absorbed
真实非连续 V 语义。

容器 UTC `23:17:51`--`23:19:59`、exit0、非 OOM、无源码 mount；容器已删除，UTC
`23:20:03` GPU 0-7 均为 0 MiB、0%、无计算进程。完整 pytest、两次 GPU 检查、
container inspect 和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/v_absorption_stride_fix_full_cuda/`。
独立镜像 CUDA 正确性已经闭环；下一步必须用同一镜像启动真实 Qwen3 服务，确认 36 层
权重折叠、prefix hit 与 materialized 路径均生效，再重新测量 TTFT 和 operator trace。

### 10.75 Stride 修复真实服务与 Prefix-Hit TTFT

本轮复用第 10.66 节字节相同的 TTFT client，SHA256 仍为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`；负载、
warm-up、首轮 5 对和 CV 超过 3% 时扩展到 15 对的规则均未改变。服务与 HTTP client
分别运行在 GPU 容器和独立无 GPU 容器中。GPU 0 在 UTC `23:21:34` 与 `23:23:12`
两次检查中均为 0 MiB、利用率 0%、无计算进程，间隔 98 秒；GPU 1-7 同样空闲。

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:materialized-v-absorption-stride-20260720` / `sha256:578c5a9b...111251b5` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers / FA | 3.6.0 / 5.13.0 / FA4 |
| `CUDA_VISIBLE_DEVICES` | `0` |

服务 UTC `23:23:27` 启动，真实权重加载日志确认 36 层 V absorption；10 GiB cache
仍规划 INT2 history 461936 tokens、BF16 prefix 128、BF16 recent 512。独立无 GPU
health client 于 UTC `23:26:02` 首次得到 200。正式 client UTC
`23:26:22`--`23:26:32`、exit0；首轮 CV 触发扩展，最终结果为：

| 路径 | Mean TTFT | Population std | CV | 请求数 |
| --- | ---: | ---: | ---: | ---: |
| Cold / 0 cached tokens | 35.287 ms | 2.213 ms | 6.272% | 15 |
| Hit / 1840 cached tokens | 44.355 ms | 3.008 ms | 6.781% | 15 |

30 个请求均 HTTP 200、prompt tokens 均为 2100；cold 全部 0 cached tokens，hit 全部
1840 cached tokens，日志确认 bounded materialized path 生效。15/15 对均为 hit
更慢，配对差均值 `+9.068 ms`、population std `2.637 ms`，范围
`+3.060`--`+12.520 ms`；hit 均值比 cold 慢 25.698%。因此 **M2 仍失败**。

第 10.66 节旧 absorption TTFT 使用了错误 V stride，不能作为数值正确路径的性能
基线；相对第 10.51 节正确的 host-fastpath `+6.884 ms`，本轮 gap 名义增加
2.184 ms，但本轮两侧 CV 均超过 6%，且属于独立服务，不能把差异归因于 stride 读取
或 absorption。可靠结论仅是修复后的同服务 prefix hit 仍显著慢于 cold。下一步必须
用同 payload trace 验证预期的 72 次 copy/launch 是否消失，并重新定位剩余 host gap。

服务 UTC `23:26:52` 请求停止，`23:27:05` exit0、非 OOM；严格运行时错误扫描为空，
环境审计只剩基础镜像 4 个 build metadata unknown 告警，不含 absorption 开关。UTC
`23:27:08` GPU 0-7 均为 0 MiB、0%、无计算进程，容器已删除。完整 client、统计、
日志、inspect、两次分配检查和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/prefill_materialized_v_absorption_stride_ttft/`。

### 10.76 Stride 修复同 Payload Operator Profiler

三份 payload SHA256 与第 10.69 节完全相同，请求顺序仍为 vector cold/hit、matrix
cold、被 profile 的 matrix hit、被 profile 的 tensor cold。GPU 0 在 UTC
`23:28:36` 与 `23:30:08` 两次检查中均为 0 MiB、利用率 0%、无计算进程，间隔
92 秒；GPU 1-7 同样空闲。本轮固定物理 GPU 0，镜像和全部环境版本与第 10.75 节
相同；使用全新 `TRITON_CACHE_DIR=/tmp/triton-cache-v-absorption-stride-profiler`
和与第 10.69 节相同的 Torch profiler 配置。

本轮新增的无 GPU HTTP runner 在发送请求前经过 pycompile/Ruff。首轮 Ruff 只要求
删除一处 import 后的空行，第二轮 format check 只要求机械展开 expected list；最终
结果为 `All checks passed!`、`1 file already formatted`。服务 UTC `23:30:24`
启动，`23:32:59` health 200；正式 runner UTC `23:35:16`--`23:35:30`、exit0。
五个推理请求和四个 profiler endpoint 全部 HTTP 200，prompt tokens 均为 2100，
cached tokens 严格为 `0/1840/0/1840/0`。Hit profile 位于 UTC
`23:35:25.946`--`23:35:29.506`，cold profile 位于
`23:35:29.770`--`23:35:30.116`。两份 trace 汇总为：

| 指标 | Cold | Hit | Hit-Cold |
| --- | ---: | ---: | ---: |
| 主 worker CPU | 38.042 ms | 43.582 ms | +5.540 ms |
| Attention CPU total | 18.373 ms | 25.860 ms | +7.487 ms |
| GPU kernel + memcpy | 19.055 ms | 6.552 ms | -12.503 ms |
| Materialize kernel | 0 次 | 36 次 / 0.479933 ms | +0.479933 ms |

相对第 10.69 节错误 stride 实现，cold/hit 两条路径的 operator 变化完全一致：`to`
减少 72 次、`_to_copy` 减少 36 次、`copy_` 减少 108 次、`clone` 减少 72 次，
`empty/empty_like/reshape` 各减少 72 次，`view/_unsafe_view` 各减少 36 次；
`cudaLaunchKernel` 与 GPU kernel 均减少 108 次。即每层实际删除了三次 device copy：
INT2 store 的 V contiguous、V BF16-to-FP32，以及 HP store 的 V contiguous。此前只按
两个 `.contiguous()` 估计 72 次，低估了中间 dtype copy；本 trace 给出了完整证据。

相对第 10.52 节正确 host-fastpath，new hit/cold 的 `mm` 为 253/217，hit 已少 36 次
V rotation；`cudaLaunchKernel` 从 304/304 降到 268/268，GPU kernel 从 822/642
降到 750/606。Host-fastpath 的 worker gap 8.253 ms 降到 5.540 ms，减少
2.713 ms/32.868%；attention gap 9.282 ms 降到 7.487 ms，减少
1.795 ms/19.339%。因此 stride 直读和 absorption 都应保留；它们显著减少了结构性
调用，但没有让 M2 通过。

剩余结构为 hit 比 cold 多 36 次 Q rotation `mm/matmul`、36 次 materialize launch，
以及 materialize/workspace/FA 调用周边的 Python dispatcher；GPU hit 仍比 cold 快
12.503 ms，所以继续微调 materialize GPU 时间不是主方向。更重要的是，当前 bounded
materialize 仍构造全 active batch BF16 K/V，与计划中的“serving mixed prefill 不
构造全历史 BF16 K/V”硬标准冲突。下一阶段应回到 fused cached-prefill 设计，针对
host launch/dispatcher 数和现有 GQA2 kernel 做结构优化，而不是继续扩大 materialize
workspace 或用已被第 10.58 节否决的慢 Triton rotation 替代 GEMM。

服务 UTC `23:38:40` 请求停止，`23:38:43` exit0、非 OOM；严格运行时错误扫描为空，
仅有基础镜像 4 个 build metadata unknown 告警。UTC `23:38:46` GPU 0-7 均为
0 MiB、0%、无计算进程，容器已删除。完整 trace、runner、派生 summary/comparison、
响应、日志、inspect、两次分配检查和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/torch_profile_materialized_v_absorption_stride/`。

### 10.77 Fused Absorbed BF16 Output 失败 Oracle

Fused cached-prefill 审计发现：`oscar_cached_prefill_attention` 当前固定分配 FP32
output；V rotation 已吸收时，backend 随后立即把它转换为 suffix 的 BF16 再执行 LSE
merge。候选语义限定为让该 absorbed 分支直接从 Triton kernel 写 BF16；非 absorbed
分支仍保留 FP32 output 和 V inverse。为避免先改实现再补测试，本轮只在现有 poisoned
prefix-hit oracle 中增加 `output_dtype=torch.bfloat16` 调用，并要求返回 dtype 为 BF16、
输出匹配既有 FP32 reference、LSE 与 FP32 输出路径逐元素一致。

测试文件先通过 `git diff --check`。首次无 GPU 静态容器错误假定镜像解释器位于
`/usr/local/bin/python`，在 pycompile 前 exit127，未形成代码检查结果；探测到实际路径
为 `/usr/bin/python3` 后，不重复错误命令，pycompile、Ruff lint 和 format check 全部
通过。

GPU 0 在 UTC `23:44:57` 与 `23:46:35` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 98 秒；GPU 1--7 同样空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用
`oscar-vllm:materialized-v-absorption-stride-20260720`，其 image ID 与第 10.73 节
一致。实际环境为 NVIDIA B200 183359 MiB、Driver 595.71.05、`nvidia-smi` CUDA
13.2、Python 3.12.13、vLLM 0.25.0、PyTorch 2.11.0+cu130 / CUDA 13.0、Triton
3.6.0、Transformers 5.13.0。

仅运行 `test_prefix_hit_does_not_read_unowned_recent_row`，结果为
`1 failed, 16 deselected in 47.03s`、pytest exit1。唯一失败发生在新增调用入口：
`TypeError: oscar_cached_prefill_attention() got an unexpected keyword argument
'output_dtype'`；既有 FP32 cached-prefill、随机非单位 V inverse 和此前 decode reference
均已先执行通过。因此这是一项边界清楚的失败 oracle，尚未产生 BF16 数值结果，也不能
据此声称性能收益。

容器 exit1、`OOMKilled=false`，失败符合预期；UTC `23:47:53` GPU 0--7 已全部恢复
0 MiB、0%、无计算进程，容器已删除。两次完整 `nvidia-smi`、静态门禁、pytest
日志/退出码、镜像与容器 inspect、版本和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/fused_absorbed_bf16_output_red/`。下一步只实现
可选 output dtype，并先让同一 oracle 转绿；在 operator trace 证明 copy/launch 消失前，
不宣称热路径成本已经降低。

### 10.78 Fused Absorbed BF16 Output 实现与静态门禁

生产改动只涉及 cached-prefill wrapper 和 backend 两处。Wrapper 新增默认值为 `None`
的 `output_dtype`；默认调用仍按既有行为分配 FP32 output。指定 dtype 时，Triton kernel
内部的 FP32 accumulator、softmax 和 LSE 均不变，仅最终 `tl.store` 写入目标 output
dtype。若调用方仍要求 `v_rotation_t` 且目标不是 FP32，wrapper 明确抛出 `ValueError`，
避免非 absorbed 路径在 V inverse 前静默降精度或触发不清晰的 GEMM dtype 错误。

Backend 仅在 `_v_rotation_absorbed(layer)` 为真时传入 `suffix_output.dtype`；这一路径
不再执行 FP32 cached output 到 suffix dtype 的转换。非 absorbed 路径不传该参数，仍由
kernel 输出 FP32、执行原有 FP32 V inverse，再在 merge 前按需转换为 suffix dtype。
因此候选不改变 cache layout、GQA2 tile/warps、direct-LSE、merge、Q rotation、
materialize fallback 或 decode。

使用与第 10.77 节相同的无 GPU Docker 镜像，三份改动文件的 pycompile、Ruff lint、
Ruff format check 全部通过，输出为 `All checks passed!` 和
`3 files already formatted`；vLLM 子仓库 `git diff --check` 同样通过。本节没有分配
GPU，也尚未把静态通过解释为 CUDA 正确或性能收益。下一步重新执行两次连续空闲检查，
固定 GPU 后让第 10.77 节同一 oracle 转绿，并同时覆盖 absorbed backend fused 分支。

### 10.79 Fused Absorbed BF16 Output CUDA Oracle

GPU 0 在 UTC `23:51:26` 与 `23:52:59` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 93 秒；GPU 1--7 同样空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用第 10.77 节相同镜像和全新
`TRITON_CACHE_DIR=/tmp/triton-cache-fused-bf16-output-green`。Python、vLLM、
PyTorch/CUDA、Triton、Transformers、GPU 和 Driver 版本均与第 10.77 节相同。

本轮把当前两个生产文件复制到镜像内已安装 package，并只读挂载当前测试；因此验证的是
当前源码候选，不是独立镜像打包结果。运行第 10.77 节 direct wrapper oracle，以及
`test_chunked_prefill_merges_cached_and_current_attention_before_store` 的
absorbed/non-absorbed 两个参数实例。结果为
`3 passed, 14 deselected, 120 warnings in 63.73s`、pytest exit0。

Direct oracle 确认 kernel 可直接返回 BF16、在 `atol=rtol=0.015` 下匹配 FP32
reference，且 LSE 与既有 FP32 output 路径逐元素一致。Backend 两个参数实例均先强制
workspace 上限走 fused GQA2，再提高上限走 materialized，并最终匹配原始域 FP32
reference；这同时覆盖 absorbed BF16 direct store、non-absorbed FP32 V inverse、
direct-LSE merge 和现有 fallback。CUDA correctness 已转绿，但本节没有 operator trace，
仍不能声称预期的 36 次 dtype copy/launch 已从真实服务消失。

测试容器 UTC `23:53:49`--`23:55:31`、exit0、`OOMKilled=false`；容器已删除，UTC
`23:55:31` GPU 0--7 均恢复 0 MiB、0%、无计算进程。两次完整 `nvidia-smi`、pytest、
镜像/容器 inspect、挂载审计和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/fused_absorbed_bf16_output_green/`。下一步构建
包含该候选的独立镜像，核对镜像内/本地源码 SHA256，再做无源码覆盖完整 CUDA 回归。

### 10.80 Fused Absorbed BF16 Output 独立镜像

首次构建组合命令错误地从 vLLM 子仓库工作目录读取根目录报告，前置文档检查立即
`FileNotFoundError`、exit1，Docker build 完全未启动。该重复编排错误已记入规划文件；
随后把根目录文档门禁与子仓库 build 拆为两个独立命令。根目录门禁确认 10.1--10.79
编号连续且唯一、交叉引用有效、无冲突标记或行尾空白，子仓库 `git diff --check` 通过。

独立镜像 `oscar-vllm:fused-absorbed-bf16-output-20260720` 于 UTC
`23:57:28`--`23:57:33` 构建完成，build exit0，完整 image ID 为
`sha256:be6b7279cdfe3c8758abc68542f23d068656b7328804dc9dc5e926f15034d18b`；
entrypoint 为 `vllm serve`，工作目录为 `/workspace`。本阶段没有分配 GPU。

镜像内/本地三个目标文件的 SHA256 逐项一致：

| 文件 | SHA256 |
| --- | --- |
| `triton_oscar_prefill.py` | `c80661a4a2705a0eb431e4d5e8cc00d57062638b8e5e72f11cb4d6343f3d96c6` |
| `oscar_attn.py` | `2660ed877d11ca98d9a797d8f22ec3e2dcfe3b22492c2a01c78459785de3395b` |
| `test_oscar.py` | `44c30d9a39d724b29a3a4bcb8026bd0c05e4faac739ed05e709b1cc74b3aa161` |

构建日志、退出码、image inspect 和两侧 hash 位于
`artifacts/oscar_vllm_optimization/20260720/fused_absorbed_bf16_output_image/`。下一步
重新执行两次 GPU 空闲检查，直接运行镜像内完整 OSCAR CUDA，不挂载或覆盖源码。

### 10.81 Fused Absorbed BF16 Output 独立镜像完整 CUDA

GPU 0 在 UTC `23:58:39` 与次日 `00:00:08` 两次检查中均为 0 MiB、利用率 0%、
无计算进程，间隔 89 秒；GPU 1--7 同样空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用第 10.80 节独立镜像和全新
`TRITON_CACHE_DIR=/tmp/triton-cache-fused-bf16-output-full`；其余实际环境版本与
第 10.77 节相同。

本轮不挂载或覆盖任何源码，直接运行镜像内完整
`tests/quantization/test_oscar.py`。结果为
`17 passed, 134 warnings in 103.46s`、pytest exit0。覆盖范围包括 INT2/BF16 store、
demote/decode、scheduler ownership、prefix hit、chunked tiers、poisoned shared-hit、
absorbed/non-absorbed fused+materialized prefill，以及新增 BF16 cached-output dtype、
数值和 LSE oracle。

容器 UTC `00:00:41`--`00:02:57`、exit0、`OOMKilled=false`，mount 数为 0；容器已
删除，UTC `00:02:57` GPU 0--7 均恢复 0 MiB、0%、无计算进程。完整 pytest、两次
GPU 检查、image/container inspect 和回收记录位于
`artifacts/oscar_vllm_optimization/20260720/fused_absorbed_bf16_output_full_cuda/`。
独立镜像 CUDA correctness 已闭环；下一步启动真实服务时显式关闭 materialize
workspace，使 prefix hit 必然走 fused GQA2，再用同 payload operator trace 验证
36 层 dtype copy/launch 是否实际消失。

### 10.82 Serving 默认禁用 Materialize 失败 Oracle

第 10.81 节计划“启动时显式关闭 materialize”，但进一步源码审计确认当前没有对应
环境变量或 CLI 参数：`_MATERIALIZE_WORKSPACE_FRACTION=0.05` 硬编码在 backend。
10 GiB、8 KV heads、D=128、BF16 时容量为 131072 tokens，因此 2100-token payload
必然走 materialize；用临时源码覆盖或 monkeypatch 得到的 fused trace 不能代表真实
production serving。

计划硬标准仍明确要求 serving mixed prefill 不构造全历史 BF16 K/V，且没有获得门槛
修订。因此候选采用最小行为变化：默认 fraction 设为 0，不新增公开 opt-in 开关；既有
materialize implementation 仅保留给 direct CUDA oracle 手动覆盖
`_materialize_max_tokens`，正常服务不可达。为先证明旧行为，本轮只在 CPU 测试增加
10 GiB 配置下 `_materialize_token_capacity(...) == 0` 的断言，尚未修改生产常量。

使用第 10.80 节镜像的无 GPU Docker 容器。测试文件 pycompile、Ruff lint/format
全部通过；定向 pytest 于 UTC `00:05:29`--`00:05:45` 运行，结果为
`1 failed, 11 deselected, 2 warnings in 12.56s`、exit1。唯一失败为
`AssertionError: 131072 != 0`，精确证明现有默认值违反预期 serving contract。
容器非 OOM、已删除，本轮没有分配 GPU。

静态日志、pytest、退出码和 container inspect 位于
`artifacts/oscar_vllm_optimization/20260721/materialize_default_disabled_red/`。下一步
只把 production fraction 从 0.05 改为 0.0，让同一 CPU oracle 转绿，再重跑完整静态、
CPU 和 CUDA 门禁；这项变更本身不代表 fused TTFT 已改善。

### 10.83 Serving 默认禁用 Materialize 实现与 CPU 回归

生产修改仅把 `_MATERIALIZE_WORKSPACE_FRACTION` 从 `0.05` 改为 `0.0`，并注明
full-history materialize 仅保留给 direct correctness oracle。未新增公开配置、未删除
materialize kernel，也未修改 cached GQA2、workspace manager、cache layout、scheduler
或 decode。现有 CUDA mixed oracle 仍可通过测试内显式设置
`impl._materialize_max_tokens` 覆盖该路径。

使用第 10.80 节镜像的无 GPU Docker 容器覆盖当前 backend。Backend、CPU 测试与
CUDA 测试三文件的 pycompile、Ruff lint/format 全部通过，输出为
`All checks passed!`、`3 files already formatted`；子仓库 `git diff --check` 通过。
完整 `test_oscar_cpu.py` 于 UTC `00:06:45`--`00:07:05` 运行，结果为
`11 passed, 1 skipped, 2 warnings, 12 subtests passed in 8.41s`、pytest exit0。

新增 capacity oracle 已从第 10.82 节的 131072 转为 0；其余 config/layout、rotation
loading 和 V absorption CPU 合约未回退。容器 exit0、非 OOM、已删除，本节没有分配
GPU。静态日志、pytest、退出码和 container inspect 位于
`artifacts/oscar_vllm_optimization/20260721/materialize_default_disabled_green/`。下一步
构建 fused-default 独立镜像，核对源码 hash，再运行无覆盖 CUDA 和真实 fused profiler；
在 profiler 前不能声称 M2 或 copy/launch 指标改善。

### 10.84 Fused-Default 独立镜像

根目录文档门禁与子仓库 build 本轮保持独立执行。构建前确认 10.1--10.83 编号连续且
唯一、交叉引用有效、无冲突标记或行尾空白，子仓库 `git diff --check` 通过。

独立镜像 `oscar-vllm:fused-default-bf16-output-20260721` 于 UTC
`00:07:55`--`00:08:01` 构建完成，build exit0，完整 image ID 为
`sha256:37bc682cc1f2e1d5b95410f60c723e95b9f929239b910d21887e0e2e09bbb84e`；
entrypoint 为 `vllm serve`，工作目录为 `/workspace`。本阶段没有分配 GPU。

镜像内/本地四个目标文件的 SHA256 逐项一致：

| 文件 | SHA256 |
| --- | --- |
| `oscar_attn.py` | `8b4037b00a1cd35c25aa29f5d66cde557deac111e98bafde0732e5af7287ce27` |
| `triton_oscar_prefill.py` | `c80661a4a2705a0eb431e4d5e8cc00d57062638b8e5e72f11cb4d6343f3d96c6` |
| `test_oscar_cpu.py` | `83be4f6d8d98e7dfca9ee770fd5d9489122762620989f39285d8d23746914eec` |
| `test_oscar.py` | `44c30d9a39d724b29a3a4bcb8026bd0c05e4faac739ed05e709b1cc74b3aa161` |

构建日志、退出码、image inspect 和两侧 hash 位于
`artifacts/oscar_vllm_optimization/20260721/fused_default_bf16_output_image/`。下一步
重新执行两次 GPU 空闲检查，先跑镜像内无覆盖完整 CUDA；通过后直接使用同一镜像启动
真实 fused profiler 服务。

### 10.85 Fused-Default 独立镜像完整 CUDA

GPU 0 在 UTC `00:09:08` 与 `00:10:34` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 86 秒；GPU 1--7 同样空闲。本轮固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`，使用第 10.84 节镜像和全新
`TRITON_CACHE_DIR=/tmp/triton-cache-fused-default-full`；其余实际环境版本与
第 10.77 节相同。

本轮不挂载或覆盖源码，直接运行镜像内完整 `test_oscar.py`，结果为
`17 passed, 134 warnings in 88.37s`、pytest exit0。新增 BF16 cached output、默认
fused serving 相关代码与全部既有 CUDA oracle 均通过；测试内显式提高
`_materialize_max_tokens` 的 absorbed/non-absorbed materialize 路径也继续通过，证明
默认禁用没有删除 test-only fallback。

容器 UTC `00:11:06`--`00:13:07`、exit0、`OOMKilled=false`，mount 数为 0；容器已
删除，UTC `00:13:07` GPU 0--7 均恢复 0 MiB、0%、无计算进程。完整 pytest、两次
GPU 检查、image/container inspect 和回收记录位于
`artifacts/oscar_vllm_optimization/20260721/fused_default_bf16_output_full_cuda/`。
下一步重新执行独立的两次 GPU 空闲检查，使用同一镜像和同 SHA payload 启动真实
Torch profiler；必须从日志和 trace 同时确认 materialize 0 次、cached GQA2 36 次。

### 10.86 Fused-Default 同 Payload Operator Profiler

本轮复用第 10.76 节三份 payload 与完全相同的请求顺序。Vector、matrix、tensor
SHA256 分别为 `0bfa9352...07ecd`、`0c33cf27...42cf3`、
`3e762929...2b92f`；HTTP runner 复制后通过 pycompile、Ruff lint/format。GPU 0 在
UTC `00:14:43` 与 `00:16:14` 两次检查中均为 0 MiB、利用率 0%、无计算进程，间隔
91 秒；GPU 1--7 同样空闲。本轮固定物理 GPU 0。

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:fused-default-bf16-output-20260721` / `sha256:37bc682c...9bbb84e` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | `0` |
| Profiler | Torch CPU+CUDA，worker only，no stack/shape/memory/gzip |

服务 UTC `00:16:50` 启动，`00:19:04` health 200。日志确认 36 层 V absorption、
INT2 history 461936 tokens、BF16 prefix 128、BF16 recent 512，并且没有 materialized
path 提示。独立无 GPU runner UTC `00:19:19`--`00:19:39`、exit0；五个推理请求与
四个 profiler endpoint 全部 HTTP 200，prompt tokens 均为 2100，cached tokens 严格为
`0/1840/0/1840/0`。Hit profile 窗口为 `00:19:31.678`--`00:19:38.319`，cold 为
`00:19:38.566`--`00:19:38.922`。

Raw trace 的同口径结果如下：

| 指标 | Cold | Hit | Hit-Cold |
| --- | ---: | ---: | ---: |
| 主 worker CPU | 37.388 ms | 50.483 ms | +13.096 ms |
| 36 层 attention CPU | 17.465 ms | 27.308 ms | +9.842 ms |
| GPU kernel + memcpy | 18.996 ms | 23.463 ms | +4.467 ms |
| Cached GQA2 kernel | 0 次 | 36 次 / 17.398783 ms | +17.398783 ms |
| Materialize kernel | 0 次 | 0 次 | 0 |
| `cudaLaunchKernel` | 268 次 | 340 次 | +72 次 |
| GPU kernel 数 | 606 次 | 786 次 | +180 次 |

相对第 10.37 节旧 GQA2 同 payload raw trace，cached kernel 从 17.855682 ms 降至
17.398783 ms，名义减少 0.456899 ms（2.558%）。更可靠的结构证据是 hit-cold 的
`aten::to/_to_copy/copy_` 增量分别从 `108/72/72` 降为 `72/36/36`：每项恰好减少
36 次，证明 absorbed cached output 的逐层 FP32-to-BF16 copy 已消失；当前剩余 36 次
`_to_copy/copy_` 对应 Q 的 BF16-to-FP32 旋转输入。`mm` 增量从 72 降为 36，则与
V inverse 被 absorption 删除一致。累计至当前的 absorption、stride 和 host-fastpath
优化后，launch 增量从 180 降为 72，GPU kernel 增量从 324 降为 180。

但端到端 host 结论仍不能越界。相对旧 GQA2，attention gap 减少 1.245 ms，GPU
kernel+memcpy gap 减少 0.605 ms；worker gap 却从 12.129 ms 增至 13.096 ms，增加
0.966 ms。两边各只有一个独立服务 profile 窗口，不能把该波动归因于候选，也不能据此
宣告 M2。可靠结论是：默认 serving 已满足不 materialize 全历史 BF16 的硬标准，且
目标 dtype copy 确实删除；fused cached kernel 仍是约 17.4 ms 的主瓶颈。

服务 UTC `00:21:46` 请求停止，`00:21:53` exit0、`OOMKilled=false`；严格运行时错误
扫描和 materialize 日志扫描均为空。UTC `00:21:54` GPU 0--7 均恢复 0 MiB、0%、
无计算进程，容器已删除。完整 `nvidia-smi`、payload、runner、响应、trace、summary、
日志、inspect 和回收记录位于
`artifacts/oscar_vllm_optimization/20260721/torch_profile_fused_default_bf16_output/`。
下一步使用同一 final image 和原 TTFT client 按 CV 规则复测 M2；若仍失败，后续优化
必须直接作用于 cached GQA2 kernel，而不是恢复 materialize。

### 10.87 Fused-Default Prefix-Hit TTFT M2

本轮复用原始 TTFT client，SHA256 仍为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`；
两对 warm-up、首轮 5 对和任一侧 CV 超过 3% 时扩展到 15 对的规则不变。Client 复制后
通过 pycompile、Ruff lint/format，并运行在共享服务网络的独立无 GPU 容器。

GPU 0 在 UTC `00:24:27` 与 `00:26:01` 两次检查中均为 0 MiB、利用率 0%、无计算
进程，间隔 94 秒；GPU 1--7 同样空闲。本轮固定物理 GPU 0，Docker image、模型、
10 GiB、BF16、OSCAR INT2、maxseq2、prefix cache、chunked prefill、absorption 及全部
环境版本与第 10.86 节相同；容器内 `CUDA_VISIBLE_DEVICES=0`，但不启用 profiler。

服务 UTC `00:26:51` 启动，`00:29:00` health 200；日志确认 36 层 absorption、三段式
容量和 fused mixed attention，materialize 日志扫描为空。正式 client UTC
`00:29:18`--`00:29:28`、exit0。首 5 对的两侧 CV 均低于 3%，因此按预定规则停止，
没有追加样本：

| 路径 | Mean TTFT | Population std | CV | 请求数 |
| --- | ---: | ---: | ---: | ---: |
| Cold / 0 cached tokens | 37.936 ms | 0.727 ms | 1.916% | 5 |
| Hit / 1840 cached tokens | 47.435 ms | 0.433 ms | 0.912% | 5 |

10 个正式请求全部 HTTP 200、prompt tokens=2100；cold 全为 0 cached tokens，hit 全为
1840 cached tokens。5/5 配对均为 hit 更慢，`hit-cold` 均值 `+9.499 ms`、population
std `0.442 ms`，范围 `+8.832`--`+9.983 ms`；hit 均值比 cold 慢 25.039%。因此
**M2 明确失败**。当前低 CV 和一致符号足以判断失败，不需要靠追加样本改变结论。

第 10.36 节旧 GQA2 的 nominal gap 为 +10.377 ms，本轮为 +9.499 ms，但旧轮两侧 CV
高且属于独立服务，不能把 0.878 ms 差解释为候选收益。第 10.86 节已经给出结构证据：
目标 copy 已删除，但 36 层 cached GQA2 kernel 仍约 17.4 ms；这与 TTFT 继续失败一致。

服务 UTC `00:29:52` 请求停止，停止前严格运行错误扫描为空。EngineCore shutdown 后，
API output handler 在最终日志记录了一次 `EngineDeadError`；它发生在全部请求结果落盘
之后、由 `docker stop` 触发的退出阶段。容器最终于 `00:30:01` exit0、
`OOMKilled=false`，所以不把它解释为 workload 失败，但最终错误扫描也不能表述为空。
UTC `00:30:01` GPU 0--7 均恢复 0 MiB、0%、无计算进程，容器已删除。

完整 client、请求、统计、日志、inspect、两次 `nvidia-smi` 和回收记录位于
`artifacts/oscar_vllm_optimization/20260721/prefill_fused_default_bf16_output_ttft/`。
下一阶段必须优化 cached GQA2 主 kernel；恢复 full-history materialize 虽可降低 GPU
时间，但违反硬标准，不再作为 M2 解法。

### 10.88 Segmented Cache-Source Load 候选与静态门禁

第 10.86 节的真实 fused trace 显示，36 层 cached GQA2 kernel 累计仍为
17.398783 ms。源码复核进一步确认：现有 kernel 在每个 32-token tile 内同时构造
INT2、prefix 和 recent 三路地址与 masked load，最后再选择实际来源。当前 M2 workload
的 `cached_len=shared_hit_len=1840`，所以 `recent_start=1840`，缓存区实际只有前 64
个 prefix token 与后 1776 个 INT2 token；INT2 tile 中的 HP 地址和 load 没有提供有效
数据。第 10.22 节 Q rotation 的单层五批均值仅 0.022043 ms，按 36 层线性外推也不足
0.8 ms，且 CV 为 32.139%，因此本轮不把外围 dtype copy 作为主优化方向。

新增 artifact-only A/B 脚本，不修改 production wrapper 或 kernel。候选保持 32x32、
GQA2、`num_warps=4`、`num_stages=2` 与 online-softmax 更新顺序，只把缓存来源拆成三个
专用 loop：prefix loop 不读 block table/INT2，INT2 loop 不构造 HP 地址，recent loop
不读 block table/INT2。测试固定使用 260 query、1840 cached、32 query heads、8 KV
heads、head dim 128 和非零 INT2/HP cache，并同时覆盖：

- `shared_hit=1840`：当前完整 prefix-hit 形状，没有 request-owned recent；
- `shared_hit=0`：`recent_start=1584`，故 recent 边界不与 32-token tile 对齐，用于防止
  分段后边界归约变化被遗漏。

脚本最终 SHA256 为
`7a2aa64d1055261f38ee02323264391ee2eff9d5346256729ccfc7886af5194f`。无 GPU 静态
门禁使用镜像 `oscar-vllm:fused-default-bf16-output-20260721`，完整 image ID 为
`sha256:37bc682cc1f2e1d5b95410f60c723e95b9f929239b910d21887e0e2e09bbb84e`。
首轮 pycompile 与 Ruff lint 已通过，format check 只报告需要机械重排；格式化后完整
重跑为 pycompile、Ruff lint、Ruff format check 全部 exit0。该阶段未分配 GPU，也尚未
产生正确性或性能结论；只有后续 CUDA A/B 同时通过数值 oracle 且给出稳定收益，候选
才允许进入 production。

脚本、两轮静态日志、退出码与 SHA256 位于
`artifacts/oscar_vllm_optimization/20260721/prefill_segmented_load_microbench/`。
下一步按两次连续空闲规则固定物理 GPU，再运行同一脚本的 CUDA correctness 与五批
交错计时。

### 10.89 Segmented A/B 的 GPU 分配阻塞

静态门禁完成后按全卡可用范围开始分配前检查。UTC `00:42:29` 至 `01:10:45` 共执行
21 次检查，相邻轮次均至少间隔一分钟；每轮都同时保存 GPU 显存/利用率、compute
process 列表和完整 `nvidia-smi`。21 轮中 GPU 0--7 始终各有一个相同 8-way 作业的
Python 子进程，因此没有任何卡形成第一次空闲记录，更不满足连续两次空闲要求。

进程树确认父进程为外部 `eval.py`，目标模型是
`/nfs/AIED/models/GLM-5.2-FP8`，任务为 `arena-hard-v2:3`，`max-new-tokens=256`；
它不属于本项目，未对其终止、抢占或注入工作。首轮每卡显存为 1160 MiB，末轮增长为
47930--56972 MiB；末轮完整环境状态为 NVIDIA B200、Driver 595.71.05、
`nvidia-smi` CUDA 13.2，8 张卡均处于 P0。即使瞬时利用率多数为 0%，存在计算进程且
显存持续增长，不能解释为空闲卡，也不适合运行亚毫秒级性能 A/B。

因此本阶段没有启动 Docker CUDA 容器，没有生成 correctness、kernel latency 或
speedup 数据，也没有修改 production kernel。当前可复现状态仅为第 10.88 节静态通过
的 artifact-only 候选；M2 仍保持失败，不得把未执行的 segmented 方案记作优化收益。
GPU 检查、完整 `nvidia-smi`、compute process 和候选脚本均位于
`artifacts/oscar_vllm_optimization/20260721/prefill_segmented_load_microbench/`。
待至少一张物理卡连续两次无进程后，下一动作仍是固定该卡运行同 SHA256 脚本，而不是
直接合入生产代码。

### 10.90 GPU 短暂释放后的恢复检查

收到 GPU 已释放的通知后，本阶段从既有 artifact 继续执行，不重建或修改候选脚本；
脚本 SHA256 仍为 `7a2aa64d...5194f`，静态门禁仍为 exit0。UTC `02:22:09` 的首次
恢复检查显示 GPU 0--7 均为 0 MiB、0%、无计算进程，形成一轮空闲记录。

但 UTC `02:23:48` 的第二次检查中，8 张卡均出现新的 DeepSpec Python 子进程
（PID 1188546--1188553），各占约 1146 MiB，所以没有任一张卡满足连续两次空闲。
UTC `02:25:14`/`02:26:34` 继续复核时，每卡显存已分别增长至约 108.8 GiB 和
118.6--119.0 GiB，GPU 利用率最高达到 100%，确认这是活跃负载而不是残留 context。

进程树显示新父进程仍为外部 GLM-5.2-FP8 `arena-hard-v2:3` 评测，但本轮
`max-new-tokens=2048`。本项目没有终止或干扰该作业，也没有在其上并跑。由于首轮空闲
状态在第二轮前失效，本轮仍未启动 Docker CUDA 容器、未产生 segmented correctness
或 latency 结果，production kernel 仍未修改。恢复检查的 CSV、完整 `nvidia-smi` 和
compute-process 列表保存在第 10.89 节同一 artifact 目录下，文件名使用
`gpu_resume_check*` 与 `nvidia_smi_resume_check*` 前缀。

下一次恢复必须从头获得同一物理卡连续两次无进程记录；若外部评测由自动调度器反复
拉起，需要先暂停该调度器或为本项目保留至少一张卡，否则无法形成可复现的独占性能
窗口。

### 10.91 Segmented Cache-Source Load CUDA A/B

按用户要求，GPU 无空闲时保持持续监控，不再结束当前工作轮次。Monitor 每 65 秒保存
一次 GPU CSV、compute-process 列表和完整 `nvidia-smi`；UTC `02:38:52` 与
`02:40:11` 连续两轮显示 GPU 0--7 均无计算进程，故自动选择最低编号的物理 GPU 0。
第二轮完整 `nvidia-smi` 于 `02:40:25` 落盘，8 张卡均为 0 MiB、0%，无运行进程。

实验直接使用第 10.88 节 SHA256 为 `7a2aa64d...5194f` 的 artifact-only 脚本，生产
代码来自独立镜像，未通过 bind mount 覆盖。实际环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 / ID | `oscar-vllm:fused-default-bf16-output-20260721` / `sha256:37bc682c...9bbb84e` |
| 固定物理 GPU / `CUDA_VISIBLE_DEVICES` | GPU 0 / `0` |
| GPU / 显存 | NVIDIA B200 / 183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| 预热 / 采样 | 20 次 / 5 批，每批 100 次 |

Docker CUDA benchmark 于 UTC `02:40:29`--`02:41:07` 完成，exit0、
`OOMKilled=false`。Baseline 与 segmented 使用相同的 32x32、GQA2、warps4/stages2
配置并交错计时，结果如下：

| 场景 | Baseline | Segmented | Delta | Speedup | 两侧 CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full prefix hit，`shared_hit=1840` | 0.473414 ms | 0.404868 ms | -0.068546 ms | 1.169304x | 0.0653% / 0.0079% |
| Recent tail，`shared_hit=0` | 0.435833 ms | 0.356561 ms | -0.079272 ms | 1.222325x | 0.0065% / 0.0164% |

四组 CV 均远低于 3%。Full prefix hit 的 output 与 LSE 相对现有 production kernel
均为 bitwise 一致；`recent_start=1584` 的未对齐 recent 边界中，output/LSE max abs
分别为 `0.0009765625` 与 `9.536743e-7`，通过预设数值 oracle。因此该候选同时具备
正确性与稳定组件收益，可以进入 production 实现和更广 CUDA oracle。

Full-hit 单层节省 0.068546 ms，按 36 层线性外推为 2.467654 ms；segmented 单层
0.404868 ms 外推为 14.575262 ms。该外推只用于组件预算，不是服务 TTFT 实测，也不足以
单独抵消第 10.87 节 +9.499 ms 的 TTFT gap。因此本轮不能宣告 M2 通过；下一步应把
候选以最小改动合入 production kernel，先覆盖 poisoned prefix/recent 与 mixed-batch
CUDA oracle，再由独立镜像 profiler 和同 client TTFT 判断端到端收益。

容器 inspect、环境版本、原始五批 JSON、完整日志、连续空闲检查和回收记录位于
`artifacts/oscar_vllm_optimization/20260721/prefill_segmented_load_microbench/`
下的 `resume_monitor_20260721T023852Z/`。容器已删除，benchmark 后 GPU 0--7 均恢复
0 MiB、0%。辅助汇总首次把 inspect 数组误作对象，`jq` 查询失败；已改用 `.[0]` 从
同一文件成功提取 exit0/非 OOM 状态，没有重复 GPU 实验。

### 10.92 Production Segmented Kernel 与静态门禁

将第 10.91 节已验证的结构以最小范围合入
`triton_oscar_prefill.py`：新增一个只负责 online-softmax 状态更新的 Triton helper，
并把 cached-prefill 原先同时构造三种来源的单 loop 替换为 prefix、INT2 history、recent
三个专用 loop。Prefix/recent loop 不再读取 block table 或 INT2 cache，INT2 loop 不再
构造 HP 地址；wrapper ABI、32x32 tile、GQA2、warps/stages、materialize oracle 和其他
backend 均未修改。

首轮无 GPU 静态容器把单文件挂在容器根目录，脱离项目 `pyproject.toml` 后误报既有
import I001；pycompile 和 Git 空白检查实际已通过。该误报没有触发源码修改。改为只读
挂载完整仓库并从 `/src` 项目根运行后，pycompile 与 Ruff lint 通过，format check 仅
要求新代码机械重排。Docker Ruff format 后完整重跑结果为：pycompile、Ruff lint、
Ruff format check 全部 exit0，目标文件直接文本扫描也确认无冲突标记、无行尾空白且
EOF 有换行。最终 production 文件 SHA256 为
`ff7edcfe3aa28d3134439b1637b148cbf96ebdec4f9d3e31d3e5a15493ff36cc`。

静态阶段结束时，该目标文件在 Git 中显示为未跟踪 `??`，index 中没有对应条目；这与
本轮开始时的暂存状态不同。没有擅自重新暂存或修改 index，后续验证直接使用磁盘上的
上述 SHA256 文件。该状态不影响 Docker 源码覆盖 CUDA oracle，但在最终提交前需要由
仓库所有者按预期整理暂存范围。

本阶段没有分配 GPU，也没有宣称 production correctness 已闭环。静态日志、退出码、
patch 快照和 SHA256 位于
`artifacts/oscar_vllm_optimization/20260721/prefill_segmented_production_static/`。
下一步继续按 65 秒监控规则分配 GPU，用磁盘当前源码覆盖镜像内 Python package，运行
poisoned prefix/recent、absorbed BF16 output 和 mixed cached/current 定向 CUDA oracle。

### 10.93 Production Segmented Kernel 定向 CUDA Oracle

持续 monitor 于 UTC `02:47:57` 与 `02:49:21` 连续确认 GPU 0--7 无计算进程，自动
固定物理 GPU 0；容器内 `CUDA_VISIBLE_DEVICES=0`。Docker image、GPU、driver、Python
和相关包版本与第 10.91 节相同。本轮只读覆盖镜像内 production prefill 文件和当前
CUDA 测试文件，二者实际 SHA256 分别为 `ff7edcfe...f36cc` 与
`44c30d9a...b3aa161`；其余 vLLM package 和编译扩展均来自独立镜像。

定向选择 3 个 pytest 实例：poisoned full prefix hit，以及 mixed cached/current
prefill 的 absorbed/non-absorbed 两种参数。它们共同覆盖不读取未拥有 recent row、
prefix page table、INT2 history、request-owned recent、cached/current LSE merge、
BF16 absorbed output、FP32 V inverse、非连续 fused-QKV V stride 和 materialize 对照。

CUDA pytest 于 UTC `02:49:46`--`02:50:59` 完成，结果为
`3 passed, 14 deselected, 120 warnings in 67.30s`、exit0。严格日志扫描未检出
traceback、runtime/CUDA/assertion、实际 OOM；容器 exit0、`OOMKilled=false`。因此
production port 的既有核心语义已闭环，未出现 artifact-only kernel 与生产 wrapper
集成差异。

本轮结束后容器已删除，完整 `nvidia-smi` 显示无运行进程；GPU 0--7 均为 4 MiB、0%，
属于驱动基线。连续空闲检查、pytest 日志/退出码、源码 hash、container inspect 和回收
记录位于
`artifacts/oscar_vllm_optimization/20260721/prefill_segmented_production_cuda/`。
下一步构建包含 production SHA 的独立镜像，核对镜像内/本地 hash 后运行无覆盖完整
OSCAR CUDA 回归；完整回归通过前不启动真实服务性能测试。

### 10.94 Production Segmented 独立镜像

独立镜像 `oscar-vllm:segmented-prefill-20260721` 于 UTC
`02:52:16`--`02:52:22` 构建完成，build exit0，完整 image ID 为
`sha256:db33e89ace8c789a87a022361e99f582e4b037a668d75ca4e6506b015a28dcef`。
本阶段没有分配 GPU。

镜像内
`/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_oscar_prefill.py`
与本地 production 文件的 SHA256 均为
`ff7edcfe3aa28d3134439b1637b148cbf96ebdec4f9d3e31d3e5a15493ff36cc`，证明 build
context 包含当前未跟踪文件的实际内容，后续无需源码覆盖。

构建命令在 `vllm/` 工作目录内展开 `$PWD/artifacts`，导致日志首次落到子仓库下的
`vllm/artifacts/`；镜像构建和 hash 校验均未受影响。没有重复 build，而是把全部文件
原样复制到 workspace 根的规范目录，并以逐文件 SHA256 比对确认两侧一致。规范 artifact
位于 `artifacts/oscar_vllm_optimization/20260721/segmented_prefill_image/`，包含 build
日志/退出码、时间、image inspect 和两侧 hash。

下一步继续保持 GPU 监控；同一物理卡连续两轮空闲后，直接使用该镜像、无 bind mount
运行完整 `test_oscar.py`。只有完整 CUDA exit0 后才进入真实服务 profiler/TTFT。

### 10.95 Production Segmented 独立镜像完整 CUDA 回归

持续 monitor 于 UTC `02:54:11` 开始；前 10 轮均没有空闲 GPU，故没有提前退出或
违规并跑，并在监控达到 10 分钟时同步了进度。UTC `03:05:11` 与 `03:06:32` 连续
两轮确认 GPU 0--7 均无计算进程，自动固定最低编号的物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`。除 Docker 镜像外，GPU、driver、Python 和相关包版本均与
第 10.91 节一致；本轮镜像为 `oscar-vllm:segmented-prefill-20260721`，完整 image ID
为 `sha256:db33e89ace8c789a87a022361e99f582e4b037a668d75ca4e6506b015a28dcef`。

完整 `tests/quantization/test_oscar.py` 于 UTC `03:07:04`--`03:09:15` 在独立镜像内
运行，未挂载任何 host 路径，`TRITON_CACHE_DIR` 使用容器内全新目录。结果为
`17 passed, 134 warnings in 90.43s`、pytest exit0；container inspect 同时确认
exit0、`OOMKilled=false`、`Mounts=[]`。严格日志扫描未检出 traceback、runtime/CUDA
error、assertion 或实际 OOM。该结果闭环了 segmented production kernel 的独立打包
与完整 CUDA correctness，但尚不构成真实服务算子或 TTFT 性能证据，M2 仍未通过。

容器已删除；实验后完整 GPU 状态显示 GPU 0--7 均为 0 MiB、0%，无计算进程。
12 轮连续监控、完整 `nvidia-smi`、pytest 日志/退出码、container inspect 和回收记录
位于
`artifacts/oscar_vllm_optimization/20260721/segmented_prefill_full_cuda/monitor_20260721T025411Z/`。
下一步使用同一独立镜像和冻结 payload 运行真实服务 operator profiler，验证 segmented
cached-prefill 的 36 层总 GPU 时间与 materialize 调用数，再以相同 token-ID client
复测 prefix-hit TTFT M2。

### 10.96 Production Segmented 真实服务 Operator Profiler

本轮严格复用第 10.86 节的三份 token-ID payload 与 runner；SHA256 仍分别为
`0bfa9352...07ecd`、`0c33cf27...42cf3`、`3e762929...2b92f` 和
`aa4893ec...5256d`。服务参数同样保持 Qwen3-4B、BF16、eager、10 GiB KV budget、
`max_num_seqs=2`、prefix caching、chunked prefill、8192-token batch 上限以及
V-rotation absorption。唯一有意变化是镜像换为第 10.94 节 segmented 镜像；实际 GPU
和软件环境与第 10.91 节相同。

持续 monitor 于 UTC `03:14:51`/`03:16:15` 连续确认 GPU 0--7 空闲后固定物理
GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`。服务于 `03:16:36` 启动、`03:18:52`
健康就绪；日志确认 36 层 V rotation 已吸收，prefix/recent 为 64/256 tokens，物理
INT2 history 容量仍为 461936 tokens。客户端使用独立无 GPU 容器；UTC
`03:19:06`--`03:19:26` 的五次请求全部 HTTP 200，观测序列严格等于预期的
`(2100, cached=0/1840/0/1840/0)`，workload exit0。

由原始 trace 按与第 10.86 节相同口径重算，结果如下；括号内为 segmented 相对
fused-default 的 cold/hit gap 变化，负值代表 gap 缩小：

| 指标 | Fused-default | Segmented | 变化 |
| --- | ---: | ---: | ---: |
| 36 层 cached kernel | 17.398783 ms | 14.867568 ms | -2.531215 ms，1.170251x |
| Worker CPU cold/hit gap | 13.095574 ms | 11.570078 ms | -1.525496 ms |
| Attention CPU cold/hit gap | 9.842419 ms | 10.968971 ms | +1.126552 ms |
| GPU kernel+memcpy cold/hit gap | 4.467286 ms | 1.916497 ms | -2.550789 ms |

Cached kernel 共 36 次，时间降低 14.5482%，与第 10.91 节组件 A/B 的 1.169304x
方向和幅度一致。Cold GPU kernel+memcpy 为 18.998604 ms，基本等于旧值
18.995556 ms；hit 从 23.462842 ms 降至 20.915101 ms，说明收益集中在命中路径，
而非由全局时序偏移伪造。Materialize kernel 在 cold/hit 中均为 0 次。两侧 GPU kernel
计数仍为 606/786、memcpy 为 48/48，`cudaLaunchKernel` 为 268/340，主要 PyTorch op
计数也逐项不变。因此本轮证据支持“同一 36 次 cached launch 内减少无效来源 load”，
不支持“靠减少 launch 数获得收益”。

需要客观看待 Attention CPU gap 增加 1.126552 ms：该 CPU span 包含异步 launch 与
调度开销，单次 trace 不是统计性 TTFT 结论。尽管 Worker 和 GPU gap 都缩小，本轮仍不
宣告 M2 通过；必须用第 10.87 节同一 token-ID TTFT client 做 warm-up 后的配对采样。

服务停止前和最终严格日志扫描均为空；容器最终 exit0、`OOMKilled=false`，并于 UTC
`03:22:49` 正常退出。UTC `03:23:16` GPU 0--7 均回到 0 MiB、0%，无计算进程。
后处理首次直接调用宿主 Ruff 时发现命令不存在，随后容器内首次 Ruff 又因只读 mount
无法创建默认 cache；设置 `/tmp` cache 后 pycompile、Ruff lint/format 全部通过。
辅助 `cudaLaunchKernelExC` 统计也已从误用 driver API 修正为 runtime API 后重算，
原始 trace 与核心 cached-kernel 时间未受这些后处理问题影响。

连续空闲记录、完整服务日志、请求响应、两份 trace、可复现汇总脚本、summary、inspect
和 GPU 回收状态均位于
`artifacts/oscar_vllm_optimization/20260721/torch_profile_segmented_prefill/`。
下一步重新按连续两次空闲规则固定 GPU，用同一独立镜像和冻结 client 复测 TTFT M2。

### 10.97 Production Segmented Prefix-Hit TTFT M2 复测

本轮直接复用第 10.87 节 token-ID client，SHA256 仍为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`；服务
参数与第 10.96 节相同，但关闭 profiler。Docker 镜像和完整 image ID 仍为
`oscar-vllm:segmented-prefill-20260721` / `sha256:db33e89a...28dcef`，GPU 与软件
环境仍同第 10.91 节。

持续 monitor 于 UTC `03:25:41`/`03:27:08` 连续确认 GPU 0--7 空闲，固定物理
GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`。服务于 `03:27:32` 启动、`03:29:47`
健康就绪；日志再次确认 OSCAR backend、prefix caching、chunked prefill、36 层
V-rotation absorption 和 461936-token INT2 history 容量。正式 benchmark 使用独立
无 GPU 客户端容器，于 UTC `03:29:59`--`03:30:16` 完成并 exit0。

客户端先运行 2 对 warm-up，再运行 5 对正式 cold/hit 配对。两侧 CV 已低于 3%，故按
冻结规则不扩展到 15 对。全部 10 个正式请求均 HTTP 200、prompt 2100 tokens；cold
均为 0 cached tokens，hit 均为 1840 cached tokens。结果如下：

| 指标 | Fused-default（第 10.87 节） | Segmented | 变化 |
| --- | ---: | ---: | ---: |
| Cold TTFT 均值 | 37.936016 ms | 32.907069 ms | -5.028948 ms |
| Hit TTFT 均值 | 47.434855 ms | 40.132638 ms | -7.302218 ms |
| Cold / hit CV | 1.916% / 0.912% | 2.677% / 2.928% | 两侧均 <3% |
| Hit-cold 配对均值 | +9.498839 ms | +7.225569 ms | -2.273270 ms，-23.9321% |

Segmented 的五个逐对 hit-cold 差为 `+3.616501`、`+8.026159`、`+7.656846`、
`+8.830209`、`+7.998130 ms`；5/5 hit 都更慢，配对差标准差为 1.845212 ms。
1840/2100 即 87.619% 的 prompt tokens 被命中，满足命中率 >75% 和 prefill tokens
减少 >=70% 两项门槛，但 TTFT 没有改善，因此 **M2 仍然失败**。

跨独立服务的 cold/hit 绝对值都下降，不能全部归因于一个 kernel 改动；更可靠的是同轮
配对 gap 缩小 2.273270 ms，它与第 10.96 节 GPU gap 缩小 2.550789 ms 方向一致。
这证明 segmented 优化确实传递到端到端路径，但现有 14.867568 ms cached kernel 和
剩余 host/attention 开销仍足以使 hit 比 cold 慢约 7.2 ms，不能据此宣告性能达标。

服务停止前和最终严格日志扫描均为空；容器最终 exit0、`OOMKilled=false`，于 UTC
`03:31:32` 正常退出。UTC `03:31:52` GPU 0--7 均回到 0 MiB、0%，无计算进程。
连续空闲记录、完整日志、原始逐请求结果、配对统计、inspect 和 GPU 回收状态位于
`artifacts/oscar_vllm_optimization/20260721/prefill_segmented_ttft/`。下一步继续定位
剩余约 7.2 ms 命中惩罚；在新的组件候选同时具备 correctness 与稳定收益前，不重复
同一 TTFT 实验，也不宣告阶段 6 完成。

### 10.98 BF16 Quantization Metadata 候选与静态门禁

第 10.97 节后重新审计 cache geometry 与官方参考。当前 vLLM `OscarConfig` 为每个
K 或 V 向量保存 FP32 `scale+zero`，metadata 为 8 B；128 维 INT2 数据为 32 B，故
K+V quant slot 为 `2 * (32+8) = 80 B`。官方 OSCAR-SGLang unified pool 的
`scale_dtype` 默认 `torch.bfloat16`，同几何对应 `2 * (32+4) = 72 B`。这不是 allocator
预留差异，而是实际量化 metadata 精度和物理布局差异；若 correctness 可接受，单看
history slot 几何即可带来 `80/72 = 1.111111x` 容量倍率，但实际三段式总容量仍必须由
production allocator 和服务日志重测，不能直接把该比值当作最终压缩率。

本轮只新增 artifact-only production-shape A/B，不修改 production。Candidate 保持
INT2 indices、64-token prefix、256-token recent、三段 source loop、32x32 tile、GQA2、
4 warps/2 stages 和 online-softmax 顺序不变；只把四个 scale/zero 值转为 BF16。为了
避免新增 metadata tensor 或复制，脚本对同一 contiguous `uint8` cache storage 建立
BF16 typed view，kernel 仍从 uint8 pointer 读取 INT2 bytes，但从 typed pointer 直接读取
BF16 metadata。A/B 覆盖 `shared_hit=1840` full hit 与 `recent_start=1584` 未对齐
recent tail，并保留五批交错 CUDA-event 测量框架。

脚本最终 SHA256 为
`450cb2230979ea6fa1f7753396e4887e71505a02246e2eb8b4fa5e896b50360f`。首次静态
容器因试图在只读 `/src` 父 mount 下叠加单文件 mount 而在 OCI 初始化阶段失败；改为
独立 `/bench.py` mount 后，Ruff 依次要求项目 import 分组空行和机械格式化。按只读
diff 修正、由容器 Ruff formatter 处理后，最终 pycompile、Ruff lint、Ruff format check
全部 exit0。上述轮次均未分配 GPU，也没有生成数值误差、latency 或实际容量结果。

脚本、全部静态日志/退出码和 hash 位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_meta_microbench/`。下一步持续
监控 GPU；同一物理卡连续两轮空闲后固定该卡运行候选。只有 full-hit/recent-tail 的
correctness 和五批稳定收益同时通过，才允许进入 production red oracle 与布局修改。

### 10.99 BF16 Quantization Metadata CUDA A/B

持续 monitor 于 UTC `03:42:56` 开始；前 6 轮均无空闲卡，未提前退出或并跑。UTC
`03:49:32`/`03:50:56` 连续确认 GPU 0--7 无计算进程后自动固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`。实验直接使用第 10.98 节 SHA256 为
`450cb223...50360f` 的 artifact-only 脚本和 segmented 独立镜像；production package
未被源码覆盖。Docker image、GPU、driver、Python 与相关包版本同第 10.91 节。

CUDA A/B 于 UTC `03:51:07`--`03:51:56` 完成，容器 exit0、
`OOMKilled=false`，严格日志扫描为空。两种 metadata layout 使用相同的 INT2 indices、
prefix/recent data、query 和调度 metadata；BF16 候选只从 FP32 baseline 的 scale/zero
舍入构造自己的 72 B slot。Correctness 结果如下：

| 场景 | Output max abs | LSE max abs | 结果 |
| --- | ---: | ---: | --- |
| Full prefix hit，`shared_hit=1840` | 0.0029296875 | 0.0008678436 | 通过 |
| Recent tail，`shared_hit=0` | 0.0029296875 | 0.0009479523 | 通过 |

每个配置预热 20 次，再执行 5 批、每批 100 次交错 CUDA-event 测量：

| 场景 | FP32 metadata | BF16 metadata | Delta | Speedup | 两侧 CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full prefix hit | 0.405104 ms | 0.234191 ms | -0.170914 ms | 1.729805x | 0.1318% / 0.1184% |
| Recent tail | 0.356572 ms | 0.210318 ms | -0.146254 ms | 1.695391x | 0.0177% / 0.0221% |

四组 CV 均远低于 3%，且 FP32 full-hit baseline 与第 10.91 节 production segmented
0.404868 ms 一致。Full-hit 按 36 层线性外推时，BF16 metadata 为 8.430863 ms，
相对本轮 FP32 baseline 节省 6.152888 ms；该值只是组件预算，不是服务 trace 或 TTFT。
结合第 10.97 节仍剩约 7.2 ms 的命中惩罚，这一候选的量级足以进入 production 验证，
但不能提前宣告 M2 会通过。

物理 quant slot 从 80 B 降至 72 B，history 几何容量倍率为 1.111111x；本轮没有运行
production allocator，因此仍不报告新的实际 token capacity 或最终压缩率。容器已删除，
实验后 GPU 0--7 均为 0 MiB、0%，无计算进程。8 轮监控、完整 `nvidia-smi`、原始
五批结果、container inspect 和回收记录位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_meta_microbench/monitor_20260721T034256Z/`。
下一步先新增 production layout/store/decode/prefill 的失败 oracle，再以最小一致改动
切换 metadata；不能只改 prefill reader，否则会造成 cache ABI 不一致。

### 10.100 BF16 Metadata Production 失败 Oracle（静态与 CPU）

在 production 实现前先修改现有测试契约。CPU geometry oracle 将 quant slot、1024-token
mixed bytes 和 padded page 的 quant 部分从 80 B 改为 72 B。CUDA reference 仍以 FP32
scale/zero 计算 INT2 indices，只在反量化时使用落盘后舍入为 BF16 的 metadata；另新增
raw-storage oracle，要求 writer 把 K/V scale+zero 写入同一 `uint8` cache 的 BF16 typed
view 对应位置。该定义没有把量化计算本身降为 BF16，也同时约束 writer 和 reader ABI。

两份测试首先通过 pycompile 与 Ruff lint；首轮 format check 只要求 CUDA 测试机械
排版。容器 Ruff formatter 处理后，最终 pycompile、Ruff lint、Ruff format check 全部
exit0。无 GPU 的定向 CPU red run 使用 segmented 独立镜像内当前 80 B production
config，结果为 `3 failed, 9 deselected in 9.36s`。三处失败分别为：

- `slot_size_aligned` 实际 80，期望 72；
- 1024-token mixed bytes 实际 1761280，期望 1716224；
- 8192 上下文 padded page 实际 14336，期望 13312。

失败全部精确落在旧 FP32 metadata geometry，没有导入、平台或无关 assertion 失败，
因此形成有效 CPU red oracle。该阶段未分配 GPU；raw writer 与 BF16 roundtrip 的 CUDA
red oracle 尚未运行，不能从 CPU geometry 失败推断 kernel 读写行为。测试 hash、静态
日志和 CPU red 输出位于
`artifacts/oscar_vllm_optimization/20260721/bf16_meta_red_oracle/`。下一步持续监控 GPU，
运行新增 writer oracle 与 store/dequant roundtrip，确认旧 ABI 在 CUDA 上也按预期失败。

### 10.101 BF16 Metadata Production CUDA 失败 Oracle

持续 monitor 于 UTC `03:57:41`/`03:59:00` 连续确认 GPU 0--7 空闲后固定物理
GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`。本轮使用 segmented 独立镜像中的旧 80 B
production package，只读挂载第 10.100 节当前 CUDA 测试；Docker image、GPU、driver、
Python 和包版本同第 10.99 节。

定向选择 raw BF16 writer oracle 与 64/128 head-dim 两个 store/dequant roundtrip，UTC
`03:59:23`--`04:00:18` 的预期结果为 `3 failed, 15 deselected in 49.90s`、pytest
exit1。失败边界如下：

- Raw writer 首先确认旧 config 的 `meta_bytes` 实际为 8，而新契约要求 4；
- 64 维 roundtrip 有 3647/10240 元素不匹配，max abs 0.0123291；
- 128 维 roundtrip 有 8884/20480 元素不匹配，max abs 0.015625。

后两项失败说明旧 reader 返回的是 FP32 metadata 反量化值，与“INT2 indices 仍按 FP32
计算、但落盘 metadata 舍入为 BF16”的新 reference 确实不同；它们不是单纯由 geometry
assertion 触发。三项 oracle 因此共同约束 config、writer storage 和 reader semantics，
可用于防止只修改单侧 ABI 的不完整实现。

容器符合预期 exit1、`OOMKilled=false`，已删除；实验后 GPU 0--7 均为 0 MiB、0%，
无计算进程。两次连续空闲记录、pytest 完整日志、container inspect 与回收状态位于
`artifacts/oscar_vllm_optimization/20260721/bf16_meta_red_oracle/cuda_monitor_20260721T035741Z/`。
下一步开始 production 一致切换：config geometry、store/demote writer、decode/full-
dequant/prefill/materialize reader 必须在同一阶段使用 BF16 typed view，并先以静态、CPU
和这三项 CUDA oracle 转绿。

### 10.102 BF16 Metadata Production ABI 与 CPU Green

Production 已按第 10.101 节失败契约做最小一致切换。`OscarConfig.meta_bytes` 从每向量
8 B FP32 `scale+zero` 改为 4 B BF16，128 维 INT2 的 K/V region 各由 40 B 降为
36 B，quant slot 因而由 80 B 降为 72 B。普通 store 和 recent demote writer 仍以
FP32 计算 min/max、scale、zero 与 INT2 indices，只在写入同一 `uint8` cache storage
时把 scale/zero 舍入为 BF16；decode、full-dequant、受限 materialize 与 fused cached-
prefill reader 均通过 `kv_cache.view(torch.bfloat16)` 直接读取。没有新增第四个 pool、
独立 metadata allocation 或随 history 长度增长的 workspace。

本轮使用无 GPU 的 `oscar-vllm:segmented-prefill-20260721` 容器，只读挂载完整仓库运行
7 个相关文件的 pycompile、Ruff lint 与 Ruff format check。首轮 pycompile/lint 通过，
format check 只要求 decode 和 prefill 两个 reader 机械排版，故退出码为 1；使用镜像内
同一 Ruff 只格式化这两个文件后，完整重跑结果为三项全部 exit0，7 个文件均已格式化。

随后仍在无 GPU 容器中，把当前 config 与第 10.100 节同一 CPU oracle 覆盖到已安装包，
定向重跑先前 3 个 geometry/容量失败断言，结果为
`3 passed, 9 deselected in 9.21s`、pytest exit0。它们分别确认默认 quant slot 为 72 B、
1024-token mixed bytes 使用 72 B history geometry，以及 8192 上下文 padded page 的
quant 部分使用 72 B。该结果与第 10.100 节 `3 failed` 构成同一 oracle 的 red/green
闭环，但只证明 CPU geometry 与容量公式，尚不能证明 CUDA writer/readers 已一致转绿。

静态日志、真实退出码、formatter 输出与 CPU pytest 日志位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_meta_production_green/`。本阶段
没有设置 `CUDA_VISIBLE_DEVICES`、没有向 Docker 透传 GPU，也没有产生新的 GPU 环境或
性能数据。下一步持续监控所有 GPU；同一物理卡连续两次空闲后固定该卡，重跑第 10.101
节同一 raw BF16 writer 与 64/128 维 roundtrip CUDA oracle。

### 10.103 BF16 Metadata Production CUDA Green

持续 monitor 于 UTC `04:08:41` 与 `04:09:58` 两轮确认 GPU 0--7 均为 0 MiB、
无计算进程，两轮间隔 77 秒；随后固定最低编号的物理 GPU 0，容器内设置
`CUDA_VISIBLE_DEVICES=0`。实验前 `nvidia-smi` 显示 GPU 0 为 NVIDIA B200、总显存
183359 MiB、使用 0 MiB、利用率 0%，Driver 595.71.05、宿主显示 CUDA 13.2，且无
运行进程。

本轮 Docker 镜像为 `oscar-vllm:segmented-prefill-20260721`，image ID
`sha256:db33e89ace8c789a87a022361e99f582e4b037a668d75ca4e6506b015a28dcef`；
当前 config、store、mixed-store、decode、prefill 与测试均以只读单文件 mount 覆盖到
镜像安装路径。容器软件环境为 Python 3.12.13、vLLM 0.25.0、PyTorch
2.11.0+cu130（`torch.version.cuda=13.0`）、Triton 3.6.0、Transformers 5.13.0。

UTC `04:10:19`--`04:11:10` 重跑第 10.101 节同一 raw BF16 writer 与 64/128 维
store/dequant roundtrip，结果为 `3 passed, 15 deselected in 44.34s`、pytest exit0。
这三项与旧 ABI 的 `3 failed` 一一对应：writer 已把 K/V scale+zero 写入同一 cache 的
BF16 typed view，两个 head dimension 的 reader 输出也已匹配“FP32 计算 INT2 indices、
BF16 metadata 落盘后反量化”的 reference。至此 config geometry、普通 store writer 与
full-dequant reader 形成 production red/green 闭环；demote、decode、materialize 与
fused prefill 的更广 CUDA 路径仍需完整回归覆盖，不能由这三项定向结果外推。

Container inspect 确认 exit0、`OOMKilled=false`；严格 traceback、runtime/CUDA error、
assertion 和实际 OOM 扫描为空。容器删除后 GPU 0--7 均为 0 MiB、0%，无计算进程。
连续空闲记录、完整 `nvidia-smi`、源码 SHA256、pytest 日志/退出码、container inspect、
软件版本与 GPU 回收状态位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_meta_production_green/cuda_green_monitor_20260721T040841Z/`。
下一步使用当前源码运行扩展 OSCAR CUDA 回归，覆盖 demote/decode/prefill/materialize，
通过后再构建不依赖源码覆盖的独立镜像。

### 10.104 BF16 Metadata Production 完整 CUDA 回归

这是新的 GPU 分配，故重新累计空闲记录。UTC `04:13:30` 与 `04:14:53` 连续确认
GPU 0--7 均为 0 MiB、无计算进程后，再次固定物理 GPU 0；容器内
`CUDA_VISIBLE_DEVICES=0`。Docker 镜像、image ID、NVIDIA B200 183359 MiB、Driver
595.71.05、宿主 CUDA 13.2，以及 Python/vLLM/PyTorch/Triton/Transformers 版本均与
第 10.103 节相同。当前五个 production 文件和测试仍以只读 mount 覆盖，因而本轮验证
的是磁盘当前源码，不是尚未构建的独立 BF16-metadata 镜像。

UTC `04:15:16`--`04:17:00` 使用全新 Triton cache 运行完整
`tests/quantization/test_oscar.py`，结果为
`18 passed, 120 warnings in 98.48s`、pytest exit0。相对第 10.95 节 segmented 镜像的
17 项完整回归，本轮多出的第 18 项是 raw BF16 metadata writer oracle；其余现有测试
同时实际执行 store/full-dequant、decode、mixed recent demote/store、mapped queries、
prefix page ownership、chunked cached/current prefill、poisoned prefix/recent、受限
materialize 和 V-rotation absorbed/non-absorbed 路径。Warnings 均为 FlashAttention/
CUTLASS/Swig 的 deprecation warning，没有测试失败或运行时错误。

Container inspect 确认 exit0、`OOMKilled=false`；严格 traceback、runtime/CUDA error、
assertion 与实际 OOM 扫描为空。容器删除后 GPU 0--7 均为 0 MiB、0%，无计算进程。
连续空闲记录、完整 `nvidia-smi`、pytest 日志/退出码、container inspect 与回收状态位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_meta_production_green/full_cuda_monitor_20260721T041330Z/`。
至此 BF16 metadata production ABI 已通过静态、CPU、定向 CUDA 与完整 CUDA 门禁；
下一步构建包含这些源码且无运行时源码覆盖的独立镜像，再运行完整 CUDA 打包回归。

### 10.105 BF16 Metadata Production 独立镜像

本阶段不分配 GPU。UTC `04:18:38` 前后使用现有
`vllm/docker/Dockerfile.oscar-vllm` 构建独立镜像
`oscar-vllm:bf16-meta-20260721`，Docker build exit0；完整 image ID 为
`sha256:087236b7fcef89d0d1704c937e35664f3f175e9e9f183be55500af9792f8d011`。
Base image 仍为 `docker.m.daocloud.io/vllm/vllm-openai:v0.25.0`，因此 Python 和相关
包版本保持第 10.103 节所列环境不变。

构建后在无 GPU 容器中逐文件计算 SHA256。镜像内 config、store、mixed-store、decode、
prefill 与完整 CUDA 测试的 hash 分别为 `7fdd9050...0bef`、`dd73482e...652`、
`d7ab6454...c30`、`dc78c434...2cf`、`a6c09d13...bb8`、`256278a7...7ff`，与构建前
磁盘当前文件逐项完全一致。该检查确认 72 B geometry、两个 BF16 writer、全部 BF16
reader 和新增 oracle 已实际打包进镜像，而不是继续引用第 10.94 节的旧 80 B 镜像。

构建日志/退出码、起止时间、image inspect 以及镜像内外完整 SHA256 位于
`artifacts/oscar_vllm_optimization/20260721/bf16_meta_image/`。本节只证明可复现打包
与源码一致性，没有 CUDA correctness 或性能数据。下一步重新按连续两次空闲规则固定
GPU，直接使用该镜像内 `/workspace/tests/quantization/test_oscar.py`，不挂载任何 host
路径，运行完整 CUDA 回归。

### 10.106 BF16 Metadata 独立镜像完整 CUDA 回归

这是另一项独立 GPU 分配。持续 monitor 于 UTC `04:20:00` 与 `04:21:29` 连续确认
GPU 0--7 均为 0 MiB、无计算进程后固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`。实验使用第 10.105 节独立镜像
`oscar-vllm:bf16-meta-20260721`，image ID `sha256:087236b7...8d011`；GPU、driver、
CUDA、Python 与相关包版本同第 10.103 节。

UTC `04:21:57`--`04:24:19` 直接运行镜像内完整 CUDA 测试，使用全新 Triton cache，
结果为 `18 passed, 134 warnings in 102.20s`、pytest exit0。Container inspect 确认
实际 image ID 为第 10.105 节 ID、`Mounts=[]`、exit0、`OOMKilled=false`；严格
traceback、runtime/CUDA error、assertion 与实际 OOM 扫描为空。因此本轮闭环了 BF16
metadata production ABI 的独立打包与完整 CUDA correctness，不再依赖任何 host 源码
覆盖。Warnings 仍全部是依赖包 deprecation warning。

容器已删除；实验后 GPU 0--7 均为 0 MiB、0%，无计算进程。连续空闲记录、完整
`nvidia-smi`、pytest 日志/退出码、container inspect 与回收状态位于
`artifacts/oscar_vllm_optimization/20260721/bf16_meta_image/full_cuda_monitor_20260721T041959Z/`。
下一步使用同一独立镜像和冻结的真实服务参数/payload，先确认 72 B quant slot 对实际
allocator token capacity 的影响，再做 operator profiler 与同 token-ID TTFT M2；不能把
第 10.99 节 artifact-only 组件加速直接当作服务结果。

### 10.107 BF16 Metadata 首轮真实服务失败与 Cache-Spec 缺口

本轮复用第 10.96 节完全相同的模型、rotation、10 GiB KV budget、prefix caching、
chunked prefill、`max_num_seqs=2`、profiler 配置，以及 SHA256 不变的三份 payload 与
runner；唯一有意变化是镜像切到第 10.105 节 BF16-meta 独立镜像。持续 monitor 于 UTC
`04:27:43`/`04:28:59` 连续确认 GPU 0--7 空闲后固定物理 GPU 0，容器内
`CUDA_VISIBLE_DEVICES=0`；GPU 与软件环境同第 10.103 节，HTTP client 使用独立无 GPU
容器。

服务于 UTC `04:29:29` 启动，但在健康就绪前于 `04:31:28` exit1、
`OOMKilled=false`。因此 workload 未发送任何模型请求，`workload.exit=99`，没有产生
有效 profiler trace、operator 时间或 TTFT，不能把本轮写成性能实验。实际根因是：

`RuntimeError: OSCAR KV allocation size mismatch: actual=298260480, expected=268696576`

`actual/expected=1.110028`，与旧 80 B/new 72 B slot 比例同量级。与此同时 planner 日志
仍报告 `INT2 history=461936 tokens (9.91 GiB)`，和第 10.96 节旧 80 B 镜像完全相同。
这证明第 10.102 节的五文件 ABI 切换仍不完整：kernel/config 已按 72 B reshape，但上游
`OscarKVCacheSpec.quant_slot_size`/page allocation 仍使用旧 geometry，导致 allocator
实际保留旧 byte 数，既没有获得容量提升，也无法启动服务。此前定向与完整 CUDA 单测
自行按 `cfg.slot_size_aligned` 构造 tensor，未经过生产 cache planner，因而没有覆盖该
跨层 contract；第 10.104/10.106 节的 correctness 结论仍有效，但不足以证明 serving。

容器和独立 client 已删除，实验后 GPU 0--7 均为 0 MiB、0%，无计算进程。连续空闲、
完整服务日志、health 失败、container inspect、错误扫描与回收记录位于
`artifacts/oscar_vllm_optimization/20260721/torch_profile_bf16_meta/`。下一步先新增
cache-spec/page allocation 合约 oracle，使旧上游 80 B 值明确失败；然后只修正该
geometry 来源，重跑 CPU/static 与真实服务。Profiler 和 TTFT 在服务成功前保持未运行。

### 10.108 BF16 Metadata Cache-Spec 失败 Oracle

为覆盖第 10.107 节遗漏的 production planner contract，本轮在现有 core 测试中新增
无 GPU oracle：构造 `quant_slot_size=72` 的 `OscarKVCacheSpec`，以 10 GiB、36 层、
8 KV heads、block size 16 和既有 prefix/recent reserve 直接计算 quant page 数，再要求
`get_kv_cache_config_from_groups` 的 `num_blocks` 与每层 backing tensor 大小使用该 spec
geometry。该计算不依赖 `OscarKVCacheGeometry` 内部默认值，能避免 planner 和 worker
各自用一份 metadata 精度配置而再次漂移。

测试文件 pycompile 与 Ruff lint 首轮通过，format check 只要求新增表达式机械排版；
使用镜像内同一 Ruff 格式化后，完整静态门禁 exit0。随后在无 GPU 的第 10.105 节镜像
上只读挂载当前测试，定向 red run 为 `1 failed, 26 deselected in 0.94s`、pytest exit1：

- planner 实际返回 `28999` quant pages；
- 72 B spec 期望 `32221` quant pages；
- planner 日志仍报告旧的 `463984` INT2 history tokens。

失败精确发生在 `result.num_blocks == expected_num_blocks`，没有 import、platform 或其他
断言错误，与第 10.107 节服务端旧 capacity/mismatch 相互印证。静态日志、formatter、
red pytest 与真实退出码位于
`artifacts/oscar_vllm_optimization/20260721/bf16_meta_cache_spec_red/`。本阶段没有分配
GPU。下一步让 planner 的 `OscarKVCacheGeometry` 显式接收
`OscarKVCacheSpec.quant_slot_size`，删除其独立推导 metadata byte 数的歧义，并把现有
10 GiB capacity oracle 全部更新到 72 B 实际 geometry。

### 10.109 BF16 Metadata Cache-Spec 修复与 CPU Green

Planner 修复严格限定为两处 production 改动：`OscarKVCacheGeometry` 新增必填
`quant_slot_size`，其 `quant_token_bytes` 直接使用该 slot；
`get_kv_cache_config_from_groups` 从当前 `OscarKVCacheSpec.quant_slot_size` 原样传入。
旧的 planner-local `scale_element_size=4` 及重复 packed-byte 推导已删除，因而 config、
attention spec、planner、backing tensor 和 worker reshape 共享同一个 geometry 来源。
Allocator 的 prefix/recent reserve、page 数整除、ownership 与回收策略均未修改。

Core oracle 同步使用 72 B Qwen3 slot：单层 quant page 从 10240 B 降至 9216 B；
10 GiB 下 maxseq 1/8/48 的 quant slots 分别为 515536/499600/408576，保证容量分别为
515521/499480/407856。单请求 physical ratio 实际为 7.079982x，仍高于 6.2x 门槛。
首轮全文件 CPU 回归有两项失败：一项是测试期望手算未按 BF16 page 向下取整，已从
7.079885 修正为实际 7.079982；另一项是无 GPU 容器中的既有 scheduler-preemption
用例无法自动推断 `DeviceConfig`，失败发生在本次 geometry 逻辑之外。

修正测试常量后，三个改动文件 pycompile、Ruff lint/format 全部 exit0；无 GPU 重跑
其余测试为 `26 passed, 1 deselected, 16 warnings in 5.31s`、pytest exit0。第 10.108
节新增 oracle 已转绿，maxseq 1/8/48 容量、三段式 backing split、allocator lifecycle 与
accounting 也全部通过。唯一 deselect 即上述需要 CUDA platform detection 的既有
preemption 用例，不是失败或跳过本次新增 oracle。

静态/CPU 日志与退出码位于
`artifacts/oscar_vllm_optimization/20260721/bf16_meta_cache_spec_green/`；本阶段没有
分配 GPU。当前 core geometry、planner 与测试 SHA256 分别为 `9ab4e52e...52e6`、
`8abfb262...607f`、`7da9752a...c114`。下一步重建独立镜像，再重新运行第 10.107 节
同一真实服务；成功标准是 history capacity 变为实际 72 B 结果、无 allocation mismatch、
服务健康且冻结 profiler workload 完成。

### 10.110 BF16 Metadata Cache-Spec 修复镜像

本阶段不分配 GPU。使用当前完整源码构建
`oscar-vllm:bf16-meta-specfix-20260721`，Docker build exit0，image ID 为
`sha256:a2898419196c6dacc854376f9e90106035368cb2ae7a8959d7dcd6f1c60ae75b`。
构建上下文和 base image 与第 10.105 节一致。

构建后在无 GPU 容器中逐项核验 core geometry、planner、OSCAR config、store、mixed-
store、decode、prefill、core 测试和 CUDA 测试共 9 个文件；镜像内外 SHA256 全部
一致。其中新 planner/geometry/test hash 为第 10.109 节记录的三项值，config 与四个
kernel hash 与第 10.105 节完全相同。这证明新镜像只增加 cache-spec/planner contract
修复及相应测试，没有改变已经通过完整 CUDA 的 kernel 字节。

构建日志/退出码、起止时间、image inspect 和完整 hash 清单位于
`artifacts/oscar_vllm_optimization/20260721/bf16_meta_specfix_image/`。本节没有服务、
CUDA correctness 或性能结果。下一步重新持续监控 GPU，使用该镜像和第 10.107 节冻结
服务/runner；先以实际 `513264` 左右的 maxseq2 history capacity、健康就绪和 workload
exit0 验证 planner，再分析 profiler trace。

### 10.111 BF16 Metadata 修复后真实容量与 Operator Profiler

本轮严格复用第 10.96 节的服务参数、三份 payload、runner 与调用顺序；runner/vector/
matrix/tensor SHA256 仍为 `aa4893ec...256d`、`0bfa9352...7ecd`、
`0c33cf27...2cf3`、`3e762929...b92f`。唯一有意变化是使用第 10.110 节 spec-fix 镜像。
持续 monitor 于 UTC `04:42:21`/`04:43:47` 连续确认 GPU 0--7 空闲后固定物理
GPU 0；容器内 `CUDA_VISIBLE_DEVICES=0`。Docker image ID、NVIDIA B200 183359 MiB、
Driver 595.71.05、宿主 CUDA 13.2，以及 Python 3.12.13、vLLM 0.25.0、PyTorch
2.11.0+cu130/CUDA 13.0、Triton 3.6.0、Transformers 5.13.0 均已落盘。

服务于 UTC `04:44:16` 启动、`04:46:31` 健康就绪。实际 planner 日志为
`INT2 history=513264 tokens (9.91 GiB), BF16 prefix=128, BF16 recent=512`，并报告
GPU KV cache size 513264 tokens；第 10.107 节 allocation mismatch 消失。相对第
10.96/10.107 节旧 80 B capacity 461936，实际增加 51328 tokens、11.111496%。若用
第 10.1 节同一 10 GiB BF16 72816 slots 作分母，当前 maxseq2 history physical ratio
为 7.048780x，旧值为 6.343880x；这是一项实际服务 allocator 结果，不是 80/72 外推。

独立无 GPU client 于 UTC `04:46:31`--`04:46:54` 完成 5 个请求并 exit0；全部 HTTP
200、prompt 2100 tokens，cached 序列严格为 `0/1840/0/1840/0`。两份原始 trace 按与
第 10.96 节相同口径重算：

| 指标 | Segmented（第 10.96 节） | BF16 metadata | 变化 |
| --- | ---: | ---: | ---: |
| 36 层 cached kernel | 14.867568 ms | 9.039926 ms | -5.827642 ms，1.644656x |
| Worker CPU hit-cold gap | 11.570078 ms | 13.153221 ms | +1.583143 ms |
| Attention CPU hit-cold gap | 10.968971 ms | 10.588453 ms | -0.380518 ms |
| GPU kernel+memcpy hit-cold gap | +1.916497 ms | -3.820392 ms | -5.736889 ms |

相对 fused-default 的 17.398783 ms，当前 cached kernel 节省 8.358857 ms、提速
1.924660x。Cold/hit GPU kernel+memcpy 分别为 18.918789/15.098397 ms；cached kernel
仍为 36 次，materialize 为 0 次，GPU kernel、memcpy、`cudaLaunchKernel` 计数仍分别
为 cold/hit 606/786、48/48、268/340。故收益来自相同 launch 内的 BF16 metadata 读取，
不是减少请求工作量、launch 数或启用 full-history materialize。需要保留 Worker CPU gap
增加这一反向信号；operator trace 仍不是统计 TTFT，M2 尚未宣告通过。

Analyzer 首轮 Ruff 因 import 后多一个空行报 I001，但同一原始 trace 已成功解析；按
Ruff 精确 diff 删除空行后，pycompile/lint/format 全部 exit0，重算 summary 与首份逐字节
一致，未重复 GPU 实验。服务停止前和最终错误扫描均为空，容器 exit0、
`OOMKilled=false`；停止后 GPU 0--7 均为 0 MiB、0%，无计算进程。连续空闲、完整
`nvidia-smi`、服务日志、请求响应、两份 trace、分析器/summary、inspect 与回收记录位于
`artifacts/oscar_vllm_optimization/20260721/torch_profile_bf16_meta_specfix/`。下一步
使用第 10.97 节同一 token-ID client 做 warm-up 后配对 TTFT，检验负 GPU gap 能否传递
到端到端 M2。

### 10.112 BF16 Metadata Prefix-Hit TTFT M2 复测

本轮直接复用第 10.97 节 token-ID client，SHA256 仍为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`；
服务参数与第 10.111 节相同但关闭 profiler，端口保持冻结 client 使用的 8441。持续
monitor 于 UTC `04:51:42`/`04:53:07` 连续确认 GPU 0--7 空闲后固定物理 GPU 0，
容器内 `CUDA_VISIBLE_DEVICES=0`。镜像、GPU、driver、CUDA、Python 与包版本同第
10.111 节，client 仍是独立无 GPU 容器。

服务于 UTC `04:53:39` 启动、`04:55:52` 健康就绪，日志再次确认实际 history capacity
513264 tokens。Client 先执行 2 对 warm-up；前 5 对 cold/hit CV 为 4.572%/1.909%，
因此按冻结规则扩展到最多 15 对。最终 30 个正式请求全部 HTTP 200、prompt 2100
tokens，所有 cold cached=0、所有 hit cached=1840；benchmark exit0。结果如下：

| 指标 | Segmented（第 10.97 节，5 对） | BF16 metadata（15 对） | 变化 |
| --- | ---: | ---: | ---: |
| Cold TTFT 均值 | 32.907069 ms | 33.649515 ms | +0.742446 ms |
| Hit TTFT 均值 | 40.132638 ms | 41.942152 ms | +1.809515 ms |
| Cold / hit CV | 2.677% / 2.928% | 4.191% / 3.383% | 当前仍 >3% |
| Hit-cold 配对均值 | +7.225569 ms | +8.292637 ms | +1.067068 ms |

15 个逐对差全部为正，范围 `+5.948790` 至 `+10.520901 ms`，配对均值
`+8.292637 ms`、标准差 1.270374 ms；首 5 对配对均值也为 `+8.172545 ms`，扩展后
方向与量级未改变。最终单侧 CV 仍超过 3%，作为实际服务噪声保留，不能选择性使用后
10 对较低绝对值；但 15/15 hit 更慢已明确表明 **M2 仍然失败**。1840/2100=87.619%
的 tokens 命中，功能门槛继续满足，失败只在端到端 TTFT。

这也说明第 10.111 节 `-3.820392 ms` GPU hit-cold gap 没有传递到客户端：Worker CPU
span 的 `+13.153221 ms` gap 和 profiler 外的调度/streaming 首 token 路径仍占主导。
不能用 cached kernel 1.644656x 加速或容量 +11.1115% 替代 M2 结论。派生 paired jq
首轮因根对象作用域错误得到空文件，绑定 `$req` 后从同一有效 summary 重算；没有重复
服务或请求。

服务停止前和最终错误扫描均为空，容器 exit0、`OOMKilled=false`；UTC `04:56:06`
停止后 GPU 0--7 均为 0 MiB、0%，无计算进程。连续空闲、完整 `nvidia-smi`、服务日志、
30 个请求、summary、paired stats、inspect 与回收状态位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_meta_ttft/`。下一步不能继续只
优化 cached GPU kernel；应以 Worker CPU/profiler 外首 token 路径为目标形成新的可测
候选，同时保留当前 72 B capacity 与 kernel 收益。

### 10.113 BF16 Metadata Cold/Hit CPU 路径细分

本阶段没有重新分配 GPU，也没有重新运行服务；只读取第 10.111 节同一 B200、同镜像、
同 payload 生成的 cold/hit 原始 Torch trace。分析器按同一 CPU thread 的时间区间建立
嵌套关系：36 个 `vllm::unified_attention_with_output` parent 的 wall time，减去其
不重叠的直接子事件区间，定义为 trace 未覆盖的 attention self time。该值包含 Python/
dispatcher、Triton launch wrapper 以及 profiler 自身未分类开销，不能直接当成纯 Python
或纯 Triton JIT 时间。

可复现细分如下：

| 指标 | Cold | Hit | Hit-cold |
| --- | ---: | ---: | ---: |
| Worker span | 34.555459 ms | 47.708680 ms | +13.153221 ms |
| 36 层 attention wall | 17.290234 ms | 27.878687 ms | +10.588453 ms |
| Worker 中 attention 外时间 | 17.265225 ms | 19.829993 ms | +2.564768 ms |
| Attention self time | 10.654629 ms | 16.656622 ms | +6.001993 ms |
| Attention 直接子事件 | 6.635605 ms | 11.222065 ms | +4.586460 ms |

可见的 4.586460 ms 直接子事件增量中，主要项是额外 36 次 Q rotation `aten::matmul`
的 1.889827 ms、72 次 `aten::to` 的 0.641332 ms、36 次
`_C::merge_attn_states` 的 0.557659 ms、72 次 `cuLaunchKernelEx` 的 0.426919 ms，
以及 `empty_like`/`empty` 的 0.582929 ms。更大的单项是 6.001993 ms attention self
增量，约占 attention wall 增量的 56.684%；因此第 10.112 节的 TTFT 回退不能只用 Q
rotation 或 cached kernel GPU 时间解释。

无 GPU 容器内读取 Triton 3.6.0 的 `JITFunction.run` 实现，确认每次普通 `kernel[grid]`
调用都会重复取得 device/stream、执行 binder、计算 cache key、检查已使用全局值、规范化
grid，再进入已编译 `CompiledKernel.run`。这为“缓存 compiled kernel 后直接启动”提供了
源码依据，但当前 trace 没有把 6.001993 ms 分解到上述各步骤，故它仍只是下一轮
artifact-only A/B 候选，不能提前写成已证实根因或 production 收益。

分析器首次静态检查的 pycompile/lint 通过、format check 要求机械排版；包装脚本又误取
`tee` 对应的 `PIPESTATUS[1]`，使首轮 `.exit` 错写为 0。保留该无效门禁日志后，使用
container Ruff 格式化，并以本命令正确的 `PIPESTATUS[0]` 重跑，最终 pycompile、Ruff
lint/format 全部 exit0。分析器与 JSON SHA256 分别为 `dd23f3d...605b`、
`67a1585c...b0a`，artifact 位于
`artifacts/oscar_vllm_optimization/20260721/bf16_meta_cpu_path_analysis/`。下一步先以
production shape 比较标准 Triton JIT launch 与 cached `CompiledKernel` direct launch
的 output/LSE、GPU 时间和 host dispatch；只有 correctness 与稳定 host 收益同时成立，
才修改 production wrapper。

### 10.114 Compiled-Kernel Launch A/B 静态门禁

已新增 artifact-only A/B 脚本，production 源码未修改。脚本直接导入第 10.110 节镜像
中的 72 B `_oscar_cached_prefill_kernel`，固定真实 full-hit shape：260 个 query token、
1840 个 cached token、32 query heads、8 KV heads、head dimension 128、32x32 tile、
GQA2。标准路径继续调用 `JITFunction.run`；候选在首次标准调用返回
`CompiledKernel` 后缓存 `compiled_kernel[grid]` runner，后续只传入当前 tensor/scalar
runtime arguments，不缓存 output、LSE、KV cache 或 stream 指针。

Correctness 将逐元素比较标准/候选的 output 与 LSE，场景同时覆盖
`shared_hit=1840` full hit 和 `shared_hit=0` 的未对齐 recent tail。性能分为两种口径：
CUDA event 使用 20 次预热后 5 批、每批 100 次；host enqueue 使用每批同步前后各隔离
20 次不含 synchronize 的 `perf_counter_ns` 测量，共 10 批并交错顺序。后者只衡量 CPU
提交调用的 wall time，不冒充 GPU latency 或服务 TTFT。

首轮 pycompile 通过，但 Ruff lint 要求按项目规则合并 import 分组；按只读 fix diff
修正后，第二轮 lint 通过而 format check 要求机械排版。使用同镜像 Ruff formatter 后，
最终 pycompile、Ruff lint/format 全部 exit0，脚本 SHA256 为
`2d5ed179e5c89c439e16a9ce049cde269fffb71f0e937a6c73c201d9c1a9346c`。所有静态日志、
退出码和脚本位于
`artifacts/oscar_vllm_optimization/20260721/prefill_compiled_launch_microbench/`。本阶段
未透传 GPU，没有 correctness、host enqueue 或 GPU 时间结果。下一步持续监控全部 GPU；
只有同一张卡连续两次空闲后才固定运行该 SHA 的脚本，若 Triton compiled ABI 拒绝直接
runtime arguments，则如实保留失败，不修改 production 规避。

### 10.115 Compiled-Kernel Launch 首轮 ABI 失败

这是新的 GPU 分配。UTC `05:09:48` 与 `05:11:19` 连续确认 GPU 0--7 为 0 MiB、
无计算进程后固定物理 GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`。实验使用第 10.110 节
镜像 `oscar-vllm:bf16-meta-specfix-20260721`，image ID
`sha256:a2898419...ae75b`；NVIDIA B200 183359 MiB、Driver 595.71.05、宿主 CUDA
13.2，以及 Python 3.12.13、vLLM 0.25.0、PyTorch 2.11.0+cu130/CUDA 13.0、Triton
3.6.0、Transformers 5.13.0 均与第 10.111 节相同。

容器于 UTC `05:11:58`--`05:12:53` 执行第 10.114 节冻结 SHA 的脚本，在 Triton 冷编译
完成后首次调用 cached compiled runner 时 exit1、`OOMKilled=false`：

`TypeError: function takes exactly 60 arguments (43 given)`

该计数精确解释了缺口：launcher 的 60 项包含 13 个 grid/stream/function/metadata/hook
内部参数和原 kernel 的 47 个参数；脚本传入 30 个 runtime tensor/scalar 后总计 43，
遗漏的正是 kernel signature 中 17 个 `tl.constexpr` 位置。普通 `JITFunction.run`
会把完整 `bound_args.values()` 传给 compiled launcher，即使 constexpr 已参与编译；
直接调用也必须按原 signature 顺序传入相同常量。失败发生在 full-hit/recent-tail
correctness、host enqueue 和 CUDA event 正式采样前，因此本轮没有有效数值或性能结果。

严格错误扫描只命中上述预期 traceback；容器删除后 GPU 0--7 均为 0 MiB、0%，无计算
进程。连续空闲记录、完整 `nvidia-smi`、脚本日志、inspect、错误扫描和回收状态位于
`artifacts/oscar_vllm_optimization/20260721/prefill_compiled_launch_microbench/monitor_20260721T050948Z/`。
下一步只修正 artifact 脚本：按 kernel 定义顺序追加 17 个完全相同的 constexpr 值，
不修改 production，也不改变 benchmark shape/次数；静态转绿后必须重新执行连续两次
GPU 空闲检查，不能复用本轮分配记录。

### 10.116 Compiled-Kernel Launch ABI 修正静态门禁

Artifact 脚本已按第 10.115 节边界修正：在每次 direct runner 调用的 30 个动态参数后，
严格按 kernel signature 顺序追加 17 个相同常量，依次为 head/geometry、量化布局、
attention scale、prefix/recent window 和 tile 参数。`num_warps=4`、`num_stages=2` 仍只
作为编译 option 传给首次 `JITFunction.run`，不错误地加入 kernel 参数。Candidate 仍
不缓存 tensor、stream 或输出地址，也没有改动 production 文件。

使用第 10.110 节镜像、无 GPU、只读挂载修正版脚本，pycompile、Ruff lint/format 一次
全部 exit0。新脚本 SHA256 为
`a0cd45ac08cfd6e7dced889fa60d93d5422b68a2bac8d45a4682f53d1437bfdd`；静态日志和退出码
仍位于
`artifacts/oscar_vllm_optimization/20260721/prefill_compiled_launch_microbench/`。
本阶段没有 CUDA correctness 或性能结果。下一步从零开始持续监控 GPU，必须重新获得
同一物理卡连续两次空闲后运行该 SHA；首轮失败的空闲记录不复用。

### 10.117 Compiled-Kernel Launch CUDA A/B 与否决

UTC `05:15:56`/`05:17:16` 重新连续确认 GPU 0--7 为 0 MiB、无计算进程后，固定
物理 GPU 0；容器内 `CUDA_VISIBLE_DEVICES=0`。镜像、image ID、GPU、driver、CUDA、
Python 和包版本均与第 10.115 节相同。修正版脚本于 UTC `05:17:43`--`05:18:36`
运行完成，容器 exit0、`OOMKilled=false`。

标准 JIT 与 cached compiled runner 在 full prefix hit 和未对齐 recent tail 两个场景中，
output/LSE max abs 均为 0，逐元素完全一致。20 次预热后，五批、每批 100 次的 CUDA
event 结果如下：

| 路径 | GPU 均值 | CV | 相对标准路径 |
| --- | ---: | ---: | ---: |
| `JITFunction.run` | 0.234351 ms | 0.1618% | 1.000000x |
| Cached `CompiledKernel[grid]` | 0.234003 ms | 0.0192% | 1.001488x |

GPU 差值只有 -0.000348 ms，符合两者启动同一个 compiled kernel 的预期。十批、每批
20 次、同步区间之间的 host enqueue 结果为：

| 路径 | Host enqueue 均值 | CV | 相对标准路径 |
| --- | ---: | ---: | ---: |
| `JITFunction.run` | 23.822000 us | 0.8764% | 1.000000x |
| Cached `CompiledKernel[grid]` | 12.666590 us | 1.4453% | 1.880696x |

候选每层实际节省 11.155410 us，按 36 层线性上限仅 0.401595 ms，分别只占第 10.113
节 6.001993 ms attention self gap 的 6.6910%，以及第 10.112 节 8.292637 ms TTFT gap
的 4.8428%。这证明 JIT binder/cache-key 确实有可测成本，但也反证它不是剩余回退的
主因。为不足半毫秒的上限引入 Triton 3.6 私有 generated-launcher ABI、手工 constexpr
顺序和 source-change 检查绕过，风险与收益不相称，故 **否决该 production 候选**；当前
production 文件保持未修改。

严格错误扫描为空；容器删除后 GPU 0--7 均为 0 MiB、0%，无计算进程。连续空闲、完整
`nvidia-smi`、原始 JSON、日志、inspect 与回收记录位于
`artifacts/oscar_vllm_optimization/20260721/prefill_compiled_launch_microbench/monitor_20260721T051556Z/`；
results JSON SHA256 为 `99cc67d1...8a6d`。下一候选必须至少消除 Q rotation、suffix/
cached merge 或多次 allocation/launch 中的组合成本；不再沿 private compiled launcher
继续优化，也不为该候选重跑 TTFT。

### 10.118 Full Mixed-Prefill Fused 候选与静态门禁

下一 artifact-only 候选不再绕过 Triton launcher，而是减少实际算子数量。当前 hit 每层
依次执行 current suffix FlashAttention、cached 三段式 Triton attention，再分配输出并用
`merge_attn_states` 合并两个 LSE。候选保留预先旋转的 Q/K 和 absorbed V，在现有 cached
kernel 的三段 loop 后追加 current K/V causal loop，使同一个 online softmax 直接覆盖
1840 cached + 260 current tokens并输出最终 attention/LSE。它不物化 cached K/V、不新增
history-sized workspace，也不改变三段式地址或 BF16 metadata ABI。

脚本固定第 10.117 节相同 production shape、32x32/GQA2 和 72 B quant slot。Baseline
实际调用 FlashAttention、当前 production `oscar_cached_prefill_attention` 和 C++ merge；
candidate 只调用新 kernel。两侧使用相同的 FP32 rotated Q、FP32 rotated current K、
BF16 absorbed current V 及三段式 cache。Correctness 同时覆盖 full prefix hit 与未对齐
recent tail，并比较最终 output 和由两侧 online-softmax 得到的最终 LSE；性能仍使用
20 次预热、五批每批 100 次 CUDA event，以及十批每批 20 次 host enqueue。

Candidate kernel 由当前 production 文件机械复制后只修改 artifact：新增 current K/V
指针及独立三维 stride、`MAX_QUERY_LEN`，并以 `current_offset <= query_position` 的二维
mask 实现 causal suffix。首轮 pycompile 通过，Ruff 只要求两个 artifact 文件调整 import
分组；修正后 lint 通过、format check 要求机械排版。使用同镜像 Ruff formatter 后，
最终两文件 pycompile、Ruff lint/format 全部 exit0。Benchmark/kernel SHA256 分别为
`b80b14b6...e67aa`、`eeb406b3...685c7`，完整脚本、静态日志与退出码位于
`artifacts/oscar_vllm_optimization/20260721/prefill_full_fused_microbench/`。

本阶段未分配 GPU，Triton candidate 尚未经历 CUDA 编译，因此不能宣称资源占用、数值
正确或性能收益。下一步持续监控所有 GPU，重新取得同一卡连续两次空闲后固定运行；若
kernel 编译失败、数值越界或 GPU/host 任一侧无稳定收益，候选不得进入 production。

### 10.119 Full Mixed-Prefill Fused CUDA A/B

UTC `05:28:05`/`05:29:37` 连续确认 GPU 0--7 为 0 MiB、无计算进程后固定物理
GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`。实验使用第 10.110 节 spec-fix 镜像；image
ID、NVIDIA B200 183359 MiB、Driver 595.71.05、宿主 CUDA 13.2，以及 Python/vLLM/
PyTorch/Triton/Transformers 版本均与第 10.115 节相同。

修正版脚本于 UTC `05:30:03`--`05:31:05` 完成，容器 exit0、`OOMKilled=false`。
Correctness 结果为：

| 场景 | Output max abs | LSE max abs | 结果 |
| --- | ---: | ---: | --- |
| Full prefix hit | 0.0009765625 | 2.861023e-6 | 通过 |
| Request recent tail | 0.0009765625 | 1.907349e-6 | 通过 |

20 次预热后，五批、每批 100 次 CUDA event 与十批 host enqueue 结果如下：

| 路径 | GPU 均值 / CV | Host enqueue 均值 / CV |
| --- | ---: | ---: |
| FA + cached + merge baseline | 0.260099 ms / 0.1415% | 156.451520 us / 2.0862% |
| Full mixed fused candidate | 0.250134 ms / 0.0628% | 33.754700 us / 2.3012% |

Candidate GPU 加速 1.039840x、每层节省 0.009965 ms；host enqueue 加速 4.634955x、
每层节省 122.696820 us。按 36 层线性预算分别为 0.358749 ms GPU 和 4.417086 ms
host，host 部分相当于第 10.113 节 attention self gap 的 73.5936%、第 10.112 节 TTFT
gap 的 53.2651%。GPU+host 相加的 4.775834 ms 只是组件线性预算，不能作为服务 TTFT
预测或把 M2 提前标记为通过。

两侧 CV 均低于 3%，correctness 与 GPU/host 收益同时满足进入 production oracle 的
门槛。严格错误扫描为空；日志中的 CUTLASS/SWIG 条目均为 deprecation warning。容器
删除后 GPU 0--7 均为 0 MiB、0%，无计算进程。连续空闲、完整 `nvidia-smi`、原始
JSON、日志、inspect 与回收记录位于
`artifacts/oscar_vllm_optimization/20260721/prefill_full_fused_microbench/monitor_20260721T052805Z/`；
results JSON SHA256 为 `a3af504a...1845`。下一步新增 production backend oracle，要求
absorbed cached-prefill 不再调用 suffix FlashAttention/merge 且仍匹配现有 mixed
reference；先形成 red，再合入最小 kernel/wrapper/backend 改动。

### 10.120 Full Mixed-Prefill Production 失败 Oracle 静态门禁

在修改 production 前，现有 absorbed/non-absorbed mixed backend oracle 已加严。它原本
覆盖一个 384-token cached request 和一个 cold request、17/13 current tokens、不同 K/V
stride、三段式 store/demote，以及 fused/materialized 两条结果对 FP32 reference。新增约束
仅作用于 `absorb_v_rotation=True` 的默认 fused 分支：暂时把
`impl._flash_attn_varlen` 替换为必抛 `AssertionError` 的函数，要求 cached+current
attention 在不调用 suffix FlashAttention 的情况下完成；首次 fused forward 后立即恢复
原方法，故 materialize 与 non-absorbed 原语义不变。

使用第 10.110 节镜像、无 GPU、只读挂载当前测试，pycompile、Ruff lint/format 一次
全部 exit0；测试 SHA256 为 `e2a0ab30...0c399`。静态日志和退出码位于
`artifacts/oscar_vllm_optimization/20260721/prefill_full_fused_production_red/`。本阶段
没有运行 pytest，尚未形成 CUDA red。下一步重新持续监控 GPU；连续两次空闲后只运行
该 mixed oracle，预期旧 production 的 absorbed 参数在目标 suffix FA 调用处失败，而
non-absorbed 参数继续通过。

### 10.121 Full Mixed-Prefill Production CUDA Red

UTC `05:34:55`/`05:36:27` 连续确认 GPU 0--7 为 0 MiB、无计算进程后固定物理
GPU 0；容器内 `CUDA_VISIBLE_DEVICES=0`。旧 production 的 spec-fix 镜像、image ID、
GPU 与软件环境同第 10.119 节。只读挂载第 10.120 节测试，UTC `05:36:43`--
`05:37:27` 定向运行两个参数，结果为
`1 failed, 1 passed, 16 deselected, 120 warnings in 40.13s`、pytest exit1。

`absorb_v_rotation=False` 继续通过；唯一失败是 `True` 参数在旧 backend
`_prefill_attention` 第 723 行调用 `_flash_attn_varlen` 时精确触发新增断言：

`AssertionError: absorbed cached prefill must fuse current attention`

失败发生在目标控制流，没有 import、CUDA、数值 reference 或无关 assertion 失败，因此
形成有效 production red oracle。Warnings 均为既有 CUTLASS/SWIG deprecation warning。
容器 exit1、`OOMKilled=false` 并已删除。

需要保留一项并发环境事实：连续空闲检查和本容器启动后，外部
`sglang.launch_server --model-path .../DeepSeek-V4-Pro --tp 8` 于 UTC `05:36:53`
启动 8 个子进程，并在本轮结束时占用 GPU 0--7 各约 124.4--124.7 GiB。因此回收快照
不是全卡空闲；这些 PID 的共同父进程为 1645210，启动时间晚于本容器，且本容器已由
Docker inspect/rm 闭环，不能把外部占用写成本实验泄漏。该竞争不改变“旧 absorbed
控制流必调用 suffix FA”的 red 结论，但后续 CUDA green 必须等待外部作业释放，并重新
获得同一卡连续两次空闲。

连续空闲、pytest 日志、container inspect、外部进程与结束 GPU 状态位于
`artifacts/oscar_vllm_optimization/20260721/prefill_full_fused_production_red/monitor_20260721T053455Z/`。
下一步可在不分配 GPU 的情况下合入最小 production kernel/wrapper/backend 改动和静态
门禁；CUDA green monitor 必须持续运行直至满足空闲规则。

### 10.122 Full Mixed-Prefill Production 合入与静态门禁

已按第 10.119 节通过的候选做最小 production 合入。现有 cached-prefill Triton kernel
新增可选 current K/V 指针、各自独立 stride 和 causal current loop；只有 wrapper 同时
收到 current K/V 时才令 `MAX_QUERY_LEN=max_query_len`，原有调用则设为 0，使该 loop
在编译期关闭。三段式 cached source 与 current source 继续共享同一 `m_i/l_i/acc` online
softmax，不物化完整历史 BF16 K/V，也不增加随 history 长度增长的 workspace。

Backend 只在 V rotation 已 absorption 且存在 cached context 时启用 full fused：forward
预先计算一次 current K rotation，将同一 tensor 同时传给 fused attention 与 mixed-cache
store；absorbed current V 直接复用原 V view。该分支从 full fused output 直接执行 V
inverse，不再调用 suffix FlashAttention 或 LSE merge。`absorb_v_rotation=False` 仍走
原 cached attention、suffix FlashAttention 和 merge；bounded materialize oracle 也保持
原路径，因此没有扩大 absorption 或 workspace 的适用边界。

使用镜像 `oscar-vllm:bf16-meta-specfix-20260721`、不透传 GPU、只读挂载当前仓库进行
静态检查。首轮 pycompile 与 Ruff lint 通过，但 format check 要求机械排版 prefill kernel；
用同镜像 Ruff formatter 只处理该文件后，完整重跑三个文件的 pycompile、Ruff lint/format
全部 exit0，目标文件 `git diff --check` 也无输出。最终 SHA256 为：

- backend：`121591df...ec5`；
- prefill kernel/wrapper：`e1c764e6...4715`；
- production oracle：`e2a0ab30...c399`。

本阶段没有分配 GPU，因此尚不能宣称 production Triton 编译或数值 green。下一步持续
监控 GPU 状态；只有同一物理卡连续两次空闲后才运行第 10.120 节两个参数的 CUDA oracle。
若暂时没有满足条件的 GPU，监控不会停止或把任务标记为 blocked。

### 10.123 Full Mixed-Prefill Production CUDA Green

从第 10.121 节外部占用开始，监控以 65 秒间隔持续运行，没有因前三轮全卡忙而退出。
UTC `05:46:50` 与 `05:48:04` 连续两轮确认 GPU 0--7 均为 0 MiB、0%、无 compute
process，随后固定物理 GPU 0；容器内 `CUDA_VISIBLE_DEVICES=0`。启动前即时快照为
GPU 0 使用 4 MiB、其余 1--4 MiB、全部 0% 且无 compute process。

实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:bf16-meta-specfix-20260721` |
| 镜像 ID | `sha256:a2898419...ae75b` |
| GPU | NVIDIA B200，183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| `CUDA_VISIBLE_DEVICES` | `0` |

当前 backend、prefill kernel 和第 10.120 节测试覆盖到镜像 installed package，保留镜像
内编译扩展，并使用全新 `TRITON_CACHE_DIR`。容器于 UTC `05:49:17`--`05:50:55`
运行两个参数，实际结果为：

`2 passed, 16 deselected, 120 warnings in 45.62s`

`absorb_v_rotation=False` 的既有 cached + suffix FA + merge 路径继续通过；
`absorb_v_rotation=True` 在 oracle 禁止调用 suffix FlashAttention 的条件下通过 full fused
路径。两者均与既有 FP32 mixed-prefill reference 一致，并同时覆盖一个 384-token cached
request、一个 cold request、17/13 current tokens、prefix/recent/INT2 三段式及非连续
fused-QKV V stride。Warnings 均为既有 CUTLASS/SWIG deprecation warning；严格 traceback/
CUDA error/runtime error/OOM 扫描为空。容器 exit0、`OOMKilled=false` 并已删除。

运行中再次发生外部竞争：新的 `sglang.launch_server --model-path .../DeepSeek-V4-Pro
--tp 8` 父进程 1676928 于 UTC `05:50:09` 启动，晚于本容器启动，并生成 GPU PID
1677569 等八个子进程。结束快照先显示每卡约 866 MiB，随后增长到约 124.4--124.7 GiB；
本容器只透传物理 GPU 0，Docker device request、exit 状态和删除记录均已落盘，因此这些
八卡进程不是本实验泄漏。该竞争不影响已经通过的控制流与数值断言，但本轮不得作为
性能数据；后续 profiler/TTFT 必须重新持续等待连续空闲。

连续监控、五轮 snapshot、完整 `nvidia-smi`、pytest、环境、container/image inspect、
严格扫描与回收记录位于
`artifacts/oscar_vllm_optimization/20260721/prefill_full_fused_production_green/monitor_20260721T054333Z/`；
pytest 日志 SHA256 为 `3b98a1e0...6f92`。下一步构建包含当前 production 的独立镜像，
再使用第 10.111/10.112 节冻结 payload 依次复测 profiler 和 TTFT M2；在真实服务数据前
不宣称 M2 已通过。

### 10.124 Full Mixed-Prefill 独立镜像冻结

在不透传 GPU 的情况下，以当前 `vllm/docker/Dockerfile.oscar-vllm` 和 vLLM 子仓库为
build context 构建独立镜像 `oscar-vllm:full-fused-prefill-20260721`。Docker build
exit0，最终镜像 ID 为：

`sha256:5babdcc22d5b27d8646205f2c781d8ac92357c5eca1d337cad4f2c0c14702c3a`

无 GPU 容器内复核的 Python/包版本仍为 Python 3.12.13、vLLM 0.25.0、PyTorch
2.11.0+cu130/CUDA 13.0、Triton 3.6.0、Transformers 5.13.0。镜像内三份关键文件
SHA256 与第 10.122 节工作区冻结值逐项一致：backend `121591df...ec5`、prefill
kernel/wrapper `e1c764e6...4715`、production oracle `e2a0ab30...c399`。

首次镜像校验的编排错误需要如实保留：build 命令从 `vllm/` workdir 展开 `$PWD`，将
build artifact 写进子仓库 `vllm/artifacts`；随后从 workspace 根运行的校验读取规范
`artifacts/` 路径，导致 `tee` 和退出码文件无法落盘。镜像构建本身已完成，且该次 stdout
已显示三份 hash 匹配。没有重复构建镜像；将原 artifact 原样复制到根目录并逐文件
SHA256 比对一致后，重新运行无 GPU 镜像校验，最终 exit0。

规范 artifact 位于
`artifacts/oscar_vllm_optimization/20260721/full_fused_prefill_image/`；build log 和
最终 image-validation log SHA256 分别为 `475f923a...2396`、`584f0afe...45aa`。
本阶段没有 GPU 实验。下一步使用该镜像和冻结 payload 做真实 production profiler；
由于外部 TP8 作业再次占满全卡，必须持续监控，重新获得同一卡连续两次空闲后才能启动。

### 10.125 Full Mixed-Prefill 真实服务 Operator Profiler

本轮严格复用第 10.111 节的 runner、vector/matrix/tensor payload 和调用顺序；四份
SHA256 仍为 `aa4893ec...256d`、`0bfa9352...7ecd`、`0c33cf27...2cf3`、
`3e762929...b92f`。唯一有意变化是使用第 10.124 节 full-fused 镜像。持续 monitor
前五轮均为全卡占用；UTC `06:01:41`/`06:03:01` 连续确认 GPU 0--7 为 0 MiB、
0%、无 compute process 后固定物理 GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`。

实验环境为镜像 `oscar-vllm:full-fused-prefill-20260721`、image ID
`sha256:5babdcc2...02c3a`、NVIDIA B200 183359 MiB、Driver 595.71.05、宿主 CUDA
13.2、Python 3.12.13、vLLM 0.25.0、PyTorch 2.11.0+cu130/CUDA 13.0、Triton
3.6.0、Transformers 5.13.0。服务于 UTC `06:04:33` 启动、`06:06:50` 健康就绪；
planner 仍实际报告 history capacity 513264 tokens，36 层 V absorption 成功，未启用
full-history materialize。

Health 与 workload 均运行在不透传 GPU、共享服务 network namespace 的独立容器中。
UTC `06:07:04`--`06:07:08` 五个请求全部 HTTP 200、prompt 2100 tokens，cached
序列严格为 `0/1840/0/1840/0`；workload validation 与 runner 均 exit0。服务期间
compute process 只有物理 GPU 0 上本容器的 `VLLM::EngineCore`，没有外部任务介入。

两份 trace 使用同口径 analyzer 与第 10.111 节 BF16-metadata trace 对比：

| 指标 | BF16 metadata | Full fused | 变化 |
| --- | ---: | ---: | ---: |
| Cold Worker CPU | 34.555459 ms | 35.021850 ms | +0.466391 ms |
| Hit Worker CPU | 47.708680 ms | 42.362153 ms | -5.346527 ms |
| Worker hit-cold gap | +13.153221 ms | +7.340303 ms | -5.812918 ms，-44.193875% |
| Cold attention CPU | 17.290234 ms | 17.961498 ms | +0.671264 ms |
| Hit attention CPU | 27.878687 ms | 22.433601 ms | -5.445086 ms |
| Attention hit-cold gap | +10.588453 ms | +4.472103 ms | -6.116350 ms，-57.764340% |
| Hit GPU kernel+memcpy | 15.098397 ms | 15.354787 ms | +0.256390 ms，+1.698127% |
| GPU hit-cold gap | -3.820392 ms | -3.559399 ms | +0.260993 ms |
| Hit GPU kernel 数 | 786 | 714 | -72 |
| Hit CUDA runtime launch 数 | 377 | 305 | -72 |

Full-fused cached kernel 为 36 次、合计 9.816432 ms，高于旧 cached-only kernel 的
9.039926 ms；这是因为新 kernel 同时执行 current causal attention，不能把两者当相同
工作量直接声称退化。总 hit GPU 时间只增加 0.256390 ms，同时 suffix FA 与 merge 被
移除：旧 trace 中 36 次 `_C::merge_attn_states` CPU op（0.557659 ms）和 36 次 merge
GPU kernel（0.108255 ms）在新 trace 中均为 0，hit GPU kernel 与 runtime launch 均减少
72 次。Cold GPU 时间只变化 -0.004603 ms，说明两轮设备侧基线可比；CPU cold 的
0.466--0.671 ms 差异作为单 trace 噪声保留。

Analyzer 首轮 pycompile/lint 通过，format check 要求机械排版；使用同镜像 formatter 后
pycompile、Ruff lint/format 全部 exit0，同一 trace 解析成功。初始错误扫描又因宽泛 `oom`
匹配配置提示 `If OOM'ed`；改用 traceback/CUDA error/RuntimeError/真实 OOM 严格模式后
为空。服务于 UTC `06:07:30` exit0、`OOMKilled=false`，容器删除后 GPU 0--7 全部
0 MiB、0%，无 compute process。

完整 monitor、环境、payload、响应、两份 trace、analyzer/summary、container/image
inspect、日志与回收状态位于
`artifacts/oscar_vllm_optimization/20260721/torch_profile_full_fused_prefill/`；analyzer、
summary、cold/hit trace SHA256 分别为 `cdb5fcfe...5e4e7`、`68d69f62...279a`、
`04b2895d...0a4b`、`b4af60dc...90b7`。Operator trace 已证明 host 路径明显缩短，但仍
不是客户端 TTFT 统计；下一步必须用第 10.112 节同 token-ID client 做配对 TTFT M2。

### 10.126 Full Mixed-Prefill Prefix-Hit TTFT M2 复测

本轮原样复用第 10.112 节 token-ID client，SHA256 仍为
`673c94eadb604071126a6951c01fd3c37cce2f8b7a2dc518de9522d9ab364f32`；服务参数
与第 10.125 节相同但关闭 profiler，端口保持冻结 client 使用的 8441。持续 monitor
前 15 轮均无空闲卡；UTC `06:29:32`/`06:30:52` 连续确认 GPU 0--7 为 0 MiB、
0%、无 compute process 后固定物理 GPU 0，容器内 `CUDA_VISIBLE_DEVICES=0`。

镜像、image ID、NVIDIA B200、driver、CUDA、Python 和包版本同第 10.125 节。服务于
UTC `06:32:11` 启动、`06:34:28` 健康就绪；日志再次确认 history capacity 513264
tokens、36 层 V absorption、OSCAR mixed attention 和三段式 store 均实际启用。Health
与 benchmark 使用不透传 GPU、共享服务 network namespace 的独立容器。

Client 先执行 2 对 warm-up；前 5 对 cold/hit CV 为 4.458061%/3.603938%，所以按冻结
规则扩展至最多 15 对。最终 30 个正式请求全部 HTTP 200、prompt 2100 tokens，所有
cold cached=0、所有 hit cached=1840；benchmark exit0。结果如下：

| 指标 | BF16 metadata（第 10.112 节） | Full fused（15 对） | 变化 |
| --- | ---: | ---: | ---: |
| Cold TTFT 均值 | 33.649515 ms | 33.111420 ms | -0.538095 ms |
| Hit TTFT 均值 | 41.942152 ms | 33.073252 ms | -8.868900 ms |
| Cold / hit CV | 4.191% / 3.383% | 3.013% / 2.464% | cold 仍略高于 3% |
| Hit-cold 配对均值 | +8.292637 ms | -0.038168 ms | -8.330805 ms |

15 对中 9 对 hit 更快、6 对 cold 更快。按 `hit TTFT - cold TTFT` 定义，配对差样本
标准差 0.528563 ms、标准误 0.136474 ms，双侧 95% t 区间为
`[-0.330877, +0.254540] ms`，包含 0。最终 mean improvement 为 0.115273%，但该量级
不能排除测量噪声；且 cold CV 在最大样本数时仍为 3.013374%。因此本轮证明 prefix hit
已从稳定回退推进到**统计持平**，但没有证明“TTFT 有统计改善”，故 **M2 仍不能标记
通过**。不能用均值方向轻微为负或第 10.125 节 operator 收益替代该结论。

功能门槛继续满足：1840/2100=87.619% prompt tokens 命中，超过 75%，实际 prefill
tokens 减少 87.619%，超过 70%。运行期间 compute process 只有 GPU 0 上本服务的
`VLLM::EngineCore`，无外部 TP8 任务介入；服务停止前严格错误扫描为空，容器 exit0、
`OOMKilled=false` 并已删除。回收快照 GPU 0--7 均为 4 MiB、0%、无 compute process。

连续 17 轮 monitor、完整 `nvidia-smi`、30 个请求、summary、paired stats、服务/客户
端 inspect、日志与回收状态位于
`artifacts/oscar_vllm_optimization/20260721/prefill_full_fused_ttft/`；summary 与 paired
stats SHA256 分别为 `df267ff8...7658`、`2d187a6d...d53f`。下一步只读取第 10.125 节
full-fused trace，分解剩余 4.472103 ms attention CPU gap并形成新的独立候选；不重复
同一 TTFT，也不因系统 goal 已被过早标成 achieved 而停止未完成的验收。

### 10.127 Full-Fused Cold/Hit CPU 路径细分

本阶段没有重新分配 GPU或运行服务，只读取第 10.125 节同一有效 cold/hit trace。复用
第 10.113 节已验证的同线程嵌套算法，将 36 个
`vllm::unified_attention_with_output` parent 拆为 direct children 与未分类 self time。

结果如下：

| 指标 | Cold | Hit | Hit-cold |
| --- | ---: | ---: | ---: |
| Worker span | 35.021850 ms | 42.362153 ms | +7.340303 ms |
| 36 层 attention wall | 17.961498 ms | 22.433601 ms | +4.472103 ms |
| Worker 中 attention 外时间 | 17.060352 ms | 19.928552 ms | +2.868200 ms |
| Attention self time | 10.699894 ms | 13.690002 ms | +2.990108 ms |
| Attention direct children | 7.261604 ms | 8.743599 ms | +1.481995 ms |

剩余 attention gap 中 self 占 66.861340%，direct children 占 33.138660%。Direct 增量的
主要项为：36 次额外 `aten::matmul` 0.796828 ms、72 次 `aten::to` 0.544495 ms、
72 次 `cuLaunchKernelEx` driver 0.409531 ms；同时少 36 次 `cudaLaunchKernelExC`
runtime 0.353402 ms，其他 direct 项均小于 0.1 ms。Worker 中 attention 外仍另有
2.868200 ms，约占 Worker gap 的 39.074681%，因此不能假设只优化 rotation 就会线性
转化为全部 TTFT 收益。

Cold/hit 的 direct `aten::matmul` 数为 72/108，每层只增加 1 次，而不是 2 次：current
K rotation 是两侧 cache store 都需要的共同工作，V inverse 也是共同工作；prefix-hit
路径唯一新增的是 cached attention 的 Q rotation。当前该 rotation 使用 FP32 `Rk`，而
Triton kernel 加载 Q 后立即转为 BF16；wrapper 还固定把输入提升为 FP32。由此形成下一
个可单独验证的候选：仅在 absorbed full-fused 分支使用已缓存的 BF16 `Rk_fast` 计算 Q，
并保持 BF16 Q 输入进入 kernel。Materialize 已使用同一 BF16 rotation，但 production
full-fused 尚未使用；数值与性能均需 artifact A/B，不能据 trace 提前认定安全或有效。

Analyzer 首轮 pycompile 与 format check 通过，但 lint 报 import 分组 I001；由于容器内
脚本遗漏 `set -e`，最后的 format check 又返回 0，包装错误地继续生成首份 JSON。该份
不作为正式门禁。删除唯一多余空行并在容器内加入 `set -e` 后，pycompile、Ruff lint/
format 全部 exit0，再从同一 trace 重算得到上述结果。最终 analyzer/JSON SHA256 为
`68526cc3...e5c`、`21c4ae4d...9035`，artifact 位于
`artifacts/oscar_vllm_optimization/20260721/full_fused_cpu_path_analysis/`。下一步先做
FP32-Q 与 BF16-Q 的 production-shape correctness、GPU 与 host A/B；未过门槛前不修改
production，也不重复 TTFT。

### 10.128 BF16-Q Rotation Artifact 与静态门禁

已新增 artifact-only A/B，production 源码未修改。候选从当前 production prefill 文件
机械复制，只改一项 wrapper 行为：当且仅当同时提供 current K/V、即 absorbed full-fused
ABI 时，保持传入 Q 的 dtype；原 cached-only/non-absorbed 调用仍执行
`contiguous().float()`。Benchmark baseline 使用 FP32 `Rk` 计算 Q 并调用当前 production
wrapper；candidate 使用同一矩阵的 BF16 `Rk_fast` 计算 Q，并以 BF16 输入调用候选
wrapper。current K 仍以 FP32 rotation 预先计算，K/V cache、三段式地址、current V、
32x32/GQA2 tile 和 online softmax 均相同。

计时范围不是单独 GEMM，而是每层完整的“Q rotation + full-fused attention + BF16 V
inverse”；这样会包含 wrapper conversion、输出/LSE allocation 和 kernel launch，避免
把局部 matmul 收益冒充 backend 收益。Correctness 同时覆盖 `shared_hit=1840` full hit
与 `shared_hit=0` recent-tail ownership，并比较最终 inverse 后 output 与 LSE。性能规则
继续使用 20 次预热、五批每批 100 次 CUDA event，以及十批每批 20 次 host enqueue。

首轮静态检查 pycompile 通过，Ruff lint 报 benchmark import 分组 I001 和候选 89 字符
E501；按精确 diff 修正后，第二轮 pycompile/lint 通过而 format check 要求机械排版
benchmark。使用同镜像 formatter 后，第三轮两个文件的 pycompile、Ruff lint/format
全部 exit0。Benchmark/candidate SHA256 为 `5776bde0...ce0`、`8cc8fa20...a5c`，完整
脚本和三轮静态日志位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_q_microbench/`。

本阶段没有分配 GPU，尚无 BF16-Q 数值或性能结论。下一步从零持续监控 GPU；取得同一卡
连续两次空闲后固定运行该 SHA。若任一 ownership 场景超出既有误差门槛，或 GPU/host
任一侧没有稳定收益，则候选不得进入 production。

### 10.129 BF16-Q 首轮 CUDA A/B 无效性能记录

本轮先修正了 monitor 的空闲判定。首个 monitor 把显存必须等于 0 MiB 作为条件，因而
错误拒绝了 4 MiB、利用率 0%、无 compute process 的驱动 bookkeeping 状态；该 monitor
在三轮后主动中止，三轮结果没有复用于后续计数。新 monitor 从零开始，空闲定义为
显存不超过 8 MiB、利用率 0%，且该 GPU UUID 没有 compute process。它没有因全卡忙或
多次只出现单轮空闲而退出，完整运行 97 轮。UTC `08:40:56` 的第 96 轮显示 GPU 1--7
空闲，`08:42:03` 的第 97 轮显示 GPU 0、2、4--7 空闲，因此 GPU 2 满足连续两次空闲
并被固定；先前所有未通过复核的单轮样本均已清零。

实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:full-fused-prefill-20260721` |
| 镜像 ID | `sha256:5babdcc22d5b27d8646205f2c781d8ac92357c5eca1d337cad4f2c0c14702c3a` |
| 固定物理 GPU | GPU 2，NVIDIA B200，183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| 容器内 `CUDA_VISIBLE_DEVICES` | `0`（Docker device request 固定物理 GPU 2） |

但是，本轮不能形成有效性能结论。外部 8 卡 Python worker 于 UTC `08:41:46` 启动；
第 97 轮 monitor 在 `08:42:03` 采样时它尚未在 GPU 2 分配可见显存，而本容器直到
`08:42:32` 才启动。启动前即时 `nvidia-smi` 已显示 GPU 2 使用 124657 MiB、利用率
10%，PID 2050892 正在运行；这明确证明正式测量从一开始就与外部任务并发。结束快照
仍显示同一 PID 使用约 124657 MiB、利用率 14%。因此，虽然分配动作依据当时两次有效
采样合法，但启动前状态已变化，本轮 GPU 与 host 性能值均作废，不能用于 production
合入、M2 判断或与无竞争实验比较。

为保证审计完整，保留本轮实际输出但明确标注为非权威数据。两个 correctness 场景在
并发条件下仍通过：full-prefix-hit 的 output/LSE 最大绝对差为
`0.000988007/0.001840591`，recent-tail 为 `0.001464844/0.001819611`。五批 CUDA event
均值为 FP32-Q `0.269073 ms`、BF16-Q `0.255881 ms`，表面速度比 `1.051558x`；十批
host enqueue 均值为 `85.070185/70.232385 us`，表面速度比 `1.211267x`。这些数值只能
说明脚本在竞争环境中完成，不能证明候选有稳定收益；正确性也将在有效重跑中复核。

容器 UTC `08:42:32`--`08:42:50` 运行，exit0、`OOMKilled=false`，严格 traceback/
CUDA error/RuntimeError/真实 OOM 扫描为空，容器已删除。本轮完整 97 轮 monitor、前后
完整 `nvidia-smi`、外部进程证据、结果、日志、container/image inspect 与 SHA256 位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_q_microbench/`；结果 JSON SHA256
为 `de0139d3...ac4`。下一步必须重新从零持续监控；再次取得同一卡连续两次空闲后，
还要在 Docker 启动前做即时占用复核，若状态变化则放弃该分配并继续监控。

### 10.130 BF16-Q 第二轮无竞争 CUDA A/B

第 10.129 节无效轮完成后，新 monitor 从零开始持续运行。它没有因周期性单轮空闲或
全卡忙而停止，共运行 202 轮；UTC `12:39:32` 与 `12:40:47` 连续确认 GPU 0--7
均为空闲，随后固定物理 GPU 0。为封闭上轮的启动竞态，Docker 启动前在同一编排命令中
增加即时门禁；UTC `12:41:12` 实测 GPU 0 为 0 MiB、0%，对应 UUID 无 compute
process，门禁通过后才启动容器。启动前和结束后的完整 `nvidia-smi` 均显示全部 GPU
空闲，正式运行期间没有外部任务证据，因此本轮可作为有效 A/B。

实验环境如下：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:full-fused-prefill-20260721` |
| 镜像 ID | `sha256:5babdcc22d5b27d8646205f2c781d8ac92357c5eca1d337cad4f2c0c14702c3a` |
| 固定物理 GPU | GPU 0，NVIDIA B200，183359 MiB |
| Driver / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| Python | 3.12.13 |
| vLLM / PyTorch / PyTorch CUDA | 0.25.0 / 2.11.0+cu130 / 13.0 |
| Triton / Transformers | 3.6.0 / 5.13.0 |
| 容器内 `CUDA_VISIBLE_DEVICES` | `0`（Docker device request 固定物理 GPU 0） |

两个 ownership correctness 场景均通过。Full-prefix-hit 的 output/LSE 最大绝对差为
`0.000988007/0.001840591`，recent-tail 为 `0.001464844/0.001819611`，均显著低于
既有 `atol=rtol=0.03` 门槛。有效性能结果如下：

| 指标 | FP32-Q baseline | BF16-Q candidate | Candidate 变化 |
| --- | ---: | ---: | ---: |
| GPU CUDA event mean | 0.272887 ms | 0.259118 ms | -0.013769 ms，1.053140x |
| GPU batch CV | 3.096420% | 3.104872% | 两侧均略高于 3% |
| Host enqueue mean | 78.210660 us | 64.875215 us | -13.335445 us，1.205555x |
| Host batch CV | 2.613781% | 1.961341% | 两侧低于 3% |

GPU 五个配对批次中 candidate 为 4/5 更快；`FP32-Q - BF16-Q` 配对差均值
0.013769 ms、样本标准差 0.014629 ms，双侧 95% t 区间为
`[-0.004394, 0.031933] ms`，包含 0。两侧各有一个约 0.02 ms 的离群批次，使各自 CV
略超 3%。Host 十个配对批次则为 10/10 更快，配对均值 13.335445 us，95% t 区间
`[11.892840, 14.778050] us`，完全为正。因此本轮证明 correctness 和 host enqueue
收益，但单轮 GPU 样本尚未排除零收益；不能仅用 1.053140x 均值直接进入 production。

容器 UTC `12:41:32`--`12:42:46` 运行，exit0、`OOMKilled=false`，严格 traceback/
CUDA error/RuntimeError/真实 OOM 扫描为空并已删除。结果、配对统计、即时门禁、完整
前后 `nvidia-smi`、container inspect 和日志位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_q_microbench/run_20260721T124047Z/`；
结果与配对统计 SHA256 分别为 `7edd1919...64d`、`eb207b83...8c7`。下一步重新从零
执行 GPU 分配并复测；若有效复测重现 correctness、host 全批收益和约 0.013 ms GPU
收益，再以跨轮一致性决定是否建立 production red/green，不提前修改 production。

### 10.131 BF16-Q 第三轮复现与跨轮判定

第 10.130 节后重新从零启动 monitor。UTC `12:46:20`/`12:47:27` 两轮均确认
GPU 0--7 为空闲，固定物理 GPU 0；随后即时门禁再次确认 GPU 0 为 0 MiB、0%、无
compute process。使用完全相同的镜像、artifact SHA、shape、预热、交替顺序和计时范围
完成第三轮。实验环境与第 10.130 节相同，容器内 `CUDA_VISIBLE_DEVICES=0`。

Correctness 精确复现上一轮：full-prefix-hit output/LSE 最大绝对差仍为
`0.000988007/0.001840591`，recent-tail 仍为 `0.001464844/0.001819611`。性能结果为：

| 指标 | FP32-Q baseline | BF16-Q candidate | Candidate 变化 |
| --- | ---: | ---: | ---: |
| GPU CUDA event mean | 0.272767 ms | 0.258788 ms | -0.013979 ms，1.054018x |
| GPU batch CV | 3.112683% | 2.903046% | baseline 略高于 3% |
| Host enqueue mean | 78.332295 us | 64.334295 us | -13.998000 us，1.217582x |
| Host batch CV | 1.508824% | 0.730723% | 两侧低于 3% |

两轮有效运行的均值与离群形态高度一致：上一轮/本轮 GPU 节省为
0.013769/0.013979 ms，host 节省为 13.335445/13.998000 us。合并十个 GPU 配对批次后，
candidate 为 8/10 更快，FP32-Q/BF16-Q 总均值为 0.272827/0.258953 ms，速度比
1.053578x；配对差均值 0.013874 ms、样本标准差 0.013661 ms，双侧 95% t 区间
`[0.004102, 0.023647] ms`，已完全为正。合并二十个 host 批次为 20/20 更快，均值
78.271478/64.604755 us、速度比 1.211544x，配对差 95% t 区间
`[12.837793, 14.495652] us`。

容器 UTC `12:47:59`--`12:49:03` 运行，前后 GPU 0 均为 0 MiB、0%、无 compute
process，exit0、`OOMKilled=false`，严格错误扫描为空并已删除。本轮结果与跨轮统计位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_q_microbench/run_20260721T124727Z/`，
SHA256 分别为 `858a8120...e2e`、`707e6de8...b05`。

因此 BF16-Q 候选已满足 artifact 阶段的三项条件：两种 ownership correctness 重复通过、
GPU 跨轮配对区间为正、host 全部配对批次为正。下一步进入 production red/green：先在
现有 absorbed oracle 中新增“full-fused Q 必须保持 BF16”的精确失败断言，再做最小
backend/wrapper 改动；non-absorbed、cached-only 与 materialize 路径必须保持原 dtype
行为。Production CUDA green 和后续真实 TTFT 未完成前，M2 状态仍不改变。

### 10.132 BF16-Q Production Red Oracle 与静态门禁

在第 10.120 节既有两参数 production oracle 内新增最小 dtype 合约，production 源码
仍未修改。测试通过 module-level wrapper 观察 backend 传给
`oscar_cached_prefill_attention` 的 Q：`absorb_v_rotation=True` 的 full-fused ABI
必须保持与原 query 相同的 BF16；`False` 的 cached-only/non-absorbed ABI 必须继续为
FP32。该断言与既有“absorbed 不得调用 suffix FlashAttention”断言同时存在，因此能
区分目标 dtype 路径而不弱化控制流或数值 reference。

使用 full-fused 独立镜像、不透传 GPU、只读挂载完整仓库执行静态门禁。首轮
pycompile 通过，Ruff 仅报新增局部 import 排序 I001；按 Ruff 精确 diff 调整后，第二轮
pycompile、Ruff lint/format 全部 exit0。最终测试 SHA256 为
`a4bff199...c676`，静态日志位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_q_production_red/`。

本阶段没有分配 GPU，尚未形成 runtime red。下一步从零检查 GPU；取得同一卡连续两次
空闲并通过即时门禁后，仅运行该测试的两个参数。有效 red 必须是 non-absorbed 通过、
absorbed 唯一失败于 `q_rot.dtype == query.dtype`，其他失败不算目标证据。

### 10.133 BF16-Q Production CUDA Red

UTC `12:54:44`/`12:56:02` 连续确认 GPU 0--7 均为空闲，随后固定物理 GPU 0；
启动前即时门禁再次得到 0 MiB、0%、无 compute process。实验使用镜像
`oscar-vllm:full-fused-prefill-20260721`、image ID `sha256:5babdcc2...02c3a`、
NVIDIA B200 183359 MiB、Driver 595.71.05、宿主 CUDA 13.2、Python 3.12.13、
vLLM 0.25.0、PyTorch 2.11.0+cu130/CUDA 13.0、Triton 3.6.0、Transformers 5.13.0；
容器内 `CUDA_VISIBLE_DEVICES=0`。

旧 production 镜像只读挂载第 10.132 节测试，定向运行两个参数，实际结果为：

`1 failed, 1 passed, 16 deselected, 120 warnings in 72.54s`

`absorb_v_rotation=False` 参数通过，证明新增 oracle 没有改变 non-absorbed FP32 Q 合约。
唯一失败是 `True` 参数在 backend 调用 cached-prefill wrapper 时得到
`q_rot.dtype=torch.float32`，而 oracle 期望原 query 的 `torch.bfloat16`；失败栈精确
落在 `assert q_rot.dtype == expected_dtype`。既有 absorbed suffix-FA 禁止断言未触发，
没有 import、CUDA、reference 或其他控制流失败，因此构成有效 production red。

容器 UTC `12:56:54`--`12:58:13` 运行，预期 exit1、`OOMKilled=false`，前后 GPU 0
均为 0 MiB、0%、无 compute process，容器已删除。Warnings 均为既有 CUTLASS/SWIG
deprecation warning。完整 monitor、即时门禁、pytest、container inspect、前后
`nvidia-smi` 与回收状态位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_q_production_red/`；pytest
SHA256 为 `de86c707...75a`。下一步只做两处最小 production 改动：absorbed full-fused
backend 使用 BF16 `Rk_fast` 计算 Q，wrapper 只在 current K/V 同时存在时保留 Q dtype；
其他路径继续使用 FP32。

### 10.134 BF16-Q Production 合入与静态门禁

Production 改动严格限定为两处。Backend 先判断 V rotation 是否 absorption：absorbed
full-fused 分支优先复用 dtype 与 query 相同的 `_oscar_Rk_fast`，缺失或 dtype 不匹配时
才从 FP32 `Rk` 转换，并用原 BF16 query 做 matmul；non-absorbed 分支继续执行
`query.float() @ FP32 Rk`。Cached-prefill wrapper 仅在 current K/V 同时存在时保留
传入 Q dtype，cached-only 调用仍强制 `contiguous().float()`。Materialize 使用独立
wrapper，未改动；current K rotation、V absorption/inverse、三段式读写与 allocator 也均
未修改。

使用 full-fused 独立镜像、不透传 GPU、只读挂载完整仓库，对 backend、prefill 与测试
三文件执行 pycompile、Ruff lint/format；首轮全部 exit0，三个文件均已符合 formatter，
目标 `git diff --check` 也无输出。最终 SHA256 为：

- backend：`87db7318...e7a6`；
- prefill kernel/wrapper：`8cc8fa20...a5c`；
- production oracle：`a4bff199...c676`。

Prefill SHA 与第 10.128--10.131 节实际测量的候选副本完全一致。本阶段没有分配 GPU，
尚不能宣称 production CUDA green。下一步重新执行连续两次空闲和即时门禁，再把当前
backend/prefill/test 覆盖到镜像 installed package，运行同一两参数 oracle；成功标准为
`2 passed`，并同时保留 non-absorbed FP32 dtype、absorbed BF16 dtype、禁止 suffix FA
与两条 FP32 reference 数值断言。

### 10.135 BF16-Q Production CUDA Green

UTC `13:02:27`/`13:03:48` 连续确认 GPU 0--7 均为空闲，固定物理 GPU 0；即时门禁
再次确认 0 MiB、0%、无 compute process。实验环境与第 10.133 节相同，容器内
`CUDA_VISIBLE_DEVICES=0`；唯一有意变化是把第 10.134 节当前 backend/prefill 和同一
测试只读覆盖到镜像 installed package。

两个参数的实际结果为：

`2 passed, 16 deselected, 120 warnings in 57.71s`

这一次通过的断言同时包括：non-absorbed cached-only Q 仍为 FP32；absorbed full-fused
Q 为 BF16；absorbed 分支不调用 suffix FlashAttention；两条路径均与既有 FP32 mixed
attention reference 在 `atol=rtol=0.015` 内一致；真实非连续 fused-QKV V stride、一个
384-token cached request、一个 cold request、17/13 current tokens 和三 pool ownership
仍全部覆盖。Warnings 均为既有 CUTLASS/SWIG deprecation warning，严格 traceback/
CUDA error/RuntimeError/assertion/OOM 扫描为空。

容器 UTC `13:04:36`--`13:05:38` 运行，exit0、`OOMKilled=false`；启动前与结束后
GPU 0 均为 0 MiB、0%、无 compute process，容器已删除。完整 monitor、即时门禁、
pytest、三份只读 mount、container inspect、前后 `nvidia-smi` 与回收状态位于
`artifacts/oscar_vllm_optimization/20260721/prefill_bf16_q_production_green/`；pytest
SHA256 为 `da1f853a...0e4`。

Production correctness 已闭环，但真实服务仍使用旧 full-fused 镜像。下一步构建包含
BF16-Q production 的新独立镜像并校验三份源码 SHA；随后复用第 10.125/10.126 节冻结
profiler 与 TTFT workload。只有无竞争真实 TTFT 显示统计改善，M2 才能改变状态。

### 10.136 BF16-Q 独立镜像冻结

在不透传 GPU 的情况下，以当前 vLLM 子仓库和 `docker/Dockerfile.oscar-vllm` 构建
独立镜像 `oscar-vllm:bf16-q-prefill-20260721`。Docker build exit0，最终 image ID 为：

`sha256:44ac3500c64c5857fd89502ea0404a5566ddfa7f9b77e26446829cd3b18e9d7a`

无 GPU 容器内采集的软件环境为 Python 3.12.13、vLLM 0.25.0、PyTorch
2.11.0+cu130/CUDA 13.0、Triton 3.6.0、Transformers 5.13.0。镜像内三份关键源码
SHA256 与第 10.134--10.135 节工作区值逐项一致：backend `87db7318...e7a6`、prefill
`8cc8fa20...a5c`、oracle `a4bff199...c676`。构建全部命中基础依赖缓存，只重新复制
当前 vLLM/tests 并导出新 image layers，没有覆盖旧 full-fused 镜像。

Build log、image inspect、镜像内外哈希和版本位于
`artifacts/oscar_vllm_optimization/20260721/bf16_q_prefill_image/`。本阶段没有运行 GPU
实验。下一步先复用第 10.125 节冻结真实服务 profiler workload，确认 production 服务
实际启用 BF16-Q 且剩余 attention/Worker gap 下降；随后再做第 10.126 节配对 TTFT。
