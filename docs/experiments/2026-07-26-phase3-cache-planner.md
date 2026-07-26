# 阶段 3 实验报告：三池 CacheSpec 与 CPU allocator

## 1. 结论

阶段 3 通过设计文档规定的出口条件：

- 单请求、多请求的 prefix/recent/history ownership 均有确定的物理归属；
- finish、abort、preemption 与 reuse 的容量守恒测试全部通过；
- history OOM 会原子回滚，不泄漏 page、slot、logical length 或 cache version；
- scheduler metadata、worker ownership 与六个无重叠 tensor view 已接入正式源码；
- 三池 tensor 总大小与联合 capacity planner 逐字节一致；
- 正式代码线定向回归为 116/116 passed。

本阶段是 CPU 规划与生命周期验收，不把理论容量比冒充 GPU 实测容量。

## 2. 固定源码与模型几何

| 项目 | 实际值 |
| --- | --- |
| 正式源码分支 | `feat/glm52-mla-cache-planner` |
| 正式源码 commit | `e75a40a294bd3127667f34ebffce8119a8ac0f3a` |
| 层数 | 78 |
| MLA head size | 576 |
| 共享 latent / RoPE | 512 / 64 |
| INT2 group size | 128 |
| block size | 16 |
| prefix / recent | 64 / 256 tokens |
| `indexer_types` | shared 57 层、full 21 层 |

源码 commit 已在远端发布，正式 submodule 与远端指向同一 commit，两个工作区均无 tracked 修改。

## 3. 三池布局与计费

每层每 token 的共享 latent 原生 BF16 成本为 `512 × 2 = 1,024` bytes。INT2 history 包含 128-byte packed data，以及 4 组 FP32 scale/zero 共 32 bytes，合计 160 bytes；因此仅 history latent 的理论压缩率和 page padding 后压缩率均为 6.4×。

64 维 RoPE 始终保持 BF16。21 个 full indexer 的原生 DSA cache 为每层每 token 132 bytes，其中 data 128 bytes、scale 4 bytes。标准 vLLM block table 继续管理 RoPE 与 DSA，并为 null block 保留 block 0；INT2 history 使用独立 page namespace。

### 3.1 14GiB 联合容量计划

固定 `max_num_seqs=16` 的实际 CPU planner 结果如下：

| 指标 | 实际值 |
| --- | ---: |
| 总预算 | 15,032,385,536 bytes |
| 标准 blocks | 36,216 |
| 扣除 null block 后 usable blocks | 36,215 |
| logical token slots | 579,440 |
| 独立 history pages | 36,216 |
| fixed BF16 prefix/recent | 408,944,640 bytes |
| INT2 history | 7,231,610,880 bytes |
| BF16 RoPE | 5,785,288,704 bytes |
| native auxiliary DSA | 1,606,252,032 bytes |
| 总分配 | 15,032,096,256 bytes |
| 未使用 | 289,280 bytes |

`总分配 + 未使用 = 总预算` 精确成立。同一预算下的理论 native 计划为 10,142 blocks、162,256 token slots，联合计划的理论容量比为 `3.5711468297×`。

## 4. Allocator 与生命周期

固定边界测试覆盖长度 0、63、64、65、319、320、321 和 32,768 tokens。allocator 为每个请求保存稳定 `hp_row`、prefix/recent 起点、history page IDs、generation、cache version 与 logical length。

实际验证包括：

- 320→337 tokens 时，新增 17 个 history tokens，物理地址跨完整 page 与 partial page；
- recent 使用环形地址，history 使用 logical position 到 page/slot 映射；
- 单页预算下从 336→337 tokens 的新 page 分配失败后，metadata、logical length、page ownership 与 version 均保持不变；
- finish、abort、preemption 均释放 history pages 与 BF16 row；
- reuse 可复用物理 row，但生成新的 generation，旧 worker metadata 会被拒绝。

## 5. Scheduler 与 worker 接入

`OscarMLAAttentionSpec` 进入 KV cache config；scheduler 通过 request-keyed `WorkerCacheMetadata` 下发 generation/version、稳定 BF16 row 与 history page IDs。worker 拒绝 generation 跳变和倒退 version，并在 finished/preempted 后释放 mirror。

每层 raw allocation 被切分为以下六个连续且无重叠的 view：

1. packed INT2 history data；
2. FP32 history scale；
3. FP32 history zero；
4. BF16 prefix；
5. BF16 recent；
6. BF16 RoPE。

测试逐一比较 `data_ptr`、shape、storage 大小和 planner 计费，没有隐式 padding 或完整 BF16 history。

## 6. 正式回归结果

正式 submodule 上的定向命令覆盖 `tests/oscar_mla`、`test_kv_cache_utils.py` 和 `test_single_type_kv_cache_manager.py`，结果为：

| 验证 | 实际结果 |
| --- | --- |
| 定向 pytest | 116 passed、0 failed，27.11 秒 |
| 完整 scheduler 强制离线回归 | 68 passed、28 failed，31.22 秒 |
| 13 个无既存 lint 债务的改动文件 | ruff 0.14.0 与 format check 通过 |
| 14 个改动 Python 文件 | compileall 通过 |
| Git whitespace | `git diff --check` 通过 |

完整 scheduler 的 28 项失败均在构造 `llava-hf/llava-1.5-7b-hf` 配置时发生，因为强制离线环境没有该仓库配置；没有 OSCAR 或通用 scheduler 断言失败。没有为消除这些离线失败下载模型。

正式首轮定向回归曾得到 104 passed、12 failed；12 项都因源码树 `.venv` 不含已安装 vLLM package metadata，自动 device detection 失败。加入项目内候选 rootfs 的已安装 vLLM metadata/dependency 路径后，这 12 项先独立得到 12/12 passed，再完成上述 116/116 全量结果，且 `torch.cuda.is_initialized()` 保持 false。

`vllm/v1/worker/gpu_model_runner.py` 在阶段 3 基线和当前版本均有相同的 6 个既存 ruff/format 问题；本阶段仅加入 import、ownership 调用与 cache reshape 分支，没有修改既存问题行，也没有为通过检查而重排整个大文件。

## 7. 阶段出口

阶段 3 已完成。代码和本报告进入 Git；本阶段没有生成需要上传的大型模型、cache 或日志产物。下一阶段必须在全新任务专用 Triton cache 上完成 SM80 cold compile 和 A800 实际 launch，CPU interpreter 结果不能替代该门禁。
