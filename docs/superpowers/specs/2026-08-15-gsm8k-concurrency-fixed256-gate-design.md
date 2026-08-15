# GSM8K 并发与 fixed256 Gate 调整设计

## 1. 目标

修改 `glm52_oscar_vllm_optimize_task_v2.md`，将 fixed256 和完整 1,319 题 GSM8K 正式评测的服务端、客户端并发统一为 64，并将 fixed256 相对初始 `oscar_baseline` 的最大正确题数回退从 10 收紧到 5。完整 1,319 题的最大回退继续保持 10。

## 2. 固定规则

调整后必须满足：

```text
fixed256_server_max_num_seqs = 64
fixed256_client_concurrency = 64
fixed256_max_correct_drop = 5
candidate_fixed256_correct >= initial_oscar_fixed256_correct - 5

full_gsm8k_server_max_num_seqs = 64
full_gsm8k_client_concurrency = 64
full_gsm8k_max_correct_drop = 10
candidate_full_correct >= initial_oscar_full_correct - 10
```

两类评测继续使用 3,600 秒单次请求超时、每题 `requested_max_tokens=8192`、失败 ID 持续补测直至零失败。只在 batch1 激活的候选继续保留服务端/客户端 1/1 例外，并必须用日志证明候选真实激活。

完整 1,319 题 Gate 的 `full_gsm8k_gate_enabled=false` 默认状态不变；本次只调整其启用时采用的正式并发，不改变启停语义和最多回退 10 题的门槛。

## 3. 修改范围

同步修改以下位置，避免任务书内部冲突：

1. `acceptance.fixed256_max_correct_drop`：从 10 改为 5；`full_gsm8k_max_correct_drop` 保持 10。
2. 第 1.4 节：两类 GSM8K 的统一正式并发从 32/32 改为 64/64；将笼统的“最多回退 10 题”拆成 fixed256 最多回退 5 题、full 最多回退 10 题。
3. 第 5.2 节：fixed256 正式并发改为 64/64，历史并发说明同步改为 64/64，公式改为减 5。
4. 第 5.3 节：完整 1,319 题正式并发和失败 ID 补测并发改为 64；公式和最多回退 10 题保持不变。
5. 全文定向搜索所有 GSM8K `32`、`10`、`concurrency` 和 `max_num_seqs` 表述，确认没有正式并发旧值或 fixed256 旧门槛残留。

不改变 LongBench Feature、性能 Gate、输出预算、超时、请求失败处理、full Gate 开关或其他章节规则。

## 4. 现有用户修改的保留方式

当前用户指定的原文件比远端 `main` 更新，已经包含 full GSM8K Gate 默认关闭、失败 ID 持续补测及 3,600 秒超时等规则。实施时以该原文件为事实源：先在任务分支完整保留这些既有修改，再单独提交本轮 64/5 调整，使审阅者可以区分既有内容与本轮改动。不得用旧 `main` 文件覆盖原路径。

## 5. 验证

1. 断言 fixed256 的配置和公式均为 5。
2. 断言 full 1,319 的配置和公式仍为 10。
3. 断言所有非 batch1 例外的 GSM8K 正式服务端/客户端并发均为 64，失败补测并发也为 64。
4. 断言 batch1-only 的 1/1 例外仍存在。
5. 检查章节编号、交叉引用和 Markdown 连贯性，并运行 `git diff --check`。
6. 按仓库规范尝试 `pytest -q`；若环境仍缺少 pytest，记录退出码和限制，不虚报通过。

## 6. 环境信息

本阶段仅修改任务书，没有运行 Docker、模型服务、GPU 实验、benchmark 或 profiling。

- Docker 镜像：未使用，不适用。
- Python：`3.8.10`。
- `CUDA_VISIBLE_DEVICES`：未设置。
- `nvidia-smi`：8 张 `NVIDIA A800-SXM4-80GB`，每张 `81920 MiB`，驱动版本 `575.57.08`。
- Torch CUDA：当前主机 Python 环境未安装 `torch`，不适用。
- 相关 Python 包：当前主机环境未安装 `torch`、`numpy` 和 `pytest`；本阶段没有 Python 运行依赖。
- Git 基线：`main` 提交 `815d392`；任务分支 `docs/update-gsm8k-gate`。
