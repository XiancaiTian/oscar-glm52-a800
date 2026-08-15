# GLM-5.2 LongBench Feature Gate 对齐设计

## 1. 目标

修改 `glm52_oscar_vllm_optimize_task_v2.md`，使 LongBench Feature 的采集范围、重复提取和相似度计算方式完全对齐 `fp8_speed_up_v6_tpot_task.md`。本次只调整 LongBench Feature gate，不改变 GLM-5.2 任务自身的模型、数据集、tokenizer、TP/PP rank、服务隔离、性能负载、GSM8K、capacity 或其他门禁语义。

## 2. Feature 采集口径

每个样本固定使用 `model.layers.4.input_layernorm` 的 input hidden states，按参考任务书的 policy 保存并组合：

```text
prompt hook chunk + first max_tokens decode hook chunks
max_tokens = 4
expected_feature_seq_len = prompt_token_len + 4
```

必须保存完整序列 hidden states，并沿用参考任务书的去重和组合方式。禁止只保存 prompt 最后 4 个 token、只保存 generated/decode 部分、只保存最后 token，或改用其他窗口、池化与 token 选择方式。每个样本必须记录 prompt token 数、实际 generated token 数、完整 Feature 序列长度、shape、dtype、finish reason 和证据路径；实际 generated token 数不是 4 时不得进入验收。

## 3. 相似度计算与通过条件

`oscar_baseline` 必须通过独立重启服务完成 3 次 Feature 提取。同一样本的 3 个完整 Feature 两两计算 global cosine similarity，所有两两 cosine 均须 `>= 0.99`，才能认定 Baseline Feature 可复现。

每个性能通过候选也必须通过独立重启服务完成 3 次 Feature 提取。对每个样本，将 3 个候选完整 Feature 与 3 个 Baseline 完整 Feature 逐一比较，形成 `3 × 3 = 9` 个 stage/candidate-vs-baseline global cosine similarity。9 个 cosine 中的最大值 `>= 0.99`，该样本通过；10 个样本必须全部通过。

这里的 global cosine 对完整的 `prompt + 4 decode` Feature 计算，不得改为逐 token cosine、只比较 decode token，或只比较部分采集位置。本设计沿用参考任务书的完整 Feature 比较语义，不额外发明与参考任务书不同的 flatten、mean pooling 或其他聚合规则；提取与比较脚本的张量选择和组合 policy 必须在 Baseline 与所有候选之间保持完全一致。

## 4. 文档修改范围

为避免同一任务书内部出现冲突，实施时同步检查并修改以下位置：

1. 第 1 节 `longbench_feature` 固定配置：补充完整保存范围、候选重复次数和 3×3 比较口径，移除与旧窗口相关的含义。
2. 第 1.4 节及 baseline 冻结描述：明确冻结 3 次 Baseline Feature 集，而不是只依赖单份 canonical tensor。
3. 第 3 节记录结构和汇总字段：保证能够记录 3 次候选、9 个 cosine、逐样本最大值和全局统计。
4. 第 4 节 required gate 表：把 LongBench Feature 通过条件指向更新后的第 5.1 节。
5. 第 5.1 节：删除 `input_window = prompt最后4个token`，改为完整 prompt 加前 4 个 decode hook chunks，并写明完整序列、去重组合、3 次 Baseline、3 次候选和逐样本 3×3 global cosine。
6. 后续验收清单、失败条件、流程图、最终完成条件及 artifact 字段：清除“候选只提取 1 次”“每个采集位置 cosine”“单一 canonical reference 比较”等旧口径。

不调整章节编号；修改后检查标题顺序、交叉引用和上下文连贯性。

## 5. 验证方法

实施完成后执行以下静态检查：

1. 搜索并确认不存在 `input_window = prompt最后4个token`、`候选提取1次`、只比较 generated、只比较最后 token 等旧口径。
2. 搜索所有 `Feature`、`cosine`、`canonical`、`repeat`、`generated_tokens` 相关段落，逐处核对一致性。
3. 确认文档明确包含 `prompt_token_len + 4`、完整序列 hidden states、Baseline 3 次、候选 3 次、逐样本 3×3 共 9 个 global cosine、逐样本最大值 `>= 0.99`。
4. 执行 `git diff --check`，检查 Markdown 空白错误。
5. 检查章节编号和交叉引用，不运行 `pytest -q`：本任务仅修改未被现有测试套件覆盖的任务书文档，pytest 不能验证其语义；以定向文本检查和人工 diff 审阅为准。

## 6. 风险与边界

- “完全对齐”只覆盖 Feature 采集和相似度计算，不复制参考任务书的 GLM-5.1 模型路径、1P1D 四机拓扑、GSM8K 阈值或 TPOT stage 规则。
- 从单次候选对 canonical reference 的比较改为 3×3 最大值判定，会改变计算量和 gate 统计含义；这是完全对齐参考任务书所必需的变化。
- 目标任务书当前在主检出目录中未被 Git 跟踪。实施时须完整保留原文件内容，只对上述相关段落做可追溯修改，并在专用 worktree 内提交。

## 7. 环境信息

本阶段仅编写文档设计，未运行 Docker、模型服务、GPU 实验、benchmark 或 profiling。

- Docker 镜像：未使用，不适用。
- Python：`3.8.10`。
- `CUDA_VISIBLE_DEVICES`：未设置。
- `nvidia-smi`：8 张 `NVIDIA A800-SXM4-80GB`，每张 `81920 MiB`，驱动版本 `575.57.08`。
- Torch CUDA：当前主机 Python 环境未安装 `torch`，因此 `torch.version.cuda` 和 `torch.cuda.is_available()` 不适用。
- 相关 Python 包：当前主机 Python 环境未安装 `torch` 和 `numpy`；本阶段没有 Python 运行依赖。
- Git 基线：`main` 提交 `eea1d87`；任务分支 `docs/align-glm52-longbench-feature`。
