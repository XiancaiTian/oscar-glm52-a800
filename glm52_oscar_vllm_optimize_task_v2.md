# GLM-5.2 OSCAR-vLLM TTFT/TPOT 自动优化执行任务书 v2

> 适用范围：GLM-5.2 的 OSCAR MLA 路径；不包含其他模型或 full-attention 主线
>
> 历史结果来源：`OSCAR_vLLM_GLM-5.2适配与性能优化报告.md`

## 0. 上层规范与执行原则

1. 开始前完整读取项目根目录的 `AGENTS.md`、`AGENTS_misc.md`、本任务书和当前报告；冲突时以上层规范为准。
2. 使用 `planning-with-files` 维护独立的 `task_plan.md`、`findings.md` 和 `progress.md`，保证中断后可恢复。
3. 所有结论必须来自实际落盘结果。历史报告中的数值只用于提出复测候选，不得直接充当新 stage 的 baseline、latest keep 或最终结论。
4. 每完成一个阶段，先更新 `OSCAR_vLLM_GLM-5.2适配与性能优化报告.md`，再进入下一阶段。
5. 长时间实验每 10 分钟输出一次进度；禁止锁定 GPU 频率，只允许只读采集时钟、温度、功耗和显存 telemetry。
6. 每个候选只验证一个清晰假设。性能不通过时立即停止，不再执行该候选的精度 gate。
7. 性能测试、LongBench Feature 提取和 GSM8K 必须使用相互独立的服务、进程、容器、端口和时间窗口；Feature hook 不得进入 formal 性能环境。
8. baseline使用`nsys`建立一次瓶颈画像；candidate profiling仅在证据不足时执行，`ncu`只回答必要的
   单kernel问题；正式性能测试中不得开启任何profiler。
9. 只对通过其余门禁、准备keep的候选执行clean deployment。discard候选不提交、不打tag、不重复
   构建clean image。
10. 本任务书不授权修改 `/nfs/AE/txc/oscar`；该目录以及其他参考工程只读。

## 1. 固定任务配置

开始 GPU 实验前，必须将本节配置写入
`oscar_vllm_opt/stages/apple800_glm52_32k1024_chunked8k_o2_v2/config.yaml`，补齐实际镜像、文件 SHA256、GPU
身份和运行时版本，并记录 `config.yaml` 的 SHA256。以下 `null` 项必须通过新 stage 的实际 baseline
实验填写，不得沿用历史报告或其他stage数值。

```yaml
stage: apple800_glm52_32k1024_chunked8k_o2_v2

paths:
  source_repo: /nfs/AE/txc/oscar-glm/glm52_oscar_vllm
  stage_root: /nfs/AE/txc/oscar-glm/oscar_vllm_opt/stages/apple800_glm52_32k1024_chunked8k_o2_v2
  worktree_root: /nfs/AE/txc/oscar-glm/oscar_vllm_opt/worktrees
  model: /nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001
  rotations: /nfs/AE/txc/oscar-glm/artifacts/phase2/20260727T2253Z_reap_rotation_fit_final
  accuracy_project: /nfs/AE/txc/vllm_turbo_baseline_acc
  unified_report: /nfs/AE/txc/oscar-glm/OSCAR_vLLM_GLM-5.2适配与性能优化报告.md

git:
  historical_branch: feat/glm52-oscar-integration
  historical_source_commit: d4494c325b5830d09a75efe00d94dc3621fb4a7d
  stage_base_branch: oscar-opt/apple800-glm52-32k1024-chunked8k-o2-v2
  keep_tag_prefix: oscar-keep/

model:
  dtype: bfloat16
  tensor_parallel_size: 8
  pipeline_parallel_size: 1

oscar:
  kv_cache_dtype: oscar_mla_int2
  compressed_latent_rank: 512
  group_size: 128
  prefix_tokens: 64
  recent_tokens: 256
  history_page_size: 16
  shared_rotation: true
  rotation_alpha: 0.25
  clip_ratio_counts:
    "0.92": 61
    "0.94": 17
  rotation_file: rotations.pt
  rotation_sha256: 256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d
  prefill_top_k: 2048
  decode_top_k: 2048

workload:
  formal_gpu_count: 8
  input_length: 32768
  output_length: 1024
  batch_size: 1
  concurrency: 1
  request_rate: inf
  ignore_eos: true
  temperature: 0
  enable_thinking: false
  prefix_cache: disabled
  chunked_prefill: enabled
  max_num_batched_tokens: 8192
  cuda_graph_policy: vllm_default_o2
  expected_effective_cudagraph_mode: FULL_AND_PIECEWISE
  explicit_compilation_config: forbidden
  explicit_capture_sizes: forbidden

performance:
  seed: 42
  warmup_requests: 1
  measured_requests: 3
  aggregation: median_of_measured_requests
  unstable_cv_pct: 0.5
  unstable_expand_measured_requests: 10
  clock_policy: default_dynamic_no_lock
  clock_telemetry: read_only

acceptance:
  keep_primary_improvement_pct: 1.0
  keep_other_max_regression_pct: 0.5
  final_ttft_vs_bf16_max_regression_pct: null
  final_tpot_vs_bf16_min_improvement_pct: null
  full_gsm8k_gate_enabled: false
  fixed256_initial_oscar_correct: null
  fixed256_max_correct_drop: 5
  full_gsm8k_initial_oscar_correct: null
  full_gsm8k_max_correct_drop: 10
  accuracy_max_request_failures: 0
  capacity_max_drop_pct_without_detailed_retest: 1.0
  route_cooldown_discards: 3
  route_cooldown_max_target_improvement_pct: 0.2
  reset_after_no_keep_candidates: 5
  reset_after_hours_without_keep: 6
  max_resets_without_new_evidence: 10

longbench_feature:
  dataset_path: null
  manifest: oscar_vllm_opt/stages/apple800_glm52_32k1024_chunked8k_o2_v2/features/manifests/longbench_longest10_le32k_manifest.json
  tokenizer: /nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001
  sample_count: 10
  max_prompt_tokens: 32768
  hook: model.layers.4.input_layernorm
  hook_semantics: input
  max_tokens: 4
  required_generated_tokens: 4
  capture_scope: prompt_hook_chunk_plus_first_max_tokens_decode_hook_chunks
  expected_feature_seq_len: prompt_token_len_plus_4
  preserve_full_sequence_hidden_states: true
  top_p: 1.0
  seed: 42
  max_concurrency: 1
  serial_inference: true
  baseline_reference: oscar_baseline
  baseline_repeats: 3
  freeze_all_baseline_repeats: true
  candidate_repeats: 3
  baseline_stability_cosine_min: 0.99
  candidate_sample_max_cosine_min: 0.99
  candidate_comparison: per_sample_3x3_global_cosine
  candidate_sample_reduction: max
  reuse_formal_service: false
  overlap_with_formal_on_same_host: false

capacity:
  gpu_memory_utilization: 0.90
  fixed_kv_cache_memory_bytes: null
  default_probe: allocator_log_plus_short_smoke
```

### 1.1 路径与身份

| 项目 | 固定值 |
|---|---|
| 工作区 | `/nfs/AE/txc/oscar-glm` |
| 开发工程 | `/nfs/AE/txc/oscar-glm/glm52_oscar_vllm` |
| 历史复测源 | `d4494c325b5830d09a75efe00d94dc3621fb4a7d` |
| 历史分支 | `feat/glm52-oscar-integration` |
| 模型 | `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001` |
| 精度工程 | `/nfs/AE/txc/vllm_turbo_baseline_acc` |
| 主报告 | `/nfs/AE/txc/oscar-glm/OSCAR_vLLM_GLM-5.2适配与性能优化报告.md` |
| 新 stage 根目录 | `/nfs/AE/txc/oscar-glm/oscar_vllm_opt/stages/apple800_glm52_32k1024_chunked8k_o2_v2` |
| rotation artifact | `/nfs/AE/txc/oscar-glm/artifacts/phase2/20260727T2253Z_reap_rotation_fit_final` |
| `rotations.pt` SHA256 | `256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d` |
| `manifest.json` SHA256 | `0275043c070c9127354997374e9bca1c70fe1308a7b2d057f992fadedef868e5` |

启动新 stage 时必须重新记录开发工程的 `HEAD`、branch、dirty status、submodule 状态、模型 manifest、镜像 digest、驱动、CUDA、PyTorch、Triton 和 vLLM 版本。不得因为 `.gitmodules` 声明 `main` 而擅自切换分支。

### 1.2 GLM-5.2 OSCAR 固定语义

以下参数不是优化变量：

| 项目 | 固定值 |
|---|---|
| attention backend | `TRITON_MLA_SPARSE` |
| OSCAR KV cache dtype | `oscar_mla_int2` |
| shared compressed latent rank | 512 |
| rotation | K/V 共用一份 rotation |
| clip ratio | 61层为0.92，17层为0.94；以冻结manifest为准 |
| BF16 prefix | 64 tokens |
| BF16 recent | 256 tokens，ring |
| INT2 history page size | 16 |
| INT2 group size | 128 |
| prefill top-k | 2048 |
| decode top-k | 2048 |
| TP / PP | 8 / 1 |
| chunked prefill | 开启，`max_num_batched_tokens=8192` |
| CUDA Graph | vLLM默认O2；预期服务级`FULL_AND_PIECEWISE` |
| prefix caching | 关闭 |
| speculative decoding | 关闭 |
| async scheduling | 关闭 |
| context parallelism / DBO | 关闭 |
| KV transfer / offloading | 关闭 |

禁止通过修改 top-k、prefix、recent、INT2 group size、输出预算、compilation config、capture sizes或
attention backend来制造性能提升。CUDA Graph固定采用vLLM默认O2策略：预期服务级模式为
`FULL_AND_PIECEWISE`，chunked prefill使用`PIECEWISE` graph，batch1单token decode使用`FULL`
graph。不得显式覆盖compilation config或capture sizes；若实际vLLM版本行为不同，必须记录服务级
解析结果和prefill/decode实际模式，不得静默回退eager或伪造模式一致。

LongBench 数据与 manifest、GSM8K 题目 ID、prompt、chat template、generation、答案抽取、评分逻辑、
请求失败口径，以及性能与 Feature 隔离方式也属于固定语义。任何固定语义确需变化时，必须暂停当前
stage，由 Shawn 确认新 stage 边界并重建 baseline。

所有BF16与OSCAR模型测试统一`temperature=0`并关闭thinking，包括formal、LongBench Feature、
GSM8K、capacity smoke和profiling；不得把sampling或thinking改动作为优化项。

### 1.3 统一性能负载

```yaml
workload:
  input_tokens: 32768
  output_tokens: 1024
  batch_size: 1
  request_rate: inf
  max_concurrency: 1
  formal_gpu_count: 8
  tensor_parallel_size: 8
  pipeline_parallel_size: 1
  enable_chunked_prefill: true
  max_num_batched_tokens: 8192
  cuda_graph_policy: vllm_default_o2
  expected_effective_cudagraph_mode: FULL_AND_PIECEWISE
  explicit_compilation_config: forbidden
  explicit_capture_sizes: forbidden
  max_model_len: 131072
  max_num_seqs: 16
  gpu_memory_utilization: 0.92
  seed: 42
  temperature: 0
  thinking: disabled
```

要求：

- 性能输入必须由固定 token ID 构造并保存 SHA256；不得依赖自然文本分词后“约等于 32K”。
- 必须生成满 1,024 token；记录实际 input/output token 数，任何不足均视为该请求无效。
- formal 使用一个服务，按顺序完成 1 次 warmup 和 3 次正式请求，即 W1+N3。
- TTFT、TPOT 取 3 个正式请求逐请求指标的 median，不取三轮 mean 的 median。
- 目标指标 CV 超过 0.5% 时，将同一服务、同一环境下的正式请求总数扩展到 10 次，并取 N10 median；不得只重跑“好看”的样本。
- 每次 baseline/candidate 对比必须使用同一机器、相同 GPU 集合、相同镜像依赖、相同服务参数和相同输入。GPU 默认动态频率，不执行任何锁频命令。
- 开始前若发现历史遗留graphics/memory clock锁，只允许在测量窗口外执行恢复默认频率的操作并保存
  原始输出；测量期间不得控制频率。clock/power/temperature/utilization只读telemetry只用于解释波动，
  任何频率值均不得成为性能、精度、keep或discard门禁。

### 1.4 精度固定语义

- 所有模型精度测试统一 `temperature=0`，关闭 thinking；必须记录实际请求体或经验证的等价 chat-template 参数。
- 精度参考是本 stage 重新测得的“原始 OSCAR 集成基线”，不是 `previous_keep`，也不是历史报告中的 107/256 或 99/256。
- GSM8K 服务端`max_model_len`、固定最大输出上限`max_tokens=8192`、prompt、few-shot、answer parser、样本顺序和数据版本在 baseline 冻结后不得变化。
- GSM8K fixed256和完整1,319题统一使用服务端`max_num_seqs=64`、客户端`concurrency=64`、单次请求
  超时3,600秒；请求失败时只重测失败ID并持续补测，直至全部题目成功判分、`failed_requests=0`。
- fixed256精度允许相对原始OSCAR集成基线最多回退5道题；完整1,319题Gate启用时仍允许最多回退10道题。
  失败请求不得当作错误答案掩盖，`failed_requests`必须为0。
- 完整1,319题GSM8K Gate由`acceptance.full_gsm8k_gate_enabled`控制，当前固定为`false`。当前required
  精度Gate及顺序为LongBench Feature → GSM8K fixed256；任一失败立即discard。
- 第5.3节完整GSM8K规范全部保留。仅由Shawn把开关改为`true`后恢复执行，届时required顺序为
  LongBench Feature → GSM8K fixed256 → 完整GSM8K 1,319，任一失败立即discard并停止后续Gate。

## 2. 新 stage 的 baseline 与历史复测

### 2.1 三个基准身份

1. `bf16_reference`：同一 GLM-5.2 源码和运行环境，`kv_cache_dtype=auto`；用于最终 OSCAR/BF16 性能与精度对照。
2. `oscar_baseline`：保留完整 GLM-5.2 OSCAR 集成语义、但不含第 2.2 节 R01～R10 性能优化的原始 OSCAR 版本；它是精度参考和初始 `previous_keep`。
3. `historical_current`：当前历史累计源码 `d4494c325...`；它是待复测候选，不自动视为 keep。

不得拆分`best_ttft_keep`、`best_tpot_keep`或其他独立代码基准；整个stage始终只有一个线性的
`previous_keep`。

为避免旧提交混入依赖差异，优先从 `d4494c325...` 新建 stage baseline 分支，精确回退第 2.2 节 R01～R10 性能机制而保留后续 GLM 兼容修复，重构 `oscar_baseline`。若无法等价重构，才允许使用历史提交，并在 `record.json` 中记录差异审计和理由。

建立baseline前必须先通过O2 CUDA Graph前置门禁：

1. BF16与OSCAR使用完全相同的vLLM默认O2策略、chunked prefill 8K和其余固定参数启动。
2. 分别记录服务级解析模式、prefill实际模式和decode实际模式；预期依次为
   `FULL_AND_PIECEWISE`、`PIECEWISE`和`FULL`。
3. OSCAR若graph capture失败、运行报错或静默回退eager，不得建立性能baseline；先以
   `baseline_policy_correction`记录并完成CUDA Graph适配，再从同一冻结源码重测BF16与OSCAR。
4. 旧报告及历史artifact中的2K/eager结果只作实现线索，不能与本stage的8K/O2结果直接计算性能
   变化，也不能充当baseline或`previous_keep`。

建立 baseline 的顺序：

1. 冻结统一源码、镜像依赖、模型、rotation artifact 和运行参数。
2. 实测 `bf16_reference` 的性能、LongBench Feature和fixed256；仅在完整GSM8K Gate开启时执行完整GSM8K。
3. 实测 `oscar_baseline` 的性能、LongBench Feature和fixed256；仅在完整GSM8K Gate开启时执行完整GSM8K。
4. `oscar_baseline` 通过运行完整性后，冻结三次LongBench Baseline Feature集、fixed256分数，以及开关
   开启时的完整GSM8K分数，设置为初始`previous_keep`。
5. 实测 `historical_current`。它必须重新走性能和当前全部required精度Gate，不能继承历史结论。

### 2.2 历史结果仅作线索

下表数值来自旧报告，负载和聚合协议不统一，全部标记为“待复测”：

| 编号 | 历史机制 | 相关提交线索 | 历史负载 | 历史 TTFT/TPOT 变化 | 新 stage 状态 |
|---|---|---|---|---|---|
| R01 | Decode 索引与 metadata 快路径 | `98ddd3f4e`、`14c768b40` | 1K/128 | 5,049.520/243.127 → 5,014.582/206.735 ms | 待复测 |
| R02 | Prefill 有效 top-k 裁剪与 split 收敛 | `a94b1f640` | 1K/128 | 5,014.582/206.735 → 3,714.821/205.908 ms | 待复测 |
| R03 | Grouped prefill | `c3728be9f`、`a0171ed6a`、`35ab18464` | 1K/128 | 3,714.821/205.908 → 1,317.120/202.668 ms | 待复测 |
| R04 | Grouped prefill Value 精度恢复 | `24938975f`、`b9626ce9f`、`b247211c9` | 32K/128 | 106,660.424/200.303 → 47,143.207/199.458 ms | 待复测 |
| R05 | Grouped prefill 使用 8 warps | `b87a401da` | 32K/128 | 47,143.207/199.458 → 41,618.560/201.347 ms | 待复测 |
| R06 | Grouped prefill 扩展到 8 heads | `a2fe02055` | 32K/128 | 41,618.560/201.347 → 36,245.415/199.205 ms | 待复测 |
| R07 | Causal loop 精简 | `fd281f5f9` | 32K/128 | 36,245.415/199.205 → 35,683.893/197.826 ms | 待复测 |
| R08 | BF16 tile gate | `ca4a404e9` | 32K/128 | 35,683.893/197.826 → 32,683.066/200.037 ms | 待复测 |
| R09 | Top-k 索引排序/布局收敛 | `1e768aef6`、`ea8ae6b77` | 32K/128 | 32,683.066/200.037 → 32,449.245/199.155 ms | 待复测 |
| R10 | Contiguous inverse rotation | `67a0e47ff` | 32K/128 | 32,449.245/199.155 → 30,539.197/202.514 ms | 待复测 |

复测规则：

- 每项都在统一 32K/1024 协议下重新验证；不得保留任何 1K 或 32K/128 数值作为新结论。
- 优先按原依赖顺序，从最新 accuracy-valid keep 构造单一机制候选；若某项依赖未 keep 的前序机制，可测试最小依赖组合，但必须明确记录组合边界。
- 旧报告“有效”不等于本stage keep。每项都必须满足本任务书的性能keep标准和当前全部required精度Gate。
- `c0bcbbbdf` compact grouped prefill history loads 与 `d0d22489b` inverse rotation output-add fusion 是历史 discard；没有新的 profiling 证据和实质不同实现时禁止原样重试。

## 3. 唯一记录体系

### 3.1 目录与事实源

```text
oscar_vllm_opt/stages/apple800_glm52_32k1024_chunked8k_o2_v2/
├── config.yaml
├── summary.md
├── analysis/
│   └── <analysis_id>.md
├── features/
│   ├── manifests/
│   ├── baseline/
│   ├── candidates/
│   └── compare/
├── baselines/
│   ├── bf16_reference/
│   │   ├── record.json
│   │   ├── logs/
│   │   ├── raw/
│   │   └── artifacts/
│   └── oscar_baseline/
│       ├── record.json
│       ├── logs/
│       ├── raw/
│       └── artifacts/
├── experiments/<candidate_id>/
│   ├── record.json
│   ├── logs/
│   ├── raw/
│   └── artifacts/
└── latest_keep.json
```

每个 baseline/candidate 只有一个 `record.json` 作为机器事实源。`summary.md`、主报告和 `latest_keep.json` 必须由它生成或逐字段核对，不得形成第二套手工真值。

`logs/`、`raw/`和`artifacts/`按需创建，允许为空或不存在。必须落盘：required gate直接证据、正式
性能与GSM8K raw、实际执行的capacity/profiling/clean deployment复核材料，以及决定`crash`、
`blocked`或`discard`的失败证据。常规静态自检全文、重复环境输出、无决策价值的调试日志、探索性
中间结果、重复副本和每10分钟进度打印默认不保存；需要时在`record.json`写结论或引用已有证据。

每个会改变候选状态或`next_action`的步骤完成后，必须先更新`record.json`。跨Agent交接、上下文压缩
或中断恢复时，接收者直接读取并核验落盘事实源；交接消息不能替代证据。

### 3.2 `summary.md` 结构

`summary.md`只包含：

1. Baseline与Keep汇总：`bf16_reference`、`oscar_baseline`和按接收顺序排列的keep。
2. 当前状态：stage、config SHA256、当前`previous_keep`、最终目标、阻塞项和下一步。

| 版本 | 类型 | TTFT median | TTFT vs OSCAR baseline | TTFT vs previous keep | TPOT median | TPOT vs OSCAR baseline | TPOT vs previous keep | LongBench gate cosine | fixed256 | full GSM8K | KV capacity | required gates | record |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `<baseline_id>` | `oscar_baseline` | `<value>` | — | — | `<value>` | — | — | `<value>` | `<value>` | `<value/disabled>` | `<value>` | `pass` | `<record.json>` |
| `<candidate_id>` | `<ttft_keep/tpot_keep/dual_keep>` | `<value>` | `<pct>` | `<pct>` | `<value>` | `<pct>` | `<pct>` | `<value>` | `<value>` | `<value/disabled>` | `<value>` | `pass` | `<record.json>` |

正改善定义为`(reference_metric - candidate_metric) / reference_metric * 100`。表中只使用正式median，
不记录discard、逐轮过程、完整日志或raw completion。
`LongBench gate cosine`列中，Baseline填写三次Feature两两cosine的全局最小值；候选填写10个样本各自
9个candidate-vs-baseline cosine最大值中的最小值。

### 3.3 `record.json` 最小结构

```json
{
  "experiment_id": "R01_retest_or_C001",
  "stage": "apple800_glm52_32k1024_chunked8k_o2_v2",
  "kind": "baseline|candidate",
  "status": "planned",
  "started_at": null,
  "ended_at": null,
  "duration_seconds": null,
  "objective": "ttft|tpot|null",
  "optimization_source": {"source_id": "", "evidence": []},
  "hypothesis": "",
  "change_summary": "",
  "impact_class": [],
  "risk_and_rollback": {"risks": [], "rollback": ""},
  "source_state": {
    "previous_keep_id": null,
    "stage_base_branch": "oscar-opt/apple800-glm52-32k1024-chunked8k-o2-v2",
    "base_commit": "",
    "worktree": "",
    "frozen_tree": "",
    "source_head_or_sha256": "",
    "diff_sha256": ""
  },
  "config_sha256": "",
  "environment": {
    "image_digest": "",
    "model_manifest_sha256": "",
    "rotation_sha256": "",
    "cuda_visible_devices": "",
    "server_args_sha256": "",
    "effective_cudagraph_mode": "",
    "prefill_cudagraph_mode": "",
    "decode_cudagraph_mode": "",
    "frequency_locked": false
  },
  "commands": [],
  "metrics": {
    "performance": {},
    "longbench_feature": {},
    "fixed256": {},
    "full_gsm8k": {},
    "kv_capacity": {}
  },
  "checks": [],
  "artifacts": [],
  "keep_type": null,
  "failure_reason": null,
  "next_action": "",
  "commit_ref": null
}
```

约束：

1. `kind`只能是`baseline`或`candidate`；候选`objective`只能是`ttft`或`tpot`。
2. `status`只能是`planned`、`running`、`keep`、`discard`、`crash`、`blocked`、
   `profiling_insufficient`或`baseline_policy_correction`。
3. `keep_type`仅在keep时填写，取`ttft_keep`、`tpot_keep`或`dual_keep`。
4. `impact_class`从`runtime_only`、`kernel`、`kv_layout`、`accuracy_semantics`、`public_path`选择，可多选。
5. `checks`每项至少含`name`、`required`、`status`、`evidence`和`notes`；状态为`pass`、`fail`、
   `blocked`或`not_required`。
6. `artifacts`每项至少含`type`、`path`、`sha256`、`created_at`、`source_head_or_sha256`和`command`。
7. GPU环境记录镜像digest、Python、关键包、驱动/CUDA/Triton、`CUDA_VISIBLE_DEVICES`、`nvidia-smi`、
   模型/rotation SHA256和实际命令。
8. 准备keep的候选必须包含`clean_deploy`和`independent_evaluator` required check。
9. 风险、回退方式和`next_action`必须落盘，不得只存在于上下文。

## 4. 正式性能与 keep 标准

### 4.1 执行顺序

每个候选使用三类角色：

- 主Agent：瓶颈分析、候选准入、编排、证据核验和最终判定。
- 实现subagent：在独立worktree实现单一建议并完成最小针对性自检。
- 验收subagent：只对已通过性能、当前全部required精度Gate和容量门禁、准备keep的候选执行一次最终独立验收；
  不得修改源码。

实现和静态分析可以并行推进，但本任务TP=8，任何GPU服务均使用8张卡，GPU阶段必须排队。并行候选
若在其他候选keep后才准备固化，必须从最新keep重新评估代码基线和性能关系，不能直接把旧keep派生
分支合并为线性keep。

每个候选按以下顺序执行：

1. 主Agent核验源码、正式raw、有效profiling和历史失败记录，确认唯一假设、latest keep、未重复测试；
   证据不足则标记`profiling_insufficient`，不得盲目实现。
2. 从最新`previous_keep`创建独立worktree；实现subagent只实现该假设。配置或机械替换只做精确diff、
   语法/导入和直接相关测试；数学、索引、layout或ABI改写增加少量覆盖目标风险的测试。默认不创建
   独立“实现合同”、不穷举mutant、不做逐阶段独立复审，也不单独运行GPU smoke。
3. 冻结源码，记录diff/tree/文件SHA256，完成静态`capacity_impact_audit`并准备可追溯运行环境；此时
   不执行clean deployment。
4. 直接执行formal W1+N3；不设置性能初筛，不重跑previous keep。服务启动、目标OSCAR路径、
   CUDA Graph解析结果、warmup和正式请求同时验证运行完整性。若CV超阈值，扩展到N10。
5. 性能不满足则discard，不运行LongBench Feature、GSM8K、capacity GPU实测或最终独立验收。
6. formal服务退出并冻结raw/日志后，依次执行LongBench Feature→fixed256；仅当
   `full_gsm8k_gate_enabled=true`时再执行完整GSM8K。任一required Gate失败立即discard。
7. 当前全部required精度Gate通过后，仅在静态审计要求时执行轻量capacity；只有改动触及公共路径或收益机制仍无法
   由formal证实时才补充公共路径回归或profiling。诊断数值不得成为第四种精度gate。
8. 其余门禁通过后才执行clean deployment；随后由非实现者一次性核验diff、镜像身份、formal raw、
   性能/Feature隔离、当前全部required精度Gate、容量及必要回归证据，不拆成逐阶段重复授权链。
9. 主Agent核验`record.json`；全部required gate通过后才创建keep commit/tag，更新`latest_keep.json`
   和报告。失败候选保存最小证据后清理，不能作为下一候选代码基础。

### 4.2 候选性能 keep 标准

相对 `previous_keep`：

- TTFT 或 TPOT 至少一项改善 `>= 1.0%`；
- 另一项回退 `<= 0.5%`；
- 请求全部成功，实际输入 32,768、输出 1,024；无 OOM、preemption、fallback、意外CUDA Graph模式/backend/top-k变化；
- 不允许以更改精度语义、输出预算、服务并发、chunked prefill 大小或 GPU 状态换取提升。

任何一项不满足即`discard`。两项均改善至少1%时为`dual_keep`，否则按主要目标记为`ttft_keep`或
`tpot_keep`。0.5%是硬回退上限，不使用统计噪声豁免。previous keep的正式结果直接沿用，不因新
候选重复测试；只有结果不属于同一stage、同型号硬件、固定负载和无锁频协议，或证据确认无效时，
才重建整个pair，不能因GPU物理编号变化而重测。

### 4.3 最终目标

GLM-5.2相对BF16的最终数值目标由Shawn在首轮统一BF16/OSCAR复测后确认并写入`config.yaml`；
在此之前仍按第4.2节迭代并报告相对BF16的实际差距，不得沿用其他模型stage的目标。

### 4.4 唯一验收矩阵

所有required gate必须写入`record.json.checks`。任一required gate为`fail`或`blocked`时不得keep、
提交或合并。

| Gate | 何时 required | 通过条件 |
|---|---|---|
| fixed semantics | 所有候选 | 满足第1.2～1.4节 |
| source identity | 所有候选 | endpoint、镜像、源码树、config、raw与experiment ID均属于当前候选 |
| OSCAR path | 所有候选 | `oscar_mla_int2`、三段式mixed-KV、rotation、store/read及DSA/MLA路径实际执行 |
| CUDA Graph policy | baseline与所有候选 | 默认O2；服务级预期`FULL_AND_PIECEWISE`，prefill实际`PIECEWISE`、decode实际`FULL`，无静默eager fallback |
| runtime integrity | 所有候选 | 正式请求shape、dtype、finite、token数、backend和fallback状态正确 |
| formal performance | 所有候选 | 正式结果完整并满足第4.2节 |
| performance/Feature isolation | baseline与所有候选 | formal无Feature instrumentation，执行窗口不重叠 |
| LongBench Baseline Feature | `oscar_baseline` | 完整 prompt 加前4个decode token的Feature独立提取3次，同一样本三份Feature两两global cosine均`>=0.99`，冻结三次Baseline Feature集 |
| LongBench Feature | 性能通过候选 | 完整Feature独立提取3次并与三次Baseline逐样本做3×3 global cosine；每个样本9个cosine的最大值`>=0.99`；失败即停止后续精度测试 |
| fixed256 | LongBench通过候选 | 满足第5.2节；失败即discard |
| full GSM8K | `full_gsm8k_gate_enabled=true`且fixed256通过 | 满足第5.3节；开关为`false`时记录`required=false`、`status=not_required`且不执行 |
| capacity impact audit | 所有候选 | 冻结源码后完成静态diff审计，不调用GPU |
| light KV capacity | formal和精度通过且审计要求重测 | 满足第6.1节 |
| baseline profiling | baseline | 满足第6.2节baseline画像要求 |
| candidate profiling | 证据不足且影响准入或归因时 | 满足第6.2节 |
| public path regression | 触及公共路径且formal未覆盖时 | legacy、BF16和非目标geometry可加载执行且shape/dtype/finite正确 |
| clean deployment | 仅准备keep的候选 | 满足第8节clean deployment要求 |
| independent evaluator | 所有准备keep候选 | 非实现者核验全部required证据且不引入第四种精度判据 |

formal除TTFT/TPOT逐请求值外，还必须记录ITL的mean/median/p99/std/CV、output token/s、wall time、
raw JSON/JSONL、serve log、bench log、实际命令和环境。CV按
`standard_deviation / arithmetic_mean * 100`计算。任一请求失败、OOM、timeout、服务异常或关键字段
缺失时该轮无效；保存失败证据，修复后整轮重跑，不得只统计成功子集。

修改数学、索引、layout或reduction顺序时，可使用reference/legacy、bitwise、最大绝对差、cosine、
PPL或loss定位问题，但必须标记`diagnostic_only`、`required=false`。LongBench Feature和fixed256始终
是required精度Gate；完整GSM8K仅在开关开启时required。任何其他自定义数值容差不得触发精度discard
或替代当前required精度Gate。

## 5. 精度 gate

### 5.1 LongBench Feature gate

首次建立`oscar_baseline`时，从LongBench或LongBenchV2官方数据中选择最终prompt token长度不超过
32,768的最长10个去重样本：

1. 使用第1节GLM-5.2 tokenizer；长度在官方prompt format叠加固定chat template并关闭thinking后
   计算，不能用原始`context`或`input`估算。
2. 按prompt内容去重，再按token长度降序选择10条；baseline与所有候选复用同一manifest。
3. manifest至少记录`sample_id`、数据集路径/row index、prompt SHA256、prompt token长度、prompt
   template、tokenizer、数据集SHA256、来源URL/revision和选择时间。
4. 若本地没有数据，只能从LongBench官方GitHub、官方Hugging Face或项目主页明确链接的渠道获取并
   记录身份；官方数据或tokenizer无法获得且自主修复失败时，stage标记`blocked`，不得换用伪造数据。

Feature提取固定口径：

```text
hook = model.layers.4.input_layernorm
hook_semantics = input
capture_scope = prompt hook chunk + first max_tokens decode hook chunks
max_tokens = 4
required_generated_tokens = 4
expected_feature_seq_len = prompt_token_len + 4
preserve_full_sequence_hidden_states = true
temperature = 0
enable_thinking = false
top_p = 1.0
server_seed = client_seed = torch_manual_seed = cuda_manual_seed_all = 42
max_concurrency = 1
serial_inference = true
prefix_cache = disabled
generation_config = vllm
deterministic_feature_mode = true
capture_pp_rank = 0
capture_tp_rank = 0
```

运行时必须记录解析后的完整模块名并证明采集的是`input`。TP=8、PP=1下固定采集TP rank 0 / PP
rank 0，baseline与候选完全一致；不得混合不同rank tensor。每个样本逐条串行执行，实际生成4个token。

Feature保存范围和组合policy必须完全对齐`fp8_speed_up_v6_tpot_task.md`：保存完整prompt hook chunk与
first `max_tokens` decode hook chunks，并沿用其去重和组合方式；不得改成只保存prompt最后4个token、只
保存generated/decode部分、只保存最后token或其他窗口口径。必须保存完整序列hidden states，记录batch
维、序列维和hidden维，最后一维必须等于模型hidden size；预期完整Feature序列长度通常应为
`prompt_token_len + 4`。若实际生成token数不是4、完整Feature序列长度或shape不符合该policy，必须标记
失败，不能参与验收。

Feature必须非空、非全0、shape一致且全部finite。每条保存Feature、prompt/generated token数、完整
Feature序列长度、finish reason、shape、dtype、token/张量选择方式、去重和组合policy、源码/配置身份
和SHA256；提取与比较脚本也必须冻结并记录SHA256。Baseline与所有候选必须使用同一hook点、hook语义、
推理参数、输入样本、输出token设置、token/张量选择、去重和组合方式。

Feature提取与formal必须严格隔离：formal结束、raw/日志冻结且服务退出后，才启动独立Feature镜像、
服务、进程、endpoint、端口和artifact目录；两阶段记录容器、PID、GPU、起止时间、源码树和镜像身份。
同一主机执行窗口不得重叠，formal环境不得存在hook或保存逻辑。若发现formal受Feature instrumentation、
I/O或资源竞争影响，性能结果无效，必须在完全关闭Feature环境后重测。

`oscar_baseline`必须独立重启服务和测试，共提取3次完整Feature。同一样本三次完整Feature两两计算
global cosine similarity，所有两两cosine均`>=0.99`，才能认定Baseline Feature可复现并冻结这三次
Baseline Feature集。稳定性不满足时先排查seed、generation config、prefix cache、hook语义、dtype、
batching、去重和组合方式、保存窗口、tokenizer和模型加载，仍无法修复则stage标记`blocked`。

性能通过候选必须对同一10个样本独立重启服务和测试，共提取3次完整Feature。每个样本的三次候选
Feature与三次Baseline Feature逐一计算global cosine similarity，形成3×3共9个candidate-vs-baseline
cosine；global cosine必须对完整的`prompt + 4 decode` Feature计算，不得改为逐token cosine、只比较
decode token或只比较部分采集位置。每个样本9个cosine中的最大值`>=0.99`即通过该样本，10个样本必须
全部通过；任一tensor无效、不足或样本未通过即discard，不执行fixed256。

固定manifest、Baseline三次完整Feature及两两比较、候选三次完整Feature、每个样本9个cosine、逐样本
最大cosine、全局最小/最大cosine和比较JSON均属于required gate证据，不能省略。

### 5.2 GSM8K fixed256

使用精度工程现有official_v5确定性selector，冻结256个题目ID、prompt hash、gold、chat template、
sampling、答案抽取和评分器。fixed256复用第5.3节固定的`requested_max_tokens=8192`，不得基于
256题子集或prompt长度重新计算，也不得使用更短输出预算。

TP=8精度服务固定使用服务端`max_num_seqs=64`、客户端`concurrency=64`，每次请求超时3,600秒。
请求失败时保留已成功样本，只按冻结ID重测失败题目，并重复该过程直至256/256全部成功判分、
`failed_requests=0`；不得把失败请求按错误答案计入，也不得通过缩短输出预算、降低并发或改变生成参数
规避失败。历史256/256与128/128门禁结果只作为并发选择依据，不得替代本任务书规定的64/64正式配置。
这不改变性能formal的batch1。
若候选机制设计上只在batch1启用，精度服务必须改为`max_num_seqs=1`、`concurrency=1`，并通过日志
证明候选真实激活，禁止用自动fallback结果冒充候选精度。

候选通过条件：

```text
candidate_fixed256_correct >= initial_oscar_fixed256_correct - 5
candidate_fixed256_scored == 256
candidate_fixed256_request_failures == 0
candidate_fixed256_requested_max_tokens_by_id == initial_oscar_fixed256_requested_max_tokens_by_id
```

`initial_oscar_fixed256_correct`必须由本stage新测`oscar_baseline`填写；历史107/256、99/256不得代入。
任一条件不满足即discard。开关关闭时，fixed256通过即完成当前GSM8K精度Gate；开关开启时，fixed256
通过才允许进入完整GSM8K。

### 5.3 完整 GSM8K

本节由`acceptance.full_gsm8k_gate_enabled`控制，规范保留但当前开关为`false`，因此不执行且在
`record.json.checks`中记录`required=false`、`status=not_required`，并在`notes`注明开关关闭。后续仅由Shawn把开关改为
`true`后恢复为required Gate；重新开启后，当前`previous_keep`及后续准备keep的候选必须补齐本节结果，
不得把关闭期间的`not_required`视为通过。

使用official_v5完整1,319题，固定：

```text
temperature = 0
enable_thinking = false
top_p = 1.0
seed = 42
server_max_model_len = 16384
requested_max_tokens = 8192
metric = exact numeric match
```

所有GSM8K题目固定设置`requested_max_tokens=8192`，表示每题最多生成8192个token，允许模型遇到
EOS提前结束，不要求强制生成满8192个token。BF16、原始OSCAR、所有候选以及fixed256中的
所有题目必须使用该同一预算；不得按prompt、shard或子集重新计算，也不得临时缩短预算。该固定值写入
`config.yaml`和manifest后冻结。当前stage使用
`run_accuracy_suite_fixed8192.py`显式覆盖official_v5 runner的动态预算逻辑；冻结该wrapper与原runner
的SHA256，并从predictions确认1,319题的`requested_max_tokens`集合恰好为`{8192}`。完整suite通过同一
TP=8服务并发评测，固定服务端`max_num_seqs=64`、客户端`concurrency=64`，每次请求超时3,600秒。
BF16、原始OSCAR和候选必须使用相同并发、超时和输出预算。batch1-only候选按第5.2节使用1/1，但
输出预算仍固定为8192。

主轮出现任何请求失败时，保留已成功样本，使用official_v5 `--resume`只重测失败ID；补测仍使用
3,600秒请求超时和正式客户端并发64。若补测后仍有失败，继续只对剩余失败ID执行同配置补测，直至
1,319题全部scored、0 request failure。持续失败时必须诊断并修复服务或客户端故障后继续，不得重跑
或替换已成功样本，也不得临时缩短输出预算、降低并发或改变生成参数。BF16、原始OSCAR与候选采用
完全相同规则。

候选通过条件：

```text
candidate_full_correct >= initial_oscar_full_correct - 10
candidate_full_scored == 1319
candidate_full_request_failures == 0
candidate_full_requested_max_tokens_by_id == initial_oscar_full_requested_max_tokens_by_id
```

`initial_oscar_full_correct`必须由本stage新测`oscar_baseline`填写。聚合前验证题目无重复、无缺失，
ID、prompt hash和gold一致。raw至少包含question、gold、completion、抽取答案、正确性、finish reason、
请求耗时和错误。开关开启后，所有准备keep候选都必须当轮实际执行完整GSM8K，runtime-only候选也不得继承。

开关开启时，BF16同协议结果必须落盘用于报告；候选精度回退门槛始终相对原始`oscar_baseline`，不相对
`previous_keep`。除LongBench Feature、fixed256和完整GSM8K外，不得新增其他精度gate。

## 6. KV capacity、profiling 与证据驱动优化

### 6.1 KV capacity

`oscar_baseline`使用轻量方案实测：固定`gpu_memory_utilization=0.90`，不设置
`kv_cache_memory_bytes`，让vLLM按实际模型和workspace分配；保存`GPU KV cache size`、OSCAR
INT2/BF16 pool、maximum concurrency和allocator日志，并运行一条短smoke确认cache可读写且无OOM。

所有候选在源码冻结后、formal前先做静态`capacity_impact_audit`，该审计只看diff、不调用GPU：

- 纯计算kernel且不改变长期buffer、layout、allocator或workspace：继承最近有效容量证据，记录来源，
  `light KV capacity=not_required`，不得声称当轮实测。
- 触及KV dtype/bitwidth/group、layout、metadata、allocator、长期workspace、rotation或mixed-KV window：
  标记`light KV capacity=required`，仅在formal和当前全部required精度Gate通过后运行轻量实测。

候选相对`oscar_baseline`或最近有效同语义容量下降不超过1%视为无明显回退；超过1%触发详细容量
压力测试，详细复测仍确认明显回退时不得keep。理论bitwidth/BPE不能替代allocator实测。任何OOM、
preemption、可用block明显下降或相同32K/1024请求无法完成均视为失败。

### 6.2 Profiling

baseline profiling为required：使用Nsight Systems分析真实32K/1024 serving时间线，warmup后采集正式
语义请求，分离prefill/TTFT和稳定decode/TPOT窗口，记录kernel、同步、CPU/GPU调度、通信和主要耗时。
instrumentation下时延只作归因，绝不写入正式性能表。

candidate profiling默认`required=false`；只有现有证据不足且缺口会影响候选准入或收益归因时才
required。先复用同源码、同负载、同路径的有效证据；仅在以下情况补采：

- 连续候选无有效提升且无法定位瓶颈；
- 性能变化与假设不一致；
- 两种kernel/并行方案间需要结构性决策；
- 需要确认kernel time、launch、通信、同步、访存或资源占用是否为主因。

先使用成本较低的已有trace、kernel计时、Triton编译信息和Nsys摘要。只有Nsys已定位具体kernel且
需要occupancy、HBM/L2、寄存器、指令或cache指标时才用NCU。每次NCU前记录：尚未回答且会改变
决策的问题、现有证据为何不足、目标kernel和所需metric；预定矩阵或“补齐轮数”本身不是理由。
raw无效、源码/语义/负载变化，或新问题无法由旧证据回答时才允许重跑。

每份profiling记录命令、工具版本、config SHA256、frozen source、环境、原始产物路径和SHA256。
源码关键路径、负载或固定配置变化后旧profiling过期；证据不足时标记`profiling_insufficient`。
profiling使用独立进程/端口，结果不得混入formal。

全局reset时，将只读`/nfs/AE/txc/oscar-glm/glm52_speed_up_v2_stable_8th`纳入受控对照，但必须先
验证它确实支持同一GLM-5.2模型、同一OSCAR语义和同一32K/1024/TP=8负载；若无法等价运行，只允许
源码/框架层归因，不得伪造绝对性能对比。可比时至少比较kernel数与launch、调度边界、KV读取与
反量化复用、融合、通信、HBM/L2、occupancy、register/local spill和主要指令路径。参考工程只读，
跨实现绝对时延只作归因，不直接充当vLLM keep门禁。

### 6.3 优化方向

每个新候选必须引用实际证据，优先检查：

1. TP=8 下 decode metadata、索引选择、demotion scratch 和跨 rank 同步开销；
2. grouped prefill 的 KV tile 共享、INT2 unpack/反量化、warp/head 映射和寄存器压力；
3. prefill/decode kernel launch 数、Python/C++ 调度、CPU-GPU 同步和重复 tensor 构造；
4. inverse rotation、Value 路径、top-k 后访问局部性与中间布局；
5. vLLM 框架接入相对高效参考实现的额外 materialization、fallback 或串行路径。

允许重写 Triton 算子或并行方式，但必须保持第 1.2 节语义，并以 profiling 或受控 A/B 证据说明为什么值得实施。

## 7. 防止无效迭代

### 7.1 候选准入与去重

每个候选实现前必须有`optimization_source`，记录证据、瓶颈、机制、目标指标、预期变化、影响范围、
风险、回退和验证方法。优化族按收益机制划分，如INT2 unpack/dequant、MLA latent reuse、attention
dataflow、inverse rotation、metadata/layout、kernel launch、TP通信或host pipeline，不按文件名划分。

主Agent比较历史候选机制、位置、参数和负载。无新证据的同机制改名、相邻参数微调或失败worktree
叠加视为重复，不得执行。microbenchmark收益不能单独构成serving候选，必须说明如何作用于当前真实
请求窗口。候选可重构kernel、KV数据流、launch融合、runtime或框架集成，但一次仍只有一个可验证主因。

### 7.2 Cooldown 与 reset

同一优化族连续3个有效discard，且各候选目标指标改善均低于0.2%时立即cooldown。记录共同失败机制、
已排除范围和解除条件；只有新的profiling、SASS、源码事实或raw benchmark改变原判断时才能解除。

自最近keep起，满足任一条件时暂停候选并reset：连续5个源码候选无keep，或累计6小时无keep。
reset由未参与当前路线实现的独立analyst执行，读取latest keep源码、固定配置、raw、有效profiling、
candidate records和失败artifact；证据过期或不足时按第6.2节补采。

reset报告保存到`analysis/`，至少包含触发原因、复核证据、TTFT/TPOT瓶颈排序、已证伪/cooldown方向、
新证据、一个可执行TTFT候选、一个可执行TPOT候选和下一步排序，并按第6.2节复核GLM只读参考实现。
新keep后候选数和小时清零；没有新测量、源码事实或可验证机制的reset记为“无新证据reset”。reset
不是重复跑previous keep，只有发现环境漂移或证据损坏时才重测pair。

### 7.3 重试边界

1. `discard`、`crash`、`blocked`和`profiling_insufficient`记录直接原因、证据、已排除范围、是否可
   重试、重试前置条件和next action。
2. 未满足条件不得重复同类候选；只有固定负载、latest keep或关键路径变化且新证据足以改变旧判断，
   重测才不属于无条件重复。
3. 环境故障、请求失败、OOM、timeout或关键字段缺失不用于判断优化机制，但必须保存并修复；针对性
   修复后可完整重跑一次。GSM8K请求失败是例外，必须按第5.2、5.3节持续只补测失败ID，直至全部成功；
   数值或稳定性能失败不得靠增加seed、挑样本或无限重复改变结论。
4. 无法提供独立实现、验收或analyst角色时，不得由主Agent伪装切换角色；候选或reset标记blocked，
   记录恢复条件。
5. 历史GLM候选若仅因reference/legacy最大差、自定义容差、PPL、loss、bitwise或非本任务书规定的
   Feature/cosine而提前discard，原精度淘汰结论无效，必须从最新previous keep重建，使用
   `reevaluation_of`指回旧记录，并按formal→LongBench→fixed256重评；仅在完整GSM8K Gate开启时追加
   完整GSM8K。已有正式性能失败
   或完整GSM8K失败者不因本规则自动重跑。

## 8. Git、worktree 与部署

### 8.1 Baseline分支

源码Git根为嵌套的`glm52_oscar_vllm`仓库。开始前必须：

1. 核验历史源`d4494c325...`的commit、tree、branch、dirty状态和已有证据。
2. 按第2.1节重构并验证原始`oscar_baseline`，从其冻结commit创建
   `oscar-opt/apple800-glm52-32k1024-chunked8k-o2-v2`，作为stage唯一基线分支和首个`previous_keep`。
3. 完成baseline运行完整性、当前全部required精度Gate、capacity、Nsys画像和身份核验；diagnostic-only数值不参与精度
   淘汰。

不得使用`git reset --hard`、`git checkout --`或其他破坏性命令清理用户修改。

### 8.2 候选worktree

1. 每个候选使用独立worktree：`oscar_vllm_opt/worktrees/YYYY-MM-DD-HH-MM-<candidate>`，从创建时
   stage基线分支最新commit派生。
2. 实现subagent可以修改；验收subagent只读。候选之间不得共享worktree、容器、端口或artifact。
3. 候选冻结后记录diff、tree、文件SHA256、运行镜像digest和实验配置。
4. discard/crash/blocked不提交；其record和必要artifact完整后才移除worktree，且不得作为下个候选
   代码基础。
5. 本模型TP=8，一次性能或精度服务占用全部8卡；不能并行两个GPU服务。无GPU实现/静态检查可并行，
   GPU阶段排队。
6. 使用GPU前按`AGENTS_misc.md`连续两次、间隔1分钟确认空闲且无非本项目进程；不得抢占他人进程。

### 8.3 Keep commit与tag

1. 只有全部required gate通过的候选才能创建keep commit。
2. 每个keep只使用一个实现commit；提交信息包含experiment ID和keep type。
3. commit tree必须与clean deployment冻结验收树一致。
4. 创建annotated tag `oscar-keep/<experiment_id>`并指向最终实现commit。
5. 合并或快进stage基线分支后更新`previous_keep`。
6. 本地commit/tag不等于发布；未经Shawn授权不得push、创建PR或改变远端状态。

### 8.4 Artifact与clean deployment

1. Git commit是可部署源码事实源；record保存环境、config和artifact SHA256。
2. 只保存参与门禁、归因、失败判定或复现所需的模型/rotation、raw、trace、`*.nsys-rep`、
   `*.sqlite`等，记录稳定路径、大小、SHA256和生成命令；非必要中间产物不落盘。
3. 模型与rotation必须位于稳定非临时路径；Docker镜像记录名称和不可变digest。
4. 仅对通过其余keep门禁、准备keep的候选执行clean deployment：从冻结候选树构建镜像，不挂载开发
   源码patch；smoke后验证镜像内源码hash与commit tree一致，并复核服务参数、OSCAR路径、一次正式
   性能结果和必要精度摘要。
5. 顶层报告和planning只在主checkout更新，不得混入候选worktree实现commit。

## 9. 报告、停止与恢复

### 9.1 实时报告

每完成一个阶段立即更新 `OSCAR_vLLM_GLM-5.2适配与性能优化报告.md`：

- 统一复测后的 BF16、原始 OSCAR、historical current；
- 每个历史复测项和新优化项的唯一改动、TTFT、TPOT、相对previous keep的变化、当前required精度Gate、最终判定；
- 只把同时通过性能和精度 gate 的候选列入“有效优化项”；
- 历史 1K/128 和 32K/128 数值保留时必须明确标注“历史参考”，不得与新 32K/1024 表格混排。

修改报告前重新全文读取；修改后检查章节编号、表格、交叉引用和汇总数据是否与各小节一致。

每产生一个同时通过正式性能和当前全部required精度Gate的新有效优化项，主报告必须同时新增对应优化小节并更新
有效优化项汇总表；不同负载、输出长度或运行边界必须显式区分，不得跨口径直接计算收益。

### 9.2 Stage 完成

满足以下全部条件才可宣布完成：

1. BF16、原始 OSCAR 和 `d4494c325...` historical current 已在统一协议下复测；
2. R01～R10 均有新 stage 判定，不存在仅沿用历史数值的项目；
3. 所有keep均通过当前全部required精度Gate、必要capacity检查和clean deployment；完整GSM8K关闭时
   必须明确记录`required=false`、`status=not_required`和关闭原因，不得写成`pass`；
4. baseline profiling、必要candidate profiling和所有准备keep候选的独立验收均已完成；
5. 最终 latest keep 与 BF16 的 TTFT/TPOT、精度和容量差距已写入主报告；
6. Shawn 已确认写入`config.yaml`的最终数值目标达到，或明确要求结束当前 stage。

### 9.3 停止与阻塞

本任务不因单次失败、普通bug或单个候选无收益停止。以下情况结束自动迭代并汇报：

1. Shawn明确要求停止或确认stage完成。
2. 达到第9.2节条件并通过终局验收。
3. 缺GPU、模型、数据、权限、固定环境或必要工具，完成合理自主修复后仍无法继续。
4. 连续10次独立analyst reset均未产生新证据或可验证候选方向。
5. 固定配置必须改变，需要Shawn确认新stage边界。

进入blocked或结束前，先安全终止本项目服务/实验，完成当前`record.json`、`summary.md`、planning和
主报告更新，保存有效/失败证据、已排除方向和解除阻塞所需输入。不得把耗时、结果不理想或待
profiling本身当作阻塞。
