背景：

https://arxiv.org/pdf/2605.17757 是一篇 KV Cache 压缩算法论文，其配套代码为 https://github.com/FutureMLS-Lab/OSCAR。 该配套代码已提供 SGLang 的 OSCAR 算法适配，但尚不支持 vLLM。

任务目标：

基于 vLLM v0.25.0，交付一个 OSCAR-vLLM 原型实现，使 `/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507` 可以在 vLLM 中启用 OSCAR KV Cache 推理路径，并完成 GSM8K 精度验收。任务优先目标是研究复现可跑通、路径可证明、实验可复现；upstream 合入质量、多模型泛化、完整性能优化不作为本任务硬性验收。

版本与参考实现：

- vLLM 基线固定为 vLLM v0.25.0。实现、测试和报告均基于该版本，不在任务过程中追逐新的 upstream 变化。
- OSCAR-vLLM 的开发、集成、测试和实验均必须基于本机现成 Docker 镜像 `docker.m.daocloud.io/vllm/vllm-openai:v0.25.0`。
- OSCAR 参考实现固定为 `FutureMLS-Lab/OSCAR` 当前 `main` 的 HEAD。任务启动时需记录该 commit hash，后续实现参考均以该 commit 为准。
- 实现过程可以参考 OSCAR 的 SGLang 实现，以及 https://github.com/vllm-project/vllm/pull/46774。

必做实现范围：

- 必做支持范围限定为 `/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507` 在 vLLM v0.25.0 下的推理路径。
- 实现应尽量沿用 vLLM 现有抽象，避免无关路径写死；但多模型、多 head_dim、多 dtype、多并发特性、prefix cache 组合、spec decode 等通用支持不作为硬性验收。
- OSCAR calibration 参数固定使用 `/data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations`：
  - `k_rotation_qqt_r_h_pbr.pt`
  - `v_rotation_sst_r_h_pbr.pt`
- vLLM 集成必须加载上述 K/V rotation 文件，并将 clipping ratio 对齐 OSCAR SGLang 配置：
  - `SGLANG_OSCAR_K_CLIP_RATIO=0.96`
  - `SGLANG_OSCAR_V_CLIP_RATIO=0.92`
- mixed KV 策略必须对齐 OSCAR SGLang：
  - `SGLANG_MIXED_KV_PREFIX_TOKENS=64`
  - `SGLANG_MIXED_KV_RECENT_TOKENS=256`
- 即 prefix/sink 前 64 个 token 保持 BF16，recent 最近 256 个 token 保持 BF16，中间 history KV 使用 OSCAR rotation + clipping + INT2 存储。
- 若实现未使用 OSCAR attention-aware rotation / clipping 参数，而是退化为随机旋转、Hadamard-only 或占位参数，即使 INT2 cache 跑通，也不能视为完成 OSCAR 集成。

Kernel 与实现方式：

- 可以先实现 PyTorch / Triton 非融合 reference correctness path，用于验证 rotation、clipping、INT2 pack/unpack、dequant、attention 输出等数值语义。
- 最终 serving path 的性能关键路径应使用 Triton kernel 实现，包括 KV 写入/量化，以及 attention 读取/反量化/逆旋转。避免使用 CUDA，除非对应功能无法用 Triton 合理实现，并在报告中说明原因。
- 若最终只保留 reference correctness path，不能视为完成集成任务。

OSCAR 路径与压缩率验收：

- 路径证明：运行日志、tracing 或等价证据必须显示启用了 OSCAR KV cache 配置、加载了 rotation / clipping 参数、调用了 OSCAR KV 写入路径和 attention 读取路径。
- 代码证明：报告中需列出 vLLM 中新增/修改的关键入口，包括 KV cache allocation、KV write/quantize、attention decode/dequantize、配置开关等。
- 压缩率证明：给出同一模型、同一 batch/seq 配置下 BF16 KV cache 与 OSCAR KV cache 的理论 bytes/token/layer 和实测显存或 cache block 占用对比。
- 压缩率必须按 prefix 64 BF16、recent 256 BF16、history INT2 的 mixed KV 布局计算，不能只报告 history INT2 的理论压缩率。
- 若实现只是将整段 KV cache 在 attention 前反量化成临时 BF16 大 cache，不能视为通过 OSCAR 推理路径验收。

精度验收：

使用 `/data/ssd1/txc/vllm_turbo_baseline_acc` 数据集中的 GSM8K 子集，使用 `/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507` 作为测试模型，验证 OSCAR-vLLM 精度是否与 OSCAR SGLang 对齐。

在运行 OSCAR-vLLM GSM8K 前，必须先用同一 vLLM v0.25.0 基线、同一模型、同一数据子集、同一评测脚本和生成参数跑 BF16 baseline，并记录 `Correct / Total`、accuracy、命令和环境。OSCAR-vLLM 的主要验收阈值仍以 OSCAR-SGLang `0.8741` 为参照，同时报告相对 vLLM BF16 baseline 的 delta。

GSM8K 是本任务唯一必做精度验收项。IFEval、LiveCodeBench v6、MultiPL-E 和 Overall 仅作为 OSCAR-SGLang 参考背景，不作为本任务必过项。若时间允许，可作为扩展实验补充。

已知 OSCAR SGLang 在同个数据集测得的精度如下：

| Benchmark | Correct / Total | OSCAR Accuracy | Baseline Accuracy | Delta |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 1153 / 1319 | 0.8741 | 0.8893 | -0.0152 |
| IFEval | 433 / 541 | 0.8004 | 0.8189 | -0.0185 |
| LiveCodeBench v6 | 66 / 175 | 0.3771 | 0.3657 | +0.0114 |
| MultiPL-E | 133 / 325 | 0.4092 | 0.4185 | -0.0092 |
| Overall | 1785 / 2360 | 0.7564 | 0.7695 | -0.0131 |

GSM8K 精度硬阈值：

- OSCAR-vLLM accuracy 应不低于 OSCAR-SGLang 参考值 `0.8741` 的 1.5 个百分点，即 `accuracy >= 0.8591`。
- 对于 `Total=1319` 的 GSM8K 子集，等价硬阈值为 `Correct >= 1134 / 1319`。
- 若低于该阈值，视为未通过精度验收，除非 Shawn 明确接受误差分析结果。
- 若未通过，必须提供误差分析，包括 prompt/template、生成参数、OSCAR 配置、rotation/clipping 加载、KV 路径证明和样本级 diff。

最小测试要求：

- rotation 文件加载测试：确认 K/V rotation shape、dtype、device 与 Qwen3-4B attention head_dim 匹配。
- INT2 pack/unpack roundtrip 测试：随机张量经过 clip、quantize、pack、unpack、dequant 后 shape 正确，误差在预期范围内。
- mixed KV layout 测试：给定 seq_len，验证 prefix 64、recent 256、history OSCAR INT2 的 token 分区正确，边界 case 至少包括 `seq_len < 64`、`64 < seq_len < 320`、`seq_len > 320`。
- 小规模 attention 对照测试：短序列下 OSCAR path 与 reference PyTorch path 输出接近，确认反量化和逆旋转位置正确。
- 端到端 smoke test：指定模型至少生成若干条样本，日志证明进入 OSCAR path。

实验与报告要求：

- 实验环境、GPU 分配、Docker 使用、长实验进度输出等要求遵循本目录 `AGENTS.md` 和 `AGENTS_misc.md`。
- 本任务直接使用上述 Docker 镜像作为 OSCAR-vLLM 集成与实验基座，除非 Shawn 明确指定其他镜像。
- 本任务的实验报告必须记录：Docker 镜像名称、Python 版本、关键 Python 包版本、CUDA_VISIBLE_DEVICES、`nvidia-smi` 输出、vLLM v0.25.0、OSCAR main HEAD commit、模型路径、数据集路径和完整运行命令。
- 所有报告文档必须使用中文。所有数据必须基于实际落地结果，不得凭空捏造。
