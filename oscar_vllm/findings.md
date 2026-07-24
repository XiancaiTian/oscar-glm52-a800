# 事实与发现

## 2026-07-17 初始核对

- 工作目录 `/data/ssd1/txc/oscar_vllm` 实际解析后的 `pwd` 为 `/data1/txc/oscar_vllm`；两者指向同一现有目录。
- 初始目录只有规范、任务、rotation 文件和无关的 `setup_vpn` 工具，没有 vLLM/OSCAR 源码，也不是 Git 仓库。
- 模型目录 `/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507` 存在。
- 数据集目录 `/data/ssd1/txc/vllm_turbo_baseline_acc` 存在。
- 两个指定 rotation 文件均存在：
  - `qwe3-4b-instruct-2507-rotations/k_rotation_qqt_r_h_pbr.pt`
  - `qwe3-4b-instruct-2507-rotations/v_rotation_sst_r_h_pbr.pt`
- GPU 编号未在现有指令中指定，因此在 Shawn 确认之前不能开始 GPU 实验。
- OSCAR `main` HEAD 已通过远端引用固定为 `41ebcdba3db5f0ce1339c3727caea80df575d437`。
- vLLM `v0.25.0` tag 已通过远端引用固定为 `702f4814fe54fabff350d43cb753ae3e47c0c276`。
- vLLM PR #46774 head 已通过远端引用固定为 `69d19ab12ced4c561666db0a2863881825d489df`，仅作为参考实现。
- 当前用户直接访问 Docker socket 被拒绝；需使用已授权的 sudo 访问 Docker daemon。
- 指定镜像已存在，本机 digest 为 `sha256:fc56161ee42a011aeee78b65d0a81b6683c7d04402fd40503d14d4d6c98f07cb`。
- vLLM 源码已在 `vllm/` 按 tag 浅克隆并处于 detached HEAD；OSCAR 固定源码已在 `oscar_reference/` 按 HEAD 浅克隆。
- OSCAR 仓库包含 `sglang-research` 完整实现；核心文件包括 `QuantKernel/oscar_rotation_clip_int2_kv.py`、attention backend、KV memory pool、Qwen3 模型入口及两组注册测试。
- PR #46774 head 是合并 upstream main 的 merge commit；直接与 v0.25.0 比较会混入大量无关变化，需隔离 OSCAR 专属提交或只抽取关键文件。
- 目标模型配置为 36 层、32 个 query heads、8 个 KV heads、`head_dim=128`、`torch_dtype=bfloat16`。
- OSCAR SGLang 默认 mixed window 是 32/128，但本任务明确要求覆盖为 prefix 64、recent 256；rotation/clipping 环境入口均在参考实现中存在。
- 参考 `kv_quant_kernels.py` 的独立 `dequantize_kv_int2_triton` 会分配完整 model-dtype 输出，仅适合作为测试 oracle/通用 helper；本任务 serving 验收不能使用“整段反量化后 attention”的方式。
- 数据集仓库当前有大量预先存在的 tracked 删除/修改和 untracked 新 suite；必须保留这些用户侧状态。实际 GSM8K 文件位于 `accuracy_suites/model_agnostic_accuracy_official_v4/samples/gsm8k.jsonl`。
- PR #46774 的 OSCAR 专属提交为 `e156a30bf528fe03645ae2042b8f6b2d4717f98a`，共新增/修改 15 个文件；可复用 calibrated rotation、Triton INT2 store 和 fused decode 骨架。
- 该 PR 明确用首尾各 2 个 BF16 layer 代替 OSCAR 的 BF16 token windows，不满足任务的 prefix 64/recent 256 mixed KV 要求，不能原样作为交付。
- vLLM v0.25.0 已包含 TurboQuant 自定义 attention backend、packed cache shape 和 `TQFullAttentionSpec`，为 OSCAR 自定义 cache spec/shape 提供当前版本内的兼容模式。
- 满足 mixed KV 需要双 tier 数据布局：有限 BF16 prefix/recent arena、INT2 history arena，以及 fused attention 中合并两个 tier 的 logits/LSE；不能为所有 token 同时保留 BF16 副本。
- 当前 GSM8K 文件实际为 1319 行，SHA256 为 `3a3e1fbcc2cdb1c2c5835b6c31d39c72df388f3a747a2b509c7bc84ee0bf78f8`。
- 当前评测配置为 chat completions，`n=1`、`seed=42`、`temperature=0.0`、`top_p=1.0`，GSM8K 超时 300 秒。
- 指定镜像实际环境：Python 3.12.13、PyTorch 2.11.0+cu130、Triton 3.6.0、vLLM 0.25.0，镜像构建 commit 元数据为 `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`。
- K/V rotation 均为含 `format_version/layers/objective/source_grouping` 的字典，含 36 层；首层 rotation shape `(128, 128)`、dtype FP32、加载 device CPU，与目标模型层数/head_dim 匹配。
- OSCAR INT2 每个 KV head/token 的 packed K+V 为 80 bytes（K/V 各 32 data + 8 bytes FP32 `(scale, zero_point)` metadata）；BF16 K+V 为 512 bytes。
- `seq_len <= 320` 时没有 INT2 history；`seq_len > 320` 时 history token 数为 `seq_len - 320`，三个 tier 无重叠且总数等于 seq_len。
- vLLM 的 `page_size_padded` 会分配完整 padded bytes，同时把 backend tensor 视图限制在 real page，并用 block stride 跳过页尾；适合把页尾用作有限 BF16 arena，而不破坏现有 INT2 slot mapping。
- 对 `max_model_len=8192`、block size 16、8 KV heads：OSCAR padded page 为 14,336 bytes（10,240 INT2 + 4,096 BF16 arena），纯 BF16 page 为 65,536 bytes。
- 派生镜像 `oscar-vllm:v0.25.0-dev` 当前 manifest list digest 为 `sha256:a0f8175c2ddb4df68157a6c250e6cab5a553e3f17daf786c5835247064cf17eb`；它以指定 digest 为 base，覆盖 Python 源码并保留官方编译扩展。
- BF16 prefix/recent arena 是每层 cache tensor 页尾组成的固定全局区域，而非请求逻辑页的一部分。原型强制 `max_num_seqs=1` 且禁用 prefix cache，因此可按固定物理页寻址；INT2 token slot 仍按请求 `block_table` 寻址。若错误地让 BF16 arena 经 `block_table` 映射，短请求只分配少量逻辑页却需保存最多 320 个 BF16 token，会越界访问 block table。代码已增加 `num_blocks * hp_slots_per_block` 的运行时容量检查。
- 在无 GPU 容器中显式注入 vLLM `CpuPlatform` 后，真实 Qwen3 模型的 `EngineArgs.create_engine_config()` 配置层测试通过：`cache_dtype=oscar_int2`、`max_num_seqs=1`、`enable_chunked_prefill=False`、模型默认 `max_model_len=262144`；该结果只证明模型解析和 OSCAR 参数校验，不代表 CUDA serving 已通过。
- OSCAR 固定参考测试把 clipping threshold 定义为 `sort(abs(x))[floor(clip_ratio * head_dim)]`；当前 Triton store/demotion 的整数索引计算与此完全一致。其余 kernel 数值仍需获准 GPU 后编译和对照验证。
- 基础镜像与派生镜像的 vLLM 包内均有 16 个原生 `.so`，路径和逐文件 SHA256 完全一致。该构建使用 stable-libtorch 扩展名（例如 `_C_stable_libtorch.abi3.so`），基础镜像本身也不能 `import vllm._C`，因此该模块名不能作为扩展保留性检查。
- 首次 Triton 离线编译发现 V 逆旋转 kernel 的 `result[0, :]` 不受 Triton 3.6 支持；改为 Triton 列 reduction 后，逆旋转、INT2 store、BF16 store、demotion、mixed decode stage1 五个 kernel 均成功为 SM90 生成 cubin，逆旋转和 mixed decode 也成功为 SM80 生成 cubin。该结果证明 lowering 可行，不证明实际 launch 或数值正确。
- `max_model_len=8192`、block size 16、8 KV heads 时，采用参考 FP32 metadata 后代码计算的每层 BF16 逻辑占用为 33,554,432 bytes，OSCAR mixed 有效数据为 6,348,800 bytes（理论 5.2852x），按 padded page 实际分配为 7,340,032 bytes（分配口径 4.5714x）；后者仍需与 GPU 启动后的真实 cache block 数结合验证。
- OSCAR 参考实现明确使用 `tensor @ R` 和 `result @ R_v.T`，与当前方向一致。真实 36 层 calibration 的最大逐元素 `R^T R-I` 误差为 K `5.364418e-7`、V `4.768372e-7`，最大 Frobenius 误差为 K `2.334956e-6`、V `2.406903e-6`。
- OSCAR 固定参考提交的 `SGLANG_MIXED_KV_SCALE_DTYPE` 默认值实际为 `float32`，量化 metadata 为 `(scale, zero_point)`，反量化公式为 `(q - zero_point) * scale`；修改前由参考 vLLM PR 演化的原型使用 FP16 `(scale, vmin)`，不是字节级相同语义。
- 在派生镜像 PyTorch 2.11.0 内以 seed `20260717` 比较 10,000 个 128 维向量：标准正态输入下，SGLang FP32 方案 MAE 为 `0.4276402`，当前 FP16 方案 MAE 为 `0.4276501`，二者输出 MAE 为 `0.0012276`、最大差 `2.62895`；窄分布偏置输入下两者 MAE 分别为 `0.0042719` 和 `0.0062889`。为优先满足与 OSCAR SGLang 的数值/精度对齐，后续实现改用 FP32 `(scale, zero_point)`；其 metadata 容量代价将计入 mixed 压缩率。
- FP32 metadata 的四字节 store/load helper 已随 INT2 store、mixed decode stage1 和 full-dequant oracle 在 Triton 3.6 下分别通过 SM80/SM90 离线编译；该证据仍只覆盖 lowering，不替代目标 GPU launch 数值测试。
- 镜像实际默认 `tq_max_kv_splits_for_cuda_graph=32`。按真实 runtime specialization 重编译后，SM90 的 INT2 store、BF16 store、demotion、mixed decode stage1、复用 stage2、V inverse rotation cubin 为 47,128、7,840、47,632、70,480、12,856、72,872 bytes，SM80 为 46,048、7,272、46,560、69,424、12,272、71,776 bytes；stage1 使用实际 `NUM_KV_SPLITS=32,BLOCK_KV=4,num_warps=1,num_stages=1`。
- 空 decode split 不会导致未初始化读取：复用的 vLLM stage2 依据 `seq_len` 和相同 `NUM_KV_SPLITS` 重新计算每个 split 的范围，只加载 `split_end > split_start` 的槽位。
- Triton CPU 解释器实际运行 INT2 store/full-dequant：seed 17、4 token、2 KV heads、head dim 128 时相对 PyTorch oracle 的最大绝对差 K `0.0009506`、V `0.0007792`，平均差 K `0.0001479`、V `0.0001594`。
- Triton CPU 解释器实际运行 321-token mixed 边界（demotion、mixed stage1、32-split stage2、identity V inverse）：seed 23、Hk=1、Hq=2、head dim 128，输出全为 finite，相对直接 PyTorch attention 最大绝对差 `0.0029296875`、平均差 `0.0006603971`。解释器证据不替代 CUDA launch。

## 待确认事实

- Triton kernel 在目标 GPU 上的实际编译结果和与 PyTorch reference 的数值误差。
