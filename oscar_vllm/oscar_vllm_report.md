# OSCAR-vLLM v0.25.0 集成与验证报告

## 1. 当前状态

截至 2026-07-20，已完成固定版本与数据核对、OSCAR mixed KV 原型实现、CPU 单元测试、vLLM 配置层验证、Triton CPU 解释器数值对照、离线编译、6 项 CUDA 单元测试、真实模型端到端 smoke、同配置 BF16/OSCAR cache 实测，以及完整 BF16/OSCAR GSM8K 精度评测。物理 GPU 1 上首次测试发现的 V 逆旋转错误已修正，并通过自包含镜像复验。BF16 baseline 为 `1161/1319`、accuracy `0.8802122820`；OSCAR 为 `1157/1319`、accuracy `0.8771796816`，通过 `Correct >= 1134/1319` 硬阈值。

本文只记录已经实际执行并获得结果的项目；未执行项不填入推测数据。

## 2. 固定版本与路径

| 项目 | 实际值 |
| --- | --- |
| vLLM 基线 | tag `v0.25.0`，commit `702f4814fe54fabff350d43cb753ae3e47c0c276` |
| 开发分支 | `oscar-vllm-v0.25.0` |
| OSCAR 参考实现 | `main` HEAD `41ebcdba3db5f0ce1339c3727caea80df575d437` |
| vLLM 参考 PR | PR #46774 head `69d19ab12ced4c561666db0a2863881825d489df`，OSCAR 专属 commit `e156a30bf528fe03645ae2042b8f6b2d4717f98a` |
| 固定基础镜像 | `docker.m.daocloud.io/vllm/vllm-openai:v0.25.0` |
| 基础镜像 digest | `sha256:fc56161ee42a011aeee78b65d0a81b6683c7d04402fd40503d14d4d6c98f07cb` |
| 当前派生镜像 | `oscar-vllm:v0.25.0-dev` |
| 派生镜像 digest | `sha256:0356873d691c97f72ccc08eda4ca7aecb9b297206763fab8ea3ad01b3b4f1e7e` |
| 模型 | `/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507` |
| 数据集仓库 | `/data/ssd1/txc/vllm_turbo_baseline_acc` |
| GSM8K 样本 | `accuracy_suites/model_agnostic_accuracy_official_v4/samples/gsm8k.jsonl` |
| GSM8K 样本数 | 1319 |
| GSM8K SHA256 | `3a3e1fbcc2cdb1c2c5835b6c31d39c72df388f3a747a2b509c7bc84ee0bf78f8` |
| K rotation | `/data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations/k_rotation_qqt_r_h_pbr.pt` |
| V rotation | `/data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations/v_rotation_sst_r_h_pbr.pt` |

数据集仓库在任务开始前已有大量 tracked 修改/删除和 untracked 文件，本任务只读使用实际 GSM8K suite，没有恢复或覆盖这些既有改动。

## 3. 验证环境

### 3.1 无 GPU 环境

本阶段所有容器均显式设置 `CUDA_VISIBLE_DEVICES=''`，没有挂载 GPU。

| 项目 | 实际值 |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| vLLM | 0.25.0 |
| ruff | 0.14.0 |
| CUDA_VISIBLE_DEVICES | 空字符串 |
| nvidia-smi | 本阶段未执行；按项目规范须在指定 GPU 后检查 |

基础镜像和派生镜像的 vLLM 包内各有 16 个原生 `.so`，逐文件路径和 SHA256 完全一致，确认源码覆盖没有丢失基础镜像的 stable-libtorch 编译扩展。

### 3.2 GPU 1 环境

GPU 实验只挂载 Shawn 指定的物理 GPU 1。Docker 的 device 过滤将其映射为容器内 `cuda:0`，因此容器内设置为 `CUDA_VISIBLE_DEVICES=0`；未查询或挂载其他 GPU。

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:v0.25.0-dev` |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| vLLM | 0.25.0 |
| 物理 GPU | GPU 1，NVIDIA B200 |
| CUDA_VISIBLE_DEVICES | `0`（容器只挂载物理 GPU 1） |
| Driver / nvidia-smi CUDA | 595.71.05 / 13.2 |

UTC `2026-07-19 11:04:05`、首次 GPU 单元测试前的完整 `nvidia-smi` 输出如下：

```text
NVIDIA-SMI 595.71.05              Driver Version: 595.71.05      CUDA Version: 13.2
GPU  Name              Bus-Id        Disp.A  Memory-Usage       GPU-Util  Compute M.
  1  NVIDIA B200       00000000:3D:00.0 Off  0MiB / 183359MiB       0%      Default
Processes: No running processes found
```

### 3.3 GPU 0 BF16 baseline 环境

Shawn 于 2026-07-20 授权使用所有空闲 GPU。GPU 0-7 经两次间隔 1 分钟检查均为空闲；为保持单卡实验可比，BF16 baseline 只挂载物理 GPU 0，并在容器内设置 `CUDA_VISIBLE_DEVICES=0`。

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `docker.m.daocloud.io/vllm/vllm-openai:v0.25.0` |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| vLLM | 0.25.0 |
| 物理 GPU | GPU 0，NVIDIA B200，Bus-Id `00000000:17:00.0` |
| CUDA_VISIBLE_DEVICES | `0`（容器只挂载物理 GPU 0） |
| Driver / nvidia-smi CUDA | 595.71.05 / 13.2 |

UTC `2026-07-20 01:29:07` 的 `nvidia-smi` 显示 GPU 0 为 `0 MiB / 183359 MiB`、利用率 0%、无运行进程；当时 GPU 0-7 均为空闲。完整原始输出保存于 `artifacts/gsm8k/20260720/nvidia_smi_before_bf16.txt`。

### 3.4 GPU 0-7 BF16 分片评测环境

两轮单卡评测受到外部生命周期因素中断后，按 `AGENTS_misc.md` 允许的精度多卡方式使用 GPU 0-7。每个服务容器只挂载一张物理卡，该卡在容器内均映射为 `cuda:0` 并设置 `CUDA_VISIBLE_DEVICES=0`；各卡配置完全相同。

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `docker.m.daocloud.io/vllm/vllm-openai:v0.25.0` |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| vLLM | 0.25.0 |
| 物理 GPU | GPU 0-7，8 张 NVIDIA B200 |
| CUDA_VISIBLE_DEVICES | 每个服务容器均为 `0` |
| Driver / nvidia-smi CUDA | 595.71.05 / 13.2 |

GPU 0-7 在 UTC `2026-07-20 02:37:52` 和 `02:38:58` 连续两次均为 0 MiB、利用率 0%、无 compute process。启动前的完整 `nvidia-smi` 输出保存于 `artifacts/gsm8k/20260720/nvidia_smi_before_bf16_8way.txt`。

### 3.5 GPU 0-7 OSCAR 分片评测环境

OSCAR 评测继续使用 GPU 0-7，每个服务容器只挂载一张物理卡，容器内均为 `CUDA_VISIBLE_DEVICES=0`。与第 3.4 节 BF16 环境的差异仅为派生镜像和 `oscar_int2` 配置。

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:v0.25.0-dev` |
| 镜像 digest | `sha256:0356873d691c97f72ccc08eda4ca7aecb9b297206763fab8ea3ad01b3b4f1e7e` |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| vLLM | 0.25.0 |
| 物理 GPU | GPU 0-7，8 张 NVIDIA B200 |
| CUDA_VISIBLE_DEVICES | 每个服务容器均为 `0` |
| Driver / nvidia-smi CUDA | 595.71.05 / 13.2 |

BF16 容器回收后，GPU 0-7 在 UTC `02:57:10`、`02:59:31` 连续两次均为空闲；为避开已观察到的整点外部作业，还在 UTC `03:01:27` 额外确认全部为 0 MiB、利用率 0%、无 compute process。启动前完整输出保存于 `artifacts/gsm8k/20260720/nvidia_smi_before_oscar_8way.txt`。

## 4. 已实现路径

### 4.1 配置与约束

- 新增 `oscar_int2` KV cache dtype、backend registry 和 CUDA backend 选择入口。
- 登记 7 个 `VLLM_OSCAR_*` 环境变量。
- 固定并校验 K/V rotation、K clip 0.96、V clip 0.92、prefix 64、recent 256、INT2 group size 128。
- 原型限制 `max_num_seqs=1`，禁用 chunked prefill、prefix cache、speculative decoding、KV skip layer 和 CUDA graph。

### 4.2 Cache allocation

INT2 cache slot 为每个 KV head/token 80 bytes，其中 K/V 各含 32 bytes INT2 数据和 8 bytes FP32 `(scale, zero_point)` metadata。量化与反量化公式分别为 `q=round(x/scale+zero_point)` 和 `x_hat=(q-zero_point)*scale`，与固定 OSCAR SGLang 提交的默认语义一致。使用 `TQFullAttentionSpec.page_size_padded` 在每个物理 cache page 的 INT2 区域后保留固定 BF16 arena 分片。

单请求 BF16 arena 使用固定物理页，INT2 token slot 使用请求 `block_table`。该设计依赖第 4.1 节列出的单请求和禁 prefix cache 约束；代码包含 BF16 arena 容量检查。

### 4.3 KV write 与 attention read

- Prefill 对当前 BF16 K/V 做标准 attention；随后加载逐层 calibration，对 K/V 旋转。
- Prefix 和 recent 写入页尾 BF16 arena。
- History 通过 Triton 完成 clipping、asymmetric INT2 quantize、pack 和 scatter。
- Decode 时先 demote 离开 recent window 的 token，再写入新的 recent token。
- Mixed decode stage1 在一个 online softmax 中直接读取 BF16 prefix/recent 和 INT2 history，并在读取时反量化；serving 路径不构造整段 BF16 临时 cache。
- Decode value weighted sum 通过独立 Triton kernel 乘 `R_v^T` 完成逆旋转。

运行时已加入三类一次性日志：calibration 路径、Triton KV write 路径、Triton mixed attention read 路径。第 5.7 节的真实 serving 日志已确认三类日志全部出现。

### 4.4 关键代码入口

| 功能 | vLLM 文件 |
| --- | --- |
| 配置、slot/page 几何 | `vllm/model_executor/layers/quantization/oscar/config.py` |
| rotation 加载与逐层选择 | `vllm/model_executor/layers/quantization/oscar/rotation.py` |
| cache spec/allocation 接入 | `vllm/model_executor/layers/attention/attention.py`、`vllm/platforms/interface.py` |
| 配置开关与 backend 选择 | `vllm/envs.py`、`vllm/engine/arg_utils.py`、`vllm/platforms/cuda.py`、`vllm/v1/attention/backends/registry.py` |
| mixed serving 调度 | `vllm/v1/attention/backends/oscar_attn.py` |
| INT2 write/quantize | `vllm/v1/attention/ops/triton_oscar_store.py` |
| BF16 write 与 recent demotion | `vllm/v1/attention/ops/triton_oscar_mixed_store.py` |
| mixed decode/dequantize 与逆旋转 | `vllm/v1/attention/ops/triton_oscar_decode.py` |

## 5. 已完成测试

### 5.1 CPU 单元测试

实际命令：

```bash
sudo docker run --rm \
  -e CUDA_VISIBLE_DEVICES='' \
  -e OSCAR_TEST_ROTATION_DIR=/data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations \
  -v /data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations:/data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations:ro \
  --entrypoint /usr/bin/python3 \
  oscar-vllm:v0.25.0-dev \
  -m pytest -q /workspace/tests/quantization/test_oscar_cpu.py
```

自包含镜像构建后的实际结果为 `8 passed, 2 warnings, 12 subtests passed in 1.93s`；全部精度实验结束后的最终复测为 `8 passed, 2 warnings, 12 subtests passed in 1.47s`。

覆盖内容包括任务默认值、slot/page 几何、mixed token 边界、mixed bytes、非法布局、缺失 calibration 拒绝、缺层拒绝，以及真实 Qwen3 K/V rotation 的 36 层 shape、dtype、CPU device 和 `R^T R ≈ I` 检查。36 层实际最大逐元素正交误差：K 为 `5.364418e-7`，V 为 `4.768372e-7`；最大 Frobenius 误差：K 为 `2.334956e-6`，V 为 `2.406903e-6`。

GPU 获准前曾在 `CUDA_VISIBLE_DEVICES=''` 下收集 GPU 测试模块 `tests/quantization/test_oscar.py`，结果为 `6 skipped, 2 warnings in 0.03s`，证明模块导入和 collection 正常；其后的真实 CUDA 结果见第 5.6 节。

### 5.2 配置层验证

在无 GPU 容器中显式注入 vLLM `CpuPlatform` 后，使用真实模型路径执行 `EngineArgs.create_engine_config()`，实际输出为：

```text
CONFIG_OK oscar_int2 1 False 262144
```

日志同时确认模型架构解析为 `Qwen3ForCausalLM`，OSCAR 配置为 prefix 64 BF16、recent 256 BF16、history rotation+clip+INT2、K/V clip 0.96/0.92。此测试只验证配置生成，不替代 CUDA serving。

### 5.3 静态与离线编译

- 全部 OSCAR 改动文件的定向 ruff 检查通过。
- Python `py_compile` 和 `git diff --check` 通过。
- 最终审计再次对全部 staged Python 文件运行 ruff 和 `py_compile`，并运行 `git diff --check`、`git diff --cached --check`，全部通过。
- 镜像内实际默认 split 数为 32。按实际 runtime 常量和 launch options 离线编译，INT2 store、BF16 store、demotion、mixed decode stage1、复用的 decode stage2、V inverse rotation 在 SM90 生成的 cubin 分别为 47,128、7,840、47,632、70,480、12,856、72,872 bytes。
- 同一组实际 runtime specialization 在 SM80 生成的 cubin 分别为 46,048、7,272、46,560、69,424、12,272、71,776 bytes。
- 仅用于测试的 full-dequant oracle 也在 SM80/SM90 生成 cubin；它不进入 serving 路径。

首次离线编译发现 V 逆旋转 kernel 使用了 Triton 3.6 不支持的常量行索引；改为 Triton 列 reduction 后，SM80/SM90 均编译通过。

### 5.4 参考量化语义核对

固定 OSCAR 提交的 `SGLANG_MIXED_KV_SCALE_DTYPE` 默认值为 `float32`，保存 `(scale, zero_point)`。在派生镜像 PyTorch 2.11.0 中以 seed `20260717` 比较 10,000 个 128 维向量：标准正态输入下，SGLang FP32 方案 MAE 为 `0.4276402`，原 FP16 `(scale, vmin)` 方案 MAE 为 `0.4276501`，两种输出间 MAE 为 `0.0012276`、最大差为 `2.62895`；窄分布偏置输入下，两者 MAE 分别为 `0.0042719` 和 `0.0062889`。因此当前实现采用参考 FP32 metadata，而非保留参考 vLLM PR 的 FP16 metadata 优化。

### 5.5 Triton CPU 解释器数值对照

在同一派生镜像中设置 `CUDA_VISIBLE_DEVICES=''` 和 `TRITON_INTERPRET=1`，使用 CPU tensor 实际执行 kernel：

- seed 17、4 token、2 KV heads、head dim 128 的 INT2 store/full-dequant 对照中，相对 PyTorch oracle 的最大绝对差为 K `0.0009506`、V `0.0007792`，平均绝对差为 K `0.0001479`、V `0.0001594`；差异包含 full-dequant oracle 输出落到 FP16 的舍入。
- seed 23、321 token、1 KV head、2 query heads、head dim 128 的 mixed 边界对照实际执行了 recent demotion、BF16/INT2 mixed stage1、32-split stage2 和 identity V inverse。输出全部为 finite，相对直接 PyTorch attention 的最大绝对差为 `0.0029296875`，平均绝对差为 `0.0006603971`。

解释器结果验证了 kernel 逻辑和 metadata 字节 pack/unpack，但不代表 CUDA kernel 的真实 launch 与数值；后续第 5.6 节记录了独立的真实 CUDA 验证。性能尚未单独测量。

### 5.6 GPU 单元测试与修正

在第 3.2 节所列环境中实际运行：

```bash
sudo docker run --rm --gpus 'device=1' \
  -e CUDA_VISIBLE_DEVICES=0 \
  --entrypoint /usr/bin/python3 \
  oscar-vllm:v0.25.0-dev \
  -m pytest -q /workspace/tests/quantization/test_oscar.py
```

实际结果为 `4 passed, 2 failed, 16 warnings in 8.47s`。失败项及最大绝对误差如下：

| 测试参数 | 最大绝对误差 | 结论 |
| --- | ---: | --- |
| head dim 128、Hq 8、Hk 2 | 1.7315640 | decode attention 与反量化 cache 的 PyTorch 参考值不一致 |
| head dim 64、Hq 4、Hk 4 | 1.6989495 | decode attention 与反量化 cache 的 PyTorch 参考值不一致 |

该误差远大于测试容差 `atol=rtol=5e-3`。最小 CUDA 对照进一步得到：不执行 V 逆旋转时，decode 输出相对参考值的最大绝对误差仅为 `0.0004594`；QR 产生的非连续 rotation 矩阵 stride 为 `(1, 64)`，函数内临时 contiguous 副本的异步生命周期使逆旋转最大误差达到 `1.1904850`，而由调用方持有 contiguous 副本时为 `0.0022181`。

修正方式是让 V 逆旋转 Triton kernel 显式接收 rotation 的行、列 stride，并直接读取调用方持有的矩阵，不再创建函数内临时 contiguous 张量。将修正版单文件只读挂载进同一派生镜像后，两个原失败用例实际结果为 `2 passed, 16 warnings in 4.83s`，完整 GPU 模块实际结果为：

```text
6 passed, 16 warnings in 7.66s
```

随后重建派生镜像，得到 manifest list digest `sha256:0356873d691c97f72ccc08eda4ca7aecb9b297206763fab8ea3ad01b3b4f1e7e`。在不挂载任何源码的条件下重跑相同命令，实际结果为：

```text
6 passed, 16 warnings in 7.19s
```

因此当前自包含镜像已通过 INT2 roundtrip、mixed BF16/INT2 store/demotion/decode、两组 attention 对照和非连续 V rotation 输入的真实 CUDA launch 验证。

全部精度实验结束后，GPU 0-7 于 UTC `03:30:07` 和 `03:32:13` 连续两次确认为空闲；只挂载物理 GPU 0、且不挂载任何源码，再次运行同一测试模块，最终结果为：

```text
6 passed, 16 warnings in 37.10s
```

### 5.7 真实模型端到端 smoke

使用物理 GPU 1 启动服务的实际命令如下；容器只挂载该卡，物理 GPU 1 在容器内映射为 `cuda:0`：

```bash
sudo docker run --rm --name oscar-vllm-smoke \
  --gpus 'device=1' --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_OSCAR_K_ROTATION_PATH=/rot/k_rotation_qqt_r_h_pbr.pt \
  -e VLLM_OSCAR_V_ROTATION_PATH=/rot/v_rotation_sst_r_h_pbr.pt \
  -e VLLM_OSCAR_K_CLIP_RATIO=0.96 \
  -e VLLM_OSCAR_V_CLIP_RATIO=0.92 \
  -e VLLM_OSCAR_GROUP_SIZE=128 \
  -e VLLM_OSCAR_PREFIX_TOKENS=64 \
  -e VLLM_OSCAR_RECENT_TOKENS=256 \
  -v /data/ssd1/checkpoints/Qwen3-4B-Instruct-2507:/model:ro \
  -v /data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations:/rot:ro \
  -p 127.0.0.1:8101:8000 \
  oscar-vllm:v0.25.0-dev /model \
  --dtype bfloat16 --kv-cache-dtype oscar_int2 \
  --max-model-len 8192 --max-num-seqs 1 \
  --no-enable-chunked-prefill --no-enable-prefix-caching \
  --enforce-eager --gpu-memory-utilization 0.1 --port 8000
```

启动日志实际确认：Qwen3ForCausalLM、BF16、`oscar_int2`、prefix 64、recent 256、K/V clip 0.96/0.92、OSCAR attention backend，以及 K/V 两个 rotation checkpoint 各加载 36 层。日志同时出现：

```text
OSCAR Triton KV write active: prefix=64 BF16, recent=256 BF16,
history=clip+INT2 (K 0.96, V 0.92)
OSCAR Triton mixed attention read active: fused BF16/INT2 read,
dequantization, online softmax, and V inverse rotation
```

顺序请求结果如下；三条 prompt 均超过 320 token，实际覆盖 history INT2 区间：

| 请求 | Prompt tokens | Completion tokens | HTTP | 输出 |
| --- | ---: | ---: | ---: | --- |
| 1 | 743 | 2 | 200 | `5` |
| 2 | 703 | 2 | 200 | `Paris` |
| 3 | 782 | 3 | 200 | `10` |

服务日志还记录了 `_demote_hp_kernel` 和 `_store_hp_kernel` 在 inference 中的首次 JIT，进一步证明 recent ring 的写入和 demotion 被真实请求触发。完整日志位于 `artifacts/smoke/oscar_server.log`，响应位于 `artifacts/smoke/response_1.json` 至 `response_3.json`。

### 5.8 GSM8K BF16 无效运行记录

BF16 首轮使用 frozen official v4 suite、`--benchmarks GSM8K --concurrency 1`，生成参数为 temperature 0、top_p 1、seed 42、n 1。评测运行到样本 386 后，前台服务所在的外层工具会话达到 30 分钟上限，于 UTC `2026-07-20 01:59:28` 向容器发送 `SIGTERM`；服务日志记录了正常 shutdown 流程，没有模型执行错误或 CUDA 错误。

该轮最终状态为 386 条 `scored`、933 条 `request_failed`；其中 919 条为 connection reset，14 条为 remote closed。仅已评分子集的 accuracy 为 `0.8963731`，不能代表 1319 条完整 baseline，因此不用于后续 OSCAR 精度比较。无效轮结果和日志分别保存在 `artifacts/gsm8k/20260720/bf16_invalid_server_timeout/`、`bf16_invalid_server_timeout_runner.log` 和 `bf16_invalid_server_timeout_server.log`。

为避免再次受到外层工具会话时限影响，完整重跑把服务和 runner 都改为 detached Docker 容器，结束后再单独抓取日志。外部 8-GPU 任务结束后，GPU 0-7 于 UTC `02:21` 和 `02:23` 连续两次确认为空闲；重跑固定使用物理 GPU 0，服务健康检查返回 HTTP 200 后，runner 于 UTC `02:28:02` 启动。启动前完整 `nvidia-smi` 保存于 `artifacts/gsm8k/20260720/nvidia_smi_before_bf16_rerun.txt`。

第二轮在 15 个请求成功后，服务于 UTC `02:29:20` 收到外部 `SIGTERM`，容器最终退出码为 137，Docker 状态明确为 `OOMKilled=false`；服务日志仍无模型执行或 CUDA 错误。同一时刻 GPU 0-7 出现另一组外部 PID `2336175` 至 `2336182`，属于其他用户随后启动的 8-GPU 任务。该轮最终为 15 条 `scored`、1304 条 `request_failed`，其中 14 条评分正确；该局部结果同样不用于 baseline。

第二轮无效产物保存在 `artifacts/gsm8k/20260720/bf16_invalid_external_preemption/`、`bf16_invalid_external_preemption_runner.log` 和 `bf16_invalid_external_preemption_server.log`；抢占后的完整 `nvidia-smi` 保存于 `nvidia_smi_after_bf16_external_preemption.txt`。

GPU 0-7 在 UTC `02:37:52` 和 `02:38:58` 再次连续两次确认为空闲。为缩短暴露于外部批任务间隔的时间，第三轮按 `AGENTS_misc.md` 允许的精度多卡方式，将 official GSM8K manifest 按原始索引模 8 分为 8 个互斥 shard，样本数为 `165×7 + 164`；合计和唯一 ID 都是 1319，ID 集与原始 GSM8K 完全一致。每张卡仍运行相同 BF16 vLLM 配置，每个 runner 仍使用 official 脚本、`concurrency=1` 和同一逐请求生成参数；完整有效结果见第 5.9 节。启动前完整环境记录位于 `nvidia_smi_before_bf16_8way.txt`。

### 5.9 GSM8K vLLM BF16 baseline

分片只改变独立样本分派，不改变模型、服务配置、官方评测脚本或逐请求参数。8 个 manifest 按 official GSM8K 原始索引模 8 生成，样本数为 `165×7 + 164`；合并前后均验证为 1319 个唯一 ID，且 ID 集与原始 GSM8K 完全一致。每个请求由 frozen official v4 runner 发出，temperature 0、top_p 1、seed 42、n 1，每个 runner 的 `concurrency=1`。

服务实际启动命令如下，其中 `gpu` 为 0 至 7、`port=8100+gpu`：

```bash
sudo docker run -d --name "oscar-bf16-$gpu" \
  --gpus "device=$gpu" --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
  -v /data/ssd1/checkpoints/Qwen3-4B-Instruct-2507:/model:ro \
  -p "127.0.0.1:$port:8000" \
  docker.m.daocloud.io/vllm/vllm-openai:v0.25.0 /model \
  --served-model-name qwen3-4b-instruct-2507 \
  --dtype bfloat16 --kv-cache-dtype auto \
  --max-model-len 8192 --max-num-seqs 1 \
  --no-enable-chunked-prefill --no-enable-prefix-caching \
  --enforce-eager --gpu-memory-utilization 0.1 --port 8000
```

每个 shard 的 runner 实际命令如下，其中 `shard` 为 0 至 7：

```bash
sudo docker run -d --name "oscar-bf16-runner-$shard" --network=host \
  -e CUDA_VISIBLE_DEVICES='' \
  -v /data/ssd1/txc/vllm_turbo_baseline_acc:/eval:ro \
  -v /data/ssd1/txc/oscar_vllm/artifacts/gsm8k/20260720:/results \
  -w /eval --entrypoint /bin/bash oscar-vllm:v0.25.0-dev -lc \
  "uv pip install --system --no-cache \
     -r requirements-accuracy-suite.txt absl-py nltk && \
   /usr/bin/python3 tools/run_accuracy_suite.py \
     --suite-dir /results/gsm8k_shards/suite_$shard \
     --output-dir /results/bf16_shards/shard_$shard \
     --base-url http://127.0.0.1:$port/v1 \
     --model qwen3-4b-instruct-2507 --concurrency 1 \
     --benchmarks GSM8K"
```

8 个 runner 均以退出码 0 完成，各 shard 都是全量 `scored` 且 request failure 为 0。按 official manifest 原顺序合并后的实际结果为：

| Correct / Total | Accuracy | Scored | Request failed | 合并评测墙钟时长 |
| ---: | ---: | ---: | ---: | ---: |
| 1161 / 1319 | 0.8802122820 | 1319 | 0 | 807.6267 秒 |

合并结果位于 `artifacts/gsm8k/20260720/bf16/`，`predictions.jsonl` 为 1319 行、1319 个唯一 ID、无空记录，SHA256 为 `c9aba24eb1c6fb9f4dee2ed6dc19a5f5362611badfba692294ff55df85a631c6`。各 shard 原始结果位于 `bf16_shards/shard_0` 至 `shard_7`，服务和 runner 日志分别为 `bf16_8way_server_0.log` 至 `_7.log`、`bf16_8way_runner_0.log` 至 `_7.log`。

### 5.10 GSM8K OSCAR-vLLM 精度验收

OSCAR 使用与第 5.9 节完全相同的 8 个 manifest shard 和 official runner 参数。服务实际启动命令如下，其中 `gpu` 为 0 至 7、`port=8200+gpu`：

```bash
sudo docker run -d --name "oscar-eval-$gpu" \
  --gpus "device=$gpu" --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_OSCAR_K_ROTATION_PATH=/rot/k_rotation_qqt_r_h_pbr.pt \
  -e VLLM_OSCAR_V_ROTATION_PATH=/rot/v_rotation_sst_r_h_pbr.pt \
  -e VLLM_OSCAR_K_CLIP_RATIO=0.96 \
  -e VLLM_OSCAR_V_CLIP_RATIO=0.92 \
  -e VLLM_OSCAR_GROUP_SIZE=128 \
  -e VLLM_OSCAR_PREFIX_TOKENS=64 \
  -e VLLM_OSCAR_RECENT_TOKENS=256 \
  -v /data/ssd1/checkpoints/Qwen3-4B-Instruct-2507:/model:ro \
  -v /data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations:/rot:ro \
  -p "127.0.0.1:$port:8000" oscar-vllm:v0.25.0-dev /model \
  --served-model-name qwen3-4b-instruct-2507 \
  --dtype bfloat16 --kv-cache-dtype oscar_int2 \
  --max-model-len 8192 --max-num-seqs 1 \
  --no-enable-chunked-prefill --no-enable-prefix-caching \
  --enforce-eager --gpu-memory-utilization 0.1 --port 8000
```

runner 的实际命令如下，其中 `shard` 为 0 至 7、`port=8200+shard`：

```bash
sudo docker run -d --name "oscar-runner-$shard" --network=host \
  -e CUDA_VISIBLE_DEVICES='' \
  -v /data/ssd1/txc/vllm_turbo_baseline_acc:/eval:ro \
  -v /data/ssd1/txc/oscar_vllm/artifacts/gsm8k/20260720:/results \
  -w /eval --entrypoint /bin/bash oscar-vllm:v0.25.0-dev -lc \
  "uv pip install --system --no-cache \
     -r requirements-accuracy-suite.txt absl-py nltk && \
   /usr/bin/python3 tools/run_accuracy_suite.py \
     --suite-dir /results/gsm8k_shards/suite_$shard \
     --output-dir /results/oscar_shards/shard_$shard \
     --base-url http://127.0.0.1:$port/v1 \
     --model qwen3-4b-instruct-2507 --concurrency 1 \
     --benchmarks GSM8K"
```

8 个 runner 均退出码 0，各 shard 全部为 `scored`、request failure 为 0。

真实 GSM8K 请求日志再次确认 `oscar_int2`、K/V 两个 36 层 rotation checkpoint、K/V clip 0.96/0.92、OSCAR backend、Triton KV write 和 mixed attention read 全部启用。最终结果为：

| 配置 | Correct / Total | Accuracy | 相对 BF16 accuracy delta |
| --- | ---: | ---: | ---: |
| vLLM BF16 | 1161 / 1319 | 0.8802122820 | - |
| OSCAR-vLLM | 1157 / 1319 | 0.8771796816 | -0.0030326005（-0.303260 个百分点） |

OSCAR 比硬阈值 `1134/1319` 多 23 条正确样本，accuracy 比 `0.8591` 高 `0.0180796816`，因此通过任务精度验收；同时比 OSCAR-SGLang 参考 accuracy `0.8741` 高 `0.0030796816`。样本级对照为共同正确 1134 条、仅 BF16 正确 27 条、仅 OSCAR 正确 23 条、共同错误 135 条，合计 1319 条。

合并评测实际墙钟时长为 `1212.5210` 秒。结果位于 `artifacts/gsm8k/20260720/oscar/`；`predictions.jsonl` 为 1319 行、1319 个唯一 ID、无空记录，SHA256 为 `f71fb0d0edc68b697165c79de426f30a666ab9b70ccc2e1bdeeb1213f47664ab`。原始 shard 结果位于 `oscar_shards/shard_0` 至 `shard_7`，服务和 runner 日志分别为 `oscar_8way_server_0.log` 至 `_7.log`、`oscar_8way_runner_0.log` 至 `_7.log`。

## 6. 压缩率理论与实测

以下为代码实际计算结果，不是 GPU 实测。配置为 `max_model_len=8192`、block size 16、8 KV heads、head dim 128、单层：

| 口径 | Bytes/layer | 相对 BF16 压缩倍数 |
| --- | ---: | ---: |
| BF16 逻辑占用 | 33,554,432 | 1.0000x |
| OSCAR mixed 有效数据 | 6,348,800 | 5.2852x |
| OSCAR padded page 分配 | 7,340,032 | 4.5714x |

OSCAR 每个物理 page 为 14,336 bytes，其中 INT2 page 为 10,240 bytes，BF16 arena 分片为 4,096 bytes；同几何纯 BF16 page 为 65,536 bytes。

BF16 与 OSCAR 均使用单张 B200、同一模型、BF16 权重、`max_model_len=8192`、`max_num_seqs=1`、关闭 chunked prefill/prefix cache、eager、`gpu_memory_utilization=0.1`。两次启动均实际报告 available KV cache 9.35 GiB：

| Cache dtype | GPU KV cache tokens | Blocks（16 tokens/block） | 8192-token 最大并发 |
| --- | ---: | ---: | ---: |
| BF16 `auto` | 68,080 | 4,255 | 8.31x |
| OSCAR `oscar_int2` | 311,280 | 19,455 | 38.00x |

OSCAR/BF16 的实测 token 与 block 容量倍率均为 `4.572268x`，与 padded page 理论分配倍率 `4.5714x` 一致；小幅差异来自可分配整数 block 的取整。BF16 cache 数据所在的启动日志位于 `artifacts/gsm8k/20260720/bf16_invalid_server_timeout_server.log`；该 cache 分配在后续服务超时前已经完成，因此不受第 5.8 节评测中断影响。OSCAR 启动日志位于 `artifacts/smoke/oscar_server.log`。

### 6.1 跨框架压缩率对比补充

独立对比报告 [`oscar_kv_compression_comparison.md`](oscar_kv_compression_comparison.md) 在同一张 B200、同一模型、TP=1、单请求和精确 10 GiB 全部 KV tensor 预算下重跑了两套框架。OSCAR-vLLM 的框架内物理容量压缩率为 `4.571523x`，论文官方 OSCAR-SGLang 固定提交的校正结果为 `6.286860x`；后者高 `37.5222%`。差异来自物理 allocator 布局，不是两者的 OSCAR mixed-KV 语义不同。

SGLang quant-only 估算器的 `6.400044x` 没有扣除独立 HP arena 和保留页，不能作为物理主结果。其默认 HP pool 在 48 请求槽、同一 10 GiB 预算下会把物理容量比降至 `0.970336x`，因此跨框架结果必须连同并发槽位和 HP pool 配置一起解释。完整环境、公式、命令、失败边界、日志路径和哈希见独立报告。

## 7. 验收结论

- 实现范围通过：Qwen3-4B-Instruct-2507 的 OSCAR mixed KV 路径实际加载 attention-aware K/V rotation 和 0.96/0.92 clipping，执行 prefix 64 BF16、recent 256 BF16、history INT2。
- Kernel 路径通过：真实请求日志证明 Triton KV write、history demotion、mixed read/dequantization/online softmax 和 V inverse rotation 均执行；serving backend 不调用仅供测试的 full-dequant oracle。
- 最小测试通过：最终 CPU 为 `8 passed`，CUDA 为 `6 passed`；mixed layout、INT2 roundtrip、attention 对照和真实模型 smoke 均已有实际结果。
- 压缩率通过：同 9.35 GiB KV budget 下，OSCAR/BF16 cache token/block 容量实测为 `4.572268x`，与 padded page 理论 `4.5714x` 一致。
- 精度通过：BF16 为 `1161/1319`，OSCAR 为 `1157/1319`；OSCAR 比 `1134/1319` 硬阈值多 23 条，相对 BF16 accuracy delta 为 `-0.0030326005`。
- 可复现性通过：固定源码、镜像 digest、模型、数据集、rotation、依赖版本、GPU 映射、`nvidia-smi`、完整命令、逐 shard 原始结果、合并结果和日志均已记录；最终无残留 OSCAR 容器。

本任务规定的硬性验收项均已完成。原型边界仍为第 4.1 节所述单请求、禁 chunked prefill/prefix cache/speculative decoding/KV skip layer/CUDA graph；这些能力不在本任务硬性范围内。
