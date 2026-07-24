# OSCAR-vLLM 与官方 OSCAR-SGLang KV Cache 压缩率对比报告

## 1. 目标与统计口径

本报告比较当前工作区的 OSCAR-vLLM 与 OSCAR 论文官方配套 SGLang 代码在 Qwen3-4B-Instruct-2507 上的 KV Cache 容量压缩率。官方 OSCAR 仓库固定为提交 `41ebcdba3db5f0ce1339c3727caea80df575d437`，实验不修改其 SGLang 实现。

主指标固定为各框架内部的容量比：

```text
KV Cache 容量压缩率 = 同框架 OSCAR 可分配 token 数 / 同框架 BF16 可分配 token 数
```

四组主实验固定使用同一张物理 GPU、同一模型、TP=1、单请求并发，并将全部 KV tensor 的物理存储预算限制为不超过 10 GiB。SGLang 将分别报告调度器标称容量和计入独立高精度池后的物理校正容量。吞吐、延迟和精度不属于本报告范围。

## 2. 固定配置

### 2.1 模型与 KV 配置

| 项目 | 值 |
| --- | --- |
| 模型路径 | `/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507` |
| 层数 | 36 |
| KV heads | 8 |
| K/V head dim | 128 / 128 |
| 模型 dtype | BF16 |
| 最大位置长度 | 262144 |
| OSCAR prefix / recent | 64 / 256 tokens |
| history dtype | INT2 |
| 量化 group size | 128 |
| scale / zero dtype | FP32 |

以上模型数据来自本地 `config.json`，OSCAR 参数来自两套实现和官方 README 的 serving 配置。

### 2.2 理论字节基线

每层、每个 KV head、每个 token 的 BF16 K+V 为：

```text
2 * 128 * 2 bytes = 512 bytes
```

INT2 history 的 packed K+V 为 64 bytes，K/V 各保存一组 FP32 `(scale, zero)`，metadata 共 16 bytes，因此合计为 80 bytes。只考虑 history 时的渐近比值为：

```text
512 / 80 = 6.4x
```

该值不包含固定的 64-token BF16 prefix 和 256-token BF16 recent window，不能直接称为 mixed-KV 实际压缩率。

### 2.3 两套物理布局的关键差异

- OSCAR-vLLM 使用 16-token block；每个物理 block 同时计入 INT2 数据和按 block 摊分的 BF16 arena。当前 Qwen3-4B、8192 最大长度下，每层每个 block 为 `10,240 + 4,096 = 14,336` bytes，理论 page 容量比为 `4.5714x`。
- 官方 OSCAR-SGLang 将 INT2 packed K/V、FP32 scale/zero、共享 BF16 prefix 池和每请求 BF16 recent ring 分配为独立 tensor。BF16 recent ring 为 `256 + 8 - 1 = 263` slots；默认共享 prefix 池为 `16 * max_running_requests * 64` slots。
- SGLang 的 `max_total_num_tokens` 估算系数只计入 INT2 packed 数据和 scale/zero，即 80 bytes/head/token；独立 BF16 池没有进入该系数。因此其启动日志可能给出接近 `6.4x` 的标称容量比，但该值并非全部 KV tensor 的物理压缩率。

## 3. 实验环境

### 3.1 主机与 GPU

UTC `2026-07-20 07:15:45` 和 `07:17:16` 连续两次检查时，GPU 0-7 均无 compute process。主实验固定使用物理 GPU 0，容器内设置 `CUDA_VISIBLE_DEVICES=0`。

| 项目 | 实际值 |
| --- | --- |
| GPU | NVIDIA B200 |
| 单卡总显存 | 183359 MiB |
| NVIDIA Driver | 595.71.05 |
| 主机 nvidia-smi CUDA | 13.2 |

完整第二次检查保存在 `artifacts/kv_compression/20260720/gpu_check_2.txt`。

### 3.2 官方 OSCAR-SGLang 容器

| 项目 | 实际值 |
| --- | --- |
| 派生镜像 | `oscar-sglang:41ebcdb` |
| 镜像 digest | `sha256:3384f67634c9f8c93413770ed1a444b293eb6e59df6377f4be20d695e28e5dd6` |
| 基础镜像 | `docker.m.daocloud.io/lmsysorg/sglang:v0.5.10.post1` |
| Python | 3.12.3 |
| PyTorch / CUDA | 2.9.1+cu129 / 12.9 |
| Transformers | 5.3.0 |
| Triton | 3.5.1 |
| sglang-kernel | 0.4.1 |
| FlashInfer | 0.6.7.post3 |

派生镜像只将固定提交的 `sglang-research/python` 放到 `PYTHONPATH` 首位。真实 GPU 导入确认 `sglang` 来自 `/opt/oscar/sglang-research/python/sglang/__init__.py`，`UnifiedInt2HPKVPool` 来自官方源码，且 CUDA 可见 GPU 为 NVIDIA B200。验证输出保存在 `artifacts/kv_compression/20260720/sglang_image_verify.txt`。

### 3.3 OSCAR-vLLM 容器

| 项目 | 实际值 |
| --- | --- |
| 派生镜像 | `oscar-vllm:v0.25.0-dev` |
| 镜像 ID | `sha256:0356873d691c97f72ccc08eda4ca7aecb9b297206763fab8ea3ad01b3b4f1e7e` |
| Python | 3.12.13 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| Transformers | 5.13.0 |
| Triton | 3.6.0 |
| vLLM | 0.25.0 |
| FlashInfer | 0.6.13 |

真实 GPU 版本检查保存在 `artifacts/kv_compression/20260720/vllm_image_verify.txt`。两组服务均设置 `CUDA_VISIBLE_DEVICES=0`，只挂载物理 GPU 0。

## 4. OSCAR-vLLM 10 GiB 实验

### 4.1 实际命令

BF16 组的核心启动参数为：

```bash
sudo docker run -d --name kvcomp-vllm-bf16 \
  --gpus 'device=0' --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
  -v /data/ssd1/checkpoints/Qwen3-4B-Instruct-2507:/model:ro \
  oscar-vllm:v0.25.0-dev /model \
  --dtype bfloat16 --kv-cache-dtype auto \
  --kv-cache-memory-bytes 10737418240 \
  --max-model-len 8192 --max-num-seqs 1 \
  --no-enable-chunked-prefill --no-enable-prefix-caching --enforce-eager
```

OSCAR 组使用同一命令，只把 KV dtype 改为 `oscar_int2`，并增加以下固定环境与 rotation 挂载：

```bash
-e VLLM_OSCAR_K_ROTATION_PATH=/rot/k_rotation_qqt_r_h_pbr.pt \
-e VLLM_OSCAR_V_ROTATION_PATH=/rot/v_rotation_sst_r_h_pbr.pt \
-e VLLM_OSCAR_K_CLIP_RATIO=0.96 -e VLLM_OSCAR_V_CLIP_RATIO=0.92 \
-e VLLM_OSCAR_GROUP_SIZE=128 \
-e VLLM_OSCAR_PREFIX_TOKENS=64 -e VLLM_OSCAR_RECENT_TOKENS=256 \
-v /data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations:/rot:ro
```

两次启动前的完整 GPU 状态分别保存于 `nvidia_smi_before_vllm_bf16.txt` 和 `nvidia_smi_before_vllm_oscar.txt`，均显示物理 GPU 0 为 0 MiB、0% 利用率、无运行进程。

### 4.2 实测结果

| 配置 | 10 GiB 下 tokens | 16-token blocks | 8192-token 并发 | 服务状态 |
| --- | ---: | ---: | ---: | --- |
| vLLM BF16 | 72,816 | 4,551 | 8.89x | 健康，正常退出 |
| OSCAR-vLLM | 332,880 | 20,805 | 40.63x | 健康，正常退出 |

实测容量压缩率为：

```text
332880 / 72816 = 4.571522742x
```

字节复算如下：

| 配置 | 每 block 全 36 层字节 | 实际分配字节 | 10 GiB 未使用字节 |
| --- | ---: | ---: | ---: |
| vLLM BF16 | 2,359,296 | 10,737,156,096 | 262,144 |
| OSCAR-vLLM | 516,096 | 10,737,377,280 | 40,960 |

理论 page 比值为 `2,359,296 / 516,096 = 4.571428571x`，与实测容量比的微小差异只来自 10 GiB 除以整 block 后的取整。OSCAR 日志还确认两个 36 层 rotation、prefix 64/recent 256、clip 0.96/0.92、Triton KV write 和 mixed attention read 均生效。

完整日志位于：

- `artifacts/kv_compression/20260720/vllm_bf16_server.log`
- `artifacts/kv_compression/20260720/vllm_oscar_server.log`

## 5. 官方 OSCAR-SGLang 10 GiB 实验

### 5.1 B200 backend 约束与实际命令

官方 README 的 H100 配方使用 FA3 prefill，但固定代码中的 FA3 backend 明确限制为 SM80-SM90，不能在 B200 SM100 上启动。首次尝试还发现默认总 backend 会把 BF16 page size 从 1 改为 64；OSCAR 的实验性 piecewise graph warmup会进入 wheel 中不支持 SM100 的 FlashAttention 路径。有效主实验因此同时固定：

```text
--attention-backend triton
--prefill-attention-backend triton
--decode-attention-backend triton
--disable-cuda-graph
--disable-piecewise-cuda-graph
```

这些设置只改变 attention/warmup backend，不改变 KV tensor 的 dtype、shape 或容量公式。两次失败尝试分别保存在 `sglang_bf16_invalid_fa3_sm100.log` 和 `sglang_oscar_req1_invalid_piecewise_sm100.log`，不计入主结果。

有效 BF16 组的核心参数为：

```bash
python3 -m sglang.launch_server \
  --model-path /model --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --context-length 8192 --tensor-parallel-size 1 \
  --max-running-requests 1 --max-total-tokens 72816 \
  --mem-fraction-static 0.2 --page-size 1 \
  --attention-backend triton --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --disable-cuda-graph --disable-piecewise-cuda-graph
```

OSCAR 组保持上述参数，只改为 `--kv-cache-dtype int2 --kv-cache-quant-group-size 128 --page-size 8 --max-total-tokens 457784`，并设置官方 mixed-KV、rotation、clip、HP dtype 和 scale dtype 环境变量。两组均在 `oscar-sglang:41ebcdb` 中设置 `CUDA_VISIBLE_DEVICES=0` 并只挂载物理 GPU 0。

### 5.2 单请求实测结果

| 配置 | scheduler tokens | quant slots | BF16 HP/padded slots | 全部 KV tensor 字节 | 服务状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| SGLang BF16 | 72,816 | 0 | 72,817 | 10,737,303,552 | `/model_info` 200，正常退出 |
| OSCAR-SGLang | 457,784 | 457,792 | 1,287 | 10,737,303,552 | `/model_info` 200，健康启动 |

OSCAR 的 1,287 个 HP slots 由 1,024 个共享 prefix slots 与 263 个单请求 recent-ring slots 组成。其精确字节为：

```text
quant = 457792 * 23040 = 10,547,527,680 bytes
HP    = 1287 * 147456 =    189,775,872 bytes
total =                     10,737,303,552 bytes
```

两组都比 10 GiB 少 114,688 bytes。计入全部独立 HP tensor 后，单请求物理容量压缩率为：

```text
457784 / 72816 = 6.286860031x
```

若只使用 SGLang 容量估算器的 quant-token 系数而不扣除 HP arena 和保留页，同一 10 GiB 会得到 BF16 `72,817`、OSCAR `466,032` 个标称 tokens，标称比值为 `6.400043946x`。这解释了为什么直接读取 quant-only 估算会看起来等于 6.4x，而实际物理容量比是 6.2869x。

OSCAR mixed pool 的通用日志显示 `KV Cache is allocated. #tokens: 1287`；源码和同一启动日志表明这里传入的是 `hp_total_slots`，不是 scheduler 容量。真正的 scheduler 容量由后续日志 `max_total_num_tokens=457784` 给出，不能把 1,287 误当成总容量。

有效日志与启动前 GPU 状态位于：

- `artifacts/kv_compression/20260720/sglang_bf16_aligned_server.log`
- `artifacts/kv_compression/20260720/sglang_oscar_req1_server.log`
- `artifacts/kv_compression/20260720/nvidia_smi_before_sglang_bf16_aligned.txt`
- `artifacts/kv_compression/20260720/nvidia_smi_before_sglang_oscar_req1_valid.txt`

OSCAR 服务已完成健康启动；回收时 30 秒停止窗口耗尽后 Docker 强制结束，状态为 exit 137、`OOMKilled=false`。该状态发生在结果和日志已经落盘之后，不是启动或 KV 分配 OOM。

## 6. SGLang 48 请求槽敏感性实验

保持同一镜像、GPU、模型、10 GiB 总 KV tensor 预算和全部 mixed-KV 参数，只把 `--max-running-requests` 从 1 改为 48。按官方默认公式，HP prefix pool 从 1,024 slots 扩大到：

```text
48 * 64 * 16 = 49,152 slots
```

每请求 recent ring 为 263 slots，48 个请求槽共 12,624 slots，因此 HP arena 合计 61,776 slots。实测日志与精确复算如下：

| 项目 | 单请求 | 48 请求槽 |
| --- | ---: | ---: |
| HP prefix slots | 1,024 | 49,152 |
| HP recent slots | 263 | 12,624 |
| HP total slots | 1,287 | 61,776 |
| HP bytes | 189,775,872 | 9,109,241,856 |
| quant slots（含保留页） | 457,792 | 70,664 |
| scheduler tokens | 457,784 | 70,656 |
| 全部 KV tensor 字节 | 10,737,303,552 | 10,737,340,416 |

48 槽下 HP arena 占全部 KV tensor 的 `84.8370%`，scheduler 容量相对单请求下降 `84.5656%`。相对于同预算的 SGLang BF16 基线，容量比为：

```text
70656 / 72816 = 0.970336190x
```

即在该默认 prefix-pool 配置下，48 个请求槽的 OSCAR-SGLang 可调度 token 容量反而比 BF16 少约 `2.9664%`。这不是 INT2 本身失效，而是 `16 * max_running_requests * prefix_tokens` 的 BF16 prefix 预留在 10 GiB 小池预算下占用了 8.48 GiB。该结论说明 SGLang 的物理压缩率必须同时报告请求槽数和 HP prefix-pool 配置。

服务实际健康启动，日志确认 `max_req_slots=48`、`hp_prefix_pool_tokens=49152`、HP arena 8.48 GB 和 `max_total_num_tokens=70656`。完整日志和 GPU 状态位于：

- `artifacts/kv_compression/20260720/sglang_oscar_req48_server.log`
- `artifacts/kv_compression/20260720/nvidia_smi_before_sglang_oscar_req48.txt`

回收时即使给出 60 秒停止窗口，SGLang supervisor 仍由 Docker 强制结束，exit 137 且 `OOMKilled=false`；这发生在 `/model_info` 返回 200 和所有容量证据落盘之后。

## 7. 理论 mixed-KV 长度矩阵

以下矩阵比较同一 OSCAR 语义下的逻辑有效数据量，不包含 vLLM block padding，也不包含 SGLang 独立 HP pool 的额外预留。设上下文长度为 `L`，单层单个 KV head 的字节数为：

```text
BF16 bytes  = L * 512
mixed bytes = min(L, 320) * 512 + max(L - 320, 0) * 80
```

其中 320 是 64-token prefix 与 256-token recent window 之和。

| 上下文长度 | BF16 bytes/head/layer | mixed bytes/head/layer | 有效压缩率 |
| ---: | ---: | ---: | ---: |
| 8K（8,192） | 4,194,304 | 793,600 | 5.285161290x |
| 32K（32,768） | 16,777,216 | 2,759,680 | 6.079406308x |
| 128K（131,072） | 67,108,864 | 10,624,000 | 6.316722892x |
| 256K（262,144） | 134,217,728 | 21,109,760 | 6.358088770x |

该矩阵对两套实现都成立，因为两者使用相同的 prefix/recent/history 数据语义。上下文越长，固定 BF16 window 的占比越小，压缩率趋近但不会在有限长度上达到 history-only 的 `6.4x`。

## 8. 主结果与解释

在同一张 B200、同一模型、TP=1、单请求和精确 10 GiB 全部 KV tensor 预算下，主结果为：

| 框架 | BF16 tokens | OSCAR tokens | 框架内物理容量压缩率 |
| --- | ---: | ---: | ---: |
| OSCAR-vLLM | 72,816 | 332,880 | 4.571522742x |
| 官方 OSCAR-SGLang | 72,816 | 457,784 | 6.286860031x |

官方 OSCAR-SGLang 的单请求物理容量压缩率比 OSCAR-vLLM 高：

```text
6.286860031 / 4.571522742 - 1 = 37.522230%
```

同一预算下对应多出 124,904 个 scheduler tokens。这个差异不能解释为 OSCAR 压缩算法不同：第 7 章的逻辑 mixed-KV 公式对两者相同。差异来自物理 allocator 布局：OSCAR-vLLM 的每个物理 block 都带有 BF16 arena，10 GiB 下共预留 20,805 个 HP slots，HP 占物理 KV page 的 `28.5714%`；官方 OSCAR-SGLang 在单请求配置下把 HP pool 独立出来，只预留 1,287 个 HP slots。因此，本报告的结论是单请求、10 GiB 条件下 SGLang 的存储布局具有更高物理容量压缩率，不是论文算法本身优于 vLLM 版本。

SGLang quant-only 估算器给出的 `6.400043946x` 未扣除独立 HP arena 和保留页，只作为标称值；主结论使用全部 KV tensor 校正后的 `6.286860031x`。第 6 章的 48 请求槽结果进一步说明，该优势依赖并发槽位和 HP pool 配置，不能外推为任意并发下的固定倍率。

## 9. 适用边界与证据

- 本任务只测 KV tensor 容量，不测吞吐、延迟或精度；服务健康启动用于确认实际 allocator 路径和 tensor shape 生效，没有把一次 inference 请求的性能或正确性纳入结论。
- 官方 H100 配方中的 FA3 和实验性 piecewise graph 路径不支持当前 B200 SM100。有效 SGLang 实验使用固定提交内的 Triton attention backend，并关闭 CUDA graph 和 piecewise CUDA graph；这些选项不改变 KV tensor 的 dtype、shape 或容量。
- `nvidia-smi` 的进程驻留显存还包含权重和 workspace，不作为 KV tensor 字节数。物理预算由源码 tensor shape、实际启动参数和 allocator 日志逐项复算。
- SGLang 48 请求槽的结果依赖默认 `16 * max_running_requests * prefix_tokens` 共享 prefix pool 公式及 10 GiB 小池预算，不能直接推广到其他池配置。
- 两个 OSCAR-SGLang 容器均在结果保存后由 Docker 强制回收，exit 137 且 `OOMKilled=false`；主结果取自回收前已健康启动的服务和落盘日志。

关键原始日志的 SHA256 为：

| 日志 | SHA256 |
| --- | --- |
| `vllm_bf16_server.log` | `8bbdf0d2e63427406a05a87a85d33d77603d9c0b82f22db7734d005033a800f1` |
| `vllm_oscar_server.log` | `cc47850e296a8a19d1f00e9ef544d7c207c2920293010c1c7133121e963887bd` |
| `sglang_bf16_aligned_server.log` | `8ad420273433cc8063d424f18ac23e8956c2fb1889042c709325374f094a3dac` |
| `sglang_oscar_req1_server.log` | `4602b1b7b38c2f6ac3e817eb60726f6b918286c9c50d7831692cdec41423a69e` |
| `sglang_oscar_req48_server.log` | `0c5edff2529c4ccab8a7a336ee2f64fdb670d90be458952c04d77ee68539a60f` |

精确计算结果另存于 `artifacts/kv_compression/20260720/final_calculations.txt` 和 `sglang_10g_capacity_calculation.json`。

## 10. 阶段状态

| 阶段 | 状态 |
| --- | --- |
| 固定源码、公式和环境 | 已完成 |
| 构建并验证官方 SGLang 镜像 | 已完成 |
| vLLM BF16 / OSCAR 10 GiB 实验 | 已完成 |
| SGLang BF16 / OSCAR 10 GiB 实验 | 已完成 |
| SGLang 48 请求槽敏感性实验 | 已完成 |
| 最终计算与证据复核 | 已完成 |
