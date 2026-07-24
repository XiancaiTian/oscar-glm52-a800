# OSCAR-vLLM 部署说明

本文说明如何在单机单卡 NVIDIA GPU 上部署当前 OSCAR-vLLM 实验实现。文档中的
版本、路径和参数均来自本工作区已经落地的构建或实验记录。

## 1. 当前版本与验收状态

截至 2026-07-23，本地最新冻结镜像为：

| 项目 | 实际值 |
| --- | --- |
| Docker 镜像 | `oscar-vllm:bf16-q-prefill-20260721` |
| Image ID | `sha256:44ac3500c64c5857fd89502ea0404a5566ddfa7f9b77e26446829cd3b18e9d7a` |
| 上游 vLLM | `0.25.0` |
| Python | `3.12.13` |
| PyTorch | `2.11.0+cu130` |
| Triton | `3.6.0` |
| Transformers | `5.13.0` |
| 开发分支 | `oscar-vllm-v0.25.0` |
| 上游基线 commit | `702f4814fe54fabff350d43cb753ae3e47c0c276` |

该镜像已经通过定向 CUDA correctness 测试，并包含 BF16 metadata、full-fused
cached prefill、V rotation absorption 和 BF16 Q rotation。它尚未完成最新代码的完整
GSM8K 回归和最终 TTFT 验收，因此当前定位是实验镜像，不是正式生产版本。

`oscar-vllm:full-fused-prefill-20260721` 是最近一个完成真实服务 profiler 和 TTFT
实验的前序镜像，但不包含最后的 BF16 Q rotation 优化。需要复现既有服务实验时使用
该前序镜像；需要运行最新代码时使用本节列出的最新镜像。

## 2. 已验证范围

当前实现的已验证部署边界如下：

- 单机、单卡、TP=1、PP=1、eager execution。
- Qwen3 decoder-only dense BF16 模型。
- K/V head dimension 128、INT2 group size 128。
- BF16 prefix 64 tokens、BF16 recent 256 tokens，其余 history 使用 OSCAR INT2。
- 支持 prefix caching、chunked prefill 和多请求调度。
- OpenAI-compatible HTTP serving。

以下配置不应在当前版本中用于部署：

- CUDA graph。
- speculative decoding。
- KV connector、KV offloading 或 disaggregated serving。
- `--kv-cache-dtype-skip-layers`。
- async scheduling 与 OSCAR prefix caching 的组合。
- TP、PP、DCP 或 PCP 大于 1。
- V rotation absorption 与 LoRA、量化模型权重的组合。
- MLA、sliding-window、attention sink、混合架构和多模态模型。

## 3. 前置条件

实际验证环境为 NVIDIA B200 183359 MiB、Driver `595.71.05`，主机
`nvidia-smi` 显示 CUDA `13.2`。其他 GPU、Driver 或 CUDA 组合尚未形成完整兼容性
矩阵。

部署前检查 Docker 和 NVIDIA Container Toolkit：

```bash
nvidia-smi
sudo docker info
sudo docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  oscar-vllm:bf16-q-prefill-20260721
```

部署时需要两个只读目录：

```text
/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507
/data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations
```

当前 rotation 文件及 SHA256 为：

```text
50ba567af7f1c34f992a3100a6463e39d0005c0e0a79eb1fa87609cd66a51ba0  k_rotation_qqt_r_h_pbr.pt
b145f52cf0f8e0b0d21aecc80f1f12fde89a6e03df1186423f25d18a3f1847fc  v_rotation_sst_r_h_pbr.pt
```

部署其他模型不能复用这两份 rotation。rotation 必须与模型、层数和 head dimension
匹配。

## 4. 使用现有镜像

确认镜像存在并核对不可变 Image ID：

```bash
IMAGE=oscar-vllm:bf16-q-prefill-20260721

sudo docker image inspect "$IMAGE" \
  --format 'ID={{.Id}} Created={{.Created}}'
```

预期 ID：

```text
sha256:44ac3500c64c5857fd89502ea0404a5566ddfa7f9b77e26446829cd3b18e9d7a
```

核对镜像内软件版本：

```bash
sudo docker run --rm \
  --entrypoint /usr/bin/python3 \
  "$IMAGE" -c \
  'import sys, torch, transformers, triton, vllm;
print(sys.version.split()[0]);
print(vllm.__version__);
print(torch.__version__);
print(triton.__version__);
print(transformers.__version__)'
```

## 5. 从本地源码构建

本地源码位于 `/data/ssd1/txc/oscar_vllm/vllm`。Dockerfile 会把本地 `vllm/`、
`tests/` 和 `pyproject.toml` 复制到基础镜像：

```bash
cd /data/ssd1/txc/oscar_vllm/vllm

sudo docker build \
  -f docker/Dockerfile.oscar-vllm \
  -t oscar-vllm:local \
  .
```

当前工作区包含尚未提交的修改，因此 Git commit 不能唯一标识构建内容。自行重建时应
同时记录 Image ID 和关键源码 SHA256。

## 6. 启动服务

以下配置对应已经使用过的 Qwen3-4B、10 GiB KV tensor、prefix caching、chunked
prefill 和双并发实验口径。最新镜像尚未用这条完整命令完成最终端到端验收，部署后必须
执行第 7 节检查。

```bash
IMAGE=oscar-vllm:bf16-q-prefill-20260721
MODEL_DIR=/data/ssd1/checkpoints/Qwen3-4B-Instruct-2507
ROTATION_DIR=/data/ssd1/txc/oscar_vllm/qwe3-4b-instruct-2507-rotations
GPU_ID=0

sudo docker run -d \
  --name oscar-vllm \
  --gpus "device=${GPU_ID}" \
  --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_OSCAR_K_ROTATION_PATH=/rot/k_rotation_qqt_r_h_pbr.pt \
  -e VLLM_OSCAR_V_ROTATION_PATH=/rot/v_rotation_sst_r_h_pbr.pt \
  -e VLLM_OSCAR_K_CLIP_RATIO=0.96 \
  -e VLLM_OSCAR_V_CLIP_RATIO=0.92 \
  -e VLLM_OSCAR_GROUP_SIZE=128 \
  -e VLLM_OSCAR_PREFIX_TOKENS=64 \
  -e VLLM_OSCAR_RECENT_TOKENS=256 \
  -e VLLM_OSCAR_ABSORB_V_ROTATION=1 \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${ROTATION_DIR}:/rot:ro" \
  -p 127.0.0.1:8000:8000 \
  "$IMAGE" /model \
  --served-model-name qwen3-4b-instruct-2507 \
  --dtype bfloat16 \
  --kv-cache-dtype oscar_int2 \
  --kv-cache-memory-bytes 10737418240 \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --enforce-eager \
  --port 8000
```

容器只挂载一张物理 GPU，该卡在容器内映射为 `CUDA_VISIBLE_DEVICES=0`。不要把
`CUDA_VISIBLE_DEVICES` 设置为主机物理编号，否则会与 Docker 的设备重映射混淆。

`VLLM_OSCAR_ABSORB_V_ROTATION=1` 启用当前性能路径，只支持 TP=1、无 LoRA、未量化
的 dense Qwen3 权重。需要排查 absorption 相关问题时可设置为 `0`，但这不是最新性能
配置。

## 7. 健康检查与请求验证

观察启动日志：

```bash
sudo docker logs -f oscar-vllm
```

健康检查：

```bash
curl --fail --silent http://127.0.0.1:8000/health
curl --fail --silent http://127.0.0.1:8000/v1/models
```

Chat Completions 请求：

```bash
curl --fail --silent \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-4b-instruct-2507",
    "messages": [
      {"role": "user", "content": "What is 2 + 3? Answer with one number."}
    ],
    "temperature": 0,
    "max_tokens": 8
  }'
```

日志至少应确认：

- `OSCAR KV cache enabled`。
- K/V rotation 文件成功加载。
- `Absorbed OSCAR V rotation into 36 Qwen3 QKV layers`。
- OSCAR Triton store 和 mixed attention read 路径在实际请求中激活。
- 没有 `CUDA error`、`EngineCore` failure、OOM 或 unsupported configuration。

检查容器状态：

```bash
sudo docker inspect oscar-vllm \
  --format 'running={{.State.Running}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}'
```

## 8. 停止与清理

```bash
sudo docker stop -t 30 oscar-vllm
sudo docker rm oscar-vllm
```

停止后使用 `nvidia-smi` 确认显存和 compute process 已回收。

## 9. 镜像迁移

导出镜像：

```bash
sudo docker save oscar-vllm:bf16-q-prefill-20260721 \
  | gzip > oscar-vllm-bf16-q-prefill-20260721.tar.gz

sha256sum oscar-vllm-bf16-q-prefill-20260721.tar.gz
```

目标机器导入：

```bash
gzip -dc oscar-vllm-bf16-q-prefill-20260721.tar.gz \
  | sudo docker load

sudo docker image inspect oscar-vllm:bf16-q-prefill-20260721 \
  --format '{{.Id}}'
```

模型和 rotation 文件不在镜像中，必须单独传输，并在目标机器重新核对 SHA256。

## 10. 常见启动失败

| 现象 | 检查项 |
| --- | --- |
| rotation path must be set | 检查两个 `VLLM_OSCAR_*_ROTATION_PATH` 环境变量及只读挂载 |
| rotation file does not exist | 检查宿主机目录、容器内 `/rot` 文件名和权限 |
| requires `--enforce-eager` | 添加 `--enforce-eager`，当前不支持 CUDA graph |
| absorption only supports TP=1 | 固定 `--tensor-parallel-size 1` |
| absorption does not support LoRA | 移除 LoRA，或关闭 absorption 后重新验证 |
| does not support speculative decoding | 移除 speculative decoding 参数 |
| does not support KV connectors/offloading | 移除 KV transfer、connector 和 offload 参数 |
| prefix caching does not support async scheduling | 关闭 async scheduling |
| KV cache allocation 失败 | 检查 GPU 剩余显存和 `--kv-cache-memory-bytes` |
| API 端口无法访问 | 检查 `-p` 映射、容器状态和服务日志 |

## 11. 证据与进一步验证

实现与构建记录见：

- `oscar_vllm_optimization_report.md` 第 10.134 至 10.136 节。
- `artifacts/oscar_vllm_optimization/20260721/bf16_q_prefill_image/`。
- `artifacts/oscar_vllm_optimization/20260721/prefill_bf16_q_production_green/`。

当前部署后仍需补齐两项最终验收：

1. 最新镜像的完整 GSM8K 1319 条精度回归。
2. 最新镜像的真实服务 profiler 和配对 prefix-hit TTFT。
