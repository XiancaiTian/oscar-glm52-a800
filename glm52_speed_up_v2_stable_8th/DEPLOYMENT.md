# GLM-5.2 8th 2P1D Deployment

本目录是从已验证的 7th 部署流程固化出的 8th 可迁移部署包。默认配置保留 7th 的 IP、模型路径和镜像作为示例；迁移到新 6 台机器时，只改 `configs/deploy_8th.env`。

## 关键文件

```text
configs/deploy_8th.env                         # 8th 主配置
configs/examples/deploy_7th_example.env        # 7th 已验证示例配置
configs/warmup_shapes_stage_90k_v2.txt         # stage90k warmup shape
source/runtime_patch_source/                   # 运行时补丁源，包含 shape padding 补丁
source/runtime_patch_source/MANIFEST.sha256
artifacts/images/vllm_openai_glm52_v2_stable_2p1d_usagefix_20260625_114616.tar
artifacts/images/vllm_openai_glm52_v2_stable_2p1d_usagefix_20260625_114616.tar.sha256
artifacts/images/latest_image_manifest.json
scripts/deploy/deploy_8th_claude204_current_verified.sh
scripts/deploy/deploy_8th_current_verified.sh
scripts/deploy/deploy_stage_90k_v2.sh
scripts/deploy/lib/manage_stack_2p1d.sh
scripts/deploy/lib/container_entry.sh
scripts/watchdog/prefill_decode_availability_watchdog.sh
```

## 修改配置

编辑 `configs/deploy_8th.env`：

```bash
export MODEL_PATH="${MODEL_PATH:-/path/to/GLM-5.2-FP8}"
export MODEL_ID="${MODEL_ID:-GLM-5.2-FP8}"
export IMAGE_TAG="${IMAGE_TAG:-registry.example.com/ae/vllm_openai_glm52:tag}"
export IMAGE_TAR="${IMAGE_TAR:-/path/to/image.tar}"
export RUNTIME_PATCH_SOURCE_DIR="${RUNTIME_PATCH_SOURCE_DIR:-${TASK_ROOT}/source/runtime_patch_source}"

export BUILD_HOST="${BUILD_HOST:-x.x.x.x}"
export PREFILL0_HEAD="${PREFILL0_HEAD:-x.x.x.x}"
export PREFILL0_WORKER="${PREFILL0_WORKER:-x.x.x.x}"
export PREFILL1_HEAD="${PREFILL1_HEAD:-x.x.x.x}"
export PREFILL1_WORKER="${PREFILL1_WORKER:-x.x.x.x}"
export DECODE_HEAD="${DECODE_HEAD:-x.x.x.x}"
export DECODE_WORKER="${DECODE_WORKER:-x.x.x.x}"
export TARGET_HOSTS="${TARGET_HOSTS:-... six unique hosts ...}"
export ALLOWED_HOSTS="${ALLOWED_HOSTS:-... same hosts, comma separated ...}"
export EXPECTED_HOSTS_CSV="${EXPECTED_HOSTS_CSV:-... same hosts, comma separated ...}"
```

`precheck` 会拒绝角色 IP、`TARGET_HOSTS`、`ALLOWED_HOSTS`、`EXPECTED_HOSTS_CSV` 不一致的配置。

`precheck` 也会检查 `source/runtime_patch_source` 是否完整，尤其是
`vllm/v1/worker/gpu_model_runner.py` 中的 shape padding 标记：
`VLLM_PREFILL_SHAPE_BUCKET`、`_pad_for_prefill_shape_bucket` 和
`Prefill shape bucket enabled`。

## Runtime Patch

8th 已将运行时补丁源固化在本目录内：

```text
source/runtime_patch_source/
```

部署时 `manage_stack_2p1d.sh` 会从该目录复制以下补丁进容器内的 `/opt/vllm_glm52_v1`：

```text
vllm/v1/engine/core.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/attention/backend.py
vllm/v1/attention/backends/mla/indexer.py
vllm/v1/attention/backends/mla/triton_mla_sparse.py
vllm/v1/attention/ops/mqa_logits_triton.py
vllm/v1/attention/ops/triton_sparse_mla_kernel.py
vllm/model_executor/layers/sparse_attn_indexer.py
vllm/entrypoints/openai/chat_completion/serving.py
tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
```

补丁完整性可在迁移后校验：

```bash
cd /path/to/glm52_speed_up_v2_stable_8th/source/runtime_patch_source
sha256sum -c MANIFEST.sha256
```

## IB 网口示例

对比 `glm52_speed_up_v2_stable_4th`、`5th`、`6th`、`7th` 后，IB/HCA 设备列表一致，都是多 rail：

```bash
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_6,mlx5_7"
export UCX_TLS="rc,cuda_copy,cuda_ipc"
export UCX_DEVICES="mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1"
export UCX_NET_DEVICES="mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1"
```

差异在 Linux 网口名，也就是 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` 使用的 `IFACE`。`4th`、`5th`、`6th` 的公共部署环境没有主机级覆盖，适用于 6 台机器网口都叫 `ens22f0` 的情况：

```bash
export IFACE="ens22f0"
export HOST_IFACE_OVERRIDES=""
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_6,mlx5_7"
export UCX_TLS="rc,cuda_copy,cuda_ipc"
export UCX_DEVICES="mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1"
export UCX_NET_DEVICES="mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1"
```

`7th` 是另一种情况：大部分机器仍使用 `ens22f0`，但 `192.168.16.68`、`192.168.16.69` 的服务网口名是 `ens22f0np0`，所以需要按 IP 覆盖：

```bash
export IFACE="ens22f0"
export HOST_IFACE_OVERRIDES="192.168.16.68=ens22f0np0,192.168.16.69=ens22f0np0"
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_6,mlx5_7"
export UCX_TLS="rc,cuda_copy,cuda_ipc"
export UCX_DEVICES="mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1"
export UCX_NET_DEVICES="mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1"
```

迁移到新的 6 台机器时，用下面命令确认每台机器承载部署 IP 的 Linux 网口名；如果只有部分机器不是默认网口名，只把那些 IP 写入 `HOST_IFACE_OVERRIDES`：

```bash
ssh <host> 'ip -br addr | egrep "ens22f0|ens22f0np0|192\\.168\\."; ibdev2netdev | egrep "mlx5_2|mlx5_3|mlx5_6|mlx5_7"'
```

## 离线镜像包

8th 目录已带上 7th 验证过的镜像 tar，便于迁移到和现有 registry 隔离的内网：

```text
artifacts/images/vllm_openai_glm52_v2_stable_2p1d_usagefix_20260625_114616.tar
artifacts/images/vllm_openai_glm52_v2_stable_2p1d_usagefix_20260625_114616.tar.sha256
artifacts/images/latest_image_manifest.json
```

在隔离环境中，先校验镜像包，再执行 `load_image_all.sh` 分发到 6 台机器：

```bash
cd /path/to/glm52_speed_up_v2_stable_8th/artifacts/images
sha256sum -c vllm_openai_glm52_v2_stable_2p1d_usagefix_20260625_114616.tar.sha256

cd /path/to/glm52_speed_up_v2_stable_8th
./scripts/deploy/load_image_all.sh
```

## 部署流程

在 8th 目录执行：

```bash
cd /nfs/AE/zhanghong/workflow/vllm_a/glm52_speed_up_v2_stable_8th

./scripts/deploy/deploy_8th_claude204_current_verified.sh print-config
./scripts/deploy/deploy_8th_claude204_current_verified.sh precheck
./scripts/deploy/load_image_all.sh
./scripts/deploy/deploy_8th_claude204_current_verified.sh full-restart-warmup
./scripts/deploy/deploy_8th_claude204_current_verified.sh verify
```

如果镜像已经在 6 台机器上存在，可以跳过 `load_image_all.sh`。

## 常用操作

```bash
# 只重启 proxy，不重启 P0/P1/D0 vLLM
./scripts/deploy/deploy_8th_claude204_current_verified.sh proxy-restart

# 只重启并 warmup 某个 Prefill 组
./scripts/deploy/deploy_8th_claude204_current_verified.sh prefill-recover P0
./scripts/deploy/deploy_8th_claude204_current_verified.sh prefill-recover P1

# 只跑 stage90k warmup
./scripts/deploy/deploy_8th_claude204_current_verified.sh warmup

# 查看当前容器和进程状态
./scripts/deploy/status_all.sh

# 停止当前配置对应的服务进程
./scripts/deploy/stop_all.sh
```

## Watchdog

主流程使用可配置的 availability watchdog：

```bash
./scripts/deploy/deploy_8th_claude204_current_verified.sh availability-watchdog-once
./scripts/deploy/deploy_8th_claude204_current_verified.sh availability-watchdog-start
./scripts/deploy/deploy_8th_claude204_current_verified.sh availability-watchdog-status
./scripts/deploy/deploy_8th_claude204_current_verified.sh availability-watchdog-stop
```

旧 7th 的 `watchdog_common/install/service/final_drill` 链路没有迁移为 8th 主流程，避免保留固定 IP 的 systemd 配置。

## 7th 示例

7th 的已验证配置保存在：

```text
configs/examples/deploy_7th_example.env
```

查看 7th 示例的展开结果：

```bash
DEPLOY_CONFIG_FILE=/nfs/AE/zhanghong/workflow/vllm_a/glm52_speed_up_v2_stable_8th/configs/examples/deploy_7th_example.env \
  ./scripts/deploy/deploy_8th_claude204_current_verified.sh print-config
```
