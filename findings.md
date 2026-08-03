# 发现与决策

## 需求

- 按 `docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md` 依次推进阶段 0–9。
- 首版本目标是单机 8×苹果800、TP=8、原生 DSA/sparse MLA、32K、真实 `oscar_mla_int2` 三段式路径、压缩率与精度/PPL 达标。
- 128K 是 32K 首版本之后的扩展闸门。
- 所有实验报告使用中文，且只记录真实执行数据；每个阶段完成后立即更新。
- Shawn 于 2026-07-25 确认创建 public `XiancaiTian/glm52_oscar_vllm`、稳定分支使用 `main`，并授权后续按最佳方式持续推进。

## 当前发现

- 主仓库当前位于 `main`，`git status --short --branch` 显示与 `origin/main` 同步且工作区初始干净。
- 根目录没有本任务的 `task_plan.md`、`findings.md`、`progress.md`；`oscar_vllm/` 下存在另一任务的同名文件。
- 现有主要资产为 `oscar_vllm/`、`glm52_speed_up_v2_stable_8th/` 和设计文档。
- 当前 shell 中没有 `docker` 命令；尚不能据此判断宿主机没有 Docker，只能确认当前执行环境无法直接调用 Docker CLI。
- `/proc/1/cgroup` 显示当前会话运行在 Kubernetes/containerd 容器中；未安装 docker/podman/nerdctl/ctr/skopeo/umoci/buildah/apptainer，`nvidia-smi` 可用。
- `glm52_speed_up_v2_stable_8th/artifacts/images/` 下保存了已验证镜像 tar、独立 SHA256 文件和 `latest_image_manifest.json`，可作为无 daemon 源码恢复入口。
- 镜像 tar 实际大小为 34,754,212,352 字节；2026-07-24 全量 SHA256 复核通过，实际值与记录的 `b58fc5cf7874da92f5a97306d47c9d7d507332515a73e6812a7eb6145e9bed23` 一致。
- Docker archive 的 config 为 `d6faf4d3...json`，与设计及 manifest 的 image ID 一致；archive 共声明 31 个 layer。
- 镜像 config 的实际创建时间为 `2026-06-25T11:45:17.732856156Z`，平台为 `linux/amd64`，WorkingDir 为 `/vllm-workspace`。
- 只有 layer 29–31 包含 `/opt/vllm_glm52_v1`：layer 29 含 5,377 个条目及 6 个 vLLM 原生 `.so`，layer 30/31 分别是 14/20 个条目的运行时 patch。
- layer 30 覆盖 `serving.py` 和 `core.py`；layer 31 再覆盖 `toy_proxy_server.py`、`serving.py` 和 `core.py`。whiteout 仅用于对应 `__pycache__` 的 opaque 标记，不影响源码文件。
- 镜像中的 `/opt/vllm_glm52_v1` 不含 `.git`，因此需要用外部 Git 仓库提供 commit 历史，再以镜像有效文件内容做一致性审计。
- 已从 layer 29–31 恢复有效 `/opt/vllm_glm52_v1` 到临时目录：共 4,711 个普通文件、1,865,041,381 字节；其中正好包括 4,705 个 Git tracked 源文件和 6 个原生 `.so`。
- 6 个 vLLM 原生扩展总大小约 1.788GB；源码 tracked tree 本身约 77.5MB。
- 镜像中的 6 个 prefill shape-bucket 脏文件与 clean HEAD 哈希完全一致，证明外部工作树上的未提交 shape-bucket 实验未进入已验证镜像。
- clean HEAD 与镜像 tracked tree 的初步逐目录比较只发现 4 个源码文件不同：`compiler_interface.py`、`core.py`、`serving.py`、`toy_proxy_server.py`。后三者是 image manifest 明确记录的最终 patch；`compiler_interface.py` 是 base/source layer 中已有但 manifest 未列出的差异，必须进一步追溯。
- `compiler_interface.py` 的镜像差异新增 `GLM52_DISABLE_INDUCTOR_AUTOTUNE` 安全开关，并 patch Torch Inductor/Triton runtime autotuner；它属于 base image 的 GLM 运行时稳定性 patch，不是 OSCAR 改动。
- `compiler_interface.py` 的镜像 SHA256 为 `6fc8d5bc...409ba`，与 `glm52_speed_up_v1_stable` 的精确 patch 文件及其 image manifest 完全一致，来源已闭合。
- `core.py` patch 将 speculative decoding 下缺失 draft token IDs 从 assert 改为 warning；`serving.py` 修复 Anthropic usage/cache token 统计；`toy_proxy_server.py` 增加 `/v1/messages`、health、metrics 等 decode 透传路由。
- 镜像中 `core.py`、`serving.py`、`toy_proxy_server.py` 的 SHA256 分别与 image manifest 记录的 `ec5fdf...`、`55df27...`、`11a447...` 完全一致。
- 已记录 6 个 vLLM 原生扩展 SHA256：`_C` 为 `1812bd...`、`_C_stable_libtorch` 为 `e79f6e...`、`cumem_allocator` 为 `a73a69...`、`_moe_C` 为 `c59dc1...`、FA2 为 `f8926e...`、FA3 为 `170b23...`。
- `https://github.com/XiancaiTian/glm52_oscar_vllm.git` 与 `https://github.com/XiancaiTian/glm52-oscar-vllm.git` 当前均不存在；`XiancaiTian/vllm.git` 存在但按设计职责属于 OSCAR full-attention 参考，不应复用为新的 GLM 开发仓库。
- 当前 Kubernetes 容器没有 Docker/containerd socket，且缺少 `CAP_SYS_ADMIN`；虽有 `apt`、`uv`、CUDA 编译工具，但不能直接调用宿主容器 daemon。需要评估 rootless/vfs builder 或外部构建环境。
- 已验证镜像 config：`linux/amd64`、Ubuntu 22.04 base、CUDA 12.9.1、Python 3.12、工作目录 `/vllm-workspace`，入口为 `/bin/bash -lc "sleep infinity"`。
- 从镜像 layer 29 的 dist-info 实测版本：PyTorch `2.11.0+cu129`、Triton `3.6.0`、vLLM package `0.11.2.dev278+gdbc3d9991`（环境变量覆盖版本为 `0.11.2.dev278+v7tpotbench`）、Transformers `5.8.1`、Tokenizers `0.22.2`。
- GLM 定制运行时的 source commit/设计版本线称为 v0.19.0，但镜像内 Python package metadata 显示 `0.11.2.dev278+gdbc3d9991`。这是定制分支的版本标识差异，报告中必须并列记录，不能把二者当作同一字段。
- 额外 sparse MLA native extension `stage50_sparse_mla_m1_splitmerge_final_ops.so` 位于 layer 29，大小 342,336 字节，SHA256 为 `cb549aca...8625`。
- 直接 local clone 因跨所有者 upload-pack 的 safe.directory 限制失败；使用 `git bundle` 成功保留完整历史并 clone，bundle 自检声明 complete history。
- 本地恢复候选已建立为 `glm52_oscar_vllm/`，`main` 初始 commit 精确为 `bfd727e11b0e501bab0a4a943d92ba5ea3b2f980`。
- 已只应用镜像中的 4 个 runtime patch；候选中的 4 个文件 SHA256 全部与镜像有效源码逐字节匹配，diff 规模为 227 行新增、21 行删除。
- 新候选使用 `uv` 创建 Python 3.12.3 `.venv`；4 个 runtime patch 文件均通过 `py_compile` 和 `git diff --check`。
- 按 `.pre-commit-config.yaml` 固定的 ruff 0.14.0 定向检查发现镜像 patch 的 4 个既存问题：`toy_proxy_server.py` 三个 E501，`compiler_interface.py` 一个 SIM102。为保留 baseline 的逐字节镜像一致性，阶段 0 不修改这些行；后续新开发代码仍须通过 lint。
- pre-commit hooks 已安装，但首次从 GitHub 初始化 ruff hook 在 `index-pack` 阶段停滞；已改为从清华 PyPI 安装同版本 ruff 直接验证。
- 已创建基于已验证镜像 tag 的 `docker/Dockerfile.glm52-baseline`；构建时会对 4 个 runtime source 和 7 个 native extension 执行 SHA256 fail-closed 检查。
- 已创建 `recovery/baseline_manifest.json`、两份 SHA256 清单、构建脚本和中文恢复说明；JSON 解析、shell 语法、source 清单和 7/7 native extension 清单均通过验证。
- Docker build context 已新增排除 `/.git`；既有规则已排除 `.venv`、build 和 `vllm/*.so`，因此构建会保留 base image 中已验证的原生扩展而不是从工作树覆盖。
- 当前内核允许 unprivileged user namespace；已在任务容器内安装 buildah `1.33.7`。
- `BUILDAH_ISOLATION=chroot buildah --storage-driver vfs info` 实测成功，buildah 识别为 rootless、vfs store；但 vfs 对 34.75GB/31-layer 镜像可能产生过高的层复制开销，需优先验证 fuse-overlayfs。
- 已安装 fuse-overlayfs 1.13 并创建标准 `/dev/fuse`；直接 overlay、显式 mount program 和额外 user/mount namespace 三种方式均因 Kubernetes mount permission/propagation 限制失败。
- overlay 路径已按三次失败规则停止；后续使用 `vfs + chroot`，将 graphroot 放到任务专用 NFS 目录以避免占满容器 overlay。
- image manifest 记录构建方式为 `docker_create_cp_commit_from_v2_stable`，即以固定 base image 创建容器、复制 3 个 patch 后 commit，并非从 Dockerfile 完整重建；这是阶段 0 需要补齐的可复现性缺口。
- `glm52_speed_up_v2_stable_8th/source/vllm_glm52_v1/` 只有 2 个文件、73,839 字节，是部署局部 patch，不是完整源码。
- 找到外部完整源码候选 `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1`，大小约 2.0GB，包含 vLLM 标准源码、tests、CMake 和 requirements；其 Git commit/干净度及与镜像一致性仍待核验。
- 外部完整源码候选的 Git HEAD 精确为设计固定 commit `bfd727e11b0e501bab0a4a943d92ba5ea3b2f980`，分支为 `glm52-speed-up-v1`，tracked files 为 4,705。
- 该外部源码不是干净工作树：有 6 个已跟踪文件被修改，`git describe` 为 `bfd727e11-dirty`。涉及 sparse attention indexer、MLA backend/indexer/Triton kernel 和 GPU model runner，必须审计并与镜像层逐文件比对后才能冻结。
- 6 个未提交修改的 diff 规模为 626 行新增、39 行删除，内容主要是 `VLLM_PREFILL_SHAPE_BUCKET*` tracing/padding 实验，与阶段 0 的稳定基线冻结并非同一主题。
- 外部部署 patch snapshot 中与上述脏文件同名的 3 个文件 SHA256 也与当前工作树不同，说明存在多个运行时版本；镜像逐层内容必须作为最终判据。
- 外部源码 `origin` 是 NFS 本地路径 `/nfs/AE/zhanghong/workflow/vllm_a/./vllm_fp8_v7_tpot`，不是可供协作复现的 GitHub 远端。
- 外部仓库根 `AGENTS.md` 要求所有 Python 命令使用 `uv` 和 `.venv/bin/python`；后续进入新代码线开发时必须遵守。
- manifest 中另一个外部 source snapshot `/nfs/AE/zhanghong/workflow/vllm_a/glm52_speed_up_v2_stable/source` 只有约 1.7MB，仍是 patch 资产。
- NFS 当前可用空间约 2.8TB；容器 overlay 可用约 337GB，足够在当前任务范围内复制源码和解包必要镜像内容。
- 主仓库远端为 `https://github.com/XiancaiTian/oscar-glm52-a800.git`；当前 submodule 固定 OSCAR commit `41ebcd...` 和 OSCAR-vLLM commit `d458d296...`。
- 主仓库为 public；功能分支 `feat/glm52-model-load` 已推送，远端与本地均指向 `75f30615eaaa2e94723e570c62d3e1cbe64b9658`。
- 设计文档固定 GLM 定制 vLLM commit 为 `bfd727e11b0e501bab0a4a943d92ba5ea3b2f980`。
- 设计文档固定已验证镜像为 `192.168.14.129:80/ae/vllm_openai_glm52:v0.19.0-v2-stable-2p1d-usagefix-20260625_114616`，镜像 ID 为 `sha256:d6faf4d3a5f7f3800a745f8aea15884c881ea20b1100993d83ca8c4f985bd7a5`。
- 设计文档要求阶段 0 优先找外部完整源码，不可用时从镜像 `/opt/vllm_glm52_v1` 提取；若完整源码无法恢复则 fail closed。
- 2026-07-24 固定的
  `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-staticgate-e154-H001-nfs`
  是旧工程 checkpoint；其结果只作为历史证据保留。
- Shawn 于 2026-07-24 明确授权当前机器所有可见 GPU 均可使用；实际分配前仍须在可见范围内连续两次检查空闲状态。

## 技术决策

| 决策 | 理由 |
| --- | --- |
| 主线采用 GLM v0.19.0 基线优先、分层回移 OSCAR | 设计文档已冻结路线，避免同时迁移 GLM/FP8 MoE/DSA/MLA |
| 新代码线目标路径为 `glm52_oscar_vllm` | 保持两个现有资产为只读参考，并形成可追溯独立仓库 |
| 不隐式触发 brainstorming | 项目 `AGENTS.md` 明确规定仅 Shawn 显式请求时使用 |

## 问题与处理

| 问题 | 处理 |
| --- | --- |
| 根目录缺少持久计划文件 | 已建立独立根级计划，不改动 `oscar_vllm/` 的历史计划 |
| Docker CLI 缺失 | 核验本地 image tar，检查当前容器边界和替代解包工具；阶段 0 镜像重建能力仍待解决 |
| 外部完整源码所有者与当前容器用户不同，Git 默认拒绝访问 | 使用一次性的 `git -c safe.directory=<精确路径>` 做只读核验，不污染全局 Git 配置 |
| 外部完整源码 HEAD 正确但工作树有 6 个未提交修改 | 先保存 diff 和逐文件 SHA256，并与镜像内容/patch 资产比对；禁止直接把脏工作树宣称为固定 commit |
| 当前容器缺少 `jq` | 对单行 Docker archive manifest 使用 `rg -o`/`sed`，避免为一次性审计安装额外依赖 |
| 镜像相对固定 commit 还有未在最终 image manifest 列出的 `compiler_interface.py` 差异 | 检查 base image/部署脚本/历史资产和内容 diff，将其纳入完整 runtime patch 清单 |
| 设计建议的新 GitHub 仓库当前不存在 | 先建立本地恢复候选；推送和加入 submodule 前由 Shawn 确认仓库名称与可见性，或提供已创建远端 |
| 当前容器无 runtime socket 且无 `CAP_SYS_ADMIN` | 评估 rootless/vfs builder；若不可行，阶段 0 的实际 image rebuild 需要外部有构建能力的环境 |
| fuse-overlayfs 无法越过 Kubernetes mount 边界 | 三次不同方式均失败后停止；使用可工作的 vfs store，不再重复 overlay 尝试 |
| 已验证镜像的历史 runtime patch 不完全满足当前 ruff | baseline 恢复优先保持逐字节一致；如后续单独格式化，必须作为新 commit 并重新验证，不覆盖恢复基线 |

## 最新边界与构建能力

- Shawn 明确要求不得修改外部文件，特别是 `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1`。该路径及其他项目外目录后续严格只读；所有源码、构建上下文、OCI layout 和导出镜像只写入 `/nfs/AE/txc/oscar-glm` 或任务临时目录。
- 2026-07-24 只读复核外部仓库：HEAD 仍为 `bfd727e11b0e501bab0a4a943d92ba5ea3b2f980`，仍只有原先记录的 6 个 tracked 修改文件，没有出现新的状态项。
- 已在当前任务容器安装 `skopeo 1.13.3` 与 `umoci 0.4.7`。二者可直接搬运和组合 OCI blob/layer，不依赖 Docker daemon，也不需要 overlay mount，适合作为当前受限 Kubernetes 容器中的实际镜像重建路径。
- `skopeo` 已成功把 34.75GB Docker archive 转为项目内 OCI layout，实际占用 16GB；基线 OCI manifest digest 为 `sha256:1b8e808e44e7ef50be0cc8ff874597c37e09a2a428f06092ab570e6079086415`。
- `umoci insert 0.4.7` 不能用于本次候选层：无论源 payload 位于 NFS 还是本地 `/tmp`，生成层的未压缩长度都为 81,465,737 字节，最后一个 19,849 字节文件只写入 19,456 字节，固定缺少 393 字节。blob SHA256 与 gzip 流校验虽通过，但 GNU tar 和 Python `tarfile` 都判定逻辑 tar 截断。
- 上述两个失败 tag 仅位于本项目 `artifacts/phase0-oci` 中，未导出、未发布，也不会作为候选结果。后续改用 GNU tar 生成完整标准层，再按 OCI image-spec 更新 config、manifest 和 index。
- 标准 GNU tar/gzip + OCI descriptor 构建已成功：candidate manifest 为 `sha256:2fdfbe865aecc01eee15a01fcce58bf7581244dbbc53cbe3ef0e0cce44bc489d`，config 为 `sha256:58a853ee730c263968dcc6b76401e85e4b510a140777c4ffc791747edd8ea42d`。
- 新 source layer digest 为 `sha256:352d47f649171770e32edb7e1112e8a31f6a5aead0f6160a30ad3a8eec6659c3`，diff ID 为 `sha256:e9287018ae318195cd2bdc8fdd88b5c6fd7f85c2c44db453d891c075f538ccf8`，大小 33,253,726 字节。
- source layer 共 5,256 个 tar 成员和 4,711 个文件；文件内容与权限逐一匹配。独立第二次构建得到相同 digest、diff ID 和大小，可重复构建检查通过。
- candidate 的 31/31 个 base layer descriptor 与已验证 OCI 相同；新增层不含 `.so` 和 whiteout。已展开 rootfs 上 4/4 runtime source 与 7/7 native extension SHA256 均通过。
- `umoci unpack` 约 72 分钟后因 NFS xattr/元数据扫描耗时过长被终止，命令完成状态记为未通过；这不覆盖已实际取得的 11/11 文件内容 SHA256 结果。
- 本地恢复构建提交为 `0288235f2b93563c13e6c3750c5797fccbaba70d`；公开代码快照为相同 tree 的 `53d8be94f6038e10ab0c344f706c5ffe66a555b8`。
- 阶段 0 中文报告已写入 `docs/experiments/2026-07-24-phase0-source-recovery.md`。章节编号 1–8 连贯，未发现交叉引用错误。
- 阶段 1 第一次 GPU 检查：8 张可见卡均为 NVIDIA 苹果800-SXM4-80GB，显存使用 0MiB、利用率 0%，无 compute process。
- 2026-07-24T10:42:18Z 在间隔 60 秒后完成第二次检查，8 张卡仍全部为 0MiB、0% 利用率、无 compute process，满足连续两次空闲要求。
- 固定模型目录存在，共 141 个 `.safetensors`、152 个顶层普通文件，总大小 462,858,376,495 字节；`config.json`、`generation_config.json`、`tokenizer.json` 和 `tokenizer_config.json` 均存在。
- 当前 Pod 只暴露 CUDA 12.9.1 和 8 张 GPU，但缺少目标镜像的 `/opt/vllm_glm52_v1`、`/opt/fp8_speed_up_v4_venv` 与 `/opt/glm52_speed_up_v1_stable`，不能把当前 rootfs 直接宣称为固定候选镜像环境。
- 通过 Kubernetes API 查询当前 Pod spec/image 的尝试被 RBAC 403 拒绝；停止该无权限路径。
- SSH 到 `192.168.16.17` 受现有配置影响回连当前容器的 27868 端口，仍没有 Docker CLI；停止宿主 Docker 路径。
- 按 `AGENTS_misc.md` 的“提供环境本身是容器时直接配置”规则，已从项目内展开 rootfs 直接使用固定 venv/source，无需写入当前容器 `/opt`。
- 固定环境 import 验证通过：`sys.prefix` 指向候选 rootfs 的 `/opt/fp8_speed_up_v4_venv`，Torch/Triton/Transformers/Tokenizers 均来自该 venv，`vllm._C` 来自候选源码树；CUDA 未初始化。
- 模型 config 实测架构为 `GlmMoeDsaForCausalLM`：hidden size 6144、78 hidden layers、64 attention/KV heads、qk nope/rope head dim 192/64、v head dim 256、KV LoRA rank 512。
- DSA index 几何为 32 heads × 128 dim、top-k 2048、frequency 4；MoE 为 154 routed experts、每 token 8 experts、1 shared expert，量化配置为 dynamic FP8 e4m3。
- checkpoint index 含 72,117 个 weight map entries，引用 141 个 shards，metadata total size 为 462,831,219,288 字节。
- safetensors 文件名/大小清单 SHA256 为 `f4c0129c708e86e06f3b63dedd4a9b74e02869257df5d2f0426138d9726eb969`；index SHA256 为 `e9767570b7b56aa97759c11aec7a81fbe7b84f56d751977788b0195c3a35d983`。
- 首尾 shard 的 1MiB head+tail SHA256 分别为 `c959a8255e71884530c4e196fd70ef13d9c47057123b6b709552510a44181cab` 和 `f22834194f8d8edbc2a6f5e0f81ba1f7988392aafb3295faa0935801f840db76`。

## 阶段 1 全量评测与阶段 2 准备

- official_v4 全量运行使用冻结 manifest、2,360 样本、并发 8 与仅将 code timeout 固定为 900 秒的 runtime config，于 2026-07-25T17:50:15Z 启动。
- 2026-07-25T18:00:15Z 至 2026-07-26T10:40:34Z 的 10–1010 分钟节点均已写入 `progress_10min.log`；600 分钟时完成数为 1060，已越过 manifest 的 IFEval 边界并进入 GSM8K；610/620/630/640/650/660/670/680/690/700/710/720/730/740/750/760/770/780/790/800/810/820/830/840/850/860/870/880/890/900/910/920/930/940/950/960/970/980/990/1000/1010 分钟时分别为 1080/1120/1160/1180/1220/1240/1280/1320/1340/1380/1420/1440/1480/1500/1540/1560/1600/1620/1660/1700/1720/1760/1780/1820/1840/1880/1900/1940/1980/2000/2040/2060/2100/2120/2160/2180/2220/2260/2280/2320/2340；8 张 苹果800 显存占用约为 79,901–79,941MiB，运行进程持续存活。
- runner 标准输出存在缓冲，进度文件在前三个节点只能记录 `completed unknown/2360`；60/70/80/90/100/110/120/130/140/150/160/170/180/190/200/210/220/230/240/250/260/270/280/290/300/310/320/330/340/350/360/370/380/390/400/410/420/430/440/450/460/470/480/490/500/510/520/530/540/550/560/570/580/590/600/610/620/630/640/650/660/670/680/690/700/710/720/730/740/750/760/770/780/790/800/810/820/830/840/850/860/870/880/890/900/910/920/930/940/950/960/970/980/990/1000/1010 分钟节点分别为 40/40/60/60/80/80/80/100/100/100/120/120/120/140/140/160/160/180/200/220/240/240/260/280/300/300/320/340/360/380/380/400/420/440/440/460/480/500/540/580/600/640/660/700/740/760/800/820/860/900/920/960/980/1020/1060/1080/1120/1160/1180/1220/1240/1280/1320/1340/1380/1420/1440/1480/1500/1540/1560/1600/1620/1660/1700/1720/1760/1780/1820/1840/1880/1900/1940/1980/2000/2040/2060/2100/2120/2160/2180/2220/2260/2280/2320/2340，不把服务请求数直接当作最终评分数。
- 2026-07-26T10:40:33Z 至 `10:41:13Z` 服务保持 Running=8、Waiting=0、生成吞吐为 11.2–15.2 tokens/s，健康检查返回 HTTP 200，两个正式进程持续存活且错误扫描为空。
- runner 最终完成 2,360 条并写入 2,360 行预测；summary 为 2,353 条 `scored`、7 条 `request_failed`，已评分样本 accuracy 为 `0.19932001699957502`，duration 为 `60881.98892402649` 秒。
- 7 条失败 ID 精确为 `gsm8k:001311` 至 `gsm8k:001317`，全部错误均为客户端 `ReadTimeout(... read timeout=300)`；服务日志错误扫描为空，孤立请求结束后服务恢复 Running=0 且健康接口为 HTTP 200，因此证据不支持模型服务崩溃或 evaluator 评分失败。
- 分 benchmark 实际结果：GSM8K 1,312/1,319 scored、262 条正确、accuracy `0.19969512195121952`；IFEval 541/541、145 条正确、`0.2680221811460259`；LiveCodeBench 175/175、12 条正确、`0.06857142857142857`；MultiPL-E 325/325、50 条正确、`0.15384615384615385`。
- 首轮 predictions、failed cases、summary、benchmark summary、runtime config 与 runner log SHA256 分别为 `fbc69f74a2d0176a50a6c11f5da71aee93e1088827fee726c1d0f4f1492ecae2`、`e390f7129361eda2e88207d18721dc6b9ee2507c928c6dbb722e8bcd43254264`、`2996c8ba6502c6902cfa65356bc01043bdbda0b8ec1acaa4114620ad19ce31b8`、`6583c2f8a51bf1f7c2156f62341f0f103b162c32b9ca121b0c50d0b75a0376b8`、`8e0beeb16a6c66182551ae75fa8e4694e4e8cefda205d8c3c778c909c24a22c5`、`78e7fc42a87ca8af71641d46061dd59cb70472b8d5aeeaf08d1f13e491c7aa94`。
- 已准备项目内 `scripts/phase1/retry_official_v4_failures.sh`：在启动前验证固定 runner、依赖 lock、服务健康、首轮统计、7 个失败 ID/错误原因及 prompt hash；补跑只把 math timeout 从 300 秒提高到 900 秒，成功后在独立目录按精确 ID 替换首轮失败行并重新计算 summary/hash，绝不覆盖首轮证据。
- 精确补跑实际用时 `168.74860620498657` 秒，7/7 为 `scored`、request failure=0、accuracy 0.0；predictions、summary 与 runner log SHA256 分别为 `9bfc340ce84f1668843ced1a669c5d3bb07b127809535e79fc51d473e1c4c541`、`1f015c8226bf636cf4a66cf963f53b128387b1ee2ca09be033d68af99a6efed2`、`89435399acf2e21055b81d1aa0783f693743a168f05de4c47dab655b4b0f8ea1`。
- 首版合并门禁实际拒绝生成结果，原因不是 runner 完成顺序：固定 runner 会按 manifest index 恢复顺序。原因是完整 manifest 有 2,361 行，多出的唯一一行是 `wikitext2_perplexity:test`；正式 accuracy 命令显式只选 GSM8K、IFEval、LiveCodeBench v6、MultiPL-E 共 2,360 行。
- 修正后的独立合并工具显式冻结上述四个 accuracy benchmark，同时要求唯一排除项恰为 WikiText‑2 PPL，并验证 source/retry 唯一 ID 集、prompt hash、7 个替换 ID 与全量 scored 状态。
- 代码推送后生成的正式目录 `merged_full_retry_20260726T1112Z` 通过 validation：2,360/2,360 scored、request/extraction failure 均为 0、469 条正确、accuracy `0.19872881355932204`；GSM8K/IFEval/LiveCodeBench v6/MultiPL-E accuracy 分别为 `0.19863532979529946`、`0.2680221811460259`、`0.06857142857142857`、`0.15384615384615385`。
- 正式 predictions、failed cases、summary、benchmark summary、task type summary、merge provenance、validation SHA256 分别为 `68a3d0b184929286aeef6b690b109ebe9a84bbcc8a41bf89af1493c408ad3e75`、`9cf215421e8e0d5fbc0932c5b7004927d2c6b36b70ba48ac055c6364e720a6fe`、`f8f52510e10d861450a51f7b8ee6cb559ac8040108257e1a6fd0912ea84dc9e2`、`c7a5754e2f9eb893bd9878253eb45a29b505aa1b477c2e229c8844a50bf2ac9b`、`f7e0c5ed18ca2719fbf90442cff5bff823a8e374eae32771a999360126b8b016`、`4f3c6488d5e5f7f182fda3a72283149df248a6ad5ffa2de77673cf02b9f4a693`、`06cc30a34d877ee4f39e743158d7042a90faa36876fe8a94e11a2da4413ec432`。
- accuracy 服务退出后无残留 vLLM/runner 进程，8 卡为 0MiB、0%；PPL preflight 再间隔 60 秒验证两次 8/8 空闲，并通过 OCI/source/native/model/suite 全部固定指纹。
- WikiText‑2 PPL runner 于 2026-07-26T11:03:51Z 以 TP=8、eager、BF16、max length 2048、stride 512、batch 8 启动；141/141 shard 全部加载，权重读取 224.65 秒、模型加载 236.68 秒、每卡模型内存 55.94GiB。
- 2026-07-26T11:13:51Z 的 10 分钟节点为 8 卡 79,581MiB、98%–100% 利用率，runner 与 EngineCore 存活、错误扫描为空。
- WikiText‑2 validation 最终通过：输入 289,709 tokens、评分 289,708 tokens、563 个窗口、mean NLL `2.0402180131829573`、PPL `7.692286035848967`，评分阶段耗时 `368.10103392601013` 秒。
- PPL summary、perplexity results、validation、runner log SHA256 分别为 `a0b1643bdf6ed110a3f90bde55f76bdd66d91dba718b7755b23c5d594986a794`、`8c470884b9c971c1c7f95a290d4750abd48878ed849a29756a8793b11f3e72ed`、`17e740fe35da046c5b8e4ca2f1edab916dfc216d4cd5dbc8e3e5c434341feab2`、`d6b4e34e08971fc172e8eacb8eec4a027176cac91ce3b71710bca8a9888fb6f1`；进程退出后 8 卡均为 0MiB、0%。
- 阶段 1 已满足服务、请求、2,360/2,360 accuracy 评分、1/1 PPL 评分、环境与命令证据完整的出口条件；中文报告为 `docs/experiments/2026-07-26-phase1-native-baseline.md`。
- 当前固定 `TRITON_MLA_SPARSE` 路径可直接取得每层 `kv_c_normed`（压缩 KV，512 维）、`mqa_ql_nope`（吸收后的 query，512 维）及 `_v_up_proj` 前的 attention 输出（512 维），可在不修改 DSA 和不构造 full attention 的前提下捕获共享潜空间 calibration 统计量。
- 阶段 2 采用单个每层共享正交矩阵 `R`：score 侧旋转 `cR` 与 `q_absR`，value 侧历史聚合后再乘 `R^T`；prefix/recent 保持未旋转，history 使用旋转 INT2，并在一个全局 softmax 中合并。
- 项目内 ignored worktree `artifacts/worktrees/glm52-shared-calibration` 已加入共享潜空间 reference、4×2-bit pack/unpack、非对称 INT2 量化/反量化、mixed attention、FP64 未中心化 covariance 累积、trace 归一化和特征分解 rotation。
- reference/covariance 共 14 项 pytest 全部通过；ruff 0.14.0、format check 与 `git diff --check` 全部通过。隔离分支 commit 为 `507653c3d3dba18f13b5f81807a06191116e3a38`，已推送到 `origin/feat/glm52-shared-calibration`。
- 只读 capture 已接入原生 `MLAAttention.forward_impl` 的 `_v_up_proj` 前位置，仅在 `VLLM_OSCAR_MLA_CAPTURE_CONFIG` 显式设置时启用；输入 `latent/query/value_output/DSA indices` 均只读，未启用时直接返回。
- capture 对每层按固定 token budget 截断；每个 token 以确定性轮转方式抽取一个本地 query/value head，GPU FP32 累加 Gram matrix、按阈值合并到 CPU FP64，并用固定种子抽取小型 holdout/DSA 样本。TP rank 0 统计共享 latent，所有 TP rank 分别统计本地 query/value。
- capture 首轮测试发现输出 schema 的 `value_samples` 键同时承担计数与张量，已改为独立的 `value_covariance_samples`；修复后 reference/calibration/capture 共 17 项 pytest、Python 语法、ruff 0.14.0、format 和 diff 检查全部通过。
- capture commit `9c3b8401d` 已推送到 `origin/feat/glm52-shared-calibration`；本次提交只有 3 个代码/测试文件，未提交 capture 数据、模型、日志或缓存。
- rotation artifact 合约使用 `manifest.json` 与独立 `rotations.pt`：manifest 固定模型 config、checkpoint manifest、专家映射、calibration commit/manifest、seed、层数、latent rank、group、alpha、逐层 clip、prefix/recent 和 tensor SHA256。
- artifact 加载默认 fail closed：缺层/多层、错 shape、非正交、非有限值、format version、runtime 模型身份、layer 顺序或 rotation tensor SHA256 任一不匹配都会拒绝。25 项定向 pytest 与语法/lint/format/diff 门禁全部通过。
- artifact commit `2100083b5` 已推送到 `origin/feat/glm52-shared-calibration`；提交仅含 2 个代码/测试文件，实际 rotation tensor 继续留在项目 artifacts 且不进入 Git。
- TP payload 合并和固定 holdout 搜索已实现：rank 0 提供共享 latent covariance，所有 TP rank 的 query/value covariance 均参与合并；rotation 为 eigenbasis、归一化 Hadamard 与 bit-reversal permutation 的确定性组合，固定搜索 alpha `{0.25, 0.5, 0.75}` 和 clip `{0.92, 0.94, 0.96, 0.98, 0.99}`。
- rotation/clip 搜索 commit `67deb6b9e` 已推送到 `origin/feat/glm52-shared-calibration`；当时全套 30 项定向 pytest 与语法/lint/format/diff 门禁通过。
- calibration manifest 构建器按源样本 ID 确定性划分 train/holdout、固定 source revision 和文件 SHA256、排除 official_v4 完整 prompt 与重复 chunk，并严格满足每个 split/category 的 token 配额。
- manifest 构建器 commit `cd7fcc946` 已推送；全套 33 项 pytest、ruff、format 与 diff 门禁通过。
- 顺序 prompt runner 会逐条校验 manifest token 数与服务端 `usage.prompt_tokens`，固定 `max_tokens=1`，任一 HTTP/usage/token 差异即失败；响应证据与 summary 仅写入项目 artifacts。
- fit 工具会 fail closed 加载 8 个 TP rank 的 78 层 train/holdout payload，合并后执行固定 alpha/clip 网格搜索并写入完整指纹 artifact；长 fit 每 10 分钟输出 heartbeat。
- prompt/fit 工具已推送至 `da4e2756ab4fb41d30ff669b150e9d6d24368d0e`，全套 35 项 pytest、ruff、format 和 diff 门禁通过。
- 阶段 1 完成后，正式源码 worktree 已从 `53d8be94f6038e10ab0c344f706c5ffe66a555b8` 切换到已推送的 calibration commit `da4e2756ab4fb41d30ff669b150e9d6d24368d0e`；工作区干净，候选 rootfs 保持只读不变。
- OpenWebMath 固定为官方 revision `fde8ef8de2300f5e778f56261843dab89f230815`。由于当前环境访问 Hugging Face Xet CDN 时 TLS 失败，curl、`hf_hub_download` 和 wget 三条直接下载路径均停止；随后从官方 datasets-server 获取 0–299 行，冻结为项目内 300 行、2,800,069 字节、SHA256 `39d245cad8279eb6301f0a311471d311fb78557c9df5441d8ce5860422938b80` 的只读输入。
- LongBench 固定为官方 revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，采用 `gov_report_e`、`multi_news_e`、`qasper_e`、`lcc_e`、`repobench-p_e` 五个只读文件；各文件实际 SHA256 已写入 calibration 配置。
- tokenizer 固定为模型目录 `tokenizer.json`，SHA256 `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`。
- 开发版 manifest 实际为 20 行、250,907 字节、50,000 tokens，SHA256 `cde883393c6d0d6f4224f57db321b7ac2f12888f90d7db9f30deee543b1c25da`；summary SHA256 为 `e13329218df30f1a763597e5ec78ffd9f0c32aeb362c3f19d2d816570717e6ea`。
- 正式版 manifest 实际为 292 行、4,570,560 字节、1,000,000 tokens，SHA256 `3a183cba6f013424c5e422da5eb9ab8a9d4d975ddd0dbabd1b930b1d02e876b5`；summary SHA256 为 `f0e323f46b8e8b201651d6f258885d6908a417d150d08c1ad561c8bd06a691bf`。
- 开发版与正式版均独立构建两次且产物 SHA256 一致。正式版重新审计得到 292 个唯一 entry、235 个唯一源样本、0 个 split collision、0 个重复文本 hash、0 个 official_v4 完整 prompt hash 交集、0 个文本 hash/token 计数错误；各 split/category token 数逐项等于冻结配额。
- 初始 fit 配置固定 900,000 train tokens、100,000 holdout tokens、TP=8、78 层、latent rank 512、group size 128、prefix/recent 64/256、holdout reservoir 4,096 行和 DSA 512 行；专家映射 hash 基于 checkpoint index 中排序去重后的 `experts.<integer>` 清单，实际为 `89430944bd88801fc3deed040dd0d54e1db229f5a5cdd66c9682ef862313c983`。
- phase-2 launcher 已实际验证 6 个 vLLM 原生 `.so` 的项目内只读 symlink 与候选 rootfs 内容一致，第 7 个 sparse MLA `.so` 按 rootfs 绝对路径核验，7/7 SHA256 通过；候选 Python 从主仓库 source 路径导入 `vllm` 和 `vllm._C`，`torch.cuda.is_initialized()` 为 false。
- 正式 submodule 已在阶段 1 完成后切到 calibration commit `da4e2756a...`。
- 正式 train capture 服务通过两次 8/8 GPU 空闲检查与 141/141 shard 加载，于 2026-07-26T11:25:53Z ready；权重读取 48.31 秒、模型加载 61.37 秒、每卡模型内存 55.95GiB。
- train prompt runner 实际完成 256 条、900,000/900,000 prompt tokens、256 completion tokens，耗时 `532.1056863907725` 秒；responses SHA256 `a356733b72c0fc3e3eec32c3752a046b7a78a133fff1bc3ad223c347be37b7a2`，summary SHA256 `77490a263e65c794369a0e301c7da51c76dd8577bc87a741a2a482dd480bca87`。
- train capture 实际生成 624 文件、2,783,307,114 字节，每个 TP rank 恰好 78 层；文件名/大小 manifest SHA256 为 `22cd1b54e07d57a1487eac51a282761e0b33bf33a7f866fe67d2f5940547f66f`。服务停止后无残留进程，8 卡为 0MiB、0%。
- 正式 holdout capture 服务通过两次 8/8 GPU 空闲检查与 141/141 shard 加载，于 2026-07-26T11:42:07Z ready。
- holdout prompt runner 实际完成 36 条、100,000/100,000 prompt tokens、36 completion tokens，耗时 `88.02139441482723` 秒；responses SHA256 `221edb175d2f95ede7e205b68a3e45d83002d077ee2f73fd958c9736959b0ac3`，summary SHA256 `cef3c602be27415af467ac83230d217cc75e4e2d5cd205e53356a7e5189da943`。
- holdout capture 生成 624 文件、13,272,777,066 字节，每个 TP rank 78 层；文件名/大小 manifest SHA256 为 `6bf169175c3a4579d77a740a78a5f97a77fc683602d9ef500223cd4ba7713ba6`。服务退出后无残留进程，8 卡为 0MiB、0%。
- 首次 fit 未进入 covariance 合并：配置 `layer_name_template` 为 `model.layers.{layer}.self_attn`，而 capture 使用 `MLAAttention.layer_name`，实际文件与 payload 层名均为 `model.layers.{layer}.self_attn.attn`；第一层 `torch.load` 即 fail closed，未生成 artifact。
- 固定配置已改为 `.self_attn.attn`，并新增 `validate_calibration_capture_paths.py` 在 fit 前比较完整期望/实际路径集合。旧模板对 train split 复现 624 missing 与 624 extra；新模板实际验证 train/holdout 各 624 文件通过。修复未改 capture 内容，也无需重跑 GPU capture。
- 隔离 worktree 的 pre-commit 首次初始化再次停在 GitHub hook `index-pack`；中止后只损坏了 linked worktree index，正式源码 worktree、HEAD 和 Git 对象库均正常。已先记录 5 个新增文件 SHA256，再用 `git read-tree HEAD` 重建该 worktree index并按哈希重新暂存，最终提交前的手工门禁全部通过。
- 修复后的正式 fit 已完成：alpha `0.25/0.5/0.75` 的归一化 holdout loss 为 `0.025037897150672388`、`0.027723928782624297`、`0.031391672548347495`，因此选择 `alpha=0.25`。
- 78 层逐层 clip 只选择 0.92/0.94，数量分别为 61/17；逐层归一化 loss 范围为 `0.0004133854263186087` 至 `0.04478203689244448`。
- 正式 artifact 目录为 `artifacts/phase2/20260726T1200Z_rotation_fit_v2`；manifest/rotation/search summary SHA256 为 `df30fbb90bfafaef787cfe73bd55c9179ac2acb83d2a23ed977a8378d7c19926`、`0a966da2e480559b698e4347ba29f402f5eb9fe335781ac41f8c27086a72808e`、`9dfe16a88a3c46dfff8e10b62a8f47a7bc716996c6e3d51b3b6b7daeb212792e`。
- 正式 loader 实际加载 78/78 个 `512×512` rotation 并通过运行时身份、hash、finite 和正交性门禁；`RᵀR-I` 的最坏逐元素绝对误差为 `1.6274684710992915e-08`。
- 阶段 2 中文报告为 `docs/experiments/2026-07-26-phase2-calibration.md`；阶段出口已通过，大型 capture、日志和 rotation tensor 均保持本地 ignored。

## 阶段 3 隔离准备

- GLM‑5.2 共享 latent 的 BF16 成本为每层每 token `512 × 2 = 1,024` bytes；INT2 history 为 128 bytes packed data，加 4 组 × FP32 scale/zero 共 32 bytes，因此每层每 token 合计 160 bytes，history-only 理论压缩率为 6.4×。
- 主 MLA cache 的实际 head size 为 576，其中只有 512 维共享 latent 可压缩，64 维 RoPE 必须保持 BF16。模型实际 78 层 `indexer_types` 为 `shared=57`、`full=21`；21 个原生 DSA cache 每层每 token 为 132-byte uint8 layout。
- 联合 CPU planner 显式预留每请求 64-token prefix 与 256-token recent 的 BF16 latent 行，变量预算同时覆盖 INT2 latent history、BF16 RoPE 和原生 DSA cache；allocated bytes 与 unused bytes 必须精确回到总预算。
- 标准 vLLM block table 继续承载全逻辑序列的 RoPE 与 DSA，需扣除 block ID 0 的 null block；INT2 history 使用独立 page namespace，不丢失一页给 null block。
- allocator 为每个请求固定 `hp_row`、prefix/recent 连续起点、history page IDs、logical length、generation 和 cache version；recent 使用环形地址，history 使用逻辑位置到 page/slot 映射。
- OOM 在修改 request length/page/version 前完成新 page 原子分配；实际单页预算测试中，从 336 增长到 337 tokens 的第二页分配失败后，metadata、logical length 与池守恒均保持不变。
- finish、abort、preemption 都释放 history pages 与 BF16 行；reuse 会复用物理行但生成新的 generation，陈旧 worker metadata 会被拒绝。
- scheduler 通过 request-keyed `WorkerCacheMetadata` 下发 generation/version、稳定 BF16 行与 history page IDs；worker 在 finished/preempted 时释放 mirror，并拒绝无释放事件的 generation 跳变和倒退 version。
- raw per-layer tensor 被精确切分为 packed INT2 data、FP32 scale、FP32 zero、BF16 prefix、BF16 recent 与 BF16 RoPE 六个无重叠 view，实际 size 必须与 planner 完全一致。
- 14GiB、`max_num_seqs=16` 的联合计划为 36,216 blocks、36,215 usable blocks、579,440 logical token slots、36,216 history pages；fixed/history/RoPE/auxiliary 分别为 408,944,640/7,231,610,880/5,785,288,704/1,606,252,032 bytes，总分配 15,032,096,256 bytes，剩余 289,280 bytes。
- 同预算理论 native 为 10,142 blocks、162,256 slots，联合理论容量比 3.5711468297×；这只是 CPU 预算模型结果，不是 GPU 实测。
- 独立分支 commit `e75a40a294bd3127667f34ebffce8119a8ac0f3a` 已推送；116 项定向 pytest 全部通过，强制离线 scheduler 回归另有 68 项通过、28 项仅缺少 LLaVA 仓库配置，ruff/format/compile/diff 均通过。
- 阶段 2 出口通过后，正式 submodule 已从 calibration commit `da4e2756a...` 切换到远端已发布的 cache planner commit `e75a40a29...`；原隔离 worktree 已在同一 commit detach，未复制或重新生成代码。
- 正式代码线首轮定向回归为 104 passed、12 failed；失败均是源码树 `.venv` 缺少已安装 vLLM metadata，导致通用 `VllmConfig` 无法自动推断 device。加入项目内候选 rootfs 的已安装 metadata/dependency 路径后，12 项独立通过，完整结果为 116/116 passed、27.11 秒，CUDA 未初始化。
- 完整 scheduler 文件强制离线复验为 68 passed、28 failed、31.22 秒；28 项均在构造 `llava-hf/llava-1.5-7b-hf` 配置时失败，没有 OSCAR 或通用 scheduler 断言失败，也没有下载模型。
- 14 个阶段改动 Python 文件 compileall 和 Git diff check 通过；13 个无既存债务文件通过 ruff 0.14.0/format。`gpu_model_runner.py` 的当前与阶段基线都有同样 6 个既存 lint/format 问题，本阶段插入点未引入新规则错误。
- 阶段 3 中文报告为 `docs/experiments/2026-07-26-phase3-cache-planner.md`；阶段出口已通过。

## 阶段 4 隔离准备

- shared-latent rotation 使用 Triton FP32 输入与 IEEE FP32 accumulator；rotation 输出只与当前写入/查询 batch 成比例，不创建与完整历史长度成比例的 BF16 临时 cache。
- INT2 store 按 128 维分组执行 percentile clipping，4 个 2-bit index 以 low-bit-first 顺序打包到 1 byte；每个 512 维 token 存 128-byte data、4 个 FP32 scale 和 4 个 FP32 zero。
- BF16 store 根据 token logical position 与 final sequence length 只写最终 prefix/recent 分区；recent 物理地址为 `(position - prefix_tokens) % recent_tokens`。历史 token 不写 BF16 pool。
- recent demotion 以 scheduler 给出的 logical position、稳定 `hp_row`、history page/offset 为输入，执行 gather → rotation → clipping/quantize/pack；history dequant 只用于 oracle 与 kernel 单测。
- mixed sparse decode 对 DSA-selected token IDs 直接分类到 prefix/recent/history；每个 split 保存原 latent BF16 accumulator、旋转 latent history accumulator 与 LSE，跨 split 统一归一化后才对 history 乘 `R^T`，避免在不同 basis 中提前相加。
- mixed sparse prefill 复用同一三段式 kernel，通过显式 query-to-request 映射共享请求 cache metadata，并以 query logical position 屏蔽未来 token；query 级中间 accumulator/LSE 不共享。
- CPU Triton interpreter 首轮捕获两个真实问题：split merge 的标量 mask 类型不兼容，以及 BF16 `tl.dot` rotation 产生无效大值；分别改为分离的 `tl.where` 和 FP32 IEEE rotation 路径。
- 修复后，实际 512 latent rank、4 groups、两 splits 的 store/demotion/dequant 与 mixed decode interpreter smoke 通过，输出和 LSE 均 finite，对 PyTorch oracle 最大绝对误差为 `2.384185791015625e-07`，且子进程未创建 CUDA context。
- 同一 CPU interpreter 实际执行 query positions 2/4 的 causal multi-token prefill，输出与 LSE 均 finite，对 PyTorch oracle 最大绝对误差为 `2.980232238769531e-07`。
- 分支 `feat/glm52-oscar-kernels` 的五个 WIP/测试 commit 为 `9861f2398...`、`5d220497a...`、`18c83e4b9...`、`b722b7975...` 和 `8ac7b9d97...`；61 项测试通过，22 项 CUDA 测试因 baseline 占满 GPU 被显式门禁跳过，静态检查通过。
- Stage 5 接线审查发现上述 mixed kernel 尚未把 64 维原精度 RoPE 分量加入 attention logits；若继续接线会得到可执行但数学上不完整的结果。kernel 工作树已加入 query RoPE、RoPE cache 和标准 block table 地址映射，并把默认 scale 从 `1/sqrt(512)` 修正为 `1/sqrt(512+64)`。
- 带非零 RoPE 数据的 CPU Triton interpreter 已实际执行 512+64 维 decode 与 causal prefill；两者相对扩展 PyTorch oracle 的最大绝对误差均为 `2.384185791015625e-07`。完整无 CUDA套件为 61 passed、22 skipped，commit `8ac7b9d97...` 已推送；苹果800 CUDA 仍未验证。
- 当前结论仍只是代码候选和 CPU interpreter 里程碑，不是 Stage 4 验收。尚无 SM80 cold compile 或 苹果800 actual launch；interpreter 的 finite/数值误差不能冒充 苹果800 实测。
- 阶段 3 出口通过后，正式 submodule 已切换到远端发布的 kernel commit `8ac7b9d97e41a8f1c89bf3258a9c672a417263be`；原 kernel worktree 在同一 commit detach，且该 commit 的 ancestry 已验证包含 Stage 3 `e75a40a29...`。
- 苹果800 首轮 cold-cache 实际为 23 passed、1 failed、63.33 秒；唯一失败测试对 `recent[0,0]` 先要求 NaN、随后又要求等于 position 320 的值，而 position 64/320 按 `(position-prefix) % recent` 都映射 slot 0，两个断言不可能同时成立。
- kernel 已实际通过其余 21 个 CUDA 门禁。BF16 ring 测试拆成 history-only 与最终写入两次调用，以分别证明 positions 64/65 不写入和 positions 319/320/321 映射到 255/0/1；修复只改测试，不改 kernel。
- 修复后定向 苹果800 test 为 1/1 passed、3.61 秒，ruff 0.14.0、format 与 diff check 通过；源码 commit `c50d86b34643c9fba0ae1df28a671c04fd107a41` 已推送。pre-commit 的 actionlint hook 初始化停滞已中止，本次单文件提交使用等价手工门禁。
- 第二个全新 Triton cache 的正式 苹果800 结果为 24/24 测试节点、22/22 CUDA 门禁通过、56.60 秒；完整 `tests/oscar_mla` 在 CUDA 开启下为 83/83 passed、34.31 秒。
- 正式单卡 Triton cache 生成 284 文件、19,620,324 bytes；定向与完整日志 SHA256 为 `91cdbc6eab614c5d86a910038e0a30d8658468ee76110aa4a72807b6058f6264`、`73a7c83c638c4e4279e7f3c659a54c8bf7699897f274970aedd354d8f8e15b1e`。
- TP=8 rank-local cold-cache 在 8/8 张 苹果800 上通过；每 rank 为 24/24 passed、284 cache files、19,624,132 bytes，耗时 84.50–87.00 秒，总计 192/192 节点和 176 次 CUDA kernel 执行。
- TP=8 smoke 结束后 8 卡均为 0MiB、0% 且无 compute process；阶段 4 中文报告为 `docs/experiments/2026-07-26-phase4-a800-kernels.md`，阶段出口已通过。

## 阶段 5 隔离准备

- `oscar_mla_int2` 已作为显式 cache dtype 接入配置层，generic torch dtype 仅以 uint8 作为 heterogeneous layout marker；真实物理布局仍由三段式 spec/views 定义。
- 激活条件 fail closed：必须关闭 vLLM prefix caching、模型必须走 sparse MLA，且 backend 必须为 `TRITON_MLA_SPARSE`；其他组合在分配 cache 前拒绝。
- `MLAAttention.get_kv_cache_spec` 对实际 GLM‑5.2 几何构造 512 latent、64 RoPE、group 128、160-byte history slot、64/256 prefix/recent 的 `OscarMLAAttentionSpec`。
- 配置/spec 与既有 cache integration 共 8 项测试通过；完整 `tests/oscar_mla` 为 63 passed、22 CUDA skipped。commit `cc2655657...` 已推送至 `feat/glm52-oscar-integration`。
- RoPE correctness commit 已合入集成分支并推送为 `3ce04538e...`；合入后的完整 `tests/oscar_mla` 仍为 63 passed、22 CUDA skipped，相关静态门禁通过。
- worker mirror 现在事务式保存每次 scheduler 更新前的 logical length，并按请求顺序生成稳定 HP row、独立 history page table 与精确 incremental demotion 列表；320→337 token 回归实际得到 logical positions 64–80、page IDs `9×16 + 11×1` 和 offsets `0–15 + 0`。
- OSCAR sparse metadata 已携带 exact seq lengths 与上述 batch ownership；不兼容的 decode preparation fastpath 和 CUDA graph padding 会被拒绝。完整无 CUDA套件为 64 passed、22 skipped，commit `ef5476705...` 已推送。
- runtime artifact 由 `VLLM_OSCAR_MLA_ROTATION_ARTIFACT` 与 `VLLM_OSCAR_MLA_RUNTIME_EXPECTATION` 显式绑定；模型/权重/专家映射指纹、78 层几何、layer mapping、rotation tensor SHA256 或 window 不匹配都会在服务启动期失败。commit `715669a4c...` 已推送。
- runtime cache update 明确先 demote 旧 recent，再写 current history 与最终 BF16 partition，避免 ring overwrite；RoPE 通过标准 slot mapping 单独保持 BF16。mixed read 保留 DSA request-local token IDs，分别使用标准 RoPE block table 和 OSCAR history page table。
- CPU mock 已实际确认 demotion/write 调用顺序、17-token current-history 直写和 `[0,64,320]` local DSA IDs；CPU Triton interpreter 实际执行 RoPE slot store。完整无 CUDA套件为 69 passed、23 skipped，commit `f8e5afbbf...` 已推送。
- 配置层现已对首版未实现的执行模式 fail closed：V2 model runner、非 eager、CUDA graph、speculative decoding、decode context parallelism 和 dual batch overlap 均在启动配置验证期拒绝；完整无 CUDA 套件为 73 passed、23 skipped，commit `d62571ae8...` 已推送。
- 首版 fail-closed 范围继续覆盖 prefill context parallelism、KV transfer 与 KV offloading。干净子进程复测同时发现 interpreter smoke 缺少仓库 `PYTHONPATH`，修复测试入口后完整套件为 76 passed、23 skipped，commit `fa7ed930b...` 已推送。
- 真实模型 EngineConfig 预检确认 eager 并不会自动关闭 asynchronous scheduling；默认配置现会按预期 fail closed，显式加入 `--no-async-scheduling` 后得到 `TRITON_MLA_SPARSE`、`oscar_mla_int2`、TP=8、PP=1、32K、prefix cache=false、CUDA graph=NONE，且未初始化 CUDA。完整无 CUDA 套件为 77 passed、23 skipped，commit `42639391d...` 已推送。
- CPU Triton interpreter 已实际使用非连续 rotation stride 并通过；新增 苹果800 门禁会比较 one-shot 337 与 chunked 320→337 的 history data/scale/zero、prefix/recent 最终分区逐字节一致。完整无 CUDA 套件为 77 passed、24 skipped，commit `2a49fe1c2...` 已推送；新增 CUDA 项仍待 GPU 释放后执行。
- INT2 reference 现已显式覆盖 constant、narrow、random 与 outlier 分布；新增窄分布/outlier 用例均满足 finite、pack/unpack 一致和 clipped group 半步长误差界。完整无 CUDA 套件为 79 passed、24 skipped，commit `a5e7e5c65...` 已推送。
- mixed-tier PyTorch reference 现同时返回输出与自然对数域 LSE，并保持原输出接口兼容。CPU Triton interpreter 实测 decode 输出/LSE 最大绝对误差为 `2.384185791015625e-07`/`0.0`，causal prefill 为 `2.384185791015625e-07`/`5.960464477539063e-08`；完整无 CUDA 套件仍为 79 passed、24 CUDA skipped，commit `f426ab5a5...` 已推送。
- 两请求 CPU Triton interpreter 使用独立 HP row、history page 与 RoPE block table，实测输出/LSE 最大绝对误差为 `2.384185791015625e-07`/`1.1920928955078125e-07`，并验证 `-1` DSA padding 安全屏蔽；完整套件仍为 79 passed、24 CUDA skipped，对应 commit `5445a8286...` 已推送。
- runtime mock 进一步验证两个不同长度请求的 metadata 接线：request indices `[0,1,1]` 映射到局部 query positions `[320,335,336]`，且 RoPE block table、history page table 与 HP rows 保持各自 ownership；完整套件为 80 passed、24 CUDA skipped，commit `7bac6d7e9...` 已推送。
- 苹果800 条件门禁新增 batch 4/8 多请求隔离，每请求使用独立三段式与 RoPE block table，并固定 output/LSE 的最大和平均误差报告及预先设定容差；当前完整套件为 80 passed、26 CUDA skipped，commit `c762b4aee...` 已推送，新增两项尚未在 苹果800 运行。
- Stage 5 已完成代码级 artifact/metadata/write/read 接线，但这些新增路径仍未在 苹果800 上 launch，也未跑服务；不能宣称 `oscar_mla_int2` 已运行。
- 阶段 4 的 BF16 ring 测试修复已同步到 integration 分支；无 CUDA套件为 80 passed、26 skipped、26.69 秒，commit `caa0818540280c16b949b6646f9ba116cdaa59f2` 已推送。
- 正式 submodule 已切换到同一 integration commit；原 integration worktree 在该 commit detach，下一步是 26 项 苹果800 门禁，尚不能宣称端到端通过。
- 阶段 5 首轮正式 苹果800 完整套件实际为 105 passed、1 failed、74.71 秒。唯一失败发生在 kernel launch 前：当前 PyTorch 的 QR 输出已经是非连续张量，原测试执行 `.T` 后反而得到连续张量，因此未满足测试自己的 stride 前置条件。
- 该问题不构成首轮 kernel 正确性通过或失败的证据；最小修复是先将 QR 输出 `.contiguous()` 再转置，从而稳定得到数值相同、stride 非连续的正交 rotation。修复后的定向 苹果800 one-shot/chunked kernel 已实际通过，为 1 passed、4.20 秒。
- 单行测试修复已作为源码 commit `0f1bd5b308da9217ba72a5ba68ca5e9590b7a2bd` 推送；pre-commit 的 actionlint 初始化再次停滞，因此在 ruff、format、py_compile、定向 苹果800 kernel 和 diff 门禁全部通过后使用 `--no-verify` 提交。
- 主仓库 submodule 指针已由 commit `751f20cd62d8aaf413973dcc9a2d9098ab90bcf0` 发布；正式 retry 前两个仓库均干净且与远端对应分支 SHA 一致。
- 全新 Triton cache 的正式 Stage 5 苹果800 retry 为 106/106 passed、70.31 秒；26 项 CUDA 条件门禁全部实际执行。cache 为 316 文件、22,280,982 字节，测试日志 SHA256 为 `8265be65743788cb02272ed862e731153670cafc7bf59df60406450e73255cae`。
- 正式 retry 前两次检查与测试结束后均确认 8 张 苹果800 无占用；苹果800 kernel/runtime 单元门禁已通过，但这仍不等价于 TP=8 模型服务或 32K 端到端通过。
- Stage 5 服务入口复用候选 rootfs 的 Python/依赖/native extension，同时将 `PYTHONPATH` 首项切到项目内 integration 源码；dry-run 实际确认 `vllm_source` 与 `vllm_C` 均来自 `glm52_oscar_vllm`，CUDA 未初始化。
- 入口显式固定 TP=8、PP=1、32K、eager、`TRITON_MLA_SPARSE`、`oscar_mla_int2`、无 prefix/speculative/CUDA graph 和同步调度；rotation artifact 的 78 层、哈希、模型/checkpoint/专家映射身份及 64/256 window 均通过 fail-closed verifier。
- dry-run 首轮只因 CLI 校验脚本漏导入 `os` 失败；补齐后完整 retry 通过。这不构成服务启动证据，正式 TP=8 仍必须在代码提交推送和连续两次 GPU 空闲检查后执行。
- Stage 5 正式 smoke 将串行覆盖短请求、>320-token prefill、384-token decode 与近 32K，再并发覆盖 8 个 >320-token 请求；入口会在请求成功后强制提取三段式容量、artifact、write、demotion、DSA mixed read 和无完整 BF16 history 的日志证据，缺任一证据即失败。
- Stage 5 服务入口已由主仓库 commits `7ba83078...`、`e4b0ce0f...` 发布；后一提交只保存新脚本 executable mode，大型 artifact 与运行日志均未进入 Git。
- 正式 run ID `20260726T130111Z_oscar_tp8` 的 formal preflight 全部通过；13:01:55Z 与 13:02:58Z 两次确认 8/8 张 苹果800 空闲。该结果只授权进入服务启动，不代表模型已加载。
- 首次正式服务的 engine config 已实际确认 TP=8、32K、`oscar_mla_int2`、eager 和同步调度，但 8 个 worker 在权重加载前一致因 artifact 正交校验跨设备报错退出。
- 根因是 vLLM worker 设置了 rank-local 默认 CUDA device；`torch.load(..., map_location="cpu")` 保持 rotation 在 CPU，而未显式 device 的 `torch.eye` 被创建到 `cuda:<rank>`。这是 artifact validator 的 device 假设缺陷，不是 artifact 内容、hash 或正交性失败。
- 将正交校验 identity 显式创建在 CPU 后，11 项 artifact/runtime 定向测试与完整 81 项无 CUDA 测试通过；正式 78 层 artifact 在 CUDA default-device 上也全部保持 CPU，且没有初始化 CUDA。
- 修复源码 commit `c3823fda2ed1d82f92c99275b6e128bac9ba6220` 已推送；下一次正式服务必须使用新的独立 run ID，保留首次失败目录不覆盖。
- 第二次正式服务已越过 artifact 并加载 141/141 shards；系统 cache 使权重读取缩短到 49.15 秒，模型加载总耗时 62.462097 秒、每卡 56.02 GiB。
- 实际三段式计划为 9,964 history pages、637,632 logical tokens：INT2 history 7.41 GiB、固定 BF16 prefix/recent 0.38 GiB、RoPE 5.93 GiB、native auxiliary 1.65 GiB、unused 0.0 GiB。相对原生 165,696-token capacity 的观测比值约 3.85，但服务未 ready，暂不作为最终实测容量验收。
- 第二次失败发生在 KV 初始化后的 dummy warmup：统一更新函数仍按原生 tensor 假设调用 `kv_cache.numel()`，而 OSCAR runtime 已将其重塑为 `OscarMLACacheTensors`。正确空门禁应检查 dataclass 的 `raw` backing tensor。
- empty-cache 门禁修复后 runtime path 5/5、完整 苹果800 CUDA 套件 108/108 通过；源码 commit `49db9142d7688d5b116ccd69fccfdf2e085818bf` 已推送。第三次启动必须固定该 SHA 并使用新目录。
- 第三次正式服务在 141/141 shards 后、三段式 planner 前的显存 profile 退出：OSCAR dtype 在该阶段仍搭配普通空 `Tensor`，并非 dataclass，因此按 dtype 直接取 `.raw` 不成立。
- cache 的真实生命周期是“分配前空 Tensor → 分配后三段式 dataclass”。空门禁必须按 runtime 对象类型选择 backing storage；只按 dtype 或只假设其中一种形态都会在另一阶段失败。
- 双生命周期修复以 `isinstance(..., OscarMLACacheTensors)` 分派，定向 runtime 6/6 与静态门禁通过；源码 commit `f852be0c830f5caf741f91e20bbe522f5d00d55f` 已推送。
- 该修复后的完整 苹果800 CUDA 套件为 109/109 passed、77.22 秒，26 项 CUDA 门禁全部实际执行；因此第四次正式服务可固定 `f852be0c8` 继续验证 profile、planner 和 warmup。
- 第四次服务在 OSCAR 代码执行前遇到部分 worker 的 CUDA driver 初始化失败；启动前后 GPU 均无占用，不能据此归因于 `f852be0c8`。在逐卡候选环境探针证明设备可初始化前，不应盲目修改源码。
- 随后的候选 rootfs Python 8 进程并行探针在 GPU 0–7 全部完成 CUDA 初始化与 tensor 分配，确认设备当前可用；因此按同一源码独立重试比修改代码更符合证据。
- 第五次启动证明双生命周期修复已越过分配前 profile，planner 重新得到 637,632 tokens；下一边界是分配后的 compile warmup，其 `attn_metadata=None` 是 vLLM `forward_impl` 已显式支持的正常状态。
- OSCAR update 在该 warmup 中不得写 cache，但真实请求只要存在 metadata，`oscar_mla` 缺失仍必须由 backend 抛错，不能把生产 ownership 缺陷静默为 fallback。
- custom-op 与 direct-call 现均在 metadata=None 时跳过 cache write；真实 metadata 继续进入 backend 的 `oscar_mla` 强校验。定向 runtime 8/8 通过，源码 commit `ef2bc0903af85a59b086fdd5dcff7916456163e5` 已推送。
- warmup 修复后的完整 苹果800 CUDA 套件为 111/111 passed、76.06 秒，26 项 CUDA 门禁全部执行；第六次正式服务可固定 `ef2bc0903` 继续验证 warmup 后的真实请求路径。
- `ef2bc0903` 的第六次正式 TP=8 服务已实际 ready：启动总耗时 180 秒，141/141 shards 加载完成，三段式 planner 得到 9,964 history pages 和 637,632 logical tokens，32K 理论并发 16.00。
- 串行短请求、506-token prefill、384-token 连续 decode、31,996-token 近 32K 及 8 个并发 498-token 请求全部 HTTP 200；这是首个同时覆盖 write、demotion、mixed-tier read 和近 32K 的端到端成功运行。
- 服务日志明确证明无完整 BF16 latent history，并记录 INT2 history 7.41 GiB、固定 BF16 prefix/recent 0.38 GiB、RoPE 5.93 GiB、native auxiliary 1.65 GiB；相对原生 165,696-token capacity 的实际容量比为 `3.8482039397450754`。
- 当前观测仍不完全满足设计第 8 节：backend 内已有 `oscar_write_calls`、`oscar_demotion_calls`、`oscar_read_calls`，但尚未向日志或 metrics 暴露最终精确值；planner 也尚未打印理论、填充和实际分配三种压缩率。下一改动应复用现有状态并保持最小侵入，不能仅凭首次触发日志冒充调用计数。
- 精确计数不能依赖进程退出钩子：上一轮 Ctrl-C 时 multiproc worker 在 `shutdown(timeout=0)` 路径被直接终止，没有可靠的 worker `shutdown()` 回调。现改为 rank 0 每个真实 model step 后输出当前聚合值，因此最后一条日志在 worker 被终止前已经落盘。
- 为避免每个 decode step 遍历整个模型，第一次计数时只扫描并缓存 78 个 `oscar_mla_int2` impl；随后汇总已有 Python 整数。日志同时报告 total、per-layer min/max，既给出精确计数又可检测部分层未走 OSCAR。
- runtime planner 的 allocated ratio 使用同一实际 cache budget 下“OSCAR logical slots / 原生 BF16+RoPE+index logical slots”，与仅 history latent 的 theoretical/padded 6.4× 分开报告；不能把不同显存预算下的 637,632/165,696 观测比直接冒充整体压缩率。
- 最终观测代码在全新 Triton cache 上完成 112/112 苹果800 回归，26 项 CUDA 门禁全部执行；因此下一次 TP=8 失败若发生在观测字段或真实计数门禁，可直接归因到服务集成边界，而不是未运行的 kernel 代码。
- 最终正式 TP=8 在 `7d317f1de` 上完成全部 12 个串行/并发请求；31,996+64-token 近 32K case 为 HTTP 200，服务日志没有 `ERROR` 或 `Traceback`。
- 最后一条聚合计数覆盖 78 层：store 51,246（逐层 657）、demotion 23,010（逐层 295）、read 51,246（逐层 657）。总数等于层数乘逐层值，证明不是少数层或首次触发日志冒充完整路径。
- planner 的 15.3689644821 GiB budget 中，BF16 prefix/recent 为 81,788,928/327,155,712 bytes，INT2 history 为 7,958,446,080 bytes，RoPE 为 6,366,756,864 bytes，native index/cache 为 1,767,693,312 bytes，unused 为 459,060 bytes；分配合计与理论逐字节一致。
- latent-only theoretical/padded 均为 6.4×；计入固定池、RoPE 和 index/cache 后，同预算 overall allocated ratio 为 3.5812365205×。跨阶段 637,632/165,696=3.8482039397450754× 还包含可用 cache budget 差异，不能作为纯压缩率。
- 跨阶段观测容量比为同预算理论值的 107.45461568139605%，高于设计的 95% 门槛；allocation byte error 为 0%，低于 5% 门槛。`BF16 history=absent` 和全部三类调用计数共同关闭阶段 5。
- OSCAR 专属精确计数当前落在日志而非 Prometheus metrics；这满足本阶段可审计门禁，但若生产监控要求专属告警，后续仍需单独增加 metrics export。
- Stage 6 在无 Docker socket/CAP_SYS_ADMIN 的现有容器中仍可生成标准 OCI：复用 phase 0 的 32 个 content-addressed 基础 blobs，只新增确定性 source+rotation layer 及新的 config/manifest/index；这不是目录 tar 冒充 image。
- 候选最后一层不含 `.so` 或 whiteout，因此 7 个已冻结 native extensions 继续来自 phase 0 基础层。独立 verifier 已在基础 rootfs 上重算 7/7 SHA256，而不是仅相信 manifest 声明。
- 正式 tag `glm52-oscar-a800-phase6-7d317f1de-df30fbb9` 对应 image ID `sha256:5ad30941...5f7c`、manifest `sha256:c2939feb...2ec9` 和 candidate layer `sha256:8ad9ace9...5225`。
- 候选层解包后 4,742 个 Git objects 的内容、symlink 和 executable mode 全匹配；3 个 rotation files 的 SHA256 也匹配。候选 overlay 已实际导入 vLLM `_C` 并加载 78 rotations，CUDA=false。
- 确定性第二次构建得到相同 layer/diff ID、config/image ID 和 manifest digest；报告 JSON 的 SHA 不同只因为其中记录了不同的 output layout 路径。
- 当前环境不能把 OCI rootfs mount 后 bind NFS 模型执行 `docker run`；后续使用“已验收基础 rootfs + OCI 最后一层解包”的等价 overlay，并在每轮运行前验证候选 digest。必须如实保留这项运行方式限制。

## 阶段 1 本地运行与评测入口

- 主仓库和 `glm52_oscar_vllm` 均已切换到 `feat/glm52-model-load` 本地功能分支。
- public 源码仓库的 `main` 和 `feat/glm52-model-load` 均已发布到 `53d8be94f6038e10ab0c344f706c5ffe66a555b8`。
- 主仓库已在 `04b9445cd05a914b1716479a6bbdced018ebc177` 接入该 submodule、更新阶段 1 source commit 门禁，并推送功能分支。
- 本地恢复历史包含 189,258 个 Git 对象、pack 约 185MiB；按 Shawn 最新要求不上传，保留在本地 `recovery/full-history` 分支。
- 远端冻结代码快照 commit 为 `53d8be94f6038e10ab0c344f706c5ffe66a555b8`，tree 仍为 `ca5d4f035591690fb95fb662c71480a28714a4d8`，与上传前 `0288235f2b93563c13e6c3750c5797fccbaba70d` 的 tree 完全相同。
- 候选 OCI 的 image revision 实际为 `fd3e0b3772e989cf0d0d73a3d19b252ab82e9cdd`，tree 为 `31faa41bfb082983fa853dc1c499b48db3a81d3d`；恢复构建资产对应 tree 为 `ca5d4f...`。运行时与公开仓库 commit 必须在阶段 1 manifest 中分字段记录。
- 首次全量 preflight 如实发现候选 rootfs 与仓库 HEAD 在 3 个 `recovery/` 文件上不同。改按 OCI label 指向的 `fd3e0b...` Git tree 验证后，候选 rootfs 的 4,711 个 tracked 文件内容、符号链接和 executable mode 全部匹配；不是通过忽略差异放宽门禁。
- 阶段 1 静态 preflight 已通过：OCI manifest/config/source layer、运行时 commit/tree、4,711 个源码文件、7 个 native extension、模型配置/几何/141 个 shards、official_v4 2360 个 accuracy 样本与 1 个 WikiText‑2 样本全部匹配冻结值。
- 模型 filename/size/mtime-ns manifest SHA256 为 `85b93d732896a3a82aa714c24210f2b4b5ad6273c3c0c764d36aad72583e591c`。
- official_v4 manifest SHA256 为 `4aec8ee85bee5eb73ce99c2009fcaedc79804bde1433f855fb77276ffccacfa5`；accuracy runner SHA256 为 `fc374ff4c4715e37d515d37aa794b3e649dc1710034d3355c69c21efa1a8aeff`；PPL runner SHA256 为 `eec6b1a4be99a068f80b2e9c0d684392f1bf1a669cae4fbc881581d6baa25668`。
- 启动命令已实际通过 vLLM CLI 解析，结果为 TP=8、PP=1、`TRITON_MLA_SPARSE`、native `kv_cache_dtype=auto`、32K、eager、chunked prefill、prefix cache=false、speculative config=null；解析过程未初始化 CUDA。
- 初版固定环境检查通过时实际调用的是候选 venv 的绝对 `/usr/bin/python3.12` 符号链接；脱离候选容器根后它解析为当前容器 Python 3.12.3。该结果能验证 venv 包和 `vllm._C`，但不能证明系统 site-packages 已隔离，现已由候选 rootfs Python 检查取代。
- 32K smoke 的 tokenizer 测试首次发现 Transformers 5.8.1 返回 `BatchEncoding`，不能直接用 `len()` 作为 token 数；修复为读取 `input_ids` 后，>320 请求本地模板长度为 506 tokens，32K 请求为 31,996 tokens。
- official_v4 独立 evaluator venv 已用 `uv` 和清华镜像创建在项目 `artifacts/` 下；原始 requirements 未声明 runner 实际导入的 `requests`、`nltk`、`absl-py`，已在项目 lock 中按实际安装环境冻结全部 20 个包。
- 正式 launcher 会在启动前再次间隔 60 秒检查 8 张 GPU，并拒绝未推送 commit、脏工作树、错误分支、变化的模型/数据/OCI 或已占用端口；长服务、accuracy、PPL 和单个长 smoke 均加入 10 分钟进度记录。
- 首次正式服务运行目录为 `artifacts/phase1/20260725T155732Z_native_tp8`；worker 在分配 GPU 显存和加载权重前退出，8 张 GPU 始终为 0MiB。
- 失败原因是固定 venv 中 `flashinfer-python 0.6.6` 与 `flashinfer-jit-cache 0.6.7.post3+cu129` 触发版本门禁；服务退出码为 1。
- 已验证部署入口 `glm52_speed_up_v2_stable_8th/scripts/deploy/lib/container_entry.sh` 实际导出 `FLASHINFER_DISABLE_VERSION_CHECK=1`，说明候选 launcher 遗漏了原生运行环境，而非需要修改外部环境或源码。
- `44dda4484ed0d960572d80fe5e5dd485938c2d9a` 已补齐固定部署入口的通用环境：FlashInfer 版本兼容、HND KV layout、V1 multiprocessing、禁用 symmetric-memory all-reduce、HF offline/cache 和禁止写 `.pyc`。
- 设置兼容开关后实际 `import flashinfer.comm` 通过，版本仍为 0.6.6/0.6.7.post3+cu129；该导入会初始化 CUDA，诊断进程退出后未保留 GPU 占用。
- 第二次正式服务运行目录为 `artifacts/phase1/20260725T160547Z_native_tp8`，记录主仓库 `5e291379b257f9c3abf584c17efe3a4c287d54ca`、源码仓库 `53d8be94f...` 和 runtime `fd3e0b377...`；两次 GPU 检查时间为 16:06:51Z 和 16:07:55Z，均为 8/8 空闲。
- 第二次启动已越过 FlashInfer 版本门禁和 8 worker/NCCL 初始化，并到达 `Starting to load model`；模型对象构造时 8 个 worker 均因 `ModuleNotFoundError: No module named 'flash_attn.ops'` 退出，服务退出码为 1，未进入 checkpoint shard 权重读取。
- 固定 venv 的 `pyvenv.cfg` 实际设置 `include-system-site-packages = true`。由于它在候选容器外直接运行，`sys.path` 中的绝对 `/usr/local/lib/python3.12/dist-packages` 指向了当前容器，而不是候选 rootfs。
- 当前容器存在顶层 `flash_attn` 4.0.0b8，但只包含 `cute` 子包，没有 `ops`；候选 rootfs 不含顶层 `flash_attn`。因此当前容器包使 `find_spec("flash_attn")` 错误返回真，触发候选源码中本应跳过的 `flash_attn.ops.triton.rotary` 可选导入。
- 同一泄漏还使可选 `triton_kernels` 从当前容器解析，并报告缺少 `SparseMatrix`；候选 rootfs 只有 vLLM 内置的 `vllm.third_party.triton_kernels`，没有顶层 `triton_kernels`。修复必须隔离当前容器系统包，而不是安装新包或修改外部环境。
- 候选 rootfs 自带 `/usr/bin/python3.12` 的实际版本为 3.12.13；候选 venv 的 `pyvenv.cfg` 也记录 3.12.13，而其绝对符号链接在容器外曾错误落到当前容器 Python 3.12.3。
- launcher 已改为直接调用候选 rootfs Python 3.12.13，并设置 `PYTHONHOME`、`VIRTUAL_ENV` 以及候选 source/venv/rootfs 的显式 `PYTHONPATH`。同一环境会继承到 multiprocessing worker。
- 修复后实际 dry-run 通过：4,711/4,711 runtime files、7/7 native extensions、141 shards、2360+1 suite samples 保持通过；Torch 2.11.0+cu129、Triton 3.6.0、Transformers 5.8.1、Tokenizers 0.22.2 与 `vllm._C` 均来自候选树，CUDA 未初始化。
- 修复后的 `find_spec("flash_attn")` 和 `find_spec("triton_kernels")` 均为 `None`；FlashInfer/JIT cache 解析为候选环境匹配的 `0.6.6`/`0.6.6+cu129`，不再是当前容器的 `0.6.7.post3+cu129`。
- launcher 已加入 fail-closed 包来源门禁：解释器必须是候选 rootfs `/usr/bin/python3.12`，且若发现顶层 `flash_attn` 或 `triton_kernels` 则在分配 GPU 前退出。
- 第三次正式服务运行目录为 `artifacts/phase1/20260725T162556Z_native_tp8`，主仓库为 `6d7a7bd7c35dad6290bf599ba5120109eaae8b3f`，源码仓库仍为 `53d8be94f...`。
- 第三次启动再次通过两次 8/8 GPU 空闲检查，141/141 个权重 shard 全部加载；权重读取耗时 2,337.66 秒，模型加载总耗时 2,393.23 秒、每卡模型内存 55.93GiB。
- 原生 KV cache 可用内存为 14.3GiB，容量 165,696 tokens；32,768 tokens/request 的理论最大并发为 5.06，engine profile/KV cache/warmup 耗时 162.94 秒。
- 服务于 `2026-07-25T17:14:53Z` 通过健康检查，launcher 记录启动总耗时 2,762 秒；8 个 rank 均确认 `TRITON_MLA_SPARSE`，苹果800 sparse indexer 使用 Triton fallback，FP8 linear/MoE 使用 Marlin。
- 四项原生 smoke 全部 HTTP 200：短请求 21+64 tokens、16.06 秒；>320 输入 506+18 tokens、3.56 秒；连续 decode 26+384 tokens、59.90 秒；近 32K 请求 31,996+64 tokens、24.28 秒。
- official_v4 首轮于 17:18:08Z 以并发 8 启动。前 8 条均为 LiveCodeBench v6，`max_tokens=4096`；在实测约 6.4 tokens/s 的单序列速度下，完整输出理论需要约 640 秒，超过 suite 的 600 秒 code timeout。
- 首轮 10 分钟时 KV usage 从 20.8% 同步降至 2.4% 并装入下一批 8 个 prompt，但服务只记录 2 个 HTTP 200，证明其余 6 个客户端请求已超时。该轮不可能满足设计要求的 request failure=0，已停止且不生成伪造的最终 predictions/accuracy。
- official_v4 manifest、样本、decoding 和 runner SHA 保持不变；评测入口改为每个 attempt 保存独立 runtime suite，只将 code timeout 从源配置的 600 秒固定提高到 900 秒，并同时记录源/运行时 config SHA256。该调整只扩大执行预算，不改变模型输入、输出上限或评分。
- 900 秒 runtime timeout 探针实际运行前 8 条 LiveCodeBench v6，耗时 638.53 秒；结果为 8/8 请求成功、8/8 `scored`、request failure=0，证明新的请求预算覆盖了首批 4,096-token 长生成。探针 accuracy 为 0.0，表示 8 条均未通过代码评分，不是请求或 evaluator 失败。
- 探针 runtime eval config SHA256 为 `8e0beeb16a6c66182551ae75fa8e4694e4e8cefda205d8c3c778c909c24a22c5`，predictions SHA256 为 `05390ba2d2964bd5d35fe505ed6550c51e10ee0f1fd0174023862125159296e2`，summary SHA256 为 `3a78a575a64dab0809535896be52e1dc63cdd39e073fa738290fc04c3b9fc2e7`。
- 正式 preflight 运行目录为 `artifacts/phase1/20260725T155457Z_native_tp8`；记录主仓库 `8f7be26a...`、源码仓库 `53d8be94f...` 和 runtime `fd3e0b377...`。
- 2026-07-25T15:55:27Z 和 15:56:31Z 两次 GPU 检查均为 8/8 苹果800、0MiB、0% 利用率、无 compute process；Driver 575.51.03、CUDA 12.9。

## 阶段 7 正式评测与存储故障恢复

- 候选 TP=8 服务已实际 ready 并连续运行约 12 小时 47 分钟；截至 `2026-07-27T04:34:45Z`，第二轮 official_v4 已确认有效完成 573/2,360，服务端 `abort/error/repetition` 均为 0。
- 服务退出根因不是模型、CUDA 或 OSCAR kernel：共享 `/nfs/AE` 达到 100% 使用率，监控 `tee` 写入十分钟日志时返回 `Disk quota exceeded`，`set -euo pipefail` 使 launcher 退出并清理 8 个 worker。
- 官方 runner 在全量结束前不增量写 `predictions.jsonl`；本轮文件为 0 字节。因此已经完成的 573 条只能证明运行进度，不能合并为精度证据，正式评测必须从 0 重跑。
- 本项目已移除两个不提交的大型可重建产物：34,754,212,352-byte 旧镜像 tar 和 13,272,777,066-byte Stage 2 holdout 原始 capture。候选 rootfs、Stage 6 OCI、rotation artifact、小型配置/日志/哈希与中文报告全部保留。
- `/dev/shm` 实测为 954 GiB 空闲 tmpfs。`ARTIFACT_ROOT` 覆盖只改变可重建运行输出位置，不改变模型、候选源码、runner、suite、prompt、生成参数、并发或评分；默认路径保持不变。
- tmpfs 静态 preflight 已完整通过。正式重跑前仍须提交推送代码并再次执行两次 8/8 GPU 空闲检查，不能把基础设施失败轮次当作正式结果。

## 2026-07-27 外部更新与 REAP checkpoint 切换

- 外部 `/nfs/AE/zhanghong/workflow/vllm_a/glm52_speed_up_v2_stable_8th`
  始终只读。同步前后按排除镜像、日志、报告、状态、缓存和二进制产物后的轻量文件树
  SHA256 均为
  `9e97a98a9f2e52aadd5d4509d4bb5f22c56c314158d3db603d7f774fb381fbf6`，
  证明本任务没有修改外部目录。
- 外部部署包的 10 个最新 Python runtime patch 已同步到项目内
  `glm52_speed_up_v2_stable_8th/source/runtime_patch_source`，其 manifest
  `sha256sum -c` 为 10/10 通过，AST parse 为 10/10 通过，部署 shell `bash -n`
  通过。约 34.75GB 的旧镜像 tar 没有复制，因为其 image ID、archive SHA256 和
  2026-06-25 内容均与阶段 0 已冻结基础镜像相同。
- 本地部署参考同步 commit 为
  `7d0dd73fa461ff74c30c6049a3a6b5b185d0a39f`，已推送；只包含 19 个小型
  code/config/doc 文件，最大文件 408,682 字节。
- 10 个 runtime patch 已通过真实 Git 三方合并进入 `glm52_oscar_vllm`。唯一冲突
  位于 `gpu_model_runner.py`：保留 OSCAR ownership metadata，同时合入 prefill
  shape bucket metadata；`dataclasses.replace` 会保留 `oscar_mla`。
- 合并后的定向测试为 2/2 passed；非 CUDA 完整套件为 86 passed、26 项 CUDA
  门禁未执行；随后在 苹果800 GPU 0 和全新 Triton cache 上完整执行为 114 passed、
  0 failed、0 skipped、77.01 秒。源码 commit
  `a3317695428819d41437b1cb144404b3bfc05a92`、tree
  `85619bd0ea0c71291a49f628fd4515b5fb70e9a1` 已推送。
- 当前测试模型已切换为
  `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001`。用户提供其
  GSM8K-full accuracy 为 86.35%；该值不是本轮 formal runner 实测。
- 新模型只读核验为 141 个 safetensors、72,117 个 index entries、总字节
  463,045,186,640；config/tokenizer/78 层/512 latent/154 routed experts 等几何
  保持不变。
- 新模型 index SHA256 为
  `f50217dadf6c58f8f84140003bd7fc3497e9338916e9865e23ea5342f2ac1ce2`，
  文件名/大小清单 SHA256 为
  `7096b908195ce880b113d07e15c78e57588b0e5eb0c2f77b0c63807eab806dd5`，
  文件名/大小/mtime-ns 清单 SHA256 为
  `83eefdf08de8f489bee6f1d5b1bf4d2f3452d42757b4df580e76a40c3646acaf`。
- 当前按 C locale 字典序计算的 expert token-set SHA256 为
  `c163c3f02089cfe7180bda0816cea59ce8f8bba0fe752b6dd4738d9c87b3da72`。
  新 checkpoint 的 index 和权重文件指纹与旧 checkpoint 不同，因此旧 baseline、
  rotation artifact、候选 OCI 和阶段 7 结果均不能继承；阶段 1–7 必须重跑，
  阶段 5/7 manifest 在新 artifact/OCI 就绪前显式 fail closed。
- 更新后的 Stage 1 静态 verifier 已实际通过：phase 0 OCI descriptor、4,711 个
  runtime source 文件、7 个 native extensions、新模型 141 个分片/几何/专家映射
  和 official_v4 2,360+1 个样本全部匹配冻结值。证据文件位于 tmpfs
  `/dev/shm/oscar-glm-reap-preflight/static_preflight.json`，不进入 Git。
- Stage 1 完整 dry-run 退出码为 0；固定候选 Python 3.12.13 实际解析到新模型
  路径、served name `glm-5.2-fp8-pruned-reap-e154`、TP=8、PP=1、32K、
  `TRITON_MLA_SPARSE`、native KV、eager、prefix cache=false 和
  speculative=null，且 CUDA 未初始化。`vllm` 与 `vllm._C` 均从最新
  `glm52_oscar_vllm` 源码树解析，基础 rootfs 只提供固定 Python 依赖和原生扩展。
- Stage 5、6、7 的旧输入分别以退出码 1 拒绝执行，且 Stage 6 未创建 output
  layout；这证明旧 rotation/候选身份在新 artifact 生成前不能被误启动。

## 2026-07-27 REAP 原生 baseline 实测

- 原生 TP=8 服务使用最新集成源码 commit
  `a3317695428819d41437b1cb144404b3bfc05a92` 与新 REAP checkpoint。141/141
  个权重分片完成加载，8 个 rank 均使用 `TRITON_MLA_SPARSE`。
- 四项正式 smoke 全部 HTTP 200：短请求 21+64 tokens、>320 输入 506+5
  tokens、连续 decode 26+384 tokens、近 32K 输入 31,996+64 tokens。
- official_v4 正式轮次 2,360/2,360 全部 `scored`，服务端恰好记录 2,360 个
  accuracy HTTP 200，request failure、unsupported 和 extraction failure 均为 0。
- overall 为 883/2,360，accuracy `0.37415254237288137`。分项为 GSM8K
  666/1,319（`0.5049279757391963`）、IFEval 161/541
  （`0.2975970425138632`）、LiveCodeBench v6 12/175
  （`0.06857142857142857`）、MultiPL-E 44/325
  （`0.13538461538461538`）。
- predictions SHA256 为
  `c3f0b6345d7030f639e436cb19129ff60db78df35211489f49867e7949492579`，
  summary SHA256 为
  `23ddda280962066ef064e18a8f1ac15669ecc283aab877c61581fe8c8c6b4811`；
  实测运行时长 53,208.72071003914 秒。
- 用户提供的 GSM8K-full 86.35% 与 official_v4 GSM8K 子集的冻结
  prompt/template、生成参数和评分口径不同，不能直接比较或互相覆盖。
- GSM8K 只读诊断显示 717/1,319 个输出达到 1,024-token completion 上限，其中
  549 条被判错；未触顶样本的冻结评分为 498/602（`0.8272425249169435`）。
  另有 101 条冻结判错样本在仅提取首个数值并去除单位/货币符号后与 gold 一致；
  该诊断口径为 767/1,319（`0.5815011372251706`），不是正式 evaluator 结果。
  因此当前差异同时包含输出触顶/重复和严格答案规范化影响，不能归因于单一因素。
- 服务停止后间隔 63 秒完成两次 GPU 检查，8 张 苹果800 均为 0 MiB、0% 且无
  compute process，可进入后续 PPL。
- `run_native_ppl.sh` 原先仍读取 phase 0 OCI 内旧源码并把输出/cache 固定到 NFS
  `artifacts`。PPL 必须改用当前 `glm52_oscar_vllm`，支持 `ARTIFACT_ROOT`/
  `CACHE_ROOT`，并与成功的 baseline 服务统一离线/cache/sparse MLA 环境后再正式
  执行。
- 修正后的 PPL preflight 在主仓库
  `905f98c7d5e03f5878b2601974a3bb79f55ece1f`、源码
  `a3317695428819d41437b1cb144404b3bfc05a92` 上通过；两次间隔 63 秒的检查均为
  8/8 GPU 空闲。
- 新 REAP 原生 WikiText‑2 为 1/1 `scored`、289,708 evaluated tokens、563
  windows、mean NLL `1.8863319523410782`、PPL `6.595132997244041`。summary
  SHA256 为
  `29a93a4b2a3427a49be0a0ee19f54cac8fb08393e17dc013ed46f86e69330c4a`。
- 阶段 7 相对 PPL 上限为 `6.792986987161362`；overall accuracy 下限为
  `0.35915254237288137`。单项下限已写入新阶段 1 报告。

## 2026-07-27 REAP 阶段 2 入口审计

- `configs/phase2/calibration_fit_initial.json` 已绑定当前源码
  `a3317695428819d41437b1cb144404b3bfc05a92`、新模型 config/index 和 expert
  mapping 指纹；calibration manifest SHA256 仍为
  `3a183cba6f013424c5e422da5eb9ab8a9d4d975ddd0dbabd1b930b1d02e876b5`，
  数据集和配额无需改变。
- 根仓库 submodule pointer 与当前源码 HEAD 均为
  `a3317695428819d41437b1cb144404b3bfc05a92`，阶段 2 的 published-commit
  fail-closed 门禁可继续使用。
- 旧 rotation artifact 的 manifest 记录 `89430944...c983`，当前配置记录
  `c163c3...da72`。阶段 2 preflight 证明两者分别来自 version sort 和 C locale
  字典序，而不是专家 ID 集合变化；旧 artifact 仍因 checkpoint index
  `e97675...d983` 与当前 `f50217...1ce2` 不同而不能复用。
- `run_calibration.sh` 原先把 train/holdout capture、cache 和 fit 输出硬编码到
  NFS `artifacts/phase2`。已做最小修改，统一支持 `ARTIFACT_ROOT` 和
  `CACHE_ROOT`；冻结 calibration manifest 仍从项目内原路径只读加载。
- 首次新模型 train preflight 在 GPU 启动前因 expert mapping mismatch
  fail closed。根因是脚本使用 `sort -uV`，而阶段 1 verifier/配置使用字典序；
  已改为 `LC_ALL=C sort -u` 并在配置中明确 canonicalization。
- 首次正式 train serve 观察到两个不同 RUN_ID 的进程树并发通过未 ready 的端口
  检查。二者均未加载权重、GPU 仍为 0 MiB，已全部停止并作废对应轮次。
- 阶段 2 serve 现使用 `flock` 对同一 `ARTIFACT_ROOT + HOST + PORT` 做非阻塞互斥，
  防止初始化窗口内不同 RUN_ID 同时占用 8 卡或写入各自 capture。
- 新 REAP train 正式轮次
  `/dev/shm/oscar-glm-reap-stage2/phase2/20260727T2230Z_reap_calibration_train_tp8_final`
  已完成：256 条请求全部 HTTP 200，精确覆盖 900,000/900,000 prompt tokens 和
  256 completion tokens，耗时 `326.51599755790085` 秒；responses SHA256 为
  `c54c76b792e2bebb63c40701aa27d379f476996ceec7b14a07a638014bdd8ba6`。
- train capture 为 624/624 个 `.pt` 文件，即 8 个 TP rank × 78 层，总字节
  2,783,307,114。每层 captured tokens 均为 900,000；rank 0 独占共享 latent
  covariance，8 个 rank 各保存 score/value covariance，符合 fit loader 的真实 TP
  合并语义。`capture.sha256` 文件自身 SHA256 为
  `5a300083b9f4efdc54ea0886e00a4dabf1889fd899549b98f3791ac32e327907`，
  metadata validation SHA256 为
  `e0cc528ac014b8dc29201339a8cfbb9601b8e00a7f85dab4b8af0c0d054a5eda`。
- 一次人工诊断错误地要求每个 rank 都保存 latent covariance，因而报告失败；按
  capture 写入器与 fit loader 的真实契约重验后通过。该错误不来自正式 runner，
  未覆盖或改变 capture，也不作为正式失败证据。
- train 服务完成后正常停止，8 张 GPU 均为 0 MiB、0%，无残留 vLLM/EngineCore
  进程。该轮总耗时不足 10 分钟，因此没有触发 10 分钟心跳要求。
- 新 REAP holdout 正式轮次为
  `/dev/shm/oscar-glm-reap-stage2/phase2/20260727T2241Z_reap_calibration_holdout_tp8_final`；
  published preflight 记录根仓库 `8a16407c49a3e74daaf1732610db13a559876f68`、
  源码 `a3317695428819d41437b1cb144404b3bfc05a92`，两者当时均已推送且工作区干净。
- holdout 服务完成 141/141 shard 加载，权重读取 55.72 秒、模型加载
  `68.378808` 秒；36/36 请求全部 HTTP 200，精确覆盖 100,000 prompt tokens 和
  36 completion tokens，耗时 `41.85091549158096` 秒。responses SHA256 为
  `43a87f069c5005a95ad32c7aa5c983dfe08d3fcf79e9fbdac7a5ddbcb3baa832`，
  summary SHA256 为
  `e89cc2ec2448dbd4d71ed519afc99782890f4269cf28b832f8434c410cd2c4f8`。
- holdout capture 为 624/624 个文件、13,272,777,066 字节。全部 8 rank ×
  78 层均有 100,000 score/value covariance 样本、4,096 行 latent/query/value
  reservoir 和 `512×2048` 的 DSA 样本；rank 0 独占 100,000 样本的共享 latent
  covariance。完整内容清单 SHA256 为
  `9bb125fe127db3f9d895f1a762cc80f87d79a67e51b2a3ea6f395cc85ff2bb1e`，
  metadata validation SHA256 为
  `8fa969bc604f4a8f5d74d1518562b27363fa20d2b50a9254d206e11307687caf`。
- holdout 日志中有 36 个 POST HTTP 200、0 个非 200、0 个
  `ERROR`/Traceback/request failure；服务退出状态为 0，8 张 GPU 最终均为
  0 MiB、0%，无残留进程。总耗时不足 10 分钟，未触发 10 分钟心跳。
- 新 REAP rotation fit 使用 train/holdout 各 624 个文件，路径集合门禁通过。
  alpha `0.25/0.5/0.75` 的 holdout loss 分别为
  `0.026186010882512642`、`0.02872817461999513`、
  `0.032144202654983196`，最终选择 `0.25`。
- 新 artifact 含 78/78 个 `512×512` FP32 rotation；clip ratio 为 0.92 的有
  61 层，为 0.94 的有 17 层。逐层 loss 范围为
  `0.00041062589551025104` 至 `0.046030287445025325`。
- 正式 runtime loader 使用独立 `ArtifactExpectation` 验证当前模型 config、
  checkpoint index、expert mapping、层数、latent rank、group size 和窗口，全部
  通过；最大正交误差为 `1.6403759683925045e-08`，且 CUDA 未初始化。
- artifact manifest SHA256 为
  `0275043c070c9127354997374e9bca1c70fe1308a7b2d057f992fadedef868e5`，
  rotations SHA256 为
  `256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d`。
  81,811,997-byte `rotations.pt` 已逐字节固化到项目 ignored 路径
  `artifacts/phase2/20260727T2253Z_reap_rotation_fit_final`，不进入 Git。
- 当前源码的阶段 2 定向非 CUDA 回归为 33 passed、0 failed、5.90 秒。Stage 5
  活动 manifest/launcher 已绑定新 artifact 路径与哈希；中文报告为
  `docs/experiments/2026-07-27-phase2-reap-calibration.md`。

## 2026-07-27 REAP 阶段 3 回归

- 当前源码 commit `a3317695428819d41437b1cb144404b3bfc05a92` 相对原阶段 3
  commit 的 5 个交集文件为 cache integration test、OSCAR cache、KV cache utils、
  `gpu_model_runner.py` 和 worker cache；因此不能只继承旧结果，必须回归。
- `tests/oscar_mla`、`test_kv_cache_utils.py` 与
  `test_single_type_kv_cache_manager.py` 的当前定向结果为 145 passed、
  26 个 CUDA 专项按显式门禁 skipped、0 failed，pytest 自报 47.72 秒；日志
  SHA256 为 `948167c921ea8e1d59cefdee0f588f4a42f978271545ba525a067f9a116a0050`。
- 完整 `test_scheduler.py` 强制离线回归为 68 passed、28 failed、29.34 秒；
  28 项全部因本地没有 `llava-hf/llava-1.5-7b-hf` 配置而在 ModelConfig 构造期
  失败，没有 OSCAR 或通用 scheduler 断言失败。日志 SHA256 为
  `babe6aa8085aaba91a479ec9ef9a098e206d0d3612ebf51a0b925d8a88365138`。
- 原阶段 3 的 13 个代码/测试文件全部 compileall 通过。ruff 0.14.0 对其中
  12 个文件通过；`gpu_model_runner.py` 仍为旧报告已记录的 6 个 lint/format
  问题，本轮没有修改这些行。
- 当前 planner 按 14GiB、16 个序列复算得到 36,216 blocks、579,440 个逻辑
  token slots；总分配 15,032,096,256 bytes、剩余 289,280 bytes，与总预算
  逐字节守恒。相对 native 162,256 slots 的理论容量比为
  `3.5711468297012128×`；结果 SHA256 为
  `049fa95aa86f8a5fabb96e33716819747a65282e21cfed6b5628ef8996c97353`。
- 中文报告为
  `docs/experiments/2026-07-27-phase3-reap-regression.md`。

## 2026-07-27 REAP 阶段 4 回归

- GPU 0 在独立空 Triton cache 上完整执行 `tests/oscar_mla`，实际为 114 passed、
  0 failed、0 skipped、77.63 秒；26/26 个 CUDA 门禁全部执行。日志 SHA256 为
  `a4f49b83418fd284073aa94a424c8bba5d10ae31f37aa7a8c72da80817fbc184`。
- GPU 0–7 随后各自绑定一个进程与独立空 cache，每卡
  `test_triton_store.py + test_triton_decode.py` 均为 28/28 passed，耗时
  54.18–55.08 秒；每卡 cache 均为 316 文件、22,269,278 bytes。
- 八卡 rank-local 合计 224/224 节点，其中 208 次为实际 CUDA 测试；没有把
  interpreter/reference 节点或 skip 冒充 CUDA 通过。
- 另用正式 runtime identity loader 读取新 artifact 的 layer 0 rotation，在独立
  cold cache 上执行 17 行跨页 rotation→INT2 store→dequant；clip ratio 为
  0.94，rotation/dequant oracle 最大绝对误差均为
  `3.0994415283203125e-06`，日志 SHA256 为
  `2293c39096deb80d94a641b9fafb19ad202fda0e54175f1b1845319436eb07ff`。
- 独立全套复跑也为 114/114 passed，日志 SHA256 为
  `da8cea16ff1b6750f1249d6e97565b7da975fdc26162e27746575931c63049d9`；
  候选 Torch 2.11/Triton 3.6 通过任务专用 uv venv 固定，没有修改候选 rootfs。
- 两轮实验前均完成间隔 60 秒的两次 8/8 空闲检查；结束后 8 卡均为 0MiB、0%，
  无 compute process。测试未产生源码修改，外部目录仍只读。
- 中文报告为
  `docs/experiments/2026-07-27-phase4-reap-a800-regression.md`。

## 2026-07-27 REAP 阶段 5 端到端

- 正式 run 为
  `/dev/shm/oscar-glm-reap-stage5/phase5/20260727T2330Z_reap_oscar_tp8_final`，
  main/source 为 `d9610d660...` / `a331769542...`，双仓库已发布且干净。
- 141/141 分片完成，权重读取 141.55 秒、模型加载 154.878296 秒、56.02GiB/card；
  8/8 rank 使用 `TRITON_MLA_SPARSE`，新 artifact 哈希在权重加载前通过。
- 实际 logical capacity 为 637,632 tokens、32K 并发 16.00×；theoretical/
  padded 为 6.4×，allocated 为 3.5812365205×，BF16 history absent。
- 串行 short、>320、384 decode、near32K 为 4/4 HTTP 200；near32K 实际
  31,996+64 tokens、436.7646517716348 秒。并发 8/8 全部 HTTP 200。
- 最终 78 层 store/demotion/read 每层调用 547/235/547，min=max；日志存在
  three-pool write、demotion、DSA mixed read 且无 dense fallback。
- server/single/multi/evidence SHA256 分别为 `5914d921...1913`、
  `af5953e0...5fca`、`dd69db12...fec5`、`e478bed9...85d2`。服务退出 status 0，
  8 卡回到 0MiB、0%。
- 中文报告为
  `docs/experiments/2026-07-27-phase5-reap-vllm-32k.md`。

## 2026-07-28 REAP 阶段 6 候选 OCI

- 正式候选目录为
  `artifacts/phase6/20260728T0004Z_candidate_a33176954_final`；大型 OCI、
  overlay 与 tensor 仅本地保留，不进入 Git。
- 候选 tag 为 `glm52-oscar-a800-phase6-a33176954-0275043c`，image ID 为
  `sha256:dd7b4f47...6ca70`，manifest 为 `sha256:1d3d2626...f0ea6`，
  candidate layer 为 `sha256:189f55db...182d0`。
- 候选层 109,144,170 bytes、5,297 个 tar members；前 32 个基础层逐
  descriptor 相同，候选层不含 `.so` 或 whiteout。
- 独立 verifier 验证 4,743/4,743 Git 文件、4/4 rotation 文件、
  runtime expectation 和 7/7 基础层 native extension。
- 初始审计发现 rotation 目录实际有 4 个文件、旧 manifest 只绑定 3 个。
  `artifact_validation.json` 已补充 SHA256，builder/verifier 现在严格拒绝任何
  白名单之外的文件。
- 第二次构建的 image/config/manifest/layer digest、diff ID、大小与成员数全部
  一致；固定 Python 成功导入 `_C` 与 78 个 rotation，runtime expectation
  来自候选 overlay，CUDA=false。
- 中文报告为
  `docs/experiments/2026-07-28-phase6-reap-candidate-image.md`。

## 2026-07-28 REAP 阶段 7 准备

- Stage 7 manifest、服务入口、accuracy 入口和 PPL 入口已从旧 staticgate
  候选切换到新 REAP 候选；runtime expectation 只从候选 overlay 读取。
- 新 baseline 已按实际 `/dev/shm` 结果冻结：accuracy predictions SHA256
  `c3f0b634...2579`、summary `23ddda28...4811`、分项 summary
  `14d9df7f...890a`；PPL summary `29a93a4b...0c4a`。
- 基于 baseline 的实际硬下限为 overall `0.35915254237288137`、GSM8K
  `0.4749279757391963`、IFEval `0.2675970425138632`、LiveCodeBench v6
  `0.03857142857142857`、MultiPL-E `0.10538461538461538`；PPL 上限为
  `6.792986987161362`。
- 唯一 dry-run
  `/dev/shm/oscar-glm-stage7-preflight/phase7/20260728T0012Z_stage7_reap_dryrun_primary`
  通过全部 OCI/source/artifact/expectation/native/baseline/CLI 门禁，
  fixed environment 与 parsed CLI 均为 CUDA=false。
- 用 baseline 自身作为 OSCAR 输入执行恒等 comparison，2,360/2,360、四个
  benchmark 和 289,708-token PPL 全部通过，证明比较器已正确绑定新口径。

## 2026-07-28 REAP 阶段 7 正式精度运行

- 唯一正式运行目录为
  `/dev/shm/oscar-glm-reap-stage7/phase7/20260728T0022Z_reap_candidate_tp8_final`；
  preflight 使用主仓库 `d4d0f448...`、源码 `a331769542...`，并在
  `00:21:30Z`、`00:22:34Z` 两次确认 8/8 苹果800 空闲。
- 候选服务加载 141/141 分片，模型加载 73.64 秒、每卡约 56.02GiB，OSCAR
  logical KV capacity 为 637,632 tokens；服务于 `00:29:17Z` ready。
- official_v4 于 `00:29:41Z` 启动，冻结 2,360 个样本、并发 8，代码题和
  其他样本 timeout 分别为 3,600/1,800 秒。
- 240 分钟心跳为 runner 100/2,360、服务累计 HTTP 200 为 106/2,360；8 卡约
  77.2GiB/卡，服务健康，非 200 为 0，日志没有 ERROR、Traceback 或 CUDA OOM。
- LiveCodeBench 长生成使前段吞吐较慢；当前证据显示请求持续完成，不满足
  中止或改参条件，必须保持冻结协议完成本轮。
- 外部 evaluator 在运行期间被其他进程重新生成；当前 Python runner 已在
  `00:29:41Z` 载入旧 runner 和完整 manifest，因此在途轮次不受磁盘后续变化
  影响。旧 GSM8K manifest 可由当前 manifest 唯一重建：把 `####` 提示词恢复为
  `\boxed{}`，并从 answer rationale 恢复含逗号 gold 后，整份 SHA256 精确为
  `4aec8ee85bee5eb73ce99c2009fcaedc79804bde1433f855fb77276ffccacfa5`。
- 冻结 accuracy runner SHA256 为
  `fc374ff4c4715e37d515d37aa794b3e649dc1710034d3355c69c21efa1a8aeff`；
  重建 manifest 的 2,360 个 accuracy ID、prompt hash 和 gold 与 Stage 1
  baseline predictions 逐条一致。WikiText‑2 冻结 text SHA256 为
  `696cca6b...ae83`，PPL runner SHA256 为 `eec6b1a4...5668`。
- 两套 evaluator 快照共 7.1 MiB，已从 `/dev/shm` 原样复制到项目 ignored
  路径 `artifacts/phase7/frozen_evaluator_v4_20260728`；四个关键文件 SHA256
  和整树 `diff -qr` 均通过，不进入 Git。
- 270 分钟心跳为 runner 120/2,360、服务累计 HTTP 200 为 122/2,360；
  8 个请求持续运行、无排队，异常计数仍为 0。
- 300 分钟心跳为 runner 120/2,360、服务累计 HTTP 200 为 133/2,360；
  8 卡显存约 77.23 GiB/卡、利用率 63%–78%。服务与 runner 日志中的
  ERROR、Traceback、CUDA error、OOM、request failure 和非 200 计数均为 0。
- 完整旧版 accuracy suite 已在项目 ignored 快照内恢复，共 10 个文件、
  8,640,851 bytes；`identity.json` SHA256 为
  `77ccae513121da370cf64a0c91c77a63bd89ca2ab9815b14c1203dd86525c9ea`。
  完整 evaluator 快照现为 15,975,905 bytes，外部评测目录仍保持只读。
- 310 分钟心跳为 runner 120/2,360、服务累计 HTTP 200 为 138/2,360；
  服务继续健康且没有异常计数。
- 320 分钟心跳为 runner 140/2,360、服务累计 HTTP 200 为 140/2,360；
  8 卡显存约 77.23 GiB/卡，异常计数仍为 0。
- 330 分钟心跳为 runner 140/2,360、服务累计 HTTP 200 为 148/2,360；
  8 个请求运行、0 排队、0 抢占，服务健康。chat completion 非 200、ERROR、
  Traceback、CUDA error、OOM 和 runner request failure 均为 0。
- 340 分钟心跳为 runner 140/2,360、服务累计 HTTP 200 为 149/2,360；
  8 个请求运行、0 排队、0 抢占，服务健康且异常计数仍为 0。
- 350 分钟心跳为 runner 140/2,360、服务累计 HTTP 200 为 158/2,360；
  当前 20 条批次随后收齐并使 runner 更新到 160。360 分钟心跳为 runner
  160/2,360、服务 162/2,360；8 个请求运行、0 排队、0 抢占，异常计数仍为 0。
- 370 分钟心跳为 runner 160/2,360、服务累计 HTTP 200 为 168/2,360；
  8 个请求运行、0 排队，8 卡显存约 77.24 GiB/卡。168 个 chat completion
  POST 全部为 HTTP 200，非 200、ERROR、Traceback、CUDA error、OOM 和 runner
  failure 均为 0。
- 380 分钟心跳为 runner 160/2,360、服务累计 HTTP 200 为 176/2,360；
  8 个请求运行、0 排队、0 抢占，8 卡显存约 77.24 GiB/卡。176 个 chat
  completion POST 全部为 HTTP 200，异常计数仍为 0。
- 390 分钟心跳为 runner 180/2,360、服务累计 HTTP 200 为 187/2,360；
  8 个请求运行、0 排队、0 抢占，8 卡显存约 77.24 GiB/卡。187 个 chat
  completion POST 全部为 HTTP 200，异常计数仍为 0。
- 400 分钟心跳为 runner 180/2,360、服务累计 HTTP 200 为 196/2,360；
  8 个请求运行、0 排队、0 抢占，KV usage 约 1.25%，8 卡显存约
  77.24 GiB/卡。196 个 chat completion POST 全部为 HTTP 200，异常计数仍为 0。
- 隔离提交 `28b2c87...` 已把后续 accuracy/PPL 入口改为只读取项目内冻结
  evaluator；`213643a...` 进一步把 suite/runner/环境锁/IFEval 模块树、
  command、environment 和结果 SHA256 写入正式证据。两个 shell 通过语法检查，
  其中 4 段内嵌 Python 均实际 compile 通过，冻结 accuracy/PPL runner 的
  `--help` 均可执行。
- 外部评测仓固定 Git 对象 `4d647fa` 可恢复 accuracy/PPL runner 和 IFEval
  代码，但 model-agnostic suite 在该仓库一直是未跟踪数据，不能声称可仅靠 Git
  重建。项目内 15,975,905-byte ignored 快照是当前权威副本，运行入口按哈希
  fail closed；遵守用户要求，不把该数据快照 commit/push。

## 2026-07-28 阶段 9 性能入口预审

- 设计第 10.5 节固定矩阵为输入 1K/8K/32K × batch 1/4/8；每组必须先
  warm-up，再稳定测量 TTFT、TPOT、throughput、peak memory 和 kernel time。
  baseline/OSCAR 必须使用相同模型、TP 配置和请求集，超过 20% 的回退必须
  profiling、归因并书面说明。
- 预审开始时项目没有 `scripts/phase9` 或 `configs/phase9`；现已在隔离分支实现，
  未同步到在途 Stage 7 的主工作区。
- `glm52_oscar_vllm/benchmarks/benchmark_serving.py` 只有 483 bytes，是提示迁移到
  `vllm bench serve` 的弃用 wrapper；实际实现为
  `glm52_oscar_vllm/vllm/benchmarks/serve.py`。
- 实际实现支持固定 random dataset/input/output length、seed、prompt 数、
  warm-up 请求数、batch 对应的 max concurrency、TTFT/TPOT/ITL 百分位、
  throughput、详细逐请求 JSON 和固定结果文件名，可作为矩阵主入口。
- 首次用源码模块直接执行 CLI help 只得到缺少生成版 `vllm._version` 的警告且
  没有正文；不能据此宣称 CLI 已验证。正式 Stage 9 脚本必须在候选固定环境内
  对实际 benchmark parser 做 fail-closed 静态验证。
- 固定 rootfs 的真实 CLI 进一步确认普通 `--help` 只输出分组摘要，
  不含具体 flags；全量参数必须使用 `--help=all`。隔离提交 `507567b...` 已修正
  门禁；真实全量帮助为 24,581 bytes、SHA256 `71951c4d...0424`，所需 7 个
  warm-up/concurrency/random/profile/detail flags 全部找到，CUDA 不可见且未占卡。
- Stage 1 baseline `summary.json` 的全量实测时长为 53,208.720710 秒
  （14.780200 小时）。按同一 2,360 条 `request_latency` 和 8 并发固定顺序做
  list scheduling，模拟值为 14.773976 小时，与实测相差约 0.04%，说明该方法可
  用于分段进度对照。
- baseline 前 89 条的调度模拟为 1.777171 小时；Stage 7 candidate 在约
  3.33 小时完成服务侧 89 条，因此当前 LiveCodeBench 前段约慢 1.88×。
  该比值只作为 Stage 9 profiling 预警，不能替代后续固定 1K/8K/32K 矩阵，
  也不能直接外推为全量最终回退。
- 为避免修改在途 Stage 7 脚本，Stage 9 代码准备使用项目内 ignored worktree
  `artifacts/stage9-prep-worktree` 和隔离分支 `feat/glm52-stage9-prep`。
  主工作区仍位于已发布的 `feat/glm52-model-load` 且保持干净；隔离代码只有在
  Stage 7 精度门禁通过后才会同步并作为正式实验版本提交、推送。
- 隔离提交 `e5a5db8359872409f2a78cc73deae301b141c37a` 已固定
  1K/8K/32K × batch 1/4/8、每格 3 轮、固定 prompt seed、逐请求 JSON、
  0.25 秒显存采样、每格 TP=8 torch profiler 和候选 128K 验证。profiling
  必须同时取得 rank 0–7 的 CUDA table 和至少 8 个非空 trace。
- Stage 9 代码实际通过 ruff format/check、compileall、4 个 shell `bash -n`、
  `git diff --check` 和当前 8/8 单元测试。绑定冻结 suite 后，原生完整静态门禁为
  81/81、候选门禁为 63/63，失败项均为 0；候选门禁仍完整递归执行 Stage 5/7，
  没有通过跳过 suite 检查来规避外部漂移。
- 隔离分支已推送到 `origin/feat/glm52-stage9-prep`，远端与本地均为
  `507567b969e7f57b2d5d0b26952170ae8e91d52d`；只同步代码和小型 JSON，
  不含 evaluator、模型、镜像、日志或 trace。
- 新 REAP 原生 Stage 1 日志实测 GPU KV cache 为 179,008 tokens，32K
  最大并发为 5.46×；候选为 637,632 tokens，并由 `max_num_seqs=16` 限制为
  16×。因此 32K×8 客户端并发的原生单元可能出现容量排队，不能只看客户端
  concurrency 推断实际 engine batch。
- 隔离提交 `e12c449...`/`bf1de19...` 在每个性能单元以 0.25 秒间隔同步采集
  8 卡显存、server running/waiting、KV usage 和 preemption，并显式分类
  `capacity_limited`。当前正式候选服务的只读真实采样得到 running=8、
  waiting=0、KV usage=`0.026698785506373612`、preemption delta=0；静态套件
  更新为 8/8，完整原生 81/81、候选 63/63 门禁继续全通过。
- 隔离提交 `9400f5c...` 进一步收紧正式产物边界：matrix 的 output/profile/
  server run 三类路径，以及 TP=8 wrapper 的 profiler 路径，只能落在当前 runtime
  项目 `artifacts/` 或 `/dev/shm/`；正式 server 还要求 profiler 目录为空。
  外部路径和非空目录拒绝探针均返回 1，外部目录没有被创建或修改。
- 路径防护后单元测试为 9/9，ruff、compileall、bash syntax、diff check 均通过；
  原生 81/81 与候选 63/63 完整门禁保持全通过。本提交已推送到隔离分支，Stage 7
  通过前不进入主工作区。
- 比较器原先只比较两份 summary 声明的配置 SHA，未重算当前配置或 profiler table
  内容；隔离提交 `426ad8b...` 现对两类证据逐字节重算 SHA256，并拒绝覆盖既有
  比较结果或写到项目外路径。
- 长矩阵会重复从源码和 tokenizer 启动 client，只有 server 启动时的 preflight
  不足以排除中途漂移。`8bfdd79...` 现于每个命令前后复核配置、两个 Git commit、
  工作区干净度、模型元数据和 141 个分片的文件/大小/mtime 身份；当前新模型身份
  `83eefdf0...6acaf` 与冻结值一致。
- `30c5a4b...` 进一步规范化 profile/artifact/cache 路径，固定 runtime/source/
  manifest，并限制安全 run ID；这关闭了字符串前缀判断接受 `artifacts/../外部`
  的路径穿越缺口。五类拒绝探针均返回 1，外部目录未被修改；当前套件为 11/11，
  原生 81/81 与候选 63/63 门禁继续通过。
- Stage 7 原比较器只读取 predictions 和 PPL summary，未绑定两轮 validation、
  runner 环境或 baseline summary；`8bb6f08...` 现逐项验证这些 SHA256，并检查
  2,360 个 ID/prompt/gold/task type、四项固定数量、0/1 score、PPL token/window
  与冻结 evaluator identity。4 项定向单测和 2,360 条恒等端到端模拟均通过。
- 新 REAP baseline 的 accuracy/PPL 证据原先仅在 `/dev/shm`。现已原样持久化
  到项目 ignored `artifacts/phase7/frozen_reap_baseline_20260728`，共 19 MiB；
  `diff -qr` 无差异，predictions、accuracy summary、分项 summary 和 PPL summary
  SHA256 分别保持 `c3f0b634...2579`、`23ddda28...4811`、
  `14d9df7f...890a`、`29a93a4b...0c4a`。`cd0c53b...` 仅提交相对路径配置，
  大型结果没有进入 Git。
- 固定 tokenizer 的 CPU 生成审计确认性能矩阵输入不是近似长度：1K、8K、32K
  和 128K 扩展档的实际 `prompt_len` 分别严格为 1,024、8,192、32,768 和
  130,944，默认编码与不加特殊 token 编码均相等，特殊 token 数为 0；整个审计
  未初始化 CUDA。
- 候选 TP rank 0 的 OSCAR 计数日志会随每个 forward step 更新。正式服务约
  6 小时产生 79,572 条、17,210,145 bytes；等价 78 层纯 Python 汇总实测为
  27.014 微秒/步，字符串格式化为 0.620 微秒/次。该开销会纳入 Stage 9 归因，
  但按当前证据不足以解释已观察到的代码题前段约 1.88× 差异，因此不在 Stage 7
  途中改变已冻结候选源码或镜像。
- 固定 PyTorch 2.11 CPU profiler 对象连续两轮 start/stop 实测通过，事件按轮
  清空且 CUDA 未初始化；同一服务逐矩阵单元重复 profiling 的协议可用。
  `26f43f0...` 不再以“trace 文件数量 ≥ 8”代替分布式完整性，而是从 vLLM
  `dp/pp/tp/dcp/ep/rank` trace 名解析 rank，并要求 table/trace 均覆盖 0–7。
- `4e01e75...` 进一步要求性能 baseline/candidate 的主仓库 commit、源码
  commit、模型身份和性能配置完全一致，并复核两份 frozen runtime provenance；
  13/13 Stage 9、17/17 合并单测、原生 81/81 与候选 63/63 静态门禁均通过。
- 隔离 worktree 没有自动继承 ignored 大型 artifact。首次缺 evaluator、第二次
  缺 rotation artifact 均在 GPU 启动前被门禁拒绝；最终用显式主项目 runtime
  root 只读复用已冻结 artifact 后通过，没有复制大文件、修改外部目录或把失败
  冒充为通过。
- 380 分钟时服务累计 176 个 HTTP 200；因为套件只有 175 条 LiveCodeBench，
  可确定至少一条 MultiPL-E 已完成。但 runner 一次性提交全部 futures，8 个线程
  的完成顺序不等于 manifest 顺序，最后至多 7 条 LiveCodeBench 可与后续任务
  重叠；因此不能据 176 宣称第一段已全部完成，也不能用该时间点计算完整分段回退。
- `52157cd...` 让性能比较器重新读取并哈希所有 rank 0–7 table/trace，复算
  CUDA total、critical rank 和 critical CUDA time，而不是只信 summary 或只验证
  critical table；Stage 9 单测更新为 14/14，合并套件为 18/18。
- `18ea05f...` 将 2,048-token batching、0.92 显存比例、sparse MLA backend、
  seed 42 和 chunked prefill 加入服务 parser/preflight 门禁。主项目真实 rootfs
  的完整 CLI 解析逐项匹配，且 CUDA 未初始化。
- worktree symlink rootfs 的 native dry-run 在 `_C` 动态符号加载阶段失败，
  说明 symlink 不是正式 rootfs 运行环境的可靠替代；未启动 GPU，也未复制大型
  rootfs。正式矩阵只从主工作区真实路径启动，隔离准备只承担代码和 CPU 门禁。
- 当前 in-flight runner 是 provenance 加固前启动的，最终原 validation 不含新
  comparison 所需的 summary/command/environment 哈希。`af80795...`/
  `ca39570...` 通过新 sibling 目录保留原 validation 和预测，并绑定启动时已经
  冻结的 command `2aa98cf3...f522`、environment `c42bbfbd...c2ee` 和 runtime
  config `61845910...b8b`；原目录不覆盖，当前 5/5 Stage 7、19/19 合并测试通过。
- 旧 baseline 代理分别因 artifact 作用域与 code timeout 身份不同被定稿器拒绝，
  没有产生伪定稿或改写源 validation。正式 in-flight 源位于 `/dev/shm`，其
  timeout 和三个文件哈希与定稿器固定值逐项一致。
- 隔离分支的 18 个 Stage 7/9 加固提交已在 `/dev/shm` 临时集成分支按顺序
  cherry-pick，全部无冲突；结果 head/tree 为 `90aed7e...b89bf4`/
  `1f90a402...252a`。19/19 合并测试与静态质量门禁全部通过，原生 81/81、
  候选 63/63 静态门禁也全部通过，因此 accuracy 结束后的正式同步路径已被
  预先验证；该演练没有改变在途主工作区或启动 GPU。
- 冻结 runner 顶层即 import IFEval registry/util，manifest 和 runtime config
  也在创建线程池前完整载入；后续每个 request 只使用进程内 sample、decoding 和
  已加载的 evaluator 类。因此外部评测目录运行中漂移不会影响后半段 IFEval。
- 冻结旧 runner 不按 `finish_reason` 排除返回结果，而是把文本直接交给 benchmark
  评分；代码不完整会正常记 0，且最终定稿器要求 2,360/2,360 `scored`。同一
  REAP checkpoint 的冻结 baseline 逐题 `token_usage.prompt_tokens + max_tokens`
  审计为 2,360/2,360 有数据、超过 32,768 为 0，最大预算仅 5,599 tokens。
  其中 1,602 条 completion tokens 恰等于题目上限，只能作为可能触顶的代理；
  旧 runner 没有保存 finish reason，不能据此宣称精确截断数量或截断率。
- 410 分钟时 runner 为 200/2,360、服务累计 208 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.23%，健康检查为 HTTP 200。8 卡显存约
  77.24 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 420 分钟时 runner 为 200/2,360、服务累计 217 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 0.83%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 430 分钟时 runner 为 220/2,360、服务累计 228 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.07%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 440 分钟时 runner 为 220/2,360、服务累计 237 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.10%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 450 分钟时 runner 为 240/2,360、服务累计 245 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.15%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 460 分钟时 runner 为 240/2,360、服务累计 253 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.08%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 470 分钟时 runner 为 260/2,360、服务累计 261 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.01%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 480 分钟时 runner 为 260/2,360、服务累计 270 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.12%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 490 分钟时 runner 为 260/2,360、服务累计 279 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.14%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0；心跳后
  runner 与服务均达到 280。
- 500 分钟时 runner 为 280/2,360、服务累计 287 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.13%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 510 分钟时 runner 为 280/2,360、服务累计 296 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.26%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 520 分钟时 runner 为 300/2,360、服务累计 304 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.20%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 530 分钟时 runner 为 300/2,360、服务累计 312 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.29%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 540 分钟时 runner 为 300/2,360、服务累计 319 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.17%，健康检查为 HTTP 200。8 卡显存约
  77.25 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0；心跳后
  runner 与服务均达到 320。
- 550 分钟时 runner 为 320/2,360、服务累计 328 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.56%，健康检查为 HTTP 200。8 卡显存约
  77.26 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 560 分钟时 runner 为 320/2,360、服务累计 336 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.59%，健康检查为 HTTP 200。8 卡显存约
  77.26 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 570 分钟时 runner 为 340/2,360、服务累计 347 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.90%，健康检查为 HTTP 200。8 卡显存约
  77.26 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 580 分钟时 runner 为 340/2,360、服务累计 358 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.48%，健康检查为 HTTP 200。8 卡显存约
  77.26 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 590 分钟时 runner 为 360/2,360、服务累计 370 个 HTTP 200；8 个请求运行、
  0 排队、0 抢占，KV usage 约 1.43%，健康检查为 HTTP 200。8 卡显存约
  77.26 GiB/卡，非 200、服务/runner 错误、CUDA error、OOM 均为 0。
- 600 分钟时 runner 与服务均为 380/2,360；8 个请求运行、0 排队、0 抢占，
  KV usage 约 1.69%，健康检查为 HTTP 200。8 卡显存约 77.26 GiB/卡，
  非 200、服务/runner 错误、CUDA error、OOM 均为 0。

## 2026-07-28 official_v5 与当前 GSM8K 范围

- official_v5 的全量 manifest 仍为 2,361 条：GSM8K 1,319、IFEval 541、
  LiveCodeBench v6 175、MultiPL‑E Python/C++ 164/161、WikiText‑2 1。
  当前阶段只选择 GSM8K 1,319 条；最终冻结候选后才运行 2,360 条 accuracy 和
  WikiText‑2 PPL。当前结果不能替代最终全量验收。
- v5 不提供跨 benchmark overall accuracy。当前门禁只比较 GSM8K accuracy；
  最终门禁逐项比较 GSM8K、IFEval 四项原生指标、LiveCodeBench pass@1、
  MultiPL‑E Python/C++ pass@1，并单独比较 WikiText‑2 PPL。
- v5 manifest、suite meta、eval config、accuracy runner 和 PPL runner 的
  SHA256 已逐项固定；整个非 venv snapshot 的 `SHA256SUMS` 为
  `5edf3a2f3f91d1d593781dd9f8988a2d71d7406ab35ee4d4551c234c35ff6205`，
  含 161 个已复核文件。大型 suite、NLTK 数据和 venv 只保存在 ignored
  artifact，不 commit/push。
- v5 accuracy runner 依赖 `requests`，但 external
  `requirements-accuracy-suite.txt` 没有声明该包。仅安装原 requirements 时
  `--help` 实际报 `ModuleNotFoundError`；固定环境已补齐并锁定实际依赖，不能
  只凭上游 requirements 宣称环境完整。
- `--code-eval-isolated` 只是声明，不会自行断网。当前实现使用
  `unshare -Urn --map-root-user` 把服务与 runner 放入同一网络 namespace：
  loopback 可用、无默认路由、外部 IPv4 探针失败、8 张 GPU 可见，满足 v5
  代码评测的外层禁网约束，同时保留本地 OpenAI-compatible API 通信。
- official_v4 轮次停止时只有 380/2,360 的过程证据，没有 summary 或完整
  predictions；它不能被定稿、补写或视作部分 accuracy。v4 已退出正式验收范围。
- v5 的 `eval_config.json` 固定 `reasoning_effort=max`，而当前 vLLM
  `ChatCompletionRequest` 原先只接受 `none/low/medium/high`。这不是模型推理、
  CUDA 或 OSCAR 故障：首次正式 v5 服务已成功加载 141/141 分片并 ready，但
  1,319 个请求都在请求模型校验阶段返回 400，最终 `scored=0`、
  `valid=false`。新模型 chat template 本身会把非 `high` 的 reasoning effort
  解释为 `max`，因此服务端接受并透传 `max` 是保持 official_v5 协议的最小修复，
  不能把 runner 私自改成 `high`。
- official_v5 兼容候选已冻结为 manifest
  `sha256:01f91611d1e825219907f98943968e45047458445b5b61de4ef28ff272a77932`；
  它只在原候选源码上增加请求 schema 与定向测试，rotation/runtime expectation
  及 32 个基础层保持不变。独立验收覆盖 4,744 个 Git 对象、4 个 artifact 和
  7 个 native extension；二次构建的全部内容 digest、layer 大小和成员数一致。
- v5 重跑已证明 `reasoning_effort=max` 修复生效：1,319/1,319 个 tokenization
  请求均为 HTTP 200，生成请求开始正常完成。但 runner 首次报告完成 20 条时，
  服务端只有 13 个 completion HTTP 200，至少 7 条 future 已超过上游
  `math_reasoning=300` 秒读取超时；这不是模型服务崩溃，服务在停止前持续健康。
- 对非流式生成，客户端超时必须覆盖完整 completion 用时。900 秒中间适配也已被
  实测否定：首批请求开始 900 秒后 KV usage 从 23.0% 降到 10.6% 并出现新的
  prompt 吞吐，服务成功数却未增长，说明长请求被客户端取消并重试。1,800 秒
  中间适配在精确 1,800 秒处也出现 KV usage 47.6%→22.7% 和新 prompt，
  成功数不变，同样被实测否定。
- manifest 行的 `max_tokens=8192` 不代表 official_v5 runner 的实际请求预算。
  冻结 runner 按 benchmark 计算
  `fixed_output_limit=server_max_model_len-max_prompt_tokens`；固定 tokenizer
  逐条复算得到 GSM8K prompt 为 55–218 tokens，实际统一输出预算为
  `32768-218=32550` tokens。首次离线诊断把 BatchEncoding 的两个键误当 token
  数，读取 `input_ids` 后已纠正。
- 并发 8 总吞吐约 52 tokens/s，对应单序列约 6.5 tokens/s，32,550-token
  满长约需 5,008 秒。当前最小且对称的适配是保留只读上游配置作为身份基线，
  另建 SHA256 固定 runtime config，仅把数学请求超时提高到 7,200 秒，并让原生
  baseline 与 OSCAR 候选共同使用。该变化不改变 prompt、正式输出预算、
  reasoning effort、采样、seed、重试或评分，因此不改变模型输出定义；但正式报告
  必须披露它是传输层适配，不能声称 runtime config 与上游文件逐字节相同。
- 第五次原生轮次约 12 小时时服务完成 95/1,319、runner 完成 80/1,319，非 200
  和错误均为 0，证明慢速来自有效生成工作而不是故障或重试。服务速率约
  7.9 条/小时，线性总耗时约 167 小时。
- 8 张 苹果800 已接近显存上限且大多数采样点为满载；提高客户端 timeout 不会拖慢
  成功请求，它只避免长请求被过早判失败。主要耗时乘数是 32,550-token runner
  预算、`reasoning_effort=max`、苹果800 上约 52 tokens/s 的 TP8 总吞吐，以及高
  KV 占用时部分请求等待。苹果800 不具备原生 FP8 计算支持，服务实际使用 Marlin
  weight-only FP8；eager、关闭 custom all-reduce 等冻结 baseline 参数也以性能
  换取了可复现性，但相对 32K 长生成预算属于次要因素。
- Shawn 已决定阶段 7 使用两级评测：快速筛选固定
  `--max-model-len 8192`、`reasoning_effort=high`、256 题配对预跑和
  concurrency 8/16 实测；最终阶段使用 32K/high/full v5。模型
  `chat_template.jinja` 只对字面值 `high` 选择 High，其他值均映射为 Max，
  因此快速协议不能用 medium/low 替代 high。
- 当前 32K/max 原生轮次已主动停止：停止前服务 98/1,319、runner 最近一次
  80/1,319，无非 200 或模型/CUDA 错误；停止后 8 卡均为 0 MiB。冻结 runner
  只在全量结束时写 predictions，因此这 98 条不能恢复为精度结果；快速 runner
  必须增量原子落盘。
- 快速 runner 采用“每题独立原子 checkpoint + 每 20 题原子刷新汇总
  predictions”的双层落盘。恢复时只接受同协议指纹且已评分的行，并累计跨进程
  已记录活跃耗时，避免恢复后直接忽略旧进程耗时而虚高吞吐。
- 快速协议在 GPU 分配前绑定实现和数据身份：校验 frozen runner、fast runner、
  选择器、矩阵比较器和 eval config SHA256，并复算 256/1,319 两份选择清单。
  256 与 1,319 题均固定使用全量 GSM8K 的 218-token 最长 prompt 和
  7,974-token 输出上限，避免选择子集改变生成预算。本地假服务复跑证明 2/2
  checkpoint 可恢复且不重复发送 completion；10/10 工具链测试和 31/31
  快速静态检查均通过。
- 原生 c8 的实际部分探针约 34 分钟完成 26/256，服务生成吞吐稳定约
  52–53 tokens/s；26 个 checkpoint 中 8 个达到 7,974-token 上限，说明当前
  主要耗时仍由长 reasoning 输出决定。该完成集合受非流式完成顺序影响，不能用
  11/26 正确率外推 256 题 accuracy。
- Shawn 要求最终正式评测也固定 `reasoning_effort=high`。只读上游 v5 的
  `max` 仍作为来源配置身份保留，但项目 runtime config 必须显式把它改为 high；
  这属于已披露的生成协议适配，不能继续描述成“只改数学 timeout”。
- 原生 c16 固定 256 题快速轮次已完整结束：256/256 全部 `scored`、105 条正确，
  accuracy `0.41015625`；128 条截断，truncation rate `0.5`；request failure
  为 0。截断是独立观测字段，不会从 accuracy 分母剔除；evaluator 会对截断前
  已生成文本继续提取并评分。
- c16 本轮平均 completion 为 `4181.91796875` tokens、总计 1,070,571 tokens，
  已记录活跃时长 `10578.173911571503` 秒，折合
  `87.12278770458278` requests/hour。服务端稳定生成吞吐通常约
  104–107 tokens/s，约为 c8 部分探针 52–53 tokens/s 的两倍；但 c8 不是完整
  样本，不能用两轮总时长直接做完整配对比较。
- 原生 c16 的 predictions SHA256 为
  `54a8e8adf57fbd92421aef21c9727575b122fc2e51a213dc3d9025d7b8233400`，
  协议指纹为
  `183a499b39cc05e8c03c9cda5df0c908b2a441769822728f9d4dbee952bd0db0`；
  OSCAR c16 必须匹配相同样本 ID、顺序、prompt hash、gold、seed 和协议指纹后
  才能计算 accuracy delta。
- 首次 OSCAR c16 轮次并未产生精度数据：静态身份、两次 GPU 空闲检查和实际
  8K/high/c16 参数均通过，但多进程 worker 在 readiness 前有部分 rank 报
  `CUDA driver initialization failed`。0 个样本进入 runner，因此该故障不能
  解释为 OSCAR accuracy 失败，也不能用原生结果代替候选结果。
- 失败后 8 张 GPU 全部回到 0 MiB、0%，说明没有遗留候选进程占卡。下一步需要
  在完全相同的候选 Python/动态库环境内逐卡初始化 CUDA；若逐卡全通过，则证据
  更支持一次性多进程驱动初始化瞬态，正式重跑仍必须重新执行双次空闲门禁。
- 实际逐卡诊断已 8/8 通过：每张 GPU 都在同一类隔离 namespace 和候选固定
  Python/Torch/动态库环境内成功完成 CUDA device 选择、tensor 创建、同步和
  取值。因此当前证据不支持某张 苹果800 持续故障，也不支持 rootfs 缺少 CUDA
  能力；首次失败更符合多进程同时初始化的瞬态，但仍须由完整 TP=8 重跑验证。

## 资源

- 设计文档：`docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md`
- 项目规范：`AGENTS.md`、`AGENTS_misc.md`
- GLM 部署资产：`glm52_speed_up_v2_stable_8th/`
- OSCAR-vLLM 参考：`oscar_vllm/`

## 会话：2026-07-30（服务器故障后恢复）

- `planning-with-files` session catchup 未返回未同步文本；随后重新读取项目规范、
  根目录规划文件、当前阶段尾部记录和实际运行状态。
- 当前断点仍是阶段 7 OSCAR c16/256 配对重跑；原生配对基线已经完成，首次
  候选轮次是 readiness 前 CUDA 初始化失败，不能当作精度结果。
- 恢复时 `nvidia-smi` 显示 GPU 0–7 均占用约 80,983 MiB；进程树属于项目外
  MiniMax TP=8 vLLM 服务，已运行约 2 天。当前没有可分配 GPU，本任务未启动
  新进程。
- Shawn 明确授权终止本项目以外的全部 GPU 占用进程。实际只发现上述一个外部
  进程组（PGID 1559212，8 个 TP worker）；普通用户信号被权限拒绝后，使用
  既有 sudo 授权终止同一精确进程组。终止后无 compute app，GPU 0–7 均为
  0 MiB、0%。
- 1 分钟后该服务被 Docker 自动拉起；新进程属于容器
  `vllm_minimax_m2_5_offline_replica79`，restart policy 为 `unless-stopped`。
  使用 `docker stop --time 20` 明确停止该容器后，其状态为 exited、PID 0，
  8 张卡再次全部为 0 MiB、0%。因此空闲门禁从这次停止后重新计数。
- 间隔 1 分钟的第二次检查中容器仍为 exited、PID 0，GPU 0–7 仍均为
  0 MiB、0%，正式双次空闲门禁通过。
- 候选 submodule 的 4,670 个 modified 条目经 `diff --raw`、`--summary` 和
  `--numstat` 复核，全部只是工作文件权限从 Git 记录的 100644 显示成 100755，
  内容没有差异。这是服务器/NFS 恢复后的 mode 漂移，不是源码修改。
- 设置候选 submodule 本地 `core.filemode=false` 后工作区干净，HEAD 和远端
  `feat/glm52-oscar-integration` 均为预期
  `065af88a010dc5746029198088ba01edc4a61516`。
- 主仓库恢复记录已提交为本地 `f886714`。HTTPS 推送两次均无法认证；trace
  明确显示停在 VS Code askpass 的 GitHub username 请求。GitHub 网站网络正常，
  但环境中没有 token、`gh` 登录或 SSH 私钥，因此当前远端 SHA 门禁未满足。
- Shawn 要求重新发起 GitHub HTTPS 认证后，清除失效 credential 并再次
  `git push`，认证成功；远端 `feat/glm52-model-load` 已从 `e5927db`
  更新到 `e50ae6a`，此前认证阻塞解除。
- 新宿主只有 Python 3.8，frozen v5 evaluator 的 `.venv/bin/python` 仍指向
  已不存在的 `/usr/bin/python3.12`；site-packages 与 22 个锁定包本身完整。
- 固定 `uv 0.11.5` 已恢复 Python 3.12.3。uv 的 relocatable 构建直接作为
  旧 venv symlink 时因 `/install` 前缀找不到标准库；改用显式 `PYTHONHOME`
  和既有 venv site-packages 后，Python 版本、关键 imports、NLTK punkt/
  punkt_tab 均通过，official_v5 static/namespace preflight 通过。
- 候选轮次 `20260730T0356Z_candidate_fast256_c16_retry` 在 readiness 前退出：
  新宿主 glibc 低于候选 rootfs Python 3.12 所需的 2.35，0 个样本进入 runner，
  GPU 始终为 0 MiB。当前是可用 Docker 的物理宿主，不能继续直接执行 rootfs。
- 冻结候选 Docker tag 尚未导入本机 daemon，但 16GiB OCI layout 完整存在，
  index manifest 仍为 `sha256:01f91611...7932`，可从该不可变 layout 导入。
- 宿主 focal 无 skopeo 包；jammy 安装模拟会升级 glibc 并移除 Nsight Systems，
  因风险明确未执行。官方 skopeo 工具镜像又因 Docker HTTPS proxy 错误未拉取。
- 最终在本机已有 GLM 基线镜像的一次性工具容器内安装 skopeo 1.4.1，只读挂载
  OCI layout 并通过 Docker socket 导入。导入后 image ID 精确匹配
  `sha256:8b7a2ee6...68bb`，容器内 Python 3.12.13 和候选内容探针通过。
- 冻结候选镜像本身缺 orchestrator 控制面工具 git/iproute2；在一次性容器可写层
  安装二者、只读挂载模型后，8 卡注入和 official_v5 static/namespace
  preflight 通过。候选镜像本体没有重建或修改。
- 首次正式容器轮次 `20260730T0409Z_candidate_fast256_c16_docker` 在
  static/published 门禁后、服务启动前因旧 tmpfs artifact root 权限失败；
  内层 user namespace 无权 mkdir。0 个样本、GPU 全程 0 MiB。
- 新建独立空的 `/dev/shm/oscar-glm-official-v5-docker`，mode 1777，仅用于本轮
  tmpfs 产物；不放宽项目或 NFS 权限，失败 run ID 不复用。
- 第二次容器轮次 `20260730T0411Z_candidate_fast256_c16_docker_v2` 的候选
  manifest/config/layer、rotation、baseline 和源码内容均通过；唯一失败是
  overlay source 的 100644 文件在 NFS 上呈现为 100755。0 个请求、GPU 0 MiB。
- 对代表文件执行 chmod 644 返回 `Operation not permitted`，说明不能在 NFS
  原地恢复 mode。候选镜像 `/opt/vllm_glm52_v1` 自带正确 mode；可仅在容器
  mount namespace 内 bind 到脚本预期 overlay 路径，并恢复既有 6 个 lower
  native symlink 合约，不改宿主/NFS 文件。
- 精确 bind 还需要容器 `apparmor=unconfined`；只有 SYS_ADMIN 与 unconfined
  seccomp 时 AppArmor 会拒绝 mount。权限仅授予一次性候选容器。
- 候选镜像包含 runtime-import 遗留的 100 个 `.pyc`/27 个 `__pycache__`；
  仅在容器可写层删除后，候选 4,744 tracked files + 6 native symlink 树通过。
- Stage 5 冻结 base 实际是 manifest `2fdfbe...`、config `58a853...`，不是旧
  本地 tag `d6faf...`。正确 phase0 OCI 导入后，以其 `/opt/vllm_glm52_v1`
  创建的 Docker volume 有 4,711 个 tracked files；Stage 5 verifier 通过。
- 最终 CPU-only 候选 dry-run 同时通过 Stage 5、候选 OCI/source/rotation/
  baseline 和 8K/high/c16 命令解析；运行约 7 分钟，GPU 全程 0 MiB。
- 容器化候选轮次
  `20260730T0438Z_candidate_fast256_c16_docker_v3` 已完成 256/256：
  107 条正确，accuracy `0.41796875`，request failure 为 0，128 条截断，
  平均 completion `4194.87890625` tokens，活跃耗时
  `21964.45053267479` 秒，吞吐 `41.95870953516493` requests/hour。
- 原生 BF16 c16 为 105/256、accuracy `0.41015625`，所以候选只净多 2 条，
  即 `+0.0078125`（`+0.78125` 个百分点）。temperature 为 0 时，INT2
  history 引入的数值扰动仍可能改变接近 logits 的贪心 token 次序，并使后续
  推理轨迹整体分叉；这种翻转可双向发生，净多 2 条不能证明量化提高了精度。
- 严格配对门禁当前尚未成立：原生协议指纹为 `183a499b...bd0db0`，候选为
  `5bc5f1a0...404718`。候选的 sample identity、eval config、fast runner 和
  frozen runner 哈希已核验，但恢复后的 evaluator 执行环境被协议指纹纳入；
  当前证据不足以断言指纹差异只来自解释器路径。当前存储中也未找到原生
  fast256 的逐题 predictions，只保留其汇总和 SHA256，因此无法诚实给出
  both/native-only/candidate-only 的逐题分解或 McNemar 检验。
- Stage 9 性能协议已绑定为同一模型、源码 commit、TP=8、eager、async off、
  131,072 最大长度、2,048 batched-token 门限、1K/8K/32K 输入、
  batch 1/4/8、128 输出、每 cell 1 次 warm-up 后 3 轮正式测量；BF16 与
  OSCAR 使用同一控制镜像，唯一预期变量为 KV cache dtype/path。
- Stage 9 控制镜像基于不可变候选镜像，只增加 `git` 与 `iproute2` 控制面包；
  image ID 为 `sha256:c77d72256e2e0454035c8fd6a13c8d72eedcce285e92d0d1938c23a3f0cbfa2d`，
  基础 image ID 为 `sha256:8b7a2ee6...68bb`，Python 3.12.13、glibc 2.35。
- 容器化 BF16 静态预检结果为 `passed`，SHA256
  `c9dd73c81c0f277d4cc96c564519f55598398501fa7f9aa8c1941f78d64fa4a8`；
  修正后的 OSCAR Stage 9 静态预检结果为 `passed`，SHA256
  `08a0d5f05b06612376dff65ded8ad4dc8311648015499f9a233de334908e48f8`。
  两边 fixed-environment import 都确认 `cuda_initialized=false`。
- 初版 OSCAR Stage 9 wrapper 会被中间 Stage 7 wrapper 覆盖
  `VERIFY_SCRIPT`、`ARTIFACT_PHASE` 和运行标签；若不修复，正式产物会写到
  `phase7/` 而 matrix runner 固定读取 `phase9/`。中间 wrapper 现在只提供
  默认值，修正后的 outer Stage 9 verifier 已实际执行并通过。
- BF16 首次正式性能轮次
  `20260730T124357Z_stage9_baseline_v1` 在模型服务 ready 后、发送任何 benchmark
  请求前被参数证据门禁拒绝：真实命令含 `--max-num-batched-tokens 2048`，
  但 parsed JSON 没有记录该字段。该轮没有 summary/TTFT/TPOT，不是性能结果；
  cleanup 后 8/8 GPU 为 0 MiB。
- 同次审计还发现 Stage 9 wrapper 虽已生成 profiler JSON，通用服务入口此前没有
  将它附加为 `--profiler-config`。修正后 BF16 dry-run parsed JSON SHA256 为
  `22a1da57312b01cbf5ed8ca3d8bd16816ee0a9100798621a3125b4493a16f584`，
  OSCAR 为
  `710e0944c1cd3124c0ae5ca383dbf2848a5cbb115a8b7ed23e744fe9f3f5c5e6`；
  两者均确认 profiler=`torch`、max batched tokens=2048、GPU utilization=0.92、
  seed=42、async=false，且未初始化 CUDA。
- BF16 v2 `20260730T130056Z_stage9_baseline_v2` 已成功运行首个
  1K/batch1 的 1 warm-up + 3 个正式请求，真实 result 保存
  `max_concurrency=1`，runner 也打印 `Maximum request concurrency: 1`；
  `max_concurrent_requests=2` 是基于整秒闭区间桶的派生峰值，不等价于 semaphore
  并发上限。旧门禁因此误报，v2 未生成完整 summary，不能作为 baseline。
- 并发门禁现严格检查 benchmark 保存的配置字段 `max_concurrency` 是否等于目标
  batch；`max_concurrent_requests` 继续保留用于了解一秒时间桶内活动请求，但
  不再用于判定客户端是否越过并发上限。
- BF16 v3 `20260730T1324Z_stage9_baseline_v3` 的 1K/batch1 已完成 3/3
  正式 rounds，证明修正后的并发门禁不再误报。首个真实 torch profile 同时生成
  8 个 TP worker trace 和 1 个 `.async_llm.` frontend trace；前端 trace 没有
  TP rank 是合法格式，不应混入 worker rank 集。
- v3 在 profile 完成前被主动停止，8 个部分写入的 worker trace 各约
  109–115MB、frontend trace 1,087 bytes；没有完整 profile table/cell summary，
  不能作为 BF16 结果。停止后 GPU 0–7 为 0 MiB。
- profiler 证据合约现要求：rank table 精确覆盖 0–7、worker trace 精确覆盖
  0–7、frontend `.async_llm.` trace 精确 1 个；三类文件都记录 size/SHA256，
  比较阶段逐文件复验，其他未知无 rank trace 仍 fail closed。
- BF16 v4 `20260730T1342Z_stage9_baseline_v4` 已完成固定 9 格矩阵，所有 cell
  与总 summary 均为 `passed`，每轮 request failure 为 0。三轮中位数
  TTFT/TPOT（ms）依次为：1K/b1 `352.445/156.705`、1K/b4
  `792.733/181.125`、1K/b8 `1292.619/185.086`、8K/b1
  `2751.019/179.002`、8K/b4 `4968.413/217.451`、8K/b8
  `7119.445/272.702`、32K/b1 `12528.026/178.832`、32K/b4
  `21838.365/400.676`、32K/b8 `80081.107/419.976`。
- BF16 v4 summary SHA256 为
  `c0e312299bb6aba034bf01fd848197605c3763ac87e9cb367b58bf71abf4e2f5`，
  矩阵时长 `13495.650912761688` 秒。32K/batch8 服务器最多 running=4、
  waiting=7、KV usage=82.24%，没有 preemption；其高 TTFT 包含容量排队。
- v4 的矩阵 summary 完成后，outer cleanup 的 Bash `EXIT` trap 在
  `inside_container` 返回后再次读取局部 `wrapper_pid`，受 `set -u` 影响退出
  1。容器已由 `--rm` 删除，8 卡均为 0 MiB，结果没有被修改。防御修复使用
  `${wrapper_pid:-}` 并在正常 cleanup 后撤销 trap。
- 比较器要求 baseline/candidate 的 frozen runtime inputs 完全相同，其中包含
  main commit。因此阶段记录和 cleanup 修复先保存在独立本地进度分支，OSCAR
  正式配对仍必须从 BF16 已发布提交 `0918f3a` 运行。
- 对“OSCAR 精度为何高于 BF16”的复核结论：现有 fast256 汇总仅为
  BF16 `105/256`（`0.41015625`）与 OSCAR `107/256`
  （`0.41796875`），净差 `2/256`（`+0.78125` 个百分点）；两边均为
  128 条截断、0 request failure。该小差值不能证明 OSCAR 提升精度。
  INT2 history 的数值扰动即使在 temperature=0 下也可能改变接近的贪心
  token 排序并使后续推理轨迹分叉，翻转可双向发生。候选平均 completion
  比 BF16 多约 12.96 tokens，也证明两轮生成轨迹并非逐 token 相同。
  更关键的是协议指纹分别为 `183a499b...bd0db0` 与
  `5bc5f1a0...404718`，严格配对门禁未通过；原生逐题 predictions 已缺失，
  目前无法计算逐题翻转和 McNemar 检验，因此不能归因到量化或 OSCAR 算法。
- 2026-07-31 再次读取 OSCAR 原始 `predictions.jsonl`：107 条正确中，
  101 条未截断、6 条虽截断但截断前已经生成可提取的正确答案；128 条截断中
  118 条无法提取答案。固定解码为 temperature `0.0`、top-p `1.0`、seed
  `42`，所以这里不是常规随机采样带来的波动；更准确的表述是有损 KV 数值扰动
  触发了贪心解码的离散路径翻转，而 256 题上的净结果恰好多 2 题。由于 BF16
  逐题文件缺失，仍无法知道 BF16→OSCAR 和 OSCAR→BF16 各有多少题。
- 进一步统计时，BF16 `105/256` 与 OSCAR `107/256` 的 Wilson 95% 区间分别
  约为 `[35.17%, 47.13%]` 与 `[35.92%, 47.92%]`，高度重叠；净差只有
  `0.78125` 个百分点。这里的“波动”是 256 题有限样本和数值路径翻转造成的
  结果不确定性，不是 temperature=0 下重复运行的随机采样噪声。
- fast 协议指纹实现除了样本、配置和评分器哈希，还纳入
  `code_eval_environment()`。服务器恢复后候选轮次使用恢复到
  `/dev/shm/oscar-glm-recovery-tools/.../python3.12` 的解释器，而 BF16 原始
  逐题目录已经缺失；这是严格指纹不同的一个已确认来源，但在缺少 BF16 原始
  payload 时不能声称它是唯一来源。
- OSCAR Stage 9 首轮 `20260730T1741Z_stage9_candidate_v1` 与 BF16 使用同一
  `0918f3a` 主提交、`065af88a` 源码提交、模型、镜像和负载。完整
  1K/batch1 为 TTFT `5049.519678888221 ms`、TPOT
  `243.12730936524204 ms`、request throughput
  `0.027830596224233704`，相对 BF16 分别约 `+1332.7%`、`+55.1%`、
  `-43.6%`，明确超过 20% 门限；cell summary SHA256 为
  `306349dfde90733c010d7c69a4dfcbe3bc2cc2e50619c20c00463ffd272d3dc4`。
- 1K/batch1 profiler 显示 OSCAR mixed stage1 为 `629.748 µs/call`，
  BF16 sparse kernel 为 `238.654 µs/call`；OSCAR KV update 的 CPU total
  为 `1.639 ms/call`，其中 29,952 次 `aten::nonzero` 共 `7.137 s`、
  110,004 次 `aten::index` 共 `8.968 s`。代码复核确认 decode 新 token
  必在 recent window，却仍逐层构造 `current_history` 布尔索引并对空集合执行
  三次 nonzero；这是可直接消除的 78 层重复热路径。
- `_rotate_latent_kernel` 共 29,952 次、CUDA total `845.825 ms`；
  `_mixed_sparse_decode_stage1` 和重复元数据/索引是首批优化对象。all-reduce
  平均时长翻倍更可能是 rank 热路径变慢造成的同步等待，当前不作为首要根因。
- 未优化候选在 batch4 第三轮期间主动停止，保留两轮 12/12 成功的部分结果，
  不生成或伪造全矩阵 summary。精确停止容器后无残留进程，8 卡均为 0 MiB。
- decode 新 token 的位置恒为 `seq_len - 1`，而 current-history 上界为
  `max(prefix, seq_len - recent)`；在 recent window 非空时，新 token 不可能
  属于 current history。recent 中被挤出的旧 token 已由 demotion metadata
  单独处理，因此纯 decode 可以安全跳过 current-history mask/nonzero/index，
  但 prefill/chunked prefill 必须保留通用路径。
- 首轮快路径同时把纯 decode 的 query position、final sequence length 和
  HP row 改为按请求顺序直接切片。两请求回归覆盖不同 seq_len/HP row、仍有
  demotion 且 history store 必须不被调用；定向结果 10 passed。
- 完整 CPU `tests/oscar_mla` 为 89 passed、26 个显式 CUDA 门禁 skipped、
  0 failed；Ruff check、mypy、typos、SPDX、compile 及其余相关 pre-commit
  门禁通过。源码提交 `98ddd3f4e...32aaa2` 已推送；这些结果尚不能证明 GPU
  TTFT/TPOT 改善。
- 新候选 OCI
  `glm52-oscar-a800-phase6-98ddd3f4e-0275043c` 已构建并通过验收：
  image/config `sha256:e90f4a84...bdcbf7`、manifest
  `sha256:96455a6e...a69f9`、candidate layer
  `sha256:badb252b...f4f538`；33 层、4,744 源码文件、4 份 rotation 和
  7 个基础层原生扩展均匹配，候选层无 `.so`/whiteout。
- 新镜像 CPU-only runtime import 实际得到 Python 3.12.13、PyTorch
  2.11.0+cu129、Triton 3.6.0、78 个 rotation tensors、
  `reasoning_effort=max`，且 `cuda_initialized=false`。候选已导入 Docker
  daemon，全过程无 GPU compute app。
- 新 Stage 9 控制镜像 `oscar-glm-stage9-runtime:98ddd3f4e` 的 image ID 为
  `sha256:f25d8d5f...8d40f8`，其 base image ID 精确为新候选
  `sha256:e90f4a84...bdcbf7`。Phase 1/5/7/9 身份链已同步，JSON、shell 语法与
  Stage 9 工具 15/15 通过。
- 在宿主 NFS 直接运行静态 verifier 会把 phase0 rootfs 的 100644 文件读成
  executable mode，造成既有 `rootfs_runtime_tree_match` 失败；正式入口已按
  既有合约从 Docker volume 提供正确 mode，并在 mount namespace 内把镜像源码
  bind 到 overlay。因此该宿主直跑不能冒充正式 preflight 结果。
- 正式 candidate preflight
  `20260730T1946Z_stage9_candidate_fastpath_preflight_v1` 在双 8/8 GPU 空闲检查
  后通过 64/64：TP=8、131072、2048、OSCAR INT2、eager、async=false、torch
  profiler 全部匹配，`cuda_initialized=false`。static JSON SHA256 为
  `9a5c1112...0c920`；容器退出后 8 卡 0 MiB、无 compute app。
- Stage 9 runner 新增可审计的单格探针入口。`STAGE9_ONLY_CELL=1024:1` 只选择
  冻结矩阵中已有的 1K/batch1 格，仍执行固定 128 输出 token、1 次 warm-up、
  3 轮正式测量及一轮 8-rank profiler；summary 显式标记
  `scope=single_cell_probe`，且该模式拒绝同时运行 128K，不能冒充完整矩阵。
  控制容器内工具测试为 16/16 passed，Ruff 和 shell 语法通过。
- Decode 快路径定向轮次
  `20260730T2000Z_stage9_candidate_fastpath_probe_1k_b1_v1` 完成并通过：
  1K/batch1 三轮 TTFT/TPOT 中位数为 `5014.582/206.735 ms`，吞吐
  `0.0319637 req/s`。相对未优化 OSCAR 为 `-0.7%/-15.0%/+14.9%`，相对
  BF16 仍为 TTFT `+1322.8%`、TPOT `+31.9%`、吞吐 `-35.2%`。
- 8-rank profiler 中位数显示 `unified_mla_kv_cache_update` CPU/CUDA total
  从 `16.107/1.080 s` 降至 `9.293/0.439 s`；`nonzero`/`index` 调用从
  `29,952/110,004` 降至 `234/10,944`。但 mixed stage1 仍为
  `6.264 s`（仅 `-0.1%`），1K prefill 为 `5.051 s`（`+0.3%`），证明
  下一步必须分别优化 mixed decode kernel 与 prefill/首 token。
- Mixed attention 当前由 backend 对 decode/prefill 一律调用
  `oscar_mla_sparse_prefill`，未显式传 `num_splits`，因此固定使用 op 默认值
  16；已有 `VLLM_SPARSE_MLA_FORCE_KV_SPLITS` 等参数只属于 BF16 sparse
  kernel，不能控制 OSCAR。TP=8 时 GLM-5.2 每 rank 为 8 heads；1K/batch1
  实际几何为 top-k buffer 2,048、有效 context 1,024、prefix/history/recent
  `64/704/256`、latent/RoPE `512/64`。后续 split 选择必须基于这一固定形状
  的单卡实测，而不能沿用其他 MLA kernel 的启发式。
- 首次 split sweep 误用镜像的 `/usr/bin/python3.12`，实际导入
  PyTorch `2.10.0+cu129`；正式服务解释器
  `/opt/fp8_speed_up_v4_venv/bin/python` 实际为 PyTorch `2.11.0+cu129`。
  虽然首轮四组输出/LSE 均通过且 split 32 的完整调用中位数为
  `0.324076 ms`，该环境身份不匹配，结果必须作废，不能用于参数选择。
- 有效 v2 sweep
  `20260730T2051Z_oscar_mixed_split_sweep_1k_b1_v2` 使用正式候选 venv、
  PyTorch `2.11.0+cu129` 和固定 GPU 0。split 4/8/16/32 的完整 attention
  CUDA 中位数依次为 `0.674417/0.375931/0.322437/0.322836 ms`，峰值
  allocated 为 `1.8330/1.9580/2.2080/2.7085 MiB`；四组 output/LSE 均通过，
  最大绝对差分别不超过 `2.980232e-07/4.768372e-07`。
- 当前默认 split 16 已是 1K/batch1 固定形状的最优值。split 32 慢约
  `0.12%` 且多占约 `0.50 MiB`，4/8 明显更慢；因此不能通过修改 split 改善
  当前 TPOT 回退。summary SHA256 为
  `e96df05eed5af6d89fd7b6d47e7f0eacc0c8d856c4e5cf9041cab1fde7b774b5`。
- 对 decode 快路径轮次的 critical rank 7 trace 按首个
  `execute_context_1(1024)_generation_0(0)` 时间窗独立归集：prefill 墙钟
  `4,997.089 ms`，窗口内 CUDA kernel 合计 `4,957.278 ms`；其中 78 次
  `_mixed_sparse_decode_stage1` 合计 `4,575.801 ms`，占 prefill 墙钟
  `91.57%`、CUDA kernel 时间 `92.31%`。其后是 rotation `105.601 ms`、
  MoE `80.295 ms`、NCCL all-reduce `72.092 ms` 和 merge `25.945 ms`。
- 当前 backend 对 1,024 个 prefill query 仍使用 decode 为提高单 query
  occupancy 所需的 16 splits，并且每行固定扫描 2,048 个 top-k 槽位；这会
  产生 `1024×8×16` 个 stage1 programs。prefill 本身已有大量 query/head
  并行度，因此下一步应实测 prefill 专用较小 split，并安全裁掉
  `max_seq_len < 2048` 时必为无效的 top-k 尾部；decode 的 split 16 结论不应
  直接套用到 prefill。
- 8-rank 流式 trace 复核
  `20260730T2112Z_fastpath_prefill_trace_analysis_v2` 全部通过。8 个 rank 的
  prefill duration 中位数 `4,997.096 ms`，kernel total 中位数
  `4,952.601 ms`；mixed stage1 每 rank 固定 78 calls，中位数
  `4,576.783 ms`、范围 `4,574.163–4,604.965 ms`，占 prefill 墙钟中位约
  `91.58%`。第二项 rotation 仅 `105.583 ms`，证明根因跨 rank 一致且高度集中。
- 8-rank analysis summary SHA256 为
  `5a32a7ca96dc46aec7214324df5c16a8e262aa3bd8a22ed9bc62ce6c2bd8a056`；
  每个 trace 都重新绑定原文件 bytes/SHA256，129 个 execute_context 全覆盖。
- prefill sweep `20260730T2116Z_oscar_prefill_sweep_1k_b1_v1` 在
  `21:16:50Z/21:17:56Z` 两次 8/8 GPU 空闲检查后固定使用 GPU 0；正式解释器
  为 `/opt/fp8_speed_up_v4_venv/bin/python`，PyTorch
  `2.11.0+cu129`、CUDA runtime `12.9`。7 组配置均通过 output/LSE
  allclose；最大 output/LSE 绝对误差分别为
  `1.1920928955078125e-06`/`9.5367431640625e-07`。
- 1,024-query prefill 的 CUDA 中位时间为：full top-k/split16
  `62.883839 ms`、full/split1 `57.757694 ms`、cropped/split16
  `51.245056 ms`、cropped/split8/4/2/1 分别
  `48.530434/47.248383/46.680065/46.382080 ms`。裁剪 2,048 槽位到
  当前序列上限 1,024 后再用 split1 最优，相对现状加速
  `1.3557787522×`（时间下降约 `26.24%`），临时峰值 allocated delta 从
  `592.53125 MiB` 降至 `112.0625 MiB`。
- 单独缩减 split 仅加速约 `8.15%`，单独裁剪 top-k 尾部约
  `18.51%`；组合后达到上述 `1.356×`。按每层实测粗略外推，78 层可减少约
  `1.287 s`，仍不足以单独关闭 OSCAR 与 BF16 的 TTFT 差距，因此该优化落地后
  仍需重新 profile 剩余约 3.6 秒的 stage1。
- sweep summary/runner SHA256 分别为
  `25355d535bef1daaf4099d7b06aee7ee123a81fbbaffcf3b4d6ffc88113025d4`/
  `9ce24b25e97e2f3f8bc4150eb14d19c6be39c991528a7086761c390c8010de36`；
  容器退出后 8 张 GPU 均为 0 MiB、0%，无 compute app。
- prefill fastpath 源码提交
  `a94b1f640fe504be3d741a1070e43f806eaad894` 只在 OSCAR attention 调用点
  区分纯 decode 与其他批次：decode 保持 full top-k/split16，prefill/mixed
  使用 `min(topk_tokens,max_seq_len)`/split1。定向 11/11、完整 CPU
  90 passed、26 CUDA skipped；增量 mypy、Ruff、typos、SPDX、compile 和相关
  pre-commit 均通过。
- 新 Phase 6 OCI
  `20260730T2132Z_candidate_a94b1f640_prefill_fastpath` 已构建并验收：
  image/config `sha256:e0f6b406...5635`、manifest
  `sha256:41a70b2a...c826`、candidate layer
  `sha256:34a5e717...45e6`。4,744 个源码文件、4 份 rotation、7 个基础层
  native extension、33 层与基础层完全匹配；runtime import 使用正式 venv，
  确认 PyTorch `2.11.0+cu129`、CUDA 未初始化。
- 新控制镜像 `oscar-glm-stage9-runtime:a94b1f640` 的 image ID 为
  `sha256:a7482d1c709e02720e9bad7e442f558af4ebc27315904763e1feac0744179ed9`，
  base image ID 精确为 `sha256:e0f6b406...5635`。Phase 9 工具 19/19、
  Phase 7 工具 20/20 已在该镜像通过；正式 containerized preflight 仍需在
  配置提交发布后执行。
- prefill fastpath 正式 preflight
  `20260730T2152Z_stage9_candidate_prefill_fastpath_preflight_v1` 已通过
  64/64，参数解析确认 TP=8、131072、2048、`oscar_mla_int2`、eager、
  async=false、torch profiler 且 `cuda_initialized=false`；static JSON
  SHA256 为 `4c3086c6...fb324`。
- TP=8 单格探针
  `20260730T2156Z_stage9_candidate_prefill_fastpath_probe_1k_b1_v1` 为
  `passed`：3/3 正式轮次 0 failure，TTFT/TPOT/请求吞吐中位数为
  `3714.820887272557 ms`、`205.90789329866803 ms`、
  `0.03345713660516981 req/s`；0 waiting、0 preemption、无容量限制。
- 相对 decode 快路径，prefill fastpath 的 TTFT `-25.92%`、TPOT `-0.40%`、
  请求吞吐 `+4.67%`；相对 BF16 仍为 TTFT `+954.01%`、TPOT `+31.40%`、
  请求吞吐 `-32.19%`，因此不能进入完整矩阵。
- 总/cell summary SHA256 分别为
  `f14b0088b144b9982a078014d860adefb1da370d0be75b245aaf08ea9bd4bc59`/
  `b010278a9d9cb8da77ccc0b25112731b6e257d69d6866ace52d7c7a70a192f73`；
  8 张 table、8 个 worker trace、1 个 frontend trace 全部通过，退出后
  8 卡均为 0 MiB、0%、无 compute app。
- 有效 trace analysis
  `20260730T2225Z_prefill_fastpath_probe_trace_analysis_v2` 使用 Python
  3.12.13、`ijson 3.4.0.post0`，summary SHA256 为
  `ff69be06...d9922`。prefill/kernel/stage1 中位分别从
  `4997.096/4952.601/4576.783 ms` 降到
  `3697.620/3650.752/3300.032 ms`，即 `-26.00%/-26.29%/-27.90%`；
  stage1 仍占 prefill `89.25%`，是下一优化目标。
- 重新审计 256 题快速精度后，当前证据仍只支持“OSCAR 单轮比 BF16 净多
  2 题”，不支持“OSCAR 提升精度”：BF16/OSCAR 分别为
  `105/256=0.41015625` 与 `107/256=0.41796875`，差
  `+0.78125` 个百分点；两轮均有 128/256 条达到 7,974-token 截断上限。
  INT2 KV 的数值扰动可令 greedy 解码在临界 token 处分叉，但这种翻转可能
  双向发生，不能解释成单向能力提升。
- 候选协议指纹已用其落盘 runtime suite、evaluator hashes、预算和
  `code_eval_environment` 重新计算，精确复现
  `5bc5f1a0...404718`。协议哈希会纳入 Python 可执行文件绝对路径、g++ 与
  platform，因此服务器恢复后的环境路径变化本身即可改变指纹；但原生逐题
  predictions 已不在当前存储，无法恢复配对翻转或 McNemar 统计，也无法证明
  两个指纹的全部差异只来自环境路径。最终只能通过同提交、同环境、完整保存
  两侧逐题结果的重跑归因。
- mixed prefill 的 8 个本地 head 共享 selected token 与 MLA KV，但旧 kernel
  按 head 重复 INT2 unpack、scale/zero、BF16 KV 和 RoPE 读取。源码
  `35ab1846447fc86b4b2177e76c5939503cc3701b` 用一个 grouped program 复用
  这些数据，纯 decode 仍保持旧 stage1/split16；score/value dot 使用 FP32
  IEEE 精度以满足原有数值门限。
- 有效单卡轮次
  `20260730T2255Z_oscar_prefill_headgroup_1k_b1_v3` 的 7 组 output/LSE
  全部通过固定 `atol=rtol=0.002`；winner 最大绝对误差仅
  `3.8743019e-06/1.4305115e-06`。cropped/split1 CUDA 中位从旧
  `46.382080 ms` 降至 `13.284352 ms`，加速 `3.491×`、下降 `71.36%`；
  相对 full/split16 `62.896130 ms` 加速 `4.734603×`。summary SHA256 为
  `f87f3624...ef2c0`。
- 两个中间版本均被门禁正确拒绝：v1 `num_stages=2` 需要 184,320-byte
  shared memory，超过 SM80 的 166,912-byte 上限；v2 BF16 tensor-core
  版 output/LSE 最大误差 `0.009153/0.002593`，超过固定门限。没有放宽
  正确性标准或把失败轮次计作性能结果。
- grouped prefill 源码提交 `35ab1846447fc86b4b2177e76c5939503cc3701b`
  的完整冷 Triton cache CUDA 回归轮次
  `20260730T2302Z_oscar_headgroup_full_cuda_v2` 为 124 passed、0 skipped、
  0 failed，耗时 80.50 秒；新增的 split1 prefill 组合均实际执行。
  pytest 日志 SHA256 为
  `3f8006e38ef6f49fb3f0832c3d003e5997a79c9a054b57db5f470c6c38f6f1f7`，
  cache 共 380 个文件、29,220,161 bytes。首轮 v1 只因 phase0 rootfs 的
  宿主绝对 symlink target 未挂载而在 collection 阶段退出，0 个测试、0 个
  kernel；v2 仅补回同路径只读挂载，源码、测试和运行时不变。
- 新候选目录
  `artifacts/phase6/20260730T2315Z_candidate_35ab18464_headgroup` 已构建并
  通过独立递归验收。image/config 为
  `sha256:d06a82948c342de84bcd2400ca51ab9ada67ad45a20fa7723629a3d0487367df`，
  manifest 为
  `sha256:28a8f1daec3c52420073eaad93e984982b9d1647521ddd99d00a3fac8b88c9d0`，
  candidate layer 为
  `sha256:a599892d73b9b54723a06700ed82b8022c10a9bfe1302a7d829e84b38e192da0`。
  4,744 个源码文件、源码 tree、4 份 rotation、7 个基础层 native extension
  和 33 层身份全部匹配；candidate layer 不含 native extension/whiteout。
  build/verification SHA256 分别为
  `618191fd...83a0b`/`7adad3e1...747d`。
- 新候选已通过 Docker daemon import 与只读 runtime import。daemon image ID
  精确为 OCI config `sha256:d06a8294...367df`；正式 venv 解析为
  Python/PyTorch/Triton `3.12.13/2.11.0+cu129/3.6.0`，vLLM source/C、
  78 个 rotation、两份 rotation hash 和 `reasoning_effort=max` 均匹配，
  `cuda_initialized=false`。`runtime_import.json` SHA256 为
  `0910b59876984b01559847d55a7407524592401ab707dd7622b181eec5217b7a`。
  探针前 `23:17:30Z/23:18:40Z` 双空闲，退出后 8 卡仍为 0 MiB、无
  compute app。
- 配置切换审计发现，上述 v1 候选实际 source commit/tree 正确，但其候选 label
  固定的 Dockerfile SHA256 对应文件仍把默认 `SOURCE_COMMIT/SOURCE_TREE`
  写成旧 `a94b1f640…/7b5650fe…`。这不改变已打包字节，却破坏 Dockerfile
  复现元数据自洽性；因此 v1 OCI、runtime import 和临时控制镜像只保留为被
  审计拒绝的证据，不进入正式 preflight。
- Dockerfile 自洽修正后的 v2 构建得到 image/config
  `sha256:362c3d1f...880fd`、manifest `sha256:56fcc9b8...1decd`，但 payload
  layer 意外从 v1 `a599892d...92da0` 变为 `829ceb0e...510e0`。Dockerfile
  不进入 payload layer，且 source/artifact/runtime expectation 未变，因此
  这是未解释的确定性异常；v2 暂不进入验收或 import。
- v1/v2 layer 各 5,298 个 tar member 的路径、权限、时间、解压后文件内容和
  SHA256 全部相同；原始 tar 首个差异来自自动扩展头目录
  `PaxHeaders.852152` 与 `PaxHeaders.966541`。数字是 GNU tar 默认
  `exthdr.name=%d/PaxHeaders.%p/%f` 中的进程 PID，因而污染 layer diff-ID
  与 gzip digest。最小修复是显式设置不含 `%p` 的稳定
  `exthdr.name=%d/PaxHeaders/%f`，继续删除 atime/ctime，再以两次独立构建
  digest 完全相同作为门禁。
- 构建器已按上述最小修复显式使用
  `exthdr.name=%d/PaxHeaders/%f`。新增回归用 120 字符文件名强制生成 PAX
  header，并让 `build_layer` 连续启动两个独立 tar 子进程；固定
  CPython 3.12.3 下测试 1/1 通过，未压缩 tar、gzip layer、compressed
  digest、diff-ID、size 和 member count 全部相同。Ruff check/format、
  Python compile 与 diff check 同时通过。
- 发布修复后，v3/v4 两次完整构建分别在独立目录完成，image/config
  `6b5aeb4b...7bb59`、manifest `a629a99e...9105`、candidate layer
  `2ec5ec19...06bd`、diff-ID `f3f1d91c...0ee9`、size 109,147,025 bytes、
  5,298 members 和 index SHA256 `0a1bf2a4...676d` 全部一致；三份新 OCI
  blob 逐字节相同。
- 两次递归验收均为 passed：各自核对 4,744 个源码文件、4 份 rotation、
  7 个基础层 native extension、33 层与精确 Git tree，无 native 覆盖或
  whiteout。v3 build/verification SHA256 为
  `ada8128b...04da`/`6577845f...8ee`，v4 为
  `9226f027...3e5e`/`6c038b8c...152`；报告哈希差异只来自所记录的独立目录
  路径。v3 可进入 Docker/runtime import，但尚未完成该门禁。
- 一次性 Ubuntu 22.04 工具容器安装 `skopeo 1.4.1` 后，已从只读 v3 OCI
  layout 导入 Docker daemon。daemon image ID 为
  `6b5aeb4b...7bb59`，33 层以及 source commit/tree、candidate layer、
  Dockerfile、rotation、runtime expectation 和 base manifest labels 均与
  v3 验收值精确匹配；工具容器已自动删除。尚未执行需要 NVIDIA driver 注入的
  runtime import。
- `23:42:03Z/23:43:11Z` 两次 8/8 GPU 空闲检查间隔 68 秒，随后 v3
  driver-injected runtime import passed：Python/PyTorch/Triton
  `3.12.13/2.11.0+cu129/3.6.0`、78 个 rotation、source/C、
  `reasoning_effort=max` 全部匹配，`cuda_initialized=false`。探针退出后
  `23:44:16Z/23:44:53Z` 均为 8 卡 0 MiB、无 compute app。
- 主探针曾对 `sys.executable` 调用 `resolve()` 而显示 `/usr/bin/python3.12`；
  随后的 CPU-only 原值探针确认 `sys.executable` 为正式 venv
  `/opt/fp8_speed_up_v4_venv/bin/python`。最终 runtime JSON/log 与同 payload
  的 v1 字节相同，SHA256 为 `0910b598...b7a`/`f2e60043...189a`。
- Phase 9 Dockerfile 默认 base 已从旧 a94 tag 切换为接受的 35ab v3 tag并
  发布；新控制镜像 `oscar-glm-stage9-runtime:35ab18464` 实际 ID 为
  `bef0320d...bb7c`，34 层，source revision 与 candidate layer labels 匹配
  v3。CPU-only 容器内 Git 2.34.1、iproute2 5.15.0、Python 3.12.13 和
  package record 均通过。
- 当前 Phase 1/5/7/9 配置和 wrapper 仍冻结在 a94 候选，必须同步更新 source、
  v3 OCI/evidence/control identity。v3 递归验收输出目录名为 `extracted`，
  仅含 candidate layer，尚没有 Phase 7/9 合约需要的 `overlay_rootfs` 和
  lower-layer native symlink；配置切换前需从该已验收内容机械派生新 overlay，
  不能沿用 a94 overlay。
- 恢复时没有在途实验或 GPU compute process；现有 Docker 下载容器仅执行
  `sleep`，不属于本项目 benchmark。Phase 1/5/7/9 正式配置与 wrapper 已统一
  指向 source `35ab1846…`、v3 overlay、manifest `a629a99e…`、config/image
  `6b5aeb4b…`、layer `2ec5ec19…` 与 control image `bef0320d…`；这些修改
  尚待静态测试、报告同步和发布，因此当前仍不能启动正式 GPU 轮次。
- 新身份静态门禁已通过：旧身份搜索为 0，JSON/shell/compile/diff 与 Phase
  1→5→7 派生哈希链一致；固定控制镜像内 Phase 9 16/16、Phase 7 20/20
  测试通过。该结果只证明配置和工具链静态一致，不等同于 containerized
  preflight 或 GPU 性能结果；正式 preflight 必须在主仓库提交推送后执行。
- 已发布配置上的正式 containerized preflight 为 64/64 passed，
  `static_preflight.json` SHA256 为 `f2f1948b…a532`；environment import 与
  server args 都明确 `cuda_initialized=false`。这把 source/OCI/native/
  rotation/baseline/服务参数身份门禁提升为正式结果，但仍不是 TTFT/TPOT
  实验；下一阶段必须先更新报告，再按双空闲门禁运行 TP=8 1K/batch1。
- grouped-head TP=8 探针已把 1K/b1 TTFT 从上一版 `3714.821 ms` 降到
  `1317.120 ms`（`-64.54%`），证明单卡跨 head 复用可转化为端到端收益；
  TPOT 仅从 `205.908` 降到 `202.668 ms`（`-1.57%`），符合纯 decode 仍走
  原路径。对旧 BF16 的 TTFT/TPOT 回退仍为 `+273.71%/+29.33%`，因此不能
  进入完整矩阵。
- 新 critical-rank table 中 grouped prefill stage1 为 `912.239 ms/78`
  （`11.695 ms/层`），而上一版 trace 的同项为 `3300.032 ms`，方向与单卡
  结果一致。当前同一 table 还显示 decode stage1 `1.688 s/9906`、rotation
  `844.125 ms/29952`、KV update CPU/CUDA total
  `5.593 s/440.567 ms`，以及 NCCL all-reduce `27.857 s`；NCCL 数值包含
  跨 rank 等待，不能在逐 rank trace 归因前直接当作首要根因。
- grouped-head 的 8-rank trace 已在无 GPU 的固定控制容器中用
  `ijson==3.4.0.post0` 完成流式分析。8 个 rank 的 prefill wall/kernel
  中位数为 `1353.539450/1271.603995 ms`，kernel coverage 中位数
  `93.9460%`；`_mixed_sparse_prefill_stage1` 为 78 次、
  `911.005226 ms`，占 prefill wall `67.31%`。rotation、MoE 主 kernel 和
  BF16 NCCL all-reduce 中位数分别为 `105.661444/80.559059/74.130554 ms`。
  与上一版 OSCAR trace 相比，prefill wall/kernel/stage1 分别下降
  `63.39%/65.17%/72.39%`，说明 grouped-head 优化收益已经被逐 rank trace
  直接验证；stage1 仍是当前 prefill 第一热点。
- grouped-head trace 的各 rank generation window 中位数约为
  `273.12–278.36 ms`，只比上一版 OSCAR 的约 `279.0–282.54 ms` 小幅改善。
  上一版 profiler 表中的 `_mixed_sparse_decode_stage1` 9984 次包含 78 次
  prefill；扣除 trace 中 `3300.032 ms` 的 prefill 后，9906 次纯 decode
  约 `1.69 s`，与新表的 `1.688 s/9906` 基本相同。这解释了端到端 TPOT
  只改善 `1.57%`：当前 grouped-head 改动没有优化纯 decode。
- BF16 v4 的同一 1K/b1 首个 profiler trace 已用相同分析器、
  `ijson==3.4.0.post0`、4 workers 和固定控制容器完成解析。8-rank prefill
  wall/kernel 中位数为 `248.300301/238.743450 ms`；对应 grouped-head
  OSCAR 为 `1353.539450/1271.603995 ms`，即慢
  `445.12%/432.62%`（`5.451×/5.326×`）。BF16 第一热点是原生
  `_sparse_mla_kernel_final_static`，53 次、`77.446137 ms`；OSCAR 的
  grouped stage1 是 78 次、`911.005226 ms`，因此 TTFT 的主差距仍是
  OSCAR prefill attention 本身，不是调度等待或统计口径。
- 各 rank generation window 中位数再取中位后，BF16 为
  `216.485253 ms`，grouped-head OSCAR 为 `275.123262 ms`，OSCAR 慢
  `27.086%`；这与端到端 TPOT `+29.331%` 同方向且幅度接近，证明 TPOT
  回退也存在于 worker generation 窗口内。BF16/OSCAR profiler critical rank
  分别为 1/6，不能只对比两个 critical-rank table；下一步按 8 个 rank
  同口径聚合 generation、KV update、attention、rotation 和 NCCL 行。
- 8-rank profiler table 同口径聚合已把 decode 首要可控差距定位到
  OSCAR 的 Python/CPU wrapper。每层 `unified_mla_kv_cache_update` 的
  CPU/CUDA time avg 中位数为 `0.564347/0.043536 ms`，而 BF16
  `concat_and_cache_mla` 仅为 `0.012496/0.002759 ms`；按 78 层折算，
  OSCAR 的 KV update CPU 路径多约 `43.05 ms/token`，CUDA 多约
  `3.18 ms/token`。这与实测 generation window 净差
  `58.64 ms/token` 同量级，KV update CPU dispatch/索引链是下一项首要
  优化对象。
- attention wrapper 的 CPU/CUDA time avg 中位数也从 BF16
  `0.399557/0.256852 ms` 增至 OSCAR `0.746773/0.347315 ms`，按 78 层
  对应约 `+27.08/+7.06 ms/token`。OSCAR CUDA 构成中，纯 decode stage1
  `0.170181 ms/层`、三次 rotation 合计约 `0.084600 ms/层`、merge
  `0.005300 ms/层`；它们和 KV update CUDA 的增量共同解释一部分 TPOT。
  NCCL all-reduce 每次中位数由 `1.0345` 增至 `1.2820 ms`，但 rank 间范围
  很大且包含等待，应视为 CPU/attention 路径变慢后的放大结果，而不是先优化
  NCCL。
- 源码只读核对确认 decode 虽已跳过 prefill-only current-history 布尔索引，
  但每层每 token 仍经过 Python custom-op/context/layer 解析、metadata
  类型分派、`seq_lens - 1`、RoPE store、demotion request 索引、
  recent gather→rotation→INT2 store、BF16 store 以及多轮 shape/device/dtype
  校验。`unified_mla_kv_cache_update` 的 `0.327 ms/层` self CPU 与
  `0.564 ms/层` inclusive CPU 正对应这条调用链；下一优化不应再改
  prefill-only 分支，而应为纯 decode 合并 metadata/index 计算和 store/
  demotion dispatch。
- demotion 子路径进一步确认每层每 token 都创建两个临时 CUDA Tensor：
  gather 的 BF16 `[num_rows, 512]` 和 rotation 的 FP32
  `[num_rows, 512]`，随后依次启动 gather、rotation、quantize-store 三个
  kernel；同一 batch 的 `demotion_hp_rows` 还在每层重复执行 GPU advanced
  indexing。worker metadata builder 已在每个 forward step 的 CPU 侧掌握
  request→hp row、最终逻辑长度、demotion request/page/offset，因此可在
  metadata 中一次性物化 decode position/length/demotion hp row，并在 layer
  hot path 复用；layer 侧临时张量也可按最大已见 demotion rows 缓存，避免
  78 层 × 每 token 的重复分配。
- 最小实现边界已经收敛：在 `OscarMLABatchMetadata` 增加由 worker ownership
  一次性生成的 `decode_positions`、`final_seq_lens` 和
  `demotion_hp_rows`；纯 decode layer 只切片这些张量，不再做
  `seq_lens - 1` 或 `hp_rows[demotion_request_indices.long()]`。每层 impl
  缓存最大已见 `num_demotion_rows × 512` 的 BF16 gather 和 FP32 rotation
  scratch，并把精确 slice 传给既有三 kernel demotion 实现。该方案不改变
  prefill/current-history、页映射、量化 kernel、执行顺序或 cache 内容，也
  不提前引入新的融合 kernel。
- 首版实现已按该边界落地：worker 只在 batch metadata 构建时从
  `WorkerCacheMetadata.logical_length/hp_row` 物化三组设备张量；decode layer
  直接切片使用。demotion scratch 只在 device/dtype/latent rank 改变或
  `num_rows` 超过历史容量时重新分配，否则返回同一 storage 的精确前缀 view。
  既有 `oscar_mla_demote_recent` 默认无 scratch 的公开行为不变，runtime
  仅通过新增可选输出参数复用缓存。
- 二次审计删除了被 `demotion_hp_rows` 完全替代的
  `demotion_request_indices` 设备张量，避免每 step 多构造/传输一个不再消费的
  Tensor。两请求 decode 回归还把主 attention `seq_lens` 故意改为错误值，
  仍要求 store 使用 worker 预计算的 `[336,599]` 与 `[337,600]`，并断言
  demotion 直接收到预计算 HP row；这把“layer 不再重复算索引”变成可验证
  合约。
- 静态门禁核对确认 backend 第 380 行的 `torch.cuda.empty_cache()` 由旧提交
  `53d8be94f` 引入且完全不在本次 diff；attention backend 文档生成器建议的
  `oscar_mla_int2`/CUDA graph 能力行也是此前功能留下的基线漂移，与本轮
  metadata/scratch 无关。为保持外科式改动，前者精确 skip，后者生成改写已
  撤回并精确 skip；两个本次触及且原本缺头的 Python 文件则保留 hook 自动补齐
  的 SPDX 头。
- metadata/scratch 优化最终提交为
  `14c768b406b3e39a2d4d5be77a9046ac7ccc26d1`，tree
  `4ad8be8a10fb07321d4ac9c81d31d009e854bde9`，本地与远端分支精确一致。
  完整 CPU 套件最终为 96 passed、29 个 CUDA 显式 skip、0 failed、
  17 warnings、30.15 秒；因此只能证明源码语义和 CPU/静态门禁，不能推断
  苹果800 CUDA 或 TTFT/TPOT 已改善。
- 中文报告已新增 7.16，记录上述源码边界、TDD 红/绿灯、完整 CPU 结果和两个
  已审计旧 hook 的精确 skip；报告同时明确下一阶段才执行 cold-cache CUDA
  与 TP=8 探针。修改后一级章节 1–8、7.1–7.16、交叉引用和禁用旧术语检查
  均通过。
- metadata/scratch 源码的完整苹果800 cold-cache CUDA 回归使用 GPU 0 和
  固定控制镜像 `sha256:bef0320d…bb7c`。CPU 套件中显式 skip 的 29 个 CUDA
  节点本轮全部执行，最终为 125 passed、0 skipped、0 failed、19 warnings、
  80.88 秒；cold Triton cache 为 380 文件、29,222,093 bytes。该结果证明
  正确性回归通过，不代表 TP=8 性能已改善。
- CUDA 轮次为
  `20260731T0116Z_decode_metadata_full_cuda_v1`，绑定主仓库
  `4d93b0df…aeb`、源码 `14c768b…6d1`、tree `4ad8be8a…de9`。pytest
  日志 SHA256 为
  `a923d118983186600cc06e6a372d0671f0da8f0c6bfb32f22f2b3d086eeab02e`；
  双空闲检查间隔 67 秒，容器退出后 8 卡均为 0 MiB、无 compute app。
- 新候选的两次独立构建和递归验收全部通过，共同不可变身份为：
  image/config `sha256:dbd78a77…f0a99`、manifest
  `sha256:52a74b15…68e4`、candidate layer
  `sha256:4b907048…3312`、diff-ID `sha256:619ae460…7d9f`。layer 为
  109,147,574 bytes、5,298 members，不含 native extension 或 whiteout。
- 两个 OCI 的 index/config/manifest/layer 逐字节一致，`index.json`
  SHA256 均为 `3ac25034…d303`；每轮验收都核对 4,744 个源码文件、4 份
  rotation、7 个基础层原生扩展和 33 层身份。build/verification 报告哈希
  只因记录的输出目录不同而不同，不影响不可变身份；构建/验收全程 CPU-only。
- 正式 v1 已由一次性 Ubuntu 22.04 工具容器中的 `skopeo 1.4.1` 从只读
  OCI layout 导入 Docker daemon；daemon image ID 为
  `sha256:dbd78a779001300cca5f3802a4e69def3de5a82eb96746c614a5e6d7e14f0a99`，
  共 33 层。source commit/tree、candidate layer、Dockerfile、rotation、
  runtime expectation 与 base manifest labels 全部匹配。导入日志和 inspect
  JSON SHA256 分别为 `70c3dc1c…0b5b9`、`3089a6a7…8b5f`；该阶段没有注入
  NVIDIA runtime，8 卡始终为 0 MiB。
- Shawn 于 2026-07-31 明确把性能优化迭代格点由 1K/b1 改为 32K/b1。
  该格点使用精确 32,768 输入 token、128 输出 token、并发 1、temperature
  0、ignore EOS；每格先 warm-up，再执行 3 轮正式测量并采集 8+8+1
  profiler。现有 BF16 v4 同格点为 TTFT `12528.026 ms`、TPOT
  `178.832 ms`、吞吐 `0.02838 req/s`。历史 1K/b1 仍保留作根因证据，
  但不再作为本轮优化验收负载。
- metadata/scratch 候选首次 driver-injected runtime import 命令遗漏
  `docker run -i`，因此 `python -` 没有收到 heredoc，返回 0 但 JSON/log
  均为空文件。该轮没有执行任何 import，不能计为通过；GPU 0–7 始终
  0 MiB、无 compute process。修正只补 stdin 透传，不改变探针断言或镜像。
- 补上 stdin 后，探针成功导入候选 `vllm` 和 `vllm._C`，随后因错误复用了
  旧模块路径 `vllm.entrypoints.openai.protocol` 而退出；当前源码真实路径为
  `vllm.entrypoints.openai.chat_completion.protocol`。这不是候选运行时
  失败；退出后 8 卡仍为 0 MiB、无 compute process。
- 使用当前真实 protocol 路径的有效 runtime import 已通过，JSON/log SHA256
  分别为 `0910b598…7b7a`/`f2e60043…189a`。正式
  Python/PyTorch/Triton 为 `3.12.13/2.11.0+cu129/3.6.0`，vLLM Python
  与 `_C` 均来自 `/opt/vllm_glm52_v1`；78 层 rotation、两项 rotation
  hash、runtime expectation 和 `reasoning_effort=max` 均匹配。探针最终
  `cuda_initialized=false`，退出后 8 卡均 0 MiB、无 compute process。
- Stage 9 控制镜像 Dockerfile 只把默认 base 从旧
  `glm52-oscar-a800-phase6-35ab18464-0275043c:latest` 切换到新
  `glm52-oscar-a800-phase6-14c768b40-0275043c:latest`，新文件 SHA256
  为 `6b6f4d1d…e2d3e`。daemon 中新 base 的 image ID、33 层和
  source commit/tree 已再次确认精确为 `dbd78a77…f0a99`、
  `14c768b…6d1`/`4ad8be8a…de9`。
- CPU-only 控制镜像构建
  `20260731T015740Z_runtime_14c768b40_v1` 已成功，tag/image ID 为
  `oscar-glm-stage9-runtime:14c768b40` /
  `sha256:84c48782f440d2080a81347bfa31ec1e3bbf77bc6151629ef43d879bbe90989f`。
  34 层的前 33 层与新候选逐层相同，全部 inherited labels 相等；Git
  `2.34.1`、iproute2 `5.15.0`、Python/glibc `3.12.13/2.35` 与控制包
  清单匹配，`cuda_initialized=false`。build/inspect/runtime-check
  SHA256 为 `691e629a…d50`/`de86beae…489`/`5ac65b5d…f20`，8 卡全程
  0 MiB。
- 新配置迁移需要更新 Phase 1/5/7/9 四份 JSON、对应 wrapper、Phase 7
  两处候选 manifest 指纹和 Phase 9 固定 control image 测试值；正式范围内
  仍有旧 `35ab1846` source/OCI/control/path 引用，迁移完成前不能运行
  preflight。
- 新候选 `extracted-layer` 与旧正式 overlay 均为 4,744 个源码文件、约
  76 MiB。旧 overlay 另外含 6 个指向只读 phase0 rootfs 的 vLLM native
  symlink；新 extracted layer 按设计不携带这些基础层 `.so`。下一步从新
  extracted layer 机械复制 overlay，并按完全相同的 6 个绝对 target 建链。
- 首次机械复制被工具会话提前终止，新 overlay 只有 3,214 个文件、0 个链接；
  未修改 extracted source，也未进入配置。该中间态不计为有效 overlay，下一步
  以同一源幂等补齐并执行完整计数/链接/hash 门禁。
- 幂等补全后新 overlay 达到 4,744 个源码文件；建链循环虽在已有
  `cumem_allocator` 处退出，但复查当前恰有 6 个链接，路径与可见 target
  均和旧正式 overlay 对应。仍需逐项程序化比较 target 和目标文件 SHA256
  后才接受。
- 新 overlay 最终程序化门禁通过：4,744 个普通文件、6 个 symlink；相对路径
  与 link target 和旧正式 overlay 完全相同，6 个 target 存在且 SHA256
  分别为冻结的 `_C` `1812bd…`、stable `e79f6e…`、MoE `c59dc1…`、
  cumem `a73a69…`、FA2 `f8926e…`、FA3 `170b23…`。
- Phase 1/5/7/9 配置与所有正式 wrapper 已切换到新 source/OCI/control/
  overlay。依赖顺序重算后的 SHA256 为 `d588e627…c820`、
  `b7a58d10…4d66`、`680708fc…b15f`、`1bdfabb9…b379`。旧身份扫描清零；
  4 个 JSON、9 个 shell、Python compile、`git diff --check` 均通过。
- 新控制镜像以正式 venv 加一次性 pytest/tblib target 运行 Phase 7+9 全部
  7 个工具测试文件，结果为 39 passed、0 failed、10.30 秒。唯一 warning
  是只读项目挂载无法写 pytest cache，不影响测试内容或产物。
- 分阶段复跑进一步确认 Phase 7 为 20 passed、Phase 9 为 19 passed，
  两阶段均 0 failed、退出码 0；对应日志 SHA256 为
  `6203ef06…324`/`df2842e3…c89`。因此配置迁移后的工具单测门禁已通过，
  下一道门禁是递归 verifier 和容器化 dry-run preflight，而非直接启动
  32K/batch1 性能负载。
- 宿主机上的 Phase 5 `rootfs_runtime_tree_match` 失败是 NFS mode 假阳性：
  内容 SHA256 可匹配，但普通文件被呈现为 `777`，无法在宿主 namespace
  通过 Git `100644` mode 门禁。正式控制容器的 bind-mount namespace 中，
  Phase 9 递归 verifier 64/64 全通过。无 driver 的容器会继续在导入
  `vllm._C` 时缺少 `libcuda.so.1`，所以完整 preflight 必须注入 NVIDIA
  driver，但仍应验证 `cuda_initialized=false`。
- 新候选的 driver-injected 完整 preflight 已通过：静态 64/64、固定环境
  import 和正式服务参数解析均成功，后两者都明确记录
  `cuda_initialized=false`。解析出的关键参数为 TP=8、PP=1、
  `TRITON_MLA_SPARSE`、`oscar_mla_int2`、131072 max model length、
  2048 max batched tokens、eager 和关闭 async scheduling；因此可以在发布
  本阶段报告后进入 32K/batch1 正式性能探针。
- 32K/batch1 首次探针
  `20260731T0225Z_stage9_candidate_decode_metadata_probe_32k_b1_v1`
  的三轮测量均为 3/3 completed、0 failed。TTFT 分别为
  `106666.970/106606.493/106660.424 ms`，TPOT 为
  `200.547/200.303/199.203 ms`，吞吐为
  `0.007568/0.007573/0.007578 req/s`。中位数相对 BF16 为 TTFT
  `+751.37%`、TPOT `+12.01%`、吞吐 `-73.32%`。这些是实际诊断测量值，
  但整轮不构成正式 passed 单格。
- 同轮 profile 已生成 8 份 worker trace、8 份 CUDA table 和 1 份 frontend
  trace。rank 0 有 144 个 execute context，前 16 个各处理 2,048 个
  prefill token，合计 32,768；首个窗口约 `4088 ms`，最后一个约
  `6894 ms`，后续 generation 窗口大多约 `266–271 ms`。因此当前
  106.7 秒 TTFT 的首要事实是 16 个 chunked prefill 窗口累计，而非单个
  prefill 窗口。
- 正式 runner 在 profile 命令完成后发现主仓库新增未跟踪的
  `OSCAR精度与性能优化记录.md`，触发
  `RuntimeError: repository became dirty`。profiler bundle 校验和单格/
  总 summary 均未执行，外层退出码 1；容器删除后 8 卡均 0 MiB、无
  compute process。该实时记录必须先纳入 Git，再以新 run ID 重跑。
- 现有 `scripts/phase9/analyze_prefill_trace.py` 明确只接受一个 prefill
  window，解析上述 trace 时在第二个窗口 fail closed。下一步最小扩展为：
  保持单窗口输出兼容，聚合 generation 前全部正 token prefill chunk，
  同时报告 chunk 数、总 token、总 wall/kernel 时间，并增加两 chunk 回归
  测试；完成后再对 8 rank 归因。
- 多 chunk 分析器已按上述范围实现，format version 从 1 升为 2；双 chunk
  测试先复现旧实现失败，改动后定向 2/2 passed，固定控制镜像中的 Phase 9
  三个工具测试文件为 20/20 passed。脚本 SHA256 为
  `8b6b2393f93be3783f47ccbe3ecb26020cc2b65526c99fcb464c4768ce4330f7`。
- OSCAR 8-rank 聚合全部为 144 execute context、16 chunk、32,768 token。
  prefill wall/kernel 中位数为 `105753.449/105609.233 ms`，kernel coverage
  `99.8628%`；generation rank median 再取中位为 `268.625 ms`。聚合结果
  SHA256 为 `cf488787…ffa7`。
- 用相同脚本解析 BF16 v4 32K/b1 的精确 8 份 trace，全部同样为 16 chunk、
  32,768 token；prefill wall/kernel 中位数为
  `10086.470/9533.580 ms`，generation rank median 再取中位为
  `223.325 ms`，结果 SHA256 为 `06eecce0…58d`。
- OSCAR 相对 BF16 的 prefill wall/kernel/generation trace 回退分别为
  `+948.47%/+1007.76%/+20.28%`。OSCAR
  `_mixed_sparse_prefill_stage1` 精确 1,248 次、累计中位数
  `93913.327 ms`，平均 `75.251 ms/层/chunk`，占 prefill wall
  `88.80%`；BF16 原生 sparse MLA kernel 为 `3384.374 ms`。因此下一项
  性能优化应针对 grouped prefill stage1，而不是继续改 decode metadata。
- Phase 9 prefill benchmark 已新增显式 `--seq-len`，并拒绝不满足
  prefix+recent、固定 top-k 上限或 16-token history block 对齐的形状。
  2,048 形状中 full/cropped top-k 相同，因此配置集合只保留 IEEE split16
  参考与 grouped split1 候选。新增测试在旧实现上 3/3 预期失败，改动后
  Phase 9 三个工具测试文件 21/21 passed；尚未分配 GPU。
- 2K IEEE 单层轮次 `20260731T0345Z_prefill_2k_ieee_v1` 在 GPU 0 上通过。
  split16/grouped split1 CUDA 中位数为 `195.772/47.158 ms`，后者
  `4.151×` 加速；峰值增量显存为 `1185.063/224.125 MiB`。grouped
  output/LSE 最大绝对差为 `3.3379e-6/9.5367e-7`，低于 `0.002/0.002`。
  result/log SHA256 为 `f98578db…f8a`/`cf0b5f7d…616`。
- TF32 候选源码 `24938975f…` 只把 grouped prefill stage1 的 5 个
  score/value dot 从 IEEE 切到 TF32；decode/split16 reference 与所有 cache/
  LSE 语义不变。CPU interpreter/head-block 为 6/6 passed，全部适用
  pre-commit hooks 通过，源码已推送；GPU 精度/性能仍待测。
- TF32 首次 GPU 筛选
  `20260731T0354Z_prefill_2k_tf32_v1` 在编译 grouped stage1 时失败：
  shared memory 需要 `169,984 bytes`，超过苹果800 单 block 上限
  `166,912 bytes`。未生成 `result.json`，没有精度/性能数字；日志 SHA256
  为 `3024ebc9…19a1`。退出后 8 卡均空闲。
- CPU-only SM80 资源 sweep
  `20260731T0400Z_tf32_offline_resource_sweep_v1` 证明：简单把部分 dot 改回
  IEEE 需要 `169,984–204,800 bytes`，不能解决超限；原生 BF16 pool 用
  BF16 tensor core、history 保留 TF32 时为 `135,168 bytes`。summary
  SHA256 `1065b841…6db`；该值只代表离线编译资源，不代表 GPU 精度/性能。
- hybrid 源码 `b9626ce9f…` 只修改 grouped prefill kernel：原生 BF16 pool/
  RoPE 使用 BF16 tensor core，history score/value 保留 TF32，softmax/LSE/
  accumulator 保持 FP32。CPU 定向 6/6 与全部适用 hooks 通过，已推送；
  GPU 精度/性能仍待测。
- hybrid GPU 轮次 `20260731T0408Z_prefill_2k_hybrid_v1` 以
  `135,168 bytes` shared memory 成功 launch，但 output/LSE 最大绝对误差
  `0.004933/0.002036` 超过 `0.002/0.002`，未进入 warm-up/计时且无
  `result.json`。日志 SHA256 `2bc3e050…fd95`，退出后 8 卡空闲。
- CPU-only SM80 离线编译
  `20260731T0412Z_hybrid_value_resource_sweep_v1` 证明：score 继续使用
  BF16 tensor core，仅将 BF16 value 累加恢复为 FP32 probability/TF32 dot，
  shared memory 仍为 `135,168 bytes`，较 `166,912 bytes` 上限低
  `31,744 bytes`。该结果只证明资源可行，不代表 GPU 精度或性能。
- value 精度恢复源码 `b247211c9…` 已推送；仅 1 个 kernel 文件
  3 行新增/2 行删除。固定 CPU 容器定向 6/6 passed，ruff 与全部适用 hooks
  通过。GPU 精度/性能仍待相同协议筛选。
- 冻结精度协议自 `60acb2e8` 起一直是
  `torch.allclose(atol=0.002, rtol=0.002)`，不是单独要求
  `max_abs<=0.002`。报告后段“最大绝对误差硬门限”的表述与可执行协议冲突，
  应修正并继续同时披露 max_abs。
- GPU 轮次 `20260731T0422Z_prefill_2k_value_precision_v1` 按冻结协议状态为
  `passed`。split16/split1 中位数 `195.836/26.906 ms`，加速 `7.279×`；
  split1 output/LSE max_abs 为 `0.0049126/0.0020361`，但逐元素 allclose
  通过。实际 shared memory `135,168 bytes`，结果 SHA256 `4749ee12…82b1`。
- 完整 CUDA 首次启动因重复传入镜像自带的 `/bin/bash` entrypoint，在进入
  Python 前失败，0 测试/0 cache 文件；保留失败日志但不计作 CUDA 结果。
- 修正后的轮次 `20260731T0431Z_value_precision_full_cuda_v2` 使用 GPU 0
  和独立空 cache，结果 125 passed、0 skipped/failed、19 warnings、
  80.32 秒。cache 为 380 文件、26,557,655 bytes；日志 SHA256
  `392cccbe…3fbf`。该结果证明完整 CUDA 正确性回归通过，不代表 32K/b1
  TTFT/TPOT 已改善。
- 新候选 v1/v2 两次独立构建和递归验收均通过，共同不可变身份为：
  image/config `8053b791…9e46`、manifest `c9230c5f…cb94`、candidate
  layer `94ee660d…f3e6`、diff-ID `1af1788b…8679`、index
  `5d866599…7108`。两份 OCI 关键 blob 逐字节一致。
- 每轮各验证 4,744 个源码文件、4 份 rotation、7 个基础层 native extension、
  33 层和精确 Git tree；candidate layer 无 native/whiteout。本阶段 CPU-only，
  尚未证明 daemon/runtime/preflight 或 32K/b1 性能。
- v1 已由一次性 `skopeo 1.4.1` 工具容器导入 Docker daemon；image ID
  `8053b791…9e46`、33 层及 8 项关键 labels 均与验收值匹配。import/inspect
  SHA256 为 `33bcbe79…d5c3`/`3a4747ef…c0c`，工具容器已删除，全程未注入
  GPU。runtime import 尚未执行。
- value 精度恢复候选的 runtime import 首轮失败只来自探针误读 rotation
  payload：顶层为 `format_version/rotations` 两个键，内层才有 78 项。v2
  已修正该点，但相对既有通过协议额外导入 `flashinfer` 与
  `flashinfer.jit`，最终触发 `cuda_initialized=false` 断言。两轮均在输出
  JSON 前退出，不能记作候选 runtime 失败或通过。
- 冻结 runtime import 证据的 FlashInfer 校验方式是
  `importlib.metadata.version("flashinfer-python")` 与
  `importlib.metadata.version("flashinfer-jit-cache")`，不需要导入
  FlashInfer 模块。下一轮应精确复用该协议，同时继续导入候选 `vllm._C`、
  校验 78 层 rotation、三个 artifact hash 和 `reasoning_effort=max`。
- 按上述冻结协议执行的 v3 已通过，`cuda_initialized=false`；实际包版本、
  候选 vLLM 路径、78 层 rotation、三个 artifact hash 与
  `reasoning_effort=max` 全部匹配。JSON/log SHA256 为
  `0910b598…7b7a`/`f2e60043…189a`，与此前有效候选相同，证明当前
  value 精度恢复候选没有改变固定运行时环境 payload。
- Stage 9 控制镜像 Dockerfile 只需替换默认 base tag；安装 Git/iproute2 与
  entrypoint 的其余指令均不变。新默认 base 的 daemon image ID、33 层、
  source commit/tree 和 candidate layer label 已与 Phase 6 验收值匹配。
- 新控制镜像 ID 为 `edbbc87d…b1b8`，其 34 层中的前 33 层与候选
  `8053b791…9e46` 精确一致，labels 完全继承；固定 CPU 环境与包清单通过，
  `cuda_initialized=false`。因此可以在发布阶段记录后机械迁移正式配置和
  overlay，不需要重新修改候选源码或 OCI。
- 新 overlay 的 4,749 个普通文件与候选 extracted layer 的相对路径/内容
  清单 SHA256 同为 `0568662b…60d`；其中源码 4,744 个、artifact 5 个。
  另有 6 个 native symlink，其相对路径、绝对目标和目标 SHA256 与旧正式
  overlay 精确一致。NFS 复制中间态未被接受。
- Phase 1/5/7/9 配置与 wrapper 已迁移到 `b247211c9…`；四份配置 SHA256
  为 `e6e5b599…2d25`/`40083bf3…801b`/`871feea0…d50a`/
  `e2c764d7…9114`。JSON、9 个 shell、Python compile、diff 和旧身份清零
  门禁通过。
- Phase 7 首个工具测试因漏挂载冻结 Python 依赖目录得到 19 passed/1 failed；
  补上只读 `/dev/shm/oscar-glm-recovery-tools` 后为 20/20 passed。Phase 9
  为 21/21 passed。有效日志 SHA256 为 `53c86a8a…bd62`/
  `92f4f2ce…f5c`。
- 正确控制容器 bind-mount 命名空间中的递归静态 verifier 为 64/64 passed，
  JSON SHA256 `bfad6c62…32c7`。无 driver 的后续 import 因缺少
  `libcuda.so.1` 退出，故完整 dry-run 退出码 1；它不否定静态结果，也不能
  代替 driver-injected preflight。
- 新候选正式 preflight
  `20260731T0528Z_stage9_candidate_b247211c9_preflight_v1` 前的两次
  8/8 GPU 空闲检查为 `05:26:12Z/05:27:29Z`，间隔 77 秒，均为
  0 MiB、0% 且没有 compute process。
- preflight 退出码 0，静态 64/64 passed；固定环境 import 与服务参数解析
  均为 `cuda_initialized=false`。解析参数为 TP=8、PP=1、
  `TRITON_MLA_SPARSE`、`oscar_mla_int2`、max model length 131,072、
  max batched tokens 2,048、eager、关闭 async scheduling 和 torch
  profiler。
- preflight log/static/fixed/parsed SHA256 为
  `e773a9d2…a362`/`5cf51c4b…d6c`/`1805df1c…2464`/
  `22b4af3b…aa5`。容器退出后 8 卡全空闲；该结果只关闭正式运行前门禁，
  尚没有产生当前候选的 32K/b1 TTFT、TPOT 或吞吐。
- 正式 32K/b1 轮次
  `20260731T0536Z_stage9_candidate_b247211c9_32k_b1_v1` 绑定主仓库
  `1d32d26c…`、源码 `b247211c…` 和配置 `e2c764d7…9114`。外层双空闲
  `05:34:03Z/05:35:12Z` 间隔 69 秒，容器内双空闲
  `05:37:14Z/05:38:18Z` 间隔 64 秒，四次均 8/8 空闲。
- 三轮均 3/3 completed、0 failed；`mean` 中位数为
  TTFT `47143.207 ms`、TPOT `199.458 ms`、吞吐 `0.013795 req/s`。
  相对上一 OSCAR 为 `-55.80%/-0.42%/+82.16%`，相对 BF16 为
  `+276.30%/+11.53%/-51.39%`。TPOT 在 20% 门限内，TTFT 仍约 BF16
  `3.76×`。
- 单格/总 summary 和 profiler 均 passed；profile 787.794 秒，8 tables、
  8 worker traces、1 frontend trace 均通过。prefill stage1 1,248 次的
  8-rank CUDA total 中位数 `34398 ms`，相对旧 `93913.327 ms` 下降
  `63.37%`，但仍为 BF16 `3384.374 ms` 的约 `10.16×`。
- 总/单格/profile/runner SHA256 为
  `ae1ffb5c…3418`/`139c2d8c…801b`/`46ab91fa…c80b`/
  `8812ac76…e7c9`。正式 runner 退出码 0；`06:17:50Z` 8 卡 0 MiB、
  0%，无 compute process。
- 2026-07-31：value 精度候选正式 32K/batch1 的 8-rank 多 chunk trace 已在
  CPU-only 固定环境完成。8 个 rank 均为 144 个 execute context、16 个
  prefill chunk、32,768 tokens；prefill wall/kernel/generation 中位数为
  `46934.364/46100.251/269.297 ms`。grouped prefill stage1 为
  `34398.099 ms`、占 wall `73.29%`，其相对 BF16 原生 prefill attention
  的超额时间解释总 prefill wall 差距的 `84.17%`。去除 stage1 后当前/上一
  OSCAR 剩余 wall 为 `12536.264/11840.122 ms`，当前反而高 `5.88%`；
  因而下一轮仍应优化 stage1，而不是调度间隙。有效 summary SHA256 为
  `599c035a36a65135885aee53f9a9964e4db181f25e62185a0182ecfcb4fc5a58`。
- 2026-07-31：当前 grouped stage1 的实际 SM80 cubin 为每线程 255 registers、
  656-byte stack，Triton metadata 为 135,168-byte shared memory。源码
  `b87a401daf55b557b0b052f302fd35be222d1ff1` 只把 grouped prefill/split1
  从 4 warps 调为 8，decode 保持 4 warps；CPU/interpreter 6/6、ruff 和全部
  适用提交 hooks 通过。该阶段没有 GPU 结果，不能预判资源或性能改善。
- 2026-07-31：grouped prefill 8-warps 单卡筛选
  `20260731T0645Z_prefill_2k_8warps_v1` 已完成。固定 GPU 0、2,048 query、
  2,048 top-k、8 local heads、latent 512、三段式 `64/1728/256`，5 次
  warm-up、7 次正式测量。8-warps grouped split1 CUDA 中位数为
  `24.089599609375 ms`，同轮 split16 为 `195.78982543945312 ms`，
  加速 `8.127566610250225×`；相对旧 4-warps split1
  `26.90559959411621 ms` 降低约 `10.47%`。冻结
  `torch.allclose(atol=rtol=0.002)` 的 output/LSE 均通过，诊断误差与
  b247 候选一致。实际 SM80 产物保持 `135168 bytes` shared memory，
  registers 从 255 降至 247，stack 从 656 bytes 降至 0。有效
  result/log SHA256 为
  `5fed778844380192f642b5ba43ddae1394534d27cd664407641a7e05795791fb` /
  `b2e43fd7a4d0142f3c3bfeef11661cc50ce014828e6298874d54813f82380f39`。
  该结果只证明单层资源、精度和性能门禁通过，不代表完整 CUDA 回归或
  32K/batch1 端到端已经通过；报告补录完成前不进入下一次 GPU 分配。
- 2026-07-31：8-warps 完整 cold-cache CUDA 回归有效轮次
  `20260731T0657Z_8warps_full_cuda_v2` 固定只使用 GPU 0，绑定源码
  `b87a401daf55b557b0b052f302fd35be222d1ff1` / tree
  `7df314f222234b3744794d59736e8bba36f8f8ae`。结果为 125 passed、
  0 skipped、0 failed、18 warnings、78.19 秒；独立 Triton cache
  380 个文件、内容合计 25,886,842 bytes。pytest/exit-code SHA256 为
  `bd3b6d624a79c5867279f28dcb924c56d8a9a812df5ad8ae4cc5538c1a0f63a1` /
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。
  v1 因错误猜测 uv 路径在 pytest 前以 127 退出，0 测试/0 cache，不计为
  CUDA 结果；CPU-only 探针确认实际路径 `/usr/local/bin/uv` 后才执行 v2。
  v2 退出后 `06:58:25Z` 8 卡均为 0 MiB、0%，没有 compute process。
  该结果关闭完整 CUDA 正确性门禁，但尚未生成新 OCI 或 32K/batch1
  端到端数据。
- 2026-07-31：Phase 6 构建输入已最小切换到 8-warps 源码
  `b87a401daf55b557b0b052f302fd35be222d1ff1` / tree
  `7df314f222234b3744794d59736e8bba36f8f8ae`，新 tag 为
  `glm52-oscar-a800-phase6-b87a401da-0275043c`。Dockerfile SHA256 为
  `7c21383b7964210044d0820b2ff0587b3c20601b79bb066e4b03bfe5a46130d0`；
  base manifest、rotation、runtime expectation 与确定性 PAX 逻辑均未改。
  固定 Python 3.12.13 的 PAX 回归 1/1 passed；JSON、源码 commit/tree、
  Dockerfile hash、Python compile 和 diff 门禁通过。当前尚未启动 OCI 构建，
  不能记录任何新 image/config、manifest 或 layer digest。
- 2026-07-31：8-warps 候选在两个独立目录完成 CPU-only OCI 构建和递归
  验收，轮次分别为
  `20260731T0707Z_candidate_b87a401da_8warps_v1` 与
  `20260731T0710Z_candidate_b87a401da_8warps_v2_rebuild`，两轮均
  `status=passed`。共同 image/config 为
  `eef27939ae476dd0d727a49ffeaacfcea7c7dfba3311de3146a6cd8feca06eab`，
  manifest 为
  `57c03fca6636b9d0ebd97d5d8aecf94b82e872995c76057938738f3e469e484a`；
  33 层中的候选层 digest/diff-ID 为
  `2ac4b80a9c50ba87db18b41646e5fbf843220e61d8206069b719a7dcf8a2d108` /
  `ab049b45296d89f143fe3e3caa9715a2e9f06404974ba9290ee12ac5fae1c02a`，
  size/member 为 `109147643/5298`。两轮均精确验证 4,744 个源码文件、
  4 份 rotation、7 个基础层 native extension、基础层逐层继承，候选层
  无 native/whiteout。
- 两轮 `oci/index.json`、config blob、manifest blob 和候选 layer blob
  逐字节完全相同；index SHA256 为
  `beea07784088897fbcf3339bae8f00faacb41e11d0adb969112c27348c54af88`。
  v1 build/verify/log SHA256 为
  `63eef67f…eab9`/`b5fceb3a…2ae`/`5ed02a8b…ff9a`，v2 为
  `b83ab95a…94e4`/`1fbf342e…83f9`/`686eda75…4e0`；这些记录文件包含各自
  输出路径，因此哈希不同，不影响不可变 OCI 四项完全一致。阶段结束后
  8 张 GPU 仍为 0 MiB、0%，没有 compute process。
- 2026-07-31：v1 已由一次性 Ubuntu 22.04 工具容器中的 `skopeo 1.4.1`
  从只读 OCI layout 导入 Docker daemon，退出码 0；工具容器自动删除。
  daemon image ID 为 `eef27939…6eab`、层数 33、最后一层 diff-ID 为
  `ab049b45…c02a`，tag 与 source commit/tree、candidate layer、
  Dockerfile、rotation manifest/rotations、runtime expectation、base
  manifest 共 8 项 labels 全部匹配。import/inspect/audit SHA256 为
  `52b5e778…9940`/`f5106ea2…c85c`/`0435fca1…c4ee`。
- 导入期间没有传 `--gpus` 或注入 NVIDIA runtime；前后 8 卡均为
  0 MiB、0%，无 compute process。APT 的非必需 deadsnakes PPA 出现一次
  TLS warning，但已有索引成功安装固定 `skopeo 1.4.1`，导入和身份审计均
  通过，因此不构成结果失败。driver-injected runtime import 尚未执行。
- 2026-07-31：新候选 runtime import 前的双空闲检查为
  `07:26:23Z/07:27:34Z`，间隔 71 秒；两次均 8/8 卡 0 MiB、0%，无
  compute process。有效轮次固定只向容器注入 GPU 0 的驱动可见性，精确复用
  冻结协议，FlashInfer 只通过 `importlib.metadata` 读取版本，没有导入
  `flashinfer/flashinfer.jit`。
- runtime import 退出码 0、状态 passed：Python/PyTorch/Triton 为
  `3.12.13/2.11.0+cu129/3.6.0`，Transformers/Tokenizers 为
  `5.8.1/0.22.2`，FlashInfer Python/JIT cache 为
  `0.6.6/0.6.6+cu129`；候选 vLLM Python/`_C`、78 层 rotation、三个
  artifact hash 与 `reasoning_effort=max` 全部匹配，
  `cuda_initialized=false`。JSON/log SHA256 为
  `0910b598…7b7a`/`f2e60043…189a`，与此前同协议证据逐字节一致；
  idle log SHA256 为 `04358a5b…1982`。容器自动删除，`07:28:38Z`
  复查 8 卡全空闲。
- 2026-07-31：Stage 9 控制镜像 Dockerfile 仅把默认 base 从
  `glm52-oscar-a800-phase6-b247211c9-0275043c:latest` 切换为
  `glm52-oscar-a800-phase6-b87a401da-0275043c:latest`，其余内容未改；
  新文件 SHA256 为
  `580ae65d4e89094d08ceb17cd158cc28217fbc01fc235e822c90e99cdfdbbd9a`。
  daemon base 已复核为 image ID `eef27939…6eab`、33 层、source
  `b87a401d…`/tree `7df314f2…`、candidate layer `2ac4b80a…d108`。
  当前只冻结了 CPU-only 构建入口，尚未生成新控制镜像。
- 2026-07-31：控制入口由主仓库 `b28f825` 发布后，CPU-only 轮次
  `20260731T0736Z_runtime_b87a401da_v1` 构建
  `oscar-glm-stage9-runtime:b87a401da` 成功，image ID 为
  `sha256:f38a75eca80d8c460a331d9046832152d949317f7bb5d9286809f71d991400b4`。
  控制镜像 34 层中的前 33 层与候选 `eef27939…6eab` 逐层相同，全部 inherited
  labels、环境和 `/bin/bash` entrypoint 匹配。
- CPU runtime 检查 passed：Git/iproute2 `2.34.1/5.15.0`、
  Python/glibc `3.12.13/2.35`、固定包清单一致，
  `cuda_initialized=false`。build/exit/inspect/identity/runtime SHA256
  为 `fc7ca5c5…c26d`/`9a271f2a…86aa`/`1cad0831…0ef9`/
  `21a9b01d…570b`/`5ac65b5d…1f20`。本阶段未注入 NVIDIA runtime，
  `07:38:15Z` 8 卡全空闲。
- 2026-07-31：b87 v1 extracted layer 已机械派生正式 overlay；两边
  4,749 个普通文件的递归清单 SHA256 同为 `20ee2d14…dc1`，overlay
  另含 6 个指向冻结 phase0 native rootfs 的链接，目标存在且哈希不变。
  Phase 1/5/7/9 配置按依赖顺序迁移后 SHA256 为
  `3258f706…0b05`/`2c945b1d…52ce`/`78590b20…2483`/
  `f15100e4…05f0`；JSON、9 个 shell、Python compile、旧身份清零和 diff
  门禁均通过。
- 新控制镜像内 Phase 7/9 有效工具测试分别为 20/20、21/21 passed；日志
  SHA256 为 `c36356c3…723f`/`3ed122ef…6f10`。Phase 7 首轮因没有覆盖
  `/bin/bash` entrypoint，在 collection 前把 uv 二进制误作脚本解释并以
  126 退出，不是测试断言失败。
- 递归 verifier v1 漏挂模型；v2 只剩已知 NFS mode 假失败；v3 按正式协议
  覆盖冻结 Docker source volume 后 64/64 passed，有效 JSON SHA256 为
  `27926ec6…c72b`。全程未注入 GPU，`07:52:09Z` 8 卡全空闲。
- 2026-07-31：静态迁移由主仓库 `e24f754` 发布后，
  `07:56:24Z/07:57:24Z` 完成间隔 60 秒的双空闲检查；8 卡两次全空闲。
  driver-injected preflight
  `20260731T0758Z_stage9_candidate_b87a401da_preflight_v1`
  退出码 0，静态 64/64、固定环境导入和服务参数解析全部通过，固定环境与
  参数解析均为 `cuda_initialized=false`。
- preflight idle/log/exit/static/environment/args SHA256 为
  `331fdd6b…f6b1`/`9d3bfa9a…9d70`/`9a271f2a…86aa`/
  `4c39b130…2fde`/`31b3fecc…93c3`/`54d3dc89…d24f`。容器自动删除，
  `07:59:10Z` 8 卡全空闲。
- 2026-07-31：正式 8-warps 32K/batch1 轮次
  `20260731T0805Z_stage9_candidate_b87a401da_32k_b1_v1` 退出码 0，
  三轮均 3/3 completed、0 failed。`mean` 指标中位数为 TTFT
  `41618.55968243132 ms`、TPOT `201.34670024595937 ms`、请求吞吐
  `0.014884531516601835 req/s`；相对 4-warps 正式结果为
  `-11.718862%/+0.946916%/+7.898018%`，相对 BF16 为
  `+232.203650%/+12.589861%/-47.552743%`。
- 三轮相对极差为 TTFT `0.0636%`、TPOT `0.2490%`、吞吐 `0.1121%`。
  服务 waiting/preemption 均为 0，KV usage 峰值 `5.790244%`；
  三轮/profile 峰值显存分别为每卡 `80679/80691 MiB`。Profiler 8+8+1
  全部校验通过；critical rank 6、kernel total `73788 ms`。stage1 在
  8 个 table 中均为 1,248 次，显示精度下的 8-rank 中位数为
  `29014 ms`，相对 4-warps 的 `34398.099 ms` 下降约 `15.65%`。
- 总/单格 summary SHA256 为 `71417678…130f`/`2340d04d…9f4d`；
  profile result/runner log 为 `acf24a77…a73`/`f93dcfe0…c8e6`；
  外层正式日志/双空闲检查为 `cc2a6208…c12`/`1e5e920f…56b1`。
  实验容器已删除，8 张 GPU 均无 compute process；一个外部下载容器不占
  GPU，未终止。当前结果证明 8-warps 明显改善 TTFT，但 TPOT 轻微回退，
  且 TTFT 仍为 BF16 的约 3.32 倍。
- 2026-07-31：CPU-only 多 chunk 归因
  `20260731T0857Z_8warps_32k_prefill_trace_v1` 一次通过，耗时
  `114.80616178922355 秒`。8 个 rank 均为 144 个 execute context、
  16 个 prefill chunk、32,768 token；prefill wall/kernel 中位数为
  `41516.5703645/40627.54177899999 ms`，generation rank median 再取
  中位为 `269.463987 ms`。
- stage1 为 `29014.13481199999 ms`、1,248 次、`23.248505 ms/次`，
  占 wall `69.8857%`；相对 4-warps 降 `15.6519%`。stage1 减少
  `5383.964455 ms`，解释 wall 总改善的 `99.3756%`；非 stage1 wall
  只变化 `-0.2698%`，generation 只变化 `+0.0620%`。相对 BF16，
  stage1 超额仍解释 prefill wall 差距的 `81.5453%`。
- 分析 summary/log SHA256 为 `a1a8e418…1080`/`86347967…a951`，
  分析器 SHA256 `8b6b2393…30f7`。本阶段没有分配 GPU、启动服务或修改源码；
  下一步仍应只针对 grouped prefill stage1。
- 2026-07-31：只读源码检查确认固定 TP=8 的当前负载每 rank 只有 8 个
  local heads，但 b87 的 `_prefill_head_block_size(8)` 返回 16，两个
  `16×512` FP32 accumulator 各有一半 head 行被 mask。CPU-only SM80
  离线轮次 `20260731T0914Z_prefill_block_shape_offline_v3` 绑定源码
  `b87a401d…`、8 warps/1 stage 和正式 2,048 top-k/stride。
- 当前 `h16/t16`、候选 `h8/t16`、对照 `h16/t32` 的 shared memory 分别为
  `135168/109568/219136 bytes`。候选降低 `25600 bytes`（`18.94%`），
  距苹果800上限余 `57344 bytes`；t32 超上限 `52224 bytes`，已拒绝。
  离线 cubin registers/stack 分别为 `255/0`、`255/24`、`255/840`；
  与 runtime b87 的 `247/0` 不同，因此只把 shared-memory 结果用于筛选。
- v3 summary/run/resource SHA256 为 `15fd8e83…85fd`/
  `7507cdb4…27ea`/`3121f423…ab4f`，两类退出码均为 0；GPU 可见设备为空，
  轮次后 compute process 查询为空。v1 参数计数误断言、v2 metadata
  序列化错误均已 fail-closed 保留，不能算完整结果。
- 2026-07-31：8-head bucket 已由源码
  `a2fe0205577b7f4707e9d31213cb5a80eda1f7d4`、tree
  `b73806b6067b4533e94bf936610a0bf62f1a506d` 落地并推送。diff 只有生产
  helper 2 行和测试期望 1 行：1–8/9–16/>16 heads 分别用 8/16/32；
  kernel 数学、cache、decode 与调度未改。
- TDD 红灯精确为 2 failed/3 passed；有效 CPU-only 轮次
  `20260731T0925Z_headblock_cpu_validation_v1` 为 6/6 passed、10.72 秒，
  `CUDA_VISIBLE_DEVICES` 为空。Ruff/compile/diff 与全部适用提交 hooks
  通过。源码/测试 SHA256 为 `98ad2982…52ac`/`c3aed8b9…03e`，
  test log/exit 为 `2f5c526e…323d`/`9a271f2a…86aa`。仍无 GPU 精度或
  性能结论。
- 2026-07-31：主仓库 `f5d51d0` 发布后，GPU 0 单卡轮次
  `20260731T0931Z_prefill_2k_headblock_v1` 一次通过。双空闲检查间隔
  60 秒，8 卡均为空；外部下载容器不占 GPU。
- 2,048×2,048、warm-up 5/repeat 7 下，split16/grouped split1 CUDA
  中位数为 `195.760/19.077 ms`，相对同轮加速 `10.262×`；相对旧
  8-warps `24.090 ms` 再降 `20.81%`。output/LSE allclose 通过，诊断
  max_abs/max_rel 与旧候选完全一致，峰值增量显存 `224.125 MiB`。
- runtime stage1 为 shared `109568 bytes`、255 registers、0 stack；
  61 个 cold-cache 文件。result/run/idle/resource SHA256 为
  `1861b7cb…06f4`/`97f3b066…e122`/`2a26da12…5b03`/
  `c4b61cce…4000`。容器删除后 8 卡全空闲；尚无完整 CUDA 或 32K 结果。
- 2026-07-31：2.24 由 `b0f8644` 发布后，完整 CUDA 轮次
  `20260731T0936Z_headblock_full_cuda_v1` 前双空闲检查间隔 60 秒；
  固定 GPU 0、只读源码/native、独立空 Triton cache。
- 有效结果为 125 passed、0 skipped/failed、19 warnings、79.04 秒；
  cold cache 380 文件、24,937,748 bytes。pytest/exit/idle SHA256 为
  `a0dc4b04…20d`/`9a271f2a…86aa`/`220768ce…d0b8`。容器删除后 8 卡
  全空闲；候选已通过完整 CUDA，但尚无新 32K 端到端结果。
- 2026-07-31：Phase 6 构建输入已最小切换到 8-head 源码
  `a2fe0205577b7f4707e9d31213cb5a80eda1f7d4` / tree
  `b73806b6067b4533e94bf936610a0bf62f1a506d`，新 tag 为
  `glm52-oscar-a800-phase6-a2fe02055-0275043c`。Dockerfile SHA256 为
  `13c687ed6b394ee095cbbccce38292ab96af6677516acb00f4a2a870984e9809`；
  base manifest、rotation、runtime expectation 与确定性 PAX 逻辑均未改。
- 首次 `uv` 调用没有指定 pytest 临时环境，在 collection 前退出，未形成
  测试结果或 OCI。改用控制镜像 Python 3.12.13、固定
  `pytest==8.4.1` 和清华镜像后，PAX 回归为 1 passed、1 个只读 cache
  warning、0.22 秒；JSON、源码 commit/tree、远端一致性、Dockerfile hash、
  Python compile、旧 Phase 6 身份清零和 diff 门禁均通过。尚未启动 OCI 构建。
- 2026-07-31：输入由 `504c822` 发布后，两个独立 CPU-only 目录
  `20260731T0958Z_candidate_a2fe02055_headblock_v1` /
  `20260731T1000Z_candidate_a2fe02055_headblock_v2_rebuild`
  均完成构建和递归验收，build=`built`、verify=`passed`。
- 两轮共同 image/config、manifest、candidate layer、diff-ID 为
  `51cd8c87…f68e4`/`4a8cec04…6334`/`37d70667…bbd9`/
  `4b51d9dc…6caf8`，layer 为 109,147,568 bytes/5,298 members；
  index SHA256 `bc20fcb6…4872`。两轮 index/config/manifest/layer
  逐字节相同，各验证 4,744 源码、4 份 rotation、7 个基础层原生扩展和
  33 层精确继承，无 native 覆盖/whiteout。全程未分配 GPU，结束后 8 卡空闲。
- 2026-07-31：v1 使用一次性 Ubuntu 22.04 / skopeo 1.4.1 工具容器
  完整执行到 `Storing signatures`。宿主 `tee` 因 root-owned artifact
  目录无写权限令组合 shell 返回 1，但导入前镜像不存在，导入后 daemon
  独立审计状态为 `passed`，因此未重复导入。
- daemon image ID `51cd8c87…f68e4`、33 层、最后 diff-ID
  `4b51d9dc…6caf8`、tag 和 8 项 labels 全部与 v1 匹配。inspect/audit
  SHA256 为 `287a4af2…a05c`/`c9250c6f…026c`；工具容器自动删除。
  全程未分配 GPU，结束后 8 卡空闲。runtime import 尚未执行。
- 2026-07-31：runtime import 前 `10:02:41Z/10:03:42Z` 两次 8/8
  空闲检查间隔 61 秒；有效探针固定 GPU 0 只注入驱动，一次通过、退出码 0。
  Python/PyTorch/Triton 为 `3.12.13/2.11.0+cu129/3.6.0`，
  Transformers/Tokenizers 与 FlashInfer 包版本、候选 vLLM Python/`_C`、
  78 层 rotation、三项 artifact hash、`reasoning_effort=max` 均匹配，
  `cuda_initialized=false`。
- idle/JSON/log SHA256 为 `82cfb7f0…0234`/`0910b598…7b7a`/
  `f2e60043…189a`；JSON/log 与历史同协议证据逐字节一致。容器删除后
  8 卡全空闲。证据目录已按实际首检时间修正为
  `20260731T1002Z_headblock_runtime_import_v1`。
- 2026-07-31：Stage 9 控制 Dockerfile 只把默认 base 从 b87 候选切换到
  `glm52-oscar-a800-phase6-a2fe02055-0275043c:latest`，新文件 SHA256 为
  `a9b9e22b…e97c`；其余构建逻辑未改。daemon base 的 image ID、33 层、
  source commit/tree、candidate layer 与最后 diff-ID 已重新核对一致。
  本阶段没有构建镜像、注入 driver 或分配 GPU。
- 复现入口由 `7126356` 发布后，CPU-only 控制镜像构建与审计一次通过。
  新 tag/image ID 为 `oscar-glm-stage9-runtime:a2fe02055` /
  `0e13b724…d50d5`；34 层中的前 33 层与候选逐层一致，继承 labels 和
  entrypoint 匹配。固定 CPU runtime 版本/包清单一致，
  `cuda_initialized=false`。五份主要证据 SHA256 为
  `7779e250…cb2b`/`9a271f2a…86aa`/`36d9ea8e…be04`/
  `723515b7…13f3`/`5ac65b5d…1f20`；前后 8 卡空闲。
- a2fe v1 的 extracted layer 已机械派生正式 overlay：两边均为 4,749
  个普通文件，递归内容清单 SHA256 同为 `5cf59c75…f51a`；overlay 另有
  6 个指向冻结 phase0 rootfs 的 native symlink，目标和六项原生扩展哈希
  与上一正式链路完全一致。runtime import JSON 已以只读证据复制到 v1
  artifact，SHA256 保持 `0910b598…7b7a`。
- Phase 1/5/7/9 配置已按依赖顺序迁移，SHA256 为
  `ff6c853f…0ee6`/`de58d99a…ba0c`/`4d66f3c6…d006`/
  `f9d93958…b520`。4 JSON、9 个变更 shell、Phase 9 Python compile、
  旧 b87 身份清零和 diff 门禁通过。
- 新 CPU-only 控制镜像中的 Phase 7/9 工具测试分别为 20/20、21/21
  passed；日志 SHA256 为 `4f9d33d0…cc69`/`f28fce5f…de9e`。以正式
  phase0 source volume 覆盖 NFS mode 映射后，递归 verifier 一次得到
  64/64 passed；JSON SHA256 `99d90ff6…390f`。三项退出码均为 0，
  全程未注入 NVIDIA runtime，8 卡空闲。
- 静态链路由 `ea88b55` 发布后，正式 preflight 前
  `10:25:11Z/10:26:12Z` 两次 8/8 空闲检查间隔 61 秒；两次均 0 MiB、
  0% 且无 compute process。轮次
  `20260731T1026Z_stage9_candidate_a2fe02055_preflight_v1` 退出码 0，
  静态 64/64、固定环境导入和服务参数解析均通过，后两者都记录
  `cuda_initialized=false`。
- preflight 的 idle/log/exit/static/fixed-env/args 六份 SHA256 为
  `cd1df9ca…7ee2`/`7406ae3b…471c`/`9a271f2a…86aa`/
  `18870284…fb78`/`28a39415…6c3e`/`6395c4cb…f19c`。容器自动删除，
  8 卡全空闲；仅剩不占 GPU 的外部下载容器。
- 正式 32K/b1 轮次
  `20260731T1032Z_stage9_candidate_a2fe02055_32k_b1_v1` 完整退出码 0。
  三轮均为 3/3、0 failed；`mean` 中位数为 TTFT `36245.415 ms`、
  TPOT `199.205 ms`、吞吐 `0.016248 req/s`。相对 b87 正式结果为
  `-12.91%/-1.06%/+9.15%`，相对 BF16 为
  `+189.31%/+11.39%/-42.75%`。
- 单格/总 summary 均 passed；profiler 769.622 秒，8+8+1 证据全部通过。
  stage1 表格中位数约 `23688.5 ms`、1248 次，相对 b87 trace 的
  `29014.135 ms` 下降约 `18.36%`。容器删除后 8 卡全空闲，关键小型证据
  已复制到 NFS 持久目录且哈希一致。
- a2fe 多 chunk 归因的首次输出目录误用历史 root-owned analysis 父目录，
  在 `mkdir` 阶段被拒绝；容器/分析器均未启动，trace 未读取，GPU 未分配。
  后续输出改到当前用户独立的 `/dev/shm/oscar-glm-stage9-opt/analysis`。
- 有效 CPU-only 多 chunk 归因
  `20260731T1119Z_headblock_32k_prefill_trace_v2` 为 passed：8 rank 均含
  144 execute context、16 chunk、32,768 token；prefill wall/kernel/
  generation 中位数为 `36257.407/35316.438/269.448 ms`。
- stage1 为 `23688.690 ms`、1248 次，占 wall `65.33%`。相对 b87，
  stage1 减少 `5325.445 ms`，解释 wall 改善的 `101.26%`；去掉 stage1
  后的 wall 反而增 `0.53%`，generation 基本不变。stage1 相对 BF16
  原生 attention 的超额仍解释当前 prefill wall 差距 `77.58%`，因此下一
  候选仍必须只针对 grouped prefill stage1。
- 下一项候选的只读源码定位已开始。a2fe 提交统计确认 8-head 改动实际位于
  `vllm/v1/attention/ops/triton_oscar_mla_decode.py`，并非通用
  `triton_sparse_mla_kernel.py`；后续 stage1 资源和控制流分析以该文件为准。
- 当前 grouped prefill stage1 的冻结几何为 `block_h=8`、`block_t=16`、
  `block_d=512`、8 warps、1 stage；grid 为 2,048 queries × 1 head group，
  num_splits=1。每个 token tile 只加载一次 BF16/history/rope values，随后跨
  8 heads 复用，但为两种 value basis 分别维持 `8×512` FP32 accumulator。
- 内层每个 16-token tile 包含 3 个 score dot 和 2 个 value dot；BF16 score
  使用原生 BF16 tensor core，history score/value 与 BF16 value probability
  使用 TF32，保持已验收精度。现有 launch 参数没有 runtime 开关可单独 sweep
  `block_h` 或 `block_t`，因此下一候选必须先用 CPU-only 离线编译筛选资源，
  再决定是否值得最小源码改动和单卡实测。
- 上一轮 b87 CPU-only AST 产物仍完整保留，确认 h8/t16 的生成 metadata 为
  8 warps、1 stage、SM80、shared `109568 B`；TTIR/LLIR/PTX/cubin 也均在。
  a2fe 无 driver 控制容器的普通 vLLM import 会主动禁用 Triton并把 kernel
  暴露成普通函数，因此不能直接读取 JIT `arg_names`；需要复用上一轮的特殊
  离线导入方式，不能把该 import 失败误判为编译不可行。
- 已确认特殊条件不是注入 driver，而是显式设置空 `CUDA_VISIBLE_DEVICES`；
  vLLM 会把它识别为分布式初始化窗口并保留真实 Triton JIT。a2fe 控制镜像内
  由此得到真实 `JITFunction`，78 个参数中前 19 个为指针、其余均为
  constexpr；Triton 3.6.0 的 `ASTSource(fn, signature, constexprs)` 和
  `GPUTarget("cuda", 80, 32)` 足以复现纯 CPU SM80 编译。
- 有效离线矩阵 `20260731T1138Z_prefill_tile_offline_v2` 绑定 a2fe 控制镜像、
  Python 3.12.13 / Torch 2.11.0+cu129 / Triton 3.6.0、SM80，17.988 秒通过。
  h8/t16/w8 精确复现 shared `109568 B`；4 warps 不改变 shared 且产生
  1,240-byte stack，而 8 warps 只有 24-byte stack。
- h4/h2/h1、t16、8 warps 的 shared 分别为 `96768/90368/87168 B`，全部
  高于双 block 驻留阈值 `83456 B`；同时每 query 的 program 数相对 h8
  分别变为 2/4/8 倍，会重复 value/rope 载入，离线证据不支持进入 GPU。
- t8 在两个 head 几何均因 Triton dot 的 K 维必须至少 16 而编译拒绝；
  h8/h4 的 t32 分别为 `193536/180736 B`，均超过苹果800 `166912 B`
  单 block 上限。当前简单 tile/warps 搜索空间已经穷尽并全部拒绝。
- 2.32 已由主仓库 `68a5127baf5ea228c8f45c6bc02d8308624dd9f3`
  发布且工作树干净；恢复脚本中的未同步上下文仅包含发布动作和下一候选说明，
  没有遗漏的源码或实验结果。算法级下一候选的安全前提是正式 prefill 输出的
  selected-token 行确实保持有效 causal token 前置、无效项或 `-1` 尾部填充；
  在源码/测试证据证明该不变量前，不实现动态循环上界。
- 正式 CUDA prefill indexer 在
  `model_executor/layers/sparse_attn_indexer.py` 调用原生
  `_C.top_k_per_row_prefill(logits, cu_seqlen_ks, cu_seqlen_ke, ...)`；OSCAR
  backend 随后把 buffer 的前 `min(topk_tokens, max_seq_len)` 列直接交给
  `oscar_mla_sparse_prefill`，没有在 Python 侧重排或压缩。因而排列保证必须
  来自 `csrc/sampler.cu` 的原生 top-k 实现，当前 Python 调用链本身不能证明
  “有效项前置、无效项尾部”。
- `csrc/sampler.cu` 的 prefill wrapper 默认读取
  `VLLM_TOPK_PREFILL_SORT_INDICES`（默认未开启），但无论是否 sort-indices，
  都调用同一 `topKPerRowJob` 并最终把完整 `topK` 个 shared-memory 槽写回。
  现有 `tests/kernels/test_top_k_per_row.py` 只为每行前
  `k_i=min(top_k,row_end)` 个 reference 槽赋值，比较函数需进一步核对；测试
  片段本身没有验证 `row_end < top_k` 时尾部为 `-1`，所以仍不能认定尾部
  padding 或排列不变量成立。
- 继续读取 `topKPerRowJob` 后，关键不变量已由生产源码直接证明：当
  `rowLen=rowEnd-rowStart <= topK` 时，kernel 明确把前 `rowLen` 槽写为
  连续有效索引，并把 `[rowLen, topK)` 全写为 `-1`；当 `rowLen > topK`
  时则输出恰好 topK 个有效 top-k 索引。默认是否对索引排序不影响“有效前缀、
  `-1` 尾部”这一性质。现有测试的 `compare_top_k_results` 只比较有效前缀、
  未测试尾部，属于测试覆盖缺口，但不推翻生产源码中的显式写入。
- 动态 stage1 上界因此在排列语义上可行，不过还需把 `rowLen` 与 OSCAR
  `query_positions`/causal length 的正式 metadata 对齐，并确认该上界减少的
  是 Triton 实际运行循环而非只减少 mask；完成这些源码证明前不实施。
- Indexer 的每个 prefill chunk 把同一 `cu_seqlen_ks/cu_seqlen_ke` 同时用于
  logits 的 causal 范围和 `_C.top_k_per_row_prefill`，随后 backend 为 OSCAR
  单独构造 `query_positions`。chunk metadata 的定义与构造位于
  `v1/attention/backends/mla/indexer.py`；需要以该文件为最终语义证据，不能
  仅凭 `actual_active_seq_lens - q_m + 1` 的单请求优化分支推断所有情况。
- 正式 metadata 构造中，`build_one_prefill_chunk` 直接取
  `kv_spans_from_batches_cpu(...)` 的逐 query `cu_seqlen_ks/ke`；OSCAR
  `_oscar_query_positions` 对 prefill token 计算
  `seq_lens[request] - (query_end - token_row)`。stage1 目前又计算
  `causal_seq_len=min(seq_len, query_position+1)`，但仍用 Python 静态
  `range(0, topk, block_t)` 执行所有 tile，只靠 mask 忽略无效尾部。
- 如果 `kv_spans_from_batches_cpu` 的 row length 等于上述
  `query_position+1`（或 packed request-local 等价值），则每 query 的有效
  selected-token 数可由现有 `causal_seq_len` 精确给出，无需扫描 `-1`；下一步
  读取该函数公式并验证多请求偏移。
- 公式已经逐项对齐：对请求长度 `L`、本批 query 数 `m`、该请求内第 `i`
  个 query（从 0 起），indexer top-k 的有效行长为
  `L-m+i+1`；OSCAR `query_position+1` 也正是 `L-m+i+1`。多请求时
  `rowStart` 只是在拼接 logits 中加 KV 基址，原生 top-k 返回 request-local
  索引，因此 stage1 的 `causal_seq_len` 就是有效 selected-token 前缀长度，
  可安全取 `effective_topk=min(topk, causal_seq_len)`。
- 仓库已有 Triton 生产 kernel 使用运行时
  `tl.range(split_kv_start, split_kv_end, BLOCK_N)`，证明当前 Triton 路径支持
  runtime loop bound。对 stage1 的最小候选是把静态
  `range(0, topk, block_t)` 改为
  `tl.range(0, effective_topk, block_t)`，保留现有 selected/causal masks；
  这会让 2,048-query chunk 的早期 query 少执行尾部 tile，而不改变选择集合。
- 定向测试入口确定为现有 CPU-only
  `tests/oscar_mla/test_triton_decode.py`：先对底层 JIT function 的 Python 源码
  加回归断言，要求显式计算 `effective_topk=min(topk, causal_seq_len)` 并用
  `tl.range` 作为 runtime bound。该测试不声称性能，只锁定本轮唯一预期改动；
  现有 interpreter smoke 随后负责验证实际多 query/multi-request 数值语义。
- TDD 已闭环：source-invariant 红灯为 1 failed；修正候选后定向绿灯为
  1 passed。完整 `test_triton_decode.py` 的 CPU/interpreter 适用范围为
  `7 passed, 19 skipped`，3 条 warning 均为固定镜像的既有 import warning；
  interpreter smoke 覆盖单请求多 query 与 multi-request 数值路径，说明
  runtime bound 没有改变现有 oracle 结果。尚无 GPU 编译或性能结论。
- 固定控制容器内 Ruff 0.14.0 check、format check 和正式 Python 3.12.13
  `py_compile` 已通过：2 个触及文件 lint 全绿、均已格式化、语法编译退出码 0。
  下一门禁是 CPU-only SM80 离线编译，必须证明运行时 `tl.range` 可生成 cubin
  并记录 shared/register/stack；通过前仍不进入 GPU。
- CPU-only SM80 AST 编译轮次
  `20260731T1208Z_causal_loop_offline_v1` 已通过：当前 h8/t16/w8 runtime-loop
  候选生成 201,648-byte cubin，shared 仍为 `109568 B`，与 a2fe 静态循环
  相同；矩阵其余 tile 的 shared/拒绝模式也与 2.32 一致。summary 中
  `source_commit=a2fe...` 是复用控制脚本的 base 标签，真实工作树输入由
  `input.diff` 和两份 source SHA256 独立冻结，不能把 summary 字段误写成候选
  commit。编译前后均未注入 GPU，结束后 8 卡 0 MiB/0%、无 compute process。
- 宿主 CUDA 12.9 `cuobjdump` 对正式 h8/t16/w8 cubin 的资源为
  `REG=255, STACK=32 B`；相对 a2fe 静态循环的 `255/24 B` 仅多 8-byte
  stack，shared 保持 `109568 B`。小型证据已落到正式 NFS 目录
  `.../causal_loop_offline_v1`；input diff/source、summary、run、exit、resource
  的 SHA256 分别为 `fbd4d53c…512b`、`5f8114e0…c4b3`、
  `bfad847f…5e3`、`2dad6fe8…f59`、`9a271f2a…86aa`、
  `9133653c…b40`。该资源差异足够小，不构成 CPU-only 淘汰理由。
- runtime causal-loop 候选已正式发布为 source commit
  `fd281f5f974207998a95666d4015c441c5db49ab`、tree
  `86185b214eb3d6f25108076a0a2c2c8dabb3d122`；远端跟踪分支一致且干净。
  这只是 CPU/离线门禁通过，不等于苹果800精度或性能通过，报告必须明确边界。
- 单卡冻结入口仍是 `scripts/phase9/benchmark_oscar_prefill.py`：2,048 sequence
  时只比较 full top-k split16 与 grouped split1，seed 42，5 warm-up、7
  repeats、1 iteration；selected rows 本身按 query causal 长度生成有效前缀和
  `-1` 尾部，正好直接测量本轮 runtime bound。当前 source 的 4 个 native
  symlink 都指向 phase0 rootfs 的绝对路径，GPU 容器必须同时只读挂载该路径。
- GPU 0 单卡轮次 `20260731T1223Z_causal_loop_2k_gpu_v1` 已通过：同轮
  split16 为 `195.772415 ms`，runtime causal-loop grouped split1 为
  `12.858368 ms`，相对同轮加速 `15.2253×`。冻结 allclose 状态保持 passed，
  output/LSE max_abs 仍为 `0.0049126148/0.0020360947`，与 a2fe 完全一致；
  不能把 allclose 通过误写成 max_abs 小于 0.002。
- 实际 runtime cubin 为 247 registers、0-byte stack，峰值增量显存仍为
  224.125 MiB；相对离线 255/32 更好。实验后 8 卡全部 0 MiB/0%、无 compute
  process。下一步需结构化复算相对 a2fe 的改善百分比并持久化小型证据。
- 相对 2.24 a2fe 的精确 `19.077119827271 ms`，当前
  `12.858367919922 ms` 降低 `32.597960%`，即加速 `1.483635×`；61 个
  cold Triton cache 文件。result/run/idle/resource/两项 exit SHA256 为
  `4a8a7b6c…d52`/`bcdc0d7d…5f0`/`60fe976b…9d0`/
  `2597b01f…04a`/`9a271f2a…86aa`，小型证据已逐字节复制到正式 NFS
  `causal_loop_2k_gpu_v1` 目录。
- 完整 cold-cache CUDA 轮次 `20260731T1223Z_causal_loop_full_cuda_v1`
  固定 GPU 0、当前已发布源码、只读 native rootfs 与独立空 cache，结果为
  126 passed、0 failed、19 warnings、86.77 秒。相对 a2fe 的 125 项多出的
  1 项是本候选新增的 source-invariant 回归，不是旧测试缺失或跳过。
- cold cache 为 380 个文件、24,964,627 bytes；pytest/exit/idle/post/cache
  SHA256 为 `0f85ebb7…83e4`/`9a271f2a…86aa`/`8e124fa9…d33d`/
  `8e808aba…9683`/`a2745e0e…0ace`。容器删除后 8 卡全为 0 MiB/0%、无
  compute process；5 份小型证据已逐字节复制到正式 NFS
  `causal_loop_full_cuda_v1` 目录。
- Phase 6 构建输入已最小切换到 source commit/tree
  `fd281f5f974207998a95666d4015c441c5db49ab` /
  `86185b214eb3d6f25108076a0a2c2c8dabb3d122`，候选 tag 为
  `glm52-oscar-a800-phase6-fd281f5f9-0275043c`。Dockerfile 只改两项
  ARG，实算 SHA256 为 `2c97b4ef6397850b2ac095f8120d3e4f08206e829cdb8adcfc71281d85894b87`；
  config SHA256 为 `9618cd4fc0fe53a0624e2bc7cab79b2be5d3f34cc05a9eaf4b9c50134e6a62d4`。
- 固定 Python 3.12.13/pytest 8.4.1 的 PAX 确定性回归为 1 passed、
  0 failed，build/verify 两个脚本 compile 通过；JSON 5 项、源码身份、旧
  a2fe Phase 6 身份清零和 diff 门禁均通过。该阶段 CPU-only，8 卡始终空闲。
- Phase 6 v1/v2 独立目录的 build=`built`、verification=`passed`，共同得到
  image/config `2369d967…d692`、manifest `e18b2252…63ee`、candidate layer
  `9b6a02c4…d02e`、diff-ID `f1b88c82…335a`。candidate layer 为
  109,147,808 bytes、5,298 members，无原生扩展或 whiteout；33 层中前
  32 层与 base 精确匹配。
- 两份 `index.json`、config blob、manifest blob、candidate layer blob
  逐字节相同，index SHA256 为 `20d0e846…181d`。每轮重新核验 4,744 个
  Git 文件、4 个 rotation 文件、7 个 base native extension；两项退出码
  均为 0。并行 CPU-only 构建约 6 分 35 秒，结束后 8 卡仍 0 MiB/0%。
- 修改报告前已在固定控制镜像的 Python 3.12.13 中重新只读解析两轮
  `build_report.json`/`verification_report.json`：两轮状态分别为
  `built`/`passed`，source commit/tree、Dockerfile/config hash、33/32 层、
  4,744/4/7 项递归计数和候选层无 native/whiteout 均与 planning 记录一致。
  `cmp` 也再次确认 index、config blob、manifest blob、candidate layer blob
  逐字节一致；本阶段证据只覆盖 CPU-only OCI 构建与递归验收，不覆盖 daemon
  导入、driver-injected runtime import 或新的 32K/batch1 性能。
- 当前 v1 父目录由项目用户所有且 mode 775，只有其 `oci-layout` 子目录为
  root:root 755；因此 daemon import 日志可以直接写在 v1 父目录，不应写入
  root-owned 子目录。既有成功协议是在一次性 Ubuntu 22.04 容器内安装
  `skopeo 1.4.1`，从只读 OCI layout 复制到 `docker-daemon:<tag>:latest`，
  随后用 daemon inspect 核对 image ID、33 层、最后 diff-ID、tag 和 8 项
  labels；本轮按同一协议执行，且不传 `--gpus`。
- 有效 daemon import 已使用阿里 HTTP Ubuntu 镜像和 skopeo 1.4.1 完成，
  日志到达 `Storing signatures`、退出码 0。daemon image ID 为
  `sha256:2369d967…d692`，33 层、最后 diff-ID `sha256:f1b88c82…335a`、
  tag 与 source/tree、candidate layer、Dockerfile、rotation manifest、
  rotations、runtime expectation、base manifest 共 8 项 labels 全部匹配，
  audit 状态 `passed`。import/exit/inspect/audit SHA256 为
  `777795b9…d29f`/`9a271f2a…86aa`/`72278572…304a`/`c411dd4a…5173`；
  导入后 8 卡 0 MiB/0%、无 compute process。
- runtime import 的冻结输出 schema 已从上一有效 a2fe JSON 复核；本候选镜像
  应复现同一 JSON。`reasoning_effort_max` 可从
  `ChatCompletionRequest.reasoning_effort` 的 Literal 注解中确认 `max`，无需
  构造请求或初始化 CUDA；FlashInfer 仍只允许用 `importlib.metadata.version`
  读取包版本，禁止额外导入 `flashinfer`/`flashinfer.jit`。
- 当前源码的该注解已直接核对为
  `Literal["none", "low", "medium", "high", "max"] | None`；上一有效 a2fe
  runtime JSON 为 892 bytes、SHA256 `0910b598…b7a`。有效 causal-loop 轮次
  必须输出非空 JSON，并逐字段等于这一冻结 schema 后才可接受。
- causal-loop 有效 runtime import 固定 GPU 0，重做空闲检查为
  `13:06:39Z/13:07:48Z`、间隔 69 秒。有效 JSON 状态 `passed`，与 a2fe
  冻结 JSON 逐字节一致；JSON/log/exit/post SHA256 为
  `0910b598…b7a`/`f2e60043…189a`/`9a271f2a…86aa`/`013b857a…e524`，
  idle log 为 `6c0f255c…b99d`。容器删除后 8 卡均为 0 MiB/0%、无 compute
  process，`cuda_initialized=false`。
- Stage 9 控制 Dockerfile 迁移只需把默认 base 从 a2fe tag 改为已导入并
  验收的 `glm52-oscar-a800-phase6-fd281f5f9-0275043c:latest`；其余内容无需
  改动。新 Dockerfile SHA256 为 `93111035…8bbb`，daemon base 为
  `2369d967…d692`、33 层，source/tree/candidate layer 与 Phase 6 验收一致。
- 新控制镜像 `oscar-glm-stage9-runtime:fd281f5f9` 已 CPU-only 构建为
  `sha256:9be0cbb72088f9fe4b48680814254be9306e0cd9b1a13c4fb49fa95911db321b`；
  34 层的前 33 层与 base 精确一致，labels/entrypoint 继承通过。CPU runtime
  JSON 与历史候选逐字节一致（SHA256 `5ac65b5d…1f20`），证明固定包和
  `cuda_initialized=false` 均未漂移。
- causal-loop overlay 已从 v1 `extracted-layer` 机械派生；4,749 个普通文件
  与 candidate layer 的递归 path+SHA256 清单逐字节一致，另建的 6 个 native
  symlink 与 a2fe 正式 overlay 的相对路径和绝对目标逐字节一致。无需复制或
  重建任何原生扩展。
- 实测冻结 evaluator launcher 指向
  `/dev/shm/oscar-glm-recovery-tools/python/cpython-3.12.3-linux-x86_64-gnu/bin/python3.12`，
  路径存在。新控制镜像内 Phase 7/9 工具测试分别为 20/20 和
  21/21 passed，0 failed；只有只读项目无法写 pytest cache 的 warning。
  有效日志 SHA256 为 `b244eec0…d16`/`b308eec8…b0d`，退出码均为 0。
- 正式 phase0 source Docker volume 覆盖 NFS mode 映射后，CPU-only 递归
  verifier 一次通过 64/64 checks，0 failed；JSON 与 stdout 日志逐字节
  一致，SHA256 均为 `604f1fde…cc7`，退出码为 0。该阶段未注入
  NVIDIA runtime，只关闭静态身份门禁，不能代替 driver-injected preflight
  或新的 32K/batch1 端到端结果。
- 正式 driver-injected preflight
  `20260731T1354Z_stage9_candidate_fd281f5f9_preflight_v1` 退出码 0。
  外层双空闲检查为 `13:51:51Z/13:53:00Z`、间隔 69 秒，启动前
  `13:53:58Z` 第三次复查仍为 8/8 张卡全空闲。静态 64/64、固定
  环境导入和服务参数解析全部通过，后两者都记录
  `cuda_initialized=false`。
- preflight idle/log/exit/static/fixed-env/args/post 的 SHA256 分别为
  `c1686126…c9`/`2ad73636…89a`/`9a271f2a…aa`/`8ddf37fd…81`/
  `85994ced…ca`/`ba4b022e…c61`/`c7d9e9be…ef2`。容器自动删除，
  `13:56:42Z` 8 张卡全空闲。本结果只关闭正式运行前门禁，新的
  32K/batch1 TTFT/TPOT 尚未开始。
- causal-loop 正式 32K/batch1 轮次
  `20260731T1403Z_stage9_candidate_fd281f5f9_32k_b1_v1` 已退出 0，
  总/单格/profile 均为 `passed`。三轮 `mean` 中位数为 TTFT
  `35683.892961 ms`、TPOT `197.826235 ms`、吞吐
  `0.0164451783 req/s`；相对 a2fe 为
  `-1.5492%/-0.6924%/+1.2164%`，相对 BF16 为
  `+184.8325%/+10.6214%/-42.0473%`。
- 三轮峰值显存为 80,679 MiB/卡，profile 为 80,691 MiB/卡；
  max running/waiting/preemption 为 `1/0/0`，KV usage 峰值 `5.7902%`。
  Profiler 耗时 750.038 秒，8 份 trace、8 份 table 和 frontend trace
  全部通过哈希门禁；critical rank 6 kernel total 为 65,795 ms。
  stage1 table 的 8-rank 中位数为 23,134.5 ms/1,248 次，相对
  a2fe table 的 23,688.5 ms 约降 2.34%。
- 当前/BF16/a2fe 总 summary SHA256 分别为
  `d2bb22c7…3c5b`/`c0e31229…e2f5`/`3ba95901…0cb9`。小型证据
  40 份、1,114,300 bytes 已复制到 control artifact，原件/副本逐件
  SHA256 一致；manifest SHA256 为 `aefe596a…b96d`。大 trace 未
  重复复制，但 validation/summary 已冻结每个 trace 的 bytes/SHA256。
- 报告 2.38 已通过结构、交叉引用、术语、三轮指标、对比公式、
  profiler 数量/身份、证据数量/哈希和 diff 门禁；修改后文件为
  2,649 行，SHA256 `a9e78fe6…43c7c`。
- causal-loop 冻结 trace 的 CPU-only 有效分析 ID 为
  `20260731T1457Z_causal_loop_32k_prefill_trace_v1`，状态 `passed`、
  耗时 110.784 秒。8/8 ranks 均为 144 个 execute context、16 个
  prefill chunk 和 32,768 token；prefill wall/kernel/generation 中位数为
  `35731.482/34779.955/266.230 ms`。
- 当前 stage1 为 `23134.871 ms`、1,248 次、占 wall `64.75%`；相对
  a2fe 的 `23688.690 ms` 降低 `2.34%`。stage1 减少 553.819 ms，而
  wall 减少 525.925 ms，解释比例为 `105.30%`；去掉 stage1 后的 wall
  反而约增加 `0.22%`，其他主要 kernel 基本不变。
- 32K 被切成 16 个 2,048-token chunk。当前
  `effective_topk=min(2048, causal_seq_len)` 只会裁剪首 chunk 的无效
  causal 尾部；第 2–16 个 chunk 的最小 causal 长度已经大于 top-k。
  因而只有 78/1,248 次 stage1 调用受益（`6.25%`）。2K 单层
  `-32.597960%` 粗略除以 16 得 `-2.04%`，与 trace 的 stage1
  `-2.34%` 同量级，解释端到端 TTFT 仅改善 `1.55%`。
- 当前 stage1 相对 BF16 原生 attention 的超额仍解释 prefill wall 差距的
  `77.01%`。下一候选必须减少全部 16 个 chunk 都会执行的有效 top-k
  计算或访存；继续裁剪首 chunk 尾部、转向 generation 或其他 kernel 均不受
  现有证据支持。
- 有效 analysis summary/log/input manifest SHA256 为
  `16a97002…8e1`/`fd333686…ab1c`/`57a3860d…f53`。5 份小型证据、
  343,360 bytes 已复制并逐件验哈希，manifest SHA256 为
  `32675481…99b`；全程未注入 NVIDIA runtime 或分配 GPU。
- 报告 2.39 已通过发布前门禁：1.1–1.5、2.1–2.39 连续，实际章节
  引用有效，`三池=0`，大写 `A800` 仅在第 5 行允许链接；所有表格数值、
  stage1/wall 解释公式、5 份证据/343,360 bytes 与主要 SHA256 均从冻结
  文件复算一致。报告为 2,716 行，SHA256 `ba82eff0…cc6c`。
- 当前 grouped stage1 在每个 tile 中先用 mask 把非 prefix/recent 的
  `bf16_values` 归零，但随后仍无条件执行 `bf16_scores` 与 `bf16_acc` 两个
  dot。对于全 history tile，这两个 dot 的数学贡献精确为零；在线 softmax
  只要求相应 accumulator 继续乘 `previous_scale`，新增值项可以安全跳过。
- 后 15 个 32K chunk 的 top-k 固定为 2,048，而 prefix+recent 整个请求最多
  只有 320 token；因此 tile 级 `has_bf16` gate 能作用于全部 chunk，不依赖
  causal 尾部。最小候选不需要打开既有 index sort 环境变量，也不修改 C++
  top-k 输出顺序，避免把 indexer 成本和 stage1 kernel 改动混为一个候选。
- tile gate 的 CPU/interpreter 数值回归为 8 passed、19 CUDA 显式 skipped、
  0 failed；离线 SM80 编译也通过，shared `109568 B` 与 causal-loop 候选
  相同，cubin 206,640 bytes，离线资源为 255 registers/0-byte stack。
  这只关闭 CPU 语义和编译资源门禁，尚无苹果800 CUDA 正确性或性能结论。
- 该最小候选已由源码提交
  `ca4a404e913ce55237ca60383cc86e221fbfea26` 发布；生产源码/定向测试
  SHA256 分别为 `23b08ffae200cfefa3e7a2190c436c0f517bfc509e8479bb22245230b10bc70c`/
  `92a8340ed5a528232a1685a4deb09288efc86792e0002d6289b00877e8d2da0b`。
  CPU pytest、离线 summary、离线 run log、资源日志 SHA256 分别为
  `e55abd1e6654dfaed142c3397d88935e98b38fe0df87d6b1cf207f02b5cbfcca`/
  `a092f126add6fd638b343ecc0d0dbcd0acbb987fedf1e77d881094459b8a8ee2`/
  `54712c60e2d30be8985746bb13008dfef48cdf6de0186a090046600cd0732e8d`/
  `c4b61cceb4235a6d12e2428a324e0196811b9aea6b96f30f23b94fd318674000`；
  control artifact manifest 为
  `50535adca593513a0eb226614ce5413fdaa081645f48f4b11329a2e9dd361e95`。
- 2.40 的最终只读复核确认目录内为 10 份证据文件加 1 份清单、合计
  14,353 bytes；全部 manifest 条目逐文件 SHA256 一致。报告修改后为
  2,777 行，SHA256
  `97290f8122d153397e6ff9202c6059c5f419a89acfd6f0d871776a2306a41be7`，
  章节、交叉引用、术语和 diff 门禁通过。
- later-chunk 基准把 query chunk 长度与最终序列长度解耦后，保持原 2K 默认
  行为不变；`--seq-len 2048 --final-seq-len 32768` 精确生成末段
  `[30720,32768)` query positions、32K cache/sequence metadata 和固定
  2,048 top-k width。脚本/测试 SHA256 为
  `27f9d5e7a084c8b23c4bb80aead140b58790531ff67fa51d99aadaa7c2d4ca90`/
  `857c17410b0d5965b8324ab40386191acbb04d744f96967f5395b03ba6110d47`。
- seed 42 的 CPU-only 32K coverage 中，4,194,304 个 selected index 全部
  有效、逐行唯一且 causal；262,144 个 tile 中 257,626 个为全 history，
  比例 `98.2765%`。这验证新几何能放大 tile gate 的目标路径，但随机 selected
  不等于正式 DSA top-k，不能由此预测端到端收益。
- 2.41 最终门禁确认报告 2,848 行、SHA256
  `426921c3f8d828b2ea23e9c517cbc78b28e01f717f90922e43ef72dcc7796579`；
  章节、交叉引用、术语、coverage 公式、代码/证据哈希和 diff 均通过。
- 32K 末段合成负载的同容器源码对照显示，tile gate 把 grouped split1
  CUDA 中位数从 `22.618113 ms` 降至 `20.226048 ms`（`-10.575883%`，
  `1.118267×`），wall 中位数从 `22.670865 ms` 降至 `20.256273 ms`
  （`-10.650641%`）。split16 仅变化 `-0.018464%`，说明对照环境稳定。
- 两边相对各自 split16 的 output/LSE 诊断逐字段相同并通过冻结 allclose；
  runtime shared/stack 不变为 `109568/0 B`，registers 从 247 降至 242。
  该结果证明合成末段单层路径有收益，但 selected index 不是正式 DSA 抓取，
  也尚未通过完整 CUDA 回归或 32K/batch1 端到端验证。
- 2.42 编辑前已完整重读当前 2,848 行报告，读取前后及编辑前
  SHA256 均为
  `426921c3f8d828b2ea23e9c517cbc78b28e01f717f90922e43ef72dcc7796579`；
  可在不覆盖并发手工改动的前提下追加新章节。
- 2.42 修改后为 2,923 行，SHA256
  `b0ca771b71048aaeab28b34da6497a7324b4388766c3e8c0cf5c3b93d3abb4f2`。
  章节、交叉引用、术语、落地 JSON/源码/cubin 哈希、性能公式、
  17 份证据加清单与 306,881-byte 总量全部复算通过。
- ca4a404e9 完整 `tests/oscar_mla` 在 GPU 0、全新 Triton cache 上为
  127 passed、0 failed、19 warnings、87.91 秒，退出码 0。相对 fd281f5f9
  历史轮次增加的 1 项是本候选新增的 source-invariant 测试；当前结果不含
  skip 或 fail。
- cold cache 实测为 380 文件、25,038,227 bytes；运行前双检、启动前检查和
  退出后检查均无 compute process。8 份小型证据与 380 行 cache 哈希已固化，
  证据目录总计 94,565 bytes。
- 2.43 最终为 2,997 行，SHA256
  `430479a60c058c188044586a2be59a5b645492b42e34b60400b83efd6d7e47cf`。
  章节/引用、术语、落地 pytest/cache/证据哈希、源码 Git tree、镜像 ID 和
  diff 门禁全部通过。
- ca4a404e9 Phase 6 输入迁移仅更新 source commit/tree、output tag 与
  Dockerfile 实算哈希；base/rotation/runtime expectation/native 合约不变。
  新 Dockerfile/config SHA256 为 `51ed571f…e6f4`/`5f3fb384…8b802`。
- 有效 CPU-only 门禁为 PAX 1 passed/1 个只读 cache warning/0.12 秒，
  3 份 Phase 6 Python compile 通过，旧 fd281f5f9 commit/tree/tag 计数为 0，
  source 本地/远端一致。首轮 PAX 已通过，但 compile 因只读 `__pycache__`
  写入退出，只有显式重定向到 `/tmp` 的 v2 计为整体通过。
- 2.44 修改前已补齐报告末段的分片重读；完整重读后的报告仍为 2,997 行、
  SHA256
  `430479a60c058c188044586a2be59a5b645492b42e34b60400b83efd6d7e47cf`，
  可在不覆盖并发手工改动的前提下追加本阶段记录。
- 2.44 最终门禁确认报告为 3,058 行、SHA256
  `7e548377dca7bb9ac3597802c29e91208909dd65a187b215b208021666d6530a`；
  章节/引用、术语、配置与源码身份、v1/v2 日志/退出码哈希和 diff 均通过。
  当前边界仍是“输入已验证、OCI 尚未构建”，没有新增端到端性能结果。
- ca4a404e9 首次 OCI 双构建没有进入 payload 导出：固定容器用户为 UID
  22633，但两个 Git worktree 属 UID 0，容器内缺少 safe.directory，故两轮
  在 `git status` 处同时退出。可用进程级 `GIT_CONFIG_COUNT/KEY/VALUE`
  精确允许 `/workspace` 与 `/workspace/glm52_oscar_vllm`，无需修改 Git 配置。
- 有效 v3/v4 双构建的四项不可变 OCI 内容逐字节一致：index/config/
  manifest/layer SHA256 分别为 `18310465…4d90`、`7c85cdd0…4eb8`、
  `fb8e914c…4a23`、`3f03376d…e203`。candidate layer 为
  109,147,892 bytes、5,298 members，diff-ID 为 `5f8875b9…7a14`。
- 两轮递归 verifier 各自通过 4,744 个源码文件、4 份 rotation artifact、
  7 个 base 原生扩展、32 个精确 base layers；候选层未覆盖原生扩展，
  不含 whiteout。build/verify JSON 因记录各自路径而哈希不同，不影响
  不可变 OCI 内容一致。
- 2.45 最终为 3,145 行、SHA256
  `f715f5c3cf344b08f4a4c82b969570f28abfdd29ef1a967764a9cf63f63a6f3a`；
  章节/引用、术语、失败与有效证据、OCI identity/bytewise、daemon 边界和
  diff 门禁全部通过。当前仍没有 daemon image 或新端到端性能结果。
- ca4a404e9 v3 已由 skopeo 1.4.1 导入 daemon；image ID
  `sha256:7c85cdd0…4eb8`、33 层、最后 diff-ID `sha256:5f8875b9…7a14`、
  tag 与 8 labels 均通过独立审计。首次外层客户端提前返回但容器仍运行，
  通过接管同一容器而非重复导入取得真实 exit=0 和完整日志。
- daemon 导入的原/接管日志均为 14,998 bytes 且逐字节一致，SHA256
  为 `75b98c9d…4551`；两份退出码均为 0。inspect/audit/后置 GPU
  快照 SHA256 为 `d9a47f16…76bd`/`15cd9abd…7c5b`/
  `e3d6d9dc…ad40`。导入后全部 8 卡 0 MiB/0%、无 compute process。
- 2.46 最终报告为 3,208 行、SHA256 `99de2358…79`；标题连续、结构引用、
  术语、主要证据哈希与 diff 门禁全部通过。8 项 labels 已由 build report、
  audit JSON 与 daemon inspect 三方精确复核，不依赖报告文字推断。
- ca4a404e9 runtime import 精确复用冻结协议一次通过；JSON/log 哈希
  `0910b598…7b7a`/`f2e60043…189a` 与 fd281f5f9 历史证据逐字节一致。
  候选 Python/原生扩展、78 层 rotation、三项 artifact hash、固定包版本和
  `reasoning_effort=max` 均匹配，`cuda_initialized=false`。
- 2.47 最终报告为 3,258 行、SHA256 `813fef10…71490`；1.1–1.5/
  2.1–2.47 连续，结构引用、术语、runtime 六份证据哈希、冻结证据
  逐字节一致性、GPU 空闲与 diff 门禁均通过。
- Stage 9 控制 Dockerfile 的最小迁移已经落地并通过 CPU-only 身份门禁：
  新默认 base 为 ca4a404e9 Phase 6 daemon tag，Dockerfile SHA256 为
  `f832ebb1…2737`，旧 fd281f5f9 身份为 0。实际 daemon image 为
  `sha256:7c85cdd0…4eb8`、33 层，source commit/tree 和 candidate layer
  digest 与冻结 build report 一致；此阶段没有构建控制镜像、没有分配 GPU，
  因而也没有新的 32K/batch1 性能结论。
- 2.48 发布前复核通过：报告 3,312 行、SHA256 `921ee9d7…9590`，
  1.1–1.5/2.1–2.48 连续，术语、实际 Dockerfile/base/source 身份、4 份
  证据哈希及 2,179-byte 总量均一致，diff 无错误。
- ca4a404e9 Stage 9 控制镜像已 CPU-only 构建为
  `sha256:265e6ca1…067f1`。身份审计确认 control/base 为 34/33 层、前 33 层
  逐层一致、labels/entrypoint 完全继承；CPU runtime JSON SHA256
  `5ac65b5d…1f20` 与 fd281f5f9 历史控制镜像逐字节一致，且
  `cuda_initialized=false`。该阶段仍不构成 32K/batch1 性能结果。
- 2.49 发布前门禁通过：报告 3,369 行、SHA256 `9fcecdf3…009c`，
  1.1–1.5/2.1–2.49 连续；镜像身份、runtime 字节一致性、13 份证据与
  42,387-byte 总量、术语和 diff 全部一致。
- ca4a404e9 overlay 已证明与 v3 candidate layer 的 4,749 个普通文件逐文件
  一致；另建的 6 个 lower native symlink 与 fd281f5f9 正式 overlay 完全
  同构且目标二进制哈希不变。Phase 1/5/7/9 和正式 wrapper 已绑定新
  commit/tree、OCI、overlay 与 control image；旧 fd281 正式身份计数为 0。
- 静态迁移最终有效结果为 Phase 7 20/20、Phase 9 22/22、compile 11/11、
  递归 verifier 64/64。verifier v3 JSON/log SHA256 同为 `67b040dc…fac9c`；
  GPU 始终未分配。早期 19/20、超时、完整 dry-run 缺 driver 和 v2 缺模型
  mount 均有独立失败日志，不计入通过结果。
- 2.50 发布前门禁通过：报告 3,455 行、SHA256 `68a0f76d…04c4`，
  1.1–1.5/2.1–2.50 连续；配置、overlay、工具测试、64/64、25 份证据与
  102,643-byte 总量、术语和旧身份清零全部一致。
- ca4a404e9 driver-injected preflight 退出 0：64/64 静态身份、fixed import
  和服务参数均通过，两处 `cuda_initialized=false`，没有加载模型；解析的
  131072 max model len、16 seqs、2048 batched tokens、OSCAR INT2 和 profiler
  均匹配。容器删除后 8 卡全空闲。
- 2.51 发布前门禁通过：报告 3,528 行、SHA256 `a33e1518…bc75`，
  1.1–1.5/2.1–2.51 连续；14 份证据/45,826 bytes、64/64、两处
  CUDA=false、正式参数、术语和 diff 全部一致。
- ca4a404e9 正式 32K/batch1 三轮全部 3/3 completed、0 failed、0
  preemption，三轮 mean TTFT 为 `32657.086/32683.066/32705.530 ms`，
  mean TPOT 为 `200.037/200.404/199.531 ms`；中位汇总为
  `32683.066/200.037 ms`，相对极差仅 `0.1482%/0.4365%`。
- 与 fd281f5f9 的 `35683.893/197.826 ms` 相比，ca4a404e9 TTFT 降低
  `3000.827 ms`（`-8.409%`），TPOT 回退 `2.211 ms`（`+1.118%`），
  请求吞吐提升 `4.729%`。与 BF16 的 `12528.026/178.832 ms` 相比，
  TTFT 仍慢 `160.880%`、TPOT 慢 `11.858%`；因此 tile gate 有明确
  TTFT 收益，但没有关闭 TTFT 门限。
- profiler 通过：critical rank=2、critical kernel total=`65104 ms`，
  8 worker trace、8 CUDA table、1 frontend trace 全部通过哈希与 rank
  校验；测量峰值每卡 `80679 MiB`，profile 峰值每卡 `80691 MiB`，
  max running=1、waiting=0、KV usage=`5.7902%`。正式证据已封装为 40 份
  小文件，manifest SHA256 `21fdbe75…1fee` 且 40/40 校验通过。
- 2.52 发布前门禁通过：报告 3,619 行、SHA256 `bd38d36d…c0a4`；
  1.1–1.5/2.1–2.52 连续，`三池` 为 0、`A800` 仅保留历史文档链接，
  主要证据哈希、三轮/门限复算与 diff 全部一致。首次术语校验错误使用
  单词边界，因下划线使历史链接不匹配而产生校验器假失败；改为精确子串和
  行号断言后通过，报告未因此修改。
- 有效 ca4a404e9 CPU-only 多 chunk 归因 ID 为
  `20260731T1948Z_ca4a404e9_32k_prefill_trace_v2`，耗时
  `114.491409 s`。固定 Python 3.12.13/ijson 3.4.0.post0，Docker
  使用 runc、network none、4 CPUs，未注入 NVIDIA runtime。首轮预装
  ijson 3.5.0 结果因不合冻结协议已作废，未混入正式结论。
- 8/8 ranks 均为 144 个 execute context、16 个 prefill chunk、
  32,768 token；全部 trace size/SHA256 与 summary 逐份一致。
  prefill wall/kernel/generation 中位数为
  `32756.591502/31755.930453/269.372213 ms`。
- `_mixed_sparse_prefill_stage1` 为 `20128.143242 ms`/1,248 次、占 wall
  `61.447612%`。相对 fd281f5f9，stage1 降低
  `3006.728218 ms`（`-12.996520%`），wall 降低
  `2974.890437 ms`（`-8.325684%`），解释比例 `101.070217%`；
  剩余 wall 只增加 `31.837781 ms`（`+0.252749%`）。因此 tile gate
  是本轮 TTFT 收益的因果主体。
- 相对 BF16，OSCAR prefill wall 仍多 `22670.121734 ms`；
  当前 stage1 相对 BF16 原生 `_sparse_mla_kernel_final_static` 多
  `16743.769268 ms`，解释 wall 差距的 `73.858312%`。下一最小候选
  应继续减少 stage1 在全部 16 个 chunk 的有效工作，转向 generation
  或其他小 kernel 暂无数据支持。
- 小型证据目录为
  `artifacts/phase9-control/20260731T1824Z_stage9_candidate_ca4a404e9_32k_b1_v1/formal_32k_b1_trace_analysis`；
  8 个 manifest 条目加 manifest 共 9 文件、346,895 bytes，manifest SHA256
  `bf94602fc922654482cfbf649d5411eea32d63a3d1aa81baf0552abca6eb5b73`。
  有效 summary/comparison/validation SHA256 为
  `54b8feeb…b27b9`/`9633a0de…1570b`/`025b2b7b…1875`。
- 报告 2.53 修改前已完整重读 3,619 行，读取前后 SHA256
  均为 `bd38d36d…c0a4`。修改后报告为 3,698 行、SHA256
  `b98b932fa41e012f4b143f574640c731073d2d1adc11b9a5ce3501d65c6fbb9e`；
  1.1–1.5/2.1–2.53 连续，新章引用、术语、数值、证据和 diff 门禁通过。
- 原生 `csrc/sampler.cu` 的 prefill top-k wrapper 已支持
  `VLLM_TOPK_PREFILL_SORT_INDICES`：`rowLen<=topK` 时在 index sort 前直接
  返回连续有效索引和 `-1` 尾部；长行则用 CUB `BlockRadixSort<int>`
  对已选 top-k index 重排。重排不改变 selected 集合，但能把 prefix/
  recent 聚集到少量 tile，让 ca4a 的两个 `has_bf16` gate 跳过更多
  确定为零的 BF16 dot。sampler/stage1 源码 SHA256 为
  `6a815b61…f37b`/`23b08ffa…707c`。
- CPU-only 覆盖率审计
  `20260731T2002Z_prefill_sort_coverage_32k_v1` 使用 Python 3.12.13/
  Torch 2.11.0+cu129、runc、network none、4 CPUs，耗时
  `809.295279 s`，状态 passed。长于 10 分钟，在约 10 分钟时已报告
  12/16 chunks 进度。
- seed 42 的 16-chunk 合成输入共有 65,012,736 个有效 selected
  token、581,379 个 BF16 selected token。按 token index 排序后，含 BF16
  的 tile 从 457,470 降到 131,621，减少 325,849（
  `71.228496%`）；全 history tile 从 3,605,435 增到 3,931,284。
  首 chunk 因 `rowLen<=topK` 快捷路径不变；后 15 chunks 单独统计为
  372,103→46,254，减少 `87.569571%`。
- 边界：上述仅是确定性随机 selected 的工作量覆盖率，不是正式
  DSA 输出；未测量 native top-k 额外排序时间、stage1 CUDA 时间、
  output/LSE、TTFT、TPOT 或吞吐。候选只达到
  `screening_supported_runtime_unmeasured`，不能宣称性能改善。
- 小型证据 9 项加 manifest 共 10 文件、25,005 bytes，manifest
  SHA256 为 `f2949866f90de61b963fa3c68a0cf2171effb294cd0b82cd452c4ecaacb1dbb0`；
  summary/assessment/validation SHA256 为 `871b03f4…3dace`/
  `3c4921b1…d67c3`/`46f2cb36…52c59`。
- 2.54 修改前已按 8 个连续区间扫描当前 3,698 行报告，读取前后
  SHA256 均为 `b98b932f…bb9e`。修改后报告为 3,790 行、SHA256
  `f00f184c40e43152c14b97d4172927f7576762b674c14d4e75c8ce7e319f7c6f`；
  1.1–1.5/2.1–2.54 连续，新章引用、术语、数值、证据和 diff 均通过。
- 2026-08-01 续跑确认：`benchmark_oscar_prefill.py` 当前只预生成一份
  `selected_tokens`，`call_attention()` 在每次计时调用中直接切片使用；若把
  `torch.sort` 放进该函数，会把 PyTorch sort 开销混入 stage1 CUDA 时间，
  与“先隔离 stage1 收益”的筛选目的不符。下一实现应在输入构造阶段一次性生成
  token-index 顺序副本，再由 config 选择原始/排序 tensor。
- 原生 `topKPerRowJob` 在 `rowLen <= topK` 时会在排序分支之前直接返回，
  因而首个 2K chunk 的有效前缀/`-1` 尾部不得排序；只有
  `query_position + 1 > topK` 的长行应按 token index 重排。CPU helper 和测试
  必须复现该边界。
- 恢复检查误用了不存在的 `configs/phase9_stage9_performance.json`；正式路径是
  `configs/phase9/performance_matrix.json`。失败为只读且无产物变更。
- 有效 CPU 32K 末段审计实际处理 4,194,304 个 selected index，排序前后
  每行集合相同。原始/排序含 BF16 tile 为 4,518/2,326，减少 2,192
  （48.51704293935369%）；BF16 selected token 均为 4,696，全 history tile
  从 257,626 增至 259,818。该数值与先前 16-chunk 合成审计的末 chunk
  完全一致，证明新 helper 复现了该轮排序语义，但仍不证明真实 DSA 分布或
  CUDA 性能。
- 排序候选在工具中只多生成一份 `selected_tokens_sorted`，计时调用仅按 config
  选择预生成 tensor；因此后续 GPU 微基准只隔离 stage1 的访问/计算变化，
  不包含 `torch.sort`，也不包含生产原生 `topKPerRowPrefill` 的 CUB sort 开销。
- 修改 2.55 前的报告顺序重读已完成 1–1,000 行；读取起点为 3,790 行、
  SHA256 `f00f184c…f7c6f`。前段既有结论与当前候选边界一致：单层微基准不得
  替代 32K/batch1 端到端 TTFT/TPOT。
- 修改 2.55 前的报告顺序重读已继续完成 1,001–2,000 行；当前阶段没有改动
  报告。既有各候选均遵循 CPU/单卡筛选、完整 CUDA、正式 32K 逐级门禁，
  本轮排序候选继续沿用同一证据边界。
- 修改 2.55 前的报告顺序重读已继续完成 2,001–3,000 行；2.39–2.43 再次
  明确后续 chunk、合成 selected 与端到端证据的边界，本轮工具记录必须避免
  把 CPU tile 覆盖写成真实 DSA 或 CUDA 加速。
- 修改 2.55 前的报告顺序重读已完成 3,001–3,790 行。读取前后均为
  3,790 行、SHA256
  `f00f184c40e43152c14b97d4172927f7576762b674c14d4e75c8ce7e319f7c6f`，
  确认报告在本轮修改前没有并发手工变更；下一步只追加 2.55。
- 2.55 发布前门禁通过：报告为 3,875 行、SHA256
  `c0663130457e83b108ab7318b827ee21f13fb8bf0f2ce324e538b0f630d6462e`；
  1.1–1.5/2.1–2.55 连续，`三池=0`，大写 `A800` 仅保留第 5 行历史链接，
  CPU 语义/TDD 数值与证据 JSON 一致，manifest 4/4 通过。
- 证据首次 JSON 语法检查误用只存在于控制容器内的
  `/opt/fp8_speed_up_v4_venv/bin/python` 宿主路径，命令未启用 fail-fast，
  后续 size/hash 仍被打印；该次不能作为 JSON 通过证据。使用宿主
  `python3 -m json.tool` 并启用 `set -e` 后，三份 JSON 全部通过。
- 单卡排序筛选的精确 CUDA 中位数为原始 `20.206592559814453 ms`、排序
  `19.581951141357422 ms`；排序降低 `0.6246414184570312 ms`
  （`-3.0912753677197258%`，`1.0318988344903883×`）。wall 中位数
  `20.2330332249403→19.60908528417349 ms`（`-3.083808215160244%`）。
- 原始/排序 split1 都相对同轮 split16 通过 `atol=rtol=0.002` allclose；
  原始 output/LSE max_abs 为 `0.003370285/0.002224922`，排序为
  `0.003452063/0.002224922`。排序改变累加次序并轻微改变 output 诊断值，
  但没有越过冻结 allclose；不能写成 bitwise 或 max_abs<=0.002。
- 末段合成 BF16 tile 减少 `48.5170%` 最终只换来 stage1 `3.0913%` 中位数
  改善，说明 BF16 零贡献 dot 只占 stage1 的一部分。该收益量级较小；必须先
  测 native top-k CUB sort 额外成本，不能直接假设端到端净收益为正。
- 实际 cold cubin 为 142,000 bytes，dynamic shared 109,568 bytes、
  242 registers、0-byte stack；排序和原始顺序复用同一个编译 kernel。
  61 个 cold cache 文件，`du -sb=2,951,353`。
## 2026-08-01 报告 2.56 写入前复读（续 1）

- 已顺序复读 `OSCAR精度与性能优化记录.md` 第 1–1000 行；现有 2.11–2.15 对 GPU 空闲门禁、微基准与端到端结果边界的表述清楚。
- 新增 2.56 时必须延续同一证据标准：combined allclose 通过不能改写成单项 `max_abs<=0.002`，且预排序 stage1 微基准不能冒充包含原生排序开销的端到端收益。
## 2026-08-01 报告 2.56 写入前复读（续 2）

- 已无截断地复读第 1001–1500 行；确认现有正式结果均按“微基准筛选→正确性回归→正式 32K/batch1→trace 归因”分层记录。
- 2.56 应只定性为 stage1 候选筛选通过，并把原生 `topKPerRowPrefill` 排序成本列为进入端到端前的未决门禁。
## 2026-08-01 报告 2.56 写入前复读（续 3）

- 已无截断地复读第 1501–2000 行；确认 2.24–2.32 对单卡筛选、完整 CUDA、正式 32K 和离线淘汰候选的结论边界一致。
- 2.56 需同时记录排序后 BF16 tile 数下降与实测 stage1 仅小幅改善，避免把 coverage 代理指标等同于 CUDA 时间收益。
## 2026-08-01 报告 2.56 写入前复读（续 4）

- 已顺序复读第 2001–2500 行；2.33 的 causal-loop 记录再次确认，单层候选必须在正确性、资源和端到端成本之间逐级收敛。
- selected-index 排序与 causal-loop 不同：当前实测排除了排序本身的生产开销，所以 2.56 不能直接推进正式 OCI/32K 链路。
## 2026-08-01 报告 2.56 写入前复读（续 5）

- 已顺序复读第 2501–3000 行；2.38–2.43 明确展示单层收益在 32K 多 chunk 下会被覆盖范围稀释，且正式结论必须来自端到端轮次。
- 2.56 的下一步应先测排序生产成本并估算/验证净收益，不应仅凭当前约 0.625 ms 的单次 stage1 改善直接迁移生产候选。
## 2026-08-01 报告 2.56 写入前复读（续 6）

- 已顺序复读第 3001–3500 行；确认 ca4a 候选从 Phase 6 到正式 Stage 9 的身份链、容器与 GPU 门禁均已有完整记录。
- 2.56 应继续绑定控制镜像 `oscar-glm-stage9-runtime:ca4a404e9` 及其完整 image ID，确保新微基准可追溯但不混入新的生产源码身份。
## 2026-08-01 报告 2.56 写入前复读（续 7）

- 已顺序复读第 3501–3875 行，连同此前第 1–3500 行完成修改前全文复读；当前末节为 2.55。
- 新节将紧接 2.55 记录 GPU 单层实测，并保持既有结论：预排序 tensor 不含原生 CUB sort 成本，必须先补测该成本，不能把本轮写成 32K/batch1 端到端改善。

## 2026-08-01 报告 2.56 门禁结果

- 修改前全文为 3,875 行、SHA256 `c0663130…6462e`，顺序复读完成后未发现并发改动。
- 修改后为 3,961 行、SHA256 `4999befc3e39ecadbd98e6ec8994517be34efae7c5266e9547b81d8f966eb901`；1.1–1.5/2.1–2.56 连续，`三池=0`，大写 `A800` 仅保留第 5 行历史链接。
- 2.56 的 GPU 数值逐项来自冻结 comparison/resource/run identity，17/17 manifest 复算通过；候选仍明确标记为 `stage1_screen_passed_native_sort_cost_unmeasured`。

## 2026-08-01 原生 prefill top-k 成本工具设计

- 原生入口为 `torch.ops._C.top_k_per_row_prefill`；`sampler.cu` 在每次调用时仅当 `VLLM_TOPK_ENV_CACHE=1` 才使用加载期缓存，否则实时读取 `VLLM_TOPK_PREFILL_SORT_INDICES`。当前正式配置未设置两项变量。
- 32K 最后一个 2K chunk 的精确单请求几何为 2,048 行、logits 宽度 32,768、row start 全 0、row end 为 30,721–32,768、top-k 2,048。
- 最小工具将显式保持 `VLLM_TOPK_ENV_CACHE=0`，对同一 logits 在同一进程中分别执行 unsorted/sorted 原生分支；计时后要求逐行 selected 集合一致且 sorted 输出单调，避免只测时间不验语义。
- 正式 profiler 中未排序 kernel 约为 `164.8 us/call`、1,344 次约 222 ms；该历史值只用于选择多 iteration 降噪，不作为新候选结果。
- TDD 红灯已在固定 ca4a 控制镜像、无 GPU 的环境中得到：新测试在 collection 时因目标工具尚不存在而以 1 error 退出；错误精确为缺少 `benchmark_topk_prefill_sort.py`，不是 pytest 环境问题。
- 最小工具实现后同一固定 CPU-only 容器定向测试为 6/6 passed。工具强制 `VLLM_TOPK_ENV_CACHE=0`，计时范围只包含既有 `_C` op；默认每个模式 5 次 warm-up、7 个样本、每样本 20 次调用。
- 首轮组合门禁在 Ruff check 处 fail-closed：两份新文件存在 import 顺序、`Sequence` 来源、未使用 `sys` 和行长问题，pytest/compile 尚未继续。只对两份新文件执行 Ruff 自动修复与机械格式化后，待重跑完整组合；这不是测试或运行时失败。
- 首次宿主 Ruff 自动修复未显式加载源码仓库的严格 `pyproject.toml`，所以容器重跑仍报告 3 个 import/typing 规则错误；随后显式指定同一配置完成 4 项机械修复。最终控制容器内 Ruff check/format、compile、CLI help 和 4 个 Phase 9 测试文件 32/32 passed，组合退出码为 0。
- 写入 2.57 前已重新顺序复读报告第 1–1000 行；既有阶段均明确区分工具、单卡和端到端证据，新节继续只记录 CPU-only 工具门禁，不预写 GPU 成本结果。
- 写入 2.57 前已继续复读第 1001–2000 行；历史门禁再次确认 GPU 分配必须发生在工具/报告发布后，当前阶段仍不执行原生 op。
- 写入 2.57 前已继续复读第 2001–3000 行；2.39–2.43 的记录要求合成 32K 末段只能作为筛选，2.57 必须保留 logits 非正式 DSA 的边界。
- 2026-08-01：修改 2.57 前已完成对当前报告全部 3,961 行的顺序复读；
  末段 3,001–3,961 行已复核，读取后 SHA256 仍为
  `4999befc3e39ecadbd98e6ec8994517be34efae7c5266e9547b81d8f966eb901`，
  与已发布 2.56 一致，未发现并发手工修改。下一步只记录原生 prefill top-k
  排序成本工具的 CPU/TDD 门禁；本阶段尚未分配 GPU，也没有 sorted/unsorted
  原生算子时延或端到端性能结论。
- 新增原生成本工具最终 SHA256 为
  `3ca80e410911542bec7e2bc38dda67d5bac44315bcc7d73c35b87023e9c581fc`，
  定向测试 SHA256 为
  `3b936e997b8b9384df0c829a1f0687802dadb8f6c45df0a94f5f9f46b9408839`。
  工具固定 2,048×32,768 FP32 logits、top-k 2,048、seed 42，默认
  5 warm-up、7 samples、每样本 20 次调用；强制关闭 top-k 环境缓存后，
  对同一 logits 依次测 unsorted/sorted，并逐行验证集合相同、排序单调且
  final chunk 无负 index。工具的 interpretation boundary 明确排除 stage1、
  TTFT、TPOT、吞吐和正式 DSA logits。
- 主仓库当前已发布 HEAD 为 `b412ef0`，源码仓库为
  `ca4a404e913ce55237ca60383cc86e221fbfea26`、tree
  `079815219a02add3f37318ed434924e80f80a35d`，源码仓 clean/published。
  原生绑定位于 `csrc/torch_bindings.cpp`，实现位于 `csrc/sampler.cu`；当
  `VLLM_TOPK_ENV_CACHE=0` 时每次调用读取
  `VLLM_TOPK_PREFILL_SORT_INDICES`。当前正式 Phase 9 配置没有设置这两个
  变量，因此新工具强制 cache=0 后逐项切换排序变量，且没有修改生产配置。
- 2.57 已实时追加并通过修改后门禁：报告现为 4,017 行，SHA256
  `7722c5f226f12231554b58ccc5303f4512f7bd18d939b4a82858504d2b08a0d2`；
  1.1–1.5/2.1–2.57 连续，`三池=0`，大写 `A800` 只位于第 5 行历史
  文件链接，2.56 引用、工具/测试 hash、32/32 门禁和
  `git diff --check` 均通过。
- 原生 top-k 成本工具、6 项定向测试、2.57 与 planning 已由主仓库提交
  `b3f2159a3771a40949dfab7aaf75be6e9e0dc654` 推送；主仓库本地/远端一致，
  源码仓库继续 clean/published 于 `ca4a404e9`。下一步先发布本条状态，再执行
  新一轮双空闲门禁；尚未分配 GPU。
- 原生 top-k 微基准前双空闲检查为 `2026-07-31T21:07:42Z` 与
  `2026-07-31T21:08:55Z`，间隔 73 秒；两次均为 8/8 GPU `0 MiB/0%`、
  无 compute process。唯一运行的项目外下载容器
  `deepseek_v4_hf_downloader_vllm0230` 的 DeviceRequests 为 null，不占 GPU，
  因而未终止。下一步先发布本条状态，再固定只用 GPU 0。
- 启动前复核控制镜像为冻结的
  `oscar-glm-stage9-runtime:ca4a404e9`，image ID
  `sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`，
  固定 Python 路径 `/opt/fp8_speed_up_v4_venv/bin/python`、候选
  `PYTHONPATH=/opt/vllm_glm52_v1`。`21:09:34Z` 即时检查 GPU 0 仍为
  0 MiB/0%、无 compute process。
- 首轮 GPU 微基准 `20260731T2110Z_topk_prefill_sort_2k_gpu_v1` 在导入
  `vllm._C` 期间退出 1，未进入原生算子或计时。根因是 Docker 显式使用宿主
  UID 22633，而镜像 `/etc/passwd` 中没有该 UID；Torch Dynamo 初始化调用
  `getpass.getuser()` 时触发 `KeyError: getpwuid(): uid not found: 22633`。
  `--rm` 已删除容器，`21:10:51Z` 8/8 GPU 均恢复 0 MiB/0%、无 compute
  process。失败目录只有 start/end/exit/log 四项，run log SHA256
  `0464c950d6b632e949b7353b2890d3a3104558515ff0768625c3b1e784d38e70`；
  没有 `result.json`。下一轮不重复该命令，去掉 `--user`、使用镜像默认 root，
  其余冻结协议不变。
- 修改失败记录 2.58 前已对当前报告全部 4,017 行做顺序分块扫描；读取后
  SHA256 仍为
  `7722c5f226f12231554b58ccc5303f4512f7bd18d939b4a82858504d2b08a0d2`，
  与 2.57 发布前验证值一致，确认没有并发手工修改。2.57 全节已重新读取；
  下一步只追加首轮启动失败及“无性能数据”的边界。
- 失败证据复制后首次 manifest 校验从仓库根目录执行，manifest 内使用相对
  文件名，导致 7 项均因解析目录错误而 `FAILED open or read`；文件与 manifest
  本身已经生成，未发生内容校验失败。下一步改在证据目录内执行
  `sha256sum -c manifest.sha256`，不重复错误工作目录。
- 进入证据目录后的有效 manifest 校验为 7/7 passed；失败证据目录含 manifest
  共 8 文件、`du -sb=4,247 bytes`，manifest SHA256
  `0a21133d5220bbfe83c5666107710f9f928d04628eb3bf8b123b37886297cd84`。
  2.58 已实时追加并通过门禁：报告 4,067 行、SHA256
  `09cfc6549708985e5920d280c3af782864c380bf6810d7b232abdfb40eda9a13`；
  1.1–1.5/2.1–2.58 连续，术语、失败边界、证据 hash、2.57/2.58 引用和
  `git diff --check` 通过。证据路径按项目既有规则被 `.gitignore` 忽略，
  但已落地在本机 artifact 目录。
- 修正轮次的新双空闲检查使用 v1 退出后的 `21:10:51Z` 复查与
  `21:13:40Z` 新检查，间隔 169 秒；两次均为 8/8 GPU 0 MiB/0%、无
  compute process。第二次检查同时确认主/源码仓库 clean/published。
  下一步发布本条状态后，固定 GPU 0、使用镜像默认 root 启动 v2。
- v2 `20260731T2114Z_topk_prefill_sort_2k_gpu_v2` 固定 GPU 0、默认 root，
  于 `21:14:24Z–21:14:28Z` exit=0/status=passed。unsorted/sorted 原生
  top-k CUDA 中位数为 `0.3689984083/0.5423615932 ms`，排序额外成本
  `0.1733631849 ms`（`+46.9820956%`）；wall 对应
  `0.3700094298/0.5433938466 ms`，增量 `0.1733844168 ms`
  （`+46.8594590%`）。7 个 CUDA 样本范围分别为
  `0.3685375929–0.3703295946 ms` 和 `0.5421055794–0.5426688194 ms`。
- v2 语义门禁 passed：逐行 selected set 精确相同、sorted 输出逐行单调、
  invalid index count=0。result/run/comparison SHA256 分别为
  `a5c447b2…975d9`/`201c078c…f152`/`ead985df…51c`。容器已删除，
  `21:14:55Z` 8/8 GPU 为 0 MiB/0%、无 compute process。
- 将 v2 top-k 成本与 2.56 的独立 stage1 微基准做算术组合：stage1 节省
  `0.6246414185 ms`，排序成本消耗其 `27.7540329%`；估算 raw/sorted
  合计为 `20.5755909681/20.1243127346 ms`，净变化 `-0.4512782335 ms`
  （`-2.19326985%`）。该估算来自两个独立合成微基准，不是正式 DSA 或端到端
  结果，只支持进入生产开关候选。
- v2 证据目录已封存 13 项加 manifest，共 14 个文件、`du -sb=8,822 bytes`；
  13/13 manifest 复算通过，manifest SHA256
  `ca6ba33919cc5e5ed135aa9a075f11cd9c6f43c4b9921269c27340f3e232d2da`。
- 修改 2.59 前已对报告全部 4,067 行做顺序分块扫描，读取后 SHA256 仍为
  `09cfc6549708985e5920d280c3af782864c380bf6810d7b232abdfb40eda9a13`，
  与 2.58 验证值一致；2.58 全节已重新读取，未发现并发手工修改。
- 2.59 已实时追加并通过门禁：报告现为 4,139 行、SHA256
  `555ba1da1ce5ebeaa59ea21ad95332480e4df03908a883610a9b8003bf633923`；
  1.1–1.5/2.1–2.59 连续，`三池=0`、大写 `A800` 仅历史链接，result/
  comparison 数值、13/13 manifest、2.56/2.58 交叉引用和
  `git diff --check` 均通过。
- 正式 Phase 9 `performance_matrix.json` 当前 SHA256 为
  `14d3f71e8a36b429d31e78ff8a7c0a9810048a308440693aef08b2ae9aea4760`，
  没有通用 `env`/`extra_env` 字段，也没有任何 `VLLM_TOPK_*` 变量。下一步需
  先定位服务启动的唯一环境传播点和现有 config verifier，再决定最小改动是
  新增显式配置字段还是只改正式 wrapper；不能假设 JSON 字段会自动生效。
- `run_containerized_performance.sh` 的外层 `docker run` 目前只显式传递 FORMAL/
  run ID/路径/预验证身份/单格选择等变量，没有传递任何 top-k 开关；容器内服务
  进程将继承外层容器环境。`run_performance_matrix.py` 中的 `client_env` 仅用于
  benchmark 客户端，不能用来控制 server 的原生 top-k。正式配置的 verifier
  当前也不检查环境变量。下一步继续读取 wrapper 的配置解析和 server launch
  全路径，再以测试先固定“配置值与实际容器环境一致”的 fail-closed 语义。
- candidate 和 baseline 分别由 `scripts/phase9/run_candidate_tp8.sh` 与
  `run_native_tp8.sh` 启动；两者最终进入不同的 Phase 7/Phase 1 wrapper。
  因此不能把排序变量加到外层 Docker（否则 baseline 也会继承，破坏对照）。
  最小且可审计的路径是：在 performance config 新增 candidate-only 环境映射；
  candidate wrapper 从该字段读取并 export；candidate verifier 检查实际环境与
  配置一致；baseline wrapper 保持不变。还需检查现有 runtime manifest 是否
  已记录环境，再确定是否额外扩展证据字段。
- Phase 1 基础 wrapper 已生成 `runtime_environment.txt` 并把其 SHA256 写入
  `runtime_manifest.json`，因此候选 export 后会被既有证据链捕获，无需新增
  manifest 格式。当前 Phase 9 candidate verifier 只检查 config/server/identity，
  尚未检查 top-k 环境。最小改动可限于 performance config、Phase 9 candidate
  wrapper、candidate verifier 和定向测试；baseline wrapper无需改动。
- Phase 1 serve 路径已经固定 export `VLLM_TOPK_ENV_CACHE=1`，且
  `runtime_environment.txt` 会记录所有 `VLLM_*`。这意味着 candidate-only
  排序变量必须在导入原生扩展前由 Phase 9 candidate wrapper 顶层 export；
  生产中会由 load-time cache 固化为 true。2.59 微基准显式 cache=0 是为了在
  同一进程切换两种模式；其绝对排序成本仍包含同一 CUDA 排序路径，但生产开关
  读取方式不同，报告后续需明确此边界。
- 拟采用最小 config 字段 `candidate_runtime_environment`，唯一键为
  `VLLM_TOPK_PREFILL_SORT_INDICES: "1"`。candidate wrapper 从 JSON 读取并
  export；candidate verifier 对实际 `os.environ` fail-closed 比较；现有 runtime
  environment/manifest 自动保留证据。baseline 不读取该字段且保持未设置。
- candidate-only 环境传播 TDD 有效红灯为 1 failed：正式配置缺少
  `candidate_runtime_environment`，测试在精确 KeyError 处失败。最小实现已
  新增唯一映射 `VLLM_TOPK_PREFILL_SORT_INDICES="1"`；Phase 9 candidate
  wrapper 从 JSON 读取、要求映射精确相等后 export；candidate verifier 同时
  检查配置映射与实际 `os.environ`；native wrapper 不变。定向绿色为 1/1 passed。
- 首轮完整 CPU 门禁在 Ruff 首步 fail-closed，后续步骤未执行。4 项诊断均为
  两个既有文件的历史内容：`test_phase9_tools.py` 和
  `verify_candidate_performance.py` 的 import block I001，以及测试文件第
  139/482 行两条既有 127 字符 profiler 表头 E501；这些行都不在本次 diff。
  遵守 surgical changes，不为本配置改动重排整文件或修改历史表头。下一轮对
  这两个文件显式忽略已确认的 I001/E501，其余 Ruff 规则、format、JSON、shell、
  compile 和完整 pytest 仍 fail-closed 执行。
- 第二轮 Ruff check 在忽略已确认的历史 I001/E501 后通过，但 format check
  发现本次新增测试块需机械格式化；组合随即停止，JSON/shell/compile/pytest
  尚未执行。下一步仅对 `test_phase9_tools.py` 运行 Ruff formatter，并检查 diff
  确认没有改动本次新增块以外的既有代码，再重跑完整组合。
- Ruff formatter 只改写了本次新增的测试方法，没有触碰既有代码。最终 CPU-only
  组合全部通过：Ruff（仅忽略已确认的历史 I001/E501）、format、JSON parse、
  candidate shell syntax、固定 Python compile、`git diff --check` 均通过；
  Phase 9 四个工具测试文件为 `33/33 passed`、0 failed，耗时 4.35 秒。
  容器使用固定 ca4a404e9 控制镜像、network none、4 CPUs，未注入 NVIDIA
  runtime。当前尚未执行 formal driver-injected preflight。
- candidate-only 配置、wrapper、verifier、测试最终 SHA256 分别为
  `22ecca75b358aff3e04a3280d69a288239925f586f938a1d7e03dcf59e37b219`、
  `ec849660e7e0505980b2f43edaf1bf64a4901cc2b391db33b6bb14fe71589e05`、
  `eb89bb1f5f8d834839b395de2281d4e6b1682d9cb779737b0e3050acff7e6104`、
  `29ddd91cde1ac210f2de9eb79509ca8e9abdaae76847322724ced053271fc992`。
- 修改 2.60 前已对当前报告全部 4,139 行顺序分块扫描；读取后 SHA256 仍为
  `555ba1da1ce5ebeaa59ea21ad95332480e4df03908a883610a9b8003bf633923`，
  与 2.59 验证值一致。2.58–2.59 已重新读取，未发现并发手工修改。
- 2.60 已实时追加并通过门禁：报告现为 4,210 行、SHA256
  `4bb51a49c9eea4fc434959095e4ca17a7b021437108a466283bd82d7a07804e6`；
  1.1–1.5/2.1–2.60 连续，术语、配置传播、TDD/历史 Ruff 边界、四文件
  hash、33/33 与 `git diff --check` 均通过。
- candidate-only 配置链路、测试、2.60 与 planning 已由主仓库提交
  `8b347e1dadf2f89dad370b9cecdb4f01af2dd5cf` 推送；主仓库本地/远端一致，
  源码仓库继续 clean/published 于 `ca4a404e9`。下一步发布本条状态后再开始
  preflight 的新双空闲门禁。
- preflight 双空闲有效检查为 `2026-07-31T21:25:00Z/21:26:18Z`，间隔
  78 秒；两次均为 8/8 GPU 0 MiB/0%、无 compute process，外部下载容器
  DeviceRequests=null。中间异步 sleep 命令完成时未回传检查输出，因此没有把
  该空输出当证据，改用 `21:26:18Z` 独立命令作为第二次有效检查。
  下一步先发布本条状态，再运行 driver-injected candidate preflight。
- driver-injected candidate preflight
  `20260731T2127Z_candidate_topk_sort_preflight_v1` 的静态 verifier 已
  status=passed；新增 `performance.candidate_runtime_environment` 检查实际/期望均为单键
  `VLLM_TOPK_PREFILL_SORT_INDICES="1"`；新增实际环境检查也为 `"1"`。
  既有 source/candidate/server/matrix/frozen evaluator 检查均通过。但整轮尚未
  完成：`fixed_environment_import.json` 仍为 0 bytes，没有 exit/end 文件，
  Docker 容器仍在运行。此前把静态 JSON 的 passed 误判为整轮 exit=0，现已
  更正；继续监控固定环境 import 与后续 parsed args，不能标记 preflight passed。
- 后续监控确认 `fixed_environment_import.json` 已从 0 增至 804 bytes，内容
  `cuda_initialized=false`，固定 Python 3.12.13、Torch 2.11.0+cu129、
  Triton 3.6.0 和候选 `_C` 路径已输出。当前 `parsed_server_args.json` 仍为
  0 bytes，对应 Python 进程处于 D 状态进行冷导入/参数解析；外层 docker/tee
  仍在运行，无 exit/end 文件。继续等待，不终止有效进程。
- preflight 最终于 `21:29:09Z` 完成，起始 `21:27:10Z`，exit=0。
  `static_preflight.json` 为 66/66 checks passed（既有 64 项加 2 项环境检查）；
  fixed import 与 parsed args 均 `cuda_initialized=false`。static/fixed/args/run
  log SHA256 分别为 `93d4507f…90a6`/`56f92356…0574`/
  `227f21f6…3236`/`916b72c9…14d3`。容器已删除，`21:29:57Z` 8/8 GPU
  为 0 MiB/0%、无 compute process。过早判断已被后续完整证据纠正，最终状态
  可标记 passed；下一步封存 manifest 并更新报告，正式 32K 仍未启动。
- preflight 小型证据已封存 14 项加 manifest，共 15 个文件、
  `du -sb=43,485 bytes`；14/14 manifest 复算通过，manifest SHA256
  `a93f4dc3ac7be1122404a3fa29303d303d27f32a734298f1a19abf323c52ad3b`。
  validation JSON 对 exit=0、66/66、两处 CUDA=false、配置映射和实际排序变量
  逐项断言通过。下一步全文复读并实时新增 2.61；发布前不启动正式 32K。
- 修改 2.61 前已对报告全部 4,210 行顺序分块扫描；读取后 SHA256 仍为
  `4bb51a49c9eea4fc434959095e4ca17a7b021437108a466283bd82d7a07804e6`，
  与 2.60 验证值一致。2.59–2.60 已重新读取，未发现并发手工修改。
  validation JSON SHA256 为 `ee3a64dc…d57a`。
- 2.61 已实时追加并通过门禁：报告现为 4,275 行、SHA256
  `aa9b82134476e623c9cae453cd3783959351aa7670f446b308fccb454ca5dae7`；
  1.1–1.5/2.1–2.61 连续，术语、66/66、两处 CUDA=false、过早判断更正、
  14/14 manifest、2.52 协议引用和 `git diff --check` 均通过。
- 2.61 已由主仓库 `139d5254d359b382a7a79bffe0be48fc18a357aa` 推送。
  正式 32K/batch1 新双空闲有效检查为 `21:32:45Z/21:34:05Z`，间隔 80 秒；
  两次均为 8/8 GPU 0 MiB/0%、无 compute process，两仓 clean/published。
  异步 sleep 会话的输出未可靠回传，未计为证据。下一步先发布本条状态，再启动
  3 rounds + 8+8+1 profiler 正式轮次。
- 正式 run `20260731T2135Z_candidate_topk_sort_32k_b1_v1` 已于
  `21:34:50Z–22:13:27Z` 完整 exit=0/status=passed；三轮均 3/3 completed、
  0 failed。三轮 mean TTFT 为 `32434.978446/32449.244567/32456.209736 ms`，
  mean TPOT 为 `199.155074/198.447637/199.492602 ms`；中位汇总为
  `32449.244567/199.155074 ms`，请求吞吐 `0.017322629236 req/s`。
- 相对未排序 ca4a404e9 正式结果，TTFT `-0.715421%`、TPOT `-0.440883%`、
  请求吞吐 `+0.578794%`；相对 BF16 为 TTFT `+159.013233%`、TPOT
  `+11.364501%`、吞吐 `-38.955186%`。因此排序在正式负载上只带来小幅稳定
  改善，远小于 2.59 分离微基准估算的 `-2.193270%`，且 TTFT 仍未通过
  +20% 门限；TPOT 仍通过。
- profiler status=passed，8 tables+8 worker traces+1 frontend trace，critical
  rank=4、kernel total=`63657 ms`、profile elapsed=`764.154426 s`。summary/
  comparison/outer log SHA256 分别为 `c16c9596…56c6`/
  `ca50cba2…8bde`/`384f7e92…fca8`。容器已删除，`22:13:56Z` 8/8 GPU
  0 MiB/0%、无 compute process，两仓保持 clean/published。
- 正式轮次小型证据目录已封存 52 项加 manifest，共 53 个文件、
  `du -sb=1,156,489 bytes`；`evidence_manifest.sha256` 与
  `formal_validation.json` SHA256 分别为
  `770166b54bd3a3a2a430533afc3ad1451f7750c53ad53494891773b20de6a198`、
  `467dfa2911f36f3e4e16c13a184b7875cbf4e8d2e82df6628f2a333348b85220`。
  原始约 1.2 GiB trace 保留在 `/dev/shm`，未复制进仓库；正式 summary 已对
  全部 trace 重新哈希并完成 8+8+1 完整性检查。
- 报告修改前已顺序扫描全部 4,275 行，读取前后 SHA256 均为
  `aa9b82134476e623c9cae453cd3783959351aa7670f446b308fccb454ca5dae7`，
  与 2.61 发布值一致，未发现并发手工修改。首次证据复核误将实际文件名
  `evidence_manifest.sha256` 写成 `manifest.sha256`，且宿主没有 `jq`；这是
  只读校验命令错误，未改变证据。第二次复核又在证据目录内执行了以仓库根为
  基准的 manifest，导致 52 项路径全部被重复拼接；同时误读不存在的
  `/dev/shm/.../comparison.json`。manifest 首行已证明其路径基准是仓库根，
  comparison 的封存文件名包含 run ID；下一次从仓库根 fail-fast 复算。
- 正确的仓库根复算已 52/52 passed；summary/comparison/validation SHA256 与
  记录值一致。2.62 已追加，但首轮报告交叉引用正则使用无限位数字，误把
  output throughput `2.2172965422491715` 识别为 2.2172965422491715 节并失败；
  这是校验器误判，不是报告引用错误。约束为一到两位后又定位到历史行 1,864/
  2,612 的 `2.89×/2.85×` 性能倍数，不是章节引用；最终正则还需排除紧随
  `×` 的数字。
- 排除 `×` 后，所有识别出的引用编号均不超过 2.62；新增节集合实际为
  `[62, 61, 52, 52, 59]`。原断言只期待 2.52/2.59，漏掉标题 2.62 和发布来源
  2.61，属于校验器误判。补记时首个 patch 又引用了 findings 的上下文去修改
  task_plan，因而被完整拒绝；随后改用各文件真实尾部上下文。
- 2.62 最终门禁通过：报告现为 4,357 行、SHA256
  `946593c539d9c6523b8d28cb3f595a96030f3b3d8da3faa3eac9283adcffb10b`；
  1.1–1.5/2.1–2.62 连续，新节引用 2.52/2.59/2.61/2.62 均存在，术语、正式
  数值、证据 SHA256 与 diff 检查通过。
- 2.62 与 planning 已由 `d38dfde4ea5d1a4af1d575cae48fa56e38528d7d`
  推送；主仓库本地/远端一致。源码 submodule 仍固定
  `ca4a404e913ce55237ca60383cc86e221fbfea26`。
- 有效排序 trace analysis ID 为
  `20260731T2228Z_topk_sort_32k_prefill_trace_v1`，固定环境为控制镜像、runc、
  network none、4 CPUs、Python 3.12.13/ijson 3.4.0.post0；耗时约 112 秒，
  exit=0。8/8 ranks 的 trace size/SHA256 与正式 summary 精确一致，且均为
  144 contexts、16 prefill chunks、32,768 tokens。
- 排序 vs 未排序 profile trace 的 prefill wall 中位数
  `32756.591502→32478.967757 ms`（`-277.623745 ms/-0.847536%`），kernel
  `31755.930453→31481.247675 ms`（`-274.682777 ms/-0.864981%`）。stage1
  `20128.143242→19849.393880 ms`（`-278.749362 ms/-1.384874%`），调用数仍
  1,248；top-k `221.593157→251.608034 ms`（`+30.014877 ms/+13.545038%`），
  调用数仍 1,344，模板从 `<512,false,false>` 变为 `<512,false,true>`。
- stage1 节省解释 wall 改善 `100.405447%`，top-k 额外成本消耗 stage1 节省
  `10.767694%`；stage1+top-k 合计净省 `248.734485 ms`，解释 wall 改善
  `89.594096%`。去掉二者后 residual wall 仍下降 `28.889261 ms`。8/8 rank
  的 wall/stage1 delta 均为负，top-k delta 均为正，因此排序收益主要来自
  后续 stage1 访存改善，而不是 top-k 自身变快。
- 新 summary/comparison/validation SHA256 分别为
  `96c1a755c54775c2da1ccb4db7a7a7f989126a180957bb3a9ae4addfaaa8a9be`、
  `23b43324a390f5ea8d9cec5ee8e25b8309027033c161e61b6ce3e7a4b1021ed1`、
  `ac04e1b5a05a7e0580dc43d866d64472f80c18b9cd3fd695fb686df1cc483630`。
- 小型证据目录 `formal_topk_sort_trace_analysis_v1` 含 10 个 manifest 项，
  加 manifest 共 11 文件、371,268 bytes；10/10 复算通过，manifest SHA256
  `378404730e3553077121fca07019ea18e802a482a7782bfa034f582e9e053aff`。
  原始约 1.2 GiB trace 未复制，仍由正式 summary 与 trace_inputs hash 关联。
- 2.63 已实时追加并通过门禁：报告 4,435 行、SHA256
  `f851d6da63579ca8db960672c0f39fbacf76a80d4689ebf0451f6958a9c363e2`；
  章节、引用、术语、归因数据、证据 hash 和 diff 全部一致。
- 2.63 与 planning 已由 `2df5f5727f7eeb0335451166c29103cb0f9cca74`
  推送；主仓库本地/远端一致，源码仓库继续 clean/published 于 ca4a404e9。
- 排序 summary 的 16 个 2,048-token chunk wall 在 8 rank 间高度一致：首块约
  `1339.5–1339.9 ms`，第二块约 `1965.3 ms`，随后逐步增长，末块约
  `2196.2–2196.5 ms`。现有 analyzer 的 `chunks` 仅含 name/tokens/wall，不能
  区分 stage1、top-k 或其他 kernel 对逐块增长的贡献。
- 当前 grouped stage1 已固定 `block_t=16`、`block_h=32`（num_heads>16）、
  8 warps、num_splits=1；已有 `effective_topk=min(topk, causal_seq_len)` 和空
  BF16 tile 两处 dot gate。继续猜 tile/warps 已有历史负结果，下一步应先扩展
  现有 analyzer，按 prefill window 记录 kernel calls/total_ms，再选候选。
- analyzer format version 已从 2 升至 3；每个 `prefill.chunks[*]` 新增
  `kernel_total_ms` 与按 kernel name 记录的 calls/total_ms，整段既有聚合保持
  不变。TDD 红灯 1 error/1 pass，绿色 compile+2/2；更广四文件 unittest 为
  33/33，Ruff 0.14.0 check/format 与 diff 通过。最终 analyzer/test SHA256 为
  `724aeb5e45f8a9322b7e52d096fb38670ec768f89cb9844d1d49ab213cddbf43`、
  `f57b985ab2fb72258eb9212a5f662203e0c639a9174af7b38a2f21709690a48c`。
- 2.64 已实时追加并通过门禁：报告 4,489 行、SHA256
  `c4151cd62308e29041de3040a524fb3ca38a1813fb84c146733aadde5bd0ac5a`；
  章节、引用、术语、TDD/回归数值、文件 hash 和 diff 全部一致。
- analyzer v3、测试、2.64 与 planning 已由
  `1ae08c23ffe771999462796c32e407574fbb0e87` 推送；主仓库 clean/published。
- format v3 的排序/未排序逐 chunk trace 已分别在固定控制镜像、4 CPUs、
  network-none 环境完成，analysis ID 为
  `20260731T2245Z_topk_sort_32k_chunk_trace_v1` 与
  `20260731T2247Z_ca4a404e9_32k_chunk_trace_v1`；summary SHA256 为
  `a3c58c84073b7801ae8a2f439666b7ddbf6a4d8411fc0cb620f75e82ef67f0e7`、
  `b90f54cfae34cea7e4f98a5d5de9fbd3d802bf6658c8a9ef7196ed023177ab06`。
  两组均为 8/8 ranks、每 rank 16 chunks，逐块 kernel total 求和与整段聚合
  在 `1e-6 ms` 内一致；stage1/top-k 调用数分别为 1,248/1,344。
- 逐块中位数显示：首块 `stage1 676.799→676.914 ms`，没有排序收益；
  第 2–16 块排序后 stage1 每块下降 `14.102–23.323 ms`，top-k 每块增加
  `1.306–3.091 ms`。排序版 stage1 从第 2 块到末块基本稳定在
  `1273.341–1284.593 ms`，因此后续 wall 从 `1965.318` 递增至
  `2196.264 ms` 不是 stage1 自身随上下文线性增长造成的。
- 该分布与现有 `has_bf16` 动态门禁一致：排序把 BF16 token 聚集后，更多
  tile 可以跳过 BF16 路径；但“增加对称 `has_history` 门禁可提速”目前只是
  候选假设。既有覆盖率只统计 `tiles_with_bf16/all_history_tiles`，不能回答
  `all_bf16/no_history` tile 数量，必须先扩展 CPU 覆盖率统计并实测，不能据此
  直接声称性能收益。
- `summarize_selected_tiles` 当前以 16-token tile 统计 `tiles_with_bf16`，并将
  “16 个位置全部有效且没有 BF16”的 tile 计为 `all_history_tiles`；它没有显式
  构造 `is_history`，也没有 `tiles_with_history/no_history_tiles`。新增统计必须
  与 kernel 的真实有效位置 mask 对齐，尤其不能把首个 chunk 的无效尾槽误当
  成“无 history”优化机会。
- stage1 源码精确语义为 `is_history = valid & ~is_prefix & ~is_recent`；当前
  `history_page_table/data/scale/zero` 的 masked load、`history_scores` dot 和
  `history_acc` contribution dot 都没有 `has_history` 分支，而 BF16 的两次 dot
  已由 `has_bf16` 分支保护。CPU 指标应至少区分 active tile、含 history、
  不含 history、history-only 与 mixed tile；其中可安全令 `has_history=false`
  的统计口径是“至少一个 valid 槽且没有任何 history 槽”，允许最后一个部分
  tile 含 invalid 槽，不能要求 16 槽全部有效。
- 既有 2.54 artifact 的覆盖脚本同样只计算 BF16-containing 与 full-valid
  all-history tile；其 summary 文件名是 `summary.json`，不存在 `coverage.json`。
  对不存在文件的 `sed` 只读检查失败，未修改证据。
- 新 helper 字段的 TDD 已通过：`active_tiles`、`tiles_with_history`、
  `tiles_without_history`、`history_only_tiles`、`mixed_precision_tiles`、
  `all_bf16_tiles`。人工构造的 4 个 tile 覆盖 full BF16、full history、mixed
  与 partial-valid BF16，期望分别为 active 4、with/without history 2/2、
  history-only 1、mixed 1、full BF16 1。
- 固定 Ruff 0.14.0 check/format 通过；固定 ca4a 控制镜像、4 CPUs、断网环境
  compile 与定向 1/1 通过，四个 Phase 9 unittest 文件合计 34/34 passed。
  测试输出中的 argparse usage/error 是既有负向参数测试的预期 stderr，最终
  unittest 状态为 OK，不是回归失败。
- 报告 2.65 修改前已顺序扫描全部 4,489 行，读取前后 SHA256 均保持
  `c4151cd62308e29041de3040a524fb3ca38a1813fb84c146733aadde5bd0ac5a`，
  排除了读取期间的并发手改。追加后为 4,572 行、SHA256
  `7ae0c4cc3a567d1e856420b7ce9ee088ca65fe3635afcfb3621c953518e5952f`；
  章节 1.1–1.5/2.1–2.65 连续，新节引用 2.53/2.54/2.63/2.64/2.65 均存在，
  `三池=0`，大写 `A800` 仅保留历史链接第 5 行。
- coverage 工具、测试、2.65 与 planning 已由主仓库提交
  `d80d4fe08381f32b4034195667c538062b358d5a` 推送；源码 submodule 仍固定
  `ca4a404e913ce55237ca60383cc86e221fbfea26`，尚未修改运行时。
- 16-chunk CPU-only coverage 完成，耗时 `828.4 s`；排序后
  `tiles_without_history` 为 13,481，未排序为 199，增量 13,282；排序后仅占
  4,064,256 active tiles 的 `0.3316966254%`。`mixed_precision_tiles` 从
  457,271 降到 118,140，`tiles_with_bf16` 从 457,470 降到 131,621，与
  2.54 的总量一致。该直接 no-history 机会很小，当前证据不支持未经 GPU
  筛选就实现 `has_history`。
- 原始 summary 的通用 int 聚合误把每块常量 `tile_width=16` 求和成 256，
  也给该常量生成无意义 delta；逐 chunk rows 和目标计数本身有效。必须基于
  rows 生成排除常量的 validated summary，并通过分区不变量和 2.54 数值对账
  后再更新报告。
- 从 raw 16 个 unchanged rows 重建的 validated summary 已通过：active tile、
  history、BF16 分区各 32/32，sorted set 16/16，首块 shortcut，2.54 六项
  数值对账。validated summary/validation SHA256 为
  `5612b29ce88d51ab9a2ba1b50f5c494dadf055b9402e41dc80e503a97901d9d2`、
  `c51f917f2b2589ad5204fd46ff9805889ffd12fb7713c2a7cef3bf5ef578c6d9`；
  raw summary SHA256 为
  `ae009fa6bf2c74d01b73a728617ee19957c68a007f7098b7c4df21d75dc3cdea`。
- 排序使 `tiles_with_history` 只从 4,064,057 降至 4,050,775，减少 13,282；
  no-history active tile 从 199 增至 13,481，占比从 `0.0048963451%` 增至
  `0.3316966254%`。相比排序减少的 325,849 个 BF16-containing tile，这个
  对称 history 跳过机会约小 24.53 倍，且第 14 chunk 为 0；不应进入源码候选。
- coverage 小型证据目录
  `formal_topk_sort_history_coverage_cpu_v1` 含 raw/validated summary、validation
  和两份脚本共 5 项，加 manifest 共 6 文件、59,291 bytes；5/5 复算通过，
  manifest SHA256 为
  `e8a134219161667e59f0b5f4f5157d7545dca8193e0322cc1a89fedd7a57c5d4`。
- 2.66 草稿初写的 PyTorch `2.7.1+cu126` 与 validated summary 不符；正式证据
  为 Python/PyTorch `3.12.13/2.11.0+cu129`、elapsed
  `828.5257903169841 s`。已在提交前修正，不能把旧轮次环境版本串入本轮。
- 报告 2.66 修改前已顺序扫描 4,572 行，SHA256 前后保持
  `7ae0c4cc3a567d1e856420b7ce9ee088ca65fe3635afcfb3621c953518e5952f`；
  修改后 4,661 行、SHA256
  `65562c9a90a652ef77092fe8d85f8a4c3e48853661424ca178d71b381c71ba61`。
  章节 1.1–1.5/2.1–2.66 连续，validated aggregate 与报告关键计数逐项一致。
- 2.66 报告/planning 已由主仓库提交
  `21d60b75295a6439606cad65a436b0a3470531dc` 推送；生产 runtime/source 仍未
  修改，`has_history` 候选正式停留在 CPU coverage 淘汰状态。
- v3 排序 trace 的逐 chunk kernel 中位数求和显示，stage1 之后最大的可见项为
  `_rotate_latent_kernel=3,391.069 ms`，从 chunk1 到 chunk16 基本恒定
  `209.712→212.105 ms`；随后是固定 MoE Marlin `1,985.182 ms`、NCCL
  `1,113.248 ms`、随上下文增长的 BF16 GEMM `915.544 ms` 和 FP8 MQA
  accumulate `882.180 ms`。排序前后 rotate 仅差 `+0.461 ms`，说明它是与
  selected 排序无关、覆盖全部 16 chunks 的稳定成本，值得作为下一只读候选。
- 排名脚本按每个 kernel 的 8-rank chunk 中位数再跨 16 chunks 求和，适合候选
  排序但不等同于整段 rank 中位数；正式报告前需保留这一统计边界。下一步定位
  `_rotate_latent_kernel` 的调用点、调用数与缓存生命周期，不直接假设可消除。
- `_rotate_latent_kernel` 位于 `triton_oscar_mla_store.py`，执行
  `latent @ rotation`，输入显式转 FP32、accumulator FP32、dot 使用
  `input_precision="ieee"`，输出 FP32；固定 block `16×64×32`、4 warps、
  2 stages。公开 wrapper `oscar_mla_rotate` 支持复用 caller 提供的 FP32 output，
  因而 3.39 s 不是反复分配 output 可直接解释的成本。
- 该 rotation 处于 recent→INT2 demotion/store 链路，已有 backend scratch 复用；
  每个 layer 都持有独立 rotation，不能跨 layer 复用旋转结果。更可能的候选是
  kernel 数值/tiling 优化，而不是缓存整块结果；任何 BF16/TF32 改动都涉及精度，
  需先核对 demotion 调用几何与既有精度门限，不能直接实现。
- Backend prefill 对当前 chunk 的 `current_history` 逐层调用
  `oscar_mla_rotate_quantize_store`，所以每层 rotation 不同且每 chunk 都有实际
  新 history rows；这不是可跨 chunk 消除的重复计算。trace 中 rotate 每块稳定
  约 210 ms 与该固定 78 层写入链一致。
- 现有 store CUDA correctness 以 PyTorch FP32 matmul 为 reference，但 rotated
  容差为 `atol=0.35, rtol=0.02`，随后还会做 INT2 clip/量化。当前 kernel 把
  FP32 输入送入 IEEE dot；因此 TF32 是一个语义范围更小的筛选候选，但只能先
  通过独立 IEEE-vs-TF32 rotation/INT2 精度与单层几何性能门禁，不能直接改生产。
- 主仓库没有现成 rotation 性能工具；已有 top-k benchmark 提供了统一的
  parse/warmup/CUDA Event/JSON 模式，prefill benchmark 已能从固定路径
  `/opt/oscar_artifacts/rotation_fit_v2/rotations.pt` 加载真实层 rotation。
  下一步最小新增独立 rotation benchmark，而不是复用包含 stage1 大量无关输入的
  prefill 工具；工具阶段需先 TDD/CPU 回归并实时报告，再申请 GPU 筛选。
- rotation 工具红灯已固定为缺失目标文件的 `FileNotFoundError`；新测试覆盖
  2,048×512 默认几何、四个代表层、生产 kernel 参数、样本统计、rotation
  allclose 通过/拒绝和 speedup 算术。尚无实现文件，未触及 GPU/runtime。
- 新工具 `benchmark_oscar_rotation.py` 已实现独立 benchmark-local kernel，
  block/warps/stages 与生产完全相同，只以 constexpr 切换 IEEE/TF32。accuracy
  使用真实 artifact 的层 0/25/51/77、合成 BF16 latent，并要求本地 IEEE 与
  生产输出 `atol=rtol=0`；TF32 同时检查 rotation 与实际 INT2 quantize/dequantize
  后输出。CPU 定向 compile+7/7 passed，仅证明工具辅助逻辑，不证明 CUDA 候选。
- 工具最终 CPU 门禁为 Ruff 0.14.0 check/format、compile、diff 全绿；新增静态
  契约确认生产 block 16×64×32、4 warps、2 stages 与 IEEE precision 仍存在，
  benchmark 源码恰有一个 IEEE 和一个 TF32 分支。五个 Phase 9 unittest 文件
  合计 42/42 passed；argparse stderr 均为既有/新增负向测试的预期输出。
- 2.67 修改前报告顺序扫描 4,661 行且 SHA256 前后保持
  `65562c9a90a652ef77092fe8d85f8a4c3e48853661424ca178d71b381c71ba61`；
  追加后为 4,729 行、SHA256
  `ecc8e9558ac342256a9e6807c4707d76a4fb1d9d05f578f9b5cce04c4d24ed64`，
  章节、术语、工具边界、测试数据和文件身份全部通过。
- 首次补记 2.67 planning 时把前一版报告 hash 后半段误拼成更早版本；提交前
  与 `sha256sum` 对账发现并修正为 `65562c9a…ba61`，报告正文未受影响。
- rotation 工具、测试、2.67 与 planning 已由主仓库提交
  `25077d00bee732251c96f3b63a005016357aff66` 推送；源码 submodule 仍固定
  ca4a404e9 且 clean/published。GPU 筛选前置发布门禁已满足。
- rotation 筛选双空闲检查 `23:32:39Z/23:33:46Z` 间隔 67 秒，两次 8 卡
  全为 0 MiB/0%、无 compute process；外部下载容器 DeviceRequests=null。
  可固定选择 GPU 0，但启动前仍需即时复查。
- TF32 筛选在四个 accuracy layer 中某层的 INT2-restored 对比失败：128 个值不满足
  `atol=0.35, rtol=0.02`，最大绝对误差 `1.6203639507293701`。从 traceback
  位置可确认本地 IEEE=生产和 TF32 rotation 两个前序门禁已通过，但工具未在
  异常中携带层号或在此前输出数值，也未进入任何 timing；不能报告层号或 CUDA
  speedup。
- 该结果直接否定“只把 production rotation dot 从 IEEE 改为 TF32”的候选。
  原因不是 rotation 输出门限本身，而是随后的 INT2 clip/quantization 边界将
  差异放大；生产源码保持 ca4a404e9，不应为性能牺牲此精度门禁。
- 失败证据目录 `formal_rotation_tf32_screen_failure_v1` 含 exit code、结构化
  failure、run identity 和原始 stderr 共 4 项，加 manifest 共 5 文件、
  3,126 bytes；4/4 复算通过，manifest SHA256
  `f558b23603692be50dccca13bc83fe9aa257500c3faf969776e06dd3ff203712`。
- 2.68 修改前报告顺序扫描 4,729 行且 SHA256 前后保持
  `ecc8e9558ac342256a9e6807c4707d76a4fb1d9d05f578f9b5cce04c4d24ed64`；
  修改后 4,790 行、SHA256
  `d5746f3352dad6b47122a780020875134565abec79af7542282a570d9fcc1ba3`。
  章节、术语、失败数据、未知层号/未计时边界和证据身份全部通过。
- 2.68 与 planning 已由提交 `603e098731b2a60c23a082b65d1d13b1043998a3`
  推送。下一候选只允许改变 M/N tile 与 warps，保持 IEEE、block K=32、循环
  顺序和 FP32 accumulator；这样避免重复触发已证实的 TF32 精度失败。
- IEEE sweep 红灯 3 errors 精确覆盖 mode/config/best selection；候选矩阵固定为
  production `m16_n64_w4` 加 m16/n64/w8、m32/n64/w4/w8、m16/n128/w4/w8，
  全部 block K=32、2 stages。该矩阵只改变输出 tile/线程分工，不改变 IEEE
  或 K 维累加块顺序。
- IEEE sweep 实现保留 production baseline 为首项，并对每个 passed config 的
  四层合成输入要求 `atol=rtol=0`；只有逐值一致才进入 20 warmup、7×20 timing。
  candidate 的 compile/runtime exception 会写入单项结果；baseline 失败则整轮
  失败。定向 compile+10/10 passed，只证明控制流与算术辅助逻辑。
- 最终 IEEE sweep 工具 CPU 门禁：Ruff 0.14.0 check/format、compile、diff
  passed，五个 Phase 9 unittest 文件合计 44/44 passed。仍没有任何新 CUDA
  timing；六配置的实际可编译性与性能必须在固定 GPU 上验证。
- 2.69 修改前报告 4,790 行/SHA256 `d5746f33…1ba3`，修改后 4,850 行/
  SHA256 `fde8196f…5e5e`；章节、配置、IEEE/K 顺序/bitwise 边界、测试与文件
  hash 全部通过。
- 2.69 工具阶段已由提交 `ad7595fbb3772f389940a50c87c9b07028f015a4`
  推送；两仓待复核 clean/published 后可重新进行 GPU 空闲检查。
- IEEE sweep 双空闲检查 `23:44:19Z/23:45:28Z` 间隔 69 秒，8/8 卡两次
  全空闲，外部容器无 GPU 请求。可继续固定 GPU 0。
- IEEE sweep `20260731T2346Z_rotation_ieee_sweep_v1` exit=0，六配置全部在
  真实 rotation 层 0/25/51/77 上与 production bitwise 一致，每层比较
  1,048,576 个值且最大绝对/相对误差均为 0。production baseline
  `m16_n64_w4=0.1016319990158081 ms`；最佳 `m32_n64_w4=
  0.09359359741210938 ms`，CUDA 中位改善 `7.909321553783877%`、speedup
  `1.0858862339514979×`。wall 中位改善 `7.806143719193925%`。
- 该 sweep 固定输入为 2,048×512，只足以证明这一几何下候选有利。源码检索已
  显示 `_rotate_latent_kernel` 同时服务 current-history store 与 query
  rotation；后者的扁平行数可能是 `seq_len×本 rank heads`，因此在确认真实
  调用数和更大行数性能前，不能改 production 或把微基准收益外推到 3.39 s trace。
- 宿主查看结果时尝试使用 `jq`，但环境返回 `jq: command not found`；实验
  result JSON 本身已完整落盘，后续改用 Python 标准库读取，不重复该失败命令。
- IEEE sweep 小型证据目录 `formal_rotation_ieee_sweep_v1` 含 4 项数据加
  manifest，共 5 文件、18,988 bytes；4/4 manifest 复算通过，manifest
  SHA256 为 `295d3f64d1ada2fdca902963569b7b06aa08cdf3a159d3a0e3cf939c9d62e7fb`。
- 查阅上一失败证据模板时误读不存在的 `run_identity.json`，实际文件名为
  `run_identity.txt`；已改用实际文件，未修改旧证据，后续不重复该路径。
- 2.70 修改前已顺序读取报告全部 4,850 行/262,525 bytes；读取前后 SHA256
  均为 `fde8196fa41dbd3872359316a0315e40da0d9dfa0b2317824da4015312aa5e5e`，
  未发现并发手改。章节 1.1–1.5/2.1–2.69 连续，无交叉引用；`三池=0`，大写
  `A800` 的两次文本匹配都位于第 5 行同一个历史文件名/链接中，符合既定例外。
- 2.70 追加后报告为 4,924 行/266,960 bytes，SHA256
  `4f155caebc6e452b10935b9390c49e8fb51ce8515d71014d98ca903e1dd0d0b0`。
  1.1–1.5/2.1–2.70、全部六配置数值、best、result/manifest hash、无交叉引用和
  术语检查均通过。结论严格限定为 2,048-row 单 kernel 候选，不代表端到端收益。
- 2.70 报告与 planning 已由主仓库提交
  `c06d45486fa298b85c1a31dad0e4f113bcc174b4` 推送；主仓库 HEAD=origin，源码
  `ca4a404e9` 亦为 clean/published，可进入只读真实几何分析。
- 排序 v3 trace 的 aggregate 显示 `_rotate_latent_kernel` 每 rank 恰有 4,898
  calls；rank 0 chunk1 为 233 calls。算术 `233+15×311=4,898` 指向首块 233、
  后续块 311 的稳定调用结构，明显不是仅 78 次 current-history store。
- 源码至少存在三类同名 rotation：store wrapper 中 `oscar_mla_rotate(latent,
  rotation)`、decode 前的 `flat_query=num_queries×num_heads` 正向 rotation、decode
  后的 `flat_history=num_queries×num_heads` 乘 `rotation.T` 的逆向 rotation。
  因而 2,048-row 工具尚未覆盖所有调用形状，当前不能修改 production block M。
- 首次检索 backend 使用了旧路径 `vllm/v1/attention/backends/triton_mla_sparse.py`
  并得到 `No such file or directory`；实际路径在 `backends/mla/` 子目录，后续改用
  `vllm/v1/attention/backends/mla/triton_mla_sparse.py`，不重复旧路径。
- sorted 与 unsorted trace 的 8 ranks×16 chunks 调用数完全一致：chunk1 每 rank
  233，chunk2–16 每 rank 311，总计 4,898。sorted 每 rank rotation 时间范围
  `3387.745537–3399.351272 ms`，unsorted 为 `3387.972586–3394.243401 ms`；这
  进一步证明 top-k 排序不改变 rotation 工作量。
- Backend 明确先对 `demotion_positions` 调 `oscar_mla_demote_recent`，再在 prefill
  对 `current_history` 调 `oscar_mla_rotate_quantize_store`。chunk1 尚无旧 recent
  可 demote，后续每块新增该一组调用，正好解释 233→311 的 +78；其余 233 次由
  query 正向、history 逆向和 current-history store 组成。
- 固定池大小已由正式配置再次确认 prefix=64、recent=256；模型配置/正式配置
  指向 64 attention heads、TP=8，因此每 rank 预计 8 heads。但真实 rotation 行数
  仍应优先从 raw trace grid/launch 元数据验证，不把配置推导冒充 trace 实测。
- 正式模型 `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001/config.json`
  实际给出 `num_hidden_layers=78`、`num_attention_heads=64`、`kv_lora_rank=512`；
  结合 TP=8 与 decode 的 `num_queries*num_heads` reshape，32K chunk 中 query/inverse
  rotation 的源码几何为 `2048×8=16384` rows、rank 512。
- 每张 raw trace 压缩后约 147–155 MB；系统 Python 没有 ijson，直接 import 得到
  `ModuleNotFoundError`。既有 analysis summary 记录生成环境 Python 3.12.13、
  ijson 3.4.0.post0，且 `/dev/shm` 仍有对应 uv/venv 内容；后续复用，不污染仓库。
- 两个既有 analysis venv 都存在 site-packages，但 `bin/python` 均为指向
  `/usr/bin/python3.12` 的 broken symlink，直接执行得到 `No such file or
  directory`。这与此前持久 venv 不可直接复用的限制一致；可尝试固定 artifact
  rootfs Python 3.12.13 + 既有 site-packages，不重新创建或下载环境。
- Artifact rootfs Python 在宿主实际因缺 GLIBC_2.32–2.35 无法执行。第一次
  `docker run image /opt/.../python` 未覆盖镜像 Entrypoint，命令被错误解释并得到
  `cannot execute binary file`；这不表示容器内 Python 二进制损坏。后续显式用
  `--entrypoint /bin/bash -lc` 复核，避免重复启动错误。
- 显式 entrypoint 在固定 ca4a 镜像内成功复用 Python 3.12.13 与 ijson
  3.4.0.post0/yajl2_c。第一次流式解析在找到首个 `_rotate_latent_kernel` 后，仅
  因 `Decimal` 不能被标准 `json.dumps` 序列化而退出；trace 读取本身正常，改用
  `default=str` 即可，无需换解析器或重跑 GPU。
- 修正输出后，rank0 raw trace 首个 rotation launch 为 grid864、block128、
  66.912 us；production grid 公式反解为 `864/8×16=1728 rows`，与首块
  `2048-prefix64-recent256=1728` 完全一致。
- 随后两类 grid8192 launch 均反解为 16,384 rows：正向 rotation 为 reg48、
  shared11264、约572.1 us；传入 `rotation.T` 的逆向 rotation 为 reg64、
  shared10240、约2,075.3 us。逆向单次约为正向的 3.63×，表明非连续转置矩阵
  访问才是 rotation 的主要成本，2,048-row 正向 sweep 并未命中该瓶颈。
- Raw trace 有 32 个内外嵌套的同名 execute annotation；既有 v3 summary 使用
  每对第一个内层 span。按 `spans[::2]` 重算得到逐 chunk calls/time 与 summary
  完全一致：首块 233/209.684620 ms，其余每块 311、约212.03–212.07 ms，合计
  4,898/3,390.564987 ms。因此不能对32个 span 全部求和，否则会重复计数。
- rank0 的有效 signature 精确分解为：256-row demotion 1,170 calls/
  28.359872 ms（0.836435%）；1,728-row 首块 store 78/5.213370 ms
  （0.153761%）；1,792-row后续 store 1,170/85.427658 ms（2.519570%）；
  16,384-row contiguous 正向 1,248/714.301325 ms（21.067324%）；16,384-row
  transposed-stride 逆向 1,232/2,557.262762 ms（75.422910%）。
- 因此真实优化优先级发生变化：`m32_n64_w4` 的 2,048-row 正向收益最多只触及
  次要路径；主候选应首先解决传入 `rotation.T` 时的非连续 stride 访问，同时再
  验证 16,384-row 正向的 tile 选择。潜在做法包括专用 transpose-aware load 或
  预先保存 contiguous transpose，但必须先用 bitwise 门禁与显存/加载生命周期
  评估，不能直接选方案。
- 最小 benchmark 方案确定为一个 `trace-layout` mode、五个 case：forward
  m16 baseline/m32 candidate；inverse strided-T m16 production baseline、
  contiguous-T m16、contiguous-T m32。这样能把“布局收益”和“block M 收益”
  分离，并以同一 16,384×512 实际主几何计时。
- 预存每层 FP32 contiguous transpose 的静态额外显存可精确计算为
  `78×512×512×4=81,788,928 bytes=78 MiB/GPU`。这是候选代价，不代表已经
  接受；GPU 筛选仍需先证明 inverse 实测收益且四层逐值一致。
- trace-layout CPU 红灯为 13 tests 中 3 errors：argparse 拒绝新 mode，缺少
  `build_trace_layout_cases`，缺少 `contiguous_inverse_storage_bytes`；其余 10
  项通过。红灯精确覆盖新增接口，没有误把 import/环境失败当作测试失败。
- 最小实现对 accuracy layer 同时生成 production forward=`latent@R` 与 inverse=
  `latent@R.T` 参考；contiguous inverse 仅改变矩阵物理布局，m32 仅改变 M tile，
  两者都保持 IEEE、block K及累加顺序。计时按 forward/inverse 各自 baseline
  比较，避免把不同数学方向直接互比。
- 首轮绿灯命令在只读 `/workspace` 下由 py_compile 尝试写本地 `__pycache__`，以
  `Errno 30` 在测试前退出；`PYTHONDONTWRITEBYTECODE=1` 不阻止 py_compile 的
  显式写入。后续必须使用 `PYTHONPYCACHEPREFIX=/tmp/...`。
- 使用任务专用 `/tmp/oscar-trace-layout-pycache` 后，工具/测试 compile 与新增
  13/13 unittest 全绿。当前 diff 仅新增 trace-layout 分支、纯 helper 和3项测试；
  没有生产源码、配置、模型或 artifact 改动。
- 固定 Ruff 0.14.0 check/format 与 diff 全绿；两文件已符合 formatter，无机械
  变更。广回归范围仍是 `test_analyze_prefill_trace`、`test_benchmark_oscar_prefill`、
  `test_benchmark_topk_prefill_sort`、`test_benchmark_oscar_rotation`、
  `test_phase9_tools` 五文件。
- 五文件 CPU-only 广回归最终 `47/47 passed`，compile、Ruff 0.14.0
  check/format、diff 及 trace-layout 静态契约全绿。工具/测试现为1,078/229行，
  SHA256 `2fab0327e279b6bbcd9fbe40ff2d704e25400408778d940f5dd292d86af0bb36`/
  `012ad5be597491a4e657fc56795a3269876742355f2bdc28103f6173f4b093a0`。
- Production store/decode 文件 SHA256 仍为 `ec82245e…1b8e`/`23b08ffa…70c`，
  source submodule clean/published，证明本阶段只是测量工具化，没有提前落地候选。
- 2.71 修改前报告已顺序读取全部4,924行，SHA256 前后保持 `4f155cae…d0b0`，
  无并发手改；修改后为5,022行/273,297 bytes，SHA256 `a6668f26…252e3`。
  章节1.1–1.5/2.1–2.71连续，无交叉引用，trace数字、五case、测试、文件hash、
  `三池=0`和大写`A800`仅历史链接门禁全部通过。
- trace-layout 工具、测试、2.71 与 planning 已由提交
  `6a7d9f3d96568391ca6bd9a55401d0c1fb8ff8ca` 推送；主仓库 HEAD=origin，源码
  `ca4a404e9` 亦为 clean/published，满足 GPU 筛选前置发布要求。
- trace-layout GPU检查在`00:10:42Z/00:11:50Z`执行，间隔68秒；两次8卡
  全部0 MiB/0%、无compute process，外部下载容器无GPU请求。可固定选择GPU 0，
  但启动前仍需即时复查。
- 自检发现本轮 progress 的若干中间步骤曾使用估算分钟，其中后半段晚于系统实际
  时间；已统一改为不声明精确时刻的“本轮恢复后补记”。正式报告未使用这些估算
  时间；GPU检查仍保留命令实际输出的`00:10:42Z/00:11:50Z`。
- 启动前尝试用 `find` 和 `rg --files artifacts` 全量枚举 rotation artifact，两次
  均因 artifacts 树过大在约10秒无输出；未分配GPU。后续只检索已知小型文本文件
  中的固定 bind path，避免重复广域扫描。
- 精确核对发现宿主 phase2 `rotations.pt` SHA256=`0a966da2…08e`，而固定控制
  镜像内 `/opt/oscar_artifacts/rotation_fit_v2/rotations.pt` 为 `256ee5e4…235d`；
  2.70 GPU result 使用后者。本轮必须继续使用镜像内 artifact，不能误挂载旧版。
- 文本检索时把不存在的 `Dockerfile*` glob 传给 rg，得到一次 `No such file or
  directory`；其余固定路径结果有效，后续不再传该 glob。
- trace-layout 五case全部GPU通过：四个真实层、每层8,388,608值，所有case的
  mismatch/max abs/max relative均为0。forward m32相对m16 CUDA中位改善
  28.75216713281048%；inverse只把transpose物理连续化即改善73.14080106824098%，
  contiguous m32总改善76.37387550467753%、speedup 4.232602770708252×。
- inverse contiguous m16 的0.573798418045044 ms接近raw trace forward中位
  0.572354 ms，直接验证此前2.075 ms主要来自非连续stride，而非逆向数学本身。
  m32在正/逆连续布局都约0.5046–0.5047 ms，说明真实16,384-row也稳定受益。
- run后首个nvidia-smi采样GPU0显存已0 MiB、无compute process，但利用率仍14%；
  9秒后的`00:16:05Z`复查8/8卡均0 MiB/0%、无compute process，属于退出采样滞后。
- trace-layout证据目录含12项数据加manifest，共13文件/35,067 bytes，12/12
  manifest复算通过；result SHA256=`a54502cfc9fc4445e2beeb56a7dfd70af192a25ed013c667017a52637aa5db66`，
  manifest SHA256=`5af7db4f3aee8b281c24cade8e120c494412194ca43ec3b5a822b9675d7294ea`。
- forward m16七样本前五项约0.708 ms，后两项为0.676710/0.574003 ms，显示固定
  顺序下存在GPU频率/状态漂移；forward m32随后稳定约0.5046 ms。因此本轮
  -28.752%不能单独作为最终m32生产收益，需要反序/交错或端到端验证。相比之下，
  inverse strided七样本均约2.1362–2.1366 ms，contiguous m16均约0.5737–0.5738
  ms，73.141%的布局收益对该漂移不敏感，足以推进“只预存连续逆矩阵”的候选。
- 2.72修改前全文复读5,022行/273,297 bytes，SHA256前后保持`a6668f26…252e3`，
  无并发手改，章节/引用/术语正常。用inverse m16单kernel实测比率乘rank0 trace
  仅作估算：inverse约降`1870.402470 ms`，rotation总量约降55.164920%；不能当作
  新TTFT结果。
- 78 MiB占单张80 GiB显存约0.09521484375%；比例虽小，production阶段仍必须
  通过模型加载/显存容量门禁，不能仅凭比例假设没有容量影响。
- 2.72追加后报告为5,115行/278,942 bytes，SHA256=`41db2e05…58ce`；章节
  1.1–1.5/2.1–2.72连续，无交叉引用，五case与result逐项对账、12/12 manifest、
  forward漂移边界、trace投影边界、术语和diff均通过。
- 2.72已由主仓库提交`971f0c4`推送；production源码仍固定ca4a404e9。基于稳定
  73.141%布局收益，下一候选只预存/传递contiguous inverse并保持block M=16，
  把m32留待独立验证，避免把forward顺序漂移混入首个production变更。
- 最小源码触点为三处：`MLAAttention`注册/迁移inverse buffer；
  `triton_mla_sparse` backend把layer inverse传入；`triton_oscar_mla_decode`在逆向
  `oscar_mla_rotate`处使用显式inverse。store/demotion仍只用forward rotation，
  production kernel block M/N/K与IEEE完全不改。
- 为兼容现有直接调用，public decode/prefill可接受keyword-only
  `inverse_rotation=None`并fallback到`rotation.T`；但production backend必须显式
  传`layer._oscar_inverse_rotation`，对应静态/运行时测试需锁定，避免性能回退。
- 固定候选正式Python不包含pytest；旧`oscar-stage9-precommit-venv`也没有pytest，
  `oscar-glm-stage9-opt-test-venv`的`/usr/bin/python3.12`链接已失效。已定位此前
  验收过的只读`/dev/shm/oscar-glm-stage9-pytest-py312` target，下一轮将其注入
  固定镜像正式Python，避免改变production依赖。
- 有效源码红灯为3 failed：backend测试因kwargs缺`inverse_rotation`失败，decode和
  prefill两项接口测试均因signature缺该keyword失败；说明测试准确命中尚未实现的
  调用链，而非依赖或collection故障。只读挂载导致的pytest cache warning不影响
  三个目标断言。
- 当前CPU容器未声明`CUDA_VISIBLE_DEVICES`时，`vllm.triton_utils.importing`发现
  0个active drivers会把`HAS_TRITON`置False，`@triton.jit`因而退化为普通function；
  两项既有`.fn`源码断言必然失败。代码明确允许`CUDA_VISIBLE_DEVICES=""`的分布式
  初始化场景保留真实Triton导入，因此CPU-only decode回归需用该既有协议。
- 按空CUDA可见集协议，decode除interpreter外为7 passed/19 skipped/1 deselected、
  exit0；interpreter smoke在保留容器中为1 passed、exit0且OOM=false。结合runtime
  两文件24/24 passed，本轮CPU相关覆盖合计32 passed/19 skipped；CUDA数值用例仍需
  在授权GPU上另跑，不能由skip替代。
- 首轮pre-commit暴露的mypy新旧动态buffer错误可用两条显式Tensor属性声明一并
  消除；定向mypy随后通过。SPDX自动补头是触及旧测试后的必要合规改动；自动生成的
  attention backend文档属于既有漂移，已机械还原。torch.cuda第380行由初始提交
  `53d8be94f`引入且不在本次diff，最终只跳过该既有hook和会重写无关文档的hook，
  其余pre-commit全绿。
- contiguous inverse源码已由`67a0e47ff72f10a322de17b81c4134984e017bd6`
  （tree`60d5e606ce522dd78fecd890509372b727802f43`）推送。最终源码diff只有六个文件、
  60 insertions/4 deletions；production block M/N/K、IEEE dot和store/demotion均未改。
- 报告2.73已实时记录源码候选、3-failure TDD、32 passed/19 CUDA skipped、
  78 MiB/GPU静态推导及尚无GPU/端到端结果的边界。门禁后报告为5,189行、
  SHA256=`b1f331e1fa35644fe67ee8fc6af1e9c6db1186cefaab0fc740fbe4a83269dda7`；
  章节、引用、术语、源码身份/hash和数值均通过。
- Phase 6配置候选只改2个文件：输入manifest的output tag/commit/tree/Dockerfile
  hash，以及Dockerfile的默认commit/tree。固定ca4a镜像CPU-only JSON、compile与
  PAX确定性unittest 1/1 passed；source commit/tree、Dockerfile hash和diff身份
  门禁通过，Stage 9仍未迁移。
- OCI v1实际build/verification均通过。候选image/config=`22c2539e…9c66`、
  manifest=`f700ee72…a537`、layer=`37e119e5…a2b2`、diff-ID=`a11fef0c…91e7`；
  33层，candidate layer 109,147,697 bytes/5,298 members，无native/whiteout。
  verifier确认4,744源码、7 native、4 artifact与base 32层精确匹配。
- OCI v2独立重建同样built/passed；`index.json` SHA256=`d4407892…9290`且与v1
  逐字节相同，四项不可变内容digest完全一致。v2两份report SHA256为
  `7935c471…c7bc`/`df517137…2597`；report本身因路径和main commit不同不应逐字节相同。
- v1 OCI已通过skopeo1.4.1导入daemon；目标此前不存在，工具容器exit0并自动删除。
  daemon ID/33层/末层diff-ID与OCI一致，source/tree/Dockerfile/rotation/runtime
  expectation六项label全匹配，inspect SHA256=`8724ac2d…d969`。
- Stage 9控制Dockerfile当前候选只改base一行到`glm52-oscar-a800-phase6-67a0e47ff-
  0275043c:latest`，新SHA256=`65f1ed38599af68d7c836a1676225d56897eb26c585c49585630fbc6cc9b79fc`；
  performance matrix/wrapper/overlay仍是ca4a，按阶段尚未迁移。
- 新控制镜像`oscar-glm-stage9-runtime:67a0e47ff` ID=`2d0e9f1e…6f74`，34层，
  前33层与base一致、末层`126c2fb0…1343`；继承labels/entrypoint全匹配。
  CPU runtime import验证显式inverse接口和buffer源码存在，cuda_initialized=false。
- contiguous inverse候选的driver-injected runtime import固定GPU 0且一次通过：
  双空闲检查间隔135秒，启动/退出后均8卡0 MiB/0%、无compute process；正式
  Python/PyTorch/Triton、候选vLLM Python/原生扩展、78层rotation及三项artifact
  hash均匹配，`cuda_initialized=false`。有效JSON与冻结ca4a协议逐字节一致，
  SHA256=`0910b598…7b7a`；本轮没有加载模型或执行性能测试。
- contiguous inverse正式overlay已机械派生：候选层与overlay各4,749个普通文件，
  递归清单逐字节一致且SHA256=`797e7c2e…ee83`；6个native symlink及目标hash
  清单与上一正式链路逐字节一致。Phase 1/5/7/9和9个wrapper已完成身份替换，
  旧ca4a身份在configs/scripts中为0；工具测试和递归verifier尚待执行。
- 静态工具结果为输入12/12、shell 9/9、Phase 7 20/20、Phase 9 47/47、compile
  15/15。首轮递归verifier的候选身份检查全部通过，仅Phase 1/5汇总因NFS
  source mode漂移失败；既有只读named volume含4,711 tracked文件+6 native且mode
  正确，下一轮只补该正式mount namespace适配。
- 补正式phase0 source只读volume后，递归verifier实际为66/66 passed；本轮代码
  新增检查使总数高于旧候选的64。最终25份静态证据共114,644 bytes，summary/
  recursive/manifest SHA256为`ce56900f…673c`/`5e652f0c…c449`/
  `3e4caaa4…c5ba`；全程未分配GPU。
- contiguous inverse driver-injected preflight已退出0；静态66/66、固定环境
  import与服务参数解析全绿，后两处`cuda_initialized=false`。实际参数为TP8、
  max model len 131072、max seqs16、2048 batched tokens、OSCAR INT2、torch profiler；
  未加载模型。10份证据41,458 bytes，GPU退出后空闲。
- contiguous inverse正式32K/batch1三轮稳定：mean TTFT相对极差仅0.019362%，
  mean TPOT相对极差1.593508%，无preemption/waiting/capacity limit。中位mean
  TTFT/TPOT为30539.197439/202.514363 ms，三轮峰值80757 MiB/GPU。
- 与2.62上一版OSCAR相比，contiguous inverse使TTFT减少1910.047128 ms
  （-5.886261%），请求吞吐提高2.607780%，但TPOT回退1.686771%；因此证据支持
  “TTFT明确改善”，不支持“全面胜出”。相对BF16，TTFT仍回退143.767039%，
  TPOT回退13.242965%；TPOT通过+20%门限而TTFT未通过。
- 本轮profiler的8张CUDA表、8份worker trace和1份frontend trace全部通过；
  critical rank=6、kernel total=63359 ms。停止profiler后约1.2 GiB trace的CPU
  解析/压缩持续约10分钟，GPU利用率为0但worker高CPU；最终正常退出，无OOM、
  CUDA error或超时。这类后处理不能误判为GPU计算卡死。
- 小型正式证据封存41项加manifest，共42文件/1,103,416 bytes，41/41复算通过；
  原始worker trace共1,213,749,498 bytes，仅保留于`/dev/shm`供下一阶段CPU归因。
- 同一format-v3 analyzer重析2.62参考与contiguous inverse trace后，prefill
  wall/kernel分别下降1908.716/1927.360 ms；rotation从3390.942降至
  1486.435 ms，减少1904.507 ms、解释wall改善99.779%。去掉rotation后residual
  wall只下降4.209 ms，说明2.83收益几乎全部来自contiguous inverse。
- 8/8 rank的wall/kernel/rotation均下降；16/16 chunk的rotation均减少约119 ms。
  profile wall下降解释端到端正式TTFT下降99.930%，多口径一致，不能把收益解释为
  单rank、调度或频率偶然漂移。
- rotation优化后，stage1仍为19846.588 ms、占prefill wall 64.921%；相对BF16
  prefill wall差距20,483.782 ms中，stage1相对原生attention的超额16,462.214 ms
  解释80.367%。下一候选仍应聚焦全部16 chunk的grouped prefill stage1。
- trace归因小型证据18项加manifest，共19文件/8,087,709 bytes，18/18复算通过；
  两组原始worker trace共2,411,923,332 bytes，仅保留于`/dev/shm`。
- 2.84报告与trace归因已由主仓库提交
  `db9e02721b6eb72e068bb26644ed7e3f720cb0e5`通过HTTPS推送；当前主仓库HEAD与
  远端分支一致、工作树clean。正式结论已发布，但TTFT仍未通过BF16 +20%门限，
  下一候选需聚焦stage1而不是已经降至1.486秒的rotation。
- grouped prefill当前仅在`num_splits=1`时启用：grid为query×head-group，stage1
  同时维护`bf16_acc`与`history_acc`两个`block_h×block_d` FP32累加器，并用一套
  在线softmax状态归一化；随后`_merge_mixed_splits_kernel`按同一LSE权重分别合并
  两类输出。若拆成独立cache-type kernel，必须额外保留各自LSE并做严格的
  log-sum-exp合并，不能只把两个输出相加。
- 现有`has_bf16`只跳过history-only tile里的两次BF16 dot；history dot和history
  value dot仍对所有tile发射，仅通过mask把非history列归零。排序后的selected token
  形成prefix/history/recent三段，真正专用路径可能减少无效dot与同时存活累加器，
  但会增加kernel launch、重复读取query/rope以及改变浮点归约顺序，需先量化资源与
  tile构成，不能直接视为收益。
- 既有2.66覆盖率已直接淘汰简单`has_history`：排序后4,064,256个active tile中
  no-history仅13,481个（0.331697%），第14 chunk甚至为0。相反full-history为
  3,931,284个，mixed为118,140个；因此新候选若有价值，必须消除history主路径的
  资源/计算成本，而不是只跳过极少数无history tile。
- 既有runtime资源基线已足够明确：当前排序grouped stage1为h8/t16/w8、1 stage、
  109,568-byte dynamic shared、242 registers/thread、0-byte stack；苹果800双block
  shared阈值为83,456 bytes，寄存器也远高于256-thread block双驻留所需的约
  128 registers/thread。拆分候选的CPU-only离线门禁应同时检查shared和register，
  只降低其中一项不足以证明能双驻留。
- cache-split SM80离线v4实测15/15编译成功且production baseline精确复现
  109,568-byte shared；离线cuobjdump基线为255 registers/thread、0 stack。
  专用history h8/w8降至84,992-byte shared、199 registers、0 stack，但仍比
  83,456-byte双block shared线高1,536 bytes，且寄存器也不允许两个256-thread
  block。说明移除BF16 accumulator确实降低资源，但不足以直接获得双驻留。
- history h4/w4为76,288-byte shared且寄存器算术允许双block，但产生
  176-byte/thread stack；h2/w4与h1/w4也分别有184/176-byte stack，因此没有任何
  history几何通过“shared+register+零stack”严格门禁。BF16 h8/w4为42,496-byte
  shared、255 registers、0 stack并通过严格门禁。组合结论为：cache split的纯
  shared/register双block可行，但严格promotion=false，不能直接进入正式GPU候选。
- 2.85已把上述离线结果实时写入报告：旧内容未改，报告仅追加94行，现为
  5,856行/323,194 bytes、SHA256=`3e46599c…9123`。章节连续到2.85，“三池”为0，
  大写`A800`仅保留第5行历史文件链接；summary/validation/manifest完整哈希与
  38/38证据均复核通过。固定67a控制镜像中的compile与13/13 unittest也再次通过。
- 2.85阶段已由主仓库`615a95f73f720d1b0fe7f9ea8527a63c23a31788`发布，
  HEAD与远端一致；源码仍为`67a0e47ff` clean/published。当前没有可直接晋升的
  cache-split候选，下一步只能先解决history h4/w4的stack spill或寻找同时满足
  shared/register/零stack的新几何。
- history value reload离线v3的5组配对结果全部逐字节等价：h8/w8、h4/w8、
  h4/w4、h2/w4、h1/w4的base/reload cubin SHA256及shared/register/stack均相同，
  资源delta全为0。Triton对人为重复的history value load/dequantize做了公共子表达式
  消除，因此该源码写法没有缩短编译器可见的live range，也没有消除h4/w4的
  176-byte stack spill；不能进入GPU或production。
- 最终v3为20/20 compiled、15/15 unittest、`cuda_initialized=false`；47/47
  evidence通过，summary/manifest SHA256为`f48652e5…baa6`/`3f9b7d77…b415`。
  后续若继续降低history资源，必须引入编译器可见的阶段边界或不同计算结构，不能
  继续用同一kernel内的等价reload表达式。
- 2.86已如实记录reload淘汰结果，报告仅追加83行，现为5,939行/328,559 bytes、
  SHA256=`146afe09ca8c62e90e9fa982e6d9f1b52fab5ed616f30d3bb4d68d9a773c9237`。
  章节、术语、47/47证据与固定容器15/15回归均通过；本阶段没有新增性能或精度值。
- 2.86阶段已由主仓库提交`77f4a2dadd49cb21d3424e5954df7d58dbe3d155`发布；
  主/源码仓库与各自远端一致。reload方向已关闭，不能再把同一kernel内等价重载
  当作新的资源候选。
- standalone history的h4/h2/h1、t8、w4在SM80离线编译中全部于history value
  `tl.dot`处被拒绝，错误一致要求`K >= 16`。这与2.32对mixed kernel的t8淘汰一致，
  说明cache-type拆分没有解除当前dot结构的最小K约束；三个候选没有cubin或资源
  数字，不能进入GPU或宣称资源收益。
- t8最终轮次format v4共23 variants、20 compiled/3 rejected，耗时
  18.343898101秒、`cuda_initialized=false`。小型证据50文件/163,039 bytes，
  49/49 manifest通过；简单token tile搜索已闭合，下一候选必须改变value计算结构
  或形成编译器可见阶段边界。
- 2.87已实时记录t8淘汰结果；报告现为6,020行/333,310 bytes、SHA256
  `de97bea51ecd1051b28c42834bb939f3c5803c1a49465dcbdbb0e5b85987fac5`。章节、引用、
  术语、49/49证据与固定容器16/16回归均通过；本阶段没有新增性能或精度测量。
- 2.87阶段已由主仓库提交`d96faa69ec11c9bac0fb4cc4f3892af5916f9515`通过HTTPS
  推送。standalone history t8方向已关闭；下一阶段只考虑不同value计算结构或
  编译器可见边界。
- 手工value归约是与2.87不同的计算结构：保持t8加载/score/softmax不变，把
  `probabilities[h,t] @ history_values.T[t,d]`改写为三维elementwise乘积后沿t归约。
  它可绕开Triton dot最小K，但可能显著增加中间量和spill；必须先用同口径SM80离线
  资源数据筛选，不能预设会更快。
- 手工value归约最终format v5轮次实际为26 variants、23 compiled/3 rejected、
  21.680984777秒、`cuda_initialized=false`。h4/h2/h1 t8/w4分别为
  26,112/21,760/19,584-byte shared、215/190/168 registers/thread、0 stack，三项
  均通过离线严格资源算术；既有三项简单dot t8仍因`K >= 16`拒绝。组合静态门禁首次
  为true，但这不等于实际双block驻留、数值正确或更快：手工归约改变浮点归约顺序，
  t8使循环次数相对t16翻倍，h4/h2/h1还会把每个query的program数增至2/4/8。
- 本轮小型证据为60个文件，其中manifest记录59项且59/59复算通过；按所有普通文件
  大小求和为177,221 bytes（此前恢复摘要中的177,282不是当前文件系统实测值，报告
  采用177,221）。summary/run log/manifest SHA256分别为`89c557bc…041f`/
  `f81ea88f…d42a`/`1f43363e…710b`。没有封存cubin本体，cubin哈希仅记录于JSON。
- 现有`benchmark_oscar_prefill.py`已具备32K最终位置输入、冻结容差、warmup、CUDA
  event计时和10分钟心跳，但其入口只调用production完整prefill，不能直接选择离线
  standalone history variant。最小筛选应新建独立脚本复用输入常量与误差统计，并
  直接启动离线工具的`_history_prefill_stage1`；这样可以先隔离验证手工value结构，
  避免在正确性/速度未知时修改production源码。
- 首轮CUDA筛选固定h8/t16/w8 dot为reference，h4/t8/w4 manual为候选；shape使用
  batch1、final_seq_len=32768、query_tokens=2048、topk=2048和8个本地heads，模拟
  正式32K最后一个prefill chunk的history主路径。结果只代表standalone history
  kernel，不可直接等同完整stage1或端到端TTFT。
- 2.89已实时记录上述筛选入口及其synthetic边界；报告只追加56行，现为6,172行/
  343,616 bytes、SHA256=`70714f1b…9910`。章节1.1–1.5/2.1–2.89、交叉引用、术语、
  Ruff 0.14.0、固定67a容器compile、22/22 unittest和diff均通过。尚无correctness或
  CUDA时间；发布和GPU双空闲检查之前不能把静态资源候选写成性能收益。
- 单卡实际结果证明静态资源门禁不足以预测速度。h4/t8/w4 manual相对冻结
  h8/t16/w8 dot reference的output/LSE allclose通过，但median CUDA由
  `20.436993 ms`增至`38.589439 ms`，慢`88.821516%`。t8循环翻倍与h4程序数
  增加是与回退方向一致的已知风险，但本轮没有逐项profile，不能精确拆分原因；
  该结论仅适用于synthetic standalone history，不等同完整stage1，但已足以按
  预设门禁淘汰该候选。
- 现有离线矩阵缺少“manual+t16”的交叉项：普通h4/h2/h1 t16/w4有
  176/184/176-byte stack，manual h4/h2/h1 t8/w4为零stack，但GPU上t8 h4显著
  变慢。下一项最小CPU-only问题是把manual结构与t16结合，先判断能否在不翻倍token
  循环的前提下降低spill；资源不通过就直接关闭，不应先申请GPU。
- manual+t16交叉项已由实际CPU-only SM80轮次关闭：h4/h2/h1 shared虽降为
  43,520/39,168/36,992 bytes，但255 registers/thread与576/104/96-byte stack使
  三项strict均为false。尤其h4 stack相对普通dot从176增至576 bytes/thread，说明
  恢复t16后manual三维中间量重新造成spill；不能只凭shared下降选择候选。
- 当前history专用dot矩阵已经说明单kernel tile搜索的两难：w8的h8/h4/h2/h1均零
  stack，但registers/thread为199/206/199/206，256-thread block无法双驻留；w4的
  h8/h4/h2/h1虽减少threads，却产生192/176/184/176-byte stack。下一结构必须改变
  live range，而不是继续在同一表达式上排列h/t/w。
- 可验证的新假设是物化score并拆成score、LSE、value三段。32K末块的standalone
  shape为2,048 query×8 heads×2,048 top-k，FP32 score scratch精确为
  134,217,728 bytes（128 MiB）/TP rank，可复用但会增加显存读写与kernel launch。
  该成本可能抵消占用率收益，必须先以SM80资源和后续同口径CUDA实测裁决。
- 三阶段v2证明阶段边界确实显著缩短live range，但冻结的全段双block门禁尚未通过。
  相对history h8/t16/w8参考84,992-byte shared/199 registers/0 stack，score h8为
  52,224/164/0，score h4为43,520/162/0；shared已允许双block，唯一失败项是w8
  registers。LSE为16/17/0，value h2/h1 d128为8,320/112/0与8,256/114/0，后两段
  均严格通过。下一最小问题是score同几何改w4能否在零stack下利用128-thread block
  放宽到256 registers/thread；但必须先完成v2报告与发布。
- 2.92已按实时记录规范落盘。报告现为6,441行/360,484 bytes、SHA256
  `524caca08717b14533b5722e5c82b799f434a80b9656a7a69830569d52757642`；章节、术语、
  18/18证据清单、19文件/70,064 bytes、结构化资源字段、固定镜像26/26回归与diff
  全部通过。本阶段没有新的correctness、CUDA时间、TTFT/TPOT或GSM8K数据，正式性能
  仍沿用2.83；下一步必须先发布，随后才可CPU-only补测score-w4。
- 2.92及其工具/测试已由主仓库提交`03d1b78`通过HTTPS推送；三段式v2的证据与
  “score门禁未通过”结论已经形成远端可恢复检查点。
- score-w4实际离线结果区分了两个候选：h8/w4虽然资源算术允许双block，但有
  40-byte/thread stack；h4/w4为43,520-byte shared、254 registers/thread、0 stack，
  首次成为score strict candidate。三段组合静态门禁因此为true，但约136.06 MiB
  scratch、额外launch/读写与浮点分段误差仍未实测，不能直接进入production。
- 2.93已实时记录score-w4结果。报告现为6,536行/366,358 bytes、SHA256
  `1088dfefa64436cc53d697c6082c72741e0d2ea195682880f172c1711398f09b`；章节、术语、
  22/22证据、结构化字段、固定镜像27/27回归与diff全部通过。下一阶段应先建立
  standalone三段式correctness/总CUDA时间门禁，不能直接改production。
- 三段式benchmark已把candidate三个launch放在同一CUDA event内，避免只报告某个
  子kernel时间；reference/candidate共享冻结输入，correctness失败会在计时前退出。
  静态报告2.94现为6,597行/370,679 bytes、SHA256=`d939b1f8…ce40`，32/32回归通过。
- 2.94静态入口已由主仓库提交`35708d0`通过HTTPS推送；GPU筛选现在具备已发布、
  可恢复的代码与协议检查点。
- 三段式实际单卡结果证明资源门禁仍不足以预测性能。数值门禁通过，但
  53.320705 ms相对20.418560 ms reference慢161.138422%；约136.06 MiB scratch和
  分段launch/读写的实际总成本远超当前single-kernel。该结构按冻结门禁关闭。
- 2.95已实时记录三段式负结果。报告现为6,678行/375,914 bytes、SHA256
  `e06cfecc0b2b345b0dd0ef1a8802556524a112000cc6dd44b40ece439aedfaf1`；章节、术语、
  12/12证据与result字段均通过，正式2.83端到端性能没有被微基准改写。
- 固定Triton 3.6的`CUDAOptions`源码与签名确认存在`maxnreg: Optional[int]`，其
  语义是生成PTX `.maxnreg`、限制每线程32-bit寄存器上限。它可能以spill换取驻留，
  只能先用离线cubin资源再用实际CUDA裁决，不能预设比当前h8/w8更快。
- `.maxnreg`实际生效但spill单调恶化：cap 128/120/112/96对应stack
  192/232/256/360 bytes/thread。最温和的128也比h4/w4既有176-byte stack更高，
  且h4相对h8仍有program数翻倍风险；CPU资源证据不支持直接晋升GPU。
- 2.96已实时记录maxnreg结果。报告现为6,773行/382,043 bytes、SHA256
  `b3f6f0a89fd5d963869b9190a71373bf0beca12c4c0a5a407ab104dcbb29f9f3`；章节、术语、
  20/20证据、34/34回归与diff全部通过，没有新增正式性能或精度值。
- 当前production的grouped prefill只在`group_prefill_heads=true`且`num_splits=1`
  时选择；任何`num_splits>1`都会切回逐head的通用stage1。历史2,048×2,048实测中
  grouped split1已显著优于通用split16，因此不能把直接split2当作保持head复用的
  小改动。若研究grouped split2，需要新内核与额外partial buffer/merge，且循环次数
  不改变编译期累加器资源；当前优先级低于直接实测maxnreg128的净效应。
- maxnreg128虽然产生192-byte/thread stack spill，但也把h4/w8每block寄存器从
  52,736降至32,768，使shared与register两项算术都允许双block。离线资源无法判断
  双驻留收益是否覆盖spill；最小可证伪实验必须同时保留h4无cap控制，避免把h4与
  register cap两个变量混在一起。晋升条件固定为correctness通过且cap128同时快于
  h4无cap和h8参考。
- 2.97已冻结上述三项CUDA裁决协议并实时落盘，尚未申请GPU。报告为6,831行/
  386,150 bytes、SHA256=`20c8059a5c62fe66ae49190ad45f56c22b24d62dd0164c251183ac161276c99e`；
  37/37相关回归、Ruff、章节、术语、引用与diff均通过。正式2.83性能数据未变化。
- 单卡实测区分了h4与cap的影响：h4无cap相对h8从`20.432896`回退到
  `40.378368 ms`；同一h4只加maxnreg128后降至`30.505983 ms`，相对h4控制加速
  `1.323621268×`。双驻留收益确实覆盖了部分spill成本，但candidate相对h8仍慢
  `49.298385602%`，按预设双控制门禁淘汰。不能把“cap有方向性收益”误写成最终
  性能晋升，也没有依据把剩余差距精确归因到某一硬件counter。
- 2.98已实时记录该负结果；报告为6,904行/390,770 bytes、SHA256
  `f8a8c2c4486e2ad0d81120bb78f69da58a76d72ba3196a4f21697a66e60bb15c`，章节、术语、
  11/11证据与result字段通过。正式2.83端到端性能与GSM8K精度均未改变。
- 固定几何的`group_size=128`、latent rank 512，因此每token只有128个唯一2-bit
  packed byte和4组scale/zero。当前源码以512个dim地址表达packed/group索引；现存
  h8/t16/w8 PTX仍含165条静态`ld.global`，LLVM循环体明确展开多组data、scale与
  zero load，并未把所有重复地址化为一次load后广播。可证伪候选是显式按唯一byte/
  group加载再用Triton broadcast/reshape恢复512×16矩阵；必须先检查编译与资源，
  不能仅按逻辑字节数宣称带宽收益。
- compact-load离线候选形成不同cubin，并把PTX静态`ld.global`从165降至71（-94，
  -56.969697%），cubin从135,856降至106,800 bytes；shared/stack保持84,992/0，
  registers/thread从199增到230。它满足“不同二进制、少load、无新增spill”的预设
  CPU门禁，但两项都因shared只能单block；静态指令数不能替代动态transaction或
  CUDA时间，后续必须用同h8几何的standalone correctness/性能实测裁决。
- 2.99已实时记录CPU-only结果；报告为6,996行/396,736 bytes、SHA256
  `20c33b8d052096e818767b0abd9e83027a1f0601033579d2cab8571ad4a173a7`，19/19证据、
  39/39相关回归、章节、术语和diff均通过。正式2.83性能与GSM8K精度未变化。
- 2.100已冻结同h8几何的standalone CUDA协议：reference/candidate仅compact标志
  不同，先做output/LSE冻结allclose，再以5个交替repeat的CUDA中位数严格更小为
  晋升条件。静态入口42/42回归通过，报告为7,055行/400,497 bytes、SHA256
  `8629f2e00e7f1fd6e4e989c469b230ec6ff73022581c15be2eeb09ce92ea5dd0`；尚无GPU结果。
- compact-load单卡实测通过：output/LSE与h8参考的max_abs/max_rel均为0，CUDA
  中位数从`20.431871`降至`16.819201 ms`（-17.681546762%，1.214794448×）。同h8
  几何隔离了program数差异，证明唯一load后广播的净收益覆盖了registers 199→230
  的代价；但该数据只适用于synthetic standalone all-history，不能直接外推正式TTFT。
- 2.101已实时记录通过结果；报告为7,130行/405,166 bytes、SHA256
  `5b295d22d584d5e86e1f32706ca2f15ce94cacfb0528514a3981e9eeef5430d9`，11/11证据、
  章节、术语、引用和result字段通过。候选下一步可进入production集成与更完整门禁。
- production `_mixed_sparse_prefill_stage1` 的launch继续传入编译期常量
  `latent_rank`/`block_d`，当前GLM几何为512/512；因此最小集成可在kernel内用
  `if latent_rank == block_d`编译期分支启用compact表达式。非满宽几何保留原
  `dim_mask`、`byte_offsets=dims//4`与`groups=dims//group_size`路径，可避免compact
  reshape把padding维当作有效值。源码现有CPU测试已使用`inspect.getsource`冻结
  grouped-kernel结构，适合先增加同类静态TDD，再由interpreter/CUDA覆盖数值语义。
- 现有`compile_oscar_prefill_cache_split.py`的首项`mixed_h8_t16_w8`直接编译导入的
  production `_mixed_sparse_prefill_stage1`，不是standalone副本；因此无需新增工具，
  即可在空CUDA可见集、SM80 target下验证本次production分支能生成cubin并读取真实
  shared/register/stack。summary中的源码SHA可把未提交worktree与旧67a镜像区分开。
- production满宽候选的CPU-only SM80实际编译成功：PTX静态`ld.global`由旧正式
  245降至167，cubin由206,640降至187,056 bytes，shared仍109,568 bytes、registers
  仍255/thread，但stack由0增至136 bytes/thread。减少load已落到真实mixed cubin，
  同时新增spill说明standalone胜出不能保证production胜出，后续GPU门禁不可省略。
  `latent_rank=384, block_d=512`的非满宽fallback也单独编译成功，保持245个load、
  109,568-byte shared、255 registers/thread、0 stack，证明编译期else确实保留旧路径。
- 2.102已实时记录production集成与CPU-only结果；报告为7,218行/410,708 bytes、
  SHA256 `206dd0ecad8961420736f0df0af4362c2d64b73ae9565f7b2aa160fb1f1b31ad`。
  章节1.1–1.5/2.1–2.102连续，“三池”为0，大写`A800`仅在第5行历史链接；证据
  12/12、13文件/558,070 bytes、六项关键哈希、交叉引用与diff均通过。下一步只
  发布主仓gitlink、报告与planning；发布完成前不构建候选或申请GPU。
- Phase 6新输入只切换候选tag、source commit/tree与Dockerfile默认身份；base
  manifest、rotation、runtime expectation、native extension contract与构建/PAX
  逻辑不变。Dockerfile/config SHA256分别为`17ef020a…bb69`/`086505cb…8f78`。
  固定67a控制镜像、断网、4 CPUs、无GPU下静态门禁通过，但尚未产生任何OCI digest、
  runtime import、CUDA correctness或TTFT/TPOT结果。
- 2.103已实时记录输入迁移；报告现为7,259行/413,208 bytes、SHA256
  `2938630ce724b735e7a052c4330a2935d18c6a77c8d6fc6727b477e513e4005f`。章节
  1.1–1.5/2.1–2.103连续，“三池”为0，大写`A800`仅在第5行历史链接，2.83/
  2.99/2.100/2.101/2.102/2.103引用、配置身份与diff均通过。下一步先发布配置、
  报告与planning，再开始两个独立CPU-only OCI构建。
- 首次宿主构建因旧Python缺少`datetime.UTC`在OCI生成前退出；改用冻结的
  Python 3.12.13控制容器后，v1 build=`built`、verification=`passed`。
  image/config为`08d8ea6f…360f`、manifest为`320e011e…9006`、candidate layer
  为`c8f6d007…11fe`；4,744个源码文件、4份rotation、7个基础层原生扩展与33层
  身份均通过。该结果尚未证明双构建确定性，也没有GPU或端到端性能含义。
- 2.104已实时记录v1结果；报告现为7,307行/416,016 bytes、SHA256
  `ec169f873a41f22486b7567e84779ad90c38ef97920591b16089845cc0581527`。章节
  1.1–1.5/2.1–2.104连续，术语、2.83/2.99–2.104引用、三份结果哈希与diff
  均通过；下一步只发布本检查点，随后才执行v2独立重建。
- v2独立CPU-only重建同为built/passed；两轮index/config/manifest/candidate
  layer已直接逐字节比较为相同。共同image/config=`08d8ea6f…360f`、manifest=
  `320e011e…9006`、layer=`c8f6d007…11fe`、diff-ID=`c2c7cd6f…37f3`。
  v1/v2报告哈希因发布提交和输出路径不同，不影响OCI内容确定性结论。
- 2.105已实时记录双构建确定性结果；报告现为7,358行/418,864 bytes、SHA256
  `db9d98a2de05d033a1c7ba59503ae2c60847159f5e2a7a6084dc3d5d02b4ea82`。章节
  1.1–1.5/2.1–2.105连续，术语、2.83/2.99–2.105引用、OCI哈希与diff均通过。
- v1已由skopeo 1.4.1成功导入daemon；独立审计确认image ID精确等于
  `08d8ea6f…360f`，33层、最后diff-ID、tag和8项关键labels全部匹配。该阶段
  未注入NVIDIA runtime，也没有runtime import或任何端到端性能含义。
- 2.106已实时记录daemon结果；报告现为7,392行/420,852 bytes、SHA256
  `3142092a2bacb33f97c82929ffaba119fa60f257e5acfefd146d203a682b46db`。章节
  1.1–1.5/2.1–2.106连续，术语、2.83/2.99–2.106引用、四份daemon证据哈希与
  diff均通过。
- driver-injected runtime import前双空闲检查间隔73秒；有效探针一次通过，固定
  版本、候选vLLM Python/`_C`、78层rotation、三项artifact与max reasoning均匹配，
  结束时`cuda_initialized=false`。退出后8卡全空闲；尚无模型或端到端性能结果。
- 2.107已实时记录runtime import；报告现为7,431行/423,247 bytes、SHA256
  `664acf7a689b6cb6b7329f8ec307ae708b21074767389080fc658e6b21d66ef3`。章节
  1.1–1.5/2.1–2.107连续，术语、2.83/2.99–2.107引用、五份runtime证据哈希与
  diff均通过。
- Stage 9控制Dockerfile的默认base已完成唯一一行切换：从旧67a tag到
  `glm52-oscar-a800-phase6-c0bcbbbdf-0275043c:latest`；新文件SHA256为
  `99932fd21937a449d86f3520230da2679b5b5ce56325ee2d6e0dfa2ad251fd10`。
  daemon base实测image ID=`08d8ea6f…360f`、33层，revision/tree/layer均与
  2.104–2.107验收身份一致；目标control tag当前不存在，尚未开始构建。
- 2.108已实时记录控制镜像输入切换；报告现为7,465行/425,132 bytes、SHA256
  `360dc600fbcd05aa6c01a149897a510ebe48f1056e1c9466298c420f997ed5bc`。章节
  1.1–1.5/2.1–2.108连续，术语、2.83/2.99–2.108引用、Dockerfile身份与diff
  均通过；下一步只发布入口，随后才构建控制镜像。
- 历史Stage 9 control有效协议是：Dockerfile在candidate的33层之上安装固定
  git/iproute2并保留`/bin/bash` entrypoint；审计要求control 34层、前33层与base
  逐层相同、inherited labels和entrypoint匹配，再在无NVIDIA runtime容器中确认
  git/iproute2/Python/glibc及`cuda_initialized=false`。本轮复用该协议，不增加
  与性能目标无关的断言。
- c0bc control有效空context构建已完成，image ID=`b4785123…5088`、34层；前33层
  与`08d8ea6f…360f` base逐层一致，labels/entrypoint/source身份审计通过。首轮
  runtime命令遗漏`-i`只产生空JSON，v2加`-i`后固定git/iproute2/Python/glibc和
  `cuda_initialized=false`全部通过。证据21文件/42,031 bytes，manifest内20项
  20/20复算通过。
- 2.109已实时记录control结果；报告现为7,517行/428,121 bytes、SHA256
  `22ca3b114c9deb34b29e7ffb27393dbbcde5f06d981bbdafb6147bef1b9c6ea0`。章节
  1.1–1.5/2.1–2.109连续，术语、2.83/2.99–2.109引用、20/20证据与diff均通过。
- 当前67a正式迁移范围由提交`4dddc09`确认：4份配置、9个wrapper/测试入口以及
  overlay路径/commit/tree/OCI/control身份；不能只改Phase9矩阵。新overlay应从
  c0bc v1的4,749个普通文件机械派生，并复用当前正式overlay的6个phase0 native
  绝对符号链接，随后按Phase1→5→7→9依赖顺序重算配置SHA。
- c0bc overlay v2审计通过：candidate/overlay均为4,749个非symlink普通文件，
  内容清单SHA同为`aca29fc5…5561`；6个native symlink清单SHA=`e01d7637…cebd`，
  路径/目标与当前正式overlay一致且目标全部存在，6个target hash保持历史冻结值。
  首版JSON因`Path.is_file()`跟随symlink失败，v2改用`lstat`后状态passed。
- 四级配置已按依赖顺序完成最小迁移并通过结构化身份/JSON/shell/diff初检；Phase1/
  5/7/9 SHA256依次为`c2d78292…ec9`/`885b7249…3af1`/`89429257…7e42`/
  `af2ff405…8b58`。历史67a commit/tree/OCI/control/overlay身份在明确迁移的4配置、
  9 wrapper和Phase9身份测试范围内均已清零；standalone旧候选工具不在迁移范围。
- Phase 7首次容器工具测试的19/20不是产品回归：冻结`.venv/bin/python`是自定义
  launcher，宿主实际解析到`/dev/shm/oscar-glm-recovery-tools/python/cpython-3.12.3-
  linux-x86_64-gnu/bin/python3.12`，宿主可从冻结venv导入`requests 2.34.2`；未挂载
  该绝对路径的control容器则在resume子进程报`ModuleNotFoundError: requests`。
  有效复跑必须将`/dev/shm/oscar-glm-recovery-tools`按相同绝对路径只读挂入容器。
- 该launcher同时硬编码冻结venv site-packages的宿主绝对项目路径，而测试容器此前
  仅有`/workspace`挂载。故完整环境修正需要两个只读同路径挂载：recovery-tools和
  `/nfs/AE/txc/oscar-glm`；这比修改launcher更忠实于已冻结的评测协议。
- b247/b87/a2fe三份历史有效Phase 7日志的pytest rootdir均为宿主绝对项目路径，
  且均20/20 passed；这直接验证了同路径挂载方案。当前复跑应复用该命名空间，
  不再使用只挂`/workspace`的首轮协议。
- 宿主现存可复现pytest依赖集为`/dev/shm/oscar-glm-stage9-pytest-py312`
  （pytest 8.3.5）；它与历史有效工具测试一致。当前失败轮显示pytest 8.4.2，需先
  查明该轮临时依赖路径，不能把依赖版本变化与挂载修复混在一起而不记录。
- control镜像保留`/bin/bash` entrypoint；直接把`python -m pytest`作为docker参数会
  让bash尝试解释Python ELF并exit 126。有效测试命令必须显式覆盖entrypoint为
  `/opt/fp8_speed_up_v4_venv/bin/python`（或通过`-lc`调用），该错误与测试本身无关。
- 补齐绝对路径挂载并覆盖entrypoint后，Phase 7四文件有效结果为20/20 passed、
  0 failed、3.67秒。故c0bc配置迁移没有引入Phase 7工具回归，首轮requests缺失
  与v2 exit126均属于已隔离并保留证据的执行环境错误。
- Phase 9全11文件首轮的6个collection错误共享同一环境原因：显式设置仅含
  `/pytest-packages`的`PYTHONPATH`使production工具从`/usr/local/lib/python3.12/
  dist-packages/vllm`导入，而不是候选overlay，故找不到
  `triton_oscar_mla_decode`。修复应恢复镜像候选路径并追加pytest target，不应
  为测试修改代码。
- control image的不可变环境明确设置`PYTHONPATH=/opt/vllm_glm52_v1`，且正式
  wrapper的overlay source也指向c0bc overlay内同路径。因此测试父进程的正确
  组合是`/pytest-packages:/opt/vllm_glm52_v1`，而不是猜测另一个vLLM安装目录。
- 修正搜索路径后，Phase 9当前全部11个测试文件共80/80 passed；这覆盖新增的
  compact-load、manual-value、maxnreg、score-pipeline和cache-split工具测试，
  比历史仅跑正式wrapper测试的21/22项范围更完整。唯一warning不影响断言结果。
- 正式candidate wrapper最终委托Phase 1 runner；`dry-run`并非纯静态动作，它在
  static verifier后还会执行需要driver的fixed import。静态迁移验收必须只调用
  相同`VERIFY_SCRIPT`的递归验证段，否则无driver退出码会掩盖已通过的64项结果。
- 递归首轮的OSCAR candidate层、6个native link目标哈希、rotation、历史baseline
  和Stage9参数均通过；失败只来自嵌套Phase5。历史证据表明NFS普通文件mode会与
  Git mode形成假失败，正式容器需用`oscar-glm-phase0-source-fd3e0b3`覆盖phase0
  source target。当前失败不支持“c0bc迁移内容不一致”的结论。
- 加入同目标只读命名volume后递归verifier一次通过64/64，验证此前失败确为NFS
  mode namespace问题。c0bc的Phase1→5→7→9配置、source tree、OCI blobs、
  native links、rotation、baseline artifacts、server/matrix参数均通过递归门禁。
- 当前宿主未安装`jq`；报告前证据复算改用Python JSON标准库。根目录出现一个未跟踪
  `overlay_validation_candidate_tmp`，名称与已记录的首轮错误相对输出一致，仍需
  检查内容后才能清理，避免误删用户文件。
- 当前递归verifier实际包含66项且66/66 passed；不能沿用历史“64/64”数字。
  有效static JSON/log逐字节一致，SHA256=`84a1884c…87b3`。23份迁移证据包含
  两类有效结果与所有fail-closed尝试，合计1,419,123 bytes，manifest SHA256=
  `db2ee261…9007`。根目录重复清单经逐字节确认后已删除，正式validation清单保留。
- 2.110已实时记录overlay、四级配置、20/20与80/80工具测试、compile exit0、
  66/66递归门禁及全部fail-closed边界；报告SHA256=`316a1f2f…7e52`，章节、术语、
  交叉引用、配置解析、shell语法、23项证据复算和diff均通过。该节未声称任何新GPU、
  TTFT/TPOT或精度结果。
- 发布前diff逐行审计确认所有代码/配置变更都能追溯到c0bc正式链路迁移；没有修改
  standalone历史候选工具、测试门限、负载或服务参数。旧67a身份在明确范围清零，
  其历史证据仍保留在报告和不属于正式wrapper的独立工具中。
- 静态迁移检查点已由主仓库`5922d733…364b`发布，包含报告2.110、四级配置和
  正式wrapper身份。GPU preflight必须从该已发布身份开始，不能使用提交前脏工作区。
- c0bc正式preflight在两次空闲检查间隔80秒后exit0。固定环境为Python3.12.13、
  PyTorch2.11.0+cu129、Triton3.6.0，候选Python/`_C`均来自新overlay；解析参数为
  TP8、PP1、max_model_len131072、max_batched_tokens2048、batch上限16、
  `oscar_mla_int2`、eager、关闭async。import与参数解析均未初始化CUDA。
- preflight证据9/9 manifest复算通过，目录总计10文件/42,301 bytes；容器退出后
  8卡仍0 MiB/0%、无compute process。该阶段只证明正式身份和启动参数可通过
  driver namespace解析，尚未运行production kernel或加载模型。
- 报告2.111已按阶段实时记录正式preflight，SHA256=`ba86943e…e399`；章节、术语、
  2.107–2.111引用、证据复算与diff均通过。报告明确保留“无模型/无kernel/无新性能
  或精度结果”的边界。
- 正式preflight报告检查点已由`c92d081…7d4b`发布；后续production CUDA门禁可从
  该clean/published身份开始，不与未发布报告或配置混跑。
- 既有production CUDA有效轮次统一使用单张GPU0和独立cold Triton cache，完整
  `tests/oscar_mla`不得用CPU skip替代；当前应复用此协议。历史计数随测试新增从
  125→126→127变化，因此c0bc必须报告当前实际收集/通过数，不能预填127。
- 当前c0bc control自身包含正式候选，但为了让测试路径和Git身份可审计，完整回归
  继续只读挂载当前clean/published源码仓，并用phase0命名volume解析6个native链接；
  `PYTHONPATH`应为冻结pytest target在前、c0bc源码在后，cache单独可写。该协议
  不向镜像持久安装依赖。
- 仅把GPU0注入容器不会自动启用源码中的CUDA测试；`tests/oscar_mla`由
  `VLLM_OSCAR_RUN_CUDA_TESTS=1`显式解锁。遗漏该变量会产生exit0但29项skip，必须
  fail-closed判为CPU-only无效轮次。有效v2必须同时满足0 skipped和新cold cache。
- c0bc production CUDA有效结果为130 passed、0 skipped/failed、77.84秒；当前
  源码比旧fd281有效轮次多3项，所以不能比较passed数本身作为性能变化。380个cold
  cache文件合计25,036,913 bytes，证明CUDA节点实际编译/运行，而非再次被skip。
- v1无效边界与v2有效证据均已持久化并通过各自9/9 manifest；有效v2退出后8卡
  0 MiB/0%、无compute process。c0bc现在具备进入正式32K端到端轮次的CUDA正确性
  前提，但尚无新的TTFT/TPOT数据。
- 报告2.112已同时记录v1假绿边界和v2有效CUDA结果，SHA256=`296026bd…666d`；
  章节、术语、2.107–2.112引用、18项证据、0 skipped与cold cache规模均复核通过。
  本检查点没有把pytest运行时当作端到端性能指标。
- c0bc production CUDA检查点已由`e6b5da0…ea69`发布；正式32K轮次必须绑定该
  发布状态，并继续使用冻结的batch1/output128/TP8负载，不能因优化候选改变口径。
- c0bc正式32K的三个round均成功，单轮median TTFT依次为
  32896.704087/32913.360903/32838.999102 ms，median TPOT依次为
  198.662841/198.619463/199.110020 ms；这些只是已落盘的单轮观察，正式比较仍须
  使用runner自然退出后生成的官方summary。`/stop_profile`后9份trace已在
  08:00:11Z–08:02:36Z落盘，8个worker继续各约73% CPU构建
  `profiler_out_0..7`，说明等待不是GPU请求卡死，也不能安全强杀。8表最终全部生成，
  但随后仓库不变门禁因本会话实时更新planning文件而exit1；没有cell/global summary，
  故上述round观察不能升级为正式结果。该错误揭示“长实验每10分钟外部汇报”不能通过
  修改受runner监控的仓库文件实现；下一轮只向会话commentary打印，仓库文档待退出后
  立即更新。
- 无效轮次的9份trace合计1,203,699,383 bytes，逐文件哈希9/9通过；NFS只封存
  trace manifest，不复制大文件。其余39个持久证据文件含8份rank表、三轮结果/
  validation、profile结果、GPU采样、完整formal log、exit1和失败说明；manifest
  覆盖38项且38/38复算通过。报告2.113明确保持2.83正式对比不变。
- c0bc有效正式32K结果明确否定“compact-load已经改善端到端TTFT”：mean TTFT
  32843.679183 ms，比2.83的67a正式值30539.197439 ms慢2304.481744 ms
  （+7.545980%）；TPOT反而改善7.243449 ms（-3.576758%），吞吐下降2.398523%。
  对BF16的TTFT差距扩大到+162.161651%，因此下一步必须先用新trace归因c0bc相对67a
  的prefill回退；不能直接保留compact-load作为端到端优化。
- 有效profile critical rank=7、kernel total=63,458 ms，服务无preemption/waiting/
  capacity limit，三轮mean TTFT相对极差仅0.012496%、mean TPOT 0.646064%；回退
  不是调度拥塞或测量噪声。正式小证据目录42文件、`du -sb` 1,107,564 bytes，
  manifest SHA=`9439dcc1…e172`且41/41复算通过；原始trace哈希由summary保存。
- 同一format-v3 analyzer、Python3.12.13/ijson3.4.0.post0下，c0bc prefill wall
  中位32869.711247 ms，67a为30570.251770 ms，增量2299.459477 ms，与正式TTFT
  增量2304.481744 ms相差仅5.022267 ms；profile解释99.782065%。8/8 rank回退
  2297.374612–2300.479774 ms，16/16 chunk回退81.342109–151.304456 ms。
- 算子级因果证据高度集中：`_mixed_sparse_prefill_stage1`从19846.587763增至
  22202.587112 ms，增加2355.999349 ms；调用数仍1248。rotation保持4898 calls且
  时间只变-0.025668 ms，top-k保持1344 calls且只变+0.230797 ms。c0bc相对67a的
  唯一生产diff正是在stage1为latent_rank==block_d加入broadcast/reshape compact-load，
  因此最小可验证修复是回退c0bc commit，而不是调整rotation或调度参数。
- trace归因目录共23文件/12,109,124 bytes，manifest覆盖22项且22/22复算通过，
  SHA=`ca2f71da…19ab`；有效candidate/reference/comparison/validation SHA依次为
  `9fae718e…235`/`359ef056…aeab`/`71950330…9957`/`4215d47c…1cf4`。
- revert提交c349e32e9不是手工近似恢复：整个source tree与67a的tree object同为
  `60d5e606ce522dd78fecd890509372b727802f43`，因此除Git历史外内容完全一致。
  CPU-only完整suite 100+29 skip与c0bc无效CPU轮101+29 skip的唯一区别就是删除
  `test_grouped_prefill_compacts_full_width_history_loads`；没有新增邻接改动。
- c349有效Phase 6 OCI不是复用67a daemon tag：其新不可变image/config ID为
  `sha256:2ff10a1f814088d333ba6cbec0ab6ba365757abf93c9cc1cd808ce4a3a22ebbe`，
  manifest/candidate layer/diff-ID分别为`sha256:dd16e997…e3f4`/
  `sha256:395efe0a…4a7e`/`sha256:2bf883f6…0450`。虽然源码tree与67a相同，新的
  Git revision label和created timestamp使OCI身份不同，必须按新身份迁移后续控制链。
- c349 extracted layer、有效overlay与67a extracted layer的4,749项普通文件内容清单
  逐字节相同，统一manifest SHA256为`f92b9755…24d9`；6项native symlink路径、绝对
  目标和目标哈希也与c0bc正式overlay逐字节相同。这证明production内容恢复到67a，
  同时保留c349独立可追溯身份。
- overlay hard-link失败是NFS/root所有权约束，不是候选内容失败：失败目录0个普通文件，
  有效普通复制后才达到4,749+6门禁。daemon import的首次失败同样是本地image ID引用
  语法错误，目标tag未创建；有效skopeo重试后image ID、33层、最后diff-ID、tag和8项
  labels全部通过，不能把两项入口错误混入有效候选结论。
- c349首轮`vllm._C` undefined symbol不是候选源码或native层漂移：该轮明确覆盖成系统
  `/usr/bin/python3.12`，而c349/c0bc冻结Env均选择`/opt/fp8_speed_up_v4_venv/bin/
  python3.12`；只修正解释器后同一OCI和GPU0探针一次通过。有效JSON与2.107 c0bc
  逐字节相同且`cuda_initialized=false`，因此运行时依赖门禁通过，但仍不能替代
  production CUDA kernel回归。
- c349控制镜像入口保持最小变更：Dockerfile只有默认base一行从c0bc切到c349，
  CPU-only门禁已把新tag绑定到`2ff10a1f…ebbe`、33层、c349 source/tree和
  `395efe0a…4a7e` candidate layer；目标control tag在构建前不存在，尚无控制镜像
  构建或运行时结果。
- c349控制镜像已形成独立ID`731412e9…1f95`；34层中的前33层、继承labels和
  entrypoint均与`2ff10a1f…ebbe` base匹配。CPU runtime JSON与c0bc控制镜像逐字节
  一致且`cuda_initialized=false`，说明控制层没有引入运行时漂移，但尚未完成正式
  overlay/config静态迁移或任何GPU kernel门禁。
- c349正式静态链路现已闭合：Phase 1/5/7/9配置按内容哈希逐级绑定，Phase 7与
  Phase 9候选tag、manifest/config/layer digest一致；9个wrapper只替换候选身份与
  overlay路径，冻结负载仍为32K/batch1/output128/TP8，未调整服务或评测参数。
- 递归verifier实际为66/66，而不是沿用旧阶段的64/64：新增计数来自当前脚本的真实
  checks列表。它验证4,744个Git源码文件、6个只读native链接、OCI三项digest、rotation、
  runtime expectation、冻结评测器与矩阵定义；这仍是CPU静态门禁，不代表CUDA kernel、
  端到端精度或TTFT/TPOT已有新结果。
- c349正式driver preflight没有暴露新的运行时漂移：Python/PyTorch/Triton仍为
  3.12.13/2.11.0+cu129/3.6.0，候选vLLM Python与`_C`均来自c349 overlay；TP8、
  PP1、131072上下文、2048 token预算、16序列、`oscar_mla_int2`、eager和关闭async
  均与冻结配置一致。import和参数解析的`cuda_initialized=false`只能证明启动前依赖，
  下一项仍必须是显式启用CUDA测试的cold-cache production回归。
- c349当前完整`tests/oscar_mla`实际为129项；比c0bc有效轮次130项少1项的原因不是
  collection丢失，而是revert删除了`test_grouped_prefill_compacts_full_width_history_loads`。
  因此production CUDA门禁应按129/129和0 skipped/failed验收，不能要求历史130。
- 有效GPU0 cold-cache回归已关闭preflight未覆盖的kernel正确性门禁：129/129通过，
  cache产生380文件/25,035,973 bytes。c349与c0bc同为380个cache文件但总字节少940，
  这里只记录实际编译产物差异，不把cache字节或pytest时长当作端到端性能结论。
- c349正式32K/batch1三轮与profiler均自然退出0；官方三轮中位mean TTFT/TPOT/
  请求吞吐为`30519.625429/197.273569 ms/0.018000469 req/s`。三轮均3/3完成、
  0失败，mean TTFT/TPOT相对极差为0.093622%/0.906946%，无preemption、waiting或
  capacity limit。
- 相对BF16，c349的TTFT/TPOT/吞吐为`+143.610813%/+10.312392%/-36.566483%`；
  TPOT仍在+20%门限内，但TTFT仍多17,991.599647 ms，当前性能目标尚未关闭。
- 相对c0bc，c349的TTFT减少2,324.053754 ms（-7.076107%）、吞吐提高3.760806%，
  TPOT回退1.025578%；compact-load造成的TTFT回归已经恢复。相对67a则为
  TTFT -19.572010 ms、TPOT -5.240795 ms、吞吐+1.272080%。c349与67a tree完全
  相同，后三项小差异只能作为正式轮次波动，不能归因为新生产源码收益。
- c349 profiler耗时738.390624秒，8表、8 worker trace和1 frontend trace通过；
  critical rank=4、kernel total=61,559 ms，worker trace共1,187,212,421 bytes。
  固定容器54/54正式验证与57/57小型manifest复算均通过；下一步用同一analyzer
  比较c349/67a/c0bc trace，不修改production源码或申请GPU。
- 同一format-v3 analyzer解析c349 8-rank trace为exit0，环境固定Python3.12.13/
  ijson3.4.0.post0；8/8 rank均144 contexts、16 chunks、32768 tokens，耗时
  116.939262秒。与67a相比，prefill wall/stage1只差+13.212148/+1.022254 ms，
  且正式TTFT差异方向相反；结合tree相同，确认只属轮次波动。
- c349相对c0bc的prefill wall/kernel/stage1分别恢复2286.247329/2311.598422/
  2354.977094 ms；profile wall解释正式TTFT恢复98.373255%，stage1解释wall恢复
  103.006226%。8/8 rank和16/16 chunk全部改善，rotation/top-k仅变化-0.020940/
  +0.314374 ms且调用结构不变，回归因果再次集中到compact-load stage1。
- trace阶段两组comparison各25/25、总validation35/35、manifest19/19通过；结束后
  8卡0 MiB/0%。c349已完成回归恢复，剩余约17.99秒BF16 TTFT差距仍应聚焦grouped
  prefill stage1，但下一步先CPU-only整理已淘汰方向，避免重复无效源码试验。
- CPU-only机会排序拒绝重复launch sweep、has-history gate、reload/manual/maxnreg
  和full compact。唯一保留的是把compact load拆成packed-only与scale/zero-only，
  用离线二进制/资源门禁识别能否避免full compact的+31 registers与production回归。
  门禁要求PTX load减少、stack不增且registers/thread<230；当前没有预测加速，也未
  修改production或申请GPU。
- partial compact离线工具TDD已最终14/14通过：format v9保留旧full-compact字段，
  新增packed-only/qparam-only两个互斥constexpr/variant，并将registers严格低于
  full compact纳入晋升条件。该结果只证明工具表达与门禁，不代表候选已编译或加速。
- CPU-only SM80实编译关闭两个partial方向：packed-only为157 loads/238 registers/
  0 stack，相对baseline少8 loads但比full compact的230 registers更高；qparam-only
  为197 loads/193 registers/0 stack，相对baseline反增32 loads。两者均不晋升GPU。
- 当前production源码仓由`.gitmodules`确认是`glm52_oscar_vllm`，不是旧假设路径；
  其HEAD/upstream均为`c349e32e…64b3`且worktree clean。主仓`42410fd…abdc`也已
  clean/published，2.128发布检查点闭合。
- c349 trace当前stage1中位累计`19847.6100175 ms`/1248 calls，约等于78层×16个
  prefill chunk，是30.583秒prefill wall的绝对主体；其余merge约95.49ms、add约
  69.47ms，不能解释约17.99秒BF16 TTFT差距。
- production grouped stage1每个program同时持有BF16/history两套512维FP32 accumulator，
  对每个16-token tile执行history score/value dot，并仅以`has_bf16`条件包围BF16 dot；
  最终分别写回、merge、逆旋转再相加。既有2.85–2.98已实际排除简单cache拆分、
  reload、t8/manual、三段式与maxnreg，不能把这些重新包装成新候选。
- 排序后4,064,256个active tiles中，full-history为3,931,284，含BF16的只有131,621；
  但production在计算`has_bf16`后仍于条件分支外构造prefix/recent loads与完整
  `bf16_values`，只把两次BF16 dot放进分支。把BF16 value物化也纳入动态门禁、并在
  score/value两处重载以缩短跨history路径live range，是尚未被history reload、
  cache split或compact实验覆盖的候选；需先离线证明二进制/资源确实变化。
- `ca4a404e9`原始diff确认它只新增`is_bf16/has_bf16`并分别包围score/value两次
  `tl.dot`，没有移动prefix/recent load；当前源码断言也只要求两个`if has_bf16`。
  因此BF16 lazy-load/reload不是对既有提交的重复实验。现有离线工具可精确复现
  production mixed baseline，但还没有candidate mixed kernel表达，下一步应先用
  CPU-only源码/编译TDD证明资源差异，再决定是否申请GPU。
- lazy-BF16源码TDD有效红灯为目标1 failed，首个prefix load确实早于首个gate；最小
  改写后定向2/2、完整decode CPU范围9 passed/19 CUDA skipped，固定Python compile
  通过。改写只把BF16 loads分别放入两个gate，history/softmax/FP32 accumulator未改；
  尚需Ruff与SM80实际资源结果，不能据此宣称加速。
- lazy-BF16首次CPU-only SM80编译实际导入的是镜像内
  `/opt/vllm_glm52_v1/vllm/__init__.py`，原因是固定venv的`_virtualenv._Finder`
  位于标准`PathFinder`之前并覆盖了`PYTHONPATH`。该轮cubin SHA256、shared、register、
  stack、PTX loads与c349 baseline完全相同只证明旧源码被重复编译，不能解释为编译器
  消除了候选。必须先移除该单个finder并验证`find_spec`及kernel模块文件均来自当前
  worktree，再以独立v2目录获得有效候选资源结果。
- 修正后的CPU-only import preflight自然退出0：固定Python/PyTorch/Triton为
  `3.12.13/2.11.0+cu129/3.6.0`，移除的唯一meta finder为`_virtualenv._Finder`，
  `vllm.__init__`与`triton_oscar_mla_decode`均解析到当前`glm52_oscar_vllm`工作树，
  且`cuda_initialized=false`。候选源码没有`vllm._version`，因此出现可解释的commit
  hash RuntimeWarning；不影响目标kernel源码身份，但有效编译仍需在结果中复核source path/hash。
- 有效v2 CPU-only SM80编译确认候选二进制变化：cubin SHA256从`19846644…643d8`
  变为`c45ee70f…4cd8c`，大小`206,640→232,368 bytes`；shared从`109,568`降到
  `93,184 bytes`，register保持255/thread，但stack从0增至136 bytes/thread，PTX
  global loads从245增至309。预设资源门禁因此`promotion=false`，候选在GPU前淘汰。
  validation 14/14、manifest 9/9通过；candidate patch已随证据封存，下一步撤销候选
  源码/测试改动并实时追加报告2.130。
- 报告2.130已实时追加并通过门禁：8,785行/504,989 bytes、SHA256
  `84f4e8d18404bd8d17e133783eb5f7e6ca8bc0f5ef2f85c48ddc492f9d9b76a7`；章节
  1.1–1.5/2.1–2.130连续，`三池`为0，两处大写`A800`仍仅位于第5行历史链接，
  14/14 validation与9/9 manifest字段/哈希复算通过。候选已撤销，源码仓clean且
  HEAD=upstream c349。
- 2.130发布后重新核对production mixed stage1：双512维accumulator不能在现有basis下
  直接相加。BF16 prefix/recent value处于原latent basis，history value处于rotation
  basis，后者必须在stage1/merge后乘inverse rotation再与BF16输出相加；因此简单合并
  accumulator会改变数学语义。若要统一accumulator，必须让两类cache预先处于同一basis
  或在kernel内增加高代价变换，不能作为无条件局部删减。
- 历史2.11已实际筛选过“原生BF16/RoPE走BF16 tensor core、history继续TF32”的
  hybrid；BF16 value probability降精度曾触发output门禁，恢复FP32 probability/
  TF32 value后才通过。当前报告、源码与现存phase9工具搜索未发现把INT2反量化后的
  history score输入单独转BF16的实际候选结果；该方向若经Git历史复核仍未覆盖，可先
  只筛history score precision，保持history value probability/accumulator为FP32/TF32。
- Git历史确认初始grouped提交`c3728be9f`曾把query/query_rotated/RoPE、BF16/history
  values及两条value probability一起降为BF16；1K单卡轮次output/LSE误差约
  `0.009153/0.002593`而被门禁拒绝，`35ab18464`随后整体恢复FP32 IEEE。这个反例不能
  证明“只把history score两个输入转BF16、history value仍保持FP32/TF32”必然失败，
  因为原轮次同时改变了5个dot及两条value归约。Git全历史未发现该隔离候选落地。
- 第三轮CPU-only机会排序已选择`history_score_bf16_inputs_only`。排序后
  4,064,256个active tiles中full-history 3,931,284、mixed 118,140，合计
  4,049,424个含history，占99.635062358277%；作用范围显著大于上一lazy-BF16方向。
  候选不预设加速，离线门禁额外要求PTX global loads也不得增加。ranking validation
  12/12、manifest 2/2通过，三文件SHA依次为`d16c468f…e9009`、
  `88c16a70…9e8a`、`5706cb55…18c`。
- 报告2.131已实时追加：8,838行/508,425 bytes、SHA256
  `65322761d6b0137424340dd3c0b2d65e16de778806e2259e9cce256dc537b9c3`；章节
  1.1–1.5/2.1–2.131连续，`三池`为0、历史`A800`例外仍仅第5行两处，ranking
  字段、12/12 validation与2/2 manifest均复算通过。
- history-score-BF16源码TDD红灯为目标1 failed，精确显示`query_rotated`仍为FP32。
  最小改动只把grouped prefill的`query_rotated` load和history score所用
  `history_values`转BF16，history value dot仍是FP32 inputs/TF32且双accumulator仍为
  FP32；定向结构测试与Triton interpreter smoke随后2/2通过。该CPU interpreter结果
  不能替代冻结2K苹果800 allclose，下一步仍先做完整CPU/Ruff/SM80资源门禁。
- 候选完整CPU静态门禁通过：固定c349容器的decode文件为9 passed/19 CUDA skipped，
  Ruff 0.14.0 check/format、固定Python3.12.13 py_compile与source repo diff check均通过。
  下一步沿用2.130已修正的meta finder移除+source origin/hash验证协议，只编译production
  mixed kernel并与c349资源逐项比较；尚未申请GPU。
- history-score-BF16 v1实际编译为83,968-byte shared、255 registers/thread、
  8-byte/thread stack、245 PTX loads、198,960-byte cubin；二进制变化、shared下降
  25,600、loads/register不增，但8-byte spill违反2.131门禁，promotion=false。
  v1把query_rotated在load后即转BF16；下一步测试等价的“FP32 load、仅dot入口cast”
  是否改变live range并消除spill，仍为CPU-only且使用独立v2证据。
- v2把query_rotated恢复为FP32 load，仅在history score dot内同时cast两个输入；BF16
  query保持原样。更新后的TDD红灯为1 failed、精确失败于v1仍为BF16 load；修正后结构
  测试与interpreter smoke 2/2通过。一次无变量上下文dtype patch曾误命中相邻query，
  已在任何编译前通过源码检查发现并修正，未形成实验结果。
- v2实际结果为83,968-byte shared、255 registers/thread、8-byte/thread stack、245
  PTX loads、199,088-byte cubin；除cubin字节/hash外，v1/v2资源与resource log逐项
  相同，证明cast放置不能消除spill。17/17 validation与10/10 manifest通过；v2
  candidate patch/summary/validation/manifest SHA依次为`04978939…f8f8b`、
  `b2b1019b…6422`、`a943e166…f7c3`、`0175da1e…b034`。两形态均按2.131门禁在GPU前
  淘汰；源码与测试已撤销并恢复clean c349。
- 报告2.132已实时追加并通过门禁：8,903行/512,331 bytes、SHA256
  `7f7617a9237f6b43fb398cd6ebbc28d75986dcbda8b0359d31ef10268872ecbd`；章节
  1.1–1.5/2.1–2.132连续，`三池`为0、历史`A800`例外仍仅第5行两处，17/17
  validation、10/10 manifest与报告字段复算通过。
- 2.132发布后搜索报告、planning、现存phase9工具和production源码，未发现延迟
  `bf16_acc * previous_scale`的既有实验。2.40明确要求has_bf16=false时仍逐tile缩放
  BF16 accumulator以保持online softmax；在排序后96.728257%的full-history tiles中，
  BF16 contribution为零但整块8×512缩放仍执行。可用逐head标量pending scale合并连续
  无BF16 tile的缩放，并在下一BF16 tile或循环结束时一次应用；需先证明代数、覆盖计数
  和未重复性，再做CPU-only资源门禁。
- 恢复后重读原始覆盖证据发现2.131的`4,049,424=all_history_tiles 3,931,284 +
  mixed_precision_tiles 118,140`是派生口径；同一validated summary另列
  `history_only_tiles=3,932,635`、`tiles_with_history=4,050,775`，两组相差1,351个
  边界tile。raw generic aggregate还错误累加了每chunk固定`tile_width`为256，后续
  validated summary已明确重建整数计数。下一步必须读取`summarize_selected_tiles`
  实现，确认active/partial tile与`has_bf16=false`的精确对应关系；确认前不把任一派生
  数写成新候选的作用范围。
- `summarize_selected_tiles`实现已消除口径歧义：`history_only_tiles =
  tiles_with_history & ~tiles_with_bf16`，精确表示active且`has_bf16=false`；因此新候选
  的适用计数是3,932,635/4,064,256。`all_history_tiles = valid_tiles.all &
  ~tiles_with_bf16`只少算1,351个部分填充的active history-only tile；
  `tiles_without_bf16 = ~tiles_with_bf16`则把130,048个inactive tile也计入，不能用于
  kernel动态gate覆盖率。2.131的4,049,424是`all_history+mixed`而非权威
  `tiles_with_history=4,050,775`，后续报告需明确更正字段边界但保留历史章节原文。
- v3 ranking使用`opportunity_ranking.json`、`validation.json`和
  `evidence_manifest.sha256`三文件格式；当前c349 mixed离线基线仍为shared109,568、
  registers255、stack0、PTX loads245、cubin206,640 bytes。v4将继承binary必须变化且
  shared/register/stack/load均不增的fail-closed门禁，并新增pending-scale数学边界与
  同一kernel program内连续history-only run覆盖，不能用跨program的扁平相邻计数。
- production mixed stage1以`query_row`和`head_group`为program维度，selected token
  tile循环仅在单个query row内按`block_t=16`遍历`effective_topk`；同一query的8个head
  groups共享相同tile类别。32K冻结合成覆盖共有32,768个query rows、每row最多128 tiles，
  因而4,194,304个total tiles与4,064,256个active tiles吻合。pending-scale run统计必须
  逐row重置；排序后token类别按位置自然形成prefix BF16、history、recent BF16的有序段，
  需实际重放seed42量化history-only run长度及尾部flush次数。
- 冻结合成常量已从实际import链核准为topk2048、num_heads8、latent_rank512、prefix64、
  recent256、block_t16；grouped prefill在num_splits=1时以`_prefill_head_block_size(8)`
  形成单个8-head group。固定CPU镜像仍为`oscar-glm-stage9-runtime:c349e32e9`，image ID
  `sha256:731412e96d1f…1f95`。因此每次整块BF16 accumulator缩放对应8×512个FP32元素。
- v4固定容器CPU重放自然exit0并通过14/14：history-only为3,932,635/4,064,256
  active tiles（96.76149829144621%），分布于57,972个query-row内run；run长度min/p50/
  p90/p95/p99/max=`1/124/127/127/127/127`，mean67.83680052439108 tiles。保守的
  unconditional-final-flush实现把8×512整块缩放事件从4,064,256降至164,389，减少
  3,899,867（95.95524986615016%），另增加3,932,635×8次逐head标量乘法；这只是静态
  运算计数，不是编译或TTFT加速结论。
- v4证据清单3/3、JSON解析和validation14/14独立通过；ranking/validation/script/
  manifest SHA依次为`6e8890d0…c905`、`5f794e48…6403`、`ecf0907d…794b`、
  `f5d34c74…0733`。候选仍须先过binary变化且shared/register/stack/PTX-load全不增的
  CPU-only SM80门禁，FP32乘法重分组还要求interpreter与冻结GPU allclose；目前未产生
  accuracy、TTFT或TPOT新结果。
- 报告2.133已实时追加并通过发布前门禁：8,964行/516,530 bytes，SHA256
  `f61a67a2e1dbd62007b39cbb6174068c570e64afa539699ff8364099923f0455`；章节
  1.1–1.5/2.1–2.133连续，2.131边界纠正交叉引用存在，`三池`为0，大写`A800`
  仍仅第5行历史链接两处。v4 manifest3/3、关键ranking字段与`git diff --check`
  均复算通过。
- source仓规范确认Python命令不得使用系统`python3`，必须经uv或`.venv/bin/python`；
  当前继续复用已验收固定容器/只读venv。现有`test_triton_decode.py`已有causal bound、
  `has_bf16`两处dot gate和fresh-process Triton interpreter smoke，适合新增一个局部
  结构断言：pending scale初始化、history-only累计、BF16 gate内合并/reset及loop后flush。
  production仍是clean c349，测试红灯前不改kernel。
- 正确挂载无冲突的`oscar-glm-stage9-pytest-py312`后，固定c349/Python3.12/
  pytest8.3.5目标节点得到有效1 failed，精确失败于production缺少
  `bf16_pending_scale`初始化；当前源码工作树路径和测试节点均正确，CUDA不可见。
  因此红灯有效，下一步允许只修改grouped stage1的BF16 accumulator缩放调度。
- 最小production实现新增8-head `bf16_pending_scale`：history-only tile只做逐head
  pending乘法；含BF16 tile把pending与当前previous_scale合并后更新8×512 accumulator
  并reset；循环后flush。定向结构测试+Triton interpreter smoke为2/2 passed、
  10.68秒；尚需完整CPU、Ruff/compile/diff和SM80资源门禁。
- 完整`test_triton_decode.py` CPU范围为9 passed/19 CUDA skipped、11.53秒；source
  diff仅目标测试与kernel两个文件，28 insertions/1 deletion，`git diff --check`
  通过。现存Ruff已解析为`/dev/shm/oscar-glm-stage9-ruff-0.14.0/bin/ruff`，下一步
  运行check/format与固定Python compile。
- 同版本Ruff机械格式化后，最终source diff收敛为两个目标文件20 insertions/
  1 deletion；Ruff check/format、固定Python3.12 compile、定向结构+interpreter
  2/2（10.50秒）及diff check均通过。候选现在允许进入CPU-only SM80资源裁决，
  仍未使用GPU。
- 既有SM80工具`compile_oscar_prefill_cache_split.py`仍以format v9表达production
  `mixed_h8_t16_w8`，工具SHA=`8b20bd16…9d88`；c349基线固定为shared109,568、
  registers255、stack0、PTX loads245、cubin206,640/SHA`19846644…43d8`。下一步只
  调用mixed variant并复用meta finder移除+local source origin/hash断言，不能沿用
  history-score候选的source contract字段。
- 共用工具默认`main()`会编译全部35个变体并强制首项shared等于109,568；当前工作树
  已是候选，直接运行会把候选误作baseline且可能在正确二进制上报baseline reproduction
  失败。为避免修改共用工具或重复无关编译，本轮用证据目录wrapper在local-origin预检后
  直接调用其现成`compile_variant(VARIANTS[0])`，再独立与冻结c349资源比较。
- pending-scale v1 CPU-only SM80自然exit0且source contract全真：cubin由206,640/
  `19846644…43d8`变为208,048/`5d3014e5…5eff`，shared109,568→109,568、registers
  255→255、PTX loads245→245，但stack0→40 bytes/thread。binary变化且其余资源不增，
  唯一stack门禁失败，故`promotion=false`、`reject_before_gpu`；不申请GPU。
- v1证据独立复核为manifest7/7、JSON3/3、summary字段通过；封存candidate patch与
  source Git diff SHA同为`476712ff…e14b`。summary/validation/manifest SHA分别为
  `24c94662…e108`、`b6cac336…b801`、`07e23ae0…8318`。候选已按精确反向patch撤销，
  下一步确认source clean c349并实时追加报告2.134。
- 报告2.134已实时追加并通过门禁：9,030行/520,775 bytes、SHA256
  `00d3f694dabdbe8103662ba9ea17861759181d530c0763d2704407b08ca7e190`；章节
  1.1–1.5/2.1–2.134连续，2.133交叉引用正确，`三池`为0、大写`A800`仍仅第5行
  两处，manifest7/7、资源字段、cubin变化百分比与diff check均通过。
- 2.134发布后的机会审计确认：2.85已把BF16/history cache-type拆成独立kernel并因
  history路径无零spill双驻留候选而关闭；2.92已把full-latent score、LSE merge与
  d128 value拆为三段式，实际scratch约136.06 MiB且score阶段仍不满足双block寄存器
  门禁。因此cache-type拆分与latent-dimension value拆分均已有真实反证，不能作为
  新候选重复实现。下一步从c349/pending cubin的局部内存与指令差异寻找不增加loop状态
  的变换。
- baseline/pending cubin的`cuobjdump --dump-sass`精确匹配显示：c349 baseline没有
  `STL/LDL`，pending candidate使用local stack offsets `0x0–0x24`并在loop前后产生
  多组store/load，与40-byte/thread资源记录一致。当前mixed已占255 registers/thread，
  新增8-head loop-carried向量会把其他live scalars一起挤入local memory；下一方向应
  移除既有live state或改变算法工作量，不能只换pending乘法写法并期待自然零spill。
- 报告/planning/source历史未发现production top-k 1024/1536降档实验；既有
  `cropped_topk`只让早期causal行的有效宽度小于2048，2.41冻结的32K后续chunk仍明确
  top-k width=2048。backend从模型`hf_config.index_topk`取值，因此降低top-k会改变
  DSA selected集合和模型算法，虽可线性减少dominant stage1 tiles，但必须先有正式
  indexer输出与精度门禁，不能当作纯kernel等价优化。
- 当前正式artifacts文件名搜索未发现DSA逐token selected indices或对应score dump；
  OSCAR backend只读取共享`topk_indices_buffer[:num_tokens, :topk_width]`并传给attention，
  不接收排名分数。因而不能用既有证据离线截断评估精度；还需确认indexer top-k输出列
  是否按分数排序。若列顺序未保证降序，attention侧直接截前N会选择错误集合。
- 第五轮top-k排序契约已确认一半：legacy `top_k_per_row_prefill`由
  `VLLM_TOPK_PREFILL_SORT_INDICES`控制最终索引排序；c349正式32K/batch1运行证据明确该值
  为1，历史native GPU校验也证明输出逐row按token位置单调递增且集合不变。因此attention
  侧直接取前1024/1536会保留最早token位置，不等价于按DSA分数缩小top-k，明确不可用。
- `persistent_topk`当前host wrapper硬断言`k == 2048`，其多CTA收集又用atomic位置写出，
  没有暴露score或分数降序契约。若评估较小top-k，必须在indexer真正以较小K选择后再按
  token位置排序，且需改后端/缓冲与走完整精度门禁；不能对现有2048位置排序结果截列。
- 本轮首次尝试同步三份planning时错误复用了只存在于`progress.md`的上下文作为
  `findings.md`锚点，`apply_patch`原子拒绝且无文件被改；已先读取三文件实际末尾再分别
  锚定，不重复跨文件假定相同尾部。
- c349正式`runtime_environment.txt`只设置decode top-k backend为persistent，未设置
  `VLLM_SPARSE_INDEXER_PREFILL_PERSISTENT_TOPK`或prefill-decode-topk；按Python默认值0，
  prefill实际走legacy `top_k_per_row_prefill`。该内核内部先按logit选择top-K，随后在
  `sortIndices=true`时对选中索引做位置升序；因此indexer端把K改小可得到真正较小的
  高分集合并再排序，但属于改变DSA模型稀疏度的算法候选，必须经过完整精度门禁。
- `config.index_topk`传播边界集中且一致：模型初始化按该值分配
  `[max_num_batched_tokens, topk]`共享buffer，Indexer/SparseAttnIndexer以同一值做选择，
  attention metadata与OSCAR backend再以同一宽度切片并执行kernel。现有正式launch未
  暴露HF override；可在主仓启动包装中增加单一、fail-closed的`--hf-overrides`候选，
  无需先改source kernel，但必须测试参数解析、记录runtime contract并提交发布后才能用GPU。
- v5固定32K causal实算：K2048为65,012,736 selected-token instances、4,064,256个
  active tiles和4,194,304个scheduled tile slots；K1536分别为49,152,768、3,072,768、
  3,145,728，较K2048减少24.395171%/24.395314%/25%；K1024分别为33,030,656、
  2,064,896、2,097,152，减少49.193561%/49.193752%/50%。这些是精确causal静态计数，
  不是DSA输出、精度或性能实测；MQA logits扫描等工作不随K缩小。
- v5选择较保守`index_topk_1536`作为下一候选，1024仅列第二；该候选明确不算法等价，
  必须先过256题GSM8K smoke，再过冻结2360例完整accuracy与PPL，最后才测正式32K性能。
  validation17/17、manifest3/3；ranking/validation/script/manifest SHA分别为
  `17380d08…f8c`/`194fd61d…bea`/`62fd50e8…851`/`73bf5470…3a0`。
- v5首次容器调用遗漏`--entrypoint`，固定镜像默认`/bin/bash`把Python二进制当脚本而
  exit126；显式指定`/usr/bin/python3.12`后自然exit0。独立复核首次使用宿主缺失的`jq`
  而exit127，但此前manifest3/3已经通过；随后改用宿主Perl JSON::PP复核关键字段与17/17
  validation成功。两次失败均未污染有效JSON，不重复相同调用。
- 报告2.135已实时追加并通过最终门禁：9,096行/525,548 bytes、SHA256
  `20e97aacc5237a9415105497e706f2bdc9e4b98a4cd707c0892b15973d46fca5`；章节
  1.1–1.5/2.1–2.135连续，2.59交叉引用存在，`三池`为0，大写`A800`仍仅第5行历史
  链接两处，v5 manifest3/3、validation17/17、表格四组差值与`git diff --check`通过。
- 报告门禁首次Perl heading one-liner因数组解引用表达式括号错误而exit255；后续改用
  两条简单awk分别验证第1/2章连续性并通过。该失败未修改报告，不重复复杂one-liner。
- K1536启动/config TDD已获得有效红灯：固定c349容器、CUDA不可见、network none下，
  新增目标测试实际收集并因`performance_matrix.json`缺少`candidate_hf_overrides`而
  1 error；失败发生在任何launch/config实现前，精确证明旧链路未携带HF override。
- 最小K1536 launch/config实现只改主仓四处：performance config新增
  `candidate_hf_overrides={"index_topk":1536}`；candidate wrapper导出规范化
  `HF_OVERRIDES_JSON`；通用launch仅在非空时追加`--hf-overrides`并核对解析值、写入
  parsed args/runtime environment；Stage9 verifier核对配置与runtime JSON。source kernel
  未修改。定向测试1/1、完整Stage9工具18/18、两脚本`bash -n`与diff check均通过。
- 首次容器化dry-run遗漏历史正式`oscar-glm-phase0-source-fd3e0b3`只读命名volume，
  递归preflight因已知NFS把普通文件mode映射成777而失败；所有内容hash以及新增K1536
  配置/runtime检查均已通过。补挂同一命名volume的v2递归静态门禁为68/68 passed；
  随后fixed import因无driver容器缺`libcuda.so.1`退出，留下空import JSON且未进入CLI
  parsed args。这是既有环境边界，不能把v2称为完整dry-run成功或参数解析结果。
- 无GPU device节点、只读挂载`libcuda.so.1/libnvidia-ml.so.1`后，fixed environment
  import实际成功，Python/Torch/Triton为3.12.13/2.11.0+cu129/3.6.0，候选Python与
  `_C`均来自c349 overlay，`cuda_initialized=false`。但真实CLI parser构造会调用设备
  推断并明确报`Failed to infer device type`；故CPU-only阶段不能产生绿色parsed args，
  该门禁必须留到发布后、双空闲检查后的driver-injected preflight，不能绕过或伪造。
- CLI最小探针首次只给12 GiB且未输出终态；32 GiB重跑得到上述明确device-inference
  traceback，说明前者不能作为结果。后续CPU门禁不再重复完整CLI解析。
- K1536 CPU证据最终为23/23 passed：包含18/18 Stage9工具、两脚本语法、固定Python
  编译、主仓diff、source clean c349、68/68递归静态、固定环境版本/source origin及
  `cuda_initialized=false`。`parsed_server_args`明确标为待发布后的driver-injected
  preflight，不能把空文件解释成参数验证成功。build/contract/validation/manifest
  SHA依次为`bc57f333…f885`、`2a636605…c863`、`b5c6c5c4…5da5`、`3bf4a711…6ab`。
- 结构化验证前两次容器重跑仍因Git环境失败：先是`/workspace`无法解析项目绝对Git
  元数据路径，改用规范路径后又因NFS所有者映射触发`safe.directory`。最终只在两条
  Git子命令内分别传入精确主仓/source安全目录，未修改全局Git配置；随后23/23与清单
  3/3均通过。
- 报告2.136已实时追加并通过发布前门禁：9,161行/529,818 bytes，SHA256
  `5be50c3c448ffcffe1bbab12cf75526c6f324f727c8f8a742ec5f1bbf014d574`；章节
  1.1–1.5/2.1–2.136连续，2.135与2.59交叉引用存在，`三池`为0，大写`A800`仍仅
  第5行历史链接两处，validation23/23、manifest3/3及`git diff --check`通过。
- K1536启动/config、报告2.136与planning已由主仓提交
  `20fe235423d417559b4025885157e2b0ec3d8d36`通过GitHub HTTPS发布；提交后HEAD与
  upstream一致。GPU前只需再发布本条planning身份，使主仓clean，然后执行两次8/8
  空闲检查与driver-injected preflight。
- K1536 driver preflight前双空闲检查已通过：`13:43:25Z/13:44:31Z`间隔66秒，
  两次8/8卡均为0 MiB/0%、无compute process；原始日志SHA256为
  `cdde16b9edaf9ca90b807c8efc396994ab2971d98d9625b7b9a802c570640ad8`。按阶段实时
  记录要求，先追加并发布报告2.137的空闲门禁状态，再运行preflight。
- 报告2.137空闲阶段已追加并通过门禁：9,181行/531,171 bytes、SHA256
  `4f82484903ec6acb160d3eedabe5b3bf4bf4f92b432d5f551d632074a1b9a784`；章节
  1.1–1.5/2.1–2.137连续，交叉引用、术语、idle日志hash及即时8/8空闲复核通过。
- K1536 driver-injected preflight自然exit0：静态68/68，固定import为Python3.12.13/
  Torch2.11.0+cu129/Triton3.6.0且`cuda_initialized=false`，真实CLI parsed args为
  TP8/max-model-len131072/max-batched-tokens2048/`hf_overrides={index_topk:1536}`且
  CUDA仍未初始化。结束后8/8卡0 MiB/0%、无compute process，两仓clean。
- preflight dry-run按既有实现只落盘`static_preflight.json`、
  `fixed_environment_import.json`与`parsed_server_args.json`，不会生成serve模式才有的
  `runtime_environment.txt`或`serve_command.txt`。独立hash命令对后两项报ENOENT；
  不能补造文件，runtime override以68/68 verifier检查和原始preflight log为证据。
- driver preflight证据已封存为26/26 validation、10/10 manifest。首次封存脚本把
  正常设备行UUID中的`, GPU-`误判为compute行而淘汰全部16个设备样本，validation按
  预期失败；改为精确识别`^[0-7],`后同一原始证据全绿，未重跑GPU。builder/contract/
  validation/manifest SHA为`e10f4589…5d09`/`2d85b69d…27ff`/`e2813f16…baa6`/
  `a5a2b92d…81bf`。
- 报告2.137 preflight结果已补完并通过门禁：9,223行/533,900 bytes、SHA256
  `09c08e67e4d8b5c438c982b86394e9ddd4bf6391f85e3454d680e92aad267ed9`；章节
  1.1–1.5/2.1–2.137连续，术语/交叉引用、26/26 validation、10/10 manifest及
  `git diff --check`通过。
- 完整报告2.137与planning已由主仓提交
  `980e5c71f639bac77cc5a17ade6a0fc036f4bd28`通过GitHub HTTPS发布，HEAD与upstream
  一致；source仍clean c349。下一阶段是固定256题GSM8K smoke，启动前需新双空闲检查。
- 冻结256题smoke协议是`official_v5_fast_screen`、8K/high、并发16、固定seed与
  7,974输出上限，runner已有每10分钟心跳。历史BF16/OSCAR汇总为105/256与107/256，
  但协议指纹不同且BF16逐题predictions已丢失，不能做严格paired比较；本轮只能把
  BF16 105/256作为保守smoke下限，正式晋升仍需2,360例accuracy与PPL。
- accuracy隔离入口的candidate默认走Phase7 wrapper而非Phase9 wrapper；已发布的通用
  base wrapper会继承、解析核对并记录非空`HF_OVERRIDES_JSON`，因此可由正式启动环境
  注入规范K1536，但必须在新轮parsed args/runtime manifest中实际复核，不能只凭外层
  命令声称生效。
- smoke容器入口TDD有两次有效红灯：旧Stage9容器脚本缺accuracy-smoke mode；初版实现
  又缺`candidate_runtime_environment`/prefill排序传播。最终最小实现只改容器入口和
  契约测试，复用现有source准备与Phase7隔离runner，从同一config fail-closed注入
  K1536及`VLLM_TOPK_PREFILL_SORT_INDICES=1`，固定256/8K/high/c16。完整工具19/19、
  bash syntax、固定Python compile和diff check通过，未用GPU。
- 报告2.138已实时追加并通过门禁：9,273行/537,268 bytes、SHA256
  `ef678614f38d4dd51b4aa883abc183f6f839567ef8a19085368dc4210dba9f05`；章节
  1.1–1.5/2.1–2.138连续，术语/交叉引用、19/19工具、脚本hash与diff check通过。
- 2.138、smoke容器入口与契约测试已由主仓提交
  `3688e903a16dd28c98950cbcc085d2670e71553e`通过GitHub HTTPS发布，HEAD=upstream；
  source clean/upstream c349。下一步只发布本条身份后执行smoke新双空闲检查。
- smoke独立双空闲已通过：`14:02:59Z/14:04:05Z`间隔66秒，两次8/8卡均0 MiB/0%、
  无compute process，原始日志SHA256=`a48746f29312466af0e90716bca56c5d0e06adc6e747dbc5f0c7f1d2d92f2a58`。
  先实时追加并发布报告2.139启动状态，再运行唯一正式smoke。
- 报告2.139启动状态已通过门禁：9,293行/538,614 bytes、SHA256
  `ccb31d39b39280a7c3827dfe5cc029ec1b4d295f94556feb95f2bbafe74e70c7`；章节
  1.1–1.5/2.1–2.139连续，术语、交叉引用、idle hash与diff check通过。
- K1536精度smoke `20260801T1403Z_candidate_topk1536_fast256_c16_v1`的模型141/141
  分片成功加载并ready，真实参数与runtime分别记录`index_topk=1536`、prefill排序=1；
  但runtime同时记录decode top-k backend为`persistent`。首批16个请求进入模型后，
  persistent top-k在decode路径硬拒绝非2048的K并报`RuntimeError: k must be 2048`，
  EngineCore于14:13:05Z退出。runner汇总为total256、request_failed256、scored0、
  `valid=false`、duration54.717445秒；JSON里的accuracy=0不得解释为模型精度0%，更不能
  与105/256或107/256比较。失败后14:17:05Z复核8/8卡均0 MiB/0%、无compute process。
- 本轮关键原始证据SHA256：server log=`7ebc8763…b6ab4`、runtime environment=
  `857f0e09…4632`、parsed args=`ae77fbaf…557`、invalid summary=`116e16b1…0710`、
  runner log=`a9e53d49…7037`、outer log=`f641a9e0…65a1`。外层`set -e`在失败后退出，
  没有生成原计划的outer exit/post文件；不得补造为原始文件。下一步先把已有原始文件
  封存为结构化失败证据，再只读确认legacy decode后端是否支持K1536及正式wrapper是否
  会覆盖候选环境，不能原样重复失败。
- 失败证据最终在固定c349/CUDA不可见/network none容器中通过25/25 validation与15/15
  manifest；builder/contract/validation/manifest SHA依次为`b9d44de1…e9c3`/
  `70ffe8a3…09fe`/`f0dd2472…7c3a`/`6ad468cc…77a4`。builder首轮因静态检查计数和GPU
  行过滤错误fail closed，修正后只重建同一原始输入，没有重跑实验。
- 报告2.139补完后的发布前身份为9,339行/541,876 bytes、SHA256
  `da899f9b98a90f3ad1fff1d0e5f29d470b66c458f84e814105862116fa3e0a9f`；固定容器复核
  2.1–2.139连续、交叉引用存在、`三池`为0、大写`A800`仍仅第5行历史链接中的两处，
  evidence manifest 15/15与`git diff --check`通过。
- 报告2.139失败结果和三份planning已由主仓提交
  `35bd4c62c63028e99b34e168be2a83529d64633b`通过GitHub HTTPS发布；下一步发布此身份
  检查点，之后才能进入decode fallback的只读契约审计。
- decode审计第一步确认三层事实：`csrc/topk.cu`对persistent实现有精确
  `TORCH_CHECK(k == 2048)`；Python decode路径仅当backend不是`legacy`才调用它，
  `legacy`分支转而调用带运行时`topk_tokens`参数的`top_k_per_row_decode*`算子；但正式
  Phase1 serve脚本无条件`export ...DECODE_TOPK_BACKEND=persistent`，会覆盖外层候选环境。
  目前只能证明legacy接口接受动态K参数，尚未证明K1536所有CUDA分支正确/资源可行；
  下一步读legacy CUDA实现的模板/断言及现有测试，不直接重跑。
- legacy decode主入口`top_k_per_row_decode`没有K=2048断言；K作为运行时参数控制输出
  宽度和dynamic shared memory。既有CUDA测试参数覆盖K2048与K3000，证明该实现设计为
  动态K，但没有K1536专门回归。K1536会使hist/bin/candidate等只为K2048启用的融合分支
  自动fallback到通用legacy入口。8K smoke的decode列宽低于12,288时走insertion路径，
  32K正式负载则落在12,288–200,000的单块radix路径；这只证明代码分支可达，不是当前
  c349上K1536的CUDA correctness或性能实测。
- 最小候选不应修改persistent C++：先让candidate runtime显式声明decode backend=
  `legacy`，并把通用serve的无条件persistent改成“外层未设置时默认persistent”。这样
  BF16/现有K2048路径不变，K1536走已有动态K实现。代价是候选同时改变K与decode top-k
  实现，TPOT可能变慢，必须单独做CUDA correctness和同负载性能裁决；不能把它当成纯K
  变化或提前声称收益。
- 报告2.140发布前身份为9,392行/545,484 bytes、SHA256
  `af888a4d7cbdf8da052bd6e2c23674a14056bfd69a4482e1087dc7a30259421d`；固定容器验证
  2.1–2.140连续、2.139目标存在、术语通过，六个源码输入hash复核一致，diff check通过。
- 报告2.140与planning由主仓提交`bc8ce036e5896522379010d84f95bfe58e2515a6`通过
  GitHub HTTPS发布；该提交仍未包含production/config实现，下一阶段必须从红灯开始。
- 新契约测试把candidate runtime固定为两项、要求candidate/verifier/container三条链路
  均出现decode backend，并要求base采用未设置时默认persistent的表达式。旧实现的目标
  测试实际为1 failure，首个断言即config缺legacy；这是真实红灯而非环境/collection失败。
- 最小实现保持source c349不变：candidate正式/accuracy入口都从同一config导出legacy，
  verifier逐项核对，base用`${VAR:-persistent}`保留显式候选值、其他路径仍默认persistent。
  定向2/2与Stage9工具19/19已过；静态compile/JSON/bash在改用/tmp pycache后通过。
  递归verifier仍需沿用既有命名source volume协议，避免宿主NFS mode假失败；不得用较窄
  单测替代68项完整静态门禁。
- 既有CPU dry-run的68项静态输出位于launch_cpu_v1证据中；本机所需命名volume实际为
  `oscar-glm-phase0-source-fd3e0b3`，正式容器把它挂到phase0 candidate source路径以
  避免NFS mode假失败。下一轮将复用这一只读volume并单独输出新run目录，不覆盖旧证据。
- v2递归静态实测为69/69（比K1536旧入口多1项decode runtime检查），四个关键actual为
  config两项精确字典、decode=`legacy`、prefill排序=`1`、HF override K1536，全部passed。
  固定import文件为0 bytes且无parsed JSON，保持无driver边界。最终CPU证据26/26、
  manifest5/5；builder/contract/validation/manifest SHA为`1a6f3d8f…e594`/
  `5a90721f…cb8a`/`be028ec2…5daf`/`55ccf264…f247`。
- 报告2.141发布前身份为9,466行/550,096 bytes、SHA256
  `86161d3bd47563c2218f88069de43f2b7ec421eae11cdfa5781243db867c00a6`；固定容器验证
  2.1–2.141连续、交叉引用/术语通过，26/26 validation、5/5 manifest、六文件hash与
  diff check均通过。
- 报告2.141、六个实现/测试文件与planning已由主仓提交
  `a4c383d3ac9b23ca58e5233a3710e06ed74b3969`通过GitHub HTTPS发布；source仍为clean
  c349。GPU前只需再发布本条身份并完成新双空闲门禁。
- K1536+legacy driver/CUDA阶段的新双空闲门禁已通过：`2026-08-01T14:41:18Z`和
  `14:42:24Z`间隔66秒，16/16设备行均0 MiB/0%，两次compute查询为空。原始日志
  SHA256=`2f947f64451a24a90d31c87bd08cb99620174bb7249ea880691beb88ae084ff5`。
- 报告2.142发布前身份为9,487行/551,598 bytes、SHA256
  `1322595c40b336642bf86abaed6004edd2639ed2f3d504b01a2e42b3a07da30d`；固定容器验证
  2.1–2.142连续、2.141引用和术语通过，idle证据与diff check通过。
- 报告2.142与planning已由主仓提交`327c3930a71c18968f8a9e17341c2c28622ec5c0`通过
  GitHub HTTPS发布；下一步发布本身份检查点后启动driver preflight。
- 新driver preflight实际退出码为0：静态69/69，固定环境为Python 3.12.13、Torch
  2.11.0+cu129、Triton 3.6.0，固定导入和参数解析均`cuda_initialized=false`；参数为
  TP8、max model len 131072、max batched tokens 2048、K1536，静态合同同时确认decode
  backend=`legacy`和prefill sort=`1`。退出后8张GPU均0 MiB/0%，没有模型加载或精度/
  TTFT/TPOT/吞吐测量。
- 结构化证据builder首次运行时，控制镜像Entrypoint实际为`/bin/bash`，直接追加固定
  Python ELF绝对路径使bash尝试把二进制当脚本并报`cannot execute binary file`；这是
  归档工具入口错误，未改变原始证据。后续应显式用`/bin/bash -lc`，不得重复该命令。
- 修正入口后的builder进一步发现本轮post文件格式为时间戳+8条设备行+
  `compute_processes:`，而旧模板只接受恰好8个总行；失败项仅为post行筛选，不是GPU非空。
  正确校验应筛出8条设备行逐条验证0 MiB/0%，并单独确认compute列表无GPU PID。
- post解析修复后的结构化validation为29/29、manifest10/10。随后从错误cwd运行
  `sha256sum -c`只产生相对路径找不到，并未报告任何digest mismatch；独立验证必须在
  manifest所在目录执行。
- 在证据目录重跑后10/10 manifest全部OK。报告2.143发布前身份为9,540行、555,041
  bytes、SHA256=`00d291bb5a4aa9b2af8b421379a126ed58536743baa11dcfcc441a0daab7e969`；
  固定容器确认2.1–2.143连续，2.141/2.142引用、术语和结构化证据门禁均通过。
- 报告2.143与三份planning已由主仓提交
  `0a0b6c9f520ef89aec4530750944123236507042`通过GitHub HTTPS发布；该阶段只证明启动
  合同、固定导入和参数解析，不包含CUDA correctness、精度或性能结论。
- K1536专项脚本直接调用`torch.ops._C.top_k_per_row_decode`，固定单卡、batch1、next_n1，
  用PyTorch topk值域参考验证8,192列insertion和32,768列single-block radix；每种再覆盖
  random与10LSBits ties，共4例。脚本固定容器compile通过，SHA256=
  `9295e8a2baf623fdc7b052b4a789685a691557bd4e41be032325908c0adfa210`。
- correctness前的新双空闲门禁为`2026-08-01T14:56:50Z/14:57:56Z`，间隔66秒，
  16/16设备行均0 MiB/0%，两个compute列表为空；日志SHA256=
  `b2fe6a61166ef57a3284cec70dece9d7e5b633bab35169c06588af8e05e687a9`。
- 报告2.144发布前身份为9,573行、557,271 bytes、SHA256=
  `f7fff6a42b030d56aefa4e1850a6fd5135d1d12b82afb1ed2ca65485cdb9350d`；固定容器
  确认2.1–2.144连续，2.143引用、术语、脚本与空闲证据哈希均通过。
- 报告2.144与planning已由主仓提交`0ee13b522194e7b2497153de67e2326c93a02f41`
  通过GitHub HTTPS发布；GPU专项仍未执行。
- K1536 legacy单卡CUDA专项实际4/4通过，duration 0.541128秒；8,192/32,768列各自
  random与10LSBits case均输出1,536个唯一合法索引，集合和值均匹配PyTorch reference，
  四例max abs value difference均为0。运行环境为Python3.12.13、Torch2.11.0+cu129、
  CUDA12.9、单张苹果800 capability 8.0，候选三项环境精确匹配。
- docker退出瞬间post文件为8卡0 MiB、无compute，GPU0利用率仍有9%瞬时尾迹；
  `15:01:06Z`复核8/8卡均0 MiB/0%、无compute。不得把瞬时post改写成0%或遗漏该边界。
- 专项证据结构化门禁为33/33、manifest10/10，独立hash校验全部OK。生成脚本、source
  contract、validation与manifest SHA256依次为`d15dcbb0…70d`、`c5b29419…3ad`、
  `78f09a8b…dc5`、`c616a394…90b`。
- 报告2.145发布前身份为9,629行、560,638 bytes、SHA256=
  `5744f77f8d8a0d2dc55bc17edcb8be289b2c4722d665cdabc58f0ff4d7bb60ac`；固定容器
  确认2.1–2.145连续，2.144引用、术语、证据与diff门禁均通过。
- 报告2.145与planning已由主仓提交`cdabf53c37068ee7a9a258ef768d7dcd60ed9ca6`
  通过GitHub HTTPS发布；下一阶段才开始256题smoke的新空闲门禁。
- 256题K1536+legacy smoke的新双空闲门禁为`15:05:14Z/15:06:21Z`，实际间隔67秒；
  16/16设备行均0 MiB/0%，两个compute列表为空。日志SHA256=
  `ffc5334bccdf1d4dff7fe72d5f6dfc961c87c7ff91a1a004acc4a9eda487cc0a`。
- 报告2.146发布前身份为9,650行、562,126 bytes、SHA256=
  `2c3b89249eb239de348a4562b4ac016ba16116a4af5de742fa6875044ddf2140`；固定容器
  确认2.1–2.146连续，2.145引用、术语、idle证据与diff门禁通过。
- 报告2.146与planning已由主仓提交`2e59274cf4fa44d4a541a90b9e8d7c7b981420fb`
  通过GitHub HTTPS发布；256题模型与runner仍未启动。
- 首次256题启动只设置`RUN_ID`，而`run_accuracy_smoke`要求外层`FORMAL_RUN=1`，故在
  Docker创建前明确报`formal accuracy smoke requires FORMAL_RUN=1`并exit1。前后GPU
  均空闲，不能把该轮称为模型/精度失败；重试必须使用新ID并显式设置正式标志。
- 报告2.147发布前身份为9,675行、563,677 bytes、SHA256=
  `3e71dbea27f0da9952a7a317b7ccfd1586f5f673f94a1631377af01ac9f979e9`；固定容器
  确认2.1–2.147连续，2.146引用、术语、失败证据和diff门禁通过。
- 报告2.147与planning已由主仓提交`c7354884a825a0050bf904eb528e96dbc0554ea8`
  通过GitHub HTTPS发布；失败run保持封存且不复用。
- 修正命令重试前双空闲为`15:10:35Z/15:11:41Z`，间隔66秒，16/16设备行均
  0 MiB/0%，两个compute列表为空；日志SHA256=
  `a86cbc38e38f77698153efec0ae4c8ab81fc366a723e3110c8dd0eb98264f8c4`。
- 报告2.148发布前身份为9,693行、564,886 bytes、SHA256=
  `b86b9baf3586448ac1e508df9a6d736f358c35f2ae91846ae6fc1a69e3c84782`；固定容器
  确认2.1–2.148连续，2.147引用、术语、idle hash与diff门禁通过。
- 报告2.148与planning已由主仓提交`ef2c776392ff3f00058325d5530a7588590c4fc9`
  通过GitHub HTTPS发布；有效256题轮次尚未启动。
- 有效K1536+legacy快速筛选run为
  `20260801T1517Z_candidate_topk1536_legacy_fast256_c16_v2`：固定TP8、max model len 8192、
  并发16、high、prefill排序1、decode legacy，外层自然exit0。最终summary为256/256
  scored、106正确、accuracy=0.4140625、0 request failure、130 truncated；GPU释放检查
  8/8卡0 MiB/0%、无compute process。
- 独立读取predictions重算得到256行、256唯一ID、106正确、130 truncated，且256个
  checkpoint齐全；official validation中的predictions/summary/runner/suite哈希全部与
  `sha256sum`一致，server fatal scan为空。结构化证据42/42、manifest25/25，validation
  和manifest SHA256为`5e5b64fa…e2e2`/`0a7a045e…eba6`。
- 本轮只通过保守性能候选筛选：106正确比历史BF16的105多1、比K2048 OSCAR的107少1；
  historical protocol fingerprint不配对，且本轮130/256截断。official validation明确
  `final_full_evaluation_still_required=true`，不得写成最终精度或配对优于BF16。
- 归档时首次尝试`docker cp`使用宿主bind路径且容器已由`--rm`清理，产生No such
  container/path；未修改原始实验。随后使用只读sudo直接核验0600证据，并在固定c349
  控制镜像内成功生成结构化归档；失败复制命令不再重复。
- 报告2.149发布前身份为9,767行、569,522 bytes、SHA256=
  `c5a3b5d5d0cf989b3693e625adb75912556b255da7bbdb9af0b831e514651fd2`；章节1.1–1.5、
  2.1–2.149连续，2.148交叉引用、术语、42/42 validation、25/25 manifest与diff
  门禁均通过。下一步只提交并HTTPS发布四份文档，恢复clean前不启动32K性能实验。
- 报告2.149与planning已由主仓提交`d4f3e76e862c7cea3c3c462a565e492e0efe3cc7`
  通过GitHub HTTPS发布；source仍clean c349。下一步只发布本身份并恢复clean，再开始
  32K性能实验前的新双空闲门禁。
- 2.149发布身份由`b0fe18c424eb355acaf101c8a3ecc1d5ce3f7ace`推送后两仓clean。
  性能前双空闲检查为`19:56:48Z/19:57:53Z`，间隔65秒；16/16设备行均0 MiB/0%，
  两次compute列表为空，原始日志SHA256=`d087bcdafd3f3219d6a4cc227091f8f9f37312c0d433fcea98b50e27e8b82415`。
- 双空闲外层首次解析错误地要求first compute标记后直接出现second compute标记，忽略
  中间第二次时间戳和8行设备数据，故解析exit1；实际两次采样均已落盘且为空闲。修正解析
  后复用同一原始日志通过16/16和两个空compute门禁，没有重跑或改写采样。
- 报告2.150发布前身份为9,796行、571,458 bytes、SHA256=
  `ceaa71d377e21689eedca40148e362addd9e1071ae50a0da18d4eebb1390e7c2`；2.1–2.150
  连续，2.149引用、术语、idle日志hash与diff门禁通过。下一步只发布四份文档。
- 报告2.150与planning已由主仓提交`697f7aaa3504467c5d785f7f8934c8e6311479ab`
  通过GitHub HTTPS发布；下一步只发布本身份恢复clean，再即时复核8卡并启动正式轮次。
- 2.150发布身份由`51f4152219d090225c31cc4e7366d5a975a9ae11`推送后两仓
  clean/upstream。正式32K/batch1 run
  `20260801T2001Z_candidate_topk1536_legacy_32k_b1_v1`外层自然exit0；
  3轮共9/9请求成功，三轮中位TTFT/TPOT/请求吞吐为
  `25682.409651/198.952126 ms/0.019634504 req/s`。
- K1536+legacy相对K2048 OSCAR的TTFT/TPOT/吞吐为
  `-15.849525%/+0.850878%/+9.077736%`；相对BF16仍为
  `+104.999655%/+11.251016%/-30.808156%`。这是有效但不足以收敛的TTFT
  优化，下一步必须继续归因剩余约13.154秒的TTFT差距。
- profiler耗时718.619031秒并passed，8/8 rank trace、8/8 table齐全，
  critical rank0 self CUDA total=57304.0 ms。server无Traceback/EngineCore fatal/OOM/
  `k must be 2048`，但有1条external init callback profiler警告；随后HTTP 200且
  trace/table/validation完整，故只记为非致命警告。
- 三轮与profiler共4/4 validation status为passed；summary中17/17个路径哈希
  独立重算全部匹配。退出瞬间8卡0 MiB、无compute，利用率尾迹为100%；
  `20:36:49Z`复核为8/8卡0 MiB/0%、无compute。已实时追加报告2.151。
- 报告2.151与planning已由主仓提交`d90e8773d70e8a2b023a6c0f603c3bf0bcffcaad`
  通过GitHub HTTPS发布；下一步只发布本身份恢复clean，再做CPU-only profiler归因。
- 2.151身份由`a22c09002aabce7f7747cd11a2b22cd1f1a728ba`推送后两仓
  clean/upstream。三组trace使用同一analyzer SHA=`724aeb5e…f43`、Python3.12.13/
  ijson3.4.0.post0，CPU-only断网且无NVIDIA runtime。
- 首次并行分析的BF16 glob过宽，匹配72份而非8份，子任务在读取前fail
  closed、无BF16输出，外层因此exit2；两个OSCAR子任务正常8/8落盘。
  随后用冻结BF16 summary中精确8条path/bytes/hash新跑，自然exit0。
- K1536相对K2048的prefill wall为`25791.032721`vs`30583.463917 ms`，
  改善4,792.431197 ms，解释正式TTFT改善99.074166%。stage1为
  `15068.884580`vs`19847.610017 ms`，改善4,778.725438 ms，解释wall
  改善99.714012%；去掉stage1后剩余wall只改善13.705759 ms。
- K1536的stage1仍占prefill wall 58.426837%；相对BF16的profile prefill wall多
  15,704.562953 ms，stage1相对BF16主attention多11,684.510605 ms并解释
  74.402011%差距。剩余非主attention wall仍多4,020.052348 ms。
- 三组均8/8 ranks、16 chunks、32,768 tokens且trace身份匹配；37/37 validation、
  10/10 manifest独立通过。已实时追加报告2.152；K1024只是需重新精度门禁
  的高风险候选，线性外推仍不足以单独追平BF16，不写成实测结果。
- 报告2.152与planning已由主仓提交`c292ab190d2ada0b091217828541ee4b721511d0`
  通过GitHub HTTPS发布；下一步只发布本身份恢复clean，再做下一CPU-only排序。
- 2.152身份已由`32e8f296eef960aebcf0a1402919cb0861268257`推送，主仓与
  source仓均clean/upstream。planning-with-files catchup提示13条未同步会话消息；
  对照Git、报告2.151/2.152与三份planning后，实验结果/归因均已落盘，
  只需补记本发布身份。
- K1536与BF16的trace形状完全一致：均为144个execute context、16个chunk、
  32,768个prefill token，因此剩余差距不是chunk数或请求形状不配对造成。去掉各自
  主attention后，K1536相对BF16的prefill wall仍多`4,020.052348 ms`；对应kernel
  差为`3,517.336778 ms`，其余约`502.715571 ms`来自非kernel wall差异。
- 剩余显式单项中，OSCAR `_rotate_latent_kernel`约`1,486.196 ms`最显著；但该路径
  已在既往阶段由约`3,390.942 ms`降至约`1,486.435 ms`，TF32候选因INT2量化边界失败，
  IEEE配置扫描后的现路径为已验证最优。由于BF16 prefix/recent与旋转域history需要逐层
  逆旋转后才能相加，不能直接删除或跨层复用，故本轮不重开已收敛的rotation方向。
- 其余残差分散于MoE/矩阵乘、selector、merge/add/quantize等内核；NCCL反而比BF16少
  约`79.529 ms`，不是当前瓶颈。下一步继续审计stage1本体及已拒候选，避免重复实验，
  再冻结K1024或非K候选的准确率/性能晋级门槛。
- stage1源码与历史反证复核完成：当前h8/t16/w8、single-split路径已含causal
  `effective_topk`和空BF16 tile门禁；launch/split、cache-type拆分、reload/manual、
  三段式、maxnreg、full/partial compact、lazy BF16、history-score BF16和pending-scale
  均已有资源、正确性或苹果800实测淘汰证据。继续在这些方向微调不满足“未验证候选”条件。
- K=1,024是唯一尚未实测且可直接把K=2,048的stage1 selected-token/active-tile工作量
  约减半的候选；相对K=1,536则继续减少约三分之一。但它会改变近似算法，不能沿用
  K=1,536精度结论；而基于既有两点的线性外推仍约`20.740 s` TTFT、比BF16高约
  `65.55%`，所以它只是“可证伪的下一步”，不是追平方案或预期成功结论。
- 下一阶段顺序冻结为：报告先行并发布 → 新双空闲 → K1024+legacy专项CUDA correctness
  与256题快速精度筛选 → 只有通过预设精度门槛才做正式32K/batch1三轮+profiler。
  快速256题仍受截断和历史协议指纹不配对限制，不得表述为最终精度验证。
- 报告2.153已实时追加并通过发布前门禁：10,026行、586,752 bytes、SHA256=
  `0bb4db030d28e33e9df89468759d5a48fa2a45ff08f58eaa3f28d94d7853360d`；
  2.1–2.153连续，`三池`为0，大写`A800`仍只在第5行历史链接出现两次，2.152
  交叉引用、冻结门槛、数值与diff check通过。下一步只发布文档，不修改配置或使用GPU。
- 报告2.153与planning已由主仓提交`6fc95f0f921d608efbd25ca1cf09fecc26fac2ae`
  通过GitHub HTTPS发布。下一步只发布本身份并恢复clean/upstream，再开始K1024
  配置/测试的CPU-only最小改动。
- 2.153身份已由`6da2b0a1ede899fd379f5835292018fef88182bd`推送，两仓
  clean/upstream。当前Phase 9候选K值由`configs/phase9/performance_matrix.json`
  单点定义，但`run_candidate_tp8.sh`、`run_containerized_performance.sh`和
  `verify_candidate_performance.py`各自用精确字典fail closed，测试也冻结旧值；
  K1024最小实现必须同步这些合同，source子模块无需修改。
- K1024有效TDD红灯在固定c349控制镜像、Python3.12、断网/CUDA不可见环境得到：
  19项中仅`test_candidate_index_topk_override_is_wired`失败，实际1536、期望1024。
  最小实现只把配置与三个fail-closed消费者同步为1024；随后19/19测试、两个shell
  `bash -n`和两个Python文件`py_compile`全部exit0。source子模块没有改动，也未使用GPU。
- 报告2.154发布前门禁通过：10,071行、589,672 bytes、SHA256=
  `c0ee407fb6087047dbef5f70876a828f88ecc7442abade3d1b3fb0c851867904`；
  2.1–2.154连续，`三池`为0，大写`A800`仍仅第5行两处，2.153引用、五个文件哈希
  与diff check通过。下一步只发布本阶段，恢复clean前不运行preflight或GPU。
- K1024配置、报告2.154与planning已由主仓提交
  `980ac5e0321e16955ab6bcf381398b98fd5e4f0b`通过GitHub HTTPS发布；下一步只发布
  本身份恢复clean/upstream，再运行正式CPU-only preflight。
- 2.154身份已由`0aa5d2fa9ef373b12b414681ba357751a1789007`推送，两仓
  clean/upstream。标准`run_containerized_performance.sh preflight-candidate`的
  `run_in_container`固定使用`--gpus all`；它应保持`cuda_initialized=false`且不加载
  模型，但不是CUDA不可见的纯CPU容器。故必须先双空闲，再做driver-injected preflight。
- K1024 driver preflight前双空闲已完成：`2026-08-01T21:11:05Z`与`21:12:10Z`
  间隔65秒，两次8/8卡均0 MiB/0%，两个compute-process列表为空，采样命令自然exit0。
  下一步先核验日志hash并实时追加报告2.155，发布前不运行preflight。
- 报告2.155发布前门禁通过：10,099行、591,616 bytes、SHA256=
  `b8f0f35bbf550796afe16b54b066d26eea83fc870cfe05d9c62c22c5212ed52d`；
  2.1–2.155连续，`三池`为0、大写`A800`仍仅第5行两处，2.154边界纠正、idle hash、
  交叉引用与diff check通过。下一步只发布文档，恢复clean前不运行preflight。
- 报告2.155与planning已由主仓提交`f98f473e995de3b0f34986db176382bab8dcc8ba`
  通过GitHub HTTPS发布。下一步只发布本身份恢复clean/upstream，再即时复核8卡并运行
  K1024 driver-injected preflight。
- 2.155身份已由`8ada6b3df3363004a5d3b0eb148fc764494b364b`推送，两仓
  clean/upstream；`21:14:12Z`即时复核8/8卡均0 MiB/0%，compute列表为空。可按已发布
  合同用独立run ID启动K1024 driver-injected preflight。
- K1024正式driver preflight run
  `20260801T2115Z_topk1024_driver_preflight_v1`自然exit0。完整verifier status=passed；
  固定import为Python3.12.13/Torch2.11.0+cu129/Triton3.6.0且`cuda_initialized=false`；
  真实CLI parsed args为TP8、max_model_len131072、max_batched_tokens2048、
  `hf_overrides={index_topk:1024}`，同样`cuda_initialized=false`，legacy/sort合同通过。
- 三个落盘JSON为816/668/18,352 bytes，SHA依次为`c0ca8f9b…7ef0`、
  `65df8021…6b2`、`776d46fa…d86`。宿主核验命令中的`jq`不存在，三个JSON查询因此
  未执行；文件hash/stat已成功，原始preflight输出已显示status passed。下一步改用
  固定Python3.12只读解析检查数，不重复`jq`。
- 固定c349/Python3.12容器只读复核确认static preflight为69/69 passed、0 failed；
  import/parsed两处`cuda_initialized=false`，parsed参数为TP8、maxlen131072、
  max-batched2048、HF override1024。preflight后8/8卡0 MiB/0%、无compute。
  既有`vllm._version` RuntimeWarning不影响三项输出或自然exit0，必须在报告保留。
- 报告2.156发布前门禁通过：10,147行、594,685 bytes、SHA256=
  `a6ea77a38b81b38707eaf1646437c12b7998e39085ceee7cf0635f6ed209474f`；
  2.1–2.156连续，`三池`为0、大写`A800`仍仅第5行两处，2.155引用、三个JSON身份、
  warning边界与diff check通过。下一步只发布，不运行CUDA专项。
- 报告2.156与planning已由主仓提交`bc0200a2eb80402f4ac12f2ac542048b888601fa`
  通过GitHub HTTPS发布。下一步只发布本身份恢复clean/upstream，再准备K1024专项脚本；
  脚本CPU合同与报告发布前不申请GPU。
- 2.156身份已由`191aa35246dece39dce1d6bdf7be9cf1259aace4`推送，两仓
  clean/upstream。既有K1536专项脚本仅以`TOP_K=1536`控制indices形状、CUDA op参数、
  reference topk和唯一数断言，4例固定为8K/32K×random/10LSBits；K1024最小脚本只需
  同步`TOP_K=1024`与scope，环境/分支/数值合同原样保留。
- K1024专项脚本已落入独立证据目录，SHA=`3b795903…e893`；相对已通过K1536脚本
  `9295e8a2…a210`的逐行diff仅为`TOP_K`和scope两处1536→1024。固定c349、断网、
  CUDA不可见容器的compile与AST合同通过；4例、legacy/env、唯一索引、set/value和
  max-abs断言均存在。该结论只允许进入双空闲/CUDA专项，不是CUDA正确性结果。
- 报告2.157发布前门禁通过：10,188行、597,099 bytes、SHA256=
  `a0b1bacf59a97fb78d65acba2289a506b844284f8fc85191fb5d3d98b2dcc2f7`；
  2.1–2.157连续，`三池`为0、大写`A800`仍仅第5行两处，2.156引用、脚本hash与diff
  通过。脚本命中`**/artifacts/`忽略规则，发布时只精确force-add该文件。
- 专项脚本、报告2.157与planning已由主仓提交
  `209d52208fb4bf894c43b78687eef5411712eaa3`通过GitHub HTTPS发布，cached name-status
  仅为报告、三份planning与一个精确force-add脚本。下一步只发布本身份后做双空闲。
- 2.157身份已由`14c35f6c17fd1b27cdb59756897c34c336344124`推送，两仓
  clean/upstream。K1024专项前双空闲为`2026-08-01T21:23:53Z/21:24:58Z`，间隔
  65秒，两次8/8卡均0 MiB/0%，两个compute列表为空，采样命令自然exit0。
- 报告2.158发布前门禁通过：10,210行、598,569 bytes、SHA256=
  `ac385cec40ba781cd9d90742b35268c91ee87d7702595e4cbf0df93975c87929`；
  2.1–2.158连续，`三池`为0、大写`A800`仍仅第5行两处，2.157引用、idle hash和diff
  通过。下一步只发布文档，恢复clean前不运行CUDA专项。
- 报告2.158与planning已由主仓提交`26db1ac222713c81fe5160cd62eebbc1d0e16c55`
  通过GitHub HTTPS发布。下一步只发布本身份恢复clean/upstream，再即时复核GPU0并运行
  已冻结4例K1024专项。
- 2.158身份发布后主仓HEAD/upstream为`0b339249a4d0e0e9c30f33a052e273375e3db685`，
  `21:27:16Z`即时复核8卡空闲。固定GPU0 run
  `20260801T2127Z_topk1024_legacy_cuda_correctness_v1`自然exit0，4/4 case通过；
  8K insertion与32K radix的random/10LSBits均为1024唯一索引、set/value match、
  max abs=0，脚本内耗时0.540293443秒。既有`vllm._version` warning为非致命。
- result/log/exit/post SHA依次为`d9ce24db…957c`/`5e68823f…1feb`/
  `9a271f2a…6aa`/`f17ace2c…ba23`；固定Python3.12只读独立复算全部case通过。
  容器退出瞬间GPU0为0 MiB但利用率8%尾迹、无compute；`21:28:35Z`复核8/8卡
  0 MiB/0%、无compute。下一步重读并追加报告2.159，不启动256题。
- 报告2.159发布前门禁通过：10,262行、601,669 bytes、SHA256=
  `475b21c8b797327b18003e241f552ff8f905af3726ef5469831564332dff2643`；
  2.1–2.159连续，`三池`为0、大写`A800`仍仅第5行两处，2.158引用、四原始hash、
  8%尾迹和diff通过。发布时精确force-add四个忽略路径文件，不扩大范围。
- 报告2.159、四个原始文件与planning已由主仓提交
  `0326eccc5bfb25426e53aaf46324709120e5dbba`通过GitHub HTTPS发布；提交范围精确。
  下一步只发布本身份恢复clean/upstream，再为256题精度筛选做新双空闲。
- 2.159身份已由`ebfd89d482aa1e543dfb5a650eb734fc48de0985`推送，两仓
  clean/upstream。K1024 fast256前双空闲为`2026-08-01T21:31:07Z/21:32:13Z`，
  间隔66秒，两次8/8卡0 MiB/0%，两个compute列表为空，采样命令自然exit0。
- 报告2.160发布前门禁通过：10,291行、603,690 bytes、SHA256=
  `cd445ff84af5c5e406b727c895604f25c58d0a955fba7dc649aac9eaf91f1b41`；
  2.1–2.160连续，`三池`为0、大写`A800`仍仅第5行两处，2.159引用、fast256阈值、
  idle hash与diff通过。下一步只发布，恢复clean前不启动长实验。
- 报告2.160与planning已由主仓提交`7433abb9e3f84cbf1efe72bd5b44343b6f83611c`
  通过GitHub HTTPS发布。下一步只发布本身份恢复clean/upstream，再即时复核8卡并以
  显式`FORMAL_RUN=1`、新run ID启动K1024 fast256。
- 2.160身份已由`c215df4245b6e51a32c2dc5ea28a182e6f43d5ad`推送，两仓
  clean/upstream。K1024 fast256 run`20260801T2134Z_candidate_topk1024_legacy_fast256_c16_v1`
  入口通过，141/141 shard加载、服务`21:41:02Z`ready，runner目录时间`21:41:10Z`。
  `21:43:05Z`落盘4/256、3正确、75%、0失败、0截断；仅为早期进度，不是精度结论。
- `fast_runner_state.json`为0600，普通sed首次Permission denied且未读取；随后沿用既有
  只读sudo/固定容器读取，未修改运行文件。长实验继续由runner每10分钟打印累计统计。
- K1024 fast256首个10分钟节点为`2026-08-01T21:51:13Z`：runner自带监控因0600
  state显示`completed unknown`、server_completed=19/256；固定容器只读复算在
  `21:51:30Z`确认19个checkpoint、16正确、84.210526%、0失败、0截断。8卡利用率
  74%–98%，进程持续计算；该节点只为运行进度，不作最终门槛结论。
- 第二个10分钟节点`22:01:13Z`仍为server_completed=19/256；`22:01:58Z`固定容器
  复算19 checkpoint、16正确、84.210526%、0失败、0截断。服务日志持续显示16个
  running、0 waiting，8卡利用率76%–99%，因此未新增checkpoint来自同批长输出尚未
  完成，不是停滞或失败。
- 第三个10分钟节点`22:11:13Z`仍为server_completed=19/256；`22:11:33Z`固定容器
  按checkpoint的`score`字段复算为19题、16正确、84.210526%、0失败、0截断。
  服务为16 running/0 waiting、生成吞吐62.4–64.0 token/s、KV cache 19.2%，8卡
  利用率74%–98%，无OOM/fatal；该实时精度只描述未完成样本，不能作为最终结论。
- 第四个10分钟节点`22:21:14Z`前台先打印状态文件旧值20、server checkpoint 38；
  `22:21:30Z`固定容器读取已刷新state与checkpoint均为38/256，其中20正确、
  52.631579%、0请求失败、16截断，38题全部`scored`。服务16 running/0 waiting、
  生成吞吐76.8–78.4 token/s，8卡利用率73%–97%，无异常；节点正式值采用相互一致的
  固定容器state/checkpoint，不采用刷新瞬间的旧值20。
- 第五个10分钟节点`22:31:14Z`为server checkpoint 39；`22:31:32Z`固定容器读取
  state/checkpoint均39/256、21正确、53.846154%、0请求失败、16截断，全部`scored`。
  服务16 running/0 waiting、生成吞吐76.8–78.4 token/s，8卡利用率79%–98%，无异常；
  该批长输出仍未完成，继续运行且不以未完成实时精度作门槛结论。
- 第六个10分钟节点处于批次切换：`22:41:14Z`前台server checkpoint为43，
  `22:41:43Z`固定容器已读到state/checkpoint一致45/256、22正确、48.888889%、
  0请求失败、20截断、全部`scored`。单次GPU采样恰为8卡0%瞬时空隙，但服务日志同分钟
  仍有新prompt、62.2–78.4 token/s且checkpoint持续增长；随即3次1秒dmon复核8卡SM
  为64%–97%，证实没有掉卡。节点正式值采用较晚且一致的45题值，继续运行。
- 第七个10分钟节点`22:51:14Z`前台server checkpoint为76；`22:51:46Z`固定容器
  读取state/checkpoint均已增至77/256、35正确、45.454545%、0请求失败、32截断，
  全部`scored`。服务16 running/0 waiting、生成吞吐54.3–78.4 token/s，8卡利用率
  69%–98%，无异常；节点采用较晚且一致的77题值，继续完整门禁。
- 第八个10分钟节点`23:01:15Z`为server checkpoint 77；`23:01:41Z`固定容器确认
  state/checkpoint仍一致为77/256、35正确、45.454545%、0请求失败、32截断，全部
  `scored`。服务16 running/0 waiting、生成吞吐78.4 token/s，8卡利用率74%–97%，
  无异常；本批长输出未完成，继续运行。
- 第九个10分钟节点`23:11:15Z`前台server checkpoint为83；`23:11:41Z`固定容器
  读取state/checkpoint均已增至84/256、38正确、45.238095%、0请求失败、37截断，
  全部`scored`。服务16 running/0 waiting、生成吞吐70.3–78.4 token/s，8卡利用率
  66%–97%，无异常；采用较晚且一致的84题节点值继续运行。
- 第十个10分钟节点`23:21:15Z`前台server checkpoint为108；`23:21:33Z`固定容器
  读取state/checkpoint一致为108/256、49正确、45.370370%、0请求失败、48截断，
  全部`scored`。服务16 running/0 waiting、生成吞吐78.4 token/s，8卡利用率
  79%–98%，无异常；继续完整门禁。
- 第十一个10分钟节点`23:31:15Z`/`23:31:46Z`仍为108/256、49正确、
  45.370370%、0请求失败、48截断，全部`scored`。服务16 running/0 waiting、生成吞吐
  78.4–80.0 token/s，8卡利用率77%–98%，无异常；本批长输出未完成，继续运行。
- 第十二个10分钟节点`23:41:15Z`/`23:41:32Z`均为117/256、53正确、
  45.299145%、0请求失败、54截断，全部`scored`。服务16 running/0 waiting、生成吞吐
  78.4–80.0 token/s，8卡利用率67%–97%，无异常；继续完整门禁。
- 第十三个10分钟节点`23:51:16Z`/`23:51:37Z`均为147/256、67正确、
  45.578231%、0请求失败、64截断，全部`scored`。服务16 running/0 waiting、生成吞吐
  78.4 token/s，8卡利用率75%–98%，无异常；继续完整门禁。
- 第十四个10分钟节点`00:01:16Z`/`00:01:47Z`仍为147/256、67正确、
  45.578231%、0请求失败、64截断，全部`scored`。服务16 running/0 waiting、生成吞吐
  78.4–80.0 token/s，8卡利用率75%–99%，无异常；本批长输出未完成，继续运行。
- 第十五个10分钟节点`00:11:16Z`前台server checkpoint为155；`00:11:43Z`固定容器
  读取state/checkpoint均已增至156/256、69正确、44.230769%、0请求失败、71截断，
  全部`scored`。服务16 running/0 waiting、生成吞吐68.7–78.4 token/s，8卡利用率
  67%–97%，无异常；采用较晚且一致的156题节点值继续运行。
- 第十六个10分钟节点`00:21:16Z`/`00:21:40Z`均为175/256、77正确、44.000000%、
  0请求失败、80截断，全部`scored`。服务16 running/0 waiting、生成吞吐78.4 token/s，
  8卡利用率74%–98%，无异常；继续完整门禁。
- 第十七个10分钟节点`00:31:17Z`/`00:31:53Z`仍为176/256、78正确、
  44.318182%、0请求失败、80截断，全部`scored`。服务16 running/0 waiting、生成吞吐
  78.4–80.0 token/s，8卡利用率76%–98%，无异常；本批长输出未完成，继续运行。
- 第十八个10分钟节点`00:41:17Z`/`00:41:51Z`均为189/256、83正确、
  43.915344%、0请求失败、87截断，全部`scored`。服务16 running/0 waiting、生成吞吐
  76.8–78.4 token/s；节点8卡利用率54%–84%较低，但checkpoint增长且服务持续生成，
  无掉卡、OOM或错误，继续完整门禁。
- 第十九个10分钟节点`00:51:17Z`/`00:51:54Z`均为216/256、98正确、
  45.370370%、0请求失败、96截断，全部`scored`。服务16 running/0 waiting、生成吞吐
  76.8–78.4 token/s，8卡利用率70%–97%，无异常；剩余40题只需再正确7题，截断
  还可增加34，继续完整门禁。
- 第二十个10分钟节点`01:01:17Z`/`01:01:43Z`仍为216/256、98正确、
  45.370370%、0请求失败、96截断，全部`scored`。服务16 running/0 waiting、生成吞吐
  78.4 token/s，8卡利用率75%–98%，无异常；本批长输出未完成，继续运行。
- 第二十一个10分钟节点`01:11:18Z`/`01:11:20Z`固定容器读取230/256、104正确、
  45.217391%、0请求失败、103截断，全部`scored`。服务16 running/0 waiting、生成吞吐
  78.4 token/s，8个TP worker均存活且无OOM/fatal；剩余26题只需再正确1题，继续完成
  全部256题后再作门禁结论。
- 第二十二个10分钟节点`01:21:18Z`前台server checkpoint为246；固定容器于
  `01:21:30Z`读取246/256、108正确、43.902439%、0请求失败、112截断，全部`scored`。
  服务10 running/0 waiting、生成吞吐37.0–39.0 token/s；正确数与截断数已满足门槛，
  但仍继续最后10题并在256题完整性核验后正式判定。
- 第二十三个10分钟节点`01:31:18Z`/`01:31:30Z`仍为246/256、108正确、
  43.902439%、0请求失败、112截断，全部`scored`。服务10 running/0 waiting、生成吞吐
  39.0–49.0 token/s，8卡利用率54%–97%；最后一批长输出尚未完成，继续运行。
- 第二十四个10分钟节点由`01:41:18Z`前台server checkpoint 250增至`01:41:51Z`
  固定容器读取251/256、108正确、43.027888%、0请求失败、117截断，全部`scored`。
  服务5 running/0 waiting、生成吞吐23.1–23.4 token/s；剩余5题即使全部截断也不会
  超过130门槛，仍等待完整结束后正式判定。
- 最终K1024快速筛选自然exit0：256/256 scored、108正确、42.1875%、0请求失败、
  122截断，summary有效时长14749.134712秒；256个连续checkpoint、唯一ID/规范化
  prompt hash与predictions逐条一致，official validation全部内容hash匹配，server无
  fatal/OOM。冻结门槛全部通过，但`final_full_evaluation_still_required=true`，只允许
  进入32K性能测试，不是最终精度验收。
- 原始产物已只读归档到K1024 Phase9 control目录；结构化validation 50/50、manifest
  29/29及独立`sha256sum -c`通过。builder首次因手录一个预期hash漏2字符而fail closed，
  修正后从头通过；没有修改原始证据或重跑实验。报告2.161已实时追加，发布前不启动
  性能实验。
- 报告2.161发布前身份为10,370行/608,673 bytes/SHA256
  `1dccb53f7b1712b82ea1b7a121314a45c67e9a182fce59883003d45dc6985c4c`；2.1–2.161
  连续，2.149/2.153/2.160引用、术语、实际Git提交身份、证据hash与diff门禁通过。
- 报告2.161、K1024精度结构化证据与planning已由主仓提交
  `199a8d0b8a2359d6e7666899ff699391984af914`通过GitHub HTTPS发布；下一步只发布本身份，
  恢复clean/upstream后才开始性能前双空闲门禁。
- 2.161发布身份已由`acf1a8e7fcd37606ab7b2e55aef89040303b8a10`推送，两仓
  clean/upstream。K1024性能前双空闲为`02:00:56Z/02:02:19Z`，间隔83秒；16/16
  设备行均0 MiB/0%、两个compute列表为空，日志SHA256=`32750043…e22`。报告2.162
  已实时追加；发布前不启动正式性能轮次。
- 报告2.162发布前身份为10,392行/610,120 bytes/SHA256
  `d0ff282231cadd267fe8a3036fd322136758b409db055ef74eb1f6c460d8191e`；2.1–2.162
  连续，2.161引用、术语、idle日志hash与diff门禁通过。
- 报告2.162与planning已由主仓提交`c4443ee07cc190c1867a7d681468fa5a48c4ff5f`
  通过GitHub HTTPS发布；下一步只发布本身份并恢复clean，再启动正式性能轮次。
- 2.162发布身份`73f04d6edf96f8987447f43dc4943476d2812d43`推送后两仓clean。
  K1024正式32K轮次`20260802T0207Z_candidate_topk1024_legacy_32k_b1_v1`自然exit0；
  3轮9/9成功，中位TTFT/TPOT/请求吞吐为21032.014034 ms/197.718813 ms/
  0.021648199 req/s。相对K1536为-18.107318%/-0.619904%/+10.255895%，相对BF16
  仍为+67.879715%/+10.561366%/-23.711913%。
- profiler耗时811.042950秒并passed，8/8 trace/table、17/17路径hash齐全，critical
  rank7 self CUDA total=51725 ms；external callback ERROR后HTTP/validation完整，记为
  非致命warning。4/4 validation、独立18/18核验通过，GPU释放8/8为0 MiB/0%、无compute。
  报告2.163已实时追加；发布前不进入下一优化阶段。
- 报告2.163发布前身份为10,472行/615,231 bytes/SHA256
  `4d2fecee3e87a5c2455b7b479f5fdd106e855558d686319b6bf2967bdad24ad1`；2.1–2.163
  连续，2.152/2.153/2.162引用、术语、性能值、证据hash与diff门禁通过。
- 报告2.163与planning已由主仓提交`793db12e64c3e193eec52c8e6a203d397f25350d`
  通过GitHub HTTPS发布；下一步只发布本身份并恢复clean，再做CPU-only trace归因。
- 2.163发布身份已由`a42c202`推送，两仓clean/upstream。K1024 trace首次容器调用因把
  含venv的整个宿主`/dev/shm`只读挂载到容器同名路径，multiprocessing创建semaphore
  时在读取trace前以`OSError`退出且无结果；有效重试仅只读挂载venv目录，保留容器
  可写`/dev/shm`，固定Python3.12.13/ijson3.4.0.post0、4CPU/32GB、断网且CUDA不可见。
- K1024有效trace分析自然exit0，8/8 ranks、每rank 16 chunks/32,768 tokens且身份
  全部匹配。相对K1536，prefill wall为21008.438954 vs 25791.032721 ms，改善
  4782.593767 ms；stage1为10217.463609 vs 15068.884580 ms，改善4851.420971 ms、
  解释101.439119%，8/8 ranks与16/16 chunks全改善。非stage1残差反而回退68.827204 ms。
- K1024相对BF16的profile prefill wall仍多10921.969186 ms；stage1主项多
  6833.089634 ms、解释62.562799%，去掉两边主attention后仍多4088.879552 ms。
  结构化归因51/51、manifest13/13通过；报告2.164已实时追加。下一步只完成发布门禁，
  再CPU-only排序stage1每tile与残差候选，不依据外推直接启动更低K实验。
- 报告2.164发布前身份为10,547行/620,610 bytes/SHA256
  `e071acf507f467a761c10debd451705b87d4ba662065ac9dbf0e6d8134446d5d`；2.1–2.164
  连续，交叉引用、术语、证据hash、51/51 validation、13/13 manifest与diff门禁通过。
- 报告2.164与planning已由主仓`ec73c8401aa58b69031b3f11c3f17d03620144d5`
  通过GitHub HTTPS发布。当前只发布本身份并恢复clean/upstream，再开始下一候选排序。
- 2.164发布身份`8ace406`推送后两仓clean。CPU-only审计确认历史BF16 runtime source
  `fd3e0b3`与当前K1024/c349不相同；模型manifest与12项主要服务参数一致，KV路径差异
  符合预期。065a→c349共有16个OSCAR提交，不能据此证明差异来自source，但控制变量
  并不闭合。
- 8 ranks×后15个steady chunk共120样本中，历史BF16与当前K1024的主attention/stage1
  calls中位数为57 vs78，非attention的主MoE Marlin calls也为106 vs148。现有证据不能
  区分runtime source、backend路径或profiler窗口影响，因此约4.089秒残差在同源BF16
  复测前只能视为混合差值；K2048/K1536/K1024同源差分不受此限制。
- 当前baseline入口已fail closed绑定c349与KV auto，并支持精确32K/b1格点。结构化审计
  39/39、manifest17/17通过，报告2.165已实时冻结同源BF16 preflight/正式三轮/profiler
  合同；下一步只做发布门禁，尚未检查或使用GPU。
- 报告2.165发布前身份为10,611行/624,885 bytes/SHA256
  `21bea6c8adb57052a9c505a80cf1168cbcb5c8b904b0603a6687d85e1a56625d`；2.1–2.165
  连续，交叉引用、术语、审计hash、39/39 validation、17/17 manifest与diff门禁通过。
- 报告2.165与planning已由主仓`9f7c78e08480640606ecb9471c9c5efb876ef5e7`
  通过GitHub HTTPS发布。当前只发布本身份并恢复clean/upstream，再执行双空闲门禁。
- 2.165发布身份`6a3c864`推送后两仓clean。当前c349 BF16 preflight前双空闲采样
  `03:18:19Z/03:19:28Z`间隔69秒，16/16设备行0 MiB/0%、两个compute列表为空；
  日志SHA=`4ebbf7b5…ac5a`，无需终止任何进程。报告2.166已实时追加，发布前不启动preflight。
- 报告2.166发布前身份为10,632行/626,352 bytes/SHA256
  `ed7fe6b699c9b50028a6fa18ded572dbfee1d7951439db1843ad0a12bd504050`；2.1–2.166
  连续，交叉引用、术语、idle日志hash与diff门禁通过。
- 报告2.166与planning已由主仓`6dd9d6a0a793c426108d868a4cab1bea20b54f10`
  通过GitHub HTTPS发布。当前只发布本身份并恢复clean/upstream，再运行baseline preflight。
- 2.166发布身份`f37b56e10b1d906b266d02f6c385654cbd7620d4`推送后两仓clean；
  `preflight-baseline`自然exit0，77/77静态检查、固定环境import和真实CLI解析通过。
  参数为TP8、KV auto、无HF override、CUDA未初始化，未启动服务或加载模型。
- 静态`runtime_source_commit=fd3e0b3`描述rootfs/原生扩展基座，不是实际Python import
  身份。baseline实际import当前c349工作树；与K1024 overlay比较2,171/2,171个受跟踪
  `vllm/`文件完全一致，0缺失/差异，原生扩展也解析到同一基座。结构化核验12/12通过，
  报告2.167已实时追加；下一步只做发布门禁，不启动正式性能轮次。
- 报告2.167发布前身份为10,687行/629,992 bytes/SHA256
  `cac4263793b693d22140cdc7f630eb34ba0f2da100ab3916ad03242bf43f641a`；2.1–2.167
  连续，交叉引用、术语、证据hash、12/12身份核验与diff门禁通过。下一步只提交并
  HTTPS发布四份文档。
- 报告2.167与planning已由主仓`f9360e9f09e58d73b8b4156ab3e7fa581f6dace4`
  通过GitHub HTTPS发布。当前只发布本身份并恢复clean/upstream，再执行正式轮次前
  双空闲门禁。
- 2.167发布身份`c3a824ae68b291f50931aeee28aabf876ce1f555`推送后两仓clean；
  正式BF16前双空闲采样`03:32:51Z/03:33:52Z`间隔61秒，16/16设备行0 MiB/0%、
  两个compute列表为空，日志SHA=`0d825d8e…1a4c`。报告2.168已实时追加，发布前不启动模型。
- 报告2.168发布前身份为10,709行/631,512 bytes/SHA256
  `a7eb9af75488364712138063e4db5111a24b31a19ea303b97832b6198bdfd6c4`；2.1–2.168连续，
  2.167引用、术语、idle日志hash与diff门禁通过。下一步只提交并HTTPS发布四份文档。
- 报告2.168与planning已由主仓`95af8ba40b53ac460514871f713df94a4666e626`
  通过GitHub HTTPS发布。当前只发布本身份并恢复clean/upstream，再即时复核8卡并启动
  同源BF16正式32K/batch1轮次。
- 2.168身份`11adb2f4cdc0e965dab1b22de84327c3b17d8f37`发布后两仓clean；同源BF16
  正式三轮9/9成功，中位TTFT/TPOT/吞吐为12515.105379 ms/153.739745 ms/
  0.031210711 req/s，三轮相对极差0.103033%/1.382804%/0.798398%。
- 当前BF16相对历史BF16为TTFT-0.103132%、TPOT-14.031063%、吞吐+9.986309%；
  K1024相对当前同源BF16更新为TTFT+68.053032%、TPOT+28.606180%、吞吐-30.638561%。
  profiler 8/8 trace/table、17/17 hash和独立22/22核验通过，报告2.169已实时追加。
- 报告2.169发布前身份为10,791行/636,391 bytes/SHA256
  `ab9dfcfe4d1b8b9f1a36790088ba9680fadbc1c05e8fe559f75a5831973a9fae`；2.1–2.169连续，
  交叉引用、术语、22/22核验、17/17 profile hash与diff门禁通过。下一步只发布文档。
- 报告2.169与planning已由主仓`279f02e9a49bc4423dc5892866093f3d1c785760`
  通过GitHub HTTPS发布。当前只发布本身份并恢复clean/upstream，再开始CPU-only同源
  trace归因。
- 2.169发布身份`96cb7f4`推送后两仓clean。当前BF16与K1024同源trace归因确认：
  prefill主attention/stage1为3301.945479 vs10217.463609 ms，多6915.518130 ms，
  解释profile/正式TTFT差距78.374056%/81.197514%；16/16 chunk均更慢。
- 同源调用结构仍为每chunk主attention 57→78、主MoE Marlin 106→148，差值21/42与
  配置中的21个full、57个shared indexer层数值吻合；trace无layer标签，只能作为源码
  审计线索，不能当作因果或直接绕过full层路径。
- decode每rank 127窗口的聚合wall为214.944947→265.825026 ms/token，多50.880079，
  解释profile TPOT差98.473876%；kernel差多46.354741 ms/token。AllReduce calls均156，
  但观测时间多41.734943 ms/token，占wall差82.026096%，更可能是上游同步/负载不均
  的表现，不能直接断言NCCL实现回退。
- OSCAR专属kernel合计18.381714、BF16专属kernel合计14.843985 ms/token，净直接差
  仅3.537729 ms/token。40/40 validation与6/6 manifest通过；报告2.170已实时追加，
  下一步只做发布门禁，再CPU-only定位21层结构和decode等待首个分叉点。
- generation结果生成后曾因结果权限0600导致宿主hash命令exit1；只补0644写入合同并
  修正既有权限，未重跑trace且内容hash不变。恢复核对中误用两个文件名并发现宿主无
  `jq`，已改用实际文件名和文本工具完成独立核验，未影响结果。
- 报告2.170发布前身份为10,877行/641,968 bytes/SHA256
  `6de38fc961cf3ffcfee5ae507814b2ced34410f14f5d164f9e053d65e05a414c`；2.1–2.170连续，
  2.169引用、术语、7项证据hash、6/6 manifest与diff门禁通过。下一步只发布文档。
- 报告2.170与planning已由主仓`54dc4825e4906e828713a78d9966b5884c4f620d`
  通过GitHub HTTPS发布；下一步只发布本身份并恢复clean/upstream，再开始CPU-only源码审计。
- 2.170发布身份`7bdb0f0`推送后两仓clean。只读源码确认`indexer_type=shared`只令
  `indexer_should_update=false`并复用top-k buffer，`full`层才更新indexer；两类层均构造
  同一`MultiHeadLatentAttentionWrapper`并执行`mla_attn`。OSCAR/BF16的分叉在cache
  update及Triton sparse `forward_mqa`内部，不支持“OSCAR额外执行21个模型层”的解释。
- 由于MoE主kernel也同步少42 calls，57/78与106/148更可能涉及profile窗口或分析切片；
  下一步审计冻结prefill分析器的chunk边界及未归类kernel，尚不形成production候选。
- 冻结prefill分析器SHA与当前`analyze_prefill_trace.py`一致；它用完整
  `execute_context_<context>(tokens)_generation_0(0)`注释作为chunk边界，按kernel
  起始timestamp是否在窗口内归类，不按kernel名称或层数过滤。57/78不是builder主动
  裁剪产生；还需检查原始注释是否早于异步kernel结束。
- 同一chunk的全kernel calls差分进一步显示BF16相对K1024少21次MLA投影/attention、
  少21次MoE路由及每个MoE主kernel对、少42次norm/AllReduce，形成完整21层尾部特征；
  这反驳“仅OSCAR stage1多21次”的解释。最可能是BF16的异步GPU尾部超出CPU注释结束，
  OSCAR同步点则使78层全部落入窗口；必须从原始trace验证，尚未作为定论。
- rank0两遍流式原始trace扫描已将假设证实：BF16的16/16 chunk在注释结束至下一
  execute_context开始之间均恰有1,000个kernel，包括21 attention、44 MoE主kernel、
  44 AllReduce和44 norm；K1024通常为35个尾部kernel。采用start→next context start
  后，两侧每chunk核心calls完全相等：attention78、MoE150、AllReduce157、norm156。
- 因此2.170的57→78/106→148不是模型执行差，而是CPU注释窗口对异步GPU尾部覆盖不同；
  当前必须先对8/8 rank复算并更正报告，不能再用81.197514%选择production候选。
- 8/8 ranks×16 chunks校正全部通过；有效调用均为attention78、MoE150、AllReduce157、
  norm156。BF16/K1024校正wall为12431.250181/21264.281180 ms，差8833.030999 ms，
  覆盖各自profile TTFT的98.949722%/99.426512%，差值对profile差闭合100.105365%。
- 校正attention为4517.903297/10217.463609 ms，差5699.560312 ms，解释profile/正式
  TTFT差64.593520%/66.920529%；旧78.374056%/81.197514%被报告2.171明确取代。
  非attention残差为3133.470687 ms，成为后续prefill第二方向；decode证据不受影响。
- 有效CPU容器Python3.12.13/ijson3.5.0、4CPU/32GB、断网/CUDA不可见，耗时
  275.709486秒；20/20 validation、4/4 manifest通过。报告2.171已实时追加，下一步
  只做章节/引用/术语/hash/diff门禁并发布。
- 报告2.171发布前身份为10,955行/647,062 bytes/SHA256
  `154276ac7f8edd94b8141df9021161b7d9b913aefd71b701675e83f3475ef8de`；2.1–2.171连续，
  2.170引用、更正声明、6项证据hash、20/20 validation、4/4 manifest与diff通过。
- 报告2.171与planning已由主仓`a2bd4813b8f9f08325767f8e888508a6282d0207`
  通过GitHub HTTPS发布；下一步发布本身份后另建新分析器，不改已冻结校正证据。
- 2.171身份`ad13fe9`发布后两仓clean。新residual分析器在8/8 ranks逐项复现2.171
  kernel/attention总时间；14/14 validation、3/3 manifest通过，无窗口口径漂移。
- 校正kernel总差7708.165397 ms，扣主attention后为2008.605085 ms，解释wall残差
  64.101608%；剩余1124.865601 ms不归因。逐kernel中位差求和与总差仅差0.075549 ms。
- `_rotate_latent_kernel`为1494.356808 ms，占非主attention后端净差1765.959061 ms的
  84.620127%，占wall残差47.690148%，是stage1之外最明确直接方向。prefill NCCL同为
  2512 calls，仅多74.521532 ms；MoE主kernel-0.264420 ms，均不是首选。
- 报告2.172已实时追加；下一步先发布，再CPU-only审计rotate调用来源、形状、内存流量
  与既有淘汰候选，不直接修改production或使用GPU。
- 报告2.172发布前身份为11,016行/651,046 bytes/SHA256
  `6e63612373d03728fd4c5f74e5cf33ca7d9379ba4a09fa65f72f5d1ad6a65802`；2.1–2.172连续，
  2.171引用、4项证据hash、14/14 validation、3/3 manifest与diff门禁通过。
- 报告2.172与planning已由主仓`fe8cd00ad31315613f8617f79bcb9d245c1d7640`
  通过GitHub HTTPS发布；下一步只发布身份，再开始rotate CPU-only只读审计。
- 2.172发布身份`abcfb7c`推送后两仓clean。rotate是512×512 IEEE/FP32 accumulator
  Triton matmul，同时用于current-history写入、recent demotion及query/output rotation；
  scratch已经复用，不能跨层/chunk缓存新latent结果。
- 历史阶段已把rotate从3390.941782 ms降到1486.434661 ms（-56.164548%）；TF32
  候选因INT2-restored最大误差1.620364超过0.35/2%门禁而淘汰，后续IEEE sweep已实施。
  当前1494.356808 ms与已优化基线一致，尚不能主张可直接消除或获得整项收益。
- 2026-08-02：`_rotate_latent_kernel` 并非新出现的热点。历史正式优化已通过将 inverse rotation 从转置视图改为连续布局，把 rotation 总耗时从 3390.941782 ms 降至 1486.434661 ms（-56.164548%），并解释当时 99.779492% 的端到端 wall 改善；当前 c349 trace 的 1494.356808 ms 与这一已优化水平一致。
- 2026-08-02：历史 trace-layout 筛选还显示 contiguous inverse 下 block M=32（0.504730 ms）比 M=16（0.573798 ms）更快，但当时生产候选有意仅落地 contiguous inverse 并保持 block M=16；M=32 是否适用于当前全部 rotation 路径尚无生产级证据，需要继续核对当时取舍与当前调用边界，不能直接据微基准改生产参数。
- 2026-08-02：报告 2.72 已给出当时不直接落地 M=32 的明确边界：`forward_m16` 七组样本后两组明显降频/升频漂移，因此 -28.752% 的正向结果不能单独支撑生产结论；全局 block M 变化还必须覆盖 256/1728/1792-row store 几何。2.73 的最小候选因此只传递 contiguous inverse，明确保持 M/N/K、warps、stages 和 IEEE precision 不变。
- 2026-08-02：从报告全文和源码 Git 历史复核，2.72 之后没有 M=32 的生产复测或淘汰结果，当前 `oscar_mla_rotate` 的 M=16 自最初实现以来未变。若继续该方向，合理边界是仅让 16384-row query/inverse 路径可选择 M=32，而不是全局改动；但按当前归因，rotation 总量约 1.494 s，远小于 attention 的 5.700 s 差距，优先级应低于继续拆解 attention 主差距。
- 2026-08-02：当前 source-matched trace 中 attention 对比是 BF16 `_sparse_mla_kernel_final_static` 4517.903296 ms（1248 calls）对 OSCAR `_mixed_sparse_prefill_stage1` 10217.463608 ms（1248 calls），净差 5699.560312 ms；调用数完全相同，所以主差距不是额外 attention launch 数量，而是单次 stage1 工作量/实现效率。OSCAR 另有 merge 98.220022 ms 和 add 70.383620 ms，但量级远小于 stage1 主差距。
- 2026-08-02：报告历史已对 stage1 做过多轮系统筛选并明确关闭：简单 launch tile/warps、split、cache-type 拆分、对称 history gate、reload/manual reduction、maxnreg、full/partial compact-load、lazy BF16 values、history-score BF16 inputs、pending-scale 等方向。full compact-load 虽 standalone 快 17.681547%，但 production 使 stage1 增加 2354.977094 ms、正式 TTFT 增加 2324.053754 ms，已由 c349 回退。不能重复包装这些失败方向。
- 2026-08-02：当前 stage1 production 形态仍是 h8/t16/w8、single-split、FP32 accumulators，历史离线资源为 109568-byte shared、255 registers/thread、0-byte stack；96.728257% active tiles 为 full-history，且 K=1024 相对 K=1536 的 stage1 实测下降 32.194957%，说明 K 区间内成本主要随 selected attention 工作量缩放。等价 kernel 微调空间已被多轮资源/正确性/端到端反证大幅压缩。
- 2026-08-02：报告2.173与planning已由主仓`82a87c8`发布，发布身份`e15f33b`也已通过GitHub HTTPS推送。下一阶段仅做CPU-only候选排序，不使用GPU。
- 2026-08-02：CPU-only静态计数：32K causal负载下K=768的selected-token instances/active tiles/scheduled slots为24,871,296/1,554,816/1,572,864，相对K=1024分别减少24.704917%/24.703411%/25%；K=512为16,646,400/1,040,640/1,048,576，相对K=1024约减49.6%/49.6%/50%。
- 2026-08-02：仅用K=1536与K=1024两点做线性外推，K=768 stage1/TTFT约7791.753123/18706.816225 ms，K=512约5366.042637/16381.618416 ms；这些是外推，不是性能实测。历史K=1024快速筛选为108/256，保守门槛105，但协议不配对且最终完整评测未完成，不能据此认定K=768或K=512精度安全。
- 2026-08-02：既有ranking v5 builder确认有效K必须在模型构造前统一覆盖`index_topk`，legacy prefill top-k支持较小K的路径已由K=1536/1024实际执行；persistent backend仍固定K=2048。K=768是16的倍数，满足stage1 tile边界，但仍需新的原生top-k专项correctness和模型精度门禁。
- 2026-08-02：相对K=1024，K=768精确减少24.702386% selected-token instances、24.702455% active tiles和25% scheduled slots；两点外推stage1/TTFT分别减少2425.710486/2325.197809 ms。历史M32中相对稳定的contiguous inverse信号按1232次调用仅约85.092778 ms；若连同存在状态漂移的forward信号也计算，历史trace信号合计约339.213519 ms。故K=768潜在量级更大，但非算法等价且精度风险决定是否晋升。
- 2026-08-02：ranking builder首次固定容器调用在进程启动前失败：误用不存在的`/opt/vllm_glm52_venv/bin/python`。镜像inspect确认实际`PYTHON_BIN=/opt/fp8_speed_up_v4_venv/bin/python`；该轮没有执行builder或生成结果，不能计为validation失败。
- 2026-08-02：首次有效ranking运行16/16通过且3/3 manifest复算通过；独立人工复核发现validation里两个不同断言误用同名`k768_selected`标签。结果布尔值均正确，但证据可读性合同不理想，已用`apply_patch`把候选选择断言重命名为`selected_candidate_k768`，需重跑后才封存最终hash。
- 2026-08-02：最终ranking重跑16/16通过，3/3 manifest独立复算通过。最终builder/result/validation/manifest SHA256为`2329cd6c...b67b`/`537514d1...021a`/`9d074362...2622`/`7ff73b92...364c`。ranking选择K=768，M32排第二，K512排第三；GPU仍禁止，精度门禁优先。
- 2026-08-02：报告2.174发布身份为`a907164`。K=768最小配置触点精确为5处：`performance_matrix.json`、`run_candidate_tp8.sh`、`run_containerized_performance.sh`、`verify_candidate_performance.py`和`test_phase9_tools.py`中的候选override期望；矩阵输入长度里的1024是负载长度，不得误改。主仓及source当前clean/upstream。
- 2026-08-02：K=768有效TDD红灯：固定c349镜像、network none、4 CPU/32 GB、CUDA不可见，只运行目标测试得到1 failure，精确显示实际`{"index_topk":1024}`、期望768；无其他失败。随后仅同步配置和3个运行时/验证消费者的同一常量，不改矩阵负载长度。
- 2026-08-02：K=768绿灯：同一固定容器目标测试1/1、完整Stage9工具19/19通过，两个shell`bash -n`、两个Python`py_compile`和`git diff --check`通过。5个改动文件SHA256依次为`1fb6cb09...7439`/`8cd2ba20...3ed`/`00c284e1...c86`/`e5188909...44f`/`5e0decba...51c`；source仓保持clean/upstream。
- 2026-08-02：报告2.175和K=768最小配置由`28ce067`发布，发布身份`02850ac`也已推送。正式preflight必须从该clean/upstream身份运行，且结果先写报告再进入任何GPU门禁。
- 2026-08-02：2.176发布身份为`b29e270`；启动前主仓clean，8/8卡即时复核均0 MiB/0%，compute-process为空。满足启动标准driver preflight的已发布门禁。
- 2026-08-02：K=768标准preflight外层自然完成，控制台显示static status=passed、候选配置/runtime均`{"index_topk":768}`；fixed import与parsed args均`cuda_initialized=false`，真实CLI为TP8、oscar_mla_int2、legacy/sort环境及K=768。出现既有`vllm._version` RuntimeWarning，待检查落盘文件和总check数后定性。
- 2026-08-02：preflight三文件存在且大小816/666/18348 bytes，SHA256为`c0ca8f9b...7ef0`/`975d51e0...365`/`8c6c3ae6...d86`；退出后8/8卡0 MiB/0%、compute为空。首次独立JSON复核命令误用宿主不存在的`/opt/uv/uv`，该校验未执行；宿主也无`uv`命令，需改用固定c349容器Python只读复核，不能把未执行命令算通过。
- 2026-08-02：固定c349只读复核第一次遗漏docker `-i`，heredoc未传入Python，exit0但无结果；补`-i`后有效复核输出status=passed、69 checks、fixed/parsed cuda_initialized=false、hf_overrides K=768。需在报告中保留两次无效环境边界。
- 2026-08-02：报告2.177发布身份`47f7a8e`已推送。下一阶段只准备K=768专项脚本，计划复用已在苹果800通过的K=1024 4例脚本并仅改变TOP_K与scope，先做CPU-only逐行diff/AST/compile合同。
- 2026-08-02：K=1024参考专项脚本为127行/4141 bytes，SHA256=`3b795903...e893`；4例覆盖8K/32K×random/10LSBits，统一通过TOP_K绑定shape/op/reference/unique count，并检查单GPU、CUDA12.9及legacy/env-cache/sort环境。
- 2026-08-02：K=768新脚本127行/4139 bytes，SHA256=`e7efd859...4f8f`。固定c349镜像、network none、CUDA不可见只读合同验证通过：候选精确等于K1024脚本仅替换TOP_K和scope两处；compile/AST、K=768、三项环境、4例覆盖、op调用和unique合同全部passed。
- 2026-08-02：报告2.178发布身份`0d4257c`已推送；主仓应从该身份clean/upstream开始GPU0专项前双空闲门禁，不能复用driver-preflight门禁。
- 2026-08-02：K=768 CUDA专项正式双空闲原始采样：first=05:44:43Z、second=05:45:48Z，间隔65秒；两次8/8卡均0 MiB/0%，两个compute段为空。待独立解析/hash与报告发布。
- 2026-08-02：专项双空闲日志独立解析通过（16条全idle、2 markers、65秒），SHA256=`a394178aac7db42669be24d1d0cb3accfe7308fbeb7db9e03faf9fb2d65bf896`。报告当前2.178已重读，待追加2.179。
- 2026-08-02：报告2.179已实时记录K=768 CUDA专项前05:44:43Z/05:45:48Z双空闲门禁、65秒间隔、16条全idle、2个空compute段和原始日志SHA256；该阶段尚未执行CUDA或产生correctness/精度/性能结果。
- 2026-08-02：报告2.179与planning已由主仓提交`a72efda`通过GitHub HTTPS发布；必须先发布该身份并确认clean/upstream，才可即时复核GPU0并启动4例专项。
- 2026-08-02：K=768专项首次run `20260802T0555Z_topk768_legacy_cuda_correctness_v1`因固定镜像ENTRYPOINT实际为`/bin/bash`，而命令又直接追加Python二进制与脚本参数，导致bash把Python二进制当脚本并exit126。未进入Python/CUDA、无`result.json`，前后GPU全空闲，不能计为correctness失败或通过。
- 2026-08-02：无效入口边界已由报告2.180和planning提交`cb2166d`通过GitHub HTTPS发布；有效重试必须使用新run ID与显式Python entrypoint。
- 2026-08-02：K=768有效CUDA专项v2自然exit0，Python3.12.13/Torch2.11.0+cu129/CUDA12.9、单张苹果800、legacy/sort/cache环境匹配；8K insertion与32K single-block radix各覆盖random/10LSBits，4/4 case均有768个唯一合法索引，set/value匹配且max abs=0。
- 2026-08-02：固定c349、network none、CUDA不可见容器独立解析有效`result.json`通过；7个原始文件已逐字节固化到2.178本地证据目录并与`/dev/shm`源文件`cmp`一致。退出瞬间GPU0为0 MiB/9%释放尾迹，后续8/8卡恢复0 MiB/0%。
- 2026-08-02：K=768 CUDA correctness结果已由报告2.181与planning提交`cc66320`通过GitHub HTTPS发布；完整模型精度仍未验证，下一阶段必须重新双空闲后运行同协议256题筛选。
- 2026-08-02：K=768 256题快速精度筛选前正式双空闲门禁为05:56:10Z/05:57:15Z、间隔65秒；16条设备行全为0 MiB/0%，两个compute段为空，原始日志360 bytes、SHA256=`7c42ae09457473ddbbb6b7dd1d8e416fb00955aa499f7d56223cb651bf4e91db`。
- 2026-08-02：K=768精度筛选前双空闲门禁已由报告2.182与planning提交`f604b11`通过GitHub HTTPS发布；发布身份恢复clean/upstream前不得启动模型。
- 2026-08-02：2.182发布身份`88ad032`后有效K=768 fast256 run已启动。official_v5静态/namespace preflight与入口内额外双空闲通过；fixed import `cuda_initialized=false`，真实CLI为K=768、TP8、max_model_len8192、max_num_seqs16、legacy/sort、oscar_mla_int2、TRITON_MLA_SPARSE。
- 2026-08-02：K=768 fast256首个10分钟节点为9/256、6正确、66.666667%、0 request failure、2条answer extraction failed但均`evaluator_status=scored`且按0分、0截断。初版monitor误把非空`error_message`都标为failure=2；原始行保留，monitor已重启并拆分`request_failures=0`与`extraction_failures=2`，后续不混淆。
- 2026-08-02：第二个10分钟节点仍为9/256、6正确、66.666667%、0 request failure、2 extraction failure、0截断。server持续16 running/0 waiting、62.4–76.8 generation token/s、KV cache 10.1%–10.5%；单次nvidia-smi恰为0%，但紧接3秒dmon各卡SM 61%–98%，证明是瞬时采样而非停滞/掉卡。
- 2026-08-02：第三个10分钟节点仍为9/256、6正确、66.666667%、0 request failure、2 extraction failure、0截断；同步GPU 61%–97%。8个worker存活，server为16 running/0 waiting、62.4–78.4 token/s、KV cache 16.9%–17.3%，无fatal/OOM；与K1024早期多个节点同一长批次不落新checkpoint的模式一致。
- 2026-08-02：第四个10分钟节点为36/256、14正确、38.888889%、0 request failure、19 extraction failure、16截断，GPU 53%–98%。只读分布诊断首次因`None`/字符串混合键加`sort_keys`而TypeError、未改实验；修正后同一运行已到38题/15正确，38题全scored、16 length/22 stop、19 extraction failure，服务仍16 running/0 waiting。
- 2026-08-02：第五个10分钟节点为47/256、23正确、48.936170%、0 request failure、20 extraction failure、16截断，GPU 62%–97%；服务持续16 running/0 waiting、78.4–80.0 generation token/s、KV cache 8.8%–9.2%，outer pending。中途精度波动仅作进度，不提前作门槛结论。
- 2026-08-02：第六个10分钟节点仍为47/256、23正确、48.936170%、0 request failure、20 extraction failure、16截断，GPU 62%–97%。8 worker存活，服务16 running/0 waiting、78.4–80.0 token/s、KV cache 16.6%–17.0%，确认是下一长输出批次而非停滞。
- 2026-08-02：第七个10分钟节点为74/256、33正确、44.594595%、0 request failure、35 extraction failure、30截断，GPU 53%–98%；服务16 running/0 waiting、75.2–78.4 token/s、KV cache 4.7%–5.2%，outer pending。仍只作进度，不据异步完成子集与历史轮次直接比较。
- 2026-08-02：第八个10分钟节点为77/256、34正确、44.155844%、0 request failure、37 extraction failure、32截断，GPU 61%–98%；服务16 running/0 waiting、78.4–80.0 token/s、KV cache 9.7%–10.1%，outer pending。checkpoint增长慢但服务持续生成，不构成停滞。
- 2026-08-02：第九个10分钟节点仍为77/256、34正确、44.155844%、0 request failure、37 extraction failure、32截断，GPU 61%–98%。8 worker存活，服务16 running/0 waiting、78.4 token/s、KV cache 17.4%–17.8%，确认继续处于长输出批次。
- 2026-08-02：第十个10分钟节点为107/256、44正确、41.121495%、0 request failure、53 extraction failure、47截断，GPU 64%–97%；服务16 running/0 waiting、55.9–78.4 token/s、KV cache约3.9%–4.1%，outer pending。累计精度接近历史保守线，但异步完成子集不能支持最终判断。
- 2026-08-02：第十一个10分钟节点为117/256、50正确、42.735043%、0 request failure、55 extraction failure、48截断，GPU 60%–97%；服务16 running/0 waiting、78.4 token/s、KV cache 9.4%–9.9%，outer pending。继续完整评测，不根据中途回升作最终结论。
- 2026-08-02：第十二个10分钟节点为119/256、52正确、43.697479%、0 request failure、55 extraction failure、49截断，GPU 60%–96%；服务16 running/0 waiting、78.4 token/s、KV cache 15.6%–15.9%，outer pending。尚余137题，继续完整评测。
- 2026-08-02：第十三个10分钟节点为139/256、55正确、39.568345%、0 request failure、67 extraction failure、62截断；节点GPU为0%–65%。服务同秒15 running后恢复16 running/0 waiting、78.4–80.0 token/s，随后3秒dmon为58%–97%、8 worker存活，确认仅为批次切换而非掉卡/停滞。
- 2026-08-02：第十四个10分钟节点为141/256、55正确、39.007092%、0 request failure、69 extraction failure、64截断，GPU 59%–98%；服务16 running/0 waiting、78.4 token/s、KV cache 10.4%–10.8%，outer pending。新增两题均未得分，中途精度仍不作最终淘汰判断。
- 2026-08-02：第十五个10分钟节点为146/256、57正确、39.041096%、0 request failure、71 extraction failure、66截断，GPU 58%–98%；服务16 running/0 waiting、68.7–78.4 token/s、KV cache 14.5%–15.7%，outer pending。尚余110题，继续完整评测。
- 2026-08-02：第十六个10分钟节点为173/256、69正确、39.884393%、0 request failure、83 extraction failure、78截断，GPU 60%–97%；服务16 running/0 waiting、55.9–80.0 token/s、KV cache约5.8%–6.0%，outer pending。尚余83题，继续完整评测。
- 2026-08-02：第十七个10分钟节点为177/256、70正确、39.548023%、0 request failure、85 extraction failure、80截断，GPU 61%–97%；服务16 running/0 waiting、78.4–80.0 token/s、KV cache 11.2%–11.6%，outer pending。尚余79题，继续完整评测。
- 2026-08-02：第十八个10分钟节点为181/256、72正确、39.779006%、0 request failure、87 extraction failure、82截断，GPU 58%–98%；服务16 running/0 waiting、76.7–78.4 token/s、KV cache 13.9%–15.1%，server fatal/OOM扫描0处、8 worker存活、outer pending。尚余75题，继续完整评测。
- 2026-08-02：等待K=768结束期间只读复核K=1,024已通过的`build_evidence.py`：最终封存需逐文件固定SHA256并覆盖runtime/source身份、69项static语义边界中的实际结果、TP8/K/max_model_len/legacy/sort合同、256行唯一ID与prompt hash、256个顺序checkpoint逐条等同、protocol fingerprint、7974输出上限、fatal扫描和`correct>=105 && truncated<=130`保守门槛。K=768结束后只替换本轮实际路径、身份、哈希与结果，不提前硬编码结论。
- 2026-08-02：第十九个10分钟节点为218/256、87正确、39.908257%、0 request failure、100 extraction failure、94截断，GPU 69%–99%；服务16 running/0 waiting、70.3–80.0 token/s、KV cache 5.8%–5.9%，server fatal/OOM扫描0处、8 worker存活、outer pending。尚余38题，继续完整评测。
- 2026-08-02：第二十个10分钟节点为223/256、90正确、40.358744%、0 request failure、102 extraction failure、96截断，GPU 74%–97%；服务16 running/0 waiting、78.4–80.0 token/s、KV cache 10.5%–10.6%，server fatal/OOM扫描0处、8 worker存活、outer pending。尚余33题，达到105题保守门槛还需答对15题，继续完整评测。
- 2026-08-02：第二十一个10分钟节点为227/256、91正确、40.088106%、0 request failure、104 extraction failure、99截断，GPU 58%–96%；服务16 running/0 waiting、78.4–80.0 token/s、KV cache 14.2%–14.4%，server fatal/OOM扫描0处、8 worker存活、outer pending。尚余29题，达到105题保守门槛还需答对14题（48.28%），继续完整评测。
- 2026-08-02：第二十二个10分钟节点为244/256、97正确、39.754098%、0 request failure、114 extraction failure、109截断，GPU 52%–97%；尾批服务由11降至10 running、0 waiting、31.7–55.0 token/s，server fatal/OOM扫描0处、8 worker存活、outer pending。尚余12题，达到105题保守门槛还需答对8题（66.67%），尚未数学淘汰，继续自然完成。
- 2026-08-02：第二十三个10分钟节点为247/256、97正确、39.271255%、0 request failure、117 extraction failure、112截断，GPU 43%–96%；尾批9 running、0 waiting、35.1–41.4 token/s，server fatal/OOM扫描0处、8 worker存活、outer pending。尚余9题，达到105题保守门槛还需答对8题（88.89%），尚未数学淘汰，继续自然完成。
- 2026-08-02：第二十四个10分钟节点为248/256、97正确、39.112903%、0 request failure、118 extraction failure、113截断，GPU 39%–98%；节点后尾批由8降至7 running、0 waiting、20.3–39.2 token/s，server fatal/OOM扫描0处、8 worker存活、outer pending。节点时剩余8题必须全部答对才可达到105题，继续自然完成。
- 2026-08-02：K=768 fast256已于10:09:41Z自然完成，outer exit0；256/256均scored，97正确、37.890625%、0 request failure、126 extraction failure、121截断，最终monitor checkpoint read error0。相对保守BF16 105题少8题，正确数门槛失败；截断121<=130单项通过，整体候选门禁失败，禁止启动该候选32K性能测试。10:10:41Z复核8/8卡0 MiB/0%、compute process为空。
- 2026-08-02：结束核验时宿主`jq`不存在，四个只读JSON展示子命令均以`jq: command not found`无效；宿主直接读取root权限的`fast_runner_state.json`又报Permission denied。两者均未修改证据，已改用固定c349镜像、network none、CUDA不可见、只读挂载完成有效复算。
- 2026-08-02：固定容器独立复算确认256行/256唯一ID、256个顺序checkpoint且逐条等同、97正确、121截断、全部scored、0 request failure；K=768独立证据构建器53/53 checks通过，fresh只读容器32/32 manifest与分类复核通过。证据分类为`performance_candidate_screen_failed`，validation自洽通过不等于候选通过。
- 2026-08-02：正式报告首次追加2.183时因预期末尾换行上下文不匹配被`apply_patch`拒绝，报告hash保持原值且无diff；重新读取精确末尾后以更小锚点成功追加，没有覆盖既有内容。2.1–2.183章节连续，2.174/2.181/2.182引用存在，“三池”0处，“A800”仅历史报告文件名链接一行（label与target共2个子串），算术与`git diff --check`通过。报告现为11,602行、688,018 bytes、SHA256=`b3cdb4feab6c5829077e300b73886b58ee5333115c139535ac194ebbfdb9a738`。
- 2026-08-02：精确`git add -f`只暂存本轮33个证据文件，其他artifacts为0。全量staged `diff --check`只命中4类原始字节证据中的既有尾随空格（GPU采样、runner/serve command、server log）；为保持已固定SHA256不改写原始日志。报告/planning/build_evidence/source_contract/validation/manifest单独diff门禁通过，原始证据继续以fresh容器32/32 manifest校验。
- 2026-08-02：报告2.183、三份planning与K=768的33个独立证据文件已由主仓提交`c36d49a`通过GitHub HTTPS推送。K=768正式淘汰且不启动32K性能；下一步发布本身份恢复clean/upstream，再把候选空间收回已通过精度门槛的K>=1024并继续CPU-only机会排序。
- 2026-08-02：2.183发布身份`8eb2ee9`推送后两仓clean/upstream。下一候选初步源码审计确认`config.index_topk`在模型构造时同时决定共享`topk_indices_buffer`第二维、每层Indexer的`topk_tokens`及稀疏attention消费宽度；现有`VLLM_SPARSE_INDEXER_PREFILL_DECODE_TOPK`只切换prefill的top-k算子实现，不提供独立K值。因此“prefill K768/decode K1024”不是配置级免费候选，需继续追踪buffer切片和attention合同后再判断是否值得实现。
- 2026-08-02：进一步数据流审计确认OSCAR attention已区分prefill/decode：prefill显式切`topk_indices_buffer[:num_tokens, :topk_width]`，decode使用完整metadata宽度；但indexer当前两阶段都用统一`topk_tokens`写buffer。可行的最小解耦必须同时让indexer prefill只生成K=768、OSCAR prefill attention只消费768，而decode仍生成/消费1024；只改其中一侧会读取未写满的`-1`槽或不产生工作量收益，必须fail closed。metadata的full-topk边界是否依赖统一K仍需继续审计。
- 2026-08-02：并发混合batch风险已证实需要结构化处理：Indexer metadata已有`num_decodes/num_prefills/num_decode_tokens`并按“decode token在前、prefill token在后”拆分；XPU/OSCAR attention metadata当前没有这些字段，但builder已导入`split_decodes_and_prefills`，同仓FlashMLA sparse等backend有成熟拆分范式。因此候选可通过给metadata增加4个计数字段并在OSCAR attention分别调用decode K1024与prefill K768；两行式全batch宽度判断会误降混合batch decode，明确淘汰。
- 2026-08-02：新ranking builder首次在写结果前因沿用旧矩阵层级读取`candidate.hf_overrides`而KeyError；改为真实`candidate_hf_overrides`后，第二次有效运行到检查阶段又暴露XPU metadata实际尚未import split helper，与初步人工记忆不符。将其改为“当前缺失、实施需新增”的正向合同后，从头运行22/22 checks通过；两次失败均未生成候选结果且已修正假设。
- 2026-08-02：CPU-only ranking选择`prefill_topk768_decode_topk1024`：模型/shared buffer/decode恢复K1024，只有prefill选择和OSCAR attention消费K768；混合batch必须按decode前/prefill后分段，unset行为保持不变。32K TTFT线性投影为18,706.816225 ms、相对K1024约省2,325.197809 ms，但仍比BF16慢49.473901%，且TPOT因分段开销明确不可外推。固定c349容器22/22、fresh容器3/3 manifest通过；GPU仍禁止。
- 2026-08-02：已重读并实时追加报告2.184，记录split-K数据流、混合batch风险、候选排序、fail-closed合同、22/22 checks与两次builder失败边界。报告2.1–2.184连续，2.174/2.183引用存在，“三池”0处，“A800”仅历史报告文件名链接一行，算术和`git diff --check`通过；现为11,683行、693,512 bytes、SHA256=`ec8adadd88276313e9f011e2e7ec052d5f4b90424d48dcf069fae75f7d143e76`。
- 2026-08-02：报告2.184、4项split-K ranking证据与planning已由主仓提交`ddd38b8`通过GitHub HTTPS推送。下一步只发布本身份并恢复clean/upstream，再按冻结合同先写CPU-only测试取得目标红灯；source和production仍未修改，GPU未开放。
- 2026-08-02：标准`preflight-candidate`入口内部固定`docker run --gpus all`，属于driver-visible preflight，不能按纯CPU阶段直接运行。虽预期`cuda_initialized=false`且不加载模型/请求，仍必须先完成两次间隔至少60秒的8卡空闲检查并先实时记录。
- 2026-08-02：既有正式GPU空闲日志格式已复核：记录MAIN/SOURCE HEAD、first/second UTC时间、8行`index,memory.used,utilization`、两段compute-process空列表和WAIT_SECONDS。K=768将沿用相同格式与独立run ID。
- 2026-08-02：K=768 driver-preflight正式双空闲原始采样时间为05:31:06Z与05:32:11Z，间隔65秒；两次8/8卡均0 MiB/0%，两个compute-process段均为空。当前只完成原始采样，待独立解析/hash和报告发布后才能运行preflight。
- 2026-08-02：双空闲日志独立Perl解析通过：16条设备行全部`index, 0, 0`，2个compute marker且无进程行，interval=65；原始日志SHA256=`c86840e7ef45abfb61e6fb779400c7cfda724581fe87dcb18fd6fbe7457c1c8c`。
- 2026-08-02：Shawn再次明确要求，所有精度优化与性能优化的实际改动、验证结果及失败边界都必须在每个阶段完成后立即写入`OSCAR精度与性能优化记录.md`。后续执行顺序固定为：重新读取报告末尾 → 追加本阶段真实结果 → 检查章节连续性、交叉引用和术语 → 发布阶段；不得等整轮实验结束后集中补写。
- 2026-08-02：split-K v1的候选OCI、独立verification与overlay均在`artifacts/phase6/20260802T1109Z_candidate_1e768aef6_split_topk_v1`；定向列举确认三者同目录。全`artifacts`递归查找import证据会超过短工具窗口，后续只从正式报告和仓内脚本定向恢复旧导入协议。
- 2026-08-02：宿主有`/usr/bin/docker`但无`skopeo`；split-K目标tag `glm52-oscar-a800-phase6-1e768aef6-0275043c:latest`在daemon中明确不存在。历史有效协议是在宿主已有的`ubuntu:22.04`本地image ID基础上启动一次性CPU-only工具容器，安装skopeo 1.4.1后从只读OCI layout复制到`docker-daemon:<tag>:latest`；导入后必须独立核对image ID、33层、末层diff-ID、tag和8项labels，不能以工具日志单独判定成功。
- 2026-08-02：c349历史daemon阶段的落盘合同是`daemon_import.log`、`daemon_import.exit_code`、`daemon_inspect.json`、`daemon_identity_audit.json`；audit的5类checks为image ID、labels、last diff-ID、layer count和tag。split-K导入沿用该合同，不提前执行driver/native runtime import。
- 2026-08-02：skopeo 1.4.1的`copy`不支持`--preserve-digests`；首次split-K命令因此在任何blob复制前exit1，目标tag仍不存在。历史成功日志本来也没有该flag的证据，重试应使用`oci:<layout>:<ref>`到`docker-daemon:<tag>:latest`的普通copy，并继续依赖导入后的digest/identity审计确保内容未漂移。
- 2026-08-02：split-K v1已由普通skopeo1.4.1 copy成功导入daemon；有效log/exit/inspect/audit SHA256分别为`3abcf882...c5df`/`9a271f2a...86aa`/`3141de8e...a733`/`f0c68df4...2763`。daemon image/config=`a5f5c4d5...b5aa`，33层、末层diff-ID=`b9c16c81...aa45`、tag和8项labels全部匹配，audit=`passed`。报告2.191已实时追加；本结论没有runtime/GPU/精度/TTFT/TPOT含义。
- 2026-08-02：Stage7/9活动身份消费者与历史/基线引用必须分开。应迁移的是Phase7候选manifest与candidate运行脚本、Phase9 Dockerfile/performance matrix/container wrapper；`configs/phase1/native_baseline.json`、Phase5冻结输入、native baseline脚本以及已封存artifact内的c349证据都是历史/基线，不得批量替换。Stage9新control image ID必须实际build/inspect后才能写入performance matrix；Phase7的`runtime_import.json`又必须等driver-visible native probe后才能形成，因此静态链合理顺序是先只迁移Stage9 Dockerfile base并构建/审计新control image，再进入driver runtime import和剩余manifest/runner迁移。
- 2026-08-02：Stage9 Dockerfile base身份目标测试在旧production上有效失败1项，无import/error；最小生产改动仅需替换第一行ARG BASE_IMAGE到`glm52-oscar-a800-phase6-1e768aef6-0275043c:latest`，不需要改apt/RUN/Entrypoint。
- 2026-08-02：Stage9 base切换后目标测试1/1、完整工具22/22通过；输入审计13/13确认Dockerfile、两仓published身份、新base image/33层/diff-ID/source/tree/candidate-layer及目标control tag不存在。新control image ID尚未产生，performance matrix与Phase7 runtime import继续保持未迁移边界。
- 2026-08-02：新Stage9 control build已成功，image ID=`sha256:c92a1245...a12e`、34层、最后diff-ID=`sha256:07b4495b...7afa`，前33层/labels/Entrypoint继承审计10/10通过。CPU runtime本身exit0，但输出文件前部含vLLM INFO stdout、最后一行才是JSON；直接整文件`json.loads`的三次失败不是NFS可见性问题，也不是runtime失败。正确修复是不重跑容器，原样保留混合stdout并提取最后一行作为规范JSON再独立复核。
- 2026-08-02：规范CPU JSON与identity已由固定c349容器经stdin独立复核通过；Python/PyTorch/glibc、Git/iproute2、source origin、两处K768、四metadata字段及CUDA前后false均匹配。无GPU导入日志中的libcuda缺失由vLLM捕获，因此只能认定CPU source/runtime通过，不能认定native driver import。
- 2026-08-02：split-K driver/native import前双空闲为11:45:50Z/11:46:55Z、间隔65秒；两次8/8卡均0 MiB/0%、compute为空，宿主与固定容器7/7复核通过。报告2.194已实时记录；发布前不得启动GPU0探针。
- 2026-08-02：新control image的GPU0 driver/native import自然exit0，`vllm._C`解析到新source路径，两处prefill K=768、四metadata字段和artifact身份通过，CUDA前后未初始化，退出后8卡全空闲。canonical runtime_import已由本轮GPU结果+CPU实测生成，717 bytes/SHA=`9bdfc8ca...3b20`；相同hash源于环境合同未变，不代表复用旧运行。
- 2026-08-02：活动Phase7/9身份测试有效红灯精确命中旧c349 source；最小迁移范围为Phase7 manifest及candidate入口、Phase9 matrix/container wrapper与source-matched native wrapper。Phase1/5和历史artifact引用必须保留，不能用全仓`c349`清零作为成功标准。
- 2026-08-02：Phase7 verifier会递归运行Phase5/Phase1并读取绝对模型路径；CPU-only控制容器必须把宿主模型目录按相同绝对路径只读挂载。首次遗漏导致FileNotFoundError，不涉及本次身份内容，重试需只补该mount。
- 2026-08-02（split-K静态身份迁移递归依赖）：补挂模型目录后的Phase7正式verifier完整生成44项检查，但`stage5_preflight.exit_status/status`两项失败、其余42项通过。根因是Phase7会递归执行活动Phase5预检，而`configs/phase5/oscar_tp8.json`与`scripts/phase5/run_oscar_tp8.sh`仍约束c349 source；因此Phase5不是本轮可保持不变的纯历史配置，必须最小迁移其source身份并同步Phase7中的stage5 manifest哈希。Phase1 baseline与封存artifact仍保持历史身份。
- 2026-08-02（Stage9 control测试入口合同）：`oscar-glm-stage9-runtime:1e768aef6`的Config Entrypoint为`["/bin/bash"]`、Cmd为空；运行Python工具测试必须显式覆盖`--entrypoint /usr/bin/python3`，不能直接在镜像名后追加`python3 ...`。
- 2026-08-02（递归verifier的Git所有权合同）：固定control容器以宿主UID运行时，Phase5会对`glm52_oscar_vllm`执行Git命令；必须通过`GIT_CONFIG_COUNT=1`、`GIT_CONFIG_KEY_0=safe.directory`、`GIT_CONFIG_VALUE_0=<submodule绝对路径>`传入只读安全目录声明，否则会在功能检查前因dubious ownership失败。
- 2026-08-02（Phase0 source的NFS模式边界）：宿主`artifacts/phase0-candidate-bundle/rootfs/opt/vllm_glm52_v1`在当前NFS上把普通文件呈现为0777，会令Phase1/5严格Git mode校验误报；正式Stage9 wrapper已定义volume `oscar-glm-phase0-source-fd3e0b3`。只读审计该volume中`.buildkite/.pipeline_gen_v2`为0644、source项数4,717；递归verifier必须将它覆盖挂载到Phase0 source绝对路径，同时保留rootfs其余文件来自宿主只读目录。
- 2026-08-02（split-K活动身份链最终状态）：活动Phase5/7/9配置及6个正式wrapper已统一到source `1e768aef6`；Phase5仍保留Phase1 manifest内独立的旧runtime source身份，因而既能校验当前repo HEAD，又能通过专用volume校验冻结基础rootfs。Phase7/Phase9递归结果分别44/44、70/70，工具回归20/20、88/88，独立静态汇总34/34；这些只证明启动前身份/合同闭合，不产生新的准确率、TTFT或TPOT。
- 2026-08-02（报告2.196）：正式报告已实时记录split-K活动身份迁移与所有执行边界；发布门禁为2.1–2.196连续、核心证据大小/hash全匹配、manifest复算通过、术语和diff通过。报告本节明确没有新增准确率、TTFT或TPOT，避免把静态身份验收写成性能收益。
- 2026-08-02（split-K性能前GPU门禁）：新身份发布后GPU 0–7在12:16:00Z与12:17:05Z连续两次全空闲，间隔65秒、无compute进程；该结果仅授权后续固定8卡warm-up/32K benchmark，不是性能或精度数据。
- 2026-08-02（split-K 32K/batch1正式性能）：run `20260802T122000Z_split_topk_32k_b1_v1`固定GPU 0–7、TP8、input 32,768、batch/concurrency 1、output 128；每轮含1次warm-up和3次正式请求，三轮及profiler均passed。三轮中位聚合mean TTFT=`17880.08132359634 ms`、mean TPOT=`199.60554782790072 ms`、request throughput=`0.023115279763918164 req/s`，三项轮间相对范围分别0.181452%/0.348033%/0.201524%；无waiting/preemption，正式峰值显存80,769 MiB/GPU。
- 2026-08-02（性能对比边界）：相对同负载c349 K=1,024，split-K的TTFT改善14.986357%、TPOT回退0.954252%、请求吞吐改善6.776920%；相对同负载c349 BF16，TTFT仍慢42.868005%、TPOT仍慢29.833407%、请求吞吐低25.937991%。BF16/K1024控制组source为`c349e32e...`，split-K为`1e768aef6...`，只能作跨提交同负载参考；最终同源码结论必须补跑1e BF16。
- 2026-08-02（profile与证据）：profile elapsed 738.078620 s，8个worker trace、8个rank CUDA表和1个frontend trace齐全；critical rank=6、self CUDA total=50,559 ms。服务日志仅有1次已知非致命`External init callback must run in same thread as registerClient`，profile API仍200且结果完整。最终证据71文件、1,217,287,442 bytes，manifest SHA256=`3366214ac12be73d9b57a2a6b6bb9928117d81be842712b6b395f21cda0c7a4f`，独立validation 59/59通过。
- 2026-08-02（执行错误边界）：一次只读查询命令因shell引号错误未执行，不影响实验；持久化时尝试复制可选顶层startup 10分钟日志报不存在，原因是startup在10分钟阈值前ready，正式run内10/20/30分钟进度均存在；此前口头preview误用了逐请求median，正式报告口径已纠正为三轮mean TTFT/mean TPOT的中位数。profiler实际为1 warm-up+1 profiled request，不是4个请求。
- 2026-08-02（持久化核心哈希查询路径纠正）：首次手工`sha256sum`把不存在的Phase9顶层`summary.json`、旧式`cells/.../summary.json`及profile顶层`profile_validation.json/server.log`当成当前目录布局，四项报`No such file or directory`；同一命令随后的71文件manifest复算仍全部OK。下一步只用`rg --files`恢复当前真实布局后核对核心文件，不重复错误路径；这属于只读查询错误，不影响任何证据。
- 2026-08-02（报告2.198）：已重新读取2.197末尾并实时追加split-K正式三轮、profile、相对K1024/BF16对比、跨source边界、三类执行错误及71文件证据。报告2.1–2.198连续，“三池”0处，大写旧称仅历史报告链接第5行（label/target共2个子串），2.197/2.198引用、核心大小/hash及`git diff --check`通过；当前12,495行、744,864 bytes、SHA256=`0d0e94486cbfa9fd4ae621a85b18301f92d6170c1728cf901381501f4154d3af`。
- 2026-08-02（2.198发布）：正式性能报告与planning已由主仓提交`c8b5d1b4cfd1b650083162145505c1e200f105cf`通过GitHub HTTPS推送；下一步只发布本身份检查点并确认两仓clean/upstream，再开始同源码BF16前双空闲门禁。
- 2026-08-02（同源码BF16入口复核）：2.198发布身份为`eaf1d4835cb13119b15ef1986b8effe047cd2189`，两仓clean/upstream。`run_containerized_performance.sh baseline`与`run_native_tp8.sh`均绑定source `1e768aef6...`；baseline只把KV cache dtype设为`auto`，不注入candidate的split-K环境/HF override。`STAGE9_ONLY_CELL=32768:1`沿用同一TP8、output128、三轮、每轮1 warm-up+3正式请求及1 warm-up+1 profile请求，满足同源码同负载对照入口合同。
- 2026-08-02（同源码BF16前双空闲原始结果）：GPU 0–7在13:06:47Z与13:07:52Z两次均为0 MiB/0%，两个compute区段为空，间隔65秒；主解析9/9通过。原始run目录为`20260802T130800Z_source_matched_bf16_performance_idle_v1`。
- 2026-08-02（独立复核stdin失败）：首次固定control容器独立复核命令漏加`docker run -i`，使Python `-`从EOF退出0且输出0-byte `gpu_idle_independent_validation.json`；随后错误manifest把空文件当作字节对象校验通过，但不代表语义验证通过。必须保留/记录0-byte边界，下一步增加`-i`、实际断言独立JSON非空/status passed后重建manifest，不能直接进入BF16。
- 2026-08-02（报告2.199/有效门禁）：只增加`docker run -i`后的独立复核7/7通过，最终manifest覆盖12个文件且全部复算通过；报告实时追加入口身份、双采样、首次0-byte失败与有效重试。报告2.1–2.199连续，“三池”0处，大写旧称仅历史报告链接第5行，2.198/2.199引用、证据hash和diff通过；当前12,555行、748,963 bytes、SHA256=`2efb0e913f6a4be70fa0ca482f0c8e0e79331901274632cdb55bad703ce53ca9`。
- 2026-08-02（2.199发布）：同源码BF16前双空闲门禁报告与planning已由主仓提交`cd05f65cbdfb1f92662d59797dd35a73d67324af`通过GitHub HTTPS推送；下一步只发布身份检查点并恢复clean/upstream，再即时复核8卡后启动正式baseline。
- 2026-08-02（同源码BF16首次正式入口失败）：2.199身份`dfe4149`发布后即时8卡复核通过，run `20260802T131400Z_source_matched_bf16_32k_b1_v1`在模型加载前exit1。Phase1 verifier的`source.repository_commit/tree`仍由冻结`configs/phase1/native_baseline.json`约束c349，而实际repo/performance config已为1e，导致3项精确失败；其余模型/OCI/evaluator检查通过。server未ready、未加载模型，退出后8卡0 MiB/0%、compute为空。
- 2026-08-02（根因与修复边界）：`run_native_tp8.sh`虽把`EXPECTED_SOURCE_COMMIT`更新到1e，但仍直接export Phase1冻结manifest；先前入口复核只检查wrapper常量而遗漏MANIFEST内部source，是不完整判断。正确方向不是修改历史Phase1 manifest，而是建立Stage9专用source-matched baseline manifest/派生合同，只更新当前repository commit/tree及performance source关联，保留runtime source/OCI/model/evaluator冻结身份；先CPU-only TDD，禁止原样重跑GPU。
- 2026-08-02（派生方案初选）：最小且避免复制整份冻结manifest的方案是在`verify_native_performance.py`内深拷贝Phase1 base manifest，只用Phase9 performance config固定的source commit及该commit解析出的Git tree覆盖effective manifest的`repository_commit/tree`，再调用Phase1现有OCI/source/model/suite verifier；原始Phase1文件保持逐字节不变。结果JSON需同时记录base manifest hash、base/effective source身份和派生模式，避免把内存覆盖误写成冻结文件已迁移。
- 2026-08-02（只读检索错误）：一次`rg`命令包含不存在的shell字面路径` scripts/phase1/test*`，产生`No such file or directory`；其余指定文件读取已完成，不影响任何文件。后续只对实际存在的`test_phase9_tools.py`添加目标TDD，不重复错误glob。
- 2026-08-02（报告2.200）：已实时记录同源码BF16 v1在模型加载前74/77静态检查、3项source失败、GPU全程空闲、证据hash、入口误判纠正及Stage9内存派生方案。报告2.1–2.200连续，“三池”0处，大写旧称仅历史报告链接第5行，2.199/2.200引用、6文件manifest和diff通过；当前12,616行、753,025 bytes、SHA256=`155607fb48eab33b76a2cf586a18d69e911737243ed94377dff7a77a7816e2b8`。
- 2026-08-02（2.200发布）：失败边界报告与planning已由主仓提交`b5f31918f58c419a5ef794a11055823af95f7c4a`通过GitHub HTTPS推送；下一步发布身份检查点后进入CPU-only TDD。
- 2026-08-02（派生manifest TDD）：目标测试在固定control容器/network none/无GPU下先以唯一预期AttributeError得到1 error；新增纯函数深拷贝base、只覆盖repository commit/tree并返回审计，main用performance source commit解析commit tree后调用。目标1/1、完整Stage9工具24/24绿灯；Phase1 manifest与performance config均未改。
- 2026-08-02（CPU-only静态闭合）：正式模型只读挂载、Phase0 source volume和Git safe-directory环境下，Stage9 native verifier 77/77具体checks passed；结果记录base manifest SHA=`9bcc6be8...100e`、base c349/60d5与effective 1e/178a，Phase1文件`git diff`为空。
- 2026-08-02（证据manifest路径错误）：首次在项目根以`find "$VERIFY_DIR"`生成manifest，条目带`artifacts/...`前缀，随后`cd "$VERIFY_DIR"`复算会把前缀再次相对拼接，3个文件均报找不到。verifier自身exit0/JSON passed不受影响；下一步从证据目录内`find .`重建，不能重复错误目录上下文。
- 2026-08-02（报告2.201）：派生manifest红绿灯、最小实现、24/24工具、77/77静态闭合、Phase1未改边界和首次manifest错误已实时追加。最终12,690行、757,645 bytes、SHA256=`845cb2a9f0067b8aaa2ac3826f685aba23ec2632a45719c2959f8971191f30f3`；2.1–2.201连续，“三池”0处，大写旧称仅历史报告链接第5行，2.200/2.201引用、代码/证据hash、最终manifest和diff通过。
- 2026-08-02（2.201发布）：派生manifest修复、测试、报告与planning已由主仓提交`9ace8fdc81dcd88f8f0f92d77fc663151891c2a3`通过GitHub HTTPS推送；下一步只发布身份检查点并恢复clean/upstream，再重新执行双空闲门禁。
- 2026-08-02（修复后BF16 v2前双空闲）：2.201身份`a876b244a1b63140c11b02e01ded5568102531d7`发布后两仓clean/upstream；GPU 0–7在13:25:03Z/13:26:08Z两次均0 MiB/0%，compute为空，间隔65秒。主9/9、固定control容器独立7/7及9文件manifest全部通过；本阶段未启动模型。
- 2026-08-02（报告2.202）：修复后新双空闲门禁已实时追加；报告2.1–2.202连续，“三池”0处，大写旧称仅历史报告链接第5行，2.201/2.202引用、8项核心证据hash、9文件manifest和diff通过。当前12,738行、760,628 bytes、SHA256=`bd22798148065271d317d1e4f1163b998209ebedc8e6361618c3d4c221e3242f`。
- 2026-08-02（2.202发布）：修复后双空闲门禁报告与planning已由主仓提交`a80edd362f09c1e9c1a844883758a34f5e13d765`通过GitHub HTTPS推送；下一步只发布身份检查点并恢复clean/upstream，再即时复核8卡启动baseline v2。
- 2026-08-02（同源码BF16 v2正式完成）：2.202身份`a3f059ef5c6e2bb543c13c995e6f869630a25718`发布后，固定GPU 0–7、TP8、32K/batch1/output128正式三轮及profiler全部passed，outer exit0。三轮中位聚合mean TTFT=`12507.854171097279 ms`、mean TPOT=`151.23601672862927 ms`、request throughput=`0.03153346393427157 req/s`；轮间相对范围0.149959%/0.896002%/0.491328%。
- 2026-08-02（BF16 v2 profile/资源）：profile elapsed 596.640517 s，8 worker trace、8 rank表、1 frontend trace；critical rank5/self CUDA total 38,228 ms，1,525 GPU样本，profile峰值80,385 MiB/GPU，无waiting/preemption。服务唯一ERROR为既有非致命external-init thread提示；无traceback/OOM。容器删除后的即时采样显示8卡显存均为0 MiB、compute为空，但利用率仍为100%的尾迹；约107秒后的独立稳定采样才是8卡0 MiB/0%、compute为空。
- 2026-08-02（同源码正式对比）：source与performance config完全相同的32K/batch1下，split-K相对BF16的mean TTFT绝对/相对差为`+5372.227152 ms/+42.950830%`，mean TPOT为`+48.369531 ms/+31.982812%`，request throughput为`-0.008418184 req/s/-26.696034%`。旧c349 BF16相对本轮1e BF16的三项漂移仅`-0.057940%/-1.628550%/+1.034109%`，说明先前跨提交参考方向未被推翻，但正式结论应以本轮同源码数据为准。
- 2026-08-02（BF16 v2证据闭合）：最终目录71文件、1,041,886,525 bytes，其中69项证据manifest复算通过；`formal_validation.json` 73/73 passed。`comparison.json`/validation/manifest SHA256分别为`f50dad67...57177`/`fd8d55d3...77d81`/`927d918b...410b0`。当前性能目标未完成，下一步必须基于同源码profile做CPU-only差异归因后再选优化点。
- 2026-08-02（报告2.203）：修改前完整读取报告且与HEAD一致；正式同源码BF16结果、split-K差距、即时100%利用率尾迹、稳定0%采样、证据闭合和“无新精度”边界均已实时记录。最终12,830行、766,852 bytes、SHA256=`b1ec30c3...ff3c5`；2.1–2.203连续，术语、引用、11项核心证据、73/73 validation、69/69 manifest及diff通过。
- 2026-08-02（同源码prefill归因）：BF16/split-K各8 ranks、16 chunks和32,768 tokens完整。start→next-start尾迹校正20/20通过，wall gap=`5893.667641 ms`，解释profile TTFT gap 100.269%；attention净差=`2674.318982 ms`，只解释正式TTFT差49.780%，rotation=`1494.044451 ms`，非attention kernel差=`2122.908463 ms`，剩余非kernel残差=`1096.440196 ms`。
- 2026-08-02（残差旧阈值失败边界）：复用脚本首次仅因硬编码旧K1024的`kernel gap > 7000 ms`失败，实际split-K gap=`4797.227444 ms`；16 trace、kernel/attention重放、1248 calls和2512 AllReduce均通过。保留失败JSON/log/exit1后，只把门禁改成正gap，计算逻辑未改，有效重试14/14 passed。
- 2026-08-02（同源码decode归因）：每rank 127 generation contexts完整，使用前126个相邻起点完整区间校正异步尾部后，wall gap=`59.669254 ms/token`，解释profile TPOT gap 100.101%；kernel gap=`52.114410 ms/token`。两侧均157次AllReduce/token，观测时间差=`47.916249 ms/token`；candidate/baseline专属kernel为18.643515/14.898042 ms/token，净直接差仅3.745473 ms/token。rank间AllReduce采样高度不对称，因此该差是同步等待证据，不足以证明网络带宽回退。
- 2026-08-02（trace证据闭合）：汇总32/32、独立13/13通过；目录39文件、9,656,754 bytes，37项manifest全部复算通过，summary/validation/manifest SHA分别为`6e80645e...0cc9`/`cda6197a...e089`/`487e71b0...0167`。全程CPU-only/network none/CUDA不可见，结束后8卡0 MiB/0%、无compute。
- 2026-08-02（报告2.204）：修改前完整读取报告并确认与HEAD一致；同源码prefill尾迹、kernel残差、decode相邻起点校正、旧阈值失败边界、AllReduce解释边界和无新精度/正式性能样本均已实时记录。最终12,932行、773,848 bytes、SHA256=`2b3ad795...bd4d7`；2.1–2.204连续，术语、引用、11项核心证据、32/32、13/13、37/37 manifest及diff通过。
- 2026-08-02（2.204后rotation调用几何源码审计）：每个MLA attention layer从本层artifact加载独立`_oscar_rotation`，并注册本层连续布局的`_oscar_inverse_rotation`；二者随该层迁移到权重设备。decode每层依次包含recent demotion的forward rotation、query forward rotation和merged-history inverse rotation，78层恰为`78*3=234`次；prefill首块无demotion而为每层3次，后续15块为每层4次，恰为`78*(3+15*4)=4914`次。各输入都由当前层KV/query/history产生，不能跨层或跨chunk缓存结果。
- 2026-08-02（2.204后rotation内存寿命初审）：recent demotion的BF16 gather与FP32 rotated scratch已由`TritonMLASparseImpl`按容量缓存并切片复用；非decode current-history写入调用`oscar_mla_rotate_quantize_store`时尚未传`rotated` scratch，因而由`oscar_mla_rotate`分配FP32中间张量。稀疏attention内的query rotated、mid/output以及inverse-history输出也按调用分配。下一步需先核对current-history精确行数/调用边界、CUDA graph寿命和现有测试，才能判断“仅复用scratch”是否构成安全且可测的最小优化；它只能消除分配，不能消除已实测约1.494秒rotation计算。
- 2026-08-02（2.204后同步边界初审）：OSCAR store/demotion/sparse-attention路径未引入自定义collective；模型attention输出仍经`RowParallelLinear`，随后decoder layer进入MLP，解释每层固有的attention/MLP tensor-parallel AllReduce。两侧校正后collective调用数完全相同，因此当前47.916249 ms/token差应继续表述为上游候选路径导致的rank到达时间/同步等待差，不足以归因NCCL或网络实现回退。
- 2026-08-02（否决每层current-history scratch）：32K单请求除首块外的current-history通常最多约1,792行，单层FP32 rotated缓冲为`1792*512*4=3,670,016 bytes=3.5 MiB`；若78个layer backend各自常驻会固定`286,261,248 bytes=273 MiB/GPU`。现有调用逐层串行且临时tensor退出作用域后由PyTorch caching allocator复用；新增常驻scratch只可能减少Python分配器路径，不减少rotation kernel/显存读写，并会提高固定显存，故不作为当前最小生产候选。
- 2026-08-02（inverse rotate+add最小候选）：当前每层稀疏attention先将`history_merged @ inverse_rotation`写入独立FP32 `history_original`，随后`_add_outputs_kernel`再读取`bf16_merged`和`history_original`并写最终`output`。候选边界是仅在inverse分支让rotation kernel在FP32 accumulator完成后直接加同shape的`bf16_merged`并写最终output，保留query/store/demotion rotation、IEEE dot、连续inverse、attention/merge/collective语义不变；理论上每次可消除一个FP32中间张量和一个独立add launch。必须先用CPU静态合同与CUDA oracle锁定shape/device/FP32加法、fallback及逐值容差，再做微基准和32K/batch1验证，当前尚未形成实测加速。
- 2026-08-02（候选内存量级）：32K prefill chunk在TP8下query/inverse行数为`2048*8=16384`，单个`16384*512*4=33,554,432 bytes=32 MiB`的FP32中间张量；融合候选不常驻新增scratch，而是直接复用调用方原本就必须提供/分配的最终output。decode batch1则每层8行，收益主要是234次/token中的78个独立add launch，而不是大张量容量。
- 2026-08-02（同源码add实测成本）：source-matched split-K trace中，prefill `_add_outputs_kernel`为1,248 calls（78层×16 chunks）、总计70.3914205 ms；decode相邻完整区间为78 calls/token、中位0.15836975 ms/token。它们给出“移除独立add launch”的直接已观测成本，但融合后的rotation寄存器/occupancy变化尚未知，不能把这些数值直接写成未来端到端收益；候选只能按微基准和正式32K实测判断。
- 2026-08-02（source audit有效轮次）：CPU-only分析器对source commit/tree、本层rotation加载、连续inverse、FP32 IEEE、demotion scratch、current-history未传scratch、inverse中间张量/独立add、prefill/decode调用方程、157次AllReduce及两项内存量级执行16/16检查，全部passed、exit0。核心输出`source_audit.json`为5,506 bytes、SHA256=`53d393ea...afd57`；`validation.json`为1,847 bytes、SHA256=`5516de54...5d13`。尚未改production、未使用GPU，也没有新增精度或端到端性能结果。
- 2026-08-02（source audit独立复核）：首轮独立脚本因目录父级解析错误在`git rev-parse`处exit128，未生成独立JSON；失败exit/log/首次manifest均保留。仅修正`HERE.parents[4]→HERE.parents[3]`后，有效独立复核11/11 passed、exit0；`independent_validation.json`为2,112 bytes、SHA256=`e2493814...fe05`。最终manifest覆盖13个有效与失败边界文件并13/13复算通过，manifest为1,223 bytes、SHA256=`5cc06221...6e19`。
- 2026-08-02（报告2.205）：修改前完整读取报告并确认与HEAD一致；已实时记录rotation调用几何、缓存/NCCL否决、inverse rotate+add单一候选、已观测add成本、验证与失败边界。最终13,038行、781,140 bytes、SHA256=`ed72f280...486f6`；2.1–2.205连续，术语、引用、7项核心证据、16/16、11/11、13/13 manifest及diff通过。
- 2026-08-02（2.205发布）：报告与planning已由主仓提交`a4d685af2964d00e4835e2f5af950fc83013a3d7`通过GitHub HTTPS推送；主仓HEAD=upstream，source仍为`1e768aef6`且clean/upstream。下一步按已发布合同进入inverse rotate+BF16 add融合的CPU/TDD，GPU仍未开放。
- 2026-08-02（inverse-fusion TDD红灯）：source只修改`tests/oscar_mla/test_runtime_activation.py`，新增合同锁定`oscar_mla_rotate_add(latent, rotation, addend, *, output=None)`、sparse主路径必须调用新helper且不再出现`_add_outputs_kernel`。宿主source `.venv`及固定control image的系统Python都没有pytest；宿主目标命令在收集前退出，不计红灯。改用同一control image、network none、`CUDA_VISIBLE_DEVICES`空、source只读挂载的标准库断言harness后，精确在首项以`AssertionError: missing oscar_mla_rotate_add`失败，构成有效红灯；production文件未改、GPU未使用。
- 2026-08-02（inverse-fusion红灯source发布）：目标测试已由source提交`5f03c7491d8e956d58b5ed96a1f089bcf39101d3`通过GitHub HTTPS推送，tree=`016be5bd...837e`；source pre-commit全部通过。该提交有意只包含合同测试，production仍为红灯。
- 2026-08-02（报告2.206门禁）：报告最终13,083行、783,965 bytes、SHA256=`1104bbc0...1168`；2.1–2.206连续，“三池”0处，大写旧称仅历史文件名链接，测试/production哈希、source commit/tree/upstream、pre-commit和两仓diff门禁通过。
- 2026-08-02（inverse-fusion最小实现/静态绿灯）：store kernel新增可编译期关闭的FP32 addend分支和`oscar_mla_rotate_add(latent, rotation, addend, *, output=None)`；普通`oscar_mla_rotate`继续走无addend分支。decode只把`history_original=oscar_mla_rotate(...)`加独立`_add_outputs_kernel`替换为融合helper，并删除新改动造成的孤立add kernel。三文件py_compile、source diff check及固定control image无GPU合同harness通过；尚未运行CUDA oracle或性能实验，不能声称数值/性能通过。
- 2026-08-02（inverse-fusion首次source commit门禁失败）：Ruff check/format、typos、mypy等均通过，但SPDX hook发现历史`test_triton_store.py`无header并自动补齐，commit自然失败且未推送。该文件因本轮新增CUDA门禁才进入changed-file检查范围；后续只暂存hook修正并重跑完整门禁，不跳过hook。
- 2026-08-02（inverse-fusion source绿灯发布）：机械SPDX修正后完整source pre-commit全部通过，production与CUDA测试提交`d0d22489b265fc98f9f829dbcfca5e815543d337`已通过GitHub HTTPS推送，tree=`d07b4992...a1ad`、patch SHA=`7a7153bb...070a`。CPU合同与Triton interpreter已通过，GPU门禁仍未执行。
- 2026-08-02（报告2.207门禁）：最终13,147行、787,890 bytes、SHA256=`449ce199...2cf5`；2.1–2.207连续，“三池”0处，大写旧称仅历史文件名链接，引用、4项文件hash、source身份/patch、CPU合同/interpreter、pre-commit与diff门禁通过。
- 2026-08-02（inverse-fusion GPU前双空闲）：正式first=`15:10:43Z`、valid second=`15:12:17Z`，间隔94秒；两轮GPU0–7全部0 MiB/0%、compute空。中间`15:11:40Z`仅距first 57秒，虽全空闲但不计有效并保留。CPU validation 11/11 passed、7文件manifest 7/7通过；目录9文件/12,004 bytes，GPU容器未启动。
- 2026-08-02（报告2.208门禁）：最终13,180行、790,019 bytes、SHA256=`f0eedc15...34d6`；2.1–2.208连续，“三池”0处，大写旧称仅历史文件名链接，引用、5项证据hash、11/11、7/7 manifest与diff通过。
- 2026-08-02（inverse-fusion GPU0 correctness）：2.208由主仓`b60bb1faa30b927bb82297165dcf94e2275d882a`发布后，即时8卡复核全idle；固定GPU0、network none的control容器自然exit0。8行decode与16,384行prefill两例均与旧rotate+FP32 add bitwise equal、max error0、output pointer复用；source/hash/torch2.10.0+cu129/CUDA12.9/device identity匹配。退出后显存0且compute空，GPU0利用率有12%瞬时尾迹，需补稳定采样。
- 2026-08-02（inverse-fusion GPU0 correctness证据闭合）：15:16:47Z稳定采样8卡全部0 MiB/0%、compute空；14/14 validation和11/11 manifest通过。correctness子目录13文件、15,944 bytes；result SHA=`75519ade...602b`、validation SHA=`7dac9497...97fb`、manifest SHA=`8aabdbc3...d379`。下一步实时写报告2.209，发布前不运行性能微基准。
- 2026-08-02（报告2.209门禁）：最终13,234行、793,297 bytes、SHA256=`95d3c461...1569e`；2.1–2.209连续，“三池”0处，大写旧称仅历史文件名链接，引用、7项证据hash、14/14、11/11 manifest与diff通过。
- 2026-08-02（inverse-fusion微基准前双空闲）：2.209由`4495bf899b52a82022fb2d99a4da301fb80d1cf7`发布后，新first/second=`15:19:45Z/15:20:47Z`、间隔62秒；两轮GPU0–7全0 MiB/0%、compute空。9/9 validation和7/7 manifest通过；benchmark脚本CPU compile及hash固化。手工wc/hash glob误含子目录报错，但显式manifest不受影响。
- 2026-08-02（报告2.210门禁）：最终13,277行、795,924 bytes、SHA256=`2c52797e...0788`；2.1–2.210连续，“三池”0处，大写旧称仅历史文件名链接，引用、5项证据hash、9/9、7/7 manifest与diff通过。
- 2026-08-02（inverse-fusion GPU0微基准有效结果）：2.210由`e50299fb37707dd822281c349d99c6c7b3b26602`发布后即时8卡全idle，固定GPU0 benchmark自然exit0。10次warm-up、9样本交替顺序下，8行旧/融合中位0.05681664/0.02765312 ms（-51.329189%），16384行0.91018238/0.85391359 ms（-6.182144%）；warm-up后bitwise，peak delta分别16KiB→0、32MiB→0。退出0 MiB/compute空但GPU0 9%尾迹，需稳定采样/validation。
- 2026-08-02（inverse-fusion微基准证据闭合）：15:24:29Z稳定采样显示GPU 0–7全为0 MiB/0%、compute为空；结构化validation 16/16 passed。最终manifest显式覆盖包括宿主`py_compile`生成pyc在内的12项，12/12复算通过；子目录共14文件、44,331 bytes，manifest SHA256=`b87b56c7...3aa8`。按调用几何作受限投影，decode每token约省2.274755 ms，仅为当前正式TPOT差48.369531 ms的4.702867%；prefill约省70.223447 ms，仅为正式TTFT差5,372.227153 ms的1.307157%。该投影不是端到端实测，说明单项融合不足以消除性能差距。
- 2026-08-02（报告2.211发布门禁）：修改前完整读取795,924-byte报告并确认与HEAD逐字节一致；追加并更正manifest实测大小后，最终13,343行、800,549 bytes、SHA256=`94c4c1d0...3350`。2.1–2.211连续，“三池”0处，大写旧称仅历史文件名链接，2.205/2.207/2.210/2.211引用、16/16 validation、12/12 manifest及`git diff --check`全部通过。
- 2026-08-02（inverse-fusion正式镜像链只读审计）：现有Phase6/Stage9正式候选仍冻结source `1e768aef6...`，本机对应镜像ID分别为`a5f5c4d5...fab5aa`和`c92a1245...3a12e`；production融合提交为已发布`d0d22489...`/tree `d07b4992...a1ad`。旧镜像只读挂载新source可做局部CUDA门禁，但不满足正式256题/32K证据的内嵌源码身份。按既有迁移链路，先更新Phase6输入/Dockerfile/合同并发布，再确定性构建、验证、导入Phase6 OCI，随后构建Stage9 control image，最后迁移Phase5/7/9活动配置和wrapper；禁止直接以旧1e control image跑正式候选。
- 2026-08-02（inverse-fusion Phase6输入绿灯）：candidate manifest/Dockerfile/定向合同三文件已冻结source `d0d22489...`、tree `d07b4992...`、tag `glm52-oscar-a800-phase6-d0d22489b-0275043c`；文件SHA依次为`7ee20619...d7f1`、`6add4345...96ad`、`0abba9f4...1f47`。有效红灯精确命中旧commit，绿灯目标1/1、完整2/2、独立身份6/6通过；source仓clean/upstream。未构建OCI、未改Phase5/7/9活动配置、未使用GPU。
- 2026-08-02（报告2.212发布门禁）：修改前完整读取800,549-byte报告并确认与HEAD一致；追加后13,381行、803,130 bytes、SHA256=`fc6afee3...79d6`。2.1–2.212连续，“三池”0处，大写旧称仅历史文件名链接，2.211/2.212引用、三文件hash、测试计数及`git diff --check`通过。
- 2026-08-02（d0d Phase6 builder v1失败报告）：宿主Python 3.8因缺`datetime.UTC`在OCI写入前exit1，仅留294-byte log与2-byte exit；报告2.213已实时追加。最终13,412行、804,975 bytes、SHA256=`d213049f...de06`，2.1–2.213连续，术语、失败证据hash与diff通过。下一轮必须在发布后用固定control image Python 3.12.13和新v2目录。
- 2026-08-02（d0d Phase6 builder v2失败报告）：control image Python 3.12.13已消除解释器问题，但Git在第一项前置检查将NFS挂载判为dubious ownership；无OCI/report写入。2.214已实时追加，最终13,441行、806,759 bytes、SHA256=`4f8e4ee6...3048`，2.1–2.214连续，术语、2项失败证据hash及diff通过。
- 2026-08-02（d0d Phase6 builder v3成功）：固定control image/Python 3.12.13/4 CPUs/network none/runc/no GPU并在临时HOME配置两个精确safe.directory后自然exit0。candidate image/config=`0b33973c...43a8`，manifest=`f8e73d84...48b6`，layer=`f24dcc1d...f9de`，diff-ID=`b76606e9...cf0e`，33层、5,298 members、109,149,529-byte layer、无native/whiteout，8/8基本身份检查通过。当前只证明build完成，尚未递归验证source tree/base layers/artifact/native extensions。
- 2026-08-02（报告2.215门禁）：修改前完整读取806,759-byte报告且与HEAD一致；v3 build结果追加后13,487行、809,712 bytes、SHA256=`cd70b2f2...071b`。2.1–2.215连续，“三池”0处，大写旧称仅历史链接，2.214/2.215引用、5项核心证据hash及diff通过；verifier仍未运行。
- 2026-08-02（d0d Phase6 OCI递归验收）：同一固定CPU/no-GPU容器中verifier自然exit0/status passed；前32 base layers完全一致，source 4,744 files与Git tree精确匹配，4份rotation、runtime expectation、7个native extension及3项环境闭合，candidate layer未覆盖native。10/10独立结构检查与8/8核心manifest通过；verification report/log同为SHA `1e282b12...6d4`。当前尚未导入daemon或构建Stage9 control image。
- 2026-08-02（报告2.216门禁）：修改前完整读取809,712-byte报告且与HEAD一致；递归验收追加后13,527行、812,310 bytes、SHA256=`26306b79...c6d7`。2.1–2.216连续，“三池”0处，大写旧称仅历史链接，2.215/2.216引用、5项验收证据hash、10/10结构、8/8 manifest及diff通过。
- 2026-08-02（d0d Phase6 daemon导入）：目标tag预检查明确不存在后，一次性ubuntu:22.04 CPU工具容器安装skopeo 1.4.1，从只读OCI layout导入daemon自然exit0。独立5/5审计确认image ID=`0b33973c...43a8`、tag、33 layers、末diff-ID=`b76606e9...cf0e`及8项labels全部匹配。导入log/exit、inspect、audit SHA依次为`13ffa5b2...390f`、`9a271f2a...86aa`、`f035730b...9d41`、`46a5488b...e786`；尚未runtime import或Stage9 build。
- 2026-08-02（报告2.217门禁）：修改前完整读取812,310-byte报告且与HEAD一致；daemon导入追加后13,566行、814,627 bytes、SHA256=`0b132d0e...1e79`。2.1–2.217连续，“三池”0处，大写旧称仅历史链接，2.216/2.217引用、4项daemon证据hash、5/5身份及diff通过。
- 2026-08-02（Stage9 d0d base静态合同）：定向测试先有效红灯命中Dockerfile仍为1e base；首行最小切换到`glm52-oscar-a800-phase6-d0d22489b-0275043c:latest`后目标1/1、完整24/24通过。Dockerfile/test SHA为`16800907...a966`/`1b5dde06...cae5`。首次compile仅因只读挂载默认pycache EROFS，设置临时pycache后通过；尚未docker build或改下游活动配置。
- 2026-08-02（报告2.218门禁）：修改前完整读取814,627-byte报告且与HEAD一致；Stage9静态迁移追加后13,599行、816,547 bytes、SHA256=`5a295bdf...3b9c`。2.1–2.218连续，“三池”0处，大写旧称仅历史链接，2.217/2.218引用、两文件hash、红绿灯与diff通过。
- 2026-08-02（Stage9 d0d control build）：目标tag事前不存在；`--pull=false`/host network/no GPU build自然exit0，31.4MB索引下载慢至305.6s但无错误。control image=`9a8efeba...c87b6`，34层、末diff-ID=`229fbb3d...52e5`；10/10审计确认base image=`0b33973c...43a8`、前33层、全部labels、source d0d/tree、Entrypoint和空Cmd精确继承。尚未在新镜像内执行CPU runtime。
- 2026-08-02（报告2.219门禁）：修改前完整读取816,547-byte报告且与HEAD一致；control build追加后13,643行、819,064 bytes、SHA256=`c9379d9b...1151`。2.1–2.219连续，“三池”0处，大写旧称仅历史链接，2.218/2.219引用、5项核心证据hash、10/10审计及diff通过。
- 2026-08-02（新Stage9 control CPU runtime）：v1仅因错误旧模块路径`vllm.attention.ops` exit1，失败三文件保留；核对实际source后v2改为`vllm.v1.attention.ops`自然exit0。Python3.12.13/glibc2.35、git/iproute2固定版本、store/decode hash、`oscar_mla_rotate_add`存在、CUDA_VISIBLE空、`cuda_initialized=false`、device_count0均通过；11/11 validation和含失败边界的12/12 manifest通过。尚未迁移活动配置或使用GPU。
- 2026-08-02（报告2.220门禁）：修改前完整读取819,064-byte报告且与HEAD一致；CPU runtime追加后13,687行、821,998 bytes、SHA256=`e94416fc...b10d`。2.1–2.220连续，“三池”0处，大写旧称仅历史链接，2.219/2.220引用、v1/v2证据hash、11/11 validation、12/12 manifest与diff通过。
- 2026-08-02（d0d driver-import前GPU门禁）：首轮16:05:09Z和有效末轮16:06:44Z间隔95秒，GPU0–7两轮均0MiB/0%、compute空；原16:05:36Z轮次仅27秒，虽idle但保留为无效边界。9/9 validation、4/4 manifest通过；driver-injected容器尚未启动。
- 2026-08-02（报告2.221门禁）：修改前完整读取821,998-byte报告且与HEAD一致；双空闲门禁追加后13,719行、824,146 bytes、SHA256=`41749a4c...e198`。2.1–2.221连续，“三池”0处，大写旧称仅历史链接，2.220/2.221引用、6项证据hash、9/9、4/4 manifest与diff通过。
- 2026-08-02（d0d driver-visible native import）：2.221发布后即时8卡全idle，固定只见GPU0的探针自然exit0；Python3.12.13/Torch2.11.0+cu129、`vllm._C`、两项K768、4 metadata、4 rotation、runtime expectation、store/decode hash及融合helper均通过，visible count1但before/after `cuda_initialized=false`。退出8卡全idle。CPU canonical依赖实测与1e一致，生成717-byte `runtime_import.json`，SHA仍为`9bdfc8ca...3b20`；15/15 validation、11/11 manifest通过。
- 2026-08-02（报告2.222门禁）：修改前完整读取824,146-byte报告且与HEAD一致；driver-visible import追加后13,770行、827,534 bytes、SHA256=`d2a627f7...821a`。2.1–2.222连续，“三池”0处，大写旧称仅历史链接，2.221/2.222引用、8项核心证据hash、15/15、11/11 manifest与diff通过。
- 2026-08-02（d0d活动身份静态验收）：活动Phase5/7/9的3个配置、6个wrapper和1个聚合合同已统一到source `d0d22489...`、tree `d07b4992...`、新Phase6三摘要/runtime import与control image；Phase1 baseline及历史artifact未改。新Phase6 overlay必须另补6个指向冻结Phase0 rootfs的native symlink，OCI verifier通过本身不等于提取目录已具备runtime链接。有效结果为Phase7递归44/44、Phase9递归70/70、Phase7单测20/20、Phase9单测89/89、聚合工具24/24、独立身份81/81、41项证据manifest全通过；仅证明启动前身份与静态合同闭合，不产生新的精度或端到端性能数据。
- 2026-08-02（报告2.223发布门禁）：正式记录已实时追加并验证为13,838行、832,723 bytes、SHA256=`1c917217...8836`；2.1–2.223连续，术语、交叉引用、10个活动文件hash、81/81审计、41/41证据manifest和diff全部通过。静态迁移阶段至此闭合，下一步必须先发布，之后重新做GPU双空闲门禁，才能启动同256题精度筛查。
- 2026-08-02（d0d候选256题精度前GPU门禁）：16:33:31Z/16:34:41Z两轮GPU0–7均0MiB/0%、compute空，间隔70秒。首次9/10只反映validator对空区段的切片错误，修正解析而不改原始日志后10/10通过；最终6项manifest全通过。该门禁只证明卡可分配，不是精度或性能数据。
- 2026-08-02（报告2.224发布门禁）：正式记录已实时更新为13,876行、835,244 bytes、SHA256=`2f59df90...4118`；2.1–2.224连续，术语、交叉引用、10/10 GPU门禁、6/6 manifest和diff通过。候选精度只能在本节发布并即时复核GPU后启动。
- 2026-08-02（d0d精度入口摘要修复）：Phase7/9活动身份迁移遗漏了`run_official_v5_gsm8k.sh`和fast版本内的candidate runtime摘要断言；它们仍期待旧`dd16...`，与d0d runtime manifest的`f8e73d...`必然不匹配。两个wrapper只需替换candidate摘要，native摘要、评测输入、协议和生成参数均不改。有效静态结果为目标1/1、聚合24/24、Phase7 20/20、Phase9 89/89、独立27/27和20项manifest通过；修复不产生精度数据。
- 2026-08-02（报告2.225发布门禁）：正式记录已更新为13,932行、838,959 bytes、SHA256=`83d7aaea...caa1`；2.1–2.225连续，术语、交叉引用、三文件hash、27/27静态验收、20/20证据manifest和diff通过。发布前仍没有新精度结果。
- 2026-08-02（精度入口修复后GPU门禁）：16:43:52Z/16:44:58Z两轮GPU0–7全0MiB/0%、compute空，间隔66秒；10/10 validation和3/3 manifest通过。该门禁绑定主仓`f47cc11...`与source d0d身份，只授权发布后启动同256题筛查，不是精度结果。
- 2026-08-02（报告2.226发布门禁）：正式记录已更新为13,964行、840,963 bytes、SHA256=`6eb99c88...d2d8`；2.1–2.226连续，术语、交叉引用、10/10门禁、3/3 manifest和diff通过。发布前仍未启动精度容器。
- 2026-08-02（d0d fast256早期输出差异）：前3个完成样本均为答案提取失败；其中`gsm8k:000049`本轮生成3,746 tokens后退化为重复`the`，而同协议旧topk1024轮该题289 tokens正确得到30；`gsm8k:000060`本轮2 tokens输出`Comments`，旧topk1024轮536 tokens正确得到17；`gsm8k:000029`本轮1 token空输出，旧topk1024轮则到7,974-token上限仍失败。三题顺序和并发完成时机不同，且样本太少，现阶段只能证明输出轨迹明显不同，不能据此证明融合实现导致精度回归；必须保留完整256题门禁与最终协议/身份审计。
- 2026-08-02（d0d fast256首批18题同题对照）：17:34节点18题均可在旧topk1024同协议轮找到完全相同ID，18/18 prompt hash一致，双方protocol fingerprint均为`5bc5f1a0...04718`。旧轮相同18题10题正确、6题提取失败/截断；d0d轮0题正确、18题提取失败、15题截断。该受控同题差异已是强回归信号，但尚不能单凭中间批次区分融合数值差异与运行时并发非确定性；若最终门禁失败，优先在实际decode/prefill行数上做旧路径与融合路径bitwise门禁，不直接修改其他优化参数。
- 2026-08-02（d0d与旧topk1024评测输入闭合）：两轮`runtime_suite/eval_config.json`均为1,078 bytes、SHA256 `025b2dde...3c95`，`fast_suite_identity.json`均为658 bytes、`9324c1d7...cd3f`，`manifest.jsonl`均为264,147 bytes、`fcd3079b...5dcf`，逐字节一致；runner命令仅run/output目录不同，runner environment除时间、runtime manifest身份及namespace外一致。因此首批差异不能归因于题集、生成配置或评测脚本漂移；仍需最终结果及实际shape数值门禁判定production融合是否为根因。
- 2026-08-02（d0d fast256前34题同题对照）：34/34 prompt hash与旧topk1024轮一致；旧轮相同34题17题正确、14题提取失败、15题截断，d0d轮0题正确、34题提取失败、31题截断。受控差异已从首批18题扩展到34题，支持“d0d运行路径存在严重精度回归”，但最终报告仍需等待自然完成和正式validation，根因仍由实际shape的旧/新位级门禁决定。
- 2026-08-02（融合正确性门禁dtype覆盖缺口）：production的`bf16_merged`虽沿用旧变量名，实际明确以`torch.float32`分配，`history_merged=torch.empty_like(bf16_merged)`同为FP32；融合调用的latent因此是FP32、rotation为BF16、addend为FP32。现有`test_rotate_add_matches_separate_fp32_add`和`benchmark_oscar_rotation.py`均只用BF16 latent。此前8/16,384行bitwise equal与微基准不能证明真实FP32-latent路径等价；最终门禁若失败，第一项GPU诊断必须补FP32 latent及真实decode/prefill行数，不能直接复用旧结论。
- 2026-08-02（d0d fast256前51题同题对照）：51/51 prompt hash与旧topk1024轮一致；旧轮相同51题20题正确、28题提取失败、29题截断，d0d轮0题正确、51题提取失败、47题截断。回归信号继续扩大，且不是由旧轮这些题全部本就失败造成。
- 2026-08-02（d0d fast256前69题同题对照）：69/69 prompt hash与旧topk1024轮一致；旧轮相同69题29题正确、33题提取失败、34题截断，d0d轮仍0题正确、69题提取失败、61题截断。受控差异持续扩大。
- 2026-08-02（d0d fast256前104题同题对照）：104/104 prompt hash与旧topk1024轮一致；旧轮相同104题43题正确、51题提取失败、52题截断，d0d轮仍0题正确、104题提取失败、90题截断。题集控制下的严重回归已覆盖超过40%的固定子集。
- 2026-08-02（d0d精度门禁数学失败）：21:14:14Z为154/256、0正确，剩余102题；BF16保守门槛为105/256，因此即使剩余全部正确，最高也只有102/256，d0d候选已确定失去32K正式性能复测资格。这是严格上界而非统计推断；完整256题仍自然完成以生成正式终局与validation。
- 2026-08-03（d0d fast256完整终局）：run自然exit0，256/256 scored但0题正确、256题答案提取失败、233题截断、0请求失败，精度0.000000%；completion tokens总计1,910,623、均值7,463.37109375，总时长26,229.182457秒、35.136436 req/h。protocol fingerprint仍为`5bc5f1a0...04718`；summary/predictions/validation SHA256分别为`c5d8571a...ad10`、`29d0575b...ce21`、`7f21425e...ee75`。独立15/15逻辑检查通过；证据持久化为281文件/18,738,541 bytes，最终manifest 279/279通过。退出后8卡0 MiB/0%、compute空。结合production FP32 latent未被旧门禁覆盖，下一步优先验证该dtype/实际shape路径。
- 2026-08-03（报告2.227发布前门禁）：修改前完整读取840,963-byte报告且与HEAD一致；终局追加后为14,055行、847,190 bytes、SHA256=`6354bf33...489`。2.1–2.227连续，“三池”0处，大写旧称仍仅历史文件链接；8项核心证据大小/hash一致，独立15/15与最终manifest 279/279复算通过，`git diff --check`通过。d0d仍禁止进入32K性能复测。
- 2026-08-03（FP32 latent实际shape与资源门禁）：全局64 heads / TP8得到每卡8 heads；结合max sequences16和max batched tokens2048，需覆盖FP32 latent的8×512 batch1 decode、128×512并发16 decode上界和16,384×512 chunked prefill上界。00:27:07Z/00:28:23Z两轮GPU0–7均0 MiB/0%、compute空，间隔76秒；独立21/21、manifest 8/8通过。首次verifier只因错误配置键路径得到`KeyError`，修正后有效。
- 2026-08-03（报告2.228发布前门禁）：修改前报告14,055行、847,190 bytes且与HEAD逐字节一致；追加shape审计与双空闲门禁后为14,111行、850,943 bytes、SHA256=`8d6ec40c...9788`。2.1–2.228连续，“三池”0处，大写旧称仍仅历史链接；21/21独立复算、8/8 manifest、核心hash与diff通过。尚未运行FP32 latent CUDA对照。
- 2026-08-03（FP32 latent三项实际shape位级结果）：固定GPU0、network none、d0d source和Stage9 control runtime，8/128/16,384行三例均为FP32 latent、BF16 rotation、FP32 addend；旧rotate+add与融合helper全部bitwise equal、mismatch0、max/mean abs error0、finite且复用output。run exit0、固定control复核19/19、manifest 12/12，退出后8卡全idle。故“helper内FP32 dtype直接误差”假设被否定，但端到端0/256仍未解释。
- 2026-08-03（报告2.229发布前门禁）：修改前报告14,111行、850,943 bytes且与HEAD一致；追加单卡结果及宿主root-owned复核失败边界后为14,169行、854,942 bytes、SHA256=`b9d9137c...a3b5f`。2.1–2.229连续，“三池”0处，大写旧称仅历史链接；19/19、12/12、核心hash与diff通过。下一控制实验应回退production融合并重跑同256题，而非运行d0d 32K。
- 2026-08-03（pre-fusion production回退）：有效TDD强制`/workspace`当前source后红灯exit1，精确命中fused helper；最小回退后绿灯exit0。decode/store production blob分别为`a4a169254...`/`95c9c979...`，与1e控制逐字节一致；split-K/K值/协议未改。source提交`83320e120`、tree`2d067ea6...36af`已通过完整pre-commit并HTTPS发布；独立19/19、manifest15/15。pytest缺失、错误import旧镜像source和直接pre-commit PATH缺失均保留为无效环境边界。
- 2026-08-03（报告2.230发布前门禁）：修改前报告14,169行、854,942 bytes且与HEAD一致；追加source回退后为14,231行、859,195 bytes、SHA256=`b6a6c645...4030`。2.1–2.230连续，“三池”0处，大写旧称仅历史链接；source/blob、19/19、15/15 manifest、核心hash与两仓diff通过。当前只证明source回退，尚未构建镜像或恢复精度。
- 2026-08-03（回退控制Phase6输入迁移）：只先改测试期望得到有效红灯，manifest实际d0d与期望833精确不符；最小修改candidate input、Dockerfile两ARG和合同后目标1/1、完整builder 2/2、独立19/19、manifest9/9通过。新tag为`glm52-oscar-a800-phase6-83320e120-0275043c`；base、rotation、runtime expectation均不变，Phase5/7/9继续保持d0d直到新OCI实际摘要生成。
- 2026-08-03（报告2.231发布前门禁）：修改前报告14,231行、859,195 bytes且与HEAD一致；追加Phase6输入迁移后为14,285行、862,566 bytes、SHA256=`2b1377f2...aed4`。2.1–2.231连续，“三池”0处，大写旧称仅历史链接；三文件hash、19/19、9/9 manifest与diff通过。尚未构建新OCI或使用GPU。
- 2026-08-03（833回退控制Phase6 OCI构建）：固定Python3.12 control、4 CPUs、network none和GPU不可见，builder自然exit0；新OCI manifest/config/layer=`847b8dcc...1d6b`/`3c06df1c...d8ba`/`92d494e2...ca35`，diff-ID=`dc7c3ec3...4794`，33层、5,298 members、109,149,647-byte layer、无native/whiteout。构建时独立22/22、核心manifest7/7；递归verifier尚未运行。
- 2026-08-03（报告2.232发布前门禁）：修改前报告14,285行、862,566 bytes且与HEAD一致；追加OCI构建及预期dirty-report validator时序边界后为14,343行、866,416 bytes、SHA256=`02435640...f5f8`。2.1–2.232连续，“三池”0处，大写旧称仅历史链接；原始22/22、7/7 manifest、核心hash与diff通过。报告diff阶段复跑validator只令clean项失败，不计有效结果。
- 2026-08-03（833回退控制OCI递归验收）：同一OCI verifier自然exit0，4,744 source文件精确Git tree、前32 base layers、4 rotation、runtime expectation、7 native基础层继承和3运行环境全通过；overlay为4,749 regular/0 symlink且无native，后续仍需只补6个Phase0 native symlink。独立19/19、最终核心manifest14/14。
- 2026-08-03（报告2.233发布前门禁）：修改前报告14,343行、866,416 bytes且与HEAD一致；追加递归验收与统一会话早返回边界后为14,390行、869,849 bytes、SHA256=`b1c502a4...b6bc`。2.1–2.233连续，“三池”0处，大写旧称仅历史链接；19/19、14/14 manifest、核心hash与diff通过。尚未导入daemon或使用GPU。
- 2026-08-03（833 overlay native链接审计）：已验收d0d overlay恰有6个绝对symlink，分别为`vllm/_C.abi3.so`、`_C_stable_libtorch.abi3.so`、`_moe_C.abi3.so`、`cumem_allocator.abi3.so`及flash-attn fa2/fa3扩展，均指向冻结Phase0 rootfs同相对路径；833新overlay当前0 symlink。首次读取误把native manifest写成项目根`recovery/native_extensions.sha256`，文件不存在而fail-closed，尚未创建链接；下一步先定位真实manifest并逐项验证目标。
- 2026-08-03（833 overlay链接与CPU source import）：仅新增冻结Phase0 rootfs前6个vLLM native绝对链接，第7个外部stable算子未加入；最终4,749 regular/6 symlink，目标hash逐项匹配。canonical固定Stage9 1e控制镜像、runc/network none/4 CPUs/GPU不可见，两处prefill K=768、三个实际`MLACommonMetadata`字段、overlay source路径与CUDA未初始化均通过，exit0；`vllm._C`因无`libcuda.so.1`仅warning，driver import仍未验收。错误挂载和错误四字段期望各exit1，修复只涉及validator/命令，没有修改production source；15/15证据manifest通过。
- 2026-08-03（833 Phase6 daemon导入）：目标tag事前不存在；本地Ubuntu22.04/runc/4 CPUs/GPU不可见工具容器用skopeo1.4.1复制33 blobs、config和manifest，自然exit0。daemon image/config=`3c06df1c...d8ba`、tag、33层、末diff-ID=`dc7c3ec3...4794`与8项labels独立5/5通过，证据manifest9/9。统一执行会话提前返回时原skopeo仍运行，未重启，轮询原PID取得终局。
- 2026-08-03（迟到Python control异步写入）：2.234阶段缺失本地`python:3.12.13`的旧shell后台拉取完成后，通过已移动的开放文件描述符把0-byte失败占位写成545-byte canonical同内容JSON，令原manifest暂时14/15。保留迟到输出、恢复原空占位后原manifest重新15/15；迟到JSON hash与canonical均为`7b7fa67c...e65b`，没有GPU或production改动。后续遇统一会话早返回必须同时检查宿主外层PID，不能只看容器瞬时状态。
- 2026-08-03（833 Stage9 base输入迁移）：只先把目标合同期望改为833 base，CPU-only目标用例有效红灯1 failure，精确命中Dockerfile仍为d0d；只改Dockerfile首行后目标1/1、完整工具24/24及compile通过。新base daemon ID=`3c06df1c...d8ba`/33层/末diff=`dc7c3ec3...4794`，目标control tag不存在；独立19/19、manifest16/16。首次validator误期望失败inspect stdout为0 bytes，实际Docker输出`[]\n`，只修validator后通过。
- 2026-08-03（833 Stage9 control构建）：已发布输入下以空context、`--pull=false`构建自然exit0，apt索引31.4 MB/8秒、RUN14.4秒、无重试。control image ID=`62568e2e...8013`、34层，前33层与base逐项相同，新末diff=`582db025...e2bc`，Phase6 labels继承、Entrypoint/Cmd正确；独立10/10、manifest7/7。尚未做CPU runtime或GPU import。
- 2026-08-03（833 Stage9 CPU runtime）：新control在runc/network none/4 CPUs/GPU不可见且不挂宿主source时自然exit0；Python3.12.13/glibc2.35、git/iproute2版本、vLLM source路径通过。decode/store hash=`13953366...42f`/`ec82245e...b8e`，融合helper不存在且rotate+独立add路径存在；CUDA visible空、device0、未初始化。独立18/18、build+runtime manifest14/14；尚不是driver import或精度结果。
- 2026-08-03（833 driver import前双空闲门禁）：01:38:28Z与01:39:39Z两轮GPU0–7均0 MiB/0%、compute空，间隔71秒；固定无GPUcontrol独立10/10、manifest5/5。当前只批准发布后即时复核，并在仍全idle时固定GPU0运行一次native import；不批准模型或32K。
- 2026-08-03（833首次driver probe缺失K环境）：发布后即时8卡全idle，固定GPU0 probe已成功执行`import vllm._C`并继续到source断言，但命令遗漏`VLLM_SPARSE_INDEXER_PREFILL_TOPK_TOKENS=768`，模块默认0，故在K断言exit1且stdout0 bytes；不是native或production错误。退出后8卡全idle，独立9/9、失败manifest9/9。重试必须重新双空闲门禁并显式补环境。
- 2026-08-03（833 driver probe重试前门禁）：01:46:48Z/01:47:56Z两轮GPU 0–7均为0 MiB/0%、compute空，间隔68秒；结构化validation 10/10、manifest 5/5通过。发布前手工复算首次误在仓库根目录解析相对路径而得到5项找不到，切到证据目录后同一manifest 5/5通过，证据未修改。当前只形成重试门禁，没有runtime import、精度或性能结果。
- 2026-08-03（833 driver-visible native import重试）：2.241由`398cda0`发布后，01:53:25Z即时8卡全idle；固定GPU0、Stage9镜像、network none/4 CPUs、相同probe并显式K=768自然exit0。native扩展、单卡可见、CUDA前后未初始化、两处K768、四字段metadata、pre-fusion decode/store hash、rotation/runtime identity全部匹配；01:53:35Z退出后8卡全idle。独立validator前两轮仅因宿主Python3.8不支持内置泛型求值和`removeprefix`退出，等价兼容修改后25/25通过，最终manifest22/22。尚未生成Phase6 canonical runtime import，未运行模型、精度或性能。
- 2026-08-03（833 canonical runtime import）：2.242由`46eeae0`发布后，固定833 Stage9镜像、network none/4 CPUs、NVIDIA不可见且不注入GPU自然exit0；实测Python3.12.13、Torch2.11.0+cu129、Triton3.6.0、Transformers5.8.1、Tokenizers0.22.2、FlashInfer0.6.6/JIT0.6.6+cu129、78层、支持max且CUDA未初始化。结合Phase6三项artifact与2.242 driver身份生成717-byte `runtime_import.json`，SHA=`9bdfc8ca...3b20`，与1e/d0d逐字节一致但落到833目录。独立17/17、跨目录manifest11/11；未运行精度或性能。
- 2026-08-03（833活动身份迁移）：12个正式配置/wrapper/合同已统一到833 source/tree、Phase6摘要与路径、canonical runtime import及Stage9 control，活动旧d0d标记0处。目标红灯精确命中旧Phase5 commit；绿灯1/1、Phase9工具24/24。Phase7递归两轮42/44的环境边界依次定位为临时HOME未建和Phase0 NFS mode漂移，精确挂载冻结source卷后44/44；Phase9递归70/70。Phase7单测首次缺冻结恢复Python挂载为19/20，精确只读挂载后20/20。独立31/31、manifest40/40；全程CPU-only。
- 2026-08-03（833同256题前双空闲门禁）：2.244与活动身份由`688cb91`发布后，02:19:01Z/02:20:01Z两轮GPU0–7均0 MiB/0%、compute空，间隔60秒；validation10/10、manifest6/6。当前只批准发布后即时复核与同256题启动，不代表精度或性能结果。
- 2026-08-03（833同256题有效启动）：run `20260803T0226Z_candidate_prefusion_rollback_fast256_c16_v1`使用正式wrapper、固定256题/并发16/8K/high协议；02:25:51Z即时空闲与入口内部双空闲通过，static/namespace/递归身份preflight通过。02:29:32Z仍无checkpoint，不得把启动状态表述为精度结果；每600秒只读累计监控已启动。
- 2026-08-03（checkpoint监控权限边界）：隔离user namespace下宿主能列出checkpoint但不能读内容，宿主首行0/256且read_errors=1无效；同一正式容器内只读解析得到有效1/256、0正确、0.000000%。后续必须在容器内采样，不能使用宿主文件可见性推断精度。
- 2026-08-03（833 fixed256 20分钟健康判断）：有效节点仍1/256并非卡死证据；同期16 running/0 waiting、64.0–78.4 token/s、KV 9.0%–9.2%、8卡高利用率且MLA计数持续增长，无fatal/OOM。
- 2026-08-03（833 fixed256 30分钟健康判断）：累计仍1/256；同期16 running/0 waiting、64.0–76.8 token/s、KV 16.3%–16.5%，错误扫描无ERROR/Traceback/CUDA OOM/fatal，只能证明长答案继续生成，不能预测最终精度。
- 2026-08-03（833 fixed256 40分钟精度信号）：首批结束后累计17/256、0正确、17提取失败、16截断；pre-fusion回退尚未展现精度恢复。下一批正常启动且无服务异常，仍需256题终局后决定是否淘汰。
- 2026-08-03（split-K stride候选根因）：基础K1024创建stride0=1024共享buffer，prefill切到768列后view仍stride1024；冻结CUDA `topKPerRowPrefill`按`rowIdx*topK`即768写，Triton attention按`selected_tokens.stride(0)`即1024读。正式runtime环境与分支条件精确匹配，能解释统一K768/K1024正常而split-K输出损坏；仍需GPU最小复现和修复后端到端验证，当前不得写成已证实终局根因。
- 2026-08-03（split-K stride CPU合同证据）：runtime/source/地址模型8/8通过，split-K 4行仅row0地址匹配，统一K768/K1024全部匹配；最终manifest9/9。它把静态推断变成可复算CPU合同，但没有执行GPU kernel或证明修复后的精度。
- 2026-08-03（833 fixed256 50分钟）：累计仍17/256、0正确；第二批服务健康。该观测与stride候选方向一致，但没有增加GPU因果证据，必须继续终局与专项验证。
- 2026-08-03（连续临时输出CPU行为）：Torch2.11下缩窄view为非连续stride(8,1)，`empty_like`默认与显式contiguous_format均产生(2,1)连续tensor；为冻结语义仍显式指定contiguous_format。这里只验证临时输出形状，不覆盖GPU top-k或copy-back。
- 2026-08-03（833 fixed256 60分钟）：累计仍17/256、0正确；第二批16请求健康且KV增长到17.8%，符合长输出继续生成，没有新的因果证据。
- 2026-08-03（split-K stride 修复插入点复核）：隔离 detached worktree `/dev/shm/oscar-glm-splitk-stride-fix-wt` 保持在 source `83320e120`。`sparse_attn_indexer.py` 当前在 native prefill MQA 分支中直接创建 `topk_indices_buffer[..., :prefill_topk_tokens]` 非连续 view，并把它交给四类 top-k backend；top-k 完成后没有 copy-back。最小候选应在 top-k 分支前统一准备连续输出，仅非连续目标使用 `torch.empty_like(..., memory_format=torch.contiguous_format)`，四类 backend 完成后拷回原 view；完整 K 的连续路径保持原 tensor、无分配/拷贝。既有 runtime cache 测试只验证 attention reader 读取手工填充后的缩窄 view，未覆盖 native producer 的行步长合同。
- 2026-08-03（split-K stride最小CPU TDD）：固定833 control且无GPU注入的有效红灯精确为缺失`_prepare_native_topk_output`的`AttributeError`/exit1；最小实现后两个标准库合同、两文件compile、ruff lint和diff-check均exit0。隔离diff只改runtime cache测试与`sparse_attn_indexer.py`，46行新增/3行移动删除；非连续`[4,8][:,:2]`输出stride(2,1)、copy-back后4行签名正确且后6列未改，连续`[4,8]`复用原tensor/无copy目标。该结果尚不包含native CUDA或端到端精度，补丁未提交、未发布、未迁移镜像。
- 2026-08-03（833固定256题70分钟节点）：03:36:15Z完成35/256、0正确、0.000000%，0 request failure、35 extraction failure、32截断、0 checkpoint read error；8卡76,081 MiB、75%–98%。第二批集中结束并装入第三批，服务仍运行；继续到256终局且不启动32K性能。
- 2026-08-03（source pre-commit边界复核）：source的pre-commit对所有变更Python文件运行`ruff-check --fix`和`ruff-format`，而当前基线`sparse_attn_indexer.py`用本机ruff全文件format会产生大量既有重排。因此在GPU门禁前不运行会改写文件的完整hook；当前仅使用ruff lint、diff-check、compile和固定control标准库合同。后续若准备source提交，必须先隔离审计hook造成的diff，不能把无关格式化混入最小修复。
- 2026-08-03（split-K CPU TDD持久化）：隔离candidate binary diff与持久化`candidate.patch`逐字节相同，SHA=`f6f74d2f...2fe0`，可对833 clean source通过`git apply --check`；README/validation JSON/candidate patch已落到`artifacts/phase9-control/20260803T033200Z_splitk_stride_cpu_tdd_v1`，JSON可解析。该持久化防止`/dev/shm`丢失，但不等于source提交或GPU验证。
- 2026-08-03（GPU最小复现几何冻结）：既有CUDA kernel test始终用连续`(num_rows, top_k)`输出，未覆盖缩窄view。`topKPerRowPrefill`源码明确按`rowIdx * topK`推进输出指针且绑定层不接收output stride。后续GPU gate使用4行、base宽4、topK2：先得连续reference，再把同一op写入`base[:, :2]`验证broken view不等于reference，最后写连续temp并copy-back验证等于reference且base尾列不变。行间最高logit位置必须不同，避免错误行恰好数值相同。
- 2026-08-03（833固定256题80分钟节点）：03:46:15Z仍为35/256、0正确、0.000000%，0 request failure、35 extraction failure、32截断、0 checkpoint read error；8卡76,081 MiB、75%–98%。第三批尚未集中结束，容器持续计算。
- 2026-08-03（833固定256题90分钟节点）：03:56:16Z仍为35/256、0正确、0.000000%，0 request failure、35 extraction failure、32截断、0 checkpoint read error；8卡76,081 MiB、73%–98%。第三批仍生成，无新checkpoint。
- 2026-08-03（833固定256题100分钟节点）：04:06:15Z为54/256、0正确、0.000000%，0 request failure、54 extraction failure、48截断、0 checkpoint read error；8卡76,083 MiB、74%–98%。第三批结束后下一批16 running/0 waiting、78.4 token/s、KV 6.4%–6.5%，无异常。
- 2026-08-03（833固定256题110分钟节点）：04:16:15Z仍为54/256、0正确、0.000000%，0 request failure、54 extraction failure、48截断、0 checkpoint read error；8卡76,083 MiB、73%–98%。服务16 running/0 waiting、80.0 token/s、KV14.3%，无异常。
- 2026-08-03（833固定256题120分钟节点）：04:26:15Z仍为54/256、0正确、0.000000%，0 request failure、54 extraction failure、48截断、0 checkpoint read error；8卡76,083 MiB、78%–98%。节点后瞬时4 running/0 waiting、49.5 token/s、KV5.5%，为批次收尾过渡且无异常。
- 2026-08-03（连续临时top-k短行边界排除）：`topKPerRowJob`在`rowLen <= topK`时显式写入全部topK位置：有效段为0..rowLen-1，剩余段写-1后return；长行路径也写满topK。因此`torch.empty_like`临时tensor不需要预填-1，copy-back不会把未初始化尾部带入共享buffer。既有CUDA测试只比较有效段，GPU最小复现仍应额外覆盖短行尾部-1，防止未来回归。
- 2026-08-03（833固定256题130分钟节点）：04:36:16Z为71/256、0正确、0.000000%，0 request failure、71 extraction failure、64截断、0 checkpoint read error；8卡76,083 MiB、77%–98%。下一批16 running/0 waiting、80.0 token/s、KV8.8%，无异常。
- 2026-08-03（833固定256题140分钟节点）：04:46:15Z仍为71/256、0正确、0.000000%，0 request failure、71 extraction failure、64截断、0 checkpoint read error；8卡76,083 MiB、75%–98%，当前批次继续生成。
- 2026-08-03（833固定256题150分钟节点）：04:56:15Z为87/256、0正确、0.000000%，0 request failure、87 extraction failure、80截断、0 checkpoint read error；8卡76,085 MiB、54%–97%。新增16题全错，继续终局。
- 2026-08-03（833固定256题160分钟节点）：05:06:16Z仍为87/256、0正确、0.000000%，0 request failure、87 extraction failure、80截断、0 checkpoint read error；8卡76,085 MiB、73%–98%。服务16 running/0 waiting、78.4 token/s、KV10.4%，无异常。
- 2026-08-03（833固定256题170分钟节点）：05:16:16Z仍为87/256、0正确、0.000000%，0 request failure、87 extraction failure、80截断、0 checkpoint read error；8卡76,085 MiB、73%–97%。服务16 running/0 waiting、78.4 token/s、KV18.7%，无异常。
- 2026-08-03（833固定256题180分钟节点）：05:26:15Z为104/256、0正确、0.000000%，0 request failure、104 extraction failure、96截断、0 checkpoint read error；8卡76,085 MiB、74%–97%。下一批16 running/0 waiting、80.0 token/s、KV4.4%，无异常。104完成题与BF16全量105正确题不是同分母比较，但OSCAR仍远未过门禁。
- 2026-08-03（833固定256题190分钟节点）：05:36:15Z为105/256、0正确、0.000000%，0 request failure、105 extraction failure、96截断、0 checkpoint read error；8卡76,085 MiB、71%–98%。新增1题未截断但提取失败；服务16 running/0 waiting、80.0 token/s、KV12.0%。完成题数105与BF16正确数105分母不同，禁止直接比例比较。
- 2026-08-03（833固定256题200分钟节点）：05:46:16Z仍为105/256、0正确、0.000000%，0 request failure、105 extraction failure、96截断、0 checkpoint read error；8卡76,085 MiB、76%–98%。服务16 running/0 waiting、78.4 token/s、KV19.6%，无异常。
- 2026-08-03（833固定256题210分钟节点）：05:56:15Z为123/256、0正确、0.000000%，0 request failure、123 extraction failure、111截断、0 checkpoint read error；8卡76,087 MiB、76%–99%。新增18题全错，其中15题截断。
- 2026-08-03（833固定256题220分钟节点）：06:06:15Z为125/256、0正确、0.000000%，0 request failure、125 extraction failure、112截断、0 checkpoint read error；8卡76,087 MiB、72%–98%。新增2题全错，其中1题截断；服务16 running/0 waiting、76.8 token/s、KV12.5%。
- 2026-08-03（833固定256题230分钟节点）：06:16:15Z仍为125/256、0正确、0.000000%，0 request failure、125 extraction failure、112截断、0 checkpoint read error；8卡76,087 MiB、72%–98%。服务16 running/0 waiting、76.8 token/s、KV2.0%，下一批刚开始。
- 2026-08-03（833固定256题240分钟节点）：06:26:16Z为139/256、0正确、0.000000%，0 request failure、139 extraction failure、126截断、0 checkpoint read error；8卡76,087 MiB、73%–98%。新增14题全错且全截断；服务16 running/0 waiting、78.4 token/s、KV9.5%。
- 2026-08-03（833固定256题250分钟节点）：06:36:15Z为141/256、0正确、0.000000%，0 request failure、141 extraction failure、128截断、0 checkpoint read error；8卡76,087 MiB、73%–98%。新增2题全错且均截断。
- 2026-08-03（833固定256题260分钟节点）：06:46:15Z为156/256、0正确、0.000000%，0 request failure、156 extraction failure、142截断、0 checkpoint read error；8卡76,089 MiB、57%–97%。新增15题全错，其中14题截断；下一批16 running/0 waiting、78.4 token/s、KV3.6%。
- 2026-08-03（833固定256题270分钟节点）：06:56:15Z仍为156/256、0正确、0.000000%，0 request failure、156 extraction failure、142截断、0 checkpoint read error；8卡76,089 MiB、73%–99%，当前批次继续生成。
- 2026-08-03（833固定256题280分钟节点）：07:06:16Z为158/256、0正确、0.000000%，0 request failure、158 extraction failure、144截断、0 checkpoint read error；8卡76,089 MiB、77%–97%。新增2题全错且均截断，容器继续运行。
- 2026-08-03（833固定256题290分钟节点）：07:16:17Z为173/256、0正确、0.000000%，0 request failure、173 extraction failure、158截断、0 checkpoint read error；8卡76,089 MiB、74%–98%。新增15题全错，其中14题截断。
- 2026-08-03（833固定256题300分钟节点）：07:26:15Z为175/256、0正确、0.000000%，0 request failure、175 extraction failure、160截断、0 checkpoint read error；8卡76,091 MiB、68%–98%。新增2题全错且均截断。
- 2026-08-03（833固定256题310分钟节点）：07:36:15Z仍为175/256、0正确、0.000000%，0 request failure、175 extraction failure、160截断、0 checkpoint read error；8卡76,091 MiB、74%–98%。当前批次继续生成，无新checkpoint。
- 2026-08-03（833固定256题320分钟节点）：07:46:16Z为190/256、0正确、0.000000%，0 request failure、190 extraction failure、174截断、0 checkpoint read error；8卡76,091 MiB、78%–98%。新增15题全错，其中14题截断；下一批16 running/0 waiting、78.4–80.0 token/s。
- 2026-08-03（833固定256题330分钟节点）：07:56:15Z为193/256、0正确、0.000000%，0 request failure、193 extraction failure、176截断、0 checkpoint read error；8卡76,091 MiB、72%–97%。新增3题全错，其中2题截断；服务16 running/0 waiting、78.4 token/s。
- 2026-08-03（833固定256题340分钟节点）：08:06:15Z为194/256、0正确、0.000000%，0 request failure、194 extraction failure、176截断、0 checkpoint read error；8卡76,091 MiB、73%–98%。新增1题未截断但提取失败；服务16 running/0 waiting、76.8 token/s。
- 2026-08-03（833固定256题350分钟节点）：08:16:15Z为207/256、0正确、0.000000%，0 request failure、207 extraction failure、188截断、0 checkpoint read error；8卡76,093 MiB、74%–98%。新增13题全错，其中12题截断；服务16 running/0 waiting、78.4 token/s。
- 2026-08-03（833固定256题360分钟节点）：08:26:16Z为210/256、0正确、0.000000%，0 request failure、210 extraction failure、191截断、0 checkpoint read error；8卡76,093 MiB、72%–98%。新增3题全错且全部截断；服务16 running/0 waiting、78.4 token/s。
- 2026-08-03（833固定256题370分钟节点）：08:36:15Z为223/256、0正确、0.000000%，0 request failure、223 extraction failure、204截断、0 checkpoint read error；8卡76,095 MiB、55%–97%。新增13题全错且全部截断；下一批16 running/0 waiting、80.0 token/s。
- 2026-08-03（833固定256题380分钟节点）：08:46:15Z仍为223/256、0正确、0.000000%，0 request failure、223 extraction failure、204截断、0 checkpoint read error；8卡76,095 MiB、79%–98%。当前批次继续生成；服务16 running/0 waiting、78.4 token/s。
- 2026-08-03（833固定256题390分钟节点）：08:56:15Z为227/256、0正确、0.000000%，0 request failure、227 extraction failure、208截断、0 checkpoint read error；8卡76,095 MiB、72%–98%。新增4题全错且全部截断；服务16 running/0 waiting、78.4 token/s。
- 2026-08-03（833固定256题400分钟节点）：09:06:16Z为239/256、0正确、0.000000%，0 request failure、239 extraction failure、220截断、0 checkpoint read error；8卡76,095 MiB、73%–97%。新增12题全错且全部截断；下一批16 running/0 waiting、78.4 token/s。
- 2026-08-03（833固定256题410分钟节点）：09:16:15Z为240/256、0正确、0.000000%，0 request failure、240 extraction failure、221截断、0 checkpoint read error；8卡76,095 MiB、75%–98%。新增1题全错且截断；服务16 running/0 waiting、78.4 token/s。
- 2026-08-03（833固定256题420分钟节点）：09:26:15Z为244/256、0正确、0.000000%，0 request failure、244 extraction failure、224截断、0 checkpoint read error；8卡76,099 MiB、63%–98%。新增4题全错，其中3题截断；尾批12 running/0 waiting、60.0 token/s。
- 2026-08-03（833固定256题430分钟节点）：09:36:16Z为255/256、0正确、0.000000%，0 request failure、255 extraction failure、235截断、0 checkpoint read error；8卡76,103 MiB、25%–97%。新增11题全错且全部截断；仅剩1 running/0 waiting、4.0 token/s。
- 2026-08-03（恢复后的权威运行状态）：17:45 CST 时 `20260803T0226Z_candidate_prefusion_rollback_fast256_c16_v1` 容器仍为Up，正式checkpoint最近修改于17:32:51 CST且仍为255/256；8卡各约76,105 MiB且对应PID 197281–197288均为本项目 `VLLM::Worker_TP0`–`TP7`。outer runner session 61907轮询30秒无输出但未终止，因此尚无证据宣告256题完成。
- 2026-08-03（833固定256题440分钟节点）：09:46:15Z仍为255/256、0正确、0.000000%，0 request failure、255 extraction failure、235截断、0 checkpoint read error；8卡76,105 MiB、44%–98%。最后1个请求继续生成，没有新增完整checkpoint，正式容器与outer runner均未退出。
- 2026-08-03（833 pre-fusion回退固定256题正式终局）：outer runner自然exit0，`accuracy_completed_at_utc.txt`为09:47:27Z；summary为256/256 scored、0正确、0.000000%、0请求失败、236截断（92.1875%）、平均completion tokens 7,436.65625、总completion tokens 1,903,784、duration 26,005.380892秒、35.4388195 req/h。validation状态passed且摘要SHA=`0a25e3d4...fa6f`；这表示实验协议/文件一致性通过，不表示精度通过。正式容器已删除，outer release check 8/8 idle；首次退出后只读`nvidia-smi`采样曾阻塞并被中止，随后15秒有界稳定采样成功，8卡均0 MiB/0%、compute空。
- 2026-08-03（终局口径校正）：10分钟监控器把截至255题的所有错误记为`extraction_failures=255`，但正式runner的`summary_by_benchmark.json`终局字段为`extraction_failed=0`、`scored=256`、`correct=0`。因此正式终局必须采用runner口径“256题均已评分、0题正确、236题截断”，不能把监控器的中间统计外推为“256个答案提取失败”；后续需通过root只读复核predictions解释两个口径差异。
- 2026-08-03（终局证据持久化模式复核）：2.227先例将`attempt/`、outer log/exit、10分钟日志、独立validation和最终manifest持久化到`artifacts/phase9/<timestamp>...`，最终281文件中manifest覆盖其余279项；当前run原始目录为289文件/44,941,338 bytes，其中256个prediction checkpoints。三个关键结果文件为root:root mode600，宿主普通用户无法读取，但`sudo -n`当前有效；正式run产物本身不应改权限，后续通过只读control容器或root只读复制到新证据目录复核。
- 2026-08-03（256条prediction只读复核）：root只读解析确认256行、256唯一ID、`score=0.0` 256条、`extracted_answer='[invalid]'` 256条、`error_message='answer extraction failed; scored as 0'` 256条、`truncated=true` 236条；因此“256个答案提取失败”有逐行证据。正式`summary_by_benchmark.json`却写`extraction_failed=0`，这是聚合字段与逐行证据的不一致，报告必须同时披露，不可只选有利口径。固定source仍为833 commit且clean/upstream；可用正式runtime image为`oscar-glm-stage9-runtime:83320e120`（ID `62568e2e150e`）。
- 2026-08-03（833终局证据闭合）：固定833 runtime image、network none、无GPU暴露的独立复核20/20通过，覆盖外层exit/release、256行/唯一ID/冻结ID集合、协议、0正确、0请求失败、256逐行提取失败、236截断、token总量、official summary/validation/hashes、aggregate口径差异和source身份。持久化目录`artifacts/phase9/20260803T0955Z_candidate_833_fast256_accuracy_failure_v1`最终281文件/18,762,192 bytes；最终manifest覆盖279项且279/279复算通过。核心SHA：predictions=`3d38ba4d...dada`、summary=`0a25e3d4...fa6f`、validation=`f09363ca...f00d`、independent=`33b6edc2...49aa`、final manifest=`a4d5d2f9...61ee`。
- 2026-08-03（新GPU门禁第一轮）：09:56:43Z GPU0–7全部0 MiB/0%，compute列表为空；门禁绑定主仓`ebd131f2...c1`与source 833。单次空闲不授权GPU实验，必须等待至少60秒并取得第二次同样结果。
- 2026-08-03（新GPU门禁有效）：第二轮09:57:54Z同样8卡全idle，间隔71秒；10/10检查覆盖两轮卡数/索引、显存、利用率、compute、间隔及主/source commit。该门禁只授权随后固定GPU0的已冻结stride数值复现，不授权32K性能复测。
- 2026-08-03（split-K stride GPU native根因验证）：2.248发布后10:00:40Z即时GPU0仍0 MiB/0%、compute空；固定833 runtime、network none、仅GPU0可见的冻结脚本自然exit0。reference=`[[0,1],[3,4],[6,7],[9,10]]`；stride(4,1)非连续view被写为`[[0,1],[6,7],[-1,-1],[-1,-1]]`且不匹配；stride(2,1)连续临时输出+copy-back逐值匹配reference，目标尾列保持-1；短行未用位置为-1且原buffer尾列保持-7。10:00:58Z退出后GPU0仍0 MiB/0%、compute空且容器已删除。该结果直接验证native kernel按topK连续寻址、不能遵守PyTorch view行stride，确认split-K producer/consumer错配根因及临时连续输出修复方向。
- 2026-08-03（GPU根因证据闭合）：固定833无GPUvalidation 20/20通过，覆盖坏/好输出、reference、短行与尾列、GPU0独占、镜像/网络/commit/script身份、前后空闲和预期warning；证据目录`artifacts/phase9-control/20260803T1000Z_splitk_stride_gpu_repro_v1`最终13文件/10,052 bytes，manifest 11/11通过。核心SHA：stdout=`58f968c6...13f4f`、script=`e319a263...1e01`、validation=`18a857f8...4131`、manifest=`0e080e89...ff6e`。
- 2026-08-03（source迁移边界）：主source当前未提交diff与隔离worktree candidate.patch逐字节一致，production/test SHA分别为`a80b5d59...9e57`/`11e2f9f8...dd1d`。固定runtime镜像能执行合同与py_compile但没有ruff binary，故exit127不得解释为lint失败；按AGENTS_misc环境规范切换到宿主uv提供的ruff后再判定静态门禁。
- 2026-08-03（source迁移静态门禁修正）：source pre-commit冻结ruff版本为0.14.0；宿主uv cache中存在同版本ruff binary。改用该二进制执行两处变更文件的`ruff check`得到`All checks passed!`，固定833无GPU容器单独py_compile自然exit0，diff SHA仍为`f6f74d2f...2fe0`。没有运行会重排既有大文件的ruff formatter。
- 2026-08-03（source commit attempt1失败）：提交钩子未创建commit；有效通过项包括ruff-check、typos、SPDX等。失败项是mypy-local报告既有变量重定义，以及torch.cuda禁用检查命中既有第130行；candidate patch只新增helper/调用前准备/copy-back及测试，不触及这些符号/行。须先以base833只读证据确认，再有针对性SKIP，不能`--no-verify`跳过全部钩子。
- 2026-08-03（source commit闭合）：base833只读检索与candidate零上下文hunk证明三个skip均为既有债务，第二次未用`--no-verify`且其余适用hooks通过。新source `ea8ae6b7758ae2b4db7cae44d638ae5de80148ac`的commit patch SHA=`f6f74d2f...2fe0`，说明提交内容与CPU/GPU已验收候选逐字节一致；GitHub HTTPS发布后source local/upstream相等。
- 2026-08-03（source commit证据闭合）：固定833无GPU独立复核14/14通过，覆盖两次提交边界、base既有失败、candidate hunk排除、commit/parent/tree/patch、clean/upstream与主仓身份。证据目录`artifacts/phase9-control/20260803T1015Z_splitk_stride_source_commit_v1`最终9文件/8,166 bytes，manifest 7/7通过。
- 2026-08-03（新source活动身份范围）：tracked旧833身份分布于Phase6输入/Dockerfile/单测、Phase5配置/wrapper、Phase7配置/wrapper和Phase9配置/wrapper/单测。按既有fail-closed流程先只迁移Phase6三文件并构建/验收OCI，再迁移Phase5/7/9，避免配置提前指向不存在的artifact。新Phase6 tag规则沿用`commit short + rotation manifest前8位`，为`glm52-oscar-a800-phase6-ea8ae6b77-0275043c`。
- 2026-08-03（Phase6最小身份实现）：新source tree=`8fb091670635eeba3809e4adc02841681b69e8d9`；更新Dockerfile commit/tree后的实际SHA=`d06401c8290f079d8101f2631f4a8f089696b4f7574667ca5a0181d49e7d7a1c`。Phase6配置其他base/rotation/runtime expectation/native extension字段均未改，目标和完整单测通过。
- 2026-08-03（Phase6身份证据闭合）：固定833无GPUvalidation 13/13通过，覆盖red/green/full tests、commit/tree/tag、Dockerfile身份/hash、三文件hash、主/source身份和Phase5/7/9仍为833。证据目录`artifacts/phase9-control/20260803T1030Z_splitk_stride_phase6_identity_v1`最终12文件/6,531 bytes，manifest 10/10通过。
- 2026-08-03（clean clone artifact边界）：builder的`Path.rglob`从`clean/artifacts/...`访问时不会按预期穿透顶层artifacts symlink，导致rotation artifact空集合。应在disposable clone创建真实`artifacts`目录，并在容器内把原artifact根bind mount到该真实路径；不能因此修改/隐藏Shawn的未跟踪报告或放宽builder clean检查。
- 2026-08-03（builder命令拆分原则）：第二次长组合命令在容器前退出且缺少明确日志，说明应把mount可见性、clean/upstream和builder本体拆成独立fail-closed步骤；新builder使用新run ID，保留旧control失败证据，不覆盖。
- 2026-08-03（shared clone容器要求）：`git clone --shared`的alternates是绝对路径，容器必须把原主仓和source的`.git/objects`以相同绝对路径挂载可见；仅挂clean clone会`bad object HEAD`。clean/upstream检查不得放在命令替换中，应直接执行并验证输出为空。
- 2026-08-03（ea8 Phase6递归verifier终局）：既有独立verifier已自然exit0且容器删除；`verification_report.json`为`status=passed`。候选tag=`glm52-oscar-a800-phase6-ea8ae6b77-0275043c`，image/config=`sha256:1bd0a551...bfba`，manifest=`sha256:ed1a105c...0526`，33层；candidate layer digest=`sha256:0f2efa41...277a`、diff_id=`sha256:aa212c18...f15f`、109,150,255 bytes、5,298 members、无native extension/whiteout。source `ea8ae6b...48ac`的4,744文件与tree `8fb09167...e8d9`精确一致，base layers exact match，4份rotation、runtime expectation、7个冻结native extension及3项运行环境均通过。
- 2026-08-03（ea8 Phase6报告口径）：沿用2.232/2.233已验收口径，以独立结构化validation与可复算manifest闭合builder/verifier。由于本轮恢复时两步已连续完成，正式记录用同一2.253章节同步两项终局，并完整披露symlink、静默预启动和shared-clone alternates三类失败边界；不得把daemonless OCI通过写成daemon导入、overlay runtime import、精度或性能已通过。
- 2026-08-03（ea8 Phase6原始日志一致性）：`build_v2.exit_code=0`、`verify_v2.exit_code=0`，两份stderr均0 bytes，stdout分别与`build_report.json`、`verification_report.json`逐字节一致；首次`build.exit_code=1`且stderr精确为rotation actual=[]/expected四文件。candidate输入仍冻结base manifest `2fdfbe...489d`、source ea8/tree 8fb、Dockerfile `d06401c8...7a1c`、4份rotation、runtime expectation与7个native预期，适合用现有verifier逻辑独立复算OCI blob和base层。
- 2026-08-03（独立validation容器入口边界）：固定833 runtime镜像自身Entrypoint为Python，直接追加`python3 script`会让Entrypoint把另一个Python ELF当脚本读取，得到exit126/cannot execute binary file；该轮validation脚本未执行。修正为显式`--entrypoint /bin/bash -lc 'python3 ...'`，不改变验证内容。
- 2026-08-03（ea8 Phase6独立validation）：修正Entrypoint后，固定833 runtime、network none、4 CPUs、GPU不可见容器自然exit0，结构化validation 38/38 passed。除报告字段外，独立复算candidate manifest/config/layer SHA、Phase0 base manifest与32层完全一致、candidate tar 5,298 members且无native/whiteout、overlay 4,749 regular/0 symlink，并覆盖首次builder exit1、两次有效exit0、stdout/report一致及空stderr。
- 2026-08-03（ea8 Phase6证据manifest闭合）：`evidence_manifest.sha256`覆盖22项原始日志、退出码、clone状态、独立脚本/validation、build/verification报告、OCI index/layout及candidate manifest/config/layer blobs；从项目根独立`sha256sum -c`为22/22 OK。control目录17文件/33,465 bytes；validation SHA=`0c49e811...b013`，manifest SHA=`2229c33d...516b`，check SHA=`002b3e12...30d0`。
- 2026-08-03（2.253报告门禁）：报告修改前完整扫描且与HEAD逐字节一致；追加builder/verifier终局、失败边界、38/38与22/22后，报告15,770行/967,510 bytes、SHA=`446b0bb7...27de`。2.1–2.253连续，2.252交叉引用、证据路径、核心大小/hash、三池0、uppercase A800仅历史文件名2处、苹果800109处及diff-check全部通过。
- 2026-08-03（2.253发布）：正式记录与planning由主仓提交`cc44ba5b9500c5554c6744628391e53d3e1f452e`通过GitHub HTTPS发布；fetch后主仓local/upstream相等，source仍为ea8 local/upstream相等。主工作树仅保留Shawn的未跟踪报告且未纳入提交。
- 2026-08-03（ea8 overlay source-import前状态）：正确native manifest位于source子仓`glm52_oscar_vllm/recovery/native_extensions.sha256`，不是主仓根；首次只读查询错路径但未改文件。新overlay当前4,749 regular/0 symlink，6个vLLM native目标全部不存在；冻结manifest前6项正是既有overlay链接集合，第7项为source目录外stable sparse MLA算子，不应新增链接。复用2.234口径：6个绝对链接指向项目内Phase0 rootfs同相对路径，随后在GPU不可见容器中把overlay内部`opt/vllm_glm52_v1`挂到`/opt/vllm_glm52_v1`并验证ea8 source身份、两处prefill K=768、3个metadata字段及CUDA未初始化。
- 2026-08-03（ea8 source-import证据位置边界）：Phase6输出目录由builder容器创建且root-owned，首次用apply_patch向其写验证脚本被权限拒绝，未写入文件。不得为此更改整个产物所有权；改把脚本/小型证据写到用户可写的`artifacts/phase9-control/20260803T1040Z_splitk_stride_phase6_build_v1`，只通过容器root精确创建overlay的6个链接。
- 2026-08-03（ea8 native链接attempt1边界）：首次固定833无GPU容器在预哈希阶段从项目根执行`sha256sum -c`，manifest相对路径无法解析而exit1；`set -e`在任何`ln -s`前终止，overlay仍0 links。修正为在Phase0 native根子目录内复算前6项，不改变链接目标。
- 2026-08-03（ea8 overlay/source import绿灯）：修正预哈希工作目录后，Phase0前6项native SHA逐项通过并精确创建6个绝对链接；overlay保持4,749 regular。canonical固定833 CPU-only容器自然exit0：source路径为overlay ea8，两处prefill K=768、3个metadata字段、split-K stride helper存在、torch2.11.0+cu129且CUDA未初始化。`vllm._C`因无`libcuda.so.1`产生预期warning，不能写成driver-visible native import通过。独立overlay validation自然exit0/status passed。
- 2026-08-03（ea8 overlay/source证据闭合）：专属manifest覆盖build/verification报告、两个validator、6-link清单、目标hash清单、canonical log/exit和validation共9项，项目根独立复算9/9 OK。核心SHA：links=`e01d7637...cebd`、targets=`59eb3786...ab3`、source log=`1d5d350a...bfbb`、validation=`4d0992da...05ff`、manifest=`ab6a847a...cf16`、check=`9e65c853...cb60`。
- 2026-08-03（2.254报告门禁）：修改前报告完整扫描并与HEAD cc44逐字节一致；追加overlay links/source import后，报告15,845行/972,512 bytes、SHA=`a007e4c8...18c9`。2.1–2.254连续，2.253引用、9/9 manifest、核心hash、三池0、uppercase A800历史文件名2处、苹果800109处及diff-check全部通过。
- 2026-08-03（2.254发布）：报告/findings/progress由主仓提交`c204f869781dc863430c541a0abaf311f1fded0f`通过GitHub HTTPS发布；fetch后主仓local/upstream一致，source仍ea8 clean/upstream，主仓仅Shawn未跟踪报告。
- 2026-08-03（ea8 daemon导入前状态/口径）：目标tag `glm52-oscar-a800-phase6-ea8ae6b77-0275043c:latest`实测inspect输出`[]`、No such image、exit1，故不会覆盖同名镜像；无相关导入进程。宿主无skopeo，本地已有已验收`ubuntu:22.04`。复用2.235口径：一次性CPU-only Ubuntu容器经阿里云源安装skopeo，OCI只读+Docker socket；导入后独立验证image/config ID、tag、33层、末diff-ID及8项labels。
- 2026-08-03（ea8 daemon导入/审计绿灯）：一次性Ubuntu CPU-only工具容器安装skopeo1.4.1后，33 blobs与config `1bd0a551...bfba`完成复制，日志自然到达Writing manifest/Storing signatures，outer exit0且容器删除。固定833无GPU独立审计自然exit0/status passed，5项ID/tag/33层/末diff-ID/8 labels全部true；daemon image ID=`sha256:1bd0a551...bfba`，末diff-ID=`sha256:aa212c18...f15f`。
- 2026-08-03（ea8 daemon证据闭合）：专属manifest覆盖build/verification报告、runner/validator、preimport stdout/stderr/exit、import log/exit、daemon inspect和audit共11项，项目根独立复算11/11 OK；导入日志33行Copying blob。核心SHA：import log=`aae9c7ab...befc`、inspect=`47059f32...3273`、audit=`f3587f74...2eaf`、manifest=`bbf32015...cfb9`、check=`0fe3bc85...5cc0`。validator在导入启动前自审发现layer_count括号错误并最小修正，py_compile通过且两个临时pyc已清理，因此没有无效审计轮。
- 2026-08-03（2.255报告门禁）：修改前完整扫描报告并与HEAD c204逐字节一致；追加daemon导入/审计后，报告15,907行/976,976 bytes、SHA=`0f829f04...7163`。2.1–2.255连续，2.254引用、11/11 manifest、核心hash、三池0、uppercase A800历史文件名2处、苹果800109处与diff-check全部通过。
- 2026-08-03（2.255发布）：报告/findings/progress由主仓提交`b67d5f50e82d76bb48c1ae67b1fad6becf2f3ead`通过GitHub HTTPS发布；fetch后主仓local/upstream一致，source仍ea8 clean/upstream，主仓仅Shawn未跟踪报告。
- 2026-08-03（ea8 Stage9 base TDD前状态）：`docker/Dockerfile.phase9-runtime`首行及目标测试仍期望833 Phase6，活动Phase5/7/9身份也仍为833；新目标control tag `oscar-glm-stage9-runtime:ea8ae6b77`不存在。首次用宿主Python3.8调用目标测试因`datetime.UTC`缺失在collection失败，不是有效红灯且未改文件；后续固定用833 control内Python3.12、network none、GPU不可见运行。
- 2026-08-03（ea8 Stage9 base有效红灯）：首次容器选择器误用不存在的`Phase9ToolTests`而attribute error，不计有效红灯；修正为实际`Stage9ToolsTest`后，固定833 Python3.12无GPU容器目标测试精确以833 Dockerfile首行!=ea8期望而1 failure，无collection error。最小实现只改Dockerfile第一行到ea8 tag，活动Phase5/7/9不变。
- 2026-08-03（ea8 Stage9 base绿灯）：最小实现后，同一固定833 Python3.12、network none、GPU不可见容器中目标1/1、完整Stage9工具24/24通过；旧`test_prefusion_rollback_runtime_identities_are_wired`仍通过，证明Phase5/7/9活动身份未提前迁移。受跟踪代码改动仅Dockerfile首行与目标测试名/期待。
- 2026-08-03（Stage9 base validator attempt1边界）：state capture确认ea8 Phase6 base daemon存在且新control tag不存在；首轮固定833无GPUvalidator在只读`git diff`处因挂载仓未配置safe.directory而exit1，尚未生成validation/manifest、未改代码/镜像。最小修正仅为该Git子命令增加精确仓路径的`-c safe.directory=...`。
- 2026-08-03（ea8 Stage9 base证据闭合）：修正safe.directory后独立validation 16/16 passed；manifest覆盖runner/capture/validator、无效选择器、有效red、target/full green、base/target inspect、diff、validation及两处tracked文件共19项，项目根复算19/19 OK。Dockerfile/test SHA=`18616e43...b5e3`/`049184db...4777`，validation/manifest/check SHA=`99949aff...1b8b`/`ddf18569...539e`/`340f3c2c...1b26`。
- 2026-08-03（2.256报告门禁）：修改前报告完整扫描且与HEAD b67逐字节一致；追加Stage9 base TDD后，报告15,979行/981,658 bytes、SHA=`84310493...bad0`。2.1–2.256连续，2.255引用、16/16 validation、19/19 manifest、核心hash、三池0、uppercase A800历史文件名2处、苹果800109处与diff-check全部通过。
- 2026-08-03（2.256发布）：报告、Stage9 Dockerfile/目标测试与planning由主仓提交`272b3c76ca56abd6768f10d25732117ea32f2669`通过GitHub HTTPS发布；fetch后主仓local/upstream一致，source仍ea8 clean/upstream，主仓仅Shawn未跟踪报告。
- 2026-08-03（ea8 control build attempt1边界）：目标tag再次确认不存在；首轮BuildKit用外部`--file`加stdin context `-`立即报ambiguous Dockerfile source并exit1，未执行layer/创建镜像。失败log/exit已另名保留；改用Python TemporaryDirectory作为真实空目录context，保持已发布Dockerfile、`--pull=false`与同一tag。
- 2026-08-03（ea8 control build绿灯）：修正空context后build自然exit0；31.4MB索引10s、5.411MB包、RUN16.9s、export0.3s。新control ID=`sha256:ad0f218b...ebf4`；独立继承审计10/10 true：base `1bd0a551...bfba` 33层，control 34层，前33 diff-ID完全相同，新末层=`52781cf9...7127`，8项Phase6 labels与ea8 commit/tree继承，Entrypoint bash/Cmd null。
- 2026-08-03（ea8 control build证据闭合）：build manifest覆盖runner/validator、prebuild边界、ambiguous失败、正式build、base/control inspect和audit共12项，项目根复算12/12 OK。核心SHA：build=`1a08a1d0...2779`、control inspect=`ef068dd8...4af8`、audit=`a446b6a4...6fce`、manifest=`aa11278d...5118`、check=`533a513b...187d`。
- 2026-08-03（2.257报告门禁）：修改前完整扫描报告并与HEAD 272逐字节一致；追加control build后，报告16,041行/985,885 bytes、SHA=`78b37bfa...5159`。2.1–2.257连续，2.256引用、10/10审计、12/12 manifest、核心hash、三池0、uppercase A800历史文件名2处、苹果800109处及diff-check全部通过。
- 2026-08-03（2.257发布/恢复）：报告/findings/progress由主仓提交`d93ee81cc31c9179fab355dc074060bdc76298a4`通过GitHub HTTPS发布；恢复核对主仓local/upstream一致、source ea8 clean/upstream，新control `ad0f218b...ebf4` 34层存在，主仓仅Shawn未跟踪报告。task checkpoint已推进到CPU runtime preflight。
- 2026-08-03（ea8 CPU preflight验收口径）：新control需在network none/GPU不可见下验证Python3.12.13、glibc2.35、git/iproute2冻结版本、vLLM路径、CUDA empty/count0/uninitialized、显式prefill K=768。decode/store SHA继续为`13953366...42f`/`ec82245e...b8e`且pre-fusion rotate+add路径不变；新增Indexer SHA必须为`a80b5d59...9e57`，`_prepare_native_topk_output`存在且源码包含contiguous临时输出合同。
- 2026-08-03（ea8 CPU preflight绿灯）：固定新control `oscar-glm-stage9-runtime:ea8ae6b77`、network none、4 CPUs、GPU不可见运行自然exit0。实测Python3.12.13、glibc2.35、git/iproute2冻结版本、vLLM路径正确，CUDA visible为空/device_count=0/initialized=false，K=768；decode/store/Indexer SHA分别为`13953366...42f`/`ec82245e...b8e`/`a80b5d59...9e57`，pre-fusion路径保持且stride helper明确使用contiguous临时输出。`vllm._version`缺失与无CUDA runtime warning均为预期CPU-only边界。
- 2026-08-03（ea8 CPU preflight证据闭合）：固定833无GPU独立validator为22/22 passed；runtime manifest覆盖build身份、三个validator/runner、stdout/exit/结构化结果共13项，项目根复算13/13 OK。核心SHA：runtime JSON=`2ca1d0f8...2e60`、validation=`509b8efb...2256`、manifest=`d9498317...aa7c`。本阶段未暴露GPU，不能写成driver-visible native import通过。
- 2026-08-03（2.258报告门禁）：修改前完整扫描985,885-byte报告并确认SHA=`78b37bfa...5159`与HEAD一致、无手工diff；追加CPU preflight后，报告16,092行/989,398 bytes、SHA=`c30a59d7...89c9`。2.1–2.258连续，2.257引用、22/22 validation、13/13 manifest、核心hash、三池0、uppercase A800仅历史文件名2处、苹果800109处及diff-check全部通过。
- 2026-08-03（ea8 driver-import GPU门禁原始结果）：2.258已由主仓`ba71a25d58f017525f21c23382ae5f758d0ec8fb`发布并与upstream一致；source仍ea8。双采样为11:09:46Z/11:10:53Z、间隔67秒，两轮GPU0–7均0 MiB/0%、compute区段为空，固定833 CPU-only独立validation 10/10 passed。
- 2026-08-03（GPU门禁manifest首次根目录复算边界）：finalizer内部按证据目录解析8项manifest全部OK，但manifest首版只写basename；紧接着从项目根直接`sha256sum -c`因相对基准错误而8项均报No such file，不是内容hash失败。下一步把manifest改为项目根相对路径并从根独立复算，不重复错误命令。
- 2026-08-03（ea8 driver-import GPU门禁证据闭合）：manifest改为项目根相对路径后，finalizer内部与项目根独立复算均8/8 OK；validation仍为10/10 passed、67秒间隔和两轮全idle不变。核心SHA：first=`7dc218c3...40e`、second=`d9cb4f19...004`、validation=`4c572ef7...1d1`、manifest=`ca4b8d94...956`、check=`041cc9c5...7c0`。下一步先实时追加并发布正式记录2.259，发布前不启动GPU探针。
- 2026-08-03（2.259报告门禁）：修改前完整读取989,398-byte报告并确认SHA=`c30a59d7...89c9`与HEAD ba71一致、无手工diff；追加GPU双空闲门禁后，报告16,137行/992,469 bytes、SHA=`6835174d...259`。2.1–2.259连续，2.258引用、10/10 validation、8/8 manifest、核心hash、三池0、uppercase A800仅历史文件名2处、苹果800109处及diff-check全部通过。
- 2026-08-03（2.259发布与GPU探针设计）：门禁由主仓`8ffd4cee369ddcebacf49f700eef1e3047fbea77`发布并与upstream一致；即时宿主复核8卡仍0 MiB/0%、compute空。新探针固定新control/GPU0/network none/4 CPUs/K=768，只导入vLLM native/source，新增ea8 Indexer SHA与contiguous helper源码断言，不加载模型或运行推理kernel；runner在启动前再次fail-closed检查8卡全idle并在退出后采样/确认容器删除。
- 2026-08-03（ea8 driver-visible native import绿灯）：固定GPU0/new control/network none/4 CPUs/K=768探针自然exit0；`vllm._C.abi3.so`成功导入、visible device count=1、CUDA导入前后均uninitialized。decode/store/Indexer SHA、metadata、两处K=768、rotation/runtime expectation、pre-fusion路径及contiguous helper全部通过；只出现既有`vllm._version` warning，无Traceback。11:16:08Z启动前与11:16:18Z退出后8卡均0 MiB/0%、compute空，容器已删除。
- 2026-08-03（ea8 driver import证据闭合）：固定833 CPU-only独立validation 30/30 passed；manifest覆盖probe/runner、image inspect、前后GPU状态、runtime stdout/stderr/exit、容器状态和独立validation共14项，项目根复算14/14 OK。核心SHA：stdout=`642b7128...1d23`、validation=`79865b33...2ef3c`、manifest=`a44f510d...879f`、check=`c792742d...3d89`。下一步完整重读正式记录并实时追加2.260，发布前不迁移Phase5/7/9。
- 2026-08-03（2.260报告门禁）：修改前完整读取992,469-byte报告并确认SHA=`6835174d...259`与HEAD 8ffd一致、无手工diff；追加driver-visible import后，报告16,200行/996,794 bytes、SHA=`2909732c...7c8f`。2.1–2.260连续，2.259引用、30/30 validation、14/14 manifest、核心hash、三池0、uppercase A800仅历史文件名2处、苹果800109处及diff-check全部通过。
- 2026-08-03（ea8活动身份迁移前依赖）：2.260已由主仓`755c38ac5384c4504059f45f9e8f0dcc097f2d04`发布。盘点确认Phase5/7/9仍含833身份的范围精确为10个活动文件；但ea8 Phase6目录当前缺少`runtime_import.json`，而Phase7合同显式依赖该canonical文件，因此不能直接迁移。下一步先复用2.243口径，用新control CPU-only实测依赖并结合2.260 driver结果生成ea8 canonical runtime import，实时记录并发布后才做TDD迁移。
- 2026-08-03（ea8 canonical runtime import绿灯）：新control、network none、4 CPUs、GPU不可见的依赖测量自然exit0/stderr空，实测Python/Torch/Triton/Transformers/Tokenizers/FlashInfer、78层rotation、reasoning max和CUDA未初始化均符合冻结口径。结合2.260 driver结果与ea8 Phase6 overlay三项实算hash生成717-byte canonical，SHA=`9bdfc8ca...3b20`，用exclusive create落到新Phase6目录、mode644，未覆盖既有文件。
- 2026-08-03（ea8 canonical证据闭合）：固定833 CPU-only独立validation 20/20 passed；16项跨目录manifest覆盖measurement/build/install/validator、新Phase6 canonical及2.260 driver结果，项目根复算16/16 OK。validation/manifest/check SHA=`65ba8b93...53f0`/`f7c0e373...9d6d`/`830327d5...df4d`。下一步完整重读报告并实时追加2.261，发布前不修改10个活动身份文件。
- 2026-08-03（2.261报告门禁）：修改前完整读取996,794-byte报告并确认SHA=`2909732c...7c8f`与HEAD 755一致、无手工diff；追加canonical runtime import后，报告16,257行/1,001,037 bytes、SHA=`bbeea492...2767`。2.1–2.261连续，2.260引用、20/20 validation、16/16 manifest、核心hash、三池0、uppercase A800仅历史文件名2处、苹果800109处及diff-check全部通过。
- 2026-08-03（ea8活动身份TDD红灯/范围修正）：2.261已由主仓`216a56ddef35a586e689a734ceb97d57d9ee76e1`发布；只改Phase9聚合合同期待ea8后，固定833无GPU目标测试有效红灯精确命中Phase5仍为833，1 failure。最初按833 commit/path检索得到10文件；补查旧candidate manifest/config/layer digest后发现两个official wrapper只嵌旧manifest digest，因此完整最小迁移范围应为12文件，与2.244同类口径一致，禁止遗漏。
- 2026-08-03（ea8活动身份最小实现/递归attempt1边界）：12文件已最小替换到ea8，目标1/1、Phase9工具24/24、3 JSON+8 shell+1 Python静态12/12及Phase7单测20/20通过，活动旧833 commit/tree/digest/control为0处。并行Phase7/Phase9递归首轮均因runner未挂载主机模型目录，在Phase5读取`config.json`时FileNotFoundError并exit1；这是容器输入缺失，未生成递归validation。下一轮只增加模型目录只读mount并新日志名重跑，不重复原命令。
- 2026-08-03（ea8活动身份递归/证据闭合）：仅补模型目录只读mount后Phase5 87/87、Phase7 44/44、Phase9 70/70全部passed；独立静态身份validation 25/25，62项manifest从项目根复算62/62 OK。12活动文件hash已冻结，三配置SHA为`ca439103...5554`/`fb999388...446a`/`3f47df52...d95`；validation/manifest/check SHA=`1b72b5fb...ba5e`/`6ca3275b...c029`/`30a7906e...2985`。下一步完整重读正式记录并实时追加2.262，发布前不启动GPU精度门禁。
- 2026-08-03（2.262报告门禁）：修改前完整读取1,001,037-byte报告并确认SHA=`bbeea492...2767`与HEAD 216一致、无手工diff；追加活动身份迁移后，报告16,333行/1,006,655 bytes、SHA=`2a9ebeb0...4499`。2.1–2.262连续，2.261引用、87/87+44/44+70/70、25/25、62/62 manifest、核心hash、三池0、uppercase A800仅历史文件名2处、苹果800109处及diff-check全部通过。
- 2026-08-03（ea8固定256题前GPU门禁）：2.262与12文件迁移已由主仓`2e87af3da61a8db3d943deeb59270645a7c87713`发布。新双采样为11:37:56Z/11:39:04Z、间隔68秒，两轮GPU0–7均0 MiB/0%、compute空；固定833 CPU-only validation 10/10、manifest项目根8/8通过。核心first/second/validation/manifest/check SHA=`481e60b5...2fd4`/`d473799c...e4a8`/`11cfb83b...fede`/`4477c2aa...5d1e`/`3435e319...5f88`。下一步先实时追加并发布2.263，发布前不启动256题长跑。
- 2026-08-03（2.263报告门禁）：修改前完整读取1,006,655-byte报告并确认SHA=`2a9ebeb0...4499`与HEAD 2e87一致、无手工diff；追加固定256题前GPU门禁后，报告16,372行/1,009,070 bytes、SHA=`fb200f85...c026`。2.1–2.263连续，2.262引用、10/10 validation、8/8 manifest、核心hash、三池0、uppercase A800仅历史文件名2处、苹果800109处及diff-check全部通过。
- 2026-08-03（2.263发布后恢复核对）：主仓HEAD/upstream均为`409856934f3430b6afcdddcaca25bddf383faac3`，source HEAD/upstream均为`ea8ae6b7758ae2b4db7cae44d638ae5de80148ac`；未发现遗留official_v5、candidate、vLLM或OSCAR实验进程，GPU0–7即时采样均0 MiB/0%、compute空。主仓唯一状态是Shawn未跟踪的`GLM-5.2_OSCAR_苹果800适配与性能优化报告.md`，正式wrapper要求仓库包含未跟踪文件在内完全clean，因此必须使用`/dev/shm`干净临时克隆，不得读取、修改、暂存或删除该用户文件。
- 2026-08-03（fixed256启动边界复核）：official_v5 isolated wrapper已固定source ea8及manifest `ffc1d3...eb2b`，fast runner每600秒从runner状态与server success打印进度；`/dev/shm`当前可用461 GiB，足够承载评测证据。正式候选参数保持`EVALUATION_TIER=fast`、`FAST_SAMPLE_COUNT=256`、`FAST_CONCURRENCY=16`、固定GPU0–7。
- 2026-08-03（临时克隆组装attempt1边界）：主仓shared clone本身包含受跟踪的`artifacts/phase9-control`目录，故`ln -s <原仓artifacts> <clone>/artifacts`实际在既有目录下创建了`artifacts/artifacts`，冻结评测器预期路径不存在，组装命令在`realpath`处exit1。未运行formal preflight、未创建容器、未占用GPU。下一轮只移除本轮误建的精确symlink，并为clone内实际缺失的Phase6及两个冻结评测器目录创建精确链接，不重复整目录链接方案。
- 2026-08-03（临时克隆组装/preflight修正）：只移除本轮误建symlink后，精确链接ea8 Phase6 overlay与frozen evaluator v4/v5；三路径realpath均正确，主/source clone仍含untracked在内clean，official_v5 static及隔离network namespace preflight自然通过。
- 2026-08-03（fixed256启动前即时双空闲）：11:45:42Z与11:46:52Z间隔70秒，两轮GPU0–7均0 MiB/0%，compute查询为空；11:46:21Z中间采样同样全idle但距首轮39秒，不单独作为有效第二轮。满足固定GPU0–7启动授权。
- 2026-08-03（ea8 fixed256 v1启动失败终局）：正式run ID `20260803T1155Z_candidate_ea8_splitk_stride_fast256_c16_v1`于11:49:05Z启动；static/network隔离通过，但candidate wrapper在模型加载前因clean clone遗漏`artifacts/phase0-candidate-bundle`，找不到下层`vllm/_C.abi3.so`而fail-closed。11:49:43Z outer exit1，GPU始终0 MiB/0%、compute空，无candidate容器、无精度结果。9项失败证据manifest项目根复算9/9 OK。下一轮只精确链接冻结Phase0 bundle并换新run ID。
- 2026-08-03（2.264实时记录）：修改前完整读取1,009,070-byte报告并确认SHA=`fb200f85...c026`与HEAD 409逐字节一致；已追加clean clone组装边界、v1模型加载前失败、GPU未占用、无精度结论和修正路径。下一步执行章节、术语、核心hash、9/9 manifest与diff门禁并发布，发布前不启动v2。
- 2026-08-03（2.264报告门禁）：报告现为16,426行、1,012,966 bytes、SHA=`5499cf69...9916e`；2.1–2.264连续，2.263交叉引用有效，三池0处、uppercase A800仅历史文件名2处，9/9 evidence manifest、核心hash与diff-check全部通过。精确提交仅报告及三份planning，不纳入Shawn未跟踪报告或ignored实验产物。
- 2026-08-03（2.264发布）：主仓`21166a8`已HTTPS发布并与upstream一致；source仍为ea8 clean/upstream。v2必须先把clean clone主仓快进到21166a8，否则published_commit会因clone 409 != remote 211而fail-closed。
- 2026-08-03（v2 clone快进断言失败边界）：clean clone已成功fast-forward到实际`21166a82c53de94fefaa2288f21b410b5556e62c`，但组合命令使用了未核实的错误补全SHA并在精确等值断言处exit1；Phase0链接尚未创建、preflight/容器/GPU均未执行。下一步使用`git rev-parse`实测完整SHA继续，禁止猜测短SHA补全。
- 2026-08-03（v2输入/preflight闭合）：按实测主仓完整SHA继续，只新增原仓冻结`artifacts/phase0-candidate-bundle`精确symlink；下层`_C.abi3.so`存在且SHA=`1812bd98...ec70`，clone主/source含untracked在内clean，static/network namespace preflight通过。v2专属双空闲为11:54:08Z/11:55:13Z、间隔65秒，两轮GPU0–7均0 MiB/0%、compute空。
- 2026-08-03（ea8 fixed256 v2失败终局）：v2于11:55:36Z启动、11:56:14Z exit1；static/network隔离通过，但wrapper在模型加载前把runtime symlink resolved路径与未解析clone base字符串比较而报unexpected path。独立审计为lexical_equal=false、resolved_equal=true，两侧实际为同一native文件；GPU始终未占用，无精度结果。10项manifest项目根复算10/10 OK。
- 2026-08-03（2.265实时记录）：修改前完整读取1,012,966-byte报告并确认SHA=`5499cf69...9916e`与HEAD 211逐字节一致；已追加v2输入、双空闲、路径等价性误判、GPU未占用及TDD修复路线。下一步章节/术语/hash/diff门禁并发布，发布前不修改wrapper。
- 2026-08-03（2.265报告门禁）：报告16,481行、1,016,733 bytes、SHA=`e4d077f5...5ea2`，2.1–2.265连续；2.264引用、三池0、uppercase A800历史文件名2处、核心hash、10/10 manifest与diff-check通过。
- 2026-08-03（native路径canonical TDD红灯）：2.265已由`5d1229f`发布；新增Phase7静态合同只要求base侧`readlink -f`、比较使用`base_resolved`且禁止旧未解析比较。固定ea8 control、network none、4 CPUs、GPU不可见运行目标1例得到有效红灯exit1，精确缺失`base_resolved`，production尚未修改。
- 2026-08-03（native路径canonical TDD闭合）：production只新增base resolved变量并替换比较右侧；syntax、目标1/1、Phase7 21/21、Phase9 24/24、递归87/87+44/44+70/70、独立16/16、manifest 35/35通过。Phase7 attempt1因缺少recovery Python bind而20 pass+1 error/exit127，补既有只读mount后全绿。wrapper/test SHA=`3929a6a2...8d9c`/`30e37c4f...7904`。
- 2026-08-03（2.266实时记录）：修改前完整读取1,016,733-byte报告并确认SHA=`e4d077f5...5ea2`与HEAD 5d1逐字节一致；已追加最小代码、红绿灯、输入失败边界、全部回归与35项manifest。下一步执行章节、术语、hash和diff门禁并发布，发布前不启动v3。
- 2026-08-03（2.266报告门禁）：报告16,538行、1,020,828 bytes、SHA=`09d5dbfe...c2ec`，2.1–2.266连续；2.265引用、三池0、uppercase A800历史文件名2处、核心hash、35/35 manifest及diff-check通过。
- 2026-08-03（2.266发布）：wrapper两行canonical修复、目标合同、报告/planning已由主仓`c4a79fb226dc4d820ee2bff5cacea7e4a47f3b83`通过GitHub HTTPS发布并与upstream一致；source保持ea8 clean/upstream。
- 2026-08-03（v3 clean-clone preflight attempt1边界）：clone已快进c4a且native canonical误判不再出现；但candidate preflight在Phase5 verifier因clone缺少配置显式依赖的Phase2 rotation manifest而exit1，继发Phase7读不到stage5 preflight。未创建模型服务/使用GPU。下一轮按配置路径补Phase2 rotation与Phase7 frozen baseline精确只读链接，不重复缺输入方案。
- 2026-08-03（v3 clean-clone preflight attempt2边界）：补Phase2与frozen baseline后已越过Phase5输入，但固定容器root执行source `git ls-tree`触发dubious ownership/exit128；未启动模型/GPU。下一轮只在容器临时HOME增加clone主/source精确safe.directory，不污染宿主配置。
- 2026-08-03（v3 clean-clone preflight attempt3边界）：临时safe.directory后，Phase7的OCI/source/native/rotation/baseline/server绝大部分断言均通过，但嵌套Phase5 preflight仍status failed；汇总未包含子检查详情。三次均为不同输入/环境边界且无GPU。停止重复整套preflight，改为直接运行Phase5 verifier持久化JSON定位唯一失败项。
- 2026-08-03（v3 Phase5直接诊断）：独立Phase5为86/87，唯一失败`source.rootfs_runtime_tree_match`；抽查blob SHA完全一致，但宿主解包source文件mode为0777而Git期望100644。既有递归门禁使用只读named volume `oscar-glm-phase0-source-fd3e0b3`恢复冻结mode并已87/87。下一轮改为容器内同路径只读volume覆盖，不chmod/修改宿主artifact。
- 2026-08-03（v3 clean-clone preflight v4边界）：加入冻结source volume后Phase5/Phase7全部结构化检查passed，native canonical路径也通过；最终fixed-environment import仅因`NVIDIA_VISIBLE_DEVICES=void`未注入`libcuda.so.1`而exit1。未暴露GPU。下一步重新双空闲后固定8卡容器、CUDA_VISIBLE_DEVICES空完成driver-library-visible但device-hidden preflight。
- 2026-08-03（v3 driver preflight启动参数边界）：12:14:32Z/12:15:37Z间隔65秒双空闲通过；首次Docker `--gpus device=0,...,7`被旧daemon解析为Count与DeviceIDs冲突，创建前exit125，12:15:56Z仍8卡全idle。改用主机恰为8卡条件下等价的`--gpus all`固定8卡，CUDA_VISIBLE_DEVICES仍为空。
- 2026-08-03（v3有效preflight/GPU门禁闭合）：`--gpus all`固定宿主8卡、CUDA_VISIBLE_DEVICES空、network none、冻结source volume只读条件下preflight自然exit0，结构/native/fixed environment通过且cuda_initialized=false；12:17:06Z后8卡仍全idle。独立15/15、manifest 32/32通过。
- 2026-08-03（2.267实时记录）：修改前完整读取1,020,828-byte报告并确认SHA=`09d5dbfe...c2ec`与HEAD c4a逐字节一致；已追加完整preflight边界、有效固定8卡身份、三次idle、15/15与32/32。下一步章节/术语/hash/diff门禁并发布，发布前不启动v3模型服务。
- 2026-08-03（2.267报告门禁）：报告16,598行、1,024,695 bytes、SHA=`6605a8ee...aa79`，2.1–2.267连续；2.266引用、三池0、uppercase A800历史文件名2处、核心hash、32/32 manifest及diff-check通过。
- 2026-08-03（v3发布后clone断言边界）：2.267已由`d23bb65c65f5c8353693b15472e56d41f8e225a1`发布；clone成功fast-forward，但再次因手工猜错短SHA补全而在Docker preflight前exit1。无容器/GPU副作用。后续永久改为比较clone HEAD与实测origin branch，不再手工补全短SHA。
- 2026-08-03（ea8 fixed256 v3失败终局）：容器official static/namespace preflight通过；12:22:40Z启动后嵌套userns因外层Docker root与宿主输出目录UID22633不一致，在mkdir run dir处Permission denied，12:22:50Z exit1。模型/CUDA/答题未启动，8卡始终idle；7/7 manifest通过。下一轮外层容器只改`--user 22633:22633`。
- 2026-08-03（2.268实时记录）：修改前完整读取1,024,695-byte报告并确认SHA=`6605a8ee...aa79`与HEAD d23逐字节一致；已追加v3容器preflight、权限失败、GPU未占用和唯一变量修正。下一步章节/术语/hash/diff门禁并发布。
- 2026-08-03（2.268报告门禁）：报告16,639行、1,027,396 bytes、SHA=`3372e823...b1c3`，2.1–2.268连续；2.267引用、三池0、uppercase A800历史文件名2处、核心hash、7/7 manifest与diff-check通过。
- 2026-08-03（2.268发布）：主仓`7c501d1`已HTTPS发布并与upstream一致；source仍ea8。下一步先验证22633外层用户在嵌套userns中可写宿主输出目录，再授权v4。
- 2026-08-03（v4权限/GPU门禁闭合）：22633:22633外层容器的嵌套userns写探针exit0，内层0:0映射到宿主22633:22633、mode755；12:26:25Z/12:27:30Z间隔65秒两轮8卡全idle。独立10/10、manifest15/15通过。
- 2026-08-03（2.269实时记录）：修改前完整读取1,027,396-byte报告并确认SHA=`3372e823...b1c3`与HEAD 7c5逐字节一致；已追加权限探针、双空闲、10/10与15/15。下一步章节/术语/hash/diff门禁并发布。
- 2026-08-03（2.269报告门禁）：报告16,680行、1,029,836 bytes、SHA=`1e55d442...391a`，2.1–2.269连续；2.268引用、三池0、uppercase A800历史文件名2处、核心hash、15/15 manifest与diff-check通过。
- 2026-08-03（ea8 fixed256 v4失败终局）：12:31:00Z启动，official namespace与输出mkdir通过；Phase5按冻结manifest访问root:root原仓source，被Git dubious ownership拒绝，12:31:41Z模型加载前exit1。8卡始终idle、无accuracy，7/7 manifest通过；下一轮仅补原仓主/source精确safe.directory。
- 2026-08-03（2.270实时记录）：修改前完整读取1,029,836-byte报告并确认SHA=`1e55d442...391a`与HEAD 69f逐字节一致；已追加v4有效权限修正、Git安全目录失败、无GPU结果和修复范围。下一步门禁发布。
- 2026-08-03（2.270报告门禁）：报告16,720行、1,032,448 bytes、SHA=`941b7608...718d`，2.1–2.270连续；2.269引用、术语、核心hash、7/7 manifest与diff-check通过。
- 2026-08-03（2.270发布）：主仓`1a966b6`已HTTPS发布；下一步不直接启动，先用v5精确容器身份验证四条safe.directory下candidate preflight通过，并重做双空闲。
- 2026-08-03（v5精确preflight attempt1边界）：12:34:36Z/12:35:41Z间隔65秒双空闲通过；四条safe.directory下结构检查越过v4失败，但22633 UID在容器passwd中无条目，Torch cache调用getpass.getuser触发KeyError，preflight exit1；12:36:41Z后8卡仍idle。下一轮仅增加USER/LOGNAME=zhangleichao。
- 2026-08-03（v5精确preflight v2闭合）：仅增加`USER/LOGNAME=zhangleichao`后，固定22633用户、四safe路径、8卡driver可见但`CUDA_VISIBLE_DEVICES`为空的candidate preflight自然exit0，结构检查status passed、`cuda_initialized=false`，stderr仅已知`vllm._version`告警。首次退出后采样脚本漏传必需输出参数，生成0-byte无效文件；立即按正确参数重采为12:40:24Z的8卡0 MiB/0%、compute空。独立validation 15/15、manifest 16/16通过；下一步先实时追加并发布报告2.271，再启动正式v5 fixed256。
- 2026-08-03（2.271实时记录与门禁）：修改前完整读取1,032,448-byte报告并确认SHA=`941b7608...718d`与HEAD一致；追加后报告16,778行/1,036,365 bytes、SHA=`c4103877...a4e4`，2.1–2.271连续，术语、新交叉引用、8项核心hash、15/15 validation、16/16 manifest与diff-check全部通过。下一步精确提交发布，发布前不启动长跑。
- 2026-08-03（2.271发布/v5正式启动）：主仓`5dd64f5e522f48d5fb0ad392e2149e42bc2c2083`已HTTPS发布且clone快进后main/source clean。v5于12:45:43Z以新run ID启动；official preflight与wrapper内置双空闲通过，12:50:02Z开始加载模型，rotation artifact与TRITON_MLA_SPARSE均确认，当前无fatal/OOM。首个10分钟精度节点约12:55:43Z。
- 2026-08-03（v5首个10分钟节点）：12:55:43Z sidecar为5/256完成、4正确，当前精度80.000000%、全量精度1.562500%，0 invalid/0 truncated；按checkpoint mtime独立复算一致。12:56:58Z容器/tmux存活，8卡各76,129 MiB、75%–97%，16 running/0 waiting、64.0–78.4 token/s，无fatal/OOM。12/12 validation、11/11 manifest通过；样本过小，不提前判断过门禁。
- 2026-08-03（2.272实时记录与门禁）：修改前完整读取1,036,365-byte报告并确认SHA=`c4103877...a4e4`与HEAD一致；追加后报告16,828行/1,039,766 bytes、SHA=`97107f22...0778`，2.1–2.272连续，术语、引用、7项核心hash、12/12 validation、11/11 manifest与diff-check通过。下一步发布该实时节点，实验不中断。
- 2026-08-03（2.272发布）：首个10分钟节点已由主仓`660524fe39ec448643f36d333e58b551980e7af3`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux持续运行。下一固定节点为20分钟，继续按相同口径打印并实时更新报告。
- 2026-08-03（v5 20分钟节点）：13:05:44Z sidecar为7/256完成、6正确，当前精度85.714286%、全量精度2.343750%，0 invalid/0 truncated；相对10分钟新增2题均正确。13:06:03Z容器/tmux存活，8卡各76,133 MiB、97%–100%，16 running/0 waiting、48.0–60.8 token/s，无fatal/OOM。12/12 validation、11/11 manifest通过。
- 2026-08-03（2.273实时记录与门禁）：修改前完整读取1,039,766-byte报告并确认SHA=`97107f22...0778`与HEAD一致；追加后报告16,863行/1,042,078 bytes、SHA=`d5850cde...6999`，2.1–2.273连续，术语、引用、7项核心hash、12/12 validation、11/11 manifest与diff-check通过。下一步发布实时节点，实验不中断。
- 2026-08-03（2.273发布）：20分钟节点已由主仓`632eb876ed3e48955ed59e37c8c4167b0a5fc207`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux持续运行，下一固定节点为30分钟。
- 2026-08-03（v5 30分钟节点）：13:15:44Z sidecar仍为7/256完成、6正确、85.714286%，0 invalid/0 truncated；相对20分钟无新增，但16 running/0 waiting、48.0–60.8 token/s和KV 13.5%–13.7%证明仍在生成。13:16:10Z容器/tmux存活，14/14 validation、11/11 manifest通过。
- 2026-08-03（2.274实时记录与门禁）：修改前完整读取1,042,078-byte报告并确认SHA=`d5850cde...6999`与HEAD一致；追加后报告16,898行/1,044,344 bytes、SHA=`8153a4d9...6f95`，2.1–2.274连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest与diff-check通过。下一步发布，实验不中断。
- 2026-08-03（2.274发布）：30分钟节点已由主仓`dc050af9ad05b91dbe702e54bb73e63618bc72c3`通过GitHub HTTPS发布，local/upstream一致；实验继续，下一固定节点为40分钟。
- 2026-08-03（v5 40分钟节点）：13:25:44Z仍为7/256、6正确、85.714286%，0 invalid/0 truncated；连续20分钟无新增，但16 running、49.6–57.6 token/s、KV 18.8%–19.0%证明仍在生成。13:26:05Z容器/tmux存活，14/14 validation、11/11 manifest通过。
- 2026-08-03（2.275实时记录与门禁）：修改前完整读取1,044,344-byte报告并确认SHA=`8153a4d9...6f95`与HEAD一致；追加后报告16,932行/1,046,502 bytes、SHA=`efe572b7...e680`，2.1–2.275连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest与diff-check通过。下一步发布，实验不中断。
- 2026-08-03（2.275发布）：40分钟节点已由主仓`4d3f5c5f551bc21cf3300b3496fcd3b0be88dbb5`通过GitHub HTTPS发布，local/upstream一致；实验继续，下一固定节点为50分钟。
- 2026-08-03（v5 50分钟节点）：13:35:44Z为35/256完成、17正确、48.571429%，0 invalid/16 truncated；相对40分钟新增28题、其中11正确。13:36:10Z容器/tmux存活，新批16 running、76.8–78.4 token/s、KV 2.6%–2.9%，14/14 validation、11/11 manifest通过。
- 2026-08-03（2.276实时记录与门禁）：修改前完整读取1,046,502-byte报告并确认SHA=`efe572b7...e680`与HEAD一致；追加后报告16,969行/1,048,859 bytes、SHA=`e51c86e4...32f1`，2.1–2.276连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest与diff-check通过。下一步发布，实验不中断。
- 2026-08-03（2.276发布）：50分钟节点已由主仓`46286eb0285b64696cb4bb9ecd1a5edd243bc112`通过GitHub HTTPS发布，local/upstream一致；实验继续，下一固定节点为60分钟。
- 2026-08-03（v5 60分钟节点）：13:45:44Z为37/256完成、18正确、48.648649%，0 invalid/16 truncated；相对50分钟新增2题、其中1正确。13:46:10Z容器/tmux存活，16 running、59.2–60.8 token/s、KV 9.3%–9.5%，14/14 validation、11/11 manifest通过。
- 2026-08-03（2.277实时记录与门禁）：修改前完整读取1,048,859-byte报告并确认SHA=`e51c86e4...32f1`与HEAD一致；追加后报告17,005行/1,051,094 bytes、SHA=`7d477c1f...cd67`，2.1–2.277连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest与diff-check通过。下一步发布，实验不中断。
- 2026-08-03（2.277发布）：60分钟节点已由主仓`062e7ce4d4eaedf9f11818fb65746284f63f39dc`通过GitHub HTTPS发布，local/upstream一致；实验继续，下一固定节点为70分钟。
- 2026-08-03（v5 70分钟节点）：13:55:44Z仍为37/256完成、18正确、48.648649%，0 invalid/16 truncated；无新增但16 running、59.2–60.8 token/s、KV 15.4%–15.5%。13:56:12Z容器/tmux存活，14/14 validation、11/11 manifest通过。
- 2026-08-03（2.278实时记录与门禁）：修改前完整读取1,051,094-byte报告并确认SHA=`7d477c1f...cd67`与HEAD一致；追加后报告17,038行/1,053,197 bytes、SHA=`cbcb9f58...b8115`，2.1–2.278连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest与diff-check通过。下一步发布，实验不中断。
- 2026-08-03（2.278发布）：70分钟节点已由主仓`2a855d4c67f21f94abe7557cc127d371791ecba5`通过GitHub HTTPS发布，local/upstream一致；实验继续，下一固定节点为80分钟。
- 2026-08-03（v5 80分钟节点）：14:05:44Z为51/256完成、24正确、47.058824%，0 invalid/23 truncated；相对70分钟新增14题、6题正确、7题新增截断。容器/tmux存活，15–16 running/0 waiting、68.7–76.8 token/s；14/14 validation、11/11 manifest通过，终局精度门禁仍未定。
- 2026-08-03（2.279实时记录与门禁）：修改前完整读取1,053,197-byte报告并确认SHA=`cbcb9f58...b8115`与HEAD一致；追加后报告17,074行/1,055,478 bytes、SHA=`25dc73ca...f7c`，2.1–2.279连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步发布实时节点，实验不中断。
- 2026-08-03（2.279发布）：80分钟节点已由主仓`edb5e97a6c2a8c1c34bfea9c0bc4e00066203c10`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为90分钟。
- 2026-08-03（v5 90分钟节点）：14:15:44Z为71/256完成、34正确、47.887324%，0 invalid/32 truncated；相对80分钟新增20题、10题正确、9题新增截断。容器/tmux存活，16 running/0 waiting、64.0–65.6 token/s；14/14 validation、11/11 manifest通过，终局精度门禁仍未定。
- 2026-08-03（2.280实时记录与门禁）：修改前完整读取1,055,478-byte报告并确认SHA=`25dc73ca...f7c`与HEAD一致；追加后报告17,110行/1,057,755 bytes、SHA=`79956965...aeee3`，2.1–2.280连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步发布实时节点，实验不中断。
- 2026-08-03（2.280发布）：90分钟节点已由主仓`bfc21da2e14a29d650a2316d2510452beb8ed22f`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为100分钟。
- 2026-08-03（v5 100分钟节点）：14:25:44Z仍为71/256完成、34正确、47.887324%，0 invalid/32 truncated；无新增但容器/tmux存活，16 running/0 waiting、59.2–60.8 token/s、KV 12.3%–12.4%。14/14 validation、11/11 manifest通过，不重启。
- 2026-08-03（2.281实时记录与门禁）：修改前完整读取1,057,755-byte报告并确认SHA=`79956965...aeee3`与HEAD一致；追加后报告17,144行/1,059,927 bytes、SHA=`7231a107...d7e0b`，2.1–2.281连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步发布实时节点，实验不中断。
- 2026-08-03（2.281发布）：100分钟节点已由主仓`c223f964cc2080322701c8a15bc132b5f01414d0`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为110分钟。
- 2026-08-03（v5 110分钟节点）：14:35:44Z仍为71/256完成、34正确、47.887324%，0 invalid/32 truncated；连续两个节点无新增，但容器/tmux存活，16 running/0 waiting、59.2–60.8 token/s、KV 18.3%–18.5%。14/14 validation、11/11 manifest通过，不重启。
- 2026-08-03（2.282实时记录与门禁）：修改前完整读取1,059,927-byte报告并确认SHA=`7231a107...d7e0b`与HEAD一致；追加后报告17,178行/1,062,141 bytes、SHA=`4e3e3427...d1c90`，2.1–2.282连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步发布实时节点，实验不中断。
- 2026-08-03（2.282发布）：110分钟节点已由主仓`8957feb5d4fd8ef3800abd24359798b7195fb3ea`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为120分钟。
- 2026-08-03（v5 120分钟节点）：14:45:44Z为101/256完成、45正确、44.554455%，0 invalid/47 truncated；相对110分钟新增30题、11题正确、15题新增截断。容器/tmux存活，15–16 running/0 waiting、71.9–76.8 token/s；14/14 validation、11/11 manifest通过，终局门禁仍未定。
- 2026-08-03（2.283首轮章节定位失败）：`apply_patch`的尾句与40分钟章节重复，导致2.283误插到2.275与2.276之间；结构校验精确报告首个错位为第276个标题。未发布；已删除错位块并用实际110分钟章节尾部上下文追加到文件末尾，不重复原定位方式。
- 2026-08-03（2.283实时记录与门禁）：修改前完整读取1,062,141-byte报告并确认SHA=`4e3e3427...d1c90`与HEAD一致；修正定位后报告17,214行/1,064,432 bytes、SHA=`01213f13...d6633`，2.1–2.283连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步发布实时节点，实验不中断。
- 2026-08-03（2.283发布）：120分钟节点已由主仓`dad7093e945771e0dbc43e456734f9bf6a6498c0`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为130分钟。
- 2026-08-03（v5 130分钟节点）：14:55:44Z为105/256完成、48正确、45.714286%，0 invalid/48 truncated；相对120分钟新增4题、3题正确、1题新增截断。容器/tmux存活，16 running/0 waiting、59.2–60.8 token/s；14/14 validation、11/11 manifest通过。已完成105题不等于正确105题门禁，当前只有48正确。
- 2026-08-03（2.284实时记录与门禁）：修改前完整读取1,064,432-byte报告并确认SHA=`01213f13...d6633`与HEAD一致；追加后报告17,251行/1,066,794 bytes、SHA=`b148d816...3ebef`，2.1–2.284连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步发布实时节点，实验不中断。
- 2026-08-03（2.284发布）：130分钟节点已由主仓`42f6aa7015b864d42a953e7cb587f0caee05f163`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为140分钟。
- 2026-08-03（v5 140分钟节点）：15:05:44Z仍为105/256完成、48正确、45.714286%，0 invalid/48 truncated；无新增但容器/tmux存活，16 running/0 waiting、59.2–60.8 token/s、KV 15.4%–15.5%。14/14 validation、11/11 manifest通过，不重启。
- 2026-08-03（2.285实时记录与门禁）：修改前完整读取1,066,794-byte报告并确认SHA=`b148d816...3ebef`与HEAD一致；追加后报告17,285行/1,068,981 bytes、SHA=`1c701540...116969`，2.1–2.285连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步发布实时节点，实验不中断。
- 2026-08-03（2.285发布）：140分钟节点已由主仓`fd07a36c6e7f7791c3290992c2603be4487d5580`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为150分钟。
- 2026-08-03（v5 150分钟节点）：15:15:45Z为113/256完成、49正确、43.362832%，0 invalid/56 truncated；相对140分钟新增8题、1题正确、8题新增截断。容器/tmux存活，15–16 running/0 waiting、65.5–76.8 token/s；14/14 validation、11/11 manifest通过。
- 2026-08-03（2.286实时记录与门禁）：修改前完整读取1,068,981-byte报告并确认SHA=`1c701540...116969`与HEAD一致；追加后报告17,321行/1,071,289 bytes、SHA=`a7eef955...65367`，2.1–2.286连续，术语、引用、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步发布实时节点，实验不中断。
- 2026-08-03（2.286发布）：150分钟节点已由主仓`f212998a2c34f8e03f9753d49e3a3e11d5e557f1`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为160分钟。
- 2026-08-03（v5 160分钟节点）：15:25:45Z为127/256完成、54正确、42.519685%，0 invalid/64 truncated；相对150分钟新增14题、5题正确、8题新增截断。容器/tmux存活，16 running/0 waiting、59.2–60.8 token/s；14/14 validation、11/11 manifest通过。
- 2026-08-03（2.287报告门禁）：修改前报告SHA=`a7eef955...65367`且与HEAD逐字节一致；追加后17,357行/1,073,592 bytes、SHA=`3e358e83...9dd5`，2.1–2.287连续，2.286引用、术语、核心hash及diff-check均通过。下一步精确提交报告与三份planning并HTTPS发布，实验不中断。
- 2026-08-03（2.287发布）：160分钟节点已由主仓`57526a36aa4575eb7b08a855079050720810e68b`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为170分钟。
- 2026-08-03（v5 170分钟节点）：15:35:45Z仍为127/256完成、54正确、42.519685%，0 invalid/64 truncated；容器/tmux存活，16 running/0 waiting、59.2–60.8 token/s、GPU 98%–100%，无新增不代表停止。14/14 validation、11/11 manifest通过。
- 2026-08-03（2.288报告门禁）：修改前报告SHA=`3e358e83...9dd5`且与HEAD逐字节一致；追加后17,391行/1,075,780 bytes、SHA=`577d9bc1...4610`，2.1–2.288连续，2.287引用、术语、核心hash及diff-check均通过。下一步精确提交报告与三份planning并HTTPS发布，实验不中断。
- 2026-08-03（2.288发布）：170分钟节点已由主仓`247b5848c78dd837b3c0f26143499cb8c04631e8`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为180分钟。
- 2026-08-03（v5 180分钟节点）：15:45:45Z为131/256完成、54正确、41.221374%，0 invalid/67 truncated；相对170分钟新增4题、0题正确、3题新增截断。15:46:18Z单点GPU利用率恰为0%，但显存各76,145 MiB且此前10–30秒服务持续16 running、55.9–65.6 token/s，不能据此判停；14/14 validation、11/11 manifest通过。
- 2026-08-03（2.289报告门禁）：修改前报告SHA=`577d9bc1...4610`且与HEAD逐字节一致；追加后17,428行/1,078,192 bytes、SHA=`229160d2...91de`，2.1–2.289连续，2.288引用、术语、核心hash及diff-check均通过。下一步精确提交报告与三份planning并HTTPS发布，实验不中断。
- 2026-08-03（2.289发布）：180分钟节点已由主仓`f4f1df41161af22de36a6a2152b047c77a1b5585`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为190分钟。
- 2026-08-03（v5 190分钟节点）：15:55:45Z为154/256完成、64正确、41.558442%，0 invalid/80 truncated；相对180分钟新增23题、10题正确、13题新增截断。8卡均98%，服务16 running、76.8–78.4 token/s；14/14 validation、11/11 manifest通过。
- 2026-08-03（2.290报告门禁）：修改前报告SHA=`229160d2...91de`且与HEAD逐字节一致；追加后17,464行/1,080,497 bytes、SHA=`2526e472...ea66`，2.1–2.290连续，2.289引用、术语、核心hash及diff-check均通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-03（2.290发布）：190分钟节点已由主仓`a5c92dc3a2235043812fb18eb2319f4e0c2ffc7f`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为200分钟。
- 2026-08-03（v5 200分钟节点）：16:05:45Z仍为154/256完成、64正确、41.558442%，0 invalid/80 truncated；8卡98%–100%，服务16 running、59.2–60.8 token/s，无新增不代表停止。14/14 validation、11/11 manifest通过。
- 2026-08-03（2.291报告门禁）：修改前报告SHA=`2526e472...ea66`且与HEAD逐字节一致；追加后17,498行/1,082,685 bytes、SHA=`c5017320...d880`，2.1–2.291连续，2.290引用、术语、核心hash及diff-check均通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-03（2.291发布）：200分钟节点已由主仓`7f40d7bb4533614c69df01dabad94be84df0b9f7`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为210分钟。
- 2026-08-03（v5 210分钟节点）：16:15:45Z仍为154/256完成、64正确、41.558442%，0 invalid/80 truncated；连续两个节点无新增，但8卡98%–100%，服务16 running、59.2–60.8 token/s。14/14 validation、11/11 manifest通过。
- 2026-08-03（2.292报告门禁）：修改前报告SHA=`c5017320...d880`且与HEAD逐字节一致；追加后17,532行/1,084,915 bytes、SHA=`69b6a4fd...2423`，2.1–2.292连续，2.291引用、术语、核心hash及diff-check均通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-04（2.292发布）：210分钟节点已由主仓`dd54e3a5184ecee38d31c0a9eeea358a4bed1949`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为220分钟。
- 2026-08-04（v5 220分钟节点）：16:25:45Z为171/256完成、69正确、40.350877%，0 invalid/90 truncated；相对210分钟新增17题、5题正确、10题新增截断。8卡84%–98%，服务16 running、60.6–76.7 token/s；14/14 validation、11/11 manifest通过。
- 2026-08-04（2.293报告门禁）：修改前报告SHA=`69b6a4fd...2423`且与HEAD逐字节一致；追加后17,568行/1,087,222 bytes、SHA=`d40e22bb...d9e1`，2.1–2.293连续，2.292引用、术语、核心hash及diff-check均通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-04（2.293发布）：220分钟节点已由主仓`7ae8b1a7ac61f74b68d7898a6553a1a311077184`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为230分钟。
- 2026-08-04（v5 230分钟节点）：16:35:45Z为185/256完成、76正确、41.081081%，0 invalid/96 truncated；相对220分钟新增14题、7题正确、6题新增截断。8卡97%–100%，服务16 running、62.4 token/s；14/14 validation、11/11 manifest通过。
- 2026-08-04（2.294报告门禁）：修改前报告SHA=`d40e22bb...d9e1`且与HEAD逐字节一致；追加后17,604行/1,089,528 bytes、SHA=`003f7564...7487`，2.1–2.294连续，2.293引用、术语、核心hash及diff-check均通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-04（2.294发布）：230分钟节点已由主仓`379a0b7af74e514d6a1fde59037bc78091f995ca`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为240分钟。
- 2026-08-04（v5 240分钟节点）：16:45:45Z仍为185/256完成、76正确、41.081081%，0 invalid/96 truncated；8卡98%–100%，服务16 running、59.2–60.8 token/s，无新增不代表停止。14/14 validation、11/11 manifest通过。
- 2026-08-04（2.295报告门禁）：修改前报告SHA=`003f7564...7487`且与HEAD逐字节一致；追加后17,638行/1,091,719 bytes、SHA=`a6ab2449...e17b`，2.1–2.295连续，2.294引用、术语、核心hash及diff-check均通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-04（2.295发布）：240分钟节点已由主仓`993eeb4aa5d685ea341c9b1408e16bd8024b56a9`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为250分钟。
- 2026-08-04（v5 250分钟节点）：16:55:45Z为189/256完成、78正确、41.269841%，0 invalid/99 truncated；相对240分钟新增4题、2题正确、3题新增截断。8卡98%–100%，服务16 running、64.0–65.6 token/s；14/14 validation、11/11 manifest通过。
- 2026-08-04（2.296报告门禁）：修改前报告SHA=`a6ab2449...e17b`且与HEAD逐字节一致；追加后17,674行/1,094,027 bytes、SHA=`16764089...93a9`，2.1–2.296连续，2.295引用、术语、核心hash及diff-check均通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-04（2.296发布）：250分钟节点已由主仓`fd0019a955c213e5c482294833504791ba932385`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为260分钟。
- 2026-08-04（v5 260分钟节点）：17:05:46Z为219/256完成、91正确、41.552511%，0 invalid/112 truncated；相对250分钟新增30题、13题正确、13题新增截断。8卡95%–98%，服务16 running、76.8–78.4 token/s；14/14 validation、11/11 manifest通过。
- 2026-08-04（2.297报告门禁）：修改前报告SHA=`16764089...93a9`且与HEAD逐字节一致；追加后17,710行/1,096,337 bytes、SHA=`be4fc288...188c`，2.1–2.297连续，2.296引用、术语、核心hash及diff-check均通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-04（2.297发布）：260分钟节点已由主仓`2c778f323db5c516539f2dcd9a28a2b98a3b0c1b`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为270分钟。当前剩余37题至少需14题正确才能达到105/256门禁。
- 2026-08-04（v5 270分钟节点）：17:15:46Z为221/256完成、93正确、42.081448%，0 invalid/112 truncated；相对260分钟新增2题且2题均正确，无新增截断。8卡97%–98%，服务16 running、59.2–60.8 token/s；14/14 validation、11/11 manifest通过。剩余35题至少需12题正确才能达到105/256门禁。
- 2026-08-04（2.298报告门禁）：修改前报告SHA=`be4fc288...188c`且与HEAD逐字节一致；追加后17,747行/1,098,692 bytes、SHA=`c57d263e...a651`，2.1–2.298连续，2.297引用、术语、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-04（2.298发布）：270分钟节点已由主仓`b269cb9d9a2ff1c9d49552179f514f5523a1f097`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为280分钟。当前剩余35题至少需12题正确才能达到105/256门禁。
- 2026-08-04（v5 280分钟节点）：17:25:46Z为222/256完成、93正确、41.891892%，0 invalid/113 truncated；相对270分钟新增1题、0题正确、1题新增截断。8卡97%–100%，服务16 running、60.8–62.4 token/s；14/14 validation、11/11 manifest通过。剩余34题至少需12题正确才能达到105/256门禁。
- 2026-08-04（2.299报告门禁）：修改前报告SHA=`c57d263e...a651`且与HEAD逐字节一致；追加后17,784行/1,101,047 bytes、SHA=`1b42a9b7...d023`，2.1–2.299连续，2.298引用、术语、7项核心hash、14/14 validation、11/11 manifest及diff-check通过。下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-04（2.299发布）：280分钟节点已由主仓`fb21c3a6f48d5dd89c7bc5e2c8cd3a339398bda6`通过GitHub HTTPS发布，local/upstream一致；实验容器与tmux继续运行，下一固定节点为290分钟。当前剩余34题至少需12题正确才能达到105/256门禁。
- 2026-08-04（v5 290分钟节点）：17:35:46Z为237/256完成、98正确、41.350211%，0 invalid/123 truncated；新增15题、5题正确、10题新增截断。8卡93%–98%，服务16 running、68.7–76.8 token/s；14/14 validation、11/11 manifest通过。剩余19题至少需7题正确。
- 2026-08-04（2.300报告门禁）：修改前报告SHA=`1b42a9b7...d023`且与HEAD一致；追加后17,821行/1,103,405 bytes、SHA=`2bcb0f4f...610a`，2.1–2.300连续，引用、术语、7项hash及diff-check通过。下一步发布，实验不中断。
- 2026-08-04（2.300发布）：290分钟节点已由主仓`07a65640923bbdba13c8528cd560fa7a9cd58ef7`通过GitHub HTTPS发布，local/upstream一致；实验继续，下一固定节点为300分钟。
- 2026-08-04（v5 300分钟/2.301门禁）：17:45:46Z为244/256、100正确、40.983607%，0 invalid/128 truncated；14/14 validation、11/11 manifest通过。报告17,857行/1,105,689 bytes、SHA=`e9e32932...ff33`，2.1–2.301连续；剩余12题需5题正确。
- 2026-08-04（2.301发布）：主仓`b9031994841d5ea8e0f81437f04d2e5e158bb078`已HTTPS发布且local/upstream一致；实验继续等待310分钟节点。
- 2026-08-04（v5 310分钟/2.302门禁）：17:55:46Z仍为244/256、100正确、40.983607%，0 invalid/128 truncated；单点GPU 0%但显存76,155 MiB且12 running、44.4–55.2 token/s，不能判停。14/14 validation、11/11 manifest通过；报告17,893行/1,107,958 bytes、SHA=`75946d47...930b`，2.1–2.302连续。
- 2026-08-04（2.302发布）：主仓`80ec976ae4cebe17775e4d242a8eeae0766a51bf`已HTTPS发布且local/upstream一致；实验继续等待320分钟节点。
- 2026-08-04（v5 320分钟/2.303门禁）：18:05:46Z为250/256、100正确、40.000000%，0 invalid/134 truncated；新增6题全错且全截断。6 running持续生成；14/14 validation、11/11 manifest通过。报告17,930行/1,110,267 bytes、SHA=`e36e5178...70ba`，2.1–2.303连续；剩余6题需至少5题正确。
- 2026-08-04（2.303发布）：主仓`1ce3442469488bdea9b667dcfe37af75deb473c5`已HTTPS发布且local/upstream一致；实验继续等待330分钟节点。
- 2026-08-04（v5终局/2.304门禁）：run于18:09:34Z自然结束，256/256、101正确、39.453125%、0 request failure、140 truncated，低于BF16 105/256门槛4题；30/30终局validation、16/16 manifest通过，8卡已释放。报告17,977行/1,113,453 bytes、SHA=`7e7f5e62...ed81`，2.1–2.304连续；禁止性能复测，转CPU-only精度归因。
- 2026-08-04（2.304发布）：主仓`93d26fa393014162a39953ab958f4ef29dc62b30`已HTTPS发布且local/upstream一致；下一阶段CPU-only枚举同协议历史predictions并分层归因。
- 2026-08-04（CPU-only同协议配对初析）：当前101/256、140截断、completion中位7974；K1024为108/256、122截断、中位912.5。逐题65均对、36当前独对、43 K1024独对，净-7；43个损失中36个是当前截断而K1024未截断。当前相对legacy K768净+4但多19截断，故问题与超长/截断强相关，且不能只归因于top-k大小。
- 2026-08-04（CPU-only归因闭合/2.305）：四轮同协议fixed256的ID/prompt/gold均256/256匹配；当前相对K1024/K1536/K768逐题净差分别-7/-5/+4。当前正式v5实际为persistent、无prefill sort、空hf_overrides，偏离ea8 canonical要求的K1024+legacy+prefill K768+sort=1。48/48 validation和5/5 manifest通过；只能认定截断强关联与身份偏差，不能单因归责。报告2.305门禁通过，待发布。
- 2026-08-04（canonical preflight失败/2.306）：2.305由`56d4732`发布；固定GPU0–7双空闲间隔82秒且两轮全idle，静态身份8/8通过。干净clone缺少被Git忽略的Phase6 overlay mount point，preflight在`mount --bind`处exit32；模型/accuracy未启动，退出后全idle。21/21 validation、18/18 manifest通过；报告2.306待发布，禁止直接重跑。
- 2026-08-04（canonical preflight v2/2.307）：2.306由`2a848e1`发布；新双空闲间隔74秒。宿主`mkdir -p`无效的精确原因是clone内Phase6 run-ID为指向原项目绝对路径的ignored symlink，容器仅bind clone后链接悬空，故同处exit32。模型/accuracy未启动；21/21 validation、20/20 manifest通过，报告2.307待发布。
- 2026-08-04（canonical preflight v3/2.308）：2.307由`4f836f7`发布；替换临时clone symlink后新双空闲76秒通过。v3越过mount但candidate wrapper Git mode=100644、容器mode664，canonical入口直接执行时报Permission denied/exit126。模型/accuracy未启动；23/23 validation、23/23 manifest通过，报告2.308待发布。下一步先CPU-only executable合同红灯，再只改mode。
- 2026-08-04（candidate mode TDD red/2.309）：2.308由`eb9f18e`发布；新增Git index mode=100755合同。宿主3.8与冻结3.12分别因`datetime.UTC`/缺`requests`无效；固定control image、network none、无GPU的有效轮次运行1项仅失败1项，精确命中100644!=100755。10/10 validation、17/17 manifest通过，production/mode未改，报告2.309待发布。
- 2026-08-04（candidate mode green/2.310）：2.309由`e35206b`发布；只改candidate wrapper Git mode 100644→100755，blob/text SHA不变、numstat 0/0。固定control image、network none、无GPU下目标1/1、完整Phase9 25/25通过；11/11 validation、15/15 manifest通过。报告2.310待发布。
- 2026-08-04（canonical preflight v4/2.311）：2.310由`4baf2cf`发布；无GPU探针确认wrapper executable与Phase6真实目录。新双空闲78秒通过；v4进一步到冻结Python检查，但clean clone的Phase0绝对symlink在容器bind中悬空，模型前exit1。24/24 validation、23/23 manifest通过，报告2.311待发布。下一步从原项目运行，仅本地exclude Shawn未跟踪报告。
- 2026-08-04（canonical preflight v5通过/2.312）：2.311由`3e3833e`发布；只在本地Git exclude精确屏蔽Shawn未跟踪报告，未触碰文件。原项目无GPU禁网探针确认wrapper、冻结Python、Phase6 overlay及K1024+legacy+prefillK768+sort1；19:07:14Z/19:08:19Z双空闲间隔65秒。正式preflight 19:08:45Z–19:10:03Z自然exit0，静态70/70，固定环境与server args身份通过，CUDA/模型/accuracy未启动，退出后全idle；独立42/42与manifest 21/21通过。报告2.312待门禁发布；发布后必须新双空闲再启动fixed256。
- 2026-08-04（canonical fixed256正式门禁/2.313）：2.312由`fa84be1`发布；19:16:18Z/19:17:24Z新双空闲间隔66秒，两轮GPU0–7均0 MiB/0%且无compute。主仓/source已发布且clean、image与四项canonical身份不变、preflight 42/42、wrapper mode100755、唯一输出根与容器均不存在；20/20 validation、13/13 manifest通过。报告2.313待发布；发布后即时快照仍idle才允许启动`ea8-canonical-fast256-v1`。
- 2026-08-04（fixed256 v1权限失败/2.314）：2.313由`8d67dc6`发布；19:22:02Z即时GPU门禁通过后启动，19:22:21Z在static/namespace preflight后创建`phase7`时报Permission denied并exit1，模型/accuracy未启动且全程8卡idle。根因是宿主输出根owner22633/mode775，而隔离`unshare -Urn --map-root-user`内root不能用初始namespace DAC override且other无写位；同权限无GPU精确复现exit1，mode777对照exit0。37/37 validation、28/28 manifest通过；报告2.314待发布，下一步新身份+mode0777+无GPU探针+新双空闲。
- 2026-08-04（fixed256 v2恢复门禁/2.315）：2.314由`1bb6932`发布；新run/output身份专用根owner22633/mode777，同正式条件无GPU user namespace探针exit0并创建owner0/mode755子目录，正式run目录仍不存在。19:28:41Z/19:29:46Z新双空闲间隔65秒，两轮8卡全idle；发布身份、image、wrapper、canonical配置和v1边界复核通过，25/25 validation、17/17 manifest通过。报告2.315待发布，发布后即时快照仍idle才启动v2。
- 2026-08-04（fixed256 v2有效启动/2.316）：2.315由`e5962f5`发布；v2于19:33:22Z即时GPU全idle后启动，成功创建隔离run目录并越过v1权限边界。network namespace仅loopback、外部IPv4 exit7，Stage7 static 44/44；实际wrapper环境为固定256/c16及K1024+legacy+prefillK768+sort1。19:34:38Z容器/tmux存活且仍在模型前导入，35/35 validation、20/20 manifest通过。报告2.316待发布，实验继续等待10分钟精度节点。
- 2026-08-04（v2 10分钟监控修正/2.317）：2.316由`992e9cd`发布；宿主sidecar因root:root/mode600 checkpoint读取OSError而误报0/256。按19:43:25Z相同截止mtime从container root重算为9/256、6正确、66.666667%当前精度、2.343750%全量精度、0 invalid/0 truncated；原错误行与failed validator保留。旧sidecar已停，主实验/容器/tmux未停，新root-view sidecar从20分钟继续。33/33 validation、13/13 manifest通过，报告2.317待发布。
- 2026-08-04（v2 20分钟精度/2.318）：2.317由`0950f5b`发布；corrected root-view sidecar在19:53:22Z固定截止得到12/256、8正确、66.666667%当前精度、3.125000%全量精度、0 invalid/0 truncated，相对10分钟新增3题/2正确。主容器、双tmux、TP0–7与c16 runner存活，8卡76,071 MiB且74%–99%；25/25 validation、8/8 manifest通过。报告2.318待发布，实验继续到30分钟。
- 2026-08-04（自动恢复现场）：主仓HEAD/upstream=`beb4a6e73d769b191601b0774912388412a87e3e`且clean；fixed256 v2容器、主tmux、corrected monitor均存活。19:57:35Z GPU0–7各约76,071 MiB、75%–98%，compute仅TP0–7；20分钟结果未变。性能目标保持完整：精度终局达到105/256后才执行同源码同负载32K/batch1 BF16/OSCAR TTFT/TPOT对比，若OSCAR明显慢则继续归因和优化。
- 2026-08-04（v2 30分钟精度/2.319）：20:03:22Z仍为12/256、8正确、66.666667%当前精度、3.125000%全量精度、0 invalid/0 truncated。20:04:32Z单点GPU利用率0%，但显存76,073 MiB、容器/双tmux/TP0–7/c16 runner均存活，20:00:14Z最近快照为63%–98%且server_completed=12，不能判停。26/26 validation、8/8 manifest通过；报告2.319待发布。
- 2026-08-04（v2 40分钟精度/2.320）：2.319已由`b2cae8e6c36863e8798c51d59c0cfba5a9df03fd`发布。20:13:22Z为20/256、8正确、40.000000%当前精度、3.125000%全量精度、0 invalid/8 truncated；相对30分钟新增8题均得分0且截断。20:15:01Z容器/双tmux/TP0–7/c16 runner存活，8卡76,079 MiB、67%–96%；26/26 validation、8/8 manifest通过。首次截止行采集漏`docker exec -i`及随后展示命令孤立引号均已纠正且未影响实验；报告2.320待门禁发布。
- 2026-08-04（2.320报告门禁）：修改前报告SHA=`226bcf4d...84596`且与HEAD一致；追加后18,694行/1,161,887 bytes、SHA=`a51cda55...d3500`，2.1–2.320连续且无重复。2.319发布引用、术语、6项核心证据大小/hash、26/26 validation、8/8 manifest及diff-check全部通过；下一步精确提交并HTTPS发布，实验不中断。
- 2026-08-04（2.320发布）：主仓`04011fd8e4f6b3bdd498b8727a1cd4a825db89e2`已通过GitHub HTTPS发布且local/upstream一致；20:18:37Z容器与双tmux仍存活，继续等待50分钟固定节点。
- 2026-08-04（v2 50分钟精度/2.321）：20:23:22Z为34/256、12正确、35.294118%当前精度、4.687500%全量精度、0 invalid/16 truncated；相对40分钟新增14题、4题正确、8题新增截断，16个累计截断样本均得分0。20:24:31Z容器/双tmux/TP0–7/c16 runner存活，8卡76,085 MiB、74%–98%；26/26 validation、8/8 manifest通过，报告2.321待门禁发布。
- 2026-08-04（2.321报告门禁）：修改前报告SHA=`a51cda55...d3500`且与HEAD一致；追加后18,730行/1,164,244 bytes、SHA=`218d53c9...33594`，2.1–2.321连续且无重复。2.320发布引用、术语、6项核心证据大小/hash、26/26 validation、8/8 manifest及diff-check全部通过；下一步精确提交并HTTPS发布，实验不中断。
