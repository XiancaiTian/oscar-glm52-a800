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
