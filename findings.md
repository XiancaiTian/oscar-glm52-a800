# 发现与决策

## 需求

- 按 `docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md` 依次推进阶段 0–9。
- 首版本目标是单机 8×A800、TP=8、原生 DSA/sparse MLA、32K、真实 `oscar_mla_int2` 三池路径、压缩率与精度/PPL 达标。
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
- 当前模型固定为 `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-staticgate-e154-H001-nfs`。
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
- 阶段 1 第一次 GPU 检查：8 张可见卡均为 NVIDIA A800-SXM4-80GB，显存使用 0MiB、利用率 0%，无 compute process。
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
- 18:00:15Z 至 22:10:20Z 的 10–260 分钟节点均已写入 `progress_10min.log`；8 张 A800 显存占用约为 79,901–79,937MiB，运行进程持续存活。
- runner 标准输出存在缓冲，进度文件在前三个节点只能记录 `completed unknown/2360`；60/70/80/90/100/110/120/130/140/150/160/170/180/190/200/210/220/230/240/250/260 分钟节点分别为 40/40/60/60/80/80/80/100/100/100/120/120/120/140/140/160/160/180/200/220/240，不把服务请求数直接当作最终评分数。
- 18:11:57Z 服务日志自全量启动后已有 17 个 POST HTTP 200，未发现 ERROR、Traceback 或 timeout；该计数只作为请求完成下界，最终以 `predictions.jsonl` 和 summary 为准。
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
- 正式源码 worktree 仍停留在 `53d8be94f6038e10ab0c344f706c5ffe66a555b8`，工作区干净；阶段 2 的准备提交没有改变正在运行的阶段 1 submodule 或 rootfs。
- OpenWebMath 固定为官方 revision `fde8ef8de2300f5e778f56261843dab89f230815`。由于当前环境访问 Hugging Face Xet CDN 时 TLS 失败，curl、`hf_hub_download` 和 wget 三条直接下载路径均停止；随后从官方 datasets-server 获取 0–299 行，冻结为项目内 300 行、2,800,069 字节、SHA256 `39d245cad8279eb6301f0a311471d311fb78557c9df5441d8ce5860422938b80` 的只读输入。
- LongBench 固定为官方 revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，采用 `gov_report_e`、`multi_news_e`、`qasper_e`、`lcc_e`、`repobench-p_e` 五个只读文件；各文件实际 SHA256 已写入 calibration 配置。
- tokenizer 固定为模型目录 `tokenizer.json`，SHA256 `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`。
- 开发版 manifest 实际为 20 行、250,907 字节、50,000 tokens，SHA256 `cde883393c6d0d6f4224f57db321b7ac2f12888f90d7db9f30deee543b1c25da`；summary SHA256 为 `e13329218df30f1a763597e5ec78ffd9f0c32aeb362c3f19d2d816570717e6ea`。
- 正式版 manifest 实际为 292 行、4,570,560 字节、1,000,000 tokens，SHA256 `3a183cba6f013424c5e422da5eb9ab8a9d4d975ddd0dbabd1b930b1d02e876b5`；summary SHA256 为 `f0e323f46b8e8b201651d6f258885d6908a417d150d08c1ad561c8bd06a691bf`。
- 开发版与正式版均独立构建两次且产物 SHA256 一致。正式版重新审计得到 292 个唯一 entry、235 个唯一源样本、0 个 split collision、0 个重复文本 hash、0 个 official_v4 完整 prompt hash 交集、0 个文本 hash/token 计数错误；各 split/category token 数逐项等于冻结配额。
- 初始 fit 配置固定 900,000 train tokens、100,000 holdout tokens、TP=8、78 层、latent rank 512、group size 128、prefix/recent 64/256、holdout reservoir 4,096 行和 DSA 512 行；专家映射 hash 基于 checkpoint index 中排序去重后的 `experts.<integer>` 清单，实际为 `89430944bd88801fc3deed040dd0d54e1db229f5a5cdd66c9682ef862313c983`。
- phase-2 launcher 已实际验证 6 个 vLLM 原生 `.so` 的项目内只读 symlink 与候选 rootfs 内容一致，第 7 个 sparse MLA `.so` 按 rootfs 绝对路径核验，7/7 SHA256 通过；候选 Python 从主仓库 source 路径导入 `vllm` 和 `vllm._C`，`torch.cuda.is_initialized()` 为 false。
- 正式 submodule 尚未从阶段 1 的 `53d8be94f...` 切到 calibration commit；这是避免污染运行中原生 baseline 的有意闸门，不把尚未执行的 TP=8 capture 记为已通过。
- 隔离 worktree 的 pre-commit 首次初始化再次停在 GitHub hook `index-pack`；中止后只损坏了 linked worktree index，正式源码 worktree、HEAD 和 Git 对象库均正常。已先记录 5 个新增文件 SHA256，再用 `git read-tree HEAD` 重建该 worktree index并按哈希重新暂存，最终提交前的手工门禁全部通过。

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
- 独立分支 commit `e75a40a294bd3127667f34ebffce8119a8ac0f3a` 已推送；116 项定向 pytest 全部通过，强制离线 scheduler 回归另有 68 项通过、28 项仅缺少 LLaVA 仓库配置，ruff/format/compile/diff 均通过。该分支没有接入正式 submodule，正式集成仍受阶段 2 出口约束。

## 阶段 4 隔离准备

- shared-latent rotation 使用 Triton FP32 输入与 IEEE FP32 accumulator；rotation 输出只与当前写入/查询 batch 成比例，不创建与完整历史长度成比例的 BF16 临时 cache。
- INT2 store 按 128 维分组执行 percentile clipping，4 个 2-bit index 以 low-bit-first 顺序打包到 1 byte；每个 512 维 token 存 128-byte data、4 个 FP32 scale 和 4 个 FP32 zero。
- BF16 store 根据 token logical position 与 final sequence length 只写最终 prefix/recent 分区；recent 物理地址为 `(position - prefix_tokens) % recent_tokens`。历史 token 不写 BF16 pool。
- recent demotion 以 scheduler 给出的 logical position、稳定 `hp_row`、history page/offset 为输入，执行 gather → rotation → clipping/quantize/pack；history dequant 只用于 oracle 与 kernel 单测。
- mixed sparse decode 对 DSA-selected token IDs 直接分类到 prefix/recent/history；每个 split 保存原 latent BF16 accumulator、旋转 latent history accumulator 与 LSE，跨 split 统一归一化后才对 history 乘 `R^T`，避免在不同 basis 中提前相加。
- mixed sparse prefill 复用同一三池 kernel，通过显式 query-to-request 映射共享请求 cache metadata，并以 query logical position 屏蔽未来 token；query 级中间 accumulator/LSE 不共享。
- CPU Triton interpreter 首轮捕获两个真实问题：split merge 的标量 mask 类型不兼容，以及 BF16 `tl.dot` rotation 产生无效大值；分别改为分离的 `tl.where` 和 FP32 IEEE rotation 路径。
- 修复后，实际 512 latent rank、4 groups、两 splits 的 store/demotion/dequant 与 mixed decode interpreter smoke 通过，输出和 LSE 均 finite，对 PyTorch oracle 最大绝对误差为 `2.384185791015625e-07`，且子进程未创建 CUDA context。
- 同一 CPU interpreter 实际执行 query positions 2/4 的 causal multi-token prefill，输出与 LSE 均 finite，对 PyTorch oracle 最大绝对误差为 `2.980232238769531e-07`。
- 分支 `feat/glm52-oscar-kernels` 的五个 WIP/测试 commit 为 `9861f2398...`、`5d220497a...`、`18c83e4b9...`、`b722b7975...` 和 `8ac7b9d97...`；61 项测试通过，22 项 CUDA 测试因 baseline 占满 GPU 被显式门禁跳过，静态检查通过。
- Stage 5 接线审查发现上述 mixed kernel 尚未把 64 维原精度 RoPE 分量加入 attention logits；若继续接线会得到可执行但数学上不完整的结果。kernel 工作树已加入 query RoPE、RoPE cache 和标准 block table 地址映射，并把默认 scale 从 `1/sqrt(512)` 修正为 `1/sqrt(512+64)`。
- 带非零 RoPE 数据的 CPU Triton interpreter 已实际执行 512+64 维 decode 与 causal prefill；两者相对扩展 PyTorch oracle 的最大绝对误差均为 `2.384185791015625e-07`。完整无 CUDA套件为 61 passed、22 skipped，commit `8ac7b9d97...` 已推送；A800 CUDA 仍未验证。
- 当前结论仍只是代码候选和 CPU interpreter 里程碑，不是 Stage 4 验收。尚无 SM80 cold compile 或 A800 actual launch；interpreter 的 finite/数值误差不能冒充 A800 实测。

## 阶段 5 隔离准备

- `oscar_mla_int2` 已作为显式 cache dtype 接入配置层，generic torch dtype 仅以 uint8 作为 heterogeneous layout marker；真实物理布局仍由三池 spec/views 定义。
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
- Stage 5 已完成代码级 artifact/metadata/write/read 接线，但这些新增路径仍未在 A800 上 launch，也未跑服务；不能宣称 `oscar_mla_int2` 已运行。

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
- 服务于 `2026-07-25T17:14:53Z` 通过健康检查，launcher 记录启动总耗时 2,762 秒；8 个 rank 均确认 `TRITON_MLA_SPARSE`，A800 sparse indexer 使用 Triton fallback，FP8 linear/MoE 使用 Marlin。
- 四项原生 smoke 全部 HTTP 200：短请求 21+64 tokens、16.06 秒；>320 输入 506+18 tokens、3.56 秒；连续 decode 26+384 tokens、59.90 秒；近 32K 请求 31,996+64 tokens、24.28 秒。
- official_v4 首轮于 17:18:08Z 以并发 8 启动。前 8 条均为 LiveCodeBench v6，`max_tokens=4096`；在实测约 6.4 tokens/s 的单序列速度下，完整输出理论需要约 640 秒，超过 suite 的 600 秒 code timeout。
- 首轮 10 分钟时 KV usage 从 20.8% 同步降至 2.4% 并装入下一批 8 个 prompt，但服务只记录 2 个 HTTP 200，证明其余 6 个客户端请求已超时。该轮不可能满足设计要求的 request failure=0，已停止且不生成伪造的最终 predictions/accuracy。
- official_v4 manifest、样本、decoding 和 runner SHA 保持不变；评测入口改为每个 attempt 保存独立 runtime suite，只将 code timeout 从源配置的 600 秒固定提高到 900 秒，并同时记录源/运行时 config SHA256。该调整只扩大执行预算，不改变模型输入、输出上限或评分。
- 900 秒 runtime timeout 探针实际运行前 8 条 LiveCodeBench v6，耗时 638.53 秒；结果为 8/8 请求成功、8/8 `scored`、request failure=0，证明新的请求预算覆盖了首批 4,096-token 长生成。探针 accuracy 为 0.0，表示 8 条均未通过代码评分，不是请求或 evaluator 失败。
- 探针 runtime eval config SHA256 为 `8e0beeb16a6c66182551ae75fa8e4694e4e8cefda205d8c3c778c909c24a22c5`，predictions SHA256 为 `05390ba2d2964bd5d35fe505ed6550c51e10ee0f1fd0174023862125159296e2`，summary SHA256 为 `3a78a575a64dab0809535896be52e1dc63cdd39e073fa738290fc04c3b9fc2e7`。
- 正式 preflight 运行目录为 `artifacts/phase1/20260725T155457Z_native_tp8`；记录主仓库 `8f7be26a...`、源码仓库 `53d8be94f...` 和 runtime `fd3e0b377...`。
- 2026-07-25T15:55:27Z 和 15:56:31Z 两次 GPU 检查均为 8/8 A800、0MiB、0% 利用率、无 compute process；Driver 575.51.03、CUDA 12.9。

## 资源

- 设计文档：`docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md`
- 项目规范：`AGENTS.md`、`AGENTS_misc.md`
- GLM 部署资产：`glm52_speed_up_v2_stable_8th/`
- OSCAR-vLLM 参考：`oscar_vllm/`
