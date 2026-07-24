# 任务计划：OSCAR × GLM‑5.2 × A800 分阶段实验

## 目标

严格按照 `docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md` 完成阶段 0–9 的实现、实验、验收与中文报告，最终满足第 17 节的 32K 首版本完成定义，并给出 128K 扩展验证或容量阻塞证据。

## 下一步

确认并创建 `glm52_oscar_vllm` 远端，将源码仓库与主仓库当前功能分支提交并推送；
随后执行已固化的阶段 1 TP=8 原生服务、四类 smoke、official_v4 和 WikiText‑2。

## 当前阶段

阶段 1：本地运行入口已验证，等待远端发布后执行正式 GPU baseline

## 阶段

### 阶段 0：恢复并冻结源码

- [x] 找到并核验完整 GLM v0.19.0 源码
- [x] 核对 commit、镜像 ID、关键文件、原生扩展与运行时 patch
- [x] 建立本地独立 `glm52_oscar_vllm` Git 仓库
- [x] 创建可复现 Dockerfile/OCI 构建脚本并重建 baseline 镜像
- [x] 更新中文阶段报告
- [ ] 确认 GitHub 远端并推送，之后再接入主仓库 submodule
- **状态：** 本地技术出口通过；协作发布待完成

### 阶段 1：GLM‑5.2/A800 原生 baseline

- [x] 已取得 Shawn 授权：当前机器所有可见 GPU 均可使用
- [x] 连续两次核验授权范围内 8 张 A800 空闲
- [x] 复核模型与 checkpoint 快速指纹
- [x] 固化 TP=8/eager/32K/native sparse MLA 启动与 fail-closed 门禁
- [x] 固化短请求、>320 tokens、连续 decode、32K smoke 和评测入口
- [ ] 在固定容器中完成 TP=8 短请求、>320 tokens、连续 decode 和 32K 验证
- [ ] 冻结 official_v4 与 WikiText‑2 baseline
- [ ] 更新中文阶段报告
- **状态：** 前置检查进行中

### 阶段 2：Calibration 与 PyTorch reference

- [ ] 构建与 official_v4 独立的 calibration manifest
- [ ] 实现 capture、共享 covariance、rotation/clip 搜索和 artifact 合约
- [ ] 完成 reference、正交性、未量化等价与 INT2 数值验证
- [ ] 更新中文阶段报告
- **状态：** 待开始

### 阶段 3：三池 CacheSpec 与 CPU allocator

- [ ] 实现 capacity planner、prefix/recent/history allocator 与生命周期
- [ ] 接入 v0.19 scheduler/worker
- [ ] 完成容量守恒、回滚和边界测试
- [ ] 更新中文阶段报告
- **状态：** 待开始

### 阶段 4：A800/SM80 Triton kernels

- [ ] 按设计顺序实现 store、demotion、mixed sparse MLA 和 inverse rotation
- [ ] 完成 SM80 cold compile、A800 launch、oracle 与边界测试
- [ ] 更新中文阶段报告
- **状态：** 待开始

### 阶段 5：vLLM 接入与 32K 端到端

- [ ] 注册并接入 `oscar_mla_int2`
- [ ] 完成单/多请求、demotion、DSA mixed read 和接近 32K 验证
- [ ] 证明无 fallback、无完整 BF16 history，并满足压缩率阈值
- [ ] 更新中文阶段报告
- **状态：** 待开始

### 阶段 6：冻结候选镜像

- [ ] 固定源码、Dockerfile、依赖、原生扩展与 rotation artifact
- [ ] 构建并记录不可变候选镜像 tag、ID 和 digest
- [ ] 更新中文阶段报告
- **状态：** 待开始

### 阶段 7：完整精度与 PPL

- [ ] 完成 official_v4 2360 个样本及 WikiText‑2
- [ ] 生成完整差异、失败分类、预测 SHA256 和硬阈值判定
- [ ] 更新中文阶段报告
- **状态：** 待开始

### 阶段 8：精度优化（仅阶段 7 未通过时）

- [ ] 严格按设计文档第 11 节顺序优化
- [ ] 每轮保存独立 manifest、artifact/镜像和中文记录
- **状态：** 待开始

### 阶段 9：性能和 128K 扩展

- [ ] warm-up 后完成固定性能矩阵
- [ ] 对超过 20% 回退完成 profiling 和归因
- [ ] 验证 128K 或形成容量阻塞证据
- [ ] 更新中文阶段报告
- **状态：** 待开始

## 关键问题

1. `glm52_oscar_vllm` 的 GitHub 仓库名称、可见性与稳定分支应如何配置？
2. calibration 独立数据源是否已有可复用资产，还是需要新建固定混合集？

## 已做决策

| 决策 | 理由 |
| --- | --- |
| 使用主仓库根目录的独立 planning-with-files 文件 | `oscar_vllm/` 下现有计划属于另一项工作，避免覆盖和串扰 |
| 首先只推进阶段 0，不使用 GPU | 设计要求分阶段闸门；GPU 授权是阶段 1 前置条件 |
| `oscar_vllm` 与 `glm52_speed_up_v2_stable_8th` 保持只读参考 | 设计文档明确要求建立新的 `glm52_oscar_vllm` 代码线 |
| GPU 授权范围为当前机器所有可见 GPU | Shawn 于 2026-07-24 明确授权；实际分配仍须连续两次检查空闲 |
| 项目外路径严格只读 | Shawn 明确要求不得修改 `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1` 等外部文件；所有构建产物仅写入本项目或临时目录 |
| 阶段 1 直接使用项目内候选 rootfs 的固定 venv/source | 当前环境本身是 Kubernetes 容器，符合 `AGENTS_misc.md` 的直接配置规则；不写当前容器 `/opt` |
| 首阶段显式禁用 speculative、prefix cache 与 CUDA graph | 设计第 2.2 节明确列为非目标；启动参数已解析验证为 eager、无 speculative、无 prefix cache |
| 原生 attention 显式固定 `TRITON_MLA_SPARSE` | A800 为 SM80，目标模型要求 sparse MLA；显式配置便于日志和结果审计 |

## 遇到的错误

| 错误 | 尝试 | 处理 |
| --- | ---: | --- |
| 当前 shell 中 `docker: command not found`，无法直接查询 daemon/镜像 | 1 | 转为核验本地 Docker image tar，并检查容器环境与可用的无 daemon 解包工具；不重复调用缺失的 Docker CLI |
| 外部源码仓库触发 Git `dubious ownership`，且在外部 workdir 用相对路径读取根计划失败 | 1 | 不修改全局 Git 配置；后续用 `git -c safe.directory=<精确路径>` 只读核验，并用绝对路径读取任务计划 |
| 镜像归档分析脚本调用 `jq`，当前容器未安装 | 1 | 改用 POSIX 文本工具解析单行 Docker archive manifest；不为一次性 JSON 读取引入依赖 |
| layer 扫描命令因包含 `rm -f` 清理临时清单而被安全策略拒绝 | 1 | 改为单次流式 `tar -tf | awk`，不创建或删除临时文件 |
| `git archive` 使用手工输入的长 SHA 时报告 `not a tree object`，导致临时 clean tree 为空 | 1 | 使用仓库解析出的 `HEAD^{tree}`/`HEAD`，先以 `git cat-file` 验证对象再归档；不使用空目录比较结果 |
| provenance 全盘 `rg` 未有效排除历史日志，输出过大且持续扫描 | 1 | 已主动中止，改为读取精确 manifest 和已知源码路径，不再遍历 cache/logs |
| `git clone --no-local` 因源仓库 `.git` 所有权触发 `dubious ownership` | 2 | 命令级 `safe.directory` 未传递给 clone 的 upload-pack 子进程；改为先用已验证可读的源仓库生成只读 Git bundle，再从 bundle clone |
| pre-commit 首次初始化 ruff hook 在 GitHub `index-pack` 阶段超过 3 分钟无进展 | 1 | 已中止；保留已安装 hook，改为按 `.pre-commit-config.yaml` 固定版本从清华 PyPI 安装 ruff 并直接运行 |
| rootless overlay/fuse-overlayfs 三种启动方式均因 mount permission/propagation 失败 | 3 | 停止 overlay 路径；使用已验证可运行的 `vfs + chroot`，graphroot 放到任务专用 NFS 目录并监控空间 |
| buildah vfs 从 docker archive 导入失败：普通 pull 需 remount `/`；buildah unshare 又缺 subuid 且 `memfd_create` 失败 | 2 | 停止 buildah import，改用不依赖 mount 的 daemonless builder（优先 Kaniko）或 OCI 组合方案 |
| `umoci insert` 从 NFS 和本地 `/tmp` payload 生成的层都截断最后 393 字节 | 2 | SHA/gzip 完整但 tar 逻辑不完整；停止使用 `umoci insert`，改为标准 GNU tar + OCI descriptor 组装 |
| 二次层重建首次使用了错误的临时归档根路径 | 1 | 在生成摘要前中止；按 `opt/vllm_glm52_v1` 精确路径重跑并得到与首次构建完全相同的 digest/diff ID/大小 |
| `umoci unpack` 在 NFS xattr/元数据阶段运行约 72 分钟仍未退出 | 1 | 终止额外强校验并如实记为未完成；已展开 rootfs 的 4/4 source 与 7/7 native SHA256 全部通过 |

## 约束提醒

- 所有报告必须使用中文，且数据只能来自实际落地结果。
- 每个阶段实验结束后先更新报告，再进入下一阶段。
- 修改已有报告前必须重新读取，并检查章节序号与交叉引用。
- 长实验每 10 分钟记录一次进度。
- GPU 实验必须在固定 Docker 容器内执行，且分配前连续两次确认授权 GPU 空闲。
- 正式实验前必须提交并推送源码与配置；不得让主仓库 submodule 指向本地-only commit。
- `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1` 及其他项目外目录仅可只读检查；禁止修改、暂存、提交、清理或重置。
