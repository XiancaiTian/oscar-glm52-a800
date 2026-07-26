# 任务计划：OSCAR × GLM‑5.2 × A800 分阶段实验

## 目标

严格按照 `docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md` 完成阶段 0–9 的实现、实验、验收与中文报告，最终满足第 17 节的 32K 首版本完成定义，并给出 128K 扩展验证或容量阻塞证据。

## 下一步

900 秒 runtime timeout 的前 8 条 LiveCodeBench 探针已通过；official_v4 全量 2,360 样本已于 2026-07-25T17:50:15Z 启动，继续每 10 分钟记录进度，完成后运行 WikiText‑2 PPL。

## 当前阶段

阶段 1：原生服务、smoke 与 official_v4 timeout 探针通过，运行全量精度基线

## 阶段

### 阶段 0：恢复并冻结源码

- [x] 找到并核验完整 GLM v0.19.0 源码
- [x] 核对 commit、镜像 ID、关键文件、原生扩展与运行时 patch
- [x] 建立本地独立 `glm52_oscar_vllm` Git 仓库
- [x] 创建可复现 Dockerfile/OCI 构建脚本并重建 baseline 镜像
- [x] 更新中文阶段报告
- [x] 确认 GitHub 远端并推送，之后再接入主仓库 submodule
- **状态：** 完成

### 阶段 1：GLM‑5.2/A800 原生 baseline

- [x] 已取得 Shawn 授权：当前机器所有可见 GPU 均可使用
- [x] 连续两次核验授权范围内 8 张 A800 空闲
- [x] 复核模型与 checkpoint 快速指纹
- [x] 固化 TP=8/eager/32K/native sparse MLA 启动与 fail-closed 门禁
- [x] 固化短请求、>320 tokens、连续 decode、32K smoke 和评测入口
- [x] 在固定容器中完成 TP=8 短请求、>320 tokens、连续 decode 和 32K 验证
- [ ] 冻结 official_v4 与 WikiText‑2 baseline
- [ ] 更新中文阶段报告
- **状态：** official_v4 8 样本探针通过；全量 2,360 样本运行中，已完成 10–550 分钟进度记录，2026-07-26T03:00:25Z runner 报告 900/2360，服务仍持续完成请求

### 阶段 2：Calibration 与 PyTorch reference

- [x] 构建与 official_v4 独立的 calibration manifest
- [x] 实现环境显式启用、逐层限额、TP 分片的只读 activation/DSA capture
- [x] 完成 rotation artifact 写入、哈希、完整性与 runtime 身份 fail-closed 合约
- [x] 完成共享 covariance 合并和 rotation/clip 搜索
- [x] 完成共享潜空间 PyTorch reference、covariance 基础、正交性、未量化等价与 INT2 数值验证
- [ ] 更新中文阶段报告
- **状态：** 正式 manifest 已冻结；prompt/capture/TP merge/fit/artifact 正式入口已完成，35 项测试通过，源码推送至 `da4e2756a`；GPU calibration 待阶段 1 出口

### 阶段 3：三池 CacheSpec 与 CPU allocator

- [x] 实现 capacity planner、prefix/recent/history allocator 与生命周期
- [x] 在隔离分支接入 v0.19 scheduler/worker
- [x] 完成容量守恒、回滚和边界测试
- [ ] 更新中文阶段报告
- **状态：** 隔离准备分支已完成联合容量规划、三池 allocator、scheduler/worker ownership 与精确 tensor views；116 项定向测试通过，另有 68 项离线通用 scheduler 回归通过，代码推送至 `e75a40a29`；正式 submodule 接入与阶段报告仍等待阶段 2 出口

### 阶段 4：A800/SM80 Triton kernels

- [ ] 按设计顺序实现 store、demotion、mixed sparse MLA 和 inverse rotation
- [ ] 完成 SM80 cold compile、A800 launch、oracle 与边界测试
- [ ] 更新中文阶段报告
- **状态：** 隔离分支已提交 rotation/INT2 store、BF16 store、demotion、history dequant、包含 64 维原精度 RoPE score 的 mixed sparse decode/prefill、global LSE merge 和 inverse rotation WIP 候选；实际 512+64 维 CPU Triton interpreter 已通过 oracle，61 项测试通过，22 项 CUDA 测试因正式 baseline 占满 GPU 而按门禁跳过；最新代码为 `8ac7b9d97`，SM80 cold compile/A800 launch 与正式验收未完成

### 阶段 5：vLLM 接入与 32K 端到端

- [ ] 注册并接入 `oscar_mla_int2`
- [ ] 完成单/多请求、demotion、DSA mixed read 和接近 32K 验证
- [ ] 证明无 fallback、无完整 BF16 history，并满足压缩率阈值
- [ ] 更新中文阶段报告
- **状态：** 隔离分支已完成配置/spec、runtime artifact fail-closed 加载、GPU batch metadata，以及三池 write/demotion/DSA mixed read 的代码接线；最新 commit `c762b4aee` 已推送，80 项测试通过、26 项 A800 测试跳过，并显式拒绝 V2 runner、非 eager、CUDA graph、speculative、DCP、PCP、DBO、async scheduling、KV transfer 和 KV offloading；真实 GLM‑5.2 EngineConfig 已验证显式关闭 async 后成功且未初始化 CUDA，CPU interpreter 已实际覆盖非连续 rotation stride、两请求独立 HP/history/RoPE ownership 和 `-1` DSA padding，runtime mock 另验证两个不同长度请求的局部 query position，并分别对输出与自然对数域 LSE 做 PyTorch oracle 对比；A800 门禁已加入 chunked/one-shot 最终分区一致性和 batch 4/8 多请求隔离，INT2 reference 已覆盖 constant/narrow/random/outlier；A800 实际 launch 和服务端到端尚未验证

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

1. 阶段 2 正式 manifest 已冻结；阶段 1 全量 accuracy/PPL 完成后，需要切换已推送的 calibration 源码并运行 TP=8 只读 capture、合并与 artifact 导出。

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
| 创建 public `XiancaiTian/glm52_oscar_vllm`，稳定分支使用 `main` | Shawn 于 2026-07-25 确认推荐方案，并授权后续按最佳方式自主推进 |
| 远端只同步当前冻结代码快照，不上传完整恢复历史和大型实验产物 | Shawn 于 2026-07-25 明确要求大文件不必 commit/push，主要同步代码文件；完整历史保留在本地追溯分支 |
| 长时间 baseline 运行期间在项目内 ignored worktree 准备阶段 2 | 不改正式 submodule 指针、不污染运行时源码；隔离分支通过测试并推送后仍由阶段闸门决定何时接入 |
| 阶段 3 将标准 vLLM block table 与独立 INT2 history page namespace 分离 | 标准 block 继续承载全序列 BF16 RoPE 与 21 层原生 DSA cache；OSCAR allocator 只管理 latent prefix/recent/history，避免丢失 vLLM null block 和辅助 cache 所有权 |
| 阶段 4 CUDA 测试必须显式设置 `VLLM_OSCAR_RUN_CUDA_TESTS=1` | 当前正式 baseline 占满 8 卡；默认跳过避免测试 import 意外创建 CUDA context，待服务退出并再次确认 GPU 空闲后执行 cold-cache A800 验收 |
| calibration 数据先以 OpenWebMath 固定 revision 和只读 LongBench 文件作为候选 | 二者可覆盖数学、通用长文本与代码；在最终 manifest 的样本 ID、token 数与 SHA256 冻结前不宣称为正式数据集 |

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
| 完整历史推送包约 185MiB，不符合最新“主要同步代码”要求 | 2 | 停止完整历史上传；以相同 tree 创建无父提交的冻结代码快照，完整历史仅保留在本地 `recovery/full-history` |
| 首次 TP=8 启动因 FlashInfer/JIT cache 版本门禁退出 | 1 | 固定 venv 实测为 0.6.6/0.6.7.post3+cu129；已验证部署入口原本设置 `FLASHINFER_DISABLE_VERSION_CHECK=1`，补齐该通用环境后重跑 |
| 第二次 TP=8 启动因误读当前容器的 `flash_attn` 包退出 | 1 | 改用候选 rootfs 自带 Python 3.12.13，以 `PYTHONHOME` 和显式候选 venv/rootfs 路径隔离当前系统包；dry-run 已通过 |
| 首轮 official_v4 的 600 秒 code timeout 导致首批 8 条中 6 条请求失败 | 1 | 10 分钟时 KV usage 从 20.8% 降至 2.4% 并装入下一批，但服务仅记录 2 个 HTTP 200；停止不可能满足 2360/2360 scored 的无效轮次，runtime code timeout 固定为 900 秒 |
| 阶段 2 worktree 首次 pre-commit 初始化停滞，中止时 linked worktree index 被 hook cache 内容覆盖 | 1 | 核对正式源码 worktree 与对象库均完好；记录 5 个新文件 SHA256，用 `git read-tree HEAD` 仅重建隔离 worktree index，再按哈希重新暂存并通过 pytest、ruff、format 和 diff 检查 |
| capture 首轮测试发现输出字典的 `value_samples` 同时作为计数和张量键 | 1 | 将统计计数重命名为无歧义的 `*_covariance_samples`；重跑后 17 项 pytest、语法、ruff、format 和 diff 全部通过 |
| Hugging Face Xet CDN 在当前网络出现 TLS `wrong version number`，固定 OpenWebMath shard 直接下载无法完成 | 3 | curl、`hf_hub_download`、wget 三条路径均失败后停止重试；改用官方 datasets-server 按固定 revision 获取 0–299 行并在项目 artifacts 内冻结 SHA256 |
| phase-0 native hash 清单第 7 项使用 rootfs 内相对路径，在主仓库 source 目录直接执行整份 `sha256sum -c` 找不到文件 | 1 | 前 6 个 vLLM `.so` 通过只读 symlink 和原清单验证；第 7 个 sparse MLA `.so` 改按候选 rootfs 绝对路径与原清单 hash 单独核验，7/7 通过 |
| 完整 scheduler 测试默认尝试下载 LLaVA 测试模型并在项目外创建 Hugging Face cache | 1 | 立即终止下载；精确删除本次新建的 3,622,499-byte 模型 cache、0-byte lock 和 36KB Xet 日志，随后固定 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` 重跑 |
| Triton interpreter 的 split merge 对标量 mask 执行位与时报类型不兼容 | 1 | 拆分为两个 `tl.where` 条件，避免不同标量类型的位运算 |
| Triton interpreter 中 BF16 `tl.dot` rotation 产生无效大值 | 1 | rotation 改为 FP32 输入与 IEEE FP32 累加；512 维 CPU oracle 复测通过，SM80/A800 仍待正式验证 |
| Stage 5 新 worktree 的空 `uv` venv 缺少 pytest conftest 依赖 `tblib` | 1 | 使用清华 PyPI 镜像通过 `uv pip` 安装 `tblib==3.2.2`，随后在该 worktree 自有 `.venv` 中重跑 8 项测试通过 |
| Stage 4 RoPE 修复 pytest 被 worktree venv 的未安装依赖阻断 | 3 | 依次补齐 `cbor2==5.8.0`、`cachetools==7.0.1` 与 `py-cpuinfo==9.0.0`；定向 interpreter 和完整 `tests/oscar_mla` 随后分别通过 |

## 约束提醒

- 所有报告必须使用中文，且数据只能来自实际落地结果。
- 每个阶段实验结束后先更新报告，再进入下一阶段。
- 修改已有报告前必须重新读取，并检查章节序号与交叉引用。
- 长实验每 10 分钟记录一次进度。
- GPU 实验必须在固定 Docker 容器内执行，且分配前连续两次确认授权 GPU 空闲。
- 正式实验前必须提交并推送源码与配置；不得让主仓库 submodule 指向本地-only commit。
- `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1` 及其他项目外目录仅可只读检查；禁止修改、暂存、提交、清理或重置。
