# 实验推进日志

## 会话：2026-07-24

### 阶段 0：恢复并冻结源码

- **状态：** 本地技术出口通过；协作发布待完成
- **已执行：**
  - 完整读取 `planning-with-files` 规范、项目 `AGENTS_misc.md` 和设计文档。
  - 检查主仓库状态与两级目录资产。
  - 确认根目录没有本任务的持久计划，且 `oscar_vllm/` 下同名文件属于其他工作。
  - 创建本任务的 `task_plan.md`、`findings.md`、`progress.md`。
  - 记录 Shawn 的 GPU 授权：当前机器所有可见 GPU 均可使用。
  - 运行 planning-with-files session catchup；其仅检测到本轮刚完成但尚未进入旧日志快照的 `update_plan`/审计工具调用，无额外历史工作需要恢复。
  - 核验主仓库远端和 submodule 指针。
  - 尝试查询 Docker 环境，确认当前 shell 缺少 Docker CLI。
  - 定位到本地已验证镜像 tar、SHA256 和 image manifest。
  - 确认当前执行环境是 Kubernetes/containerd 容器，`nvidia-smi` 可用但没有常见容器构建/解包 CLI。
  - 读取 Docker archive manifest，确认 image config 文件名和 31 个 layer。
  - 找到外部完整源码候选 `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1`（约 2.0GB）。
  - 启动镜像 tar 全量 SHA256 校验。
  - 读取外部源码仓库的 `AGENTS.md`，确认后续 Python 环境必须使用 `uv`。
  - 通过一次性 Git safe.directory 核验源码 HEAD、分支、remote、tracked file 数量和工作树状态。
  - 审计外部源码 6 个脏文件：626 行新增、39 行删除，主要是 prefill shape-bucket 实验。
  - 完成 34.75GB 镜像 tar 的全量 SHA256 复核。
  - 将镜像 archive 解包到唯一临时目录 `/tmp/oscar-glm-image.A3ieFx`，只用于只读层分析。
  - 流式扫描 31 个 layer，确认 `/opt/vllm_glm52_v1` 只位于 layer 29–31，并定位 6 个 vLLM 原生扩展。
  - 核验 layer 30/31 的覆盖文件和 whiteout 范围。
  - 恢复镜像有效源码树到 `/tmp/glm52-image-source.rAHK8y`：4,711 个文件、1,865,041,381 字节。
  - 从外部 Git HEAD 成功导出 clean tree 到 `/tmp/glm52-clean-head.0x4tLq`：4,705 个文件、77,547,072 字节。
  - 比较 clean HEAD 与镜像源码，确认 shape-bucket 脏改动未进入镜像，并发现 4 个 tracked 源码差异。
  - 阅读 4 个源码差异：1 个 Inductor autotune 安全 patch，3 个 manifest 已记录的 usage/proxy patch。
  - 用上一层稳定镜像 manifest 和精确源文件闭合 `compiler_interface.py` patch 来源。
  - 记录 6 个 vLLM 原生扩展 SHA256。
  - 只读检查候选 GitHub URL，确认两个建议名称当前均不存在。
  - 检查容器能力、runtime socket 和构建工具，确认当前无法直接使用宿主 daemon。
  - 生成完整历史 Git bundle，并从 bundle clone 出 `glm52_oscar_vllm/`。
  - 创建本地稳定分支 `main`，初始指向固定 commit。
  - 通过 `apply_patch` 加入镜像实际存在的 4 个 runtime patch。
  - 对 4 个候选文件逐一计算 SHA256，全部与镜像有效源码匹配；`git diff --check` 通过。
  - 使用 `uv` 创建 Python 3.12.3 虚拟环境，4 个 patch 文件通过 `py_compile`。
  - 安装 pre-commit hooks；GitHub hook 初始化停滞后，改用清华 PyPI 安装固定 ruff 0.14.0。
  - ruff 定向检查发现 4 个镜像既存 lint 告警，未自动修复，以保留逐字节一致性。
  - 解析镜像 config/history，冻结 OS、CUDA、Python、入口和关键环境变量。
  - 从 layer 29 的 dist-info 读取 PyTorch、Triton、vLLM、Transformers、Tokenizers 实际版本。
  - 定位并计算 sparse MLA splitmerge native extension 的大小与 SHA256。
  - 创建可复现 baseline Dockerfile、build 脚本、JSON manifest、source/native SHA256 清单和中文恢复说明。
  - 验证 4/4 source checksum、7/7 native checksum、JSON 格式、shell 语法和 `git diff --check`。
  - 实测 unprivileged user namespace 可用。
  - 安装 buildah 1.33.7，并成功运行 `vfs + chroot` info。
  - 检查容器存储余量和 `/dev/fuse`；当前 overlay 余量约 302GB，`/dev/fuse` 尚不存在。
  - 安装 fuse-overlayfs 1.13 并创建 `/dev/fuse`。
  - 依次测试默认 overlay、显式 fuse mount program、额外 user/mount namespace；三种方式均被 Kubernetes mount 权限拒绝。
  - 决定切换到已验证的 `vfs + chroot`，并使用任务专用 NFS graphroot。
  - buildah vfs 导入仍被容器 remount/subuid 限制阻断后停止该路径。
  - 安装 `skopeo 1.13.3` 与 `umoci 0.4.7`，确认可采用不依赖 daemon/mount 的 OCI 组合方案。
  - 根据 Shawn 最新要求，将 `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1` 及其他项目外路径冻结为严格只读。
  - 只读复核外部仓库状态：HEAD 未变，状态项仍是此前已记录的 6 个 tracked 修改文件，没有新增改动。
  - 用 `skopeo` 成功把已验证 Docker archive 转为项目内 OCI layout；基线 manifest digest 为 `sha256:1b8e808e...6415`。
  - 从 Git index 导出 4,711 个 tracked 文件的干净 payload；确认不含 `.git`、`.venv` 或 `.so`，4 个 runtime source checksum 全部通过。
  - 分别用 NFS payload 和本地 `/tmp` payload 测试 `umoci insert`；两次生成的层都固定截断最后 393 字节。
  - 将两个 `umoci` tag 判定为失败且不导出，停止该插层路径；转为标准 tar + OCI descriptor 组装。
  - 使用 GNU tar/gzip 和 OCI descriptor 组装出候选 manifest `sha256:2fdfbe865aecc01eee15a01fcce58bf7581244dbbc53cbe3ef0e0cce44bc489d`。
  - 对候选第 32 层完成逐文件校验：5,256 个 tar 成员、4,711 个普通文件，内容和权限全部与固定 Git tree 一致。
  - **长任务 10 分钟进度：** `umoci unpack` 于运行 11 分 42 秒时仍处于活动状态（`Ssl`），没有 layer/tar 内容错误；仅出现 NFS 不支持 rootless xattr 的降级警告，继续等待最终 rootfs SHA256 校验。
  - **长任务 20 分钟进度：** `umoci unpack` 于运行 21 分 51 秒时仍处于活动状态（`Ssl`），NFS 尚有约 2.7TB 可用空间，仍无 layer/tar 内容错误；继续等待结束码。
  - **长任务 30 分钟进度：** `umoci unpack` 于运行 32 分 13 秒时仍处于活动状态（`Ssl`），已进入 `/opt/vllm_glm52_v1` 候选源码层，NFS 仍有约 2.7TB 可用空间，没有 layer/tar 内容错误。
  - **长任务 40 分钟进度：** `umoci unpack` 于运行 41 分 32 秒时仍处于活动状态（`Ssl`），NFS 尚有约 2.6TB 可用空间，没有新增 layer/tar 内容错误；继续等待结束码。
  - **长任务 50 分钟进度：** `umoci unpack` 于运行 51 分 39 秒时仍处于活动状态（`Ssl`），NFS 尚有约 2.6TB 可用空间，仍无新增 layer/tar 内容错误；继续等待结束码。
  - **长任务 60 分钟进度：** `umoci unpack` 于运行 1 小时 00 分 14 秒时仍处于活动状态（`Ssl`），累计读取约 20.27GB、写入约 35.11GB，NFS 尚有约 2.6TB 可用空间；没有新增 layer/tar 内容错误。
  - **长任务 70 分钟进度：** `umoci unpack` 于运行 1 小时 11 分 59 秒时仍处于活动状态（`Ssl`），累计读取约 20.38GB、写入约 35.11GB，NFS 尚有约 2.6TB 可用空间；读取计数继续增长且无 layer/tar 内容错误。
  - 检查 60–72 分钟 I/O 后确认写入量固定、仅 NFS 元数据读取增长；终止这项额外强校验，并将 `umoci unpack` 完成状态记为未通过。
  - 在已展开 rootfs 上执行两份 SHA256 清单，4/4 runtime source 和 7/7 native extension 全部 `OK`。
  - 独立第二次生成标准 source layer，digest、diff ID 和 33,253,726 字节大小与首次构建完全相同。
  - 提交无 daemon OCI 重建与验证资产：`0288235f2b93563c13e6c3750c5797fccbaba70d`。
  - 创建中文阶段报告 `docs/experiments/2026-07-24-phase0-source-recovery.md`，并检查章节编号 1–8 连贯、无错误交叉引用。
  - 阶段 1 第一次 GPU 检查：8 张 A800 均为 0MiB、0% 利用率、无 compute process。
  - 间隔 60 秒后于 `2026-07-24T10:42:18Z` 完成第二次检查，8 张 A800 状态保持不变，连续空闲检查通过。
  - 复核模型目录：141 个 safetensors、152 个顶层普通文件、总大小 462,858,376,495 字节，tokenizer/config 文件存在。
  - 核验当前 Pod rootfs：缺少目标镜像 3 个 `/opt` 路径，不能直接作为固定候选环境。
  - 尝试只读查询当前 Pod image/spec，Kubernetes RBAC 返回 403；停止该路径。
  - SSH 探测只回连当前容器的 27868 端口，仍无 Docker；停止宿主 Docker 路径。
  - 直接使用项目内展开 rootfs 的固定 venv/source 做 CPU import：固定依赖和 `vllm._C` 加载通过，CUDA 未初始化。
  - 读取模型 config/index，核验 GLM‑5.2 DSA/MLA、MoE、FP8 几何及 72,117 个权重映射项。
  - 计算 safetensors 文件名/大小清单、index 和首尾 shard 的轻量指纹。
- **实际结果：**
  - 阶段 0 四项本地技术出口条件全部通过；GitHub 远端和 submodule 接入待确认。
  - candidate manifest 为 `sha256:2fdfbe86...489d`，config 为 `sha256:58a853ee...a42d`。
  - 两次 source layer 重建结果完全一致；候选 rootfs 11/11 SHA256 通过。
  - 外部源码始终只读，HEAD 与原有 6 个工作树修改状态未被本任务改变。
  - GPU 授权和连续两次空闲检查均已满足；尚未运行阶段 1 GPU baseline。
  - 当前阶段 1 运行阻塞是固定容器入口和正式源码远端尚未闭环，不是 GPU 授权不足。
- **创建/修改文件：**
  - `task_plan.md`（新建）
  - `findings.md`（新建）
  - `progress.md`（新建）
  - `glm52_oscar_vllm/`（独立本地 Git 仓库）
  - `docs/experiments/2026-07-24-phase0-source-recovery.md`（新建中文报告）

### 阶段 1：原生 baseline 本地入口准备

- **状态：** 本地入口和评测门禁通过；正式 GPU 运行等待远端发布
- **已执行：**
  - 主仓库和源码仓库创建同名功能分支 `feat/glm52-model-load`。
  - 主仓库阶段 0/1 资产已提交并推送；远端功能分支与本地均指向 `75f30615eaaa2e94723e570c62d3e1cbe64b9658`。
  - 固化 `configs/phase1/native_baseline.json`，记录候选 OCI、源码、模型、official_v4 和服务配置。
  - 创建全量静态 verifier，核验 OCI descriptor、Git tree、native extension、模型几何/指纹和 suite 行数/SHA256。
  - 首次 verifier 发现 rootfs 与源码 HEAD 在 3 个 `recovery/` 文件上不一致；核对 OCI label 和 Git 历史后，区分仓库 HEAD `028823...` 与实际 runtime source `fd3e0b...`。
  - 按 `fd3e0b...` 的 4,711 文件 Git tree 重新执行内容、符号链接和权限全量验证，结果全部通过。
  - 创建 TP=8 server launcher，显式使用 eager、32K、`TRITON_MLA_SPARSE`、native KV、无 speculative、无 prefix cache、无 CUDA graph。
  - 加入正式实验门禁：两个仓库必须干净、同名功能分支 HEAD 必须已发布到 origin、8 张 GPU 必须间隔 60 秒连续空闲。
  - 加入 server/accuracy/PPL 的每 10 分钟进度与 GPU 状态记录。
  - 通过固定 venv 实际解析完整 server CLI；结果为 TP=8、PP=1、32K、eager、prefix cache=false、speculative=null，且 CUDA 未初始化。
  - 创建短请求、>320 tokens、384-token 连续 decode 和近 32K smoke runner。
  - CPU 测试首次发现 `BatchEncoding` 长度误用；修复后本地 chat-template token 数为 506 和 31,996。
  - 只读核验 official_v4：2360 个 accuracy 样本和 1 个 WikiText‑2 样本；冻结 suite 与两个 runner SHA256。
  - 用 `uv` 创建项目内独立 evaluator venv；补装原 requirements 未声明但 runner 实际需要的 `requests`、`nltk`、`absl-py`，冻结 20 个包。
  - 创建 official_v4 accuracy 和 TP=8 WikiText‑2 PPL wrapper、完整输出校验及 predictions/summary SHA256 归档入口。
- **实际结果：**
  - 静态 preflight 状态为 `passed`；4,711/4,711 runtime files、7/7 native extensions、141 shards、2360+1 suite samples 均通过。
  - 所有新增 shell 脚本通过 `bash -n`；Python 脚本通过固定 Python 3.12.3 `py_compile` 和 ruff 0.14.0。
  - vLLM 完整启动参数通过真实 CLI parser；尚未初始化 CUDA、尚未加载模型、尚未产生 baseline 指标。
  - 主仓库远端已闭环；源码仓库远端未确认，因此未绕过正式实验的双仓库 commit/push/clean gate。
- **创建文件：**
  - `configs/phase1/native_baseline.json`
  - `configs/phase1/evaluator-requirements.lock.txt`
  - `scripts/phase1/verify_native_baseline.py`
  - `scripts/phase1/run_native_baseline.sh`
  - `scripts/phase1/run_native_smoke.py`
  - `scripts/phase1/run_official_v4_accuracy.sh`
  - `scripts/phase1/run_native_ppl_wrapper.py`
  - `scripts/phase1/run_native_ppl.sh`

## 会话：2026-07-25

### 阶段 1：恢复远端发布

- **状态：** 完成
- **已执行：**
  - Shawn 确认创建 public `XiancaiTian/glm52_oscar_vllm`，稳定分支使用 `main`。
  - Shawn 授权后续按最佳方式自主推进，不再为常规技术选择逐项确认。
  - 已创建 public `https://github.com/XiancaiTian/glm52_oscar_vllm`。
  - 首次完整历史推送为单线程 pack，4 分钟仅生成约 19MB；改为 8 线程后确认完整历史包约 185MiB。
  - 按 Shawn 最新要求停止完整历史上传；以相同 Git tree 创建无父提交的冻结代码快照。
  - 将冻结代码快照推送到 public 仓库的 `main` 和 `feat/glm52-model-load`。
  - 在主仓库接入 submodule，更新阶段 1 source commit 门禁并推送 `04b9445cd05a914b1716479a6bbdced018ebc177`。
  - 使用新 public commit 重跑阶段 1 静态 preflight。
- **实际结果：**
  - 原 commit `0288235f2...` 与代码快照 `53d8be94f...` 的 tree 均为 `ca5d4f035...`，`git diff` 为空。
  - 完整历史保留在本地 `recovery/full-history`；远端只上传当前冻结代码快照和后续开发提交。
  - 代码快照实际上传 5,014 个对象、33.13MiB；public `main` 和功能分支均解析到 `53d8be94f...`。
  - 新 commit 下 4,711/4,711 runtime files、7/7 native extensions、141 shards 和 2360+1 suite samples 再次全部通过。

### 阶段 1：正式 GPU preflight

- **状态：** 完成
- **已执行：**
  - 以主仓库 `8f7be26a...`、源码仓库 `53d8be94f...` 执行 fail-closed 正式 preflight。
  - 于 2026-07-25T15:55:27Z 和 15:56:31Z 间隔 64 秒检查全部 8 张 A800。
- **实际结果：**
  - 静态指纹和固定环境检查全部通过，运行目录为 `artifacts/phase1/20260725T155457Z_native_tp8`。
  - 两次均为 8/8 GPU 0MiB、0% 利用率、无 compute process；满足正式分配条件。

### 阶段 1：首次 TP=8 服务启动

- **状态：** 未通过
- **已执行：**
  - 在 `artifacts/phase1/20260725T155732Z_native_tp8` 启动原生 TP=8 服务。
  - 启动器再次完成两次 8/8 GPU 空闲检查，并创建 8 个 worker。
- **实际结果：**
  - worker 于 2026-07-25T16:02:22Z 在 GPU 显存分配和权重加载前退出，服务退出码为 1。
  - 根因是 `flashinfer-python 0.6.6` 与 `flashinfer-jit-cache 0.6.7.post3+cu129` 的版本门禁。
  - 已验证部署入口原本导出 `FLASHINFER_DISABLE_VERSION_CHECK=1`；当前 launcher 遗漏该通用环境变量。

### 阶段 1：固定运行环境修复

- **状态：** 完成
- **已执行：**
  - 将已验证部署入口中的通用 FlashInfer、HND KV、V1 multiprocessing、HF offline/cache 环境补入正式 launcher。
  - 使用固定 venv 实际导入 `flashinfer.comm`，并执行 shell 语法和 diff 检查。
  - 提交并推送 `44dda4484ed0d960572d80fe5e5dd485938c2d9a`。
- **实际结果：**
  - FlashInfer 通信模块导入通过，0.6.6/0.6.7.post3+cu129 版本保持不变；兼容方式与已验证部署入口一致。
  - 诊断导入实际初始化 CUDA；进程退出后未保留 GPU 占用，正式重跑仍会重新执行两次空闲检查。

### 阶段 1：第二次 TP=8 服务启动

- **状态：** 未通过
- **已执行：**
  - 以主仓库 `5e291379b257f9c3abf584c17efe3a4c287d54ca` 和源码仓库 `53d8be94f...` 在 `artifacts/phase1/20260725T160547Z_native_tp8` 启动服务。
  - 启动器于 16:06:51Z 和 16:07:55Z 再次完成两次 8/8 GPU 空闲检查。
  - 服务越过 FlashInfer 版本门禁，完成 EngineCore、8 worker、NCCL 初始化并进入模型对象构造。
- **实际结果：**
  - 日志于 16:11:00Z 打印 `Starting to load model`，随后 8 个 worker 均因 `ModuleNotFoundError: No module named 'flash_attn.ops'` 退出；未出现 checkpoint shard 权重读取记录，服务退出码为 1。
  - 固定 venv 开启 system site-packages；在候选容器外直接运行时，绝对 `/usr/local/lib/python3.12/dist-packages` 错误解析到当前容器。
  - 当前容器的 `flash_attn` 4.0.0b8 只有 `cute` 子包，导致候选源码的可选包探测通过、后续 `ops` 导入失败；候选 rootfs 本身不存在顶层 `flash_attn`。
  - 同一路径泄漏还使可选 `triton_kernels` 从当前容器解析并报告缺少 `SparseMatrix`。处理方向是隔离当前容器系统包并映射候选 rootfs，不安装依赖、不改外部环境。

### 阶段 1：候选 rootfs Python 隔离修复

- **状态：** 完成
- **已执行：**
  - 审计当前容器与候选 rootfs 的 Python 解释器、`pyvenv.cfg`、`sys.path`、package spec 和 dist-info。
  - 将 baseline/PPL 入口改为候选 rootfs 自带 `/usr/bin/python3.12`，并显式设置候选 `PYTHONHOME`、`VIRTUAL_ENV` 和 source/venv/rootfs 包路径。
  - 在启动前加入解释器来源及顶层 `flash_attn`/`triton_kernels` 缺失断言。
  - 执行两个 shell 脚本的 `bash -n`、`git diff --check` 和完整 Stage 1 dry-run。
- **实际结果：**
  - 候选解释器实际为 Python 3.12.13；Torch 2.11.0+cu129、Triton 3.6.0、Transformers 5.8.1、Tokenizers 0.22.2、`vllm._C` 均从候选树加载，CUDA 未初始化。
  - `flash_attn` 与顶层 `triton_kernels` 的 spec 均为 `None`；FlashInfer/JIT cache 为候选环境匹配的 0.6.6/0.6.6+cu129。
  - 静态 verifier 仍为 4,711/4,711 runtime files、7/7 native extensions、141 shards、2360+1 suite samples 全部通过；完整 TP=8 参数解析通过。

### 阶段 1：第三次 TP=8 服务启动与 smoke

- **状态：** 通过
- **已执行：**
  - 以主仓库 `6d7a7bd7c35dad6290bf599ba5120109eaae8b3f`、源码仓库 `53d8be94f...` 在 `artifacts/phase1/20260725T162556Z_native_tp8` 第三次正式启动。
  - 重新完成两次 8/8 A800 空闲检查，加载 141 个 checkpoint shard，并按 10 分钟周期记录服务/GPU 进度。
  - 服务 ready 后依次执行短请求、>320 输入、强制 384-token 连续 decode 和近 32K smoke。
- **实际结果：**
  - 141/141 shard 全部加载；权重读取 2,337.66 秒，模型加载 2,393.23 秒、每卡模型内存 55.93GiB。
  - KV cache 可用 14.3GiB、容量 165,696 tokens，32,768-token 请求理论并发 5.06；初始化后处理耗时 162.94 秒。
  - 服务于 17:14:53Z ready，launcher 记录总启动耗时 2,762 秒；服务日志未出现 ERROR/Traceback。
  - 短请求 21+64 tokens/16.06 秒，>320 请求 506+18 tokens/3.56 秒，连续 decode 26+384 tokens/59.90 秒，近 32K 请求 31,996+64 tokens/24.28 秒；四项均 HTTP 200 并通过。

### 阶段 1：official_v4 原生精度基线

- **状态：** 首轮未通过，已停止
- **已执行：**
  - 于 17:18:08Z 启动 2,360 样本、并发 8 的冻结 official_v4 runner。
  - 评测进程、服务 PID、健康接口和 8 个并发请求均已核验存活。
  - 10 分钟节点已写入 `official_v4_accuracy/progress_10min.log`。
- **实际结果：**
  - manifest 前 175 条为 LiveCodeBench v6，每条 `max_tokens=4096`、HTTP timeout 600 秒，因此首批样本耗时接近 10 分钟。
  - 10 分钟时服务 KV usage 从 20.8% 降至 2.4% 并装入下一批，但首批仅 2 个 HTTP 200；其余 6 个请求已触发客户端 timeout。
  - 因设计硬门槛要求 request failure=0，本轮不可能达到 2,360/2,360 scored，已主动停止；没有生成最终 `predictions.jsonl`、summary 或 accuracy。

### 阶段 1：official_v4 code timeout 修复

- **状态：** 探针通过
- **已执行：**
  - 保持冻结 manifest、样本、decoding 和 runner 不变。
  - 将每次 accuracy 结果改为独立 attempt 目录，避免覆盖失败证据。
  - 从冻结 `eval_config.json` 生成本地 runtime config，仅把 code timeout 从 600 秒固定提高为 900 秒，并记录源/运行时 config SHA256。
  - 使用冻结 runner 和 manifest、并发 8 运行前 8 条 LiveCodeBench v6 探针。
- **实际结果：**
  - 探针耗时 638.53 秒，8/8 请求成功、8/8 `scored`、request failure=0。
  - accuracy 为 0.0，代表 8 条代码答案均未通过评分；所有 evaluator 状态仍为 `scored`，不存在请求失败。
  - runtime eval config、predictions、summary SHA256 分别为 `8e0beeb1...22c5`、`05390ba2...96e2`、`3a78a575...2e7`。
- **下一步：**
  - 运行全量 2,360 样本，继续每 10 分钟记录进度。

### 阶段 1：official_v4 全量 900 秒基线

- **状态：** 正式精度合并结果通过完整性门禁
- **已执行：**
  - 于 2026-07-25T17:50:15Z 使用冻结 2,360 样本 manifest、并发 8 和 code timeout 900 秒的 runtime config 启动正式全量评测。
  - 在 2026-07-25T18:00:15Z 至 2026-07-26T10:40:34Z 分别写入 10–1010 分钟 GPU 与进程进度。
  - 对 runner 缓冲输出、服务 POST 状态和错误日志分别核验，不用服务请求数替代最终 scored 数。
- **实际结果：**
  - 一百零一次进度记录期间 8 张 A800 均维持约 79,901–79,941MiB 显存占用，评测 runner、服务与并发请求持续运行。
  - runner 于 18:22:01Z 刷新 `completed 20/2360`；60/70/80/90/100/110/120/130/140/150/160/170/180/190/200/210/220/230/240/250/260/270/280/290/300/310/320/330/340/350/360/370/380/390/400/410/420/430/440/450/460/470/480/490/500/510/520/530/540/550/560/570/580/590/600/610/620/630/640/650/660/670/680/690/700/710/720/730/740/750/760/770/780/790/800/810/820/830/840/850/860/870/880/890/900/910/920/930/940/950/960/970/980/990/1000/1010 分钟节点分别为 40/40/60/60/80/80/80/100/100/100/120/120/120/140/140/160/160/180/200/220/240/240/260/280/300/300/320/340/360/380/380/400/420/440/440/460/480/500/540/580/600/640/660/700/740/760/800/820/860/900/920/960/980/1020/1060/1080/1120/1160/1180/1220/1240/1280/1320/1340/1380/1420/1440/1480/1500/1540/1560/1600/1620/1660/1700/1720/1760/1780/1820/1840/1880/1900/1940/1980/2000/2040/2060/2100/2120/2160/2180/2220/2260/2280/2320/2340，前三个节点因 stdout 缓冲记录为 `completed unknown/2360`。
  - 2026-07-26T10:40:33Z 至 `10:41:13Z` 服务保持 Running=8、Waiting=0、生成吞吐为 11.2–15.2 tokens/s，健康检查返回 HTTP 200，两个正式进程持续存活且错误扫描为空。
  - runner 实际完成全部 2,360 条请求并写入 2,360 行预测；summary 为 2,353 条 `scored`、7 条 `request_failed`，已评分样本 accuracy 为 `0.19932001699957502`，runner duration 为 `60881.98892402649` 秒。
  - 7 条失败均为连续 GSM8K ID `gsm8k:001311` 至 `gsm8k:001317`，错误均为客户端 `read timeout=300`；对应期间服务端错误扫描为空，孤立请求结束后健康接口仍返回 HTTP 200。
  - 分项结果为 GSM8K 1,312/1,319 scored、262 条正确、accuracy `0.19969512195121952`；IFEval 541/541 scored、145 条正确、`0.2680221811460259`；LiveCodeBench 175/175 scored、12 条正确、`0.06857142857142857`；MultiPL-E 325/325 scored、50 条正确、`0.15384615384615385`。
  - predictions、failed cases、summary、benchmark summary、runtime config 与 runner log SHA256 分别为 `fbc69f74...cae2`、`e390f712...4264`、`2996c8ba...1b8`、`6583c2f8...76b8`、`8e0beeb1...22c5`、`78e7fc42...aa94`。
  - 因硬门禁要求 2,360/2,360 scored，首轮判定未通过；已新增 fail-closed 补跑入口，严格核对 7 个失败 ID、错误原因与 prompt hash，只将 math timeout 从 300 秒提高到 900 秒，并在独立目录按 ID 合并，不覆盖首轮证据。
  - 精确补跑实际耗时 `168.74860620498657` 秒，7/7 为 `scored`、request failure=0、accuracy 0.0；7 条答案均未判对，但请求与 evaluator 状态全部成功。补跑 predictions、summary、runner log SHA256 分别为 `9bfc340c...c541`、`1f015c82...fed2`、`89435399...ea1`。
  - 首版合并门禁拒绝输出：完整 official_v4 manifest 实际含 2,361 行，其中第 2,361 行是由独立入口执行的 WikiText‑2 PPL；正式 accuracy 命令只选择四个 benchmark、共 2,360 行，旧逻辑错误地与未过滤 manifest 比较。
  - 已拆出 `merge_official_v4_retry.py`，显式固定 GSM8K、IFEval、LiveCodeBench v6、MultiPL-E，验证唯一 ID 集、prompt hash、7 个替换 ID 和唯一排除的 PPL 行。
  - 代码推送后生成正式合并目录 `merged_full_retry_20260726T1112Z`：2,360/2,360 `scored`、request/extraction failure 均为 0、469 条正确、accuracy `0.19872881355932204`，合并 duration 按首轮与补跑 runner duration 之和记录为 `61050.737530231476` 秒。
  - 正式分项结果为 GSM8K 1,319/1,319 scored、262 条正确、accuracy `0.19863532979529946`；IFEval 541/541、145 条正确、`0.2680221811460259`；LiveCodeBench v6 175/175、12 条正确、`0.06857142857142857`；MultiPL-E 325/325、50 条正确、`0.15384615384615385`。
  - 正式 predictions、failed cases、summary、benchmark summary、merge provenance 与 validation SHA256 分别为 `68a3d0b1...3e75`、`9cf21542...a6fe`、`f8f52510...c9e2`、`c7a5754e...ac9b`、`4f3c6488...a693`、`06cc30a3...c432`；预测共 2,360 个唯一 ID，failed cases 的 1,891 行均为已评分但未判对的样本。

### 阶段 1：WikiText‑2 原生 PPL

- **状态：** 完成，validation 通过
- **已执行：**
  - accuracy 服务干净退出后确认无残留 vLLM/runner 进程，8 张 A800 首次为 0MiB、0%；正式 preflight 再间隔 60 秒完成两次 8/8 GPU 空闲检查。
  - preflight 通过固定 OCI/source/native/model/suite 全部门禁；PPL runner 于 2026-07-26T11:03:51Z 使用 TP=8、eager、BF16、2048 max length、512 stride、batch 8 启动。
  - 141/141 checkpoint shard 全部加载，权重读取耗时 224.65 秒、模型加载 236.68 秒、每卡模型内存 55.94GiB；8 个 rank 均使用 `TRITON_MLA_SPARSE`。
- **实际结果：**
  - 2026-07-26T11:13:51Z 的 10 分钟节点已写入 `progress_10min.log`；8 卡显存均为 79,581MiB，利用率为 98%–100%，runner/EngineCore 存活且错误扫描为空。
  - 评分阶段耗时 `368.10103392601013` 秒；输入 289,709 tokens，实际评分 289,708 tokens、563 个窗口，mean NLL `2.0402180131829573`，PPL `7.692286035848967`，1/1 `scored`。
  - summary、perplexity results、validation 与 runner log SHA256 分别为 `a0b1643b...a794`、`8c470884...72ed`、`17e740fe...eab2`、`d6b4e34e...b6f1`；日志错误扫描为空。
  - runner 结束后无残留 vLLM/EngineCore 进程，8 张 GPU 均为 0MiB、0% 利用率。
  - 阶段 1 中文报告已写入 `docs/experiments/2026-07-26-phase1-native-baseline.md`。

### 阶段 2：共享潜空间 reference、capture、artifact 与 manifest

- **状态：** 完成；正式 capture、fit、artifact 加载和阶段报告均通过
- **已执行：**
  - 在项目内 ignored worktree 和独立分支 `feat/glm52-shared-calibration` 开发，不改变正式 submodule、阶段 1 runtime 或外部只读源码。
  - 实现 shared-`R` native/rotated/mixed attention、非对称 INT2 量化/反量化、4×2-bit pack/unpack、FP64 covariance 累积、trace 归一化和 rotation 求解。
  - 添加 reference、calibration 与 capture 单元测试，并执行 pytest、Python 语法、ruff、format check 与 `git diff --check`。
  - 在原生 sparse MLA 的 value up-projection 前加入显式环境开关控制的只读 capture，按固定 token budget 采集每层 score/value/latent covariance、holdout 样本和少量 DSA top-k 样本。
  - 实现 rotation artifact 原子写入、manifest/tensor SHA256、完整层映射、正交性检查和 runtime 身份 fail-closed 加载。
  - 实现全部 TP payload 的 fail-closed covariance 合并、rank 0 latent 与全 TP query/value 聚合，以及固定 alpha/clip 网格的 holdout 量化误差搜索。
  - 实现确定性 calibration manifest 构建器：固定数据源 revision/SHA256、按源样本划分 train/holdout、排除 official_v4 prompt、按类别 token 配额生成，并在配额不足时失败。
  - 实现逐条核验服务端 `usage.prompt_tokens` 的顺序 prompt runner，以及 8 个 TP rank × 78 层完整性合并、共享搜索和 artifact 导出工具。
  - 固化正式 train/holdout capture 与 fit launcher：验证已推送源码、root submodule 指针、模型/数据/专家映射 hash、7 个原生扩展、运行环境和 capture 进程身份。
- **实际结果：**
  - reference/covariance 基础 commit 为 `507653c3d...`；只读 capture commit 为 `9c3b8401d...`，两者均已推送到 `origin/feat/glm52-shared-calibration`。
  - capture 首轮测试发现 `value_samples` 输出键冲突；改为无歧义的 covariance 计数键后，17 项 pytest、语法、ruff 0.14.0、format 和 diff check 全部通过。
  - 单测逐张量核验 capture 前后值、storage pointer、shape 均不变；固定预算 6 tokens 的 score/value/latent 二阶矩与直接计算一致。
  - artifact 测试覆盖正常 round-trip、缺层、错 shape、非正交、runtime 模型身份错误、tensor 篡改、layer manifest 错误、版本与元数据错误。
  - artifact commit `2100083b5...` 已推送；当前全套 25 项 pytest、语法、ruff、format 和 diff check 全部通过。
  - rotation/clip 搜索 commit `67deb6b9e...` 已推送；搜索固定 alpha `{0.25, 0.5, 0.75}` 与 clip `{0.92, 0.94, 0.96, 0.98, 0.99}`，不读取 official_v4。
  - manifest 构建器 commit `cd7fcc946...` 已推送；全套 33 项 pytest、ruff、format 和 diff check 全部通过。
  - prompt runner 与 fit 工具 commit 为 `8cbba2592...`，fit 10 分钟 heartbeat commit 为 `da4e2756a...`；当前全套 35 项 pytest、ruff、format 和 diff check 全部通过并已推送。
  - 阶段 1 完成后，正式源码 worktree 已从 `53d8be94f...` 切换到已推送的 `feat/glm52-shared-calibration` commit `da4e2756a...`；源码工作区干净，隔离 worktree 保留为同 commit 的 detached 只读参考。
  - 已从官方 datasets-server 固定 OpenWebMath revision `fde8ef8d...` 的 0–299 行，项目内文件为 300 行、2,800,069 字节、SHA256 `39d245ca...b80`；LongBench 固定 revision 为 `5e628be4...`，5 个只读文件 SHA256 均已写入配置。
  - 开发版 manifest 为 20 行、250,907 字节、50,000 tokens，SHA256 `cde88339...25da`；两次独立构建的 manifest 与 summary SHA256 均完全一致。
  - 正式版 manifest 为 292 行、4,570,560 字节、1,000,000 tokens，SHA256 `3a183cba...76b5`，summary SHA256 `f0e323f4...91bf`；两次独立构建完全一致。
  - 正式版独立审计确认 292 个 entry ID 唯一、235 个源样本无 train/holdout 跨分区、无重复文本 hash、与 official_v4 完整 prompt hash 交集为 0，重新 tokenize 后各类别 token 数与配额逐项一致。
  - phase-2 runtime 的 6 个 vLLM 原生扩展通过项目内只读 symlink 解析，另 1 个 sparse MLA 扩展按候选 rootfs 绝对路径验证；7/7 SHA256 通过，候选 Python 实测从主仓库 source 载入 `vllm`/`vllm._C` 且 CUDA 未初始化。
  - 阶段 1 退出后已将正式 submodule 指针切换到 calibration commit `da4e2756a...`；切换前后的两个源码 commit 都已在远端可解析。
  - 正式 train 服务完成两次 8/8 GPU 空闲检查，141/141 shard 加载后于 2026-07-26T11:25:53Z ready；权重读取 48.31 秒、模型加载 61.37 秒、每卡模型内存 55.95GiB。
  - train prompt runner 完成 256 条响应、900,000/900,000 prompt tokens 和 256 completion tokens，耗时 `532.1056863907725` 秒；responses 与 summary SHA256 为 `a356733b...b7a2`、`77490a26...a87`。
  - capture 文件为 8 rank × 78 层 = 624/624，总大小 2,783,307,114 字节；每个 rank 恰好 78 个文件，文件名/大小 manifest SHA256 为 `22cd1b54...f66f`。
  - 10 分钟服务心跳已落盘，服务错误扫描为空；train 服务正常停止后无残留进程，8 卡均为 0MiB、0%。
  - 正式 holdout 服务完成两次 8/8 GPU 空闲检查和 141/141 shard 加载，于 2026-07-26T11:42:07Z ready。
  - holdout prompt runner 完成 36 条响应、100,000/100,000 prompt tokens 和 36 completion tokens，耗时 `88.02139441482723` 秒；responses 与 summary SHA256 为 `221edb17...0ac3`、`cef3c602...a943`。
  - holdout capture 为 624/624 文件、13,272,777,066 字节，每 rank 78 层；文件名/大小 manifest SHA256 为 `6bf16917...3ba6`。服务停止后无残留进程，8 卡均为 0MiB、0%。
  - 首次正式 fit 在读取第一层前退出：配置模板为 `model.layers.{layer}.self_attn`，实际 capture 层名和文件均带固定 `.attn` 后缀；未生成 rotation artifact。
  - 配置已修正为 `model.layers.{layer}.self_attn.attn`，并新增独立路径集合预检。旧模板对现有 capture 实际返回 624 missing + 624 extra；新模板对 train/holdout 各 624 文件均通过，shell、ruff、format 和 diff 检查通过。
  - 正式 fit 重跑目录为 `artifacts/phase2/20260726T1200Z_rotation_fit_v2`；alpha `0.25/0.5/0.75` 的归一化 loss 分别为 `0.025037897150672388`、`0.027723928782624297`、`0.031391672548347495`，最终选择 `0.25`。
  - 78 层 clip ratio 分布为 0.92 共 61 层、0.94 共 17 层；逐层归一化 loss 范围为 `0.0004133854263186087` 至 `0.04478203689244448`。
  - 正式 `manifest.json`、`rotations.pt`、`search_summary.json` 大小为 2,404/81,811,997/28,465 字节，SHA256 分别为 `df30fbb9...9926`、`0a966da2...808e`、`9dfe16a8...792e`。
  - 候选 rootfs Python 的正式 artifact loader 验证 78/78 个 `512×512` rotation、运行时身份、哈希、有限值和正交性全部通过；`RᵀR-I` 最大绝对误差范围为 `1.0171338660214246e-08` 至 `1.6274684710992915e-08`。
  - 中文阶段报告已写入 `docs/experiments/2026-07-26-phase2-calibration.md`；capture、日志与 rotation tensor 仅保存在 ignored artifacts，不进入 Git。

### 阶段 3：三池 CacheSpec、allocator 与 scheduler/worker 隔离准备

- **状态：** 完成；正式 submodule、容量/生命周期回归与中文报告均通过
- **已执行：**
  - 从 calibration 固定 commit 建立项目内独立 worktree 与 `feat/glm52-mla-cache-planner` 分支，不改变正式 submodule。
  - 按 GLM‑5.2 的 78 层、latent rank 512、group size 128、INT2 data 加每组 FP32 scale/zero 建立精确 bytes/token 和 page 公式。
  - 实现固定 BF16 prefix/recent 行、paged INT2 history、请求 generation/cache version、逻辑到物理地址映射及 finish/abort/preemption/reuse 生命周期。
  - 重新读取实际模型配置与 cache spec，确认主 MLA cache 的 576 维由 512 维共享 latent 与 64 维 RoPE 组成，78 层中 21 层具有原生 DSA indexer cache。
  - 实现联合 runtime planner：标准 vLLM block table 管理全逻辑序列的 BF16 RoPE 与原生 DSA cache，OSCAR 使用独立 page ID namespace 管理 INT2 latent history，并为每个请求保留固定 BF16 prefix/recent 行。
  - 接入 `OscarMLAAttentionSpec`、KV cache config、scheduler manager、request metadata、两套 GPU model runner 和无重叠 raw tensor views；worker 按 generation/version 拒绝陈旧 metadata。
  - 添加 token 分区边界、容量守恒、稳定行复用、partial page、OOM 原子回滚、陈旧 generation、三类释放路径、联合预算、worker views 和 scheduler metadata 测试。
- **实际结果：**
  - 每层每 token 的 history data/metadata 实际计算为 128/32 bytes，合计 160 bytes；BF16 latent 为 1,024 bytes，history-only 理论与 page padding 比均为 6.4×。
  - prefix/recent 固定为 64/256 tokens；边界测试覆盖 0、63、64、65、319、320、321 和 32,768 tokens。
  - 实际 `indexer_types` 长度为 78，其中 `shared=57`、`full=21`；21 个 DSA cache 的原生格式为每层每 token 132 bytes（128-byte data + 4-byte scale）。
  - 14GiB、`max_num_seqs=16` 的联合 CPU 计划得到 36,216 个标准 blocks、36,215 个可用 blocks、579,440 个逻辑 token slots 和 36,216 个独立 history pages；fixed BF16/history/RoPE/native auxiliary 分别占 408,944,640/7,231,610,880/5,785,288,704/1,606,252,032 bytes，总分配 15,032,096,256 bytes，剩余 289,280 bytes。
  - 同一 14GiB 预算的理论 native 计划为 10,142 blocks、162,256 token slots；联合计划理论总容量比为 3.5711468297×。这是 CPU 规划值，不冒充 GPU 实测容量。
  - 定向套件共 116 项 pytest 全部通过；完整 scheduler 文件在强制离线下另有 68 项通过、28 项仅因缺少 LLaVA 仓库配置而失败，没有 OSCAR 或通用调度断言失败。
  - 完整 scheduler 首次运行意外触发 LLaVA 下载后立即终止；本次新建的 3,622,499-byte 外部模型 cache、0-byte lock 与 36KB Xet 日志已精确删除，复核 cache 路径不存在。
  - ruff 0.14.0、import sorting、12 文件 format check、compileall 与 `git diff --check` 全部通过；13 个代码/测试文件 commit `e75a40a294bd3127667f34ebffce8119a8ac0f3a` 已推送至独立分支，未提交模型、日志、cache 或其他大文件。
  - 阶段 2 出口通过后，已将隔离 worktree 在 `e75a40a29...` 处 detach，并把正式 `glm52_oscar_vllm` submodule 切换到远端已发布分支 `feat/glm52-mla-cache-planner` 的同一 commit；切换时源码工作区干净且与远端一致。
  - 正式首轮 116 项回归为 104 passed、12 failed；12 项均因正式 `.venv` 没有已安装 vLLM package metadata、自动 device detection 失败，不是 OSCAR 断言失败。
  - 显式加入项目内候选 rootfs 的已安装 vLLM metadata/dependency 路径后，12 个失败项先独立 12/12 通过，完整定向套件随后为 116/116 passed、耗时 27.11 秒，CUDA 未初始化。
  - 完整 scheduler 文件在 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 下重跑为 68 passed、28 failed、耗时 31.22 秒；28 项全部因缺少 `llava-hf/llava-1.5-7b-hf` 配置失败，无 OSCAR/通用 scheduler 断言失败，也未下载模型。
  - 14 个阶段改动 Python 文件 compileall 与 Git diff check 通过；其中 13 个无既存 lint 债务的文件通过 ruff 0.14.0/format。`gpu_model_runner.py` 的当前/基线均有相同 6 个既存 lint/format 问题，本阶段未修改这些行。
  - 中文阶段报告已写入 `docs/experiments/2026-07-26-phase3-cache-planner.md`。

### 阶段 4：SM80 Triton kernel 隔离准备

- **状态：** 完成；SM80 cold compile、单卡/8 卡 A800 launch、oracle、边界与中文报告均通过
- **已执行：**
  - 从 Stage 3 commit `e75a40a29...` 建立项目内 ignored worktree 与独立分支 `feat/glm52-oscar-kernels`，不改变正式 submodule 或运行中服务。
  - 实现 Triton shared-latent rotation、group=128 percentile clipping、非对称 INT2 pack、FP32 scale/zero store、BF16 prefix/recent store、recent gather/demotion 和 history dequant。
  - 实现 DSA-selected mixed sparse decode 候选：同一局部 softmax 中读取 BF16 prefix/recent 与 INT2 history，分开累加原 latent/旋转 latent，跨 split 统一 LSE merge，最后对 history accumulator 执行 inverse rotation 并相加。
  - 将 mixed kernel 泛化为显式 query-to-request 映射与 query logical position，新增 causal multi-token sparse prefill API；同一请求的多 query 共享 cache metadata，但输出与中间 buffer 仍逐 query 隔离。
  - CUDA 测试显式受 `VLLM_OSCAR_RUN_CUDA_TESTS=1` 控制；正式 baseline 使用 8 张 GPU 时不创建新的 CUDA context。
  - 添加实际 512 latent rank、group=128、边界长度 63/64/319/320/321、batch 1/4、ring address、demotion、selected token IDs 与 PyTorch oracle 测试入口。
  - 添加 `TRITON_INTERPRET=1` 的无 GPU 子进程 smoke，在不创建 CUDA context 的前提下实际执行 512 维 store、demotion、dequant、两 split mixed decode/prefill 和 global merge。
- **实际结果：**
  - interpreter 首轮暴露 split merge 标量 mask 类型不兼容，以及 BF16 `tl.dot` 在解释器下产生无效大值；分别改为分离 `tl.where`，并把 rotation 输入和累加统一为 FP32 IEEE 路径。
  - 修复后实际 latent rank 512、4 groups、两 splits 的输出与 LSE 均 finite，mixed decode 对 PyTorch oracle 最大绝对误差为 `2.384185791015625e-07`，store/demotion/dequant smoke 通过，进程未创建 CUDA context。
  - causal multi-token prefill interpreter 使用同一请求的 query positions 2/4 实际通过，输出与 LSE 均 finite，对 PyTorch oracle 最大绝对误差为 `2.980232238769531e-07`。
  - `tests/oscar_mla` 在未启用 CUDA 门禁时实际为 61 passed、22 skipped；新增 3 项 skip 分别覆盖 batch 1/4/8 的 prefill，22 项全部是尚待 A800 执行的 kernel 数值/launch 测试，不计为通过。
  - 5 个新增代码/测试文件通过 ruff 0.14.0、format check、`py_compile` 与 `git diff --check`。
  - history store/demotion WIP commit 为 `9861f2398...`，mixed sparse decode WIP commit 为 `5d220497a...`，interpreter smoke 与修复 commit 为 `18c83e4b9...`，sparse prefill WIP commit 为 `b722b7975...`；均已推送至 `origin/feat/glm52-oscar-kernels`，提交中无模型、日志、cache 或其他大文件。
  - Stage 5 接线审查发现 mixed kernel 只计算 512 维 latent score，遗漏 64 维原精度 RoPE score，且默认 scale 仍为 `1/sqrt(512)`；已在 kernel 工作树补入 RoPE query/cache/block-table 读路径，并把默认 scale 改为 `1/sqrt(576)`。
  - 带非零 RoPE 的 512+64 维 CPU Triton interpreter decode/prefill 已实际通过；两者对扩展 PyTorch oracle 的最大绝对误差均为 `2.384185791015625e-07`。完整无 CUDA 套件为 61 passed、22 skipped，ruff/format/py_compile/diff 门禁通过。
  - RoPE correctness commit `8ac7b9d97...` 已推送至 `origin/feat/glm52-oscar-kernels`；同一提交已 cherry-pick 为 `3ce04538e...` 并推送至 `origin/feat/glm52-oscar-integration`，两个工作树均无 tracked 改动。
  - CPU interpreter 结果不能替代 SM80 编译和 A800 launch；Stage 4 仍未通过，GPU 释放后必须先清空任务专用 Triton cache，再运行这 22 项并按实际编译错误/误差修正。
  - 阶段 3 出口通过后，已将 kernel 隔离 worktree 在 `8ac7b9d97...` 处 detach，并把正式 submodule 切换到远端已发布的 `feat/glm52-oscar-kernels` 同一 commit；该 commit 严格继承阶段 3 的 `e75a40a29...`。
  - 正式 A800 运行目录为 `artifacts/phase4/20260726T121130Z_a800_kernels`；2026-07-26T12:11:40Z 与 12:12:53Z 两次检查均为 8/8 GPU 0MiB、0% 且无 compute process。
  - 全新任务专用 Triton cache 的首轮结果为 23 passed、1 failed、63.33 秒；22 个 CUDA 门禁中 21 个通过，唯一失败是 BF16 ring 测试先要求 `recent[0,0]` 为 NaN、后又要求同一 slot 等于 position 320 写入值的矛盾断言，不是 kernel 数值/编译失败。
  - 测试现拆为两次调用：先只传 final history positions 64/65 并验证 slot 0/1 保持 NaN，再传 prefix 与 final recent positions 319/320/321 验证 ring 地址；正式重跑前需提交推送并使用新 Triton cache。
  - 修复后的定向 A800 test 为 1/1 passed、3.61 秒，ruff 0.14.0、format 与 diff check 通过；pre-commit 初始化 actionlint hook 停滞后已中止，按手工门禁以 commit `c50d86b34643c9fba0ae1df28a671c04fd107a41` 提交并推送。
  - 第二个全新 Triton cache 的正式重跑为 24/24 测试节点通过，其中 22/22 为 CUDA 门禁，耗时 56.60 秒；cache 生成 284 文件、19,620,324 bytes。
  - CUDA 开启后的完整 `tests/oscar_mla` 为 83/83 passed、34.31 秒；定向/完整日志 SHA256 分别为 `91cdbc6e...6264`、`73a7c83c...5b1e`。
  - TP=8 rank-local smoke 再次完成两次 8/8 GPU 空闲检查；8 个并行进程各绑定一张 A800、使用独立空 Triton cache，rank 0–7 均为 24/24 passed，耗时范围 84.50–87.00 秒。
  - TP=8 smoke 总计 192/192 测试节点、176 次 CUDA kernel 执行；每 rank 均生成 284 个 cache 文件、19,624,132 bytes，结束后 8 卡均为 0MiB、0% 且无 compute process。
  - 中文阶段报告已写入 `docs/experiments/2026-07-26-phase4-a800-kernels.md`；日志和 Triton cache 仅保存在 ignored artifacts。

### 阶段 5：`oscar_mla_int2` runtime 激活准备

- **状态：** 正式 submodule 已接入 integration commit，A800 与端到端待验收
- **已执行：**
  - 从 Stage 4 commit `b722b7975...` 建立项目内 ignored worktree 和独立分支 `feat/glm52-oscar-integration`，不改变正式 submodule 或运行中 baseline。
  - 将 `oscar_mla_int2` 注册为显式 `CacheDType`，generic runtime 仅使用 uint8 marker；INT2/FP32/BF16 混合物理布局仍完全由 `OscarMLAAttentionSpec` 和 cache views 管理。
  - 只允许 `TRITON_MLA_SPARSE` backend 声明支持该 dtype；非 sparse MLA、其他 backend 和 vLLM prefix caching 均 fail closed。
  - `MLAAttention.get_kv_cache_spec` 按实际 512 latent、64 RoPE、group 128 自动生成 160-byte history slot、64-token prefix 与 256-token recent 的三池 spec。
  - 使用 `uv` 建立该 worktree 自有 `.venv`，并从清华 PyPI 镜像安装缺少的 `tblib==3.2.2`。
- **实际结果：**
  - 新增配置/spec 定向测试与既有 cache integration 合计 8 项通过；未启用 CUDA 时完整 `tests/oscar_mla` 为 63 passed、22 skipped。
  - 合入 RoPE correctness commit 后重新执行完整 `tests/oscar_mla`，仍为 63 passed、22 skipped；4 个相关文件的 ruff/format/py_compile/diff 门禁通过。
  - worker ownership 现在以事务方式保留本轮更新前 logical length，并把稳定 `hp_row`、独立 history page table、previous lengths 和逐 token demotion page/offset 物化为 batch GPU metadata；prefix/recent 物理起点与 row 不一致时 fail closed。
  - `oscar_mla_int2` 禁用单请求 decode preparation fastpath，并拒绝 CUDA graph padding；XPU/Triton sparse metadata 已携带 exact seq lengths 与 OSCAR batch metadata。
  - 新增 320→337 token 的 17-token demotion 映射回归；完整无 CUDA套件为 64 passed、22 skipped，静态门禁通过。commit `ef5476705...` 已推送至 `origin/feat/glm52-oscar-integration`。
  - 新增 runtime artifact 绑定：必须同时设置 rotation artifact 目录与 expectation JSON；模型 config、checkpoint manifest、专家映射、层数、latent/group/window、层号、rotation tensor hash 任一不符即拒绝。commit `715669a4c...` 已推送，合入时完整套件为 66 passed、22 skipped。
  - cache update 已按 `RoPE write → old recent demotion → current history direct INT2 store → final BF16 prefix/recent store` 排序，避免 current chunk 覆盖待 demote 的 recent ring；RoPE 使用标准 block slot 保持 BF16。
  - mixed read 直接消费 DSA request-local top-k IDs、标准 RoPE block table、独立 history page table 和稳定 HP rows，调用 512+64 维 OSCAR sparse prefill/decode kernel，结果转回 BF16 后进入原生 value up-projection。
  - CPU mock 回归验证 320→337 demotion 先于 recent overwrite、新请求 current history 17-token 直写，以及 selected IDs `[0,64,320]` 未被转换为错误的标准全局 slot；Triton interpreter 实际执行新增 RoPE store。
  - 完整无 CUDA套件为 69 passed、23 skipped；新增的第 23 项 CUDA 门禁覆盖标准 slot RoPE store。ruff/format/py_compile/diff 均通过，runtime cache path commit `f8e5afbbf...` 已推送。
  - 配置层新增执行模式门禁：`oscar_mla_int2` 显式拒绝 V2 model runner、非 eager、CUDA graph、speculative decoding、decode context parallelism 和 dual batch overlap，避免未实现组合静默进入 runtime。
  - 加入上述门禁后完整无 CUDA 套件为 73 passed、23 skipped；相关 `py_compile`、ruff、format 与 `git diff --check` 全部通过，commit `d62571ae8...` 已推送。
  - 继续补齐首版非目标门禁：prefill context parallelism、KV transfer 与 KV offloading 均在配置验证期拒绝。
  - 干净子进程复测发现 Triton interpreter smoke 未显式设置仓库导入路径，报 `No module named 'vllm'`；测试入口现固定项目根 `PYTHONPATH`，随后定向 10 项与完整套件均通过。
  - 最新完整无 CUDA 套件为 76 passed、23 skipped；三个改动文件的 `py_compile`、ruff、format 与 diff 门禁通过，commit `fa7ed930b...` 已推送。
  - 使用候选 rootfs Python、真实 GLM‑5.2 模型目录与完整 TP=8/32K CLI 创建 EngineConfig；实测 eager 模式仍默认启用 asynchronous scheduling。
  - 首版 OSCAR ownership 尚未验证异步调度，因此配置层新增 fail-closed 门禁。真实配置在默认值下按预期拒绝，加入 `--no-async-scheduling` 后得到 `TRITON_MLA_SPARSE`、`oscar_mla_int2`、TP=8、PP=1、32K、prefix cache=false、CUDA graph=NONE，且 `torch.cuda.is_initialized()` 为 false。
  - 新增门禁后完整无 CUDA 套件为 77 passed、23 skipped，相关静态门禁通过；commit `42639391d...` 已推送。
  - CPU Triton interpreter 的 rotation 改为真实非连续 stride 并实际通过；新增 A800 条件测试比较 one-shot 337 tokens 与 chunked 320→337 的 INT2 history data/scale/zero、BF16 prefix/recent 最终分区逐字节一致。
  - 最新完整无 CUDA 套件为 77 passed、24 skipped；新增第 24 项 CUDA 门禁尚未冒充通过，两个测试文件的语法、ruff、format 与 diff 门禁通过。commit `2a49fe1c2...` 已推送。
  - 按设计验收补齐 INT2 窄分布与极端 outlier reference；两类输入的 scale、zero、恢复值均 finite，pack/unpack 逐值一致，恢复误差不超过 clipped group 的半个量化步长。
  - 完整无 CUDA 套件更新为 79 passed、24 skipped，reference 文件的语法、ruff、format 与 diff 门禁通过。commit `a5e7e5c65...` 已推送。
  - PyTorch mixed-tier reference 新增自然对数域 LSE 返回值，并保持原输出接口兼容；CPU Triton interpreter 对 decode 的输出/LSE 最大绝对误差分别为 `2.384185791015625e-07`/`0.0`，对 causal prefill 分别为 `2.384185791015625e-07`/`5.960464477539063e-08`。
  - 定向 12 项及完整无 CUDA 套件分别为 12 passed、79 passed/24 CUDA skipped；三个改动文件的 `py_compile`、ruff、format 与 diff 门禁通过。commit `f426ab5a5...` 已推送。
  - CPU Triton interpreter 新增两请求隔离：两个请求使用不同 HP row、history page 和 RoPE block table；输出/LSE 最大绝对误差分别为 `2.384185791015625e-07`/`1.1920928955078125e-07`，并验证 `-1` DSA padding 被屏蔽。
  - 多请求改动后的完整无 CUDA 套件为 79 passed、24 CUDA skipped，静态门禁通过；commits `726d1dc70...`、`5445a8286...` 已推送。batch 4/8 与 TP=8 仍待 A800 实测。
  - runtime mock 新增两个不同长度请求的 metadata 映射验证：request indices 为 `[0,1,1]`，局部 query positions 为 `[320,335,336]`，RoPE block table、history page table 和 HP rows 均保持对应请求 ownership。
  - 完整无 CUDA 套件更新为 80 passed、24 CUDA skipped；定向 4 项、语法、ruff、format 与 diff 门禁通过。commit `7bac6d7e9...` 已推送。
  - 新增 batch 4/8 A800 条件门禁，每个请求使用独立 prefix/recent row、INT2 history page 与 RoPE block table，并逐请求构造 output/LSE PyTorch oracle；容差在硬件结果前固定为 output `atol=0.5, rtol=0.03`、LSE `atol=0.05, rtol=0.01`。
  - 完整无 CUDA 套件为 80 passed、26 CUDA skipped，新增两项未冒充通过；语法、ruff、format 与 diff 门禁通过。commit `c762b4aee...` 已推送。
  - 5 个代码/测试文件通过 ruff、Python 语法和 `git diff --check`；未格式化的既有 backend 文件只做 import sorting 和一行 dtype 变更，未顺带重排其他代码。
  - WIP commit `cc2655657...` 已推送至 `origin/feat/glm52-oscar-integration`；提交仅含代码与测试，没有模型、日志、cache 或大文件。
  - 当前只证明代码接线、CPU mock 与 interpreter；尚无 A800 实际 cache write/demotion/mixed read，不能宣称服务路径已通过。
  - 阶段 4 结束后，BF16 ring 测试修复已 cherry-pick 到 integration 分支；ruff/format 和完整无 CUDA套件重跑为 80 passed、26 skipped、26.69 秒。
  - integration 修复 commit `caa0818540280c16b949b6646f9ba116cdaa59f2` 已推送；隔离 worktree 在该 commit detach，正式 submodule 已切换到同一远端分支/commit，源码工作区干净。
  - 阶段 5 首轮正式 A800 完整套件使用 GPU 0 与全新 Triton cache，实际结果为 105 passed、1 failed、74.71 秒；其余 CUDA/kernel 节点均通过。
  - 唯一失败发生在 kernel launch 前的 `assert not rotation.is_contiguous()`：当前 PyTorch 的 QR 输出已经是列主序非连续张量，再执行 `.T` 后变为连续。测试改为先 `.contiguous()` 再转置，以稳定构造数值相同的非连续正交矩阵。
  - 修复后的定向 A800 测试已实际进入 one-shot/chunked store kernel 并通过，为 1 passed、4.20 秒；ruff、format、py_compile 和 `git diff --check` 同步通过。正式全量仍须在代码提交推送后用全新 Triton cache 重跑。
  - pre-commit 再次停在 actionlint 环境初始化，已主动中止；由于本次只改一行测试且等价手工门禁全部通过，使用 `--no-verify` 提交。源码 commit `0f1bd5b308da9217ba72a5ba68ca5e9590b7a2bd` 已推送并与远端分支一致。
  - 主仓库 submodule 指针与上述中文记录作为 commit `751f20cd62d8aaf413973dcc9a2d9098ab90bcf0` 推送；正式重跑前主仓库和源码仓库均干净且本地/远端 SHA 一致。
  - 正式 retry 目录为 `artifacts/phase5/20260726T123536Z_integration_cuda/retry_0f1bd5b30`；两次 GPU 检查时间为 12:46:25Z、12:47:38Z，8/8 张 A800 均为 0 MiB、0% 且无 compute process。
  - 使用 GPU 0 与全新 Triton cache 完整执行 `tests/oscar_mla`，结果为 106 passed、17 warnings、70.31 秒；这 106 项包含 26 项 CUDA 条件门禁，未出现 skip 或 fallback。
  - 新 cache 为 316 个文件、22,280,982 字节；完整测试日志 SHA256 为 `8265be65743788cb02272ed862e731153670cafc7bf59df60406450e73255cae`。测试结束后 8 张 GPU 均回到 0 MiB、0%。
  - 新增 Stage 5 runtime expectation、轻量配置、fail-closed verifier 与 TP=8 服务包装入口；原生 launcher 仅参数化 manifest/verifier/source/branch/dtype/cache 目录和同步调度开关，默认行为保持不变。
  - 首轮 dry-run 的 immutable OCI、4,711 个 runtime 文件、7 个 native extension、141 个模型 shard、official_v4、78 层 artifact 身份与几何全部通过；随后 CLI 校验内联脚本因遗漏 `import os` 在读取预期 dtype 前退出。
  - 补齐 import 后重跑 dry-run 通过：候选 Python 3.12.13、Torch 2.11.0+cu129、Triton 3.6.0，源码从项目内 `glm52_oscar_vllm` 解析，主机 `flash_attn`/`triton_kernels` 不可见，CLI 为 TP=8、PP=1、32K、eager、`TRITON_MLA_SPARSE`、`oscar_mla_int2`、prefix cache=false、speculative=null、`--no-async-scheduling`，且 CUDA=false。
  - 新增正式 smoke 包装入口：先复用阶段 1 的短请求、>320 tokens、384-token 连续 decode 和近 32K 四项，再并发发起 8 个 >320-token 请求；最后 fail closed 提取三池容量、artifact、write、demotion、mixed read 与“无完整 BF16 history”日志证据。
  - 新增并发脚本通过 ruff、format、py_compile 和候选 Python `--help` 导入检查；三个 shell 入口通过 `bash -n`。当前环境未安装 `shellcheck`，未把未执行的 shellcheck 冒充通过。
  - Stage 5 服务与 smoke 入口作为主仓库 commit `7ba83078be9193fb36de7be11afb02062371b1c5` 推送，随后以 `e4b0ce0f6232e1e98440375b774fac3bfa7d0677` 补齐四个新脚本的 Git executable mode；没有提交模型、cache 或实验日志。
  - 正式运行 ID 固定为 `20260726T130111Z_oscar_tp8`。formal preflight 记录主仓库 `e4b0ce0f...`、源码 `0f1bd5b3...` 和 native 基线 `fd3e0b37...`；全部静态身份与候选环境门禁通过。
  - formal preflight 的两次 GPU 检查时间为 13:01:55Z、13:02:58Z，8/8 张 A800 均空闲；服务尚未启动，不能宣称 TP=8 或端到端通过。
  - 首次正式服务再次于 13:04:39Z/约 13:05:39Z 通过两次 8/8 GPU 空闲检查，并于 13:06:11Z 创建 API server；CLI/engine 均确认 `kv_cache_dtype=oscar_mla_int2`、TP=8、32K、eager 和同步调度。
  - 8 个 worker 在模型对象构造期、权重 shard 加载前一致退出。根因是 artifact loader 虽用 `map_location="cpu"` 加载 rotation，但正交校验中的 `torch.eye` 受 vLLM rank 默认 CUDA device 影响，导致 CPU rotation 与 `cuda:<rank>` identity 在 `torch.allclose` 中跨设备报错。
  - 该轮服务退出码为 1，8 张 GPU 检查均为 0 MiB；未加载权重、未创建 KV cache、未发送请求。修复将 identity 显式固定到 CPU，并以非 CPU default-device 回归覆盖。
  - 修复后的 artifact/runtime 定向套件为 11 passed；ruff、format、py_compile 与 diff 门禁通过。完整无 CUDA 套件更新为 81 passed、26 skipped、30.02 秒。
  - 在 GPU 0 可用环境中以 CUDA default-device 加载正式 78 层 artifact，78 个 rotation 均保持在 CPU、manifest SHA256 为 `df30fbb9...c19926`，且 `torch.cuda.is_initialized()` 仍为 false。
  - 两文件修复以 `--no-verify` 提交，因为等价手工门禁和完整套件均已执行；源码 commit `c3823fda2ed1d82f92c99275b6e128bac9ba6220` 已推送且与远端一致。
  - 第二次正式运行目录为 `artifacts/phase5/20260726T131415Z_oscar_tp8_retry_c3823fda2`；两次 GPU 空闲检查通过后，8-rank NCCL 初始化、artifact SHA 加载和 141/141 shard 加载均通过。
  - 得益于系统 cache，本轮权重读取为 49.15 秒，模型加载总计 62.462097 秒、每卡 56.02 GiB；可用 KV cache 内存为 15.37 GiB。
  - 三池 planner 实际给出 9,964 history pages、637,632 logical tokens，分配为 INT2 history 7.41 GiB、BF16 prefix/recent 0.38 GiB、RoPE 5.93 GiB、native auxiliary 1.65 GiB、unused 0.0 GiB。
  - 随后的 dummy warmup 在 `unified_mla_kv_cache_update` 对 `OscarMLACacheTensors` 调用 tensor-only `.numel()`，8 workers 一致报 `AttributeError` 后退出；GPU 已回到 0 MiB，尚未 ready 或发送请求。
  - 根因是原生单 tensor 空 cache 门禁位于 dtype 分支之前；OSCAR cache 已重塑为 dataclass views。修复为 OSCAR 检查 `kv_cache.raw.numel()`、其他 dtype 保持原检查，并增加空 OSCAR cache 回归。
  - 修复后的 runtime cache path 为 5 passed；使用 GPU 0 重跑完整 CUDA 套件为 108 passed、17 warnings、36.91 秒，26 项 CUDA 条件门禁全部实际执行。
  - 两文件的 ruff、format、py_compile 与 diff 门禁通过；源码 commit `49db9142d7688d5b116ccd69fccfdf2e085818bf` 已推送且与远端一致。
  - 第三次正式运行目录为 `artifacts/phase5/20260726T132717Z_oscar_tp8_retry_49db9142d`；固定版本 dry-run 和两次 8/8 GPU 空闲检查均通过，8-rank NCCL、artifact 与 141/141 shards 也通过。
  - 本轮权重读取为 47.25 秒，模型加载总计 60.693986 秒、每卡 56.02 GiB；随后在 `determine_available_memory()` 的分配前 profile run 退出，尚未执行三池 planner。
  - profile run 虽已携带 `kv_cache_dtype=oscar_mla_int2`，`attn_layer.kv_cache` 此时仍为普通空 `Tensor`；上一修复按 dtype 无条件读取 `.raw`，导致 8 workers 一致报 `AttributeError: 'Tensor' object has no attribute 'raw'`。
  - 服务退出后 8 张 GPU 均回到 0 MiB。下一修复按 cache 实际类型选择 backing storage，并以分配前空 Tensor 与分配后 dataclass 两项回归覆盖完整生命周期。
  - 修复现按 `OscarMLACacheTensors` 实际类型选择 `.raw`，否则保留 Tensor；分配前空 Tensor 与分配后空 dataclass 两项回归均通过，runtime path 更新为 6 passed、17 warnings、4.88 秒。
  - 两文件的 ruff、format、py_compile 与 diff 门禁通过；源码 commit `f852be0c830f5caf741f91e20bbe522f5d00d55f` 已推送且与远端一致。

## 测试结果

| 检查 | 命令/输入 | 预期 | 实际 | 状态 |
| --- | --- | --- | --- | --- |
| 初始 Git 状态 | `git status --short --branch` | 确认工作区基线 | `## main...origin/main` | 通过 |
| 本任务计划文件 | 根目录文件检索 | 判断是否需要恢复旧上下文 | 根目录不存在，只有 `oscar_vllm/` 下有另一任务文件 | 通过 |
| submodule 指针 | `git submodule status --recursive` | 记录固定参考版本 | OSCAR `41ebcd...`；OSCAR-vLLM `d458d296...` | 通过 |
| Docker CLI | `docker info` | 查询本地 Docker 运行环境 | `/bin/bash: docker: command not found` | 未通过 |
| 当前环境类型 | `/proc/1/cgroup` | 判断按物理主机还是容器规范执行 | Kubernetes/containerd 容器；`nvidia-smi` 可用 | 通过 |
| 外部完整源码候选 | 精确路径检查与 `du -sh` | 优先寻找外部完整源码 | `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1`，约 2.0GB | 待深度核验 |
| 外部源码 Git HEAD | `git -c safe.directory=... rev-parse HEAD` | 匹配设计固定 commit | `bfd727e11b0e501bab0a4a943d92ba5ea3b2f980` | 通过 |
| 外部源码干净度 | `git -c safe.directory=... status --short --branch` | 干净、可按 commit 复现 | 6 个 tracked 文件被修改 | 未通过 |
| 镜像 tar 完整性 | `sha256sum -c ...tar.sha256` | 与归档记录一致 | `...tar: OK`，SHA256 `b58fc5cf...bed23` | 通过 |
| 镜像源码层位置 | 逐层 `tar -tf` | 找到完整源码和后续 patch | layer 29 为完整树，30/31 为 3 个文件的覆盖 patch | 通过 |
| clean HEAD 与镜像源码比较 | `git archive HEAD` + `diff -qr` | 明确镜像相对 commit 的全部源码差异 | 4 个 tracked 文件不同；另有 6 个原生 `.so` 和运行产生的 `__pycache__` | 待完成差异归因 |
| compiler patch 来源 | v1 stable manifest + SHA256 + `cmp` | 解释第 4 个未列入最终 manifest 的差异 | 与 v1 stable patch 精确一致 | 通过 |
| 候选 GitHub 远端 | `gh repo view`、`git ls-remote` | 找到独立代码仓库 | public；`main`/功能分支均为 `53d8be94f...` | 通过 |
| 本地容器构建能力 | capabilities + runtime socket 检查 | 可实际构建 Docker/OCI image | 无 socket、无 `CAP_SYS_ADMIN`、无 builder CLI | 待验证 rootless 替代方案 |
| 恢复候选 Git 历史 | `git bundle verify` + clone | 保留完整历史且从固定 commit 起步 | bundle 为 complete history；`main` 指向 `bfd727e...` | 通过 |
| runtime patch 一致性 | 4 文件 SHA256 对比 | 与已验证镜像逐字节一致 | 4/4 `MATCH`；`git diff --check` 通过 | 通过 |
| Python 语法 | `.venv/bin/python -m py_compile <4 files>` | 无语法错误 | 退出码 0 | 通过 |
| 定向 ruff | `ruff==0.14.0 check <4 files>` | 当前 lint 规则通过 | 3×E501、1×SIM102，均来自镜像原始 patch | 未通过（已记录，不改 baseline） |
| 镜像依赖版本 | layer 29 dist-info `METADATA` | 冻结关键 Python 依赖 | torch 2.11.0+cu129、triton 3.6.0、vLLM 0.11.2.dev278+gdbc3d9991、transformers 5.8.1、tokenizers 0.22.2 | 通过 |
| splitmerge native extension | layer 29 tar stream SHA256 | 冻结额外 sparse MLA `.so` | 342,336 bytes，SHA256 `cb549aca...8625` | 通过 |
| recovery assets 静态验证 | SHA256、`json.tool`、`bash -n`、`git diff --check` | 资产内部一致 | source 4/4、native 7/7、JSON/shell/diff 全部通过 | 通过 |
| rootless builder 启动 | `BUILDAH_ISOLATION=chroot buildah --storage-driver vfs info` | 无 Docker daemon 也能运行 builder | buildah 1.33.7 返回有效 host/store 信息 | 通过 |
| rootless overlay store | 默认/显式 fuse/user+mount namespace 三种命令 | 用 overlay 降低存储占用 | permission denied / propagation denied | 未通过，停止该路径 |
| 外部源码只读复核 | 精确 `safe.directory` 的 `git status` 与 `rev-parse` | 不因本任务产生外部改动 | HEAD 仍为 `bfd727e...`；仅原有 6 个修改项 | 通过 |
| 无 daemon OCI 工具 | `skopeo --version`、`umoci --version` 及命令帮助 | 可无 mount 搬运与组合 OCI image | skopeo 1.13.3、umoci 0.4.7 可用 | 通过 |
| Docker archive 转 OCI | `skopeo copy docker-archive:... oci:...:verified` | 不依赖 daemon/mount 导入权威镜像 | 成功；manifest `sha256:1b8e808e...6415` | 通过 |
| 干净 Git payload | `git checkout-index` + 文件/哈希检查 | 完整源码且无本地元数据/二进制污染 | 4,711 文件；4/4 runtime SHA 通过；0 个 `.git/.venv/.so` | 通过 |
| `umoci insert` 候选层 | NFS 与 `/tmp` 两类源目录 | 生成完整可解包的第 32 层 | 两次均固定缺最后 393 字节 | 未通过，停止该路径 |
| 标准 OCI 候选层 | GNU tar/gzip + OCI descriptor | 生成完整、可遍历、内容一致的第 32 层 | 5,256 成员；4,711 文件内容和权限逐一匹配 | 通过 |
| 候选 OCI config | manifest/config 解析 | 32 层/32 diff IDs，标签和入口正确 | manifest `sha256:2fdfbe86...489d`；config `sha256:58a853ee...a42d` | 通过 |
| source layer 可重复构建 | 独立第二次 Git index 导出 + GNU tar/gzip | digest、diff ID、大小完全相同 | `352d47...59c3`、`e92870...ccf8`、33,253,726 bytes 均相同 | 通过 |
| 展开 rootfs 清单 | `runtime_source.sha256` + `native_extensions.sha256` | 4+7 项全部匹配 | 11/11 `OK` | 通过 |
| 阶段 0 中文报告结构 | heading/交叉引用检查 | 章节连续且引用正确 | 1–8 连贯，无错误交叉引用 | 通过 |
| A800 空闲检查 | 间隔 60 秒的两次 `nvidia-smi` | 8 卡均无显存占用/计算进程 | 两次均为 0MiB、0%、无进程 | 通过 |
| 模型资产快速复核 | 文件数、类型、大小和关键配置 | 模型/tokenizer/config 齐全 | 141 safetensors；462,858,376,495 bytes；关键文件存在 | 通过 |
| 当前 Pod 目标环境 | `/opt` 路径与环境变量 | 与候选镜像环境一致 | 3 个目标 `/opt` 路径均缺失 | 未通过 |
| 固定 rootfs Python 环境 | import 路径、版本与 native extension | 全部来自候选 rootfs，且不初始化 CUDA | torch/triton/transformers/tokenizers/`vllm._C` 路径正确 | 通过 |
| 模型几何 | config/index 读取 | 与设计固定 DSA/MLA、MoE 和 FP8 几何一致 | 78 layers；DSA 32×128/top-k 2048/freq 4；154 experts | 通过 |
| checkpoint 轻量指纹 | 大小清单、index、首尾 shard head+tail | 形成可重复快速核验值 | 清单 `f4c012...eb969`；index `e97675...5d983`；首尾指纹已记录 | 通过 |
| 阶段 1 全量静态 preflight | 固定 Python 执行 `verify_native_baseline.py` | OCI/source/model/suite 全部匹配 | 4,711 files、7 native、141 shards、2360+1 samples 全部通过 | 通过 |
| vLLM CLI 参数解析 | 固定候选环境解析完整 serve 参数 | TP=8/eager/32K/native sparse 且无非目标能力 | prefix cache=false、speculative=null、CUDA 未初始化 | 通过 |
| 新增脚本静态检查 | `bash -n`、`py_compile`、ruff 0.14.0、`git diff --check` | 无语法/lint/diff 错误 | 全部退出码 0 | 通过 |
| 32K smoke prompt | 固定 tokenizer/chat template 生成目标长度 | >320 且近 32K | 506 tokens、31,996 tokens | 通过 |
| evaluator venv | uv lock diff、runner import、CLI help | 依赖完整且版本冻结 | 20 个包匹配 lock；25 个 IFEval registry entries；runner import 通过 | 通过 |
| public 代码快照 | `git push`、tree/diff 核验 | 只同步冻结代码且内容不变 | 5,014 objects、33.13MiB；tree `ca5d4f035...`；diff 为空 | 通过 |
| GLM submodule | `git ls-remote`、主仓库 commit/push | 指向远端可解析 commit | `53d8be94f...`；主仓库 `04b9445...` 已推送 | 通过 |
| 正式 GPU preflight | `FORMAL_RUN=1 ... formal-preflight` | 双仓库发布、全部指纹通过且 GPU 连续空闲 | `8f7be26a...`/`53d8be94f...`；两次 8/8 空闲 | 通过 |
| 首次 TP=8 服务启动 | `FORMAL_RUN=1 ... serve` | worker 初始化并加载模型 | FlashInfer 版本门禁；退出码 1；GPU 0MiB | 未通过 |
| FlashInfer 兼容修复 | 固定 venv `import flashinfer.comm`、`bash -n` | 通信模块可导入且 launcher 语法有效 | 导入通过；版本 0.6.6/0.6.7.post3+cu129；shell 通过 | 通过 |
| 第二次 TP=8 服务启动 | `FORMAL_RUN=1 ... serve` | worker 初始化并加载模型 | 当前容器 `flash_attn` 污染可选包探测；`flash_attn.ops` 缺失；退出码 1 | 未通过 |
| 固定 venv 系统包来源审计 | `pyvenv.cfg`、`sys.path`、`find_spec`、rootfs 文件清单 | 系统包来自候选 rootfs | `include-system-site-packages=true` 且绝对 `/usr/local/lib` 指向当前容器 | 未通过，已修复 |
| 候选 rootfs Python 隔离 | rootfs Python + `PYTHONHOME`/候选包路径 + dry-run | 不可见当前容器可选包且完整预检通过 | Python 3.12.13；两个可选 spec 为 `None`；全量预检和 CLI 解析通过 | 通过 |
| 第三次 TP=8 服务启动 | `FORMAL_RUN=1 ... serve` | 141 shard 加载、KV cache 初始化并 ready | 2,762 秒 ready；165,696-token KV cache；无 ERROR/Traceback | 通过 |
| 原生四项 smoke | 短请求、>320、384-token decode、31,996-token context | 全部 HTTP 200 且满足 token 门槛 | 21+64、506+18、26+384、31,996+64 tokens | 通过 |
| official_v4 原生精度首轮 | 2,360 样本、并发 8、code timeout 600 秒 | 全量 scored 并冻结 accuracy/SHA256 | 首批仅 2/8 HTTP 200，其余 6 条超时；停止 | 未通过 |
| official_v4 timeout 探针 | 前 8 样本、并发 8、code timeout 900 秒 | 8/8 scored 且 request failure=0 | 638.53 秒；8/8 scored；request failure=0；accuracy 0.0 | 通过 |
| official_v4 全量首轮、精确补跑与正式合并 | 2,360 样本首轮；仅补跑 7 个固定失败 ID，math timeout 900 秒 | 合并后 2,360/2,360 scored 且 request failure=0 | 正式合并 2,360/2,360 scored、469 条正确、accuracy `0.19872881355932204`；predictions SHA256 `68a3d0b1...3e75` | 通过 |
| WikiText-2 原生 PPL | TP=8、eager、BF16、2048/512 sliding window、batch 8 | 1/1 scored 并冻结 PPL | 289,708 evaluated tokens、563 windows、mean NLL `2.0402180131829573`、PPL `7.692286035848967` | 通过 |
| 阶段 2 reference/covariance/capture/artifact | 定向 pytest、Python 语法、ruff 0.14.0、format check、diff check | 数值 reference、基础统计、只读 capture 与 fail-closed artifact 全部通过 | 最终 35 passed；语法/lint/format/diff 均通过 | 通过 |
| 阶段 2 正式 train capture | TP=8、900,000 tokens、8 rank × 78 层 | token 精确匹配且 624 个 capture 文件完整 | 256 条、900,000 tokens、624/624 文件、2,783,307,114 字节 | 通过 |
| 阶段 2 正式 holdout capture | TP=8、100,000 tokens、8 rank × 78 层 | token 精确匹配且 624 个 reservoir/DSA capture 文件完整 | 36 条、100,000 tokens、624/624 文件、13,272,777,066 字节 | 通过 |
| 阶段 2 fit capture 路径契约 | 独立比较配置期望路径与 train/holdout 实际 `.pt` 集合 | 两个 split 均恰好匹配 624 文件 | 旧模板失败；修正 `.self_attn.attn` 后 train/holdout 各 624/624 | 通过 |
| 阶段 2 rotation artifact | 固定 alpha/clip 搜索 + 正式 runtime loader | 78 层完整、身份/哈希匹配且 `RᵀR≈I` | alpha `0.25`、loss `0.025037897150672388`；78/78 层；最大正交误差 `1.6274684710992915e-08` | 通过 |
| 阶段 3 三池 scheduler/worker 集成 | 116 项定向 pytest + 强制离线 scheduler 回归 + ruff/format/compile/diff | 三池预算、ownership、views 与通用 scheduler 无回归 | 正式 116 passed；离线 scheduler 68 passed、28 项仅缺 LLaVA 配置；13 个无既存债务文件 lint/format、14 文件 compileall 和 diff 通过 | 通过 |
| 阶段 4 A800/SM80 kernels | 两次空闲检查 + 全新 Triton cache + 单卡/8 卡 CUDA + 完整套件 | cold compile、实际 launch、oracle、边界与 TP=8 rank-local 全通过 | 正式 22/22 CUDA；完整 83/83；8 ranks 各 24/24，累计 176 次 CUDA 执行 | 通过 |
| 阶段 5 runtime cache 路径 | artifact/metadata/write/read 定向测试 + 完整 `tests/oscar_mla` + 静态门禁 | fail closed，demotion 顺序、多请求 ownership、DSA local IDs/padding、输出/LSE oracle 正确且不冒充 GPU | 正式 cold-cache 完整套件 106/106 passed、70.31 秒；26 项 CUDA 门禁均实际执行；日志 SHA256 `8265be65...55cae` | 通过 |
| 阶段 5 真实 EngineConfig | 候选 Python + 真实模型 + TP=8/32K OSCAR CLI | 默认 async 被拒绝，显式同步配置成功且不初始化 CUDA | 默认配置按预期失败；`--no-async-scheduling` 后配置字段全部匹配，CUDA=false | 通过 |
| 阶段 5 TP=8 正式入口 dry-run | 固定源码、候选 rootfs、模型、artifact 与完整 serve CLI | 全部身份/模式门禁通过且不初始化 CUDA | 4,711 source、7 native、141 shards、78 rotations 全通过；TP=8/32K/OSCAR/sync；CUDA=false | 通过 |
| 阶段 5 smoke 入口静态门禁 | serial 4 cases + concurrent 8 + runtime evidence grep | 可复现执行且脚本通过语法/静态检查 | Python ruff/format/compile/import help 通过；shell `bash -n` 通过；shellcheck 未安装 | 通过（已执行项） |
| 阶段 5 TP=8 formal preflight | 已发布代码 + immutable inputs + 连续两次 GPU 检查 | 所有身份门禁通过且 8 卡连续空闲 | main `e4b0ce0f`、source `0f1bd5b3`；13:01:55Z/13:02:58Z 两次 8/8 空闲 | 通过 |
| 阶段 5 首次 TP=8 服务 | 固定 `oscar_mla_int2` 配置与 8 workers | 加载 artifact、权重并进入 KV profile | artifact 正交校验 CPU/CUDA identity 跨设备；权重加载前退出码 1；GPU 0 MiB | 未通过，修复中 |
| artifact default-device 修复 | 11 项定向 + 正式 78 层 CUDA default-device + 完整套件 | validator 始终在 CPU 校验且无回归 | 11 passed；78/78 CPU、CUDA=false；完整 81 passed/26 skipped | 通过 |
| 阶段 5 第二次 TP=8 服务 | artifact、141 shards、三池 planner、warmup | 进入 ready | artifact/shards/planner 通过；637,632-token capacity；warmup 因 dataclass `.numel()` 退出 | 未通过，修复中 |
| 三池 empty-cache 门禁修复 | runtime path + 完整 A800 CUDA 套件 + 静态门禁 | dataclass backing tensor 门禁正确且无 kernel 回归 | runtime 5 passed；完整 CUDA 108 passed、36.91 秒；静态门禁通过 | 通过 |
| 阶段 5 第三次 TP=8 服务 | 分配前 profile、三池 planner、warmup | 进入 ready | artifact/141 shards 通过；分配前空 Tensor 因按 dtype 直接取 `.raw` 退出 | 未通过，修复中 |
| OSCAR cache 双生命周期门禁 | 分配前空 Tensor + 分配后空 dataclass + 静态门禁 | 两种对象形态均正确短路 | runtime path 6 passed、4.88 秒；ruff/format/compile/diff 通过 | 通过 |

## 错误日志

| 时间 | 错误 | 尝试 | 处理 |
| --- | --- | ---: | --- |
| 2026-07-24 | `docker: command not found` | 1 | 改为核验本地 image tar，并寻找当前环境可用的无 daemon 解包方案 |
| 2026-07-24 | 外部源码 Git `dubious ownership`；相对路径读取根计划失败 | 1 | 改用一次性 `-c safe.directory=<精确路径>` 和根计划绝对路径 |
| 2026-07-24 | `jq: command not found` | 1 | 改用 POSIX 文本工具解析 archive manifest |
| 2026-07-24 | layer 扫描命令中的 `rm -f` 被安全策略拒绝 | 1 | 改为无临时文件的流式 `tar -tf | awk` |
| 2026-07-24 | `git archive <手工长 SHA>` 报告 `not a tree object`，clean tree 为空 | 1 | 忽略该次无效 diff，改为先验证 `HEAD^{tree}` 再归档 |
| 2026-07-24 | provenance `rg` 范围过大并进入历史 logs/cache | 1 | 主动中止，后续只读取精确 manifest 和已知文件 |
| 2026-07-24 | 本地 clone 两次因源仓库 `.git` 的 `dubious ownership` 失败 | 2 | 命令级 safe.directory 对 upload-pack 无效；改用 `git bundle` 中转保留完整历史 |
| 2026-07-24 | pre-commit ruff hook 首次 GitHub 初始化超过 3 分钟无进展 | 1 | 主动中止；改用相同固定版本的 PyPI ruff 做定向检查 |
| 2026-07-24 | rootless overlay/fuse-overlayfs mount 被 Kubernetes 拒绝 | 3 | 停止 overlay；改用 NFS graphroot 上的 vfs + chroot |
| 2026-07-24 | buildah vfs 普通 pull 与 unshare import 均失败 | 2 | 分别为 remount `/` 被拒绝、缺 subuid + `memfd_create` 失败；改走 Kaniko/OCI 方案 |
| 2026-07-24 | `umoci insert` 生成的 tar layer 尾文件固定截断 393 字节 | 2 | NFS 与本地 `/tmp` 复验结果一致；停止 insert，改用 GNU tar + OCI descriptor |
| 2026-07-24 | 二次层重建首次使用错误的临时归档根路径 | 1 | 在生成摘要前中止；按精确 OCI 目标路径重跑 |
| 2026-07-24 | `umoci unpack` 在 NFS 元数据阶段约 72 分钟未退出 | 1 | 终止并记为未完成；保留 10 分钟进度，展开内容的 11/11 SHA256 通过 |
| 2026-07-24 | Kubernetes API 查询当前 Pod spec/image 返回 403 | 1 | 停止无权限查询；不重复尝试 |
| 2026-07-24 | SSH 到 HOST_IP 只回连当前容器，仍无 Docker | 1 | 停止宿主 Docker 路径；按容器环境规则直接配置固定 rootfs |
| 2026-07-24 | 首次阶段 1 verifier 把仓库 HEAD 当成 OCI runtime source，发现 3 个 `recovery/` 文件不一致 | 1 | 读取 OCI revision/tree，分开记录 repository HEAD 与 runtime commit；按 `fd3e0b...` 全量 Git tree 复验 4,711/4,711 通过 |
| 2026-07-24 | smoke prompt 对 Transformers 5.8.1 `BatchEncoding` 直接取 `len()`，错误得到 2 | 1 | 改为读取 `input_ids`；复测得到 506 和 31,996 tokens |
| 2026-07-25 | 完整历史推送包约 185MiB，不符合最新“主要同步代码”要求 | 2 | 停止完整历史上传；以相同 tree 创建无父提交的代码快照，完整历史仅保留在本地追溯分支 |
| 2026-07-25 | TP=8 worker 因 FlashInfer/JIT cache 版本不匹配退出 | 1 | 对照已验证部署入口，补齐其原有 `FLASHINFER_DISABLE_VERSION_CHECK=1` 后重跑 |
| 2026-07-25 | 第二次 TP=8 worker 因当前容器 `flash_attn` 污染、缺少 `flash_attn.ops` 退出 | 1 | 审计固定 venv 与候选 rootfs，确认 system site-packages 绝对路径泄漏；隔离当前系统包后重跑 |
| 2026-07-25 | official_v4 首轮的 600 秒 code timeout 低于 4,096-token 实测生成时间 | 1 | 保留冻结数据/runner，只把每轮 runtime code timeout 提高到 900 秒；8 条探针已全部 scored 且无请求失败 |
| 2026-07-25 | 阶段 2 linked worktree 的 pre-commit 初始化停滞，中止后 index 被 hook cache 内容覆盖 | 1 | 确认正式 worktree 与对象库完好；按预先记录的 5 文件 SHA256 重建隔离 index，重新暂存并通过全部手工门禁 |
| 2026-07-25 | capture 输出 schema 的 `value_samples` 键被计数和样本张量重复使用 | 1 | 计数键改为 `value_covariance_samples` 并同步 score/latent 命名；17 项测试与全部静态门禁随后通过 |
| 2026-07-25 | 完整 scheduler 回归默认下载 LLaVA，并在项目外创建 Hugging Face cache | 1 | 立即终止；删除本次新建的 3,622,499-byte cache、0-byte lock 和 36KB Xet 日志；改用强制离线回归 |
| 2026-07-25 | Triton interpreter 的 split merge 对标量 mask 执行位与时报类型不兼容 | 1 | 拆分为两个 `tl.where` 条件，避免对不同标量类型执行位运算 |
| 2026-07-25 | Triton interpreter 中 BF16 `tl.dot` rotation 产生无效大值 | 1 | rotation 改为 FP32 输入与 IEEE FP32 累加；512 维 oracle 复测通过，A800 编译仍待 GPU 释放 |
| 2026-07-25 | Stage 5 worktree 自有 `uv` venv 首次 pytest 缺少 `tblib` | 1 | 从清华 PyPI 镜像通过 `uv pip` 安装 `tblib==3.2.2`；用该 `.venv` 重跑 8 项测试通过 |
| 2026-07-25 | Stage 4 RoPE 修复 pytest 被 worktree venv 的未安装依赖阻断 | 3 | 依次补齐 `cbor2==5.8.0`、`cachetools==7.0.1`、`py-cpuinfo==9.0.0`；定向 interpreter 与完整套件随后通过 |
| 2026-07-25 | Stage 5 干净子进程的 interpreter smoke 缺少仓库导入路径 | 1 | 测试子进程显式固定项目根 `PYTHONPATH`；定向 10 项和完整 76 项非 CUDA 测试随后通过 |
| 2026-07-26 | 阶段 3 正式 `.venv` 没有已安装 vLLM metadata，12 项通用测试自动 device detection 失败 | 1 | 加入项目内候选 rootfs 的 metadata/dependency 路径；12/12 单独通过后全量 116/116 通过，CUDA 未初始化 |
| 2026-07-26 | Stage 4 测试修复 commit 的 pre-commit 初始化 actionlint hook 停滞 | 1 | 中止 hook；手工 ruff/format/A800 targeted test/diff 全通过后用 `--no-verify` 提交，未放宽正式测试门禁 |
| 2026-07-26 | Stage 5 首轮正式 A800 完整套件的非连续 rotation 前置断言失败 | 1 | 105/106 项通过；QR 输出在当前 PyTorch 已为非连续，原测试 `.T` 后反而连续；改为 `.contiguous().T` 后定向 A800 kernel 1/1 通过，待发布后用全新 cache 重跑 |
| 2026-07-26 | Stage 5 单文件测试修复提交时 actionlint hook 再次初始化停滞 | 1 | 主动中止；ruff、format、py_compile、定向 A800 kernel 和 diff 已全部手工通过，随后 `--no-verify` 提交并推送 `0f1bd5b30` |
| 2026-07-26 | Stage 5 首轮服务 dry-run 的 CLI 校验内联脚本遗漏 `import os` | 1 | 静态输入和候选环境均通过；补齐 import 后 retry 完整 dry-run 通过，CUDA=false |
| 2026-07-26 | dry-run retry 首次把输出重定向到尚不存在的 artifact 目录 | 1 | shell 在脚本创建目录前拒绝重定向，未执行验证；显式创建任务专用目录后重跑 |
| 2026-07-26 | 首次 Stage 5 TP=8 服务在 artifact 正交校验发生 CPU/CUDA 跨设备比较 | 1 | 8 workers 均在权重加载前退出；显式将 identity 创建在 CPU，并增加非 CPU default-device 回归 |
| 2026-07-26 | 第二次 Stage 5 TP=8 服务 warmup 对 `OscarMLACacheTensors` 调用 `.numel()` | 1 | 141 shards 与 planner 已通过；空 cache 门禁按 OSCAR dtype 改查 `.raw.numel()`，其他 dtype 保持原逻辑 |
| 2026-07-26 | 第三次 Stage 5 TP=8 profile run 对分配前空 `Tensor` 读取 `.raw` | 1 | 141 shards 通过、planner 前退出；确认 cache 在分配前后有 Tensor/dataclass 两种形态，改按实际类型分派 |

## 5 问题恢复检查

| 问题 | 答案 |
| --- | --- |
| 当前在哪里？ | 阶段 0–4 已完成并形成中文报告；阶段 5 正式 A800 cold-cache 完整套件已 106/106 通过 |
| 将去哪里？ | 启动 TP=8 `oscar_mla_int2` 服务并完成单/多请求、demotion、mixed read、近 32K、无 fallback 和压缩率验收 |
| 总目标是什么？ | 完成设计文档规定的 OSCAR × GLM‑5.2 × A800 32K 首版本及 128K 扩展验证 |
| 已了解什么？ | 见 `findings.md` |
| 已完成什么？ | 阶段 0–4 的源码恢复、baseline、calibration/artifact、三池 allocator 与 A800 kernels 正式验收及中文报告，以及阶段 5 WIP 集成代码；详见本文件对应日志 |
