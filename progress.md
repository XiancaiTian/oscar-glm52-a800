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

- **状态：** 运行中
- **已执行：**
  - 于 2026-07-25T17:50:15Z 使用冻结 2,360 样本 manifest、并发 8 和 code timeout 900 秒的 runtime config 启动正式全量评测。
  - 在 2026-07-25T18:00:15Z 至 2026-07-26T01:20:23Z 分别写入 10–450 分钟 GPU 与进程进度。
  - 对 runner 缓冲输出、服务 POST 状态和错误日志分别核验，不用服务请求数替代最终 scored 数。
- **实际结果：**
  - 四十五次进度记录期间 8 张 A800 均维持约 79,901–79,941MiB 显存占用，评测 runner、服务与并发请求持续运行。
  - runner 于 18:22:01Z 刷新 `completed 20/2360`；60/70/80/90/100/110/120/130/140/150/160/170/180/190/200/210/220/230/240/250/260/270/280/290/300/310/320/330/340/350/360/370/380/390/400/410/420/430/440/450 分钟节点分别为 40/40/60/60/80/80/80/100/100/100/120/120/120/140/140/160/160/180/200/220/240/240/260/280/300/300/320/340/360/380/380/400/420/440/440/460/480/500/540/580，前三个节点因 stdout 缓冲记录为 `completed unknown/2360`。
  - 2026-07-26T00:56:07Z 至 `00:56:27Z` 服务保持 Running=8、Waiting=0、生成吞吐约 47.0–52.0 tokens/s，累计 HTTP 200 为 518；错误检索仅命中启动期正常 NCCL 初始化信息，未发现运行时异常。
  - 本轮尚未结束，accuracy、失败分类和产物 SHA256 不提前填报。

### 阶段 2：共享潜空间 reference、capture、artifact 与 manifest

- **状态：** 准备里程碑通过，正式 calibration 待阶段 1 出口
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
  - 正式源码 worktree 仍为 `53d8be94f...` 且干净；阶段 1 运行未受影响。
  - 已从官方 datasets-server 固定 OpenWebMath revision `fde8ef8d...` 的 0–299 行，项目内文件为 300 行、2,800,069 字节、SHA256 `39d245ca...b80`；LongBench 固定 revision 为 `5e628be4...`，5 个只读文件 SHA256 均已写入配置。
  - 开发版 manifest 为 20 行、250,907 字节、50,000 tokens，SHA256 `cde88339...25da`；两次独立构建的 manifest 与 summary SHA256 均完全一致。
  - 正式版 manifest 为 292 行、4,570,560 字节、1,000,000 tokens，SHA256 `3a183cba...76b5`，summary SHA256 `f0e323f4...91bf`；两次独立构建完全一致。
  - 正式版独立审计确认 292 个 entry ID 唯一、235 个源样本无 train/holdout 跨分区、无重复文本 hash、与 official_v4 完整 prompt hash 交集为 0，重新 tokenize 后各类别 token 数与配额逐项一致。
  - phase-2 runtime 的 6 个 vLLM 原生扩展通过项目内只读 symlink 解析，另 1 个 sparse MLA 扩展按候选 rootfs 绝对路径验证；7/7 SHA256 通过，候选 Python 实测从主仓库 source 载入 `vllm`/`vllm._C` 且 CUDA 未初始化。
  - 正式源码 submodule 仍停留在阶段 1 commit `53d8be94f...`；完整 phase-2 preflight 明确等待阶段 1 退出后再切换和提交指针，未干扰当前服务。

### 阶段 3：三池 CacheSpec、allocator 与 scheduler/worker 隔离准备

- **状态：** 隔离代码里程碑通过，正式 submodule 接入与阶段报告等待阶段 2 出口
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

### 阶段 4：SM80 Triton kernel 隔离准备

- **状态：** WIP 代码已同步；decode/prefill CPU interpreter 通过，A800 cold compile/launch 尚未验收
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

### 阶段 5：`oscar_mla_int2` runtime 激活准备

- **状态：** WIP runtime cache 路径已同步；A800 launch 与端到端未验证
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
| official_v4 全量运行进度 | 2,360 样本、并发 8、code timeout 900 秒 | 每 10 分钟有记录且进程无请求错误 | 10–450 分钟记录已落盘；2026-07-26T01:20:23Z 为 580/2360，服务与 8 卡持续活动 | 运行中 |
| 阶段 2 reference/covariance/capture/artifact | 定向 pytest、Python 语法、ruff 0.14.0、format check、diff check | 数值 reference、基础统计、只读 capture 与 fail-closed artifact 全部通过 | 25 passed；语法/lint/format/diff 均通过 | 通过 |
| 阶段 3 三池 scheduler/worker 集成 | 116 项定向 pytest + 强制离线 scheduler 回归 + ruff/format/compile/diff | 三池预算、ownership、views 与通用 scheduler 无回归 | 116 passed；离线 scheduler 68 passed，28 项仅缺 LLaVA 配置；静态门禁全通过 | 通过 |
| 阶段 4 kernel 隔离准备 | `tests/oscar_mla` + Triton interpreter + ruff/format/py_compile/diff，CUDA 门禁未启用 | CPU 回归及 decode/prefill interpreter oracle 通过且 CUDA 结果不冒充 | 61 passed、22 CUDA skipped；512+64 维 decode/prefill 最大误差均为 `2.384185791015625e-07`；commit `8ac7b9d97...` 已推送；A800 未运行 | WIP |
| 阶段 5 runtime cache 路径 | artifact/metadata/write/read 定向测试 + 完整 `tests/oscar_mla` + 静态门禁 | fail closed，demotion 顺序、多请求 ownership、DSA local IDs/padding、输出/LSE oracle 正确且不冒充 GPU | 完整套件 80 passed、26 CUDA skipped；CPU interpreter 与多请求 metadata mock 已通过，batch 4/8 待 A800；commit `c762b4aee...` 已推送 | WIP |
| 阶段 5 真实 EngineConfig | 候选 Python + 真实模型 + TP=8/32K OSCAR CLI | 默认 async 被拒绝，显式同步配置成功且不初始化 CUDA | 默认配置按预期失败；`--no-async-scheduling` 后配置字段全部匹配，CUDA=false | 通过 |

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

## 5 问题恢复检查

| 问题 | 答案 |
| --- | --- |
| 当前在哪里？ | 阶段 1 official_v4 全量 900 秒基线运行中；阶段 2/3 隔离代码里程碑已通过，阶段 4 kernel 待 A800 验证，阶段 5 已完成 cache runtime 代码接线并等待 A800 验证 |
| 将去哪里？ | 完成 official_v4 全量精度和 WikiText-2 PPL，再冻结独立 calibration manifest 并运行 capture/calibration |
| 总目标是什么？ | 完成设计文档规定的 OSCAR × GLM‑5.2 × A800 32K 首版本及 128K 扩展验证 |
| 已了解什么？ | 见 `findings.md` |
| 已完成什么？ | 阶段 0、阶段 1 原生服务与四项 smoke、timeout 探针、阶段 2 calibration 全部入口、阶段 3 三池 planner/allocator/scheduler/worker 隔离实现，以及阶段 4 store/demotion/decode WIP 代码；详见本文件对应日志 |
