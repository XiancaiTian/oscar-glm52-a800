# OSCAR-vLLM 集成计划

## 目标

基于固定镜像 `docker.m.daocloud.io/vllm/vllm-openai:v0.25.0` 和 vLLM v0.25.0，为 Qwen3-4B-Instruct-2507 实现可证明的 OSCAR mixed KV serving 路径，并完成 `task.md` 规定的全部测试、路径/压缩率证明及 GSM8K 精度验收。

## 成功标准

- 固定并记录 vLLM v0.25.0、OSCAR `main` HEAD、镜像、模型、rotation 和数据集版本/路径。
- K/V rotation 与 clipping 参数真实参与计算；mixed KV 分区为 prefix 64 BF16、recent 256 BF16、history INT2。
- serving 性能关键路径使用 Triton 完成 KV 写入/量化和 attention 读取/反量化/逆旋转，不构造整段临时 BF16 cache。
- 通过 rotation 加载、INT2 roundtrip、mixed layout、attention 对照和端到端 smoke 五类测试。
- 提供日志/trace 的运行路径证明，以及 BF16 与 mixed OSCAR 的理论和实测 cache 占用对比。
- 先完成同配置 vLLM BF16 GSM8K，再完成 OSCAR-vLLM GSM8K；OSCAR `Correct >= 1134/1319`。
- 中文报告仅记录实际结果，并包含 `task.md` 和 `AGENTS.md` 要求的完整环境与命令。

## 阶段

| 阶段 | 状态 | 验证方式 |
| --- | --- | --- |
| 1. 环境、源码与验收基线 | 完成 | 源码 commit、镜像、模型/数据/rotation 存在性和工作区状态有记录 |
| 2. OSCAR 数值语义与最小单测 | 完成 | rotation/layout CPU 测试及 Triton 解释器 INT2 roundtrip/mixed attention 对照通过 |
| 3. vLLM 配置、分配与 Triton serving 集成 | 完成 | 配置/registry、静态检查、实际 specialization 离线编译和解释器路径通过；serving 无整段 BF16 临时 cache |
| 4. 端到端 smoke、路径与压缩率证明 | 完成 | 自包含镜像 GPU 单测和真实 serving smoke 已通过；同 9.35 GiB KV budget 下 OSCAR/BF16 实测 token/block 容量为 4.572268x |
| 5. GSM8K BF16 与 OSCAR 精度验收 | 完成 | BF16 `1161/1319`；OSCAR `1157/1319`，相对 delta -0.0030326005，超过硬阈值 23 条 |
| 6. 中文报告与逐项完成审计 | 完成 | 每条显式要求均有当前证据，报告章节/交叉引用和最终测试通过 |

## 当前约束与决策

- 顶层初始目录没有 vLLM 或 OSCAR 源码，也不是 Git 仓库；阶段 1 需获取固定源码。
- Shawn 已指定仅可使用 GPU 1；2026-07-19 首次检查发现该卡被其他进程占用，在连续两次确认空闲前不执行 GPU 实验或测试。
- GPU 分配后，每次实验前按规范检查占用并连续两次确认可用；长实验每 10 分钟输出进度。
- 所有开发和实验以指定 Docker 镜像为基座；主机只用于源码/文件管理和无 GPU 的初步检查。
- 原型明确限制 `max_num_seqs=1`、禁用 prefix cache/spec decode；这符合任务中多并发与组合能力不作为硬性验收的范围，并显著收敛 mixed recent ring 的正确性边界。

## 错误记录

| 错误 | 次数 | 处理 |
| --- | ---: | --- |
| 顶层执行 `git status` 报非 Git 仓库 | 1 | 已确认目录仅含任务材料，转为阶段 1 获取源码 |
| 当前用户访问 Docker socket permission denied | 1 | 后续使用规范提供的 sudo 授权访问 Docker daemon |
| OSCAR 完整历史 clone 持续下载大 pack | 1 | 终止后改为固定 HEAD 的 shallow clone，不获取无关历史 |
| partial clone 上 `git grep` 触发逐对象下载 | 1 | 终止后一次性 shallow fetch HEAD 全部对象 |
| 只读检查数据集仓库时报 dubious ownership | 1 | 使用单命令 `-c safe.directory=...`，不改全局 Git 配置 |
| 覆盖镜像 entrypoint 后 `python: command not found` | 1 | 检查镜像 Config/entrypoint 后使用其实际 venv Python 路径 |
| 只读挂载下 `compileall` 无法写 `__pycache__` | 1 | 设置 `PYTHONPYCACHEPREFIX=/tmp/pycache`，后续语法检查通过 |
| 运行镜像缺 pytest 依赖 `tblib` | 1 | 当前用直接 assertions 验证 CPU 逻辑；正式 pytest 使用同镜像派生测试环境补齐 requirements |
| 派生镜像首轮 ruff 报 7 个文件需格式化 | 1 | 使用镜像固定 ruff 0.14.0 对工作区做机械格式化后复查 |
| pytest 中 `tests.models` 无法导入 | 1 | Dockerfile 将 tests 改为复制到 `/workspace/tests`，保持包路径 |
| 无 GPU 容器内构造 `EngineArgs`/CLI 报空 device string | 2 | CUDA 镜像在未挂载 GPU/NVML 时识别为 unspecified platform；显式 `CpuPlatform` 已验证 EngineArgs 配置层，不再重复 CLI 构造，CUDA serving 留待获准 GPU 实测 |
| 初次误将固定 BF16 arena 改为经请求 `block_table` 映射 | 1 | 短请求逻辑页不足以承载 320 个全局 BF16 slot；已立即撤回，明确单请求固定物理 arena 设计并增加容量断言 |
| 用 `import vllm._C` 误判派生镜像丢失原生扩展 | 1 | 基础镜像也采用 stable-libtorch 模块名；逐文件对比 16 个 `.so` 路径和 SHA256，确认全部保留 |
| Triton V 逆旋转离线编译不支持 `result[0, :]` | 1 | 改用 Triton 列 reduction；SM80/SM90 离线编译均通过 |
| 首次检索 OSCAR kernel 使用了错误子目录 | 1 | 根据 `rg` 返回的真实路径改读 `sglang-research/python/sglang/QuantKernel/`，未重复错误命令 |
| 复核 metadata dtype 时沿用简写路径 `oscar_reference/sglang/...` 导致只读命令未命中 | 1 | 重新枚举目录并改用实际路径 `oscar_reference/sglang-research/python/sglang/...` |
| FP32 metadata 首个跨文件补丁因 decode 格式化上下文不匹配而整体拒绝 | 1 | 确认未产生部分修改，改为读取精确行段后按文件拆分补丁 |
| 通过 `ruff.__version__` 查询版本触发 `AttributeError` | 1 | ruff 模块不暴露该属性，后续使用 CLI `python -m ruff --version` |
| 镜像内 ruff 发现两个测试文件 import 排序不一致 | 1 | 先前从 `/workspace` 检查 `/src` 导致项目根/first-party 分类不同；改在挂载仓库根 `/src` 下用自身 `pyproject.toml` 修复并复查 |
| FP32 metadata 离线编译使用的 mixed decode 常量与实际 runtime specialization 不同 | 2 | 首次修正 tiling 后又发现 split 数沿用了函数默认 16，而 backend 实际配置默认 32；最终按 `NUM_KV_SPLITS=32,BLOCK_KV=4,num_warps=1,num_stages=1` 重编译 stage1/stage2，并补编译引用新 helper 的 demotion |
| 追加 GPU 检查记录时补丁上下文少了一个空格 | 1 | 补丁整体拒绝且未产生部分修改；重新读取实际文本后拆分写入 |
| 只读仓库挂载下 ruff/py_compile 尝试写 cache | 2 | ruff 加 `--no-cache`，py_compile 设置 `PYTHONPYCACHEPREFIX=/tmp/pycache` 后通过 |
| V 逆旋转 stride 修补首次仍保留临时 `.contiguous()` | 1 | 目标回归仍失败后检查调用参数，删除残留调用；同一两个回归与完整 GPU 模块均通过 |
| frozen runner requirements 未声明 `absl-py`/`nltk` | 2 | 无 GPU 同基座容器中显式安装两项；CLI 和全部 imports 通过，已记录解析版本 |
| BF16 server 前台工具会话满 30 分钟收到 SIGTERM | 1 | 386 条后服务退出，933 条连接失败；保留无效产物，改用 detached Docker container 后完整重跑 |
| detached BF16 server 被后续外部 8-GPU 任务终止 | 1 | 15 条后服务收到 SIGTERM、退出码 137 且 OOMKilled=false；隔离无效产物，重新等待连续两次空闲确认 |
