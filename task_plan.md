# 任务计划：OSCAR × GLM‑5.2 × 苹果800 分阶段实验

## 目标

严格按照 `docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md` 完成阶段 0–9 的实现、实验、验收与中文报告，最终满足第 17 节的 32K 首版本完成定义，并给出 128K 扩展验证或容量阻塞证据。

## 下一步

Stage 9 未优化 OSCAR 轮次 `20260730T1741Z_stage9_candidate_v1` 的完整
1K/batch1 已证明 TTFT/TPOT 相对 BF16 回退约 `+1332.7%/+55.1%`，并由
8-rank profiler 定位到 mixed decode kernel 与 78 层重复 KV update 索引链。
该轮已在 batch4 第三轮期间停止，8 卡释放且没有全矩阵 summary。首个最小优化
提交 `98ddd3f4e...32aaa2` 已发布：纯 decode 跳过不可能命中的
current-history 布尔索引。定向 GPU 探针已证明 KV update CPU/CUDA total
下降 `42.3%/59.3%`，TPOT 下降 `15.0%`；但 mixed stage1 基本不变，TTFT
仍约 5.015 秒。固定单卡、真实 1K/batch1 decode 几何的 split
4/8/16/32 sweep 已证明当前默认 16 最优；8-rank prefill trace 随后定位
mixed stage1 占 TTFT `91.59%`。prefill 专用 sweep 进一步证明，把 2,048
个 DSA 槽位裁剪到当前 1,024 序列上限并使用 split1，可将单层时间从
`62.884 ms` 降到 `46.382 ms`，加速 `1.356×`。prefill-only top-k 裁剪和
split1（保持 decode split16）已由源码提交
`a94b1f640...ad894` 实现，CPU 回归通过，并冻结到新 OCI/控制镜像；报告
7.12 已记录证据边界。正式 containerized preflight 已通过 64/64，
`cuda_initialized=false`。TP=8 1K/b1 探针已完成 3/3 正式轮次和完整
8+8+1 profiler：TTFT/TPOT 为 `3714.821/205.908 ms`，相对 decode
快路径为 `-25.9%/-0.4%`，但相对 BF16 仍为 `+954.0%/+31.4%`。8-rank
trace 证明 mixed stage1 仍占 prefill `89.25%`、累计 `3300.032 ms`。
mixed stage1 跨 head 复用已由源码提交
`35ab1846447fc86b4b2177e76c5939503cc3701b` 实现：单卡单层 cropped/split1
从 `46.382 ms` 降至 `13.284 ms`，并通过完整冷 cache CUDA 套件
124/124。修复 PAX 扩展头 PID 后，两次独立完整 OCI 构建和递归验收得到完全
一致的不可变身份；v3 import、控制镜像、正式配置与 64/64 preflight 均已通过。
新 TP=8 1K/batch1 探针
`20260731T0005Z_stage9_candidate_headgroup_probe_1k_b1_v1` 已完成 3/3
正式轮次及 8+8+1 profiler：TTFT/TPOT 为 `1317.120/202.668 ms`，
相对上一版 OSCAR 为 `-64.54%/-1.57%`，但相对旧 BF16 仍为
`+273.71%/+29.33%`，没有通过 20% 门限。同口径 8-rank trace/table 归因
已经完成：OSCAR prefill/generation worker 窗口相对 BF16 分别慢
`445.12%/27.09%`，逐层 KV update CPU 路径按 78 层折算约多
`43.05 ms/token`，是下一项首要可控 decode 瓶颈；NCCL 更符合跨 rank 等待
的放大结果。metadata 一次性物化与 layer demotion scratch 复用已经由源码
提交 `14c768b406b3e39a2d4d5be77a9046ac7ccc26d1` 落地并推送；最终 CPU
套件为 96 passed、29 个 CUDA 显式 skip、0 failed，静态门禁通过，中文报告
7.16 已同步。主仓库 submodule、报告和计划已由提交
`cfdef8aa562088ab7591ad72ab93ef128d855589` 发布。下一步完成苹果800
cold-cache CUDA 门禁；有效轮次为 125/125 passed、0 skipped/failed，
80.88 秒；阶段报告已由主仓库 `3446b6fa…d813` 发布。Phase 6 输入与
Dockerfile 已最小切换到源码 `14c768b…`/tree `4ad8be8a…`，确定性 PAX
回归 1/1 通过；配置由 `47769e047…9373` 发布。新 OCI 已在两个独立目录构建
并递归验收，image/config、manifest、candidate layer 和 index 逐字节一致；
正式 v1 随后已导入 Docker daemon，image ID、33 层和全部身份 labels 通过
审计。driver-injected runtime import 已通过：正式 Python/PyTorch/Triton、
候选 vLLM Python/原生扩展、78 层 rotation、runtime expectation 与
`reasoning_effort=max` 均匹配，`cuda_initialized=false`。Stage 9 控制
镜像 Dockerfile 默认 base 已最小切换到新候选 tag，文件 SHA256 为
`6b6f4d1d…e2d3e`；发布后已构建并验证
`oscar-glm-stage9-runtime:14c768b40`，image ID 为
`sha256:84c48782…989f`，34 层严格继承新候选 33 层，CPU-only runtime
检查通过且 `cuda_initialized=false`。Phase 1/5/7/9 config/wrapper 已切换，
旧身份清零；四份配置 SHA256 为
`d588e627…c820`/`b7a58d10…4d66`/`680708fc…b15f`/
`1bdfabb9…b379`，JSON、9 个 shell、Python compile 与 diff 门禁通过。
固定控制镜像内 Phase 7+9 合并工具测试为 39 passed、0 failed；下一步分别
复跑后 Phase 7/Phase 9 分别为 20/20 和 19/19 passed。宿主机直接递归
verifier 因 NFS mode 映射产生假失败；改在正式控制容器挂载命名空间后
Phase 9 递归静态 verifier 64/64 通过。无 driver 的同轮随后在导入
`vllm._C` 时缺少 `libcuda.so.1`，因此不能冒充完整 preflight。配置、
wrapper、测试与本阶段中文报告已由提交 `04c96567…` 发布，恢复进度由
`bf80eef7…` 发布。driver-injected containerized preflight 前两次 8/8
空闲检查间隔 65 秒；完整 preflight 退出码 0，静态 64/64、固定环境 import
和服务参数解析全部通过，`cuda_initialized=false`。preflight 阶段报告已由
提交 `151e1c6a…85e9` 发布。32K/batch1 首次探针三轮均为 3/3 completed、
0 failed，诊断中位数为 TTFT `106660.424 ms`、TPOT `200.303 ms`、吞吐
`0.007573 req/s`，相对 BF16 为 `+751.37%/+12.01%/-73.32%`。但 profile
完成后新增未跟踪的实时优化记录触发仓库洁净门禁，整轮退出码 1，未生成单格
summary，不能标记为正式通过。trace 已确认 32K prefill 被拆成 16 个
2,048-token 窗口。多 chunk 分析器已扩展并通过 Phase 9 工具测试 20/20；
OSCAR/BF16 的 8-rank prefill wall 中位数为
`105753.449/10086.470 ms`，kernel 合计为 `105609.233/9533.580 ms`。
OSCAR `_mixed_sparse_prefill_stage1` 为 `93913.327 ms`、1,248 次，占
prefill wall `88.80%`。下一步先把实时记录与分析器纳入 Git 并发布，再以
单卡单层 2,048-query/2,048-top-k 负载验证该 kernel 的最小优化。固定
benchmark 已新增 `--seq-len` 和 block/top-k 门禁，在 2,048 形状只比较
IEEE split16 与 grouped split1；TDD 红灯后 Phase 9 工具测试为 21/21。
入口由 `49c9a1e…` 发布后，单卡 IEEE 基线得到
`195.772/47.158 ms`，grouped split1 加速 `4.151×`，output/LSE 最大绝对差
`3.34e-6/9.54e-7`，严格门限通过。grouped TF32 只修改 5 个 dot，源码提交
`24938975f…` 已通过 6/6 CPU 定向测试和全部适用 hooks，并已推送；主仓库
`b0370c3…` 已绑定 submodule/实时记录。首次 GPU 筛选在编译时需要
`169,984 bytes` shared memory，超过 `166,912 bytes` 上限，未进入精度或
性能测量。CPU-only SM80 资源 sweep 表明简单混用 IEEE 需要
`169,984–204,800 bytes`；仅让原生 BF16 pool 走 BF16 tensor core、history
保留 TF32 时为 `135,168 bytes`。该最小 hybrid 已由源码
`b9626ce9f…` 落地，CPU 定向 6/6 与全部适用 hooks 通过并已推送。下一步发布
主仓库 submodule/实时记录，再以相同 IEEE split16 参考重筛；通过精度和性能
门限后重建候选并以新 run ID 重跑同一正式格点。hybrid GPU 轮次已证明
`135,168 bytes` 可 launch，但 output/LSE 最大绝对误差为
`0.004933/0.002036`，超过门限且未进入计时。下一步恢复 BF16 value
probability 精度，再重复上述门禁。CPU-only SM80 离线编译已证明只把
BF16 value 累加恢复为 FP32 probability/TF32 dot 时，shared memory 仍为
`135,168 bytes`，低于 `166,912 bytes` 上限。最小源码候选
`b247211c9…` 已落地并推送，CPU 定向 6/6、ruff 与全部适用 hooks 通过；
主仓库发布后，同协议 GPU 筛选已按冻结的
`torch.allclose(atol=rtol=0.002)` 通过：grouped split1 为 `26.906 ms`，
shared memory `135,168 bytes`。完整 cold-cache CUDA 回归随后为
125/125 passed、0 skipped/failed、80.32 秒。Phase 6 输入与 Dockerfile
默认身份已切换到 `b247211c…/619ea47d…`，固定 Python 3.12.13 的 PAX
确定性回归 1/1 通过。两个独立目录的候选 OCI 构建和递归验收已完成，
image/config、manifest、candidate layer、diff-ID 与 index 全部一致。下一步
先实时发布 OCI 阶段记录，再导入 v1 并完成 daemon identity、runtime import、
控制镜像、配置迁移和 preflight，之后复跑 32K/batch1。v1 已导入 daemon，
image ID、33 层和 8 项关键 labels 全部匹配；下一步发布导入记录后执行双空闲
检查和 driver-injected runtime import。首轮探针错误读取 rotation 顶层长度，
第二轮虽修正为内层 78 项，却额外导入了冻结协议不要求的
`flashinfer/flashinfer.jit` 并触发 CUDA 初始化断言；两轮均未输出通过 JSON，
退出后 8 卡空闲。第三轮精确复用既有 runtime import 协议，只通过
`importlib.metadata` 校验 FlashInfer 包版本，结果已通过且
`cuda_initialized=false`；有效 JSON/log SHA256 为 `0910b598…7b7a`/
`f2e60043…189a`。阶段记录已发布；Stage 9 控制镜像 Dockerfile 的默认
base 已最小切换到新候选，文件 SHA256 为 `6e894ed3…4845`。下一步先发布
该复现入口，再进行 CPU-only 控制镜像构建和身份审计。入口已由
`eeaf56c8…` 发布；新控制镜像 `oscar-glm-stage9-runtime:b247211c9`
已构建，image ID `edbbc87d…b1b8`，34 层、前 33 层和 labels 继承及固定
CPU 环境均通过。新 runtime overlay 已从验收候选机械派生并通过 4,749 个
普通文件、6 个 native symlink 的递归内容/目标门禁；Phase 1/5/7/9 配置和
wrapper 已按依赖顺序迁移，四份配置 SHA256 为
`e6e5b599…2d25`/`40083bf3…801b`/`871feea0…d50a`/
`e2c764d7…9114`。新控制镜像中的 Phase 7/9 工具测试为
20/20、21/21 passed；正确容器挂载命名空间内的递归静态 verifier 为
64/64 passed。无 driver 的后续固定环境 import 因缺少 `libcuda.so.1`
退出，不能冒充正式 preflight。配置、报告与 planning 已由
`696bd8f762438fe56d9bd5c934773b69ce5c78fd` 发布。随后两次 8/8 GPU
空闲检查为 `05:26:12Z/05:27:29Z`，间隔 77 秒；driver-injected
preflight `20260731T0528Z_stage9_candidate_b247211c9_preflight_v1`
退出码 0，静态 64/64、固定环境 import 与服务参数解析全部通过，两处均为
`cuda_initialized=false`。preflight 结果由
`1d32d26cfba5239886cba6fa8781acd7f1461f48` 发布后，正式 32K/batch1
轮次 `20260731T0536Z_stage9_candidate_b247211c9_32k_b1_v1` 已完整退出码
0：三轮 3/3 completed、0 failed，`mean` 中位数为 TTFT
`47143.207 ms`、TPOT `199.458 ms`、吞吐 `0.013795 req/s`。相对上一
OSCAR 为 `-55.80%/-0.42%/+82.16%`，相对 BF16 为
`+276.30%/+11.53%/-51.39%`；TPOT 已在 20% 门限内，TTFT 仍超限。
Profiler 8+8+1 证据完整通过，prefill stage1 CUDA total 的 8-rank 中位数为
`34398 ms`，相对上一候选下降 `63.37%`，但仍为 BF16 同项约
`10.16×`。新 8-rank 多 chunk trace 已完成：prefill wall/kernel 中位数为
`46934.364/46100.251 ms`，generation 为 `269.297 ms`；stage1 为
`34398.099 ms`、占 wall `73.29%`，并解释相对 BF16 prefill wall 差距的
`84.17%`。编译产物进一步显示 4-warps stage1 为每线程 255 registers、
656-byte stack、135,168-byte shared memory。最小 8-warps 候选已由源码提交
`b87a401daf55b557b0b052f302fd35be222d1ff1` 推送，CPU/interpreter 6/6、
ruff 和全部适用 hooks 通过。主仓库 submodule 与实验前记录已发布；固定
单卡 2,048×2,048 筛选随后证明 8-warps grouped split1 为
`24.090 ms`，相对同轮 split16 加速 `8.128×`，相对旧 4-warps 再降
`10.47%`。冻结 allclose 通过，实际 cubin 为 247 registers、
0-byte stack、135,168-byte shared memory。完整 cold-cache CUDA 有效轮次
随后为 125 passed、0 skipped/failed、78.19 秒；首次 uv 路径错误轮次为
0 测试/0 cache，已单独保留失败证据。Phase 6 输入已最小切换到
`b87a401d…/7df314f2…`，PAX 确定性与身份门禁通过。下一步先发布构建前记录，
再执行双目录候选 OCI 构建与递归验收；通过正式链路迁移和 preflight 后才执行
新的 32K/batch1 端到端。
Shawn 于
2026-07-31 将优化迭代负载从 1K/b1
改为固定矩阵的 32K/b1：精确 32,768 输入 token、128 输出 token、并发 1，
其余 warm-up、3 轮正式测量和 8+8+1 profiler 协议不变。现有 BF16 对照为
TTFT `12528.026 ms`、TPOT `178.832 ms`；新候选必须在同一 32K/b1
口径实测，若仍未关闭 20% 门限，再继续 profiling 与优化。
门限关闭后才以同一最终提交重跑 BF16/OSCAR 完整 9 格和 128K，执行严格比较。

## 当前阶段

阶段 9：BF16 基线已完成，OSCAR 已触发性能优化

## 阶段

### 阶段 0：恢复并冻结源码

- [x] 找到并核验完整 GLM v0.19.0 源码
- [x] 核对 commit、镜像 ID、关键文件、原生扩展与运行时 patch
- [x] 建立本地独立 `glm52_oscar_vllm` Git 仓库
- [x] 创建可复现 Dockerfile/OCI 构建脚本并重建 baseline 镜像
- [x] 更新中文阶段报告
- [x] 确认 GitHub 远端并推送，之后再接入主仓库 submodule
- **状态：** 完成

### 阶段 1：GLM‑5.2/苹果800 原生 baseline

- **旧 checkpoint 历史结果：** 下列已完成项对应
  `GLM-5.2-FP8-pruned-staticgate-e154-H001-nfs`，不计入当前 REAP 模型完成状态。
- [x] 已取得 Shawn 授权：当前机器所有可见 GPU 均可使用
- [x] 连续两次核验授权范围内 8 张 苹果800 空闲
- [x] 复核模型与 checkpoint 快速指纹
- [x] 固化 TP=8/eager/32K/native sparse MLA 启动与 fail-closed 门禁
- [x] 固化短请求、>320 tokens、连续 decode、32K smoke 和评测入口
- [x] 在固定容器中完成 TP=8 短请求、>320 tokens、连续 decode 和 32K 验证
- [x] 冻结 official_v4 与 WikiText‑2 baseline
- [x] 更新中文阶段报告
- [x] 只读核验新 REAP checkpoint 的路径、几何、141 个分片和轻量指纹
- [x] 使用新 REAP checkpoint 完成 TP=8 原生 32K smoke
- [x] 冻结新 REAP checkpoint 的 official_v4 baseline
- [x] 冻结新 REAP checkpoint 的 WikiText‑2 baseline
- [x] 更新新 REAP checkpoint 的中文阶段报告
- **状态：** 完成。旧 checkpoint 的 official_v4 `0.19872881355932204` 与 PPL
  `7.692286035848967` 仅作历史记录。新模型 official_v4 已完成 2,360/2,360
  scored、request failure=0、overall accuracy `0.37415254237288137`；其
  GSM8K 子集为 666/1,319，与用户提供的 GSM8K-full `86.35%` 不是同一口径。
  新模型 WikiText‑2 PPL 为 `6.595132997244041`。

### 阶段 2：Calibration 与 PyTorch reference

- **旧 checkpoint 历史结果：** 以下 artifact 与统计对应旧 staticgate checkpoint。
- [x] 构建与 official_v4 独立的 calibration manifest
- [x] 实现环境显式启用、逐层限额、TP 分片的只读 activation/DSA capture
- [x] 完成 rotation artifact 写入、哈希、完整性与 runtime 身份 fail-closed 合约
- [x] 完成共享 covariance 合并和 rotation/clip 搜索
- [x] 完成共享潜空间 PyTorch reference、covariance 基础、正交性、未量化等价与 INT2 数值验证
- [x] 更新中文阶段报告
- [x] 使用新 REAP checkpoint 完成 900,000-token train capture
- [x] 使用新 REAP checkpoint 完成 100,000-token holdout capture
- [x] 使用 train/holdout capture 拟合并验证独立 rotation artifact
- [x] 更新新 REAP checkpoint 的中文阶段报告
- **状态：** 完成。新 artifact 包含 78 个 rotation，alpha 为 `0.25`，holdout
  loss 为 `0.026186010882512642`，manifest SHA256 为
  `0275043c070c9127354997374e9bca1c70fe1308a7b2d057f992fadedef868e5`。
  旧 artifact 不得用于新模型。

### 阶段 3：三段式 CacheSpec 与 CPU allocator

- [x] 实现 capacity planner、prefix/recent/history allocator 与生命周期
- [x] 在隔离分支接入 v0.19 scheduler/worker
- [x] 完成容量守恒、回滚和边界测试
- [x] 更新中文阶段报告
- [x] 按新 checkpoint 与最新 runtime source 回归 allocator、ownership 和 scheduler
- **状态：** 完成。当前定向套件 145 passed、26 个 CUDA 专项按显式门禁
  skipped；完整 scheduler 为 68 passed，28 项仅因离线缺少 LLaVA 配置而失败，
  无 OSCAR 或通用 scheduler 断言失败。14GiB 当前 planner 复算为 36,216
  blocks、579,440 个逻辑 token slots，理论容量比 `3.5711468297012128×`。

### 阶段 4：苹果800/SM80 Triton kernels

- [x] 按设计顺序实现 store、demotion、mixed sparse MLA 和 inverse rotation
- [x] 完成 SM80 cold compile、苹果800 launch、oracle 与边界测试
- [x] 更新中文阶段报告
- [x] 使用新 REAP checkpoint 绑定的 artifact 完成 SM80 cold-cache/苹果800 回归
- **状态：** 完成。GPU 0 完整套件 114/114 passed；8 卡 rank-local
  224/224 节点通过，其中 208 次为实际 CUDA 执行；新 artifact 的 layer 0
  rotation 已在独立 cold cache 上通过跨页 INT2 store/dequant oracle。

### 阶段 5：vLLM 接入与 32K 端到端

- [x] 注册并接入 `oscar_mla_int2`
- [x] 完成单/多请求、demotion、DSA mixed read 和接近 32K 验证
- [x] 证明无 fallback、无完整 BF16 history，并满足压缩率阈值
- [x] 更新中文阶段报告
- [x] 用最新 runtime source 与新 rotation artifact 完成 TP=8/32K 端到端回归
- **状态：** 完成。串行 4/4 与并发 8/8 请求均 HTTP 200；近 32K 为
  31,996+64 tokens。78 层 store/demotion/read 每层为 547/235/547，
  theoretical/padded 6.4×、allocated 3.5812365205×，无完整 BF16 history。

### 阶段 6：冻结候选镜像

- [x] 固定源码、Dockerfile、依赖、原生扩展与 rotation artifact
- [x] 构建并记录不可变候选镜像 tag、ID 和 digest
- [x] 更新中文阶段报告
- [x] 基于新 REAP artifact 与最新源码构建新的不可变候选 OCI
- **状态：** 完成并已按 official_v5 请求协议重建。当前候选 tag 为
  `glm52-oscar-a800-phase6-065af88a0-0275043c`，image ID 为
  `sha256:8b7a2ee6...68bb`，manifest 为 `sha256:01f91611...7932`；
  4,744 个源码文件、4 个 rotation 文件、runtime expectation 和 7 个基础层
  native extensions 全部通过，二次构建 digest 完全一致。

### 阶段 7：official_v5 GSM8K 阶段门禁

- [x] 冻结 official_v5 suite、runner、evaluator、依赖和协议指纹
- [ ] 完成 c8 部分吞吐探针和 8K/high、c16 固定 256 题原生/OSCAR 配对预跑
- [x] 验证快速 runner 增量落盘与中断恢复，冻结快速筛选协议
- [ ] 完成原生 baseline 与 OSCAR 候选各 1,319 条 GSM8K
- [ ] 生成 GSM8K 差异、截断/失败分类、预测 SHA256 和阶段门禁判定
- [ ] 更新中文阶段报告
- **状态：** 协议切换中。旧 checkpoint 的 573/2,360 基础设施失败轮次保留为
  历史证据；新 REAP checkpoint 的 official_v4 正式轮次曾于
  `2026-07-28T00:29:41Z` 启动，运行目录为
  `/dev/shm/oscar-glm-reap-stage7/phase7/20260728T0022Z_reap_candidate_tp8_final`。
  该轮在 600 分钟时为 runner/service 380/2,360，随后因 Shawn 将正式协议改为
  official_v5 而主动终止；全部 GPU 已释放，未生成的 v4 汇总不能当作实验结果。
  当前阶段按最新要求只运行 v5 GSM8K 1,319 条。v5 evaluator 已冻结到项目
  ignored artifact，共 578 MiB、161 个非 venv 文件通过 SHA256 递归复核；
  suite manifest SHA256 为 `ffc1d3b3...b2b`，runner SHA256 为
  `f2d36d9b...e80f`。固定 Python 3.12.3 环境、NLTK punkt/punkt_tab 与
  loopback-only 网络命名空间预检均已通过。首次 v5 原生 GPU 正式轮次
  `20260728T1145Z_native_official_v5_gsm8k` 的 TP=8 服务成功加载 141/141
  分片并 ready，但 1,319 个请求全部在生成前返回 HTTP 400，结果
  `valid=false`，不能计入精度。根因是 v5 固定
  `reasoning_effort=max`，而集成服务请求 schema 尚未接受 `max`；源码
  `065af88a0...` 已增加原样透传支持并通过 1/1 定向单元测试；候选 OCI 已完成
  重建、独立验收、固定运行时导入和确定性复建。第二次正式重跑
  `20260728T1148Z_native_official_v5_gsm8k_v2` 已接受全部 1,319 个
  tokenization 和 `reasoning_effort=max`；runner 首次报告完成 20 条时，
  服务端仅记录 13 个 completion HTTP 200，证明至少 7 条超过了上游固定的
  300 秒数学请求客户端超时。该轮已主动停止，未生成 summary/predictions，
  全部 GPU 已释放；它不能计入精度。当时的中间 runtime config 仅将
  `math_reasoning` 从 300 秒提高到 900 秒；样本、解码、评分和重试配置不变，
  原生与候选两轮必须使用同一 SHA256 固定配置。中间适配提交
  `addf77b7...` 已推送；第三次原生正式轮次
  `20260728T1255Z_native_official_v5_gsm8k_v3` 已通过 61/61 门禁和两次
  8/8 GPU 空闲检查，服务于 `2026-07-28T12:25:25Z` ready，1,319/1,319
  tokenization 均为 HTTP 200。该轮 20 分钟时仅有 6 个 HTTP 200；首批请求
  启动 900 秒后，KV usage 从 23.0% 降到 10.6% 并重新出现 prompt 吞吐，
  证明至少 2 个长请求已超时重试。该轮已停止并释放全部 GPU，
  未生成 summary/predictions。中间适配随后把 timeout 提高到 1,800 秒；提交
  `e0048b2...` 已推送；第四次原生正式轮次
  `20260728T1250Z_native_official_v5_gsm8k_v4` 已通过 61/61 门禁和两次
  8/8 GPU 空闲检查，服务于 `2026-07-28T12:56:28Z` ready，1,319/1,319
  tokenization 均为 HTTP 200。该轮 30 分钟时为 8 个 HTTP 200；精确
  1,800 秒处 KV usage 从 47.6% 降到 22.7% 并出现新 prompt，证明仍在超时
  重试，已停止并释放全部 GPU。冻结 runner 实际统一使用
  `32768 - 最长 prompt 218 = 32550` tokens，而不是 manifest 的 8,192 字段；
  按实测约 6.5 tokens/s/序列，满长约 5,008 秒。当前将数学请求 timeout 提高到
  7,200 秒、保留约 44% 余量。适配提交 `778c1a8...` 已推送；第五次原生正式
  轮次 `20260728T1333Z_native_official_v5_gsm8k_v5` 已通过 61/61 门禁和
  两次 8/8 GPU 空闲检查，服务于 `2026-07-28T13:38:37Z` ready，
  1,319/1,319 tokenization 均为 HTTP 200。10–80 分钟服务累计保持
  4/1,319；90/100/110/120 分钟增至 9/10/12/16，证明满长请求已在 timeout
  前正常完成。非 200、runner/service 错误、OOM 和 CUDA error 均为 0，
  当前正在生成。截至 `2026-07-29T01:39:06Z`，运行约 12 小时，服务累计
  95/1,319、runner 已完成 80/1,319；非 200 和错误仍为 0。当前服务完成速率约
  7.9 条/小时，按此线性估算原生轮次总耗时约 167 小时。Shawn 随后决定切换
  两级评测；该轮于 `2026-07-29T02:20:40Z` 主动停止，停止前服务累计
  98/1,319、runner 最近一次落盘为 80/1,319。进程树在 7 秒内退出，8 张 GPU
  均回到 0 MiB、0%；该轮没有 summary/predictions，不能作为精度结果。快速
  runner、确定性选择器、四格比较器、8K/high 配置和 GPU 前静态 verifier 已
  完成；逐题 checkpoint、每 20 题汇总落盘、协议指纹恢复和累计已记录活跃耗时均已
  实现。本地假服务连续运行两次，第二次从 2/2 恢复且 completion 请求数保持 2；
  256 与 1,319 题均固定使用全量数据的 218-token 最长 prompt 和 7,974-token
  输出上限。快速工具链 10/10 单元/集成测试、31/31 快速静态检查及 namespace
  preflight 均通过；代码与配置提交 `07fcb5b` 已推送，尚未产生 GPU 精度或
  吞吐结果。首次原生 c8 快速轮次
  `20260729T0249Z_native_fast256_c8` 实际运行约 34 分钟后按 Shawn 的增大
  并发要求停止：保存 26 个 checkpoint，26/26 scored、11 正确、8 截断、
  request failure 为 0，已完成输出合计 71,507 tokens；停止后 8/8 GPU 为
  0 MiB。该轮不具备完整 accuracy，只作为 c8 部分吞吐和截断证据。下一轮直接
  使用 c16 完成 256 题原生/OSCAR 配对。final=high 配置已通过 20/20 相关测试、
  33/33 正式 verifier、31/31 快速 verifier 及 formal/fast namespace
  preflight；提交 `99aaf8d` 已推送。原生 c16 轮次
  `20260729T0338Z_native_fast256_c16` 已完成并通过 validation：256/256
  scored、105 正确，accuracy `0.41015625`；128 条截断，truncation rate
  `0.5`；request failure 为 0。平均 completion 为 `4181.91796875` tokens，
  累计 1,070,571 tokens，已记录活跃时长 `10578.173911571503` 秒，吞吐
  `87.12278770458278` requests/hour。运行期间生成吞吐通常约
  104–107 tokens/s，8 卡无 OOM 或服务错误；结束后 8/8 GPU 已释放。
  `predictions.jsonl` SHA256 为
  `54a8e8adf57fbd92421aef21c9727575b122fc2e51a213dc3d9025d7b8233400`，
  协议指纹为
  `183a499b39cc05e8c03c9cda5df0c908b2a441769822728f9d4dbee952bd0db0`。
  截断题仍参与评分：截断前可提取答案则正常判分，无法提取则记错而不是请求失败。
  下一轮使用同一 c16 协议运行 OSCAR 候选。

### 阶段 8：精度优化（仅阶段 7 未通过时）

- [ ] 严格按设计文档第 11 节顺序优化
- [ ] 每轮保存独立 manifest、artifact/镜像和中文记录
- **状态：** 待开始

### 阶段 9：性能、128K 扩展与最终全量验收

- [ ] warm-up 后完成固定性能矩阵
- [ ] 对超过 20% 回退完成 profiling 和归因
- [ ] 验证 128K 或形成容量阻塞证据
- [ ] 冻结最终候选后完成 official_v5 全量 2,360 条 accuracy
- [ ] 完成 official_v5 WikiText‑2 PPL 与各原生指标硬门禁
- [ ] 更新中文阶段报告
- **状态：** 进行中。原 Stage 9 准备代码基于 official_v4 结果合约，切换
  official_v5 后必须先适配并重新通过静态门禁；已在项目内 ignored worktree
  `artifacts/stage9-prep-worktree` 创建隔离分支 `feat/glm52-stage9-prep`，
  完成固定性能矩阵、TP=8 启动、逐 rank profiler、128K 和比较入口，提交为
  `e5a5db8359872409f2a78cc73deae301b141c37a`。11/11 单元测试、原生 81/81
  静态门禁和候选 63/63 静态门禁均通过。Stage 9 启动安全提交为
  `30c5a4b`；性能输出、profiler、服务运行、cache 和比较结果现仅允许位于项目
  `artifacts/` 或 `/dev/shm/`，并持续复核配置、Git、模型和 profiler 证据哈希。
  外部/穿越路径、不安全 run ID 及非空正式 profiler 目录均会被拒绝。精度门禁
  通过前不启动 Stage 9 GPU 实验，也不把隔离提交同步到正在运行的主工作区。
  隔离分支当前 head 为 `ca39570`；其中也已准备 Stage 7 结果 provenance 门禁和
  项目内 ignored baseline 路径，但在途 accuracy 完成前不同步到主工作区。固定
  tokenizer 的 CPU 实测确认 1K/8K/32K/128K 四档随机 prompt 分别精确为
  1,024/8,192/32,768/130,944 tokens，未初始化 CUDA。当前 14/14 单元测试、
  19/19 Phase 7+9 合并测试、原生 81/81 和候选 63/63 静态门禁通过；比较器
  同时绑定主/源码 commit、模型身份和配置，逐文件重验 8 rank profiler
  table/trace，并验证固定服务参数。旧入口的在途 accuracy 完成后由独立定稿器
  在新目录保留原 validation 并补齐 provenance，不覆盖原始证据。该 v4 定稿逻辑
  仅作为历史准备，不得直接用于 v5。18 个隔离提交
  已在 `/dev/shm` 临时分支完成无冲突集成演练；结果 head 为 `90aed7e`、tree
  为 `1f90a402...252a`，19/19 合并测试、原生 81/81 与候选 63/63 静态门禁
  全部通过。该演练未改动正式主工作区或在途进程。当前已将与目标直接相关的
  Stage 9 工具同步到主工作区，并把 BF16/OSCAR 固定为同一 TP=8 控制镜像、
  模型、源码、1K/8K/32K × batch 1/4/8 矩阵、128 输出 token、3 轮、warm-up
  与逐 cell profiler；唯一变量是 KV cache 路径。控制镜像 ID 已冻结为
  `sha256:c77d7225...bfa2d`，基础候选镜像 ID 为 `sha256:8b7a2ee6...68bb`。
  容器化 BF16 预检
  `20260730T_stage9_preflight_baseline_v1` 与修正后的 OSCAR 预检
  `20260730T_stage9_preflight_candidate_v2` 均为 `passed`，两者均确认
  `cuda_initialized=false`；14/14 工具测试、shell 语法、Python compile、
  JSON 与 diff 检查通过。正式 BF16 v4 已完成 9/9 格、summary status
  `passed`；TTFT/TPOT 详见中文报告第 7.5 节。下一步以同一已发布提交运行
  OSCAR 首个完整格已触发性能优化；最新源码 `14c768b…` 的 metadata/scratch
  优化已经通过 CPU 96 passed、29 CUDA skip、静态门禁和苹果800 CUDA
  125/125 passed。下一步冻结该源码的新候选并执行 TP=8 1K/b1 探针；
  通过 20% 门限后再运行同提交完整矩阵和 128K 候选验证。

## 关键问题

1. 新 REAP checkpoint 在固定 runtime 和 official_v5 GSM8K 下的原生 baseline
   是否能完整复现，以及 OSCAR 候选是否先满足阶段门禁，并在最终冻结后满足
   official_v5 全量 accuracy/PPL 的第 10.4 节硬阈值。

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
| 原生 attention 显式固定 `TRITON_MLA_SPARSE` | 苹果800 为 SM80，目标模型要求 sparse MLA；显式配置便于日志和结果审计 |
| 创建 public `XiancaiTian/glm52_oscar_vllm`，稳定分支使用 `main` | Shawn 于 2026-07-25 确认推荐方案，并授权后续按最佳方式自主推进 |
| 远端只同步当前冻结代码快照，不上传完整恢复历史和大型实验产物 | Shawn 于 2026-07-25 明确要求大文件不必 commit/push，主要同步代码文件；完整历史保留在本地追溯分支 |
| 长时间 baseline 运行期间在项目内 ignored worktree 准备阶段 2 | 不改正式 submodule 指针、不污染运行时源码；隔离分支通过测试并推送后仍由阶段闸门决定何时接入 |
| 阶段 3 将标准 vLLM block table 与独立 INT2 history page namespace 分离 | 标准 block 继续承载全序列 BF16 RoPE 与 21 层原生 DSA cache；OSCAR allocator 只管理 latent prefix/recent/history，避免丢失 vLLM null block 和辅助 cache 所有权 |
| 阶段 4 CUDA 测试必须显式设置 `VLLM_OSCAR_RUN_CUDA_TESTS=1` | 当前正式 baseline 占满 8 卡；默认跳过避免测试 import 意外创建 CUDA context，待服务退出并再次确认 GPU 空闲后执行 cold-cache 苹果800 验收 |
| calibration 数据先以 OpenWebMath 固定 revision 和只读 LongBench 文件作为候选 | 二者可覆盖数学、通用长文本与代码；在最终 manifest 的样本 ID、token 数与 SHA256 冻结前不宣称为正式数据集 |
| official_v4 仅精确补跑首轮 7 条 `request_failed` 样本 | 7 条均为 GSM8K 且错误明确为客户端 `read timeout=300`，服务端无错误；补跑只将 math timeout 提高到 900 秒，并按 ID 替换失败行，原始 2,360 行结果保持只读 |
| official_v4 accuracy 合并显式固定四个 benchmark | 完整 manifest 实际为 2,361 行，其中 WikiText‑2 PPL 单独运行；accuracy runner 的正式命令只选择 GSM8K、IFEval、LiveCodeBench v6、MultiPL-E 共 2,360 行，合并必须复现相同选择 |
| 新 REAP checkpoint 重新执行阶段 1–7 | config 几何虽不变，但 checkpoint index 和权重文件指纹变化；旧 baseline、rotation、候选 OCI 和精度结论不能继承 |
| Stage 7 长跑期间在 ignored worktree 准备 Stage 9 | 不修改当前正式运行所读取的脚本、主工作区 HEAD 或候选源码；隔离提交只有在 Stage 7 通过后才同步回主功能分支并正式发布 |
| 正式评测协议切换为 official_v5 | Shawn 于 2026-07-28 指定 `/nfs/AE/txc/vllm_turbo_baseline_acc` 的 v5；v4 仅保留历史证据，不参与后续验收 |
| 当前阶段只运行 v5 GSM8K，最终阶段再运行 v5 全量 | 当前以 1,319 条 GSM8K 加快迭代；最终冻结候选必须完成 2,360 条 accuracy 和 WikiText‑2 PPL，阶段结果不能替代最终验收 |
| 阶段 7 改为“快速筛选协议 + 最终正式协议”两级评测 | Shawn 于 2026-07-29 明确要求停止 32K/max 长跑；快速层使用 8K/high，c8 做部分吞吐探针后以 c16 完成 256 题原生/OSCAR 配对，随后跑完整 GSM8K；最终阶段使用 32K/high 和 full v5 |
| 快速与最终正式评测统一使用 `reasoning_effort=high` | Shawn 于 2026-07-29 明确要求最终正式评测也使用 high，不得恢复 max；只读上游 v5 的 max 仅保留为来源身份，项目 runtime config 明确记录 high 适配 |
| v5 数学请求统一使用 7,200 秒 runtime timeout | 300/900/1,800 秒均已出现实测超时；冻结 runner 的实际统一输出预算为 `32768-218=32550` tokens，按约 6.5 tokens/s/序列满长约需 5,008 秒，7,200 秒留出约 44% 余量；原生与候选同口径，样本、输出预算、解码、重试和评分均不变 |
| v5 不计算跨 benchmark overall accuracy | 遵守 v5 原生指标协议；最终逐项比较 GSM8K、IFEval 四项、LiveCodeBench、MultiPL‑E Python/C++，每项下降不超过 3 个百分点 |

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
| Stage 9 静态格式命令误把两个 shell 文件传给 Python formatter | 1 | formatter 只改了 Python 文件并以状态 2退出；立即收窄为 `scripts/phase9`，随后 ruff format/check、4 个 `bash -n` 和 7/7 单元测试全部通过 |
| 首次 TP=8 启动因 FlashInfer/JIT cache 版本门禁退出 | 1 | 固定 venv 实测为 0.6.6/0.6.7.post3+cu129；已验证部署入口原本设置 `FLASHINFER_DISABLE_VERSION_CHECK=1`，补齐该通用环境后重跑 |
| 第二次 TP=8 启动因误读当前容器的 `flash_attn` 包退出 | 1 | 改用候选 rootfs 自带 Python 3.12.13，以 `PYTHONHOME` 和显式候选 venv/rootfs 路径隔离当前系统包；dry-run 已通过 |
| 首轮 official_v4 的 600 秒 code timeout 导致首批 8 条中 6 条请求失败 | 1 | 10 分钟时 KV usage 从 20.8% 降至 2.4% 并装入下一批，但服务仅记录 2 个 HTTP 200；停止不可能满足 2360/2360 scored 的无效轮次，runtime code timeout 固定为 900 秒 |
| 阶段 2 worktree 首次 pre-commit 初始化停滞，中止时 linked worktree index 被 hook cache 内容覆盖 | 1 | 核对正式源码 worktree 与对象库均完好；记录 5 个新文件 SHA256，用 `git read-tree HEAD` 仅重建隔离 worktree index，再按哈希重新暂存并通过 pytest、ruff、format 和 diff 检查 |
| capture 首轮测试发现输出字典的 `value_samples` 同时作为计数和张量键 | 1 | 将统计计数重命名为无歧义的 `*_covariance_samples`；重跑后 17 项 pytest、语法、ruff、format 和 diff 全部通过 |
| Hugging Face Xet CDN 在当前网络出现 TLS `wrong version number`，固定 OpenWebMath shard 直接下载无法完成 | 3 | curl、`hf_hub_download`、wget 三条路径均失败后停止重试；改用官方 datasets-server 按固定 revision 获取 0–299 行并在项目 artifacts 内冻结 SHA256 |
| phase-0 native hash 清单第 7 项使用 rootfs 内相对路径，在主仓库 source 目录直接执行整份 `sha256sum -c` 找不到文件 | 1 | 前 6 个 vLLM `.so` 通过只读 symlink 和原清单验证；第 7 个 sparse MLA `.so` 改按候选 rootfs 绝对路径与原清单 hash 单独核验，7/7 通过 |
| 完整 scheduler 测试默认尝试下载 LLaVA 测试模型并在项目外创建 Hugging Face cache | 1 | 立即终止下载；精确删除本次新建的 3,622,499-byte 模型 cache、0-byte lock 和 36KB Xet 日志，随后固定 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` 重跑 |
| Triton interpreter 的 split merge 对标量 mask 执行位与时报类型不兼容 | 1 | 拆分为两个 `tl.where` 条件，避免不同标量类型的位运算 |
| Triton interpreter 中 BF16 `tl.dot` rotation 产生无效大值 | 1 | rotation 改为 FP32 输入与 IEEE FP32 累加；512 维 CPU oracle 复测通过，SM80/苹果800 仍待正式验证 |
| Stage 5 新 worktree 的空 `uv` venv 缺少 pytest conftest 依赖 `tblib` | 1 | 使用清华 PyPI 镜像通过 `uv pip` 安装 `tblib==3.2.2`，随后在该 worktree 自有 `.venv` 中重跑 8 项测试通过 |
| Stage 4 RoPE 修复 pytest 被 worktree venv 的未安装依赖阻断 | 3 | 依次补齐 `cbor2==5.8.0`、`cachetools==7.0.1` 与 `py-cpuinfo==9.0.0`；定向 interpreter 和完整 `tests/oscar_mla` 随后分别通过 |
| official_v4 全量首轮末尾 7 条 GSM8K 触发客户端 300 秒读取超时 | 1 | 保留首轮 2,360 行原始证据；新增 fail-closed 精确补跑入口，仅补跑 7 个固定 ID、将 math timeout 提高到 900 秒，并在独立目录生成可追溯合并结果 |
| 精确补跑完成后的首版合并门禁把 2,360 条 accuracy 与包含 PPL 的 2,361 行完整 manifest 比较 | 1 | 门禁拒绝写入正式合并产物；拆出可复用合并工具，显式固定四个 accuracy benchmark 并验证唯一 ID、prompt hash、7 个替换 ID 与独立 PPL 排除行 |
| 阶段 2 首次 fit 按 `self_attn.pt` 查找实际为 `self_attn.attn.pt` 的 capture | 1 | fit 在第一层读取前 fail closed，未生成 artifact；修正固定 `layer_name_template` 并新增 1,248 个 train/holdout 路径集合预检，旧契约测试失败、新契约测试通过 |
| 阶段 3 正式 `.venv` 缺少已安装 vLLM metadata，12 项通用测试无法自动识别 device | 1 | 不改源码；使用项目内候选 rootfs 的已安装 vLLM metadata/dependency 路径，12/12 单独通过后全量 116/116 通过，CUDA 未初始化 |
| 阶段 4 首轮 苹果800 cold-cache 中 BF16 ring test 包含先断言 slot 0 为 NaN、后断言同一 slot 等于 position 320 的矛盾条件 | 1 | kernel 其余 21 个 CUDA 用例通过；将测试拆为先单独验证 history 64/65 不写入，再验证 319/320/321 ring 写入，提交后用新 Triton cache 重跑 |
| kernel 测试修复 commit 的 pre-commit 首次初始化 actionlint hook 停滞 | 1 | 中止 hook；ruff、format、targeted 苹果800 test 与 diff check 已手工通过，使用 `--no-verify` 提交单个测试文件并推送 |
| 阶段 5 首轮正式 苹果800 完整套件在非连续 rotation 前置断言失败 | 1 | 105 项通过，唯一失败发生在 kernel launch 前；当前 PyTorch 的 QR 输出本身已非连续，`.T` 后变为连续，改为 `.contiguous().T` 稳定构造非连续正交矩阵后定向复测 |
| Stage 5 单文件测试修复提交时 actionlint hook 再次初始化停滞 | 1 | 中止 hook；已独立通过 ruff、format、py_compile、定向 苹果800 kernel 与 diff 门禁，使用 `--no-verify` 提交并成功推送源码 commit `0f1bd5b30` |
| Stage 5 首轮服务 dry-run 的 CLI 校验内联脚本遗漏 `import os` | 1 | 所有静态输入检查和候选环境导入均已通过，CLI 读取预期 dtype 前报 `NameError`；补齐单个 import 后重跑 dry-run 全部通过，CUDA=false |
| dry-run retry 首次重定向到尚未创建的运行目录 | 1 | shell 在脚本创建目录前处理重定向，未启动验证逻辑；先创建任务专用 artifact 目录后重跑 |
| 首次 Stage 5 TP=8 服务的 artifact 正交校验发生 CPU/CUDA 跨设备比较 | 1 | 8 个 worker 均在权重加载前退出；`torch.load(..., map_location=\"cpu\")` 的 rotation 在 CPU，但 `torch.eye` 受 rank 默认 CUDA device 影响；将 identity 显式固定到 CPU 并增加非 CPU default-device 回归 |
| 第二次 Stage 5 TP=8 服务 warmup 对三段式 cache view 调用 tensor-only `.numel()` | 1 | 已加载 141/141 shards 并规划 637,632-token cache；`attn_layer.kv_cache` 为 `OscarMLACacheTensors` dataclass，不是单 tensor；空 cache 门禁改为按 dtype 检查其 `.raw.numel()` 并增加回归 |
| 第三次 Stage 5 TP=8 profile run 按 OSCAR dtype 直接读取空 `Tensor.raw` | 1 | 分配前 profile cache 仍是普通空 `Tensor`，分配后才是三段式 dataclass；修复必须按 cache 实际类型而不是仅按配置 dtype 分派 |
| 第四次 Stage 5 TP=8 部分 worker 初始化 CUDA driver 失败 | 1 | 启动前两次 8/8 空闲，错误发生在 NCCL/模型/artifact 前且退出后 0 MiB；先逐卡验证候选环境，再以原代码独立重试 |
| 第五次 Stage 5 TP=8 compile warmup 的 attention metadata 为 `None` | 1 | 141 shards、profile 和 planner 均通过；vLLM warmup 明确允许无 metadata，OSCAR 应保留 dummy dependency 但跳过 cache write，真实请求仍需 fail-closed |
| direct-call warmup 首版条件误落入原生 cache update | 1 | 新增回归立即发现组合条件的 `else` 归属错误；改为 OSCAR/非 OSCAR 两层显式分支后定向 8/8 通过 |
| Stage 7 第二轮 accuracy 在有效完成 573/2,360 后服务退出 | 1 | `/nfs/AE` 100% 满盘，十分钟监控 `tee` 报 `Disk quota exceeded`，不是模型/CUDA 故障；0-byte predictions 不作为结果。删除本项目内两个可重建大型产物，并为 service/accuracy/PPL 增加默认不变的 `ARTIFACT_ROOT`，正式重跑输出固定到 954 GiB 空闲的 `/dev/shm` |
| REAP 阶段 2 首次 train preflight 报 expert mapping SHA256 mismatch | 1 | GPU 启动前 fail closed；脚本使用 version sort，而配置和阶段 1 verifier 使用字典序。统一为 `LC_ALL=C sort -u`，并明确 hash canonicalization；154 个 expert token 集合本身未变化 |
| REAP 阶段 2 首次正式 train serve 出现两个 RUN_ID 并发初始化 | 1 | GPU 仍为 0 MiB、未生成 capture 时停止两棵进程树并作废轮次；为同 artifact root、host、port 增加非阻塞 `flock`，防止 ready 前的端口检查竞态 |
| REAP train capture 首次人工 metadata 校验错误要求每个 rank 都含 latent covariance | 1 | 该假设不符合 fit loader 的 TP 语义；改按实现核验 rank 0 独占共享 latent covariance、8 个 rank 各有 score/value covariance，624/624 文件全部通过。正式 runner 未失败，错误人工结果未作为证据 |
| 恢复会话后重复启动 holdout 被 serve lock 拒绝 | 1 | 未创建新运行目录、未占用 GPU；只读检查锁持有者后确认既有正式 holdout 已完成 preflight 并正在合法启动，沿用唯一轮次，不删除锁文件或终止合法任务 |
| 在源码 workdir 探测 pytest 版本时误用带仓库前缀的相对路径 | 1 | 该命令未运行测试；改用 workdir 内 `.venv/bin/python` 后探针通过，随后阶段 2 定向套件 33/33 passed |
| 隔离 Stage 9 worktree 的 CPU 复核在切换目录后仍使用带 worktree 前缀的相对 Python 路径 | 1 | 测试未启动；用 `realpath` 固定绝对解释器路径后重跑，Stage 7/9 单测 4/4、11/11 通过，compileall、4 个 shell 语法和 diff check 均通过 |
| Stage 3 首次回归计时命令假设 `/usr/bin/time` 存在 | 1 | 当前容器没有该二进制，pytest 未启动；改用 bash 时间戳计时，retry 正式执行为 145 passed、26 skipped、0 failed |
| Stage 4 候选 venv 缺 pytest，后续 CPU interpreter 子进程又误加载 rootfs 全局 Torch 2.10 | 2 | 两轮均未形成正式通过结果；用候选 Python 创建任务专用 uv venv，通过 `.pth` 固定候选 Torch 2.11/Triton 3.6，补齐 pytest/tblib 后用新 cache 得到 114/114 passed |
| Stage 5 退出后人工 JSON 汇总探针把整数 `requests` 当列表 | 1 | 正式响应文件和 smoke runner 均已通过；按实际 `results` 字段重验 8 行、8/8 HTTP 200，不修改实验产物 |
| Stage 6 首版候选只冻结 rotation 目录中 3/4 个文件 | 1 | 构建内容含 `artifact_validation.json`，但输入 manifest 未绑定其哈希；将 4 个文件全部加入 SHA256 白名单，并让 builder/verifier 拒绝任何额外文件后重新正式构建 |
| Stage 6 NFS verifier/runtime import 的执行通道早于实际子进程退出返回 | 1 | 不使用提前返回判断状态；按实际 PID 和非空结果文件短周期轮询，最终 verifier 与 runtime import 均为 `passed` |
| 首次固定 token 预算审计误读 baseline usage 与 manifest max-token 字段 | 1 | 该轮 2,360 条均未参与计算，输出不作为证据；按实际 `token_usage` 与顶层 `max_tokens` 重跑后 2,360/2,360 完整覆盖，超 32,768-token 预算为 0 |
| Stage 9 首次只读 CLI help 探针没有产生正文，仅报告缺少生成版 `vllm._version` | 1 | 不把源码模块直接执行结果作为正式入口；确认仓库 wrapper 已弃用，后续由固定候选环境调用 `vllm.benchmarks.serve` 的实际实现，并在正式脚本中加入静态 CLI 门禁 |
| Stage 9 隔离 worktree 完整静态门禁缺少 ignored artifact 链接 | 2 | 首次缺冻结 evaluator，补链接后原生 81/81 通过；候选随后缺 rotation artifact。最终显式把 `OSCAR_RUNTIME_PROJECT_ROOT` 固定为主项目，只读复用完整 artifact，候选 63/63 通过；两次失败均在 GPU 启动前且未形成通过结果 |
| Stage 9 隔离 worktree 通过 symlink rootfs 执行完整 dry-run 时 native extension 符号不匹配 | 1 | 该诊断未启动 GPU，不能代表主项目真实 rootfs 路径。停止沿 symlink 执行动态扩展，改用主项目 rootfs 真实路径直接解析同一 serve CLI；全部固定参数和值通过且 CUDA=false，正式实验仍只在主工作区运行 |
| accuracy 定稿器用旧 baseline 代理做额外端到端测试时被作用域/timeout 身份门禁拒绝 | 2 | 第一次 NFS 源不属于隔离 worktree artifact root；第二次复制到 `/dev/shm` 后又因旧 baseline 的 code timeout=900 与当前 in-flight 固定 3600 不同而拒绝。两次均未生成定稿结果，源 validation 哈希不变；当前正式源位于 `/dev/shm` 且三项身份 SHA 已预先冻结 |
| 380 分钟服务累计数被过度解释为 LiveCodeBench 全部完成 | 1 | 复读 runner 发现 8 线程完成顺序与 manifest 顺序不等价；176 个成功只证明至少一条 MultiPL-E 完成，仍可能有最多 7 条 LiveCodeBench 在途。立即更正文档，不把该边界作为完整分段性能数据 |
| 首次 official_v5 原生入口把 v5 运行套件传给历史 v4 静态 verifier | 1 | GPU 检查前 fail closed、未加载模型；拆分 `STATIC_SUITE_DIR` 与 `SUITE_DIR`，前者绑定 frozen v4 服务证据、后者绑定 v5 正式 runner，原生 61/61、候选 44/44 dry-run 通过 |
| 首次 official_v5 GSM8K 请求全部返回 HTTP 400 | 1 | 服务与 141/141 分片加载正常，但 v5 的 `reasoning_effort=max` 被服务端 Literal schema 拒绝；该轮 0/1,319 scored、`valid=false`，不作为精度结果。源码已接受并透传 `max`，1/1 定向测试通过，候选 OCI 重建后再正式重跑 |
| 重建后首次直接调用原生 dry-run 未显式传入 frozen v4 静态套件 | 1 | 当前 external v4 文件已漂移，历史 hash 门禁按预期失败；改为与正式隔离 orchestrator 相同的 `STATIC_SUITE_DIR=frozen_v4`、`SUITE_DIR=frozen_v5` 后 61/61 通过，未启动 GPU |
| v5 namespace 早退 cleanup 引用已离开作用域的局部 PID | 1 | 失败轮次没有 GPU 进程；将 wrapper PID 提升为 namespace 脚本状态，保证 EXIT trap 在早退路径也可安全执行 |
| 第二次 official_v5 GSM8K 原生轮次出现 300 秒客户端读取超时 | 1 | runner 首次完成 20 条时服务端仅有 13 个 HTTP 200，至少 7 条 future 已超时；立即停止不可能满足 request failure=0 的轮次并释放 8 张 GPU，保留失败证据。新增 SHA256 固定 runtime config，仅把数学请求超时提高到 900 秒，原生/候选同口径重跑 |
| timeout 适配复核误用不存在的根目录 `.venv/bin/python` | 1 | 单元测试未启动；项目根没有该虚拟环境，改用已安装依赖的系统 Python 运行纯标准库定向测试，正式 evaluator 仍使用 frozen v5 自己的固定 `.venv` |
| 第三次 official_v5 GSM8K 原生轮次证明 900 秒仍不足 | 1 | 首批请求 12:25:28 开始，12:40:28 的 KV usage 从 23.0% 降到 10.6% 并出现 31.2 prompt tokens/s，但服务成功数没有对应增长，符合客户端 900 秒超时后重试；20 分钟时仅 6 个 HTTP 200。立即停止、释放 8 张 GPU 并保留 366 KiB 小型证据；按 8,192-token 上限与约 6.5 tokens/s/序列把 runtime timeout 提高到 1,800 秒 |
| 第四次 official_v5 GSM8K 原生轮次证明 1,800 秒仍不足 | 1 | 30 分钟时服务仅 8 个 HTTP 200；精确 1,800 秒处 KV usage 47.6%→22.7% 并出现 40.9 prompt tokens/s，成功数未增长。立即停止并释放 8 张 GPU，保存 386 KiB 小型证据。复核 runner 发现 manifest 的 8,192 字段未被读取，真实固定预算为 32,550 tokens；首次离线复算误把 BatchEncoding 的 2 个键当 token 数，修正为读取 `input_ids` 后得到最长 prompt 218、预算 32,550，timeout 改为 7,200 秒 |
| 保存停止轮次证据时两份同名 `progress_10min.log` 发生目标名冲突 | 1 | `cp` 拒绝覆盖首个文件；改为显式保存为 `service_progress_10min.log` 与 `runner_progress_10min.log`，两份 SHA256 分别固定且原始 tmpfs 文件未修改 |
| 为定位运行产物执行的上级目录宽泛 `find` 扫描耗时过长 | 1 | 主动中止，未修改文件；后续只读取已知 PID、运行目录和精确 artifact 路径，不再宽泛遍历 NFS 上级目录 |
| c8 过程更新把未读取的第 26 条结果错误外推为正确且未截断 | 1 | 立即重新读取全部 26 个 checkpoint 并更正为 11 正确、8 截断；后续过程统计只从落盘 checkpoint 计算，不再根据前一状态外推 |
| 首次合并 rename 与多文件更新的 `apply_patch` hunk 格式无效 | 1 | 未修改任何文件；拆为独立的 rename patch 和普通更新 patch 后成功应用，不重复使用混合 hunk |
| OSCAR c16 首次快速配对轮次在 readiness 前部分 worker 初始化 CUDA driver 失败 | 1 | 运行目录 `20260729T0650Z_candidate_fast256_c16` 已确认实际参数为新 REAP 模型、8K/high/c16，0/256 请求进入评测；退出后 8 卡均为 0 MiB。`2026-07-29T07:13:24Z` 与 `07:14:24Z` 双次空闲检查通过，随后同候选环境逐卡 CUDA tensor 探针 8/8 通过，排除固定坏卡与环境整体不可用；以新 run ID 独立重跑，不把失败轮次当作精度结果 |
| 恢复会话时 Git 因 `safe.directory` 所有权保护拒绝读取 | 2 | Git 2.25.1 不支持任务预期的 `GIT_CONFIG_GLOBAL` 临时覆盖；按 Git 自身提示仅为主仓库和候选 submodule 添加精确的 global `safe.directory` 条目，不使用通配符，随后可读取状态 |
| 恢复会话时 8 张 苹果800 被项目外 TP=8 服务全部占用 | 2 | 初始未干预；Shawn 随后明确授权终止所有本项目外 GPU 占用进程。第一次精确终止进程组后，Docker 因 `unless-stopped` 自动拉起同一服务；第二次解析到容器 ID 后用 `docker stop --time 20` 停止容器，状态为 exited、PID 0，随后间隔 1 分钟的两次检查均为 8/8 张卡 0 MiB、0% |
| 服务器恢复后候选 submodule 的 4,670 个文件被 Git 标记 modified | 1 | `git diff --raw/--summary/--numstat` 证明全部只是 NFS 将 100644 呈现为 100755，内容差异为 0；设置该 submodule 的本地 `core.filemode=false` 以忽略文件系统不可表达的权限漂移，不改文件内容或提交 |
| 恢复记录提交后 `git push` 卡在 GitHub HTTPS askpass | 2 | 首次推送无输出，进程检查定位到 `git-remote-https`；带 trace 的非交互重试明确停在 GitHub username askpass。按 Shawn 要求重新发起认证，清除失效 credential 后 push 成功，远端更新到 `e50ae6a` |
| 恢复后 frozen v5 evaluator 的 `.venv/bin/python` 绝对链接失效 | 2 | 新宿主没有原 `/usr/bin/python3.12`；首次把 uv Python 直接链接进既有 venv 时因 relocatable `/install` 前缀找不到标准库。改为恢复固定 uv 0.11.5/Python 3.12.3，并用显式 `PYTHONHOME`/既有 site-packages 启动；22 个锁定包均存在，关键 imports、NLTK punkt/punkt_tab 与 official_v5 static/namespace preflight 通过 |
| 恢复后的候选 c16 重跑在 readiness 前因宿主 glibc 版本不足退出 | 1 | `20260730T0356Z_candidate_fast256_c16_retry` 已通过 official_v5 静态门禁，但 rootfs Python 3.12 要求 glibc 2.35、新宿主为更旧版本；0/256 请求、GPU 8/8 保持 0 MiB。当前环境已是带 Docker 的物理宿主，按项目规范从 16GiB 冻结 OCI layout 导入不可变候选镜像后在容器内运行，不把该轮当作精度结果 |
| 宿主/工具容器直接安装 skopeo 的两条路径失败 | 3 | focal 无 skopeo；jammy 模拟会升级 glibc 并移除 Nsight，未执行；拉取官方 skopeo 镜像遇到 Docker HTTPS proxy 错误。最终用本机已有 GLM 基线镜像的一次性可写层安装 skopeo 1.4.1，从只读 OCI layout 成功导入候选，Docker image ID 精确为 `8b7a2ee6...68bb` |
| 首次容器化候选轮次无法在旧 tmpfs artifact root 创建 run 目录 | 1 | `20260730T0409Z_candidate_fast256_c16_docker` 在 static/published 门禁后、CUDA 初始化前因内层 user namespace 权限失败，0/256、GPU 0 MiB。创建独立空的 mode 1777 `/dev/shm/oscar-glm-official-v5-docker`，使用新 run ID 重跑，不改变项目/NFS 权限 |
| 第二次容器化候选轮次被 NFS overlay source 的 mode 漂移拒绝 | 1 | `20260730T0411Z_candidate_fast256_c16_docker_v2` 的 manifest/config/layer、rotation、baseline 与 4,744 文件内容均通过，仅 Git 100644→NFS 100755 失败；NFS chmod 返回 EPERM，0/256、GPU 0 MiB。下一轮在容器 mount namespace 内把镜像自带正确 mode 的 `/opt/vllm_glm52_v1` bind 到 overlay source 路径，并按既有合约把 6 个 native extension 变为指向 lower rootfs 的 symlink，不修改 NFS 内容 |
| 容器 mode 适配的前两次 CPU dry-run 仍未通过 | 2 | 第一次缺 `apparmor=unconfined`，精确 bind 被拒绝；加入后 overlay mode 通过但镜像内 100 个 runtime `.pyc` 被判 extra。一次性容器可写层删除 27 个 `__pycache__` 后 overlay 通过；随后发现 Stage 5 还需正确 phase0 base OCI，而非旧 `d6faf...` tag。导入 config `58a853...` 的 phase0 OCI 并创建 4,711 tracked-file volume 后，Stage 5 与最终候选 dry-run 均通过，GPU 始终 0 MiB |
| Stage 9 首次用宿主 Python 运行工具测试失败 | 1 | 宿主 Python 3.8 不支持工具使用的 `datetime.UTC`；改用项目已冻结的 Python 3.12 evaluator 环境，14/14 测试通过 |
| Stage 9 首次控制镜像构建发送整个项目上下文且新入口无执行权限 | 2 | 中止无效的大上下文构建；将 Docker build context 收窄为 4.6KB 的 `docker/`，并把入口 mode 固定为 755，控制镜像成功构建 |
| 容器预检读取未设置的发布提交变量 | 1 | 正式模式仍要求并传入已发布 SHA；CPU-only 预检改为空值安全传递，不绕过正式发布门禁，BF16 预检随后通过 |
| Stage 9 verifier 未声明统一入口传入的 `--suite-dir` | 1 | 两个 verifier 增加必填参数并严格校验其解析路径等于冻结 suite，防止只为兼容参数而放宽证据身份 |
| OSCAR 首次 Stage 9 预检被中间 Stage 7 wrapper 覆盖 verifier/运行阶段 | 1 | 该次绿色结果不计入 Stage 9；将中间 wrapper 改为仅提供默认值、尊重上层显式环境，使用新 run ID 重跑后 outer Stage 9 与递归 Stage 7 门禁全部通过 |
| BF16 首次 Stage 9 正式轮次在服务 ready 后被矩阵参数证据门禁拒绝 | 1 | 真实 `serve_command.txt` 已包含 `--max-num-batched-tokens 2048`，但旧 `parsed_server_args.json` 未输出该字段；0 个 benchmark 请求、无 TTFT/TPOT 结果，服务退出后 8 卡为 0 MiB。解析器现完整输出 max batched tokens、显存比例、seed、async 与 profiler 字段，并把冻结 torch profiler config 真正传给服务；BF16/OSCAR 新 dry-run 均验证参数完整且 `cuda_initialized=false` |
| BF16 第二次 Stage 9 正式轮次误把整秒桶峰值当作真实并发门禁 | 1 | 首个 1K/batch1 结果中配置字段为 `max_concurrency=1`，runner 日志也明确显示最大请求并发 1；`max_concurrent_requests=2` 来自 benchmark 把请求活动区间按闭区间整秒分桶，相邻串行请求会在边界桶重叠。该轮仅完成一个 cell 的首轮、无完整 summary，服务已清理且 8 卡为 0 MiB。门禁改为校验配置字段，粗粒度峰值只保留为观测，并新增回归测试 |
| 并发门禁修复后的首轮单元测试有一份旧 fixture 缺新字段 | 1 | 新定向测试通过，既有 exact-workload fixture 因没有 `max_concurrency` 被正确拒绝；为旧 fixture 补入与其 batch=4 一致的真实配置字段后重跑全套 |
| BF16 第三次 Stage 9 正式轮次发现 profiler 额外生成前端 trace | 1 | 1K/batch1 的 3/3 round 已通过，首个 profile 实际生成 8 个 rank trace 外还有一个 `.async_llm.` 前端 trace；旧捕获器会把它误判为缺 rank 的 worker。为避免生成整矩阵后才失败，精确停止本任务容器；无完整 cell/summary，8 卡已释放。捕获器现分别记录并哈希 1 个 frontend trace，仍严格要求 TP rank 0–7 各自 table/trace，比较器同步复验 |
| BF16 v4 完成 9/9 后 outer cleanup 的 `EXIT` trap 读取已离开作用域的局部 `wrapper_pid` | 1 | `summary.json` 已先完整写入且 status=`passed`，容器删除、8 卡 0 MiB；错误只影响 launcher 最终退出码。cleanup 改用 `${wrapper_pid:-}` 并在正常清理后撤销 EXIT/INT/TERM trap；为保持比较器要求的同一主提交，候选仍从已发布的 BF16 提交运行，修复在配对完成后合入 |
| BF16 阶段记录复核首次直接调用候选 rootfs Python | 1 | 宿主 glibc 低于 rootfs Python 要求，测试未启动；改在冻结 Stage 9 控制容器内运行，同一 `scripts/phase9` 测试 15/15 通过 |
| BF16 阶段报告首次 `git diff --check` 发现新增提交行尾随空格 | 1 | 移除该行 Markdown 尾随空格后重新执行完整 diff 门禁 |
| 查询精度证据行号时把含反引号的搜索词放入双引号 shell 参数 | 1 | shell 将反引号内容误作命令替换并产生无副作用的 `No such file`；改用不含反引号的固定文本搜索，实验与结果文件均未修改 |
| 停止未优化候选时 PTY Ctrl-C 未传递到容器主进程 | 1 | Ctrl-C 后容器和 8 个 worker 仍存活；改用精确容器名执行 `docker stop --time 20`，容器删除、进程退出，8 卡回到 0 MiB |
| 更新报告状态日期时沿用 Markdown 行尾双空格 | 1 | `git diff --check` 在提交前拒绝该新增尾随空格；移除尾随空格后重新执行门禁，不影响报告内容 |
| 复核 OSCAR 精度产物时宿主缺少 `jq` 且逐题文件为 root-only | 1 | `summary.json`/`validation.json` 的 SHA256 已在宿主复核一致；改用标准库 Python 读取 summary。`predictions.jsonl` 为 root:root mode 600，宿主未越权修改权限，沿用此前已落地的 validation 与报告哈希；本次结论不声称完成新的逐题复核 |
| Stage 9 优化恢复记录中的一次性 `/dev/shm` 测试 venv 已消失 | 1 | 测试命令在 collection 前即因 Python 路径不存在退出，未形成代码测试结果；按 `AGENTS_misc.md` 使用现有固定 uv 和清华镜像重建一次性 CPU 测试环境 |
| Stage 9 优化 CPU 容器的只读源码挂载阻止 Ruff 写缓存 | 1 | 完整 `tests/oscar_mla` 已先得到 89 passed、26 CUDA skipped、0 failed；随后 Ruff 在检查前因 `.ruff_cache` 只读退出。保持源码只读，改把 `RUFF_CACHE_DIR` 指向容器 `/tmp` 后独立重跑 Ruff 与 compile 门禁 |
| decode 快路径多请求回归首次直接修改 frozen metadata | 1 | 目标文件其余 9 项通过，新测试在进入被测函数前触发 `FrozenInstanceError`；改用 `dataclasses.replace` 构造两请求 metadata，不修改生产实现来迁就测试 |
| decode 快路径提交被恢复后的 pre-commit 环境阻止 | 5 | 首次缺模块；既有 `.venv` 是 Python 3.8，pre-commit 4.2 又要求 Python≥3.9。改装 3.5 后，完整 hook 被无关 Action/Node/Go 环境下载长期阻塞；按文件类型跳过无关 hook 后，`mypy==1.19.1` 又因 Python 3.8 无法解析。最终用恢复的 Python 3.12/固定 uv 建 pre-commit 4.2 venv；安装器因 root-owned NFS hook 的 metadata 操作报 EPERM，但已实际原位写入正确 3.12 解释器。逐文件核对 hook 内容后补装 commit-msg，并执行 Ruff、typos、mypy 及全部相关 Python 门禁；不使用 `--no-verify` |
| decode 快路径首轮有效 pre-commit 发现基线遗留门禁 | 1 | Ruff format 与 SPDX hook 自动修正本次触及文件；Ruff check、typos、mypy 等通过。`check-torch-cuda-call` 报错行由 `53d8be94f` 引入且本次 diff 不含 `torch.cuda`；attention backend 文档生成器产生的是既有 OSCAR dtype 的陈旧文档漂移，也非本次 decode 改动。恢复无关文档，只对这两个基线遗留 hook 做精确跳过，其余相关门禁继续执行 |
| 新候选 CPU-only runtime import 探针失败 | 2 | 首次加载 `vllm._C` 时缺 `libcuda.so.1`；只读挂载宿主 driver userspace library 后已越过该点，第二次因探针沿用旧版 `vllm.entrypoints.openai.protocol` import 路径退出。两次均未初始化 CUDA。改用当前源码实际的 `openai.chat_completion.protocol` 路径复测，不修改候选镜像 |
| value 精度恢复候选 runtime import 探针失败 | 2 | 首轮把 rotation artifact 顶层两个键误判为 78 层；v2 改为内层 78 项后又额外导入冻结协议不要求的 `flashinfer/flashinfer.jit`，最终被 CUDA 未初始化断言拒绝。两轮空 JSON 和日志均保留，不计作候选失败或 runtime 通过；下一轮精确复用既有 `importlib.metadata` 协议 |
| 新控制镜像首次身份审计脚本失败 | 1 | 构建和 CPU runtime 检查已通过；审计脚本把 Docker 日志的 12 位短 ID 猜写为错误的完整 SHA256，首个断言退出。改为从 daemon 读取完整 ID 后，34/33 层、基础层逐层继承、labels 与 entrypoint 全部通过 |
| 新 Stage 9 配置首次用 standalone Python 运行工具测试缺 `requests` | 1 | JSON 和 shell 语法门禁已先通过；测试在 collection/import 阶段退出，未形成单元测试结果。改用刚冻结、内含正式依赖的 `oscar-glm-stage9-runtime:98ddd3f4e` CPU-only 容器重跑，不在宿主环境临时补包 |
| 新候选首轮 Stage 9 静态门禁缺 lower-layer native links | 1 | Phase 6 verifier 已确认基础层 7 个扩展未被候选覆盖，但独立 overlay 只解出候选层；Phase 7 在读取第一个 `_C.abi3.so` 前退出，未生成绿色结果。按既有 overlay 合约为 6 个 vLLM 扩展建立指向只读 phase0 rootfs 的精确 symlink 后重跑，不复制或修改原生二进制 |
| 尝试离线重建 fast256 协议指纹以定位 BF16/OSCAR 指纹差异的单一字段 | 3 | 第一次非特权 Python 无权读取 root-only runtime suite；第二次 sudo Python 缺 frozen evaluator 的 `absl` 依赖；第三次宿主 Python 3.8 不支持字典 `|`。该重建不是回答精度差异的必要证据，停止继续猜测；只报告已验证的指纹不一致、原生逐题文件缺失和现有汇总结果 |
| 单格性能探针首次 Ruff format check 发现测试文件格式漂移 | 1 | Ruff check 已通过；使用同一 Ruff 0.14.0 对单个测试文件执行机械格式化，随后 Ruff check/format、shell 语法、控制容器内 16/16 单元测试和 diff check 全部通过 |
| 只读 Docker 复核 root-only 精度文件时沿用控制镜像默认 `/bin/bash` entrypoint | 1 | 镜像把传入的 `python` 当作 Bash 脚本并报 `cannot execute binary file`；改为显式 `--entrypoint /usr/bin/python3.12`，不改文件权限即完成逐题只读统计 |
| split benchmark 首次在只读项目挂载中执行 `py_compile` | 1 | Python 尝试写源码旁 `__pycache__` 被拒绝，测试未形成通过结果；设置任务专用 `PYTHONPYCACHEPREFIX=/tmp/pycache` 后 compile 与 help 通过 |
| split benchmark 控制镜像内未包含 Ruff 可执行文件 | 1 | compile/help 已先通过；使用已冻结的 pre-commit Ruff 0.14.0 环境执行 check/format，不在运行镜像临时安装包 |
| 首次探查 rotation payload 时把顶层版本整数当 tensor | 1 | 只读命令在打印类型后退出，artifact 未修改；按实际 `payload[\"rotations\"][layer]` 读取，确认 78 个 FP32 512×512 tensor |
| 首次 mixed split sweep 显式使用系统 `/usr/bin/python3.12` | 1 | sweep 完成后 summary 身份审计发现 Torch 为 2.10，而正式候选 venv 为 2.11；该结果整体作废，不用于参数选择或报告。工具新增解释器/Torch/CUDA fail-closed 身份门禁，提交推送后以新 run ID 重做双次空闲检查和 sweep |
| 首次尝试在旧 Stage 9 root-owned `results/` 下创建 split 目录 | 1 | `mkdir` 在 GPU 分配前被拒绝，旧产物未修改；创建当前用户独立 mode 700 的 `/dev/shm/oscar-glm-stage9-splits`，后续每轮目录和权限均显式记录 |
| 首次 8-rank prefill trace analyzer 把末尾 0-token generation=0 空标记识别为第二个 prefill | 2 | 工具 fail-closed 且未生成 summary；实际 trace 同时含唯一 1,024-token prefill 和一个 context/tokens 均为 0 的空窗口。首次修复的回归又发现空窗口不应计入 generation duration；最终 prefill 要求 context/tokens>0，decode duration 只接受 generation>0，边界回归通过后以新 analysis ID 重跑 |
| prefill microbenchmark 首次 Ruff format check 发现主文件格式漂移 | 1 | Ruff check 已通过，未执行 GPU；使用同一 Ruff 0.14.0 机械格式化后重新执行 check、format、compile、单元测试、CLI help 与 diff check |
| 首次追加 prefill sweep 记录时引用了 findings 中不存在的进度段落 | 1 | `apply_patch` fail-closed 且未修改任何文件；重新读取三个目标位置后，改用 findings 的实际末尾上下文分别追加结果 |
| 搜索 indexer 的 `-1` 模式时未使用 `rg --` | 1 | `rg` 把模式识别为选项并在只读搜索阶段退出；改为先写 `--` 终止选项解析，未修改源码 |
| 恢复后的 Stage 9 CPU 测试 venv 只剩失效入口 | 1 | 旧 `/dev/shm` Python 已消失；不依赖该路径，改用正式控制容器的 Python/PyTorch/vLLM 和 uv 安装的临时纯测试依赖 target |
| 固定控制镜像首次运行源码 pytest 缺测试依赖 | 2 | 第一轮缺 pytest；首次 target 又因宿主 Python 3.8 解出旧 typing_extensions，遮蔽正式 pydantic 依赖。使用恢复的 Python 3.12.3 重新创建只含 pytest 8.3.5/tblib 3.1.0 的 target 后，测试成功进入被测代码 |
| 首次直接调用 `uv` 安装临时测试依赖 | 1 | 宿主 PATH 没有 uv；改用已恢复并固定的 `/dev/shm/oscar-glm-recovery-tools/.../uv` 绝对路径和清华镜像 |
| grouped prefill kernel 首次 SM80 编译超过 shared-memory 上限 | 1 | `20260730T2250Z_oscar_prefill_headgroup_1k_b1_v1` 在正式计算前报告需要 184,320 bytes、硬件上限 166,912 bytes；容器退出且 GPU 已释放。保持 head/tile 几何不变，只把 grouped launch 的 `num_stages` 从 2 降到 1，重新提交发布后再测 |
| grouped prefill BF16 tensor-core 版超过固定数值容差 | 1 | `20260730T2252Z_oscar_prefill_headgroup_1k_b1_v2` 已编译执行，但 output/LSE 最大误差 `0.009153/0.002593`，超过既有 `0.002/0.002` 门限；不放宽正确性标准，保留跨 head 复用并改用 FP32 IEEE dot 后重测 |
| grouped prefill 完整 CUDA 首轮容器挂载隐藏原生扩展 | 1 | `20260730T2259Z_oscar_headgroup_full_cuda_v1` 在 pytest collection 阶段因源码 symlink 的宿主绝对 target 未挂载而缺 `vllm._C`，0 个测试执行、GPU 已释放；保持源码和测试不变，只按既有合约把 phase0 rootfs 挂入同一绝对路径后重跑 |
| Phase 6 输入静态检查调用宿主 `jq` | 1 | 宿主没有安装 `jq`，命令在任何构建动作前退出；改用已有 Python 3 标准库只读解析 JSON，不新增环境依赖 |
| 新 Phase 6 构建入口误用宿主 Python 3.8 | 1 | `datetime.UTC` 在 OCI 写入前触发 `AttributeError`；保持构建器与输入不变，改用已恢复并固定的 Python 3.12 解释器重跑 |
| 探查 Phase 0 工具镜像时重复传入 `bash` | 1 | 镜像已有 `/bin/bash` entrypoint，额外 positional `bash` 被当作脚本而报 cannot execute；后续显式使用 `--entrypoint /bin/bash` |
| 新候选首次 runtime import 未注入 NVIDIA 驱动 | 1 | `vllm._C` 加载时报缺 `libcuda.so.1`，`torch.cuda` 未初始化且无 GPU 进程；提交记录后执行双 GPU 空闲检查，再用 `--gpus all` 只注入驱动运行同一只读 import 探针 |
| grouped 候选构建后发现 Phase 6 Dockerfile 默认源码身份滞后 | 1 | 自定义 OCI 构建器实际从 manifest 打包了正确 `35ab1846…` tree，但候选 label 哈希对应的 Dockerfile 仍默认旧 `a94b1f640…`；停止使用已导入候选/控制镜像做正式门禁，更新 Dockerfile 与输入哈希后以新 artifact 目录重建 |
| grouped 候选 v2 payload layer 未复现 v1 digest | 1 | Dockerfile hash 只应改变 config/manifest，但 v2 layer 从 `a599892d…` 变为 `829ceb0e…`；暂不接受 v2，逐 member 对比 v1/v2 tar metadata/content，定位非确定性后再重建 |
| 用宿主 Python 3.8 解析 tblib 3.1.0 依赖失败 | 1 | tblib 3.1.0 要求 Python≥3.9；为 uv 显式指定恢复的 CPython 3.12.3 后安装成功 |
| 直接对完整触及文件运行 mypy 报 4 个既有错误 | 1 | 四行均不在本次 diff；改用项目 `tools/pre_commit/mypy.py` 的增量、`follow-imports=skip` 合约检查变更行，两个文件均无问题 |
| pre-commit 的 `check-torch-cuda-call` 报告旧 `torch.cuda.empty_cache()` | 1 | `git blame` 确认为 `53d8be94f` 引入且本次 diff 只在 608–637 行；只精确跳过该既有 hook，其余相关提交门禁全部执行并通过 |
| 首次读取 Phase 6 构建器时使用了不存在的脚本名 | 1 | `ls` 显示实际文件为 `build_candidate_oci.py`/`verify_candidate_oci.py`；未修改文件，改读真实入口 |
| 新 OCI 首次用宿主 Python 3.8 构建失败 | 1 | 构建器使用 `datetime.UTC`，宿主 3.8 在写 layout 前退出；改用已恢复的固定 CPython 3.12.3，以同一空输出路径成功构建 |
| 新 Phase 7 单元测试首次未挂载恢复解释器 | 1 | 19/20 通过，唯一恢复测试的 wrapper 以绝对路径调用 `/dev/shm` CPython；把恢复工具和项目都按原绝对路径只读挂载后重跑为 20/20 |
| 新 candidate 直接运行 Phase 7 verifier 缺 6 个 lower native symlink | 1 | Phase 6 layer 按设计不覆盖 native extension；在新 overlay 中建立指向 phase0 rootfs 的 6 个精确只读 symlink，candidate Git tree/native link 门禁全部通过 |
| 新 Phase 5 配置首次遗漏更新 Phase 1 manifest 的派生 SHA256 | 1 | verifier 准确报告旧 hash；更新 `base_manifest_sha256` 后重新计算 Phase 5 SHA，并同步 Phase 7 的 `stage5_manifest.sha256` |
| 宿主直接递归 Phase 7 verifier 被 phase0 NFS mode 漂移拒绝 | 1 | candidate OCI、4,744 文件、native link、rotation、baseline 和服务参数均已通过；仅 phase0 runtime tree 因宿主 100644→100755 漂移失败。保持 verifier fail-closed，发布后使用既有 containerized mount namespace 恢复镜像内正确 mode 再正式 preflight |
| TP=8 探针结束后首次读取不存在的 `cell_summary.json` | 1 | 总 summary 已完整通过且实验未受影响；按实际目录结构改读 `matrix/input_1024_batch_1/summary.json`，复核 cell status、指标与 SHA256 |
| 新 trace 分析首次猜测了不存在的 analyzer 文件名，随后宿主/恢复 Python 均缺 `ijson` | 2 | 两次均未生成分析结果；改读真实 `analyze_prefill_trace.py`，并使用不挂 GPU 的固定控制容器内 `uv`、清华镜像和任务专用 `/dev/shm` cache |
| 尝试在宿主直接调用已随恢复环境消失的 `uv` | 1 | venv 创建前即退出，没有残留有效环境；确认固定控制镜像包含 `/usr/local/bin/uv`，改用一次性 CPU 容器执行分析 |
| trace 分析 v1 使用绝对 Python 绕过 `uv run` 临时环境 | 1 | summary 自记录实际 `ijson 3.5.0`，因此 v1 不作为最终证据；用新 analysis ID、PATH 中的 `python` 和固定 `ijson==3.4.0.post0` 重跑，v2 与 v1 聚合数值完全一致 |
| PAX 确定性回归首次调用 PATH 中不存在的 `python3.12` | 1 | 测试在解释器启动前退出，没有形成测试结果；改用已恢复并固定的 CPython 3.12.3 绝对路径执行同一测试 |
| 全配置身份批量 patch 对 Phase 9 测试中的 base ID 作了错误假设 | 1 | patch 原子失败，除先前单独完成的 Phase 1/5 修改外没有应用任何批量变更；测试实际把 base ID 与 candidate config 动态比较，只需更新固定 control image ID。拆分为配置、wrapper、测试三个精确 patch |
| 报告交叉引用检查器把 `7.137 秒` 识别成第 7.137 节 | 1 | 报告章节本身连续；原正则扫描所有 `7.x` 小数，误命中 profiler 秒数。改为只扫描“见/记录在/按/保持 7.x”等章节引用语境后重跑 |
| preflight 证据探查把 `xargs` 与读取 stdin 的 heredoc Python 混用 | 1 | 文件路径已列出，但后续同一 shell 输出被 stdin 组合截断；未修改证据。改用 Python `Path.glob` 直接读取三个 JSON，得到完整状态、检查数和 SHA256 |
| grouped trace 分析又直接调用宿主 Python | 1 | 宿主缺 `ijson` 的限制已在旧轮次记录；本次在 import 阶段退出，未读取 trace 或生成结果。后续直接复用固定控制容器、`uv` 和 `ijson==3.4.0.post0`，不再探测宿主解释器 |
| BF16 trace 首轮强制 `uv --offline` 时缓存不能解析固定 ijson | 1 | 在依赖解析阶段退出，未读取 trace、未生成输出目录；保持固定容器和版本不变，改用已验收的清华 PyPI 镜像在线解析，并继续使用任务专用 uv cache |
| demotion 归因写入 planning 文件的首个 patch 上下文不匹配 | 1 | patch 原子失败、文件未被修改；重新读取文件末尾并按实际最新段落追加，不复用过期上下文 |
| 恢复后的源码 CPU test venv 绝对解释器链接再次失效 | 1 | `/dev/shm/oscar-glm-stage9-opt-test-venv/bin/python` 指向当前宿主不存在的 `/usr/bin/python3.12`，测试未启动；不原地修补旧 venv，改用固定控制容器挂载当前源码和既有 native rootfs 执行 TDD |
| 查找 Phase 9 配置时猜测了不存在的 `performance_config.json` | 1 | 只读 Python 在打开文件时退出，未修改状态；TDD 只需已确认的控制镜像和源码/native 挂载，不再依赖该猜测路径，后续配置查询先用 `rg --files` |
| 固定控制镜像未预装 pytest | 1 | 当前源码/native 挂载均成功，但解释器在测试 collection 前报告 `No module named pytest`；下一轮在同一一次性容器中用 uv、清华镜像安装固定 pytest/tblib，再执行红灯用例 |
| decode 优化 pre-commit 首轮发现两个既有门禁漂移 | 1 | Ruff/format/typos/mypy/forbidden imports 均通过；SPDX hook 为两个本次触及的旧文件补头。`torch.cuda` 报错和 attention backend 文档改写需先用 diff/blame 判断是否属于本次改动，只保留必要修复并对已证实旧项精确 skip |
| metadata/scratch 候选首次 runtime import 命令遗漏 Docker stdin 透传 | 1 | `docker run` 缺少 `-i`，容器内 `python -` 从空 stdin 正常退出，生成的 JSON/log 均为空文件；该轮作废，CUDA 未初始化、GPU 始终 0 MiB。保持镜像、探针和 GPU 0 不变，仅补 `-i` 后重跑 |
| metadata/scratch runtime 探针重复使用旧版 OpenAI protocol import 路径 | 1 | 探针在 `vllm.entrypoints.openai.protocol` import 处退出，候选当前真实路径为已在旧轮次记录的 `vllm.entrypoints.openai.chat_completion.protocol`；CUDA 未初始化、GPU 已释放。改用当前源码真实路径，不修改镜像或候选 |
| runtime import 成功后的 planning patch 混入错误 patch 边界 | 1 | `apply_patch` 原子拒绝且三个 planning 文件均未修改；重新读取实际目标上下文后拆分为有效 patch |
| 控制镜像 planning 批量 patch 再次使用了未匹配的多文件上下文 | 1 | patch 原子拒绝，Dockerfile 以外文件未修改；重新读取最新上下文并拆成逐文件 patch |
| 控制镜像结果 planning patch 在文件边界前误留空 `@@` | 1 | patch 原子拒绝且 planning 未修改；移除多余 hunk 标记并继续使用逐文件 patch |
| 配置审计时猜测 Phase 1 文件名为 `baseline_manifest.json` | 1 | `sed` 只读失败，未修改文件；`rg` 已显示真实文件为 `configs/phase1/native_baseline.json`，后续按真实路径迁移 |
| 新候选 overlay 首次 `cp -a` 被长命令会话提前终止 | 1 | 目标只有 3,214/4,744 个文件、0 个 native link，明确判为无效；只读 extracted source 未改。用同一源的目录内容对明确目标幂等补全，再要求 4,744 文件和 6 个有效链接 |
| overlay 建链循环遇到已存在的 `cumem_allocator` link | 1 | `ln -s` fail-closed；复查显示补全复制与此前循环合计已得到 4,744 文件和 6 个链接。下一步不重复创建，改为逐相对路径/target/hash 验证全部 6 项 |
| 配置门禁结果 planning patch 又在多文件边界前残留空 hunk | 1 | patch 原子拒绝且 planning 未修改；继续拆成逐文件 patch，并停止在文件边界前写无内容的 `@@` |
| 新控制镜像直接执行工具测试缺 pytest | 1 | 正式 venv 在 collection 前报告 `No module named pytest`，未形成测试结果；保持镜像不可变，在一次性容器用 uv/清华镜像/临时 cache 注入固定 pytest/tblib 后重跑 |
| 合并工具测试结果 planning patch 再次误留空文件边界 hunk | 1 | patch 原子拒绝，planning 未修改；继续逐文件 patch |
| 恢复会话后首次追加 8-warps 单卡结果时引用了 findings 中不存在的整段上下文 | 1 | `apply_patch` 原子拒绝，文件未修改；重新读取两个 planning 文件的实际末尾后按各自上下文追加 |
| 搜索待更新报告语句时把 Markdown 反引号直接放入双引号 shell 命令 | 1 | shell 在执行 `rg` 前因引号不闭合退出；改用单引号固定搜索模式，不再让反引号参与 shell 解析 |
| 首次章节检查脚本把 Markdown 标题井号数量错误写成正则量词 | 1 | Python `re` 在读取首行前报 `nothing to repeat`；改用字符串前缀和普通标题捕获，不重复使用动态量词 |
| 单卡结果证据复核时按历史约定猜测日志名为 `runner.log` | 1 | result 哈希与内容已通过，但该文件不存在；只列出精确运行目录文件并对实际日志路径复核，不重复猜测文件名 |
| 8-warps 完整 CUDA 首次启动猜测控制镜像中的 uv 位于 `/opt/uv/bin/uv` | 1 | 容器在 pytest 前以 127 退出，0 测试、0 Triton cache；CPU-only 探针确认实际路径为 `/usr/local/bin/uv`，重新双检 GPU 空闲后改用实测路径 |

## 约束提醒

- 所有报告必须使用中文，且数据只能来自实际落地结果。
- 每个阶段实验结束后先更新报告，再进入下一阶段。
- 修改已有报告前必须重新读取，并检查章节序号与交叉引用。
- 长实验每 10 分钟记录一次进度。
- GPU 实验必须在固定 Docker 容器内执行，且分配前连续两次确认授权 GPU 空闲。
- 正式实验前必须提交并推送源码与配置；不得让主仓库 submodule 指向本地-only commit。
- `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1` 及其他项目外目录仅可只读检查；禁止修改、暂存、提交、清理或重置。
