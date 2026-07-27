# 阶段 3 实验报告：REAP 模型三池 Cache 与 Scheduler 回归

## 1. 结论

新 REAP checkpoint 与最新 runtime source 的阶段 3 回归通过：

- 当前 `tests/oscar_mla` 与两个 KV cache core 文件共 145 passed；
- 26 个 CUDA 专项按显式门禁 skipped，留给阶段 4 在 A800 上实际执行；
- 完整 scheduler 文件中 68 项通用测试通过；
- 其余 28 项均在测试体执行前因离线环境缺少
  `llava-hf/llava-1.5-7b-hf` 配置而失败，没有 OSCAR 或通用 scheduler 断言失败；
- 测试过程没有初始化 CUDA，8 张 A800 均保持 0MiB、0%。

本阶段只验证 CPU 规划、ownership 与 scheduler/worker metadata，不把理论容量比
冒充 GPU 实测值。新 checkpoint 的层数、latent rank、RoPE、group size 与
indexer 几何和旧 checkpoint 相同，因此不需要改动 allocator 代码。

## 2. 固定输入

| 项目 | 实际值 |
| --- | --- |
| 主仓库 commit | `38a3179` |
| 源码 commit | `a3317695428819d41437b1cb144404b3bfc05a92` |
| 模型 | `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001` |
| rotation manifest SHA256 | `0275043c070c9127354997374e9bca1c70fe1308a7b2d057f992fadedef868e5` |
| 层数 / latent rank / RoPE | 78 / 512 / 64 |
| group size | 128 |
| prefix / recent | 64 / 256 tokens |
| `indexer_types` | shared 57 层、full 21 层 |

主仓库与源码仓库在测试前均干净，HEAD 与远端分支一致。测试使用源码 worktree
已有的 uv `.venv`，固定依赖从项目内候选 rootfs 读取。

## 3. 容量复算

使用当前源码的 capacity planner，按 78 层、latent rank 512、14GiB 总预算和
`max_num_seqs=16` 重新计算，实际结果如下：

| 指标 | 实际值 |
| --- | ---: |
| 总预算 | 15,032,385,536 bytes |
| 标准 blocks / usable blocks | 36,216 / 36,215 |
| logical token slots | 579,440 |
| history pages / slots | 36,216 / 579,456 |
| fixed BF16 prefix/recent | 408,944,640 bytes |
| INT2 history | 7,231,610,880 bytes |
| BF16 RoPE | 5,785,288,704 bytes |
| native auxiliary DSA | 1,606,252,032 bytes |
| 总分配 / 未使用 | 15,032,096,256 / 289,280 bytes |
| native logical token slots | 162,256 |
| 理论容量比 | 3.5711468297012128× |

`总分配 + 未使用 = 总预算` 精确成立，history 理论与 page padding 后压缩率均为
6.4×。结果文件 SHA256 为
`049fa95aa86f8a5fabb96e33716819747a65282e21cfed6b5628ef8996c97353`。
这是 CPU planner 的确定性结果，不是 GPU 峰值容量实测。

## 4. 定向回归

定向范围为完整 `tests/oscar_mla`、`test_kv_cache_utils.py` 和
`test_single_type_kv_cache_manager.py`。

正式结果为：

| 指标 | 实际值 |
| --- | --- |
| passed | 145 |
| failed / skipped | 0 / 26 |
| 耗时 | 47.72 秒 |
| 日志 SHA256 | `948167c921ea8e1d59cefdee0f588f4a42f978271545ba525a067f9a116a0050` |

26 个 skip 全部是必须显式设置 `VLLM_OSCAR_RUN_CUDA_TESTS=1` 的 CUDA 专项，不把
skip 记作 CUDA 通过。另一次只选择 allocator/ownership 四文件的隔离重跑为
82/82 passed。

## 5. 完整 Scheduler 离线回归

`tests/v1/core/test_scheduler.py` 在相同离线环境下得到：

| 指标 | 实际值 |
| --- | --- |
| collected / passed | 96 / 68 |
| failed | 28 |
| 耗时 | 29.34 秒 |
| 日志 SHA256 | `babe6aa8085aaba91a479ec9ef9a098e206d0d3612ebf51a0b925d8a88365138` |

28 项失败全部引用 `llava-hf/llava-1.5-7b-hf`，并在 `ModelConfig` 构造阶段以同一
离线配置缺失错误退出；这些测试覆盖多模态 encoder cache 与 EC connector，不是
OSCAR MLA 三池路径。该结果与旧 checkpoint 阶段 3 的 68 passed / 28
配置缺失一致。没有为追求表面全绿下载 LLaVA。

## 6. 资源与代码状态

- `torch.cuda.is_initialized()` 为 `False`；
- 测试后 GPU 0–7 均为 0MiB、0%，无 compute process；
- 主仓库与源码仓库没有测试产生的改动；
- 日志保存在 `/dev/shm/oscar-glm-reap-stage3/`，不进入 Git；
- 外部 `/nfs/AE/zhanghong/workflow/vllm_a/` 路径未被修改。

## 7. 阶段出口

阶段 3 回归完成，无需修改代码。下一阶段在全新 Triton cache 上执行 A800/SM80
cold compile、真实 CUDA launch、oracle 与边界回归；CPU 测试不能替代该门禁。
