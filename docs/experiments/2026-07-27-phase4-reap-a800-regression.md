# 阶段 4 实验报告：REAP 模型 A800/SM80 Kernel 回归

## 1. 结论

新 REAP checkpoint、最新集成源码与新 rotation artifact 的阶段 4 回归通过：

- GPU 0 在全新 Triton cache 上完成完整 `tests/oscar_mla`，114/114 passed；
- 26/26 个 CUDA 门禁全部实际执行，0 failed、0 skipped；
- GPU 0–7 各自使用独立空 cache 完成 28/28 个 rank-local kernel 节点；
- 8 卡累计 224/224 节点，其中 208 次为实际 CUDA kernel 测试；
- 测试后 8 张 GPU 均为 0MiB、0%，无 compute process。

Triton cache 与测试日志位于 `/dev/shm`，不提交或推送。外部部署目录和模型目录
始终只读。

## 2. 固定输入与启动门禁

| 项目 | 实际值 |
| --- | --- |
| 主仓库 commit | `18e0684` |
| 源码 commit | `a3317695428819d41437b1cb144404b3bfc05a92` |
| 模型 | `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001` |
| rotation manifest SHA256 | `0275043c070c9127354997374e9bca1c70fe1308a7b2d057f992fadedef868e5` |
| rotations SHA256 | `256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d` |
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| Compute capability | 8.0 |
| CUDA 测试开关 | `VLLM_OSCAR_RUN_CUDA_TESTS=1` |

首次 GPU 0 全套前，两次空闲 snapshot 间隔 60 秒，SHA256 分别为
`29d3fdf5f00586010f9e723f9e3ae3d32ba7afcf52d2554e8e93239ee1a2d844`
和 `8106349eea5b07aed31c46ea8320efc790149b4a819fc95b4b3d7e429e462240`。
两次均为 8/8 GPU 0MiB、0%、无 compute process；主仓库与源码均干净且已推送。

## 3. GPU 0 完整 Cold-cache 回归

GPU 0 使用独立空目录
`/dev/shm/oscar-glm-reap-stage4/20260727T2322Z_a800_kernels/triton_gpu0_cold`。
实际结果为：

| 指标 | 实际值 |
| --- | ---: |
| 完整套件 | 114 passed |
| 实际 CUDA 门禁 | 26 / 26 |
| failed / skipped | 0 / 0 |
| 耗时 | 77.63 秒 |
| Triton cache | 316 files、22,268,746 bytes |
| 日志 SHA256 | `a4f49b83418fd284073aa94a424c8bba5d10ae31f37aa7a8c72da80817fbc184` |

覆盖 rotation、INT2 pack/store、BF16 prefix/recent、demotion、history dequant、
mixed sparse decode/prefill、DSA selected IDs、RoPE、global LSE merge 与 inverse
rotation；oracle、finite、shape 和边界断言全部通过。

## 4. 八卡 Rank-local Cold-cache

GPU 0 全套结束后再次完成两次 8/8 空闲检查，snapshot SHA256 为
`6fe7791f6f6a21277413f4828c43784816e53412f550966116a90cb7da4b5d0c`
和 `a7cceb3f49e47e1d562500ecc606f109c7d48ce49426f14185f99cbcb2b458ad`。

随后并行启动 8 个进程；每个进程固定绑定一张卡，并使用互不共享的空 Triton
cache。每卡执行 `test_triton_store.py` 与 `test_triton_decode.py` 的 28 个节点。

| GPU | 结果 | 耗时 | cache files / bytes |
| ---: | --- | ---: | ---: |
| 0 | 28 passed | 54.73 秒 | 316 / 22,269,278 |
| 1 | 28 passed | 55.08 秒 | 316 / 22,269,278 |
| 2 | 28 passed | 54.64 秒 | 316 / 22,269,278 |
| 3 | 28 passed | 54.73 秒 | 316 / 22,269,278 |
| 4 | 28 passed | 55.06 秒 | 316 / 22,269,278 |
| 5 | 28 passed | 54.18 秒 | 316 / 22,269,278 |
| 6 | 28 passed | 54.61 秒 | 316 / 22,269,278 |
| 7 | 28 passed | 54.52 秒 | 316 / 22,269,278 |

每卡 28 个节点包含 26 个 CUDA 门禁和 2 个非 CUDA/reference 节点，因此合计为
224/224 测试节点、208 次实际 CUDA 执行。8 个 cache 的文件数和字节数完全一致，
证明每张卡均独立完成相同 cold compile。

## 5. 资源与代码状态

- 测试结束后 GPU 0–7 均为 0MiB、0%，无 compute process；
- 主仓库与源码仓库均无测试产生的修改；
- 本轮没有修改 kernel、测试或相邻代码；
- 日志和 cache 仅保存在 `/dev/shm/oscar-glm-reap-stage4/`；
- `/nfs/AE/zhanghong/workflow/vllm_a/` 下外部文件未被修改。

## 6. 阶段出口

阶段 4 回归完成，无需修改代码。该结果证明 rank-local kernel 在 8 张 SM80 A800
上可独立 cold compile、launch 并通过 oracle/边界门禁；TP=8 scheduler、
collective、模型层接线、三池实际分配与接近 32K 请求必须在阶段 5 单独验证。
