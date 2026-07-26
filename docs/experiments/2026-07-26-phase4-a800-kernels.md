# 阶段 4 实验报告：A800/SM80 Triton kernels

## 1. 结论

阶段 4 通过设计文档规定的出口条件：

- rotation、clipping、INT2 pack/store、BF16 store、recent demotion、history dequant、mixed sparse decode/prefill、global LSE merge 与 inverse rotation 均已实现；
- 正式代码在全新 Triton cache 上完成 SM80 cold compile 和 A800 实际 launch；
- 22/22 项 CUDA 门禁全部通过，完整 `tests/oscar_mla` 在 CUDA 开启下为 83/83 passed；
- TP=8 rank-local cold-cache smoke 在 8/8 张 A800 上全部通过，累计执行 176 次 CUDA kernel 测试；
- 测试结束后 8 张 GPU 均回到 0MiB、0%，无 compute process。

大型 Triton cache 和测试日志仅保存在项目 `artifacts/phase4/`，不提交或推送。

## 2. 固定源码与环境

| 项目 | 实际值 |
| --- | --- |
| 源码分支 | `feat/glm52-oscar-kernels` |
| 正式源码 commit | `c50d86b34643c9fba0ae1df28a671c04fd107a41` |
| 正式主仓库 commit | `a2b36988d7e786ba28798e23239ffca8f6616d10` |
| GPU | NVIDIA A800-SXM4-80GB |
| Compute capability | 8.0 |
| PyTorch | 2.11.0+cu129 |
| Triton | 3.6.0 |
| CUDA 测试开关 | `VLLM_OSCAR_RUN_CUDA_TESTS=1` |

正式重跑前两次 GPU 检查时间为 2026-07-26T12:19:40Z 和 12:20:56Z，均为 8/8 GPU 0MiB、0% 且无 compute process。源码与主仓库在启动前均为干净、已推送状态。

## 3. Kernel 与覆盖范围

| Kernel/路径 | A800 覆盖 |
| --- | --- |
| shared latent rotation | 512 维、FP32 IEEE accumulator |
| INT2 quantize/pack/store | group 128、clip 0.92/1.0、rows 1/4/17 |
| BF16 prefix/recent store | 0/63/64/65/319/320/321 边界与 ring 地址 |
| recent demotion | 多请求 HP row、page/offset、direct-store oracle |
| history dequant | packed data、scale、zero 与 PyTorch oracle |
| mixed sparse decode | sequence 63/64/319/320/321、heads 1/4、4 splits |
| DSA selected IDs | prefix/history/recent 混合选择 |
| mixed sparse prefill | causal query batch 1/4/8 |
| RoPE | 512 latent + 64 原精度 RoPE，scale 为 `1/sqrt(576)` |
| global merge/inverse rotation | 输出与 LSE finite，并通过 PyTorch tolerance |

CUDA tests 对 rotation/dequant 使用固定 `atol=0.35, rtol=0.02`，对 mixed decode/prefill 使用 `atol=0.5, rtol=0.03`；所有 shape、finite 与误差断言均通过。CPU Triton interpreter 的 512+64 decode/prefill 相对 PyTorch oracle 最大绝对误差均为 `2.384185791015625e-07`。

## 4. 首轮失败与测试修复

首轮目录为 `artifacts/phase4/20260726T121130Z_a800_kernels`，使用全新 Triton cache，结果为 23 passed、1 failed、63.33 秒。唯一失败测试先断言 `recent[0,0]` 为 NaN，随后又断言同一 slot 等于 position 320 的写入值；position 64 和 320 按 `(position-prefix) % recent` 都映射 slot 0，因此两个断言无法同时成立。

修复将测试拆成两次调用：

1. 只传 final history positions 64/65，验证 recent slot 0/1 保持 NaN；
2. 再传 prefix 与 final recent positions 319/320/321，验证 ring slot 255/0/1。

修复不改 kernel。定向 A800 test 为 1/1 passed、3.61 秒，ruff 0.14.0、format 和 diff check 通过；源码修复 commit 为 `c50d86b34...`。

## 5. 正式 cold-cache 结果

正式重跑目录为 `artifacts/phase4/20260726T121130Z_a800_kernels/retry_c50d86b`，使用第二个全新 Triton cache。

| 验证 | 实际结果 |
| --- | --- |
| kernel 文件定向套件 | 24 passed、0 failed，56.60 秒 |
| 其中 CUDA 门禁 | 22/22 passed |
| 完整 `tests/oscar_mla`，CUDA 开启 | 83 passed、0 failed，34.31 秒 |
| Triton cache | 284 files、19,620,324 bytes |
| 运行后 GPU | 8/8 为 0MiB、0%，无 compute process |

kernel 定向日志 SHA256 为 `91cdbc6eab614c5d86a910038e0a30d8658468ee76110aa4a72807b6058f6264`；完整套件日志 SHA256 为 `73a7c83c638c4e4279e7f3c659a54c8bf7699897f274970aedd354d8f8e15b1e`；preflight SHA256 为 `1d29a2df9df46283a28168dfa566ae741c0f9cc7e6e0ab42bc076749b21d986d`。

## 6. TP=8 rank-local cold-cache

在再次完成两次 8/8 空闲检查后，同时启动 8 个独立进程；每个进程绑定一张 A800，并使用各自全新的 Triton cache 运行相同 24 个测试节点。

| Rank | 结果 | 耗时 | cache files / bytes |
| ---: | --- | ---: | ---: |
| 0 | 24 passed | 86.76 秒 | 284 / 19,624,132 |
| 1 | 24 passed | 86.90 秒 | 284 / 19,624,132 |
| 2 | 24 passed | 86.76 秒 | 284 / 19,624,132 |
| 3 | 24 passed | 87.00 秒 | 284 / 19,624,132 |
| 4 | 24 passed | 84.50 秒 | 284 / 19,624,132 |
| 5 | 24 passed | 86.65 秒 | 284 / 19,624,132 |
| 6 | 24 passed | 86.77 秒 | 284 / 19,624,132 |
| 7 | 24 passed | 84.76 秒 | 284 / 19,624,132 |

总计为 192/192 测试节点，其中 CUDA 测试执行数为 `8 × 22 = 176`。`summary.txt` SHA256 为 `3376708ba54b84a1862aa4920a33b6aaac64aaee96294e99742bebadfae35391`；运行前两次 GPU snapshot SHA256 为 `ff45db5562d7b0be784255d9e868d02e0230e18ecdf071693a7dbaab10fd7886` 和 `c0ef3526db3ff2d7ab26b945da500bb01ad3b4f74d8cedc8de7439e26a4a19ef`。

## 7. 阶段出口

阶段 4 已完成。该结果证明 rank-local kernels 可在 8 张 SM80 A800 上独立 cold compile、launch 并通过 oracle/边界门禁；完整 TP=8 服务中的 scheduler、collective、模型层接线与接近 32K 请求属于阶段 5，不能由本阶段结果替代。
