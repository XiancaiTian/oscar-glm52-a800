# 阶段 1 实验报告：REAP 剪枝模型原生 baseline

## 1. 结论

新测试模型
`/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001`
已完成阶段 1 原生 baseline：

- A800 TP=8 原生 `TRITON_MLA_SPARSE` 服务成功启动；
- 短请求、超过 320 tokens、连续 384-token decode 和近 32K 请求全部成功；
- official_v4 为 2,360/2,360 `scored`，request failure 为 0，overall accuracy
  为 `0.37415254237288137`；
- WikiText‑2 为 1/1 `scored`，PPL 为 `6.595132997244041`；
- checkpoint、源码、原生扩展、suite、命令、环境和结果哈希均已核验。

阶段 1 出口条件通过。这里的“通过”表示原生 baseline 服务和评测流程完整、结果
可作为同 checkpoint 的 OSCAR 对照，不表示对模型绝对业务精度作额外承诺。

## 2. 固定输入与环境

| 项目 | 实际值 |
| --- | --- |
| 模型路径 | `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001` |
| 权重 | 141 个 safetensors，463,045,186,640 字节 |
| config SHA256 | `21a509ab82dad35a8584b724f41aa25c775a3c8f0c7604f7b10d21ee53e5f7fc` |
| index SHA256 | `f50217dadf6c58f8f84140003bd7fc3497e9338916e9865e23ea5342f2ac1ce2` |
| expert mapping SHA256 | `c163c3f02089cfe7180bda0816cea59ce8f8bba0fe752b6dd4738d9c87b3da72` |
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| Driver / CUDA | 575.51.03 / 12.9 |
| 并行与执行模式 | TP=8、eager、禁用 prefix cache 与 CUDA graph |
| Attention backend | `TRITON_MLA_SPARSE` |
| Python | 3.12.13 |
| PyTorch / Triton | 2.11.0+cu129 / 3.6.0 |
| Transformers / Tokenizers | 5.8.1 / 0.22.2 |
| 服务与 accuracy 主仓库 commit | `91a6c0aeb1a97c312733a422620cc826be877166` |
| PPL 主仓库 commit | `905f98c7d5e03f5878b2601974a3bb79f55ece1f` |
| 集成源码 commit | `a3317695428819d41437b1cb144404b3bfc05a92` |
| OCI runtime source commit | `fd3e0b3772e989cf0d0d73a3d19b252ab82e9cdd` |

正式 preflight 核验了 4,711 个 runtime 源码文件、7 个原生扩展、141 个权重
分片、72,117 个 index entries、模型几何、expert mapping 和 official_v4
2,360+1 个样本。PPL 前两次 GPU 检查分别在 2026-07-27 21:59:16Z 和
22:00:19Z 完成，均为 8/8 空闲。

## 3. 服务与功能验证

141/141 个 checkpoint shard 全部加载。权重读取耗时 1,509.73 秒，模型加载总耗时
1,535.878352 秒，每卡模型内存 55.95GiB；服务启动总耗时 1,681 秒。

原生 KV cache 可用内存为 15.45GiB，容量为 179,008 tokens；以 32,768
tokens/request 计算的理论最大并发为 5.46。

| 用例 | 输入 + 输出 tokens | 耗时 | 结果 |
| --- | ---: | ---: | --- |
| 短请求 | 21 + 64 | 10.4668 秒 | HTTP 200 |
| 超过 320 tokens | 506 + 5 | 0.8002 秒 | HTTP 200 |
| 连续 decode | 26 + 384 | 58.8068 秒 | HTTP 200 |
| 近 32K | 31,996 + 64 | 22.4608 秒 | HTTP 200 |

8 个 rank 均记录 `TRITON_MLA_SPARSE`。smoke 结果文件状态为 `passed`。

## 4. official_v4 精度

### 4.1 正式结果

| Benchmark | scored / total | 正确数 | Accuracy |
| --- | ---: | ---: | ---: |
| GSM8K | 1,319 / 1,319 | 666 | 0.5049279757391963 |
| IFEval | 541 / 541 | 161 | 0.2975970425138632 |
| LiveCodeBench v6 | 175 / 175 | 12 | 0.06857142857142857 |
| MultiPL-E | 325 / 325 | 44 | 0.13538461538461538 |
| 总计 | 2,360 / 2,360 | 883 | 0.37415254237288137 |

正式运行时长为 53,208.72071003914 秒。服务端恰好记录 2,360 个正式 accuracy
HTTP 200，没有非 200 响应或 Traceback；所有样本的 evaluator status 均为
`scored`。

IFEval 对一个只包含点号的模型输出打印过一次非致命语言检测诊断。冻结 evaluator
仍将该样本正常计为 `scored` 并按未满足指令给 0 分，因此它不是请求失败、服务失败
或漏评。

### 4.2 评测口径说明

用户提供该模型在 GSM8K-full 上的已知 accuracy 为 86.35%。本轮 official_v4
GSM8K 子集为 666/1,319，即 `0.5049279757391963`。两者使用的
prompt/template、生成参数和评分口径不同，不能直接比较，也不能用其中一个覆盖
另一个。后续 OSCAR 候选只与本报告中完全相同环境的原生 baseline 比较。

### 4.3 精度产物

正式目录：

`/dev/shm/oscar-glm-reap-stage1/phase1/20260727T_reap_native_tp8/official_v4_accuracy/20260727T0707Z_reap_full`

| 产物 | SHA256 |
| --- | --- |
| `predictions.jsonl` | `c3f0b6345d7030f639e436cb19129ff60db78df35211489f49867e7949492579` |
| `failed_cases.jsonl` | `4c00a2fbafd8132427dae08f36616b0d63db34e555b119f2f58df12765cb4f35` |
| `summary.json` | `23ddda280962066ef064e18a8f1ac15669ecc283aab877c61581fe8c8c6b4811` |
| `summary_by_benchmark.json` | `14d9df7fcf60fa967cdeb8deebe516b8b3116100a407af410243e34e7cef890a` |
| `validation.json` | `764536b807603d35c804acd0a3d17a6f262fe147de2b6a7a75e37bd2b5110cd7` |
| `runner.log` | `f409ab24fed389ae738bc5232b34acc5cd691cf7984faa72e0d3fb71f72c58bd` |

## 5. WikiText‑2 PPL

固定配置为 TP=8、eager、BF16、max length 2,048、stride 512、batch 8、KV
cache dtype auto。141/141 个 shard 加载耗时 120.08 秒，模型加载总耗时
130.624503 秒；评分阶段耗时 363.5904743671417 秒。

| 指标 | 实际值 |
| --- | ---: |
| scored / total | 1 / 1 |
| input tokens | 289,709 |
| evaluated tokens | 289,708 |
| windows | 563 |
| mean NLL | 1.8863319523410782 |
| perplexity | 6.595132997244041 |

正式目录：

`/dev/shm/oscar-glm-reap-stage1/phase1/20260727T2200Z_reap_native_ppl/wikitext2_ppl`

| 产物 | SHA256 |
| --- | --- |
| `summary.json` | `29a93a4b2a3427a49be0a0ee19f54cac8fb08393e17dc013ed46f86e69330c4a` |
| `perplexity_results.json` | `577ff50077e650d936a60c6800a9b6b095bdb0079dca169fc0fc7bf6ff3997e2` |
| `validation.json` | `dfb5e180653dc56ad9fa4db4bfc275b57cf16de7b5828b7defbd2b39b081980a` |
| `runner.log` | `1f2fdbdec5f5f8789b8beb0359f2d7739564aff24fcd6cbc59dd14012f08e6e2` |
| `runtime_environment.txt` | `7105706d9d4ac4fc375c055fdaf07c75f7d26682bc964d2af8f37037bb9ec4b7` |

runner 正常关闭所有 EngineCore/worker 进程；日志没有 `ERROR` 或 Traceback。

## 6. 后续硬阈值

后续 OSCAR 候选必须使用相同 checkpoint、TP=8、prompt/template、decoding、runner
和 suite。阶段 7 的硬阈值为：

- 2,360/2,360 全部评分且 request failure 为 0；
- overall accuracy 不低于 `0.35915254237288137`；
- GSM8K 不低于 `0.4749279757391963`；
- IFEval 不低于 `0.2675970425138632`；
- LiveCodeBench v6 不低于 `0.03857142857142857`；
- MultiPL-E 不低于 `0.10538461538461538`；
- WikiText‑2 PPL 不高于 `6.792986987161362`。

这些阈值只用于衡量 OSCAR 相对同 checkpoint 原生 KV baseline 的退化，不用于把
用户提供的 GSM8K-full 86.35% 转换成另一套评测的门槛。

## 7. 存储与交付边界

模型、cache、完整预测、服务日志和其他大型实验产物不 commit、不 push。正式运行
输出保存在 `/dev/shm`，代码、小型配置、哈希和本报告同步到 Git。
