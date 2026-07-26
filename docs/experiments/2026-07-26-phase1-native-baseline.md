# 阶段 1 实验报告：GLM‑5.2/A800 原生 baseline

## 1. 结论

阶段 1 通过设计文档规定的出口条件：

- A800 TP=8 原生 `TRITON_MLA_SPARSE` 服务成功启动；
- 短请求、超过 320 tokens、连续 384-token decode 和近 32K 请求均成功；
- official_v4 正式结果为 2,360/2,360 `scored`，request/extraction failure 均为 0；
- WikiText‑2 PPL 为 `7.692286035848967`，1/1 `scored`；
- 命令、环境、配置、结果与 SHA256 均保存在项目内 `artifacts/phase1/`。

这里的“通过”表示原生 baseline 服务与评测流程有效且证据可复现，不表示当前剪枝模型达到业务绝对精度要求。

## 2. 固定环境

| 项目 | 实际值 |
| --- | --- |
| 模型 | `GLM-5.2-FP8-pruned-staticgate-e154-H001-nfs` |
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| Driver / CUDA | 575.51.03 / 12.9 |
| 并行与执行模式 | TP=8、eager、禁用 prefix cache 与 CUDA graph |
| Attention backend | `TRITON_MLA_SPARSE` |
| Python | 3.12.13 |
| PyTorch / Triton | 2.11.0+cu129 / 3.6.0 |
| Transformers / Tokenizers | 5.8.1 / 0.22.2 |
| 固定源码仓库 commit | `53d8be94f6038e10ab0c344f706c5ffe66a555b8` |
| OCI runtime source commit | `fd3e0b3772e989cf0d0d73a3d19b252ab82e9cdd` |
| accuracy 服务主仓库 commit | `6d7a7bd7c35dad6290bf599ba5120109eaae8b3f` |
| PPL 正式 preflight 主仓库 commit | `0df8c3fbf95fbb8073966e739c2fc2a391ef73b7` |

正式 PPL preflight 复核了 4,711 个 runtime 源码文件、7 个原生扩展、141 个权重分片、模型几何和 suite 指纹；两次间隔 60 秒的 GPU 检查均为 8/8 空闲。

## 3. 服务与功能验证

141/141 个 checkpoint shard 全部加载。首次服务权重读取耗时 2,337.66 秒，模型加载耗时 2,393.23 秒，每卡模型内存 55.93GiB；engine profile、KV cache 与 warmup 后，服务总启动耗时 2,762 秒。

原生 KV cache 可用内存为 14.3GiB，容量为 165,696 tokens；以 32,768 tokens/request 计算的理论最大并发为 5.06。

| 用例 | 输入 + 输出 tokens | 耗时 | 结果 |
| --- | ---: | ---: | --- |
| 短请求 | 21 + 64 | 16.06 秒 | HTTP 200 |
| 超过 320 tokens | 506 + 18 | 3.56 秒 | HTTP 200 |
| 连续 decode | 26 + 384 | 59.90 秒 | HTTP 200 |
| 近 32K | 31,996 + 64 | 24.28 秒 | HTTP 200 |

## 4. official_v4 精度

### 4.1 正式结果

| Benchmark | scored / total | 正确数 | Accuracy |
| --- | ---: | ---: | ---: |
| GSM8K | 1,319 / 1,319 | 262 | 0.19863532979529946 |
| IFEval | 541 / 541 | 145 | 0.2680221811460259 |
| LiveCodeBench v6 | 175 / 175 | 12 | 0.06857142857142857 |
| MultiPL-E | 325 / 325 | 50 | 0.15384615384615385 |
| 总计 | 2,360 / 2,360 | 469 | 0.19872881355932204 |

正式合并目录：

`artifacts/phase1/20260725T162556Z_native_tp8/official_v4_accuracy/merged_full_retry_20260726T1112Z`

### 4.2 超时补跑说明

全量首轮完成 2,360 条请求，但末尾 7 条 GSM8K 因客户端 math timeout 300 秒成为 `request_failed`；服务端错误扫描为空。随后只对固定 ID `gsm8k:001311` 至 `gsm8k:001317` 将 math timeout 提高到 900 秒补跑，7/7 `scored`、request failure=0、accuracy 0.0。

合并仅按精确 ID 替换首轮 `request_failed` 行，首轮原始证据没有覆盖。完整 manifest 的第 2,361 行是由独立入口执行的 WikiText‑2 PPL，因此 accuracy 合并显式固定四个 benchmark 共 2,360 行。

### 4.3 精度产物 SHA256

| 产物 | SHA256 |
| --- | --- |
| `predictions.jsonl` | `68a3d0b184929286aeef6b690b109ebe9a84bbcc8a41bf89af1493c408ad3e75` |
| `failed_cases.jsonl` | `9cf215421e8e0d5fbc0932c5b7004927d2c6b36b70ba48ac055c6364e720a6fe` |
| `summary.json` | `f8f52510e10d861450a51f7b8ee6cb559ac8040108257e1a6fd0912ea84dc9e2` |
| `summary_by_benchmark.json` | `c7a5754e2f9eb893bd9878253eb45a29b505aa1b477c2e229c8844a50bf2ac9b` |
| `merge_provenance.json` | `4f3c6488d5e5f7f182fda3a72283149df248a6ad5ffa2de77673cf02b9f4a693` |
| `validation.json` | `06cc30a34d877ee4f39e743158d7042a90faa36876fe8a94e11a2da4413ec432` |

## 5. WikiText‑2 PPL

固定配置为 TP=8、eager、BF16、max length 2,048、stride 512、batch 8、KV cache dtype auto。PPL runner 完成 141/141 shard 加载，权重读取耗时 224.65 秒、模型加载耗时 236.68 秒；评分阶段耗时 `368.10103392601013` 秒。

| 指标 | 实际值 |
| --- | ---: |
| scored / total | 1 / 1 |
| input tokens | 289,709 |
| evaluated tokens | 289,708 |
| windows | 563 |
| mean NLL | 2.0402180131829573 |
| perplexity | 7.692286035848967 |

正式 PPL 目录：

`artifacts/phase1/20260726T1115Z_native_ppl/wikitext2_ppl`

| 产物 | SHA256 |
| --- | --- |
| `summary.json` | `a0b1643bdf6ed110a3f90bde55f76bdd66d91dba718b7755b23c5d594986a794` |
| `perplexity_results.json` | `8c470884b9c971c1c7f95a290d4750abd48878ed849a29756a8793b11f3e72ed` |
| `validation.json` | `17e740fe35da046c5b8e4ca2f1edab916dfc216d4cd5dbc8e3e5c434341feab2` |
| `runner.log` | `d6b4e34e08971fc172e8eacb8eec4a027176cac91ce3b71710bca8a9888fb6f1` |

runner 结束后无残留 vLLM/EngineCore 进程，8 张 GPU 均为 0MiB、0% 利用率。

## 6. 后续比较基线

后续 OSCAR 候选必须在相同模型、TP=8、prompt/template、decoding、runner 和 suite 下比较：

- overall accuracy 相对本报告 baseline 下降不得超过 1.5 个百分点；
- 任一 benchmark 下降不得超过 3 个百分点；
- WikiText‑2 PPL 相对 `7.692286035848967` 上升不得超过 3%。

精度和 PPL 大型产物仅保存在项目 `artifacts/`，不提交或推送；Git 仅同步代码、小型配置和本报告。
