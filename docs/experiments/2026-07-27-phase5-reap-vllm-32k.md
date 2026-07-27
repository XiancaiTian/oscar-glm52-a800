# 阶段 5 实验报告：REAP 模型 vLLM TP=8/32K 端到端

## 1. 结论

新 REAP checkpoint、当前集成源码和新 rotation artifact 的阶段 5 通过：

- TP=8 服务完成 141/141 分片加载并正常 ready；
- 短请求、>320-token 输入、384-token 连续 decode、31,996-token 上下文均
  HTTP 200；
- 8 个并发请求全部 HTTP 200；
- 78 层 store、recent→INT2 demotion 与 DSA-selected mixed read 均实际执行；
- 日志明确记录 BF16 history absent，无 dense/full-attention fallback；
- 32K 容量、三池压缩率和请求级生命周期全部满足设计门禁；
- 服务正常退出，8 张 A800 均回到 0MiB、0%。

模型、Triton/cache、服务日志和响应均位于 `/dev/shm` 或已有 ignored artifact，
不提交或推送；外部目录保持只读。

## 2. 固定输入与 Preflight

| 项目 | 实际值 |
| --- | --- |
| 主仓库 commit | `d9610d66073a2f438b60365d2bee232559d2ed64` |
| 源码 commit | `a3317695428819d41437b1cb144404b3bfc05a92` |
| 模型 | `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001` |
| rotation manifest | `0275043c070c9127354997374e9bca1c70fe1308a7b2d057f992fadedef868e5` |
| rotations | `256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d` |
| Python / PyTorch / Triton | 3.12.13 / 2.11.0+cu129 / 3.6.0 |
| attention / KV cache | `TRITON_MLA_SPARSE` / `oscar_mla_int2` |
| TP / PP / max length / max seqs | 8 / 1 / 32,768 / 16 |

正式 preflight 逐项验证已发布且干净的双仓库、固定 OCI、4,711 个源码文件、
7 个原生扩展、141 个模型分片、模型几何、official_v4 manifest 与 78 层
rotation identity。两次 GPU 检查间隔超过 60 秒，均为 8/8 空闲。静态 preflight
SHA256 为
`a6cc5b9ddd28d0c10a83afc1240fe65de8386ed6ab8598639b0aac8926dea4cd`。

## 3. 服务启动与三池容量

141 个分片用时 141.55 秒；完整模型加载用时 154.878296 秒，每卡模型内存
56.02GiB。监控器在启动后 300 秒记录 ready；服务实际于
2026-07-27T23:39:07Z 开始接受请求。

每个 TP rank 的实际三池计划一致：

| 指标 | 实际值 |
| --- | ---: |
| logical capacity | 637,632 tokens |
| 32K 最大并发 | 16.00× |
| BF16 prefix | 1,024 slots / 81,788,928 bytes |
| BF16 recent | 4,096 slots / 327,155,712 bytes |
| INT2 history | 637,696 slots / 9,964 pages / 7,958,446,080 bytes |
| BF16 RoPE | 6,366,756,864 bytes |
| native index/cache | 1,767,693,312 bytes |
| unused | 459,060 bytes |
| 理论 / page padding 压缩率 | 6.4× / 6.4× |
| allocated capacity ratio | 3.5812365205× |

日志明确输出 `BF16 history=absent`。

## 4. 请求结果

| 请求 | prompt + completion | HTTP | 用时 |
| --- | ---: | ---: | ---: |
| short | 21 + 14 | 200 | 4.456394490785897 秒 |
| >320 input | 506 + 5 | 200 | 2.9591034967452288 秒 |
| continuous decode | 26 + 384 | 200 | 88.33960080239922 秒 |
| near 32K | 31,996 + 64 | 200 | 436.7646517716348 秒 |
| concurrent | 8/8 requests passed | 8 × 200 | — |

串行与并发结果 SHA256 分别为
`af5953e0dc06d8ab35a434210ff73f46601c73e90d67157602155945b42f5fca`
和
`dd69db120ec06cf065408a94ba5d5806a77d9c5148f7ababacfd13c9e1a9fec5`。

## 5. Runtime 证据

运行时实际记录：

- 新 artifact manifest/tensor SHA256 匹配；
- three-pool write active，且没有完整 BF16 latent history；
- recent-to-INT2 demotion active；
- DSA-selected mixed prefix/recent/INT2 read active；
- 最终 78 层 call counts 为 store `42,666`（每层 547）、demotion `18,330`
  （每层 235）、read `42,666`（每层 547），三项的 min=max。

证据文件 SHA256 为
`e478bed908a088e5a1f4bf4be86ecc4a0f6e1f22ec1d584bad4594a4f89085d2`；
完整 server log SHA256 为
`5914d921c3292b221634e73568269149fa04fdd663b7461fdd21d9d6ad321913`。
错误扫描未发现 ERROR、Traceback、RuntimeError 或 ValueError。

服务超过 10 分钟，监控器按规范于 2026-07-27T23:44:16Z 写入心跳；当时
8 卡均为 77,213MiB、100%。心跳 SHA256 为
`83defac6d5e2ccad525947aa28bdd8f3bd06300a06c0f6676341b96fb68fddba`。

## 6. 非阻塞问题

- 一个 worker 报告任务专用 Torch kernel cache 目录无法创建，因此关闭该进程的
  可选 JIT cache；模型、OSCAR/Triton kernel 和全部请求仍通过，没有由此产生
  fallback 或错误。
- 退出后的人工 JSON 汇总探针首次误把整数 `requests` 当作列表；该探针在正式
  响应文件之外失败。按实际 `results` 字段重验为 8 行、8/8 HTTP 200。

## 7. 阶段出口

服务收到 SIGTERM 后正常退出，`server_exit_status=0`；8 张 GPU 均为
0MiB、0%，无残留 APIServer、EngineCore 或 worker。阶段 5 完成，无需修改源码。
下一阶段基于相同源码、模型与新 artifact 构建新的不可变候选 OCI。
