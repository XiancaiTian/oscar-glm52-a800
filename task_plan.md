# 任务计划：OSCAR × GLM‑5.2 × 苹果800 分阶段实验

## 目标

严格按照 `docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md` 完成阶段 0–9 的实现、实验、验收与中文报告，最终满足第 17 节的 32K 首版本完成定义，并给出 128K 扩展验证或容量阻塞证据。

## 当前恢复检查点（2026-08-04 07:24 CST）

- [ ] 当前进行中：2.345已由`1fde94a3a8484db74733e3e4146fd09d6e018ac2`发布；canonical fixed256 v2终局为OSCAR 93/256（36.328125%），相对BF16 baseline 105/256少12题、低4.6875个百分点，精度门禁失败。wrapper exit0，容器/双tmux退出，GPU0–7全idle；54/54 final validation与29/29 manifest通过。当前转入CPU-only canonical精度配对诊断与优化设计，继续禁止32K/batch1性能复测。

- [x] CPU-only归因证据已闭合：固定矩阵后的validation 85/85、独立矩阵4/4、manifest 10/10通过；确认当前相对四轮OSCAR历史对照净差-8/-15/-13/-4，且c349 K1024对照混有source与prefill K两项变化。
- [x] 正式报告2.346已实时追加并通过章节、术语、交叉引用、证据hash、85/85 validation、10/10 manifest及diff门禁。
- [x] 报告2.346与planning已由主仓`899c7fb47d1029824746a52c7c5ba3c2256e5c97`通过GitHub HTTPS发布；主仓与source仓均clean/upstream。
- [ ] 当前进行中：对ea8 canonical fixed256入口执行最小TDD，只把prefill top-k从768提高到1,024并验证其余身份不变；代码/配置阶段结果须先实时写入报告，随后才申请新GPU双空闲门禁。

- [ ] TDD错误边界：宿主Python3.8因缺`datetime.UTC`产生导入错误，未构成有效红灯；后续测试统一使用冻结`artifacts/phase0-candidate-bundle/rootfs/usr/bin/python3.12`。
- [ ] 冻结Python3.12直接宿主执行又因glibc ABI不足而在启动前失败；有效测试改在固定control容器内执行，不能继续把宿主运行失败计作合同红灯。
- [ ] control镜像首次命令未覆盖既有entrypoint，导致Python执行自身二进制；最后一次环境修正为先inspect并显式设置entrypoint，仍失败则停止测试环境尝试。
- [x] 固定control容器显式entrypoint后取得有效红灯：目标1 failure仅命中prefill K实际768、期望1024；现在实施4文件最小production/config变更。
- [x] prefill K1024最小变更CPU门禁闭合：目标1/1、工具25/25、Phase9递归90/90、结构化24/24与manifest 9/9通过；首次验证器布尔期望错误20/24已原样保留。
- [ ] 当前进行中：完整重读并实时追加正式报告2.347，记录TDD红绿灯、5文件最小改动、三次启动环境边界及验证器首次失败；发布完成前不做GPU双空闲门禁。
- [x] 正式报告2.347已实时追加并通过章节、引用、术语、14项身份、24/24 validation、9/9 manifest与diff门禁。
- [ ] 当前进行中：精确提交并HTTPS发布5文件prefill K1024最小改动、报告2.347及planning；发布完成前不执行GPU双空闲门禁。
- [x] 5文件prefill K1024候选、报告2.347及planning已由主仓`90f8d9f1144f298262a1e4bd0fabb80461b688af`通过GitHub HTTPS发布；两仓clean/upstream。
- [ ] 当前进行中：为同一fixed256候选执行GPU0–7新双空闲门禁（两次间隔至少60秒），门禁结果先实时写入并发布报告，随后才允许启动长跑。
- [ ] GPU门禁v1结果：两轮GPU事实通过但主仓clean身份失败，24/25；该轮不得授权启动。当前先实时追加并发布报告2.348失败边界，再用新run/output身份执行完整v2门禁。
- [ ] 报告2.348已实时追加并通过门禁；当前只提交并HTTPS发布该失败边界，发布后用`ea8-prefill1024-fast256-v2`与全新输出根重做双空闲门禁。
- [x] v2双空闲门禁已通过：00:53:32Z/00:54:47Z间隔75秒，两轮GPU0–7全idle；主/source clean/upstream、25/25 validation与17/17 manifest通过。
- [ ] 当前进行中：完整重读并实时追加报告2.349，发布门禁通过结果；2.349发布完成前不得启动`ea8-prefill1024-fast256-v2`长跑。
- [x] 报告2.349已实时追加并通过章节、引用、证据、25/25 validation、17/17 manifest与diff门禁。
- [ ] 当前进行中：提交并HTTPS发布2.349；发布后即时复核GPU仍空闲，再启动`ea8-prefill1024-fast256-v2`并每10分钟打印精度。
- [x] 2.349已由`c36b5ce15176198c2a537c52502a68f5f96dd87c`发布；`ea8-prefill1024-fast256-v2`于00:58:30Z有效启动，实际K1024身份、44/44 static、容器/tmux/TP0–7已确认。
- [ ] 当前进行中：修正并闭合启动证据validator（首次31/32仅network格式期望错误），实时追加启动报告2.350；实验继续运行并由monitor每10分钟输出精度。
- [x] 启动证据最终33/33、manifest18/18，报告2.350已实时追加并通过全部门禁。
- [ ] 当前进行中：发布2.350并等待600秒固定精度节点；实验持续运行，monitor每10分钟输出。
- [x] 2.350已由`2c1649be268a2afd2f6e000f17daa1119302f12c`发布；10分钟节点为11/256、10正确、0 invalid/0 truncated，22/22 validation、9/9 manifest通过。
- [ ] 当前进行中：实时追加并发布报告2.351；实验继续运行，等待20分钟固定节点。
- [x] 报告2.351已实时追加并通过章节、引用、证据、22/22 validation、9/9 manifest与diff门禁。
- [ ] 当前进行中：发布2.351；实验继续运行，等待20分钟固定节点并实时打印。
- [x] 2.351已由`b40f4010d2ee797912836fed07c85a65b36726cb`发布；10分钟结果已实时落盘和发布。
- [ ] 当前进行中：实验继续运行，等待01:18:30Z的20分钟固定节点并实时打印/更新报告。
- [x] 20分钟固定节点为17/256、15正确、88.235294%当前精度/5.859375%全量精度、0 invalid/0 truncated。
- [ ] 当前进行中：闭合20分钟节点证据并实时追加/发布报告2.352；实验继续运行，下一节点为01:28:30Z。
- [x] 20分钟节点截止逐题复算一致；22/22 validation与9/9 manifest通过，运行态健康。
- [ ] 当前进行中：完整重读并追加报告2.352，校验后精确发布；实验继续运行。
- [x] 报告2.352已实时追加，并通过章节、引用、证据、22/22 validation、9/9 manifest、术语与diff门禁。
- [ ] 当前进行中：发布2.352；实验继续运行，等待01:28:30Z的30分钟固定节点。
- [x] 2.352已由`b6fba23b32c30934c0984897f163ffbf7d09c7f8`发布，主仓local/upstream一致。
- [ ] 当前进行中：实验继续运行，等待01:28:30Z的30分钟固定节点并实时打印/更新报告。
- [x] 30分钟固定节点为17/256、15正确、88.235294%当前精度/5.859375%全量精度、0 invalid/0 truncated；相对20分钟无新完成题。
- [ ] 当前进行中：闭合30分钟节点证据并检查长请求还是异常卡死；结果实时更新报告。
- [x] 30分钟节点健康：8个TP worker存活，8卡利用率75%–97%，容器/tmux/c16 runner存活且无fatal；22/22 validation、9/9 manifest通过，判定为长请求执行中而非卡死。
- [ ] 当前进行中：完整重读并追加报告2.353，校验发布；实验继续运行。
- [x] 报告2.353已实时追加并通过章节、引用、证据、22/22 validation、9/9 manifest、术语与diff门禁。
- [ ] 当前进行中：发布2.353；实验继续运行，等待01:38:30Z的40分钟固定节点。
- [x] 2.353已由`5f9c30498992ab18dbccb28cdbf1c6e463de79ef`发布，主仓local/upstream一致。
- [ ] 当前进行中：实验继续运行，等待01:38:30Z的40分钟固定节点并实时打印/更新报告。
- [ ] 新约束：prefill与decode top-k必须与BF16 baseline实际值完全一致，top-k不得作为优化项。当前先用baseline落盘产物核对；若K1024不baseline不一致，则终止当前run并回退后重跑门禁。
- [x] baseline top-k审计确认prefill/decode均为2,048：Phase1配置、13份static preflight及实kernel `topk=2048`相互佐证；当前run实际为1,024/1,024，身份无效。
- [x] K1024无效run已人工中止；40分钟节点24/256、15正确、7 truncated仅作无效边界。容器/tmux消失，GPU0–7全部0 MiB/0%且compute空；人工中止路径未生成`wrapper.exit`，不得写为自然终局。
- [ ] 当前进行中：按新文档口径完整重读`GLM-5.2_OSCAR_苹果800适配与性能优化报告.md`，将top-k身份审计与无效run作为第二部分小节实时记录；旧两报告不再追加，暂不删除。
- [x] top-k审计证据闭合：22/22 validation、8/8 manifest通过；baseline 2048/2048、K1024无效身份、40分钟人工终止与GPU释放均已落盘。builder首轮因旧JSON键KeyError失败已保留。
- [ ] 当前进行中：已完整重读新统一报告，将第二部分top-k降低/split-K四个历史小节合并改写为“非优化项的身份纠正”，并修正后续编号、交叉引用与当前结论。
- [x] 新统一报告已完成top-k口径纠正：第二部分2.1–2.18连续，2.15明确为“非优化项”，历史K变化仅作无效对比边界；6项核心证据、22/22 validation、8/8 manifest、术语与whitespace门禁通过。
- [ ] 当前进行中：将原本被本地exclude的新统一报告强制纳入Git，与planning一起提交/HTTPS发布；发布后用TDD将活动prefill/decode top-k从1024回退至2048。
- [x] 新统一报告已由`832f03de3e77112f76ecc5584fca0714aace55a2`强制纳入Git并HTTPS发布，local/upstream一致。
- [x] K2048/2048 TDD有效红灯：只修两个合同测试期望，固定control容器、network none、无GPU下2 tests/2 failures，分别精确命中当前prefill=1024与decode/index=1024；production未改。
- [ ] 当前进行中：完整重读并在统一报告2.15实时记录K2048合同红灯；发布后只修配置与3个fail-closed消费者。
- [x] 统一报告2.15已实时追加K2048/2048 TDD有效红灯；全文2.1–2.18连续，术语、引用与whitespace门禁通过。
- [ ] 当前进行中：发布测试红灯与报告阶段；发布后只修配置与3个fail-closed消费者，再执行绿灯。
- [x] 红灯报告阶段已由`19b570f0feaf90de437ac16b3ba61f9414fa64c0`HTTPS发布。
- [x] K2048/2048最小production回退与CPU绿灯闭合：只改4个production/config文件，目标2/2、工具25/25、递归90/90、syntax通过；结构化最终24/24、9/9 manifest通过，首轮23/24唯diff数量期望错误已保留。
- [x] 统一报告2.15已追加K2048五文件TDD绿灯与证据：2.1–2.18连续，10项证据大小/hash、24/24 validation、9/9 manifest、术语、交叉引用与diff门禁通过。
- [x] 4个production/config回退、统一报告与planning已由`b8331c5a19f64b5392d15e6578f030a8f2cbef2c`通过GitHub HTTPS发布；主仓与source仓均clean/upstream。
- [ ] 当前进行中：为K2048 fixed256建立全新run/output身份并执行GPU0–7双空闲门禁，两次采样间隔至少60秒；门禁结果必须先实时写入统一报告并发布，才允许启动长跑。

- [x] split-K Phase 5/7/9活动身份迁移完成：10个配置/wrapper/测试文件统一到
  source `1e768aef6`、新Phase 6 OCI与control image；Phase1 baseline和冻结结果未改。
- [x] 正式CPU-only递归门禁闭合：Phase7 44/44、Phase9 70/70，工具20/20、
  88/88，独立静态汇总34/34，20文件证据manifest复算通过。
- [x] 迁移改动、v1–v4失败边界及v5有效结果已实时写入正式报告2.196；报告现为
  12,349行、735,972 bytes、SHA256 `ad3beced...01bc`，2.1–2.196连续，术语、
  交叉引用、证据hash和diff门禁通过。
- [x] 2.196及身份迁移已由`9924f885...00f`通过GitHub HTTPS发布并恢复
  clean/upstream。
- [x] split-K正式性能实验前双空闲门禁已通过：12:16:00Z/12:17:05Z间隔65秒，
  16/16设备行为0 MiB/0%，两个compute区段为空；结果已实时写入报告2.197。
- [x] 2.197已由`f07a3eabed307e8b9db393216030269e27965b3b`发布并恢复
  clean/upstream；固定GPU 0–7的split-K 32K/batch1正式三轮与profiler均已完成，
  独立验证59/59通过，GPU退出后全部释放。
- [x] split-K三轮中位聚合mean TTFT/mean TPOT/请求吞吐为
  `17880.081324 ms/199.605548 ms/0.023115280 req/s`；相对c349 K=1,024
  为`-14.986357%/+0.954252%/+6.776920%`，相对c349 BF16仍为
  `+42.868005%/+29.833407%/-25.937991%`，性能尚未收敛。
- [x] 正式结果已实时写入报告2.198；报告12,495行、744,864 bytes、SHA256
  `0d0e9448...d3af`，2.1–2.198连续，引用、术语、证据hash和diff门禁通过。
- [x] 报告2.198与planning已由主仓提交`c8b5d1b4cfd1b650083162145505c1e200f105cf`
  通过GitHub HTTPS发布。
- [x] 2.198身份已发布并恢复clean/upstream；随后完成新双空闲门禁及
  source `1e768aef6`的同源码BF16 32K/batch1正式对照，不能把c349 BF16写成同源码结论。
- [x] 同源码BF16前双空闲原始采样通过：13:06:47Z/13:07:52Z间隔65秒、16/16
  设备行为0 MiB/0%、两个compute区段为空；主9/9、有效独立7/7和最终12文件manifest
  全部通过。首次独立容器复核遗漏`-i`的0-byte边界已原样保留并记录。
- [x] 门禁结果已实时写入报告2.199；报告12,555行、748,963 bytes、SHA256
  `2efb0e91...3ca9`，2.1–2.199连续，引用、术语、证据hash和diff门禁通过。
- [x] 报告2.199与planning已由主仓提交`cd05f65cbdfb1f92662d59797dd35a73d67324af`
  通过GitHub HTTPS发布。
- [x] 2.199身份已发布并恢复clean/upstream；随后即时复核GPU 0–7并启动同源码
  BF16正式run。
- [x] 首次同源码BF16正式run `20260802T131400Z_source_matched_bf16_32k_b1_v1`
  在模型加载前fail-closed退出1：Phase1冻结manifest仍期望c349 repository commit/tree，
  与Stage9当前1e source及performance config不匹配，3项source检查失败；GPU始终未加载且
  退出后8卡全释放。禁止原样重跑；下一步设计Stage9专用派生baseline manifest，保留
  Phase1冻结文件不改，先做CPU-only TDD/静态门禁并实时更新报告。
- [x] 失败轮次、根因、派生manifest设计和6文件证据已实时写入报告2.200；报告12,616行、
  753,025 bytes、SHA256 `155607fb...e2b8`，2.1–2.200连续，引用、术语、hash和diff门禁通过。
- [x] 报告2.200与planning已由主仓提交`b5f31918f58c419a5ef794a11055823af95f7c4a`
  通过GitHub HTTPS发布。
- [x] 2.200身份已发布并恢复clean/upstream；随后先写Stage9 source-matched派生合同测试
  取得有效红灯，再做最小实现。
- [x] CPU-only TDD有效：目标测试先因缺失`derive_source_matched_manifest`得到1 error，
  最小实现后目标1/1、完整Stage9工具24/24通过；只改Stage9 verifier和测试，Phase1
  manifest/config及performance config未改。
- [x] 固定control容器完整静态verifier 77/77 passed，base c349→effective 1e审计和
  Phase1原始SHA/diff正确；从证据目录内重建的4文件manifest全部复算通过，首次路径错误
  manifest保留为失败边界。
- [x] TDD、最小代码、77/77静态闭合与错误边界已实时写入报告2.201；报告12,690行、
  757,645 bytes、SHA256 `845cb2a9...30f3`，2.1–2.201连续，引用、术语、hash和diff通过。
- [x] 报告2.201、两处代码与planning已由主仓提交
  `9ace8fdc81dcd88f8f0f92d77fc663151891c2a3`通过GitHub HTTPS发布。
- [x] 2.201身份已发布并恢复clean/upstream；随后新建run ID执行双空闲GPU门禁，
  门禁结果先实时写报告并发布，再启动baseline v2。
- [x] 修复后baseline v2前双空闲门禁通过：13:25:03Z/13:26:08Z间隔65秒，16/16
  设备行为0 MiB/0%、两个compute区段为空；主9/9、独立7/7及9文件manifest全通过。
- [x] 修复后双空闲结果已实时写入报告2.202；报告12,738行、760,628 bytes、SHA256
  `bd227981...3242`，2.1–2.202连续，引用、术语、证据hash和diff门禁通过。
- [x] 报告2.202与planning已由主仓提交`a80edd362f09c1e9c1a844883758a34f5e13d765`
  通过GitHub HTTPS发布。
- [x] 2.202身份已发布并恢复clean/upstream；随后即时复核GPU 0–7并完成baseline v2。
- [x] 同源码BF16正式run `20260802T133300Z_source_matched_bf16_32k_b1_v2`
  自然exit0/matrix passed：三轮中位聚合mean TTFT/mean TPOT/request throughput为
  `12507.854171 ms/151.236017 ms/0.031533464 req/s`；profile passed、8 trace/8 table、
  critical rank5/38,228 ms，退出后8卡显存全释放且无compute进程、容器已删除；即时采样
  利用率仍为100%的尾迹，约107秒后的独立稳定采样为8卡0 MiB/0%。
- [x] 正式结果已持久化为71文件、1,041,886,525 bytes（含manifest及其复核日志）；
  `formal_validation.json` 73/73 passed，69项证据manifest全部复算通过。
- [x] 同源码comparison闭合：split-K相对BF16的mean TTFT/mean TPOT/request throughput
  分别为`+42.950830%/+31.982812%/-26.696034%`，性能候选仍不合格。
- [x] 报告修改前已完整读取并确认与HEAD一致；2.203已实时追加。最终12,830行、
  766,852 bytes、SHA256 `b1ec30c3...ff3c5`；2.1–2.203连续，引用、术语、11项核心
  证据大小/hash、73/73 validation、69/69 manifest与diff门禁全部通过。
- [x] 报告2.203与planning已由主仓提交`a7f803676bd1a9bcc91e3ae65060270a9399912e`
  通过GitHub HTTPS发布；主仓与source仓均clean/upstream。
- [x] 同源码trace归因完成：两侧各8 ranks/16 prefill chunks/32,768 tokens及127个
  generation context完整；prefill尾迹20/20、残差14/14、decode interval 6/6、汇总
  32/32及独立13/13全部passed，37项证据manifest复算通过，全程CPU-only。
- [x] prefill校正wall gap `5893.667641 ms`，与profile TTFT gap吻合100.269%；主attention
  净差`2674.318982 ms`只解释正式TTFT差49.780%，rotation为`1494.044451 ms`，另有
  `2122.908463 ms`非attention kernel差和`1096.440196 ms`非kernel残差。
- [x] 连续generation起点的126个完整区间给出wall gap `59.669254 ms/token`，与profile
  TPOT gap吻合100.101%；两侧均157次AllReduce/token，观测等待差`47.916249 ms/token`，
  backend专属kernel净直接差仅`3.745473 ms/token`，不能直接归因为网络带宽回退。
- [x] 报告修改前完整读取且与HEAD一致；2.204已实时追加。最终12,932行、773,848 bytes、
  SHA256 `2b3ad795...bd4d7`；2.1–2.204连续，术语、引用、11项核心证据、32/32、
  13/13、37/37 manifest及diff门禁全部通过。
- [x] 报告2.204与planning已由主仓提交`5394d8e`并通过GitHub HTTPS发布；两仓恢复
  clean/upstream。
- [x] rank同步边界、rotation调用几何与缓存寿命只读审计完成；16/16主验证、11/11
  独立复核及13/13 manifest通过。否决跨层/chunk缓存、每层current-history常驻scratch和
  NCCL修改；最小候选冻结为inverse rotation+BF16 add融合，尚未宣称实测收益。
- [x] 修改前完整重读报告且确认与HEAD一致；2.205已实时追加。最终13,038行、781,140
  bytes、SHA256 `ed72f280...486f6`；2.1–2.205连续，术语、引用、7项核心证据、
  16/16、11/11、13/13 manifest及diff门禁全部通过。
- [x] 报告2.205与planning已由主仓提交`a4d685af2964d00e4835e2f5af950fc83013a3d7`
  并通过GitHub HTTPS发布；两仓恢复clean/upstream。
- [x] inverse rotation+BF16 add融合目标合同测试已新增；固定control image、network none、
  无GPU暴露的有效红灯精确命中缺失`oscar_mla_rotate_add`，production未改。
- [x] 目标测试已由source提交`5f03c7491d8e956d58b5ed96a1f089bcf39101d3`通过
  GitHub HTTPS发布，source pre-commit全部通过；报告2.206已实时追加。
- [x] 报告2.206门禁通过：13,083行、783,965 bytes、SHA256
  `1104bbc0...1168`；2.1–2.206连续，术语、引用、测试/production哈希、source身份、
  source pre-commit与两仓diff门禁全部通过。
- [x] 主仓gitlink、报告2.206与planning已由`9171e24a1700de499f2648ce523a22684e9fc64f`
  发布；两仓恢复clean/upstream后完成最小production实现。
- [x] inverse rotate+add实现通过CPU合同、Triton interpreter与source完整pre-commit；
  source提交`d0d22489b265fc98f9f829dbcfca5e815543d337`已HTTPS发布，苹果800 CUDA
  8/16,384行逐值门禁尚未执行。
- [x] 报告2.207门禁通过：13,147行、787,890 bytes、SHA256
  `449ce199...2cf5`；2.1–2.207连续，术语、引用、4项文件hash、source commit/tree/patch、
  CPU合同/interpreter、pre-commit与两仓diff门禁全部通过。
- [x] 主仓gitlink、报告2.207与planning已由提交
  `090f376572f915c4925111f1b18ed1a29d85010e`通过GitHub HTTPS发布；两仓恢复
  clean/upstream。
- [x] 苹果800 CUDA门禁前有效双空闲为15:10:43Z/15:12:17Z、间隔94秒，两轮
  GPU0–7均0 MiB/0%、compute空；11/11 validation和7/7 manifest通过。57秒中间轮次
  保留但不计有效。
- [x] 报告2.208门禁通过：13,180行、790,019 bytes、SHA256
  `f0eedc15...34d6`；2.1–2.208连续，术语、引用、5项证据hash、11/11 validation、
  7/7 manifest与两仓diff门禁全部通过。
- [x] 报告2.208与planning已由主仓提交`b60bb1faa30b927bb82297165dcf94e2275d882a`
  通过GitHub HTTPS发布；两仓恢复clean/upstream。
- [x] 固定GPU0 correctness gate自然exit0；8行decode和16,384行prefill均相对旧路径
  bitwise equal、max error=0、output pointer复用。退出后显存/compute已释放但有12%利用率尾迹。
- [x] 15:16:47Z稳定采样8卡全部0 MiB/0%、compute空；GPU0 correctness的14/14
  validation与11/11 manifest通过，证据13文件/15,944 bytes。
- [x] 报告2.209门禁通过：13,234行、793,297 bytes、SHA256
  `95d3c461...1569e`；2.1–2.209连续，术语、引用、7项证据hash、14/14 validation、
  11/11 manifest与两仓diff门禁全部通过。
- [x] 报告2.209与planning已由主仓提交`4495bf899b52a82022fb2d99a4da301fb80d1cf7`
  通过GitHub HTTPS发布；两仓恢复clean/upstream。
- [x] 固定GPU0 warm-up微基准前新双空闲为15:19:45Z/15:20:47Z、间隔62秒；两轮
  GPU0–7全idle，9/9 validation与7/7 manifest通过，benchmark脚本CPU compile通过。
- [x] 报告2.210门禁通过：13,277行、795,924 bytes、SHA256
  `2c52797e...0788`；2.1–2.210连续，术语、引用、5项证据hash、9/9 validation、
  7/7 manifest与两仓diff门禁全部通过。
- [x] 报告2.210与planning已由主仓提交`e50299fb37707dd822281c349d99c6c7b3b26602`
  通过GitHub HTTPS发布；两仓恢复clean/upstream。
- [x] 固定GPU0 benchmark自然exit0：8行中位0.05681664→0.02765312 ms/call，16,384行
  0.91018238→0.85391359 ms/call；bitwise保持，peak临时分配分别少16KiB/32MiB。
- [x] 微基准稳定采样、16/16 validation与最终12/12 manifest已闭合；最终子目录
  14文件、44,331 bytes，生成的pyc已显式纳入manifest。
- [x] 报告修改前已完整读取且与HEAD一致；2.211已实时追加。最终13,343行、
  800,549 bytes、SHA256 `94c4c1d0...3350`；2.1–2.211连续，术语、引用、
  16/16 validation、12/12 manifest与diff门禁全部通过。
- [x] 报告2.211与planning已由主仓提交
  `296a117d9b137e785684bea9f1dd8f75b56e9269`通过GitHub HTTPS发布；主仓恢复
  clean/upstream。
- [x] CPU-only审计与正式候选镜像/运行身份已迁移到source `d0d22489...`；报告
  2.223已实时追加并通过章节、术语、引用、证据hash和diff门禁，发布前不启动256题精度。
- [x] 报告2.223与活动身份迁移已由主仓提交`8fd853f3beb9048af66f26d9efc509bf55f03031`
  通过GitHub HTTPS发布，两仓clean/upstream。
- [x] 同256题精度前双空闲门禁为16:33:31Z/16:34:41Z、间隔70秒，两轮8卡
  全idle；有效validation 10/10、最终manifest 6/6通过。报告2.224已实时追加并通过
  章节、术语、引用、证据hash和diff门禁；下一步只发布，发布完成前不启动候选精度。
- [x] 2.224已由`2ccdb7ee0fae343e2660ccbafd09010ba471c450`发布；启动审计发现两个
  official_v5 wrapper的candidate manifest仍为旧`dd16...`。TDD最小迁移到d0d
  `f8e73d...`后，目标1/1、聚合24/24、Phase7 20/20、Phase9 89/89、独立27/27通过。
- [x] 精度入口身份修复和失败边界已实时写入报告2.225；章节、术语、引用、三文件hash、
  27/27 validation、20/20 manifest与diff门禁通过。下一步只发布；发布后必须重新执行
  双空闲门禁，不能复用2.224授权启动长跑。
- [x] 报告2.225与修复已由`f47cc11dcad71196f7fde7e736abf39d3e57ecb6`发布；新双空闲
  为16:43:52Z/16:44:58Z、间隔66秒，两轮全idle，validation 10/10、manifest 3/3
  通过。报告2.226已实时写入并通过章节、术语、引用、证据hash和diff门禁；下一步只
  发布，之后才启动256题长跑。
- [x] 报告2.226已由`ec7d828668666f4edf00779ac7604992ad7dd6b3`发布；正式run
  `20260802T1648Z_candidate_inverse_fusion_splitk_fast256_c16_v1`自然exit0。固定256题
  结果为0/256、0.000000%，256个答案提取失败、233个截断、0个请求失败；completion
  tokens均值7,463.37109375，总时长26,229.182457秒。独立15/15、持久化证据最终
  manifest 279/279通过，退出后8卡0 MiB/0%、compute空。候选确定失去32K复测资格。
- [x] 完整重读正式记录并实时追加2.227精度失败终局；报告现为14,055行、847,190 bytes、
  SHA256=`6354bf3366195f22ec68672518ee3b77e464ebfa7364fbd8d8cbc63973cc5489`，2.1–2.227连续，
  术语、交叉引用、8项核心证据hash、独立15/15、最终manifest 279/279和diff门禁通过。
- [x] 报告2.227与planning已由主仓提交`951ded274024d65faf4012b6edaaa27f4630cfe5`
  通过GitHub HTTPS发布；主仓与source均已恢复clean/upstream。
- [x] FP32 latent位级诊断的实际shape审计与GPU双空闲门禁完成：每卡8个head，固定
  8/128/16,384行×512；00:27:07Z/00:28:23Z两轮8卡全idle、间隔76秒，独立
  21/21、manifest 8/8通过。报告2.228已实时追加为14,111行、850,943 bytes、
  SHA256=`8d6ec40c5e825f606839cdf89e6fe07bb5b8c2613850976bbd68bf0b07179788`，
  2.1–2.228连续，术语、引用、核心hash和diff门禁通过。
- [x] 报告2.228与planning已由主仓提交`875d75a80f427f4e133b1c55ab8ad25c00e22f48`
  发布；即时8卡全idle后固定GPU0完成FP32 latent 8/128/16,384行对照，三例均
  bitwise equal、0 mismatch、max/mean error 0、output指针复用，run exit0；固定control
  独立19/19、manifest 12/12通过，退出后8卡全idle。该结果否定helper内直接dtype误差，
  不能解释端到端0/256。
- [x] 实时追加报告2.229；当前14,169行、854,942 bytes、
  SHA256=`b9d9137ca9b74f3cfd42fba3a7be321676ed8a65a079033d39b06ef4c29a3b5f`，
  2.1–2.229连续，术语、引用、核心hash、19/19、12/12与diff门禁通过。
- [x] 报告2.229与planning已由主仓提交`444adac`发布；production已最小回退到2.183
  pre-fusion路径。有效TDD为红灯exit1/绿灯exit0，decode/store blob与1e逐字节一致；
  source `83320e1205b65b551633eb4e32c4858987ba0516`、tree `2d067ea6...36af`
  已通过pre-commit和GitHub HTTPS发布，独立19/19、manifest 15/15通过。
- [x] 实时追加报告2.230；当前14,231行、859,195 bytes、
  SHA256=`b6a6c6457f61328376f5182c35206b76d0e6528c8eaba427c15290c1cad64030`，
  2.1–2.230连续，术语、引用、source/blob/hash、19/19、15/15与diff门禁通过。
- [x] 报告2.230、source gitlink与planning已由主仓提交
  `365a7a64824d950d59cb51fd167366b563bc9663`通过GitHub HTTPS发布；两仓恢复
  clean/upstream。
- [x] Phase6输入迁移完成：目标红灯1/1、绿灯1/1、完整builder单测2/2、独立19/19、
  manifest 9/9；只更新833 commit/tree/tag和Dockerfile hash，base/rotation/runtime
  expectation不变，Phase5/7/9尚未提前替换。
- [x] 实时追加报告2.231；当前14,285行、862,566 bytes、
  SHA256=`2b1377f2e4a9cd883ac120dce1100ba0456e89f812b4aff510f5398f8352aed4`，
  2.1–2.231连续，术语、引用、三文件hash、19/19、9/9与diff门禁通过。
- [x] 报告2.231、Phase6输入与planning已由主仓提交`79328a28383f5608c13c4ff208d77b47a653d0b0`
  发布；CPU-only builder自然exit0，新OCI manifest/config/layer为`847b8dcc...1d6b` /
  `3c06df1c...d8ba` / `92d494e2...ca35`，33层、5,298 members、无native/whiteout，
  构建时独立22/22、核心manifest 7/7通过。
- [x] 实时追加报告2.232；当前14,343行、866,416 bytes、
  SHA256=`024356402e49bcd6c0019c571cf120b1dac52f84e08cc136335cd28499fbf5f8`，
  2.1–2.232连续，术语、引用、OCI/核心hash、22/22、7/7与diff门禁通过。
- [x] 报告2.232与planning已由主仓提交`87c7597`发布；同一OCI递归verifier自然exit0，
  source4,744文件精确tree、32个base layers、4份rotation、runtime expectation、7个native
  继承和3项环境均通过；独立19/19、最终核心manifest 14/14。
- [x] 实时追加报告2.233；当前14,390行、869,849 bytes、
  SHA256=`b1c502a48f6c0ad4c93af220b910321a7e7ebe7986d2ce3db8df385efd6db6bc`，
  2.1–2.233连续，术语、引用、递归证据hash、19/19、14/14与diff门禁通过。
- [x] 报告2.233与planning已由主仓提交`ddf0ce8a1d55072343cb8e04132158bddd344314`
  发布；833 overlay只补6个冻结Phase0 native链接，最终4,749 regular/6 symlink；CPU-only
  canonical source import自然exit0，两处prefill K=768、三个实际metadata字段和CUDA未初始化
  均通过，结构化validation及15/15 manifest通过。
- [x] 实时追加报告2.234；当前14,459行、874,787 bytes、
  SHA256=`d062984ad3f70c73cc4073c369c8b5cabccf91b90a8b89fa2526200e2486986b`，
  2.1–2.234连续，术语、引用、9项核心证据hash、15/15 manifest与diff门禁通过。
- [x] 报告2.234与planning已由主仓提交`6781369f8e02cd82b30ddc8ba9d4f5d9c56ac114`
  发布；目标tag预先不存在后，CPU-only skopeo导入自然exit0，33 blobs/config/manifest完整。
  daemon image ID/tag/33层/末diff-ID/8项labels独立5/5通过，daemon证据manifest 9/9；
  异步迟到写入边界已保留，2.234原manifest恢复15/15。
- [x] 实时追加报告2.235；当前14,522行、879,394 bytes、
  SHA256=`51052a4ce5491896376f305bc39e934e84fbc1975050a28a99c5aad74556d89b`，
  2.1–2.235连续，术语、引用、9项核心证据hash、9/9与15/15 manifest及diff门禁通过。
- [x] 报告2.235与planning已由主仓提交`e180b95b8bf0e46d8f0a002c0020a17ae246a5b2`
  发布；Stage9 base输入先做有效红灯1/1，再只迁移Dockerfile首行到833，目标1/1、完整
  工具24/24、compile、独立19/19与manifest 16/16通过；目标control tag仍不存在。
- [x] 实时追加报告2.236；当前14,577行、883,231 bytes、
  SHA256=`8cc024da34f19131e39c082cd0da19ac30c66bb31244bbe25c005af1371a4b38`，
  2.1–2.236连续，术语、引用、10项核心证据hash、16/16 manifest与diff门禁通过。
- [x] 报告2.236、Stage9输入与planning已由主仓提交
  `b804d4d46a45c5c108928ef2a6a2aeff38033323`发布；空context、`--pull=false`构建
  `oscar-glm-stage9-runtime:83320e120`自然exit0。control ID=`62568e2e...8013`、34层，
  前33层与base一致，独立10/10、build manifest 7/7通过。
- [x] 实时追加报告2.237；当前14,625行、886,207 bytes、
  SHA256=`1f4bc67d9253067214b2d8c6d41536c3fd72502aaf74841cb363f7edd9fc7430`，
  2.1–2.237连续，术语、引用、8项核心证据hash、7/7 manifest与diff门禁通过。
- [x] 报告2.237与planning已由主仓提交`1372aed20d3ce5c7c80213a71519b7f26eb13433`
  发布；833 Stage9 CPU runtime自然exit0。Python/glibc、控制包、pre-fusion decode/store
  blob、rotate-then-add、source路径及CUDA未初始化均通过；独立18/18、合并manifest14/14。
- [x] 实时追加报告2.238；当前14,674行、889,322 bytes、
  SHA256=`5172b435240594494861567fe136c2ce491f494a012b7cac68f51754e272cd49`，
  2.1–2.238连续，术语、引用、8项核心证据hash、14/14 manifest与diff门禁通过。
- [x] 报告2.238与planning已由主仓提交`3a55badd7d7764a730f40a1205ae7529dfa84aa8`
  发布；driver import前双空闲门禁为01:38:28Z/01:39:39Z、间隔71秒，两轮8卡均
  0 MiB/0%、compute空；独立10/10、manifest5/5。
- [x] 实时追加报告2.239；当前14,704行、891,349 bytes、
  SHA256=`9d947516fe55a48a977a297d308a8519e69fa4953a1c18d6158e8f4b3085c1c1`，
  2.1–2.239连续，术语、引用、7项核心证据hash、5/5 manifest与diff门禁通过。
- [x] 报告2.239与planning已由主仓提交`554e0763d3f299e28318d2861840330930cfabee`
  发布；即时8卡全idle后固定GPU0运行首次probe。命令遗漏prefill K=768环境，native导入后
  在默认K=0断言exit1，stdout0 bytes；退出后8卡全idle，独立9/9、失败manifest9/9。
- [x] 实时追加报告2.240；当前14,743行、894,223 bytes、
  SHA256=`d1ed1836dc1d0af45b34ba6725cb88371b7f00100f4fe5fccfe28eda89c05088`，
  2.1–2.240连续，术语、引用、10项核心证据hash、9/9 manifest与diff门禁通过。
- [x] 报告2.240与planning已由主仓提交`3298dca349f300c322220eb6f826e59a50569d12`
  发布；重试前双空闲门禁为01:46:48Z/01:47:56Z、间隔68秒，两轮8卡均0 MiB/0%、
  compute空；独立10/10、manifest5/5。
- [x] 实时追加报告2.241；当前14,779行、896,628 bytes、
  SHA256=`4926a169fb60d690b43183a729aac1a8af4ed62c115b46eed7b5e4acb4b16064`，
  2.1–2.241连续，术语、引用、7项核心证据hash、5/5 manifest与diff门禁通过。
- [x] 报告2.241与planning已由主仓提交`398cda026afd61e3f88f3a9ce25fc6cd3d44896d`
  发布；即时8卡全idle后固定GPU0、相同probe和显式K=768重试自然exit0，退出后8卡
  全释放。native/production/K/metadata/rotation/runtime identity通过，独立25/25、最终
  manifest22/22；validator两轮Python3.8兼容性失败已保留。
- [x] 实时追加报告2.242；当前14,832行、900,643 bytes、
  SHA256=`8a3aabe76939ebbb1bcfd890cf4e2d4777d687385b418da2546fae5698597d6f`，
  2.1–2.242连续，术语、引用、13项核心证据hash、25/25、22/22 manifest与diff门禁通过。
- [x] 报告2.242与planning已由主仓提交`46eeae0b21018ca7a5385f6c27bc82155265a9c4`
  发布；无GPU控制容器实测依赖、78层与max合同，生成833 Phase6 717-byte canonical
  runtime import，SHA=`9bdfc8ca...3b20`；独立17/17、跨目录manifest11/11。
- [x] 实时追加报告2.243；当前14,884行、904,380 bytes、
  SHA256=`d8d6dfa83abb2db0852b359402d2d9810084267c562b3178a821d5f713a21e03`，
  2.1–2.243连续，术语、引用、10项核心证据hash、17/17、11/11 manifest与diff门禁通过。
- [x] 报告2.243与planning已由主仓提交`bcba16d0fd23dc03269bb980cae102e2b104549a`
  发布；12个活动文件最小迁移到833，目标1/1、Phase9工具24/24、Phase7单测20/20、
  Phase7/9递归44/44与70/70、独立31/31、manifest40/40通过。
- [x] 实时追加报告2.244；首个patch因非唯一尾句把本节插到2.222后，发布前章节门禁捕获并
  移到文件末尾。当前14,956行、909,359 bytes、
  SHA256=`e05b2d759581388620c149361ef9ae84f8b597f7f9fef61b402a17ad61a96ec9`，
  2.1–2.244连续，术语、引用、17项核心证据hash、31/31、40/40 manifest与diff门禁通过。
- [x] 报告2.244、12个活动文件与planning已由主仓提交`688cb910e124cabcb200c2dbb4b86121ce7b7b92`
  发布；02:19:01Z/02:20:01Z两轮8卡全0 MiB/0%、compute空，间隔60秒，10/10、manifest6/6。
- [x] 实时追加报告2.245；当前14,984行、911,282 bytes、
  SHA256=`07eef11206926d3c38cca593bb1893359d84926ef0c705e92b5f15cb4e7e4544`，
  2.1–2.245连续，术语、引用、8项核心证据hash、10/10、6/6 manifest与diff门禁通过。
- [x] 报告2.245与planning已由主仓提交`1f56915229436e34834d0730285d1eae6f1aadc3`
  通过GitHub HTTPS发布；主仓/source均clean且HEAD等于upstream。
- [ ] 即时复核8卡仍全空闲，再启动833同256题精度；每10分钟同步completed/correct/accuracy
  与GPU状态。精度通过前不得运行32K。
- [x] d0d活动Phase5/7/9的10个配置/wrapper/合同已完成最小迁移；补齐新overlay的
  6个Phase0 native链接后，Phase7/9递归44/44、70/70，单测20/20、89/89，聚合
  24/24、独立身份81/81和41项manifest均通过。失败边界与有效结果已实时写入报告
  2.223；下一步只发布本阶段，发布完成前不启动256题精度。
- [x] Phase6目标合同已先改为期望d0d身份，有效红灯精确命中candidate manifest仍为
  1e source；下一步只改Phase6输入与Dockerfile取得绿灯。
- [x] Phase6输入、Dockerfile与定向合同已最小迁移到d0d；目标1/1、完整2/2、独立
  6/6和compile/JSON/diff门禁通过，尚未构建OCI。
- [x] 报告修改前已完整读取且与HEAD一致；2.212已实时追加。最终13,381行、
  803,130 bytes、SHA256 `fc6afee3...79d6`；2.1–2.212连续，术语、引用、文件hash
  与diff门禁通过。
- [x] 2.212与Phase6输入合同已由主仓提交
  `4eec294e9e151af77d5da3dc962d99f0bc9e1a0b`通过GitHub HTTPS发布；两仓
  clean/upstream。
- [x] d0d Phase6 builder v1解释器失败已实时写入报告2.213；报告13,412行、
  804,975 bytes、SHA256 `d213049f...de06`，章节/术语/证据/diff门禁通过。
- [x] 2.213已发布；恢复clean/upstream后固定control image Python 3.12.13
  新建v2构建目录，成功结果先写报告再运行verifier。
- [x] v2 safe.directory失败已实时写入报告2.214；最终13,441行、806,759 bytes、
  SHA256 `4f8e4ee6...3048`，章节/术语/证据/diff门禁通过。
- [x] 2.214已发布；恢复clean/upstream后新建v3并精确配置两个safe.directory。
- [x] v3固定Python 3.12.13/4 CPUs/network none/no GPU构建成功；build状态built、
  8/8身份检查通过，候选image/config `0b33973c...43a8`，尚未运行递归verifier。
- [x] 报告修改前完整读取且与HEAD一致；2.215已实时追加v3 build结果。最终
  13,487行、809,712 bytes、SHA256 `cd70b2f2...071b`，章节/术语/引用/hash/diff通过。
- [x] 2.215已发布；恢复clean/upstream后运行递归verifier。
- [x] v3递归verifier自然passed；10/10独立结构检查与8/8核心manifest通过，4,744
  source、32 base layers、4 rotation、runtime expectation、7 native均闭合。
- [x] 报告修改前完整读取且与HEAD一致；2.216已实时追加递归验收。最终13,527行、
  812,310 bytes、SHA256 `26306b79...c6d7`，章节/术语/引用/hash/diff通过。
- [x] 2.216已发布；恢复clean/upstream后导入daemon并独立审计。
- [x] d0d Phase6 OCI已由skopeo 1.4.1导入daemon；5/5独立身份检查通过，image ID、
  tag、33层、末层diff-ID与8项labels全部匹配。
- [x] 报告修改前完整读取且与HEAD一致；2.217已实时追加daemon导入。最终13,566行、
  814,627 bytes、SHA256 `0b132d0e...1e79`，章节/术语/引用/hash/diff通过。
- [x] 2.217已发布；恢复clean/upstream后迁移Stage9 Dockerfile静态合同。
- [x] Stage9 base目标合同已先改为期望d0d Phase6 tag，有效红灯精确命中Dockerfile仍为
  1e base；下一步只改Dockerfile首行取得绿灯。
- [x] Stage9 Dockerfile首行与定向合同已最小迁移；目标1/1、完整24/24、临时pycache
  compile及diff门禁通过，尚未构建control image。
- [x] 报告修改前完整读取且与HEAD一致；2.218已实时追加Stage9静态迁移。最终
  13,599行、816,547 bytes、SHA256 `5a295bdf...3b9c`，章节/术语/引用/hash/diff通过。
- [x] 2.218已发布；恢复clean/upstream后构建Stage9 control image。
- [x] Stage9 control image构建自然exit0；image `9a8efeba...c87b6`，10/10身份继承
  审计通过，34层精确继承d0d Phase6的33层，尚未CPU runtime验证。
- [x] 报告修改前完整读取且与HEAD一致；2.219已实时追加control build。最终13,643行、
  819,064 bytes、SHA256 `c9379d9b...1151`，章节/术语/引用/hash/diff通过。
- [x] 2.219已发布；恢复clean/upstream后运行新control CPU runtime preflight。
- [x] 新control CPU runtime v2自然exit0；Python/glibc/包版本/两文件hash/helper/CUDA未初始化
  11/11通过，含失败边界的最终12/12 manifest通过。
- [x] 报告修改前完整读取且与HEAD一致；2.220已实时追加CPU runtime。最终13,687行、
  821,998 bytes、SHA256 `e94416fc...b10d`，章节/术语/引用/hash/validation/manifest/diff通过。
- [x] 2.220已发布；恢复clean/upstream后进入活动身份迁移前的driver-injected门禁。
- [x] driver-injected runtime import前双空闲有效first/valid为16:05:09Z/16:06:44Z，
  间隔95秒；两轮8卡全idle，9/9 validation与4/4 manifest通过，27秒轮次不计有效。
- [x] 报告修改前完整读取且与HEAD一致；2.221已实时追加双空闲门禁。最终13,719行、
  824,146 bytes、SHA256 `41749a4c...e198`，章节/术语/引用/hash/9/9/4/4/diff通过。
- [x] 2.221已发布；恢复clean/upstream后即时复核并运行driver-visible import探针。
- [x] 固定GPU0 driver-visible probe自然exit0；15/15 validation与11/11 manifest通过，
  before/after CUDA均未初始化、退出8卡全idle；canonical runtime import已生成。
- [x] 报告修改前完整读取且与HEAD一致；2.222已实时追加driver-visible import。最终
  13,770行、827,534 bytes、SHA256 `d2a627f7...821a`，章节/术语/引用/hash/15/15/11/11/diff通过。
- [ ] 当前只发布2.222；恢复clean/upstream后迁移Phase5/7/9活动身份并跑CPU递归门禁。
- [x] d0d活动身份聚合合同已先更新，有效红灯精确命中Phase5配置仍为1e source；
  下一步统一迁移活动配置与wrapper。

- [x] 当前c349 BF16与K=1,024 OSCAR的CPU-only同源trace归因完成：prefill stage1
  多6,915.518130 ms，解释正式TTFT差距81.197514%；decode generation wall多
  50.880079 ms/token，其中相同156次AllReduce的观测等待多41.734943 ms/token，
  OSCAR/BF16专属kernel净直接差仅3.537729 ms/token。
- [x] 8/8 rank、16/16 prefill chunk、每rank 127个generation窗口完整；40/40
  validation与6/6 manifest通过，结果已实时写入报告2.170。
- [x] 当前只完成报告2.170的章节、引用、术语、hash和diff门禁并发布；恢复
  clean/upstream后再做21个`full` indexer层调用结构与decode等待首个分叉点的
  CPU-only源码审计，未形成正确性合同前不修改production或申请GPU。
- [x] 报告2.170发布前门禁通过：10,877行、641,968 bytes、SHA256
  `6de38fc961cf3ffcfee5ae507814b2ced34410f14f5d164f9e053d65e05a414c`；2.1–2.170连续，
  2.169交叉引用、术语、7项证据hash、6/6 manifest与diff check全部通过。
- [x] 报告2.170与planning已由主仓提交
  `54dc4825e4906e828713a78d9966b5884c4f620d`通过GitHub HTTPS发布。
- [x] 2.170发布身份已由`7bdb0f08893a3da87b05a0a9061d37149b4ad375`推送，两仓
  clean/upstream；已开始CPU-only源码审计。
- [ ] 初步源码反证：`full/shared`只控制indexer是否更新，全部层仍调用同一MLA
  attention wrapper；OSCAR/BF16在cache update和`forward_mqa`内才分叉。因此先审计
  trace分析器窗口边界，不能把57→78/106→148解释成OSCAR多执行21个模型层。
- [ ] 已确认prefill分析器按原始`execute_context_*_generation_0`注释窗口直接归类，
  只以kernel起始时间是否落入窗口判断，不按kernel名或层数裁剪。下一步比较两侧原始
  嵌套注释和chunk尾部，检查BF16是否有异步kernel落在注释窗口之外。
- [ ] chunk级全kernel差分显示BF16不只少21次attention，还同步少21层的MoE、norm及
  42次AllReduce等整套kernel；这不能由OSCAR attention分支解释。当前首要假设改为
  BF16异步GPU尾部落在CPU execute_context注释结束后，先用原始trace核验窗口外尾部；
  若证实，须实时更正2.170的81.197514%解释边界后才能选择候选。
- [ ] rank0原始trace已证实：BF16每个chunk注释后固定有1,000个尾部kernel，其中
  attention21、MoE44、AllReduce44、norm44；K1024通常有35个尾部kernel。扩展到下一
  execute_context起点后，两侧每chunk均为attention78、MoE150、AllReduce157、norm156。
  下一步用固定CPU容器覆盖8 ranks×16 chunks并要求128/128结构一致，再校正归因。
- [x] 8 ranks×16 chunks的GPU尾部校正完成，20/20 validation与4/4 manifest通过；
  校正wall差8,833.030999 ms对profile TTFT差闭合100.105365%。主attention差改为
  5,699.560312 ms，解释正式TTFT差66.920529%，非attention残差3,133.470687 ms。
- [x] 2.170的57→78结构解释与81.197514%占比已在报告2.171中明确更正；decode结论
  不受prefill窗口校正影响。当前只做发布门禁，不修改production或使用GPU。
- [x] 报告2.171发布前门禁通过：10,955行、647,062 bytes、SHA256
  `154276ac7f8edd94b8141df9021161b7d9b913aefd71b701675e83f3475ef8de`；2.1–2.171连续，
  2.170引用、纠正边界、6项证据hash、20/20 validation、4/4 manifest及diff通过。
- [x] 报告2.171与planning已由主仓提交
  `a2bd4813b8f9f08325767f8e888508a6282d0207`通过GitHub HTTPS发布。
- [ ] 当前只发布本身份并恢复clean/upstream；随后另建独立CPU-only residual分析器，
  不修改2.171已冻结脚本/结果，先分解3,133.470687 ms prefill残差。
- [x] 2.171发布身份`ad13fe91398e0da56123a1983d2b8ca47295b815`推送后两仓clean；
  独立residual分析器精确复现2.171窗口，14/14 validation与3/3 manifest通过。
- [x] 非attention kernel差2,008.605085 ms解释wall残差64.101608%，剩余
  1,124.865601 ms不作无证据归因。`_rotate_latent_kernel`为1,494.356808 ms，占
  后端非主attention净差84.620127%；prefill NCCL仅多74.521532 ms，MoE基本持平。
- [x] 上述结果已实时写入报告2.172；当前只做发布门禁，发布前不审计或修改production。
- [x] 报告2.172发布前门禁通过：11,016行、651,046 bytes、SHA256
  `6e63612373d03728fd4c5f74e5cf33ca7d9379ba4a09fa65f72f5d1ad6a65802`；2.1–2.172连续，
  2.171引用、4项证据hash、14/14 validation、3/3 manifest及diff通过。
- [x] 报告2.172与planning已由主仓提交
  `fe8cd00ad31315613f8617f79bcb9d245c1d7640`通过GitHub HTTPS发布。
- [ ] 当前只发布本身份并恢复clean/upstream；随后只读审计rotate源码与历史候选。
- [x] 2.172发布身份`abcfb7c63e80e9d6eda946aad62347983e97377d`推送后两仓clean；
  rotate只读审计已开始。
- [ ] 历史证据确认rotate已从约3,390.942 ms优化到1,486.435 ms（-56.164548%）；
  TF32因INT2-restored精度失败淘汰，IEEE参数sweep已实施。当前1,494.357 ms接近已优化
  水平，下一步继续核对融合/复用/真实几何候选，不能把整项视为未优化收益。

- [x] c349 Phase 6 OCI、overlay、daemon identity、driver runtime import 已完成并发布。
- [x] c349 Stage 9 控制镜像已完成并发布。
- [x] Phase 1/5/7/9 四级配置、9 个正式 wrapper 与 Phase 9 身份测试已迁移；
  Phase 7/9 工具测试分别为 20/20、80/80，11 个 Python 文件编译、9 个 shell
  语法门禁和递归 verifier 66/66 全部通过。
- [x] 上述静态迁移结果已实时写入 `OSCAR精度与性能优化记录.md` 2.121，并由主仓
  提交 `4abd9fcab3a761862195fe1a387de91d7d2cacd2` 发布。
- [x] 从 clean/published 检查点完成两次间隔 86 秒的 8 卡空闲检查和
  driver-injected preflight：66/66 静态检查、固定环境 import 与服务参数解析均
  通过，`cuda_initialized=false`。
- [x] preflight 结果已实时写入报告 2.122，并由主仓提交
  `013baf64743239f356c3f34056295dad5147d5e6` 发布。
- [x] 从 clean/published 检查点完成固定 GPU0 的单卡 cold-cache production CUDA
  回归：当前129个节点全部通过，0 skipped/failed，独立cache为380文件。
- [x] CUDA 结果已实时写入报告 2.123，并由主仓提交
  `0ff69a519efb818fbdcb2a3f966942404e922f99` 发布。
- [x] 从新的 clean/published 检查点复跑同一 32K/batch1/output128/TP8 正式三轮；
  三轮、profiler 与 54/54 正式验证均通过，已与 BF16、67a 和 c0bc 完成复算。
- [x] c349 已恢复并略优于 67a 的实测水平：TTFT/TPOT/吞吐相对 67a 为
  `-0.064088%/-2.587863%/+1.272080%`；因两者 source tree 完全相同，不把小差异
  归因于源码优化。相对 BF16 的 TTFT 仍回退 `143.610813%`。
- [x] 正式结果已实时写入报告 2.124；章节、证据、术语、交叉引用与 diff 门禁均
  已通过，下一步只发布本检查点。
- [x] 用固定 Python/analyzer 对 c349、67a 与 c0bc 的 32K trace 完成 CPU-only
  同口径归因：profile 解释 c0bc→c349 正式 TTFT 恢复的 98.373255%，stage1 解释
  prefill wall 恢复的 103.006226%；8/8 rank、16/16 chunk 均恢复。
- [x] trace 归因结果已实时写入报告 2.125；章节、证据、术语、引用与 diff 门禁均
  已通过，下一步只发布本检查点。
- [x] 汇总现有 c349 trace 与已淘汰候选证据，CPU-only 排序剩余 stage1 机会；
  launch sweep、has-history gate、reload/manual/maxnreg 与 full compact 均不重试。
- [x] 唯一入选下一筛选的是 partial compact-load 分解：packed-only 与
  scale/zero-only；先做 CPU-only SM80 编译门禁，不修改 production 或申请 GPU。
- [x] 报告 2.126、5/5 ranking validation、2/2 evidence manifest、章节/引用/术语与
  diff 门禁均已通过，下一步只提交并通过 HTTPS 发布本阶段。
- [x] 发布报告 2.126 后扩展现有离线工具/测试形成 TDD 红灯；预期红灯为14项中
  1 failure/2 errors，确认旧工具缺少两个partial variant和对应门禁。
- [x] 最小实现format v9、两个互斥partial constexpr/variant及逐项门禁；修正一次
  新增函数边界错误后，固定c349容器最终14/14 tests passed，8卡全空闲。
- [x] 报告2.127共8,636行、章节1.1–1.5/2.1–2.127连续，引用、术语、哈希与diff
  门禁通过；下一步只提交并通过HTTPS发布本阶段。
- [x] 2.127及工具/测试已由`1698c27…dbfd`发布；固定c349容器CPU-only SM80编译
  33成功/3个既有t8 variant拒绝，baseline shared复现，validation 20/20通过。
- [x] packed-only虽减少8条load但寄存器升至238；scale/zero-only虽降至193寄存器
  但load增加32条。两者promotion均false，按预设门禁关闭，不申请GPU。
- [x] 报告2.128共8,680行，章节1.1–1.5/2.1–2.128连续，validation 20/20、
  manifest 14/14、引用、术语与diff门禁均通过；下一步只提交并HTTPS发布。
- [x] 2.128与planning已由`42410fd…abdc`通过HTTPS发布；主仓库与真实源码
  submodule `glm52_oscar_vllm`均clean，HEAD分别等于upstream。
- [x] 发布后重新基于c349 trace做CPU-only机会排序；排序后96.728257%的active tile
  为full-history，而现有`has_bf16`只门禁两次dot，不门禁BF16 value物化。
- [x] 唯一选择`lazy_reload_bf16_values_under_has_bf16`：保留FP32 accumulator与
  history数学，先要求二进制变化且stack/shared/register均不增加；门禁前不申请GPU。
- [x] 报告2.129共8,729行，章节1.1–1.5/2.1–2.129连续，ranking 10/10、
  manifest 2/2、引用、术语与diff门禁通过；下一步只提交并HTTPS发布。
- [x] K=1,536 + legacy decode 的有效 256 题快速筛选轮次
  `20260801T1517Z_candidate_topk1536_legacy_fast256_c16_v2` 已自然退出码 0：
  256/256 scored、106 题正确、精度 41.40625%、0 request failure、130 条截断；
  退出后 8/8 张 GPU 为 0 MiB/0%，无 compute process。
- [x] 独立重算确认 256 行预测、256 个唯一 ID、256 个 checkpoint、106 题正确；
  validation 中 predictions/summary/runner/suite 哈希逐项匹配，服务器 fatal-error 扫描为空。
  结构化证据为 42/42 validation、25/25 manifest，独立 `sha256sum -c` 全部通过。
- [x] 本轮只通过保守的性能候选快速筛选门禁：正确数相对历史 BF16 105/256 为 +1，
  相对历史 OSCAR K=2,048 的 107/256 为 -1；协议指纹不配对且 130/256 截断，
  official validation 明确标记后续完整评测仍需执行，不能宣称最终精度已验证。
- [x] 报告 2.149 与 planning 已由主仓提交
  `d4f3e76e862c7cea3c3c462a565e492e0efe3cc7`通过 GitHub HTTPS 发布。
- [x] 2.149 发布身份已由 planning 提交
  `b0fe18c424eb355acaf101c8a3ecc1d5ce3f7ace`推送，主仓与 source 恢复 clean/upstream。
- [x] 32K 性能前双空闲检查在`19:56:48Z/19:57:53Z`完成，间隔 65 秒；16/16
  设备行均为 0 MiB/0%，两次 compute-process 列表为空。
- [x] 报告 2.150 与 planning 已由主仓提交
  `697f7aaa3504467c5d785f7f8934c8e6311479ab`通过 GitHub HTTPS 发布。
- [x] 2.150 发布身份已由`51f4152219d090225c31cc4e7366d5a975a9ae11`
  推送；随后同一 32K/batch1/output128/TP8 正式三轮与 profiler 全部通过。
- [x] K=1,536 + legacy 的正式中位 TTFT/TPOT/请求吞吐为
  `25682.409651/198.952126 ms/0.019634504 req/s`；相对 K=2,048 为
  `-15.849525%/+0.850878%/+9.077736%`。
- [x] 相对 BF16，K=1,536 + legacy 仍为 TTFT `+104.999655%`、TPOT
  `+11.251016%`、请求吞吐`-30.808156%`，尚未性能收敛。结果已实时写入
  报告 2.151。
- [x] 报告 2.151 与 planning 已由主仓库提交
  `d90e8773d70e8a2b023a6c0f603c3bf0bcffcaad`通过 GitHub HTTPS 发布。
- [x] 2.151 发布身份已由 planning 提交
  `a22c09002aabce7f7747cd11a2b22cd1f1a728ba`推送，两仓恢复 clean/upstream。
- [x] CPU-only 同分析器对比完成：K=1,536 相对 K=2,048 的 prefill wall
  改善 4,792.431197 ms，其中 stage1 改善 4,778.725438 ms，解释
  99.714012%；相对 BF16 剩余 prefill wall 差距 15,704.562953 ms。
- [x] 三组均 8/8 ranks、16/16 chunks、32,768 tokens，trace 身份全部匹配；
  结构化证据 37/37 validation、10/10 manifest 通过，结论已实时写入报告 2.152。
- [x] 报告 2.152 与 planning 已由主仓库提交
  `c292ab190d2ada0b091217828541ee4b721511d0`通过 GitHub HTTPS 发布。
- [x] 2.152 发布身份已由 planning 提交
  `32e8f296eef960aebcf0a1402919cb0861268257`推送，两仓恢复 clean/upstream。
- [x] CPU-only 排序下一候选，同时评估 K=1,024 精度风险与非降 K 的
  stage1/剩余 wall 优化；现有非降 K 方向均已有资源、正确性或实测反证，唯一未实测且
  能线性减少stage1主循环工作的候选为K=1,024 + legacy decode，但线性外推仍不足以
  追平BF16，且算法精度风险高。
- [x] 重读并实时追加报告2.153，冻结K=1,024候选的快速精度淘汰门槛、专项CUDA
  correctness、正式32K性能顺序与“不得写成最终精度”的边界；10,026行、
  2.1–2.153连续，术语、交叉引用与diff门禁通过。
- [x] 报告2.153与三份planning已由主仓提交
  `6fc95f0f921d608efbd25ca1cf09fecc26fac2ae`通过GitHub HTTPS发布。
- [x] 2.153发布身份已由planning提交
  `6da2b0a1ede899fd379f5835292018fef88182bd`推送，主仓与source仓均clean/upstream。
- [x] 将Phase 9当前候选的唯一HF override由K=1,536最小更新为K=1,024，并同步
  三处fail-closed读取/验证与定向测试；固定c349/Python3.12有效红灯为19项中目标1失败，
  最小实现后19/19通过，shell语法和Python编译通过，未使用GPU。
- [x] 重读并实时追加报告2.154，记录TDD、最小diff与两次无效环境调用；报告现为
  10,071行，2.1–2.154连续，术语、引用、文件哈希与diff门禁通过。
- [x] K1024配置、报告2.154与planning已由主仓提交
  `980ac5e0321e16955ab6bcf381398b98fd5e4f0b`通过GitHub HTTPS发布。
- [x] 2.154发布身份已由planning提交
  `0aa5d2fa9ef373b12b414681ba357751a1789007`推送，两仓恢复clean/upstream。
- [x] 正式`preflight-candidate`会注入NVIDIA driver，虽预期不初始化CUDA/不加载模型，
  仍先执行两次间隔至少60秒的8卡空闲门禁并实时更新报告；双采样已于
  `21:11:05Z/21:12:10Z`完成，间隔65秒，16/16设备行均0 MiB/0%、两个compute列表为空。
  报告2.155已追加并通过门禁：10,099行、2.1–2.155连续，边界纠正、日志hash、术语、
  引用与diff均通过，并由主仓提交`f98f473e995de3b0f34986db176382bab8dcc8ba`
  通过GitHub HTTPS发布；发布身份已由`8ada6b3df3363004a5d3b0eb148fc764494b364b`
  推送，两仓clean/upstream。`21:14:12Z`即时复核8/8卡0 MiB/0%、无compute；下一步
  独立run`20260801T2115Z_topk1024_driver_preflight_v1`已自然exit0：69/69静态检查、
  固定import、真实CLI与K1024/legacy/sort合同通过，两个CUDA初始化字段均为false，
  GPU释放8/8为0 MiB/0%、无compute。
- [x] 重读并实时追加报告2.156，封存preflight结果、三个JSON身份、既有RuntimeWarning、
  两次核验失误与无精度/性能结果边界；报告现为10,147行、2.1–2.156连续，术语、
  引用、hash与diff门禁通过。
- [x] 报告2.156与planning已由主仓提交
  `bc0200a2eb80402f4ac12f2ac542048b888601fa`通过GitHub HTTPS发布。
- [x] 2.156发布身份已由planning提交
  `191aa35246dece39dce1d6bdf7be9cf1259aace4`推送，两仓恢复clean/upstream。
- [x] 基于已通过K1536的4例专项脚本只改`TOP_K`与scope为1024，落入独立K1024证据目录；
  固定c349/CUDA不可见容器compile/AST合同通过，逐行diff只有两处目标改动，脚本SHA=
  `3b79590315354db9c7244b50793365871105d9d09a427fdd202638058035e893`。
- [x] 重读并实时追加报告2.157，发布脚本身份与CPU合同；报告现为10,188行、
  2.1–2.157连续，术语、引用、脚本hash与diff门禁通过。脚本位于默认忽略的artifacts
  路径，提交时必须只对该精确文件使用`git add -f`，不得扩大忽略范围。
- [x] 专项脚本、报告2.157与planning已由主仓提交
  `209d52208fb4bf894c43b78687eef5411712eaa3`通过GitHub HTTPS发布。
- [x] 2.157发布身份已由planning提交
  `14c35f6c17fd1b27cdb59756897c34c336344124`推送，两仓恢复clean/upstream。
- [x] 专项correctness前双空闲已于`21:23:53Z/21:24:58Z`完成，间隔65秒，16/16
  设备行均0 MiB/0%、两个compute列表为空；报告2.158已追加并通过门禁：10,210行、
  2.1–2.158连续，日志hash、术语、引用与diff通过，并由主仓提交
  `26db1ac222713c81fe5160cd62eebbc1d0e16c55`通过GitHub HTTPS发布。下一步只发布本身份，
  恢复clean后即时复核GPU0并运行CUDA专项。
- [x] 报告2.156发布后，按冻结顺序执行K=1,024专项CUDA correctness与快速精度筛选；
  专项run`20260801T2127Z_topk1024_legacy_cuda_correctness_v1`已自然exit0，4/4通过；
  8K/32K×random/10LSBits均1024唯一索引、set/value match、max abs=0。下一步封存证据
  并先更新报告；报告2.159现为10,262行、2.1–2.159连续，四个原始文件已归档，术语、
  引用、hash与diff门禁通过，并由主仓提交`0326eccc5bfb25426e53aaf46324709120e5dbba`
  通过GitHub HTTPS发布。下一步只发布本身份；仅在后续精度门槛通过后才申请正式32K性能。
- [x] 2.159发布身份已由planning提交`ebfd89d482aa1e543dfb5a650eb734fc48de0985`
  推送，两仓恢复clean/upstream；K1024的256题筛选前双空闲已于
  `21:31:07Z/21:32:13Z`完成，间隔66秒，16/16设备行0 MiB/0%、两个compute列表为空。
  报告2.160已追加并通过门禁：10,291行、2.1–2.160连续，协议/阈值、idle hash、
  术语、引用与diff通过，并由主仓提交`7433abb9e3f84cbf1efe72bd5b44343b6f83611c`
  通过GitHub HTTPS发布。下一步只发布本身份；恢复clean后才启动长实验，每10分钟打印累计精度。
- [x] 2.160发布身份已由planning提交`c215df4245b6e51a32c2dc5ea28a182e6f43d5ad`
  推送，两仓clean/upstream。有效run
  `20260801T2134Z_candidate_topk1024_legacy_fast256_c16_v1`已启动：入口preflight、额外
  双空闲、141/141权重加载和服务ready通过；runner于`21:41:10Z`开始。`21:43:05Z`
  为4/256、3正确、75%、0失败、0截断；样本过少，不作门槛判断，继续每10分钟打印。
- [x] K=1,024快速筛选自然完成：256/256 scored、108正确、42.1875%、0请求失败、
  122截断；256个连续checkpoint、唯一ID与规范化prompt hash通过，server无fatal/OOM，
  正确数和截断数均通过冻结门槛。结构化证据50/50、manifest29/29通过，报告2.161已
  实时追加；发布报告与证据前不启动性能实验。
- [x] 报告2.161、结构化精度证据与planning已由主仓提交
  `199a8d0b8a2359d6e7666899ff699391984af914`通过GitHub HTTPS发布；source仍固定
  clean/upstream c349。下一步只发布本身份，恢复主仓clean/upstream后开始性能前门禁。
- [x] K=1,024性能前双空闲已于`02:00:56Z/02:02:19Z`完成，间隔83秒；16/16设备行
  均0 MiB/0%、两个compute列表为空。报告2.162已实时追加；下一步只发布门禁。
- [x] 报告2.162与planning已由主仓提交
  `c4443ee07cc190c1867a7d681468fa5a48c4ff5f`通过GitHub HTTPS发布；下一步只发布本身份。
- [x] K=1,024同口径32K/batch1/output128/TP8正式三轮与profiler自然exit0；三轮
  9/9成功，中位TTFT/TPOT/请求吞吐为`21032.014034 ms`/`197.718813 ms`/
  `0.021648199 req/s`。profiler 8/8 trace/table、4/4 validation与独立18/18核验通过，
  报告2.163已实时追加；发布前不进入下一优化阶段。
- [x] 报告2.163与planning已由主仓提交
  `793db12e64c3e193eec52c8e6a203d397f25350d`通过GitHub HTTPS发布；下一步只发布本身份。
- [x] 恢复clean/upstream后以固定Python 3.12.13/ijson 3.4.0.post0分析器完成
  K1024/K1536/K2048/BF16的CPU-only同口径trace对比；四组均8/8 ranks、每rank
  16 chunks、32,768 tokens且冻结trace身份匹配。K1024相对K1536的prefill wall
  改善4,782.593767 ms，其中stage1改善4,851.420971 ms、解释101.439119%；相对BF16
  仍多10,921.969186 ms，其中stage1主项解释62.562799%，非主attention残差仍多
  4,088.879552 ms。结构化证据51/51、manifest13/13通过，报告2.164已实时追加。
- [x] 报告2.164发布前门禁通过：10,547行、620,610 bytes、SHA256
  `e071acf507f467a761c10debd451705b87d4ba662065ac9dbf0e6d8134446d5d`；2.1–2.164
  连续，2.152/2.153/2.163交叉引用、术语、结构化证据身份与diff检查均通过。
- [x] 报告2.164与planning已由主仓提交
  `ec73c8401aa58b69031b3f11c3f17d03620144d5`通过GitHub HTTPS发布。
- [x] 2.164发布身份已由planning提交`8ace406`推送，两仓恢复clean/upstream。
- [x] CPU-only残差可比性审计完成：历史BF16 runtime source为`fd3e0b3`，当前K1024
  为`c349e32`；模型和主要服务参数一致，但8 ranks×15 steady chunks中非attention的
  主MoE Marlin calls中位数为106 vs 148，不能把约4.089秒残差直接作OSCAR因果归因。
  结构化审计39/39、manifest17/17通过，报告2.165已实时追加。
- [x] 报告2.165发布前门禁通过：10,611行、624,885 bytes、SHA256
  `21bea6c8adb57052a9c505a80cf1168cbcb5c8b904b0603a6687d85e1a56625d`；2.1–2.165
  连续，2.164交叉引用、术语、审计身份、39/39 validation、17/17 manifest与diff
  检查均通过。
- [x] 报告2.165与planning已由主仓提交
  `9f7c78e08480640606ecb9471c9c5efb876ef5e7`通过GitHub HTTPS发布。
- [x] 2.165发布身份已由planning提交`6a3c864`推送，两仓恢复clean/upstream。
- [x] 当前c349 BF16 preflight前双空闲于`03:18:19Z/03:19:28Z`完成，间隔69秒；
  16/16设备行均0 MiB/0%、两个compute列表为空，日志SHA256=`4ebbf7b5…ac5a`。
  报告2.166已实时追加，尚未启动preflight。
- [x] 报告2.166发布前门禁通过：10,632行、626,352 bytes、SHA256
  `ed7fe6b699c9b50028a6fa18ded572dbfee1d7951439db1843ad0a12bd504050`；2.1–2.166
  连续，2.165引用、术语、idle日志身份与diff检查均通过。
- [x] 报告2.166与planning已由主仓提交
  `6dd9d6a0a793c426108d868a4cab1bea20b54f10`通过GitHub HTTPS发布。
- [x] 2.166发布身份已由`f37b56e10b1d906b266d02f6c385654cbd7620d4`推送，两仓
  clean/upstream；即时复核后`preflight-baseline`自然exit0，77/77静态检查、固定
  import与真实CLI解析通过，KV=`auto`且CUDA未初始化。
- [x] 独立核验确认baseline当前工作树与K1024候选overlay的2,171/2,171个受跟踪
  `vllm/`文件字节一致，原生扩展解析到同一固定基座；12/12身份检查通过。报告2.167
  已实时追加，尚未启动正式BF16轮次。
- [x] 报告2.167发布前门禁通过：10,687行、629,992 bytes、SHA256
  `cac4263793b693d22140cdc7f630eb34ba0f2da100ab3916ad03242bf43f641a`；2.1–2.167
  连续，2.165/2.166引用、术语、证据hash、12/12核验与diff检查均通过。
- [x] 报告2.167与planning已由主仓提交
  `f9360e9f09e58d73b8b4156ab3e7fa581f6dace4`通过GitHub HTTPS发布。
- [x] 2.167发布身份已由`c3a824ae68b291f50931aeee28aabf876ce1f555`推送，两仓
  clean/upstream；正式轮次前双空闲采样`03:32:51Z/03:33:52Z`间隔61秒，16/16
  设备行0 MiB/0%、两个compute列表为空，报告2.168已实时追加。
- [x] 报告2.168发布前门禁通过：10,709行、631,512 bytes、SHA256
  `a7eb9af75488364712138063e4db5111a24b31a19ea303b97832b6198bdfd6c4`；2.1–2.168
  连续，2.167引用、术语、idle日志hash与diff检查均通过。
- [x] 报告2.168与planning已由主仓提交
  `95af8ba40b53ac460514871f713df94a4666e626`通过GitHub HTTPS发布。
- [x] 2.168发布身份已由`11adb2f4cdc0e965dab1b22de84327c3b17d8f37`推送，两仓
  clean/upstream；同源BF16正式32K/batch1三轮+profiler自然exit0，三轮9/9成功，
  中位TTFT/TPOT/吞吐为12515.105379 ms/153.739745 ms/0.031210711 req/s。
- [x] profiler 8/8 trace/table、17/17路径hash及独立22/22核验通过；相对当前同源
  BF16，K1024为TTFT+68.053032%、TPOT+28.606180%、吞吐-30.638561%。报告2.169已
  实时追加，尚未开始CPU-only trace归因。
- [x] 报告2.169发布前门禁通过：10,791行、636,391 bytes、SHA256
  `ab9dfcfe4d1b8b9f1a36790088ba9680fadbc1c05e8fe559f75a5831973a9fae`；2.1–2.169
  连续，2.163/2.168引用、术语、22/22核验、17/17 profile hash与diff检查均通过。
- [x] 报告2.169与planning已由主仓提交
  `279f02e9a49bc4423dc5892866093f3d1c785760`通过GitHub HTTPS发布。
- [ ] 发布2.169身份并恢复clean/upstream；随后用同一分析器归因当前BF16 vs K1024
  trace，先更新报告再选择production优化候选。

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
`b87a401d…/7df314f2…`，PAX 确定性与身份门禁通过。两个独立目录的 OCI
构建和递归验收均已通过，image/config、manifest、candidate layer 和 index
逐字节一致；共同 image/config 为 `eef27939…6eab`。OCI 阶段记录已发布，
v1 已导入 daemon 并通过 image ID、33 层、diff-ID 与 8 项 labels 审计。
导入阶段记录已发布，driver-injected runtime import 也已通过且
`cuda_initialized=false`。runtime 记录已发布；Stage 9 控制 Dockerfile
默认 base 已最小切换到 b87 候选，复现入口已发布；CPU-only 控制镜像构建、
34/33 层继承审计和 runtime 检查均通过。正式 overlay/配置已迁移，
Phase 7/9 工具测试和 64/64 递归 verifier 均通过。静态迁移记录已发布，
driver-injected preflight 也已通过，preflight 结果已由主仓库
`085f0b1ecbfed37198cff0d98517983032438a25` 发布。新的正式 32K/batch1
轮次 `20260731T0805Z_stage9_candidate_b87a401da_32k_b1_v1` 已完整
退出码 0：三轮均为 3/3 completed、0 failed，`mean` 中位数为 TTFT
`41618.560 ms`、TPOT `201.347 ms`、吞吐 `0.014885 req/s`。相对
4-warps 正式结果为 `-11.72%/+0.95%/+7.90%`，相对 BF16 为
`+232.20%/+12.59%/-47.55%`；TPOT 仍在 20% 门限内，TTFT 仍超限。
Profiler 8+8+1 证据通过，table 中 stage1 的 8-rank 中位数约
`29014 ms`，相对 4-warps 的 `34398.099 ms` 下降约 `15.65%`。正式结果
记录已由 `271e029` 发布；随后完成 CPU-only 多 chunk 归因，8-rank prefill
wall/kernel 中位数为 `41516.570/40627.542 ms`，stage1 精确为
`29014.135 ms`。stage1 的下降解释 4→8 warps wall 改善的 `99.38%`，
且其相对 BF16 的超额仍解释当前 prefill wall 差距的 `81.55%`。归因记录已
由 `29c1abf` 发布。只读源码检查确认固定 TP=8 负载每 rank 为 8 heads，
当前却使用 `block_h=16`；CPU-only SM80 离线轮次
`20260731T0914Z_prefill_block_shape_offline_v3` 证明把 `block_h` 精确降为
8 时 shared memory 从 `135168` 降至 `109568 bytes`（`-18.94%`），而
`block_t=32` 需要 `219136 bytes`、超过苹果800上限 `52224 bytes`。该结果
尚无 GPU 精度/性能含义。8-head block 最小源码候选已由
`a2fe0205577b7f4707e9d31213cb5a80eda1f7d4` 落地并推送，生产逻辑只新增
`num_heads<=8` bucket；TDD 红灯为 2 failed/3 passed，正式 CPU-only
参数与 interpreter 为 6/6 passed，全部适用提交 hooks 通过。下一步先发布
主仓库 submodule、2.23 与 planning。主仓库 `f5d51d0` 发布后，同一
2,048×2,048 冻结协议的单卡筛选已通过：grouped split1 从旧 8-warps 的
`24.090 ms` 降至 `19.077 ms`（`-20.81%`），allclose 诊断值保持不变，
runtime shared/registers/stack 为 `109568 bytes/255/0`。下一步先发布
2.24。完整 cold-cache CUDA 随后为 125/125 passed、0 skipped/failed、
79.04 秒，380 个 cache 文件；容器删除后 8 卡全空闲。Phase 6 输入已
最小切换到 `a2fe0205…/b73806b6…`，固定 Python 3.12.13 的 PAX
确定性回归 1/1 passed，静态身份门禁通过。下一步先发布 2.26，再执行
双目录候选 OCI 构建与递归验收。两轮现均为 passed，index/config/
manifest/candidate layer 逐字节一致；下一步先实时更新并发布 2.26，
v1 随后已导入 daemon，image ID、33 层、最后 diff-ID、tag 和 8 项 labels
审计通过。driver-injected runtime import 也已一次通过且
`cuda_initialized=false`；运行时版本、候选 vLLM Python/`_C`、78 层
rotation、三项 artifact hash 与 `reasoning_effort=max` 均匹配，容器退出后
8 卡空闲。2.26 已由主仓库 `062c910` 发布。Stage 9 控制 Dockerfile 已
完成唯一一行默认 base 切换，新 SHA256 为 `a9b9e22b…e97c`；2.27 与入口已
由 `7126356` 发布。CPU-only 控制镜像构建和身份审计均已通过，新 image ID
为 `0e13b724…d50d5`，34/33 层继承与固定 runtime 一致且
`cuda_initialized=false`。下一步先实时更新并发布 2.27，再派生正式 overlay、
迁移配置、运行工具测试和 64/64 verifier；当前 overlay 已通过
4,749 普通文件、6 个 native symlink 和递归内容哈希门禁，Phase 1/5/7/9
配置及 9 个 wrapper 已迁移并通过静态语法/身份清零检查。下一步运行
Phase 7/9 工具测试和 64/64 verifier；当前两组工具测试已为
20/20、21/21 passed，递归 verifier 为 64/64 passed。下一步完整重读并
实时更新 2.28，静态链路已由 `ea88b55` 发布；driver-injected preflight
随后已通过 64/64、固定环境与服务参数门禁，两处
`cuda_initialized=false`。2.29 已由 `c7cd7ed` 发布；新的正式
32K/batch1 已为 passed，三轮中位数 TTFT/TPOT/吞吐为
`36245.415 ms/199.205 ms/0.016248 req/s`。2.30 已由 `a358e15` 发布；
冻结 trace 的 CPU-only 多 chunk 归因随后通过：prefill wall/kernel/
generation 中位数为 `36257.407/35316.438/269.448 ms`，stage1 为
`23688.690 ms`、占 wall `65.33%`，其下降解释相对 b87 wall 改善的
`101.26%`。2.31 已由 `bd17f51` 发布；随后的 CPU-only tile/warps 矩阵
已把 h1/h2/h4、t8/t32 和 4-warps 全部按编译资源淘汰，记录于 2.32。
2.32 已由主仓库 `68a5127baf5ea228c8f45c6bc02d8308624dd9f3`
发布。当前只围绕 grouped prefill stage1 筛选能够减少实际无效工作的最小
算法候选：原生 prefill top-k 源码已证明短行输出为有效前缀加 `-1` 尾部，
且 `rowLen` 已与正式 chunk metadata/query position 逐 query 对齐。最小
runtime causal-loop 候选已经完成 TDD、7/7 CPU/interpreter、Ruff/compile
和 CPU-only SM80 门禁；h8/t16/w8 shared 仍为 `109568 B`，资源为
255 registers/32-byte stack。源码已由 `fd281f5f9` 发布，2.33 与 submodule
第一部分已由主仓库 `26ebefd` 发布。同一 2,048×2,048 冻结协议的单卡门禁
随后通过：split1 CUDA 中位数为 `12.858368 ms`，相对 a2fe 的
`19.077120 ms` 降低 `32.597960%`，allclose 诊断值保持不变，实际资源为
247 registers/0-byte stack。结果已实时补入 2.33；下一步发布报告与 planning，
再重新执行两次至少间隔 60 秒的 8 卡空闲检查，固定 GPU 0 运行独立 cold
Triton cache 的完整 CUDA 回归。当前回归已通过：126 passed、0 failed，
86.77 秒，380 个 cache 文件；全文重读后已新增 2.34 实时记录。下一步先
发布报告与 planning，再最小迁移 Phase 6 候选 OCI 输入。当前输入已切换到
`fd281f5f9`/tree `86185b21`，Dockerfile SHA256 为 `2c97b4ef…4b87`，
确定性 PAX 回归 1/1 通过；下一步全文重读并实时新增 2.35，再发布配置后执行
CPU-only 双目录 OCI 构建。配置已由 `8b414a8` 发布，两份构建和递归验收现均
通过，四项不可变 OCI 内容逐字节一致；2.35 已在全文重读后实时补入结果并
通过章节、术语、结构化 JSON、哈希和逐字节门禁，记录由主仓库 `e1078ec`
发布。CPU-only v1 daemon 导入现已退出 0，image ID、33 层、最后 diff-ID、
tag 和 8 项 labels 的身份审计均通过。2.35 已在全文重读后实时补入 daemon
import、身份审计和两轮 fail-closed 边界，并通过章节、术语、JSON、daemon
live identity、哈希和 diff 门禁，记录由主仓库 `152a26c` 发布。下一步重新
执行两次至少间隔 60 秒的 8 卡空闲检查，固定 GPU 0 运行 driver-injected
runtime import。有效轮次现已退出 0、JSON 与冻结协议逐字节一致且
`cuda_initialized=false`。2.35 已在全文重读后实时补入有效轮次、无效空输入
边界和全部证据哈希，并通过章节、术语、冻结 JSON 逐字节、哈希与 diff 门禁。
该阶段记录已由主仓库 `4af2aca` 发布。下一步最小切换 Stage 9 控制镜像默认
base。当前 Dockerfile 已完成单行切换，SHA256 为 `93111035…8bbb`；2.35 已在
全文复读后实时补入入口和 daemon base 身份，并通过章节、术语、身份与 diff
门禁，入口已由 `efba906` 发布。CPU-only 控制镜像现已构建并通过正式身份与
runtime 审计：image ID `9be0cbb7…321b`、34/33 层继承、labels/entrypoint、
固定包和 `cuda_initialized=false` 均匹配。2.35 已在全文重读后实时补入
构建、有效审计和额外 `Cmd` 断言失败边界，并通过章节、术语、落地 artifact、
镜像身份、哈希和 diff 门禁。下一步只发布报告与 planning；结果发布前不执行
正式 overlay/配置迁移、driver-injected preflight 或 32K/batch1。控制镜像
记录现已由 `a64dbc0` 发布；下一步派生正式 overlay，并按 Phase 1→5→7→9
迁移配置、运行工具测试与 64/64 verifier，结果发布前不进入 GPU preflight。
overlay 现已通过 4,749 个普通文件与 6 个 native symlink 门禁，
Phase 1/5/7/9 配置与 9 个 wrapper 已迁移；工具测试为
20/20 与 21/21 passed，正式挂载命名空间中的递归 verifier 为
64/64 passed。2.36 已在全文复读后实时写入并通过章节、交叉引用、
术语、配置/证据哈希、shell 语法、旧身份清零和 diff 门禁。本阶段已由
主仓库提交 `df51df6` 发布；下一步先发布本条 planning 状态，确保主/源码
仓库 clean/published，再完成 driver-injected preflight 前两次至少间隔
60 秒的 8/8 GPU 空闲检查。planning 发布已由 `f243c26` 完成；外层
GPU 空闲检查为 `13:51:51Z/13:53:00Z`、间隔 69 秒，两次均为 8/8 张卡
0 MiB、0% 且无 compute process。下一步发布本条状态并在启动前复查
GPU，然后运行新 run ID 的 driver-injected candidate preflight。当前
preflight 已退出 0：静态 64/64、固定环境和服务参数全部通过，后两者
均为 `cuda_initialized=false`；容器退出后 8 张卡全空闲。2.37 已在全文
复读后实时写入，并通过章节、交叉引用、术语、7 份证据哈希、
64/64、两处 CUDA 未初始化和 diff 门禁。下一步只发布本阶段记录；
发布完成前不启动新 32K/batch1。当前 2.37 已由主仓库提交
`3d51cc1` 发布；下一步发布本条 planning 状态并保持两仓
clean/published，然后为正式 causal-loop 32K/batch1 新轮次重做两次至少
间隔 60 秒的 8/8 GPU 空闲检查。外层双检现已于
`14:01:00Z/14:02:08Z` 完成，间隔 68 秒，两次都是 8/8 张卡 0 MiB、0%
且无 compute process。下一步发布本条 planning 并在启动前即时复查，
然后用新 run ID 运行同一 32,768 输入/128 输出/batch1 正式单格。
当前正式轮次
`20260731T1403Z_stage9_candidate_fd281f5f9_32k_b1_v1` 已完整
退出码 0：三轮均为 3/3 completed、0 failed，`mean` 中位数为
TTFT `35683.893 ms`、TPOT `197.826 ms`、吞吐 `0.016445 req/s`。
相对 a2fe 为 `-1.55%/-0.69%/+1.22%`，相对 BF16 为
`+184.83%/+10.62%/-42.05%`；TPOT 仍在 20% 门限内，TTFT 未过门限。
Profiler 为 passed，8+8+1 证据齐全；stage1 table 的 8-rank 中位数为
`23134.5 ms`。小型证据 40 份已复制并逐文件验哈希。2.38 与 planning 已由
主仓库提交 `7001b05` 发布；下一步先发布本条状态恢复两仓
clean/published，再对本轮冻结 8-rank trace 执行 CPU-only 多 chunk 归因。
发布状态已由 `f33b184` 固化。CPU-only 归因轮次
`20260731T1457Z_causal_loop_32k_prefill_trace_v1` 已通过：prefill
wall/kernel/generation 中位数为
`35731.482/34779.955/266.230 ms`；stage1 为 `23134.871 ms`、
1,248 次，相对 a2fe 降低 `2.34%`，解释 wall 改善的 `105.30%`。
32K 共 16 个 chunk，而当前 causal 尾部裁剪只覆盖首 chunk 的 78 次调用，
即全部 stage1 调用的 `6.25%`；单层 `-32.597960%` 粗略折算为全 32K
约 `-2.04%`，与 trace 实测一致。下一步先校验并发布报告 2.39；发布前
不修改源码或启动新 GPU 实验。之后只筛选能够减少全部 16 个 chunk
有效 top-k 计算/访存的最小 stage1 候选。
2.39 发布前门禁现已通过：报告为 2,716 行、SHA256
`ba82eff0aec78bfac64ce2174b19452517471d7328eba8ce69e4e896f1a4cc6c`；
1.1–1.5、2.1–2.39 连续，2.39 的实际章节引用有效，`三池=0`，正文
大写 `A800` 仅在第 5 行允许的历史链接。wall/kernel/generation、stage1、
解释比例、5 份证据/343,360 bytes 与全部主要 SHA256 均从冻结文件重算
一致，`git diff --check` 通过。下一步只提交推送报告与 planning；发布前
不进入下一候选。
2.39 与 trace 归因 planning 已由主仓库提交 `792a118` 推送，远端分支
已快进。下一步发布本条状态恢复 clean/published，再只读检查 stage1 的
全 16-chunk 有效计算/访存路径并提出最小候选；提出候选前不分配 GPU。
只读检查确认 stage1 每个 16-token tile 都无条件执行 BF16 score/value 两个
dot，即使该 tile 的 selected tokens 全部属于 INT2 history；prefix/recent 的
masked load 会归零，但 tensor-core dot 仍执行。后 15 个 32K chunk 中
prefix+recent 总容量最多 320，而 top-k 为 2,048，因此这是覆盖全部 chunk 的
最小可控冗余。下一候选只在 grouped prefill stage1 增加 tile 级
`has_bf16` runtime gate：无 BF16 token 时跳过这两个确定为零的 dot，同时仍
对 BF16 accumulator 应用在线 softmax 的 `previous_scale`。不改变 selected
顺序、top-k、C++ indexer、history/RoPE 计算、精度模式、launch 几何、decode
或三段式 cache。先做 source-invariant TDD 红灯，再实现并运行 CPU/interpreter、
Ruff/compile 和 CPU-only SM80 资源门禁；这些通过并实时发布报告前不分配 GPU。
定向红灯已按预期为 1 failed；生产实现现只新增 `is_bf16/has_bf16` 和两处
runtime gate。修正一次通用 patch 上下文误命中 decode 后，定向绿灯为
1 passed，完整 `test_triton_decode.py` CPU/interpreter 适用范围为
8 passed、19 skipped、3 warnings、16.15 秒。下一步执行 Ruff/format、固定
Python compile 与 CPU-only SM80 编译，资源门禁通过前仍不分配 GPU。
Ruff 0.14.0 check/format、固定 Python 3.12.13 compile 和 diff check 已通过。
CPU-only SM80 AST 编译轮次
`20260731T1521Z_bf16_tile_gate_offline_v1` 状态 `passed`：正式
h8/t16/w8 生成 206,640-byte cubin，shared 保持 `109,568 B`，离线
cuobjdump 为 255 registers/0-byte stack；未触发 `166,912 B` 上限。
下一步固化小型证据并运行全部适用提交 hooks；源码提交推送和报告实时发布
完成前不分配 GPU。
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
- **状态：** 完成并已按当前 8-warps 源码重建。候选 tag 为
  `glm52-oscar-a800-phase6-b87a401da-0275043c`，image/config 为
  `sha256:eef27939...6eab`，manifest 为 `sha256:57c03fca...484a`；
  两次独立构建各自验证 4,744 个源码文件、4 个 rotation 文件、runtime
  expectation 和 7 个基础层 native extensions，index/config/manifest/
  candidate layer 四项逐字节一致。v1 已导入 daemon，image ID、33 层、
  candidate diff-ID 和 8 项 labels 审计通过；driver-injected runtime
  import 已通过且 `cuda_initialized=false`。正式控制与运行时链路仍属
  阶段 9 当前活动步骤；新控制镜像已构建并通过 CPU-only 身份/runtime
  门禁，正式配置迁移、工具测试、64/64 递归 verifier 和 driver-injected
  preflight 已通过。

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
  OSCAR 首个完整格已触发性能优化；当前源码 `a2fe0205…` 的 8-head block
  候选已完成单卡 2K、完整苹果800 CUDA 125/125、双 OCI、daemon/runtime、
  正式链路、preflight 和 32K/batch1 端到端验收。三轮中位数 TTFT/TPOT 为
  `36245.415/199.205 ms`，相对上一 OSCAR 为 `-12.91%/-1.06%`，相对
  BF16 仍为 `+189.31%/+11.39%`。CPU-only trace 归因进一步确认 stage1
  占 prefill wall `65.33%`，并解释本轮相对上一候选 wall 改善的
  `101.26%`；causal-loop 候选随后完成同格正式验收，三轮中位数
  TTFT/TPOT 为 `35683.893/197.826 ms`，相对 a2fe 为
  `-1.55%/-0.69%`，相对 BF16 仍为 `+184.83%/+10.62%`。
  stage1 profiler table 中位数为 `23134.5 ms`，相对 a2fe 约降
  `2.34%`。CPU-only 冻结 trace 归因进一步得到 prefill
  wall/kernel/generation 为 `35731.482/34779.955/266.230 ms`，stage1
  为 `23134.871 ms`、相对 a2fe 降 `2.34%`，且解释 wall 改善的
  `105.30%`。后续 BF16 tile gate 候选 ca4a404e9 已完成同格正式验收：
  TTFT/TPOT 为 `32683.066/200.037 ms`，相对 fd281f5f9 为
  `-8.41%/+1.12%`。冻结 trace 的固定 Python 3.12.13/
  ijson 3.4.0.post0 CPU-only 归因进一步确认：prefill wall/kernel/
  generation 为 `32756.592/31755.930/269.372 ms`，stage1 为
  `20128.143 ms`、占 wall `61.45%`；stage1 降低 `3006.728 ms`，
  解释 wall 改善的 `101.07%`。下一步先完整重读并实时发布
  报告 2.53；发布前不修改下一候选源码。之后仅围绕仍解释相对
  BF16 prefill 差距 `73.86%` 的 stage1 选择下一项最小优化。
  TTFT 关闭 20%
  门限后，才运行同一最终提交的完整矩阵和 128K 候选验证。

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
| 2.205源码审计只读查询调用`jq`失败：宿主未安装该命令 | 1 | 行号/源码输出已成功；结构化JSON后续改用项目现有Python运行时读取，不安装依赖、不重复同一失败命令 |
| 前两次记录2.205查询错误的`apply_patch`分别误假设英文标题和错误的Markdown分隔格式，均校验失败且文件未改 | 2 | 用`rg`与`sed`定位真实中文标题及精确分隔格式后完成最小patch |
| 2.205独立复核首轮把目录对象`HERE`沿用脚本文件对象的`parents[4]`，项目根误解析为`/nfs/AE/txc`，`git rev-parse`退出128且未生成独立validation；随后首次manifest也因目标JSON不存在报错 | 1 | 保留失败exit/log/manifest；只把`HERE.parents[4]`改为`HERE.parents[3]`，用新有效文件名重试并要求独立JSON非空、manifest全部通过 |
| inverse-fusion目标测试首次用宿主source `.venv`运行，但该环境没有安装`pytest`，在收集前以`No module named pytest`退出 | 1 | 不计TDD红灯、不安装临时依赖；改用已冻结Stage 9控制镜像、network none且无GPU暴露执行同一测试 |
| inverse-fusion production首次source commit被SPDX hook拒绝：新增测试令历史无header的`test_triton_store.py`进入检查范围，hook自动补header | 1 | 未生成commit/push；保留失败边界，只暂存hook的机械修正后重跑完整pre-commit，不跳过任何门禁 |
| inverse-fusion GPU门禁第二采样为15:11:40Z，与15:10:43Z首轮仅间隔57秒 | 1 | 两轮虽均全idle，但第二轮不计有效；保留原始日志，追加第三轮并要求首轮至有效末轮>=60秒 |
| 微基准idle manifest后手工`wc/sha256sum "$DIR"/*`把子目录本身传给工具，产生`Is a directory`错误 | 1 | 7/7显式manifest与validation均已通过，不受影响；核心大小/hash只按明确文件路径统计，不重复错误glob |
| 微基准结果manifest后手工glob再次把`__pycache__`目录传给wc/sha，打印`Is a directory` | 2 | 11项语义manifest和16/16 validation已通过；改用`find -type f`精确枚举，并把生成pyc显式纳入最终manifest，不再用目录glob |
| 报告2.211首轮门禁从项目根执行`sha256sum -c`，manifest相对路径因工作目录错误而12项找不到 | 1 | manifest内容未变；改为进入证据子目录后复算，并先更正报告中manifest实测大小1010 bytes |
| d0d Phase6 builder首次由宿主Python 3.8运行，`datetime.UTC`不存在而在写OCI前退出 | 1 | 保留失败目录；沿用既有Phase6已验收解释器边界，查明并改用Python>=3.11，不修改builder或原样重跑 |
| d0d Phase6 builder v2固定control container以宿主UID运行，但git将NFS挂载判为dubious ownership | 1 | Python 3.12.13边界正确、OCI仍未写入；保留v2，下一轮在临时容器全局配置两个精确safe.directory后新建v3，不修改宿主Git配置或builder |
| Stage9 d0d合同绿灯命令在只读源码挂载上直接`py_compile`，写默认`__pycache__`时报EROFS | 1 | 目标1/1与完整24/24已先通过；只为compile设置`PYTHONPYCACHEPREFIX=/tmp/pycache`重跑，不改变测试结论或源码 |
| 新control CPU preflight首次按旧路径导入`vllm.attention.ops`，实际production模块位于`vllm.v1.attention.ops` | 1 | 容器exit1且未初始化CUDA；保留失败stdout/exit/尾行，核对source路径后仅修正模块名运行v2 |
| d0d runtime import GPU空闲第二采样距首轮仅27秒 | 1 | 两轮虽均全idle但不满足>=60秒；第二轮改名保留为invalid_27s，补采第三轮并以first到third闭合 |
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
| 报告交叉引用检查器把普通小数或版本号识别成章节号 | 2 | 先后误命中 `7.137 秒` 和环境版本 `2.34.1/2.35`；报告章节本身连续。只扫描“见/记录在/按/保持/相对”等章节引用语境后重跑 |
| preflight 证据探查把 `xargs` 与读取 stdin 的 heredoc Python 混用 | 1 | 文件路径已列出，但后续同一 shell 输出被 stdin 组合截断；未修改证据。改用 Python `Path.glob` 直接读取三个 JSON，得到完整状态、检查数和 SHA256 |
| grouped trace 分析又直接调用宿主 Python | 1 | 宿主缺 `ijson` 的限制已在旧轮次记录；本次在 import 阶段退出，未读取 trace 或生成结果。后续直接复用固定控制容器、`uv` 和 `ijson==3.4.0.post0`，不再探测宿主解释器 |
| BF16 trace 首轮强制 `uv --offline` 时缓存不能解析固定 ijson | 1 | 在依赖解析阶段退出，未读取 trace、未生成输出目录；保持固定容器和版本不变，改用已验收的清华 PyPI 镜像在线解析，并继续使用任务专用 uv cache |
| demotion 归因写入 planning 文件的首个 patch 上下文不匹配 | 1 | patch 原子失败、文件未被修改；重新读取文件末尾并按实际最新段落追加，不复用过期上下文 |
| 恢复后的源码 CPU test venv 绝对解释器链接再次失效 | 1 | `/dev/shm/oscar-glm-stage9-opt-test-venv/bin/python` 指向当前宿主不存在的 `/usr/bin/python3.12`，测试未启动；不原地修补旧 venv，改用固定控制容器挂载当前源码和既有 native rootfs 执行 TDD |
| 查找 Phase 9 配置时猜测了不存在的 `performance_config.json` | 1 | 只读 Python 在打开文件时退出，未修改状态；TDD 只需已确认的控制镜像和源码/native 挂载，不再依赖该猜测路径，后续配置查询先用 `rg --files` |
| 固定控制镜像未预装 pytest | 1 | 当前源码/native 挂载均成功，但解释器在测试 collection 前报告 `No module named pytest`；下一轮在同一一次性容器中用 uv、清华镜像安装固定 pytest/tblib，再执行红灯用例 |
| contiguous inverse 源码 TDD 首次直接探测正式候选 venv | 1 | `/opt/fp8_speed_up_v4_venv/bin/python` 可运行生产依赖但没有 pytest，测试在 collection 前未启动；两个猜测的旧 `/dev/shm` venv 又分别缺 pytest/解释器链接失效。保持候选环境不变，复用已验收的只读 `pytest==8.3.5` target 注入固定镜像 Python，不把环境失败算作代码红灯 |
| contiguous inverse CPU 回归未声明空 CUDA 可见集 | 1 | 新增契约与 runtime 测试已通过，但两个既有 source-invariant 用例看到 `_mixed_sparse_prefill_stage1` 为无 `.fn` 的普通 function；独立进程复现并证明不是测试顺序污染。`HAS_TRITON` 因无 active driver fail-closed 到 placeholder；下一轮按代码内置的分布式初始化协议显式 `CUDA_VISIBLE_DEVICES=""`，保持无 GPU，再运行相同 decode 文件 |
| Triton function 类型探针 shell 引号错误 | 1 | Python `-c` 的字符串引号被多余反斜线破坏，解释器在 import 前报 SyntaxError；未读取/修改源码。停止拼接复杂 `-c`，改用容器 stdin heredoc并显式 `docker run -i`，同时先读取 `HAS_TRITON` 判定逻辑 |
| 空 CUDA 可见集 decode 全文件回归未打印终态 | 1 | 前置探针已证明真实 Triton JITFunction，pytest打印7个通过点后未给summary/exit证据，容器已消失且主机内存充足；不能记作通过。下一轮先排除独立 interpreter subprocess验证其余collection/source-invariant，再单独运行interpreter smoke并显式打印Docker退出码，避免重复不透明组合 |
| contiguous inverse 首轮完整 pre-commit 命中既有门禁与新动态 buffer 类型 | 1 | Ruff/format/typos等通过；mypy列出8个既有动态属性错误和1个本轮新增inverse属性错误，SPDX hook为触及的旧测试补头，attention docs hook改写文档，torch.cuda hook命中backend既有第380行。先审计diff/blame；本轮inverse属性补显式类型，保留必要SPDX，机械还原与本候选无关的docs，再按历史协议只跳过已证明既有项 |
| contiguous inverse 首次 commit 未传已审计 hook 跳过列表 | 1 | commit hook再次命中既有`torch.cuda`并重写无关attention docs，提交被拒、源码六文件仍在index。按已验证策略再次机械还原docs，并在commit命令显式设置`SKIP=check-torch-cuda-call,attention-backend-docs`；不使用`--no-verify`，其余hooks仍全部执行 |
| 2.73 草稿手工扩写旧主仓库提交哈希错误 | 1 | 把已知缩写`971f0c4`错误补成未经验证的全值；在发布前立即用`git rev-parse 971f0c4`实算为`971f0c4f3a57bbd73e89f7cf7dff63c928e9aff2`并修正。错值未提交或推送，后续所有完整哈希均从Git/文件实算 |
| contiguous inverse 构建链初查猜错Stage 9 Dockerfile名 | 1 | Phase 6输入与Dockerfile已成功读取，随后不存在的`docker/Dockerfile.stage9-control`令组合命令停止，identity检索未执行；未修改文件。下一步先用`rg --files docker configs`定位真实入口，不再手写文件名 |
| contiguous inverse OCI v1读取verification report过早 | 1 | build report先完整生成；组合工具返回后立即读取时report尚未可见而报FileNotFoundError。随后目录审计发现原verifier继续完成，文件mtime晚于build 38秒，status=passed、4,744源码文件/7原生扩展/4 artifact均通过。没有重跑或覆盖；后续长组合必须轮询目标文件/进程终态后再读 |
| daemon导入前再次误用`docker ps` HostConfig模板 | 1 | OCI ref与目标tag不存在已先确认；随后Docker formatter不支持`.HostConfig.DeviceRequests`而退出，尚未启动工具容器或导入。改为`docker ps -q`后逐个`docker inspect --format '{{json .HostConfig.DeviceRequests}}'`，不重复错误模板 |
| Stage 9控制镜像首次build误用项目根context | 1 | 进程运行10分钟仍无image，FD诊断证明正在遍历NFS artifacts/phase0 rootfs；正式wrapper实际用`${PROJECT_ROOT}/docker`。已向本轮docker build PID发送SIGTERM，进程退出且目标tag仍不存在。下一轮使用docker目录context并显式传已审计base build-arg，不重复全项目context |
| decode 优化 pre-commit 首轮发现两个既有门禁漂移 | 1 | Ruff/format/typos/mypy/forbidden imports 均通过；SPDX hook 为两个本次触及的旧文件补头。`torch.cuda` 报错和 attention backend 文档改写需先用 diff/blame 判断是否属于本次改动，只保留必要修复并对已证实旧项精确 skip |
| runtime import 命令遗漏 Docker stdin 透传 | 2 | metadata/scratch 候选首次发生；causal-loop 恢复轮次又误复用缺少 `-i` 的 heredoc 命令。两轮容器内 `python -` 都从空 stdin 正常退出，生成空 JSON/log，不能记为通过；CUDA 未初始化。causal-loop 空文件单独保留，后续命令固定使用 `docker run -i`，并在接受退出码前强制断言 JSON 非空、`status=passed` |
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
| 双 OCI 逐字节比对后的结构化摘要命令假设宿主存在 `jq` | 2 | 首轮已确认宿主不含 `jq`；会话恢复后的证据复核又误复用了该失败方式，两次都只影响只读摘要、没有修改产物。后续固定改用已有容器/固定环境 Python 标准库解析 JSON，不安装依赖、不再调用宿主 `jq`，且不影响已通过的四项 `cmp` 与复现性结论 |
| 会话恢复后把容器内固定 Python 路径误当成宿主路径 | 1 | `/opt/fp8_speed_up_v4_venv/bin/python` 在宿主不存在，解释器启动前即退出、未读取或修改产物。改用已验收控制镜像的 `/usr/bin/python3.12`，以 CPU-only 只读挂载解析同一 JSON；不再在宿主猜测容器路径 |
| 更新 OCI 阶段状态的首次多文件 patch 使用了错误的 Markdown 列表上下文 | 1 | `apply_patch` 原子拒绝，所有目标均未修改；重新读取实际 `- **状态：**` 上下文后拆分为精确 patch |
| 控制 Dockerfile 旧 tag 清零检查用 `rg -c` 读取无匹配输出 | 1 | 文件中实际为 0 个旧 tag，但 `rg` 无匹配时不输出数字且返回 1，空字符串被脚本误判；改用 `if rg ...; then fail; else pass` 的显式语义重跑 |
| head-block 离线资源固化 v1 参数计数断言错误 | 1 | 内核实际为 19 个指针参数加 59 个 constexpr，共 78 个；脚本误断言为 77，在任何编译前 fail-closed 退出。保留日志，以新 run ID 修正计数 |
| head-block 离线资源固化 v2 误把 Triton metadata 当 dataclass | 1 | 第一组 cubin 已编译，但在 `dataclasses.asdict` 序列化处退出，未形成完整三组结果。保留日志；v3 改为读取 Triton 生成的 JSON metadata，三组编译与资源审计均退出码 0 |
| 8-head Phase 6 PAX 回归首次未指定 pytest 环境 | 1 | 控制镜像中的 `uv` 已启动，但在测试 collection 前因 PATH 中没有 `pytest` 退出，未形成测试结果、未生成 OCI；改用镜像内实测的固定 Python/pytest 环境执行同一用例 |
| 双 OCI 报告术语计数在 `pipefail` 下被零匹配提前终止 | 1 | `三池_count=0` 已打印，但 `rg` 的无匹配返回码令后续只读校验未执行；改用显式容忍零匹配的计数方式后继续完整证据校验 |
| 8-head OCI 导入日志写入 root-owned artifact 目录失败 | 1 | 宿主 `tee` 在容器启动后报告 permission denied，令组合 shell 最终返回 1；`skopeo` 仍完整执行到 `Storing signatures`。不重复导入，改以导入前镜像不存在、导入后 daemon image/层/labels 的只读审计固化成功状态，并把证据写入可写的 `/dev/shm` |
| runtime import 双空闲检查目录名预填为错误的未来时间 | 1 | 检查内容和 UTC 时间戳均正确，尚未启动候选容器；检查完成后把证据目录从错误的 `T1017Z` 原子移动为实际首检时间 `T1002Z`，后续只引用修正后的目录 |
| 新 overlay 链接审计使用了宿主 Python 3.8 不支持的 `Path.readlink()` | 1 | 4,749 个普通文件的 extracted/overlay 递归清单已先逐字节一致；链接审计在只读打印阶段退出，overlay 未修改。改用 `os.readlink()` 重跑 6 个链接目标与原生扩展哈希，并补查 GPU 状态 |
| a2fe trace 归因首次沿用 root-owned analysis 父目录 | 1 | `mkdir` 在分析器、容器和 trace 读取前被权限拒绝，没有生成结果或分配 GPU；改用当前用户独立且已验收的 `/dev/shm/oscar-glm-stage9-opt/analysis`，以新 analysis ID 重跑 |
| a2fe 对比脚本误读 1K BF16 旧 schema summary | 1 | a2fe/b87/b247 数据已先打印且不受影响；脚本在读取缺少多 chunk 字段的旧 1K BF16 文件时只读退出。按 planning 记录定位真实 32K BF16 summary `06eecce0…58d` 后重新计算全部对比 |
| 2.31 报告验证环境没有 `jq` | 1 | 命令在读取 JSON 前即报 `jq: command not found`；不安装新依赖，改用宿主只读 Python `json` 解析 |
| 2.31 对比复核猜错 b87 analysis 目录时间戳 | 1 | BF16 SHA256 已成功实算，Python 在打开不存在的 b87 路径时只读退出；使用 `find`/既有 planning 定位真实目录 `20260731T0857Z_8warps_32k_prefill_trace_v1` 后重跑，不修改任何证据 |
| 2.31 术语断言把 Markdown 链接标签和目标中的 `A800` 当成两个违规位置 | 1 | 报告仍只有第 5 行这一处允许的历史文件链接；把断言从 token 次数改为匹配行数和完整链接结构，再继续其余数值门禁 |
| 2.31 复核脚本错误地从 aggregate 读取 generation | 1 | 当前 analyzer schema 只在各 rank trace 内保存 `generation_duration_ms`；脚本在 KeyError 处只读退出。改为复用归因口径，对 8 个 rank 的 generation median 再取中位数 |
| 2.31 组合验证中的 `rg` 没有匹配运行参数 | 1 | 数值、章节和 JSON 门禁已先通过；`rg` 因 run log 只记录逐 rank 结果、不记录命令行而返回 1，使后续哈希命令未执行。分拆命令后已独立完成 run/input 内容、五份 SHA256 和 `git diff --check` 复核 |
| 下一候选源码检索先查错 sparse MLA 文件 | 1 | 精确 stage1 符号在 `triton_sparse_mla_kernel.py` 无匹配，使 `&&` 后的只读 diff 未执行；源码提交统计确认 a2fe 实际修改 `triton_oscar_mla_decode.py`，后续检索转到该文件 |
| a2fe 无 driver 控制容器直接 import 后 stage1 被替换为普通函数 | 1 | vLLM 平台探测发现 0 active driver 后禁用 Triton，`_mixed_sparse_prefill_stage1` 没有 `arg_names`；该轮只读退出。后续复用上一轮可工作的 CPU-only AST 导入方式或直接从冻结 JIT 源构造编译，不注入 GPU |
| 查找 Triton driver 门禁时 `sed` 使用了旧工具路径 | 1 | `rg` 已先定位真实文件为 `vllm/triton_utils/importing.py`，随后 `sed vllm/utils/importing.py` 报不存在；改读真实路径，确认空 `CUDA_VISIBLE_DEVICES` 是允许 0 driver 的离线导入条件 |
| tile 离线矩阵 v1 使用宿主 UID 后镜像 passwd 无该用户 | 1 | PyTorch 在 import 期用 `getpass.getuser()` 构造 cache 路径并触发 `KeyError: getpwuid()`，未进入任何 Triton 编译；保留 v1 script/log/exit，v2 继续使用非 root UID 但显式设置 `USER/LOGNAME`，避免 root-owned 证据且不复用失败命令 |
| 2.32 JSON 验证容器漏传 `-i` | 1 | heredoc 没有进入容器 stdin，Python 以空输入退出 0，不能视为 JSON 门禁；同一组合命令中的 `git diff --check/status/stat` 已独立执行。下一轮加入 `docker run -i` 并要求输出 6 项 passed 标记 |
| 动态 causal-loop 红灯首次直接使用控制镜像正式 venv | 1 | 控制镜像未预装 pytest，collection 前以 `No module named pytest` 退出；不重复该命令，改用镜像内固定 Python 加 `/usr/local/bin/uv --with pytest==8.3.5` 的一次性依赖环境 |
| 探查 source `.venv` 时其 Python symlink 已退化为系统 `/usr/bin/python3` | 1 | 版本探针随即显示 pytest 缺失；该环境不符合 source `AGENTS.md`，立即停止使用，不在其上安装或运行测试，后续只用固定控制容器的正式 Python/uv |
| 动态 causal-loop 红灯的 `uv run --isolated` 隔离了正式运行时依赖 | 1 | pytest 已安装但 `tests/conftest.py` 导入 numpy 时退出；不在隔离 venv 补齐整个 runtime，改用历史已验收的 `uv pip --target` 只安装 pytest/tblib，并把 target 前置到固定正式 Python 的 PYTHONPATH |
| 动态 causal-loop 首轮绿灯仍读取旧 JIT 函数源码 | 1 | bind-mounted 文件已含新代码，固定 Python 单独导入也确认 `effective_topk=True`；pytest 进程仍命中镜像旧 code/source cache。下一轮设置任务专用 `PYTHONPYCACHEPREFIX`，并在 pytest 前打印导入路径/源码断言，避免复用镜像 bytecode cache |
| 记录首轮绿灯失败的 planning patch 使用了过期 progress 上下文 | 1 | `apply_patch` 原子拒绝，task/progress 均未修改；已重读两文件实际末尾并用精确上下文重试 |
| 首次 production patch 的通用上下文把 `effective_topk` 插入 decode stage1 | 1 | 定向绿灯仍失败且 diff 证明 prefill 只有 loop 引用、变量却落在前一个同名 causal 片段；测试有效阻止错误通过。用 `split_len`/`token_offsets` 唯一邻接上下文把变量从 decode 移到 prefill，不改变 decode |
| causal-loop 首轮 `uvx ruff` 使用未映射 UID 的默认工具目录 | 1 | uv 尝试创建 `/.local/share/uv/tools` 并在任何 lint/compile 前权限拒绝；保持只读源码挂载，下一轮显式设置任务专用 `UV_TOOL_DIR`/`UV_CACHE_DIR` 到 `/tmp` 后重跑 |
| causal-loop 第二轮 ruff 在只读源码挂载下创建项目 cache | 1 | ruff 本体已正确安装，但尝试写 `/workspace/vllm/.ruff_cache` 时退出，lint 未完成；不改源码挂载权限，下一轮显式设置 `RUFF_CACHE_DIR=/tmp/...` 后重跑 |
| causal-loop 首轮资源审计假设控制镜像内含 `cuobjdump` | 1 | CPU-only 控制镜像中命令不存在，cubin 未修改；宿主已定位兼容的 `/usr/local/cuda/bin/cuobjdump`，下一轮直接对只读 SM80 cubin 执行资源解析 |
| causal-loop source 首次 push 沿用失效的当前 VS Code Git IPC socket | 1 | commit `fd281f5f9` 已成功形成且 hooks 全过，但 `/run/user/22633/vscode-git-69732924ca.sock` 拒绝连接，远端未更新；已只读确认既定有效 socket `vscode-git-5d76bad75c.sock` 存在，下一轮显式覆盖 `VSCODE_GIT_IPC_HANDLE` 后重推，不重复失效环境 |
| 查找历史单卡目录时 `find /dev/shm` 命中外部 `multipath` 权限拒绝 | 1 | 该无关目录只读报错，不影响已列出的项目目录；后续把搜索范围收窄到 `/dev/shm/oscar-glm-stage9-opt`，不再扫描整个共享内存根目录 |
| causal-loop 单卡结构化复算手写改善百分比常量有舍入错误 | 1 | 当前/旧两份精确中位数断言已通过，脚本在手写 improvement 常量处退出；不改任何实验数据，下一轮从两份 JSON 动态计算并打印，再只对区间/公式作断言，不硬编码派生小数 |
| causal-loop 完整 CUDA 证据目录预填了错误的未来时间 | 1 | 第一份空闲检查内容与 `12:23:08Z` 时间戳均正确，尚未启动容器；随即把目录从错误的 `T1248Z` 原子移动为实际首检时间 `T1223Z`，后续只引用修正后的目录 |
| 阶段相关文件搜索包含不存在的顶层 `tests` 目录 | 2 | Phase 6 检索首次发生；控制镜像协议恢复时又把同一不存在路径传给 `rg`。两次均为只读报错，已有匹配结果和文件均不受影响；后续只对 `rg --files` 实际列出的 `scripts`、`docs`、配置与 Dockerfile 搜索，不再手写顶层 `tests` |
| Phase 6 PAX 定向 pytest 首轮写错 unittest 类名 | 1 | pytest 在 collection 后报告 node 不存在，0 个测试执行且后续 compile 未运行；读取真实类名 `BuildCandidateOciTest` 后重跑为 1 passed，并完成固定 Python compile |
| 双 OCI 结果的 planning 批量 patch 使用了过期 progress 上下文 | 1 | `apply_patch` 原子拒绝，task/findings/progress 均未修改；重新读取三个文件的实际末尾后拆分为精确 patch |
| 恢复后补写双 OCI 证据的 findings patch 再次使用过期措辞 | 1 | `apply_patch` 因预期段落与磁盘实际措辞不符而原子拒绝，findings 未修改；已重读文件末尾，改为只在当前最后一项之后追加本轮只读复核结果 |
| 2.35 结构化术语门禁把唯一允许链接中的 `A800` 次数误断言为 1 | 1 | 链接 label 与 target 在同一第 5 行各含一次该字符串，因此断言只在术语检查处退出；报告和证据均未修改。改为断言所有命中均严格位于允许的第 5 行，继续执行其余 JSON、哈希和 OCI 逐字节门禁 |
| causal-loop daemon import 工具容器首次把阿里 Ubuntu 镜像切为 HTTPS | 1 | 最小 Ubuntu 22.04 镜像尚无 CA 证书，`apt-get update` 因证书链不可用而无法定位 skopeo，退出码 100；命令在 skopeo 安装和 OCI 导入前退出，目标 tag 仍应不存在。保留失败日志，下一轮改用阿里 HTTP 镜像完成同一 CPU-only 工具安装，不重复 HTTPS 失败路径 |
| causal-loop daemon import 第二轮手写错 OCI ref name | 1 | 阿里 HTTP 镜像和 skopeo 1.4.1 安装已成功，但 source ref 被误写为不存在的 `...-causal-loop`，skopeo 在读取 descriptor 时退出 1，尚未复制任何 blob。保留失败日志并复核目标 tag 不存在；第三轮从 `index.json` 已冻结 ref name 原样使用 `...-fd281f5f9-0275043c`，不再手写别名 |
| 控制镜像入口报告全文复读把 800 行一次输出 | 1 | 第 801–1,600 行的工具输出超过预算并在中间截断，不能视为完整读取；报告尚未修改。改为每 400 行读取并在每段后更新 progress，直到 EOF 后再复核修改前 SHA256 |
| causal-loop 控制镜像身份审计额外断言 base/control `Cmd` 相等 | 1 | 34/33 层、前 33 层、labels 与 entrypoint 已先通过，但新增的非冻结 `Cmd` 断言失败，runtime check 因 fail-closed 尚未执行；保留 v1 audit/exit 证据，读取两边实际配置后只按上一 a2fe 有效协议审计 34/33 层继承、labels、entrypoint 和 CPU runtime，不把未经约定的 `Cmd` 加入门禁 |
| causal-loop 静态迁移 compile 猜测不存在的 Phase 9 verifier 文件名 | 1 | 宿主 Python 在 `scripts/phase9/verify_profile_bundle.py` 读取前报告文件不存在；9 个 shell 语法、4 个 JSON、旧身份清零和 diff 门禁已独立完成。后续先用 `rg --files scripts/phase9` 选择实际存在的 Python 入口，再重跑 compile，不再猜测文件名 |
| causal-loop 工具测试 readiness 猜测 recovery tools 内有 `.venv/bin/python` | 1 | `/dev/shm/oscar-glm-recovery-tools` 目录存在，但猜测的解释器路径不存在，只读 `test -x` 在正式 pytest 前退出；先列出实际目录并读取测试所引用的 launcher，再使用实测路径，不安装或改写共享工具 |
| causal-loop 递归 verifier 命令含预清理 `rm -f` | 1 | 执行器在创建容器前按安全规则拒绝整条命令，没有删除文件、没有创建容器、没有运行 verifier；改为 fail-closed 断言三个目标文件均不存在，然后执行不含删除的同一 CPU-only 验证 |
| 递归 verifier 结果 planning 批量 patch 使用了错误文件尾部上下文 | 1 | `apply_patch` 原子拒绝，findings/progress 都未修改；分别读取两个文件实际尾部，再按单文件精确追加，不再用未区分的合并 tail 输出作为上下文 |
| 2.36 交叉引用检查器把性能倍数 `2.89×` 误当章节号 | 1 | 章节序号、术语、JSON/证据、9 个 shell、旧身份清零和 diff 已独立通过；报告未被该检查修改。将引用扫描收窄到“见 2.x”、“2.x 的”等实际语境，不再把小数倍数当章节引用 |
| 正式 32K 双空闲结果 planning patch 使用了未精确匹配的换行上下文 | 1 | `apply_patch` 原子拒绝，task/progress 都未修改；重新读取实际上下文后分别精确追加，不影响已落盘的 `14:01:00Z/14:02:08Z` GPU 检查证据 |
| causal-loop 正式 summary 首次只读摘要又调用宿主 `jq` | 1 | 宿主立即报 `jq: command not found`，未修改证据；改用 Python 标准库读取同一 result/validation，后续不再调用宿主 `jq` |
| 历史 summary 搜索对整个 `artifacts/` 执行无界 `find` | 1 | `/dev/shm` 精确目录已找到 BF16/a2fe/当前 summary；`artifacts` 分支持续扫描且无额外价值，因此主动中止，未修改文件。后续只查已知精确路径 |
| 小型证据复制后验证脚本把期望文件数写为 33 | 1 | 40 个文件已全部复制；脚本在计数断言处退出，尚未进入错误结论。按实际选定集合更正为 40 后，逐文件原件/副本 SHA256 和 40 行 manifest 全部通过 |
| 发布状态 planning 首次批量 patch 上下文不匹配 | 1 | `apply_patch` 原子拒绝，task/progress 均未修改；重新读取精确末尾后拆成小 patch 成功，不影响已推送的 `7001b05` |
| causal-loop trace 首次对比脚本只匹配 OSCAR stage1 符号 | 1 | 新候选与 a2fe 数据已只读打印，读取 BF16 原生 attention 时触发 `StopIteration`；冻结 trace 和结果均未修改。改为同时识别 OSCAR/BF16 符号后重算，全部指标与有效 analysis summary 一致 |
| 2.39 数值复核脚本对 generation schema 与剩余 wall 精确值作了错误假设 | 2 | 首轮把每 rank 的统计字典当数值列表；第二轮把未用于报告的剩余 wall 相对变化硬编码为略有舍入偏差的值。两次均只读退出且不修改报告/证据；改按 `generation_duration_ms.median` 聚合，并由冻结 summary 现场计算后只校验报告使用的 `+0.22%` 舍入值 |
| 下一候选首次只读源码检索沿用缺少 `v1/` 的旧路径 | 1 | `rg/sed` 在打开文件前报告不存在，未修改源码；先用 `rg --files` 定位真实文件为 `vllm/v1/attention/ops/triton_oscar_mla_decode.py`，并在继续审查前完整读取 submodule `AGENTS.md` |
| BF16 tile gate 首次 production patch 的通用 mask 上下文命中 decode | 1 | diff 审计在绿灯前发现 `is_bf16/has_bf16` 被插入 decode，而 gate 位于 prefill；测试尚未重跑、GPU 未分配。立即恢复 decode 原样，并用 prefill 独有的 `[None, :]` 张量布局上下文精确放置变量 |
| 补录 BF16 tile gate 发布状态的批量 planning patch 使用过期上下文 | 1 | `apply_patch` 原子拒绝，task/findings/progress 均未修改；已重读三个文件实际末尾，改为逐文件精确追加，不影响源码提交和已冻结实验产物 |
| 2.40 首轮结构化校验在宿主调用容器内固定 Python 路径 | 1 | `/opt/fp8_speed_up_v4_venv/bin/python` 在宿主不存在，解释器启动前退出；`git diff --check` 已独立通过且文件未修改。改用不注入 GPU 的固定控制容器和同一路径执行只读校验，不使用系统 Python |
| later-chunk 基准首轮工具门禁发现 Ruff format 漂移 | 1 | 22/22 pytest 与 Ruff check 已通过，但 format check 报主脚本需机械格式化，轮次退出 1，后续 compile/CLI 未执行；使用同一 Ruff 0.14.0 只格式化该脚本后完整重跑，不改变算法 |
| later-chunk 32K coverage 首轮 Docker 命令遗漏 stdin 透传 | 1 | `python -` 从空 stdin 正常退出 0，run log 为空，不能记为通过；保留无效 v1，下一轮固定使用 `docker run -i`，并在接受退出码前强制断言日志非空和 `status=passed` |
| 2.41 首轮结构化校验硬编码了不同的证据边界措辞 | 1 | 章节和术语已通过；脚本期待“当前尚无苹果800”，报告实际为同义且更准确的“本节尚无苹果800”，因此只读退出，报告/证据未修改。改为校验实际落盘措辞后重跑全部剩余门禁 |
| 2.42 重读状态首次多文件 patch 使用了不精确的缩写上下文 | 1 | `apply_patch` 原子拒绝，task/findings/progress 均未修改；已用 `rg -n -C` 重读三个实际位置，改为逐文件精确追加 |
| 2.42 初稿把 planning 摘要中的缩写哈希错误扩写为未验证全值 | 1 | 在发布前哈希门禁中立即从 17 份落盘证据和源码 Git 实算，发现并修正 control/candidate result、cubin 与 fd281f5f9 tree 全值；错值未提交或发布 |
| 2.42 纠正初稿的首个 patch 携带了不存在的额外上下文 | 1 | `apply_patch` 原子拒绝，报告未部分修改；改为只替换 5 个经实算的精确值后成功 |
| ca4a404e9 Phase 6 门禁检索又包含不存在的顶层 `tests` 目录 | 1 | `rg` 已在同一只读命令中定位真实用例为 `scripts/phase6/test_build_candidate_oci.py`；没有修改任何文件或启动测试，后续只使用 `rg --files scripts/phase6` 的实际清单 |
| ca4a404e9 Phase 6 首轮组合门禁的 `py_compile` 尝试写只读源码目录 | 1 | 结构化身份检查与 PAX 确定性 pytest 已先通过 1/1；compile 因 `__pycache__` 写入被拒而使整轮非绿，没有修改源码。新 run ID 显式设置 `PYTHONPYCACHEPREFIX=/tmp/pycache` 后完整重跑 |
| ca4a404e9 静态迁移的 Phase 7 工具测试首轮未挂载冻结 evaluator launcher 的 `/dev/shm` 解释器 | 1 | 19/20 通过，唯一失败进程以 127 报容器内固定 Python 路径不存在；宿主路径已确认可执行。保留首轮日志，以相同绝对路径只读挂载 recovery tools 后新日志重跑，不修改测试或实现。 |
| ca4a404e9 Phase 7 工具测试第二轮的 frozen evaluator 恢复用例冷启动超过固定 30 秒 | 1 | 解释器路径已正确解析，失败变为同一子进程 `TimeoutExpired`，其余 19 项通过；不放宽超时、不改测试，保留 retry 日志，在文件/解释器缓存已预热后用新日志重跑完整 20 项。 |
| ca4a404e9 CPU-only 递归门禁首轮误调用完整 `inside-preflight` | 1 | 64/64 静态 verifier 已通过，但 wrapper 随后继续执行需要 driver 的固定环境 import，并因本轮刻意不注入 GPU 而缺 `libcuda.so.1` 退出；组合轮次不计绿色。保留日志，以新输出只运行递归 verifier，driver import 留到发布后的正式 preflight。 |
| ca4a404e9 单独递归 verifier v2 遗漏模型目录只读挂载 | 1 | verifier 在读取模型 `config.json` 前以 `FileNotFoundError` 退出，未生成有效 JSON；overlay/config 未修改。v3 仅补正式协议已有的模型 bind-readonly，保持无 GPU和其余命令不变。 |
| ca4a404e9 preflight 首版汇总脚本手工补全了错误的 `77b5aa5` 完整哈希 | 1 | 正式 preflight 已退出 0，汇总脚本只在 Git identity 断言处失败并留下 0-byte validation log、无 JSON；保留失败汇总和首版 manifest，v2 直接读取 Git 实际完整哈希，不再手工扩写缩写。 |
| contiguous inverse runtime artifact 只读探针误用宿主 `python` | 1 | 宿主默认解释器缺少 `pathlib`，命令在读取 JSON 前退出且未修改证据；改用明确的 `python3` 后成功读取 manifest/runtime expectation，再进入正式容器探针。 |
| contiguous inverse overlay 首轮误判候选层包含 native 文件 | 1 | 4,749 个候选普通文件已完整复制；循环在第一个不存在的 native 目标断言处退出，未删除任何文件。复核确认候选层按设计不含 6 个 lower native 文件，随后仅为这 6 个精确路径创建指向已验收 base 的 symlink。 |
| contiguous inverse 静态输入首轮手工补全两个缩写哈希错误 | 1 | 其余10项检查通过；build/verification 两项 hash 精确失败，shell/测试尚未启动。直接从落盘文件重新实算为 `a2e1c9d1731f…bb54`/`285ed9978bd3…185a` 并修正配置，保留失败JSON，不把首轮记作通过。 |
| contiguous inverse 首轮递归 verifier 未覆盖 NFS mode 漂移 | 1 | 66项中62项通过，仅Phase 1/5的4个汇总状态失败；定向Stage 5诊断确认唯一根因是phase0 source的NFS 100644→100755漂移。下一轮在同一容器命名空间把既有只读`oscar-glm-phase0-source-fd3e0b3`卷挂到精确base source路径，不修改文件或放宽verifier。 |

## 当前阶段状态（BF16 tile gate）

- BF16 tile gate 源码已由提交
  `ca4a404e913ce55237ca60383cc86e221fbfea26`（tree
  `079815219a02add3f37318ed434924e80f80a35d`）发布到源码远端，源码仓库
  clean/published。CPU/离线证据已复制到 control artifact，证据清单
  SHA256 为 `50535adca593513a0eb226614ce5413fdaa081645f48f4b11329a2e9dd361e95`。
- 修改报告前已完整重读当前 2,716 行，读取前后 SHA256 均为
  `ba82eff0aec78bfac64ce2174b19452517471d7328eba8ce69e4e896f1a4cc6c`，
  确认期间无并发手工修改。下一步实时新增 2.40，明确本阶段尚无苹果800
  CUDA 正确性或性能结果；通过报告门禁并发布主仓库前不分配 GPU。
- 2.40 已完成并通过发布前门禁：报告为 2,777 行，SHA256
  `97290f8122d153397e6ff9202c6059c5f419a89acfd6f0d871776a2306a41be7`；
  1.1–1.5、2.1–2.40 连续，交叉引用有效，`三池=0`，正文大写 `A800`
  仅在第 5 行允许链接。10 份证据加清单共 14,353 bytes、源码/测试哈希
  与全部 manifest 条目均从落地文件复算一致，`git diff --check` 通过。
  下一步提交并推送报告、submodule 指针与 planning；发布前不分配 GPU。
- 2.40、submodule 指针与 planning 已由主仓库提交 `70af7e8` 推送到
  `feat/glm52-model-load`，远端已从 `ccd5cd3` 快进。下一步发布本条状态，
  恢复两仓 clean/published 后再执行新的双 GPU 空闲检查。
- 发布状态已由主仓库 `1ade59d` 固化，两仓 clean/published。只读核对证明
  既有 2K 基准把 query chunk 与最终序列长度混为同一值，无法代表后 15 个
  32K chunk；因此只给工具新增独立 `--final-seq-len`，生产源码不变。
  TDD 红灯 1 error；有效定向 unittest 4/4、Phase 9 工具测试 22/22、Ruff
  0.14.0 check/format、固定 Python compile、CLI help/非法边界和 diff 均通过。
- 有效 CPU coverage ID 为
  `20260731T1622Z_later_chunk_coverage_cpu_v2`：2,048 queries 位于
  `[30720,32768)`，4,194,304 个 selected index 均有效、逐行唯一且 causal。
  共 262,144 个 16-token tile，其中 257,626 个为全 history、无 BF16 token，
  占 `98.2765%`；该比例仅描述 seed 42 的合成随机 selected 分布，不能冒充
  正式 DSA 分布或性能收益。8 份证据加清单共 2,614 bytes，manifest SHA256
  为 `9cea5c83d4adb30cc85e4823b3ebcd81bffef3b558bbd5f3bb3ad6aa96b5d7f7`。
  下一步全文重读并实时新增 2.41，发布基准协议前不分配 GPU。
- 修改 2.41 前已完整重读当前 2,777 行报告，覆盖 1–400、401–800、
  801–1,200、1,201–1,600、1,601–2,000、2,001–2,400 和
  2,401–2,777；读取前后 SHA256 均为
  `97290f8122d153397e6ff9202c6059c5f419a89acfd6f0d871776a2306a41be7`，
  确认期间无并发手工修改。现在新增 2.41；发布前仍不分配 GPU。
- 2.41 发布前门禁已通过：报告为 2,848 行，SHA256
  `426921c3f8d828b2ea23e9c517cbc78b28e01f717f90922e43ef72dcc7796579`；
  1.1–1.5、2.1–2.41 连续，交叉引用和术语有效。coverage 公式、脚本/测试
  哈希、8 份证据加清单的 2,614 bytes 与 manifest 全部从落地文件复算一致，
  `git diff --check` 通过。下一步提交推送工具、测试、2.41 与 planning；
  发布前不分配 GPU。
- 32K later-chunk 工具、测试、2.41 与 planning 已由主仓库提交
  `6469cbf` 推送到 `feat/glm52-model-load`。下一步发布本条状态并确认
  两仓 clean/published；随后才为 fd281f5f9/ca4a404e9 对照轮次执行新的
  双 GPU 空闲检查。
- 发布状态已由 `bd2a67f` 固化，两仓 clean/published。对照轮次外层空闲
  检查为 `15:43:02Z/15:44:36Z`，间隔 94 秒；两次均为 8/8 张苹果800
  `0 MiB/0%`、无 compute process。外部下载容器 DeviceRequests 为 null，
  不占 GPU，因此无需终止。fd281f5f9 临时源码快照的 kernel SHA256 为
  `e8b1baabc43b080e0992dab8901ff4de347fb68c905793f906934effdacb11ca`，
  6 个原生链接均有效。下一步发布本条状态、启动前即时复查，再固定 GPU 0
  先运行 fd281f5f9 控制轮次。
- 空闲状态已由 `8c4eb2a` 发布；启动前即时复查仍为 8/8 空闲。单次 GPU 0
  容器中的对照轮次 `20260731T1637Z_later_chunk_tile_gate_compare_v1`
  退出码 0，fd281f5f9/ca4a404e9 均为 passed。split1 CUDA 中位数为
  `22.618113/20.226048 ms`，候选降低 `10.575883%`、加速
  `1.118267×`；wall 中位数降低 `10.650641%`。两边 split16 为
  `338.324493/338.262024 ms`，只差 `-0.018464%`。
- 两边同源 split16 参考的 allclose 均通过，output/LSE max_abs
  `0.003370285/0.002224922` 且诊断逐字段相同。候选实际 runtime 资源相对
  控制为 registers `247→242`、stack `0→0 B`、shared
  `109568→109568 B`。控制 7 个样本含一个 `25.821184 ms` 高值，median
  仍由其余约 `22.54–22.68 ms` 样本决定；候选范围为
  `20.185087–20.427776 ms`。退出后 8 卡全空闲。
- 17 份 GPU/结果/资源证据加清单共 306,881 bytes 已复制到 control
  artifact，comparison summary/manifest SHA256 为
  `95cf66b18ac8e4bc7f00f44ceee99b4985feb333487ebefc97f3e9306181cc53`/
  `b5fa7e93c2046ee704da6a199cc91e649b92b5301edd9a02f20f5d408ea56783`。
  下一步全文重读并实时新增 2.42；发布本阶段前不启动完整 CUDA 回归。
- 修改 2.42 前已按 7 个连续区间完整重读当前 2,848 行报告；
  读取前后 SHA256 均为
  `426921c3f8d828b2ea23e9c517cbc78b28e01f717f90922e43ef72dcc7796579`，
  且编辑前再次实算仍一致，确认期间无并发手工修改。下一步只新增
  2.42，然后执行章节、交叉引用、术语、数值、证据哈希和 diff 门禁。
- 2.42 已实时写入并通过发布前门禁：报告为 2,923 行，SHA256
  `b0ca771b71048aaeab28b34da6497a7324b4388766c3e8c0cf5c3b93d3abb4f2`；
  1.1–1.5、2.1–2.42 连续，35 个引用语境全部指向现有章节；
  `三池=0`，大写 `A800` 仅在第 5 行允许的历史链接。
  17 份证据加清单、306,881-byte 总量、所有哈希、指标公式、Git
  tree/kernel 字节和 `git diff --check` 均实算通过。下一步只提交并
  推送报告与 planning；发布前不启动完整 CUDA 回归。
- 2.42 与 planning 已由主仓库提交 `0a8a559` 推送，远端分支已快进。
  下一步发布本条状态以恢复 clean/published，然后为 ca4a404e9
  完整 cold-cache CUDA 回归重新执行两次间隔至少 60 秒的 GPU 空闲检查。
- 发布状态已由 `7dbb455` 推送，两仓 clean/published。完整 CUDA
  回归的新双空闲检查为 `16:00:27Z/16:01:33Z`，间隔 66 秒；
  两次均为 8/8 张苹果800 `0 MiB/0%`、无 compute process。唯一项目外
  下载容器的 DeviceRequests 为 null，不占 GPU，因此无需终止。下一步先
  发布该空闲状态，然后启动前即时复查，固定 GPU 0 与独立 cold
  Triton cache 运行 ca4a404e9 完整 `tests/oscar_mla`。
- 空闲状态已由 `14a55ba` 发布。启动前 `16:04:55Z` 即时复查仍为
  8/8 卡空闲。完整 cold-cache CUDA 轮次
  `20260731T1604Z_bf16_tile_gate_full_cuda_v1` 退出码 0：
  127 passed、0 failed、19 warnings、87.91 秒；全新 Triton cache 为 380 个文件、
  25,038,227 bytes。容器删除后 `16:06:44Z` 为 8/8 卡空闲、无
  compute process。8 份证据加清单共 94,565 bytes，manifest/pytest SHA256
  为 `d55a7235…0c3`/`3e57bee7…bfd`，380 行 cache 哈希已入清单。
  下一步完整重读当前报告后实时新增 2.43；发布前不进入后续 OCI
  或 32K/batch1 轮次。
- 修改 2.43 前已按 5 个连续区间重读当前 2,923 行报告，覆盖
  1–600、601–1,200、1,201–1,800、1,801–2,400 与 2,401–2,923；
  读取前后 SHA256 均为
  `b0ca771b71048aaeab28b34da6497a7324b4388766c3e8c0cf5c3b93d3abb4f2`，
  确认期间无并发手工修改。下一步只追加 2.43 并执行发布前门禁。
- 2.43 已实时追加并通过发布前门禁：报告为 2,997 行，SHA256
  `430479a60c058c188044586a2be59a5b645492b42e34b60400b83efd6d7e47cf`；
  1.1–1.5、2.1–2.43 连续，37 个章节引用语境全部有效；旧术语实际词项
  计数为 0，大写 `A800` 仅在第 5 行允许的历史链接。8 份证据
  加清单、94,565-byte 总量、380 行 cache 哈希、127 项 pytest、Git/tree/
  image 身份和 `git diff --check` 全部实算通过。下一步只提交推送
  报告与 planning；发布前不进入 Phase 6 迁移。
- 2.43 与 planning 已由主仓库提交 `b6407f0` 推送，远端分支已快进。
  下一步发布本条状态使两仓恢复 clean/published，然后才最小迁移
  Phase 6 候选输入到 ca4a404e9，执行 JSON/Dockerfile/PAX 确定性与身份门禁。
- 发布状态已由 `15a3076` 推送。Phase 6 输入已最小切换到
  `ca4a404e913ce55237ca60383cc86e221fbfea26`/tree
  `079815219a02add3f37318ed434924e80f80a35d`，新 tag 为
  `glm52-oscar-a800-phase6-ca4a404e9-0275043c`。Dockerfile/config SHA256 为
  `51ed571f615559a0008631c17244d88878ef0c8437c3c3161a61d1e4c022e6f4`/
  `5f3fb384f5591011db8a4c8f511cb7ea79d2c14b1d7c8186e32d3151bcb8b802`。
  有效 CPU-only v2 的 JSON/旧身份清零/本地远端一致、PAX 1/1、3 份
  Python compile 与 diff 门禁全部通过；GPU 为 8/8 `0 MiB/0%`。
  下一步完整重读报告并新增 2.44，发布配置前不启动 OCI 构建。
- 修改 2.44 前已完成当前 2,997 行报告的全文重读；末段因合并输出截断，
  又独立重读 2,251–2,500、2,501–2,750 与 2,751–2,997 行。读取后
  SHA256 仍为
  `430479a60c058c188044586a2be59a5b645492b42e34b60400b83efd6d7e47cf`，
  确认期间无并发手工修改。下一步只追加 2.44 并执行章节、引用、术语、
  配置身份、落地日志哈希和 diff 门禁；发布前不启动 OCI 构建。
- 本阶段第一次 planning `apply_patch` 因给 `findings.md` 使用了不存在的
  尾部上下文而在写入前整体 fail-closed；重新读取三个 planning 文件尾部后，
  改用各自实际上下文，不影响报告或实验输入。
- 2.44 首次结构化校验误在宿主机调用仅存在于固定控制镜像内的
  `/opt/fp8_speed_up_v4_venv/bin/python`，在读取报告前以 127 退出；没有
  修改任何文件。改为使用禁网、只读挂载、不分配 GPU 的固定控制镜像重跑。
- 固定控制镜像内的第一次结构化校验把 `1.06%`、`2.85×` 等性能小数误判
  为章节引用，因 11 个假阳性触发断言；镜像身份正确且未分配 GPU，也没有
  修改文件。下一轮只把引用识别收紧到实际章节语境，并保留未知候选审计。
- 2.44 已实时追加并通过发布前门禁：报告为 3,058 行，SHA256
  `7e548377dca7bb9ac3597802c29e91208909dd65a187b215b208021666d6530a`；
  1.1–1.5、2.1–2.44 连续，87 个已存在章节引用语境有效，11 个未知数字
  候选逐项确认均为百分比/倍数/秒数；`三池=0`，大写 `A800` 仅在第 5 行
  历史链接。配置/Dockerfile/source tree、旧身份清零、v1/v2 落地日志哈希、
  v2 退出码和 `git diff --check` 均实算通过。下一步只提交推送配置、
  Dockerfile、2.44 与 planning；发布前不启动 OCI 构建。
- Phase 6 ca4a404e9 输入、2.44 与 planning 已由主仓库提交 `323671c`
  推送到 `feat/glm52-model-load`。下一步只发布本条状态并确认两仓
  clean/published；随后执行两轮相互独立的 CPU-only 确定性 OCI 构建。
- 发布状态已由 `a1720a0` 固化，两仓 clean/published。已完整读取 Phase 6
  build/verify 实现并复核上一 fd281f5f9 双构建协议；固定控制镜像内 Python
  为 3.12.13，运行时禁网、显式清空 `CUDA_VISIBLE_DEVICES`、不传 `--gpus`。
  下一步并行运行 ca4a404e9 v1/v2 独立 layout 的 build+递归 verify，超过
  10 分钟时打印心跳；完成后先更新报告，daemon 导入仍禁止。
- 首次并行双构建在约 31 秒内均于 build 的首个 Git 状态检查 fail-closed：
  `/workspace` 和源码仓库实际属 UID 0，而固定容器按宿主用户 UID 22633
  运行，容器内没有宿主的 `safe.directory` 配置，因此 Git 以 dubious
  ownership 退出 128；两个组合退出码均为 1，尚未创建 OCI layout。
  两份失败日志 SHA256 同为
  `b52115e991a6bfafbd0ac1f3afa1ebabf267a6713f4760389c060a07513e3f6e`，
  退出码文件 SHA256 同为
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`。
  下一轮保留失败目录，使用进程级 `GIT_CONFIG_*` 分别声明两个精确
  safe.directory，不写全局配置、不改变仓库所有权，再以新 v3/v4 目录重试。
- 进程级 safe.directory 预检通过后，v3/v4 CPU-only 双构建与递归验收在
  1,171 秒完成，两个组合退出码均为 0；10 分钟时已打印正式心跳
  `elapsed_seconds=600`。首次只读汇总在打印 v3 哈希后因宿主机没有 `jq`
  退出，未修改产物；改用固定控制镜像 Python 完成两轮 JSON/OCI 比较。
- 固定 Python 的首次确定性比较又因手工错误补全 `0051472` 的完整哈希而
  fail-closed；Git/build report 实际完整值为
  `00514720643346092066cf29acdbece87651500c`。该脚本只读、未修改产物；
  同轮结束后 GPU 快照为 8/8 `0 MiB/0%`、无 compute process。下一轮
  直接从 Git/report 取值比较，不再手工扩写缩写。
- 从 Git/report 实算后，v3/v4 确定性复核状态为 passed。两轮分别为
  build=`built`、verification=`passed`、exit=0，绑定主仓库
  `00514720643346092066cf29acdbece87651500c` 和源码 ca4a404e9/tree
  `07981521…a35d`。候选 image/config 为 `sha256:7c85cdd0…4eb8`，
  manifest 为 `sha256:fb8e914c…4a23`，candidate layer 为
  `sha256:3f03376d…e203`、diff-ID `sha256:5f8875b9…7a14`、
  109,147,892 bytes/5,298 members。
- v3/v4 的 `index.json`、config blob、manifest blob 和 candidate layer
  已逐字节比较完全一致；`index.json` SHA256 为 `18310465…4d90`。
  两轮各自验收 4,744 个 Git 文件、4 份 rotation artifact、7 个 base
  原生扩展，前 32 个 base layers 精确匹配，candidate layer 不含原生扩展
  或 whiteout。下一步完整重读 3,058 行报告后新增 2.45；发布前禁止
  daemon import。
- 修改 2.45 前已按 1–500、501–1,000、1,001–1,500、1,501–2,000、
  2,001–2,500、2,501–3,058 六个连续区间完整重读报告；读取后仍为
  3,058 行，SHA256
  `7e548377dca7bb9ac3597802c29e91208909dd65a187b215b208021666d6530a`，
  确认期间无并发手工修改。daemon 中新 ca4a404e9 tag 实测不存在；唯一
  下载容器 DeviceRequests=null。下一步只新增 2.45 并做发布前门禁。
- 2.45 已实时追加并通过发布前门禁：报告为 3,145 行，SHA256
  `f715f5c3cf344b08f4a4c82b969570f28abfdd29ef1a967764a9cf63f63a6f3a`；
  1.1–1.5、2.1–2.45 连续，89 个已存在章节引用语境有效，11 个未知数字
  候选均为性能小数；术语门禁通过。v1/v2 失败日志/退出码、v3/v4
  build/verify/log/exit、4 个 OCI blob 的 size/hash/逐字节一致性和 daemon
  tag 不存在均从落地文件实算通过，`git diff --check` 通过。下一步只提交
  推送 2.45 与 planning；发布前不执行 daemon import。
- 2.45 与 planning 已由主仓库提交 `9fbba33` 推送。下一步只发布本条状态、
  确认两仓 clean/published；然后才从已验收 v3 的只读 OCI layout 执行
  CPU-only daemon import 与独立身份审计，并在进入 runtime import 前先
  更新报告。
- 发布状态 `11b8ccd` 后，从 v3 `index.json` 原样读取 ref name，使用一次性
  Ubuntu 22.04 / skopeo 1.4.1 将只读 OCI layout 导入 daemon。首次外层
  工具调用在容器仍运行时提前返回，未生成预期退出码；没有重复导入，而是
  对同一容器执行 `docker logs -f` + `docker wait` 接管。容器最终退出 0，
  原日志与恢复日志均完整到 `Storing signatures` 且逐字节相同，SHA256
  `75b98c9d9140e9c2d217d3660f868d7e2cd740dbdae66cda1bb02719d7094551`；
  恢复退出码文件 SHA256 为 `9a271f2a…86aa`。
- daemon identity audit 状态为 passed：image ID `sha256:7c85cdd0…4eb8`、
  33 层、最后 diff-ID `sha256:5f8875b9…7a14`、目标 tag 和 8 项 labels
  全部匹配 v3 build report。inspect/audit/post-GPU SHA256 为
  `d9a47f16…76bd`/`15cd9abd…7c5b`/`e3d6d9dc…ad40`；结束后 8 卡
  0 MiB/0%、无 compute process。下一步全文重读 3,145 行报告并新增
  2.46；发布前禁止 runtime import。
- 修改 2.46 前已按连续区间完成 3,145 行报告全文重读；对界面首次
  截断的末段又按 2,201–2,450、2,451–2,700、2,701–2,950 和
  2,951–3,145 行逐段补读。读取后报告仍为 3,145 行，SHA256
  `f715f5c3cf344b08f4a4c82b969570f28abfdd29ef1a967764a9cf63f63a6f3a`，
  确认期间无并发手工修改。
- daemon 证据的首次只读复核写反 GPU 快照文件名，并把容器内固定
  Python 路径用于宿主机；同命令的 `docker ps` 模板也不支持直接
  展开 HostConfig。这些命令均为只读，未修改任何产物。改用实际文件名
  `daemon_import_post_gpu.log`、直接读取 audit JSON 和逐容器
  `docker inspect` 后复核通过；当前唯一下载容器 DeviceRequests=null。
- 2.46 首轮结构化门禁用宽泛子串禁止 `2.47/2.48`，误命中历史性能
  数值 `882.474/882.489 ms` 的尾部子串而 fail-closed。报告、证据和 daemon
  均未被修改；下一轮只按章节语境提取引用，避免将小数误判为章节。
- 第二轮章节语境提取已正确识别 `2.40–2.43`，但错误要求报告
  必须含 2.46 的正文自引用。进一步定位确认“2.46 前”实际位于 planning，
  报告内的 2.46 只应出现在新章标题；标题已由 1.1–1.5/2.1–2.46 精确
  连续性断言覆盖。下一轮删除无依据的自引用要求，其余断言不放宽。
- 第三轮已通过报告结构/引用/术语和主要文件哈希，但校验器过度要求
  中文报告逐字展开 8 个 label 的每个值；报告实际只展开关键身份，
  其余值由 audit JSON 及其 SHA256 封存。该轮只读、未改产物。下一轮
  改为强制 audit 的 8 项 labels 与 build report 派生期望值、daemon inspect
  完全一致，不强迫报告重复展开全部值。
- 2.46 最终结构化门禁通过：报告 3,208 行、SHA256
  `99de2358b0a488dfb04e204d0171baa0ad3e1b04df9aad5afaf0e8a127f84679`；
  1.1–1.5/2.1–2.46 标题连续，61 个章节语境引用和 1 个范围引用
  全部有效，术语门禁通过。导入/接管日志、两份退出码、inspect/
  audit/GPU 快照哈希全部与报告一致；8 项 labels 与 build report
  派生值、daemon inspect 三方精确一致，`git diff --check` 通过。
  下一步只提交推送 2.46 与 planning；发布前不执行 runtime import。
- 2.46 与 planning 已由主仓库提交 `5498fee` 推送到
  `origin/feat/glm52-model-load`。下一步只发布本条状态并确认
  两仓 clean/published；随后复用冻结 runtime import 协议，在分配 GPU 0
  前重新完成两次间隔至少 60 秒的 8/8 GPU 空闲检查。
- 发布状态 `07a2fa3` 后，runtime import 前新双检已于
  `2026-07-31T17:21:52Z/17:23:08Z` 完成，间隔 76 秒。两次均为
  8/8 张苹果800 `0 MiB/0%`、无 compute process；唯一项目外下载容器
  DeviceRequests=null，不占用 GPU。idle log SHA256 为
  `65a6f94ae124b26a8f3b3452eb1c82bc3980828d27728af7858f54c5fa1df733`。
  下一步先发布本空闲状态；发布后做启动前即时复查，只分配 GPU 0
  并精确复用冻结 runtime import 协议。
- 双空闲状态由 `1770f2b` 发布后，启动前复查仍为 8/8 卡
  `0 MiB/0%`、无 compute process。有效 runtime import 只分配 GPU 0、
  只注入 driver，不加载模型或执行 CUDA kernel；容器退出码为 0。
- `runtime_import.json` 状态 passed：Python/PyTorch/Triton 为
  `3.12.13/2.11.0+cu129/3.6.0`，Transformers/Tokenizers 为
  `5.8.1/0.22.2`，FlashInfer Python/JIT cache 为
  `0.6.6/0.6.6+cu129`；候选 vLLM Python/`_C`、78 层 rotation、
  rotation manifest/rotations/runtime expectation 与 `reasoning_effort=max`
  全部匹配，`cuda_initialized=false`。JSON/log 与 fd281f5f9 冻结证据
  逐字节一致。
- idle/startup/JSON/log/exit/post-GPU SHA256 依次为
  `65a6f94a…f733`/`bad6a83f…2e51`/`0910b598…7b7a`/
  `f2e60043…189a`/`9a271f2a…86aa`/`3749a54e…35e9`。有效容器已自动
  删除，`17:25:04Z` 退出复查为 8/8 卡空闲。下一步全文重读
  3,208 行报告并新增 2.47；发布前不切换 Stage 9 控制镜像。
- 修改 2.47 前已按 1–550、551–1,100、1,101–1,650、1,651–2,200、
  2,201–2,750 和 2,751–3,208 六个连续区间完整重读报告。读取后
  仍为 3,208 行，SHA256
  `99de2358b0a488dfb04e204d0171baa0ad3e1b04df9aad5afaf0e8a127f84679`，
  确认期间无并发手工修改。下一步只追加 2.47 并执行发布前门禁。
- 2.47 已追加并通过固定容器内结构化门禁：报告 3,258 行、SHA256
  `813fef10f4aadf30d9c6943e4ac39a288bd9441b185ba632f1e6397e33371490`；
  1.1–1.5/2.1–2.47 标题连续，64 个章节语境引用和 1 个范围引用
  均有效，术语门禁通过。六份 runtime 证据哈希、双检间隔 76 秒、
  三份 GPU 快照、结构化版本/artifact 值、JSON/log 与历史证据逐字节一致、
  daemon/source 身份和 `git diff --check` 全部通过。下一步只提交推送
  2.47 与 planning；发布前不切换 Stage 9 控制镜像。
- 2.47 与 planning 已由主仓库提交 `53c55b0` 推送到
  `origin/feat/glm52-model-load`。下一步发布本条状态并确认两仓
  clean/published；然后才将 Stage 9 控制 Dockerfile 的默认 base 最小切换
  到 ca4a404e9 候选，先做 CPU-only 身份复核和入口发布。
- 2026-08-01 会话恢复复核：主仓库 `40e0624`、源码仓库
  `ca4a404e9` 均与各自 upstream 一致且工作区干净；优化记录仍为
  3,258 行、SHA256 `813fef10…71490`。8/8 张 GPU 均为
  `0 MiB/0%`、无 compute process，唯一项目外下载容器不占 GPU。
  当前仍按原计划进入 ca4a404e9 Stage 9 控制镜像与正式链路迁移，
  后续以同一 32K/batch1 负载与 BF16 baseline 对比 TTFT/TPOT。
- Stage 9 控制入口已完成最小 CPU-only 迁移：
  `docker/Dockerfile.phase9-runtime` 仅将默认 base 切换为
  `glm52-oscar-a800-phase6-ca4a404e9-0275043c:latest`，新文件 SHA256 为
  `f832ebb19cf7ffe28e19b26a3978aa80b59a5fd4b2f07dff40d04b6aeb342737`，
  旧 `fd281f5f9` 身份计数为 0。落地验证
  `20260731T1735Z_runtime_ca4a404e9_input_v1` 状态 passed：daemon base
  image ID `sha256:7c85cdd0…4eb8`、33 层、source commit/tree、candidate
  layer digest 与 Phase 6 build report 全部一致；源码仓库 clean/published，
  未分配 GPU，检查时 8/8 卡 0 MiB/0%。下一步全文重读报告并实时新增
  2.48；发布入口前不构建新控制镜像。
- 2.48 已实时追加并通过发布前门禁：报告为 3,312 行、SHA256
  `921ee9d774c8082810d6e123ab62da7e30732e047241a2f48de6308695359590`；
  1.1–1.5/2.1–2.48 连续，术语、Dockerfile 新旧哈希、daemon/base/source
  身份、4 份证据及 2,179-byte 总量和 `git diff --check` 全部实算通过。
  下一步只提交推送 Dockerfile、2.48 与 planning；发布前不构建控制镜像。
- 入口与 2.48 已由提交 `a8cb54b` 推送后，CPU-only 控制镜像
  `oscar-glm-stage9-runtime:ca4a404e9` 构建成功，image ID
  `sha256:265e6ca1fb1b9947a125e58e1ec1243e241628d2f25d5412982bbf15ad9067f1`。
  34/33 层继承、前 33 层、labels 与 entrypoint 全部 matched；runtime
  为 passed，Git/iproute2/Python/glibc/固定包与 fd281f5f9 控制镜像一致，
  `cuda_initialized=false`，runtime JSON 逐字节一致。构建前后 8 卡均
  0 MiB/0%，无 compute process。下一步全文重读 3,312 行报告并追加
  2.49；发布前不迁移 Phase 1/5/7/9 配置。
- 2.49 已实时追加并通过发布前门禁：报告为 3,369 行、SHA256
  `9fcecdf3b875666d729f00a40e2bbc09dbedf48a9f488960315a54ce78e3009c`；
  1.1–1.5/2.1–2.49 连续，术语、镜像身份、13 份证据/42,387-byte 总量、
  runtime 逐字节一致性和 diff 全部通过。下一步只提交推送 2.49 与
  planning；发布前不迁移正式配置。
- ca4a404e9 正式 overlay 已从 v3 `extracted-layer` 单份机械复制完成：
  4,749 个普通文件的 source/overlay 递归 SHA256 清单逐字节一致，清单
  SHA256 均为 `45eb2d62…eeb6`；6 个原生扩展 symlink 与 fd281f5f9
  正式 overlay 的相对路径/绝对目标逐字节一致，链接清单 SHA256
  `f1794994…76b2`，目标哈希清单 `c2a5c7c2…9468`。
- Phase 1→5→7→9 配置与 9 个正式 shell、Phase 9 工具测试期望已最小迁移；
  四份 config SHA256 为 `84163a19…04e`/`aea9d51b…5613`/
  `2f725eee…9678`/`14d3f71e…4760`。4 个 JSON 解析、9 个 shell 语法、正式
  范围旧 fd281 身份清零和 diff 均通过。下一步在新控制镜像内执行 Phase 7/9
  工具测试和 CPU-only 递归 verifier；结果发布前不分配 GPU。
- 有效工具与递归门禁已全部通过：Phase 7 v3 为 20/20、Phase 9 为
  22/22、固定 Python compile 为 11/11；CPU-only verifier v3 为
  64/64 checks、0 failed、exit=0，JSON/log 逐字节一致且 SHA256
  `67b040dc…fac9c`。静态 summary JSON/log SHA256 均为
  `d940e52c…d838`，证据 manifest 为 `ec772942…24a5a`；25 份文件共
  102,643 bytes。前后 8 卡 0 MiB/0%、无 compute process。下一步全文重读
  3,369 行报告并新增 2.50；发布前不执行 driver-injected preflight。
- 2.50 已实时追加并通过发布前门禁：报告为 3,455 行、SHA256
  `68a0f76d6f78a172016c51479c64aa81fffb005206d137adf6100f36c7cf04c4`；
  1.1–1.5/2.1–2.50 连续，术语、4 份配置、overlay 清单、25 份静态证据/
  102,643-byte 总量、64/64 和旧身份清零均实算通过，diff 无错误。
  下一步只提交推送配置、wrapper、2.50 与 planning；发布前不分配 GPU。
- 静态迁移已由提交 `59e8d1a` 推送，两仓 clean/published。正式 preflight
  外层双空闲检查为 `18:13:35Z/18:14:47Z`，间隔 72 秒；两次均为
  8/8 张苹果800 `0 MiB/0%`、无 compute process，idle log SHA256
  `28991d3d…6c5c`。下一步先发布本空闲状态，再做启动前即时复查并执行
  driver-injected preflight。
- 双空闲状态由 `77b5aa5` 发布后，有效 preflight run ID
  `20260731T1816Z_candidate_ca4a404e9_preflight_v1` 退出 0：静态 64/64、
  fixed environment import 和服务参数解析全部通过，后两者均为
  `cuda_initialized=false`；解析值固定 131072/16/2048、OSCAR INT2、
  eager、torch profiler。未加载模型，容器自动删除，退出后 8 卡全空闲。
- 有效 static/fixed/args SHA256 为 `d8db6b23…126c`/
  `56f92356…0574`/`6365691f…9f2c`；preflight log/exit/post 为
  `34c09065…7a15`/`9a271f2a…86aa`/`bc78eb1a…2b4a`。有效 v2 validation
  JSON/log 为 `f6ad93ac…7572`，manifest 为 `28dd8109…2665`；含首版汇总
  失败边界共 14 份文件、45,826 bytes。下一步全文重读报告并新增 2.51；
  发布前不启动 32K/batch1 正式性能轮次。
- 2.51 已实时追加并通过发布前门禁：报告为 3,528 行、SHA256
  `a33e151837a0ce00a38ca1282a96c153197740416637b64285411eee50a6bc75`；
  1.1–1.5/2.1–2.51 连续，术语、14 份证据/45,826-byte 总量、64/64、
  两处 CUDA=false、参数与失败汇总边界全部实算通过，diff 无错误。
  下一步只提交推送 2.51 与 planning；发布前不启动正式性能轮次。
- 2.51 已由提交 `cedeb16` 推送，两仓 clean/published。正式 32K/batch1
  OSCAR 外层双空闲检查为 `18:24:29Z/18:25:45Z`，间隔 76 秒；两次均为
  8/8 张苹果800 0 MiB/0%、无 compute process，idle log SHA256
  `c5abd679…a811`。下一步发布空闲状态后做启动前复查，并运行固定 3 轮加
  8-rank profiler 的同负载正式 cell。
- ca4a404e9 的正式 32K/batch1 单格运行
  `20260731T1826Z_candidate_ca4a404e9_32k_b1_v1` 已 exit=0，单格与总
  summary 均为 `passed`。三轮 `mean` 指标中位数为 TTFT
  `32683.066 ms`、TPOT `200.037 ms`、请求吞吐 `0.01722294 req/s`；
  相对 fd281f5f9 分别为 `-8.41%/+1.12%/+4.73%`，相对 BF16 为
  `+160.88%/+11.86%/-39.31%`。TPOT 仍在 BF16 +20% 门限内，TTFT
  仍高出门限 `17649.435 ms`，性能尚未收敛。
- profiler 状态 `passed`，前端 1 份、8-rank trace 与 8 份 CUDA table
  全部齐全；证据仅复制小型 summary/validation/table/server log，不复制
  约 1.2 GiB 原始 trace。证据 manifest 40/40 校验通过，SHA256
  `21fdbe75…1fee`；退出后容器已删除、8 卡 0 MiB/0%、无 compute process。
  下一步先全文重读并实时追加报告 2.52；发布本阶段后，再做 CPU-only trace
  归因并据此选择下一项最小优化。
- 修改 2.52 前已顺序重读报告全部 3,528 行，读取前后 SHA256 均为
  `a33e1518…bc75`，确认没有并发手工修改。2.52 已追加并通过门禁：报告为
  3,619 行、SHA256 `bd38d36d…c0a4`；1.1–1.5/2.1–2.52 连续，术语、
  三轮指标、BF16/旧候选差值、8-rank profiler、40/40 证据和 diff 均通过。
  下一步只提交推送本阶段记录；发布前不启动 trace 归因。
- 有效 CPU-only 归因 ID 为
  `20260731T1948Z_ca4a404e9_32k_prefill_trace_v2`，固定环境为
  Python 3.12.13/ijson 3.4.0.post0、runc、network none、4 CPU worker；
  summary/validation 均为 passed，8/8 ranks 均为 144 contexts、16 chunks、
  32,768 tokens。prefill wall/kernel/generation 中位数为
  `32756.592/31755.930/269.372 ms`，stage1 为 `20128.143 ms`/
  1,248 次、占 wall `61.45%`。
- 相对 fd281f5f9，wall/stage1 分别下降
  `2974.890/3006.728 ms`，stage1 解释 wall 改善的 `101.07%`；
  去掉 stage1 后的剩余 wall 仅增加 `31.838 ms`。相对 BF16，
  当前 stage1 减原生 attention 的超额仍解释 prefill wall 差距的
  `73.86%`。小型证据 8 项加 manifest 共 346,895 bytes，manifest
  SHA256 `bf94602f…5b73`，未复制原始 trace。下一步先全文重读并
  实时新增 2.53，发布前不进入下一源码候选。
- 修改 2.53 前已顺序重读报告全部 3,619 行，读取前后 SHA256 均为
  `bd38d36d…c0a4`，确认没有并发手工修改。2.53 已实时追加并通过门禁：
  报告 3,698 行、SHA256 `b98b932f…bb9e`；1.1–1.5/2.1–2.53 连续，
  章节引用、术语、归因公式、8/8 证据、环境锁定和 diff 均通过。
  下一步只提交推送本阶段；发布完成前不修改下一候选源码。
- 2.53 与 planning 已由主仓库提交 `a66db02` 推送；主仓库与源码仓库
  本地/远端分别精确一致于 `a66db021…da`/`ca4a404e9…a26`，两仓
  工作区均干净，8 张 GPU 均为 0 MiB/0%。trace 归因阶段完成；
  下一步只读检查 stage1 的全 16-chunk 有效计算/访存路径，选择下一
  项最小候选；候选实施前不分配 GPU。
- 只读源码检查确认原生 prefill top-k 已有
  `VLLM_TOPK_PREFILL_SORT_INDICES=1` 路径：短行保持有效前缀/
  `-1` 尾部，长行对同一 selected 集合按 token index 重排；当前正式
  Phase 9 配置未启用该开关。CPU-only 合成覆盖率轮次
  `20260731T2002Z_prefill_sort_coverage_32k_v1` 已 passed，耗时
  809.295 秒，并在约 10 分钟时同步 12/16 进度。
- seed 42 的 16-chunk 合成 selected 中，含 BF16 的 tile 从
  457,470 降到 131,621，减少 325,849（`71.2285%`）；首 chunk
  因原生短行快捷路径不变，后 15 chunks 从 372,103 降到 46,254，
  减少 `87.5696%`。该结果只是合成工作量覆盖，不是正式 DSA 分布，也
  未测 sort kernel 成本、CUDA 时间或 TTFT。下一步先完整重读报告并
  实时新增 2.54；发布前不修改工具或正式配置。
- 修改 2.54 前已对报告全部 3,698 行按 8 个连续区间重新扫描，
  读取前后 SHA256 均为 `b98b932f…bb9e`。2.54 已实时追加并通过
  门禁：报告 3,790 行、SHA256 `f00f184c…f7c6f`；1.1–1.5/
  2.1–2.54 连续，术语、引用、覆盖率复算、9/9 证据和 diff 均通过。
  下一步只提交推送本阶段；发布完成前不修改工具或配置。

## 当前续跑状态（2026-08-01）

- **状态：** selected-index 排序候选工具化验证进行中。
- **已恢复证据：** 主仓库/源码仓库分别 clean/published 于 `8e01df5`/
  `ca4a404e9`；实时报告为 3,790 行、SHA256 `f00f184c…f7c6f`。
- **本阶段成功条件：** benchmark 默认行为保持不变；可显式比较同一 selected
  集合的原始顺序与按 token index 排序顺序；短行快捷路径与原生 prefill top-k
  一致；先得到失败测试，再在固定 ca4a 控制镜像内通过 CPU-only 定向/完整工具、
  Ruff、compile 与 CLI 门禁；阶段结束先全文重读并更新中文报告，再申请 GPU。
- **当前边界：** 工具内预排序只用于隔离测量 stage1 收益，不代表原生 top-k
  sort 开销；后者必须由后续正式候选轮次单独或端到端计入。
- **错误记录：** 恢复只读检查曾误用不存在的
  `configs/phase9_stage9_performance.json`；实际正式配置路径为
  `configs/phase9/performance_matrix.json`。该命令只读失败，未修改产物；后续
  不再重复错误路径。
- **TDD 红灯：** 首次固定控制容器命令直接调用候选 venv，因其中未安装
  pytest 而退出，未形成有效红灯；随后只读挂载既有 uv 安装的
  `pytest==8.3.5` target，固定 Python 3.12.13 下得到预期
  `4 failed, 4 passed`。四个失败分别覆盖 CLI、config 和两条排序语义，均为
  旧实现缺少新能力，不是环境错误。
- **实现后验证：** 定向 8/8、Phase 9 完整工具集 26/26 passed；Ruff 0.14.0
  check/format、固定 Python compile 与 CLI help 均通过。首轮 Ruff 因只读仓库
  默认 cache 路径失败，改用 `/tmp/ruff-cache`；随后 format check 真实发现主脚本
  需机械格式化并已修正；compile 首轮又因只读 `__pycache__` 失败，改用
  `/tmp/pycache` 后有效组合退出 0。
- **CPU 32K 末段审计：** Python 3.12.13/Torch 2.11.0+cu129、无可见 CUDA，
  2,048×2,048 共 4,194,304 个 selected index。排序前后逐行 selected 集合
  完全相同；含 BF16 tile 从 4,518 降至 2,326，减少 2,192（48.5170%）；
  默认两项 config 精确不变。耗时 89.436 秒，状态 passed。
- **下一步：** 全文重读 3,790 行实时报告，新增 2.55 记录工具/TDD/CPU 语义
  与“尚未测 CUDA/原生 sort 成本”的边界；发布代码和文档后才能申请 GPU。
- **报告门禁：** 修改前已完整重读 3,790 行，读取前后 SHA256 均为
  `f00f184c…f7c6f`。2.55 已追加并通过结构化门禁；报告现为 3,875 行、
  SHA256 `c0663130…6462e`，1.1–1.5/2.1–2.55 连续，术语、引用、证据值、
  4/4 manifest 与 diff 全部通过。
- **当前下一步：** 只提交并推送工具、测试、2.55 与 planning；两仓恢复
  clean/published 前不执行 GPU 双空闲检查或候选筛选。
- **发布状态：** 工具、测试、2.55 与 planning 已由主仓库提交 `c0c7b44`
  推送到 `origin/feat/glm52-model-load`。下一步发布本条状态并确认两仓
  clean/published；随后才启动新的双空闲检查，固定 GPU 0 做单层筛选。
- **GPU 双空闲门禁：** 状态提交 `9ccad9d` 发布后，`20:41:52Z/20:43:14Z`
  两次检查间隔 82 秒；两次均为 8/8 卡 `0 MiB/0%`、无 compute process。
  GPU/compute 文件 SHA256 分别为 `bc163a60…7d1fd`/空文件哈希
  `e3b0c442…b855`。下一步先发布本条状态，再做启动前即时复查并固定只使用
  GPU 0。
- **单卡筛选结果：** run `20260731T2044Z_prefill_sort_2k_gpu_v1` 固定 GPU 0、
  cold cache、5 warm-up+7 samples，exit=0/status=passed。原始/排序 split1
  CUDA 中位数为 `20.206593/19.581951 ms`，排序降低 `0.624641 ms`
  （`-3.091275%`，`1.031899×`）；两者均相对 split16 通过冻结 allclose。
  排序 output/LSE max_abs 为 `0.003452063/0.002224922`。stage1 资源为
  109,568 B shared、242 registers、0-byte stack。
- **证据边界：** 本轮把 selected 在计时前预排序，只测 stage1 数据顺序收益；
  没有计入原生 CUB sort。17 项证据加 manifest 共 18 文件/189,482 bytes，
  manifest SHA256 `62084dac…efe2`。候选状态为
  `stage1_screen_passed_native_sort_cost_unmeasured`。
- **下一步：** 重新全文读取当前 3,875 行报告并实时新增 2.56；发布完成前
  不执行下一实验。随后先测原生 `topKPerRowPrefill` sort 成本，再决定是否
  值得进入端到端 32K/batch1。
- **额外错误记录：** 证据 JSON 首次语法检查误用了只存在于控制容器内的
  `/opt/fp8_speed_up_v4_venv/bin/python` 宿主路径；随后改用宿主 python3
  并启用 fail-fast，三份 JSON 全部验证通过。一次合并 planning patch 又因
  上下文不精确被 `apply_patch` 拒绝，未产生部分修改；本条使用精确上下文
  分文件更新。
- **2.56 报告门禁：** 修改前已顺序完整复读 3,875 行，读取前后 SHA256
  均为 `c0663130…6462e`。2.56 已实时追加；报告现为 3,961 行、SHA256
  `4999befc3e39ecadbd98e6ec8994517be34efae7c5266e9547b81d8f966eb901`。
  1.1–1.5/2.1–2.56 连续，章节引用、术语、GPU timing/allclose/resource
  数值、17/17 manifest 与 `git diff --check` 均通过。
- **当前下一步：** 只提交并推送 2.56 与 planning，确认两仓
  clean/published；随后实现并验证原生 `topKPerRowPrefill` sorted/unsorted
  成本微基准。只有 stage1 收益扣除排序成本后仍为正，才进入生产开关和正式
  32K/batch1。
- **2.56 发布状态：** 报告与 planning 已由主仓库提交 `810593b` 推送；
  主仓库和源码仓库本地/远端分别一致于 `810593b`/`ca4a404e9`，两仓
  工作区干净。下一阶段可开始 CPU-only 的原生排序成本工具门禁。
- **原生成本工具口径：** 新工具只调用既有 `_C::top_k_per_row_prefill`，固定
  32K 最后 2K chunk 的 2,048×32,768 logits、row end 30,721–32,768 和
  top-k 2,048；同一输入分别切换 sorted/unsorted。先以 TDD 固化参数、行边界、
  timing 汇总与结果语义，再发布工具；不修改生产源码或正式配置。
- **原生成本工具 CPU 门禁：** TDD 红灯为脚本缺失时 collection 1 error；
  实现后定向 6/6 passed。最终固定 ca4a404e9 控制镜像组合为 Ruff/format、
  compile、CLI help 通过，Phase 9 四个工具测试文件 `32/32 passed`。没有
  分配 GPU 或执行原生 CUDA 算子。
- **2.57 报告门禁：** 修改前已完整复读 3,961 行且 SHA256 保持
  `4999befc…eb901`；追加后为 4,017 行、SHA256 `7722c5f2…a0d2`，
  1.1–1.5/2.1–2.57 连续，术语、引用、hash 与 diff 检查通过。
- **当前下一步：** 只提交并推送原生成本工具、测试、2.57 与 planning；
  两仓 clean/published 后重新完成两次至少间隔 60 秒的 GPU 空闲检查，随后
  固定 GPU 0 实测 sorted/unsorted 原生 top-k 成本。
- **2.57 发布状态：** 工具、测试、报告与 planning 已由主仓库提交
  `b3f2159` 推送；主/源码仓库本地与远端分别一致于 `b3f2159`/`ca4a404e9`。
  下一步只发布本条状态，再开始新一轮 GPU 双空闲门禁。
- **原生 top-k GPU 双空闲门禁：** `21:07:42Z/21:08:55Z` 两次检查间隔
  73 秒，8/8 GPU 均为 0 MiB/0%、无 compute process；外部下载容器不占
  GPU。下一步先发布本条状态，再固定 GPU 0 运行原生 sorted/unsorted 微基准。
- **启动前即时门禁：** 固定 ca4a404e9 控制镜像 image ID 为
  `sha256:265e6ca1…067f1`；`21:09:34Z` GPU 0 仍为 0 MiB/0%、无 compute
  process。发布本条状态后立即运行微基准。
- **GPU 微基准 v1 失败边界：** run `20260731T2110Z_topk_prefill_sort_2k_gpu_v1`
  exit=1；使用宿主 UID 导致镜像内 `getpwuid()` 失败，发生在导入阶段，未执行
  原生算子且没有 result/timing。容器已删除、8 卡全空闲。下一步先更新实时
  报告，再以新 run ID 使用镜像默认 root，保持其余协议不变。
- **2.58 报告门禁：** 修改前顺序扫描 4,017 行且 SHA256 保持
  `7722c5f2…a0d2`；追加后为 4,067 行、SHA256 `09cfc654…9a13`。
  章节连续、失败边界、7/7 manifest、术语和 diff 检查通过。下一步只发布
  2.58/planning；发布完成前不启动修正轮次。
- **v2 双空闲门禁：** `21:10:51Z/21:13:40Z` 两次检查间隔 169 秒，8/8
  GPU 均为空闲；主/源码仓库 clean/published。下一步发布状态，随后固定
  GPU 0、移除错误 `--user` 参数，保持其他协议不变。
- **原生 top-k GPU v2：** `20260731T2114Z_topk_prefill_sort_2k_gpu_v2`
  exit=0/passed；CUDA 中位数 unsorted/sorted 为 `0.368998/0.542362 ms`，
  排序成本 `0.173363 ms`。逐行集合相同、排序单调、invalid=0。与 2.56
  stage1 分离结果的算术组合估算净省 `0.451278 ms`（`2.193270%`），但不是
  正式 DSA/端到端结果。下一步封存证据、全文复读并实时新增 2.59；此前不修改
  生产配置。
- **2.59 报告门禁：** 修改前顺序扫描 4,067 行且 SHA256 保持
  `09cfc654…9a13`；追加后为 4,139 行、SHA256 `555ba1da…3923`。
  章节连续、术语、实测/估算边界、13/13 manifest 与 diff 检查通过。
- **当前下一步：** 只提交推送 2.59 与 planning；恢复 clean/published 后，
  最小启用正式 Phase 9 的 `VLLM_TOPK_PREFILL_SORT_INDICES=1`，先做静态和
  CPU 门禁并实时更新报告，再申请正式 32K/batch1 GPU 轮次。
- **candidate-only 配置 TDD：** 红灯为配置缺字段时 1 failed；最小实现新增
  config 唯一环境映射、candidate wrapper 精确读取/export、candidate verifier
  比较实际环境，baseline wrapper 不变。定向结果 1/1 passed；下一步执行完整
  CPU-only 静态/工具门禁并实时更新报告。
- **CPU 门禁错误记录：** 首轮 Ruff 命中两个既有文件的历史 I001 和两条既有
  E501，共 4 项，均不在当前 diff；组合随即停止。下一轮忽略这两个已确认的
  历史规则并执行其余 Ruff/format/JSON/shell/compile/完整 pytest，不修改无关行。
- **candidate-only CPU 门禁：** formatter 仅改新增测试块；最终 Ruff/format、
  JSON、shell、compile、diff 通过，Phase 9 四测试文件 `33/33 passed`。
  未注入 GPU runtime。下一步全文复读并实时新增 2.60，发布后再做 preflight。
- **2.60 报告门禁：** 修改前报告 4,139 行、SHA256 `555ba1da…3923`；追加后
  为 4,210 行、SHA256 `4bb51a49…04e6`。1.1–1.5/2.1–2.60 连续，术语、
  引用、配置传播、33/33 和 diff 检查通过。
- **当前下一步：** 只提交推送 candidate-only 配置链路、测试、2.60 与
  planning；两仓 clean/published 后重新完成 GPU 双空闲门禁，再运行正式
  driver-injected candidate preflight。
- **2.60 发布状态：** candidate-only 配置链路、测试、报告与 planning 已由
  `8b347e1` 推送；两仓 clean/published。下一步发布本条状态，再开始 preflight
  的新双空闲门禁。
- **candidate preflight 双空闲：** `21:25:00Z/21:26:18Z` 两次有效检查
  间隔 78 秒，8/8 GPU 均为空闲；异步空输出不计证据。下一步发布本条状态，
  再运行 driver-injected preflight。
- **candidate preflight 进行中：** run
  `20260731T2127Z_candidate_topk_sort_preflight_v1` 仅静态 verifier passed；新增
  配置映射与实际环境变量两项检查通过。fixed import 尚未完成、无整轮退出码；
  已纠正过早完成判断。继续监控并按 10 分钟规则打印进度，正式 32K 保持阻塞。
- **candidate preflight 最终结果：** `21:27:10Z–21:29:09Z`、exit=0；静态
  66/66 passed，fixed import/parsed args 均 CUDA=false，新增两项环境检查通过。
  容器已删除、8 卡全空闲。下一步封存证据、全文复读并更新 2.61；发布完成前
  不启动正式 32K。
- **2.61 报告门禁：** 修改前 4,210 行、SHA256 `4bb51a49…04e6`；追加后
  4,275 行、SHA256 `aa9b8213…dae7`。章节连续，66/66、两处 CUDA=false、
  14/14 manifest、引用、术语和 diff 检查通过。
- **当前下一步：** 只提交推送正式排序结果、2.62 与 planning；恢复
  clean/published 后，在不分配 GPU 的前提下对排序/未排序正式 trace 做阶段
  归因。归因结果先实时更新报告，再决定下一项 GPU 优化实验。
- **正式 32K 双空闲门禁：** 2.61 由 `139d525` 发布；
  `21:32:45Z/21:34:05Z` 两次有效检查间隔 80 秒，8/8 GPU 全空闲、两仓
  clean/published。下一步发布本条状态，再启动 3 rounds + 8+8+1 profiler。
- **排序候选正式 32K 结果：** run
  `20260731T2135Z_candidate_topk_sort_32k_b1_v1` exit=0/passed；三轮中位
  TTFT/TPOT/吞吐 `32449.245/199.155 ms/0.0173226 req/s`。相对未排序 ca4a
  `-0.715%/-0.441%/+0.579%`，相对 BF16 `+159.013%/+11.365%/-38.955%`；
  TPOT 过门限，TTFT 仍失败。profiler 8+8+1 passed。下一步封存证据、全文
  复读并实时新增 2.62；此前不做 trace 归因或下一优化。
- **正式证据封存：** 已筛选封存 52 项小型证据，连同
  `evidence_manifest.sha256` 共 53 个文件、`1,156,489 bytes`；manifest 与
  `formal_validation.json` SHA256 分别为 `770166b5…a198`、
  `467dfa29…82520`。报告复读前的首次复核误用了不存在的
  `manifest.sha256`，且宿主无 `jq`；两项均为只读校验命令错误，未修改证据。
  第二次又错误进入证据目录校验了以仓库根为基准的 manifest，并误用不存在的
  `/dev/shm/.../comparison.json`；仍未修改证据。下一步固定从仓库根运行
  `sha256sum -c`，并读取封存的带 run ID comparison，再新增 2.62。
- **2.62 初次报告门禁：** 交叉引用正则误把 token throughput `2.217…`
  当成 `2.217…` 节，因而 fail-closed；报告内容没有该章节引用。下一轮把
  引用编号限定为一到两位小节号后，仍把历史性能倍数 `2.89×/2.85×` 误判为
  2.89/2.85 节。下一轮再排除紧随 `×` 的倍数，只审计实际章节引用，并重跑
  章节、术语、数值、引用和 diff 门禁。
- 排除倍数后的引用范围门禁已证明最大编号为 2.62；新增节引用集合断言漏列
  新节标题 2.62 与正文发布来源 2.61，导致第三次校验器误判。补记本错误的
  首个 planning patch 又因上下文取自错误文件被完整拒绝、无部分修改。最终
  预期集合修正为 `{2.52, 2.59, 2.61, 2.62}` 后重跑完整门禁。
- **2.62 最终报告门禁：** 已通过。报告 4,357 行、SHA256
  `946593c539d9c6523b8d28cb3f595a96030f3b3d8da3faa3eac9283adcffb10b`；
  1.1–1.5/2.1–2.62 连续，新节引用均指向已存在章节，`三池=0`、大写
  `A800` 仅历史文件链接；正式数值、证据 hash、术语和 `git diff --check`
  全部通过。下一步只提交并推送本阶段，不先做 trace 归因。
- **2.62 发布状态：** 报告/planning 已由主仓库提交
  `d38dfde4ea5d1a4af1d575cae48fa56e38528d7d` 推送；本地与
  `origin/feat/glm52-model-load` 一致。下一步只发布本条状态，之后再开始
  CPU-only trace 归因。
- **排序/未排序 trace 归因计划：** 复用已验收的固定控制镜像、runc、
  network none、4 CPUs 与 `/dev/shm/oscar-glm-stage9-ca4a404e9-formal/analysis/
  venv-python3.12.13-ijson3.4.0.post0`；先核对 8 个排序 trace 与 summary 身份，
  再用 `analyze_prefill_trace.py` 流式分析。成功后只读对比 ca4a404e9 未排序
  summary，封存小型证据并先更新报告 2.63；不注入 NVIDIA runtime、不分配 GPU。
- trace 归因首个组合预检命令返回 exit=0 但输出为空，无法证明镜像、trace 或
  Python/ijson 身份，因此判为无效检查且未启动分析。下一步拆分为独立、带显式
  输出的 fail-fast 检查，不复用空输出结论。
- 拆分检查确认镜像 ID 与 analyzer SHA 正确，目录含 8 worker traces 加 1
  frontend trace；但又直接调用了宿主上的容器 venv，因其解释器只存在于控制
  镜像而报 `No such file or directory`。这是已知限制，未读取 trace；下一步
  仅在固定断网控制容器内调用该 venv，不再从宿主执行。
- 固定断网控制容器内调用旧 venv 又报 `cannot execute binary file`，证明该
  持久环境当前不可复用，分析仍未启动。下一步只读检查 venv/uv 解释器布局；
  如旧解释器失效，则用固定控制容器和已有任务专用 uv cache 新建独立 venv，
  仍锁定 Python 3.12.13/ijson 3.4.0.post0。
- 旧 venv 已确认为指向 `/usr/bin/python3.12` 的断链。探测固定 Python 时又
  未考虑镜像既有 entrypoint，把 `/bin/bash` 作为附加命令传入，报
  `cannot execute binary file`；未创建环境。下一步读取镜像 Entrypoint/Cmd，
  按真实入口调用，不再嵌套 bash。
- 镜像入口已确认是 `/bin/bash`，正确调用验证固定 Python 3.12.13/uv 0.11.5。
  新 v2 venv 创建成功，但随后又尝试了历史已知不可行的 offline cache 解析，
  `ijson==3.4.0.post0` 不可用而退出；这是重复的已知失败，trace 仍未读取。
  保留空 venv，不删除；下一步按既有成功路径用清华 PyPI 在线锁定安装，再以
  独立 network-none 容器验证。
- v2 已经清华源安装并在独立断网容器内精确验证 Python
  3.12.13/ijson 3.4.0.post0。组合末尾的 `docker ps` 模板误用
  `.HostConfig.DeviceRequests` 而报字段不存在；不影响前两步验收。下一步用正确
  只读容器检查后启动固定 4 CPU 流式分析。
- CPU-only 分析已启动。首次轮询误把外层 functions cell 当成 unified exec
  session，并传入无效 i32 session id；调用在参数解析前失败，分析未受影响。
  外层 cell 随后返回真实 inner session `91973`，后续仅用该 session 轮询。
- 8/8 trace 分析 exit=0。首个 comparison 生成脚本在宿主 Python 3.8 因
  `zip(strict=True)` 不受支持而在写文件前退出；summary 未变。下一轮移除该
  3.10+ 语法并显式断言两边 trace 长度均为 8，再生成 comparison/validation。
- **排序 trace 归因结果：** 有效 analysis
  `20260731T2228Z_topk_sort_32k_prefill_trace_v1` exit=0，8/8 ranks 均
  144 contexts、16 chunks、32,768 tokens。相对未排序，prefill wall/kernel
  中位数下降 `277.624/274.683 ms`；stage1 下降 `278.749 ms`，top-k 增加
  `30.015 ms`，两者合计净省 `248.734 ms`、解释 wall 改善 `89.594%`。
  8/8 rank 的 wall/stage1 均下降，top-k 均上升。下一步封存小型证据并先更新
  报告 2.63；不进入下一优化。
- **排序 trace 证据：** 已封存 10 项加 manifest，共 11 文件、371,268 bytes；
  10/10 manifest 复算通过，manifest SHA256 为 `37840473…3aff`。下一步全文
  复读当前报告并新增 2.63；报告发布前不做下一优化决策。
- **2.63 报告门禁：** 已通过。报告 4,435 行、SHA256
  `f851d6da63579ca8db960672c0f39fbacf76a80d4689ebf0451f6958a9c363e2`；
  1.1–1.5/2.1–2.63 连续，新节引用 2.53/2.62/2.63 均有效，术语、归因数值、
  证据 hash 与 diff 全绿。下一步只提交推送本阶段；发布前不检查下一源码候选。
- **2.63 发布状态：** 报告/planning 已由主仓库提交
  `2df5f5727f7eeb0335451166c29103cb0f9cca74` 推送；主仓库与源码仓库均
  clean/published。下一步只读检查 stage1 的 16-chunk 工作分布与源码路径；
  形成候选后先更新 planning，不直接分配 GPU。
- **16-chunk 只读初筛：** 排序 trace 的 chunk wall 中位走势从首块约
  `1.340 s` 增至末块约 `2.196 s`；stage1 当前按整段聚合，无法证明逐块增长
  来源。源码确认 stage1 仍用 16-token tile、32-head block、8 warps，并已具备
  causal loop 与空 BF16 tile gate。下一步以 TDD 给既有 trace analyzer 增加
  每 chunk kernel calls/total_ms，不改运行时源码、不使用 GPU。
- TDD 红灯为 1 error/1 pass，精确缺少 `chunks[*].kernel_total_ms`。首次实现
  patch 在 chunks 字典中多留一个提前闭合大括号，静态查看时发现、尚未运行
  绿灯；下一步精确修正后先 compile，再运行同一测试。
- 语法修正后 compile passed、同一测试 2/2 passed、diff check passed；format
  version 从 2 升至 3，每个 chunk 新增 kernel_total_ms 与 kernels calls/total_ms。
  下一步运行 Ruff/format 和 Phase 9 CPU-only 工具回归，再重跑两组 trace。
- 广回归首个依赖探测发现固定控制镜像 Python 没有 pytest，PATH 也无 Ruff，
  因 `No module named pytest` 在测试前退出。下一步只读定位既有 pre-commit/uv
  工具环境；不修改项目依赖，不把未运行测试记作通过。
- 已定位宿主固定 Ruff 0.14.0，Ruff check 通过；format check 发现 analyzer
  新增 comprehension 需要格式化，组合 fail-fast 停止，四个广回归 unittest
  尚未运行。下一步仅机械格式化两个改动文件并审查 diff，再重跑组合。
- formatter 仅重排新增块及其相邻长条件；最终 Ruff check/format、compile、
  diff 与四个 unittest 文件合计 33/33 全部通过。analyzer/test SHA256 为
  `724aeb5e…bf43`/`f57b985a…a48c`。下一步先全文复读并实时新增报告 2.64；
  发布工具阶段前不重跑正式 trace。
- **2.64 报告门禁：** 已通过。报告 4,489 行、SHA256
  `c4151cd62308e29041de3040a524fb3ca38a1813fb84c146733aadde5bd0ac5a`；
  1.1–1.5/2.1–2.64 连续，新节引用 2.53/2.63/2.64 有效，术语、TDD、33/33、
  文件 hash 和 diff 全绿。下一步只提交推送工具阶段；发布前不跑 trace。
- **2.64 发布状态：** analyzer v3、测试、报告与 planning 已由
  `1ae08c23ffe771999462796c32e407574fbb0e87` 推送，主仓库本地/远端一致。
  下一步发布本条状态，再固定 4 CPUs、断网顺序重跑排序/未排序 trace。
- **逐 chunk trace 分析完成：** 排序与未排序两组 format v3 分析均已
  8/8 ranks、16/16 chunks 通过一致性门禁；analysis ID 分别为
  `20260731T2245Z_topk_sort_32k_chunk_trace_v1` 与
  `20260731T2247Z_ca4a404e9_32k_chunk_trace_v1`，summary SHA256 分别为
  `a3c58c84073b7801ae8a2f439666b7ddbf6a4d8411fc0cb620f75e82ef67f0e7`、
  `b90f54cfae34cea7e4f98a5d5de9fbd3d802bf6658c8a9ef7196ed023177ab06`。
  每 rank 的逐块 kernel 累计与整段聚合在 `1e-6 ms` 内一致，stage1/top-k
  调用数分别为 1,248/1,344。首块 stage1 几乎不变；第 2–16 块排序后
  stage1 每块下降约 `14.102–23.323 ms`，top-k 每块增加约
  `1.306–3.091 ms`。当前只形成“纯 BF16 tile 可能允许跳过 history 两次
  dot”的候选假设，尚未证明覆盖率，也未修改运行时源码或分配 GPU。
- **当前阶段：** 先用 CPU/TDD 扩展既有 tile 覆盖率统计，实测排序后
  `all_bf16/no_history` tile 的数量与比例；完成后先全文复读并实时更新
  `OSCAR精度与性能优化记录.md`，再决定是否实现对称 `has_history` gate。
- **coverage TDD 红灯：** 固定控制镜像 CPU-only 定向测试按预期以
  `KeyError: 'active_tiles'` 失败（0 passed/1 error），证明当前 helper 尚未
  提供 history 机会口径。下一步只实现上述六个统计字段，再跑同一绿灯与广回归。
- 首轮绿灯组合在 `py_compile` 写只读 bind mount 的 `__pycache__` 时以
  `Errno 30` 退出，测试尚未运行；实现文件未被容器修改。下一轮设置任务专用
  `PYTHONPYCACHEPREFIX=/tmp/...` 后重跑，不改挂载权限或仓库缓存。
- 设置任务专用 pycache 后 compile 与定向绿灯 1/1 passed。宿主默认 `PATH`
  没有 Ruff（`command not found`）；下一步复用此前已验证的固定 Ruff 0.14.0
  绝对路径，并运行四个 Phase 9 unittest 文件，不安装新项目依赖。
- **coverage 工具门禁通过：** Ruff 0.14.0 check/format、compile、定向测试
  1/1 及四文件广回归 34/34 均通过。下一步先重新顺序读取当前中文记录，追加
  本工具阶段及尚未实测收益的边界并完成章节/引用校验；报告发布前不运行 16-chunk
  coverage，也不改运行时源码。
- **2.65 报告门禁通过：** 修改前已顺序扫描全部 4,489 行，读取前后 SHA256
  均为 `c4151cd62308e29041de3040a524fb3ca38a1813fb84c146733aadde5bd0ac5a`，
  与 2.64 发布值一致。追加后为 4,572 行、SHA256
  `7ae0c4cc3a567d1e856420b7ce9ee088ca65fe3635afcfb3621c953518e5952f`；
  1.1–1.5/2.1–2.65 连续，新增引用、术语、analysis 身份、逐块数据、测试数与
  文件 hash 全部通过。下一步只提交推送工具/报告/planning；发布前不跑 coverage。
- **2.65 发布状态：** coverage 工具、测试、报告与 planning 已由主仓库提交
  `d80d4fe08381f32b4034195667c538062b358d5a` 推送。下一步再次确认主/源码仓库
  clean/published，再启动固定 CPU-only 16-chunk coverage；不分配 GPU。
- **16-chunk coverage 已完成：** analysis
  `20260731T2301Z_topk_sort_history_coverage_cpu_v1` exit=0，16/16 chunks，
  实测 `828.4 s`，并在 600 秒打印 `13/16` 定时进度。原始 aggregate 错把
  每块常量 `tile_width=16` 求和为 256；逐块 rows 与目标计数未受影响，但该
  raw summary 不能直接作为正式证据。下一步只从已落盘 rows 重建排除常量的
  aggregate 并做分区/2.54 对账校验，不重跑计算、不改源码。
- **coverage 重建校验通过：** validated summary 从 raw 的 16 个 unchanged rows
  重建；active/history/BF16 分区各 32/32、sorted set 16/16、首块 shortcut、
  2.54 六项对账全部 passed。排序后 no-history tile 为 `13,481/4,064,256`
  （`0.3316966254%`），未排序为 `199`（`0.0048963451%`），增量仅 13,282。
  该覆盖规模不足以支持立即实现 `has_history`；下一步封存小型证据，并先全文
  复读、实时新增报告 2.66，发布前不进入运行时改动。
- **coverage 证据封存：** 小型目录含 5 项数据/脚本加 manifest，共 6 文件、
  59,291 bytes；5/5 manifest 复算通过，manifest SHA256 为
  `e8a134219161667e59f0b5f4f5157d7545dca8193e0322cc1a89fedd7a57c5d4`。
  下一步全文复读当前 2.65 报告并新增 2.66；此前不检查下一候选。
- 2.66 首轮草稿沿用了旧环境印象，把 PyTorch 写成 `2.7.1+cu126`；证据字段
  复核显示实际为 `2.11.0+cu129`，并在发布前修正，同时把耗时改为 summary
  精确值 `828.5257903169841 s`。下一步重跑完整报告门禁。
- **2.66 报告门禁通过：** 修改前顺序扫描全部 4,572 行，读取前后 SHA256
  均为 `7ae0c4cc3a567d1e856420b7ce9ee088ca65fe3635afcfb3621c953518e5952f`；
  追加并修正环境后为 4,661 行、SHA256
  `65562c9a90a652ef77092fe8d85f8a4c3e48853661424ca178d71b381c71ba61`。
  1.1–1.5/2.1–2.66、证据数据、术语和 diff 全部通过。下一步只提交推送本阶段。
- **2.66 发布状态：** coverage 结果、候选淘汰结论、报告与 planning 已由
  主仓库提交 `21d60b75295a6439606cad65a436b0a3470531dc` 推送；主仓库 clean/published。
  下一步只读分析 v3 逐 chunk 的其他 kernel 覆盖面，选择下一最小候选；不直接
  修改源码或分配 GPU。
- **下一只读候选：** 排序 trace 中 stage1 之外最大的稳定项为
  `_rotate_latent_kernel`，逐块 8-rank 中位数跨 16 块求和约 3,391.069 ms。
  下一步核对其调用点、调用数和张量生命周期，只有证明存在语义等价复用后才
  进入 CPU/TDD 候选；当前仍不修改源码。
- 调用链已排除跨层/跨 chunk 缓存：rotate 是每层 current-history 新行的
  FP32/IEEE matmul，caller 已复用 scratch。主仓库没有专用 rotation benchmark；
  下一步以 TDD 新增独立 IEEE-vs-TF32 工具，复用真实 rotation artifact、
  2,048×512 几何和既有 0.35/2% correctness 边界。工具阶段先更新报告并发布，
  此前不修改 production kernel、不申请 GPU。
- **rotation 工具 TDD 红灯：** 固定 ca4a 控制镜像、4 CPUs、network none
  中定向 unittest 在导入缺失的 `benchmark_oscar_rotation.py` 时以
  `FileNotFoundError` 失败，测试未执行，符合先写门禁的预期。下一步实现测试
  要求的最小独立工具，再跑同一 7 项绿灯。
- **rotation 定向绿灯：** 最小工具已实现；固定控制镜像 CPU-only compile 和
  7/7 unittest passed。工具用本地同参数 IEEE kernel 对齐生产输出，再比较
  TF32，并对四个真实 rotation 层执行 rotation 与 INT2-restored allclose；
  timing 隔离 2,048×512 单 kernel。下一步做 Ruff/format、源码契约审查和
  Phase 9 广回归，尚未申请 GPU。
- 首轮 Ruff check 与 `git diff --check` 通过，但 Ruff format check 要求机械
  格式化新工具和测试后退出；广回归尚未运行。下一步仅格式化这两个新文件并
  审查格式 diff，再执行完整组合。
- **rotation 工具 CPU 门禁通过：** 机械格式化后 Ruff 0.14.0 check/format、
  compile、生产 IEEE 静态契约与五个 Phase 9 unittest 文件合计 42/42 全绿，
  `git diff --check` passed。下一步计算最终文件身份，重新全文读取并实时新增
  报告 2.67；发布前不做 GPU 空闲检查。
- **2.67 报告门禁通过：** 修改前已顺序扫描全部 4,661 行，读取前后 SHA256
  均为 `65562c9a90a652ef77092fe8d85f8a4c3e48853661424ca178d71b381c71ba61`；
  追加后为 4,729 行、SHA256
  `ecc8e9558ac342256a9e6807c4707d76a4fb1d9d05f578f9b5cce04c4d24ed64`。
  1.1–1.5/2.1–2.67、引用、术语、TDD/42/42、文件 hash 和 diff 全绿。
  下一步只提交推送工具阶段，发布前不做 GPU 检查。
- 补记 findings 时一度误拼前一版报告 SHA256 后半段；提交前复核已修正，正文
  与 task_plan 的实际 hash 始终正确。下一步最终复核并提交。
- **2.67 发布状态：** rotation 工具、测试、报告与 planning 已由主仓库提交
  `25077d00bee732251c96f3b63a005016357aff66` 推送。下一步先发布本条状态，再
  执行授权 8 卡的双空闲检查；筛选固定只使用一张空闲 GPU。
- **rotation GPU 双空闲门禁通过：** `23:32:39Z/23:33:46Z` 间隔 67 秒，
  两次 8/8 卡均 0 MiB/0%、无 compute process；外部下载容器 DeviceRequests
  为 null。下一步发布本条状态后，启动前即时复查 GPU 0，并固定单卡运行筛选。
- **rotation TF32 筛选精度失败：** run
  `20260731T2334Z_rotation_tf32_screen_v1` 固定 GPU 0，启动前 0 MiB/0%。
  benchmark-local IEEE 契约与 TF32 rotation 检查先通过，但某层 INT2-restored
  门禁出现 128 个超限值、max abs `1.6203639507293701`，超过 0.35/2%，
  exit=1；异常未携带 layer，不能断言是四层中的哪一层。未进入 timing、未写
  result JSON。容器已删除，`23:35:00Z` 8 卡全空闲。下一步封存失败证据并
  先实时新增报告 2.68，不修改生产源码。
- 首次纠正层号边界的 planning patch 漏写文件切换标记，因上下文不匹配被完整
  拒绝、未修改任何文件；随后按三个文件的真实上下文分别修正。
- **rotation 失败证据封存：** exit/failure/identity/stderr 共 4 项，加 manifest
  共 5 文件、3,126 bytes；4/4 复算通过，manifest SHA256 为
  `f558b23603692be50dccca13bc83fe9aa257500c3faf969776e06dd3ff203712`。
  下一步全文复读当前报告并新增 2.68；发布前不修改工具或进入下一实验。
- **2.68 报告门禁通过：** 修改前顺序扫描全部 4,729 行，读取前后 SHA256
  均为 `ecc8e9558ac342256a9e6807c4707d76a4fb1d9d05f578f9b5cce04c4d24ed64`；
  追加后为 4,790 行、SHA256
  `d5746f3352dad6b47122a780020875134565abec79af7542282a570d9fcc1ba3`。
  1.1–1.5/2.1–2.68、层号未知/未计时边界、证据 hash、术语和 diff 全绿。
  下一步只提交推送 2.68/planning。
- **2.68 发布状态：** TF32 精度淘汰结果、报告与 planning 已由主仓库提交
  `603e098731b2a60c23a082b65d1d13b1043998a3` 推送。下一步只扩展主仓库工具，
  筛选保持 IEEE 与 block K=32 不变的 block M/N/warps；不改 production 源码。
- **IEEE sweep TDD 红灯：** rotation 测试现为 7 passed/3 errors；精确缺少
  `mode`、`build_ieee_sweep_configs` 与 `select_best_ieee_config`，证明现有工具
  尚不能运行 IEEE 参数矩阵。下一步只实现这三项及 main 的 ieee-sweep 分支。
- **IEEE sweep 定向绿灯：** compile 与 rotation unittest 10/10 passed。工具
  保留原 TF32 模式，并新增六配置 IEEE sweep；每个配置先在四层要求与 production
  bitwise equal，再计时，candidate 编译失败会单独记录而不掩盖 baseline。
  下一步 Ruff/format 与五文件广回归，尚未使用 GPU。
- IEEE sweep 首轮 Ruff check/diff passed，但 format check 要求机械格式化工具
  后停止；广回归尚未运行。下一步仅格式化该文件并重跑完整组合。
- **IEEE sweep 工具门禁通过：** formatter 后 Ruff check/format、compile、diff
  与五文件 unittest 44/44 全绿。下一步计算文件身份，全文复读并实时新增
  报告 2.69；发布前不做新一轮 GPU 空闲检查。
- **2.69 报告门禁通过：** 修改前顺序扫描 4,790 行且 SHA256 前后为
  `d5746f3352dad6b47122a780020875134565abec79af7542282a570d9fcc1ba3`；
  修改后 4,850 行、SHA256
  `fde8196fa41dbd3872359316a0315e40da0d9dfa0b2317824da4015312aa5e5e`。
  1.1–1.5/2.1–2.69、六配置、bitwise 边界、44/44、hash 和术语全绿。
  下一步只提交推送本阶段。
- **2.69 发布状态：** IEEE sweep 工具、测试、报告与 planning 已由提交
  `ad7595fbb3772f389940a50c87c9b07028f015a4` 推送。下一步发布本条后重新执行
  8 卡双空闲门禁；正式 sweep 固定单卡。
- **IEEE sweep 双空闲门禁通过：** `23:44:19Z/23:45:28Z` 间隔 69 秒，
  两次 8 卡均 0 MiB/0%、无 compute process，外部容器 DeviceRequests=null。
  下一步发布状态，启动前即时复查 GPU 0 后运行固定单卡 sweep。
- **IEEE sweep GPU 门禁通过：** run
  `20260731T2346Z_rotation_ieee_sweep_v1` 在固定 GPU 0、固定 ca4a 控制镜像中
  exit=0；六配置在真实 rotation 层 0/25/51/77、每层 1,048,576 个值上均与
  production bitwise 一致。`m32_n64_w4` 最快，CUDA 中位数由
  `0.1016319990158081 ms` 降至 `0.09359359741210938 ms`，改善
  `7.909321553783877%`。退出后 8/8 GPU 均为 0 MiB/0%、无 compute process。
- **当前阶段：** IEEE-only 参数 sweep 已完成；开始固定单卡候选的真实几何
  覆盖审查。现有 2,048-row 微基准只覆盖 current-history 典型几何，而
  `_rotate_latent_kernel` 也由 query rotation 共用，不能把 7.909% 直接外推为
  32K 端到端收益。下一步先封存结果、全文复读并新增报告 2.70，再以 CPU/trace
  核对调用数和 2,048/16,384-row 几何；此前不修改 production kernel。
- **IEEE sweep 证据封存：** 结果、退出码、运行身份和退出后 GPU 状态共 4 项，
  加 manifest 共 5 文件、18,988 bytes；4/4 复算通过，manifest SHA256
  `295d3f64d1ada2fdca902963569b7b06aa08cdf3a159d3a0e3cf939c9d62e7fb`。
  下一步全文复读当前报告并追加 2.70。
- **2.70 修改前门禁：** 已顺序读取报告全部 4,850 行/262,525 bytes，读取前后
  SHA256 均为 `fde8196fa41dbd3872359316a0315e40da0d9dfa0b2317824da4015312aa5e5e`，
  未发现并发手改；1.1–1.5/2.1–2.69 连续，无交叉引用，`三池=0`，大写
  `A800` 仅出现在第 5 行历史文件链接。下一步追加 2.70 并做完整报告门禁。
- **2.70 报告门禁通过：** 报告现为 4,924 行/266,960 bytes，SHA256
  `4f155caebc6e452b10935b9390c49e8fb51ce8515d71014d98ca903e1dd0d0b0`；
  1.1–1.5/2.1–2.70 连续，无交叉引用，六配置、best、原始 result SHA、证据
  manifest、术语和 `git diff --check` 全绿。下一步只提交推送本阶段；发布完成前
  不进入真实几何分析。
- **2.70 发布状态：** IEEE sweep GPU 结果、报告与 planning 已由主仓库提交
  `c06d45486fa298b85c1a31dad0e4f113bcc174b4` 推送；主仓库和源码仓库均
  clean/published。下一步发布本条状态，再开始 CPU/trace 的真实几何分析。
- **真实调用覆盖初查：** 排序 trace 每 rank 的 `_rotate_latent_kernel` 总调用数
  为 4,898；rank 0 的 chunk1 为 233，而 `233 + 15×311 = 4,898`，说明每个后续
  chunk 有 311 次，不是此前按 78 层隐含假设的单次/层。源码还显示 query 入口
  rotation 与 attention 输出的 inverse rotation 都复用同名 kernel。下一步精确
  汇总 8 ranks×16 chunks，并核对 backend 的 current-history/recent demotion 路径。
- **调用数精确复核：** sorted/unsorted 的 8/8 ranks 都严格为 chunk1=233、
  chunk2–16=311、每 rank 合计 4,898；sorted 每 rank总耗时
  `3387.745537–3399.351272 ms`，unsorted 为 `3387.972586–3394.243401 ms`。
  后续 chunk 的两类 store 路径由 current-history 和 recent demotion 各贡献一次
  rotation，decode 则有 query 正向与 history 逆向 rotation。下一步直接读取 raw
  trace kernel launch/grid 元数据，避免只靠调用算术推断 num_rows。
- **模型几何身份：** 正式模型 `config.json` 实际为 78 layers、64 attention
  heads、KV LoRA rank 512；固定 TP=8，故源码推导每 rank `num_heads=8`，2,048
  query rows 会展平为 16,384×512。下一步仍用 raw trace 元数据验证 launch grid。
  系统 Python 缺 `ijson`；改用先前 analysis 生成且 summary 标识为
  Python 3.12.13/ijson 3.4.0.post0 的已有环境，不重新安装依赖。
- **analysis venv 限制：** 两个既有 venv 的 `bin/python` 都是指向宿主不存在
  `/usr/bin/python3.12` 的 broken symlink，不能直接执行。下一步复用固定 artifact
  rootfs 的 Python 3.12.13，并只把既有 venv site-packages 作为 `PYTHONPATH`；若
  import 通过再流式读 trace，不联网安装。
- **解释器复用修正：** artifact rootfs Python 在宿主因 GLIBC 版本不足不能执行；
  首次直接作为 fixed image command 又受镜像 Entrypoint 语义影响，报
  `cannot execute binary file`。下一步显式覆盖 `--entrypoint /bin/bash`，先在
  容器内核对 Python 身份和 ijson import，再解析，不重复上述两种启动方式。
- **流式环境已恢复：** 显式 entrypoint 后固定镜像内实际为 Python 3.12.13、
  ijson 3.4.0.post0/yajl2_c。首轮 raw trace 解析已找到 rotation 事件，但输出
  `json.dumps` 因 ijson 数值类型 `Decimal` 不可序列化而退出；下一轮只对输出使用
  `default=str`，不改变解析或 trace，不重复原打印方式。
- **raw launch 直接证据：** rank0 首批事件出现 grid 864 与 8,192；按 production
  `grid=ceil(rows/16)×8`，分别精确对应 1,728-row current-history 与 16,384-row
  query/inverse。16,384-row 又分成 reg48/shared11264/duration约572 us 的正向矩阵
  变体，以及 reg64/shared10240/duration约2,075 us 的 `rotation.T` 变体；后者是
  当前主要 rotation 成本。下一步完整聚合 rank0 的 grid/signature 数量与时间。
- **rank0 signature 归因完成：** 与 v3 summary 对齐的 16 个内层 span 合计
  4,898 calls/3,390.564987 ms。256-row demotion 占 28.359872 ms/0.836435%，
  1,728-row 首块 store 占 5.213370 ms/0.153761%，1,792-row 后续 store 占
  85.427658 ms/2.519570%，16,384-row 正向占 714.301325 ms/21.067324%，
  16,384-row `rotation.T` 逆向占 2,557.262762 ms/75.422910%。下一步围绕真实
  16,384-row 正/逆向设计 CPU/TDD benchmark 扩展，不再把 2,048 当主几何。
- **trace-layout 工具设计冻结：** 新 mode 只扩展主仓库 benchmark，不改源码
  submodule。固定 `--rows 16384` 时比较 forward 的 m16/m32，以及 inverse 的
  production strided-T m16、contiguous-T m16/m32；四个真实层全部要求相对
  production `atol=rtol=0` 后才计时。工具同时报告 78×512×512×FP32 contiguous
  inverse 的 81,788,928 bytes（78 MiB/GPU）静态开销。下一步先写纯 CPU 测试并
  得到红灯，再最小实现。
- **trace-layout TDD 红灯：** 固定 ca4a 控制镜像、4 CPUs、network none 中
  13 项测试为 10 passed/3 errors；精确缺失 CLI mode、case builder 与 storage
  helper。既有 10 项保持通过，负向 argparse stderr 仍是预期输出。下一步只实现
  测试要求的三项和最小运行分支，再跑同一测试，不改 production。
- **trace-layout 最小实现：** 已增加 mode、五 case、78 MiB helper、四层 bitwise
  accuracy、按 direction baseline 的计时比较与 JSON 分支；未改 production。
  首轮 compile+test 因仓库只读挂载下 py_compile 写 `scripts/phase9/__pycache__`
  得到 `Errno 30`，测试未执行。下一轮设置任务专用 `PYTHONPYCACHEPREFIX=/tmp/`
  后运行相同门禁，不重复只设 `PYTHONDONTWRITEBYTECODE` 的失败方式。
- **trace-layout 定向绿灯：** 任务专用 `/tmp` pycache 后 compile 与 13/13
  unittest passed；负向 argparse stderr 为预期。diff 审查确认只改主仓库工具、
  测试与 planning，源码 submodule 未动。下一步运行固定 Ruff 0.14.0
  check/format 和 Phase 9 五文件广回归；尚未申请 GPU。
- **trace-layout 静态门禁：** Ruff 0.14.0 check/format 与 `git diff --check`
  全绿，两文件无需 formatter 改写。五文件范围已核对为 analyzer、OSCAR prefill、
  top-k sort、rotation 与 Phase 9 tools；下一步在固定 CPU-only 控制镜像运行组合。
- **trace-layout CPU 门禁完成：** 固定控制镜像 compile 与五文件 unittest
  `47/47 passed`；最终 Ruff check/format、diff、五 case、16,384 rows gate、
  bitwise 和78层storage静态契约全绿。工具/测试为1,078/229行，SHA256
  `2fab0327…bb36`/`012ad5be…3a0`；源码 submodule clean/published 且未改。
  下一步全文复读并实时追加报告 2.71；发布前不检查或分配 GPU。
- **2.71 报告门禁通过：** 修改前顺序读取全部4,924行/266,960 bytes，读取前后
  SHA256 均为 `4f155caebc6e452b10935b9390c49e8fb51ce8515d71014d98ca903e1dd0d0b0`，
  无并发手改。修改后为5,022行/273,297 bytes，SHA256
  `a6668f26a0a52c8b643a0523e2dafd1253b5cc39f70afb3d1bf1ccf1ed7252e3`；
  1.1–1.5/2.1–2.71、调用/耗时、五case、47/47、hash、术语与 diff 全绿。
  下一步只提交推送本阶段，发布前不执行 GPU 检查。
- **2.71 发布状态：** trace-layout工具、测试、报告与planning已由主仓库提交
  `6a7d9f3d96568391ca6bd9a55401d0c1fb8ff8ca` 推送；主/源码仓库均
  clean/published。下一步发布本状态后重新执行8卡双空闲门禁，正式筛选固定单卡。
- **trace-layout GPU 双空闲门禁：** `00:10:42Z/00:11:50Z` 间隔68秒，两次
  8/8卡均0 MiB/0%、无compute process；外部下载容器DeviceRequests=null。
  下一步发布本状态，随后启动前即时复查GPU 0并固定单卡运行五case筛选。
- **启动路径核对受限：** 两次从庞大 artifacts 树全量枚举 rotation 文件的只读
  命令均在约10秒无输出，未定位路径、未启动GPU。下一步不再全树扫描，改从已
  发布脚本/配置/planning中的固定 bind path 精确解析，再做即时GPU 0复查。
- **启动输入已固定：** 控制镜像已内置 `/opt/oscar_artifacts/rotation_fit_v2`
  且 rotations SHA256=`256ee5e4…235d`，与2.70实际结果一致，不挂载宿主旧版
  `0a966da2…08e` artifact。一次 `rg Dockerfile*` 因无匹配文件报错，已改用精确
  文本范围。下一步先发布本补记恢复 clean，再即时复查GPU 0并启动。
- **trace-layout GPU 筛选通过：** run
  `20260801T0017Z_rotation_trace_layout_v1` 固定GPU 0 exit=0，5/5 cases在四层、
  每层8,388,608值上全部bitwise一致。forward m16→m32为
  `0.7081984043→0.5045760155 ms`（-28.752167%）；inverse strided m16→
  contiguous m16为`2.1363199234→0.5737984180 ms`（-73.140801%），再到
  contiguous m32为`0.5047296047 ms`（-76.373876%，4.232603×）。退出首采样
  GPU0为0 MiB/14%瞬时利用率，9秒后8/8卡均0 MiB/0%、无compute process。
  下一步封存结果并全文更新2.72；此前不改production。
- **trace-layout 证据封存：** 12项原始数据加manifest，共13文件/35,067 bytes，
  12/12复算通过；result/manifest SHA256为`a54502cf…db66`/`5af7db4f…94ea`。
  forward m16样本后两项从约0.708降至0.6767/0.5740 ms，存在顺序/频率漂移，
  因而28.752%只作为本轮顺序测量值；inverse三组样本稳定，73.141%布局收益仍
  远大于漂移。下一步全文复读并如实写2.72，不夸大forward结论。
- **2.72 修改前门禁：** 已顺序读取当前报告全部5,022行/273,297 bytes，读取
  前后SHA256均为`a6668f26a0a52c8b643a0523e2dafd1253b5cc39f70afb3d1bf1ccf1ed7252e3`，
  无并发手改；1.1–1.5/2.1–2.71、引用和术语均正常。按inverse m16实测比率
  0.2685919893仅作trace投影，可把rank0 inverse `2557.262762→686.860292 ms`，
  rotation总量`3390.564987→1520.162517 ms`（-55.164920%）；这不是端到端实测。
  下一步追加2.72并明确投影边界。
- **2.72 报告门禁通过：** 报告现为5,115行/278,942 bytes，SHA256
  `41db2e05148b03ae3d325901360cf2047b9feb633abff3265e07b7142a7458ce`；
  1.1–1.5/2.1–2.72、五case/四层bitwise、样本漂移、投影边界、证据、术语和
  `git diff --check`全绿。下一步只提交推送2.72/planning；发布前不修改源码。
- **2.72 发布状态：** trace-layout GPU结果、报告与planning已由主仓库提交
  `971f0c4`推送，主仓库clean/published；源码仍为`ca4a404e9`且clean/published。
  下一阶段只推进最小contiguous inverse候选，保持block M=16；先写源码CPU/TDD
  门禁，不直接申请GPU。
- **production最小调用链设计：** 每层注册`_oscar_inverse_rotation=
  rotation.T.contiguous()`非持久buffer，与forward rotation一起迁移到设备；backend
  以keyword传给sparse attention，内部inverse输出改用该buffer。公共decode/prefill
  保留`inverse_rotation=None`兼容现有测试/外部调用，fallback仍是`rotation.T`；
  production静态门禁要求不得走fallback。下一步针对buffer identity和backend传递
  先写源码测试红灯。
- **contiguous inverse源码发布状态：** TDD有效红灯3 failed；最终CPU相关覆盖
  32 passed/19 CUDA skipped，Ruff/format/mypy/SPDX/其余适用pre-commit全绿。
  源码提交`67a0e47ff72f10a322de17b81c4134984e017bd6`（tree
  `60d5e606ce522dd78fecd890509372b727802f43`）已推送，源码仓库clean/published。
  下一步先全文重读并实时新增报告2.73，明确尚无新苹果800数值或32K端到端结果；
  报告发布前不构建镜像、不申请GPU。
- **2.73报告门禁：** 修改前报告5,115行/278,942 bytes、SHA256
  `41db2e05148b03ae3d325901360cf2047b9feb633abff3265e07b7142a7458ce`；
  修改后5,189行/283,488 bytes、SHA256
  `b1f331e1fa35644fe67ee8fc6af1e9c6db1186cefaab0fc740fbe4a83269dda7`。
  章节1.1–1.5/2.1–2.73、交叉引用、术语、source identity、3/3文件hash、
  数据字面量与diff门禁全绿。下一步只发布主仓库submodule/报告/planning；发布前
  不构建镜像或分配GPU。
- **2.73发布状态：** 主仓库提交
  `83a1df0ea1ca6eb1d56403fbcf0aca4f121da115`已推送；主/源码仓库均
  clean/published。下一步只读核对Phase 6/Stage 9现有构建链，形成绑定
  `67a0e47ff`的不可变候选输入；配置发布与CPU构建/验收报告完成前不申请GPU。
- **Phase 6输入候选：** 已最小切换output tag、源码commit/tree与Dockerfile默认
  commit/tree到`67a0e47ff`/`60d5e606`；Dockerfile实算SHA256为
  `42b772b0f322b884d426e0b4c67b65b2aa83bbe2b1f6632b274cf11c68d9bf26`。
  Stage 9 Dockerfile/wrapper/config尚未修改。下一步运行JSON/身份/确定性构建测试与
  diff门禁，结果写入2.74后再发布配置。
- **2.74报告门禁：** 报告5,189→5,231行、283,488→285,645 bytes，SHA256
  `b1f331e1fa35644fe67ee8fc6af1e9c6db1186cefaab0fc740fbe4a83269dda7→
  946a6bd3da1c9f8f4995e055760a17208eb054a09a441aac2b2f77b41ead8e57`；
  章节1.1–1.5/2.1–2.74、引用、术语、两输入hash、1/1测试和diff全绿。
  下一步只提交推送配置阶段，发布前不执行OCI构建。
- **2.74发布状态：** Phase 6输入/报告/planning已由主仓库提交
  `abf870d237f24a831652dce0136a8c9bd71912b2`推送；两仓clean/published。
  下一步在两个独立新目录顺序构建并递归验收OCI，记录实际内容摘要与逐字节
  一致性；全程CPU-only，不申请GPU。
- **OCI v1/2.75：** v1 build/verification为built/passed；image`22c2539e…9c66`、
  manifest`f700ee72…a537`、layer`37e119e5…a2b2`。报告现5,273行、SHA256
  `c7d292b4e11ba1f6bc4bee5c84ddfd57929feda589a325ebc817f3c365b917e9`，
  章节至2.75和证据对账通过。下一步只发布v1记录，再做v2独立重建。
- **OCI v2重建：** 独立目录`20260801T004846Z…v2_rebuild`的build/verification
  亦为built/passed；index逐字节一致，image/config、manifest、candidate layer、
  diff-ID四项与v1完全相同。v2 build/verification SHA256为`7935c471…c7bc`/
  `df517137…2597`（路径/main commit字段不同故报告hash不同）。下一步全文更新2.76，
  发布前不导入daemon或迁移Stage 9。
- **2.76门禁：** 报告5,273→5,309行、SHA256
  `c7d292b4e11ba1f6bc4bee5c84ddfd57929feda589a325ebc817f3c365b917e9→
  b3436a5d6850f864504b8a49231b0b2d1f17fe43cf0fd5a644aa81cd4e76d4a3`；
  章节至2.76、术语、四项逐字节内容和两份v2 report hash对账通过。下一步只发布
  双构建记录，发布前不导入daemon。
- **daemon导入：** 一次性Ubuntu22.04/skopeo1.4.1工具容器CPU-only完成导入，exit0；
  daemon image ID=`22c2539e…9c66`、33层、末层diff-ID=`a11fef0c…91e7`，6项
  labels通过。inspect SHA256=`8724ac2d…d969`；工具容器已自动删除，外部容器
  DeviceRequests=null。下一步全文更新2.77并发布，之前不迁移Stage 9。
- **2.77门禁：** 报告5,309→5,344行、SHA256
  `b3436a5d6850f864504b8a49231b0b2d1f17fe43cf0fd5a644aa81cd4e76d4a3→
  2fa577af7239f3c66d8cbd1bf69f0cbb45d765434c6d1a0be3e922dab223ff25`；
  章节至2.77、术语、daemon身份与inspect hash通过。下一步只发布导入记录。
- **2.78控制输入门禁：** Dockerfile只切base一行，新hash`65f1ed38…b79fc`；
  报告5,344→5,365行、SHA256`2fa577af…ff25→6231e144…743f`，章节至2.78、
  术语/hash/diff通过。下一步只发布控制输入，之后CPU-only构建新控制镜像。
- **控制镜像CPU验收：** 首次误用项目根context运行10分钟后已停止且无image；按正式
  docker/ context重建成功。新image ID=`2d0e9f1e…6f74`，base/control 33/34层且
  前33层一致，末层diff-ID=`126c2fb0…1343`，labels/entrypoint通过；CPU runtime
  确认git2.34.1/iproute2-5.15.0/Torch2.11、inverse signature/buffer source通过、
  cuda_initialized=false。下一步全文更新2.79并发布，之前不申请GPU。
- **2.79门禁：** 报告5,365→5,404行、SHA256
  `6231e14426f63185c917daa3192b6c5a0459b359b71b746afb61d771f6a1743f→
  ee4c0f6055dab7acbe96bbc1acca5b811ddfa4bcbe5fc3041e3ad59e0e2f9669`；
  章节至2.79、术语、image/layer/runtime数据与diff通过。下一步只发布控制镜像记录。
- **contiguous inverse runtime import：** GPU 双空闲检查为
  `2026-08-01T01:11:53Z/01:14:08Z`，间隔135秒，8/8卡均0 MiB/0%、无
  compute process。固定GPU 0、driver-only探针一次通过，退出码0；Python/
  PyTorch/Triton为3.12.13/2.11.0+cu129/3.6.0，候选vLLM Python与`_C`路径、
  78层rotation和三项artifact hash匹配，`cuda_initialized=false`。有效JSON/log
  SHA256为`0910b598…7b7a`/`f2e60043…189a`，JSON与冻结协议逐字节一致；
  `01:16:16Z`退出快照及`01:16:24Z`复查均8/8空闲。下一步全文重读并更新2.80，
  发布前不迁移Phase 1/5/7/9配置。
- **2.80报告门禁：** 修改前报告5,404行/294,943 bytes、SHA256
  `ee4c0f6055dab7acbe96bbc1acca5b811ddfa4bcbe5fc3041e3ad59e0e2f9669`，
  全文分段读取前后hash稳定；修改后5,461行/298,481 bytes、SHA256
  `3029ea2a12b3c7422c4e4dc303a79e7175e32157b2646b886b6acad6f5e03f04`。
  章节1.1–1.5/2.1–2.80、术语、7份证据hash、JSON状态/字段、冻结协议逐字节
  一致性、时间戳与diff check全绿。下一步只提交推送2.80/planning；发布前不迁移配置。
- **2.80发布状态：** runtime import报告与planning已由主仓库提交`7bfee80`
  推送，主仓库clean/published。下一步从验收候选机械派生新overlay，并按
  Phase 1→5→7→9顺序迁移配置/wrapper；完成工具测试和64/64递归门禁后先更新报告，
  再发布静态链路。
- **contiguous inverse静态迁移候选：** 新overlay为4,749个普通文件和6个native
  symlink；候选层/overlay递归清单逐字节一致，SHA256均为`797e7c2e…ee83`，
  links/target清单与上一正式链路逐字节一致。Phase 1/5/7/9与9个正式wrapper
  已绑定`67a0e47ff`、新OCI/overlay/control；四配置当前SHA256为
  `1d33af7f…ee1`/`a9508b0d…ed15`/`0c3c97cf…6783`/`a478b72d…3f4b`，
  旧ca4a身份在configs/scripts中清零。下一步运行JSON/shell/compile、Phase 7/9
  工具测试和容器内64/64递归verifier，尚未宣称静态链路通过。
- **静态测试当前结果：** 输入v2 12/12、shell 9/9、Phase 7工具20/20、Phase 9
  工具47/47、compile 15/15通过。首轮递归verifier为62/66，唯一根因是未用正式
  phase0 source卷覆盖NFS mode漂移；其余候选OCI/source/overlay/artifact/config/
  evaluator检查全绿。下一步补精确只读volume mount重跑，不修改实现或门限。
- **contiguous inverse静态迁移通过：** 补正式base source只读卷后递归verifier
  为66/66、exit0；最终静态汇总为输入12/12、shell 9/9、Phase 7 20/20、
  Phase 9 47/47、compile 15/15、recursive 66/66，GPU未分配。25份证据
  合计114,644 bytes，summary/recursive/manifest SHA256为`ce56900f…673c`/
  `5e652f0c…c449`/`3e4caaa4…c5ba`。下一步全文重读并新增2.81，发布前不申请GPU。
- **2.81报告门禁：** 修改前报告5,461行/298,481 bytes、SHA256
  `3029ea2a12b3c7422c4e4dc303a79e7175e32157b2646b886b6acad6f5e03f04`，
  全文分段读取前后稳定；修改后5,527行/302,653 bytes、SHA256
  `ff6010799d1d45edb65c2fd4639df93a87803fa760b81a50882525bf6fe11292`。
  章节至2.81、术语、配置依赖hash、overlay清单、25/25证据、66/66递归、
  114,644 bytes与diff均通过。下一步只发布静态链路；发布前不申请GPU。
- **2.81发布状态：** 静态配置、wrapper、报告与planning已由主仓库提交
  `4dddc09`推送；主/源码仓库clean/published。下一步执行新的双GPU空闲检查，
  然后用固定控制镜像运行driver-injected candidate preflight；不加载模型。
- **contiguous inverse preflight通过：** 双空闲检查`01:33:27Z/01:34:40Z`
  间隔73秒，8/8卡0 MiB/0%。正式preflight退出0：static 66/66、固定环境import、
  服务参数解析均通过，两处`cuda_initialized=false`；解析到TP8、131072 model len、
  max seqs16、batched tokens2048、OSCAR INT2与torch profiler。10份证据共
  41,458 bytes，validation/manifest SHA256为`5a3760c8…3628`/`df26f028…dc83`；
  `01:36:15Z`退出与最终复查GPU均空闲。下一步全文更新2.82并发布，之后才启动模型。
- **2.82报告门禁：** 修改前报告5,527行/302,653 bytes、SHA256
  `ff6010799d1d45edb65c2fd4639df93a87803fa760b81a50882525bf6fe11292`，
  全文分段读取前后稳定；修改后5,584行/305,969 bytes、SHA256
  `f6c84f21041c486ca9b921fe8814999db9934e54f22d32d1fee244f1c4843334`。
  章节至2.82、术语、10/10 evidence、66/66、两处CUDA=false、正式参数、
  41,458 bytes与diff全绿。下一步只发布preflight记录，发布前不加载模型。
- **2.82发布状态：** preflight报告与planning已由主仓库提交`e02fd5f`推送，
  两仓clean/published。下一步为正式32K/batch1单格创建新run ID，重新双检8卡空闲，
  再运行3轮+8-rank profiler；长轮次每10分钟输出进度。
- **contiguous inverse正式32K结果：** run
  `20260801T013914Z_stage9_candidate_67a0e47ff_32k_b1_v1` 已 exit=0；
  summary/cell/三轮/profile 全部 passed。三轮中位 mean TTFT/TPOT/请求吞吐为
  `30539.197439017396 ms`/`202.51436330123838 ms`/
  `0.0177743652305229 req/s`。相对2.62上一版OSCAR为
  `-5.886260691%/+1.686770700%/+2.607779616%`；相对BF16的TTFT/TPOT为
  `+143.767038567%/+13.242965274%`。8 tables+8 worker traces+1 frontend trace
  均通过；容器已删除，8卡恢复0 MiB/0%。
- **2.83报告门禁：** 修改前按450行连续区间顺序读取报告全部5,584行，SHA256
  保持`f6c84f21…3334`；修改后为5,673行/311,191 bytes、SHA256
  `3d74d48298ceefb847cc2bf6983d04887a10d53b51c19511fc0dbc18b451fa37`。
  章节1.1–1.5/2.1–2.83连续，交叉引用、术语、正式summary/comparison、41/41
  evidence、三轮数据、profiler身份和diff全部通过。
- **当前下一步：** 只提交并推送2.83、comparison与planning；发布完成前不进入
  trace归因。发布后使用冻结的8份worker trace执行CPU-only归因，按实际结果选择
  下一项最小候选；TTFT尚未达到BF16 +20%门限，阶段9仍保持进行中。
- **2.83发布状态：** 正式结果报告与planning已由主仓库提交`fcdcac0`通过HTTPS
  推送；主仓库与源码仓库均clean/published。下一步进入冻结trace的CPU-only归因，
  不申请GPU、不修改production源码；归因阶段结束后先实时更新报告再选候选。
- **contiguous inverse trace归因：** 当前与2.62参考的8-rank trace均用同一
  format-v3 analyzer重析，exit0/passed。prefill wall/kernel下降
  `1908.716/1927.360 ms`，rotation下降`1904.507 ms`并解释wall改善
  `99.779%`；8/8 rank和16/16 chunk方向一致，profile wall解释正式TTFT改善
  `99.930%`。stage1仍为`19846.588 ms`、占wall`64.921%`。
- **2.84报告门禁：** 修改前报告5,673行/311,191 bytes、SHA256
  `3d74d482…fa37`，按450行连续区间读取前后稳定；修改后5,762行/
  316,952 bytes、SHA256`a14ce0563cf29436835caf50a5b2ef0b3c7f312967aa932f810fdf6aee92c082`。
  章节至2.84、引用、术语、18/18 evidence、25/25 validation、逐rank/chunk方向
  和diff全绿。下一步只发布2.84/planning；发布前不进入下一源码候选。
- **2.84发布状态：** 报告与trace归因planning已由主仓库提交
  `db9e02721b6eb72e068bb26644ed7e3f720cb0e5`通过HTTPS推送；主仓库HEAD与
  `origin/feat/glm52-model-load`一致且工作树clean。下一步只读筛选stage1候选，
  重点验证是否能按cache类型拆分路径以降低约19.85秒stage1成本；在形成实际证据前
  不修改production源码、不申请GPU。
- **stage1只读候选筛选：** 已复核现有资源与覆盖率证据。简单`has_history`因
  排序后no-history active tile仅`0.331697%`继续保持淘汰；当前h8/t16/w8资源为
  109,568-byte shared、242 registers/thread，无法双block驻留。下一候选仅保留
  “BF16/history独立累加+独立LSE合并”的离线资源筛选；先验证拆分kernel是否同时
  降到shared/register双驻留线，不通过则不申请GPU、不修改production。
- **cache-split离线资源结果：** 最终v4为15/15 compiled、baseline精确复现。
  history h8/w8降到84,992-byte shared/199 registers/0 stack，但仍超过双block
  shared线1,536 bytes；history h4/w4虽shared/register允许双block，却产生
  176-byte/thread stack。BF16 h8/w4严格通过，但history严格候选为0，组合严格
  promotion=false。本阶段先实时更新报告2.85并发布；发布前不修改production或
  申请GPU。后续只对history路径做最小资源修正，不直接落地当前拆分形态。
- **2.85报告门禁：** 修改前报告5,762行/316,952 bytes、SHA256
  `a14ce056…c082`，按无截断窗口全文重读后稳定；修改后5,856行/323,194 bytes、
  SHA256=`3e46599c53ac3c35b6432055d720bde96f136a5208de09ecad6f528de1659123`。
  报告diff为94 insertions/0 deletions；章节1.1–1.5/2.1–2.85、交叉引用、术语、
  38/38 evidence、关键资源字段与diff均通过。固定67a控制镜像复跑compile和
  13/13 unittest通过。下一步只发布工具、测试、报告与planning；发布前不继续
  history路径实验。
- **2.85发布状态：** 离线工具、测试、报告与planning已由主仓库提交
  `615a95f73f720d1b0fe7f9ea8527a63c23a31788`通过HTTPS推送；主仓库HEAD与远端
  一致，源码仓库仍为`67a0e47ff`且两仓clean/published。下一步只做history路径的
  CPU-only SM80资源修正；严格门禁通过前不申请GPU、不改production源码。
- **history value reload离线筛选：** 最终v3为20/20 compiled、15/15 unittest、
  `cuda_initialized=false`。5个reload variant与对应base的cubin和资源逐对完全
  相同，shared/register/stack delta均为0；Triton已消除重复load，history严格
  candidate仍为空。47/47 evidence通过，候选淘汰，不申请GPU、不改production。
- **当前下一步：** 全文重读已经完成；只追加2.86记录reload淘汰结果并完成章节、
  交叉引用、术语、证据和固定容器回归门禁。2.86发布前不开始新的资源候选；发布后
  再筛选具有编译器可见阶段边界的history路径结构。
- **2.86报告门禁：** 报告仅追加83行，现为5,939行/328,559 bytes、SHA256
  `146afe09…9237`；章节1.1–1.5/2.1–2.86、术语、交叉引用、47/47 evidence、
  Ruff/compile/固定容器15/15 unittest和diff全部通过。下一步只提交推送工具、测试、
  2.86与planning；发布完成前不开始下一候选。
- **2.86发布状态：** 工具、测试、报告与planning已由主仓库提交`77f4a2d`通过
  HTTPS推送，主仓库HEAD与远端一致，源码仓库仍为`67a0e47ff` clean/published。
  下一步只筛选编译器可见的history阶段边界；在新阶段落地实际结果后仍先更新报告。
- **history narrow-token tile 筛选（进行中）：** 只在离线工具中增加
  h4/h2/h1、t8、w4 history variants，检验更小token tile能否降低反量化中间量
  live range并消除w4 stack spill。成功标准为SM80离线编译通过、shared/register
  双block资源算术通过且stack=0；任一条件失败即淘汰，不修改production、不申请GPU。
- **history narrow-token tile 结果：** 最终format v4轮次为23 variants、20 compiled/
  3 rejected；h4/h2/h1 t8/w4均在history value `tl.dot`处因`K >= 16`编译拒绝，
  没有cubin或资源数字。候选已淘汰，不申请GPU、不修改production。
- **当前下一步：** 2.87已追加到实时报告；先完成章节、交叉引用、术语、49/49证据、
  固定容器回归与diff门禁，再只提交推送本阶段。发布完成前不开始下一候选；后续不再
  重复简单t8 tile搜索，只筛选改变value计算结构或具有编译器可见阶段边界的方案。
- **2.87报告门禁：** 报告仅追加81行，现为6,020行/333,310 bytes、SHA256
  `de97bea5…fac5`；章节1.1–1.5/2.1–2.87、交叉引用、术语、49/49 evidence、
  Ruff/compile/固定容器16/16 unittest和diff全部通过。下一步只提交推送工具、测试、
  2.87与planning；发布完成前不开始下一候选。
- **2.87发布状态：** 工具、测试、报告与planning已由主仓库提交`d96faa6`通过HTTPS
  推送。下一步只固化本发布状态并复核两仓clean/published；随后不再重复简单t8
  tile，转向改变value计算结构或具有编译器可见阶段边界的最小CPU-only筛选。
- **history t8手工value归约筛选结果：** 最终format v5为26 variants、23 compiled/
  3 rejected；h4/h2/h1 t8/w4手工elementwise+sum均编译通过，shared分别为
  26,112/21,760/19,584 bytes、registers/thread为215/190/168、stack均为0，组合
  严格资源门禁首次为true。该结果只证明CPU-only SM80静态资源可行，尚未验证
  output/LSE、实际双block驻留、CUDA性能、TTFT/TPOT或GSM8K，不改production。
- **2.88报告门禁：** 修改前全文6,020行/333,310 bytes且SHA256稳定；修改后为
  6,116行/339,362 bytes、SHA256=`0ddb9fd7…ea91`。章节1.1–1.5/2.1–2.88、术语、
  交叉引用、59/59 evidence、Ruff、固定容器compile、17/17 unittest与diff均通过。
  下一步只提交推送本阶段；发布完成前不启动output/LSE或GPU筛选。
- **2.88发布状态：** 手工value归约离线工具、测试、报告与planning已由主仓提交
  `389bb26`通过HTTPS推送；源码仓`glm52_oscar_vllm`仍为`67a0e47ff`且clean/published。
  下一步先为h4候选建立冻结reference的output/LSE correctness与小型CUDA计时筛选；
  production修改或正式32K前仍须先完成源码/配置提交推送和GPU双空闲检查。
- **history手工归约CUDA筛选（进行中）：** 新建独立microbenchmark，只调用已发布
  离线工具中的standalone history kernel；以h8/t16/w8 dot为冻结reference，对比
  h4/t8/w4 manual，在32K最终位置、2,048 query tokens、batch1、单卡上同时验证
  output/LSE allclose与预热后CUDA时间。先TDD/固定CPU容器/提交推送，再做两次间隔
  至少60秒的GPU空闲检查；不修改production。成功标准为correctness通过且候选实际
  CUDA时间优于reference，否则淘汰，不根据离线资源表晋升。
- **2.89静态准备门禁：** 实时报告已追加筛选范围与边界，现为6,172行/
  343,616 bytes、SHA256=`70714f1b…9910`；章节、交叉引用、术语、Ruff 0.14.0、
  固定容器compile、22/22 unittest与diff均通过。当前尚无GPU结果。下一步只提交推送
  benchmark、测试、报告与planning；两仓恢复clean/published后才做GPU双空闲检查。
- **2.89发布状态：** benchmark、测试、报告与planning已由主仓库提交`bcc0577`
  通过HTTPS推送。下一步固化本状态并确认主仓库HEAD与upstream一致、源码仓继续为
  `67a0e47ff` clean/published；随后开始GPU双空闲检查，不提前运行benchmark。
- **history手工归约CUDA筛选结果：** 双空闲间隔72秒后固定GPU0运行；output/LSE
  correctness通过，但h4/t8/w4 manual median CUDA为`38.589439 ms`，相对
  h8/t16/w8 reference的`20.436993 ms`慢`88.821516%`，promotion=false，候选
  淘汰且不改production。下一步先把实际结果追加为报告2.90并发布；发布前不筛选
  新候选。
- **2.90报告门禁：** 报告现为6,255行/348,766 bytes、SHA256=
  `73725d95…22ef`；章节、引用、术语、结构化result、5/5 evidence和diff均通过。
  下一步只提交推送报告/planning；恢复clean/published后，再依据2.84的stage1
  瓶颈与本轮淘汰结论只读选择下一候选。
- **2.90发布状态：** 淘汰结果与planning已由主仓库提交`9aaa906`通过HTTPS推送。
  下一步固化状态并复核两仓clean/published；随后只读检查保留t16循环的手工value
  归约资源机会，形成实际CPU-only门禁前不申请GPU。
- **history t16手工value归约筛选结果：** format v6 CPU-only SM80轮次为29项、
  26 compiled/3 rejected；h4/h2/h1 t16/w4 manual的shared为43,520/39,168/
  36,992 bytes、registers/thread均255，但stack为576/104/96 bytes，三项strict均
  为false并淘汰。该阶段没有申请GPU或修改production；顶层strict=true仅来自2.88
  的旧t8项，不能写成新候选通过。
- **2.91报告门禁：** 修改前报告已全文重读且与发布HEAD一致；修改后为6,344行/
  354,211 bytes、SHA256=`0336af7b…2e09`。章节1.1–1.5/2.1–2.91连续，交叉引用、
  术语、结构化summary、61/61 evidence、62文件/189,284 bytes与diff均通过。
  Ruff、固定镜像compile、22/22 unittest与diff均通过；工具、测试、报告与planning
  已由主仓提交`9e79108`通过HTTPS推送。下一步只固化发布状态并复核两仓
  clean/published；发布状态提交`420875c`推送后，主仓HEAD/upstream精确一致，
  源码仓与gitlink均为`67a0e47ff`且clean/published。本阶段完成。
- **当前下一步：** 继续围绕2.84确认的约19.85秒stage1瓶颈做只读候选筛选；排除
  已实测变慢的t8 manual和有spill的t16 manual，只有形成新的可验证结构与冻结门禁
  后才修改离线工具。下一阶段完成后仍须先全文重读并实时更新报告，再发布或申请GPU。
- **history score物化三阶段候选（进行中）：** 现有history专用w8几何虽零stack，
  但199–206 registers/thread仍只能单block驻留；w4则产生176–192-byte stack。
  下一候选不再搜索同构tile，而以score→LSE→value三个kernel形成真正编译器边界：
  score仅计算并物化每个query/head/top-k分数，LSE独立归约，value按更小latent-dim tile
  读取最终LSE并累加。第一阶段只在standalone CPU-only SM80工具中编译与核算资源；
  不修改production、不申请GPU。32K末块/batch1/TP8每rank的FP32 score scratch约
  128 MiB，额外显存流量是明确风险，只有三个kernel均编译、无stack且资源显著低于
  当前history h8/t16/w8后，才进入冻结output/LSE与CUDA时间筛选。
- **三阶段TDD红灯：** 新测试先冻结5项score/LSE/value variant、精确scratch字节、
  编译器边界源码约束与三段strict gate；固定67a只读容器中按预期因实现脚本不存在
  得到`FileNotFoundError`。下一步实现最小standalone CPU-only编译工具，不接入runtime。
- **三阶段首轮静态错误：** 最小工具实现后Ruff check通过，但format-check要求机械
  重排实现文件并返回非零，按`set -e`容器compile/tests尚未执行。下一步只用同一
  Ruff格式化两个目标文件，再完整重跑；机械格式化后Ruff、固定67a容器compile、
  4/4 unittest与diff已通过。下一步只运行CPU-only SM80离线编译，尚无资源结果。
- **三阶段离线v1错误：** 固定容器直接以文件启动工具时，`sys.path[0]`为
  `scripts/phase9`而非项目根，`from scripts.phase9 ...`触发`ModuleNotFoundError`；
  在任何variant编译前退出，没有summary/CUDA/GPU使用。保留v1失败目录，下一次改为
  同目录本地导入并让测试显式加入脚本目录，以新v2 run ID重跑。
- **三阶段离线v2结果：** 5/5 variants编译、0 rejected、CUDA未初始化。LSE为
  16-byte shared/17 registers/0 stack，value h2/h1 d128为8,320/8,256-byte
  shared、112/114 registers/0 stack，均strict=true；score h8/h4虽降至52,224/
  43,520-byte shared、164/162 registers/0 stack，但w8的256-thread block仍不能
  双驻留，score strict列表为空，pipeline gate=false。scratch总142,671,872 bytes。
  下一步先封存证据、全文重读并实时更新2.92，发布前不追加score-w4或申请GPU。
- **2.92报告门禁：** 修改前报告已完整重读并确认与发布HEAD一致；追加后为
  6,441行/360,484 bytes、SHA256=`524caca0…7642`。章节1.1–1.5/2.1–2.92连续，
  “三池”为0，大写`A800`仅第5行；18/18 manifest、19文件/70,064 bytes、summary
  精确字段、Ruff、固定镜像compile、26/26 unittest与diff均通过。下一步只提交推送
  工具、测试、证据、2.92与planning；发布完成前不补测score-w4。
- **2.92发布状态：** 三段式离线工具、测试、2.92报告与planning已由主仓库提交
  `03d1b78`通过既有GitHub HTTPS认证推送。下一步只固化本发布状态并复核两仓
  clean/published；完成后才开始score-w4的CPU-only TDD与资源补测，不申请GPU。
- **score-w4补测（进行中）：** 两仓已复核clean/published；只为score阶段增加
  h8/h4、t16、w4两个离线variant，并把summary schema升版。冻结门禁为SM80编译、
  shared/register双block算术与stack=0；不修改kernel语义、production或申请GPU。
- **score-w4 CPU-only结果：** format-v2为7/7 compiled、0 rejected、3.924135983秒、
  CUDA未初始化。h8/w4为52,224-byte shared/255 regs/40-byte stack，strict=false；
  h4/w4为43,520/254/0，strict=true。score首次出现严格候选，结合既有LSE/value，
  pipeline gate=true；这仍只是离线资源算术，不代表correctness或性能，不申请GPU。
- **当前下一步：** v3小型证据23文件/76,967 bytes、manifest 22/22通过。先全文重读
  当前6,441行报告并实时追加2.93；报告发布前不设计或运行CUDA correctness/计时。
- **v3目录标签错误：** 启动命令误写未来`051800Z`，而summary实际完成时间为
  `04:44:08Z`；已只重命名为`20260801T0444Z_history_score_w4_offline_v3`，未修改
  summary/log或证据哈希。报告固定使用纠正后的目录名。
- **2.93术语校验错误：** 首轮脚本把“大写`A800`只允许历史链接第5行”误写成
  “全文字符只出现1次”；该行label/target各有1次，导致只读断言失败且后续门禁未跑。
  改为检查命中行集合精确为第5行，不修改报告。
- **2.93报告门禁：** 修改前报告重新读取并确认与发布HEAD一致；追加后为6,536行/
  366,358 bytes、SHA256=`1088dfef…f09b`。章节1.1–1.5/2.1–2.93连续，术语与引用、
  22/22 manifest、23文件/76,967 bytes、summary精确字段、Ruff、固定镜像compile、
  27/27 unittest与diff均通过。下一步只提交推送本阶段；发布完成前不运行GPU。
- **2.93发布状态：** score-w4工具、测试、报告与planning已由主仓库提交`cdd820a`
  通过HTTPS推送；主仓HEAD/upstream一致，源码仓仍为`67a0e47ff` clean/published。
- **三段式CUDA筛选入口（进行中）：** 复用2.89冻结的32K末段synthetic all-history
  输入，以h8/t16/w8单kernel为reference；candidate固定score h4/t16/w4、LSE
  tiles128/w4、value h2/d128/t16/w4，并把三个launch的合计CUDA event时间与
  output/LSE allclose作为门禁。先TDD/CPU静态/2.94发布，再双空闲检查；尚不运行GPU。
- **2.94首轮静态错误：** 最小benchmark实现Ruff check通过，但format-check要求
  机械格式化并退出，后续compile/tests未运行。只机械format后重跑完整静态门禁。
- **2.94静态报告门禁：** 修改前报告全量读取且与发布HEAD一致；追加后为6,597行/
  370,679 bytes、SHA256=`d939b1f8…ce40`。章节1.1–1.5/2.1–2.94连续，术语、引用、
  benchmark/test哈希、Ruff、固定镜像compile、32/32 unittest与diff均通过；尚无GPU
  结果。下一步只提交推送2.94，发布完成前不做空闲检查或运行benchmark。
- **2.94发布状态：** benchmark、测试、2.94报告与planning已由主仓库提交`35708d0`
  通过HTTPS推送。下一步只固化本状态并复核两仓clean/published；随后执行两次间隔
  至少60秒的8卡空闲检查，固定只使用GPU 0运行已发布benchmark。
- **三段式单卡筛选结果：** 两次8卡空闲检查间隔79秒，固定GPU0运行已发布入口。
  output/LSE allclose通过，max_abs为`8.72612e-05/1.90735e-06`；但candidate三个
  launch合计CUDA中位数`53.320705 ms`，reference为`20.418560 ms`，慢
  `161.138422%`，promotion=false并淘汰，不进入production或完整stage1。
- **当前下一步：** CUDA证据13文件/74,950 bytes、manifest 12/12通过，退出后8卡
  均0 MiB/0%。先全文重读并实时追加2.95，发布前不运行下一候选或端到端实验。
- **2.95报告门禁：** 修改前报告全量读取且与发布HEAD一致；追加后为6,678行/
  375,914 bytes、SHA256=`e06cfecc…faf1`。章节1.1–1.5/2.1–2.95连续，术语、引用、
  12/12 evidence、13文件/74,950 bytes、result精确字段与diff均通过。下一步只提交
  推送淘汰结果；发布完成前不运行新实验。
- **2.95发布状态：** 三段式单卡淘汰结果与planning已由主仓库提交`e050f11`通过
  HTTPS推送；两仓恢复clean/published。当前三段式方向关闭。
- **下一只读候选：** 固定Triton 3.6 `CUDAOptions`实查确认支持`maxnreg`，可对
  h4/t16/w8 single-kernel尝试显式寄存器上限，检验双驻留与spill权衡。下一阶段仍
  先TDD和CPU-only SM80资源筛选，形成实际结果后先更新报告；不直接申请GPU。
- **h4/w8 maxnreg筛选（进行中）：** 只扩展standalone history离线矩阵，固定
  h4/t16/w8并补测`.maxnreg=128/120/112/96`。先证明cap实际进入编译选项，再读取
  cuobjdump的register/stack/shared；不修改kernel语义、production或申请GPU。
- **maxnreg CPU-only结果：** format-v7为30 compiled/3既有t8 rejected、32.533861秒、
  CUDA未初始化。128/120/112/96 cap均实际生效，register降至对应值，但stack升至
  192/232/256/360 bytes/thread；baseline h4/w8为206 regs/0 stack，h4/w4为
  255 regs/176 stack。四项均strict=false，没有零spill候选，不直接申请GPU。
- **当前下一步：** 证据21文件/188,448 bytes、manifest 20/20通过。先完成合并静态
  回归，再全文重读并实时追加2.96；发布前不建立GPU入口。
- **2.96报告门禁：** 修改前报告全量读取且与发布HEAD一致；追加后为6,773行/
  382,043 bytes、SHA256=`b3f6f0a8…f9f3`。章节1.1–1.5/2.1–2.96连续，术语、引用、
  20/20 evidence、21文件/188,448 bytes、summary精确字段、Ruff、固定镜像compile、
  34/34 unittest与diff均通过。下一步只提交推送；发布前不运行GPU。
- **2.96发布状态：** maxnreg CPU-only工具、测试、报告与planning已由主仓库提交
  `c709e10`通过HTTPS推送；主仓HEAD与upstream一致，源码仓继续固定为
  `67a0e47ff`且clean。
- **maxnreg128 CUDA裁决（进行中）：** 静态资源不能直接预测spill与双驻留的净效应。
  下一阶段在32K末段standalone history冻结输入上同时比较h8/t16/w8参考、
  h4/t16/w8无cap控制和h4/t16/w8/maxnreg128候选。只有correctness通过，且cap128
  同时严格快于两项控制时才可晋升；先TDD、静态门禁、实时追加2.97并发布，发布前
  不申请GPU。
- **2.97静态报告门禁：** 新入口只对candidate传`maxnreg=128`，旧默认variant保持
  不变；固定镜像compile与合并37/37 unittest（0.095秒）、Ruff、diff均通过。
  修改前报告全量读取且与发布HEAD一致；追加后为6,831行/386,150 bytes、SHA256
  `20c8059a…c99e`，章节1.1–1.5/2.1–2.97连续，术语、引用和三项脚本哈希均通过。
  下一步只提交推送，发布完成前不做GPU空闲检查。
- **2.97发布状态：** benchmark、测试、报告与planning已由主仓库提交`125d693`
  通过HTTPS推送。下一步先固化本发布状态并复核两仓clean/published，再开始两次
  间隔至少60秒的8卡空闲检查。
- **maxnreg128单卡结果：** 双空闲检查间隔65秒，固定GPU0完成三项筛选；candidate
  与h8 reference的output/LSE误差均为0。h8/h4无cap/h4-cap128 CUDA中位数分别为
  `20.432896/40.378368/30.505983 ms`。cap128相对h4无cap快`24.449688%`，但相对h8
  仍慢`49.298386%`，promotion=false，正式淘汰且不改production。
- **2.98报告门禁：** 证据12文件/53,237 bytes、manifest 11/11通过，退出后8卡均
  0 MiB/0%。修改前报告全量读取且与HEAD一致；追加后为6,904行/390,770 bytes、
  SHA256=`f8a8c2c4…b15c`，章节1.1–1.5/2.1–2.98连续，术语、引用、result字段与
  evidence均通过。下一步只提交推送，恢复clean/published前不运行下一候选。
- **2.98发布状态：** 单卡淘汰结果与planning已由主仓库提交`ef4b115`通过HTTPS
  推送，HEAD/upstream一致。下一CPU-only候选只在standalone工具中把history每个
  token的128个唯一packed byte与4组scale/zero各加载一次再广播；先要求不同cubin且
  不增加stack，不修改production、不申请GPU。
- **compact-load CPU-only结果：** 有效v2为31 compiled/3既有t8 rejected、
  `cuda_initialized=false`。h8 reference/candidate PTX `ld.global`为165/71，cubin
  135,856/106,800 bytes，shared/stack同为84,992/0；candidate寄存器199→230，
  offline promotion=true，但仍是单block资源形态，不能宣称实际加速。
- **2.99报告门禁：** 证据20文件/685,084 bytes、manifest 19/19通过；修改前报告
  全量读取且与HEAD一致，追加后为6,996行/396,736 bytes、SHA256=`20c33b8d…73a7`，
  章节1.1–1.5/2.1–2.99连续，术语、引用、summary字段、Ruff、固定镜像compile、
  合并39/39 unittest与diff均通过。下一步只提交推送；发布前不建立GPU入口。
- **2.99发布状态：** compact-load CPU-only工具、报告与planning已由主仓库提交
  `432ee5f`通过HTTPS推送，HEAD/upstream一致。
- **2.100静态入口：** 新benchmark只比较同h8/t16/w8 reference与compact candidate，
  唯一变量为compact标志；correctness通过且CUDA中位数严格更小才晋升。TDD红灯后
  定向3/3、合并42/42 unittest（0.092秒）、Ruff/compile/diff通过。修改前报告全量
  读取且与HEAD一致；追加后为7,055行/400,497 bytes、SHA256=`8629f2e0…5dd0`，
  章节1.1–1.5/2.1–2.100连续，术语、引用和四项文件哈希通过。下一步只提交推送；
  发布前不做GPU空闲检查。
- **2.100发布状态：** benchmark、测试、报告与planning已由主仓库提交`ce2c4fa`
  通过HTTPS推送。下一步先固化发布状态并复核两仓clean/published，再开始两次间隔
  至少60秒的8卡空闲检查。
- **compact-load单卡结果：** 双空闲检查间隔65秒，固定GPU0完成筛选；candidate
  相对reference的output/LSE误差均为0。CUDA中位数`20.431871→16.819201 ms`，
  降低`17.681547%`、加速`1.214794×`，promotion=true，获得production集成资格；
  尚不能改写端到端TTFT。
- **2.101报告门禁：** 证据12文件/51,111 bytes、manifest 11/11通过，退出后8卡
  均0 MiB/0%。首次追加把2.101误放在2.97后，被章节校验拦截且未发布；原样移动到
  2.100后再验证。最终报告为7,130行/405,166 bytes、SHA256=`5b295d22…30d9`，
  章节1.1–1.5/2.1–2.101连续，术语、引用、result字段与evidence通过。下一步只
  提交推送；恢复clean/published前不改production。
- **2.92全文读取错误：** 首次把1–1,200行合并输出时工具发生截断，不能作为全文
  重读证据；报告未修改。下一轮从第1行重新按单个600行窗口读取并确认无截断。
- **2.92证据复核错误：** 全文重读完成后的首轮只读复核误用宿主缺失的`jq`，并将
  `failed_v1_run.log`/`post_gpu_state.txt`错写为另外两个文件名；JSON与这两个哈希
  未完成，其余文件未受影响。后续固定使用已有Perl JSON::PP与`find`确认的实际文件名，
  不安装依赖、不重复错误路径。
- **2.92固定镜像回归错误：** Ruff已通过，但首轮`docker run`没有覆盖镜像
  ENTRYPOINT，导致传入的Python绝对路径被当作脚本并报`cannot execute binary file`；
  compile/unittest没有执行。后续先检查镜像Config，再显式设置正确ENTRYPOINT重跑。
- **2.92只读挂载compile错误：** 显式Python ENTRYPOINT纠正后，`py_compile`尝试在
  只读仓库内写`__pycache__`而因Errno 30退出，测试未运行。保持只读挂载不变，后续
  用容器内任务专属`PYTHONPYCACHEPREFIX`承接字节码输出。
- **2.92报告manifest复核错误：** 追加报告后的首轮检查从仓库根直接执行清单，导致
  18个相对路径均解析错误并fail-closed；报告章节/术语检查已经通过，证据未变。
  下一轮固定在证据目录内执行`sha256sum -c evidence_manifest.sha256`。
- **修复编辑错误：** 首次同时修改工具与测试的`apply_patch`因文件分段hunk格式无效
  被整体拒绝，两个文件均未改变。下一次使用两个合法独立hunk，不重复错误格式。
- **本阶段校验错误：** 首轮证据复核把报告相对路径错误拼在仓库根目录，并继续假设
  `results/summary.json`层级，触发`FileNotFoundError`。已用现存正式证据目录的实际
  根层`summary.json`修正，完成61/61复核；后续不重复错误路径。
- **本阶段静态回归错误：** Ruff 0.14.0已通过，但固定镜像首轮命令错误使用不存在的
  `/opt/vllm/bin/python`，因此compile/tests未执行。下一次从镜像固定`PATH`解析实际
  Python入口后重跑，不重复硬编码错误路径；该错误不代表源码或测试失败。纠正为
  镜像`PATH`内Python 3.12.13后，compile、22/22 unittest与diff均通过，发布门禁完成。

- **2.102 production集成门禁：** 源码提交`c0bcbbbdf`已HTTPS推送；满宽mixed
  production CPU-only SM80编译把PTX load 245→167、cubin 206,640→187,056 bytes，
  但新增136-byte/thread stack spill；latent384/block512 fallback保持245 loads与
  0 stack。报告已实时追加2.102，章节与12/12证据门禁通过。当前下一步只提交并推送
  主仓库gitlink、报告与planning；两仓clean/published前不构建新runtime或申请GPU。
- **2.102发布状态：** 主仓库提交`ca33b4f`已通过HTTPS推送，包含源码gitlink、报告
  与planning。当前下一步先固化本状态并复核两仓clean/published，再构建绑定
  `c0bcbbbdf`的候选runtime并执行CPU-only递归身份门禁；静态发布完成前不申请GPU。
- **2.103 Phase 6输入迁移：** candidate tag、源码commit/tree与Dockerfile默认身份
  已最小切换到`c0bcbbbdf`/`061c294d`；固定67a控制镜像内JSON、3个Python compile、
  PAX确定性1/1、源码身份、旧值清零与diff均通过。报告已实时追加2.103并通过章节、
  术语、引用和哈希门禁。下一步只发布本检查点；两仓恢复clean/published后才在两个
  独立目录构建并递归验证OCI，CPU-only构建结果不得冒充GPU或端到端性能结果。
- **2.104 OCI v1：** 固定Python 3.12.13控制容器中的第一次有效CPU-only构建为
  built，递归验收为passed；image/config=`08d8ea6f…360f`、manifest=
  `320e011e…9006`，4,744个源码文件与7个基础层原生扩展门禁通过。下一步先发布
  2.104；clean/published后才在独立目录重建并比较四项OCI不可变内容。
- **2.105 OCI确定性：** v2独立重建同为built/passed；两轮index、config、
  manifest与candidate layer逐字节完全相同，image/config=`08d8ea6f…360f`。
  下一步先发布2.105；clean/published后才从已验收v1导入daemon并审计身份。
- **2.106 daemon身份：** v1已由skopeo 1.4.1导入，image ID、33层、最后diff-ID、
  tag与8项关键labels全部通过独立审计。下一步先发布2.106；随后双空闲检查并执行
  driver-injected、无kernel的runtime import。
- **2.107 runtime import：** 两次8卡空闲检查间隔73秒；只注入GPU0驱动的探针
  一次通过，固定版本、候选vLLM Python/`_C`、78层rotation、三项artifact与
  `reasoning_effort=max`匹配，`cuda_initialized=false`。下一步先发布2.107，
  再最小切换并CPU-only构建Stage 9控制镜像。
- **2.108 控制镜像入口：** Stage 9 Dockerfile只把默认base切换到c0bc候选，
  新SHA256=`99932fd2…fd10`；daemon base身份匹配且目标control tag尚不存在。
  下一步先发布2.108；clean/published后才CPU-only构建并审计34/33层继承。
- **2.109 控制镜像结果：** 空context有效构建得到control `b4785123…5088`、
  34层；前33层、labels、entrypoint、source身份与固定CPU runtime均通过，证据
  20/20复算。下一步先发布2.109，再派生overlay并迁移四级正式配置。
- **2.117 c349 Phase 6恢复链：** c0bc回退提交`c349e32e9`已形成新OCI
  `2ff10a1f…ebbe`并通过33/32层、4,744源码文件、rotation/runtime/native递归门禁。
  有效overlay为4,749普通文件+6个native symlink，普通文件与67a逐字节一致；daemon
  image ID、最后diff-ID、tag和8项labels审计通过。一次EXDEV构建入口、一次NFS
  hard-link失败和一次本地image ID引用失败均独立保留且未混入有效结果。2.117已实时
  写入报告并通过章节/证据门禁，提交`0e9047c`已HTTPS推送。下一步固化发布状态并
  恢复两仓clean/published；随后才执行runtime import前两次至少间隔60秒的8卡空闲检查。
- **2.118 c349 runtime import：** 首轮错误使用系统Python导致`vllm._C` ABI符号失败，
  已保留为无效边界；两组双空闲检查各间隔65秒。有效轮只修正到冻结venv Python，
  固定GPU0且不运行kernel，结果exit0、`cuda_initialized=false`，版本、78层rotation、
  三项artifact、候选Python/native与`reasoning_effort=max`全部匹配。2.118已实时写入；
  下一步完成门禁并发布，随后最小切换Stage 9控制Dockerfile默认base并CPU-only构建。
- **2.119 c349控制入口：** Dockerfile只切换默认base一行，新SHA=`1a9f1df3…68ae`；
  CPU-only输入门禁10/10通过，新base OCI/source/tree/layer身份一致，目标control tag
  尚不存在。2.119已实时写入；下一步完成门禁并发布，clean/published后才构建并审计
  34/33层控制镜像。
- **2.120 c349控制结果：** CPU-only构建得到control ID`731412e9…1f95`、34/33层；
  前33层、labels、entrypoint和source/tree匹配，固定runtime通过且CUDA未初始化。
  2.120已实时写入，11/11 evidence通过。下一步完成门禁并发布；随后按Phase
  1→5→7→9迁移正式配置/wrapper并执行CPU-only工具测试与递归verifier。

## 约束提醒

- 所有报告必须使用中文，且数据只能来自实际落地结果。
- 每个阶段实验结束后先更新报告，再进入下一阶段。
- 修改已有报告前必须重新读取，并检查章节序号与交叉引用。
- 长实验每 10 分钟记录一次进度。
- GPU 实验必须在固定 Docker 容器内执行，且分配前连续两次确认授权 GPU 空闲。
- 正式实验前必须提交并推送源码与配置；不得让主仓库 submodule 指向本地-only commit。
- `/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1` 及其他项目外目录仅可只读检查；禁止修改、暂存、提交、清理或重置。

## 当前阶段错误记录

- 第一次原始trace流式扫描的`docker run`遗漏`-i`，heredoc未传入Python，容器0.6秒
  退出且未读取trace、无结果/文件改动；修正stdin参数后的新命令完成rank0扫描。
- 8-rank正式校正第一次把项目只读挂载到`/workspace`，但冻结analysis记录宿主绝对
  路径，子进程在读取trace前`FileNotFoundError`退出、无结果。有效轮次改为相同绝对
  路径挂载并仅覆盖证据子目录rw，不修改分析逻辑。
- 同源trace归因证据完整生成后，generation外层命令仅因root创建的JSON权限为`0600`，
  宿主`sha256sum` permission denied而exit1。未重跑trace；新增原子写入`0644`合同、
  修正既有文件权限并验证运行版本hash可复现，JSON内容SHA保持不变。
- 2026-08-02恢复核对时误假设证据文件名为`manifest.sha256`/`build.log`，且假设宿主
  安装`jq`；实际为`evidence_manifest.sha256`/`build_attribution.log`且无`jq`。
  已改用实际文件名与`rg`/`sed`核对，6/6 manifest通过，未修改证据。

- 恢复审计首次沿用不存在的`/nfs/AE/txc/vllm`作为源码仓路径，`git -C`立即失败且
  没有文件改动；已从`.gitmodules`解析真实路径为`glm52_oscar_vllm`，后续不重复
  该路径假设。
- lazy-BF16 TDD红灯前检查发现固定c349/phase0镜像都未安装pytest；源码仓现存
  `.venv/bin/python`链接系统Python3.8且同样无法import pytest，不能作为有效环境。
  这些探针没有执行测试或修改环境；后续复用已验收的Python3.12 pytest依赖只读挂载，
  不重复上述入口。
- 扩大门禁时沿用的`/dev/shm/oscar-stage9-precommit-venv/bin/ruff`已经不存在，Ruff
  job在lint前exit127；同批固定容器compile已通过、完整CPU pytest为9 passed/
  19 skipped。后续从现存路径解析Ruff 0.14.0，不重复失效绝对路径。
- lazy-BF16首轮CPU-only SM80离线编译为无效轮次：固定venv中的
  `_virtualenv._Finder`优先把`vllm`解析到镜像内`/opt/vllm_glm52_v1`，而非当前
  `glm52_oscar_vllm`候选源码，因此产物与c349 baseline逐字节一致不能用于候选裁决。
  该轮证据保留在`formal_32k_b1_stage1_lazy_bf16_offline_v1`；下一步先CPU-only移除
  该单个meta path finder并预检实际import origin，仅在origin明确指向当前候选源码后
  以新run ID重编译。
- 记录上述无效轮次时，首次`apply_patch`因使用了不匹配的中间上下文而整体拒绝，三个
  planning文件均未改变；本次改为以各文件末尾实际内容为锚点，不重复该失败编辑方式。
- lazy-BF16结构化验证脚本首次因shell内嵌Python f-string转义产生`SyntaxError`，第二次
  因对v1兄弟目录使用`Path.relative_to(v2)`产生`ValueError`；两轮均未改变实验产物，
  第三次分别改为百分号格式化与`os.path.relpath`后14/14通过，不重复前两种写法。
- 本次恢复按`planning-with-files`先尝试用`uv run --no-project`执行
  `session-catchup.py`，但宿主机没有安装`uv`而exit127；命令未修改项目或实验状态。
  下一次改用该技能明确提供的系统`python3`回退入口，不重复不存在的`uv`命令。
- 覆盖定义过程记录首次把尚未实算的inactive tile数写成占位文本`129,?`；在任何报告
  编辑或发布前立即发现，并由`total-active=4,194,304-4,064,256=130,048`以及
  `tiles_without_bf16-history_only=4,062,683-3,932,635=130,048`双重实算后修正。
  后续派生数字一律先现场计算并交叉验证，不把占位值写入正式报告。
- 读取v3 ranking清单时命令误用不存在的`manifest.sha256`，前两份JSON已成功读取，
  最后一项才exit2；目录实际清单名为`evidence_manifest.sha256`。后续按`find`实列文件名
  读取，不重复猜测清单名；既有证据未被修改。
- 核对冻结常量时误猜`scripts/phase9/benchmark_oscar_decode.py`路径，`rg`该子命令
  exit2；同一只读命令的production launch和固定镜像检查已成功。后续从
  `benchmark_oscar_prefill.py`实际import目标解析常量，不重复不存在的路径。
- pending-scale红灯前环境探针误猜pytest target使用`lib/python3.12/site-packages`，
  该`ls`子命令exit2；venv目录和`bin/python`实际存在，尚未执行测试。下一步从venv
  实际目录结构或其解释器`sys.path`解析依赖位置，不重复硬编码该site-packages路径。
- pending-scale红灯首次容器命令把扁平pytest8.3.5 target整体置于`PYTHONPATH`最前，
  其中`typing_extensions 4.13.2`遮蔽固定镜像Pydantic所需`Sentinel`版本，collection
  在conftest导入时exit4，未执行目标测试。下一步核对已验收的另一份Python3.12
  pytest target或只注入pytest依赖而保留镜像typing_extensions，不重复当前组合。
- pytest8.3.5扁平target的`bin/python`也是指向宿主缺失`/usr/bin/python3.12`的坏入口，
  直接探针exit127；后续只把target作为只读包目录挂载，并使用固定镜像内Python3.12。
- pending-scale静态门禁首次Ruff0.14.0 check通过、固定Python compile通过，但
  format-check要求机械重排两个本轮触及文件而exit1；下一步只用同版本Ruff格式化这
  两个文件并审计diff，不手工猜格式或扩大范围。
- 记录静态全绿的首次三文件planning patch误把task_plan中的句子当作progress上下文，
  `apply_patch`原子拒绝且三文件均未修改；重新读取各文件真实末尾后分开锚定追加，
  不重复混用上下文。
- SASS首次通用opcode awk把谓词/寄存器列误识别为opcode，虽命令exit0但输出直方图无效；
  后续只用明确`STL/LDL`正则及resource log。精确结果仍确认candidate有0x0–0x24 local
  访问而baseline无local指令，不引用错误直方图计数。
- 第五轮top-k历史搜索模式含未转义反引号，shell在执行`rg`前因引号未闭合exit2；没有
  搜索结果或文件改动。下一次移除反引号并使用双引号安全模式，不重复原命令。

## 当前检查点（2.130）

- lazy-BF16候选已在CPU-only SM80门禁淘汰：stack `0→136 bytes/thread`、PTX loads
  `245→309`，promotion=false，未申请GPU；候选源码/测试改动已撤销，源码仓重新
  clean且HEAD=upstream=`c349e32e9…64b3`。
- `OSCAR精度与性能优化记录.md`已实时追加2.130；报告为8,785行/504,989 bytes、
  SHA256=`84f4e8d1…76a7`，章节1.1–1.5/2.1–2.130连续，术语、交叉引用、14/14
  validation、9/9 manifest及`git diff --check`通过。下一步只发布本检查点；发布前
  不开始下一轮机会排序。

## 当前候选（2.131 待报告）

- 第三轮CPU-only机会排序唯一选择`history_score_bf16_inputs_only`：仅把
  `query_rotated/history_values`在history score dot入口转为BF16；history value
  probability/value仍保持FP32/TF32，softmax/LSE、双FP32 accumulator、inverse
  rotation、BF16/RoPE与cache语义不变。
- 排序后的4,064,256个active tiles中4,049,424个含history，占99.635062358277%；
  旧c372 broad-BF16失败同时改变3个score和2个value dot，不能替代该隔离候选。
- ranking validation 12/12、manifest 2/2通过。离线晋升门禁要求二进制变化，
  shared/register/stack/PTX loads均不增；通过前禁止GPU。下一步先实时追加并发布
  报告2.131，发布前不改production源码。
- 报告2.131已追加并通过门禁：8,838行/508,425 bytes、SHA256=`65322761…b9c3`；
  章节1.1–1.5/2.1–2.131连续，术语、交叉引用、12/12 validation、2/2 manifest
  和`git diff --check`通过。下一步只提交并HTTPS推送四个文档。
- history-score-BF16 v1有效CPU-only SM80编译为shared 83,968、registers 255、
  stack 8 bytes/thread、PTX loads 245、cubin 198,960；因stack相对baseline 0增加而
  promotion=false。该候选描述允许把query_rotated保留FP32 load、只在score dot入口
  cast的等价v2 lowering；下一步以独立目录编译v2，不能覆盖或把v1写成通过。
- v2首次production patch的单行hunk误命中相邻`query` load，把BF16 query改成FP32而
  没有改变query_rotated；只读检查在测试/编译前发现并用带变量上下文的精确hunk修正。
  后续不使用无变量上下文的通用dtype替换。

## 当前候选（2.133 待报告）

- CPU-only v4 ranking选择`defer_bf16_accumulator_scale_across_history_only_tiles`：
  不改任何dot精度或softmax/LSE/history accumulator，只用逐head pending scale合并
  连续history-only tile对BF16 accumulator的缩放，并在下一BF16 tile或循环末尾应用。
- seed42冻结合成覆盖重放为history-only 3,932,635/4,064,256（96.761498%），
  57,972个run，p50/max 124/127 tiles；保守末尾无条件flush的整块缩放事件理论由
  4,064,256降为164,389（-95.955250%）。该数只属静态事件计数，不是性能实测。
- validation14/14、manifest3/3通过；下一步先重读报告并追加2.133，明确纠正2.131
  `all_history+mixed`与权威active-field的边界。报告发布前不修改production或用GPU。
- 报告2.133已追加并通过门禁：8,964行/516,530 bytes、SHA256=`f61a67a2…0455`；
  章节1.1–1.5/2.1–2.133连续，术语、交叉引用、14/14 validation、3/3 manifest与
  `git diff --check`通过。下一步只发布四个文档，发布前不开始production TDD。
- 2.133与planning已由主仓库`87c4fd6`通过GitHub HTTPS发布。下一步对pending-scale
  候选执行结构TDD、interpreter/完整CPU回归及SM80离线资源门禁；promotion前禁止GPU。
- pending-scale候选SM80资源为shared109,568、registers255、stack40、PTX loads245、
  cubin208,048；因stack0→40，promotion=false并已在GPU前淘汰。源码/测试已撤销，
  source clean c349。报告2.134已追加并通过9,030行/SHA`00d3f694…e190`、章节、
  术语、交叉引用、16/16 validation与7/7 manifest门禁；下一步只发布四个文档。
- 2.134与planning已由主仓库`f12df29`通过GitHub HTTPS发布；source保持clean c349，
  本候选关闭。后续如继续优化，先做下一轮CPU-only机会排序，阶段完成后仍先实时更新
  `OSCAR精度与性能优化记录.md`。
- history-score-BF16 v2资源与v1完全相同：shared 83,968、registers 255、stack 8、
  PTX loads 245，resource log SHA也相同；promotion=false。validation 17/17、
  manifest 10/10通过，候选源码/测试已撤销，源码仓clean且HEAD=upstream c349。
  下一步重读报告并追加2.132；报告发布前不开始下一轮排序。
- 报告2.132门禁通过：8,903行/512,331 bytes、SHA256=`7f7617a9…ecbd`；章节
  1.1–1.5/2.1–2.132连续，术语、交叉引用、17/17 validation、10/10 manifest与
  diff check通过。下一步只提交并HTTPS推送四个文档。
- c349正式负载已启用prefill top-k索引按token位置升序；attention侧截前N不保持DSA
  最高分集合，已排除。persistent top-k又硬固定K=2048；下一步只读确认正式后端，并对
  indexer端减K所需后端/缓冲/完整精度门禁做fail-closed排序，阶段完成后先更新报告。
- 正式prefill已确认使用支持动态K的legacy top-k；下一步追踪`index_topk`配置传播并形成
  top-k降档候选的改动面、准确性风险和可验证门禁。仅当候选定义与证据完整后更新报告
  2.135；报告发布前不改production、不申请GPU。
- `index_topk`已确认端到端一致传播；先生成1536/1024的32K理论减算与风险排序证据，
  首选较保守的1536仅在证据支持时进入实现。v5 ranking完成后重读并更新报告2.135，
  发布前不改正式launch；后续GPU候选需先过参数/配置测试与源码、配置已发布门禁。
- v5 ranking已完成：K1536被选中，但只形成静态工作量和契约证据，尚无精度/TTFT结论。
  当前步骤切换为重读并追加报告2.135、校验章节/术语/交叉引用/manifest后发布；完成发布
  前禁止修改正式launch或启动GPU。随后以TDD增加fail-closed HF override与runtime记录。
- 报告2.135门禁已通过；当前只发布`OSCAR精度与性能优化记录.md`与三份planning。
  发布成功并确认clean/upstream后，才进入K1536 launch/config TDD；GPU仍需配置发布和
  双空闲检查，当前不允许启动。
- 2.135已由主仓`44f7a25`发布。当前步骤：先为K1536写launch/config契约测试并获得有效
  红灯，再做最小实现、CPU静态回归、报告2.136与发布。完成这些门禁前不申请GPU。
- K1536契约测试有效红灯已取得。当前实现范围固定为四处：performance config声明
  `index_topk=1536`、candidate wrapper导出规范JSON、通用launch追加并解析校验
  `--hf-overrides`、Stage9 verifier核对配置与runtime；不得修改source kernel。
- K1536最小实现及18/18 CPU回归已通过。当前只做固定容器candidate dry-run与静态证据
  封存；成功后重读报告追加2.136并发布主仓改动。该阶段仍无GPU、精度或性能结果。
- 递归静态已68/68通过；完整dry-run仍停在无driver fixed import，不能记绿。下一步尝试
  只读driver library且无GPU device的CLI解析；不改变测试门限。无论结果如何，本阶段
  报告2.136必须区分静态通过与driver preflight待执行，并在发布前保持GPU禁用。
- CPU-only边界已确定：fixed import可绿但CLI设备推断必须有driver-injected设备环境。
  当前先生成结构化validation/manifest、重读并追加报告2.136、发布全部主仓改动；发布
  后才双检GPU并跑preflight，parsed args必须实际记录K1536才允许进入精度实验。
- K1536结构化CPU门禁已23/23通过且manifest3/3通过；当前阶段切换为修改前重读报告、
  只追加2.136并校验章节/术语/交叉引用/证据哈希。报告与launch/config/planning全部
  发布并确认clean前，不执行GPU空闲检查或driver-injected preflight。
- 报告2.136门禁已通过；当前只发布9个预期文件。发布后先补记发布身份并再次提交使
  主仓clean/upstream，再执行两次间隔至少60秒的8卡空闲检查；driver preflight必须
  实际解析`index_topk=1536`，否则禁止256题GSM8K smoke。
- 2.136与K1536启动/config已由主仓`20fe235`发布。当前只发布三份planning的发布身份；
  确认主仓与source均clean/upstream后，进入8卡双空闲检查和driver-injected preflight。
  preflight失败或parsed args不含精确K1536时，禁止启动精度实验。
- K1536 preflight双空闲已通过：`13:43:25Z/13:44:31Z`、间隔66秒、两次8/8卡
  0 MiB/0%且无compute process。当前先实时追加并发布报告2.137的空闲状态，恢复clean
  后只运行driver-injected preflight；parsed args门禁未通过前禁止精度实验。
- 报告2.137空闲阶段门禁已通过；当前只提交并HTTPS推送报告与三份planning。发布后
  即时复核GPU仍空闲，再以run ID `20260801T1344Z_candidate_topk1536_preflight_v1`
  运行固定8卡candidate preflight；不得同时启动模型服务或精度请求。
- K1536 driver-injected preflight已exit0，静态68/68、fixed import CUDA=false且
  parsed args精确K1536，GPU已释放。当前先封存实际3个JSON与外层log/exit/post证据，
  重读并补完报告2.137后发布；发布完成前不启动256题GSM8K smoke。
- preflight证据已26/26且manifest10/10通过；当前只重读并补完报告2.137，校验章节、
  术语、交叉引用和全部hash后发布报告/planning。主仓再次clean/upstream前禁止启动
  256题GSM8K smoke。
- 报告2.137完整preflight结果门禁已通过；当前只提交并HTTPS推送报告与三份planning。
  发布后先补记提交身份恢复clean，再为256题GSM8K smoke执行新的8卡双空闲检查；
  smoke期间每10分钟打印累计完成数与准确率。
- 2.137完整结果已由主仓`980e5c7`发布。当前只发布三份planning身份并恢复clean；随后
  读取冻结runner与既有256题baseline/OSCAR口径，确认fail-closed阈值后执行新双空闲
  检查。不得改变题集、seed、解码参数或评分器。
- smoke口径已固定为official_v5 fast 256题/8K/high/c16；保守门限为不低于历史BF16
  105/256，但该历史对照不满足严格paired，禁止声称统计等价。当前先确认Docker入口和
  K1536环境传播/落盘；无真实parsed args证据不得启动runner，正式晋升仍需全量精度/PPL。
- smoke容器入口TDD已完成：K1536与排序=1均从config fail-closed注入，固定协议与现有
  隔离runner复用，19/19工具及静态门禁通过。当前先重读并追加报告2.138、发布全部主仓
  改动；发布并clean前不执行smoke双空闲检查。
- 报告2.138门禁已通过；当前只提交并HTTPS推送6个预期文件。发布身份补记并再次发布
  后才开始smoke的新双空闲；唯一正式run必须实际记录K1536、排序=1与固定256/c16。
- 2.138与smoke入口已由主仓`3688e90`发布。当前只发布三份planning身份并恢复clean；
  然后以新run ID做两次至少间隔60秒的8卡空闲检查，报告更新发布后才启动正式smoke。
- smoke双空闲已通过：`14:02:59Z/14:04:05Z`、间隔66秒、两次8/8卡空闲。当前先
  追加并发布报告2.139启动状态；发布后即时复查空闲，再用唯一run ID
  `20260801T1403Z_candidate_topk1536_fast256_c16_v1`启动，按10分钟心跳监控。
- 报告2.139启动状态门禁已通过；当前只提交并HTTPS推送报告与三份planning。恢复
  clean/upstream后即时复查GPU并启动唯一smoke；运行时不得再修改主仓或报告。
- K1536精度smoke已启动但未形成精度结果：模型141/141分片加载并ready后，首批16个
  请求进入decode时因正式环境仍为persistent decode top-k而触发`k must be 2048`；
  EngineCore退出，runner记录256个`request_failed`、0 scored、`valid=false`。该轮
  只能判定为候选配置不兼容，不能判定精度升降。当前先封存失败证据并补完报告2.139；
  在报告发布和decode后端契约只读审计完成前，不复用run ID、不重跑GPU。
- 失败证据builder首轮validation按预期停止：错误把本轮Phase7静态检查数写成68而实际
  JSON为44，并用首字符筛GPU行时误纳入以`2`开头的时间戳。下一轮改为真实44项和
  `^[0-7],`设备行正则，对同一原始证据重建；不得把首轮称为通过。
- 报告2.139失败结果已补完并通过发布前门禁：9,339行/541,876 bytes、SHA256
  `da899f9b…0a9f`，章节1.1–1.5/2.1–2.139连续，术语、交叉引用、25/25 validation、
  15/15 manifest及diff check通过。当前只提交并HTTPS推送报告与三份planning；发布
  并恢复clean前不实施decode fallback或启动新的GPU轮次。
- 2.139失败结果与planning已由主仓`35bd4c6`通过GitHub HTTPS发布。当前只发布本条
  planning身份并恢复clean/upstream；随后只读审计legacy decode对K1536的支持、既有
  优化环境与wrapper覆盖关系，形成CPU-only候选前不使用GPU。
- decode只读审计已选择组合候选“K1536+legacy decode”：配置显式声明legacy，candidate
  链路逐项fail closed，通用serve仅在未设置时默认persistent，保持既有默认路径不变。
  当前先发布报告2.140；发布前不写测试或实现，后续必须先做K1536专门correctness，
  且TPOT可能因legacy回退，不能提前声称性能收益。
- 审计命令中曾调用宿主缺失的`jq`而输出command not found；同一配置值此前及随后均由
  Python/源码读取确认，未依赖该失败命令形成结论。后续不再使用宿主`jq`。
- 报告2.140门禁通过：9,392行/545,484 bytes、SHA256=`af888a4d…421d`，章节
  1.1–1.5/2.1–2.140连续，术语、2.139引用、源码hash与diff check通过。当前只发布
  报告和三份planning；恢复clean/upstream前不写契约测试。
- 2.140与planning已由主仓`bc8ce03`通过GitHub HTTPS发布。下一步先发布本条身份并
  恢复clean/upstream；后续从契约测试红灯开始实现K1536+legacy decode候选，每个阶段
  仍先实时更新报告，GPU前必须再发布CPU门禁结果。
- K1536+legacy decode契约测试已在固定c349、network none、CUDA不可见容器获得有效
  1 failed/0 errors红灯，失败精确指向config仅有prefill排序、缺decode backend=legacy。
  当前进入最小实现：只改config、candidate/accuracy入口、verifier和base默认保留逻辑，
  不改source CUDA/C++；实现后先做完整CPU回归并更新报告2.141。
- 最小实现后定向2/2与完整Stage9工具19/19已通过；首轮组合静态命令在py_compile尝试
  写只读project挂载的`__pycache__`时exit1，故后续JSON/bash项未执行，不能记全绿。
  下一步设置`PYTHONPYCACHEPREFIX=/tmp/pycache`重跑compile+JSON+shell，并做递归verifier
  dry-run；仍不使用GPU。
- CPU递归dry-run v1 stdout显示新旧candidate runtime检查通过，随后按预期因无
  `libcuda.so.1`停在fixed import；但输出root未绑定宿主，容器删除后JSON丢失，不能
  封存为证据。紧接的宿主检查又误用容器专属Python路径而失败。v2将新建run ID、显式
  bind宿主/dev/shm输出并用固定容器解析；不得声称v1已持久化或重复原命令。
- v2已持久化18,352-byte static JSON，fixed import为空并因缺libcuda退出；首次固定容器
  JSON解析命令再次漏`docker run -i`而没有执行Python，不能用其空stdout当验证。立即补
  `-i`对已落盘同一JSON复核，不重跑dry-run。
- K1536+legacy启动/config的CPU阶段最终通过：定向2/2、Stage9工具19/19、compile/JSON/
  三脚本语法/diff、递归静态69/69，结构化validation26/26、manifest5/5。source仍clean
  c349，未暴露GPU；parsed args与CUDA correctness仍deferred。当前先重读并追加报告
  2.141，发布报告与六个实现文件前不做driver preflight或GPU实验。
- 报告2.141门禁通过：9,466行/550,096 bytes、SHA256=`86161d3b…00a6`，章节
  1.1–1.5/2.1–2.141连续，术语、2.136/2.140引用、26/26 validation、5/5 manifest、
  六文件hash和diff check通过。当前只发布报告、planning与六个实现/测试文件；发布并
  恢复clean前不检查或使用GPU。
- 2.141、K1536+legacy实现与planning已由主仓`a4c383d`通过GitHub HTTPS发布。当前只
  发布本条身份并恢复clean/upstream；随后执行两次间隔至少60秒的8卡空闲检查，先实时
  更新报告2.142空闲阶段，再做driver-injected preflight和专项CUDA correctness。
- 发布身份已由`7378661`推送，两仓clean/upstream。K1536+legacy GPU阶段双空闲检查
  `14:41:18Z/14:42:24Z`间隔66秒，两次8/8卡均0 MiB/0%、无compute process，日志SHA
  `2f947f64…4ff5`。当前先追加并发布报告2.142空闲阶段；恢复clean前不启动preflight。
- 报告2.142门禁通过：9,487行/551,598 bytes、SHA256=`1322595c…a30d`，章节
  1.1–1.5/2.1–2.142连续，术语、2.141引用、idle日志与diff check通过。当前只发布
  报告和三份planning；恢复clean后即时复核GPU并运行driver preflight。
- 2.142与planning已由主仓`327c393`通过GitHub HTTPS发布。当前只发布本条身份恢复
  clean/upstream；随后以新run ID运行固定8卡driver-injected preflight，真实记录
  K1536+legacy+排序1且CUDA=false才进入专项correctness。
- K1536+legacy driver preflight `20260801T1444Z_topk1536_legacy_preflight_v1`
  已自然exit0：递归静态69/69、固定Python/Torch/Triton导入、TP8服务参数解析均通过，
  K1536、decode legacy和prefill排序1均匹配，CUDA未初始化，退出后8/8 GPU为0 MiB/0%。
  归档脚本首次执行误把固定Python ELF路径直接交给镜像`/bin/bash`入口，因而exit126；
  下一轮改用`/bin/bash -lc`执行同一脚本，不重跑preflight，也不重复该错误入口。
- builder第二轮已进入固定Python，但旧逻辑把含时间戳和`compute_processes:`标记的post
  文件总行数当GPU行数，正确地在`post_gpu_count/post_gpu_zero`失败。下一轮只筛8条设备
  行并独立核对compute列表为空；仍复用同一原始证据，不重跑preflight。
- 修正post解析后builder已29/29、manifest10/10通过。其后独立`sha256sum -c`误在项目
  根目录执行，而manifest条目是证据目录相对文件名，故只报10个文件找不到；下一轮切换
  到证据目录验证，不重建或修改证据。
- 正确cwd下10/10 manifest已全部OK。报告2.143已实时追加并通过门禁：9,540行、
  555,041 bytes、SHA256=`00d291bb…e969`，2.1–2.143连续，2.141/2.142交叉引用、术语、
  29/29 validation、10/10 manifest和diff check通过。下一步只发布四份文档；恢复clean
  前不运行专项CUDA correctness。
- 报告2.143与planning已由主仓提交`0a0b6c9f520ef89aec4530750944123236507042`通过
  GitHub HTTPS发布。当前只发布本条身份并恢复clean/upstream；下一步重新执行双空闲
  门禁，再以独立run ID运行K1536 legacy专项CUDA correctness。
- 2.143发布身份已由`f91beed`推送，两仓clean/upstream。专项CUDA脚本固定单卡GPU0、
  K1536、8K insertion/32K radix及random/10LSBits共4例，固定容器compile通过，脚本SHA
  `9295e8a2…a210`。新双空闲检查`14:56:50Z/14:57:56Z`间隔66秒，两次8/8卡均
  0 MiB/0%、无compute process，日志SHA=`b2fe6a61…87a9`。当前先实时更新并发布报告
  2.144；恢复clean前不启动correctness。
- 报告2.144门禁通过：9,573行、557,271 bytes、SHA256=`f7fff6a4…350d`，2.1–2.144
  连续，2.143交叉引用、术语、脚本/idle hash和diff check通过。下一步只发布报告与三份
  planning；恢复clean/upstream前不运行CUDA专项。
- 报告2.144与planning已由主仓提交`0ee13b522194e7b2497153de67e2326c93a02f41`通过
  GitHub HTTPS发布。当前只发布本条身份并恢复clean/upstream；随后即时复核GPU0并运行
  已冻结的4例K1536 legacy CUDA专项。
- 2.144发布身份由`fb6d6d8`推送后两仓clean。单卡专项run
  `20260801T1500Z_topk1536_legacy_cuda_correctness_v1`自然exit0，4/4 case通过；8K
  insertion与32K radix的random/10LSBits均为1536唯一索引、sets/value match、max abs=0。
  退出瞬间GPU0显存0且无compute但利用率采样9%；15秒后的`15:01:06Z`复核8/8卡均
  0 MiB/0%、无compute。当前先封存证据并实时更新报告2.145，不启动256题smoke。
- 专项证据已在固定c349、network none、CUDA不可见容器完成33/33 validation与10/10
  manifest，独立`sha256sum -c`全通过；builder/contract/validation/manifest SHA依次为
  `d15dcbb0…70d`/`c5b29419…3ad`/`78f09a8b…dc5`/`c616a394…90b`。下一步重读报告并
  追加2.145，发布前不启动256题smoke。
- 报告2.145门禁通过：9,629行、560,638 bytes、SHA256=`5744f77f…60ac`，2.1–2.145
  连续，2.144引用、术语、33/33 validation、10/10 manifest、原始hash与diff check
  通过。下一步只发布四份文档；恢复clean前不启动256题smoke。
- 报告2.145与planning已由主仓提交`cdabf53c37068ee7a9a258ef768d7dcd60ed9ca6`通过
  GitHub HTTPS发布。当前只发布本条身份恢复clean/upstream；下一步为256题smoke执行
  新双空闲门禁并先实时更新报告。
- 2.145发布身份由`55804a9`推送后两仓clean。256题smoke双空闲检查为
  `15:05:14Z/15:06:21Z`，间隔67秒，两次8/8卡均0 MiB/0%、无compute，日志SHA
  `ffc5334b…cc0a`。当前先实时追加并发布报告2.146；恢复clean前不启动模型。
- 报告2.146门禁通过：9,650行、562,126 bytes、SHA256=`2c3b8924…2140`，2.1–2.146
  连续，2.145引用、术语、idle hash与diff check通过。下一步只发布四份文档；恢复
  clean/upstream前不启动模型。
- 报告2.146与planning已由主仓提交`2e59274cf4fa44d4a541a90b9e8d7c7b981420fb`通过
  GitHub HTTPS发布。当前只发布本条身份恢复clean/upstream；随后即时复核8卡并以新
  run ID启动256题smoke，按10分钟打印进度与累计精度。
- 256题首次启动run `20260801T1511Z_candidate_topk1536_legacy_fast256_c16_v1`
  在0.3秒内由外层正式门禁拒绝：遗漏`FORMAL_RUN=1`，exit1，未创建容器或加载模型；
  前后8/8卡均0 MiB/0%、无compute，没有精度结果。下一步先实时追加并发布报告2.147，
  之后用新run ID和显式`FORMAL_RUN=1`，不得复用或重复失败命令。
- 报告2.147门禁通过：9,675行、563,677 bytes、SHA256=`3e71dbea…79e9`，2.1–2.147
  连续，2.146引用、术语、失败证据hash与diff check通过。下一步只发布四份文档；恢复
  clean/upstream后重新双空闲并使用修正命令。
- 报告2.147与planning已由主仓提交`c7354884a825a0050bf904eb528e96dbc0554ea8`通过
  GitHub HTTPS发布。当前只发布本条身份恢复clean/upstream；随后新双空闲并用显式
  `FORMAL_RUN=1`、新run ID启动有效256题轮次。
- 2.147发布身份由`f3a4219`推送后两仓clean。重试双空闲检查为
  `15:10:35Z/15:11:41Z`，间隔66秒，两次8/8卡均0 MiB/0%、无compute，日志SHA
  `a86cbc38…f8c4`。当前先实时追加并发布报告2.148；恢复clean前不启动模型。
- 报告2.148门禁通过：9,693行、564,886 bytes、SHA256=`b86b9baf…4782`，2.1–2.148
  连续，2.147引用、术语、idle hash、发布身份和diff check通过。下一步只发布四份文档；
  恢复clean/upstream前不启动模型。
- 报告2.148与planning已由主仓提交`ef2c776392ff3f00058325d5530a7588590c4fc9`通过
  GitHub HTTPS发布。当前只发布本条身份恢复clean/upstream；随后立即以显式正式标志和
  新run ID启动有效256题轮次。
- [x] 完成 `_rotate_latent_kernel` 当前源码/历史取舍审计：确认既有 contiguous-inverse 优化已生效；M=32 仅保留为 16384-row 路径候选，优先级低于 attention 主差距
- [x] 继续拆解当前 source-matched prefill attention 的 5699.560312 ms 差距：确认差距集中于 1248 次 `_mixed_sparse_prefill_stage1` 的单次成本，并复核既有等价优化反证
- [x] 将 rotation/stage1 当前源码与历史反证审计形成正式报告阶段并发布：主仓提交`82a87c8`已通过GitHub HTTPS推送
- [x] 发布2.173身份`e15f33b`并恢复clean/upstream
- [ ] CPU-only排序更低K与仅16,384-row M=32两个未闭合方向，冻结单一下一候选合同
  - [x] 新增独立ranking builder，明确K=768算法不等价、GPU未开放和外推非实测边界
  - [x] 固定c349控制环境最终builder 16/16通过，3/3 manifest独立复算通过并封存最终hash
  - [x] CPU-only ranking与K=768 fail-closed合同已作为报告2.174发布：主仓提交`f4c9fbe`已通过GitHub HTTPS推送（结构化证据按既有规则保留在本地忽略目录）
  - [x] 发布2.174身份`a907164`并恢复clean/upstream
  - [x] K=768最小配置CPU-only TDD：目标红灯1 failure；绿灯目标1/1、完整19/19、bash/python语法与diff门禁全部通过
  - [x] 报告2.175与K=768最小配置已由主仓提交`28ce067`通过GitHub HTTPS发布
  - [x] 发布2.175身份`02850ac`并恢复clean/upstream
  - [ ] K=768标准preflight固定`--gpus all`：先完成并发布两次间隔>=60秒的8卡空闲门禁，再运行driver preflight；模型/CUDA实验仍未开放
    - [x] 正式first/second与报告2.176已由主仓提交`1d7df68`通过GitHub HTTPS发布
    - [x] 发布2.176身份`b29e270`并即时复核8/8卡全空闲
    - [x] K=768 preflight有效独立复核69/69、K=768、两处cuda_initialized=false，三JSON/hash及post-GPU完整
    - [x] 报告2.177 preflight结果已由主仓提交`c2e9cb0`通过GitHub HTTPS发布
    - [x] 发布2.177身份`47f7a8e`并恢复clean/upstream
    - [x] K=768 4例脚本已创建，CPU-only逐行等价、compile/AST、环境与断言合同全部通过
    - [x] 报告2.178专项脚本合同已由主仓提交`79a3a0b`通过GitHub HTTPS发布
    - [x] 发布2.178身份`0d4257c`并恢复clean/upstream
    - [ ] 执行K=768 GPU0专项前新的双空闲门禁，结果先实时写报告并发布
      - [x] 正式first/second已独立解析通过并固化hash，报告2.179已追加且一致性检查通过
      - [x] 报告2.179与planning已由主仓提交`a72efda`通过GitHub HTTPS发布
      - [x] 发布2.179身份`23a1954`并恢复clean/upstream；即时复核8/8卡全空闲
      - [ ] K=768 GPU0 4例专项
        - [x] 首次run因镜像`/bin/bash` ENTRYPOINT与显式Python参数冲突而exit126，未进入Python/CUDA且无result；报告2.180已实时追加
        - [x] 报告2.180与planning已由主仓提交`cb2166d`通过GitHub HTTPS发布，发布身份`1b3fdbf`
        - [x] 有效v2用显式Python entrypoint自然exit0；K=768在8K/32K×random/10LSBits四例均与reference一致，独立JSON复核通过
        - [x] 报告2.181与planning已由主仓提交`cc66320`通过GitHub HTTPS发布，发布身份`54e8ad4`
      - [ ] K=768同协议256题快速精度筛选
        - [x] 新双空闲门禁05:56:10Z/05:57:15Z、间隔65秒，16条全idle且compute为空；报告2.182已追加
        - [x] 报告2.182与planning已由主仓提交`f604b11`通过GitHub HTTPS发布
        - [x] 发布2.182身份`88ad032`并恢复clean/upstream；启动前8/8卡全空闲
        - [x] 有效run `20260802T0600Z_candidate_topk768_legacy_fast256_c16_v1`已完成并因97<105淘汰
          - [x] official_v5静态/namespace preflight、入口内额外双空闲与真实CLI解析通过；K=768、TP8、8K、batch并发16、legacy/sort合同匹配
          - [x] 只读累计精度monitor已启动，每600秒记录completed/correct/accuracy/failures/truncated及GPU状态
          - [x] 启动状态planning已由主仓提交`410fd46`通过GitHub HTTPS发布（runtime仍固定启动时主仓`88ad032`）
          - [x] 首个10分钟节点9/256、6正确、66.666667%、0 request failure、2 extraction failure、0截断；已纠正monitor的failure标签语义并保留原始误标签行
          - [x] 第二个10分钟节点仍为9/256、6正确、66.666667%；server 16 running/0 waiting、62.4–76.8 token/s，3秒dmon 61%–98%，确认长输出批次仍在计算
          - [x] 第三个10分钟节点仍为9/256、6正确、66.666667%；8 worker存活，server 16 running/0 waiting、62.4–78.4 token/s、KV cache 16.9%–17.3%，无fatal/OOM
          - [x] 第四个10分钟节点36/256、14正确、38.888889%、0 request failure、19 extraction failure、16截断；稍后只读38题为15正确，服务持续16 running/0 waiting
          - [x] 第五个10分钟节点47/256、23正确、48.936170%、0 request failure、20 extraction failure、16截断；服务16 running/0 waiting、78.4–80.0 token/s
          - [x] 第六个10分钟节点仍为47/256、23正确、48.936170%；8 worker存活，服务16 running/0 waiting、78.4–80.0 token/s、KV cache 16.6%–17.0%
          - [x] 第七个10分钟节点74/256、33正确、44.594595%、0 request failure、35 extraction failure、30截断；服务16 running/0 waiting、75.2–78.4 token/s
          - [x] 第八个10分钟节点77/256、34正确、44.155844%、0 request failure、37 extraction failure、32截断；服务16 running/0 waiting、78.4–80.0 token/s
          - [x] 第九个10分钟节点仍为77/256、34正确、44.155844%；8 worker存活，服务16 running/0 waiting、78.4 token/s、KV cache 17.4%–17.8%
          - [x] 第十个10分钟节点107/256、44正确、41.121495%、0 request failure、53 extraction failure、47截断；服务16 running/0 waiting、55.9–78.4 token/s
          - [x] 第十一个10分钟节点117/256、50正确、42.735043%、0 request failure、55 extraction failure、48截断；服务16 running/0 waiting、78.4 token/s
          - [x] 第十二个10分钟节点119/256、52正确、43.697479%、0 request failure、55 extraction failure、49截断；服务16 running/0 waiting、78.4 token/s
          - [x] 第十三个10分钟节点139/256、55正确、39.568345%、0 request failure、67 extraction failure、62截断；低GPU节点经服务与3秒dmon确认仅为批次切换
          - [x] 第十四个10分钟节点141/256、55正确、39.007092%、0 request failure、69 extraction failure、64截断；服务16 running/0 waiting、78.4 token/s
          - [x] 第十五个10分钟节点146/256、57正确、39.041096%、0 request failure、71 extraction failure、66截断；服务16 running/0 waiting、68.7–78.4 token/s
          - [x] 第十六个10分钟节点173/256、69正确、39.884393%、0 request failure、83 extraction failure、78截断；服务16 running/0 waiting、55.9–80.0 token/s
          - [x] 第十七个10分钟节点177/256、70正确、39.548023%、0 request failure、85 extraction failure、80截断；服务16 running/0 waiting、78.4–80.0 token/s
          - [x] 第十八个10分钟节点181/256、72正确、39.779006%、0 request failure、87 extraction failure、82截断；服务16 running/0 waiting、76.7–78.4 token/s
          - [x] 第十九个10分钟节点218/256、87正确、39.908257%、0 request failure、100 extraction failure、94截断；服务16 running/0 waiting、70.3–80.0 token/s
          - [x] 第二十个10分钟节点223/256、90正确、40.358744%、0 request failure、102 extraction failure、96截断；服务16 running/0 waiting、78.4–80.0 token/s
          - [x] 第二十一个10分钟节点227/256、91正确、40.088106%、0 request failure、104 extraction failure、99截断；服务16 running/0 waiting、78.4–80.0 token/s
          - [x] 第二十二个10分钟节点244/256、97正确、39.754098%、0 request failure、114 extraction failure、109截断；尾批服务由11降至10 running、0 waiting
          - [x] 第二十三个10分钟节点247/256、97正确、39.271255%、0 request failure、117 extraction failure、112截断；尾批9 running、0 waiting
          - [x] 第二十四个10分钟节点248/256、97正确、39.112903%、0 request failure、118 extraction failure、113截断；尾批由8降至7 running、0 waiting
          - [x] 256/256自然完成：97正确、37.890625%、0 request failure、126 extraction failure、121截断；outer exit0且8卡全释放
          - [x] 固定c349容器独立复算与证据封存通过：53/53 validation、32/32 manifest，分类为`performance_candidate_screen_failed`
          - [x] 重读并实时更新报告2.183；K=768正确数低于105门槛，不启动其32K性能测试
          - [x] 报告2.183、planning与33个独立证据文件已由主仓提交`c36d49a`通过GitHub HTTPS发布
          - [x] 发布身份`8eb2ee9`并恢复两仓clean/upstream
        - [ ] 下一候选`prefill K768 / decode K1024`的CPU-only合同与排序
          - [x] 统一K数据流、混合batch风险及仓内split参考实现审计完成
          - [x] 固定c349容器22/22 ranking checks、fresh容器3/3 manifest与数值复核通过
          - [x] 重读并实时追加报告2.184；2.1–2.184连续，引用、术语、算术与diff门禁通过
          - [x] 报告2.184、ranking与planning已由主仓提交`ddd38b8`通过GitHub HTTPS发布
          - [x] 发布本身份并恢复clean/upstream：身份`1dc7236`已通过GitHub HTTPS推送
          - [x] CPU-only TDD与最小实现：基础K1024、prefill K768、decode K1024、mixed batch分段及控制面透传；source`1e768aef6`已发布
          - [x] 重读并实时追加报告2.185；2.1–2.185连续，引用、术语、实测边界与diff门禁通过
          - [x] 发布报告2.185、控制面、planning和source gitlink：主仓`b65c9b0`、source`1e768aef6`均已通过GitHub HTTPS推送
          - [x] 发布2.185身份并恢复clean/upstream：身份`087d2f5`已通过GitHub HTTPS推送
          - [x] CPU-only审计新镜像链：必须依次重建Phase 6 source layer/overlay与Stage 9控制镜像
          - [x] 重读并实时追加报告2.186；2.1–2.186连续，引用、术语与diff门禁通过
          - [x] 发布2.186审计结论：主仓`3496e4b`已通过GitHub HTTPS推送
          - [x] 发布2.186身份并恢复clean/upstream：身份`38a7604`已通过GitHub HTTPS推送
          - [x] Phase 6输入CPU-only TDD与最小迁移：目标红灯1 failure，绿灯2/2及compile/JSON/hash/diff通过
          - [x] 重读并实时追加报告2.187；2.1–2.187连续，引用、术语与diff门禁通过
          - [x] 发布2.187与Phase 6输入：主仓`f4ec6e6`已通过GitHub HTTPS推送
          - [x] 发布2.187身份并恢复clean/upstream：身份`3fab103`已推送；无效rootfs Python边界由`c1ad82e`发布
          - [x] 两次daemonless OCI构建自然exit0；manifest/config/layer/diff ID/index逐字节确定性一致
          - [x] 重读并实时追加报告2.188；2.1–2.188连续，引用、术语与diff门禁通过
          - [x] 发布2.188双构建：主仓`3417b2e`已通过GitHub HTTPS推送
          - [x] 发布2.188身份并恢复clean/upstream：身份`40be3ff`已通过GitHub HTTPS推送
          - [x] v1独立verifier自然exit0：base/source/rotation/runtime/native/无whiteout全通过
          - [x] 重读并实时追加报告2.189；2.1–2.189连续，引用、术语与diff门禁通过
          - [x] 发布2.189 verification：主仓`6c831df`已通过GitHub HTTPS推送
          - [x] 发布2.189身份并恢复clean/upstream：身份`214771a`已通过GitHub HTTPS推送
          - [x] 新overlay精确创建6个Phase0 native symlink；4,744 regular/6 symlink与目标hash通过
          - [x] CPU-only source import通过，split-K常量/metadata正确且CUDA未初始化；native import因无libcuda延后
          - [x] 重读并实时追加报告2.190；2.1–2.190连续，引用、术语与diff门禁通过
          - [x] 发布2.190 overlay/source import：主仓`df7e9d2`已通过GitHub HTTPS推送
          - [x] 发布2.190身份并恢复clean/upstream：身份`7d397ac`已通过GitHub HTTPS推送
          - [x] split-K v1 OCI已导入daemon；image ID、tag、33层、末层diff-ID与8项labels审计全部通过
          - [x] 重读并实时追加报告2.191；2.1–2.191连续，证据hash、引用、术语与diff门禁通过
          - [x] 发布2.191 daemon导入结果：主仓`a2cb329`已通过GitHub HTTPS推送
          - [x] 发布2.191身份并恢复clean/upstream：身份`1d0dfb1`已通过GitHub HTTPS推送
          - [x] Stage 9 Dockerfile只迁移新Phase 6 base：目标红灯1 failure，绿灯1/1及完整22/22，输入审计13/13通过
          - [x] 重读并实时追加报告2.192；下一步完成一致性门禁并发布
          - [x] 发布2.192与Stage 9 Dockerfile输入切换：主仓`2d5bf95`已通过GitHub HTTPS推送
          - [x] 发布2.192身份并恢复clean/upstream：身份`f2b0108`已通过GitHub HTTPS推送
          - [x] CPU-only构建新control image：`c92a1245...a12e`、34/33层继承、identity 10/10与CPU runtime复核通过
          - [x] 重读并实时追加报告2.193；下一步完成一致性门禁并发布
          - [x] 发布2.193 Stage 9 control构建结果：主仓`ce4d19b`已通过GitHub HTTPS推送
          - [x] 发布2.193身份并恢复clean/upstream：身份`e311d60`已通过GitHub HTTPS推送
          - [x] driver/native import前双空闲门禁65秒、16条全idle、两个compute区段为空，独立7/7通过
          - [x] 重读并实时追加报告2.194；下一步完成一致性门禁并发布
          - [x] 发布2.194双空闲门禁：主仓`9f46669`已通过GitHub HTTPS推送
          - [x] 发布2.194身份并恢复clean/upstream：身份`07ddf0e`已通过GitHub HTTPS推送
          - [x] 即时复核8卡全空闲；固定GPU0 driver/native import、13/13 validation和退出后全释放通过
          - [x] CPU-only实测补全并生成canonical `runtime_import.json`，固定容器复核通过
          - [x] 重读并实时追加报告2.195；下一步完成一致性门禁并发布
          - [x] 发布2.195 native runtime import结果：主仓`ebb1500`已通过GitHub HTTPS推送
          - [ ] 发布2.195身份并恢复clean/upstream；随后迁移Phase 7 manifest/入口与Phase 9 performance matrix/container身份
- [ ] driver-visible native import后再迁移Phase 7 manifest及剩余Stage 7/9运行身份
- [x] 833固定256题正式run
  `20260803T0226Z_candidate_prefusion_rollback_fast256_c16_v1`已于02:26:34Z启动；
  外部即时空闲与入口内部双空闲均通过，static/namespace/递归身份preflight通过。
- [ ] 每600秒记录该run的completed/correct/accuracy/failures/truncated/GPU状态；精度达到
  BF16 baseline 105/256前不得启动32K/batch1性能复测。
  - [x] 首个有效纠正节点：1/256完成、0正确、0.000000%，0 request failure、1 answer
    extraction failure、0截断；宿主权限导致的无效0/256行已保留并明确排除。
  - [x] 20分钟节点仍为1/256、0正确、0.000000%；16 running/0 waiting、64.0–78.4
    token/s，8卡74%–98%，无fatal/OOM。
  - [x] 30分钟节点仍为1/256、0正确、0.000000%；16 running/0 waiting、64.0–76.8
    token/s、KV 16.3%–16.5%，错误扫描无异常。
  - [x] 40分钟节点17/256、0正确、0.000000%，17提取失败、16截断；下一批16题已启动，
    pre-fusion回退暂未展现精度恢复，继续跑完256题。
  - [x] 高置信根因候选已定位：`[M,1024][:,:768]`的stride0仍为1024，而native prefill
    top-k按`rowIdx*768`写，Triton attention按stride1024读；正式runtime确实走该分支。
  - [ ] run终局后以CPU TDD、GPU最小stride复现验证候选根因，再测试连续768临时输出并
    拷回共享buffer的最小Python修复；修复精度门禁通过前不得重新运行32K性能。
    - [x] source/runtime地址合同CPU-only 8/8通过；split-K仅第0行offset匹配，统一K768/
      K1024各4/4匹配。首次错误路径manifest保留，最终9/9通过；GPU验证仍未执行。
  - [x] 50分钟节点仍为17/256、0正确、0.000000%；第二批16 running/0 waiting、78.4
    token/s、KV 9.5%–9.8%，无异常。
  - [x] 无GPU control image实测非连续view stride(8,1)，默认/显式empty_like均连续
    stride(2,1)；候选冻结显式contiguous_format。宿主无Torch失败保留，manifest7/7。
  - [x] 60分钟节点仍为17/256、0正确、0.000000%；第二批16 running/0 waiting、
    78.4–80.0 token/s、KV 17.5%–17.8%，无异常。
  - [x] 280分钟节点为158/256、0正确、0.000000%，158提取失败、144截断；
    8卡77%–97%，容器继续运行。报告章节校验首次错用`## 2.x`规则而得到0个标题，
    该轮无效且未改文档；已改为报告实际的`### 2.x`规则重新校验。
  - [x] 290分钟节点为173/256、0正确、0.000000%，173提取失败、158截断；
    8卡74%–98%，相对280分钟新增15题全错。
  - [x] 300分钟节点为175/256、0正确、0.000000%，175提取失败、160截断；
    8卡68%–98%，相对290分钟新增2题全错。
  - [x] 310分钟节点仍为175/256、0正确、0.000000%，175提取失败、160截断；
    8卡74%–98%，当前批次继续生成。
  - [x] 320分钟节点为190/256、0正确、0.000000%，190提取失败、174截断；
    8卡78%–98%，新增15题全错。
  - [x] 330分钟节点为193/256、0正确、0.000000%，193提取失败、176截断；
    8卡72%–97%，新增3题全错。
  - [x] 340分钟节点为194/256、0正确、0.000000%，194提取失败、176截断；
    8卡73%–98%，新增1题未截断但提取失败。
  - [x] 350分钟节点为207/256、0正确、0.000000%，207提取失败、188截断；
    8卡74%–98%，新增13题全错。
  - [x] 360分钟节点为210/256、0正确、0.000000%，210提取失败、191截断；
    8卡72%–98%，新增3题全错且全部截断。
  - [x] 370分钟节点为223/256、0正确、0.000000%，223提取失败、204截断；
    8卡55%–97%，新增13题全错且全部截断。
  - [x] 380分钟节点仍为223/256、0正确、0.000000%，223提取失败、204截断；
    8卡79%–98%，当前批次继续生成。
  - [x] 390分钟节点为227/256、0正确、0.000000%，227提取失败、208截断；
    8卡72%–98%，新增4题全错且全部截断。
  - [x] 400分钟节点为239/256、0正确、0.000000%，239提取失败、220截断；
    8卡73%–97%，新增12题全错且全部截断。
  - [x] 410分钟节点为240/256、0正确、0.000000%，240提取失败、221截断；
    8卡75%–98%，新增1题全错且截断。
  - [x] 420分钟节点为244/256、0正确、0.000000%，244提取失败、224截断；
    8卡63%–98%，新增4题全错、其中3题截断，尾批12 running。
  - [x] 430分钟节点为255/256、0正确、0.000000%，255提取失败、235截断；
    8卡25%–97%，新增11题全错且全部截断，仅剩1 running。
  - [x] 440分钟节点仍为255/256、0正确、0.000000%，255提取失败、235截断；
    8卡44%–98%，最后1个请求继续生成，尚无终局产物。
  - [x] 正式run于09:47:27Z自然exit0：256/256、0正确、0.000000%、0请求失败、
    236截断、平均completion tokens 7,436.65625、总时长26,005.380892秒；容器已删除，
    outer release check 8/8 idle，随后稳定采样8卡0 MiB/0%且compute空。终局未过精度门禁。
  - [x] 终局独立复核20/20、最终manifest 279/279通过；正式报告2.247已实时追加，
    待章节、术语、证据hash和diff门禁通过后发布。发布前不启动GPU stride复现。
  - [x] 2.247已由`ebd131f2...c1`发布；新GPU双空闲门禁为09:56:43Z/09:57:54Z、
    间隔71秒，两轮8卡均0 MiB/0%、compute空，固定833镜像验证10/10通过。
    下一步先实时写入并发布门禁报告，再运行固定GPU0 stride最小复现。
  - [x] 2.248已由`8ea44d6d...39e`发布；固定GPU0 native复现exit0，非连续stride(4,1)
    输出确定错位，连续stride(2,1)临时张量+copy-back与reference完全一致，短行sentinel与
    buffer尾列保护通过。split-K stride已由静态候选升级为GPU数值验证根因；下一步先实时
    发布专项证据，再迁移最小补丁。
  - [x] GPU专项独立validation 20/20、manifest 11/11通过；正式报告2.249已实时追加，
    待章节、术语、hash和diff门禁通过后发布。发布前source主工作树保持833 clean/upstream。
  - [x] 冻结patch已逐字节迁移到833 source主工作树；CPU合同2/2、compile、ruff 0.14.0、
    diff-check及独立15/15通过，manifest 19/19通过。报告2.250已实时追加；下一步校验发布，
    然后精确提交source两文件。
  - [x] source `ea8ae6b7758ae2b4db7cae44d638ae5de80148ac`已提交并HTTPS发布；commit
    patch与candidate一致，14/14 validation、7/7 manifest通过。报告2.251已实时追加，
    主仓下一提交一并更新gitlink；之后迁移活动身份并构建新runtime。
  - [x] Phase6输入TDD有效红灯1/1、绿灯1/1、完整builder 2/2；仅候选manifest、
    Dockerfile和目标测试迁移到ea8 commit/tree/tag，Phase5/7/9保持833。下一步闭合证据、
    实时发布报告，再执行CPU-only OCI builder。
  - [x] Phase6输入独立validation 13/13、manifest 10/10通过；报告2.252已实时追加，
    下一步校验并发布三处输入与planning，随后运行CPU-only builder/verifier。
- [ ] ea8 canonical fixed256 v2终局精度门禁与后续32K/batch1性能复测
  - [x] 2.319的30分钟节点已由`b2cae8e6c36863e8798c51d59c0cfba5a9df03fd`通过
    GitHub HTTPS发布，local/upstream一致。
  - [x] 40分钟固定截止为20/256、8正确、40.000000%当前精度、3.125000%全量精度、
    0 invalid、8 truncated；26/26 validation与8/8 manifest通过，报告2.320已实时追加。
  - [x] 2.320章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁通过；
    报告18,694行/1,161,887 bytes、SHA=`a51cda55...d3500`。
  - [x] 2.320已由`04011fd8e4f6b3bdd498b8727a1cd4a825db89e2`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为50分钟。
  - [ ] 继续每600秒记录并发布精度；正式终局达到105/256前禁止32K/batch1性能复测。
  - [x] 50分钟固定截止为34/256、12正确、35.294118%当前精度、4.687500%全量精度、
    0 invalid、16 truncated；26/26 validation与8/8 manifest通过，报告2.321已实时追加。
  - [x] 2.321章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁通过；
    报告18,730行/1,164,244 bytes、SHA=`218d53c9...33594`。
  - [x] 2.321已由`9f106a1bd40b427c42f770ee08cce6783fe3ccf3`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为60分钟。
  - [x] 60分钟固定截止仍为34/256、12正确、35.294118%当前精度、4.687500%全量精度、
    0 invalid、16 truncated；26/26 validation与8/8 manifest通过，报告2.322已实时追加。
  - [x] 2.322章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁通过；
    报告18,764行/1,166,454 bytes、SHA=`bae96fde...53176`。
  - [x] 2.322已由`cc1c1a426b54ccd55b72853efda6cd9e91597262`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为70分钟。
  - [x] 70分钟固定截止为48/256、16正确、33.333333%当前精度、6.250000%全量精度、
    0 invalid、26 truncated；26/26 validation与8/8 manifest通过，报告2.323已实时追加。
  - [x] 2.323章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁通过；
    报告18,800行/1,168,770 bytes、SHA=`19a11376...86a40`。
  - [x] 2.323已由`c0f07d892359a3d8cdd48ba7c013927294bc20f8`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为80分钟。
  - [x] 80分钟固定截止为67/256、25正确、37.313433%当前精度、9.765625%全量精度、
    0 invalid、32 truncated；26/26 validation与8/8 manifest通过，报告2.324已实时追加。
  - [x] 2.324章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁通过；
    报告18,836行/1,171,132 bytes、SHA=`f8d72822...08b73`。
  - [x] 2.324已由`ffe2bba2286d34224425cc0460521c4d535497ad`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为90分钟。
  - [x] 90分钟固定截止仍为67/256、25正确、37.313433%当前精度、9.765625%全量精度、
    0 invalid、32 truncated；26/26 validation与8/8 manifest通过，报告2.325已实时追加。
  - [x] 2.325章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁通过；
    报告18,870行/1,173,338 bytes、SHA=`73055ba8...b3900`。
  - [x] 2.325已由`a84cf0eabccdda14e141c6769438d2b76eb77717`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为100分钟。
  - [x] 100分钟固定截止为77/256、28正确、36.363636%当前精度、10.937500%全量精度、
    0 invalid、40 truncated；26/26 validation与8/8 manifest通过，报告2.326已实时追加。
  - [x] 2.326章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁通过；
    报告18,906行/1,175,740 bytes、SHA=`40f8594a...56126`。
  - [x] 2.326已由`1f1af26c6360b434eca6a4f4d4dd648fa46e13a6`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为110分钟。
  - [x] 110分钟固定截止为101/256、40正确、39.603960%当前精度、15.625000%全量精度、
    0 invalid、48 truncated；26/26 validation与8/8 manifest通过，报告2.327已实时追加。
  - [x] 2.327章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁通过；
    报告18,942行/1,178,170 bytes、SHA=`4775836e...a8139`。
  - [x] 2.327已由`885e460fdbc46943e9df1dd112e2297383adb7d6`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为120分钟。
  - [x] 120分钟固定截止为102/256、41正确、40.196078%当前精度、16.015625%全量精度、
    0 invalid、48 truncated；26/26 validation与8/8 manifest通过，报告2.328已实时追加。
  - [x] 2.328章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁通过；
    报告18,978行/1,180,524 bytes、SHA=`f233f583...af8d0`。
  - [x] 2.328已由`5092197659d445199ccada858f4b29ce7ad4e1ec`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为130分钟。
  - [x] 130分钟固定截止为111/256、42正确、37.837838%当前精度、16.406250%全量精度、
    0 invalid、56 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.329已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,014行/1,182,970 bytes、SHA=`a7d497a3...36aad`。
  - [x] 2.329已由`ae0d1ec8f0ca313a57cc383d7f1de868a1a4e6ed`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为140分钟。
  - [x] 140分钟固定截止为133/256、48正确、36.090226%当前精度、18.750000%全量精度、
    0 invalid、64 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.330已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,050行/1,185,339 bytes、SHA=`aa7634c2...7a2cd`。
  - [x] 2.330已由`def16597ff245c9ceee6c8e37671a60ade663300`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为150分钟。
  - [x] 150分钟固定截止为138/256、52正确、37.681159%当前精度、20.312500%全量精度、
    0 invalid、64 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.331已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,086行/1,187,700 bytes、SHA=`c1ab612c...d48cc`。
  - [x] 2.331已由`3bb3b77a334064cad1a9b38e3a0129b5e11aa6f8`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为160分钟。
  - [x] 160分钟固定截止为144/256、53正确、36.805556%当前精度、20.703125%全量精度、
    0 invalid、69 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.332已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,122行/1,190,069 bytes、SHA=`3f4c2df2...31df7e`。
  - [x] 2.332已由`62f95b0eebed5470ee895b253a84de5eacb7a923`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为170分钟。
  - [x] 170分钟固定截止为162/256、60正确、37.037037%当前精度、23.437500%全量精度、
    0 invalid、78 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.333已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,158行/1,192,441 bytes、SHA=`992ba7a6...a13abf`。
  - [x] 2.333已由`4f91b00caf5ae9d5b09514aa79b0e1fc565b4023`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为180分钟。
  - [x] 180分钟固定截止为166/256、62正确、37.349398%当前精度、24.218750%全量精度、
    0 invalid、80 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.334已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,194行/1,194,812 bytes、SHA=`06fbe0bb...74e1a3`。
  - [x] 2.334已由`ef13ccf6fc279837dffb0151025d6618d37119a0`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为190分钟。
  - [x] 190分钟固定截止为175/256、67正确、38.285714%当前精度、26.171875%全量精度、
    0 invalid、84 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.335已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,230行/1,197,183 bytes、SHA=`b48e64dd...5c36e8`。
  - [x] 2.335已由`fce323d4752f732b007562417014de5993fd046b`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为200分钟。
  - [x] 200分钟固定截止为196/256、75正确、38.265306%当前精度、29.296875%全量精度、
    0 invalid、94 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.336已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,266行/1,199,556 bytes、SHA=`56441db5...31db82`。
  - [x] 2.336已由`141188e20a7b5c4ca13d7a42d7f02ba2b39d88fb`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为210分钟。
  - [x] 210分钟固定截止为201/256、77正确、38.308458%当前精度、30.078125%全量精度、
    0 invalid、96 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.337已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,302行/1,201,927 bytes、SHA=`7ac33bd6...ff466b`。
  - [x] 2.337已由`b4d360a88e7abaf4fd865ffc9eb7cf42c657d1e0`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为220分钟。
  - [x] 220分钟固定截止为207/256、77正确、37.198068%当前精度、30.078125%全量精度、
    0 invalid、101 truncated；26/26 validation与8/8 manifest通过。
  - [x] 2.338已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,338行/1,204,364 bytes、SHA=`9bb42642...b48b67d`。
  - [x] 2.338已由`f9af2615e47f18bee5cae031613120266c65956a`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为230分钟。
  - [x] 230分钟固定截止为227/256、87正确、38.325991%当前精度、33.984375%全量精度、
    0 invalid、110 truncated；其中4个截断样本评分正确，26/26 validation与8/8 manifest通过。
  - [x] 报告2.339已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,374行/1,206,795 bytes、SHA=`f431616e...65a05`。
  - [x] 2.339已由`77789ce7ac4b894be7c260f6852838fac5d098b9`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为240分钟。
  - [x] 240分钟固定截止为229/256、87正确、37.991266%当前精度、33.984375%全量精度、
    0 invalid、112 truncated；其中4个截断样本评分正确，26/26 validation与8/8 manifest通过。
  - [x] 报告2.340已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,410行/1,209,219 bytes、SHA=`6483cc8e...5ed22`。
  - [x] 2.340已由`47be031a60de1cd9900b564e17fa9fff2d47fbe0`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为250分钟。
  - [x] 250分钟固定截止为236/256、90正确、38.135593%当前精度、35.156250%全量精度、
    0 invalid、115 truncated；其中4个截断样本评分正确，26/26 validation与8/8 manifest通过。
  - [x] 报告2.341已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,446行/1,211,654 bytes、SHA=`5c80f144...c4983`。
  - [x] 2.341已由`0231c0236aaba8f7c03d31a2db3027df4e813e93`通过GitHub HTTPS发布，
    local/upstream一致；实验不中断，下一固定节点为260分钟。
  - [x] 260分钟固定截止为250/256、93正确、37.200000%当前精度、36.328125%全量精度、
    0 invalid、126 truncated；其中4个截断样本评分正确，26/26 validation与8/8 manifest通过。
  - [x] 仅余6题，理论最好终局为99/256，低于BF16 baseline的105/256；本候选已确定失去
    32K/batch1性能复测资格，仍需等待并发布自然终局，再转入精度诊断/优化。
  - [x] 报告2.342已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,485行/1,214,410 bytes、SHA=`ad5b26de...56063`。
  - [x] 2.342已由`c0d5c1bc518857568bdd55d1e37f9f1e9c4d298d`通过GitHub HTTPS发布，
    local/upstream一致；主实验继续到自然终局。
  - [x] 270分钟固定截止为252/256、93正确、36.904762%当前精度、36.328125%全量精度、
    0 invalid、128 truncated；理论最好终局降至97/256，26/26 validation与8/8 manifest通过。
  - [x] 报告2.343已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,523行/1,217,051 bytes、SHA=`991af2fc...0f7dc`。
  - [x] 2.343已由`62979c1d8d99a8cc7eb2280cd2e1989bd2ae0024`通过GitHub HTTPS发布，
    local/upstream一致；主实验继续到自然终局。
  - [x] 280分钟固定截止为253/256、93正确、36.758893%当前精度、36.328125%全量精度、
    0 invalid、129 truncated；理论最好终局降至96/256，26/26 validation与8/8 manifest通过。
  - [x] 报告2.344已实时追加；章节、交叉引用、术语、6项证据hash、8/8 manifest和diff门禁
    全部通过。报告19,561行/1,219,725 bytes、SHA=`bc995c94...d1e3e`。
  - [x] 2.344已由`50ffdbdefec6a89a651191b7538f8eed7a87c7b5`通过GitHub HTTPS发布，
    local/upstream一致；主实验继续到自然终局。
  - [x] canonical fixed256 v2自然终局为93/256、36.328125%，256/256 scored、0 invalid、
    0请求失败、132 truncated；相对BF16 baseline少12题、低4.6875个百分点，门禁失败。
  - [x] wrapper自然exit0，容器/双tmux退出，GPU0–7为0 MiB/0%且compute空；终局证据
    54/54 validation与29/29 manifest通过，final目录21文件、7,365,324 bytes。
  - [x] 报告2.345已实时追加；章节、交叉引用、术语、10项证据hash、54/54 validation、
    29/29 manifest和diff门禁全部通过。报告19,608行/1,223,143 bytes、
    SHA=`3cf4b521...ff9c7`。
  - [x] 2.345已由`1fde94a3a8484db74733e3e4146fd09d6e018ac2`通过GitHub HTTPS发布，
    local/upstream一致；下一步转入canonical精度诊断与优化，继续禁止32K/batch1性能复测。
  - [x] 错误记录：2.329首次追加复用了多处存在的通用尾句锚点，导致章节插到2.327之前；
    章节门禁在提交前发现。当前按完整区块原样移除并以2.328末尾唯一证据行重新追加，禁止
    再使用通用尾句作为报告追加锚点。
  - [x] 错误记录：2.328首次报告表格验证脚本按无单位整数匹配大小，但报告沿用既有
    `N bytes`格式，导致`checkpoint_120min.log`表格字符串断言失败；证据文件实际大小与
    SHA256均正确。已改为按报告实际格式复核，不改报告、不重复错误验证规则。
  - [x] 错误记录：40分钟截止行JSON首次采集因`docker exec`漏`-i`得到空文件，已改用
    container-root标准输入重采并由validator闭合；证据校验后的存活展示命令有孤立引号，
    已以独立只读命令复核容器、双tmux与GPU正常。两项错误均未影响主实验。
  - [x] 错误记录：终局普通用户首次复制容器root创建的0600结果文件时得到permission denied；
    未重复原命令，改用`sudo -n cp`只读复制并转交当前用户，54/54 hash/语义验证与29/29
    manifest闭合，原始结果未改。
  - [x] 错误记录：一次只读`rg`命令把含反引号的搜索式直接置于双引号中，shell先执行了
    `scored`和`0.41015625`并报command not found；随后以已有设计文档内容和final validator
    精确复核baseline合同，未写文件、未影响实验或证据。
