# 阶段 2 实验报告：REAP 模型 Calibration 与 Rotation Artifact

## 1. 结论

新 REAP checkpoint 的阶段 2 通过设计文档规定的出口条件：

- train/holdout 分别精确捕获 900,000/100,000 prompt tokens；
- 两个 split 均得到 8 个 TP rank × 78 层，即 624/624 个 capture 文件；
- holdout 的 4,096 行 reservoir、`512 × 2048` DSA 样本和 TP covariance
  语义均通过逐文件验证；
- 独立 holdout 搜索选择 `alpha=0.25`，归一化 loss 为
  `0.026186010882512642`；
- artifact 包含 78 个 `512 × 512` rotation，新 checkpoint 身份、文件哈希、
  层集合、有限值和正式 runtime loader 门禁全部通过；
- `RᵀR-I` 的逐层最大绝对误差上限为 `1.6403759683925045e-08`；
- 当前集成源码的阶段 2 定向非 CUDA 回归为 33/33 passed。

81,811,997 字节的 `rotations.pt` 已固化到项目 ignored artifacts，供后续阶段使用，
但不会提交或推送。约 16.06GB 的两个可重建 capture 继续仅保存在 `/dev/shm`。
train/holdout 合计 292 条请求全部 HTTP 200，服务日志无 `ERROR`、Traceback 或
`RuntimeError`。

## 2. 固定输入与环境

| 项目 | 实际值 |
| --- | --- |
| 模型路径 | `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001` |
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| 并行与执行模式 | TP=8、eager、`TRITON_MLA_SPARSE` |
| 正式 capture 根仓库 commit | `8a16407c49a3e74daaf1732610db13a559876f68` |
| calibration 源码 commit | `a3317695428819d41437b1cb144404b3bfc05a92` |
| calibration manifest | 292 行、1,000,000 tokens |
| calibration manifest SHA256 | `3a183cba6f013424c5e422da5eb9ab8a9d4d975ddd0dbabd1b930b1d02e876b5` |
| 模型配置 SHA256 | `21a509ab82dad35a8584b724f41aa25c775a3c8f0c7604f7b10d21ee53e5f7fc` |
| checkpoint manifest SHA256 | `f50217dadf6c58f8f84140003bd7fc3497e9338916e9865e23ea5342f2ac1ce2` |
| expert mapping SHA256 | `c163c3f02089cfe7180bda0816cea59ce8f8bba0fe752b6dd4738d9c87b3da72` |
| latent rank / group size | 512 / 128 |
| prefix / recent | 64 / 256 tokens |
| seed | 20260725 |

calibration manifest 与阶段 1 一致，且独立于 official_v4；checkpoint index 和
expert mapping 使用当前 REAP 模型的冻结指纹。旧 staticgate artifact 没有参与本轮
capture、fit 或 runtime loader 验证。

外部部署目录在本阶段始终只读；没有复制旧镜像文件，也没有修改
`/nfs/AE/zhanghong/workflow/vllm_a/glm52_speed_up_v2_stable_8th`。

## 3. 正式 Capture

### 3.1 Train

正式目录为
`/dev/shm/oscar-glm-reap-stage2/phase2/20260727T2230Z_reap_calibration_train_tp8_final`。
服务完成两次 8/8 GPU 空闲检查，141/141 个 checkpoint shard 全部加载；权重读取
47.59 秒，模型加载 60.161158 秒，每卡模型内存 55.95GiB。

| 指标 | 实际值 |
| --- | ---: |
| HTTP 200 / 请求数 | 256 / 256 |
| prompt tokens | 900,000 / 900,000 |
| completion tokens | 256 |
| prompt runner 耗时 | 326.51599755790085 秒 |
| capture 文件 | 624 / 624 |
| capture 总大小 | 2,783,307,114 字节 |

responses 与 summary SHA256 分别为
`c54c76b792e2bebb63c40701aa27d379f476996ceec7b14a07a638014bdd8ba6`、
`5958d95a59755c925a9de49201c62f4509599421ba4f20c750a30a687260b225`。
完整内容清单及 metadata validation SHA256 分别为
`5a300083b9f4efdc54ea0886e00a4dabf1889fd899549b98f3791ac32e327907`、
`e0cc528ac014b8dc29201339a8cfbb9601b8e00a7f85dab4b8af0c0d054a5eda`。

逐文件验证确认每层 captured tokens 均为 900,000；rank 0 独占共享 latent
covariance，8 个 rank 各保存 score/value covariance，符合 fit loader 的 TP 合并
契约。

### 3.2 Holdout

正式目录为
`/dev/shm/oscar-glm-reap-stage2/phase2/20260727T2241Z_reap_calibration_holdout_tp8_final`。
服务独立完成两次 8/8 GPU 空闲检查和 141/141 个 shard 加载；权重读取 55.72 秒，
模型加载 `68.378808` 秒，每卡模型内存 55.95GiB。

| 指标 | 实际值 |
| --- | ---: |
| HTTP 200 / 请求数 | 36 / 36 |
| prompt tokens | 100,000 / 100,000 |
| completion tokens | 36 |
| prompt runner 耗时 | 41.85091549158096 秒 |
| capture 文件 | 624 / 624 |
| capture 总大小 | 13,272,777,066 字节 |

responses 与 summary SHA256 分别为
`43a87f069c5005a95ad32c7aa5c983dfe08d3fcf79e9fbdac7a5ddbcb3baa832`、
`e89cc2ec2448dbd4d71ed519afc99782890f4269cf28b832f8434c410cd2c4f8`。
完整内容清单及 metadata validation SHA256 分别为
`9bb125fe127db3f9d895f1a762cc80f87d79a67e51b2a3ea6f395cc85ff2bb1e`、
`8fa969bc604f4a8f5d74d1518562b27363fa20d2b50a9254d206e11307687caf`。

全部 624 个文件均有 4,096 行 latent/query/value reservoir 和
`512 × 2048` DSA 样本；8 个 rank 各有 100,000 个 score/value covariance
样本，仅 rank 0 持有 100,000 个共享 latent covariance 样本。

两个服务日志均无非 200、request failure、`ERROR` 或 Traceback，退出后
8 张 GPU 均为 0MiB、0%，无残留 vLLM/EngineCore 进程。各轮总耗时均不足
10 分钟，因此没有触发 10 分钟 heartbeat。

## 4. Rotation 与 Clip 搜索

fit 路径门禁首先确认 train/holdout 各 624 个预期文件与实际文件集合完全一致。
正式输出目录为
`/dev/shm/oscar-glm-reap-stage2/phase2/20260727T2253Z_reap_rotation_fit_final`。

| Alpha | 归一化 holdout loss |
| ---: | ---: |
| 0.25 | 0.026186010882512642 |
| 0.5 | 0.02872817461999513 |
| 0.75 | 0.032144202654983196 |

最终选择 `alpha=0.25`。78 层逐层归一化 loss 范围为
`0.00041062589551025104` 至 `0.046030287445025325`；clip ratio 为 0.92 的
有 61 层，为 0.94 的有 17 层。

每层实际合并 900,000 个 train latent 样本、7,200,000 个 train score/value
样本、100,000 个 holdout latent 样本、800,000 个 holdout score/value 样本，
并使用 4,096 行 holdout reservoir。fit 总耗时不足 10 分钟，未触发 heartbeat。

## 5. Artifact 验收与固化

正式 artifact 已逐字节复制到项目 ignored 路径
`artifacts/phase2/20260727T2253Z_reap_rotation_fit_final`，源与目标四个文件均通过
`cmp`，SHA256 保持不变。

| 产物 | 大小 | SHA256 |
| --- | ---: | --- |
| `manifest.json` | 2,404 字节 | `0275043c070c9127354997374e9bca1c70fe1308a7b2d057f992fadedef868e5` |
| `rotations.pt` | 81,811,997 字节 | `256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d` |
| `search_summary.json` | 28,466 字节 | `61790fecdb02c789d3f530a5d6fed3346ba12707e30c33025d045b065ebaec10` |
| `artifact_validation.json` | 346 字节 | `978834a7baf02fe9e54253c84d86be69380c5fb01e02a5fb10d188503c782749` |

候选 rootfs Python 和当前 calibration 源码的正式 runtime loader 实测结果为：

- 当前模型 config、checkpoint index、expert mapping 与窗口配置全部匹配；
- 78/78 层齐全，每层 shape 为 `512 × 512`、FP32、CPU contiguous 且 finite；
- `RᵀR-I` 最大绝对误差范围为 `9.648358112457345e-09` 至
  `1.6403759683925045e-08`；
- loader 期间 `torch.cuda.is_initialized()` 为 false；
- fit 日志的 `ERROR`/Traceback 计数为 0。

## 6. Reference 回归与过程门禁

在源码 commit `a3317695428819d41437b1cb144404b3bfc05a92` 上重新执行
`test_reference.py`、`test_calibration.py`、`test_capture.py`、
`test_artifact.py` 和 `test_fit.py`，实际结果为 33 passed、0 failed，
耗时 5.90 秒。

测试覆盖共享 rotation 未量化等价、INT2 pack/unpack 数值界、mixed
prefix/recent/history reference、covariance 合并、只读 capture，以及 artifact
身份、哈希、层集合、shape、正交性和篡改拒绝门禁。

正式 train 前发现两个不同 RUN_ID 的 launcher 在服务尚未 ready 时并发通过端口
检查。两个作废轮次均未加载权重、GPU 仍为 0MiB。随后加入按
`ARTIFACT_ROOT + HOST + PORT` 的非阻塞 `flock`；正式 train 和 holdout 均只存在
一个服务实例。

holdout ready 后，两个控制流程几乎同时调用 prompt 入口；后到的入口在发送请求前
被“输出目录已存在”门禁拒绝。唯一 runner 完成 36/36 请求，没有覆盖或重复写入
正式响应与 capture。

## 7. 阶段出口

阶段 2 完成。后续阶段唯一允许使用的 rotation artifact 是
`artifacts/phase2/20260727T2253Z_reap_rotation_fit_final`，其 manifest 和
rotations SHA256 分别为 `0275043c...68e5` 与 `256ee5e4...235d`。

Git 只同步代码、小型配置、持久进度和本报告；`rotations.pt`、capture、服务日志和
缓存不进入 commit/push。旧 checkpoint artifact 继续保留为历史记录，但不得用于
当前模型。
