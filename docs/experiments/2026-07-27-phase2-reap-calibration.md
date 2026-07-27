# 阶段 2 实验报告：REAP 模型 Calibration 与 Rotation Artifact

## 1. 结论

新 REAP checkpoint 的阶段 2 已通过：

- train/holdout 分别精确捕获 900,000/100,000 prompt tokens；
- 两个 split 均生成 8 个 TP rank × 78 层，即 624/624 个 capture 文件；
- train/holdout 的 292 条请求全部 HTTP 200，服务日志无
  `ERROR`、`Traceback` 或 `RuntimeError`；
- fit 输出 78 个 `512 × 512` rotation，并通过正式 runtime loader 的模型身份、
  哈希、层集合、shape、finite 和正交性门禁；
- 最大正交误差为 `1.6403759683925045e-08`。

旧 staticgate checkpoint 的 rotation artifact 不适用于本模型。本报告中的大型
capture、响应、日志和 `rotations.pt` 位于 `/dev/shm`，不提交或推送；Git 只同步
小型配置、代码和中文报告。

## 2. 固定输入与环境

| 项目 | 实际值 |
| --- | --- |
| 模型 | `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001` |
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| 并行与执行模式 | TP=8、eager、`TRITON_MLA_SPARSE` |
| calibration 源码 commit | `a3317695428819d41437b1cb144404b3bfc05a92` |
| calibration manifest | 292 行、1,000,000 tokens |
| calibration manifest SHA256 | `3a183cba6f013424c5e422da5eb9ab8a9d4d975ddd0dbabd1b930b1d02e876b5` |
| 模型配置 SHA256 | `21a509ab82dad35a8584b724f41aa25c775a3c8f0c7604f7b10d21ee53e5f7fc` |
| checkpoint manifest SHA256 | `f50217dadf6c58f8f84140003bd7fc3497e9338916e9865e23ea5342f2ac1ce2` |
| expert mapping SHA256 | `c163c3f02089cfe7180bda0816cea59ce8f8bba0fe752b6dd4738d9c87b3da72` |
| latent rank / group size | 512 / 128 |
| prefix / recent | 64 / 256 tokens |
| seed | 20260725 |

外部部署目录在本阶段始终只读；没有复制旧镜像文件，也没有修改
`/nfs/AE/zhanghong/workflow/vllm_a/glm52_speed_up_v2_stable_8th`。

## 3. 正式 Capture

### 3.1 Train

正式目录为
`/dev/shm/oscar-glm-reap-stage2/phase2/20260727T2230Z_reap_calibration_train_tp8_final`。
服务完成两次 8/8 GPU 空闲检查并加载 141/141 个 checkpoint shard；权重读取
47.59 秒，模型加载 60.161158 秒，每卡模型内存 55.95GiB。

| 指标 | 实际值 |
| --- | ---: |
| responses / HTTP 200 | 256 / 256 |
| prompt tokens | 900,000 / 900,000 |
| completion tokens | 256 |
| prompt runner 耗时 | 326.51599755790085 秒 |
| capture 文件 | 624 / 624 |
| capture 总大小 | 2,783,307,114 字节 |

`responses.jsonl` SHA256 为
`c54c76b792e2bebb63c40701aa27d379f476996ceec7b14a07a638014bdd8ba6`；
`summary.json` SHA256 为
`5958d95a59755c925a9de49201c62f4509599421ba4f20c750a30a687260b225`。

### 3.2 Holdout

正式目录为
`/dev/shm/oscar-glm-reap-stage2/phase2/20260727T2241Z_reap_calibration_holdout_tp8_final`。
服务独立完成两次 8/8 GPU 空闲检查和 141/141 shard 加载；权重读取 55.72 秒，
模型加载 68.378808 秒，每卡模型内存 55.95GiB。

| 指标 | 实际值 |
| --- | ---: |
| responses / HTTP 200 | 36 / 36 |
| prompt tokens | 100,000 / 100,000 |
| completion tokens | 36 |
| prompt runner 耗时 | 41.85091549158096 秒 |
| capture 文件 | 624 / 624 |
| capture 总大小 | 13,272,777,066 字节 |

`responses.jsonl` SHA256 为
`43a87f069c5005a95ad32c7aa5c983dfe08d3fcf79e9fbdac7a5ddbcb3baa832`；
`summary.json` SHA256 为
`e89cc2ec2448dbd4d71ed519afc99782890f4269cf28b832f8434c410cd2c4f8`。

两个 split 均为每个 TP rank 78 层。rank 0 保存共享 latent covariance，全部
8 个 rank 保存 score/value covariance；holdout 另含每层 4,096 行 reservoir 和
512 行 DSA 样本。服务退出后 8 张 GPU 均为 0MiB、0%，无残留 vLLM/EngineCore
进程。两个正式服务均不足 10 分钟，因此没有触发十分钟心跳。

## 4. Rotation 与 Clip 搜索

正式 fit 目录为
`/dev/shm/oscar-glm-reap-stage2/phase2/20260727T2253Z_reap_rotation_fit_final`。
路径门禁先确认 train/holdout 各自的期望文件集合与实际 624 个文件完全一致。

| Alpha | 归一化 holdout loss |
| ---: | ---: |
| 0.25 | 0.026186010882512642 |
| 0.5 | 0.02872817461999513 |
| 0.75 | 0.032144202654983196 |

最终选择 `alpha=0.25`。78 层逐层归一化 loss 范围为
`0.00041062589551025104` 至 `0.046030287445025325`；clip ratio 为 0.92 的有
61 层，为 0.94 的有 17 层。每层使用 900,000 个 train latent 样本、
7,200,000 个 train score/value 样本、100,000 个 holdout latent 样本和
800,000 个 holdout score/value 样本。

## 5. Artifact 验收

| 产物 | 大小 | SHA256 |
| --- | ---: | --- |
| `manifest.json` | 2,404 字节 | `0275043c070c9127354997374e9bca1c70fe1308a7b2d057f992fadedef868e5` |
| `rotations.pt` | 81,811,997 字节 | `256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d` |
| `search_summary.json` | 28,466 字节 | `61790fecdb02c789d3f530a5d6fed3346ba12707e30c33025d045b065ebaec10` |

候选 rootfs Python 从当前源码加载 artifact，并使用阶段 2 配置构造独立
`ArtifactExpectation`。实际结果为：

- 78/78 层齐全，模型 config、checkpoint index、expert mapping、窗口和 group
  size 全部匹配；
- manifest 与 rotation tensor SHA256 匹配；
- 最大 `RᵀR-I` 绝对误差为 `1.6403759683925045e-08`；
- loader 返回 `passed`，且 CUDA 未初始化。

## 6. 过程门禁

正式 train 前发现两个不同 RUN_ID 的 launcher 在服务尚未 ready 时并发通过端口
检查。两个作废轮次均未加载权重、GPU 仍为 0MiB。随后加入按
`ARTIFACT_ROOT + HOST + PORT` 的非阻塞 `flock`，正式 train 和 holdout 均只存在
一个服务实例。

holdout ready 后，两个控制流程几乎同时调用 prompt 入口；后到的入口在发送请求前
被“输出目录已存在”门禁拒绝。唯一 runner 完成 36/36 请求，未覆盖或重复写入正式
响应与 capture。

## 7. 阶段出口

新 REAP checkpoint 的阶段 2 已完成。阶段 3–5 回归和阶段 6 候选 OCI 必须绑定
本报告的 artifact 身份；旧 checkpoint artifact 继续保留为历史记录，但不得用于
当前模型。
