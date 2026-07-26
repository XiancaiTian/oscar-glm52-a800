# 阶段 2 实验报告：Calibration 与 PyTorch reference

## 1. 结论

阶段 2 通过设计文档规定的出口条件：

- calibration manifest 与 official_v4 的完整 prompt hash 交集为 0；
- 正式 train/holdout 分别捕获 900,000/100,000 prompt tokens；
- 8 个 TP rank × 78 层的两个 split 均得到 624/624 个 capture 文件；
- 共享 latent reference、未量化等价、INT2 pack/unpack、mixed tier 和 DSA 输入只读测试均通过；
- 正式 artifact 包含 78 个 `512 × 512` rotation，运行时身份、哈希、层集合和正交性加载门禁通过；
- 最大正交误差为 `1.6274684710992915e-08`。

大型 capture、日志和 `rotations.pt` 仅保存在项目 `artifacts/phase2/`，不提交或推送。

## 2. 固定输入与环境

| 项目 | 实际值 |
| --- | --- |
| 模型 | `GLM-5.2-FP8-pruned-staticgate-e154-H001-nfs` |
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| 并行与执行模式 | TP=8、eager、`TRITON_MLA_SPARSE` |
| calibration 源码 commit | `da4e2756ab4fb41d30ff669b150e9d6d24368d0e` |
| calibration manifest | 292 行、1,000,000 tokens |
| calibration manifest SHA256 | `3a183cba6f013424c5e422da5eb9ab8a9d4d975ddd0dbabd1b930b1d02e876b5` |
| 模型配置 SHA256 | `21a509ab82dad35a8584b724f41aa25c775a3c8f0c7604f7b10d21ee53e5f7fc` |
| checkpoint manifest SHA256 | `e9767570b7b56aa97759c11aec7a81fbe7b84f56d751977788b0195c3a35d983` |
| expert mapping SHA256 | `89430944bd88801fc3deed040dd0d54e1db229f5a5cdd66c9682ef862313c983` |
| latent rank / group size | 512 / 128 |
| prefix / recent | 64 / 256 tokens |
| seed | 20260725 |

manifest 的独立审计结果为：292 个 entry ID 唯一、235 个源样本无 train/holdout 交叉、无重复文本 hash、与 official_v4 完整 prompt hash 交集为 0，各 split/category 重新 tokenize 后均精确满足冻结 token 配额。

## 3. 正式 capture

### 3.1 Train

正式目录为 `artifacts/phase2/20260726T1120Z_calibration_train_tp8`。服务完成两次 8/8 GPU 空闲检查并加载 141/141 个 checkpoint shard；权重读取 48.31 秒、模型加载 61.37 秒，每卡模型内存 55.95GiB。

| 指标 | 实际值 |
| --- | ---: |
| responses | 256 |
| prompt tokens | 900,000 / 900,000 |
| completion tokens | 256 |
| prompt runner 耗时 | 532.1056863907725 秒 |
| capture 文件 | 624 / 624 |
| capture 总大小 | 2,783,307,114 字节 |

`responses.jsonl` 与 summary 的 SHA256 分别为 `a356733b72c0fc3e3eec32c3752a046b7a78a133fff1bc3ad223c347be37b7a2` 和 `77490a263e65c794369a0e301c7da51c76dd8577bc87a741a2a482dd480bca87`；文件名/大小 manifest SHA256 为 `22cd1b54e07d57a1487eac51a282761e0b33bf33a7f866fe67d2f5940547f66f`。

### 3.2 Holdout

正式目录为 `artifacts/phase2/20260726T1140Z_calibration_holdout_tp8`。服务同样完成两次空闲检查和 141/141 shard 加载。

| 指标 | 实际值 |
| --- | ---: |
| responses | 36 |
| prompt tokens | 100,000 / 100,000 |
| completion tokens | 36 |
| prompt runner 耗时 | 88.02139441482723 秒 |
| capture 文件 | 624 / 624 |
| capture 总大小 | 13,272,777,066 字节 |

`responses.jsonl` 与 summary 的 SHA256 分别为 `221edb175d2f95ede7e205b68a3e45d83002d077ee2f73fd958c9736959b0ac3` 和 `cef3c602be27415af467ac83230d217cc75e4e2d5cd205e53356a7e5189da943`；文件名/大小 manifest SHA256 为 `6bf169175c3a4579d77a740a78a5f97a77fc683602d9ef500223cd4ba7713ba6`。

两个服务正常停止后均无残留 vLLM/EngineCore 进程，8 张 GPU 均为 0MiB、0% 利用率。

## 4. Rotation 与 clip 搜索

首次 fit 在读取第一层前按预期 fail closed：配置要求 `model.layers.{layer}.self_attn.pt`，实际 capture 为 `model.layers.{layer}.self_attn.attn.pt`，因此没有生成 artifact。修复后，独立路径门禁确认 train/holdout 各 624 个期望文件与实际文件集合完全一致；原有 capture 未被修改或重跑。

正式重跑目录为 `artifacts/phase2/20260726T1200Z_rotation_fit_v2`。固定 alpha 网格为 `{0.25, 0.5, 0.75}`，结果如下：

| Alpha | 归一化 holdout loss |
| ---: | ---: |
| 0.25 | 0.025037897150672388 |
| 0.5 | 0.027723928782624297 |
| 0.75 | 0.031391672548347495 |

最终选择 `alpha=0.25`。78 层的逐层归一化 loss 范围为 `0.0004133854263186087` 至 `0.04478203689244448`；clip ratio 为 0.92 的有 61 层，为 0.94 的有 17 层。每层实际使用 900,000 个 train latent 样本、7,200,000 个 train score/value 样本、100,000 个 holdout latent 样本、800,000 个 holdout score/value 样本和 4,096 行 holdout reservoir。

## 5. Artifact 验收

| 产物 | 大小 | SHA256 |
| --- | ---: | --- |
| `manifest.json` | 2,404 字节 | `df30fbb90bfafaef787cfe73bd55c9179ac2acb83d2a23ed977a8378d7c19926` |
| `rotations.pt` | 81,811,997 字节 | `0a966da2e480559b698e4347ba29f402f5eb9fe335781ac41f8c27086a72808e` |
| `search_summary.json` | 28,465 字节 | `9dfe16a88a3c46dfff8e10b62a8f47a7bc716996c6e3d51b3b6b7daeb212792e` |

使用候选 rootfs Python 和正式 calibration 源码执行运行时 loader，实际结果为：

- manifest、rotation tensor SHA256 和运行时模型身份完全匹配；
- 78/78 层齐全，每层 shape 为 `512 × 512`，元素均 finite；
- `RᵀR` 相对单位阵的逐层最大绝对误差范围为 `1.0171338660214246e-08` 至 `1.6274684710992915e-08`；
- loader 返回 `PASS`。

## 6. Reference 与回归验证

源码定向套件实际为 35 项 pytest 全部通过，ruff 0.14.0、format 和 `git diff --check` 均通过。测试覆盖：

- 未量化 shared rotation 路径与原生 MLA 等价；
- 4×2-bit pack/unpack 与非对称 INT2 半步长误差界；
- prefix/recent BF16 与 history rotated INT2 的同一 softmax reference；
- score/value covariance 合并与 artifact fail-closed 合约；
- capture 前后 DSA index、输入 tensor 的值、shape 和 storage pointer 不变。

## 7. 阶段出口

阶段 2 已完成，后续阶段以本报告的 artifact 身份和窗口配置为唯一正式输入。大文件不进入 Git；Git 仅同步代码、小型配置、持久进度和本报告。
