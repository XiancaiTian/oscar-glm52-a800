# 阶段 6 实验报告：REAP 不可变候选镜像

## 1. 结论

阶段 6 已基于当前 REAP 剪枝模型、最新 OSCAR-vLLM 源码和重新拟合的 rotation
artifact 完成：

- 标准 OCI image layout 已构建并由独立 verifier 验收；
- 4,744 个源码文件逐 Git blob、符号链接和 executable mode 完全匹配；
- rotation 目录的 4 个文件全部纳入 SHA256 白名单，任何额外文件都会被拒绝；
- `oscar_runtime_expectation.json` 已写入候选层并通过 SHA256、OCI label 和运行时
  环境变量三重绑定，候选不再依赖宿主机配置文件；
- 前 32 个基础层逐 descriptor 保持不变，候选层不含 native extension 或
  whiteout；
- 第二次独立构建的 image ID、manifest/config/layer digest、diff ID、大小和成员数
  均与第一次一致；
- 固定 rootfs Python 成功导入候选源码、基础层 `vllm._C` 和 78 层 rotation，
  runtime expectation 校验通过；official_v5 的 `reasoning_effort=max` 可被
  请求 schema 接受并原样保留，CUDA 未初始化。

候选 OCI、展开目录和 rotation tensor 属于大型本地产物，仅保存在
`artifacts/phase6/`，未 commit 或 push。Git 只同步 Dockerfile、构建/验收脚本、
小型 manifest、运行时契约和中文报告。

## 2. 构建方法与边界

当前 Kubernetes 容器没有 Docker daemon/socket，也没有执行 OCI mount 所需权限。
本阶段沿用已经验收的 daemonless 标准 OCI 构建方法：

1. 验证 phase 0 baseline OCI manifest；
2. 用 hardlink 复用 32 个不可变基础 layer blob；
3. 从已发布 Git commit 导出源码；
4. 复制严格白名单化的 4 个 rotation 文件和一个运行时 expectation 文件；
5. 用固定排序、owner、mtime 与 gzip header 生成确定性候选层；
6. 生成 OCI config、manifest、index、labels 和运行时环境；
7. 用独立 verifier 重读 descriptor、展开候选层并逐项验收。

这证明的是标准 OCI 已构建和验证，不冒充未执行的 `docker build` 或
`docker run`。本阶段不使用 GPU。

## 3. 冻结输入

| 项目 | 实际值 |
| --- | --- |
| 测试模型 | `/nfs/AE/txc/model_files/GLM-5.2-FP8-pruned-reap-e154-H001` |
| 正式构建主仓库 commit | `990dfd5397bf98c4b91f4e4d1943317fd5269a13` |
| 源码 commit | `065af88a010dc5746029198088ba01edc4a61516` |
| 源码 tree | `3bdd2793c560b5c0c0460bccaa5abdfd5f23f96e` |
| phase 0 base manifest | `sha256:2fdfbe865aecc01eee15a01fcce58bf7581244dbbc53cbe3ef0e0cce44bc489d` |
| phase 0 base config | `sha256:58a853ee730c263968dcc6b76401e85e4b510a140777c4ffc791747edd8ea42d` |
| Dockerfile SHA256 | `510200945cb23b1d16ad98b7f5edbb34b726ca37105ecfc65c447820f04b59ca` |
| 输入 manifest SHA256 | `a5465acee46846ae6ab3d3890b1fb6b6c8097b33a8c989c16f204291c551562b` |
| runtime expectation SHA256 | `9d992c7028fd1f746566e101be57a0102d5c97a9816a1737f5ec3a7ffdeda98f` |
| rotation manifest SHA256 | `0275043c070c9127354997374e9bca1c70fe1308a7b2d057f992fadedef868e5` |
| rotations SHA256 | `256ee5e4e92a2f28fa54a537daab543a6f1d54d87a569370325288186156235d` |
| search summary SHA256 | `61790fecdb02c789d3f530a5d6fed3346ba12707e30c33025d045b065ebaec10` |
| artifact validation SHA256 | `978834a7baf02fe9e54253c84d86be69380c5fb01e02a5fb10d188503c782749` |
| native manifest SHA256 | `bc167e80cb20734cbd137d77912e28a222391f42b1427fc38cc06057df706050` |

正式构建开始前，主仓库和源码仓库均为干净状态，HEAD 与各自 upstream 完全一致。
运行时 expectation 绑定新模型 config、checkpoint index、专家映射、78 层、
512 latent rank、128 group size、64 prefix tokens 和 256 recent tokens。

## 4. 候选镜像身份

| 字段 | 实际值 |
| --- | --- |
| Tag | `glm52-oscar-a800-phase6-065af88a0-0275043c` |
| Image ID / config digest | `sha256:8b7a2ee66645dfb08dace9568ca0551c01a79da143afbe0c2f21beba8fb968bb` |
| Manifest digest | `sha256:01f91611d1e825219907f98943968e45047458445b5b61de4ef28ff272a77932` |
| Architecture / OS | `amd64` / `linux` |
| Layers | 33 |
| Reproducible created 字段 | `2026-07-28T11:27:42Z` |
| Candidate layer digest | `sha256:7e220c08cdf09032d862e95179cf27f13daf245156b9e6b3dc28e685068d178e` |
| Candidate layer diff ID | `sha256:ae08e7f0b06d7651a6559da8c59cec0db6526417e1e1737350326bb9a69a0224` |
| Candidate layer size | 109,144,108 bytes |
| Candidate layer tar members | 5,298 |

候选环境固定为：

```text
PYTHONPATH=/opt/vllm_glm52_v1
VLLM_OSCAR_MLA_ROTATION_ARTIFACT=/opt/oscar_artifacts/rotation_fit_v2
VLLM_OSCAR_MLA_RUNTIME_EXPECTATION=/opt/oscar_artifacts/oscar_runtime_expectation.json
```

## 5. 独立验收与确定性

正式 verifier 的实际结果：

| 检查 | 实际结果 |
| --- | --- |
| 基础层 | 前 32 个 layer descriptor 与 phase 0 完全一致 |
| 候选层安全性 | 无 `.so`、无 OCI whiteout、无绝对路径或 `..` 路径 |
| 源码 | 4,744/4,744 Git blob、symlink、executable mode 全部匹配 |
| Rotation artifact | 4/4 文件集合和 SHA256 匹配 |
| Runtime expectation | 文件 SHA256、label 和 ENV 全部匹配 |
| Native extensions | 7/7 基础层 SHA256 匹配，候选层未覆盖 |
| Verifier 状态 | `passed` |

相同输入的第二次构建得到完全相同的 input manifest SHA256、image ID、
manifest/config/layer digest、layer diff ID、layer size 和 tar member count。
两个 build report 的文件 SHA256 不同，仅因为各自记录了不同的输出 layout 路径。

## 6. 固定运行时

候选层与 phase 0 基础 rootfs 组成等价 overlay 后，实际导入结果如下：

| 组件 | 实际版本/结果 |
| --- | --- |
| Python | `3.12.13` |
| PyTorch | `2.11.0+cu129` |
| Triton | `3.6.0` |
| Transformers | `5.8.1` |
| Tokenizers | `0.22.2` |
| vLLM dist-info | `0.11.2.dev278+gdbc3d9991` |
| vLLM version override | `dev` |
| FlashInfer Python | `0.6.6` |
| FlashInfer JIT cache | `0.6.6+cu129` |
| `vllm._C` | 从 phase 0 基础层成功导入 |
| Rotation tensors | 78 |
| Runtime expectation | 从候选 overlay 成功加载并校验 |
| official_v5 reasoning effort | `max` 可被请求 schema 接受并保持为 `max` |
| CUDA initialized | `false` |

候选源码通过 overlay 路径导入；6 个 vLLM native extension 在导入期间临时指向
已经验收的基础层文件，完成后链接全部清理。运行过程中设置
`PYTHONDONTWRITEBYTECODE=1`，overlay 内未生成 `.pyc`。

## 7. 本地证据

正式证据目录为
`artifacts/phase6/20260728T112942Z_candidate_065af88a0_v5compat`：

| 文件 | SHA256 |
| --- | --- |
| `build_result.json` | `17a0d0942eff626b79d6487836e9afb510931b9c2183a701234b976d40555587` |
| `verification.json` | `d000c76dde7a4095813e6dcffce58793704d0ae36a8efcbdbcb205386667575e` |
| `runtime_import.json` | `ed68ebe3f4ad794a948254584230ec5116035ffb7e03cd57418a9e61eabcb644` |
| `rebuild_result.json` | `d18dcf67825b7c8a223bf10b2b917aa0bcda9027e694756f378e019bbd22ffdd` |

这些大型/运行时产物被 Git 忽略，仅上述身份和结果写入报告。

## 8. 阶段出口

阶段 6 的 official_v5 兼容候选已完成重建。此前
`a331769542...` 候选 OCI 仅保留为历史证据，不得用于当前正式评测。阶段 7 必须
从本报告冻结的新候选 overlay 加载源码、rotation 和 runtime expectation，并与
同一 REAP 模型的 official_v5 原生 GSM8K baseline 执行当前阶段门禁；最终阶段再
运行 official_v5 全量 accuracy 与 WikiText‑2 PPL。
