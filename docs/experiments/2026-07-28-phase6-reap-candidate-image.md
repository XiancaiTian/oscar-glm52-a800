# 阶段 6 实验报告：REAP 不可变候选镜像

## 1. 结论

阶段 6 已基于当前 REAP 剪枝模型、最新 OSCAR-vLLM 源码和重新拟合的 rotation
artifact 完成：

- 标准 OCI image layout 已构建并由独立 verifier 验收；
- 4,743 个源码文件逐 Git blob、符号链接和 executable mode 完全匹配；
- rotation 目录的 4 个文件全部纳入 SHA256 白名单，任何额外文件都会被拒绝；
- `oscar_runtime_expectation.json` 已写入候选层并通过 SHA256、OCI label 和运行时
  环境变量三重绑定，候选不再依赖宿主机配置文件；
- 前 32 个基础层逐 descriptor 保持不变，候选层不含 native extension 或
  whiteout；
- 第二次独立构建的 image ID、manifest/config/layer digest、diff ID、大小和成员数
  均与第一次一致；
- 固定 rootfs Python 成功导入候选源码、基础层 `vllm._C` 和 78 层 rotation，
  runtime expectation 校验通过，CUDA 未初始化。

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
| 正式构建主仓库 commit | `cfbcaa5f5c4aed2d1692051a355e02d5640d3b24` |
| 源码 commit | `a3317695428819d41437b1cb144404b3bfc05a92` |
| 源码 tree | `85619bd0ea0c71291a49f628fd4515b5fb70e9a1` |
| phase 0 base manifest | `sha256:2fdfbe865aecc01eee15a01fcce58bf7581244dbbc53cbe3ef0e0cce44bc489d` |
| phase 0 base config | `sha256:58a853ee730c263968dcc6b76401e85e4b510a140777c4ffc791747edd8ea42d` |
| Dockerfile SHA256 | `dde86a9338dc338d255d615baa15f19f63c53dd09041cd7d702e075f0ef8502f` |
| 输入 manifest SHA256 | `78d18f1a04ecda22cb4358c452a26e080ef541ad3c6821b4837f0dcf1224adcb` |
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
| Tag | `glm52-oscar-a800-phase6-a33176954-0275043c` |
| Image ID / config digest | `sha256:dd7b4f47a900dfc59f599cd99ca9c3e25456d9fd5de29753d1f85fd4f256ca70` |
| Manifest digest | `sha256:1d3d26262fd6abe51ee271d99584091fcef3ca2cd35a585204ac14c6340f0ea6` |
| Architecture / OS | `amd64` / `linux` |
| Layers | 33 |
| Reproducible created 字段 | `2026-07-27T06:13:10Z` |
| Candidate layer digest | `sha256:189f55db6bd54114fc1a86f956704e7e4eb13b80f4d1f826add697aad40182d0` |
| Candidate layer diff ID | `sha256:50cf0f9ad5abaedd22dfeb17d1ee1c0c3719e105d40933ce0605bbeb44b38717` |
| Candidate layer size | 109,144,170 bytes |
| Candidate layer tar members | 5,297 |

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
| 源码 | 4,743/4,743 Git blob、symlink、executable mode 全部匹配 |
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
| CUDA initialized | `false` |

候选源码通过 overlay 路径导入；6 个 vLLM native extension 在导入期间临时指向
已经验收的基础层文件，完成后链接全部清理。运行过程中设置
`PYTHONDONTWRITEBYTECODE=1`，overlay 内未生成 `.pyc`。

## 7. 本地证据

正式证据目录为
`artifacts/phase6/20260728T0004Z_candidate_a33176954_final`：

| 文件 | SHA256 |
| --- | --- |
| `build_result.json` | `373b46e9529c28e4b3c34ac1bde01583c7b3546b07d054456d294d835b1180e7` |
| `verification.json` | `e7cfd0a7c59b21938c53d1f3473876c2b2f0f8184da9c42c721f9de5386359da` |
| `runtime_import.json` | `5735e3dbc706d341fe8a293f62b69044bc1d5f25ad2eb9aa06d56a7cc44dcb2e` |
| `rebuild_result.json` | `197406182b158a6cc26c9ee67948b00fe3aa8a4b00ebe60476f388d6e47b1d97` |

这些大型/运行时产物被 Git 忽略，仅上述身份和结果写入报告。

## 8. 阶段出口

阶段 6 完成。旧 staticgate checkpoint 的候选 OCI 仅保留为历史证据，不得用于
当前评测。阶段 7 必须从本报告冻结的新候选 overlay 加载源码、rotation 和 runtime
expectation，并相对新 REAP 模型的阶段 1 official_v4 与 WikiText‑2 baseline
执行完整精度门禁。
