# 阶段 6 实验报告：不可变候选镜像

## 1. 结论

阶段 6 已完成并通过设计文档规定的冻结要求：

- 固定源码 commit/tree、Dockerfile、基础 OCI、Python 依赖、7 个 native
  extensions 和正式 rotation artifact；
- 生成新的标准 OCI image layout，记录唯一 tag、image ID、manifest digest 和
  config digest；
- 独立 verifier 证明前 32 个基础层逐 descriptor 不变，候选最后一层不含 `.so`
  或 whiteout；
- 解包候选层后，4,742 个源码文件逐 Git blob、符号链接和 executable mode
  匹配，3 个 rotation artifact 文件和 7 个基础层 native extensions 的 SHA256
  全部通过；
- 独立第二次构建得到相同 candidate layer digest/diff ID、image ID 和 manifest
  digest；
- 候选解包运行时成功导入固定 Python/PyTorch/Triton/Transformers/vLLM native
  extension，加载 78 层 rotations，且没有初始化 CUDA。

候选 OCI、解包 rootfs 和 rotation tensors 是大型本地产物，仅保存在
`artifacts/phase6/`，没有 commit 或 push。Git 只同步 Dockerfile、构建/验收代码、
小型输入 manifest 和本中文报告。

## 2. 构建环境与方法

当前 Kubernetes 容器没有 Docker/containerd socket，也没有 `CAP_SYS_ADMIN`；
`docker`、`podman`、`nerdctl` 均不可用。阶段 0 已证明 `buildah` 无法在该环境
完成 mount/unshare。因此本阶段使用 Python、GNU tar/gzip 和 OCI descriptor
直接构建标准 OCI layout，没有把未执行的 Docker daemon 构建描述成成功。

构建方法为：

1. 读取并验证 phase 0 baseline OCI manifest；
2. 用 hardlink 复用 32 个不可变基础 layer blobs；
3. 从已发布 Git commit 导出源码，不读取工作树中的 ignored `.so`；
4. 加入正式 rotation artifact；
5. 用固定排序、owner、mtime 和 gzip header 生成确定性候选层；
6. 生成新的 OCI config、manifest、index 和 labels；
7. 由独立脚本重新读取 descriptor、解包候选层并逐项验收。

hardlink 只用于本地节省重复存储；OCI descriptor 指向的 blobs 仍按内容 SHA256
寻址。抽查的 1,751,391,348-byte 基础 blob 在 phase 0 和 phase 6 layout 中 inode
相同且 link count 为 3；109,133,958-byte 新候选 layer 是独立 blob。

## 3. 冻结输入

| 项目 | 实际值 |
| --- | --- |
| 正式构建主仓库 commit | `95582255ec82c7f336036ded240f3c8208e8e22d` |
| 源码 commit | `7d317f1dee21af9d49445878bcc9c2d181d041c9` |
| 源码 tree | `e7c8792b80b169c0f06c91296e12bf23b9c908e5` |
| phase 0 base manifest | `sha256:2fdfbe865aecc01eee15a01fcce58bf7581244dbbc53cbe3ef0e0cce44bc489d` |
| phase 0 base config | `sha256:58a853ee730c263968dcc6b76401e85e4b510a140777c4ffc791747edd8ea42d` |
| Dockerfile SHA256 | `b3138a9be71fe1f5de8a1b983918e04dac8b7ff78b86a1a596dbdcdb0c4b8bba` |
| 输入 manifest SHA256 | `5159086bdfbba5a26215aa2c30cd93db8f46d45b2388011b43ced63a6af6a0c7` |
| rotation manifest SHA256 | `df30fbb90bfafaef787cfe73bd55c9179ac2acb83d2a23ed977a8378d7c19926` |
| rotations SHA256 | `0a966da2e480559b698e4347ba29f402f5eb9fe335781ac41f8c27086a72808e` |
| search summary SHA256 | `9dfe16a88a3c46dfff8e10b62a8f47a7bc716996c6e3d51b3b6b7daeb212792e` |
| native manifest SHA256 | `bc167e80cb20734cbd137d77912e28a222391f42b1427fc38cc06057df706050` |

正式构建开始前，主仓库和源码仓库均为干净状态，HEAD 与各自 upstream 完全一致。

## 4. 候选镜像身份

| 字段 | 实际值 |
| --- | --- |
| Tag | `glm52-oscar-a800-phase6-7d317f1de-df30fbb9` |
| Image ID / config digest | `sha256:5ad3094114d68778cd743e971653aaf62f3c2e0464a2e69c62c28120e2145f7c` |
| Manifest digest | `sha256:c2939feb779757c8f4c7a500300085b1ddee287975941588a19c305e3b602ec9` |
| Architecture / OS | `amd64` / `linux` |
| Layers | 33 |
| Reproducible created 字段 | `2026-07-26T14:42:11Z` |
| Candidate layer digest | `sha256:8ad9ace913c624cee36efef0d514197c48ec0e8cadf01b8c7ffd253c46805225` |
| Candidate layer diff ID | `sha256:215580945865363fb8a3b53b88f6e90dc5ac9e27b7dbee3de2dfd005bed3c044` |
| Candidate layer size | 109,133,958 bytes |
| Candidate layer tar members | 5,294 |

`created` 使用冻结源码 commit 时间，而不是每次构建墙钟时间，使相同输入能够得到
相同 config 和 manifest digest。候选环境固定：

```text
PYTHONPATH=/opt/vllm_glm52_v1
VLLM_OSCAR_MLA_ROTATION_ARTIFACT=/opt/oscar_artifacts/rotation_fit_v2
```

## 5. 独立解包与内容验收

正式 verifier 的实际结果：

| 检查 | 实际结果 |
| --- | --- |
| 基础层 | 前 32 个 layer descriptors 与 phase 0 完全一致 |
| 候选层安全性 | 无 `.so`、无 OCI whiteout、无绝对路径或 `..` 路径 |
| 源码 | 4,742/4,742 Git blobs、symlinks、executable modes 全部匹配 |
| Rotation artifact | 3/3 SHA256 匹配 |
| Native extensions | 7/7 基础层 SHA256 匹配 |
| Native 覆盖 | 候选层未覆盖任何 native extension |
| Verifier 状态 | `passed` |

本地证据目录为
`artifacts/phase6/20260726T151933Z_candidate_7d317f1de`。其中：

| 文件 | SHA256 |
| --- | --- |
| `build_result.json` | `48b8b12e877f4904c9b7fb93fbe19042262797ffa6e821ffe976e8e05a7b1700` |
| `verification.json` | `eaad43a4c846404d5183e72352fa89258fa4d53192423af6d5f55521b374280c` |
| `runtime_import.json` | `d3a67dba2d18b2a198653c3f3bbd0f0e8529c759bead84dd5e72acce6c17ab3b` |
| `rebuild_result.json` | `8fd9bcd1979152049571657ecf776fad25d3536b93ed0ff058b5404ef13d19ab` |

第二次构建报告与第一次的文件 SHA256 不同，是因为报告忠实记录了不同的输出 layout
路径。删除该路径字段后两个报告完全一致；更重要的是，两次 layer digest、diff ID、
image ID、config digest 和 manifest digest 逐项相同。

## 6. 固定运行时

候选解包层与 phase 0 基础 rootfs 组成等价 overlay 后，实际导入结果如下：

| 组件 | 实际版本/结果 |
| --- | --- |
| CUDA image environment | 12.9.1 |
| Python | 3.12.13 |
| PyTorch | `2.11.0+cu129` |
| Triton | `3.6.0` |
| Transformers | `5.8.1` |
| Tokenizers | `0.22.2` |
| vLLM dist-info | `0.11.2.dev278+gdbc3d9991` |
| vLLM version override | `0.11.2.dev278+v7tpotbench` |
| FlashInfer Python | `0.6.6` |
| FlashInfer JIT cache | `0.6.6+cu129` |
| `vllm._C` | 从候选合并视图成功导入 |
| Rotation tensors | 78 |
| CUDA initialized | `false` |

源码快照未包含由 setuptools-scm 生成的 `vllm._version`，所以直接读取
`vllm.__version__` 显示 `dev` 并产生已有 warning；package dist-info 和候选
环境的 version override 如上。这与阶段 5 已通过的运行时行为一致，不是 native
extension 导入失败。

## 7. 运行边界

由于当前容器不能 mount OCI rootfs 并把 NFS 模型路径 bind 进子容器，阶段 7 将使用
本次 OCI 最后一层的已验收解包目录与 phase 0 基础 rootfs 构成等价 overlay，并在每次
服务前重新校验 manifest/config/layer digest。源码和 rotation 只能从该解包目录读取，
不再从可变 Git 工作树或阶段 2 原始 artifact 路径读取。

这项限制只改变本机启动 OCI 的方式，不改变候选内容身份；但报告会继续明确区分
“标准 OCI 已构建并校验”和“没有 Docker daemon 可直接 `docker run`”，不把后者
伪装成已执行。

## 8. 阶段出口

阶段 6 已完成。候选镜像身份、源码、依赖、native extensions 和 rotation artifact
均已冻结且可重复构建。任何后续代码或 artifact 变化都必须生成新的 tag、image ID
和 manifest digest，不能覆盖当前候选。下一阶段只使用该候选 overlay 执行
official_v4 2,360 个 accuracy 样本和 WikiText‑2 PPL。
