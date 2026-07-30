# 阶段 0 实验报告：恢复并冻结 GLM‑5.2 基线

## 1. 结论

2026-07-24 已满足设计文档中阶段 0 的四项本地技术出口条件：

1. 完整源码可追溯；
2. 关键文件与已验证镜像一致；
3. baseline OCI 镜像可重复构建；
4. 依赖版本和原生扩展清单已冻结。

2026-07-25 已完成独立源码仓库的 public 远端发布和主仓库 submodule 接入。
阶段 0 的本地技术出口与协作发布出口均判定为通过。

## 2. 固定输入

### 2.1 源码

- 原始源码 commit：
  `bfd727e11b0e501bab0a4a943d92ba5ea3b2f980`；
- 原始源码 tree：
  `9b1b1bf2a2f20459a1de4c0dc2ba73f23f6c2923`；
- 镜像 runtime patch commit：
  `a5a2ddfc3fb1b221a6eb41023b254ac54a98bd2c`；
- baseline 构建资产 commit：
  `fd3e0b3772e989cf0d0d73a3d19b252ab82e9cdd`；
- 无 daemon OCI 重建与验证记录 commit：
  `0288235f2b93563c13e6c3750c5797fccbaba70d`；
- public 冻结代码快照 commit：
  `53d8be94f6038e10ab0c344f706c5ffe66a555b8`；
- public 冻结代码快照 tree：
  `ca5d4f035591690fb95fb662c71480a28714a4d8`；
- public 源码仓库：
  `https://github.com/XiancaiTian/glm52_oscar_vllm`；
- 主仓库 submodule 接入 commit：
  `04b9445cd05a914b1716479a6bbdced018ebc177`。

`0288235f2...` 与 `53d8be94f...` 的 tree 均为 `ca5d4f035...`，实际
`git diff` 为空。按 Shawn 的代码同步要求，189,258 个对象、约 185MiB 的恢复
历史仅保留在本地 `recovery/full-history`，远端上传当前冻结代码快照和后续开发
提交；本次快照实际 Git pack 为 33.13MiB。

外部源码路径
`/nfs/AE/zhanghong/workflow/vllm_a/vllm_glm52_v1` 仅用于只读核验。
该仓库原有 6 个未提交的 prefill shape-bucket 修改；镜像中对应文件与 clean HEAD
一致，证明这些未提交修改没有进入本次基线。整个阶段没有修改、暂存、提交、清理或
重置该外部仓库。

### 2.2 已验证镜像

- tag：
  `192.168.14.129:80/ae/vllm_openai_glm52:v0.19.0-v2-stable-2p1d-usagefix-20260625_114616`；
- image ID：
  `sha256:d6faf4d3a5f7f3800a745f8aea15884c881ea20b1100993d83ca8c4f985bd7a5`；
- Docker archive 大小：34,754,212,352 字节；
- Docker archive SHA256：
  `b58fc5cf7874da92f5a97306d47c9d7d507332515a73e6812a7eb6145e9bed23`；
- archive 全量 SHA256 复核结果：通过。

## 3. 源码恢复与一致性审计

从已验证镜像 layer 29–31 恢复出的有效 `/opt/vllm_glm52_v1` 包含 4,711 个
普通文件，其中 4,705 个是原始 commit 的 tracked 文件，另有 6 个 vLLM 原生
扩展。

clean HEAD 与镜像 tracked tree 之间实际只有 4 个源码差异：

| 文件 | 镜像有效 SHA256 | 结果 |
| --- | --- | --- |
| `vllm/compilation/compiler_interface.py` | `6fc8d5bcb187f22cba6f9ba6e397a0242637d9b701102bb56d0635909b9409ba` | 匹配 |
| `vllm/v1/engine/core.py` | `ec5fdf64219a6fd3717dcd8af7835a99601b410a98c87d8f5e9216178dad1888` | 匹配 |
| `vllm/entrypoints/anthropic/serving.py` | `55df276287acdc55e46217bbfe4efc3e79e6a33865ad7b07e54448710cb1fa11` | 匹配 |
| `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` | `11a447a4b243e6e2df410278884c40d578e38bcf493d4860d9d9ef9517a8d4d7` | 匹配 |

`compiler_interface.py` 已通过上一版稳定镜像 manifest 和精确文件 SHA256 闭合
来源；其余 3 个文件与最终 image manifest 记录一致。

## 4. 依赖与原生扩展冻结

镜像内实际读取到的关键版本为：

| 组件 | 实际版本 |
| --- | --- |
| CUDA | 12.9.1 |
| Python | 3.12 |
| PyTorch | `2.11.0+cu129` |
| Triton | `3.6.0` |
| vLLM dist-info | `0.11.2.dev278+gdbc3d9991` |
| vLLM version override | `0.11.2.dev278+v7tpotbench` |
| Transformers | `5.8.1` |
| Tokenizers | `0.22.2` |

设计和部署资产把该定制代码线称为 `custom v0.19.0`，但 Python package metadata
不是 `0.19.0`。这两个字段在本报告中分开记录，不视为同一版本字段。

冻结的 6 个 vLLM 原生扩展和 1 个 sparse MLA splitmerge 扩展，均在展开的候选
rootfs 上通过 SHA256，实际结果为 7/7 `OK`。完整清单位于
`glm52_oscar_vllm/recovery/native_extensions.sha256`。

## 5. 候选镜像重建结果

当前 Kubernetes 容器没有 Docker/containerd socket，也没有 `CAP_SYS_ADMIN`。
因此实际采用 `skopeo + GNU tar/gzip + OCI descriptor` 的无 daemon 路径重建，
而不是声称执行了不可用的 Docker daemon。

候选结果如下：

| 字段 | 实际值 |
| --- | --- |
| base OCI manifest | `sha256:1b8e808e44e7ef50be0cc8ff874597c37e09a2a428f06092ab570e6079086415` |
| candidate OCI manifest | `sha256:2fdfbe865aecc01eee15a01fcce58bf7581244dbbc53cbe3ef0e0cce44bc489d` |
| candidate OCI config | `sha256:58a853ee730c263968dcc6b76401e85e4b510a140777c4ffc791747edd8ea42d` |
| source layer digest | `sha256:352d47f649171770e32edb7e1112e8a31f6a5aead0f6160a30ad3a8eec6659c3` |
| source layer diff ID | `sha256:e9287018ae318195cd2bdc8fdd88b5c6fd7f85c2c44db453d891c075f538ccf8` |
| source layer 大小 | 33,253,726 字节 |
| source layer 内容 | 5,256 个 tar 成员、4,711 个文件 |

31/31 个基础层 descriptor 与已验证 OCI 完全相同。新增 source layer 不含
`.so` 和 whiteout，因此不会覆盖或删除已冻结原生扩展。独立第二次生成得到完全
相同的 layer digest、diff ID 和大小，重复构建检查通过。

在已展开候选 rootfs 上再次执行清单，4 个 runtime source 与 7 个 native
extension 实际结果为 11/11 `OK`。

## 6. 未通过项与处理

### 6.1 历史 patch lint

定向 ruff 0.14.0 检查发现镜像历史 patch 中存在 3 个 E501 和 1 个 SIM102。
为了保持恢复基线与已验证镜像逐字节一致，本阶段没有修改这些历史行。Python 3.12
`py_compile` 和 `git diff --check` 均通过。

### 6.2 `umoci insert`

`umoci insert 0.4.7` 分别从 NFS 和本地 `/tmp` payload 生成 source layer，
两次都固定截断最后 393 字节。两个失败 tag 没有导出或发布，后续停止使用该路径，
改为标准 GNU tar/gzip 组装。

### 6.3 完整 rootfs 解包

`umoci unpack` 在 NFS 上运行约 72 分钟后终止。70 分钟节点累计读取约
20.38GB、写入约 35.11GB；文件内容已展开且 11/11 SHA256 通过，但命令仍停留在
xattr/元数据处理，因此“unpack 命令完成”明确记录为未通过，不伪造完成状态。

## 7. 阶段出口判定

| 出口条件 | 判定 | 证据 |
| --- | --- | --- |
| 完整源码可追溯 | 通过 | 原始 commit/tree、完整 Git 历史、4 个 runtime patch commit |
| 关键文件与已验证镜像一致 | 通过 | 4/4 runtime source SHA256；展开 rootfs 复核通过 |
| baseline 镜像可重复构建 | 通过 | 两次 source layer digest、diff ID、大小完全相同 |
| 依赖版本和原生扩展清单冻结 | 通过 | 关键 package metadata；7/7 native SHA256 |
| GitHub 远端和 submodule 接入 | 通过 | public 仓库的 `main`/功能分支均解析到 `53d8be94f...`；主仓库 `04b9445...` 已推送 |

阶段 0 全部出口通过。主仓库 submodule 只指向已在 public 远端解析成功的 commit。

## 8. 下一步

按设计文档的阶段顺序，下一步是阶段 1 原生 GLM‑5.2/苹果800 baseline，不直接跳到
OSCAR 适配：

1. 在分配前连续两次检查 8 张 苹果800 空闲状态；
2. 在固定候选环境中完成 TP=8 短请求、超过 320 tokens、连续 decode 和 32K；
3. 冻结 official_v4 与 WikiText‑2 原生 baseline；
4. 阶段 1 出口通过后再开始 calibration、allocator、SM80 kernel 和 vLLM 接入。
