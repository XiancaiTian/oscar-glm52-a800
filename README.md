# OSCAR × GLM-5.2 × 苹果800

本仓库用于在 vLLM 中集成 OSCAR、适配 GLM-5.2 的 DSA/MLA 推理，并在单机
8×NVIDIA A800 80GB 环境完成运行、性能与精度验证。

实现路线和验收门槛见
[`docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md`](docs/superpowers/specs/2026-07-24-oscar-glm52-a800-design.md)。

## 目录

- `glm52_speed_up_v2_stable_8th/`：已经完成 苹果800 适配的 GLM-5.2 参考实现与部署脚本。
- `oscar_vllm/`：已有 OSCAR-vLLM 实现、实验报告和复现资料。
- `oscar_vllm/vllm/`：固定到 `XiancaiTian/vllm` 的 OSCAR 开发分支。
- `oscar_vllm/oscar_reference/`：固定到 FutureMLS-Lab/OSCAR 的参考版本。

模型权重、容器镜像、旋转矩阵、实验产物、日志和机器凭据只保留在本地，不提交到
GitHub。

## 获取代码

```bash
git clone --recurse-submodules https://github.com/XiancaiTian/oscar-glm52-a800.git
cd oscar-glm52-a800
git submodule update --init --recursive
```
