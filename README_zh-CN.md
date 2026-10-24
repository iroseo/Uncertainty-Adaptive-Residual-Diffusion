# UARD: Uncertainty-Adaptive Residual Diffusion for One-Step SAR-to-Optical Image Translation

**当前为源码预览版，仅暂不公开两个核心函数的函数体。**

仓库保留真实的模型结构、光学锚点与预测尺度网络、训练基础设施、数据加载器、配置模板和评测工具。`UARDiffusion.q_sample` 与 `UARDiffusion.generate` 的函数体暂不公开，调用时明确抛出 `NotImplementedError`。

**S0 可训练并导出锚点；已有图像可独立评测。S1/S2 训练和扩散生成入口会提前阻止执行，当前版本不能端到端复现论文结果。**

[English](README.md) · [发布范围](RELEASE_SCOPE.md) · [配置与数据说明](docs/CONFIGURATION.md)

## 当前可以做什么

| 操作 | 状态 |
| --- | --- |
| 阅读真实模型、骨干、锚点/尺度、损失、调度器及训练封装 | 已提供 |
| 使用自己的数据训练 S0 锚点/尺度预测器 | 可用 |
| 检查数据、准备配置 | 可用 |
| 对已有预测图像计算五项标准指标 | 可用 |
| 训练 S1/S2、执行扩散生成或运行 S2 模型评测 | 暂不支持，入口明确阻止 |
| 独立复现完整论文结果 | 当前版本不支持 |

所有示例均从仓库根目录执行：

```bash
python release_status.py
python release_status.py --check s0
python release_status.py --check metrics
python release_status.py --check prepare
```

不带参数时输出 JSON。`s0`、`metrics`、`prepare` 检查通过；`s1`、`s2`、`inference` 返回退出码 2。该检查表示发布能力是否开放，不是数据、依赖和 GPU 自检。相关入口保留 `--help`。

## 环境

建议 Python 3.10。先安装适合当前 CPU/CUDA 的 PyTorch 与 torchvision，再执行：

```bash
python -m pip install -r requirements.txt
```

这些版本是建议依赖组合，不代表已核实的历史实验环境，也不是已验证所有平台的完整锁定环境。公开包未完成 GPU 端到端训练验证。`wandb`、`lmdb` 见可选依赖：

```bash
python -m pip install -r requirements-optional.txt
```

配置使用 `portable_pool: true`，无需编译 SoftPool。模板未启用 GAN 损失，默认依赖不包含 `vision-aided-loss`。

LPIPS/FID 的外部预训练特征网络可能在首次运行时自动下载权重。离线使用需准备对应缓存；仓库不附带预训练权重。

## S0 训练

先按 [配置说明](docs/CONFIGURATION.md) 放置预处理后的 8 位图像对：

```bash
python train_anchor.py --config config/UARD_anchor_s0.json --output experiments/s0 --device cuda:0
```

QXS 和 SAR2Opt 的独立 S0 训练可分别改用 `config/UARD_QXS_s0.json` 和 `config/UARD_SAR2Opt_512_s0.json`，并设置不同的输出目录。S0 的真实实现保留，无需被暂不公开的两个函数参与训练。

九个配置为开发模板，不代表论文最终实验配方。SEN1-2/QXS 的 S1 模板预算为 600,000 步，不应据此认定某个已评测的 420,000 步检查点训练了 600,000 步。

## 仅准备配置

当前 QXS/SAR2Opt 联合流水线只允许 `--prepare-only`。不带该参数会在训练 S0 前停止，避免训练到 S1 才失败：

```bash
python train_qxs.py --data-root datasets/QXS-SAROPT --sar-dir SAR --optical-dir Optical --run-dir experiments/qxs_prepare --gpu 0 --prepare-only
python train_sar2opt.py --data-root datasets/SAR2Opt --sar-dir SAR --optical-dir Optical --run-dir experiments/sar2opt_prepare --gpu 0 --prepare-only
```

此操作检查 `train/val/test` 并生成运行配置，不复制原图、不训练模型。仓库内训练模板只直接引用 `train/val`。SAR2Opt 默认还检查 1305/145/627 的划分数量，实际数量不同需显式提供 `--expected-counts`。详见 [数据说明](docs/CONFIGURATION.md)。

## 对已有图像评测

```bash
python tools/check_pairs.py --pred-root predictions --gt-dir targets --expected-count 1600 --sha256 --output pair_audit.json
python tools/evaluate.py --pred_root predictions --gt_dir targets --device cpu --expected-count 1600 --output results/evaluation
```

两个目录均应为平面目录，以完整文件主名配对，不自动移除 `test_` 前缀。`1600` 仅为数量示例，应换成实际评测数量；数量相同不能证明是同一测试集。输出目录须为新目录或空目录。

五项指标为 FID、LPIPS-Alex、PSNR、SSIM 和 RGB MSE。MSE 输出字段名为 `l2`，不是欧氏范数或 RMSE；此版本不提供 SSF。成对指标使用 RGB float32 `[0,1]`，不额外裁剪、缩放、配准或调整颜色。FID 由 torch-fidelity 对配对原文件计算，特征网络预处理由其内部完成。

输出包括逐图 CSV、汇总 JSON 和 TXT。完整五指标汇总仅在 FID 成功后写入。`evaluate_uard.py` 属于需要生成模型的旧评测入口，当前被阻止；已有图像请使用 `tools/evaluate.py`。

## 发布边界

本次保留真实源码，仅用明确的异常替代两个暂不公开的函数体，不提供伪造的替代算法。数据、权重、实验输出及私人开发历史不随包发布。后续发布时间不作承诺。

第三方代码与依赖遵循其各自许可及随附归属说明。此预览版尚未选定额外的 UARD 项目许可证，发布不构成对未公开函数体、数据和权重的授权。

## 代码来源

[Third-party notices](THIRD_PARTY_NOTICES.md) · [Source-preview modifications](docs/SOURCE_CHANGES.md)
