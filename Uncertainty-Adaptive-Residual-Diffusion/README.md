# UARD: Uncertainty-Adaptive Residual Diffusion for One-Step SAR-to-Optical Image Translation

**Source preview with two core function bodies withheld.**

This repository contains the actual UARD model structure, optical-anchor and predictive-scale implementation, training infrastructure, data loaders, configuration templates, and evaluation utilities. The bodies of `UARDiffusion.q_sample` and `UARDiffusion.generate` are not public in this preview. They raise `NotImplementedError` explicitly.

**S0 anchor training/export and evaluation of existing image outputs are available. S1/S2 training and diffusion generation are blocked. This is not a complete reproduction release.**

[中文说明](README_zh-CN.md) · [Release scope](RELEASE_SCOPE.md) · [Configuration and data](docs/CONFIGURATION.md)

## Availability

| Capability | Status |
| --- | --- |
| Read the model, backbone, anchor/scale, losses, schedules, wrappers, and data-loading source | Included |
| Train the S0 anchor/scale predictor with `train_anchor.py` | Available with your own data and dependencies |
| Inspect or prepare configuration templates | Available |
| Evaluate existing predictions with five standard metrics | Available |
| Train S1 or S2 | Blocked: forward-process body withheld |
| Generate images or run model-based evaluation | Blocked: generation body withheld |
| Reproduce the paper end to end | Not supported by this preview |

Check the release status from the repository root:

```bash
python release_status.py
python release_status.py --check s0
python release_status.py --check metrics
python release_status.py --check prepare
```

No argument prints JSON. Checks for `s0`, `metrics`, and `prepare` pass; checks for `s1`, `s2`, and `inference` exit with status 2. These checks report release availability, not whether your dependencies, data, or GPU are ready. Training and inference entry points also stop unsupported operations early. Their `--help` remains available.

## Installation

Python 3.10 is recommended. Install a matching PyTorch/torchvision build for your CPU or CUDA environment, then:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` provides a suggested dependency profile; it is not a verified historical environment or a fully validated cross-platform lockfile. GPU end-to-end training has not been validated for this public package. Optional logging and LMDB dependencies are listed in `requirements-optional.txt`:

```bash
python -m pip install -r requirements-optional.txt
```

The supplied templates set `portable_pool: true`, so compiling the SoftPool extension is unnecessary. GAN loss is disabled in these templates; `vision-aided-loss` is not a default dependency.

LPIPS and FID use external pretrained feature networks. Their libraries may download weights on first use. Offline evaluation requires the appropriate library caches. No pretrained weights are bundled.

## S0 training

All commands below run from the repository root. Prepare normalized 8-bit image pairs and review [the dataset layouts](docs/CONFIGURATION.md) first.

```bash
python train_anchor.py --config config/UARD_anchor_s0.json --output experiments/s0 --device cuda:0
```

This trains the S0 optical-anchor and predictive-scale predictor. To use the QXS or SAR2Opt S0 template, replace the configuration with `config/UARD_QXS_s0.json` or `config/UARD_SAR2Opt_512_s0.json`, and choose a separate output directory.

The checked-in JSON files are **development templates**, not a claim of the paper's final training recipe. In particular, the SEN1-2/QXS S1 template budget is 600,000 updates; it should not be confused with a 420,000-update checkpoint evaluated in an experiment.

## Dataset inspection and configuration preparation

The QXS and SAR2Opt pipeline entry points allow **`--prepare-only` only** in this preview. A request to run the full pipeline stops before training S0.

```bash
python train_qxs.py --data-root datasets/QXS-SAROPT --sar-dir SAR --optical-dir Optical --run-dir experiments/qxs_prepare --gpu 0 --prepare-only
python train_sar2opt.py --data-root datasets/SAR2Opt --sar-dir SAR --optical-dir Optical --run-dir experiments/sar2opt_prepare --gpu 0 --prepare-only
```

These preparation commands inspect `train`, `val`, and `test`; the checked-in training templates reference only `train` and `val`. Preparation does not copy the images or train a model. See [configuration details](docs/CONFIGURATION.md) for required folder names, pairing rules, image sizes, and the SAR2Opt count check.

## Evaluate existing image outputs

The standalone utilities operate on image files you already have. They do not call UARD generation.

```bash
python tools/check_pairs.py --pred-root predictions --gt-dir targets --expected-count 1600 --sha256 --output pair_audit.json
python tools/evaluate.py --pred_root predictions --gt_dir targets --device cpu --expected-count 1600 --output results/evaluation
```

Use flat image directories with matching complete filename stems. The utilities do not remove prefixes such as `test_`. Replace `1600` with your evaluation count, and use a new or empty metric output directory. A matching count does not prove that the test split matches the paper.

The evaluator computes **FID, LPIPS-Alex, PSNR, SSIM, and RGB MSE**. The MSE output field is `l2`; it is not a Euclidean norm or RMSE. SSF is not included.

Paired metrics use Pillow RGB float32 in `[0,1]` with no crop, resize, registration, or color adjustment. PSNR and SSIM use scikit-image with `data_range=1.0`; SSIM uses `channel_axis=2` and otherwise default settings. LPIPS inputs are mapped to `[-1,1]`. Each paired score is averaged over images. FID uses torch-fidelity defaults on matched original files; the library performs its feature-network preprocessing.

Outputs are `per_image_metrics.csv`, `summary.json`, and `summary.txt`. Only successful FID completion produces the complete five-metric summary. On a later FID failure, the paired CSV may already exist.

`evaluate_uard.py` is the retained **model-based** evaluation entry point and is blocked in this preview. For already generated images, use `tools/evaluate.py` as shown above.

## Source map

| Location | Content |
| --- | --- |
| `model/` | UARD model, backbones, anchor/scale code, schedules, losses, and training wrapper |
| `data/`, `core/`, `uard_io.py`, `qxs_data.py` | Data loading and supporting infrastructure |
| `main.py`, `train_anchor.py`, `train_qxs.py`, `train_sar2opt.py` | Training/preparation entry points with release guards |
| `infer_uard.py`, `evaluate_uard.py` | Retained generation/evaluation entry points, currently guarded |
| `config/` | Nine dataset/stage configuration templates |
| `tools/` | Independent pairing and five-metric evaluation |

## Licensing and attribution

See the accompanying attribution material for source origins. Third-party dependencies retain their own licenses. Publication of this preview does not grant access or rights to the withheld function bodies, datasets, or checkpoints. No additional UARD-specific license has been selected for this preview.

## Code provenance

[Third-party notices](THIRD_PARTY_NOTICES.md) · [Source-preview modifications](docs/SOURCE_CHANGES.md)
