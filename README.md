# UARD: Uncertainty-Adaptive Residual Diffusion

**One-step SAR-to-optical image translation with an optical-domain anchor and spatially adaptive perturbations.**

[中文说明](README_zh-CN.md) · [Configuration and data](docs/CONFIGURATION.md) · [Release scope](RELEASE_SCOPE.md)

## Motivation

SAR-to-optical translation must recover optical appearance from a different sensing modality. The prediction difficulty varies across an image. UARD first estimates an optical reference, then uses a spatial predictive scale to guide the perturbations used for generative refinement.

![Conceptual motivation for UARD](assets/figures/motivation.png)

*Conceptual illustration. The optical anchor sets the center of the terminal distribution. The predictive scale controls local noise amplitudes. The illustrated surface and paths are not measured diffusion trajectories.*

## Method overview

UARD connects three components:

1. **Optical anchor and predictive scale.** Heteroscedastic regression learns an optical-domain anchor **A** and a spatial predictive scale **U** from SAR. The scale represents cross-modal prediction uncertainty.
2. **Uncertainty-adaptive residual process.** The residual **A − Y** defines a correction path within the optical domain. Spatial weights **W(U)** modulate Gaussian perturbations along this path.
3. **One-step reconstruction.** A shared denoiser predicts residual and noise. Terminal-state reconstruction and perceptual refinement train the final one-call reconstruction.

![UARD framework schematic](assets/figures/framework.png)

*Framework schematic. S0 learns the anchor and scale. S1 learns residual/noise prediction. S2 refines terminal-state reconstruction. “One step” means one denoising-network evaluation; total inference also includes the S0 predictor.*

## Qualitative comparisons

### SEN1-2

![Qualitative comparison on SEN1-2](assets/figures/qualitative_sen12.png)

<details>
<summary><strong>QXS-SAROPT</strong></summary>

![Qualitative comparison on QXS-SAROPT](assets/figures/qualitative_qxs.png)

</details>

<details>
<summary><strong>SAR2Opt</strong></summary>

![Qualitative comparison on SAR2Opt](assets/figures/qualitative_sar2opt.png)

</details>

## Source preview

S0 training, anchor export, and independent evaluation of existing image outputs are available. The function bodies of `UARDiffusion.q_sample` and `UARDiffusion.generate` are withheld and raise `NotImplementedError`, so S1/S2 training, diffusion generation, and full paper reproduction are unavailable in this preview; see [release scope](RELEASE_SCOPE.md).

The repository retains the model architecture, training loops, data loaders, and nine configuration templates. Run `python release_status.py` to inspect availability.

## Installation

Python 3.10 is recommended. Use matching PyTorch/torchvision builds for your CPU or CUDA environment.

```bash
git clone https://github.com/iroseo/Uncertainty-Adaptive-Residual-Diffusion.git
cd Uncertainty-Adaptive-Residual-Diffusion
python -m pip install -r requirements.txt
```

The requirements provide a suggested dependency profile, not a verified historical environment or a fully GPU-validated lockfile. The supplied configurations use `portable_pool: true`; no SoftPool extension compilation is needed. Optional logging/LMDB packages are listed in [requirements-optional.txt](requirements-optional.txt). LPIPS and FID may download pretrained feature weights on first use.

## S0 training

Prepare aligned 8-bit image pairs using the [dataset layouts](docs/CONFIGURATION.md), then run from the repository root:

```bash
python train_anchor.py --config config/UARD_anchor_s0.json --output experiments/s0 --device cuda:0
```

QXS-SAROPT and SAR2Opt use `config/UARD_QXS_s0.json` and `config/UARD_SAR2Opt_512_s0.json`, respectively. Review dataset paths before training. The checked-in JSON files are development templates, not the final paper recipe. Dataset inspection, preparation-only commands, stage settings, and pairing rules are documented in the [configuration guide](docs/CONFIGURATION.md).

## Evaluate existing predictions

These utilities evaluate image files without invoking UARD generation:

```bash
python tools/check_pairs.py --pred-root predictions --gt-dir targets --expected-count 1600 --sha256 --output pair_audit.json
python tools/evaluate.py --pred_root predictions --gt_dir targets --device cpu --expected-count 1600 --output results/evaluation
```

Use flat directories with matching complete filename stems. Replace `1600` with the actual evaluation count and choose new output paths. Prefixes such as `test_` are not removed. The model-based `evaluate_uard.py` entry point requires the withheld generator and is unavailable in this preview.

The standalone evaluator reports **FID, LPIPS-Alex, PSNR, SSIM, and RGB MSE** (`l2`); SSF is not included. Its LPIPS-Alex scores differ from the VGG-based legacy protocol. See the [metric protocol](docs/CONFIGURATION.md#metric-protocol) for preprocessing and output details.

## Repository guide

| Location | Contents |
| --- | --- |
| `model/` | Anchor/scale predictor, denoiser, diffusion source, losses, and training wrapper |
| `main.py`, `train_anchor.py`, `train_qxs.py`, `train_sar2opt.py` | Training and preparation entry points |
| `config/` | Nine dataset/stage development templates |
| `data/`, `core/`, `uard_io.py`, `qxs_data.py` | Data loading and supporting utilities |
| `tools/` | Independent image pairing and evaluation |
| `assets/figures/` | Method illustrations and qualitative comparisons |

## Acknowledgments and licensing

The implementation builds on [E3Diff](https://github.com/DeepSARRS/E3Diff) and its upstream [SR3 PyTorch implementation](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement), with pooling infrastructure from [SoftPool](https://github.com/alexandrosstergiou/SoftPool). Source origins, applicable third-party notices, and preview modifications are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [licenses/](licenses/), and [SOURCE_CHANGES.md](docs/SOURCE_CHANGES.md).

No additional UARD-specific project license has been selected for this preview. Third-party code retains its applicable terms. Datasets and checkpoints are not bundled.
