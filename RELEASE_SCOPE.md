# Source-preview release scope

This release makes the actual UARD source structure inspectable while withholding two core function bodies.

## Included source

- UARD model and backbone code, including optical-anchor and predictive-scale prediction.
- Supporting diffusion infrastructure, losses, schedules, model wrapper, and common modules.
- S0 training code, S1/S2 training infrastructure, and generation/evaluation entry points.
- Dataset loaders, image I/O, logging and utility infrastructure.
- Nine dataset/stage configuration templates.
- Independent image-pair checking and five-metric evaluation utilities.

## Exact implementation redactions

Only the bodies of these methods are withheld:

1. `UARDiffusion.q_sample`: the UARD forward-state construction.
2. `UARDiffusion.generate`: UARD generation.

Their public bodies raise `NotImplementedError`. No substitute algorithm is presented as the original implementation.

S1/S2 and generation entry points have release guards so unsupported commands stop early. The QXS/SAR2Opt pipeline allows `--prepare-only`; full-pipeline execution stops before S0 training begins. Command-line help remains available. The stage-0 configuration can still be used with `infer_uard.py` to export the S0 optical anchor and predictive maps; it does not run diffusion generation.

## Availability checks

`python release_status.py` prints release status as JSON. From the repository root:

| Check | Result |
| --- | --- |
| `python release_status.py --check s0` | Available |
| `python release_status.py --check metrics` | Available |
| `python release_status.py --check prepare` | Available |
| `python release_status.py --check s1` | Unavailable; exit status 2 |
| `python release_status.py --check s2` | Unavailable; exit status 2 |
| `python release_status.py --check inference` | Unavailable; exit status 2 |

An available status does not certify installed dependencies, data validity, numerical performance, or hardware compatibility.

## What can be run

The dedicated `train_anchor.py` entry point can train the S0 anchor/scale predictor with user-provided data and dependencies. The independent tools can check and evaluate existing image outputs without the UARD generator. Dataset preparation can inspect images and write run configurations.

The release cannot train UARD S1/S2, generate UARD optical predictions, or independently reproduce the paper's end-to-end results. No checkpoints, datasets, generated collections, private experiment records, or private Git history are included.

## Templates and validation limits

Configuration budgets and settings are development templates, not a declaration of the final paper recipe. The dependency pins provide a suggested installation profile and are not a verified historical environment or a fully tested platform lock. GPU end-to-end training has not been validated for this packaged release.

Independent evaluation includes FID, LPIPS-Alex, PSNR, SSIM, and RGB MSE only. SSF is not included. Feature-network weights are external and may be downloaded by their libraries on first use.

## Availability and licensing

No date is promised for additional implementation. Third-party code and dependencies retain their applicable licenses and attribution. No additional UARD-specific license has been selected for this preview. Public availability does not grant access or rights to withheld function bodies, private data, or model weights.
