# Configuration and dataset guide

Run all commands from the repository root. Relative dataset paths in the checked-in templates are interpreted from that working directory.

## Configuration templates

| Dataset | S0 | S1 | S2 | Patch size | S0 / S1 / S2 updates |
| --- | --- | --- | --- | --- | --- |
| SEN1-2 | `UARD_anchor_s0.json` | `UARD_E_full_s1.json` | `UARD_E_full_s2.json` | 256 × 256 | 100,000 / 600,000 / 200,000 |
| QXS-SAROPT | `UARD_QXS_s0.json` | `UARD_QXS_s1.json` | `UARD_QXS_s2.json` | 256 × 256 | 100,000 / 600,000 / 200,000 |
| SAR2Opt | `UARD_SAR2Opt_512_s0.json` | `UARD_SAR2Opt_512_s1.json` | `UARD_SAR2Opt_512_s2.json` | 512 × 512 | 20,000 / 100,000 / 50,000 |

All files reside in `config/`. They are development templates, not the final paper recipe. In particular, an S1 template budget of 600,000 updates does not change the actual step count of a historical 420,000-update checkpoint.

All checked-in `gpu_ids` values are `[0]`. Templates use `portable_pool: true` and disable GAN loss. No compiled SoftPool extension or default `vision-aided-loss` installation is needed.

Only S0 training is executable in this source preview. S1/S2 templates are provided for inspection. Changing configuration flags does not restore the two withheld function bodies.

## Dataset roots

The templates use:

```text
datasets/
  SEN12/
    train/
      sen1/scene_001.png
      sen2/scene_001.png
    val/
      sen1/scene_002.png
      sen2/scene_002.png
  QXS-SAROPT/
    train/
      SAR/scene_001.png
      Optical/scene_001.png
    val/
      SAR/scene_002.png
      Optical/scene_002.png
  SAR2Opt/
    train/
      SAR/scene_001.png
      Optical/scene_001.png
    val/
      SAR/scene_002.png
      Optical/scene_002.png
```

The QXS/SAR2Opt `--prepare-only` pipeline additionally requires a separate `test/` split with matching modality subdirectories. The checked-in S0 training templates need only the splits they actually load. Do not use test images as validation data for checkpoint selection.

No dataset images or split manifests are included. The templates do not establish a particular split or guarantee spatial independence. Users are responsible for the split, preprocessing, alignment, and any dataset permissions.

## SEN1-2 pairing

`train_anchor.py` uses `uard_io.py`. For SEN1-2, each SAR file under `sen1/` is paired with the **same relative path, filename, and extension** under `sen2/`. Nested subdirectories are supported.

The retained legacy loader in `data/LRHR_dataset.py` constructs SEN1-2 target paths by replacing every `sen1` token with `sen2` in the source path. Therefore, filenames such as `ROIs_sen1_001.png` imply `ROIs_sen2_001.png` for that loader, while S0's lightweight loader expects identical filenames. The neutral `scene_001.png` layout shown above avoids this distinction. Inspect the intended loader rather than assuming the two pairing conventions are interchangeable; no automatic renaming is performed.

## QXS-SAROPT and SAR2Opt pairing

`qxs_data.py` supports explicit `sar_dir` and `optical_dir` configuration fields. If omitted, it looks for exactly one modality directory among these case-insensitive aliases:

- SAR: `SAR`, `sen1`, `s1`.
- Optical: `Opt`, `Optical`, `EO`, `RGB`, `sen2`, `s2`.

Ambiguous or missing modality directories cause an error. `A/` and `B/` are not automatic aliases. Set explicit fields if those names are used, after checking which modality they contain. The SAR2Opt templates explicitly use `SAR/` and `Optical/`.

Files are indexed recursively by relative path without the extension. For example, `SAR/area/scene.png` can pair with `Optical/area/scene.jpg`. Duplicate relative stems, missing pairs, and unexpected paired dimensions are rejected. This training-data convention differs from the independent evaluation tools, which expect flat directories.

Images must already have the intended aligned patch size. QXS and SAR2Opt loaders do not silently resize a mismatched patch. The preparation audit rejects raw integer high-bit-depth or floating-point images that require a different preprocessing procedure. The provided I/O path converts stored images to RGB and normalizes 8-bit image values to `[-1,1]`; it is not a raw calibrated-SAR preprocessing pipeline.

## S0 training settings

```bash
python train_anchor.py --config config/UARD_anchor_s0.json --output experiments/s0 --device cuda:0
```

Useful fields to review before a run:

| Field | Meaning |
| --- | --- |
| `stage` | Must be 0 for `train_anchor.py`. |
| `datasets.train.dataroot` / `datasets.val.dataroot` | Dataset split roots. |
| `datasets.train.batch_size` / `num_workers` | Training loader settings. |
| `datasets.*.data_len` | `-1` uses all pairs found by the loader. |
| `train.n_iter` | Target cumulative update count. |
| `train.validation_enabled` | Whether the dedicated S0 trainer loads validation data and selects by validation likelihood. |
| `train.val_freq` | Validation interval when validation is enabled. |
| `train.save_checkpoint_freq` | Checkpoint interval when validation is disabled. |
| `model.uard.portable_pool` | Portable pooling path used by the supplied templates. |

The SEN1-2 S0 template disables validation. Its checkpoints follow the fixed training budget. QXS and SAR2Opt S0 templates enable validation; their dedicated S0 trainer records the minimum validation Laplace negative log-likelihood checkpoint. A negative NLL value is not, by itself, an error for a continuous density.

The `--resume` argument takes a same-stage checkpoint **prefix**, without `_gen.pth`. Resume only with the matching configuration and optimizer state. Do not treat a weights-only checkpoint as a complete training-resume record.

## Preparation-only pipelines

```bash
python train_qxs.py --data-root datasets/QXS-SAROPT --sar-dir SAR --optical-dir Optical --run-dir experiments/qxs_prepare --gpu 0 --prepare-only
python train_sar2opt.py --data-root datasets/SAR2Opt --sar-dir SAR --optical-dir Optical --run-dir experiments/sar2opt_prepare --gpu 0 --prepare-only
```

Both inspect `train`, `val`, and `test`, check paired dimensions, and write run configuration/state files without copying images. Use a new run directory.

The SAR2Opt entry point defaults to 1,305 training, 145 validation, and 627 test pairs. If your valid split has other counts, supply the actual values with `--expected-counts TRAIN VAL TEST`. The SAR2Opt audit also rejects a relative sample ID appearing in multiple splits. That ID check does not, by itself, detect adjacent or overlapping geographic content.

The preparation command does not run S0. Train S0 separately with `train_anchor.py` if desired. A full pipeline invocation without `--prepare-only` is intentionally blocked before training in this release.

## Independent image evaluation

```bash
python tools/check_pairs.py --pred-root predictions --gt-dir targets --expected-count 1600 --sha256 --output pair_audit.json
python tools/evaluate.py --pred_root predictions --gt_dir targets --device cpu --expected-count 1600 --output results/evaluation
```

The hyphenated checker options and underscored evaluator options are intentionally different. These commands require already generated predictions. They do not use a model configuration, load a UARD checkpoint, or restore the withheld generator.

Use matching complete filename stems and a new/empty metric output directory. No prefixes are stripped. Verify split membership and provenance independently; an image count alone cannot establish benchmark equivalence. For the exact five-metric protocol, see the main README.
