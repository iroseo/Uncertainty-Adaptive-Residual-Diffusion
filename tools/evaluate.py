"""Evaluate existing paired images with five standard metrics (Python 3.8+).

Protocol: RGB float32 [0,1], scikit-image PSNR/SSIM defaults, LPIPS-Alex,
L2=MSE, and torch-fidelity FID on the matched original image files.
This utility contains no UARD model, training, or generation implementation.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError('Image directory does not exist: ' + str(directory))
    return sorted((p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS), key=lambda p: p.name)


def collect_pairs(pred_root, gt_dir, expected_count=None):
    """Match full stems, including UARD's non-numeric SEN12 filenames."""
    predictions = image_files(pred_root)
    targets = image_files(gt_dir)
    gt_by_stem = {}
    for target in targets:
        if target.stem in gt_by_stem:
            raise ValueError('Duplicate GT stem: ' + target.stem)
        gt_by_stem[target.stem] = target
    pairs, seen, missing = [], set(), []
    for prediction in predictions:
        name = prediction.stem
        if name in seen:
            raise ValueError('Duplicate prediction stem: ' + name)
        seen.add(name)
        target = gt_by_stem.get(name)
        if target is None:
            missing.append(prediction.name)
        else:
            pairs.append((prediction, target, name))
    if missing:
        raise ValueError('Missing GT for {} predictions (not silently skipped): {}'.format(
            len(missing), ', '.join(missing[:10])))
    if not pairs:
        raise ValueError('No prediction/GT pairs found.')
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError('Expected {} pairs, found {}.'.format(expected_count, len(pairs)))
    if len(pairs) < 2:
        raise ValueError('FID requires at least two images per distribution.')
    print('Prediction images: {} | Paired images: {} | Unused GT: {}'.format(
        len(predictions), len(pairs), len(targets) - len(pairs)), flush=True)
    return pairs


def read_rgb(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert('RGB')
        return np.asarray(image, dtype=np.float32) / 255.0


def img_to_lpips_tensor(img, device):
    import torch
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def paired_metrics(pairs, device, lpips_fn):
    import numpy as np
    import torch
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    rows = []
    with torch.no_grad():
        for index, (fake_path, real_path, name) in enumerate(pairs, start=1):
            fake = read_rgb(fake_path)
            real = read_rgb(real_path)
            if fake.shape != real.shape:
                raise RuntimeError('Shape mismatch for {}: {} {} vs {} {}'.format(
                    name, fake_path, fake.shape, real_path, real.shape))

            # EXACT reference protocol. No crop, resize, Y-channel, extra SSIM
            # options, square root of L2, or LPIPS normalization shortcut.
            psnr = peak_signal_noise_ratio(real, fake, data_range=1.0)
            ssim = structural_similarity(real, fake, data_range=1.0, channel_axis=2)
            l2 = np.mean((fake - real) ** 2)
            fake_tensor = img_to_lpips_tensor(fake, device)
            real_tensor = img_to_lpips_tensor(real, device)
            lpips_value = lpips_fn(fake_tensor, real_tensor).view(-1).mean().item()
            rows.append(dict(sample=name, prediction=str(fake_path.resolve()),
                             ground_truth=str(real_path.resolve()), psnr=float(psnr),
                             ssim=float(ssim), lpips=float(lpips_value), l2=float(l2)))
            if index % 100 == 0 or index == len(pairs):
                print('Paired metrics: {}/{}'.format(index, len(pairs)), flush=True)
    return rows


def link_or_copy(source, destination):
    # Preserve the original image bytes, exactly as in the supplied FID script.
    try:
        os.symlink(Path(source).resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def calculate_fid(pairs, device, tmp_dir=None):
    from torch_fidelity import calculate_metrics
    with tempfile.TemporaryDirectory(dir=tmp_dir) as folder:
        fake_dir = Path(folder) / 'fake'
        real_dir = Path(folder) / 'real'
        fake_dir.mkdir()
        real_dir.mkdir()
        for index, (fake_path, real_path, _) in enumerate(pairs):
            link_or_copy(fake_path, fake_dir / '{:08d}.png'.format(index))
            link_or_copy(real_path, real_dir / '{:08d}.png'.format(index))
        print('FID images: {} per distribution'.format(len(pairs)), flush=True)
        fid_metrics = calculate_metrics(
            input1=str(fake_dir), input2=str(real_dir),
            cuda=(device.type == 'cuda'), isc=False, fid=True, kid=False, verbose=False)
        return float(fid_metrics['frechet_inception_distance'])


def summarize(rows, fid_value):
    import numpy as np
    return dict(count=len(rows), fid=float(fid_value), **{
        name: float(np.mean([row[name] for row in rows]))
        for name in ('lpips', 'psnr', 'ssim', 'l2')})


def save_per_image(rows, output):
    with (output / 'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample', 'prediction', 'ground_truth', 'psnr', 'ssim', 'lpips', 'l2'])
        writer.writeheader()
        writer.writerows(rows)


def package_versions():
    versions = {}
    for name in ('numpy', 'Pillow', 'scikit-image', 'torch', 'lpips', 'torch-fidelity'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unknown'
    return versions


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--result_root', help='UARD directory containing sample/ and tgt/')
    inputs.add_argument('--pred_root', help='Flat directory of predictions; requires --gt_dir')
    parser.add_argument('--gt_dir', help='Flat GT directory; defaults to result_root/tgt')
    parser.add_argument('--device', default='cuda:0', help='cpu or cuda:N, e.g. cuda:2')
    parser.add_argument('--expected-count', type=int, help='Use 1600 to require the full SEN12 test set')
    parser.add_argument('--output', help='New/empty output directory; default: result_root/metrics_five')
    parser.add_argument('--tmp_dir', help='Existing temporary parent for FID; default: system temp')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_count is not None and args.expected_count < 2:
        parser.error('--expected-count must be at least 2 for FID')
    if args.result_root:
        root = Path(args.result_root).resolve()
        pred_root = root / 'sample'
        gt_dir = Path(args.gt_dir).resolve() if args.gt_dir else root / 'tgt'
        default_output = root / 'metrics_five'
    else:
        if not args.gt_dir:
            parser.error('--pred_root requires --gt_dir')
        pred_root = Path(args.pred_root).resolve()
        gt_dir = Path(args.gt_dir).resolve()
        default_output = pred_root.parent / (pred_root.name + '_metrics_five')
    output = Path(args.output).resolve() if args.output else default_output
    if output in (pred_root, gt_dir):
        raise ValueError('Output must be separate from the image directories.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Output is not empty; choose another --output: ' + str(output))
    if args.tmp_dir and not Path(args.tmp_dir).is_dir():
        raise FileNotFoundError('Temporary parent does not exist: ' + args.tmp_dir)
    pairs = collect_pairs(pred_root, gt_dir, args.expected_count)

    # Lazy imports let --help and pairing checks run without loading pretrained
    # models. Never substitute a different metric when a dependency is missing.
    import torch
    import lpips
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('--device must be cpu or cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable.')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        # torch-fidelity's cuda=True uses the current CUDA device. Select it so
        # --device cuda:2 is respected by BOTH LPIPS and FID, rather than GPU 0.
        torch.cuda.set_device(device)
    print('UARD Evaluation | LPIPS-Alex | skimage SSIM | RGB [0,1] MSE', flush=True)
    print('Prediction:', pred_root, '\nGT:', gt_dir, '\nDevice:', device, flush=True)
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    rows = paired_metrics(pairs, device, lpips_fn)
    save_per_image(rows, output)
    # On FID failure, only the paired CSV exists; no completed five-metric
    # summary is written, and the error is allowed to propagate.
    fid_value = calculate_fid(pairs, device, args.tmp_dir)
    means = summarize(rows, fid_value)
    summary = dict(means)
    summary.update(complete=True, prediction_directory=str(pred_root),
                   gt_directory=str(gt_dir), device=str(device), versions=package_versions(),
                   protocol=dict(
                       images='PIL RGB, float32 / 255, no resize/crop; full filename-stem pairing',
                       psnr='skimage peak_signal_noise_ratio(real, fake, data_range=1.0)',
                       ssim='skimage structural_similarity(real, fake, data_range=1.0, channel_axis=2)',
                       lpips='lpips.LPIPS(net=alex), inputs manually mapped to [-1,1]',
                       l2='mean((fake-real)**2) on RGB float32 [0,1] (MSE, not RMSE)',
                       fid='torch_fidelity.calculate_metrics defaults; cuda as requested; '
                           'isc=False, fid=True, kid=False, verbose=False; paired originals only',
                       averaging='Arithmetic mean of per-image PSNR, SSIM, LPIPS, L2'))
    # Keep mathematically infinite PSNR valid in JSON without nonstandard JSON
    # numbers. Finite metric values are stored without rounding.
    if summary['psnr'] == float('inf'):
        summary['psnr'] = 'Infinity'
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = ['UARD Evaluation Results', 'Evaluated images: {}'.format(len(rows)),
             'FID   : {:.6f}'.format(fid_value),
             'LPIPS : {:.6f}'.format(means['lpips']),
             'PSNR  : {:.6f} dB'.format(means['psnr']),
             'SSIM  : {:.6f}'.format(means['ssim']),
             'L2    : {:.6f}'.format(means['l2'])]
    text = '\n'.join(lines) + '\n'
    (output / 'summary.txt').write_text(text, encoding='utf-8')
    print('\n' + text + 'Saved to: ' + str(output), flush=True)
    return summary


if __name__ == '__main__':
    main()
