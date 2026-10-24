"""Evaluate a fixed UARD S2 checkpoint on every paired test image (Python 3.8+)."""
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time


def evaluate_loader(loader, model, metrics, output, expected_count, seed):
    import cv2
    import numpy as np
    import torch
    from core import metrics as image_metrics

    # A fresh directory prevents mixing checkpoints, subsets, or interrupted runs.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for kind in ('sample', 'tgt', 'src'):
        (output / kind).mkdir()
    rows, seen = [], set()
    started = time.perf_counter()
    fields = ['index', 'filename', 'psnr', 'ssim', 'lpips', 'mse', 'inference_seconds']
    with (output / 'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index, data in enumerate(loader, 1):
            if data['HR'].shape[0] != 1:
                raise ValueError('Evaluation requires batch_size=1')
            name = data['filename'][0]
            if name in seen or Path(name).name != name or '\\' in name:
                raise ValueError('Duplicate or unsafe output filename: ' + name)
            seen.add(name)
            model.feed_data(data)
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)
            tick = time.perf_counter()
            with torch.no_grad():
                model.test(continous=False, seed=seed)
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)
            elapsed = time.perf_counter() - tick
            visuals = model.get_current_visuals()
            images = {kind: image_metrics.tensor2img(visuals[key])
                      for kind, key in [('sample', 'SR'), ('tgt', 'HR'), ('src', 'LR')]}
            prediction, target = images['sample'], images['tgt']
            if prediction.shape != target.shape:
                raise ValueError('Prediction/target shape mismatch: ' + name)
            with torch.no_grad():
                row = dict(index=index, filename=name,
                           psnr=float(image_metrics.calculate_psnr(prediction, target)),
                           ssim=float(image_metrics.calculate_ssim(prediction, target)),
                           lpips=float(metrics.cal_lpips(prediction, target)),
                           mse=float(metrics.cal_l2(prediction, target)),
                           inference_seconds=elapsed)
            if any(not np.isfinite(row[k]) for k in ('ssim', 'lpips', 'mse')):
                raise ValueError('Non-finite metric: ' + name)
            for kind, array in images.items():
                if not cv2.imwrite(str(output / kind / name), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
                    raise OSError('Image write failed: ' + str(output / kind / name))
            writer.writerow(row)
            f.flush()
            rows.append(row)
            if index == 1 or index % 50 == 0 or index == expected_count:
                total = time.perf_counter() - started
                eta = total / index * max(0, expected_count - index)
                print('[{}/{}] elapsed {:.1f}s, ETA {:.1f}s'.format(
                    index, expected_count, total, eta), flush=True)
    if len(rows) != expected_count:
        raise ValueError('Expected {} images, evaluated {}; no complete summary written'.format(
            expected_count, len(rows)))
    return dict(count=len(rows), **{
        key: float(np.mean([r[key] for r in rows]))
        for key in ('psnr', 'ssim', 'lpips', 'mse', 'inference_seconds')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='New output directory; must not already exist')
    parser.add_argument('--gpu_ids', default='2', help='One physical GPU index, or cpu')
    parser.add_argument('--expected-count', type=int, default=1600)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    from release_status import require_available
    require_available("inference")
    if args.gpu_ids != 'cpu':
        int(args.gpu_ids)  # One GPU only, same batch-one evaluation as main.py.
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.gpu_ids == 'cpu' else args.gpu_ids

    import numpy as np
    import torch
    import torchvision
    import data as Data
    import model as Model
    from core import logger as Logger
    from core import metrics as Metrics

    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    cfg.update(phase='val', distributed=False, seed=args.seed, xT_noise_r=0,
               gpu_ids=[] if args.gpu_ids == 'cpu' else [int(args.gpu_ids)])
    cfg['path']['resume_training'] = False
    if cfg['stage'] != 2 or cfg['ddim_steps'] != 1:
        raise ValueError('Expected S2 one-step configuration')
    if cfg['datasets']['val']['data_len'] != -1:
        raise ValueError('Full evaluation requires data_len=-1')
    if cfg['model']['beta_schedule']['val']['n_timestep'] != 1:
        raise ValueError('Expected one sampling step')
    checkpoint = Path(cfg['path']['resume_state'] + '_gen.pth')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu_ids != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this environment')
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = Data.create_dataset(cfg['datasets']['val'], 'val')
    if len(dataset) != args.expected_count:
        raise ValueError('Expected {} pairs, found {}'.format(args.expected_count, len(dataset)))
    loader = Data.create_dataloader(dataset, cfg['datasets']['val'], 'val')
    print('Checkpoint:', checkpoint, flush=True)
    print('Full test pairs:', len(dataset), '| Output:', output, flush=True)
    model = Model.create_model(Logger.dict_to_nonedict(cfg))
    metrics = Metrics.cal_metrics(model.device)
    metrics.lpips_vgg.eval()
    metrics.lpips_vgg.requires_grad_(False)
    summary = evaluate_loader(loader, model, metrics, output, args.expected_count, args.seed)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    summary.update(complete=True, checkpoint=str(checkpoint), checkpoint_sha256=digest.hexdigest(),
                   seed=args.seed, sampling_steps=1, torch_version=torch.__version__,
                   torchvision_version=torchvision.__version__,
                   cuda_version=torch.version.cuda,
                   device=torch.cuda.get_device_name(0) if model.device.type == 'cuda' else 'cpu',
                   metric_protocol='RGB uint8 after clamp [-1,1]; existing core.metrics PSNR/SSIM; '
                                   'LPIPS VGG v0.1 normalize=True; arithmetic mean per image; '
                                   'MSE on [0,1]. Timing includes first-call warmup, excludes metrics/I/O.')
    (output / 'config_used.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    temporary.replace(output / 'summary.json')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
