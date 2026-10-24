"""SAR-only inference; exports optical anchor and uncertainty beside the result."""
import argparse
import json
from pathlib import Path




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True, help='Full path to *_gen.pth')
    parser.add_argument('--input', required=True, help='SAR image or directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1)
    args = parser.parse_args()
    from release_status import check_config
    check_config(args.config)
    import numpy as np
    from PIL import Image
    import torch
    from model.networks import define_G
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if not (cfg['model'].get('uard') or {}).get('enabled'):
        parser.error('This entry point requires a UARD B/C/D/E configuration')
    cfg.update(phase='val', gpu_ids=[], distributed=False, seed=args.seed, xT_noise_r=0)
    # Stage 1 allows a requested multi-step trajectory; one-step uses the same formula.
    if cfg['stage'] != 0:
        cfg['stage'] = 1
    cfg['model']['beta_schedule']['val'].update(ddim=1, n_timestep=args.steps)
    net = define_G(cfg).to(args.device).eval()
    net.set_new_noise_schedule(cfg['model']['beta_schedule']['val'], args.device,
                               cfg['model']['beta_schedule']['train']['n_timestep'])
    net.load_state_dict(torch.load(args.weights, map_location=args.device, weights_only=True), strict=True)
    source, destination = Path(args.input), Path(args.output)
    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    images = sorted(p for p in source.rglob('*') if p.suffix.lower() in extensions) if source.is_dir() else [source]
    if not images:
        parser.error('No SAR images found')
    destination.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, path in enumerate(images):
            # Match the repository's normalized 8-bit RGB loader (not raw calibrated SAR).
            array = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 127.5 - 1
            if array.shape[0] % 16 or array.shape[1] % 16:
                raise ValueError(f'{path}: height and width must be multiples of 16; crop consistently with training')
            sar = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(args.device)
            result, _ = net.super_resolution(sar, seed=args.seed + index)
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
            folder = destination / relative.parent / relative.name
            folder.mkdir(parents=True, exist_ok=True)
            tensors = {'prediction': result, **net.last_maps}
            for name in ('prediction', 'anchor', 'uncertainty', 'noise_weight'):
                value = tensors[name][0].detach().cpu().float()
                if name in ('prediction', 'anchor'):
                    value = (value.clamp(-1, 1) + 1) / 2
                image = (value.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                if image.shape[-1] == 1:
                    image = image[..., 0]
                Image.fromarray(image).save(folder / f'{name}.png')
            np.savez_compressed(folder / 'predictive_maps.npz', **{
                name: value[0].cpu().numpy() for name, value in net.last_maps.items()})
            print(folder)


if __name__ == '__main__':
    main()
