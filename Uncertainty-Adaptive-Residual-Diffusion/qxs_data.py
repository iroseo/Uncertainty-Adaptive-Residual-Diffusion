"""Strict QXS pairing without copying or changing the source dataset."""
from pathlib import Path
from PIL import Image

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def resolve_modalities(root, sar_dir=None, optical_dir=None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError('Missing QXS split: ' + str(root))

    def resolve(explicit, aliases, role):
        if explicit:
            result = root / explicit
            if not result.is_dir():
                raise FileNotFoundError(str(result))
            return result
        candidates = [p for p in root.iterdir() if p.is_dir() and p.name.lower() in aliases]
        if len(candidates) != 1:
            raise ValueError('Cannot uniquely identify {} under {}. Found {}. '
                             'Specify --sar-dir / --optical-dir explicitly.'.format(
                                 role, root, [p.name for p in candidates]))
        return candidates[0]

    sar = resolve(sar_dir, {'sar', 'sen1', 's1'}, 'SAR')
    optical = resolve(optical_dir, {'opt', 'optical', 'eo', 'rgb', 'sen2', 's2'}, 'optical')
    if sar.resolve() == optical.resolve():
        raise ValueError('SAR and optical directories must be different')
    return sar, optical


def collect_pairs(sar_root, optical_root):
    def index(root):
        root = Path(root)
        found = {}
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                key = path.relative_to(root).with_suffix('').as_posix()
                if key in found:
                    raise ValueError('Duplicate sample stem: ' + str(path))
                found[key] = path
        if not found:
            raise ValueError('No images in ' + str(root))
        return found

    sar, optical = index(sar_root), index(optical_root)
    missing_optical, missing_sar = set(sar) - set(optical), set(optical) - set(sar)
    if missing_optical or missing_sar:
        raise ValueError('Unpaired QXS images: missing optical={}, missing SAR={}'.format(
            sorted(missing_optical)[:5], sorted(missing_sar)[:5]))
    return [(sar[key], optical[key]) for key in sorted(sar)]


def inspect_dataset(root, sar_dir=None, optical_dir=None, image_size=256):
    report = {}
    resolved_splits = set()
    for split in ('train', 'val', 'test'):
        split_root = Path(root) / split
        canonical = split_root.resolve()
        if canonical in resolved_splits:
            raise ValueError('Two splits resolve to the same directory: ' + str(canonical))
        resolved_splits.add(canonical)
        sar, optical = resolve_modalities(split_root, sar_dir, optical_dir)
        pairs = collect_pairs(sar, optical)
        for index, (s_path, y_path) in enumerate(pairs, 1):
            with Image.open(s_path) as s, Image.open(y_path) as y:
                if s.size != y.size or s.size != (image_size, image_size):
                    raise ValueError('Expected aligned {}x{} pair: {} {} vs {} {}'.format(
                        image_size, image_size, s_path, s.size, y_path, y.size))
                if s.mode.startswith('I') or s.mode == 'F' or y.mode.startswith('I') or y.mode == 'F':
                    raise ValueError('Raw high-bit-depth image requires the same preprocessing '
                                     'as E3Diff before training: ' + str(s_path))
            if index % 2000 == 0:
                print('Checking {}: {}/{}'.format(split, index, len(pairs)), flush=True)
        report[split] = dict(count=len(pairs), sar_dir=sar.name, optical_dir=optical.name,
                             root=str(split_root.resolve()), image_size=image_size)
        print('Paired {}: {} pairs, {} -> {}'.format(split, len(pairs), sar.name, optical.name), flush=True)
    return report
