"""Paired normalized RGB image I/O for lightweight pretraining and diagnostics."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def read_rgb(path):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'), dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array).permute(2, 0, 1)


class PairedImages(Dataset):
    def __init__(self, sar_root, optical_root, limit=-1, augment=False, pairs=None):
        self.sar_root, self.optical_root = Path(sar_root), Path(optical_root)
        self.augment = augment
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        if pairs is None:
            sar_paths = sorted(p for p in self.sar_root.rglob('*') if p.suffix.lower() in extensions)
            self.pairs = [(p, self.optical_root/p.relative_to(self.sar_root)) for p in sar_paths]
        else:
            self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError(f'No images under {self.sar_root}')
        missing = [str(y) for _, y in self.pairs if not y.is_file()]
        if missing:
            raise ValueError(f'Missing optical pair: {missing[0]} ({len(missing)} missing)')
        if limit > 0:
            self.pairs = self.pairs[:limit]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        s_path, y_path = self.pairs[index]
        sar, optical = read_rgb(s_path), read_rgb(y_path)
        if sar.shape != optical.shape:
            raise ValueError(f'Pair dimensions differ: {s_path}')
        if self.augment:
            for axis in (-1, -2):
                if torch.rand(()) < 0.5:
                    sar, optical = sar.flip(axis), optical.flip(axis)
        return {'SR': sar, 'HR': optical, 'LR': sar, 'filename': str(s_path.relative_to(self.sar_root))}


class QXSPairedImages(PairedImages):
    def __init__(self, options, augment=False):
        from qxs_data import resolve_modalities, collect_pairs
        source, target = resolve_modalities(options['dataroot'], options.get('sar_dir'),
                                            options.get('optical_dir'))
        super().__init__(source, target, options.get('data_len', -1), augment,
                         pairs=collect_pairs(source, target))
        self.image_size = options.get('r_resolution', 256)

    def __getitem__(self, index):
        from urllib.parse import quote
        item = super().__getitem__(index)
        if item['HR'].shape[-2:] != (self.image_size, self.image_size):
            raise ValueError('Paired image size differs from configuration: ' + item['filename'])
        if self.augment:
            turns = int(torch.randint(0, 4, ()).item())
            for key in ('HR', 'LR', 'SR'):
                item[key] = item[key].rot90(turns, dims=(-2, -1))
        # A flat, collision-free output filename also supports nested datasets.
        item['filename'] = quote(self.pairs[index][0].relative_to(self.sar_root).as_posix(), safe='')
        return item


def dataset_from_options(options, augment=False):
    root = Path(options['dataroot'])
    if options['name'] in ('QXS', 'SAR2Opt'):
        return QXSPairedImages(options, augment=augment)
    if options['name'] == 'SEN12':
        source, target = root/'sen1', root/'sen2'
    elif options['name'] == 'SAR_EO':
        source, target = root/'SAR', root/'EO'
    else:
        raise ValueError('Lightweight loader supports SEN12 and SAR_EO normalized RGB image pairs')
    return PairedImages(source, target, options.get('data_len', -1), augment=augment)
