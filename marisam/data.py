"""Read explicitly paired GeoTIFF patches without hard-coded paths or random splitting."""
import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def read_image(path, band_order=(1, 2, 3, 4)):
    import rasterio
    with rasterio.open(path) as source:
        image = source.read(list(band_order)).astype(np.float32)
        profile = source.profile.copy()
    if not np.isfinite(image).all():
        raise ValueError('Image contains non-finite values; preprocess nodata before using it.')
    return image, profile


def normalize(image, mean, std):
    if len(mean) != 4 or len(std) != 4 or any(value <= 0 for value in std):
        raise ValueError('Provide four training-derived means and positive standard deviations.')
    return (torch.as_tensor(image, dtype=torch.float32) - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


class PatchDataset(Dataset):
    def __init__(self, manifest, root, mean, std, band_order=(1, 2, 3, 4),
                 num_classes=2, flip_probability=0.0, rotation_probability=0.0):
        self.root = Path(root)
        with open(manifest, encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'image', 'label'} <= set(reader.fieldnames or []):
                raise ValueError('Manifest needs image and label columns.')
            self.rows = list(reader)
        if not self.rows or any(not row['image'] or not row['label'] for row in self.rows):
            raise ValueError('Every sample needs an explicit image and label path.')
        self.mean, self.std, self.band_order = mean, std, band_order
        self.num_classes = num_classes
        self.flip_probability, self.rotation_probability = flip_probability, rotation_probability

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import rasterio
        row = self.rows[index]
        image, profile = read_image(self.root / row['image'], self.band_order)
        with rasterio.open(self.root / row['label']) as source:
            label = source.read(1)
            if source.crs != profile['crs'] or source.transform != profile['transform']:
                raise ValueError('Image and label must use the same CRS and pixel grid.')
        if image.shape[-2:] != label.shape or label.shape[0] != label.shape[1]:
            raise ValueError('Use co-registered square image/label patches.')
        if not np.isin(label, np.arange(self.num_classes + 1)).all():
            raise ValueError('Labels must be integer class IDs from 0 to num_classes.')
        image = normalize(image, self.mean, self.std)
        label = torch.from_numpy(label.astype(np.int64))
        for dim in (-1, -2):
            if random.random() < self.flip_probability:
                image, label = image.flip(dim), label.flip(dim)
        if random.random() < self.rotation_probability:
            turns = random.randint(1, 3)
            image, label = torch.rot90(image, turns, (-2, -1)), torch.rot90(label, turns, (-2, -1))
        return image, label
