'''The dataset. Given a directory, it collects all images within the directory
into a torch Tensor for faster access. This obviously assumes that the dataset
fits into memory. To save space, images are resized to the needed resolution
and stored using 8-bit unsigned integers for their RGB values.'''
import glob
from pathlib import Path
from itertools import chain
from logging import getLogger
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm  # type: ignore
import torchvision.io as io  # type: ignore
from torchvision import transforms
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .utils import AttrDict


logger = getLogger('noisy.dataset')


@dataclass
class ImgDataset(Dataset[Tensor]):
    cfg: AttrDict
    cache: Tensor = field(init=False)
    dynamic_transform: transforms.Compose = field(init=False)
    static_transform: transforms.Compose = field(init=False)

    def __post_init__(self) -> None:
        # Transforms applied *after* the image is loaded from the cache
        self.dynamic_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            self.normalize,
        ])
        # Transforms applied *before* the image is saved to the cache
        static_transform_steps = [
            transforms.Resize(self.cfg.img.size),
            transforms.CenterCrop(self.cfg.img.size),
        ]
        if self.cfg.img.channels == 1:
            static_transform_steps.append(transforms.transforms.Grayscale())
        self.static_transform = transforms.Compose(static_transform_steps)
        self.cache = self._get_cache()

    def _get_cache(self) -> Tensor:
        c: int = self.cfg.img.channels
        s: int = self.cfg.img.size
        path = Path(self.cfg.data.path)
        cache_file = path / f'.{c}x{s}x{s}.cache'
        if cache_file.exists():
            logger.info(f'Loading cache file: {cache_file}')
            cache = torch.load(cache_file)  # type: ignore
            assert isinstance(cache, Tensor)
            assert cache.size(1) == c
            assert cache.size(2) == cache.size(3) == s
            return cache
        logger.info(f'Populating cache file: {cache_file}')
        exts = self.cfg.data.extensions
        patterns = (path / f'**/*{ext}' for ext in exts)
        file_its = (glob.iglob(str(patt), recursive=True)
                    for patt in patterns)
        files = list(chain(*file_its))
        mb = round((len(files) * c * s * s) / 1e6, 2)
        logger.info(f'Estimated cache size: {mb} MB')
        imgs_it = (self._load_img(f) for f in tqdm(files))
        imgs = [img.unsqueeze(0) for img in imgs_it if img is not None]
        cache = torch.cat(imgs)
        assert isinstance(cache, Tensor)
        torch.save(cache, cache_file)
        logger.info('Cache saved.')
        return cache

    def _load_img(self, path: str) -> Optional[Tensor]:
        try:
            img = io.read_image(path)
        except RuntimeError as e:
            logger.warning(f'Got RuntimeError whilst loading {path}: {e}')
            return None
        img = img[:3, :, :]
        img = self.static_transform(img)
        if img.size(0) != self.cfg.img.channels:
            logger.warning(f'Expected {self.cfg.img.channels} channels but got'
                           f' {img.size(0)} whilst loading {path}')
            return None
        return img  # type: ignore

    def __getitem__(self, index: int) -> Tensor:
        return self.dynamic_transform(self.cache[index])  # type: ignore

    def __len__(self) -> int:
        return len(self.cache)

    @staticmethod
    def normalize(img: Tensor) -> Tensor:
        return img.float() / 127.5 - 1.
