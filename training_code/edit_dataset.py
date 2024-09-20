from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from glob import glob 

from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from input_transform import *
from functools import partial
from resizer import Resizer
import os 
import glm
import rgbd_3d
import cv2

class EditDataset_IN(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        # Load fractal data
        self.mixing_set = glob('../../data/Pixmix_dataset/fractals/images/*')
        self.mixing_set2 = glob('../../data/Pixmix_dataset/first_layers_resized256_onevis/images/*')

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256)
        ]) 
        if split == 'train':
            with open(Path(self.path, "save_train_names.json")) as f:
                self.load_path = json.load(f)
        elif split == 'val':
            with open(Path(self.path, "save_val_names.json")) as f:
                self.load_path = json.load(f)

    def __len__(self) -> int:
        return len(self.load_path)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt = 'Clean the image'
        item = os.path.join(self.path, self.load_path[idx])
        image_clean = self.transform(Image.open(item).convert(mode='RGB'))

        choice = np.random.randint(5)
        # perform nothing
        if choice == 0:
            img = image_clean
        elif choice == 1 or choice == 2 or choice == 3:
            if np.random.random() < 0.5:
                mixing = self.mixing_set
            else:
                mixing = self.mixing_set2
            # 1) PIXMIX
            rnd_idx  = np.random.choice(len(mixing))
            image_mix = self.transform(Image.open(mixing[rnd_idx]))
            if np.random.random() < 0.5:
                mixing = self.mixing_set
            else:
                mixing = self.mixing_set2
            rnd_idx2 = np.random.choice(len(mixing))
            image_mix2 = self.transform(Image.open(mixing[rnd_idx2]))
            img = pixmix(image_clean, image_mix, image_mix2, {'tensorize': transforms.ToTensor()}) 
        else: 
            img = Simsiam_transform(image_clean)
        
        if not choice == 0:
            img = to_pil_image(img)
        image_mix   = rearrange(2 * torch.tensor(np.array(img)).float() / 255 - 1, "h w c -> c h w")
        image_clean = rearrange(2 * torch.tensor(np.array(image_clean)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        image_0, image_1 = flip(crop(torch.cat((image_mix, image_clean)))).chunk(2)
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))

class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            edit = prompt["edit"]
            input_prompt = prompt["input"]
            output_prompt = prompt["output"]

        image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))

        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        return dict(image_0=image_0, input_prompt=input_prompt, edit=edit, output_prompt=output_prompt)
