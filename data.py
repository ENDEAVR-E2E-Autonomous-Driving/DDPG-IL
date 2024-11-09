import os
import h5py
import numpy as np
import time
import torch

from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class DataManager:
    def __init__(self, folder, name):
        self.folder = folder
        self.name = name
        self.file_path = self.get_file_path()
        self.h5file = None
        self.initialize_folder()
        self.open_file()

    def initialize_folder(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def get_file_path(self):
        return os.path.join(self.folder, f"{self.name}_{int(time.time())}.h5")

    def open_file(self):
        self.h5file = h5py.File(self.file_path, 'a')
        self.initialize_datasets()

    def initialize_datasets(self):
        datasets = {
            'images': ((0, 88, 200, 3), np.float32),
            'scalars': ((0, 3), np.float32),
            'targets': ((0, 3), np.float32),
            'commands': ((0, 1), np.uint8),
        }
        for name, (shape, dtype) in datasets.items():
            if name not in self.h5file:
                self.h5file.create_dataset(name, shape, maxshape=(None, *shape[1:]), dtype=dtype, chunks=(1, *shape[1:]))

    def save(self, image, scalars, targets, commands):
        image = np.array(image, dtype=np.float32)
        scalars = np.array(scalars, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        commands = np.array(commands, dtype=np.uint8)

        try:
            for dataset_name, data in zip(['images', 'scalars', 'targets', 'commands'], [image, scalars, targets, commands]):
                dataset = self.h5file[dataset_name]
                dataset.resize(dataset.shape[0] + 1, axis=0)
                dataset[-1:] = data
        except Exception as e:
            print(f"An error occurred: {e}")


class ImitationDataset(Dataset):
    def __init__(self, folder, cache_size=10, include_image=True):
        self.file_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')])
        self.file_lengths = [self.get_file_length(f) for f in self.file_paths]
        self.cumulative_lengths = np.cumsum(self.file_lengths)
        self.file_cache = OrderedDict()
        self.cache_size = cache_size
        self.include_image = include_image

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((88, 200)),
            transforms.ColorJitter(0.2, 0.2),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.RandomErasing(0.5, scale=(0.005, 0.01), inplace=True),
            transforms.RandomErasing(0.5, scale=(0.005, 0.01), inplace=True),
            transforms.RandomErasing(0.5, scale=(0.005, 0.01), inplace=True),
            transforms.RandomErasing(0.5, scale=(0.005, 0.01), inplace=True),
        ])

    def get_file_length(self, file_path):
        with h5py.File(file_path, 'r') as file:
            return file['images'].shape[0]

    def __len__(self):
        return sum(self.file_lengths)

    def get_file_for_index(self, idx):
        file_index = np.searchsorted(self.cumulative_lengths, idx, side='right')
        file_path = self.file_paths[file_index]

        return file_path.split("/")[-1]

    def __getitem__(self, idx):
        file_index = np.searchsorted(self.cumulative_lengths, idx, side='right')
        local_index = idx - (self.cumulative_lengths[file_index - 1] if file_index > 0 else 0)
        file_path = self.file_paths[file_index]

        if file_path not in self.file_cache:
            if len(self.file_cache) >= self.cache_size:
                self.file_cache.popitem(last=False)
            self.file_cache[file_path] = h5py.File(file_path, 'r')

        file = self.file_cache[file_path]

        if self.include_image:
            image = file['images'][local_index]
            image = np.array(image, dtype=np.float32)
            image = self.transforms(image)
        else:
            image = None

        scalars = file['scalars'][local_index]
        scalars = np.array(scalars, dtype=np.float32)

        targets = file['targets'][local_index]
        targets = np.array(targets, dtype=np.float32)

        commands = file['commands'][local_index]
        commands = np.array(commands, dtype=np.uint8)

        return image, scalars, targets, commands

    def __del__(self):
        for file in self.file_cache.values():
            file.close()
        self.file_cache.clear()
