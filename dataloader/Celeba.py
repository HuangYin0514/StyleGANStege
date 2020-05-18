
from functools import partial
from torch.utils import data
import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path


EXTS = ['jpg', 'png']


# dataset
class Dataset(data.Dataset):
    def __init__(self, folder, transform=None):
        super().__init__()
        self.folder = folder
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
