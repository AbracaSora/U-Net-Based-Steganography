import os

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, cover_path, secret_path, transform=None):
        self.cover_path = cover_path
        self.secret_path = secret_path
        self.transform = transform
        self.cover = os.listdir(cover_path)
        self.secret = os.listdir(secret_path)

    def __len__(self):
        return len(self.cover)

    def __getitem__(self, idx):
        cover_image = os.path.join(self.cover_path, self.cover[idx])
        cover_image = Image.open(cover_image).convert('RGB')
        secret_image = os.path.join(self.secret_path, self.secret[idx])
        secret_image = Image.open(secret_image).convert('RGB')

        cover = self.transform(cover_image) if self.transform else cover_image
        secret = self.transform(secret_image) if self.transform else secret_image

        return cover, secret
