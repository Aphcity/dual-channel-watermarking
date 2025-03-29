import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import phash


class MyData(Dataset):
    def __init__(self, data_path, secret_size=64, size=(400, 400)):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        self.files_list = glob(os.path.join(self.data_path, '*.png')) 
        self.to_tensor = transforms.ToTensor()
    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]

        img_cover = Image.open(img_cover_path).convert('RGB')
        img_cover = ImageOps.fit(img_cover, self.size)
        img_cover = self.to_tensor(img_cover)
        # img_cover = np.array(img_cover, dtype=np.float32) / 255.

        secret_hash = phash.calculate_phash(img_cover_path, hash_size=int(self.secret_size**0.5))
        secret = np.array([int(bit) for bit in bin(int(secret_hash, 16))[2:].zfill(self.secret_size)])
        secret = torch.from_numpy(secret).float()
        return img_cover, secret

    def __len__(self):
        return len(self.files_list)


if __name__ == '__main__':
    

    dataset = MyData(data_path=r'E:\dataset\mirflickr', secret_size=64, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    image_input, secret_input = next(iter(dataloader))
    print(type(image_input), type(secret_input))
    print(image_input.shape, secret_input.shape)
    print(image_input.max())
