import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Subset
import torch
import phash
from model import Dense


class MyData(Dataset):
    """
    Dataset class to handle image data and generate associated secret hashes.

    This class is designed for loading images from a specified directory, transforming them
    into tensors, and generating secret hashes based on perceptual hashing (pHash) for each
    image. It supports custom secret sizes and resizing dimensions. The dataset is compatible
    with PyTorch's Dataset class, enabling seamless integration into data pipelines for
    deep learning tasks.

    :ivar data_path: Path to the directory containing image files.
    :type data_path: str
    :ivar secret_size: Size of the secret hash in bits. Defaults to 64.
    :type secret_size: int
    :ivar size: Tuple representing the target dimensions for resizing images (width, height).
    :type size: tuple
    :ivar files_list: List of file paths for all images in the specified data directory.
    :type files_list: list
    :ivar to_tensor: Transformation that converts images to PyTorch tensors.
    :type to_tensor: transforms.ToTensor
    """

    def __init__(self, data_path, secret_size=64, size=(400, 400)):
        """
        Represents the initialization of the object that processes image data from a specified
        directory with optional settings for secret size and image size. It also sets up a tensor
        transformation utility and compiles a list of image files in the directory.

        :param data_path: The file path to the directory containing image data.
        :type data_path: str
        :param secret_size: The size of the secret data segment (default is 64).
        :type secret_size: int, optional
        :param size: The width and height tuple specifying the expected dimensions of the images
            (default is (400, 400)).
        :type size: tuple[int, int], optional
        """
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        self.files_list = glob(os.path.join(self.data_path, '*.jpg'))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        """
        Fetch the cover image and its corresponding secret value represented by a perceptual hash.

        The method retrieves an image based on the provided index, processes it to fit the
        specified size, converts it into a tensor format, and calculates a perceptual hash
        value for the image. The hash is then transformed into a binary representation which
        is returned alongside the processed image.

        :param idx: Index of the file in the files list
        :type idx: int
        :return: A tuple containing the processed cover image as a tensor and the secret value
                 as a float tensor
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        img_cover_path = self.files_list[idx]

        img_cover = Image.open(img_cover_path).convert('RGB')
        img_cover = ImageOps.fit(img_cover, self.size)
        img_cover = self.to_tensor(img_cover)
        # img_cover = np.array(img_cover, dtype=np.float32) / 255.

        secret_hash = phash.calculate_phash(img_cover_path, hash_size=int(self.secret_size ** 0.5))
        secret = np.array([int(bit) for bit in bin(int(secret_hash, 16))[2:].zfill(self.secret_size)])
        secret = torch.from_numpy(secret).float()
        return img_cover, secret

    def __len__(self):
        return len(self.files_list)


def train_test_dataset(dataset, test_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    datasets = {'train': Subset(dataset, train_idx), 'test': Subset(dataset, val_idx)}
    print(f"Train dataset size: {len(datasets['train'])}, Test dataset size: {len(datasets['test'])}")
    return datasets

if __name__ == '__main__':
    dataset = MyData(data_path=r'D:\CaseStudy\CSDS490\final\mirflickr', secret_size=64, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    image_input, secret_input = next(iter(dataloader))
    print(type(image_input), type(secret_input))
    print(image_input.shape, secret_input.shape)
    # # Pass the secret_input tensor through the Dense layer
    # dense_layer = Dense(64, 256 * 256, activation='relu', kernel_initializer='he_normal')
    # secret_input = dense_layer(secret_input)
    # # Apply the view operation to the output tensor
    # secret_input = secret_input.view(-1, 1, 256, 256)
    # # Perform upsampling
    # secret_input_upsampled = torch.nn.Upsample(size=(400, 400), mode='bilinear', align_corners=False)(secret_input)
    # # Concatenate tensors
    # inputs = torch.cat([image_input, secret_input_upsampled], dim=1)
    # print(inputs.shape)
    print(image_input.max())
