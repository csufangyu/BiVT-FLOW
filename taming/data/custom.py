import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTactile(CustomBase):
    def __init__(self, size, images_list_file):
        super().__init__()
        with open(images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torchvision
    from tqdm import tqdm
    file="/raid/datasets/touch2vision/train/img.txt"
    datasets=CustomTrain(256,file)
    dataloader = DataLoader(datasets, batch_size=16, shuffle=False, num_workers=16, pin_memory=True,
                                drop_last=False)
   
    pbar = tqdm(dataloader)
    for i, sample in enumerate(pbar):           
        img=sample["image"]
        # print(torch.max(tact),torch.min(tact))
        # print(torch.max(img),torch.min(img))
        print(img.size())
        assert 1==0
