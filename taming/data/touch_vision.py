from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch

class TouchDataset(Dataset):
    def __init__(self, root_dir, isTrain=True,transform=None):
        """Initialize Touch and Go dataset class.

        """
        # save the option and dataset root
        
        if isTrain:
            with open(os.path.join(root_dir, 'train.txt'),'r') as f:
                data = f.read().split('\n')
        else:
            with open(os.path.join(root_dir, 'test.txt'),'r') as f:
                data = f.read().split('\n')
        
        self.length = len(data)
        self.data = data
        self.root = os.path.join(root_dir, 'dataset')
        self.transform=transform
    def __getitem__(self, index):
        """Return a data point containing (Image A, Gelsight A, Image B and Gelsight B).
        """
        index_A = index
        # Random generate index_B different from index_A
        assert index_A < self.length,'index_A out of range'
        A_raw_path, target = self.data[index_A].strip().split(',') # mother path for A
        A_dir, A_idx = os.path.join(self.root, A_raw_path[:15]) , A_raw_path[16:]
        # Read A images and gelsight
        A_gelsight_path = os.path.join(A_dir, 'gelsight_frame', A_idx)

        A_gel = Image.open(A_gelsight_path).convert('RGB')

        if A_gel is None:
            print(A_gelsight_path)
        A_gel = self.transform(A_gel)

        return A_gel

    def __len__(self):
        """Return the total number of images."""
        return self.length


class TouchAndGoDataset(Dataset):
    
    def __init__(self, root_dir, isTrain=True,transform=None):
        """Initialize Touch and Go dataset class.

        """
        # save the option and dataset root
        
        if isTrain:
            with open(os.path.join(root_dir, 'train.txt'),'r') as f:
                data = f.read().split('\n')
        else:
            with open(os.path.join(root_dir, 'test.txt'),'r') as f:
                data = f.read().split('\n')
        
        self.length = len(data)
        self.data = data
        self.root = os.path.join(root_dir, 'dataset')
        self.transform=transform
    def __getitem__(self, index):
        """Return a data point containing (Image A, Gelsight A, Image B and Gelsight B).
        """
        index_A = index

        # Random generate index_B different from index_A
        assert index_A < self.length,'index_A out of range'
        A_raw_path, target = self.data[index_A].strip().split(',') # mother path for A
        A_dir, A_idx = os.path.join(self.root, A_raw_path[:15]) , A_raw_path[16:]
        # Read A images and gelsight
        A_img_path = os.path.join(A_dir, 'video_frame', A_idx)
        A_gelsight_path = os.path.join(A_dir, 'gelsight_frame', A_idx)

        A_img = Image.open(A_img_path).convert('RGB')
        A_gel = Image.open(A_gelsight_path).convert('RGB')

        # Read B images and gelsight

        
        # transform
        # if self.transform is not None:
        if A_img is None:
            print(A_img_path)
        if A_gel is None:
            print(A_gelsight_path)
        A_img = self.transform(A_img)
        A_gel = self.transform(A_gel)

        return {'image': A_img, 'tact': A_gel, "target":int(target)}

    def __len__(self):
        """Return the total number of images."""
        return self.length

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torchvision
    from tqdm import tqdm
    def data_centered(x,recover=False):
        if recover:
            return (x + 1.) / 2.
        else:
            return x * 2. - 1.

    trainTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize([256,256]),
            torchvision.transforms.ToTensor()
        ]
    )
    root_dir="/raid/datasets/touch2vision/"
    dataset = TouchAndGoDataset(root_dir,transform=trainTransform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True,
                                drop_last=False)
    pbar = tqdm(dataloader)
    for i, sample in enumerate(pbar):           
        img=sample["image"]
        tact=sample["tact"]
        target=sample["target"]
        img=data_centered(img)
        # print(torch.max(tact),torch.min(tact))
        # print(torch.max(img),torch.min(img))
        print(tact.size())
        print(img.size())
        print(target)
        assert 1==0