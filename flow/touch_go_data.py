from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch
import torchvision
import albumentations

class TouchAndGo_Clip_Latent_Resize(Dataset):
    def __init__(self, root_dir, isTrain=True):
        """Initialize Touch and Go dataset class.
        """
        if isTrain:
            with open(os.path.join(root_dir, 'train_change.txt'),'r') as f:
                data = f.read().split('\n')
        else:
            with open(os.path.join(root_dir, 'test_change.txt'),'r') as f:
                data = f.read().split('\n')
        
        self.length = len(data)
        self.data = data
        self.root = os.path.join(root_dir, 'dataset')
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
        
        A_img_path=A_img_path.replace('.jpg', '.npy')
        A_gelsight_path=A_gelsight_path.replace('.jpg', '.npy')
        A_gelsight_letent_path=A_gelsight_path.replace('.npy', 'resize_clip_latent.npy')
        A_img_letent_path=A_img_path.replace('.npy', 'resize_clip_latent.npy')
        img = np.load(A_img_path)
        img=torch.from_numpy(img).float()
        gelsight = np.load(A_gelsight_path)
        gelsight=torch.from_numpy(gelsight).float()
        
        touch_clip_latent=np.load(A_gelsight_letent_path)
        touch_clip_latent=torch.from_numpy(touch_clip_latent).float()
        img_clip_latent=np.load(A_img_letent_path)
        img_clip_latent=torch.from_numpy(img_clip_latent).float()
        # transform
        # if self.transform is not None:
        if img is None:
            print(A_img_path)
        if gelsight is None:
            print(A_gelsight_path)

        return {'image': img, 
                'tact': gelsight, 
                "image_clip_encode":img_clip_latent, 
                "touch_clip_encode":touch_clip_latent, 
                "target":int(target), 
                "path": self.data[index_A]
                }
    def __len__(self):
        """Return the total number of images."""
        return self.length
