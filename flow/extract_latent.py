import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModelInterface
from taming.models.autoae import AutoencoderKL_Interface
from touch_go_data import TouchGoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import torchvision
DEVICE="cuda:7"

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  model = VQModelInterface(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()
def load_AutoKL(config, ckpt_path=None, is_gumbel=False):
  model = AutoencoderKL_Interface(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

@torch.no_grad()
def get_latent(img_encoder,tactile_encoder):
    root_dir="/raid/datasets/touch2vision/"
    save_root=os.path.join(root_dir, 'dataset')

    datasets = TouchGoDataset(root_dir,isTrain=False)
    dataloader = DataLoader(datasets, batch_size=64, shuffle=False, num_workers=16, pin_memory=True,
                            drop_last=False)
    pbar = tqdm(dataloader)

    for i, data in enumerate(pbar):
        img = data['image'].to(DEVICE)
        tactile=data["tact"].to(DEVICE)
        path=data["path"]

        img_latent=img_encoder.encode(img)
        tactile_latent=tactile_encoder.encode(tactile)
        batch_size=tactile_latent.shape[0]
        for i in range(batch_size):
          img_=img_latent[i].detach().cpu().numpy()
          tactile_=tactile_latent[i].detach().cpu().numpy()
          A_raw_path, target = path[i].strip().split(',') # mother path for A
          A_dir, A_idx = os.path.join(save_root, A_raw_path[:15]) , A_raw_path[16:]
          img_path = os.path.join(A_dir, 'video_frame', A_idx)
          gelsight_path = os.path.join(A_dir, 'gelsight_frame', A_idx)
          img_latent_path=img_path.replace('.jpg', '.npy')
          gelsight_latent_path=gelsight_path.replace('.jpg', '.npy')
          np.save(img_latent_path, img_)
          np.save(gelsight_latent_path, tactile_)
          # with open('test.txt', 'a') as f:
          #    f.write(path[i] + '\n')   
             
img_config = load_config("img_encoderkl/configs/kl_32.yaml", display=False)
img_encoder = load_AutoKL(img_config, ckpt_path="img_encoderkl/kl_32_checkpoints/epoch=000191.ckpt").to(DEVICE)

tactile_config = load_config("gel_encoderkl/configs/kl_32.yaml", display=False)
tactile_encoder = load_AutoKL(tactile_config, ckpt_path="gel_encoderkl/kl_32_checkpoints/epoch=000159.ckpt").to(DEVICE)

get_latent(img_encoder,tactile_encoder)
