import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import setup_logging,DiagonalGaussianDistribution,SSIM
from scipy import integrate
from touch_go_data import TouchAndGo_Clip_Latent_Resize
from bak_datasets import Test_TouchAndGo
import logging
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
from unet_clip_class import UNetModel
from taming.modules.losses.lpips import LPIPS
import random
import math
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cosine_lr_scheduler(optimizer, epoch, total_epochs, initial_lr):
    t = epoch % (total_epochs // 2)
    lr = 0.5 * initial_lr * (1 + math.cos(math.pi * t / (total_epochs // 2)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def data_centered(x,recover=False):
    if recover:
        return (x + 1.) / 2.
    else:
        return x * 2. - 1.

class RectifiedFlow(nn.Module):
    def __init__(self, model,n_timesteps=1000, img_size=256, device="cuda"):
        super(RectifiedFlow, self).__init__()
        self.n_timesteps = n_timesteps
        self.img_size = img_size
        self.device = device
        self.model = model.to(device)
        self.criterion=nn.MSELoss() 
        self.lpips=LPIPS().eval().to(device)
    @property
    def T(self):
      return 1.

    @torch.no_grad()
    def euler_ode(self, tactle,label, N,latent):
      ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
      eps=1e-3
      dt = 1./N

      # Initial sample
      x = tactle.detach().clone()
      shape = tactle.shape
      device = tactle.device

      for i in range(N):  
        num_t = i / N * (self.T - eps) + eps      
        t = torch.ones(shape[0], device=device) * num_t
        pred = self.model(x, t*999,y=label,latent=latent)
        x = x.detach().clone() + pred * dt         
      return x
    
    @torch.no_grad()
    def euler_ode_reverse(self, img, label, N,latent):
      ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
      eps=1e-6
      dt = 1./N
      # Initial sample
      x = img.detach().clone()
      shape = img.shape
      device = img.device
      for i in reversed(range(N)):  
        num_t = i / N * (self.T - eps) + eps      
        t = torch.ones(shape[0], device=device) * num_t
        pred = self.model(x, t*999,y=label,latent=latent)
        x = x.detach().clone() - pred * dt           
      return x
    @torch.no_grad()
    def implicit_euler_reverse(self, img, label, N,latent):
        eps=1e-6
        dt = 1./N
        # Initial sample
        x = img.detach().clone()
        shape = img.shape
        device = img.device
        for i in reversed(range(N)):  
            num_t = i / N * (self.T - eps) + eps      
            t = torch.ones(shape[0], device=device) * num_t
            pred = self.model(x, t*999,y=label,latent=latent)
            y = x.detach().clone() - pred * dt
            eps_y=10
            while eps_y>1e-6:
                pred_y=y
                pred = self.model(y, t*999,y=label,latent=latent)
                y = x.detach().clone() - pred * dt
                eps_y=torch.mean(torch.abs(y-pred_y))
            x=y
        return x
    
    @torch.no_grad()
    def euler_ode_reverse_rk(self, img, label, N,latent):
      ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
      eps=1e-6
      dt = 1./N
      # Initial sample
      x = img.detach().clone()
      shape = img.shape
      device = img.device
      h=1/N * (self.T - eps) + eps  
      for i in reversed(range(N)):  
        num_t = i / N * (self.T - eps) + eps      
        t = torch.ones(shape[0], device=device) * num_t
        f1 = self.model(x, t*999, y=label,latent=latent)
        f2 = self.model(x + 0.5*h*f1,(t+0.5*h)*999,y=label,latent=latent)
        f3 = self.model(x + 0.5*h*f2,(t+0.5*h)*999,y=label,latent=latent)
        f4 = self.model(x + h*f3,(t+h)*999,y=label,latent=latent)
        pred=(1/6.0)*(f1 + 2*f2 + 2*f3 + f4)
        x=x.detach().clone() - pred * dt      
      return x
  

    @torch.no_grad()
    def sample(self, tactle,label,latent):
        img = self.euler_ode(tactle,label,N=1000,latent=latent)
        return img
    @torch.no_grad()
    def sample_reverse(self,img,label,latent):
        tactle=self.euler_ode_reverse(img,label,N=1000,latent=latent)
        return tactle
    
    def loss(self,tactle,img,label,latent,vision=True):
        N, _,_,_ = tactle.shape
        t = torch.rand((N,), device=tactle.device)
        xt = t.view(N,1,1,1) * img + (1 - t.view(N,1,1,1)) * tactle
        vector = img - tactle

        pre = self.model(xt, t*999,y=label,latent=latent)

        loss = self.criterion(pre, vector)
        return loss
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@torch.no_grad()
def val(args):
    setup_logging(args.run_name)
    device = args.device
    root_dir="/raid/datasets/touch2vision/"
    datasets = TouchAndGo_Clip_Latent_Resize(root_dir,isTrain=False)
    dataloader = DataLoader(datasets, batch_size=args.batch_size*4, shuffle=True, num_workers=16, pin_memory=True,
                            drop_last=False)
    
    model = UNetModel(
            image_size=args.image_size,
            in_channels=4,
            model_channels=224,
            out_channels=4,
            num_res_blocks=2,
            attention_resolutions=(8,4,2),
            channel_mult=(1,2,3,4),
            num_head_channels=32,
            dropout=0.1,
            num_classes=20)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"best_ckpt.pt")))

    flow = RectifiedFlow(model=model,img_size=args.image_size, device=device)
    flow.eval()
    step=0
    pbar = tqdm(dataloader)
    for i, data in enumerate(pbar):
        img = data['image'].to(device)
        tactile=data["tact"].to(device)
        label=data["target"].to(device)
        path=data["path"]
        target=data["target"]
        image_clip_encode=data["image_clip_encode"].to(device)
        touch_clip_encode=data["touch_clip_encode"].to(device)
        img=DiagonalGaussianDistribution(img).sample()
        tactile=DiagonalGaussianDistribution(tactile).sample()
        tactile_to_img = flow.sample(tactile,label,touch_clip_encode)
        img_to_tactile = flow.sample(img,label,image_clip_encode)

        batch_size=img_to_tactile.shape[0]
        for i in range(batch_size):
            img_=tactile_to_img[i].detach().cpu().numpy()
            tactile_=img_to_tactile[i].detach().cpu().numpy()
            img_latent_path=os.path.join("results", "UnetFlow_noper_val","gen", f"val_{step}_tactile_to_img.npy")
            gelsight_latent_path=os.path.join("results", "UnetFlow_noper_val","gen", f"val_{step}_img_to_tactile.npy")
            np.save(img_latent_path, img_)
            np.save(gelsight_latent_path, tactile_)
            org_img_path,org_gelsight_path=copy_file(path[i])
            img_path=os.path.join("results", "UnetFlow_noper_val","real", f"val_{step}_img.jpg")
            gel_path=os.path.join("results", "UnetFlow_noper_val","real", f"val_{step}_tactile.jpg")
            image = Image.open(org_img_path)
            gel_img=Image.open(org_gelsight_path)
            new_image = image.resize((256, 256))
            new_image.save(img_path) 
            new_gel_img = gel_img.resize((256, 256))
            new_gel_img.save(gel_path) 
            step+=1
            with open(os.path.join("results", "UnetFlow_noper_val",'target.txt'), 'a') as f:
                f.write(str(target[i].item()) + '\n')


def copy_file(path):
    root="/raid/datasets/touch2vision/dataset/"
    A_raw_path, target = path.strip().split(',') # mother path for A
    A_dir, A_idx = os.path.join(root, A_raw_path[:15]) , A_raw_path[16:]
    org_img_path = os.path.join(A_dir, 'video_frame', A_idx)
    org_gelsight_path = os.path.join(A_dir, 'gelsight_frame', A_idx)
    return org_img_path,org_gelsight_path

@torch.no_grad()
def eval(flow,val_dataloader,args,epoch,logger,global_val_step):
    flow.eval()
    device=args.device
    pbar=tqdm(val_dataloader)
    l=len(val_dataloader)
    avg_loss=AverageMeter()
    avg_ssim_loss=AverageMeter()
    for i, data in enumerate(pbar):
        img = data['image'].to(device)
        tactile=data["tact"].to(device)
        label=data["target"].to(device)
        image_clip_encode=data["image_clip_encode"].to(device)
        touch_clip_encode=data["touch_clip_encode"].to(device)
        path=data["path"]
        img=DiagonalGaussianDistribution(img).mean
        tactile=DiagonalGaussianDistribution(tactile).mean
 
        loss_1=flow.loss(tactle=tactile,img=img,label=label,latent=touch_clip_encode)
        loss_2=flow.loss(tactle=img,img=tactile,label=label,latent=image_clip_encode)
        loss=(loss_1+loss_2)*0.5
        if torch.isnan(loss):
            print(data["class_name"])
            continue

        avg_loss.update(loss.item(),img.size(0))
        pbar.set_postfix({'TEST MSE': '{:.5f}'.format(loss.item()),
                        'MSE_LOSS': '{:.5f}'.format(avg_loss.avg),
                        })
        logger.add_scalar("VAL MSE", loss.item(), global_step=global_val_step)
        global_val_step+=1
    if epoch % args.eval_frq==0:
        tactile_to_img = flow.sample(tactile,label,touch_clip_encode)
        img_to_tactile = flow.sample(img,label,image_clip_encode)
        for i in [0,1]:
            img_=tactile_to_img[i].detach().cpu().numpy()
            tactile_=img_to_tactile[i].detach().cpu().numpy()
            img_latent_path=os.path.join("results", args.run_name, f"val_{epoch}_{i}_tactile_to_img.npy")
            gelsight_latent_path=os.path.join("results", args.run_name, f"val_{epoch}_{i}_img_to_tactile.npy")
            np.save(img_latent_path, img_)
            np.save(gelsight_latent_path, tactile_)
            org_img_path,org_gelsight_path=copy_file(path[i])
            img_path=os.path.join("results", args.run_name, f"val_{epoch}_{i}_img.jpg")
            gel_path=os.path.join("results", args.run_name, f"val_{epoch}_{i}_tactile.jpg")
            image = Image.open(org_img_path)
            gel_img=Image.open(org_gelsight_path)
            new_image = image.resize((256, 256))
            new_image.save(img_path) 
            new_gel_img = gel_img.resize((256, 256))
            new_gel_img.save(gel_path) 
    flow.train()
    return global_val_step,avg_loss.avg-avg_ssim_loss.avg
def train(args):
    setup_logging(args.run_name)
    device = args.device
    root_dir="/raid/datasets/touch2vision/"

    datasets = TouchAndGo_Clip_Latent_Resize(root_dir)
    dataloader = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True,
                            drop_last=True)
    val_datasets = TouchAndGo_Clip_Latent_Resize(root_dir,isTrain=False)
    val_dataloader = DataLoader(val_datasets, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True,
                            drop_last=False)

    model = UNetModel(
            image_size=args.image_size,
            in_channels=4,
            model_channels=64,
            out_channels=4,
            num_res_blocks=2,
            attention_resolutions=(8,4,2),
            channel_mult=(1,2,2,4),
            num_head_channels=32,
            dropout=0.1,
            num_classes=20)
    global_step=0
    global_val_step=0
    best_loss=100
    flow = RectifiedFlow(model=model,img_size=args.image_size, device=device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    global_val_step,_=eval(flow,val_dataloader,args,1,logger,global_val_step)
    for epoch in range(args.epochs):
        cosine_lr_scheduler(optimizer,epoch,args.epochs,args.lr)
        logging.info(f"Starting epoch {epoch}:")
        avg_loss=AverageMeter()
        mse_loss=AverageMeter()
        ssim_avg_loss=AverageMeter()
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            img = data['image'].to(device)
            tactile=data["tact"].to(device)
            label=data["target"].to(device)
            image_clip_encode=data["image_clip_encode"].to(device)
            touch_clip_encode=data["touch_clip_encode"].to(device)
            img=DiagonalGaussianDistribution(img).sample()
            tactile=DiagonalGaussianDistribution(tactile).sample()
            loss_1=flow.loss(tactle=tactile,img=img,label=label,latent=touch_clip_encode)
            loss_2=flow.loss(tactle=img,img=tactile,label=label,latent=image_clip_encode)
            loss=(loss_1+loss_2)*0.5
            if torch.isnan(loss):
                print(data["class_name"])
                continue
            optimizer.zero_grad()
            if args.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            loss.backward()
            optimizer.step()
            avg_loss.update(loss.item(),img.size(0))
            pbar.set_postfix({'MSE': '{:.5f}'.format(loss.item()),
                             'AVG_MSE': '{:.5f}'.format(avg_loss.avg)
                             })
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            global_step+=1

        if (epoch+1) % 5==0:
            global_val_step,val_loss=eval(flow,val_dataloader,args,epoch,logger,global_val_step)
            if val_loss<best_loss:
                best_loss=val_loss
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"best_ckpt.pt"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"opt_ckpt.pt"))
        
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    set_seed_everywhere(2)
    args.run_name = "bivtflow"
    args.epochs = 800
    args.batch_size = 128
    args.image_size = 256
    args.beta1= 0.9
    args.eps = 1e-8
    args.weight_decay=0.001
    args.warmup = 0.
    args.grad_clip = 1.
    args.device = "cuda:0"
    args.lr = 1e-3 # 1e-4 for batch size 64 2e-4 for w=0.1 dropout=0.1
    args.eval_frq=50
    train(args)

if __name__ == '__main__':
    launch()

