import os
import numpy as np
from tqdm import tqdm

import torch,logging
import torchvision.utils as tvu
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from models.diffusion import Model
from runners.diffusion_utils import *
from runners.matting_utils import *

from runners.deltablock import *
from runners.deltablock_unet import *

from model import UNet
from model_wo_t import UNet_wo_T

from runners.dataset import *
from networks_swtu.vision_transformer import SwinUnet

import cv2
from tqdm import trange


def generator_tensor_dict(image_path, trimap_path, alpha_path):
    # read images

    image = cv2.imread(image_path)
    alpha = cv2.imread(alpha_path, 0)
    trimap = cv2.imread(trimap_path, 0)

    mask = (trimap > 128)
    
    sample = {'image': image, 'trimap':trimap,'mask':mask, 'alpha':alpha, 'image_shape':(image.shape[0], image.shape[1])}
    
    # reshape
    h, w = sample["image_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        pad_h = 0
        pad_w = 0

        padded_image = np.pad(sample['image'], ((32,0), (32, 0), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,0), (32, 0)), mode="reflect")
        padded_alpha = np.pad(sample['alpha'], ((32,0), (32, 0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32,0), (32, 0)), mode="reflect")
        
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
        sample['alpha'] = padded_alpha
        sample['mask'] = padded_mask
        


    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(sample['image'], ((pad_h+32, 0), (pad_w+32, 0), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((pad_h+32, 0), (pad_w+32, 0)), mode="reflect")
        padded_alpha = np.pad(sample['alpha'], ((pad_h+32, 0), (pad_w+32, 0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((pad_h+32, 0), (pad_w+32, 0)), mode="reflect")
        

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
        sample['alpha'] = padded_alpha
        sample['mask'] = padded_mask
    

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

                    
                    
    # convert GBR images to RGB
    image, trimap, alpha, mask = sample['image'][:,:,::-1], sample['trimap'], sample['alpha'], sample['mask']

    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)
    trimap = trimap.astype(np.float32)
    alpha = alpha.astype(np.float32)
    mask = mask.astype(np.float32)
    

    # normalize image
    image /= 255.
    trimap /= 255.
    alpha /= 255.


    # to tensor
    sample['image'], sample['trimap'], sample['alpha'], sample['mask'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.float), torch.from_numpy(alpha).to(torch.float), torch.from_numpy(mask).to(torch.float)
    sample['image'] = sample['image'].sub_(mean).div_(std)
    sample['image'], sample['trimap'], sample['alpha'], sample['mask']  = sample['image'][None, ...], sample['trimap'][None, None,...], sample['alpha'][None, None,...], sample['mask'][None, None,...]
    
    return sample




class Diffusion(object):
    def __init__(self, args, config, delta_config, device=None):
        self.args = args
        self.config = config
        self.delta_config = delta_config

        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]


        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))
        


    def image_matting_train(self):

        matting_model = Model(self.config)

        matting_model = UNet(in_channel=1,
        T=1000, ch=64, ch_mult=[1, 1, 2, 2], attn=[1],
        num_res_blocks=1, dropout=0.1).to(self.device)

        
        ckpt = torch.load("./pretrained_models/ckpt_diffusion.pt")['net_model']
        matting_model.load_state_dict(ckpt)
        matting_model = torch.nn.DataParallel(matting_model)
        matting_model.to(self.device)
        matting_model.eval()
        ckpt = torch.load("./pretrained_models/ckpt_adobe.pth")
        
        
        
        
        image_model = SwinUnet().to(self.device)
        image_model = torch.nn.DataParallel(image_model)
        image_model.eval()
        image_model.load_state_dict(ckpt[0])
        image_model.eval()

      
        delta_model = UNet_wo_T(in_channel=33, T=1000, ch=64, ch_mult=[1, 1, 2, 2], attn=[1],
        num_res_blocks=1, dropout=0.1).to(self.device)
        delta_model = torch.nn.DataParallel(delta_model)
        delta_model.load_state_dict(ckpt[1])
        delta_model.eval()



        image_dir = "./samples/merged/"
        trimap_dir = "./samples/trimaps/"
        alpha_dir = "./samples/alpha_copy/"



        with torch.no_grad():

            for iter, image_name in enumerate(os.listdir(image_dir)):
                        

                        image_path = os.path.join(image_dir, image_name)

                        # matte_name = image_name.split('_'+image_name.split("_")[-1])[0]+'.png'
                        # print (image_path)
                        trimap_path = os.path.join(trimap_dir, image_name)
                        alpha_path = os.path.join(alpha_dir, image_name)

                        sample = generator_tensor_dict(image_path, trimap_path, alpha_path)

                        image = sample['image']
                        trimap = sample['trimap']
                        alpha = sample['alpha']

                        image_shape = sample['image_shape']


                        image = image.to(self.device).float()
                        alpha = alpha.to(self.device).float()
                        trimap = trimap.to(self.device).float()

                        unknown = torch.ones_like(trimap)
                        unknown[trimap==0]=0
                        unknown[trimap==1]=0



                        trimap_ = torch.stack([trimap,trimap,trimap],dim=1).squeeze(2)

                        image_fea = image_model(image)

                        x0 = trimap

                        e = torch.randn_like(trimap)

                        total_noise_levels = self.args.t
                        sample_step = self.args.sample_step + 1

                        seq_inv_ = np.linspace(0, 1, sample_step) * total_noise_levels 
                        seq_inv = []

                        for s in list(seq_inv_):
                            seq_inv.append(int(s))   
                        seq_inv_next = [-1] + list(seq_inv[:-1])


                        a = (1 - self.betas).cumprod(dim=0)
                        x_start = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                        x = x_start
                        x_inv = trimap

                        x_invs = {}
                        x_invs[0] = x_inv


                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(1) * i).to(self.device).long()
                            t_prev = (torch.ones(1) * j).to(self.device).long()
                            x_inv,_ = ddim_denoising_step(x_inv, t=t, t_next=t_prev, model=matting_model,
                                    betas=self.betas)
                            x_invs[j] = x_inv



                        for it, (i, j) in enumerate(zip(reversed((seq_inv[1:])), reversed((seq_inv_next[1:])))):
                            t = (torch.ones(x_start.shape[0]) * i).to(self.device).long()
                            t_next = (torch.ones(x_start.shape[0]) * j).to(self.device).long()
                            trimap_inv = x_invs[j]
                            x = diffusionmat_module_test(x, image_fea = image_fea, mask=unknown, xt_next_truth=trimap_inv, t=t, t_next=t_next, model=matting_model, model_delta=delta_model, betas=self.betas)


                            

                        x[trimap==0]=0
                        x[trimap==1]=1
                        x = x[:,:,-image_shape[0]:,-image_shape[1]:]
                        tvu.save_image(x, os.path.join(self.args.exp, image_name),normalize=False)


                    
   
