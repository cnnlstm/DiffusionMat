import os,time
import numpy as np
from tqdm import tqdm

import torch,logging
import torchvision.utils as tvu
from PIL import Image
from torchvision import transforms

from model import UNet
from model_wo_t import UNet_wo_T

from networks_swtu.vision_transformer import SwinUnet

from runners.diffusion_utils import *
from runners.matting_utils import *

from runners.dataset import *


from torch.utils.data import Dataset, DataLoader
from dataloader.data_generator import *





import torch.optim.lr_scheduler as lr_scheduler

from tensorboardX import SummaryWriter
import random
import itertools
import cv2 as cv
from tqdm import trange


def update_lr(lr, optimizer):
    """
    update learning rates
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr(init_lr, step, iter_num):
    """
    Warm up learning rate
    """
    return step/iter_num*init_lr


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


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
        writer = SummaryWriter(self.args.log_folder)


        matting_model = UNet(in_channel=1,
        T=1000, ch=64, ch_mult=[1, 1, 2, 2], attn=[1],
        num_res_blocks=1, dropout=0.1).to(self.device)


        ckpt = torch.load("./pretrained_models/ckpt_diffusion.pt")['net_model']


        matting_model.load_state_dict(ckpt)
        matting_model.to(self.device)
        matting_model.eval()
        matting_model = torch.nn.DataParallel(matting_model)

        
        image_model = SwinUnet().to(self.device)
        
        
        delta_model = UNet_wo_T(in_channel=33, T=1000, ch=64, ch_mult=[1, 1, 2, 2], attn=[1],
        num_res_blocks=1, dropout=0.1).to(self.device)

        pretrained_path = './pretrained_models/swin_tiny_patch4_window7_224.pth'
        pretrained_dict = torch.load(pretrained_path, map_location='cuda')['model']



        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        
        for k in list(full_dict.keys()):
            new_name = 'swin_unet.'+k
            full_dict[new_name] =  full_dict[k]
            del full_dict[k]


        i = 0
        model_dict = delta_model.state_dict()
        for k in list(full_dict.keys()):
            if k in model_dict:
                i = i+1
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]

                    
        image_model.load_state_dict(full_dict, strict=False)

        image_model = torch.nn.DataParallel(image_model)
        image_model.train()
        
        delta_model = torch.nn.DataParallel(delta_model)
        delta_model.train()

        optimizer = torch.optim.Adam(itertools.chain(image_model.parameters(), delta_model.parameters()),
                          lr=self.delta_config.optim.lr, weight_decay=self.delta_config.optim.weight_decay,
                          betas=(self.delta_config.optim.beta1, 0.999), amsgrad=self.delta_config.optim.amsgrad,
                          eps=self.delta_config.optim.eps)



        step = 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.ckpt_folder, "ckpt.pth"))
            image_model.load_state_dict(states[0])
            delta_model.load_state_dict(states[1])
            optimizer.load_state_dict(states[2])
            step = states[3]
        
        
        train_set = DataGenerator(fg_path="./fg_path", 
                                  bg_path="./bg_path/", 
                                  alpha_path="./alpha_path/", phase='train')
            
        train_loader = DataLoader(dataset=train_set, num_workers=16, batch_size=self.delta_config.training.batch_size, shuffle=True, drop_last=True)
        
        
        datalooper = infiniteloop(train_loader)
        with torch.no_grad():
            with trange(self.args.total_step, dynamic_ncols=True) as pbar:
                for step_ in pbar:

                    step = step + 1
                    time0 = time.time()
                    data = next(datalooper)
                    time1 = time.time()
                    
                    
                        
                    image = data['image']
                    alpha = data['alpha']
                    trimap = data['trimap']
                    fg = data['fg']
                    bg = data['bg']
                    
                    

                    image = image.to(self.device)
                    alpha = alpha.to(self.device)
                    trimap = trimap.to(self.device)
                    fg = fg.to(self.device)
                    bg = bg.to(self.device)
                    
                    
                    
                    unknown = torch.ones_like(trimap)
                    unknown[trimap==0.0]=0
                    unknown[trimap==255.0]=0
                    
                    
                    trimap = trimap/255.0 
                    x0 = trimap
                    
                    e = torch.randn_like(trimap)
                    
                    total_noise_levels = self.args.t
                    sample_step = self.args.sample_step + 1

                    seq_inv_ = np.linspace(0, 1, sample_step) * total_noise_levels #coarse steps
                    seq_inv = []

                    for s in list(seq_inv_):
                        seq_inv.append(int(s))   
                    seq_inv_next = [-1] + list(seq_inv[:-1])
                    

                    
                    a = (1 - self.betas).cumprod(dim=0)
                    x_start = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                    
                    x = x_start
                    x_inv = torch.cat([alpha,trimap],dim=0)


                    x_invs = {}
                    x_invs[0] = x_inv


                    for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                        t = (torch.ones(self.delta_config.training.batch_size * 2) * i).to(self.device).long()
                        t_prev = (torch.ones(self.delta_config.training.batch_size * 2) * j).to(self.device).long()
                        x_inv,_ = ddim_denoising_step(x_inv, t=t, t_next=t_prev, model=matting_model,
                                betas=self.betas)
                        x_invs[j] = x_inv


                    xs = []
                    for it, (i, j) in enumerate(zip(reversed((seq_inv[1:])), reversed((seq_inv_next[1:])))):
                        t = (torch.ones(self.delta_config.training.batch_size) * i).to(self.device).long()
                        t_next = (torch.ones(self.delta_config.training.batch_size) * j).to(self.device).long()

                        with torch.enable_grad():
                            optimizer.zero_grad()

                            x_inv = x_invs[j][:self.delta_config.training.batch_size]
                            trimap_inv = x_invs[j][self.delta_config.training.batch_size:] 
                            

                            
                            x, x_0 = diffusionmat_module(x, img=image, mask=unknown, xt_next_truth=trimap_inv, t=t, t_next=t_next, model=matting_model, model_img = image_model, model_delta=delta_model, betas=self.betas)


                            image_comp = fg*x_0 + bg*(1-x_0)
                            
                            
                            
                            loss_comp = F.l1_loss(image_comp*unknown, image*unknown)
                            loss_alpha = F.l1_loss(x_0*unknown, alpha*unknown)
                            loss_inv = F.mse_loss(x*unknown, x_inv*unknown)

                            loss =  self.args.w_com * loss_comp + self.args.w_alpha * loss_alpha + self.args.w_inv * loss_inv
                            
                            

                            loss.backward()
                            optimizer.step() 
                            
                            print(
                                f"step: {step}, sample_step: {sample_step}, iter: {it}, lr: {self.delta_config.optim.lr}, i: {i}, j: {j},  loss_comp: {loss_comp.item()}, loss_alpha: {loss_alpha.item()}, loss_inv: {loss_inv.item()}, loss: {loss.item()}, w_com: {self.args.w_com}, w_alpha: {self.args.w_alpha}, w_inv: {self.args.w_inv}."
                            )

                            x = x.detach()
                            
                            if step%50==0:
                                xs.append(x)
                    
                        writer.add_scalar('loss', loss, step)
                        writer.add_scalar('loss_alpha', loss_alpha, step)
                        writer.add_scalar('loss_inv', loss_inv, step)
                        writer.add_scalar('loss_comp', loss_comp, step)
                        

                    if step%50==0:
                        x_modif = x
                        x = x_start

                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(self.delta_config.training.batch_size) * i).to(self.device).long()
                            t_next = (torch.ones(self.delta_config.training.batch_size) * j).to(self.device).long()
                            x,_ = ddim_denoising_step(x, t=t, t_next=t_next, model=matting_model,
                                        betas=self.betas)

                        x_ddim = x
                        unknown_ = torch.stack([unknown,unknown,unknown],dim=1).squeeze(2)
                        trimap_ = torch.stack([trimap,trimap,trimap],dim=1).squeeze(2)
                        x_ddim_ = torch.stack([x_ddim,x_ddim,x_ddim],dim=1).squeeze(2)
                        x_modif_ = torch.stack([x_modif,x_modif,x_modif],dim=1).squeeze(2)
                        x_0_ = torch.stack([x_0,x_0,x_0],dim=1).squeeze(2)

                        alpha_ = torch.stack([alpha,alpha,alpha],dim=1).squeeze(2)
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).cuda()
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).cuda()
                        
                        image = image.multiply_(std).add_(mean)
                        image_comp = image_comp.multiply_(std).add_(mean)
                        
                        
                        sample = torch.cat([image,image_comp])
                        
                        sample = torch.cat([sample,trimap_])
                        
                        sample = torch.cat([sample,unknown_])
                        sample = torch.cat([sample,x_modif_])
                        sample = torch.cat([sample,x_0_])
                        sample = torch.cat([sample,x_ddim_])
                        sample = torch.cat([sample,alpha_])
                        

                        tvu.save_image(sample, os.path.join(self.args.image_folder,  "sample_{}.jpg".format(step)), nrow=self.delta_config.training.batch_size)
                        
                        x = xs[0]
                        
                        for x_ in xs[1:]:
                            x = torch.cat([x,x_])
                            print (x.shape)
                        tvu.save_image(x, os.path.join(self.args.image_folder,  "x_{}.jpg".format(step)), nrow=self.delta_config.training.batch_size)
                            
                            


                    if step % 5000==0:

                        states = [
                            image_model.state_dict(),
                            delta_model.state_dict(),
                            optimizer.state_dict(),
                            step,
                        ]

                        torch.save(
                            states,
                            os.path.join(self.args.ckpt_folder, "ckpt_{}.pth".format(step)),
                        )

                    if step % 100==0:

                        states = [
                            image_model.state_dict(),
                            delta_model.state_dict(),
                            optimizer.state_dict(),
                            step,
                        ]    
                        torch.save(states, os.path.join(self.args.ckpt_folder, "ckpt.pth"))
