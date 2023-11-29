import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os,math
from torch.nn import init



def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def ddim_denoising_step(xt, t, t_next, *,
                   model,
                   # logvar,
                   betas,
                   ):
    et = model(xt, t)
    # logvar = extract(logvar, t, xt.shape)

    # Compute the next x
    bt = extract(betas, t, xt.shape)
    at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)

    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)

    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    return xt_next, et



def diffusionmat_module(xt, img, mask, xt_next_truth, t, t_next, *,
                   model, model_img, model_delta,
                   betas,
                   ):
    
    # ddim step
    with torch.no_grad():
        et = model(xt, t)
        bt = extract(betas, t, xt.shape)
        at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
        if t_next.sum() == -t_next.shape[0]:
            at_next = torch.ones_like(at)
        else:
            at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    
    
    image_fea = model_img(img)
    inputs = torch.cat([image_fea,xt_next],dim=1)#.detach()
    
    
    xt_next = model_delta(inputs)
    xt_next = xt_next * mask + xt_next_truth * (1-mask) #fuse the xt_next with gt according to mask

    if t_next.sum()!=0:
        et_next = model(xt_next, t_next)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod = extract(alphas_cumprod, t_next, xt_next.shape)
        x0 = xt_next / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et_next
    else:
        x0 = xt_next
    return xt_next, x0


def diffusionmat_module_test(xt, image_fea, mask, xt_next_truth, t, t_next, *,
                   model, model_delta,
                   betas,
                   ):
    
    # ddim step
    with torch.no_grad():
        et = model(xt, t)
        bt = extract(betas, t, xt.shape)
        at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
        if t_next.sum() == -t_next.shape[0]:
            at_next = torch.ones_like(at)
        else:
            at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    
    
    #
    inputs = torch.cat([image_fea,xt_next],dim=1)#.detach()
    
    
    xt_next = model_delta(inputs)
    xt_next = xt_next * mask + xt_next_truth * (1-mask) #fuse the xt_next with gt according to mask

    
    return xt_next#, x0




# def denoising_in_one_step_and_ddim_denoising_step_modifying_et(xt, img, t, t_next, *,
#                    model, model_delta,
#                    # logvar,
#                    betas,
#                    ):
    
    
#     et = model(xt, t)
#     inputs = torch.cat([img,et],dim=1).detach()
#     # et = model_delta(inputs, temb.detach())
#     et = model_delta(inputs, t)

#     # print ('bbbbbbbb',et.shape)

#     # in one step
#     alphas = 1.0 - betas
#     alphas_cumprod = alphas.cumprod(dim=0)
#     alphas_cumprod = extract(alphas_cumprod, t, xt.shape)
#     onestep = xt / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et
    

#     # ddim step
#     # logvar = extract(logvar, t, xt.shape)
#     bt = extract(betas, t, xt.shape)
#     at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#     if t_next.sum() == -t_next.shape[0]:
#         at_next = torch.ones_like(at)
#     else:
#         at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#     x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#     #print ('ccccc',x0_t.shape)
#     xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    


#     return onestep, xt_next, et



# def ddim_denoising_step_modifying_xt(xt, img, t, t_next, *,
#                    model, model_delta,
#                    # logvar,
#                    betas,
#                    ):
    
    
#     et = model(xt, t)
#     # inputs = torch.cat([img,et],dim=1).detach()
#     # et = model_delta(inputs, t)


#     # in one step
#     '''
#     alphas = 1.0 - betas
#     alphas_cumprod = alphas.cumprod(dim=0)
#     alphas_cumprod = extract(alphas_cumprod, t, xt.shape)
#     onestep = xt / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et
#     '''

#     # ddim step
#     with torch.no_grad():

#         # logvar = extract(logvar, t, xt.shape)
#         bt = extract(betas, t, xt.shape)
#         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#         if t_next.sum() == -t_next.shape[0]:
#             at_next = torch.ones_like(at)
#         else:
#             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
#         inputs = torch.cat([img,xt_next],dim=1).detach()
        
#     xt_next = model_delta(inputs, t)
    
#     return xt_next

#     # return onestep, xt_next, et



# def ddim_denoising_step_modifying_xt_x0(xt, img, t, t_next, *,
#                    model, model_delta,
#                    betas,
#                    ):
    
#     # ddim step
#     with torch.no_grad():
#         et = model(xt, t)
#         bt = extract(betas, t, xt.shape)
#         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#         if t_next.sum() == -t_next.shape[0]:
#             at_next = torch.ones_like(at)
#         else:
#             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
#         inputs = torch.cat([img,xt_next],dim=1).detach()
        
        
#     xt_next = model_delta(inputs, t)

#     if t_next.sum()!=0:
#         et_next = model(xt_next, t_next)
#         alphas = 1.0 - betas
#         alphas_cumprod = alphas.cumprod(dim=0)
#         alphas_cumprod = extract(alphas_cumprod, t_next, xt_next.shape)
#         x0 = xt_next / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et_next
#     else:
#         x0 = xt_next
#     return xt_next, x0

#     # in one step
#     '''
#     alphas = 1.0 - betas
#     alphas_cumprod = alphas.cumprod(dim=0)
#     alphas_cumprod = extract(alphas_cumprod, t, xt.shape)
#     onestep = xt / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et
#     '''



# def ddim_denoising_step_modifying_xt_x0_mask(xt, img, mask, xt_next_truth, t, t_next, *,
#                    model, model_delta,
#                    betas,
#                    ):
    
#     # ddim step
#     with torch.no_grad():
#         et = model(xt, t)
#         bt = extract(betas, t, xt.shape)
#         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#         if t_next.sum() == -t_next.shape[0]:
#             at_next = torch.ones_like(at)
#         else:
#             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
#         inputs = torch.cat([img,xt_next],dim=1).detach()
        
        
#     xt_next = model_delta(inputs, t)
#     # xt_next = model_delta(inputs)


#     xt_next = xt_next * mask + xt_next_truth * (1-mask) #fuse the xt_next with gt according to mask


#     if t_next.sum()!=0:
#         et_next = model(xt_next, t_next)
#         alphas = 1.0 - betas
#         alphas_cumprod = alphas.cumprod(dim=0)
#         alphas_cumprod = extract(alphas_cumprod, t_next, xt_next.shape)
#         x0 = xt_next / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et_next
#     else:
#         x0 = xt_next
#     return xt_next, x0





# def ddim_denoising_step_modifying_xt_x0_mask_imgE(xt, img, mask, xt_next_truth, t, t_next, *,
#                    model, model_img, model_delta,
#                    betas,
#                    ):
    
#     # ddim step
#     with torch.no_grad():
#         et = model(xt, t)
#         bt = extract(betas, t, xt.shape)
#         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#         if t_next.sum() == -t_next.shape[0]:
#             at_next = torch.ones_like(at)
#         else:
#             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    
    
#     image_fea = model_img(img)
#     inputs = torch.cat([image_fea,xt_next],dim=1)#.detach()
    
    
#     xt_next = model_delta(inputs, t)
#     xt_next = xt_next * mask + xt_next_truth * (1-mask) #fuse the xt_next with gt according to mask


#     if t_next.sum()!=0:
#         et_next = model(xt_next, t_next)
#         alphas = 1.0 - betas
#         alphas_cumprod = alphas.cumprod(dim=0)
#         alphas_cumprod = extract(alphas_cumprod, t_next, xt_next.shape)
#         x0 = xt_next / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et_next
#     else:
#         x0 = xt_next
#     return xt_next, x0





# def ddim_denoising_step_modifying_xt_x0_mask_imgE_wo_t_test(xt, image_fea, mask, xt_next_truth, t, t_next, *,
#                    model, model_delta,
#                    betas,
#                    ):
    
#     # ddim step
#     with torch.no_grad():
#         et = model(xt, t)
#         bt = extract(betas, t, xt.shape)
#         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#         if t_next.sum() == -t_next.shape[0]:
#             at_next = torch.ones_like(at)
#         else:
#             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    
    
#     #
#     inputs = torch.cat([image_fea,xt_next],dim=1)#.detach()
    
    
#     xt_next = model_delta(inputs)
#     xt_next = xt_next * mask + xt_next_truth * (1-mask) #fuse the xt_next with gt according to mask

    
#     return xt_next#, x0


# def ddim_denoising_step_modifying_xt_x0_nomask_imgE_wo_t_test(xt, image_fea, t, t_next, *,
#                    model, model_delta,
#                    betas,
#                    ):
    
#     # ddim step
#     with torch.no_grad():
#         et = model(xt, t)
#         bt = extract(betas, t, xt.shape)
#         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#         if t_next.sum() == -t_next.shape[0]:
#             at_next = torch.ones_like(at)
#         else:
#             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    
    
#     #
#     inputs = torch.cat([image_fea,xt_next],dim=1)#.detach()
    
#     xt_next = model_delta(inputs)
#     #xt_next = xt_next * mask + xt_next_truth * (1-mask) #fuse the xt_next with gt according to mask

    
#     return xt_next#, x0



# def ddim_denoising_step_modifying_xt_x0_nomask_imgE_wo_t(xt, img, mask, xt_next_truth, t, t_next, *,
#                    model, model_img, model_delta,
#                    betas,
#                    ):
    
#     # ddim step
#     with torch.no_grad():
#         et = model(xt, t)
#         bt = extract(betas, t, xt.shape)
#         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#         if t_next.sum() == -t_next.shape[0]:
#             at_next = torch.ones_like(at)
#         else:
#             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    
    
#     image_fea = model_img(img)
#     inputs = torch.cat([image_fea,xt_next],dim=1)#.detach()
    
    
#     xt_next = model_delta(inputs)
#     #xt_next = xt_next * mask + xt_next_truth * (1-mask) #fuse the xt_next with gt according to mask

#     print (t_next.sum())
#     if t_next.sum()!=0:
#         et_next = model(xt_next, t_next)
#         alphas = 1.0 - betas
#         alphas_cumprod = alphas.cumprod(dim=0)
#         alphas_cumprod = extract(alphas_cumprod, t_next, xt_next.shape)
#         x0 = xt_next / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et_next
#     else:
#         x0 = xt_next
#     return xt_next, x0



# def ddim_denoising_step_modifying_xt_x0_mask_imgE_wo_t_mask_guided(xt, img, t, t_next, *,
#                    model, model_img, model_delta,
#                    betas,
#                    ):
    
#     # ddim step
#     with torch.no_grad():
#         et = model(xt, t)
#         bt = extract(betas, t, xt.shape)
#         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#         if t_next.sum() == -t_next.shape[0]:
#             at_next = torch.ones_like(at)
#         else:
#             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    
    
#     image_fea = model_img(img)
#     inputs = torch.cat([image_fea,xt_next],dim=1)#.detach()
    
    
#     xt_next = model_delta(inputs)
#     #xt_next = xt_next * mask + xt_next_truth * (1-mask) #fuse the xt_next with gt according to mask


#     if t_next.sum()!=0:
#         et_next = model(xt_next, t_next)
#         alphas = 1.0 - betas
#         alphas_cumprod = alphas.cumprod(dim=0)
#         alphas_cumprod = extract(alphas_cumprod, t_next, xt_next.shape)
#         x0 = xt_next / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et_next
#     else:
#         x0 = xt_next
#     return xt_next, x0




# # def ddim_denoising_step_modifying_xt_x0_mask_imgE(xt, img, mask, xt_next_truth, t, t_next, *,
# #                    model, model_img, model_delta,
# #                    betas,
# #                    ):
    
# #     # ddim step
# #     with torch.no_grad():
# #         et = model(xt, t)
# #         bt = extract(betas, t, xt.shape)
# #         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
# #         if t_next.sum() == -t_next.shape[0]:
# #             at_next = torch.ones_like(at)
# #         else:
# #             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
# #         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
# #         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    
    
# #     image_fea = model_img(img)
# #     inputs = torch.cat([image_fea,xt_next],dim=1)#.detach()
    
    
# #     xt_next = model_delta(inputs, t)
# #     xt_next = xt_next * mask + xt_next_truth * (1-mask) #fuse the xt_next with gt according to mask


# #     if t_next.sum()!=0:
# #         et_next = model(xt_next, t_next)
# #         alphas = 1.0 - betas
# #         alphas_cumprod = alphas.cumprod(dim=0)
# #         alphas_cumprod = extract(alphas_cumprod, t_next, xt_next.shape)
# #         x0 = xt_next / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et_next
# #     else:
# #         x0 = xt_next
# #     return xt_next, x0




# def ddim_denoising_step_modifying_xt_x0_imgE(xt, img, t, t_next, *,
#                    model, model_img, model_delta,
#                    betas,
#                    ):
    
#     # ddim step
#     with torch.no_grad():
#         et = model(xt, t)
#         bt = extract(betas, t, xt.shape)
#         at = extract((1.0 - betas).cumprod(dim=0), t, xt.shape)
#         if t_next.sum() == -t_next.shape[0]:
#             at_next = torch.ones_like(at)
#         else:
#             at_next = extract((1.0 - betas).cumprod(dim=0), t_next, xt.shape)
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
#         xt_next = xt_next.detach()
    
    
#     image_fea = model_img(img)
#     inputs = torch.cat([image_fea,xt_next],dim=1)
#     xt_next = model_delta(inputs, t)


#     if t_next.sum()!=0:
#         et_next = model(xt_next, t_next)
#         alphas = 1.0 - betas
#         alphas_cumprod = alphas.cumprod(dim=0)
#         alphas_cumprod = extract(alphas_cumprod, t_next, xt_next.shape)
#         x0 = xt_next / torch.sqrt(alphas_cumprod) - (torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)) * et_next
#     else:
#         x0 = xt_next
#     return xt_next, x0
