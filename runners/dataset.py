import copy
import json
import os
import warnings
from PIL import Image
from absl import app, flags
import random

import torch
import numpy as np
import cv2 as cv
from torchvision import transforms
import torch.utils.data as data

def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


# def cv2_to_pil_exp(open_cv_image):
#     open_cv_image = np.expand_dims(np.array(open_cv_image),axis=0)
#     return Image.fromarray(open_cv_image[:, :, ::-1].copy())

def cv2_to_pil(open_cv_image):
    return Image.fromarray(open_cv_image[:, :, ::-1].copy())

# def pil_to_cv2_exp(pil_image):
#     open_cv_image = np.expand_dims(np.array(pil_image),axis=0)
#     return open_cv_image[:, :, ::-1].copy()

# def pil_to_cv2(pil_image):
#     open_cv_image = np.array(pil_image)
#     return open_cv_image[:, :, ::-1].copy()




class img_matting_dataset(data.Dataset):
    def __init__(self, image_path, matte_path, transform1, transform2):
        self.image_path = image_path
        self.matte_path = matte_path

        self.transform1 = transform1
        self.transform2 = transform2

        self.images = self._walkFile(self.image_path)
        self.mattes = self._walkFile(self.matte_path)

        assert len(os.listdir(self.image_path))==len(os.listdir(self.matte_path))#==len(os.listdir(self.trimap_path))

    def _walkFile(self,file):
        samples = []
        for x in os.listdir(file):
            samples.append(file+"/"+x)
        return sorted(samples)

    def _generate_trimap(self,alpha):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
        trimap = fg * 255 + (unknown - fg) * 128
        return Image.fromarray(trimap.astype(np.uint8))


    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        matte = Image.open(self.mattes[idx])
        trimap = self._generate_trimap(np.array(matte))
        img = self.transform1(img)
        trimap = self.transform2(trimap)        
        matte = self.transform2(matte)
        return img,matte,trimap,self.images[idx]
    def __len__(self):
        return len(os.listdir(self.image_path))






class img_matting_dataset_fg_bg(data.Dataset):
    def __init__(self, image_path, fg_path, bg_path, matte_path, transform1, transform2):
        self.image_path = image_path
        self.fg_path = fg_path
        self.bg_path = bg_path

        self.matte_path = matte_path

        self.transform1 = transform1
        self.transform2 = transform2

        self.images = self._walkFile(self.image_path)
        # self.fgs = self._walkFile(self.fg_path)
        self.bgs = self._walkFile(self.bg_path)
        #self.mattes = self._walkFile(self.matte_path)

        #assert len(os.listdir(self.image_path))==len(os.listdir(self.matte_path))#==len(os.listdir(self.trimap_path))

    def _walkFile(self,file):
        samples = []
        for x in os.listdir(file):
            samples.append(file+"/"+x)
        return sorted(samples)

    def _generate_trimap(self,alpha):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
        trimap = fg * 255 + (unknown - fg) * 128

        # mask = np.array(np.equal(trimap, 0).astype(np.float32))
        mask = np.ones_like(trimap)*255
        mask[trimap==255]=0
        mask[trimap==0]=0


        return Image.fromarray(trimap.astype(np.uint8)),Image.fromarray(mask.astype(np.uint8))


    def _generate_trimap_2(self,alpha):
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        # unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
        trimap = fg * 255 + (unknown - fg) * 128

        # mask = np.array(np.equal(trimap, 0).astype(np.float32))
        mask = np.ones_like(trimap)*255
        mask[trimap==255]=0
        mask[trimap==0]=0


        return Image.fromarray(trimap.astype(np.uint8)),Image.fromarray(mask.astype(np.uint8))


    # def __getitem__(self, idx):
    #     img = Image.open(self.images[idx])
    #     matte = Image.open(self.mattes[idx])
    #     trimap = self._generate_trimap(np.array(matte))
    #     img = self.transform1(img)
    #     trimap = self.transform2(trimap)        
    #     matte = self.transform2(matte)
    #     return img,matte,trimap,self.images[idx]

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        bg = Image.open(self.bgs[idx])

        img_name = self.images[idx].split('/')[-1]
        name = img_name.split('_'+img_name.split("_")[-1])[0]+'.jpg'
        #print (name)
        matte = Image.open(self.matte_path+name)

        fg = Image.open(self.fg_path+name)


        trimap,unknown = self._generate_trimap(np.array(matte))


        img = self.transform1(img)
        fg = self.transform1(fg)
        bg = self.transform1(bg)

        trimap = self.transform2(trimap)      
        unknown = self.transform2(unknown)      

        matte = self.transform2(matte)

        return img,matte,trimap,unknown, fg, bg,self.images[idx]

    def __len__(self):
        return len(os.listdir(self.image_path))




class img_matting_dataset_fg_bg_random_crop(data.Dataset):
    def __init__(self, image_path, fg_path, bg_path, matte_path, transform1, transform2):
        self.image_path = image_path
        self.fg_path = fg_path
        self.bg_path = bg_path
        self.matte_path = matte_path
        self.transform1 = transform1
        self.transform2 = transform2
        self.images = self._walkFile(self.image_path)
        self.bgs = self._walkFile(self.bg_path)
        self.crop_size=(256,256)

    def _walkFile(self,file):
        samples = []
        for x in os.listdir(file):
            samples.append(file+"/"+x)
        return sorted(samples)

    def _generate_trimap(self,alpha):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
        trimap = fg * 255 + (unknown - fg) * 128
        mask = np.ones_like(trimap)*255
        mask[trimap==255]=0
        mask[trimap==0]=0
        return trimap.astype(np.uint8),mask.astype(np.uint8)


    def _random_choice(self, trimap, crop_size=(256,256)):
        # print ('xxxx',trimap.shape)
        crop_height, crop_width = crop_size
        y_indices, x_indices = np.where(trimap == 128)
        num_unknowns = len(y_indices)
        x, y = 0, 0
        if num_unknowns > 0:
            ix = np.random.choice(range(num_unknowns))
            center_x = x_indices[ix]
            center_y = y_indices[ix]
            x = max(0, center_x - int(crop_width / 2))
            y = max(0, center_y - int(crop_height / 2))
        return x, y

    def _safe_crop(self, mat, x, y, crop_size):
        im_size = crop_size[0]
        crop_height, crop_width = crop_size
        if len(mat.shape) == 2:
            ret = np.zeros((crop_height, crop_width), np.uint8)
        else:
            ret = np.zeros((crop_height, crop_width, 3), np.uint8)
        crop = mat[y:y + crop_height, x:x + crop_width]
        h, w = crop.shape[:2]
        ret[0:h, 0:w] = crop
        if crop_size != (im_size, im_size):
            ret = cv.resize(ret, dsize=(im_size, im_size), interpolation=cv.INTER_NEAREST)
        return ret


    def __getitem__(self, idx):

        img = cv.imread(self.images[idx])
        bg = cv.imread(self.bgs[idx])

        img_name = self.images[idx].split('/')[-1]
        name = img_name.split('_'+img_name.split("_")[-1])[0]+'.jpg'
        matte = cv.imread(self.matte_path+name)[:,:,0]
        fg = cv.imread(self.fg_path+name)
        trimap,_ = self._generate_trimap(matte)
        x, y = self._random_choice(trimap, self.crop_size)
        img = self._safe_crop(img, x, y, self.crop_size)
        matte = self._safe_crop(matte, x, y, self.crop_size)
        fg = self._safe_crop(fg, x, y, self.crop_size)
        bg = self._safe_crop(bg, x, y, self.crop_size)
        trimap,unknown = self._generate_trimap(np.array(matte))


        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            fg = np.fliplr(fg)
            bg = np.fliplr(bg)
            trimap = np.fliplr(trimap)
            unknown = np.fliplr(unknown)
            matte = np.fliplr(matte)

        # img = transforms.ToPILImage()(img)
        # fg = transforms.ToPILImage()(fg)
        # bg = transforms.ToPILImage()(bg)
        # trimap = transforms.ToPILImage()(trimap)
        # unknown = transforms.ToPILImage()(unknown)
        # matte = transforms.ToPILImage()(matte)
        # print (trimap.shape,unknown.shape,matte.shape)
        img = cv2_to_pil(img)
        fg = cv2_to_pil(fg)
        bg = cv2_to_pil(bg)
        trimap = transforms.ToPILImage()(trimap)
        unknown = transforms.ToPILImage()(unknown)
        matte = transforms.ToPILImage()(matte)
        

        img = self.transform1(img)
        fg = self.transform1(fg)
        bg = self.transform1(bg)
        trimap = self.transform2(trimap)      
        unknown = self.transform2(unknown)      
        matte = self.transform2(matte)

        return img,matte,trimap,unknown, fg, bg,self.images[idx]

    def __len__(self):
        return len(os.listdir(self.image_path))





class img_matting_dataset_full(data.Dataset):
    def __init__(self, image_path, trimap_path, matte_path, transform1, transform2):
        self.image_path = image_path
        self.trimap_path = trimap_path
        self.matte_path = matte_path
        self.transform1 = transform1
        self.transform2 = transform2
        self.images = self._walkFile(self.image_path)
        self.trimaps = self._walkFile(self.trimap_path)

        #self.crop_size=(256,256)

    def _walkFile(self,file):
        samples = []
        for x in os.listdir(file):
            samples.append(file+"/"+x)
        return sorted(samples)

    # def _generate_trimap(self,alpha):
    #     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    #     fg = np.array(np.equal(alpha, 255).astype(np.float32))
    #     unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    #     unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
    #     trimap = fg * 255 + (unknown - fg) * 128
    #     mask = np.ones_like(trimap)*255
    #     mask[trimap==255]=0
    #     mask[trimap==0]=0
    #     return trimap.astype(np.uint8),mask.astype(np.uint8)

    

    def __getitem__(self, idx):

        img = cv.imread(self.images[idx])
        trimap = cv.imread(self.trimaps[idx])[:,:,0]

        # bg = cv.imread(self.bgs[idx])

        img_name = self.images[idx].split('/')[-1]
        name = img_name.split('_'+img_name.split("_")[-1])[0]+'.png'
        
        matte = cv.imread(self.matte_path+name)[:,:,0]

        img = cv2_to_pil(img)

        trimap = transforms.ToPILImage()(trimap)
        matte = transforms.ToPILImage()(matte)
        

        img = self.transform1(img)
        trimap = self.transform2(trimap)      
        matte = self.transform2(matte)

        return img,matte,trimap,self.images[idx]

    def __len__(self):
        return len(os.listdir(self.image_path))






class img_matting_dataset_full2(data.Dataset):
    def __init__(self, image_path, matte_path, transform1, transform2):
        self.image_path = image_path
        self.matte_path = matte_path
        self.transform1 = transform1
        self.transform2 = transform2
        self.images = self._walkFile(self.image_path)
        #self.crop_size=(256,256)

    def _walkFile(self,file):
        samples = []
        for x in os.listdir(file):
            samples.append(file+"/"+x)
        return sorted(samples)

    def _generate_trimap(self,alpha):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
        trimap = fg * 255 + (unknown - fg) * 128
        mask = np.ones_like(trimap)*255
        mask[trimap==255]=0
        mask[trimap==0]=0
        return trimap.astype(np.uint8),mask.astype(np.uint8)



    def __getitem__(self, idx):

        img = cv.imread(self.images[idx])
        # bg = cv.imread(self.bgs[idx])

        img_name = self.images[idx].split('/')[-1]
        name = img_name.split('_'+img_name.split("_")[-1])[0]+'.jpg'
        
        print (self.matte_path+name)
        matte = cv.imread(self.matte_path+name)[:,:,0]
        #fg = cv.imread(self.fg_path+name)
        trimap,_ = self._generate_trimap(matte)

        img = cv2_to_pil(img)
        trimap = transforms.ToPILImage()(trimap)
        matte = transforms.ToPILImage()(matte)
        

        img = self.transform1(img)
        trimap = self.transform2(trimap)      
        matte = self.transform2(matte)

        return img,matte,trimap,self.images[idx]

    def __len__(self):
        return len(os.listdir(self.image_path))






class img_matting_dataset_fullx2(data.Dataset):
    def __init__(self, image_path, matte_path, transform1, transform2):
        self.image_path = image_path
        self.matte_path = matte_path
        self.transform1 = transform1
        self.transform2 = transform2
        self.images = self._walkFile(self.image_path)
        self.alphas = self._walkFile(self.matte_path)


    def _walkFile(self,file):
        samples = []
        for x in os.listdir(file):
            samples.append(file+"/"+x)
        return sorted(samples)

    def _generate_trimap(self,alpha):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
        trimap = fg * 255 + (unknown - fg) * 128
        mask = np.ones_like(trimap)*255
        mask[trimap==255]=0
        mask[trimap==0]=0
        return trimap.astype(np.uint8),mask.astype(np.uint8)


    def __getitem__(self, idx):

        img = cv.imread(self.images[idx])

        img_name = self.images[idx].split('/')[-1]
        name = img_name.split('_'+img_name.split("_")[-1])[0]+'.jpg'
        matte = cv.imread(self.alphas[idx])[:,:,0]
        trimap,_ = self._generate_trimap(matte)


        img = cv2_to_pil(img)
        trimap = transforms.ToPILImage()(trimap)
        matte = transforms.ToPILImage()(matte)
        
        img = self.transform1(img)
        trimap = self.transform2(trimap)      
        matte = self.transform2(matte)

        return img,matte,trimap,self.images[idx]

    def __len__(self):
        return len(os.listdir(self.image_path))





