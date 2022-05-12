from torch.utils.data import Dataset
import cv2 as cv
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
import random
from os import path, listdir

class BaseDataset(Dataset):
    def __init__(self,
                 root_path=None,
                 splits_path=None,
                 split='train',
                 split_extension='txt',
                 split_separator=' ',
                 split_skiplines=0,
                 resize_to=None,
                 crop_to=None,
                 augment_data=True,
                 sensors=['rgb'],
                 return_grayscale=False,
                 depth_mode='log',
                 **kwargs): # whether to use city19 or city36 class set

        self.root_path = root_path
        self.sensors = sensors
        self.resize_to = resize_to
        self.crop_to = crop_to
        self.kwargs = kwargs
        self.augment_data = augment_data
        self.return_grayscale = return_grayscale
        self.depth_mode = depth_mode

        with open(path.join(splits_path, split+'.'+split_extension)) as f:
            #self.items = [l.rstrip('\n').split(split_separator) for l in f][split_skiplines:]
            self.items = [[e.lstrip('/') for e in l.rstrip('\n').split(split_separator)] for l in f][split_skiplines:]

        self.init_ids()
        self.init_cmap()
        self.init_cnames()

    # to be overridden
    def init_ids(self):
        self.raw_to_train = {i:i for i in range(256)}
        self.ignore_index = -1

    # to be overridden
    def init_cmap(self):
        self.cmap = np.array([[i,i,i] for i in range(256)])
        
    def init_cnames(self):
        self.cnames = ["c%03d"%i for i in range(256)]

    def resize_and_crop(self, rgb=None, gt=None, depth=None):
        if self.resize_to is not None:
            if '' in self.resize_to and not (rgb is None and gt is None and depth is None):
                if rgb is not None: H, W, _ = rgb.shape
                if gt is not None: H, W = gt.shape
                if depth is not None: H, W = depth.shape
                if self.resize_to.index('') == 0:
                    resize_to = (int(W*self.resize_to[1]/H), self.resize_to[1])
                else:
                    resize_to = (self.resize_to[0], int(H*self.resize_to[0]/W))

            if rgb is not None: rgb = cv.resize(rgb, resize_to, interpolation=cv.INTER_AREA) # usually images are downsized, best results obtained with inter_area
            if gt is not None: gt = cv.resize(gt, resize_to, interpolation=cv.INTER_NEAREST_EXACT) # labels must be mapped as-is
            if depth is not None: depth = cv.resize(depth, resize_to, interpolation=cv.INTER_NEAREST_EXACT)

        if self.crop_to is not None:
            if rgb is not None: H, W, _ = rgb.shape
            if gt is not None: H, W = gt.shape
            if depth is not None: H, W = depth.shape
            if not (rgb is None and gt is None and depth is None):
                dh, dw = H-self.crop_to[1], W-self.crop_to[0]
                assert dh>=0 and dw >= 0, "Incompatible crop size: (%d, %d), images have dimensions: (%d, %d)"%(self.crop_to[0], self.crop_to[1], W, H)
                h0, w0 = random.randint(0, dh) if dh>0 else 0, random.randint(0, dw) if dw>0 else 0
            if rgb is not None: rgb = (rgb[h0:h0+self.crop_to[1], w0:w0+self.crop_to[0], ...]).copy()
            if gt is not None: gt = (gt[h0:h0+self.crop_to[1], w0:w0+self.crop_to[0], ...]).copy()
            if depth is not None: depth = (depth[h0:h0+self.crop_to[1], w0:w0+self.crop_to[0], ...]).copy()

        return rgb, gt, depth

    def data_augment(self, rgb=None, gt=None, depth=None):
        if rgb is not None and self.kwargs['gaussian_noise'] and random.random() <.5:
            stride1 = self.kwargs['noise_mul']*(np.random.rand(rgb.shape[0], rgb.shape[1], rgb.shape[2])-.5)
            stride2 = self.kwargs['noise_mul']**(cv.resize(
                            np.random.rand(rgb.shape[0]//2, rgb.shape[1]//2, rgb.shape[2]).astype(np.float32),
                            (rgb.shape[1], rgb.shape[0])
                        )-.5)
            stride4 = self.kwargs['noise_mul']**(cv.resize(
                            np.random.rand(rgb.shape[0]//4, rgb.shape[1]//4, rgb.shape[2]).astype(np.float32),
                            (rgb.shape[1], rgb.shape[0])
                        )-.5)
            stride8 = self.kwargs['noise_mul']**(cv.resize(
                            np.random.rand(rgb.shape[0]//8, rgb.shape[1]//8, rgb.shape[2]).astype(np.float32),
                            (rgb.shape[1], rgb.shape[0])
                        )-.5)
            stride16 = self.kwargs['noise_mul']**(cv.resize(
                            np.random.rand(rgb.shape[0]//16, rgb.shape[1]//16, rgb.shape[2]).astype(np.float32),
                            (rgb.shape[1], rgb.shape[0])
                        )-.5)
            stride32 = self.kwargs['noise_mul']**(cv.resize(
                            np.random.rand(rgb.shape[0]//32, rgb.shape[1]//32, rgb.shape[2]).astype(np.float32),
                            (rgb.shape[1], rgb.shape[0])
                        )-.5)
            rgb = np.round(np.clip(rgb+stride1+stride2+stride4+stride8+stride16+stride32, a_min=0, a_max=255)).astype(np.uint8)
            
        if rgb is not None and self.kwargs['color_shift'] and random.random() <.5:
            ch = random.randrange(3)
            shift_x = random.randrange(11)-5
            shift_y = random.randrange(11)-5
            rgb[...,ch] = np.roll(rgb[...,ch], shift_x, axis=1)
            rgb[...,ch] = np.roll(rgb[...,ch], shift_y, axis=0)
            
        if rgb is not None and self.kwargs['color_jitter'] and random.random() <.5:
            nw = np.random.randint(255-self.kwargs['wshift_intensity'], 255+self.kwargs['wshift_intensity'], size=(3,))
            rgb = rgb*(nw/255.) # shift white point
            rgb += np.random.randint(-self.kwargs['cshift_intensity'], self.kwargs['cshift_intensity'], size=(3,)) # add random color shift
            rgb = np.round(np.clip(rgb, a_min=0, a_max=255)).astype(np.uint8)
            
        if self.kwargs['flip'] and random.random() <.5:
            if rgb is not None: rgb = (rgb[:,::-1,...]).copy()
            if gt is not None: gt = (gt[:,::-1,...]).copy()
            if depth is not None: depth = (depth[:,::-1,...]).copy()
    
        if rgb is not None and self.kwargs['gaussian_blur'] and random.random() <.5:
            sigma = random.random()*self.kwargs['blur_mul']
            rgb = cv.GaussianBlur(rgb, (0,0), sigma)
            
        if depth is not None and self.kwargs['depth_noise']:
            if self.kwargs['depth_noise_mode'] == 'awgn':
                noise = np.random.normal(size=depth.shape)/self.kwargs['depth_noise_scale'] # the rescaling is needed to align distances, this way std=.1 meter
            if self.kwargs['depth_noise_mode'] == 'poisson':
                noise = np.random.poisson(size=depth.shape)/self.kwargs['depth_noise_scale'] # the rescaling is needed to align distances, this way std=.1 meter
            if self.kwargs['depth_noise_mode'] == 'awgn_weighted':
                noise = np.random.normal(size=depth.shape)*depth/(self.kwargs['depth_noise_scale']/1000)
            if self.kwargs['depth_noise_mode'] == 'poisson_weighted':
                noise = np.random.poisson(size=depth.shape)*depth/(self.kwargs['depth_noise_scale']/1000)
            depth = np.clip(depth+noise, a_min=0, a_max=1)
        return rgb, gt, depth

    def __getitem__(self, item):
        rgb_path, gt_path = self.items[item]

        rgb = self.load_rgb(path.join(self.root_path, rgb_path)) if 'rgb' in self.sensors else None
        gt = self.map_to_train(self.load_semantic(path.join(self.root_path, gt_path))) if 'semantic' in self.sensors else None

        rgb, gt, _ = self.resize_and_crop(rgb=rgb, gt=gt)
        if self.augment_data:
            rgb, gt, _ = self.data_augment(rgb=rgb, gt=gt)
        rgb, gt, _ = self.to_pytorch(rgb=rgb, gt=gt)

        out_dict = {}
        if rgb is not None: out_dict['rgb'] = rgb
        if gt is not None: out_dict['semantic'] = gt
        #if depth is not None: out_dict['depth'] = depth

        return out_dict, item

    def __len__(self):
        return len(self.items)

    def to_pytorch(self, rgb=None, gt=None, depth=None):
        if not self.return_grayscale:
            if rgb is not None:
                rgb = torch.from_numpy(np.transpose((rgb[...,::-1]-[104.00698793, 116.66876762, 122.67891434]), (2, 0, 1)).astype(np.float32))
                #rgb = torch.from_numpy(np.transpose((rgb[...,::-1]/255.-[0.485, 0.456, 0.406])/[0.485, 0.456, 0.406], (2, 0, 1)).astype(np.float32))
        else:
            if rgb is not None:
                rgb = torch.from_numpy(np.transpose(np.expand_dims(cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)/127.5 -1, -1), (2, 0, 1)).astype(np.float32))
        if gt is not None:
            gt = torch.from_numpy(gt.astype(np.long))
        if depth is not None:
            if self.depth_mode == 'linear':
                depth = torch.from_numpy((2*depth-1.).astype(np.float32)).unsqueeze(0) # depth should be normalized in [0,1] before input
            elif self.depth_mode == 'log':
                depth = torch.from_numpy((2*(np.log2(depth+1.))-1.).astype(np.float32)).unsqueeze(0) # depth should be normalized in [0,1] before input
            elif self.depth_mode == 'root4':
                depth = torch.from_numpy((2*np.power(depth, 1/4)-1.).astype(np.float32)).unsqueeze(0) # depth should be normalized in [0,1] before input
            else:
                depth = torch.from_numpy((2*np.sqrt(depth)-1.).astype(np.float32)).unsqueeze(0) # depth should be normalized in [0,1] before input
        return rgb, gt, depth

    def to_rgb(self, tensor, force_gray=False):
        if not (self.return_grayscale or force_gray):
            t = np.array(tensor.transpose(0,1).transpose(1,2))+[104.00698793, 116.66876762, 122.67891434]
        else:
            t = (np.array(tensor.transpose(0,1).transpose(1,2))+1.)*127.5
        t = np.round(t).astype(np.uint8) # rgb
        return t

    def color_label(self, gt):
        return self.cmap[np.array(gt)]

    @staticmethod
    def load_rgb(im_path):
        #image is read in bgr
        return cv.imread(im_path)
    
    @staticmethod
    def load_semantic(im_path):
        #image should be grayscale
        return cv.imread(im_path, cv.IMREAD_UNCHANGED)

    @staticmethod
    def load_depth(im_path):
        #image should be grayscale
        return (cv.imread(im_path, cv.IMREAD_UNCHANGED)/255.).astype(np.float32)

    def map_to_train(self, gt):
        gt_clone = self.ignore_index*np.ones(gt.shape, dtype=np.long)
        if self.raw_to_train is not None:
            for k,v in self.raw_to_train.items():
                gt_clone[gt==k] = v
        return gt_clone
