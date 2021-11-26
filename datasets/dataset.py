from torch.utils.data import Dataset
import cv2 as cv
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
                 **kwargs): # whether to use city19 or city36 class set

        self.root_path = root_path
        self.sensors = sensors
        self.resize_to = resize_to
        self.crop_to = crop_to
        self.kwargs = kwargs
        self.augment_data = augment_data

        with open(path.join(splits_path, split+'.'+split_extension)) as f:
            self.items = [l.rstrip('\n').split(split_separator) for l in f][split_skiplines:]

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
                    self.resize_to[0] = int(W*self.resize_to[1]/H)
                else:
                    self.resize_to[1] = int(H*self.resize_to[0]/W)

            if rgb is not None: rgb = cv.resize(rgb, self.resize_to, interpolation=cv.INTER_AREA) # usually images are downsized, best results obtained with inter_area
            if gt is not None: gt = cv.resize(gt, self.resize_to, interpolation=cv.INTER_NEAREST_EXACT) # labels must be mapped as-is
            if depth is not None: depth = cv.resize(depth, self.resize_to, interpolation=cv.INTER_NEAREST_EXACT)

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
        if self.kwargs['flip'] and random.random()<.5:
            if rgb is not None: rgb = (rgb[:,::-1,...]).copy()
            if gt is not None: gt = (gt[:,::-1,...]).copy()
            if depth is not None: depth = (depth[:,::-1,...]).copy()
        if self.kwargs['gaussian_blur']:
            sigma = random.random()*self.kwargs['blur_mul']
            if rgb is not None: rgb = cv.GaussianBlur(rgb, (0,0), sigma)
            # not sure if it makes sense....
            #depth = cv.GaussianBlur(depth, (0,0), sigma) if depth is not None
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

    @staticmethod
    def to_pytorch(rgb=None, gt=None, depth=None):
        if rgb is not None: rgb = torch.from_numpy(np.transpose(rgb-[104.00698793, 116.66876762, 122.6789143], (2, 0, 1)).astype(np.float32))
        if gt is not None: torch.from_numpy(gt.astype(np.long))
        if depth is not None: torch.from_numpy(depth.astype(np.float32))
        return rgb, gt, depth

    def color_label(self, gt):
        return self.cmap[np.array(gt)]

    @staticmethod
    def to_rgb(tensor):
        t = np.array(tensor.transpose(0,1).transpose(1,2))+[104.00698793, 116.66876762, 122.6789143] # bgr
        t = np.round(t[...,::-1]).astype(np.uint8) # rgb
        return t

    @staticmethod
    def load_rgb(im_path):
        # image is read in bgr
        return cv.imread(im_path)

    @staticmethod
    def load_semantic(im_path):
        # image should be grayscale
        return cv.imread(im_path, cv.IMREAD_UNCHANGED)

    @staticmethod
    def load_depth(im_path):
        # image should be grayscale
        return cv.imread(im_path, cv.IMREAD_UNCHANGED)

    def map_to_train(self, gt):
        gt_clone = self.ignore_index*np.ones(gt.shape, dtype=np.long)
        if self.raw_to_train is not None:
            for k,v in self.raw_to_train.items():
                gt_clone[gt==k] = v
        return gt_clone
