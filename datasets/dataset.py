from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import torch
import random
from os import path, listdir

class BaseDataset(Dataset):
    def __init__(self,
                 root_path=None,
                 split='train',
                 split_extension='txt',
                 split_separator=' ',
                 resize_to=None,
                 crop_to=None,
                 sensors=['rgb'],
                 **kwargs): # whether to use city19 or city36 class set

        self.root_path = root_path
        self.sensors = sensors
        self.resize_to = resize_to
        self.crop_to = crop_to

        with open(path.join(root_path,split+'.'+split_extension)) as f:
            self.items = [l.rstrip('\n').split(split_separator) for l in f]

        self.init_ids()

    # to be overridden
    def init_ids(self):
        self.raw_to_train = {i:i for i in range(256)}
        self.cmap = np.array([[i,i,i] for i in range(256)])
        self.ignore_index = -1

    def resize_and_crop(self, rgb=None, gt=None, depth=None):
        if self.resize_to is not None:
            if rgb is not None: rgb = cv.resize(rgb, self.resize_to, interpolation=cv.INTER_AREA) # usually images are downsized, best results obtained with inter_area
            if gt is not None: gt = cv.resize(gt, self.resize_to, interpolation=cv.INTER_NEAREST_EXACT) # labels must be mapped as-is
            if depth is not None: depth = cv.resize(depth, self.resize_to, interpolation=cv.INTER_NEAREST_EXACT)

        if self.crop_size is not None:
            if rgb is not None: H, W, _ = rgb.shape
            if gt is not None: H, W = gt.shape
            if depth is not None: H, W = depth.shape
            if not (rgb is None and gt is None and depth is None):
                dh, dw = H-self.crop_size[1], W-self.crop_size[0]
                assert dh>=0 and dw >= 0, "Incompatible crop size: (%d, %d), images have dimensions: (%d, %d)"%(self.crop_size[0], self.crop_size[1], W, H)
                h0, w0 = random.randint(0, dh) if dh>0 else 0, random.randint(0, dw) if dw>0 else 0
            if rgb is not None: rgb = (rgb[h0:h0+self.crop_size[1], w0:w0+self.crop_size[0], ...]).copy()
            if gt is not None: gt = (gt[h0:h0+self.crop_size[1], w0:w0+self.crop_size[0], ...]).copy()
            if depth is not None: depth = (depth[h0:h0+self.crop_size[1], w0:w0+self.crop_size[0], ...]).copy()

        return rgb, gt, depth

    def data_augment(self, rgb=None, gt=None, depth=None):
        if self.kwargs['flip'] and random.random()<.5:
            if rgb is not None: rgb = (rgb[:,::-1,...]).copy()
            if gt is not None: gt = (gt[:,::-1,...]).copy()
            if depth is not None: depth = (gt[:,::-1,...]).copy()
        if self.kwargs['gaussian_blur']:
            sigma = random.random()*self.kwargs['blur_mul']
            if rgb is not None: rgb = cv.GaussianBlur(rgb, (0,0), sigma)
            # not sure if it makes sense....
            #depth = cv.GaussianBlur(depth, (0,0), sigma) if depth is not None
        return rgb, gt, depth

    @staticmethod
    def to_pytorch(bgr, gt):
        bgr = np.transpose(bgr.astype(np.float32)-[104.00698793, 116.66876762, 122.6789143], (2, 0, 1))
        return torch.from_numpy(bgr), torch.from_numpy(gt)

    @staticmethod
    def to_rgb(tensor):
        t = np.array(tensor.transpose(0,1).transpose(1,2))+[104.00698793, 116.66876762, 122.6789143] # bgr
        t = np.round(t[...,::-1]).astype(np.uint8) # rgb
        return t

    @staticmethod
    def read_im(im_path):
        # image is read in bgr
        return cv.imread(im_path)

    @staticmethod
    def read_gt(im_path):
        # image should be grayscale
        return cv.imread(im_path, cv.IMREAD_UNCHANGED)

    @staticmethod
    def load_depth(im_path, rescale=True): # return the depth in meters
        t = cv.imread(im_path).astype(int)*np.array([256*256, 256, 1])
        t = t.sum(axis=2)/(256 * 256 * 256 - 1.)
        return 1000.*t if rescale else t

    def map_to_train(self, gt):
        gt_clone = self.ignore_index*np.ones(gt.shape, dtype=np.long)
        if self.raw_to_train is not None:
            for k,v in self.raw_to_train.items():
                gt_clone[gt==k] = v
        return gt_clone
