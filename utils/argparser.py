import logging
import sys
import os
import argparse

from datasets.carlaLTTM import LTTMDataset
from datasets.cityscapes import CityDataset
from datasets.gta5 import GTAVDataset
from datasets.idd import IDDDataset
from datasets.mapillary import MapillaryDataset

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("Unrecognized boolean: "+v)

def str2str_none_num(v,t=float):
    if v.lower() in ('none',):
        return None
    else:
        try:
            return t(v)
        except ValueError:
            return v

def str2intlist(v):
    l = v.split(',')
    l = [str2str_none_num(el,t=int) for el in l]
    return l if len(l)>1 else l[0]

def str2floatlist(v):
    l = v.split(',')
    l = [str2str_none_num(el,t=float) for el in l]
    return l if len(l)>1 else l[0]

def parse_dataset(dname):
    if dname=='lttm':
        return LTTMDataset
    elif dname=='city':
        return CityDataset
    elif dname=='gta':
        return GTAVDataset
    elif dname=='idd':
        return IDDDataset
    else:
        return MapillaryDataset

def init_params():

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--dataset', default="gta", type=parse_dataset,
                           choices=['lttm', 'city', 'gta', 'idd', 'mapi'],
                           help='The dataset used for supervised training')
    argparser.add_argument('--rescale_size', default=[1280,720], type=str2intlist,
                           help='Size the images will be resized to during loading, before crop - syntax:"1280,720"')
    argparser.add_argument('--crop_images', default=False, type=str2bool,
                           help='Whether to crop the images or not')
    argparser.add_argument('--crop_size', default=[512,512], type=str2intlist,
                           help='Size the images will be cropped to - syntax:"1280,720"')
    argparser.add_argument('--root_path', type=str, help='Path to the dataset root folder')
    argparser.add_argument('--splits_path', type=str, help='Path to the dataset split lists')
    argparser.add_argument('--train_split', default='train', type=str,
                           help='Split file to be used for training samples')
    argparser.add_argument('--val_split', default='val', type=str,
                           help='Split file to be used for validation samples')
                           
    argparser.add_argument('--augment_data', default=True, type=str2bool,
                           help='Whether to augment the (training) images with flip & Gaussian Blur')
    argparser.add_argument('--random_flip', default=True, type=str2bool,
                           help='Whether to randomly flip images l/r')
    argparser.add_argument('--gaussian_blur', default=True, type=str2bool,
                           help='Whether to apply random gaussian blurring')
    argparser.add_argument('--batch_size', default=1, type=int,
                           help='Training batch size')
    argparser.add_argument('--dataloader_workers', default=4, type=int,
                           help='Number of workers to use for each dataloader (significantly affects RAM consumption)')
    argparser.add_argument('--blur_mul', default=1, type=int)
    argparser.add_argument('--pin_memory', default=True, type=str2bool)
    
    argparser.add_argument('--classifier', default='DeepLabV3', type=str,
                           choices=['DeepLabV2', 'DeepLabV3', 'DeepLabV2MSIW', 'FCN', 'PSPNet', 'uNet'],
                           help='Which classifier head to use in the model')
    argparser.add_argument('--seed', default=12345, type=int,
                               help='Seed for the RNGs, for repeatability')
    argparser.add_argument('--lr', default=2.5e-4, type=float,
                           help='The learning rate to be used')
    argparser.add_argument('--poly_power', default=.9, type=float,
                               help='lr polynomial decay rate')
    argparser.add_argument('--decay_over_iterations', default=250000, type=int,
                           help='lr polynomial decay max_steps')
    argparser.add_argument('--iterations', default=50000, type=int,
                           help='Number of iterations performed')
    argparser.add_argument('--momentum', default=.9, type=float,
                               help='SGD optimizer momentum')
    argparser.add_argument('--weight_decay', default=1e-4, type=float,
                           help='SGD optimizer weight decay')
                           
    argparser.add_argument('--validate_every_steps', default=5000, type=int,
                       help='Number of iterations every which a validation is run, <= 0 disables validation')
    argparser.add_argument('--logdir', default="log/%d"%(int(time.time())), type=str,
                   help='Path to the log directory')
                   
    return argparser.parse_args()
    
    
def init_logger(args):
    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    # create the log path
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(args.logdir, flush_secs=.5)

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.logdir, 'train_log.txt'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Global configuration as follows:")
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
        
    return writer, logger