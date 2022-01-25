import argparse
import os
import time
import shutil
from tensorboardX import SummaryWriter
import logging

from datasets.carlaLTTM import LTTMDataset
from datasets.cityscapes_white import CityDataset
from datasets.gta5 import GTAVDataset
from datasets.idd import IDDDataset
from datasets.idda import IDDADataset
from datasets.synthia import SYNTHIADataset
from datasets.mapillary import MapillaryDataset
from datasets.acdc import ACDCDataset
from datasets.tipnt import TIPNTDataset

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
    elif dname=='idda':
        return IDDADataset
    elif dname=='synthia':
        return SYNTHIADataset
    elif dname=='acdc':
        return ACDCDataset
    elif dname=='tipnt':
        return TIPNTDataset
    else:
        return MapillaryDataset

def init_params(train_type='source'):

    argparser = argparse.ArgumentParser()

    if train_type in ['source', 'uda', 'uda_fs', 'test', 'mixed', 'depth', 'depthtest']:
        argparser.add_argument('--dataset', default="gta", type=parse_dataset,
                               choices=[LTTMDataset, CityDataset, GTAVDataset, IDDDataset, IDDADataset, SYNTHIADataset, ACDCDataset, MapillaryDataset, TIPNTDataset],
                               help="The dataset used for supervised training, choose from ['lttm', 'city', 'gta', 'idd', 'synthia', 'acdc', 'mapi', 'tipnt']")
        argparser.add_argument('--rescale_size', default=[1280,''], type=str2intlist,
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

    if train_type in ['source', 'uda', 'uda_fs', 'mixed']:
        argparser.add_argument('--target_dataset', default="city", type=parse_dataset,
                               choices=[LTTMDataset, CityDataset, GTAVDataset, IDDDataset, MapillaryDataset],
                               help="The dataset used as target, choose from ['lttm', 'city', 'gta', 'idd', 'mapi']")
        argparser.add_argument('--target_rescale_size', default=[1280,''], type=str2intlist,
                               help='Size the images will be resized to during loading, before crop - syntax:"1280,720"')
        argparser.add_argument('--target_crop_images', default=False, type=str2bool,
                               help='Whether to crop the images or not')
        argparser.add_argument('--target_crop_size', default=[512,512], type=str2intlist,
                               help='Size the images will be cropped to - syntax:"1280,720"')
        argparser.add_argument('--target_root_path', type=str, help='Path to the dataset root folder')
        argparser.add_argument('--target_splits_path', type=str, help='Path to the dataset split lists')
        argparser.add_argument('--target_train_split', default='train', type=str,
                               help='Split file to be used for training samples')
        argparser.add_argument('--target_val_split', default='val', type=str,
                               help='Split file to be used for validation samples')
        

    if train_type in ['test', 'depthtest']:
        argparser.add_argument('--test_split', default='test', type=str,
                               help='Split file to be used for test samples')

    # argparser.add_argument('--sensors', default='rgb,semantic', type=str2intlist,
                           # help='Sensors to be used - syntax:"sen1,sen2,..."')
    # argparser.add_argument('--target_sensors', default='rgb,semantic', type=str2intlist,
                           # help='Sensors to be used - syntax:"sen1,sen2,..."')

    argparser.add_argument('--positions', default='D', type=str2intlist,
                           help='Positions of the sensors, only for lttm set - syntax:"pos1,pos2,..."')
    argparser.add_argument('--town', default=None, type=str,
                           help='Override town on lttm dataset')
    argparser.add_argument('--time_of_day', default=None, type=str2str_none_num,
                           help='Override Time-of-Day on lttm dataset')
    argparser.add_argument('--weather', default=None, type=str2str_none_num,
                           help='Override Weather on lttm dataset')

    argparser.add_argument('--class_set', default='city19', type=str,
                           help='Which class set to use.', choices=['city19', 'idd17', 'synthia16', 'idda16', 'sii15', 'crosscity13', 'cci12'])

    if train_type in ['source', 'uda', 'uda_fs', 'mixed', 'depth']:
        argparser.add_argument('--augment_data', default=True, type=str2bool,
                               help='Whether to augment the (training) images with flip & Gaussian Blur')
        argparser.add_argument('--random_flip', default=True, type=str2bool,
                               help='Whether to randomly flip images l/r')
        argparser.add_argument('--gaussian_blur', default=True, type=str2bool,
                               help='Whether to apply random gaussian blurring')
        argparser.add_argument('--blur_mul', default=5, type=int)
        argparser.add_argument('--gaussian_noise', default=True, type=str2bool,
                               help='Whether to apply random gaussian noise')
        argparser.add_argument('--noise_mul', default=20, type=int)
        argparser.add_argument('--color_shift', default=True, type=str2bool,
                               help='Whether to randomly shift color channels')
        argparser.add_argument('--color_jitter', default=True, type=str2bool,
                               help='Whether to jitter color channels')
        argparser.add_argument('--cshift_intensity', default=20, type=int)
        argparser.add_argument('--wshift_intensity', default=55, type=int)

    argparser.add_argument('--batch_size', default=1, type=int,
                           help='Training batch size')
    argparser.add_argument('--dataloader_workers', default=8, type=int,
                           help='Number of workers to use for each dataloader (significantly affects RAM consumption)')
    argparser.add_argument('--pin_memory', default=False, type=str2bool)
    
    argparser.add_argument('--input_channels', default=3, type=int,
                           help='Number of input channels of the network')
    argparser.add_argument('--classifier', default='DeepLabV3', type=str,
                           choices=['DeepLabV2', 'DeepLabV3', 'DeepLabV2MSIW', 'FCN', 'PSPNet', 'UNet'],
                           help='Which classifier head to use in the model')
    argparser.add_argument('--seed', default=12345, type=int,
                               help='Seed for the RNGs, for repeatability')
   
    if train_type in ['source', 'uda', 'uda_fs', 'mixed', 'depth']:
        argparser.add_argument('--sup_loss', default='ce', type=str, choices=['ce', 'msiw'],
                               help='The supervised loss to be used for optimimization')
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
        argparser.add_argument('--validate_every_steps', default=2500, type=int,
                               help='Number of iterations every which a validation is run, <= 0 disables validation')
        argparser.add_argument('--ce_use_weights', default=False, type=str2bool,
                               help='Whether to use pixel frequencies to normalize the cross-entropy')

    if train_type in ['source']:
        argparser.add_argument('--validate_on_target', default=False, type=str2bool,
                               help='Whether to also validate on target dataset')

    argparser.add_argument('--logdir', default="log/%d"%(int(time.time())), type=str,
                   help='Path to the log directory')

    if train_type in ['uda', 'test', 'depthtest']:
        argparser.add_argument('--ckpt_file', default=None, type=str,
                       help='Path to the model checkpoint, used in test script')

    if train_type in ['uda', 'uda_fs']:
        argparser.add_argument('--lambda_msiw', default=1e-1, type=float,
                               help='UDA MaxSquareIW loss coefficient')
        argparser.add_argument('--alpha_msiw', default=0, type=float,
                               help='MaxSquareIW alpha coefficient')
        argparser.add_argument('--beta_msiw', default=0, type=float,
                               help='MaxSquareIWEX beta coefficient')

    if train_type in ['mixed']:
        argparser.add_argument('--target_sample_prob', default=0, type=float,
                               help='Probability of choosing a target dataset sample during training')

    if train_type in ['depth', 'depthtest']:
        argparser.add_argument('--depth_mode', default='root4', type=str,
                               choices=['log', 'root4', 'linear', 'sqrt'],
                               help='type of prepreocessing for the depth')
        argparser.add_argument('--depth_feed_mode', default='input', type=str,
                               choices=['input', 'layer1'],
                               help='How to provide depth information to the netowork')

    return argparser.parse_args()
    
class StripColorsFormatter(logging.Formatter):
    def format(self,record):
        fmt = super(StripColorsFormatter, self).format(record)
        return fmt.replace('\033[91m','').replace('\033[92m','').replace('\033[93m','').replace('\033[96m','').replace('\033[0m','')
    
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
    fhformatter = StripColorsFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    chformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fhformatter)
    ch.setFormatter(chformatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Global configuration as follows:")
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
        
    return writer, logger