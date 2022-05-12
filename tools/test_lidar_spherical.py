import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
import random

import torch
torch.backends.cudnn.benchmark = True

from torch.nn import CrossEntropyLoss
from utils.losses import MSIW

from torch.utils import data
from tqdm import tqdm

from utils.idsmask import ids_dict
from utils.argparser import init_params, init_logger
from utils.metrics import Metrics
from models.model import SegmentationModel


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
def lr_scheduler(optimizer, iteration, init_lr, decay_over, poly_power, batch_size):
    max_iters = decay_over//batch_size
    lr = init_lr * (1 - float(iteration) / max_iters) ** poly_power
    optimizer.param_groups[0]["lr"] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]["lr"] = 10 * lr
    return lr

class Tester():
    def __init__(self, args, writer, logger):
        self.args = args
        self.writer = writer
        self.logger = logger

        self.tset = args.dataset(root_path=args.root_path,
                                 splits_path=args.splits_path,
                                 split=args.test_split,
                                 resize_to=args.rescale_size,
                                 crop_to=None,
                                 augment_data=False,
                                 sensors=['lidar'],
                                 town=args.town,
                                 weather=args.weather,
                                 time_of_day=args.time_of_day,
                                 sensor_positions=['T'],
                                 class_set=args.class_set,
                                 return_grayscale=args.input_channels==1)
        self.tloader = data.DataLoader(self.tset,
                                       shuffle=False,
                                       num_workers=args.dataloader_workers,
                                       batch_size=1,
                                       drop_last=True,
                                       pin_memory=args.pin_memory)

        
        num_classes = len(self.tset.cnames)
        self.logger.info("Training on class set: %s, Classes: %d"%(args.class_set, num_classes))
        
        self.model = SegmentationModel(args.input_channels, num_classes, args.classifier)
        assert os.path.exists(args.ckpt_file), "Checkpoint [%s] not found, aborting..."%(args.ckpt_file)
        
        self.logger.info("Loading checkpoint")
        self.model.load_state_dict(torch.load(args.ckpt_file))
        self.logger.info("Checkpoint loaded successfully")
        self.model.to('cuda')
        self.model.eval()

    def to_spherical(self, x_raw, y_raw):
        x, y = torch.ones(x_raw.shape[0],1,64,1563, dtype=torch.float32), -torch.ones(x_raw.shape[0],64,1563, dtype=torch.long)
        r = torch.norm(x_raw, dim=-1)
        mask = r > 0
        t = torch.round(31*(9*torch.arccos(x_raw[...,2]/r)/np.pi - 4.)).to(torch.long)
        p = torch.round(781.5*torch.atan2(x_raw[...,1], x_raw[...,0])/np.pi + 780.5).to(torch.long)
        
        x[:,0,t[mask],p[mask]] = 2*(torch.pow(r[mask]/100, 1/4) - .5)
        y[:,t[mask],p[mask]] = y_raw[mask]
            
        return x.to('cuda', dtype=torch.float32), y.to('cuda', torch.long)

    def test(self):
        pbar = tqdm(self.tloader, total=len(self.tset), desc="Testing")
        metrics = Metrics(self.args.class_set, log_colors=True)
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(pbar):

                x_raw, y_raw = sample[0]['lidar']['T'][0], sample[0]['lidar']['T'][1].to(torch.long)
                x, y = self.to_spherical(x_raw, y_raw)
                
                out = self.model(x)
                if type(out) is tuple:
                    out, feats = out
                pred = torch.argmax(out.detach(), dim=1)
                metrics.add_sample(pred, y)
                # same testing as maxsquare, check also flipped image (should increase performance)
                x, y = x[...,list(range(x.shape[-1]-1,-1,-1))], y[...,list(range(y.shape[-1]-1,-1,-1))]
                out = self.model(x) 
                if type(out) is tuple:
                    out, feats = out
                pred = torch.argmax(out.detach(), dim=1)
                metrics.add_sample(pred, y) # check also shape

        self.writer.add_image("test_input", self.tset.to_rgb(x[0].cpu()), 0, dataformats='HWC')
        self.writer.add_image("test_label", self.tset.color_label(y[0].cpu()), 0, dataformats='HWC')
        self.writer.add_image("test_pred", self.tset.color_label(pred[0].cpu()), 0, dataformats='HWC')
        
        self.logger.info("Evaluation on Class Set: %s\n"%self.args.class_set + str(metrics))
        self.writer.add_scalar("test_mIoU_%s"%self.args.class_set, metrics.percent_mIoU(), 0)
        for cset in ids_dict[self.args.class_set]:
            self.writer.add_scalar("test_mIoU_%s"%cset, metrics.percent_mIoU(cset), 0)
            self.logger.info("Evaluation on Class Set: %s\n"%cset + metrics.str_class_set(cset))


if __name__ == "__main__":
    
    args = init_params('test')
    writer, logger = init_logger(args)
    
    tester = Tester(args, writer, logger)
    tester.test()
    
    tester.writer.flush()
    tester.writer.close()