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
                                 sensors=['rgb', 'semantic'],
                                 town=args.town,
                                 weather=args.weather,
                                 time_of_day=args.time_of_day,
                                 sensor_positions=['FL', 'F', 'FR'],
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
        self.model.to('cuda')
        assert os.path.exists(args.ckpt_file), "Checkpoint [%s] not found, aborting..."%(args.ckpt_file)
        
        self.logger.info("Loading checkpoint")
        self.model.load_state_dict(torch.load(args.ckpt_file))
        self.logger.info("Checkpoint loaded successfully")
        self.model.to('cuda')
        self.model.eval()

    def test(self):
        pbar = tqdm(self.tloader, total=len(self.tset), desc="Testing")
        metrics = Metrics(self.args.class_set, log_colors=True)
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(pbar):

                x, y = sample[0]['rgb'], sample[0]['semantic']
                x_c = x['F'].to('cuda', dtype=torch.float32) if type(x) is dict else x.to('cuda', dtype=torch.float32)
                x_l = x['FL'].to('cuda', dtype=torch.float32) if type(x) is dict else x.to('cuda', dtype=torch.float32)
                x_r = x['FR'].to('cuda', dtype=torch.float32) if type(x) is dict else x.to('cuda', dtype=torch.float32)
                y = y['F'].to('cuda', dtype=torch.long) if type(y) is dict else y.to('cuda', dtype=torch.long)
                
                x = torch.cat([x_l,x_c,x_r], dim=1)
                
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

        self.writer.add_image("test_input", self.tset.to_rgb(x_c[0].cpu()), 0, dataformats='HWC')
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
    
    set_seed(args.seed)
    
    tester = Tester(args, writer, logger)
    tester.test()
    
    tester.writer.flush()
    tester.writer.close()