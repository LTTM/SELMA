import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
import random
import torch
torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.utils import data
from tqdm import tqdm

from utils.argparser import init_params, init_logger
from utils.metrics import Metrics
from models.model import SegmentationModel

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
                                 sensors=args.sensors,
                                 town=args.town,
                                 weather=args.weather,
                                 time_of_day=args.time_of_day,
                                 sensors_positions=args.positions)
        self.tloader = data.DataLoader(self.tset,
                                       shuffle=False,
                                       num_workers=args.dataloader_workers,
                                       batch_size=1,
                                       drop_last=True,
                                       pin_memory=args.pin_memory)

        
        # to be changed when support to different class sets is added
        self.model = SegmentationModel(19, args.classifier)
        assert os.path.exists(args.ckpt_file), "Checkpoint [%s] not found, aborting..."%(args.ckpt_file)
        
        self.logger.info("Loading checkpoint")
        self.model.load_state_dict(torch.load(args.ckpt_file))
        self.logger.info("Checkpoint loaded successfully")
        self.model.to('cuda')
        self.model.eval()
        
    def test(self):
        pbar = tqdm(self.tloader, total=len(self.tset), desc="Testing")
        metrics = Metrics(self.tset.cnames, log_colors=True)
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(pbar):

                x, y = sample[0]['rgb'], sample[0]['semantic']
                x = x['D'].to('cuda', dtype=torch.float32) if type(x) is dict else x.to('cuda', dtype=torch.float32)
                y = y['D'].to('cuda', dtype=torch.long) if type(y) is dict else y.to('cuda', dtype=torch.long)
                
                out = self.model(x)
                if type(out) is tuple:
                    out, feats = out
                pred = torch.argmax(out.detach(), dim=1)
                metrics.add_sample(pred, y)

        self.writer.add_image("test_input", self.tset.to_rgb(x[0].cpu()), 0, dataformats='HWC')
        self.writer.add_image("test_label", self.tset.color_label(y[0].cpu()), 0, dataformats='HWC')
        self.writer.add_image("test_pred", self.tset.color_label(pred[0].cpu()), 0, dataformats='HWC')
        self.writer.add_scalar("test_mIoU", metrics.percent_mIoU(), 0)
        self.logger.info("\n"+str(metrics))

if __name__ == "__main__":
    
    args = init_params()
    writer, logger = init_logger(args)
    
    tester = Tester(args, writer, logger)
    tester.test()
    
    tester.writer.flush()
    tester.writer.close()