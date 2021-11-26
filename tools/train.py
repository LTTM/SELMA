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

class Trainer():
    def __init__(self, args, writer, logger):
        self.args = args
        self.writer = writer
        self.logger = logger

        self.tset = args.dataset(root_path=args.root_path,
                                 splits_path=args.splits_path,
                                 split=args.train_split,
                                 resize_to=args.rescale_size,
                                 crop_to=args.crop_size if args.crop_images else None,
                                 augment_data=args.augment_data,
                                 flip=args.random_flip,
                                 gaussian_blur=args.gaussian_blur,
                                 blur_mul=args.blur_mul,
                                 sensors=args.sensors,
                                 town=args.town,
                                 weather=args.weather,
                                 time_of_day=args.time_of_day,
                                 sensors_positions=args.positions)
        self.tloader = data.DataLoader(self.tset,
                                       shuffle=True,
                                       num_workers=args.dataloader_workers,
                                       batch_size=args.batch_size,
                                       drop_last=True,
                                       pin_memory=args.pin_memory)
                                 
        self.vset = args.dataset(root_path=args.root_path,
                                 splits_path=args.splits_path,
                                 split=args.val_split,
                                 resize_to=args.rescale_size,
                                 crop_to=None,
                                 augment_data=False,
                                 sensors=args.sensors,
                                 town=args.town,
                                 weather=args.weather,
                                 time_of_day=args.time_of_day,
                                 sensors_positions=args.positions)
        self.vloader = data.DataLoader(self.vset,
                                       shuffle=False,
                                       num_workers=args.dataloader_workers,
                                       batch_size=1,
                                       drop_last=True,
                                       pin_memory=args.pin_memory)

        self.args.validate_every_steps = min(self.args.validate_every_steps, len(self.tset))

        # to be changed when support to different class sets is added
        self.model = SegmentationModel(19, args.classifier)
        self.model.to('cuda')
        
        self.optim = torch.optim.SGD(params=self.model.parameters_dict,
                                     momentum=self.args.momentum,
                                     weight_decay=self.args.weight_decay)
                                     
        self.loss = CrossEntropyLoss(ignore_index=-1)
        self.loss.to('cuda')

        self.best_miou = -1
        self.best_epoch = -1

    def train(self):
        epochs = int(np.ceil(self.args.iterations/self.args.validate_every_steps))
        for epoch in range(epochs):
            self.logger.info("Starting epoch %d of %d"%(epoch+1, epochs))
            self.train_epoch(epoch)
            miou = self.validate(epoch)
            torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "latest.pth"))
            self.logger.info("Validation score at epoch %d is %.2f"%(epoch, miou))
            self.writer.add_scalar('val_mIoU', miou, epoch)
            if miou > self.best_miou:
                self.best_miou = miou
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "val_best.pth"))
            self.logger.info("Best validation score is %.2f at epoch %d"%(self.best_miou, self.best_epoch))

    def train_epoch(self, epoch):
        pbar = tqdm(self.tloader, total=self.args.validate_every_steps, desc="Training Epoch %d"%(epoch+1))
        metrics = Metrics(self.tset.cnames)
        for i, sample in enumerate(pbar):
            
            curr_iter = self.args.validate_every_steps*epoch + i
            lr = lr_scheduler(self.optim, curr_iter, self.args.lr, self.args.decay_over_iterations, self.args.poly_power, self.args.batch_size)
            self.writer.add_scalar('lr', lr, curr_iter)
        
            x, y = sample[0]['rgb'], sample[0]['semantic']
            x = x['D'].to('cuda', dtype=torch.float32) if type(x) is dict else x.to('cuda', dtype=torch.float32)
            y = y['D'].to('cuda', dtype=torch.long) if type(y) is dict else y.to('cuda', dtype=torch.long)
            
            self.optim.zero_grad()
            
            out = self.model(x)
            if type(out) is tuple:
                out, feats = out
            l = self.loss(out, y)
            l.backward()
            self.writer.add_scalar('ce', l.item(), curr_iter)
            
            pred = torch.argmax(out.detach(), dim=1)
            metrics.add_sample(pred, y)
            self.writer.add_scalar('step_mIoU', metrics.percent_mIoU(), curr_iter)
            
            self.optim.step()
            
            if i == self.args.validate_every_steps:
                break
            
            if curr_iter == self.args.iterations-1:
                break
        
        self.writer.add_image("train_input", self.tset.to_rgb(x[0].cpu()), epoch, dataformats='HWC')
        self.writer.add_image("train_label", self.tset.color_label(y[0].cpu()), epoch, dataformats='HWC')
        self.writer.add_image("train_pred", self.tset.color_label(pred[0].cpu()), epoch, dataformats='HWC')
        
        miou = metrics.percent_mIoU()
        self.writer.add_scalar('train_mIoU', miou, epoch)
        self.logger.info("Epoch %d done, score: %.2f -- Starting Validation"%(epoch, miou))
        
    def validate(self, epoch):
        pbar = tqdm(self.vloader, total=len(self.vset), desc="Validation")
        metrics = Metrics(self.tset.cnames)
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

        self.writer.add_image("val_input", self.tset.to_rgb(x[0].cpu()), epoch, dataformats='HWC')
        self.writer.add_image("val_label", self.tset.color_label(y[0].cpu()), epoch, dataformats='HWC')
        self.writer.add_image("val_pred", self.tset.color_label(pred[0].cpu()), epoch, dataformats='HWC')

        self.model.train()
        return metrics.percent_mIoU()

if __name__ == "__main__":
    
    args = init_params()
    writer, logger = init_logger(args)
    
    set_seed(args.seed)
    
    trainer = Trainer(args, writer, logger)
    trainer.train()
    
    trainer.writer.flush()
    trainer.writer.close()