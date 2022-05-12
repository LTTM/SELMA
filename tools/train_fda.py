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

from datasets.selma_fda import SELMA_FDA
from datasets.carlaLTTM import LTTMDataset
from datasets.cityscapes_white import CityDataset

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

        self.tset = SELMA_FDA()
        self.tloader = data.DataLoader(self.tset,
                                       shuffle=True,
                                       num_workers=args.dataloader_workers,
                                       batch_size=args.batch_size,
                                       drop_last=True,
                                       pin_memory=args.pin_memory)
                                 
        self.vset = LTTMDataset(root_path = "D:/Datasets/SELMA/data",
                                 splits_path = "D:/Datasets/SELMA/splits",
                                 split = "val_rand",
                                 sensors=['rgb', 'semantic'],
                                 augment_data=False,
                                 resize_to=(1280,640))
        self.vloader = data.DataLoader(self.vset,
                                       shuffle=False,
                                       num_workers=args.dataloader_workers,
                                       batch_size=1,
                                       drop_last=True,
                                       pin_memory=args.pin_memory)

        self.args.validate_every_steps = min(self.args.validate_every_steps, len(self.tset)//args.batch_size)
        
        
        if hasattr(args, 'validate_on_target') and args.validate_on_target:
            self.tvset = CityDataset(root_path = "F:/Dataset/Cityscapes_extra",
                                     splits_path = "F:/Dataset/Cityscapes_extra",
                                     sensors=['rgb', 'semantic'],
                                     split='val',
                                     augment_data=False,
                                     resize_to=(1280,640))
            self.tvloader = data.DataLoader(self.tvset,
                                             shuffle=False,
                                             num_workers=args.dataloader_workers,
                                             batch_size=1,
                                             drop_last=True,
                                             pin_memory=args.pin_memory)

        # to be changed when support to different class sets is added
        num_classes = len(self.vset.cnames)
        self.logger.info("Training on class set: %s, Classes: %d"%(args.class_set, num_classes))
        
        self.model = SegmentationModel(args.input_channels, num_classes, args.classifier)
        self.model.to('cuda')
        
        self.optim = torch.optim.SGD(params=self.model.parameters_dict,
                                     momentum=self.args.momentum,
                                     weight_decay=self.args.weight_decay)
        
        if self.args.ce_use_weights:
            ws = np.load("datasets/frequencies/%s_%s.npy"%(args.dataset.__name__[:-7], args.class_set))[:-1]
            ws = ws.sum()/ws # i.e. 1/(ws/ws.sum())
            self.logger.info("Using CE-Class Weights:\n"+str(ws))
        
        self.loss = CrossEntropyLoss(ignore_index=-1, weight=torch.FloatTensor(ws) if self.args.ce_use_weights else None)
        self.loss_t = MSIW(.2, ignore_index=-1)
        
        self.loss.to('cuda')
        self.loss_t.to('cuda')

        self.best_miou = -1
        self.best_epoch = -1
        
        if hasattr(args, 'validate_on_target') and self.args.validate_on_target:
            self.target_best_miou = -1
            self.target_best_epoch = -1

    def train(self):
        epochs = int(np.ceil(self.args.iterations/self.args.validate_every_steps))
        for epoch in range(epochs):
            self.logger.info("Starting epoch %d of %d"%(epoch+1, epochs))
            self.train_epoch(epoch)
            torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "latest.pth"))
            
            miou = self.validate(epoch)
            self.logger.info("Validation score at epoch %d is %.2f"%(epoch+1, miou))
            self.writer.add_scalar('val_mIoU', miou, epoch+1)
            if miou > self.best_miou:
                self.best_miou = miou
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "val_best.pth"))
            self.logger.info("Best validation score is %.2f at epoch %d"%(self.best_miou, self.best_epoch+1))
            
            if hasattr(args, 'validate_on_target') and self.args.validate_on_target:
                miou = self.validate_target(epoch)
                self.logger.info("Target Validation score at epoch %d is %.2f"%(epoch+1, miou))
                if miou > self.target_best_miou:
                    self.target_best_miou = miou
                    self.target_best_epoch = epoch
                    torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "val_target_best.pth"))
                self.logger.info("Best target validation score is %.2f at epoch %d"%(self.target_best_miou, self.target_best_epoch+1))

    def train_epoch(self, epoch):
        pbar = tqdm(self.tloader, total=self.args.validate_every_steps, desc="Training Epoch %d"%(epoch+1))
        metrics = Metrics(self.args.class_set)
        for i, sample in enumerate(pbar):
            
            curr_iter = self.args.validate_every_steps*epoch + i
            lr = lr_scheduler(self.optim, curr_iter, self.args.lr, self.args.decay_over_iterations, self.args.poly_power, self.args.batch_size)
            self.writer.add_scalar('lr', lr, curr_iter)
        
            x, y, x_t = sample
            x, y = x.to('cuda', dtype=torch.float32), y.to('cuda', dtype=torch.long)
            
            self.optim.zero_grad()
            
            out, _ = self.model(x)
            l = self.loss(out, y)
            l.backward()
            self.writer.add_scalar('sup_loss', l.item(), curr_iter)
            
            pred = torch.argmax(out.detach(), dim=1)
            metrics.add_sample(pred, y)
            self.writer.add_scalar('step_mIoU', metrics.percent_mIoU(), curr_iter)
            
            if epoch > 4:
                x_t = x_t.to('cuda', dtype=torch.float32)
                out_t, _ = self.model(x_t)
                lt = self.loss_t(out_t)
                lt.backward()
                self.writer.add_scalar('msiw_loss', lt.item(), curr_iter)
            
            self.optim.step()
            
            if i == self.args.validate_every_steps:
                break
            
            if curr_iter == self.args.iterations-1:
                break
        
        self.writer.add_image("train_input", self.vset.to_rgb(x[0].cpu()), epoch+1, dataformats='HWC')
        self.writer.add_image("train_label", self.vset.color_label(y[0].cpu()), epoch+1, dataformats='HWC')
        self.writer.add_image("train_pred", self.vset.color_label(pred[0].cpu()), epoch+1, dataformats='HWC')
        
        miou = metrics.percent_mIoU()
        self.writer.add_scalar('train_mIoU', miou, epoch+1)
        self.logger.info("Epoch %d done, score: %.2f -- Starting Validation"%(epoch+1, miou))
        
    def validate(self, epoch):
        pbar = tqdm(self.vloader, total=len(self.vset), desc="Validation")
        metrics = Metrics(self.args.class_set)
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

        self.writer.add_image("val_input", self.vset.to_rgb(x[0].cpu()), epoch+1, dataformats='HWC')
        self.writer.add_image("val_label", self.vset.color_label(y[0].cpu()), epoch+1, dataformats='HWC')
        self.writer.add_image("val_pred", self.vset.color_label(pred[0].cpu()), epoch+1, dataformats='HWC')

        self.model.train()
        return metrics.percent_mIoU()
        
    def validate_target(self, epoch):
        pbar = tqdm(self.tvloader, total=len(self.tvset), desc="Validation")
        metrics = Metrics(self.args.class_set)
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

        self.writer.add_image("val_target_input", self.tvset.to_rgb(x[0].cpu()), epoch+1, dataformats='HWC')
        self.writer.add_image("val_target_label", self.tvset.color_label(y[0].cpu()), epoch+1, dataformats='HWC')
        self.writer.add_image("val_target_pred", self.tvset.color_label(pred[0].cpu()), epoch+1, dataformats='HWC')
        
        miou = metrics.percent_mIoU()
        self.writer.add_scalar('val_target_mIoU_%s'%self.args.class_set, miou, epoch+1)
        for cset in ids_dict[self.args.class_set]:
            self.writer.add_scalar("val_target_mIoU_%s"%cset, metrics.percent_mIoU(cset), epoch+1)

        self.model.train()
        return miou

if __name__ == "__main__":
    
    args = init_params('source')
    writer, logger = init_logger(args)
    
    set_seed(args.seed)
    
    trainer = Trainer(args, writer, logger)
    trainer.train()
    
    trainer.writer.flush()
    trainer.writer.close()