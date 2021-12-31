import sys, os
sys.path.append(os.path.abspath('.'))

from tools.train import *

class TrainerMixed(Trainer):
    def __init__(self, args, writer, logger):
        super(TrainerMixed, self).__init__(args, writer, logger)

        self.ttset = args.target_dataset(root_path=args.target_root_path,
                                         splits_path=args.target_splits_path,
                                         split=args.target_train_split,
                                         resize_to=args.target_rescale_size,
                                         crop_to=args.target_crop_size if args.target_crop_images else None,
                                         augment_data=args.augment_data,
                                         flip=args.random_flip,
                                         gaussian_blur=args.gaussian_blur,
                                         blur_mul=args.blur_mul//2,
                                         gaussian_noise=False,
                                         noise_mul=False,
                                         color_shift=False,
                                         color_jitter=False,
                                         sensors=args.sensors,
                                         town=args.town,
                                         weather=args.weather,
                                         time_of_day=args.time_of_day,
                                         sensors_positions=args.positions,
                                         class_set=args.class_set)
        self.ttloader = data.DataLoader(self.ttset,
                                         shuffle=True,
                                         num_workers=args.dataloader_workers,
                                         batch_size=args.batch_size,
                                         drop_last=True,
                                         pin_memory=args.pin_memory)
        
        self.tvset = args.target_dataset(root_path=args.target_root_path,
                                         splits_path=args.target_splits_path,
                                         split=args.target_val_split,
                                         resize_to=args.target_rescale_size,
                                         crop_to=None,
                                         augment_data=False,
                                         sensors=args.target_sensors,
                                         town=args.town,
                                         weather=args.weather,
                                         time_of_day=args.time_of_day,
                                         sensors_positions=args.positions,
                                         class_set=args.class_set)
        self.tvloader = data.DataLoader(self.tvset,
                                         shuffle=False,
                                         num_workers=args.dataloader_workers,
                                         batch_size=1,
                                         drop_last=True,
                                         pin_memory=args.pin_memory)
        
        self.target_best_miou = -1
        self.target_best_epoch = -1
        
    def train(self):
        epochs = int(np.ceil(self.args.iterations/self.args.validate_every_steps))
        for epoch in range(epochs):
            self.logger.info("Starting epoch %d of %d"%(epoch+1, epochs))
            self.train_mixed_epoch(epoch)
            torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "latest.pth"))
            
            miou = self.validate(epoch)
            self.logger.info("Validation score at epoch %d is %.2f"%(epoch+1, miou))
            self.writer.add_scalar('val_mIoU', miou, epoch+1)
            if miou > self.best_miou:
                self.best_miou = miou
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "val_best.pth"))
            self.logger.info("Best validation score is %.2f at epoch %d"%(self.best_miou, self.best_epoch+1))

            miou = self.validate_target(epoch)
            self.logger.info("Target Validation score at epoch %d is %.2f"%(epoch+1, miou))
            if miou > self.target_best_miou:
                self.target_best_miou = miou
                self.target_best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "val_target_best.pth"))
            self.logger.info("Best target validation score is %.2f at epoch %d"%(self.target_best_miou, self.target_best_epoch+1))

    def train_mixed_epoch(self, epoch):
        pbar = tqdm(self.tloader, total=self.args.validate_every_steps, desc="Training Epoch %d"%(epoch+1))
        metrics = Metrics(self.args.class_set)
        it_tt = iter(self.ttloader)
        for i, sample in enumerate(pbar):
            
            curr_iter = self.args.validate_every_steps*epoch + i
            lr = lr_scheduler(self.optim, curr_iter, self.args.lr, self.args.decay_over_iterations, self.args.poly_power, self.args.batch_size)
            self.writer.add_scalar('lr', lr, curr_iter)
            
            if np.random.random() >= self.args.target_sample_prob:

                x, y = sample[0]['rgb'], sample[0]['semantic']
                x = x['D'].to('cuda', dtype=torch.float32) if type(x) is dict else x.to('cuda', dtype=torch.float32)
                y = y['D'].to('cuda', dtype=torch.long) if type(y) is dict else y.to('cuda', dtype=torch.long)
                
                self.optim.zero_grad()
                
                out = self.model(x)
                if type(out) is tuple:
                    out, feats = out
                
                l = self.loss(out, y)
                
                self.writer.add_scalar('source_loss', l.item(), curr_iter)
            
                l.backward()
                
                pred = torch.argmax(out.detach(), dim=1)
                metrics.add_sample(pred, y)
                self.writer.add_scalar('step_mIoU', metrics.percent_mIoU(), curr_iter)

            else:
            
                # target losses
                try:
                    t_sample = next(it_tt)
                except:
                    it_tt = iter(self.ttloader)
                    t_sample = next(it_tt)
                
                x, y = t_sample[0]['rgb'], t_sample[0]['semantic']
                x = x['D'].to('cuda', dtype=torch.float32) if type(x) is dict else x.to('cuda', dtype=torch.float32)
                y = y['D'].to('cuda', dtype=torch.long) if type(y) is dict else y.to('cuda', dtype=torch.long)
                
                out = self.model(x)
                if type(out) is tuple:
                    out, feats = out
                    
                l = self.loss(out, y)

                self.writer.add_scalar('target_loss', l.item(), curr_iter)
            
                l.backward()
            
                pred = torch.argmax(out.detach(), dim=1)
                metrics.add_sample(pred, y)
                
                self.optim.step()
                
                if i == self.args.validate_every_steps:
                    break
                
                if curr_iter == self.args.iterations-1:
                    break
        
        self.writer.add_image("target_train_input", self.tset.to_rgb(x[0].cpu()), epoch+1, dataformats='HWC')
        self.writer.add_image("target_train_label", self.tset.color_label(y[0].cpu()), epoch+1, dataformats='HWC')
        self.writer.add_image("target_train_pred", self.tset.color_label(pred[0].cpu()), epoch+1, dataformats='HWC')
        
        miou = metrics.percent_mIoU()
        self.writer.add_scalar('train_mIoU', miou, epoch+1)
        self.logger.info("Epoch %d done, score: %.2f -- Starting Validation"%(epoch+1, miou))

if __name__ == "__main__":
    
    args = init_params('mixed')
    writer, logger = init_logger(args)
    
    set_seed(args.seed)
    
    trainer = TrainerMixed(args, writer, logger)
    trainer.train()
    
    trainer.writer.flush()
    trainer.writer.close()