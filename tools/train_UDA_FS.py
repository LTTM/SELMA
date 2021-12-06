import sys, os
sys.path.append(os.path.abspath('.'))

from tools.train_UDA import *

class TrainerUDAFS(TrainerUDA):
    def train(self):
        self.logger.info("Starting Source Only epoch")
        self.train_epoch(0)
        torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "so.pth"))
            
        miou = self.validate(0)
        self.logger.info("Validation of Source Only Epoch %.2f"%(miou))
        self.writer.add_scalar('val_mIoU', miou, 1)
        miou = self.validate_target(0)
        self.logger.info("Target Validation of Source Only Epoch %.2f"%(miou))
        
        self.logger.info("Starting Adaptation epoch")
        self.train_uda_epoch(1)
        torch.save(self.model.state_dict(), os.path.join(self.args.logdir, "uda.pth"))
            
        miou = self.validate(1)
        self.logger.info("Validation of Adaptation Epoch %.2f"%(miou))
        self.writer.add_scalar('val_mIoU', miou, 2)
        miou = self.validate_target(1)
        self.logger.info("Target Validation of Adaptation Epoch %.2f"%(miou))


if __name__ == "__main__":
    
    args = init_params('uda_fs')
    writer, logger = init_logger(args)
    
    set_seed(args.seed)
    
    trainer = TrainerUDAFS(args, writer, logger)
    trainer.train()
    
    trainer.writer.flush()
    trainer.writer.close()