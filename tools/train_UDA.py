import sys, os
sys.path.append(os.path.abspath('.'))

from tools.train import *

class TrainerUDA(Trainer):
    def __init__(self, args, writer, logger):
        super(TrainerUDA, self).__init__(args, writer, logger)
        
        # add uda thingies

if __name__ == "__main__":
    
    args = init_params()
    writer, logger = init_logger(args)
    
    set_seed(args.seed)
    
    trainer = TrainerUDA(args, writer, logger)
    trainer.train()
    
    trainer.writer.flush()
    trainer.writer.close()