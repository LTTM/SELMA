import numpy as np
import random
import torch
torch.backends.cudnn.benchmark = True

from utils.argparser import init_params, init_logger

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

class Trainer():
    def __init__(self, args, writer, logger):
        self.args = args
        self.writer = writer
        self.logger = logger





if __name__ = "__main__":
    
    args = init_params()
    writer, logger = init_logger(args)
    
    set_seed(args.seed)
    
    trainer = Trainer(args, writer, logger)
    trainer.train()
    