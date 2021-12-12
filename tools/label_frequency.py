import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
import random

import torch
torch.backends.cudnn.benchmark = True
from torch.utils import data
from tqdm import tqdm

from utils.argparser import init_params

if __name__ == "__main__":
    
    args = init_params('test')
    
    tset = args.dataset(root_path=args.root_path,
                        splits_path=args.splits_path,
                        split=args.train_split,
                        resize_to=args.rescale_size,
                        crop_to=args.crop_size if args.crop_images else None,
                        augment_data=False,
                        sensors=['semantic'],
                        town=args.town,
                        weather=args.weather,
                        time_of_day=args.time_of_day,
                        sensors_positions=args.positions,
                        class_set=args.class_set)
    tloader = data.DataLoader(tset,
                              shuffle=False,
                              num_workers=args.dataloader_workers,
                              batch_size=args.batch_size,
                              drop_last=True,
                              pin_memory=args.pin_memory)
                              
    num_classes = len(tset.cnames)
    
    counts = torch.zeros(num_classes+1, dtype=torch.long, device='cuda')
    
    for sample in tqdm(tloader, total=len(tset)):
        y = sample[0]['semantic']
        y = y['D'].to('cuda', dtype=torch.long) if type(y) is dict else y.to('cuda', dtype=torch.long)
        y = y.flatten() + 1 # -1:num_classes-1 -> 0:num_classes
        counts += torch.bincount(y, minlength=num_classes+1)
        
    print(counts)
    np.save("datasets/frequencies/%s_%s.npy"%(args.dataset.__name__[:-7], args.class_set),
            np.roll(counts.detach().cpu().numpy(), -1)) # shift indexes back to -1:num_classes-1
