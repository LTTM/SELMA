import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
from matplotlib import pyplot as plt

from utils.argparser import init_params

if __name__ == '__main__':

    args = init_params('test')
    
    dset = args.dataset(root_path=args.root_path,
                        splits_path=args.splits_path,
                        split=args.test_split,
                        resize_to=args.rescale_size,
                        crop_to=None,
                        augment_data=False,
                        sensors=args.sensors,
                        town=args.town,
                        weather=args.weather,
                        time_of_day=args.time_of_day,
                        sensors_positions=args.positions,
                        class_set=args.class_set)

    for data in dset:
        
        sample, imid = data if type(data) is tuple else data, 0
        x, y = sample[0]['rgb'], sample[0]['semantic']
        x = x['D'] if type(x) is dict else x
        y = y['D'] if type(y) is dict else y
        
        rgb, gt = dset.to_rgb(x), dset.color_label(y)
        
        plt.imshow(np.uint8(.5*rgb+.5*gt))
        plt.title(imid)
        plt.show()