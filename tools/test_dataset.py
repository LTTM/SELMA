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

    for sample, imid in dset:
        rgb, gt = dset.to_rgb(sample['rgb']), dset.color_label(sample['semantic'])
        
        plt.imshow(np.uint8(.5*rgb+.5*gt))
        plt.title(imid)
        plt.show()
        
        # fig, axs = plt.subplots(2,1)
        # axs[0].imshow(rgb)
        # axs[1].imshow(gt)
        # fig.suptitle(imid)
        # plt.show()