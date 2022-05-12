import sys, os
sys.path.append(os.path.abspath('.'))

from matplotlib import pyplot as plt
from datasets.selma_fda import *
import torch

a = SELMA_FDA()
rgb = a[0]

plt.imshow(a.selma.to_rgb(rgb))
plt.show()

# rgb_s, rgb_c, (ffss_m, ffss_a), (ffsc_m, ffsc_a) = a[0]

# ffss, ffsc = torch.fft.fftshift(ffss_m), torch.fft.fftshift(ffsc_m)
# ffss, ffsc = torch.log(ffss+1.), torch.log(ffsc+1.)
# ffss, ffsc = ffss/ffss.max(), ffsc/ffsc.max()

# ffss = a.selma.to_rgb(2*(ffss-.5))
# ffsc = a.selma.to_rgb(2*(ffsc-.5))

# fig, axs = plt.subplots(2,2)
# axs[0,0].imshow(a.selma.to_rgb(rgb_s))
# axs[0,1].imshow(ffss)
# axs[1,0].imshow(a.selma.to_rgb(rgb_c))
# axs[1,1].imshow(ffsc)
# plt.show()
#plt.imshow(ffss)
#plt.show()