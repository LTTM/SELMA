import sys, os
sys.path.append(os.path.abspath('.'))

import torch
torch.backends.cudnn.benchmark = True
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from itertools import chain
from matplotlib import pyplot as plt

from datasets.cityscapes import CityDataset
from utils.losses import PSNR_Like, VGG16PartialLoss
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    batch_size = 2
    
    data = CityDataset(sensors=['rgb', 'semantic'], split='train_plus', resize_to=(1024,512))
    
    writer = SummaryWriter()
    
    optim = torch.optim.Adam(chain(enc.parameters(), im_dec.parameters(), ss_dec.parameters()), lr=2.5e-4, betas=(0.5, 0.98), weight_decay=1e-5)
    mcce = nn.CrossEntropyLoss(ignore_index=-1)
    mcce.to('cuda')
    loss = nn.MSELoss()
    loss.to('cuda')
    
    for epoch in range(20):
        if epoch == 5:
            loss = PSNR_Like()
            loss.to('cuda')
        if epoch == 10:
            loss = VGG16PartialLoss()
            loss.to('cuda')
        loader = DataLoader(data, batch_size=batch_size, num_workers=8, shuffle=True)
        pbar = tqdm(loader, total=len(data)//batch_size, desc="Epoch: %d, Reconstruction Loss: 0.00, MCCE: 0.00, Progress"%(epoch+1))
        
        for i, (sample, _) in enumerate(pbar):
            im, label = sample['rgb'].to('cuda', dtype=torch.float32), sample['semantic'].to('cuda', dtype=torch.long)
            optim.zero_grad()
            
            feat = enc(im)
            out = im_dec(feat)
            seg = ss_dec(feat)
            
            l = loss(out, im)
            if type(l) is tuple:
                l, lvgg, lstyle = l
            
            ce = mcce(seg, label)
            l.backward(retain_graph=True)
            ce.backward()
            pbar.set_description("Epoch: %d, Reconstruction Loss: %.2f, MCCE; %.2f, Progress"%(epoch+1, l.item(), ce.item()))
            
            if i%50==0:
                rgb = data.to_rgb(im[0].detach().cpu())
                writer.add_image("input", rgb, len(data)*epoch+i, dataformats='HWC')
                rgb = data.to_rgb(out[0].detach().cpu())
                writer.add_image("reconstruction", rgb, len(data)*epoch+i, dataformats='HWC')
                #plt.imsave("out/e%d_i%d.png"%(epoch+1,i), rgb)
                optim.param_groups[0]["lr"] /= 1.001
                
            writer.add_scalar('lr', optim.param_groups[0]["lr"], len(data)//batch_size*epoch+i)
            writer.add_scalar('loss', l, len(data)//batch_size*epoch+i)
            writer.add_scalar('mcce', ce, len(data)//batch_size*epoch+i)
            if 'lvgg' in globals():
                writer.add_scalar('loss_vgg', lvgg, len(data)//batch_size*epoch+i)
                writer.add_scalar('loss_style', lstyle, len(data)//batch_size*epoch+i)
            
            writer.flush()
            optim.step()
        torch.save(enc.state_dict(), "enc.pth")
        torch.save(im_dec.state_dict(), "im_dec.pth")
        torch.save(ss_dec.state_dict(), "ss_dec.pth")
