import torch
from torch import nn

class MSIW(nn.Module):
    def __init__(self, ratio=.2, ignore_index=-1):
        super(MSIW, self).__init__()
        self.iw = ratio
        self.ignore_index = ignore_index
        
    def forward(self, nw_out, label=None):
        # extract dimensions
        N, C, H, W = nw_out.shape
        # compute probabilities and predicted segmentation map
        prob = torch.softmax(nw_out, dim=1)
        pred = torch.argmax(prob.detach(), dim=1, keepdim=True) if label is None else label.unsqueeze(1) # <- argmax, shape N x 1 x H x W
        mask = (pred != self.ignore_index).to(dtype=torch.float32)
        # compute the predicted class frequencies
        hist = torch.histc(pred, bins=C, min=0, max=C-1) # 1-dimensional vector of length C, skips -1s
        Np = hist.sum()
        # compute class weights array
        den = torch.clamp(torch.pow(hist, self.iw)*torch.pow(Np, 1-self.iw), min=1.)[pred] # <- cast to Nx1xHxW
        # compute the loss
        return -torch.sum(mask*torch.pow(prob, 2)/den)/(N*C)