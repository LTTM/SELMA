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

class MSIWEX(nn.Module):
    def __init__(self, weights=None, alpha=.2, beta=.2, ignore_index=-1):
        super(MSIWEX, self).__init__()
        self.alpha = alpha
        self.beta = beta
        assert beta == 0 or weights is not None, "Weights must be provided when beta > 0."
        assert alpha+beta <= 1, "The sum of coefficients Alpha and Beta must not exceed 1"
        self.weights = weights if beta > 0 else torch.tensor(1.)
        self.ignore_index = ignore_index
    
    def to(self, *args, **kwargs):
        super(MSIWEX, self).to(*args, **kwargs)
        self.weights = self.weights.to(*args, **kwargs)
    
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
        weights = torch.pow(self.weights*H*W, self.beta)
        den = torch.clamp(torch.pow(hist, self.alpha)*weights*torch.pow(Np, 1-(self.alpha+self.beta)), min=1.)[pred] # <- cast to Nx1xHxW
        # compute the loss
        return -torch.sum(mask*torch.pow(prob, 2)/den)/(N*C)