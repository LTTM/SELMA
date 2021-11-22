import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class ResnetEmbeed(nn.Module):
    def __init__(self):
        super(ResnetEmbeed, self).__init__()
        rn101 = models.resnet101()
        rn101.load_state_dict(torch.load('models/resnet101-5d3b4d8f.pth'))
        self.model = nn.Sequential(*([k for k in rn101.children()][:-1])) # create sequential model from all children
                                                                          # except the fully connected layer
        self.model.eval() # initialize the model in eval mode
    
    def forward(self, x):
        return self.model(x)