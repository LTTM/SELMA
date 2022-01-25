import torch
from models.randlanet import RandLANet

def SegmentationModel(inchs, num_classes, classifier):
    if classifier.lower() == 'RandLANet'.lower():
        model = RandLANet(3, num_classes, device=torch.device('cuda'))
    else:
        ValueError("Unrecognized Classifier:"+classifier)

    model.parameters_dict = [{'params': model.parameters(), 'lr': 1}]

    return model
