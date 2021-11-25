import torch
from models.resnet import DeeplabResnet, Bottleneck
from models.deeplab import DeepLabV2Classifier, DeepLabV3Classifier, MSIWDeepLabV2Classifier
from models.fcn import FCNClassifier
from models.pspnet import PSPNetClassifier
from models.unet import UNet

def SegmentationModel(num_classes, classifier, pretrained=True):
    if classifier.lower() == 'DeepLabV2'.lower():
        clas = DeepLabV2Classifier
    elif classifier.lower() == 'DeepLabV2MSIW'.lower():
        clas = MSIWDeepLabV2Classifier
    elif classifier.lower() == 'DeepLabV3'.lower():
        clas = DeepLabV3Classifier
    elif classifier.lower() == 'FCN'.lower():
        clas = FCNClassifier
    elif classifier.lower() == 'PSPNet'.lower():
        clas = PSPNetClassifier
    elif classifier.lower() == 'UNet'.lower():
        model = UNet(3, num_classes)
    else:
        ValueError("Unrecognized Classifier:"+classifier)

    if not classifier.lower() == 'UNet'.lower():
        model = DeeplabResnet(Bottleneck, [3, 4, 23, 3], num_classes, clas)
        if pretrained:
            restore_from = './models/backbone_checkpoints/resnet101-5d3b4d8f.pth'
            saved_state_dict = torch.load(restore_from)

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5' and not i_parts[0] == 'fc':
                    new_params[i] = saved_state_dict[i]
            model.load_state_dict(new_params)

        # add backbone to model for later use
        model.parameters_dict = [{'params': model.get_1x_lr_params_NOscale(), 'lr': 1},
                                 {'params': model.get_10x_lr_params(), 'lr': 10 * 1}]
     else:
        model.parameters_dict = [{'params': model.parameters(), 'lr': 1}]

    return model
