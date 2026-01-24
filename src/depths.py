"""
Layer-enumeration utilities for intrinsic-dimension computation

Depth extraction utilities (getDepths, getResNetsDepths, getDepths_cifar_resnet)

Traverse model layers and record a small set of “checkpoints” (input, pooling / residual blocks, classifier) together 
with a monotonically increasing depth index.

These depth indices are later used to plot intrinsic dimension vs. relative depth.

"""


import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

def getDepths(model):
    count = 0
    modules = []
    names = []
    depths = []
    modules.append('input')
    names.append('input')
    depths.append(0)

    for i,module in enumerate(model.features):
        name = module.__class__.__name__
        if 'Conv2d' in name or 'Linear' in name:
            count += 1
        if 'MaxPool2d' in name:
            modules.append(module)
            depths.append(count)
            names.append('MaxPool2d')

    clf = model.classifier
    if isinstance(clf, nn.Sequential):
        classifier_layers = clf
    else:
        classifier_layers = [clf]

    for i,module in enumerate(classifier_layers):
        name = module.__class__.__name__
        if 'Linear' in name:
            modules.append(module)
            count += 1
            depths.append(count + 1)
            names.append('Linear')
    depths = np.array(depths)
    return modules, names, depths




def getLayerDepth(layer):
    count = 0
    for m in layer:
        for c in m.children():
            name = c.__class__.__name__
            if 'Conv' in name:
                count += 1
    return count

def getResNetsDepths(model):
    modules = []
    names = []
    depths = []

    # input
    count = 0
    modules.append('input')
    names.append('input')
    depths.append(count)
    # maxpooling
    count += 1
    modules.append(model.maxpool)
    names.append('maxpool')
    depths.append(count)
    # 1
    count += getLayerDepth(model.layer1)
    modules.append(model.layer1)
    names.append('layer1')
    depths.append(count)
    # 2
    count += getLayerDepth(model.layer2)
    modules.append(model.layer2)
    names.append('layer2')
    depths.append(count)
    # 3
    count += getLayerDepth(model.layer3)
    modules.append(model.layer3)
    names.append('layer3')
    depths.append(count)
    # 4
    count += getLayerDepth(model.layer4)
    modules.append(model.layer4)
    names.append('layer4')
    depths.append(count)
    # average pooling
    count += 1
    modules.append(model.avgpool)
    names.append('avgpool')
    depths.append(count)
    # output
    count += 1
    modules.append(model.fc)
    names.append('fc')
    depths.append(count)
    depths = np.array(depths)
    return modules, names, depths


def getDepths_cifar_resnet(model):
    modules = []
    names = []
    depths = []

    # input
    count = 0
    modules.append("input")
    names.append("input")
    depths.append(count)

    # conv1
    count += 1
    modules.append(model.conv1)
    names.append("conv1")
    depths.append(count)

    # layer1~4
    for layer, lname in [
        (model.layer1, "layer1"),
        (model.layer2, "layer2"),
        (model.layer3, "layer3"),
        (model.layer4, "layer4"),
    ]:
        count += getLayerDepth(layer)
        modules.append(layer)
        names.append(lname)
        depths.append(count)


    count += 1
    modules.append(model.feature_identity)
    names.append("feature_identity")
    depths.append(count)

    count += 1
    modules.append(model.linear)
    names.append("linear")
    depths.append(count)

    depths = np.array(depths)
    return modules, names, depths

class ImagesOnlyDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img

