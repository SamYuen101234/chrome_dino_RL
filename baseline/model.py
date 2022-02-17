
from distutils.command.build import build
import torch.nn as nn
import torchvision
import timm
import torch
from typing import Tuple, Optional

import torch.nn.functional as F

def buildmodel(img_channels, ACTIONS):
    model = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4, padding=1),
            nn.MaxPool2d(2,2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2,2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(2,2),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.Dropout(0.1),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=512, out_features=ACTIONS)
        )
    #model = timm.create_model('efficientnet_b3',pretrained=True)
    return model

#print(buildmodel(4, 2))