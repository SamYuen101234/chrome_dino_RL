import torch.nn as nn
import torch


'''def buildmodel(img_channels, ACTIONS):
    model = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4, padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=ACTIONS)
        )
    return model'''

def buildmodel(img_channels, ACTIONS):
    model = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, ACTIONS),
        )
    
    return model
'''data = torch.rand(1,4,80,80)
model = buildmodel(4, 2)
print(model(data))'''