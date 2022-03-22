import torch.nn as nn
import torch
import copy


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


class DinosaurNet(nn.Module):
    def __init__(self, img_channels, ACTIONS):
        super().__init__()
        self.online = nn.Sequential(
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
        
        self.target = copy.deepcopy(self.online)
        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def sync_Q_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)     
'''data = torch.rand(1,4,80,80)
model = buildmodel(4, 2)
print(model(data))'''