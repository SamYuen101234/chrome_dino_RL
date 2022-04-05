from turtle import forward
import torch.nn as nn
import torch
import copy

import numpy as np

class ChromeDinoAgent(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def sync_target(self):
        pass

    def forward(self, input, model):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError

    def step(self, state_t, action_t, reward_t, state_t1, terminal, GAMMA):
        raise NotImplementedError
        
class Baseline(ChromeDinoAgent):

    def __init__(self, img_channels, ACTIONS):
        super().__init__()
        
        self.model = nn.Sequential(
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

    def forward(self, x, model):
        return self.model(x)

    def get_action(self, state):
        with torch.no_grad():
            action_values = self.model(state.to(self.device)) # input a stack of 4 images, get the prediction
        action_idx = torch.argmax(action_values)

        return action_idx

    def step(self, state_t, action_t, reward_t, state_t1, terminal, GAMMA):
        td_estimate = self.model(state_t.to(self.device))
        with torch.no_grad():
            td_target = self.model(state_t.to(self.device))
            next_Q = self.model(state_t1.to(self.device)) # Q_sa
            td_target[torch.arange(len(td_target)), action_t] = \
                    (reward_t + (1 - terminal.float())*GAMMA *torch.amax(next_Q, axis=1)).float() # put a mask on the action_t

        return td_estimate, td_target, next_Q

class DoubleDQN(ChromeDinoAgent):
    def __init__(self, img_channels, ACTIONS):
        super().__init__()
        '''self.online = nn.Sequential(
                nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2304, 512),
                #nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(512, ACTIONS),
            )'''
        self.online = nn.Sequential(
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
        
        self.target = copy.deepcopy(self.online)
        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def forward(self, x, model):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)

    def get_action(self, state):
        with torch.no_grad():
            action_values = self.online(state.to(self.device)) # input a stack of 4 images, get the prediction
        action_idx = torch.argmax(action_values).item()

        return action_idx

    def step(self, state_t, action_t, reward_t, state_t1, terminal, GAMMA):
        # td_estimate/current Q
        td_estimate = self.model(state_t.to(self.device), model='online')[
            np.arange(0, self.batch_size), action_t
        ]  # Q_online(s,a)

        # td_target
        with torch.no_grad():
            next_state_Q = self.model(state_t1.to(self.device), model='online')
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.model(state_t1, model="target")[
                np.arange(0, self.batch_size), best_action
            ]
            td_target = (reward_t + (1 - terminal.float())*GAMMA *next_Q).float()

        return td_estimate, td_target, next_Q