from turtle import forward
import torch.nn as nn
import torch
import copy

import numpy as np

from torch import optim

class ChromeDinoAgent:
    def __init__(self, img_channels, ACTIONS, lr, batch_size, gamma, device):
        # prototype constructor
        self.img_channels = img_channels
        self.num_actions = ACTIONS
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

    def sync_target(self):
        pass

    def save_model(self):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError

    def step(self, state_t, action_t, reward_t, state_t1, terminal):
        raise NotImplementedError
        
class Baseline(ChromeDinoAgent):

    def __init__(self, img_channels, ACTIONS, lr, batch_size, gamma, device):
        super().__init__(img_channels, ACTIONS, lr, batch_size, gamma, device)
        
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def get_action(self, state):
        with torch.no_grad():
            action_values = self.model(state.to(self.device)) # input a stack of 4 images, get the prediction
        action_idx = torch.argmax(action_values)

        return action_idx

    def save_model(self):
        torch.save(self.model, "./weights/baseline.pth")

    def step(self, state_t, action_t, reward_t, state_t1, terminal):
        td_estimate = self.model(state_t.to(self.device))
        with torch.no_grad():
            td_target = self.model(state_t.to(self.device))
            next_Q = self.model(state_t1.to(self.device)) # Q_sa
            td_target[torch.arange(len(td_target)), action_t] = \
                    (reward_t + (1 - terminal.float())*self.gamma *torch.amax(next_Q, axis=1)).float() # put a mask on the action_t

        loss = self.criterion(td_estimate.to(self.device), td_target.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        avg_loss = loss.detach().cpu().item()
        avg_q_max = torch.mean(torch.amax(next_Q)).detach().cpu().item()

        return avg_loss, avg_q_max

class DoubleDQN(ChromeDinoAgent):
    def __init__(self, img_channels, ACTIONS, lr, batch_size, gamma, device):
        super().__init__(img_channels, ACTIONS, lr, batch_size, gamma, device)

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

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save_model(self):
        torch.save(self.model, "./weights/double_dqn.pth")

    def get_action(self, state):
        with torch.no_grad():
            action_values = self.online(state.to(self.device)) # input a stack of 4 images, get the prediction
        action_idx = torch.argmax(action_values).item()

        return action_idx

    def compute_loss(self, state_t, action_t, reward_t, state_t1, terminal):
        td_estimate = self.online(state_t.to(self.device))[
            np.arange(0, self.batch_size), action_t
        ]  # Q_online(s,a)

        # td_target
        with torch.no_grad():
            next_state_Q = self.online(state_t1.to(self.device))
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.target(state_t1)[
                np.arange(0, self.batch_size), best_action
            ]
            td_target = (reward_t + (1 - terminal.float())* self.gamma *next_Q).float()
        
        return self.criterion(td_estimate.to(self.device), td_target.to(self.device))

    def step(self, state_t, action_t, reward_t, state_t1, terminal):
        # td_estimate/current Q
        # print("Inside double dqn step, batch size: ", self.batch_size)
        td_estimate = self.online(state_t.to(self.device))[
            np.arange(0, self.batch_size), action_t
        ]  # Q_online(s,a)

        # td_target
        with torch.no_grad():
            next_state_Q = self.online(state_t1.to(self.device))
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.target(state_t1)[
                np.arange(0, self.batch_size), best_action
            ]
            td_target = (reward_t + (1 - terminal.float())* self.gamma *next_Q).float()

        loss = self.criterion(td_estimate.to(self.device), td_target.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        avg_loss = loss.detach().cpu().item()
        avg_q_max = torch.mean(torch.amax(next_Q)).detach().cpu().item()

        return avg_loss, avg_q_max