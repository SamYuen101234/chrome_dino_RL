#from turtle import forward
from multiprocessing import dummy
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
        ''' A function for double DQN'''
        pass

    def save_model(self):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError

    def compute_loss(self, state_t, action_t, reward_t, state_t1, terminal):
        pass

    def step(self, state_t, action_t, reward_t, state_t1, terminal):
        raise NotImplementedError

    def last_layer(self):
        ''' A function to retrieve the last convolutional layer of the deep model for Grad CAM visualization'''
        raise NotImplementedError

    def gradient_2norm(self):
        ''' 
            return the 2 norm of gradient vector
            use this function to check the norm and fine tune the max norm of norm clipping
        '''

        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm
    


# Deep Q-Network
class Baseline(ChromeDinoAgent):

    def __init__(self, img_channels, ACTIONS, lr, weight_decay, batch_size, gamma, device, grad_norm_clipping):
        super().__init__(img_channels, ACTIONS, lr, batch_size, gamma, device)
        
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(2304, 512),
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),
                nn.Linear(512, ACTIONS),
            )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = nn.SmoothL1Loss()
        self.grad_norm_clipping = grad_norm_clipping

    def get_action(self, state):
        with torch.no_grad():
            action_values = self.model(state.to(self.device)) # input a stack of 4 images, get the prediction
        action_idx = torch.argmax(action_values)

        return action_idx

    def save_model(self):
        torch.save(self.model, "./weights/baseline.pth")

    def compute_loss(self, state_t, action_t, reward_t, state_t1, terminal):
        pass

    def step(self, state_t, action_t, reward_t, state_t1, terminal, importance=None):
        td_estimate = self.model(state_t.to(self.device))
        with torch.no_grad():
            td_target = self.model(state_t.to(self.device))
            next_Q = self.model(state_t1.to(self.device)) # Q_sa
            td_target[torch.arange(len(td_target)), action_t] = \
                    (reward_t + (1 - terminal.float())*self.gamma *torch.amax(next_Q, axis=1)).float() # put a mask on the action_t

        loss = self.criterion(td_estimate.to(self.device), td_target.to(self.device))

        # weighted loss, if we use prioritized reply buffer
        if importance is not None: loss = (torch.FloatTensor(importance)*loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        avg_loss = loss.detach().cpu().item()
        avg_q_max = torch.mean(torch.amax(next_Q)).detach().cpu().item()

        return avg_loss, avg_q_max

    def last_layer(self):
        pass

# Double Deep Q-Network
class DoubleDQN(ChromeDinoAgent):
    def __init__(self, img_channels, ACTIONS, lr, weight_decay, batch_size, gamma, device, grad_norm_clipping=10):
        super().__init__(img_channels, ACTIONS, lr, batch_size, gamma, device)

        self.online = nn.Sequential(
                nn.BatchNorm2d(img_channels),
                nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(2304, 512),
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),
                nn.Linear(512, ACTIONS),
            )
        
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = nn.SmoothL1Loss()
        self.grad_norm_clipping = grad_norm_clipping

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save_model(self):
        torch.save(self.online, "./weights/double_dqn.pth")

    def get_action(self, state):
        with torch.no_grad():
            action_values = self.online(state.to(self.device)) # input a stack of 4 images, get the prediction
        action_idx = torch.argmax(action_values).item()

        return action_idx

    def compute_loss(self, state_t, action_t, reward_t, state_t1, terminal):
        self.online.train()

        td_estimate = self.online(state_t.to(self.device))[
            np.arange(0, self.batch_size), action_t
        ]  # Q_online(s,a)

        self.online.eval()
        self.target.eval()

        # td_target
        with torch.no_grad():
            next_state_Q = self.online(state_t1.to(self.device))
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.target(state_t1.to(self.device))[
                np.arange(0, self.batch_size), best_action
            ]
            td_target = (reward_t + (1 - terminal.float())* self.gamma *next_Q).float()
        
        return self.criterion(td_estimate.to(self.device), td_target.to(self.device)), next_Q, torch.abs(td_estimate - td_target).cpu().data.numpy()

    def step(self, state_t, action_t, reward_t, state_t1, terminal, importance=None):
        # error: used for prioritized replay buffer
        loss, next_Q, error = self.compute_loss(state_t, action_t, reward_t, state_t1, terminal)

        # weighted loss, if we use prioritized reply buffer
        if importance is not None: loss = (importance*loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.grad_norm_clipping)
        self.optimizer.step()

        avg_loss = loss.detach().cpu().item()
        avg_q_max = torch.mean(torch.amax(next_Q)).detach().cpu().item()

        return avg_loss, avg_q_max, error

    def last_layer(self):
        pass
        
