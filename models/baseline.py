import torch.nn as nn
import torch

from torch import optim

from models.agent import ChromeDinoAgent

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