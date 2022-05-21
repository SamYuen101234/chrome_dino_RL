import numpy as np
import random
from collections import deque # double ended queue (can append and pop in both end)
import torch
from operator import itemgetter 

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
    
    def recall(self, batch_size):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.buffer, batch_size)
        state, action_idx, reward, next_state, terminal = map(torch.stack, zip(*batch))
        state = torch.squeeze(state)
        next_state = torch.squeeze(next_state)
        reward = torch.squeeze(reward)
        terminal = torch.squeeze(terminal)
        return state, action_idx, reward, next_state, terminal

    def append(self, experience):
        self.buffer.append(experience)

class PrioritizedReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        
    def append(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))
        
    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities
    
    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized
        
    def recall(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        batch = itemgetter(*sample_indices)(self.buffer)
        state, action_idx, reward, next_state, terminal = map(torch.stack, zip(*batch))
        state, next_state, reward, terminal= torch.squeeze(state), torch.squeeze(next_state),\
            torch.squeeze(reward), torch.squeeze(terminal)

        importance = torch.from_numpy(self.get_importance(sample_probs[sample_indices]))
        return state, action_idx, reward, next_state, terminal, importance, sample_indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset