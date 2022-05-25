import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
from Grad_CAM.grad_cam import CAM
from tqdm import tqdm
from utils.utils import AverageMeter
import time

def test_agent(agent, game, ACTIONS, device, episodes=50):
    agent.eval()
    last_time = time.time()
    scores = []

    do_nothing = np.zeros(ACTIONS) # the action array, 0: do nothing, 1: jump, 2: duck
    do_nothing[0] = 1 
    x_t, r_0, terminal = game.get_state(do_nothing) # perform this action and get the next state
    # state at t
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # stack 4 images to create placeholder input
    
    s_t = s_t.reshape(1, s_t.shape[2], s_t.shape[0], s_t.shape[1])  #1*4*80*80
    s_t = torch.from_numpy(s_t).float()
    initial_state = copy.deepcopy(s_t)
    avg_fps = AverageMeter()
    with tqdm(range(episodes), unit="episode", total=len(range(episodes))) as tepoch:
        for episode in tepoch:
            while not game.get_crashed():
                a_t = np.zeros([ACTIONS])
                action_values = agent(s_t.to(device))
                action_idx = torch.argmax(action_values).item()
                a_t[action_idx] = 1
                x_t1, r_t, terminal = game.get_state(a_t)
                x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1]) #1x1x80x80
                s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
                s_t1 = torch.from_numpy(s_t1)
                s_t = copy.deepcopy(s_t1)
                avg_fps.update(1 / (time.time()-last_time), 1)
                last_time = time.time()
                tepoch.set_postfix(fps=avg_fps.avg)
                time.sleep(0.04)
            scores.append(game.get_score())
            s_t = copy.deepcopy(initial_state)
            game.restart()

    print('Average scores in {} episodes is {:.2f}'.format(episodes, np.array(scores).mean()))
    print('Median scores in {} episodes is {:.2f}'.format(episodes, np.median(np.array(scores))))
    return np.array(scores).mean(), np.median(np.array(scores))
