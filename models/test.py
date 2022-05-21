import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
from Grad_CAM.grad_cam import CAM


def test_agent(agents, game, ACTIONS, episodes=50):
    scores = []

    do_nothing = np.zeros(ACTIONS) # the action array, 0: do nothing, 1: jump, 2: duck
    do_nothing[0] = 1 
    num_action_0 += 1
    x_t, r_0, terminal = game.get_state(do_nothing) # perform this action and get the next state
    # state at t
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # stack 4 images to create placeholder input
    
    s_t = s_t.reshape(1, s_t.shape[2], s_t.shape[0], s_t.shape[1])  #1*4*80*80
    s_t = torch.from_numpy(s_t).float()
    initial_state = copy.deepcopy(s_t)

    for episode in range(episodes):
        while not game.get_crashed():
            pass

        scores.append(game.get_score())
        game.restart()

    print('-------------------------------------Finish Testing-------------------------------------')
    plt.boxplot(scores)
    plt.savefig('test_scores.png')
    plt.clf()
    print('Average scores in {} is {}'.format(episodes, np.array(scores).mean()))

