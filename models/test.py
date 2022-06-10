import copy
import torch
import numpy as np
import sys
sys.path.append("../")
from tqdm import tqdm
from utils.utils import AverageMeter
from utils.game import Game
import time
import imageio
import pickle

def test_agent(agent, args, device):
    print('-------------------------------------Create a new random env for testing------------------------------------')
    game = Game(args.game_url, args.chrome_driver_path, args.init_script, args.cam_visualization) # create a random env for testing
    agent.eval()
    scores = []
    
    game.press_up() # start the game
    game.screen_shot()
    last_time = time.time()
    do_nothing = np.zeros(args.ACTIONS) # the action array, 0: do nothing, 1: jump, 2: duck
    do_nothing[0] = 1 
    x_t, r_0, terminal = game.get_state(do_nothing) # perform this action and get the next state
    # state at t
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # stack 4 images to create placeholder input
    
    s_t = s_t.reshape(1, s_t.shape[2], s_t.shape[0], s_t.shape[1])  #1*4*80*80
    s_t = torch.from_numpy(s_t).float()
    initial_state = copy.deepcopy(s_t)
    avg_fps = AverageMeter()

    with tqdm(range(args.num_test_episode), unit="episode", total=len(range(args.num_test_episode))) as tepoch:
        for episode in tepoch:
            if args.SAVE_GIF:
                images = [x_t]
                canvas_images = [game.canvas_image]

            if args.cam_visualization:
                states = [s_t]
                actions = [0]
            while not game.get_crashed():
                a_t = np.zeros([args.ACTIONS])
                action_values = agent(s_t.to(device))
                action_idx = torch.argmax(action_values).item()
                a_t[action_idx] = 1
                x_t1, r_t, terminal = game.get_state(a_t)
                x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1]) #1x1x80x80
                s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
                s_t1 = torch.from_numpy(s_t1)
                s_t = copy.deepcopy(s_t1)
                if args.SAVE_GIF:
                    images.append(x_t1[0,0])
                    canvas_images.append(game.canvas_image)
                if args.cam_visualization:
                    states.append(s_t)
                    actions.append(action_idx)
                avg_fps.update(1 / (time.time()-last_time), 1)
                last_time = time.time()
                tepoch.set_postfix(fps=avg_fps.avg)
                
                
                time.sleep(args.SLEEP) # 0.007: 50 fps for non prioritized replay buffer, 0.04: 14-17fps for prioritized replay buffer
            scores.append(game.get_score())
            s_t = copy.deepcopy(initial_state)
            if args.SAVE_GIF:
                imageio.mimsave('./img/double_dqn/dino' + str(episode) + '.gif', [np.array(img) for i, img in enumerate(images)], fps=50)
                imageio.mimsave('./img/double_dqn/dino_canvas' + str(episode) + '.gif', [np.array(img) for i, img in enumerate(canvas_images)], fps=50)
            
            if args.cam_visualization:
                with open("./test_states/dino_states" + str(episode) + ".pickle", "wb") as f:
                    pickle.dump({'states': states, 'actions': actions}, f) # save for grad_cam


            game.restart()
    game.end()
    print('Average scores in {} episodes is {:.2f}'.format(args.num_test_episode, np.array(scores).mean()))
    print('Median scores in {} episodes is {:.2f}'.format(args.num_test_episode, np.median(np.array(scores))))
    np.save('./test_scores/' + args.algorithm + '.npy', scores)
    return np.array(scores).mean(), np.median(np.array(scores))
