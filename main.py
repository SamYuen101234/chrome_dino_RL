from multiprocessing.sharedctypes import Value
import os
from utils.game import Game
import datetime
import sys
import importlib
import argparse
import torch
import torch.nn as nn

from types import SimpleNamespace
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from models.model import Baseline, DoubleDQN
from models.train import trainNetwork, init_cache, load_obj

def get_dino_agent(algo):
    if algo == "Baseline":
        print("Using algorithm Baseline.")
        return Baseline
    elif algo == "DoubleDQN":
        print("Using algorithm DoubleDQN.")
        return DoubleDQN
    else:
        raise ValueError

def parse_args():
    # import config 
    # sys.path.append("config")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", "--config", help="config filename")
    parser_args, _ = parser.parse_known_args(sys.argv)
    print("Using config file", parser_args.config)
    args = importlib.import_module(parser_args.config).args
    args["experiment_name"] = parser_args.config
    args =  SimpleNamespace(**args)

    return args

# run with: python3 main.py -c config1
# turn on the cloud log: tensorboard dev upload --logdir runs
if __name__ == '__main__':
    args = parse_args()

    # create a log folder for tensorboard
    if not os.path.isdir('runs'):
        os.makedirs('runs')

    # create a folder to save the buffer, epsilon for continuous training
    if not os.path.isdir('result'):
        os.makedirs('result')

    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tb_writer = tf.summary.create_file_writer(log_dir)
    writer = SummaryWriter(comment=log_dir)
    game = Game(args.game_url, args.chrome_driver_path, args.init_script)
    DinoAgent = get_dino_agent(args.algorithm)
    # training the DQN agent
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    agent = DinoAgent(args.img_channels, args.ACTIONS, args.lr, args.BATCH, args.GAMMA, device)
    print("Device:",device)

    # criterion = nn.SmoothL1Loss() # we follow pytorch example to use smoothL1Loss not MSE
    # optimizer = optim.Adam(DQN_agent.parameters(), lr=args.lr)

    if args.train == 'train': # train a model from scratch
        init_cache(args.INITIAL_EPSILON, args.REPLAY_MEMORY) # create a pkl to save the epsilon, current step
    else: # continue training a model or ask the agent to play
        print ("Now we load weight")
        #DQN_agent.load_weights(args.checkpoint) # load the model to continue training or play
        agent = torch.load(args.checkpoint, map_location=device)
        print ("Weight load successfully")
    
    set_up = load_obj("set_up")
    step, epsilon, Deque, highest_score = set_up['step'], set_up['epsilon'], set_up['D'], set_up['highest_score']
    OBSERVE = args.OBSERVATION
    if args.train == 'test':
        epsilon = 0
        OBSERVE = float('inf')
    game.screen_shot()
    train = trainNetwork(agent, game, writer, Deque, args.BATCH, device)
    game.press_up() # start the game
    train.start(epsilon, step, highest_score, 
            OBSERVE, args.ACTIONS, args.EPSILON_DECAY, args.FINAL_EPSILON, 
            args.GAMMA, args.FRAME_PER_ACTION, args.EPISODE, 
            args.SAVE_EVERY, args.SYNC_EVERY, args.TRAIN_EVERY)

    game.end()
    print("Exit")
        