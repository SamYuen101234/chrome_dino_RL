from game import Game
import datetime
from conf import *
#import tensorflow as tf 
from model import buildmodel
from train import trainNetwork, init_cache, load_obj
#from tensorflow.keras.optimizers import Adam
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch import optim

# run with: python3 main.py -c config1
if __name__ == '__main__':
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tb_writer = tf.summary.create_file_writer(log_dir)
    writer = SummaryWriter(comment=log_dir)
    game = Game(args.game_url, args.chrome_driver_path, args.init_script)
    DQN_agent = buildmodel( args.img_channels, args.ACTIONS)
    # training the DQN agent
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)
    DQN_agent.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(DQN_agent.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.train == 'train': # train a model from scratch
        init_cache(args.INITIAL_EPSILON) # create a pkl to save the epsilon, current step
    else: # continue training a model or ask the agent to play
        print ("Now we load weight")
        #DQN_agent.load_weights(args.checkpoint) # load the model to continue training or play
        DQN_agent = torch.load(args.checkpoint, map=device)
        print ("Weight load successfully")
    set_up = load_obj("set_up")
    step, epsilon, Deque, highest_score = set_up['step'], set_up['epsilon'], set_up['D'], set_up['highest_score']
    OBSERVE = args.OBSERVATION
    if args.train == 'test':
        epsilon = args.FINAL_EPSILON
        OBSERVE = float('inf')
            
    game.screen_shot()
    train = trainNetwork(DQN_agent, game, optimizer, criterion, writer, device)
    train.start(epsilon, step, Deque, highest_score, 
            OBSERVE, args.ACTIONS, args.INITIAL_EPSILON, args.FINAL_EPSILON, 
            args.GAMMA, args.FRAME_PER_ACTION, args.EXPLORE, args.EPISODE, 
            args.SAVE_EVERY, args.BATCH, args.REPLAY_MEMORY) # observe = False (training)

    game.end()
    print("Exit")
        