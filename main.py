from game import Game
import datetime
from conf import *
import tensorflow as tf 
from model import buildmodel
from train import train, init_cache, load_obj
from tensorflow.keras.optimizers import Adam
#from tensorflow_addons.optimizers import AdamW

# run with: python3 main.py -c config1
if __name__ == '__main__':
    '''if tf.test.gpu_device_name(): # check GPU with tf
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")'''
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_writer = tf.summary.create_file_writer(log_dir)
    game = Game(args.game_url, args.chrome_driver_path, args.init_script, tb_writer)
    DQN_agent = buildmodel(args.img_cols, args.img_rows, 
                       args.img_channels, args.ACTIONS, 
                       args.lr, args.checkpoint)
    #DQN_agent = DinoAgent(game)
    # training the DQN agent

    if args.train == 'train': # train a model from scratch
        init_cache(args.INITIAL_EPSILON) # create a pkl to save the epsilon, current step
    else: # continue training a model or ask the agent to play
        print ("Now we load weight")
        DQN_agent.load_weights("model.h5") # load the model to continue training or play
        print ("Weight load successfully")
    set_up = load_obj("set_up")
    step, epsilon, Deque, highest_score = set_up['step'], set_up['epsilon'], set_up['D'], set_up['highest_score']
    OBSERVE = args.OBSERVATION
    if args.train == 'test':
        epsilon = args.FINAL_EPSILON
        OBSERVE = float('inf')
            
    adam = Adam(learning_rate=args.lr)
    DQN_agent.compile(loss='mse',optimizer=adam)
    game.screen_shot()
    train(DQN_agent, game, epsilon, step, Deque, highest_score, 
            OBSERVE, args.ACTIONS, args.INITIAL_EPSILON, args.FINAL_EPSILON, 
            args.GAMMA, args.FRAME_PER_ACTION, args.EXPLORE, args.EPISODE, 
            args.SAVE_EVERY, args.BATCH, args.REPLAY_MEMORY) # observe = False (training)

    game.end()
    print("Exit")
        