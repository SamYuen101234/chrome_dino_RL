#path variables
import os
abs_path = os.path.dirname(__file__)

args = {
    "game_url": "http://www.aboutsamyuen.com/projects/chrome_dino_js/index.html",
    "chrome_driver_path": "/usr/local/bin/chromedriver",
    "loss_file_path": "./objects/loss_df.csv",
    "actions_file_path": "./objects/actions_df.csv",
    "q_value_file_path": "./objects/q_values.csv",
    "scores_file_path": "./objects/scores_df.csv",
    "train": 'train', # 'train', 'continue_train', 'test'

    #create id for canvas for faster selection from DOM
    "init_script": "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'",

    #get image from canvas
    "getbase64Script": "canvasRunner = document.getElementById('runner-canvas'); \
    return canvasRunner.toDataURL().substring(22)",

    # hypyerparameter
    "EPISODE": 20000,
    "ACTIONS": 3, # possible actions: jump, duck () and do nothing
    "GAMMA": 0.99, # EPSILON decay (encourage exploration): decay rate of past observations original 0.99
    "OBSERVATION": 100, # timesteps to observe before training
    "EXPLORE": 100000,  # frames over which to anneal epsilon
    "FINAL_EPSILON": 0.0001, # final value of epsilon
    "INITIAL_EPSILON": 0.2, # starting value of epsilon (initial randomness)
    "REPLAY_MEMORY": 50000, # number of previous transitions to remember
    "BATCH": 16, # size of minibatch
    "FRAME_PER_ACTION": 1,
    "img_rows": 80,
    "img_cols": 80,
    "img_channels": 4, #We stack 4 frames
    "lr": 1e-4,
    "weight_decay": 0,
    "dropout": 0.2,
    "model": None,   # 'resnet' for resnet 18 model or None for customized model
    "checkpoint" : "./model/model.h5",
    "SAVE_EVERY": 5000
}