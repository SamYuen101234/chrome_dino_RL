#path variables
import os
abs_path = os.path.dirname(__file__)

args = {
    "game_url": "http://www.aboutsamyuen.com/projects/chrome_dino_js/index.html",
    "chrome_driver_path": "/usr/lib/chromium-browser/chromedriver",
    "train": 'train', # 'train', 'continue_train', 'test'

    #create id for canvas for faster selection from DOM
    "init_script": "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'",

    #get image from canvas
    "getbase64Script": "canvasRunner = document.getElementById('runner-canvas'); \
    return canvasRunner.toDataURL().substring(22)",

    # hypyerparameter
    "EPISODE": 50000,
    "ACTIONS": 2, # possible actions: jump, duck () and do nothing
    "GAMMA": 0.99, # decay rate of past observations original 0.99
    "OBSERVATION": 100, # timesteps to observe before training
    "EXPLORE": 100000,  # frames over which to anneal epsilon
    "FINAL_EPSILON": 0.0001, # final value of epsilon
    "INITIAL_EPSILON": 0.1, # starting value of epsilon (initial randomness)
    "REPLAY_MEMORY": 10000, # number of previous transitions to remember
    "BATCH": 16, # size of minibatch
    "FRAME_PER_ACTION": 1,
    "img_rows": 80,
    "img_cols": 80,
    "img_channels": 4, #We stack 4 frames
    "lr": 1e-4,
    "weight_decay": 0,
    "dropout": 0.2,
    "model": None,   # 'resnet' for resnet 18 model or None for customized model
    "checkpoint" : "./model/model.pth",
    "SAVE_EVERY": 100000
}