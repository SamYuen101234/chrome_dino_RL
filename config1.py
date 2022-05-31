#path variables
import os
abs_path = os.path.dirname(__file__)

args = {

    # env setting
    "game_url": "http://www.aboutsamyuen.com/projects/chrome_dino_js/index.html",
    "chrome_driver_path": "/usr/lib/chromium-browser/chromedriver",
    "train": 'test', # 'train', 'test'
    "init_script": "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'", #create id for canvas for faster selection from DOM
    "getbase64Script": "canvasRunner = document.getElementById('runner-canvas'); \
    return canvasRunner.toDataURL().substring(22)", #get image from canvas
    "img_rows": 80,
    "img_cols": 80,
    "img_channels": 4, #We stack 4 frames
    "checkpoint" : "./weights/double_dqn.pth",
    "SAVE_EVERY": 100000,
    "num_test_episode": 20,
    "cam_visualization": False, # real-time Grad CAM visualization (XAI)
    "TEST_EVERY": 50,

    # hypyerparameter
    "algorithm": "DoubleDQN",
    "EPISODE": 5000,
    "ACTIONS": 2, # possible actions: jump, duck () and do nothing
    "GAMMA": 0.99, # decay rate of past observations original 0.99
    "OBSERVATION": 2000, # timesteps to observe before training
    "FINAL_EPSILON": 1e-1, # final value of epsilon
    "INITIAL_EPSILON": 1, # starting value of epsilon (initial randomness)
    "EPSILON_DECAY": 0.9999925, #0.999975
    "REPLAY_MEMORY": 100000, # number of previous transitions to remember
    "BATCH": 32, # size of minibatch
    "FRAME_PER_ACTION": 1,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "SYNC_EVERY": 1e4,   # no. of experiences between Q_target & Q_online sync
    "TRAIN_EVERY": 3,
    "prioritized_replay": False, # FPS is slower than unprioritized and needs larger RAM (>40GB) otherwise killed
    "grad_norm_clipping": 10, # prevent gradien explosion





    
}