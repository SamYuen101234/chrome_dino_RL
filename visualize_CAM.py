from Grad_CAM.grad_cam import CAM
import pickle
import mgzip
from tqdm import tqdm
import torch
import argparse
from types import SimpleNamespace
import importlib
import sys
import imageio
import numpy as np

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

if __name__=="__main__":
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agent = torch.load(args.checkpoint, map_location=device)
    with open('./test_states/dino_states7.pickle', 'rb') as f:
        pkl = pickle.load(f)
        states, actions = pkl['states'], pkl['actions']
    heat_maps = []
    for state, action in tqdm(zip(states, states), total=len(states)):
        heat_maps.append(CAM(agent, state, action, use_cuda=True))
    imageio.mimsave('./img/double_dqn/dino_grad_cam.gif', [np.array(img) for i, img in enumerate(heat_maps)], fps=30)
