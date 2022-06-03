
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import imageio

def CAM(agent, state, action, use_cuda=True):
    target_layers = list(agent.children())[:9] # retrieve the bottleneck
    target_layers = torch.nn.Sequential(*target_layers)
    
    # Construct the CAM object once, and then re-use it on many images:
    cam = EigenCAM(model=agent, target_layers=target_layers, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(action)]
    grayscale_cam = cam(input_tensor=state, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image((state[0,0,:,:].unsqueeze(2).cpu().numpy()/255).astype('float32'), grayscale_cam, use_rgb=True)
    #Image.fromarray(visualization)

    return visualization