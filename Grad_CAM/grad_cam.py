
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

def CAM(agent, state, action, use_cuda=True):
    target_layers = [agent.last_layer()] # retrieve the last convolutional layer

    # Construct the CAM object once, and then re-use it on many images:
    cam = EigenCAM(model=agent, target_layers=target_layers, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(action)]
    grayscale_cam = cam(input_tensor=state, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(state, grayscale_cam, use_rgb=True)

    return grayscale_cam