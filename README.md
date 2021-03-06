# Tasks to be implemented

- [x] Deep Q Network Agent
- [x] Double Deep Q Network Agent
- [x] Prioritized Experience Replay
- [x] Grad-Cam Visualization (Double DQN)
- [ ] Rainbox
- [ ] Policy Gradient
- [ ] Actor-Critic Algorithms
- [ ] Deployment (onnx, opencv4nodejs, nodejs, target FPS for the agent in browser: 50 fps)

<img src="./img/double_dqn/dino8.gif" width="250" height="250" /><img src="./img/double_dqn/dino_canvas2.gif" width="600" height="150" />

Testing in 50 FPS. A higher scores' version in [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zyuen_connect_ust_hk/Ej2-jJx7EKNCsIJdjTESonEBtwS6whKM7QThAW7I9VKlKA?e=eJKdcK).

<img src="./img/double_dqn/dino_grad_cam.gif" width="250" height="250" />

Using Grad_CAM, an explainable/interpretable AI approach for deep learning, to examine whether the agent treats the game as human. See [Grad_CAM Visualization](#user-content-grad_cam-visualization-double-dqn) for details.

# Quick Start
1. Create two directories manually or created by main.py automatically
```
mkdir result
mkdir weights
```
2. Learning or Testing
```py
python3 main.py -c config1
```

# The most detailed experiments and explaination of Chrome dinosaur in Deep Reinforcement Learning on GitHub

Chome dinosaur is a game very suitable for beginners in deep reinforcement learning because of its easy rules and environment setting. Although the game is easy for human but it is difficult for computer agent to learning it. Through this project, We will not only show the result of baseline DQN, but also compare its results with double DQN, Rainbow, policy gradient and Actor-Critic Algorithms.

* [Baseline DQN](#user-content-baseline-dqn) 
* [Double DQN](#user-content-Double-DQN)
* [Rainbow](#user-content-Rainbow)
* [Policy Gradient](#user-content-Policy-Gradient)
* [Actor-Critic Algorithms](#user-content-Actor-Critic-Algorithms)

We have also implemented a real-time browser demo here.
If you are not familar with Q-learing, you can visit a more fundamental project, Q-learning for Tic-Tac-Toe (GitHub Repo) and the real-time interactive streamlit demo.


The following is a detailed explaination of each approach and their environment setting in Chrome Dinosaur.

### Game Environment in learning
* No acceleration and no birds in the game for simplicity. If you want them, you can set the acceleration in game.py.
* Two actions only: up and nothing. There are three actions in the game actually, down to evade the birds if there is acceleration.
* Reward:
  * Hit the obstacle: -1
  * Otherwise: 0.1
* Using selenium in python to capture the images from the game.
* Using the version of Chrome Dinosaur in here: 

### Baseline DQN
* [Paper: Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Paper: Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
* [Paper: Self-Improving Reactive Agents Based On Reinforcement Learning, Planning and Teaching ](https://link.springer.com/content/pdf/10.1007%2FBF00992699.pdf)
* [Paper: Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
* [Link: Original implementation of this baseline in Keras](https://github.com/Paperspace/DinoRunTutorial)


### Double DQN
* [Reference code: TRAIN A MARIO-PLAYING RL AGENT from Pytorch Official](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
* [Similar project's report: Chrome Dino Run using Reinforcement Learning](https://arxiv.org/abs/2008.06799)
* All references in [Baseline DQN](#user-content-baseline-dqn) 

### Important Settings and Hyper-parameters
1. Training GPU: Nvidia RTX 3080 (12GB)
2. CPU:
3. Memory: 64 GB (if using prioritize replay buffer, should use at least 45 GB RAM)
4. Batch size: 32 (if too large, the overfitting will happen)
5. Buffer size: 100,000
6. **Final epsilon: 0.1**
7. FPS:
   * Slow mode: 14.xx - 18.xx fps (with prioritize replay buffer) 
   * Fast mode: 50 fps (without prioritizied replay buffer)

#### Result:


<img src="./img/double_dqn/max_scores.png" width="400" height="300" /><img src="./img/double_dqn/mean_scores.png" width="400" height="300" />
<img src="./img/double_dqn/median_scores.png" width="400" height="300" />


##### Epsilon Decay
For the final epsilon, we believe that it should be the most important hyper-parameters to affect the learning process. We tried 0.03 and 0.01 and 0.0001 before but the agent is not stable. The scores achieved by the agents are very obsolete during learning and the agent in testing is totally garbage if the epsilon is too small. Giving more exploration to the agent in this game seems better. I tried to follow the hyparameter in [this report](https://arxiv.org/abs/2008.06799) first but the problem occurs in what I have mentioned before. The training score (epsilon = 0.0001) is shown in figure . The average and median score of this agent in testing for 20 episodes are 50.xx only.

Later, we tried the final epsilon = 0.1. Although the max score in learning is smaller than 1,000, the test score is very higher when we test the agent for 20 episodes after training 100 episodes each time.

##### FPS
The FPS in here refers to the number of frames the agent to predict the action per second instead of the FPS of the game rendered by javascript in browser.

Since the computation of prioritized replay buffer is much higher, **our pc** in this experiment can only achieve $\approx 15$ FPS during learning process. If we use normal replay buffer only, the FPS in learning is faster, $\approx 50$ FPS. Higher FPS seems to be more general in lower FPS also but not the reverse. However, to obtain the similar performance, keeping the fps in both training and test is preferred. The FPS in testing is much faster without the learning process. The FPS is $\approx 90$ FPS in our PC, which is even faster than the game rendered by javascript. Thus, we add a sleep() in test function to slow down the FPS as close as learning.

<figure>
<img align="center" src="./img/double_dqn/testing_scores.png" width="400" height="300" />
<figcaption align = "center"> Fig.4 - Testing scores of a stable agents in 100 episodes (53.7 Avg FPS) </figcaption>
</figure>

### Rainbow

### Policy Gradient

### Actor-Critic Algorithms
[Paper: Actor-Critic Algorithms](https://papers.nips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

### Grad_CAM Visualization in Double DQN
<img src="./img/double_dqn/dino_grad_cam.gif" width="250" height="250" />

* [Paper: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
* [Paper of Grad-CAM's demo](http://gradcam.cloudcv.org)
* [The Grad-CAM package in python](https://github.com/jacobgil/pytorch-grad-cam)

Since the computation time of Grad-CAM is very long, every second can only produce one heat map only even though we use GPU to speed up. Therefore, it is hard to produce real-time Gram-CAM images during test. Instead of visualizing the heat maps in real time, we save the states during testing and produce the Grad-Cam heat maps after testing.

1. Set *cam_visualization* in config1.py to be True. The states from the game will be save to the folder, test_states, for each testing episode.
```json
"cam_visualization": True
```
2. Create heat_maps in GIF
```py
python3 visualize_CAM.py -c config1
```
You can choose which testing episode you want to generate the heat maps under visualize_CAM
```py
with open('./test_states/dino_states7.pickle', 'rb') as f:   # the 7-th test episode
```

The warmer area in the visualization is the area with higher weights, i.e. larger impact for these area's pixels to the final output of the agent. From the heat maps' visualization, it is very obvious that the area of the dinosaur and the obstacles always has higher impacts to the agent's action.
