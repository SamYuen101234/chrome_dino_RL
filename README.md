# Tasks to be implemented

- [x] Deep Q Network Agent
- [x] Double Deep Q Network Agent
- [x] Prioritized Experience Replay
- [ ] Real-time Grad-Cam Visualization
- [ ] Rainbox
- [ ] Policy Gradient
- [ ] Actor-Critic Algorithms
- [ ] Deployment

# Quick Start
1. 
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

* [Baseline DQN](#Baseline-DQN) 
* [Double DQN](Double-DQN)
* [Rainbow](Rainbow)
* [Policy Gradient](Policy-Gradient)
* [Actor-Critic Algorithms](Actor-Critic-Algorithms)

We have also implemented a real-time browser demo here.
If you are not familar with Q-learing, you can visit a more fundamental project, Q-learning for Tic-Tac-Toe (GitHub Repo) and the real-time interactive streamlit demo.


The following is a detailed explaination of each approach and their environment setting in Chrome Dinosaur.

### Baseline DQN
* [Paper: Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Paper: Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
* [Paper: Self-Improving Reactive Agents Based On Reinforcement Learning, Planning and Teaching ](https://link.springer.com/content/pdf/10.1007%2FBF00992699.pdf)
* [Link: Original implementation of this baseline in Keras](https://github.com/Paperspace/DinoRunTutorial)

### Double DQN
* [Reference code: TRAIN A MARIO-PLAYING RL AGENT from Pytorch Official](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

### Rainbow

### Policy Gradient

### Actor-Critic Algorithms
[Paper: Actor-Critic Algorithms](https://papers.nips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
