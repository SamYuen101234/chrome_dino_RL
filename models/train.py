from cgitb import reset
import time
import numpy as np
import random
import torch
import copy
import sys
sys.path.append("../")
from utils.utils import AverageMeter, save_obj
from models.test import test_agent

class trainNetwork:
    def __init__(self, agent,game, writer, buffer, BATCH, device):
        self.agent = agent
        self.agent.online.to(device)
        self.agent.target.to(device)
        self.game = game
        self.device = device
        self.writer = writer
        self.memory = buffer
        self.batch_size = BATCH
        self.eps = 1.0

    def cache(self, state, action_idx, reward, next_state, terminal):
        """
        Store the experience to self.memory (replay buffer)
        """

        state = state.detach().clone().cpu()
        action_idx = torch.tensor(action_idx).detach().clone().cpu()
        next_state = next_state.detach().clone().cpu()
        reward = torch.tensor([reward]).detach().clone().cpu()
        terminal = torch.tensor([terminal]).detach().clone().cpu()

        self.memory.append((state, action_idx, reward, next_state, terminal))


    def save(self, epsilon, step, highest_score):
        #self.game.pause() # pause game while saving to filesystem
        print("Now we save model")
        self.agent.save_model()
        # set_up_dict = {"epsilon": epsilon, "step": step, "D": self.memory, "highest_score": highest_score}
        # save_obj(set_up_dict, "set_up") # save the buffer to disk, need lots of space if the buffer size is large
        #self.game.resume()

    def early_stopping(self):
        pass

    def start(self, epsilon, step, highest_score, 
            OBSERVE, ACTIONS, EPSILON_DECAY, FINAL_EPSILON, GAMMA,
            FRAME_PER_ACTION, EPISODE, SAVE_EVERY, SYNC_EVERY, TRAIN_EVERY, 
            prioritized_replay, TEST_EVERY, args):
        last_time = time.time() # for computing fps
        current_episode = 0
        num_action_0 = 0
        num_action_1 = 0
        max_avg_test_score = 0
        max_median_test_score = 0
        avg_loss = AverageMeter()
        avg_Q_max = AverageMeter()
        avg_reward = AverageMeter()
        avg_score = AverageMeter()
        avg_fps = AverageMeter()
        # first action is do nothing
        do_nothing = np.zeros(ACTIONS) # the action array, 0: do nothing, 1: jump, 2: duck
        do_nothing[0] = 1 
        num_action_0 += 1
        x_t, r_0, terminal = self.game.get_state(do_nothing) # perform this action and get the next state
        # state at t
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # stack 4 images to create placeholder input
        
        s_t = s_t.reshape(1, s_t.shape[2], s_t.shape[0], s_t.shape[1])  #1*4*80*80
        s_t = torch.from_numpy(s_t).float()
        initial_state = copy.deepcopy(s_t)

        while current_episode < EPISODE:
            total_reward = 0
            start_time = time.time()
            while not self.game.get_crashed(): # not game over yet
                #time.sleep(.03) # make the fps lower for small batch number
                loss = torch.zeros(1)
                Q_sa = torch.zeros((1,2))
                action_idx = 0
                r_t = 0 # reward at t
                a_t = np.zeros([ACTIONS]) # action at t
            
                # if the sample is smaller than epsilon, we perform random action
                if step % FRAME_PER_ACTION == 0: # perform action every n frames
                    if np.random.rand() < epsilon:
                        action_idx = np.random.randint(ACTIONS)
                        a_t[action_idx] = 1
                    else:
                        action_idx = self.agent.get_action(s_t)
                        a_t[action_idx] = 1 # set the action's prediction to 1

                if action_idx == 0:
                    num_action_0 += 1
                else:
                    num_action_1 += 1

                # perform the action, and observe next state and receive reward
                x_t1, r_t, terminal = self.game.get_state(a_t)
                total_reward += r_t
                
                avg_fps.update(1 / (time.time()-last_time), 1)
                last_time = time.time()
                
                x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1]) #1x1x80x80
                s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
                s_t1 = torch.from_numpy(s_t1)
                
                # store the transition (experience tuple) in a deque for experience replay
                self.cache(s_t, action_idx, r_t, s_t1, terminal)
                # only train for train =='train' or 'continue_train'
                if step > OBSERVE:
                    # epsilon decay: reduce the chance to random explore 
                    epsilon *= EPSILON_DECAY
                    epsilon = max(FINAL_EPSILON, epsilon)

                    #sample a minibatch to train
                    if prioritized_replay:
                        # prioritization exponent
                        alpha = 0.7 # 0.7 was suggested by the prioritized_replay paper
                        state_t, action_t, reward_t, state_t1, terminal, importance, indices =\
                        self.memory.recall(self.batch_size, priority_scale=alpha)
                    else:
                        importance = None
                        state_t, action_t, reward_t, state_t1, terminal = self.memory.recall(self.batch_size)
                    #td_target = torch.zeros((self.batch_size, ACTIONS)) # (batch_size, 2)
                    
                    if step % SYNC_EVERY == 0:
                        self.agent.sync_target()

                    if step % TRAIN_EVERY == 0:
                        if importance is not None: importance = importance.to(self.device)**(1-self.eps) # use prioritized buffer
                        loss, avg_q_max, error = self.agent.step(state_t.to(self.device), action_t.to(self.device), reward_t.to(self.device),\
                                                        state_t1.to(self.device), terminal.to(self.device),\
                                                        importance)

                        if prioritized_replay: self.memory.set_priorities(indices, error) # update the priorities of priorities buffer
                        if torch.any(terminal): self.eps = max(0.1, 0.999*self.eps)
                        
                        # record the log
                        avg_loss.update(loss, self.batch_size)
                        avg_Q_max.update(avg_q_max, self.batch_size)
                        self.writer.add_scalar("Train/loss", avg_loss.avg, step)
                        
                    #################################################################################################################################

                    '''# save info
                    if step % SAVE_EVERY == 0:
                        self.save(epsilon, step, highest_score)'''
                
                s_t = copy.deepcopy(s_t1) # assign next state to current state
                step += 1
                
            # update the log after the end of each episode
            current_episode += 1

            # test 
            if current_episode % TEST_EVERY == 0:
                self.game.pause() # pause game for learning to open a new random env for testing
                with torch.no_grad():
                    avg_test_scores, median_test_scores = test_agent(self.agent.online, args, self.device)

                self.writer.add_scalar("Test/mean_score", avg_test_scores, current_episode)
                self.writer.add_scalar("Test/median_score", avg_test_scores, current_episode)

                if avg_test_scores > max_avg_test_score:
                    max_avg_test_score = avg_test_scores

                if median_test_scores > max_median_test_score:
                    max_median_test_score = median_test_scores
                    self.save(epsilon, step, highest_score) # save agent

                self.game.resume()
            # check highest point
            current_score = self.game.get_score()
            if current_score > highest_score:
                highest_score = current_score
            self.writer.add_scalar("Train/score", current_score, current_episode)
            self.writer.add_scalar("Train/reward", total_reward, current_episode)
            avg_score.update(current_score, 1)
            avg_reward.update(total_reward, 1)
            print('Episode [{}/{}], Step {}, Epsilon: {:.5f}, EPS: {:.5f} Action: (0:{},1:{}), Avg FPS: {:.4f},  Avg Reward: {:.4f}, Avg Q_MAX: {:.4f}, Avg Loss: {:.4f}, Max Train Score: {}, Max Avg Test: {:.2f}, Max Median Test: {}'
                .format(current_episode, EPISODE, step, epsilon, self.eps, num_action_0, num_action_1, avg_fps.avg,
                avg_reward.avg, avg_Q_max.avg, avg_loss.avg, highest_score, max_avg_test_score, max_median_test_score))
            
            avg_fps.reset()
            s_t = copy.deepcopy(initial_state) # reinitialize the first state
            self.game.restart() # restart the game if gameover
            
        