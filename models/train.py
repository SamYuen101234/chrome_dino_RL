from cgitb import reset
import time
import pickle
from collections import deque # double ended queue (can append and pop in both end)
import numpy as np
import random
import torch
import copy


def init_cache(INITIAL_EPSILON, REPLAY_MEMORY):
    """initial variable caching, done only once"""
    t, D = 0, deque(maxlen=REPLAY_MEMORY) # for experience replay
    set_up_dict = {"epsilon": INITIAL_EPSILON, "step": t, "D": D, "highest_score": 0}
    save_obj(set_up_dict, "set_up")

def save_obj(obj, name):
    with open('./result/'+ name + '.pkl', 'wb') as f: #dump files into objects folder
        # Use the latest protocol that supports the lowest Python version you want to support reading the data
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open('result/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class trainNetwork:
    def __init__(self, agent,game, writer, Deque, BATCH, device):
        self.agent = agent
        self.agent.online.to(device)
        self.agent.target.to(device)
        self.game = game
        self.device = device
        self.writer = writer
        self.memory = Deque
        self.batch_size = BATCH

    def cache(self, state, action_idx, reward, next_state, terminal):
        """
        Store the experience to self.memory (replay buffer)
        """

        #state = state.__array__()
        #next_state = next_state.__array__()
        state = state.detach().clone().cpu()
        action_idx = torch.tensor(action_idx).detach().clone().cpu()
        next_state = next_state.detach().clone().cpu()
        reward = torch.tensor([reward]).detach().clone().cpu()
        terminal = torch.tensor([terminal]).detach().clone().cpu()

        self.memory.append((state, action_idx, reward, next_state, terminal))


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, action_idx, reward, next_state, terminal = map(torch.stack, zip(*batch))
        state = torch.squeeze(state)
        next_state = torch.squeeze(next_state)
        reward = torch.squeeze(reward)
        terminal = torch.squeeze(terminal)
        return state.to(self.device), action_idx.to(self.device), reward.to(self.device), next_state.to(self.device), terminal.to(self.device)

    def td_estimate(self, state, action):
        pass

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        pass

    def update_Q_online(self, td_estimate, td_target):
        pass
    
    def sync_Q_target(self):
        pass

    def save(self):
        pass

    def start(self, epsilon, step, highest_score, 
            OBSERVE, ACTIONS, EPSILON_DECAY, FINAL_EPSILON, GAMMA,
            FRAME_PER_ACTION, EPISODE, SAVE_EVERY, SYNC_EVERY, TRAIN_EVERY):
        last_time = time.time() # for computing fps
        current_episode = 0
        num_action_0 = 0
        num_action_1 = 0
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
                time.sleep(.03) # make the fps lower for small batch number
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
                    state_t, action_t, reward_t, state_t1, terminal = self.recall()
                    #td_target = torch.zeros((self.batch_size, ACTIONS)) # (batch_size, 2)
                    
                    if step % SYNC_EVERY == 0:
                        self.agent.sync_target()

                    if step % TRAIN_EVERY == 0:
                        loss, avg_q_max= self.agent.step(state_t.to(self.device), action_t.to(self.device), reward_t.to(self.device), state_t1.to(self.device), terminal.to(self.device))
                        
                        # record the log
                        avg_loss.update(loss, self.batch_size)
                        avg_Q_max.update(avg_q_max, self.batch_size)
                        self.writer.add_scalar("Train/loss", avg_loss.avg, step)
                    
                    # save model
                    if step % SAVE_EVERY == 0:
                        self.game.pause() #pause game while saving to filesystem
                        print("Now we save model")
                        self.agent.save_model()
                        set_up_dict = {"epsilon": epsilon, "step": step, "D": self.memory, "highest_score": highest_score}
                        save_obj(set_up_dict, "set_up")
                        self.game.resume()
                
                s_t = copy.deepcopy(s_t1) # assign next state to current state
                step += 1
                
            
            # update the log after the end of each episode
            current_episode += 1

            # check highest point
            current_score = self.game.get_score()
            if current_score > highest_score:
                highest_score = current_score
            self.writer.add_scalar("Train/score", current_score, current_episode)
            self.writer.add_scalar("Train/reward", total_reward, current_episode)
            avg_score.update(current_score, 1)
            avg_reward.update(total_reward, 1)
            print('Episode [{}/{}], Step {}, Epsilon {:.5f}, Action: (0:{},1:{}), Avg FPS: {:.4f},  Avg Reward: {:.4f}, Avg Q_MAX: {:.4f}, Avg Loss: {:.4f}, Hihgest Score: {}'
                .format(current_episode, EPISODE, step, epsilon, num_action_0, num_action_1, avg_fps.avg,
                avg_reward.avg, avg_Q_max.avg, avg_loss.avg, highest_score))
            
            avg_fps.reset()
            s_t = copy.deepcopy(initial_state) # reinitialize the first state
            self.game.restart() # restart the game if gameover
            
        print("Training Finished!")
        print("************************")