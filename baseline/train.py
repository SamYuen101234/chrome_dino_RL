import time
import pickle
from collections import deque # double ended queue (can append and pop in both end)
import numpy as np
import random
import torch

def init_cache(INITIAL_EPSILON):
    """initial variable caching, done only once"""
    t, D = 0, deque() # for experience replay
    set_up_dict = {"epsilon": INITIAL_EPSILON, "step": t, "D": D, "highest_score": 0}
    save_obj(set_up_dict, "set_up")

def save_obj(obj, name):
    with open('./result/'+ name + '.pkl', 'wb') as f: #dump files into objects folder
        # Use the latest protocol that supports the lowest Python version you want to support reading the data
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open('result/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

'''
    ACTIONS: num of outputs/actions
    OBSERVATION: steps for the agent to explore randomly before following the DQN
    FINAL_EPSILON: the min value of epsilon for epsilon decay
'''

class trainNetwork:
    def __init__(self,model,game, optimizer, criterion, writer, device):
        self.model = model
        self.game = game
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = writer

    def start(self, epsilon, step, Deque, highest_score, 
            OBSERVE, ACTIONS, INITIAL_EPSILON, FINAL_EPSILON, GAMMA,
            FRAME_PER_ACTION, EXPLORE, EPISODE, SAVE_EVERY, BATCH,
            REPLAY_MEMORY):
        last_time = time.time()
        # first action is do nothing
        do_nothing = np.zeros(ACTIONS) # the action array, 0: do nothing, 1: jump, 2: duck
        do_nothing[0] = 1 
        x_t, r_0, terminal = self.game.get_state(do_nothing) # perform this action and get the next state
        # state at t
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # stack 4 images to create placeholder input
        s_t = s_t.reshape(1, s_t.shape[2], s_t.shape[0], s_t.shape[1])  #1*20*40*4
        s_t = torch.from_numpy(s_t).float()
        initial_state = s_t
        current_episode = 0
        while current_episode < EPISODE:
            total_reward = 0
            while not self.game.get_crashed(): # not game over yet
                loss = torch.zeros(1)
                Q_sa = torch.zeros((1,2))
                action_index = 0
                r_t = 0 # reward at t
                a_t = np.zeros([ACTIONS]) # action at t
            
                # if the sample is smaller than epsilon, we perform random action
                if step % FRAME_PER_ACTION == 0: # perform action every n frames
                    if random.random() < epsilon:
                        # print("----------Random Action----------")
                        action_index = random.randrange(ACTIONS)
                        a_t[action_index] = 1
                    else:
                        with torch.no_grad():
                            self.model.eval()
                            q = self.model(s_t.to(self.device)) #input a stack of 4 images, get the prediction
                        max_Q = torch.argmax(q)
                        action_index = max_Q 
                        a_t[action_index] = 1 # set the action's prediction to 1
                
                # epsilon decay: reduce the chance to random explore and depends more on the model's decision 
                if epsilon > FINAL_EPSILON and step > OBSERVE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                
                
                # perform the action, and observe next state and receive reward
                x_t1, r_t, terminal = self.game.get_state(a_t)
                total_reward += r_t
                print('fps: {:.4f}'.format(1 / (time.time()-last_time)))
                last_time = time.time()
                x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1]) #1x20x40x1
                s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
                s_t1 = torch.from_numpy(s_t1)
                
                # store the transition (experience tuple) in D
                Deque.append((s_t, action_index, r_t, s_t1, terminal)) # for experience replay
                if len(Deque) > REPLAY_MEMORY: # when the buffer is full
                    Deque.popleft()
                
                # only train for train =='train' or 'continue_train'
                if step > OBSERVE:
                    #sample a minibatch to train on
                    #print(len(Deque), OBSERVE)
                    minibatch = random.sample(Deque, BATCH)
                    inputs = torch.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #16, 4, 80, 80
                    targets = torch.zeros((inputs.shape[0], ACTIONS))                         #32, 2
                    
                    #Now we do the experience replay
                    for i in range(0, len(minibatch)):
                        state_t = minibatch[i][0]    # 4D stack of images
                        action_t = minibatch[i][1]   #This is action index
                        reward_t = minibatch[i][2]   #reward at state_t due to action_t
                        state_t1 = minibatch[i][3]   #next state
                        terminal = minibatch[i][4]   #wheather the agent died or survided due the action

                        inputs[i:i + 1] = state_t    
                        with torch.no_grad():
                            self.model.eval()
                            targets[i] = self.model(state_t.to(self.device))  # predicted q values
                            Q_sa = self.model(state_t1.to(self.device))      #predict q values for next step
                        
                        if terminal:
                            targets[i, action_t] = reward_t # if terminated, only equals reward
                        else:
                            targets[i, action_t] = reward_t + GAMMA * torch.max(Q_sa)
                    self.model.train()
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.to(self.device), targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # record the info
                    self.writer.add_scalar("Train/loss", loss.item(), step)
                    
                    # save model
                    if step % SAVE_EVERY == 0:
                        self.game.pause() #pause game while saving to filesystem
                        print("Now we save model")
                        torch.save(self.model, "./model/model.pth")
                        set_up_dict = {"epsilon": epsilon, "step": step, "D": Deque, "highest_score": highest_score}
                        save_obj(set_up_dict, "set_up")
                        self.game.resume()
                
                s_t = initial_state if terminal else s_t1 #reset game to initial frame if terminate
                step += 1
                state = ""
                if step <= OBSERVE:
                    state = "observe"
                elif step > OBSERVE and step <= OBSERVE + EXPLORE:
                    state = "DQN exploring"
                else:
                    state = "DQN training" # Epsilon = 0.0001, training depends on the records in deque
                
                print('Episode [{}/{}], Step {}, State: {}, Epsilon {}, Action: {},  Reward: {}, Q_MAX: {:.4f}, Loss: {:.4f}, Hihgest Score: {}'
                    .format(current_episode+1, EPISODE, step, state, epsilon, 
                    action_index,  r_t, torch.max(Q_sa).item(), loss.item(), highest_score))
                
            # end of an episode
            current_episode += 1
            # check highest point
            current_score = self.game.get_score()
            if current_score > highest_score:
                highest_score = current_score
            

            self.writer.add_scalar("Train/score", current_score, current_episode)
            self.writer.add_scalar("Train/reward", total_reward, current_episode)
            self.game.restart() # restart the game if gameover
            
        print("Training Finished!")
        print("************************")