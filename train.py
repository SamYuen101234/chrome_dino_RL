import time
import pickle
from collections import deque # double ended queue (can append and pop in both end)
import numpy as np
import random
import json
import tensorflow as tf

def init_cache(INITIAL_EPSILON):
    """initial variable caching, done only once"""
    t, D = 0, deque()
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
def train(model, game, epsilon, step, Deque, highest_score, 
          OBSERVE, ACTIONS, INITIAL_EPSILON, FINAL_EPSILON, GAMMA,
          FRAME_PER_ACTION, EXPLORE, EPISODE, SAVE_EVERY, BATCH,
          REPLAY_MEMORY):
    last_time = time.time()
    # first action is do nothing
    do_nothing = np.zeros(ACTIONS) # the action array, 0: do nothing, 1: jump, 2: duck
    do_nothing[0] = 1 
    x_t, r_0, terminal = game.get_state(do_nothing) # perform this action and get the next state
    # state at t
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # stack 4 images to create placeholder input
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*20*40*4
    initial_state = s_t
    current_episode = 0
    while current_episode < EPISODE:
        total_reward = r_0
        while not game.get_crashed(): # not game over yet
            loss = 0
            Q_sa = 0
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
                    q = model(s_t) #input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)
                    action_index = max_Q 
                    a_t[action_index] = 1 # set the action's prediction to 1
            
            # epsilon decay: reduce the chance to random explore and depends more on the model's decision 
            if epsilon > FINAL_EPSILON and step > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
            
            # perform the action, and observe next state and receive reward
            x_t1, r_t, terminal = game.get_state(a_t)
            total_reward += r_t
            print('fps: {:.4f}'.format(1 / (time.time()-last_time)))
            last_time = time.time()
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x20x40x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
            
            # store the transition (experience tuple) in D
            Deque.append((s_t, action_index, r_t, s_t1, terminal))
            if len(Deque) > REPLAY_MEMORY:
                Deque.popleft()
            
            # only train for train =='train' or 'continue_train'
            if step > OBSERVE:
                #sample a minibatch to train on
                #print(len(Deque), OBSERVE)
                minibatch = random.sample(Deque, BATCH)
                inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 20, 40, 4
                targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2
                
                #Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]    # 4D stack of images
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]   #reward at state_t due to action_t
                    state_t1 = minibatch[i][3]   #next state
                    terminal = minibatch[i][4]   #wheather the agent died or survided due the action

                    inputs[i:i + 1] = state_t    
                    
                    targets[i] = model(state_t, training=False)  # predicted q values
                    Q_sa = model(state_t1, training=False)      #predict q values for next step
                    
                    if terminal:
                        targets[i, action_t] = reward_t # if terminated, only equals reward
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                temp_s = time.time()
                
                loss += model.train_on_batch(inputs, targets)
                
                # record the info
                '''with game.writer.as_default():
                    tf.summary.scalar('loss', loss, step=step)
                    tf.summary.scalar('reward', total_reward, step=step)'''
                
                # save model
                if step % SAVE_EVERY == 0:
                    game.pause() #pause game while saving to filesystem
                    print("Now we save model and upload info")
                    model.save_weights("model.h5", overwrite=True)
                    set_up_dict = {"epsilon": epsilon, "step": step, "D": Deque, "highest_score": highest_score}
                    save_obj(set_up_dict, "set_up")
                    with open("./model/model.json", "w") as outfile:
                        json.dump(model.to_json(), outfile)
                    game.resume()
            
            s_t = initial_state if terminal else s_t1 #reset game to initial frame if terminate
            step += 1
            state = ""
            if step <= OBSERVE:
                state = "observe"
            elif step > OBSERVE and step <= OBSERVE + EXPLORE:
                state = "DQN exploring"
            else:
                state = "DQN playing" # Epsilon = 0.0001, nearly all depends on DQN
            
            
            print('Episode [{}/{}], Step {}, State: {}, Epsilon {}, Action: {},  Reward: {}, Q_MAX: {:.4f}, Loss: {:.4f}, Hihgest Score: {}'
                  .format(current_episode+1, EPISODE, step, state, epsilon, 
                  action_index,  r_t, np.max(Q_sa), loss, highest_score))
            
        # end of an episode
        current_episode += 1
        # check highest point
        current_score = game.get_score()
        if current_score > highest_score:
            highest_score = current_score
        
        with game.writer.as_default():
            tf.summary.scalar('score', current_score, step=current_episode)
            tf.summary.scalar('reward', total_reward, step=current_episode)
        game.restart() # restart the game if gameover
        
    print("Training Finished!")
    print("************************")