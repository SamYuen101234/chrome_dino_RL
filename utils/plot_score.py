import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load the log file
summaries1 = tf.compat.v1.train.summary_iterator('/home/samyuen/Desktop/chrome_dino_RL/runs/May15_16-19-47_samyuen-System-Product-Name20220515-161947/events.out.tfevents.1652602788.samyuen-System-Product-Name.287412.0')
summaries2 = tf.compat.v1.train.summary_iterator('/home/samyuen/Desktop/chrome_dino_RL/runs/May23_18-52-23_samyuen-System-Product-Name20220523-185223/events.out.tfevents.1653303145.samyuen-System-Product-Name.226774.0')

scores1 = []

try:
    for e in tqdm(summaries1):
        for v in e.summary.value:
            if v.tag == 'Train/score':
                scores1.append(v.simple_value)
except:
    pass

max_list1 = []
mean_list1 = []
median_list1 = []
min_list1 = []
std_list1 = []
for i in range(len(scores1[:-1])//20 + 1):
    if i == len(scores1)//20 + 1:
        rolling = np.array(scores1[i*20:])
        max_list1.append(np.max(rolling))
        mean_list1.append(rolling.mean())
        median_list1.append(np.median(rolling))
        min_list1.append(np.min(rolling))
        std_list1.append(np.std(rolling))
    else:
        rolling = np.array(scores1[i*20:i*20+20])
        max_list1.append(np.max(rolling))
        mean_list1.append(rolling.mean())
        median_list1.append(np.median(rolling))
        min_list1.append(np.min(rolling))
        std_list1.append(np.std(rolling))



scores2 = []

try:
    for e in tqdm(summaries2):
        for v in e.summary.value:
            if v.tag == 'Train/score':
                scores2.append(v.simple_value)
except:
    pass

max_list2 = []
mean_list2 = []
median_list2 = []
min_list2 = []
std_list2 = []
for i in range(len(scores2[:-1])//20 + 1):
    if i == len(scores2)//20 + 1:
        rolling = np.array(scores2[i*20:])
        max_list2.append(np.max(rolling))
        mean_list2.append(rolling.mean())
        median_list2.append(np.median(rolling))
        min_list2.append(np.min(rolling))
        std_list2.append(np.std(rolling))
    else:
        rolling = np.array(scores2[i*20:i*20+20])
        max_list2.append(np.max(rolling))
        mean_list2.append(rolling.mean())
        median_list2.append(np.median(rolling))
        min_list2.append(np.min(rolling))
        std_list2.append(np.std(rolling))

plt.plot(max_list1)
plt.plot(max_list2)
plt.savefig('./img/double_dqn/max_scores.png')
plt.clf()

plt.plot(mean_list1)
plt.plot(mean_list2)
plt.savefig('./img/double_dqn/mean_scores.png')
plt.clf()

plt.plot(median_list1)
plt.plot(median_list2)
plt.savefig('./img/double_dqn/median_scores.png')
plt.clf()

plt.plot(min_list1)
plt.plot(min_list2)
plt.savefig('./img/double_dqn/min_scores.png')
plt.clf()

plt.plot(std_list1)
plt.plot(std_list2)
plt.savefig('./img/double_dqn/std_scores.png')
plt.clf()

plt.hist(scores1)
plt.hist(scores2)
plt.savefig('./img/double_dqn/hist.png')
plt.clf()

plt.boxplot(scores1)
plt.boxplot(scores2)
plt.savefig('./img/double_dqn/boxplot.png')
plt.clf()

plt.plot(np.log(scores1))
plt.plot(np.log(scores2))
plt.savefig('./img/double_dqn/scores.png')
plt.clf()

np_scores = np.array(scores2)
print('Number of episodes < 250 scores:', len(np_scores[np_scores < 250]))
print('Number of episodes between 250 and 500:', len(np_scores[(np_scores >= 250) & (np_scores < 500)]))
print('Number of episodes between 500 and 1000:', len(np_scores[(np_scores >= 500) & (np_scores < 1000)]))
print('Number of episodes between 1000 and 2000:', len(np_scores[(np_scores >= 1000) & (np_scores < 2000)]))
print('Number of episodes >= 2000:', len(np_scores[np_scores >= 2000]))
