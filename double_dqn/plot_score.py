import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load the log file
#summaries = tf.compat.v1.train.summary_iterator('/home/samyuen/Desktop/vscode/chrome_dino_training/double_dqn/runs/Mar29_14-00-48_samyuen-System-Product-Name20220329-140048/events.out.tfevents.1648533648.samyuen-System-Product-Name.2917626.0')
summaries = tf.compat.v1.train.summary_iterator('/home/samyuen/Desktop/vscode/chrome_dino_training/double_dqn/runs/Apr03_23-07-33_samyuen-System-Product-Name20220403-230733/events.out.tfevents.1648998454.samyuen-System-Product-Name.1904013.0')
scores = []

try:
    for e in tqdm(summaries):
        for v in e.summary.value:
            if v.tag == 'Train/score':
                scores.append(v.simple_value)
except:
    pass
max_list = pd.Series(scores).rolling(10).max().dropna().tolist()
plt.plot(max_list)
plt.savefig('./img/max_scores.png')
plt.clf()
mean_list = pd.Series(scores).rolling(10).mean().dropna().tolist()
plt.plot(mean_list)
plt.savefig('./img/mean_scores.png')
plt.clf()
min_list = pd.Series(scores).rolling(20).min().dropna().tolist()
plt.plot(min_list)
plt.savefig('./img/min_scores.png')
plt.clf()
median_list = pd.Series(scores).rolling(10).median().dropna().tolist()
plt.plot(median_list)
plt.savefig('./img/median_scores.png')
plt.clf()
std_list = pd.Series(scores).rolling(250).std().dropna().tolist()
plt.plot(std_list)
plt.savefig('./img/std_scores.png')
plt.clf()
plt.hist(scores)
plt.savefig('./img/hist.png')
plt.clf()

plt.boxplot(scores)
plt.savefig('./img/boxplot.png')
plt.clf()

plt.plot(scores)
plt.savefig('./img/scores.png')
plt.clf()

np_scores = np.array(scores)
print('Number of episodes < 250 scores:', len(np_scores[np_scores < 250]))
print('Number of episodes between 250 and 500:', len(np_scores[(np_scores >= 250) & (np_scores < 500)]))
print('Number of episodes between 500 and 1000:', len(np_scores[(np_scores >= 500) & (np_scores < 1000)]))
print('Number of episodes between 1000 and 2000:', len(np_scores[(np_scores >= 1000) & (np_scores < 2000)]))
print('Number of episodes >= 2000:', len(np_scores[np_scores >= 2000]))
