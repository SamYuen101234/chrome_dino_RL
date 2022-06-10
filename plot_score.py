from cProfile import label
from utils.utils import export_onxx
import torch
import numpy as np
import onnx
import matplotlib.pyplot as plt
#export_onxx('./weights/double_dqn2.pth')
scores = np.load('./test_scores/DoubleDQN_test_score.npy')
#print(scores)
plt.plot(scores, label='socre', alpha=0.7)
plt.text(85, np.mean(scores)+1000, 'mean (' + str(np.mean(scores)) +')')
plt.text(85, np.median(scores)-5000, 'median (' + str(np.median(scores)) + ')')
plt.text(85, np.min(scores)-5000, 'min (' + str(np.min(scores))+')')

#plt.axhline(y=np.mean(scores), xmin=0.1, xmax=0.8, color='r', linestyle='-.', linewidth=3, label='mean')
plt.axhline(y=np.mean(scores), color='r', linestyle='-.', label='mean')
plt.axhline(y=np.median(scores), color='g', linestyle='--', label='median')
plt.axhline(y=np.min(scores), color='purple', linestyle=':', label='min')
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.title('Testing scores in 100 episodes')
plt.legend()
plt.savefig('./img/double_dqn/testing_scores.png')
plt.clf()

