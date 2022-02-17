import tensorboard as tb
import matplotlib.pyplot as plt
import pandas as pd
# plot the score from tensorboard cloud
experiment_id = "4rnq0VEUR5SknDeXgaANvg"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
#print(df)
df = df[df['run'] == 'Feb16_12-32-35_samyuen-System-Product-Name20220216-123235']
df[df['tag']=='Train/score'].to_csv('score.csv', index=False)
df = pd.read_csv('score.csv')
#df[['value']].to_csv('score.csv')
lst1 = pd.Series(df['value']).rolling(100).max().dropna().tolist()
plt.plot(lst1)
plt.savefig('temp.png')
plt.clf()