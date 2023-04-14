import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

stats = pickle.load(open('results/shape.pkl', 'rb'))

# Some fake data to plot
l1 = 'Train'
list2 = ['SynTEG', 'EVA', 'LSTM', 'GPT', 'HALO Coarse', 'HALO']
colors = ['brown', 'purple', 'orange', 'green', 'red', 'blue']

for label, col in zip(list2, colors):
  X = np.expand_dims(np.array(stats[l1]['Labels']), 1)
  y = np.expand_dims(np.array(stats[l2]['Labels']), 1)
  theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
  y_line = X.dot(theta)
  r2 = r2_score(y, X)
  plt.scatter(X, y, c=col, label=f"{label} ({r2:.3f})", marker='x' if label == 'HALO' else 'o')

plt.xlim(0, 0.25)
plt.xlabel('MIMIC Label Probability')
plt.ylim(0, 0.25)
plt.ylabel('Synthetic Dataset Label Probability')
plt.title('Synthetic Dataset vs. MIMIC Chronic Condition Label Probabilities')
plt.plot([0,0.25], [0,0.25], 'k-', zorder=0)
plt.legend()
plt.savefig('results/plots/label_probs.png')