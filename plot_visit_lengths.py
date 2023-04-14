import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

stats = pickle.load(open('results/shape.pkl', 'rb'))

l = ['EVA', 'SynTEG', 'LSTM', 'HALO Coarse', 'GPT', 'HALO', 'Train']
stats['SynTEG']['Visit Lengths'] = np.random.choice(stats['SynTEG']['Visit Lengths'], len(stats['Train']['Visit Lengths']), replace=False)
B = pd.DataFrame({key: pd.Series(stats[key]['Visit Lengths']) for key in l})
B.plot.kde()
plt.xlim(-5,45)
plt.xlabel('Number of Codes')
plt.title('Inpatient EHR Visit Lengths Probability Density')
plt.savefig("results/plots/visit_lengths.png")