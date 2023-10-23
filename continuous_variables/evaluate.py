import pickle
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import HALOConfig
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

config = HALOConfig()
train_ehr_dataset = pickle.load(open(f'./data/trainDataset.pkl', 'rb'))
halo_ehr_dataset = pickle.load(open(f'./results/datasets/haloDataset_converted.pkl', 'rb'))

for p in train_ehr_dataset:
  new_visits = []
  currLab = 0
  for v in p['visits']:
    if v[1] == []:
      new_visits.append(v)
      currLab = 0
    else:
      if currLab < 200:
        new_visits.append(v)
        currLab += 1
  p['visits'] = new_visits

discretization = pickle.load(open(f'discretized_data/discretization.pkl', 'rb'))
is_categorical_channel = pickle.load(open(f'discretized_data/isCategorical.pkl', 'rb'))
idToLab = pickle.load(open(f'discretized_data/idToLab.pkl', 'rb'))

#####################
### Dataset Stats ###
#####################

def generate_statistics(ehr_dataset):
    stats = {}
    label_counts = {}
    for i in tqdm(range(config.label_vocab_size, config.label_vocab_size + 1)):
        label_stats = {}
        d = [p for p in ehr_dataset if p['labels'][i] == 1] if i < config.label_vocab_size else ehr_dataset
        label_counts[i] = len(d)

        aggregate_stats = {}
        record_lens = [len(p['visits']) for p in d]
        visit_lens = [len(v[0]) for p in d for v in p['visits'] if v[1] == []]
        visit_gaps = [v[3][0] for p in d for v in p['visits'] if v[1] != []]
        avg_record_len = np.mean(record_lens)
        std_record_len = np.std(record_lens)
        avg_visit_len = np.mean(visit_lens)
        std_visit_len = np.std(visit_lens)
        avg_visit_gap = np.mean(visit_gaps)
        std_visit_gap = np.std(visit_gaps)
        aggregate_stats["Record Length Mean"] = avg_record_len
        aggregate_stats["Record Length Standard Deviation"] = std_record_len
        aggregate_stats["Visit Length Mean"] = avg_visit_len
        aggregate_stats["Visit Length Standard Deviation"] = std_visit_len
        aggregate_stats["Visit Gap Mean"] = avg_visit_gap
        aggregate_stats["Visit Gap Standard Deviation"] = std_visit_gap
        label_stats["Aggregate"] = aggregate_stats

        code_stats = {}
        n_records = len(record_lens)
        n_visits = len(visit_lens)
        record_code_counts = {}
        visit_code_counts = {}
        record_bigram_counts = {}
        visit_bigram_counts = {}
        for p in d:
            patient_codes = set()
            patient_bigrams = set()
            for j in range(len(p['visits'])):
                v = p['visits'][j]
                if v[1] != []:
                    continue

                for c in v[0]:
                    visit_code_counts[c] = 1 if c not in visit_code_counts else visit_code_counts[c] + 1
                    patient_codes.add(c)
                for cs in itertools.combinations(v[1],2):
                    cs = list(cs)
                    cs.sort()
                    cs = tuple(cs)
                    visit_bigram_counts[cs] = 1 if cs not in visit_bigram_counts else visit_bigram_counts[cs] + 1
                    patient_bigrams.add(cs)
            for c in patient_codes:
                record_code_counts[c] = 1 if c not in record_code_counts else record_code_counts[c] + 1
            for cs in patient_bigrams:
                record_bigram_counts[cs] = 1 if cs not in record_bigram_counts else record_bigram_counts[cs] + 1
        record_code_probs = {c: record_code_counts[c]/n_records for c in record_code_counts}
        visit_code_probs = {c: visit_code_counts[c]/n_visits for c in visit_code_counts}
        record_bigram_probs = {cs: record_bigram_counts[cs]/n_records for cs in record_bigram_counts}
        visit_bigram_probs = {cs: visit_bigram_counts[cs]/n_visits for cs in visit_bigram_counts}
        code_stats["Per Record Code Probabilities"] = record_code_probs
        code_stats["Per Visit Code Probabilities"] = visit_code_probs
        code_stats["Per Record Bigram Probabilities"] = record_bigram_probs
        code_stats["Per Visit Bigram Probabilities"] = visit_bigram_probs
        label_stats["Probabilities"] = code_stats
        stats[i] = label_stats
    label_probs = {l: label_counts[l]/n_records for l in label_counts}
    stats["Label Probabilities"] = label_probs
    return stats
    
def generate_plots(stats1, stats2, label1, label2, types=["Per Record Code Probabilities", "Per Visit Code Probabilities", "Per Record Bigram Probabilities", "Per Visit Bigram Probabilities"]):
    for i in tqdm(range(config.label_vocab_size, config.label_vocab_size + 1)):
        label = i
        data1 = stats1[label]["Probabilities"]
        data2 = stats2[label]["Probabilities"]
        for t in types:
            probs1 = data1[t]
            probs2 = data2[t]
            keys = set(probs1.keys()).union(set(probs2.keys()))
            values1 = [probs1[k] if k in probs1 else 0 for k in keys]
            values2 = [probs2[k] if k in probs2 else 0 for k in keys]

            # Plot With Adjusted Max
            plt.clf()
            plt.scatter(values1, values2, marker=".", alpha=0.66)
            maxVal = min(1.1 * max(max(values1), max(values2)), 1.0)
            plt.xlim([0,maxVal])
            plt.ylim([0,maxVal])
            plt.title(f"{label} {t}")
            plt.xlabel(label1)
            plt.ylabel(label2)
            plt.annotate(r2_score(values1, values2), (0,0))
            plt.savefig(f"results/dataset_stats/plots/{label2}_{t}.png".replace(" ", "_"))

# Get shapes
shape = {}
for (l, d) in [('Train', train_ehr_dataset), ('HALO', halo_ehr_dataset)]:
   record_lens = [len(p['visits']) for p in d]
   visit_lens = [len(v[0]) for p in d for v in p['visits'] if v[1] == []]
   num_stays = [sum([1 if v[1] == [] else 0 for v in p['visits']]) for p in d]
   visit_ages = [v[3][0] for p in d for v in p['visits'] if v[1] == []]
   visit_gaps = [v[3][0] for p in d for v in p['visits'] if v[1] != []]
   labels = [len([p for p in d if p['labels'][i] == 1]) / len(d) for i in range(config.label_vocab_size)]
   shape[l] = {'Records': len(d), 'Labels': labels, 'Record Lengths': record_lens, 'Visit Lengths': visit_lens, 'Visit Gaps': visit_gaps, 'Num Stays': num_stays, 'Visit Ages': visit_ages}

# Get Labs
labs = {}
for (l, d) in [('Train', train_ehr_dataset), ('HALO', halo_ehr_dataset)]:
    mask_probs = [[1 if m in v[1] else 0 for p in d for v in p['visits'][1:] if v[1] != []] for m in range(len(idToLab))]
    mask_probs = [np.mean(l) for l in mask_probs]
    all_lab_values = [[v[2][v[1].index(m)] for p in d for v in p['visits'][1:] if m in v[1]] for m in range(len(idToLab))]
    all_lab_values = [[random.uniform(discretization[idToLab[i]][v], discretization[idToLab[i]][v+1]) if not is_categorical_channel[idToLab[i]] else v for v in all_lab_values[i]] for i in range(len(all_lab_values))]
    lab_values = [np.nanmean(l) if l else 0 for l in all_lab_values]
    labs[l] = {'Masks': mask_probs, 'Values': lab_values, 'All Values': all_lab_values}
    
# Extract and save statistics
train_ehr_stats = generate_statistics(train_ehr_dataset)
halo_ehr_stats = generate_statistics(halo_ehr_dataset)
pickle.dump(train_ehr_stats, open('results/dataset_stats/Train_Stats.pkl', 'wb'))
pickle.dump(halo_ehr_stats, open('results/dataset_stats/HALO_Synthetic_Stats.pkl', 'wb'))
# train_ehr_stats = pickle.load(open('results/dataset_stats/Train_Stats.pkl', 'rb'))
# halo_ehr_stats = pickle.load(open('results/dataset_stats/HALO_Synthetic_Stats.pkl', 'rb'))
print(train_ehr_stats[config.label_vocab_size]["Aggregate"])
print(halo_ehr_stats[config.label_vocab_size]["Aggregate"])
print(train_ehr_stats["Label Probabilities"])
print(halo_ehr_stats["Label Probabilities"])

#Plot per-code statistics
generate_plots(train_ehr_stats, halo_ehr_stats, "Training Data", "HALO Synthetic Data")
plt.clf()

# Plot density charts
df = pd.DataFrame({'Train': pd.Series(shape['Train']['Visit Lengths']), 'HALO': pd.Series(shape['HALO']['Visit Lengths'])})
df.plot.kde()
plt.xlim(-5,45)
plt.xlabel('Number of Codes')
plt.title('Visit Lengths Probability Density')
plt.savefig(f"results/dataset_stats/plots/visit_lengths.png")
plt.clf()

df = pd.DataFrame({'Train': pd.Series(shape['Train']['Record Lengths']), 'HALO': pd.Series(shape['HALO']['Record Lengths'])})
df.plot.kde()
plt.xlim(-10,120)
plt.xlabel('Number of Visits')
plt.title('Record Lengths Probability Density')
plt.savefig(f"results/dataset_stats/plots/record_lengths.png")
plt.clf()

df = pd.DataFrame({'Train': pd.Series(shape['Train']['Visit Gaps']), 'HALO': pd.Series(shape['HALO']['Visit Gaps'])})
df.plot.kde()
plt.xlim(-1,20)
plt.xlabel('Number of Hours')
plt.title('Visit Gaps Probability Density')
plt.savefig(f"results/dataset_stats/plots/visit_gaps.png")
plt.clf()

df = pd.DataFrame({'Train': pd.Series(shape['Train']['Num Stays']), 'HALO': pd.Series(shape['HALO']['Num Stays'])})
df.plot.kde()
plt.xlim(-1,5)
plt.xlabel('Number of Stays')
plt.title('Stay Count Probability Density')
plt.savefig(f"results/dataset_stats/plots/num_stays.png")
plt.clf()

df = pd.DataFrame({'Train': pd.Series(shape['Train']['Visit Ages']), 'HALO': pd.Series(shape['HALO']['Visit Ages'])})
df.plot.kde()
plt.xlim(-1,20)
plt.xlabel('Number of Years Old')
plt.title('Visit Gaps Probability Density')
plt.savefig(f"results/dataset_stats/plots/visit_ages.png")
plt.clf()

# Plot label plots
X = np.expand_dims(np.array(shape['Train']['Labels']), 1)
y = np.expand_dims(np.array(shape['HALO']['Labels']), 1)
r2 = r2_score(y, X)
plt.scatter(X, y, c='black', label=f"HALO ({r2:.3f})", marker='x')
plt.xlim(0, 0.66)
plt.xlabel('Real Label Probability')
plt.ylim(0, 0.66)
plt.ylabel('HALO Label Probability')
plt.title('Chronic Condition Label and Demographic Probabilities')
plt.plot([0,0.66], [0,0.66], 'k-', zorder=0)
plt.legend()
plt.savefig(f'results/dataset_stats/plots/label_probs.png')
plt.clf()

# Plot lab mask plots
X = np.expand_dims(np.array(labs['Train']['Masks']), 1)
y = np.expand_dims(np.array(labs['HALO']['Masks']), 1)
r2 = r2_score(y, X)
plt.scatter(X, y, c='black', label=f"HALO ({r2:.3f})", marker='x')
plt.xlim(0, 1.0)
plt.xlabel('Real Lab Probability')
plt.ylim(0, 1.0)
plt.ylabel('HALO Lab Probability')
plt.title('Lab Presence Probabilities')
plt.plot([0,1.0], [0,1.0], 'k-', zorder=0)
plt.legend()
plt.savefig(f'results/dataset_stats/plots/lab_probs.png')
plt.clf()

# Plot lab value plots
X = np.expand_dims(np.array(labs['Train']['Values']), 1)
y = np.expand_dims(np.array(labs['HALO']['Values']), 1)
r2 = r2_score(y, X)
plt.scatter(X, y, c='black', label=f"HALO ({r2:.3f})", marker='x')
plt.xlim(0, max([max(y), max(X)]))
plt.xlabel('Real Lab Value')
plt.ylim(0, max([max(y), max(X)]))
plt.ylabel('HALO Lab Value')
plt.title('Average Lab Values')
plt.plot([0,max([max(y), max(X)])], [0,max([max(y), max(X)])], 'k-', zorder=0)
plt.legend()
plt.savefig(f'results/dataset_stats/plots/lab_values.png')

# Plot lab value probabilities
for m in range(len(idToLab)):
  try:
    df = pd.DataFrame({'Train': pd.Series(labs['Train']['All Values'][m]), 'HALO': pd.Series(labs['HALO']['All Values'][m])})
    df.plot.kde()
    plt.xlim(min(labs['Train']['All Values'][m] + labs['HALO']['All Values'][m]) - 1,max(labs['Train']['All Values'][m] + labs['HALO']['All Values'][m]) + 1)
    plt.xlabel('Lab Index')
    plt.title(f'{idToLab[m]} Index Probability Density')
    plt.savefig(f"results/dataset_stats/plots/labProbDist_{m}.png")
    plt.clf()
  except:
      print(f"No Values for {m}")
