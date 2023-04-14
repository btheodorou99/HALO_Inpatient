import pickle
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import HALOConfig
from sklearn.metrics import r2_score

config = HALOConfig()

label_mapping = pickle.load(open("./data/idToLabel.pkl", "rb"))
label_mapping[25] = 'Overall'

config = HALOConfig()
train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
halo_ehr_dataset = pickle.load(open('./results/datasets/haloDataset.pkl', 'rb'))
halo_ehr_dataset = [{'labels': p['labels'], 'visits': p['visits']} for p in halo_ehr_dataset if len(p['visits']) > 0]
haloCoarse_ehr_dataset = pickle.load(open('./results/datasets/haloCoarseDataset.pkl', 'rb'))
haloCoarse_ehr_dataset = [{'labels': p['labels'], 'visits': p['visits']} for p in haloCoarse_ehr_dataset if len(p['visits']) > 0]
lstm_ehr_dataset = pickle.load(open('./results/datasets/lstmDataset.pkl', 'rb'))
lstm_ehr_dataset = [{'labels': p['labels'], 'visits': p['visits']} for p in lstm_ehr_dataset if len(p['visits']) > 0]
synteg_ehr_dataset = pickle.load(open('./results/datasets/syntegDataset.pkl', 'rb'))
synteg_ehr_dataset = [{'labels': p['labels'], 'visits': p['visits']} for p in synteg_ehr_dataset if len(p['visits']) > 0]
eva_ehr_dataset = pickle.load(open('./results/datasets/evaDataset.pkl', 'rb'))
eva_ehr_dataset = [{'labels': p['labels'], 'visits': p['visits']} for p in eva_ehr_dataset if len(p['visits']) > 0]
gpt_ehr_dataset = pickle.load(open('./results/datasets/gpt_baseDataset.pkl', 'rb'))
gpt_ehr_dataset = [{'labels': p['labels'], 'visits': p['visits']} for p in gpt_ehr_dataset if len(p['visits']) > 0]

stats = {}
for (l,d) in [('Train', train_ehr_dataset), ('HALO', halo_ehr_dataset), ('HALO Coarse', haloCoarse_ehr_dataset), ('LSTM', lstm_ehr_dataset), ('SynTEG', synteg_ehr_dataset), ('EVA', eva_ehr_dataset), ('GPT', gpt_ehr_dataset)]:
  aggregate_stats = {}
  label_probs = [len([p for p in d if p['labels'][i] == 1])/len(d) for i in range(25)]
  aggregate_stats['Labels'] = label_probs
  record_lens = [len(p['visits']) for p in d]
  visit_lens = [len(v) for p in d for v in p['visits']]
  aggregate_stats['Record Lengths'] = record_lens
  aggregate_stats['Visit Lengths'] = visit_lens
  stats[l] = aggregate_stats

pickle.dump(stats, open('results/shape.pkl', 'wb'))

def generate_statistics(ehr_dataset):
    stats = {}
    label_counts = {}
    for i in tqdm(range(config.label_vocab_size + 1)):
        label_stats = {}
        d = [p for p in ehr_dataset if p['labels'][i] == 1] if i < config.label_vocab_size else ehr_dataset
        label_counts[label_mapping[i]] = len(d)


    for i in tqdm(range(config.label_vocab_size, config.label_vocab_size + 1)):
        label_stats = {}
        d = [p for p in ehr_dataset if p['labels'][i] == 1] if i < config.label_vocab_size else ehr_dataset
        label_counts[label_mapping[i]] = len(d)

        aggregate_stats = {}
        record_lens = [len(p['visits']) for p in d]
        visit_lens = [len(v) for p in d for v in p['visits']]
        avg_record_len = np.mean(record_lens)
        std_record_len = np.std(record_lens)
        avg_visit_len = np.mean(visit_lens)
        std_visit_len = np.std(visit_lens)
        aggregate_stats["Record Length Mean"] = avg_record_len
        aggregate_stats["Record Length Standard Deviation"] = std_record_len
        aggregate_stats["Visit Length Mean"] = avg_visit_len
        aggregate_stats["Visit Length Standard Deviation"] = std_visit_len
        label_stats["Aggregate"] = aggregate_stats

        code_stats = {}
        n_records = len(record_lens)
        n_visits = len(visit_lens)
        record_code_counts = {}
        visit_code_counts = {}
        record_bigram_counts = {}
        visit_bigram_counts = {}
        record_sequential_bigram_counts = {}
        visit_sequential_bigram_counts = {}
        for p in d:
            patient_codes = set()
            patient_bigrams = set()
            sequential_bigrams = set()
            for j in range(len(p['visits'])):
                v = p['visits'][j]
                for c in v:
                    visit_code_counts[c] = 1 if c not in visit_code_counts else visit_code_counts[c] + 1
                    patient_codes.add(c)
                for cs in itertools.combinations(v,2):
                    cs = list(cs)
                    cs.sort()
                    cs = tuple(cs)
                    visit_bigram_counts[cs] = 1 if cs not in visit_bigram_counts else visit_bigram_counts[cs] + 1
                    patient_bigrams.add(cs)
                if j > 0:
                    v0 = p['visits'][j-1]
                    for c0 in v0:
                        for c in v:
                            sc = (c0, c)
                            visit_sequential_bigram_counts[sc] = 1 if sc not in visit_sequential_bigram_counts else visit_sequential_bigram_counts[sc] + 1
                            sequential_bigrams.add(sc)
            for c in patient_codes:
                record_code_counts[c] = 1 if c not in record_code_counts else record_code_counts[c] + 1
            for cs in patient_bigrams:
                record_bigram_counts[cs] = 1 if cs not in record_bigram_counts else record_bigram_counts[cs] + 1
            for sc in sequential_bigrams:
                record_sequential_bigram_counts[sc] = 1 if sc not in record_sequential_bigram_counts else record_sequential_bigram_counts[sc] + 1
        record_code_probs = {c: record_code_counts[c]/n_records for c in record_code_counts}
        visit_code_probs = {c: visit_code_counts[c]/n_visits for c in visit_code_counts}
        record_bigram_probs = {cs: record_bigram_counts[cs]/n_records for cs in record_bigram_counts}
        visit_bigram_probs = {cs: visit_bigram_counts[cs]/n_visits for cs in visit_bigram_counts}
        record_sequential_bigram_probs = {sc: record_sequential_bigram_counts[sc]/n_records for sc in record_sequential_bigram_counts}
        visit_sequential_bigram_probs = {sc: visit_sequential_bigram_counts[sc]/(n_visits - len(d)) for sc in visit_sequential_bigram_counts}
        code_stats["Per Record Code Probabilities"] = record_code_probs
        code_stats["Per Visit Code Probabilities"] = visit_code_probs
        code_stats["Per Record Bigram Probabilities"] = record_bigram_probs
        code_stats["Per Visit Bigram Probabilities"] = visit_bigram_probs
        code_stats["Per Record Sequential Visit Bigram Probabilities"] = record_sequential_bigram_probs
        code_stats["Per Visit Sequential Visit Bigram Probabilities"] = visit_sequential_bigram_probs
        label_stats["Probabilities"] = code_stats
        stats[label_mapping[i]] = label_stats
    label_probs = {l: label_counts[l]/n_records for l in label_counts}
    stats["Label Probabilities"] = label_probs
    return stats
    
def generate_plots(stats1, stats2, label1, label2, types=["Per Record Code Probabilities", "Per Visit Code Probabilities", "Per Record Bigram Probabilities", "Per Visit Bigram Probabilities", "Per Record Sequential Visit Bigram Probabilities", "Per Visit Sequential Visit Bigram Probabilities"]):
    for i in tqdm(range(config.label_vocab_size, config.label_vocab_size + 1)):
        print("\n")
        label = label_mapping[i]
        data1 = stats1[label]["Probabilities"]
        data2 = stats2[label]["Probabilities"]
        for t in types:
            probs1 = data1[t]
            probs2 = data2[t]
            keys = set(probs1.keys()).union(set(probs2.keys()))
            values1 = [probs1[k] if k in probs1 else 0 for k in keys]
            values2 = [probs2[k] if k in probs2 else 0 for k in keys]
            r2score = r2_score(values1, values2)
            print(f'{t}: {r2score}')

            plt.clf()
            plt.scatter(values1, values2, marker=".", alpha=0.66)
            maxVal = min(1.1 * max(max(values1), max(values2)), 1.0)
            plt.xlim([0,maxVal])
            plt.ylim([0,maxVal])
            plt.title(f"{label} {t}")
            plt.xlabel(label1)
            plt.ylabel(label2)
            plt.savefig(f"results/dataset_stats/plots/{label2}_{label.split(':')[-1]}_{t}_adjMax".replace(" ", "_"))


# Extract and save statistics
train_ehr_stats = generate_statistics(train_ehr_dataset)
halo_ehr_stats = generate_statistics(halo_ehr_dataset)
haloCoarse_ehr_stats = generate_statistics(haloCoarse_ehr_dataset)
lstm_ehr_stats = generate_statistics(lstm_ehr_dataset)
synteg_ehr_stats = generate_statistics(synteg_ehr_dataset)
eva_ehr_stats = generate_statistics(eva_ehr_dataset)
gpt_ehr_stats = generate_statistics(gpt_ehr_dataset)
pickle.dump(train_ehr_stats, open('results/dataset_stats/Train_Stats.pkl', 'wb'))
pickle.dump(halo_ehr_stats, open('results/dataset_stats/HALO_Synthetic_Stats.pkl', 'wb'))
pickle.dump(haloCoarse_ehr_stats, open('results/dataset_stats/GPT_Synthetic_Stats.pkl', 'wb'))
pickle.dump(lstm_ehr_stats, open('results/dataset_stats/LSTM_Synthetic_Stats.pkl', 'wb'))
pickle.dump(synteg_ehr_stats, open('results/dataset_stats/SynTEG_Synthetic_Stats.pkl', 'wb'))
pickle.dump(eva_ehr_stats, open('results/dataset_stats/EVA_Synthetic_Stats.pkl', 'wb'))
pickle.dump(gpt_ehr_stats, open('results/dataset_stats/GPT_Synthetic_Stats.pkl', 'wb'))
print(train_ehr_stats["Overall"]["Aggregate"])
print(halo_ehr_stats["Overall"]["Aggregate"])
print(haloCoarse_ehr_stats["Overall"]["Aggregate"])
print(lstm_ehr_stats["Overall"]["Aggregate"])
print(synteg_ehr_stats["Overall"]["Aggregate"])
print(eva_ehr_stats["Overall"]["Aggregate"])
print(gpt_ehr_stats["Overall"]["Aggregate"])
print(train_ehr_stats["Label Probabilities"])
print(halo_ehr_stats["Label Probabilities"])
print(haloCoarse_ehr_stats["Label Probabilities"])
print(lstm_ehr_stats["Label Probabilities"])
print(synteg_ehr_stats["Label Probabilities"])
print(eva_ehr_stats["Label Probabilities"])
print(gpt_ehr_stats["Label Probabilities"])

# Plot per-code statistics
generate_plots(train_ehr_stats, halo_ehr_stats, "MIMIC Training Data", "HALO Synthetic Data")
generate_plots(train_ehr_stats, haloCoarse_ehr_stats, "MIMIC Training Data", "HALO Coarse Synthetic Data")
generate_plots(train_ehr_stats, lstm_ehr_stats, "MIMIC Training Data", "LSTM Synthetic Data")
generate_plots(train_ehr_stats, synteg_ehr_stats, "MIMIC Training Data", "SynTEG Synthetic Data")
generate_plots(train_ehr_stats, eva_ehr_stats, "MIMIC Training Data", "EVA Synthetic Data")
generate_plots(train_ehr_stats, gpt_ehr_stats, "MIMIC Training Data", "GPT Synthetic Data")