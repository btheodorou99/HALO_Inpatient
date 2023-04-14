import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from config import HALOConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

local_rank = -1
fp16 = False
if local_rank == -1:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
else:
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda", local_rank)
  n_gpu = 1
  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
  torch.distributed.init_process_group(backend='nccl')
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

config = HALOConfig()
test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
test_ehr_dataset = [{'labels': p['labels'], 'visits': set([c for v in p['visits'] for c in v])} for p in test_ehr_dataset]
train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
train_ehr_dataset = [{'labels': p['labels'], 'visits': set([c for v in p['visits'] for c in v])} for p in train_ehr_dataset]
train_ehr_dataset = np.random.choice(train_ehr_dataset, len(test_ehr_dataset), replace=False)
synthetic_ehr_dataset = pickle.load(open('./results/datasets/haloDataset.pkl', 'rb'))
synthetic_ehr_dataset = [{'labels': p['labels'], 'visits': set([c for v in p['visits'] for c in v])} for p in synthetic_ehr_dataset if len(p['visits']) > 0]
synthetic_ehr_dataset = np.random.choice(synthetic_ehr_dataset, len(test_ehr_dataset), replace=False)

common_codes = set([cd for cd, _ in Counter([c for p in train_ehr_dataset for c in p['visits']]).most_common()[0:100]])

test_ehr_dataset = [{'labels': set([c for c in p['labels'].nonzero()[0].tolist()] + [c + config.label_vocab_size for c in p['visits'] if c in common_codes]), 'codes': set([c for c in p['visits'] if c not in common_codes])} for p in test_ehr_dataset]
train_ehr_dataset = [{'labels': set([c for c in p['labels'].nonzero()[0].tolist()] + [c + config.label_vocab_size for c in p['visits'] if c in common_codes]), 'codes': set([c for c in p['visits'] if c not in common_codes])} for p in train_ehr_dataset]
synthetic_ehr_dataset = [{'labels': set([c for c in p['labels'].nonzero()[0].tolist()] + [c + config.label_vocab_size for c in p['visits'] if c in common_codes]), 'codes': set([c for c in p['visits'] if c not in common_codes])} for p in synthetic_ehr_dataset]

def calc_dist(lab1, lab2):
    return len(lab1.union(lab2)) - len(lab1.intersection(lab2))

def find_closest(patient, data, k):
    cond = patient['labels']
    dists = [(calc_dist(cond, ehr['labels']), ehr['codes']) for ehr in data]
    dists.sort(key= lambda x: x[0], reverse=False)
    options = [o[1] for o in dists[:k]]
    return options

def calc_attribute_risk(train_dataset, reference_dataset, k):
    tp = 0
    fp = 0
    fn = 0
    for p in tqdm(train_dataset):
        closest_k = find_closest(p, reference_dataset, k)
        pred_codes = set([cd for cd, cnt in Counter([c for p in closest_k for c in p]).items() if cnt > k/2])
        true_pos = len(pred_codes.intersection(p['codes']))
        false_pos = len(pred_codes) - true_pos 
        false_neg = len(p['codes']) - true_pos
        tp += true_pos
        fp += false_pos
        fn += false_neg
        
    f1 = tp / (tp + (0.5 * (fp + fn)))
    return f1

K = 1
att_risk = calc_attribute_risk(train_ehr_dataset, synthetic_ehr_dataset, K)
baseline_risk = calc_attribute_risk(train_ehr_dataset, test_ehr_dataset, K)
results = {
    "Attribute Attack F1 Score": att_risk,
    "Baseline Attack F1 Score": baseline_risk
}
pickle.dump(results, open("results/privacy_evaluation/attribute_inference.pkl", "wb"))
print(results)