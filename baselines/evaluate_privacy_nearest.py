import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from config import HALOConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
NUM_SAMPLES = 5000

key = 'haloCoarse'

config = HALOConfig()
train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
train_ehr_dataset = np.random.choice(train_ehr_dataset, NUM_SAMPLES)
train_ehr_dataset = [{'labels': p['labels'], 'visits': [set(v) for v in p['visits']]} for p in train_ehr_dataset]
test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
test_ehr_dataset = np.random.choice(test_ehr_dataset, NUM_SAMPLES)
test_ehr_dataset = [{'labels': p['labels'], 'visits': [set(v) for v in p['visits']]} for p in test_ehr_dataset]
synthetic_ehr_dataset = pickle.load(open(f'./results/datasets/{key}Dataset.pkl', 'rb'))
synthetic_ehr_dataset = np.random.choice([p for p in synthetic_ehr_dataset if len(p['visits']) > 0], NUM_SAMPLES)
synthetic_ehr_dataset = [{'labels': p['labels'], 'visits': [set(v) for v in p['visits']]} for p in synthetic_ehr_dataset]

def find_hamming(ehr, dataset):
    min_d = 1e10
    visits = ehr['visits']
    labels = ehr['labels']
    for p in dataset:
        d = 0 if len(visits) == len(p['visits']) else 1
        l = p['labels']
        d += ((labels + l) == 1).sum()
        for i in range(len(visits)):
            v = visits[i]
            if i >= len(p['visits']):
                d += len(v)
            else:
                v2 = p['visits'][i]
                d += len(v) + len(v2) - (2 * len(v.intersection(v2)))
                
        min_d = d if d < min_d and d > 0 else min_d
    return min_d

def calc_nnaar(train, evaluation, synthetic):
    val1 = 0
    val2 = 0
    val3 = 0
    val4 = 0
    for p in tqdm(evaluation):
        des = find_hamming(p, synthetic)
        dee = find_hamming(p, evaluation)
        if des > dee:
            val1 += 1
    
    for p in tqdm(train):
        dts = find_hamming(p, synthetic)
        dtt = find_hamming(p, train)
        if dts > dtt:
            val3 += 1

    for p in tqdm(synthetic):
        dse = find_hamming(p, evaluation)
        dst = find_hamming(p, train)
        dss = find_hamming(p, synthetic)
        if dse > dss:
            val2 += 1
        if dst > dss:
            val4 += 1

    val1 = val1 / NUM_SAMPLES
    val2 = val2 / NUM_SAMPLES
    val3 = val3 / NUM_SAMPLES
    val4 = val4 / NUM_SAMPLES

    aaes = (0.5 * val1) + (0.5 * val2)
    aaet = (0.5 * val3) + (0.5 * val4)
    return aaes - aaet

nnaar = calc_nnaar(train_ehr_dataset, test_ehr_dataset, synthetic_ehr_dataset)
results = {
    "NNAAE": nnaar
}
pickle.dump(results, open("results/privacy_evaluation/nnaar_{key}.pkl", "wb"))
print(results)