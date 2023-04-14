import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from config import SyntegConfig
from synteg import DependencyModel
from sklearn.model_selection import train_test_split

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = SyntegConfig()

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

train_ehr_dataset = pickle.load(open('../../data/trainDataset.pkl', 'rb'))

def get_batch(loc, batch_size, mode):
    # EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
    #   Where each patient P is [V_1, V_2, ... , V_j]
    #     Where each visit V is [C_1, C_2, ... , C_k]
    #   And where each Label L is a binary vector [L_1 ... L_11]
    if mode == 'train':
        ehr = train_ehr_dataset[loc:loc+batch_size]
    elif mode == 'valid':
        ehr = val_ehr_dataset[loc:loc+batch_size]
    else:
        ehr = test_ehr_dataset[loc:loc+batch_size]
    
    batch_ehr = np.zeros((len(ehr), config.max_num_visit, config.max_length_visit))
    batch_ehr[:,:,:] = config.vocab_dim # Initialize each code to the padding code
    batch_lens = np.ones((len(ehr), config.max_num_visit, 1))
    batch_mask = np.zeros((len(ehr), config.max_num_visit, 1))
    batch_num_visits = np.zeros(len(ehr))
    for i, p in enumerate(ehr):
        visits = p['visits']
        for j, v in enumerate(visits):
            batch_mask[i,j+2] = 1
            batch_lens[i,j+2] = len(v) + 1
            for k, c in enumerate(v):
                batch_ehr[i,j+2,k+1] = c
        batch_ehr[i,j+2,len(v)+1] = config.code_vocab_dim + config.label_vocab_dim + 1 # Set the last code in the last visit to be the end record code
        batch_lens[i,j+2] = len(v) + 2
        for l_idx, l in enumerate(np.nonzero(p['labels'])[0]):
            batch_ehr[i,1,l_idx+1] = config.code_vocab_dim + l
            batch_lens[i,1] = l_idx+2
        batch_num_visits[i] = len(visits)
    
    batch_mask[:,1] = 1  # Set the mask to cover the labels
    batch_ehr[:,:,0] = config.code_vocab_dim + config.label_vocab_dim # Set the first code in each visit to be the start/class token
    batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
    return batch_ehr, batch_lens, batch_mask, batch_num_visits
    
LR = 1e-4
model = DependencyModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
checkpoint = torch.load("../../save/synteg_dependency_model", map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

condition_dataset = []
for i in tqdm(range(0, len(train_ehr_dataset), config.dependency_batchsize)):
    model.train()
    
    batch_ehr, batch_lens, _, batch_num_visits = get_batch(i, config.dependency_batchsize, 'train')
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.int).to(device) # bs * visit * code
    batch_lens = torch.tensor(batch_lens, dtype=torch.int).to(device) # bs * visit
    condition_vector = model(batch_ehr, batch_lens, export=True) # bs * visit * 256
    batch_ehr = batch_ehr.detach().cpu().numpy()
    condition_vector = condition_vector.detach().cpu().numpy()

    for b, num_visits in enumerate(batch_num_visits-1):
        for v in range(int(num_visits+1)):
            ehr_tmp = batch_ehr[b, v+1, :]
            condition_vector_tmp = condition_vector[b, v, :]
            datum = {"ehr": ehr_tmp, "condition": condition_vector_tmp}
            condition_dataset.append(datum)

pickle.dump(condition_dataset, open("data/conditionDataset.pkl", "wb"))
