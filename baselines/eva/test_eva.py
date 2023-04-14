import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from eva import Eva
from config import EVAConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = EVAConfig()

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
test_ehr_dataset = pickle.load(open('../../data/testDataset.pkl', 'rb'))
index_to_code = pickle.load(open("../../data/indexToCode.pkl", "rb"))
id_to_label = pickle.load(open("../../data/idToLabel.pkl", "rb"))
train_c = set([c for p in train_ehr_dataset for v in p['visits'] for c in v])
test_ehr_dataset = [{'labels': p['labels'], 'visits': [[c for c in v if c in train_c] for v in p['visits']]} for p in test_ehr_dataset]

# Add the labels to the index_to_code mapping
for k, l in id_to_label.items():
  index_to_code[config.code_vocab_size+k] = f"Chronic Condition: {l}"
  
# Add the labels to the index_to_code mapping
index_to_code[config.code_vocab_size] = "Chronic Condition: Alzheimer or related disorders or senile"
index_to_code[config.code_vocab_size+1] = "Chronic Condition: Heart Failure"
index_to_code[config.code_vocab_size+2] = "Chronic Condition: Chronic Kidney Disease"
index_to_code[config.code_vocab_size+3] = "Chronic Condition: Cancer"
index_to_code[config.code_vocab_size+4] = "Chronic Condition: Chronic Obstructive Pulmonary Disease"
index_to_code[config.code_vocab_size+5] = "Chronic Condition: Depression"
index_to_code[config.code_vocab_size+6] = "Chronic Condition: Diabetes"
index_to_code[config.code_vocab_size+7] = "Chronic Condition: Ischemic Heart Disease"
index_to_code[config.code_vocab_size+8] = "Chronic Condition: Osteoporosis"
index_to_code[config.code_vocab_size+9] = "Chronic Condition: rheumatoid arthritis and osteoarthritis (RA/OA)"
index_to_code[config.code_vocab_size+10] = "Chronic Condition: Stroke/transient Ischemic Attack"

def get_batch(loc, batch_size, mode):
  # EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
  #   Where each patient P is [V_1, V_2, ... , V_j]
  #     Where each visit V is [C_1, C_2, ... , C_k]
  #   And where each Label L is a binary vector [L_1 ... L_n]
  if mode == 'train':
    ehr = train_ehr_dataset[loc:loc+batch_size]
  elif mode == 'valid':
    ehr = val_ehr_dataset[loc:loc+batch_size]
  else:
    ehr = test_ehr_dataset[loc:loc+batch_size]
    
  batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
  batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
  batch_lens = np.zeros(len(ehr))
  for i, p in enumerate(ehr):
    visits = p['visits']
    batch_lens[i] = len(visits)
    for j, v in enumerate(visits):
      batch_ehr[i,j+2][v] = 1
      batch_mask[i,j+2] = 1
    batch_ehr[i,1,config.code_vocab_size:config.code_vocab_size+config.label_vocab_size] = np.array(p['labels']) # Set the patient labels
    batch_ehr[i,len(visits)+1,config.code_vocab_size+config.label_vocab_size+1] = 1 # Set the final visit to have the end token
    batch_ehr[i,len(visits)+2:,config.code_vocab_size+config.label_vocab_size+2] = 1 # Set the rest to the padded visit token
  
  batch_mask[:,1] = 1 # Set the mask to cover the labels
  batch_ehr[:,0,config.code_vocab_size+config.label_vocab_size] = 1 # Set the first visits to be the start token
  batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
  return batch_ehr, batch_lens, batch_mask

model = Eva(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

checkpoint = torch.load('../../save/eva_model', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

def convert_ehr(ehrs, index_to_code=None):
  ehr_outputs = []
  for i in range(len(ehrs)):
    ehr = ehrs[i]
    ehr_output = []
    labels_output = ehr[0][config.code_vocab_size:config.code_vocab_size+config.label_vocab_size]
    if index_to_code is not None:
      labels_output = [index_to_code[idx + config.code_vocab_size] for idx in np.nonzero(labels_output)[0]]
    for j in range(1, len(ehr)):
      visit = ehr[j]
      visit_output = []
      indices = np.nonzero(visit)
      if len(indices) > 0:
        indices = indices[0]
      else:
        continue
      end = False
      for idx in indices:
        if idx < config.code_vocab_size: 
          visit_output.append(index_to_code[idx] if index_to_code is not None else idx)
        elif idx == config.code_vocab_size+config.label_vocab_size+1:
          end = True
      if visit_output != []:
        ehr_output.append(visit_output)
      if end:
        break
    ehr_outputs.append({'visits': ehr_output, 'labels': labels_output})
  ehr = None
  ehr_output = None
  labels_output = None
  visit = None
  visit_output = None
  indices = None
  return ehr_outputs

# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
for i in tqdm(range(0, len(train_ehr_dataset), config.batch_size)):
  bs = min([len(train_ehr_dataset)-i, config.batch_size])
  batch_synthetic_ehrs = model.sample(bs, device)
  batch_synthetic_ehrs = torch.bernoulli(batch_synthetic_ehrs)
  batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs.detach().cpu().numpy())
  synthetic_ehr_dataset += batch_synthetic_ehrs

pickle.dump(synthetic_ehr_dataset, open(f'../../results/datasets/evaDataset.pkl', 'wb'))