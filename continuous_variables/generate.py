import json
import torch
import pickle
import random
import numpy as np
from sys import argv
from tqdm import tqdm
from discretized_model import HALOModel
from discretized_config import HALOConfig

config = HALOConfig()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = HALOModel(config).to(device)
checkpoint = torch.load('./save/halo_model', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])

labelProbs = pickle.load(open('./discretized_data/labelProbs.pkl', 'rb'))
idxToId = pickle.load(open('discretized_data/idxToId.pkl', 'rb'))
idToLab = pickle.load(open('discretized_data/idToLab.pkl', 'rb'))
beginPos = pickle.load(open('discretized_data/beginPos.pkl', 'rb')) 
isCategorical = pickle.load(open('discretized_data/isCategorical.pkl', 'rb')) 
possible_values = pickle.load(open('discretized_data/possibleValues.pkl', 'rb'))
discretization = pickle.load(open('discretized_data/discretization.pkl', 'rb'))
indexToCode = pickle.load(open('discretized_data/indexToCode.pkl', 'rb'))
idToLabel = pickle.load(open('discretized_data/idToLabel.pkl', 'rb'))

def sample_sequence(model, length, context, batch_size, device='cuda', sample=True):
  empty = torch.zeros((1,1,config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
  context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
  prev = context.unsqueeze(1)
  context = None
  with torch.no_grad():
    for _ in range(length-1):
      prev = model.sample(torch.cat((prev,empty), dim=1), sample)
      if torch.sum(torch.sum(prev[:,:,config.code_vocab_size+config.lab_vocab_size+config.continuous_vocab_size+config.label_vocab_size+1], dim=1).bool().int(), dim=0).item() == batch_size:
        break
  ehr = prev.cpu().detach().numpy()
  prev = None
  empty = None
  return ehr

def convert_ehr(ehrs, index_to_code=None):
  ehr_outputs = []
  for i in range(len(ehrs)):
    ehr = ehrs[i]
    ehr_output = []

    labels_output = ehr[1][config.code_vocab_size+config.lab_vocab_size+config.continuous_vocab_size:config.code_vocab_size+config.lab_vocab_size+config.continuous_vocab_size+config.label_vocab_size]
    if index_to_code is not None:
      labels_output = [idToLabel[idx] for idx in np.nonzero(labels_output)[0]]

    for j in range(2, len(ehr)):
      visit = ehr[j]
      visit_output = []
      lab_mask = []
      lab_values = []
      cont_idx = -1
      indices = np.nonzero(visit)[0]
      end = False
      for idx in indices:
        if idx < config.code_vocab_size: 
          visit_output.append(index_to_code[idx] if index_to_code is not None else idx)
        elif idx < config.code_vocab_size+config.lab_vocab_size:
          lab_idx = idx - (config.code_vocab_size)
          lab_num = idxToId[lab_idx]
          if lab_num in lab_mask:
            continue
          else:
            lab_mask.append(lab_num)
            lab_values.append(lab_idx - beginPos[lab_num])
        elif idx < config.code_vocab_size+config.lab_vocab_size+config.continuous_vocab_size:
            cont_idx = cont_idx if cont_idx != -1 else idx - (config.code_vocab_size+config.lab_vocab_size)
        elif idx == config.code_vocab_size+config.lab_vocab_size+config.continuous_vocab_size+config.label_vocab_size+1:
          end = True
      
      if cont_idx == -1:
        cont_idx = random.randint(0, config.continuous_vocab_size) - 1
      if visit_output != [] or lab_mask != []:
        ehr_output.append((visit_output, lab_mask, lab_values, [cont_idx]))
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
totEHRs = len(pickle.load(open('data/trainDataset.pkl', 'rb')))
stoken = np.zeros(config.total_vocab_size)
stoken[config.code_vocab_size+config.lab_vocab_size+config.continuous_vocab_size+config.label_vocab_size] = 1
synthetic_ehr_dataset = []
for i in tqdm(range(0, totEHRs, config.sample_batch_size)):
  bs = min([totEHRs-i, config.sample_batch_size])
  batch_synthetic_ehrs = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
  batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
  synthetic_ehr_dataset += batch_synthetic_ehrs

pickle.dump(synthetic_ehr_dataset, open(f'./results/datasets/haloDataset.pkl', 'wb'))
