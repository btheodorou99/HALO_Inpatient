import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from gpt import GPTModel
from config import GPTConfig
import torch.nn.functional as F

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = GPTConfig()

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
index_to_code = pickle.load(open("../../data/indexToCode.pkl", "rb"))

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

model = GPTModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

checkpoint = torch.load('../../save/gpt_model', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

def sample_sequence(model, length, context, batch_size=None, device='cuda', sample=True):
  context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
  prev = context
  ehr = context
  past = None
  with torch.no_grad():
    for _ in range(length):
      code_logits, past = model(prev, past=past)
      code_logits = code_logits[:, -1, :]
      log_probs = F.softmax(code_logits, dim=-1)
      if sample:
        prev = torch.multinomial(log_probs, num_samples=1)
      else:
        prev = torch.argmax(log_probs, dim=1)
      ehr = torch.cat((ehr, prev), dim=1)
      
      if all([config.code_vocab_size + config.label_vocab_size + 3 in ehr[i] for i in range(batch_size)]): # early stopping
        break
  ehr = ehr.cpu().detach().numpy()
  next = None
  prev = None
  return ehr

def convert_ehr(ehrs, index_to_code=None):
  ehr_outputs = []
  for i in range(len(ehrs)):
    ehr = ehrs[i]
    ehr_output = []
    visit_output = []
    labels_output = np.zeros(config.label_vocab_size)
    started_visits = False
    for j in range(1, len(ehr)):
      code = ehr[j]
      if not started_visits:
        if code == config.code_vocab_size + config.label_vocab_size + 1:
          started_visits = True
        elif code >= config.code_vocab_size and code < config.code_vocab_size + config.label_vocab_size:
          labels_output[code - config.code_vocab_size] = 1
          
      else:
        if code < config.code_vocab_size:
          if code not in visit_output:
            visit_output.append(index_to_code[code] if index_to_code is not None else code)
        elif code == config.code_vocab_size + config.label_vocab_size + 2:
          if visit_output != []:
            ehr_output.append(visit_output)
            visit_output = []
        elif code == config.code_vocab_size + config.label_vocab_size + 3:
          break
        
    if visit_output != []:
      ehr_output.append(visit_output)
      
    if index_to_code is not None:
      labels_output = [index_to_code[idx + config.code_vocab_size] for idx in np.nonzero(labels_output)[0]]

    ehr_outputs.append({'visits': ehr_output, 'labels': labels_output})
  ehr = None
  ehr_output = None
  labels_output = None
  visit_output = None
  return ehr_outputs

# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
stoken = [config.code_vocab_size+config.label_vocab_size]
for i in tqdm(range(0, len(train_ehr_dataset), 2*config.batch_size)):
  bs = min([len(train_ehr_dataset)-i, 2*config.batch_size])
  batch_synthetic_ehrs = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
  batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
  synthetic_ehr_dataset += batch_synthetic_ehrs

pickle.dump(synthetic_ehr_dataset, open(f'../../results/datasets/gptDataset.pkl', 'wb'))