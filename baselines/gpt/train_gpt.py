import os
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from gpt import GPTModel
from config import GPTConfig

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

orig_train_ehr_dataset = pickle.load(open('../../data/trainDataset.pkl', 'rb'))
orig_val_ehr_dataset = pickle.load(open('../../data/valDataset.pkl', 'rb'))

train_ehr_dataset = []
for orig_ehr in orig_train_ehr_dataset:
  new_ehr = [config.total_vocab_size - 1] * config.n_ctx # Pad Codes
  new_ehr[0] = config.code_vocab_size + config.label_vocab_size # Start Record
  idx = 1
  
  # Add Labels
  for l in orig_ehr['labels'].nonzero()[0]:
    new_ehr[idx] = l + config.code_vocab_size
    idx += 1

  new_ehr[idx] = config.code_vocab_size + config.label_vocab_size + 1 # End Labels
  idx += 1

  # Add Visits
  for v in orig_ehr['visits']:
    for c in v:
      new_ehr[idx] = c
      idx += 1
    new_ehr[idx] = config.code_vocab_size + config.label_vocab_size + 2 # End Visit
    idx += 1

  new_ehr[idx] = config.code_vocab_size + config.label_vocab_size + 3 # End Record
  train_ehr_dataset.append(new_ehr)

val_ehr_dataset = []
for orig_ehr in orig_val_ehr_dataset:
  new_ehr = [config.total_vocab_size - 1] * config.n_ctx # Pad Codes
  new_ehr[0] = config.code_vocab_size + config.label_vocab_size # Start Record
  idx = 1
  
  # Add Labels
  for l in orig_ehr['labels'].nonzero()[0]:
    new_ehr[idx] = l + config.code_vocab_size
    idx += 1

  new_ehr[idx] = config.code_vocab_size + config.label_vocab_size + 1 # End Labels
  idx += 1

  # Add Visits
  for v in orig_ehr['visits']:
    for c in v:
      new_ehr[idx] = c
      idx += 1
    new_ehr[idx] = config.code_vocab_size + config.label_vocab_size + 2 # End Visit
    idx += 1

  new_ehr[idx] = config.code_vocab_size + config.label_vocab_size + 3 # End Record
  val_ehr_dataset.append(new_ehr)

def get_batch(loc, batch_size, mode):
  if mode == 'train':
    ehr = train_ehr_dataset[loc:loc+batch_size]
  elif mode == 'valid':
    ehr = val_ehr_dataset[loc:loc+batch_size]
  else:
    ehr = test_ehr_dataset[loc:loc+batch_size]
    
  batch_ehr = np.array(ehr)
  return batch_ehr

def shuffle_training_data(train_ehr_dataset):
  np.random.shuffle(train_ehr_dataset)

model = GPTModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists("../../save/gpt_model"):
  print("Loading previous model")
  checkpoint = torch.load('../../save/gpt_model', map_location=torch.device(device))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  model.set_tied()

# Train
global_loss = 1e10
for e in tqdm(range(config.epoch)):
  shuffle_training_data(train_ehr_dataset)
  for i in range(0, len(train_ehr_dataset), config.batch_size):
    model.train()
    
    batch_ehr = get_batch(i, config.batch_size, 'train')
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.long).to(device)
    
    optimizer.zero_grad()
    loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr)
    loss.backward()
    optimizer.step()
    
    if i % (100*config.batch_size) == 0:
      print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss))
    if i % (250*config.batch_size) == 0:
      if i == 0:
        continue
    
      model.eval()
      with torch.no_grad():
        val_l = []
        for v_i in range(0, len(val_ehr_dataset), config.batch_size):
          batch_ehr = get_batch(v_i, config.batch_size, 'valid')
          batch_ehr = torch.tensor(batch_ehr, dtype=torch.long).to(device)
  
          val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr)
          val_l.append((val_loss).cpu().detach().numpy())
          
        cur_val_loss = np.mean(val_l)
        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
        if cur_val_loss < global_loss:
          global_loss = cur_val_loss
          state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': i
            }
          torch.save(state, '../../save/gpt_model')
          print('\n------------ Save best model ------------\n')