import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from lstm import LSTMBaseline
from config import LSTMConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = LSTMConfig()

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
  for i, p in enumerate(ehr):
    visits = p['visits']
    for j, v in enumerate(visits):
      batch_ehr[i,j+2][v] = 1
      batch_mask[i,j+2] = 1
    batch_ehr[i,1,config.code_vocab_size:config.code_vocab_size+config.label_vocab_size] = np.array(p['labels']) # Set the patient labels
    batch_ehr[i,len(visits)+1,config.code_vocab_size+config.label_vocab_size+1] = 1 # Set the final visit to have the end token
    batch_ehr[i,len(visits)+2:,config.code_vocab_size+config.label_vocab_size+2] = 1 # Set the rest to the padded visit token
  
  batch_mask[:,1] = 1 # Set the mask to cover the labels
  batch_ehr[:,0,config.code_vocab_size+config.label_vocab_size] = 1 # Set the first visits to be the start token
  batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
  return batch_ehr, batch_mask

def conf_mat(x, y):
  totaltrue = np.sum(x)
  totalfalse = len(x) - totaltrue
  truepos, totalpos = np.sum(x & y), np.sum(y)
  falsepos = totalpos - truepos
  return np.array([[totalfalse - falsepos, falsepos], #true negatives, false positives
                   [totaltrue - truepos, truepos]]) #false negatives, true positives

model = LSTMBaseline(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

checkpoint = torch.load('../../save/lstm_model', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

confusion_matrix = None
probability_list = []
loss_list = []
n_visits = 0
n_pos_codes = 0
n_total_codes = 0
model.eval()
with torch.no_grad():
  for v_i in tqdm(range(0, len(test_ehr_dataset), config.batch_size)):
    # Get batch inputs
    batch_ehr, batch_mask = get_batch(v_i, config.batch_size, 'test')
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
    
    # Get batch outputs
    test_loss, predictions, labels = model(batch_ehr, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
    batch_mask_array = batch_mask.squeeze().cpu().detach().numpy()
    rounded_preds = np.around(predictions.squeeze().cpu().detach().numpy()).transpose((2,0,1))
    rounded_preds = rounded_preds + batch_mask_array - 1 # Setting the masked visits to be -1 to be ignored by the confusion matrix
    rounded_preds = rounded_preds.flatten()
    true_values = labels.squeeze().cpu().detach().numpy().transpose((2,0,1))
    true_values = true_values + batch_mask_array - 1 # Setting the masked visits to be -1 to be ignored by the confusion matrix
    true_values = true_values.flatten()

    # Append test lost
    loss_list.append(test_loss.cpu().detach().numpy())
    
    # Add confusion matrix
    batch_cmatrix = conf_mat(true_values == 1, rounded_preds == 1)
    batch_cmatrix[0][0] = torch.sum(batch_mask) * config.total_vocab_size - batch_cmatrix[0][1] - batch_cmatrix[1][0] - batch_cmatrix[1][1] # Remove the masked values
    confusion_matrix = batch_cmatrix if confusion_matrix is None else confusion_matrix + batch_cmatrix
    
    # Add number of visits and codes
    n_visits += torch.sum(batch_mask)
    n_pos_codes += torch.sum(labels)
    n_total_codes += torch.sum(batch_mask) * config.total_vocab_size

    # Calculate and add probabilities 
    # Note that the masked codes will have probability 1 and be ignored
    label_probs = torch.abs(labels - 1.0 + predictions)
    log_prob = torch.sum(torch.log(label_probs)).cpu().item()
    probability_list.append(log_prob)

  n_visits = n_visits.cpu().item()
  n_pos_codes = n_pos_codes.cpu().item()
  n_total_codes = n_total_codes.cpu().item()

# Save intermediate values in case of error
intermediate = {}
intermediate["Losses"] = loss_list
intermediate["Confusion Matrix"] = confusion_matrix
intermediate["Probabilities"] = probability_list
intermediate["Num Visits"] = n_visits
intermediate["Num Positive Codes"] = n_pos_codes
intermediate["Num Total Codes"] = n_total_codes
pickle.dump(intermediate, open("../../results/testing_stats/LSTMBaseline_intermediate_results.pkl", "wb"))

#Extract, save, and display test metrics
avg_loss = np.mean(loss_list)
tn, fp, fn, tp = confusion_matrix.ravel()
acc = (tn + tp)/(tn+fp+fn+tp)
prc = tp/(tp+fp)
rec = tp/(tp+fn)
f1 = (2 * prc * rec)/(prc + rec)
log_probability = np.sum(probability_list)
pp_visit = np.exp(-log_probability/n_visits)
pp_positive = np.exp(-log_probability/n_pos_codes)
pp_possible = np.exp(-log_probability/n_total_codes)

metrics_dict = {}
metrics_dict['Test Loss'] = avg_loss
metrics_dict['Confusion Matrix'] = confusion_matrix
metrics_dict['Accuracy'] = acc
metrics_dict['Precision'] = prc
metrics_dict['Recall'] = rec
metrics_dict['F1 Score'] = f1
metrics_dict['Test Log Probability'] = log_probability
metrics_dict['Perplexity Per Visit'] = pp_visit
metrics_dict['Perplexity Per Positive Code'] = pp_positive
metrics_dict['Perplexity Per Possible Code'] = pp_possible
pickle.dump(metrics_dict, open("../../results/testing_stats/LSTMBaseline_Metrics.pkl", "wb"))

print("Average Test Loss: ", avg_loss)
print("Confusion Matrix: ", confusion_matrix)
print('Accuracy: ', acc)
print('Precision: ', prc)
print('Recall: ', rec)
print('F1 Score: ', f1)
print('Test Log Probability: ', log_probability)
print('Perplexity Per Visit: ', pp_visit)
print('Perplexity Per Positive Code: ', pp_positive)
print('Perplexity Per Possible Code: ', pp_possible)

def sample_sequence(model, length, context, batch_size=None, device='cuda', sample=True):
  context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
  prev = context.unsqueeze(1)
  with torch.no_grad():
    for i in range(length-1):
      code_probs = model(prev)
      code_probs = code_probs[:, -1, :].unsqueeze(1)
      if sample:
        visit = torch.bernoulli(code_probs)
      else:
        visit = torch.round(code_probs)
      prev = torch.cat((prev, visit), dim=1)
  ehr = prev.cpu().detach().numpy()
  visit = None
  prev = None
  return ehr

def convert_ehr(ehrs, index_to_code=None):
  ehr_outputs = []
  for i in range(len(ehrs)):
    ehr = ehrs[i]
    ehr_output = []
    labels_output = ehr[1][config.code_vocab_size:config.code_vocab_size+config.label_vocab_size]
    if index_to_code is not None:
      labels_output = [index_to_code[idx + config.code_vocab_size] for idx in np.nonzero(labels_output)[0]]
    for j in range(2, len(ehr)):
      visit = ehr[j]
      visit_output = []
      indices = np.nonzero(visit)[0]
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
stoken = np.zeros(config.total_vocab_size)
stoken[config.code_vocab_size+config.label_vocab_size] = 1
for i in tqdm(range(0, len(train_ehr_dataset), config.batch_size)):
  bs = min([len(train_ehr_dataset)-i, config.batch_size])
  batch_synthetic_ehrs = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
  batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
  synthetic_ehr_dataset += batch_synthetic_ehrs
  
pickle.dump(synthetic_ehr_dataset, open(f'../../results/datasets/lstmDataset.pkl', 'wb'))