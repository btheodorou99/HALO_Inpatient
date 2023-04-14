import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from model import HALOModel
from config import HALOConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
LR = 0.00001
EPOCHS = 50
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TEST_EXAMPLES = 7500
NUM_TOT_EXAMPLES = 7500
NUM_VAL_EXAMPLES = 2500

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
train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
train_ehr_dataset = [(p,1) for p in train_ehr_dataset]
test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
test_ehr_dataset = [(p,0) for p in test_ehr_dataset]
synthetic_ehr_dataset = pickle.load(open('./results/datasets/haloDataset.pkl', 'rb'))
synthetic_ehr_dataset = [p for p in synthetic_ehr_dataset if len(p['visits']) > 0]

attack_dataset_pos = list(random.sample(train_ehr_dataset, NUM_TOT_EXAMPLES))
attack_dataset_neg = list(random.sample(test_ehr_dataset, NUM_TOT_EXAMPLES))
np.random.shuffle(attack_dataset_pos)
np.random.shuffle(attack_dataset_neg)
test_attack_dataset = attack_dataset_pos[:NUM_TEST_EXAMPLES] + attack_dataset_neg[:NUM_TEST_EXAMPLES]
val_attack_dataset = attack_dataset_pos[NUM_TEST_EXAMPLES:NUM_TEST_EXAMPLES+NUM_VAL_EXAMPLES] + attack_dataset_neg[NUM_TEST_EXAMPLES:NUM_TEST_EXAMPLES+NUM_VAL_EXAMPLES]
np.random.shuffle(test_attack_dataset)
np.random.shuffle(val_attack_dataset)
attack_dataset_pos = attack_dataset_pos[NUM_TEST_EXAMPLES+NUM_VAL_EXAMPLES:]
attack_dataset_neg = attack_dataset_neg[NUM_TEST_EXAMPLES+NUM_VAL_EXAMPLES:]

def get_batch(loc, batch_size, dataset):
    # EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
    #   Where each patient P is [V_1, V_2, ... , V_j]
    #     Where each visit V is [C_1, C_2, ... , C_k]
    #   And where each Label L is a binary vector [L_1 ... L_11]
    ehr = dataset[loc:loc+batch_size]
    attack_labels = [l for (e,l) in ehr]
    ehr = [e for (e,l) in ehr]

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
    return batch_ehr, batch_mask, attack_labels

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
                d += len(v) + len(v2) - (2 * len(set(v) & set(v2)))
                
        min_d = d if d < min_d else min_d
    return min_d

# Perform the Hamming Distance experiment
ds = [(find_hamming(ehr, synthetic_ehr_dataset), l) for (ehr, l) in tqdm(test_attack_dataset)]
median_dist = np.median([d for (d,l) in ds])
preds = [1 if d < median_dist else 0 for (d,l) in ds]
labels = [l for (d,l) in ds]
results = {
    "Accuracy": metrics.accuracy_score(labels, preds),
    "Precision": metrics.precision_score(labels, preds),
    "Recall": metrics.recall_score(labels, preds),
    "F1": metrics.f1_score(labels, preds)
}
pickle.dump(results, open("results/privacy_evaluation/hamming_model.pkl", "wb"))
print(results)



# Perform the Log Likelihood experiment
model = HALOModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
checkpoint = torch.load('./save/halo_model', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()

probabilities = []
with torch.no_grad():
    for i in tqdm(range(0, len(test_attack_dataset), config.batch_size)):
        # Get batch inputs
        batch_ehr, batch_mask, attack_labels = get_batch(i, 2*config.batch_size, test_attack_dataset)
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
        
        # Get batch outputs
        test_loss, predictions, labels = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)

        # Calculate and add probabilities 
        # Note that the masked codes will have probability 1 and be ignored
        label_probs = torch.abs(labels - 1.0 + predictions)
        log_prob = torch.log(label_probs).sum(dim=[1,2]).cpu().detach().numpy().tolist()
        final_probs = [(p,l) for p,l in zip(log_prob, attack_labels)]
        probabilities.extend(final_probs)

median_prob = np.median([p for (p,l) in probabilities])
preds = [1 if p > median_prob else 0 for (p,l) in probabilities]
labels = [l for (p,l) in probabilities]
results = {
    "Accuracy": metrics.accuracy_score(labels, preds),
    "Precision": metrics.precision_score(labels, preds),
    "Recall": metrics.recall_score(labels, preds),
    "F1": metrics.f1_score(labels, preds)
}
pickle.dump(results, open("results/privacy_evaluation/perplexity_model.pkl", "wb"))
print(results)