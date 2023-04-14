import torch
import pickle
import random
import itertools
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt
from config import HALOConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
LR = 0.001
EPOCHS = 25
LABEL_IDX_LIST = list(range(25))
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TRAIN_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 500

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

# Add the labels to the index_to_code mapping
index_to_code = pickle.load(open("./data/idToLabel.pkl", "rb"))

config = HALOConfig()
train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('./data/valDataset.pkl', 'rb'))
test_ehr_dataset = pickle.load(open('./data/testDataset.pkl', 'rb'))
halo_ehr_dataset = pickle.load(open('./results/datasets/haloDataset.pkl', 'rb'))
haloCoarse_ehr_dataset = pickle.load(open('./results/datasets/haloCoarseDataset.pkl', 'rb'))
lstm_ehr_dataset = pickle.load(open('./results/datasets/lstmDataset.pkl', 'rb'))
synteg_ehr_dataset = pickle.load(open('./results/datasets/syntegDataset.pkl', 'rb'))
eva_ehr_dataset = pickle.load(open('./results/datasets/evaDataset.pkl', 'rb'))
gpt_ehr_dataset = pickle.load(open('./results/datasets/gptDataset.pkl', 'rb'))

class DiagnosisModel(nn.Module):
    def __init__(self, config):
        super(DiagnosisModel, self).__init__()
        self.embedding = nn.Linear(config.code_vocab_size, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=LSTM_HIDDEN_DIM,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2*LSTM_HIDDEN_DIM, 1)

    def forward(self, input_visits, lengths):
        visit_emb = self.embedding(input_visits)
        visit_emb = self.dropout(visit_emb)
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :LSTM_HIDDEN_DIM]
        out_reverse = output[:, 0, LSTM_HIDDEN_DIM:]
        out_combined = torch.cat((out_forward, out_reverse), 1)

        patient_embedding = self.fc(out_combined)
        patient_embedding = torch.squeeze(patient_embedding, 1)
        prob = torch.sigmoid(patient_embedding)
        
        return prob

def get_batch(ehr_dataset, loc, batch_size, label_idx):
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.code_vocab_size))
    batch_labels = np.array([p['labels'][label_idx] for p in ehr])
    batch_lens = np.zeros(len(ehr))
    for i, p in enumerate(ehr):
        visits = p['visits']
        batch_lens[i] = len(visits)
        for j, v in enumerate(visits):
            batch_ehr[i,j][v] = 1

    return batch_ehr, batch_labels, batch_lens

def train_model(model, train_dataset, val_dataset, save_name, label_idx):
    global_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    for e in range(EPOCHS):
        np.random.shuffle(train_dataset)
        train_losses = []
        for i in range(0, len(train_dataset), BATCH_SIZE):
            model.train()
            batch_ehr, batch_labels, batch_lens = get_batch(train_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            prob = model(batch_ehr, batch_lens)
            loss = bce(prob, batch_labels)
            train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        cur_train_loss = np.mean(train_losses)
        print("Epoch %d Training Loss:%.5f"%(e, cur_train_loss))
    
        model.eval()
        with torch.no_grad():
            val_losses = []
            for v_i in range(0, len(val_dataset), BATCH_SIZE):
                batch_ehr, batch_labels, batch_lens = get_batch(val_dataset, v_i, BATCH_SIZE, label_idx)
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
                prob = model(batch_ehr, batch_lens)
                val_loss = bce(prob, batch_labels)
                val_losses.append(val_loss.cpu().detach().numpy())
            cur_val_loss = np.mean(val_losses)
            print("Epoch %d Validation Loss:%.5f"%(e, cur_val_loss))
            if cur_val_loss < global_loss:
                global_loss = cur_val_loss
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, f'./save/{save_name}')
                print('------------ Save best model ------------')

    model.load_state_dict(state['model'])

def test_model(model, test_dataset, label_idx):
    loss_list = []
    pred_list = []
    true_list = []
    bce = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_dataset), BATCH_SIZE):  
            batch_ehr, batch_labels, batch_lens = get_batch(test_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            prob = model(batch_ehr, batch_lens)
            val_loss = bce(prob, batch_labels)
            loss_list.append(val_loss.cpu().detach().numpy())
            pred_list += list(prob.cpu().detach().numpy())
            true_list += list(batch_labels.cpu().detach().numpy())
    
    round_list = np.around(pred_list)

    # Extract, save, and display test metrics
    avg_loss = np.mean(loss_list)
    cmatrix = metrics.confusion_matrix(true_list, round_list)
    acc = metrics.accuracy_score(true_list, round_list)
    prc = metrics.precision_score(true_list, round_list)
    rec = metrics.recall_score(true_list, round_list)
    f1 = metrics.f1_score(true_list, round_list)
    auroc = metrics.roc_auc_score(true_list, pred_list)
    (precisions, recalls, _) = metrics.precision_recall_curve(true_list, pred_list)
    auprc = metrics.auc(recalls, precisions)
    
    metrics_dict = {}
    metrics_dict['Test Loss'] = avg_loss
    metrics_dict['Confusion Matrix'] = cmatrix
    metrics_dict['Accuracy'] = acc
    metrics_dict['Precision'] = prc
    metrics_dict['Recall'] = rec
    metrics_dict['F1 Score'] = f1
    metrics_dict['AUROC'] = auroc
    metrics_dict['AUPRC'] = auprc
    
    print('Test Loss: ', avg_loss)
    print('Confusion Matrix: ', cmatrix)
    print('Accuracy: ', acc)
    print('Precision: ', prc)
    print('Recall: ', rec)
    print('F1 Score: ', f1)
    print('AUROC: ', auroc)
    print('AUPRC: ', auprc)
    print("\n")

    return metrics_dict

results = {}
for i in LABEL_IDX_LIST:
    label_results = {}

    # Prepare datasets
    halo_pos_label_dataset = [p for p in halo_ehr_dataset if p['labels'][i] == 1]
    halo_neg_label_dataset = [p for p in halo_ehr_dataset if p['labels'][i] == 0]
    haloCoarse_pos_label_dataset = [p for p in haloCoarse_ehr_dataset if p['labels'][i] == 1]
    haloCoarse_neg_label_dataset = [p for p in haloCoarse_ehr_dataset if p['labels'][i] == 0]
    lstm_pos_label_dataset = [p for p in lstm_ehr_dataset if p['labels'][i] == 1]
    lstm_neg_label_dataset = [p for p in lstm_ehr_dataset if p['labels'][i] == 0]
    synteg_pos_label_dataset = [p for p in synteg_ehr_dataset if p['labels'][i] == 1]
    synteg_neg_label_dataset = [p for p in synteg_ehr_dataset if p['labels'][i] == 0]
    eva_pos_label_dataset = [p for p in eva_ehr_dataset if p['labels'][i] == 1]
    eva_neg_label_dataset = [p for p in eva_ehr_dataset if p['labels'][i] == 0]
    gpt_pos_label_dataset = [p for p in gpt_ehr_dataset if p['labels'][i] == 1]
    gpt_neg_label_dataset = [p for p in gpt_ehr_dataset if p['labels'][i] == 0]
    train_pos_label_dataset = [p for p in train_ehr_dataset if p['labels'][i] == 1]
    train_neg_label_dataset = [p for p in train_ehr_dataset if p['labels'][i] == 0]
    val_pos_label_dataset = [p for p in val_ehr_dataset if p['labels'][i] == 1]
    val_neg_label_dataset = [p for p in val_ehr_dataset if p['labels'][i] == 0]
    test_pos_label_dataset = [p for p in test_ehr_dataset if p['labels'][i] == 1]
    test_neg_label_dataset = [p for p in test_ehr_dataset if p['labels'][i] == 0]
  
    val_dataset = list(np.random.choice(val_pos_label_dataset, int(NUM_VAL_EXAMPLES/2), replace=(False if len(val_pos_label_dataset) >= NUM_VAL_EXAMPLES else True))) + list(np.random.choice(val_neg_label_dataset, int(NUM_VAL_EXAMPLES/2), replace=False))
    test_dataset = list(np.random.choice(test_pos_label_dataset, int(NUM_TEST_EXAMPLES/2), replace=(False if len(test_pos_label_dataset) >= NUM_TEST_EXAMPLES else True))) + list(np.random.choice(test_neg_label_dataset, int(NUM_TEST_EXAMPLES/2), replace=False))

    train_dataset_real = list(np.random.choice(train_pos_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=(False if len(test_pos_label_dataset) >= int(NUM_TRAIN_EXAMPLES/2) else True))) + list(np.random.choice(train_neg_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=False))
    train_dataset_halo = list(np.random.choice(halo_pos_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=(False if len(halo_pos_label_dataset) >= int(NUM_TRAIN_EXAMPLES/2) else True))) + list(np.random.choice(halo_neg_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=False))
    train_dataset_haloCoarse = list(np.random.choice(haloCoarse_pos_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=(False if len(haloCoarse_pos_label_dataset) >= int(NUM_TRAIN_EXAMPLES/2) else True))) + list(np.random.choice(haloCoarse_neg_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=False))
    train_dataset_lstm = list(np.random.choice(lstm_pos_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=(False if len(lstm_pos_label_dataset) >= int(NUM_TRAIN_EXAMPLES/2) else True))) + list(np.random.choice(lstm_neg_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=False))
    train_dataset_synteg = list(np.random.choice(synteg_pos_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=(False if len(synteg_pos_label_dataset) >= int(NUM_TRAIN_EXAMPLES/2) else True))) + list(np.random.choice(synteg_neg_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=False))
    train_dataset_eva = list(np.random.choice(eva_pos_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=(False if len(eva_pos_label_dataset) >= int(NUM_TRAIN_EXAMPLES/2) else True))) + list(np.random.choice(eva_neg_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=False))
    train_dataset_gpt = list(np.random.choice(gpt_pos_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=(False if len(gpt_pos_label_dataset) >= int(NUM_TRAIN_EXAMPLES/2) else True))) + list(np.random.choice(gpt_neg_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=False))

    # Perform the different experiments
    model_real = DiagnosisModel(config).to(device)
    train_model(model_real, train_dataset_real, val_dataset, f"syn_diag_Real_{i}", i)
    state = torch.load(f'./save/syn_diag_Real_{i}')
    model_real.load_state_dict(state['model'])
    test_results_real = test_model(model_real, test_dataset, i)
    label_results[f'Real'] = test_results_real

    model_halo = DiagnosisModel(config).to(device)
    train_model(model_halo, train_dataset_halo, val_dataset, f"syn_diag_HALO_{i}", i)
    state = torch.load(f'./save/syn_diag_HALO_{i}')
    model_halo.load_state_dict(state['model'])
    test_results_halo = test_model(model_halo, test_dataset, i)
    label_results[f'HALO'] = test_results_halo

    model_haloCoarse = DiagnosisModel(config).to(device)
    train_model(model_haloCoarse, train_dataset_haloCoarse, val_dataset, f"syn_diag_HALOCoarse_{i}", i)
    state = torch.load(f'./save/syn_diag_HALOCoarse_{i}')
    model_haloCoarse.load_state_dict(state['model'])
    test_results_haloCoarse = test_model(model_haloCoarse, test_dataset, i)
    label_results[f'HALO Coarse'] = test_results_haloCoarse

    model_lstm = DiagnosisModel(config).to(device)
    train_model(model_lstm, train_dataset_lstm, val_dataset, f"syn_diag_LSTM_{i}", i)
    state = torch.load(f'./save/syn_diag_LSTM_{i}')
    model_lstm.load_state_dict(state['model'])
    test_results_lstm = test_model(model_lstm, test_dataset, i)
    label_results[f'LSTM'] = test_results_lstm

    model_eva = DiagnosisModel(config).to(device)
    train_model(model_eva, train_dataset_eva, val_dataset, f"syn_diag_EVA_{i}", i)
    state = torch.load(f'./save/syn_diag_EVA_{i}')
    model_eva.load_state_dict(state['model'])
    test_results_eva = test_model(model_eva, test_dataset, i)
    label_results[f'EVA'] = test_results_eva

    model_synteg = DiagnosisModel(config).to(device)
    train_model(model_synteg, train_dataset_synteg, val_dataset, f"syn_diag_SynTEG_{i}", i)
    state = torch.load(f'./save/syn_diag_SynTEG_{i}')
    model_synteg.load_state_dict(state['model'])
    test_results_synteg = test_model(model_synteg, test_dataset, i)
    label_results[f'SynTEG'] = test_results_synteg
    results[index_to_code[i]][f'SynTEG'] = test_results_synteg

    model_gpt = DiagnosisModel(config).to(device)
    train_model(model_gpt, train_dataset_gpt, val_dataset, f"syn_diag_GPT_{i}", i)
    state = torch.load(f'./save/syn_diag_GPT_{i}')
    model_gpt.load_state_dict(state['model'])
    test_results_gpt = test_model(model_gpt, test_dataset, i)
    label_results[f'GPT'] = test_results_gpt

    results[index_to_code[i]] = label_results

pickle.dump(results, open(f"results/synthetic_training_stats/fully_synthetic_stats.pkl", "wb"))