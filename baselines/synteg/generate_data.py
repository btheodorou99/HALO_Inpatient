import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from config import SyntegConfig
from synteg import Generator, DependencyModel

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
index_to_code = pickle.load(open("../../data/indexToCode.pkl", "rb"))
id_to_label = pickle.load(open("../../data/idToLabel.pkl", "rb"))
model = DependencyModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
generator = Generator(config).to(device)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=4e-6, weight_decay=1e-5)
checkpoint1 = torch.load("../../save/synteg_dependency_model", map_location=torch.device(device))
checkpoint2 = torch.load("../../save/synteg_condition_model", map_location=torch.device(device))
model.load_state_dict(checkpoint1['model'])
optimizer.load_state_dict(checkpoint1['optimizer'])
generator.load_state_dict(checkpoint2['generator'])
generator_optimizer.load_state_dict(checkpoint2['generator_optimizer'])

# Add the labels to the index_to_code mapping
for k, l in id_to_label.items():
  index_to_code[config.code_vocab_dim+k] = f"Chronic Condition: {l}"
  
def sample_sequence(length, context, batch_size, device='cuda'):
    context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).to(device)
    ehr = context.unsqueeze(1).to(device)
    batch_ehr = torch.tensor(np.ones((batch_size, config.max_num_visit, config.max_length_visit)) * config.vocab_dim, dtype=torch.long).to(device)
    batch_ehr[:,0,0] = config.code_vocab_dim + config.label_vocab_dim
    batch_lens = torch.zeros((batch_size, config.max_num_visit, 1), dtype=torch.int).to(device)
    batch_lens[:,0,0] = 1
    with torch.no_grad():
        for j in range(length-1):
            for i in range(batch_size):
                codes = torch.nonzero(ehr[i,j]).squeeze(1)
                batch_ehr[i,j,0:len(codes)] = codes[0:config.max_length_visit]
                batch_lens[i,j] = min(len(codes), config.max_length_visit)
            condition_vector = model(batch_ehr, batch_lens, export=True)
            condition = condition_vector[:,j,:]
            z = torch.randn((batch_size, config.z_dim)).to(device)
            visit = generator(z, condition)
            visit = torch.bernoulli(visit).unsqueeze(1)
            ehr = torch.cat((ehr, visit), dim=1)
    ehr = ehr.cpu().detach().numpy()
    return ehr

def convert_ehr(ehrs, index_to_code=None):
    ehr_outputs = []
    for i in range(len(ehrs)):
        ehr = ehrs[i]
        ehr_output = []
        labels_output = ehr[1][config.code_vocab_dim:config.code_vocab_dim+config.label_vocab_dim]
        if index_to_code is not None:
            labels_output = [index_to_code[idx + config.code_vocab_dim] for idx in np.nonzero(labels_output)[0]]
        for j in range(2, len(ehr)):
            visit = ehr[j]
            visit_output = []
            indices = np.nonzero(visit)[0]
            end = False
            for idx in indices:
                if idx < config.code_vocab_dim: 
                    visit_output.append(index_to_code[idx] if index_to_code is not None else idx)
                elif idx == config.code_vocab_dim+config.label_vocab_dim+1:
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

# Generate a few sampled EHR for examinations
#stoken = np.zeros(config.vocab_dim)
#synthetic_ehrs = sample_sequence(config.max_num_visit, stoken, batch_size=3, device=device)
#synthetic_ehrs = convert_ehr(synthetic_ehrs, index_to_code)
#print("Sampled Synthetic EHRs: ")
#for i in range(3):
#    print("Labels: ")
#    print(synthetic_ehrs[i]['labels'])
#    print("Visits: ")
#    for v in synthetic_ehrs[i]['visits']:
#        print(v)
#    print("\n\n")

# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
count = 0
stoken = np.zeros(config.vocab_dim)
for i in tqdm(range(0, len(train_ehr_dataset), config.dependency_batchsize)):
    bs = min([len(train_ehr_dataset)-i, config.dependency_batchsize])
    batch_synthetic_ehrs = sample_sequence(config.max_num_visit, stoken, batch_size=bs, device=device)
    batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
    synthetic_ehr_dataset += batch_synthetic_ehrs
    if len(synthetic_ehr_dataset) > 10000:
        pickle.dump(synthetic_ehr_dataset, open(f'../../temp_synteg/syntegDataset_{count}.pkl', 'wb'))
        synthetic_ehr_dataset = []
        count += 1
    
pickle.dump(synthetic_ehr_dataset, open(f'../../temp_synteg/syntegDataset_{count}.pkl', 'wb'))