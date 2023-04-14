import os
import time
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from config import SyntegConfig
import torch.nn.functional as F
from synteg import Generator, Discriminator
from torch.autograd import grad as torch_grad

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

condition_dataset = pickle.load(open('data/conditionDataset.pkl', 'rb'))

def get_batch(loc, batch_size):
    data = condition_dataset[loc:loc+batch_size]
    visits = [d['ehr'] for d in data]
    conditions = [d['condition'] for d in data]
    visits = torch.tensor(visits, dtype=torch.int64).to(device)
    conditions = torch.tensor(conditions).to(device)
    return (visits, conditions)

def shuffle_dataset(dataset):
    np.random.shuffle(dataset)

EPOCHS = 600
generator = Generator(config).to(device)
discriminator = Discriminator(config).to(device)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=4e-6, weight_decay=1e-5)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-5, weight_decay=1e-5)
if os.path.exists("../../save/synteg_condition_model"):
    print("Loading previous model")
    checkpoint = torch.load("../../save/synteg_condition_model", map_location=torch.device(device))
    generator.load_state_dict(checkpoint['generator'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

def d_step(visits, conditions):
    discriminator.train()
    generator.eval()
    discriminator_optimizer.zero_grad()
    
    real = visits
    z = torch.randn((len(visits), config.z_dim)).to(device)
    epsilon = torch.rand((len(visits), 1)).to(device)

    synthetic = generator(z, conditions)
    real_output = discriminator(real, conditions)
    fake_output = discriminator(synthetic, conditions)
    w_distance = -torch.mean(real_output) + torch.mean(fake_output)

    interpolate = real + epsilon * (synthetic - real)
    interpolate_output = discriminator(interpolate, conditions)
    
    gradients = torch_grad(outputs=interpolate_output, inputs=interpolate,
                           grad_outputs=torch.ones(interpolate_output.size()).to(device),
                           create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(len(visits), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = config.gp_weight * ((gradients_norm - 1) ** 2).mean()
    
    disc_loss = gradient_penalty + w_distance
    disc_loss.backward()
    discriminator_optimizer.step()
    return disc_loss, w_distance

def g_step(conditions):
    z = torch.randn((len(conditions), config.z_dim)).to(device)
    generator.train()
    discriminator.eval()
    generator_optimizer.zero_grad()
    synthetic = generator(z, conditions)
    fake_output = discriminator(synthetic, conditions)
    gen_loss = -torch.mean(fake_output)
    gen_loss.backward()
    generator_optimizer.step()

def train_step(batch):
    visits, conditions = batch # bs * codes, bs * condition
    visits = torch.sum(F.one_hot(visits, num_classes=config.vocab_dim+1), dim=-2)[:,:-1] # bs * vocab
    disc_loss, w_distance = d_step(visits, conditions)
    g_step(conditions)
    return disc_loss, w_distance

print('training start')
for e in tqdm(range(EPOCHS)):
    total_loss = 0
    total_w = 0
    step = 0
    shuffle_dataset(condition_dataset)
    for i in range(0, len(condition_dataset), config.gan_batchsize):
        batch = get_batch(i, config.gan_batchsize)
        loss, w = train_step(batch)
        total_loss += loss
        total_w += w
        step += 1
    format_str = 'epoch: %d, loss = %f, w = %f'
    print(format_str % (e, -total_loss / step, -total_w / step))
    if e % 50 == 49:
        state = {
                    'generator': generator.state_dict(),
                    'generator_optimizer': generator_optimizer.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'discriminator_optimizer': discriminator_optimizer.state_dict(),
                    'epoch': e
                }
        torch.save(state, '../../save/synteg_condition_model')
        print('\n------------ Save newest model ------------\n')