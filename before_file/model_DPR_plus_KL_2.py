#%%
import math
import numpy as np
import torch
import torch.nn as nn
# from torch.nn.parameter import Parameter
import torch.nn.functional as F
import random
from preprocess_10 import *
from metric import *

import torch.optim as optim

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import random
from tqdm import tqdm

from metric import *
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader


# seed
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# device
# device

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# load data
train_root = 'data/train_coo_record.npz'
valid_root = 'data/val_coo_record.npz'
test_root = 'data/test_coo_record.npz'
pop_root = 'data/popularity.npy'
skew_train_root = 'data/train_skew_coo_record.npz'
skew_pop_root = 'data/popularity_skew.npy'
#train
train_coo_record = sp.load_npz(train_root)
train_dok = train_coo_record.todok(copy=True)
train_lil = train_coo_record.tolil(copy=True)
#train skew
train_skew_coo_record = sp.load_npz(skew_train_root)
train_skew_dok = train_skew_coo_record.todok(copy = True)
train_skew_lil = train_skew_coo_record.tolil(copy = True)
# valid
val_coo_record = sp.load_npz(valid_root)
valid_dok = val_coo_record.todok(copy = True)
valid_lil = val_coo_record.tolil(copy = True)

blend_user = np.hstack((train_coo_record.row, train_skew_coo_record.row))
blend_item = np.hstack((train_coo_record.col, train_skew_coo_record.col))
blend_value = np.hstack((train_coo_record.data, train_skew_coo_record.data))
blend_coo_record = sp.coo_matrix((blend_value, (blend_user, blend_item)),shape = train_coo_record.shape)

uni_train_lil_record = blend_coo_record.tolil(copy = True)
uni_train_interaction_count = np.array([len(row) for row in uni_train_lil_record.rows], dtype = np.int64)
max_train_interaction = int(max(uni_train_interaction_count))

val_interaction_count = np.array([len(row) for row in valid_lil.rows], dtype = np.int64)
max_val_interaction = int(max(val_interaction_count))

unify_train_pos = np.full(max_train_interaction, -1, dtype = np.int64)
unify_valid_pos = np.full(max_val_interaction, -1, dtype = np.int64)
# test
test_coo_record = sp.load_npz(test_root)
test_dok = test_coo_record.todok(copy = True)
test_lil = test_coo_record.tolil(copy = True)

popularity = np.load(pop_root)
skew_pop = np.load(skew_pop_root)

# calculate blend_item_popularity
unique_item, counts = np.unique(blend_item, return_counts = True)
all_item = np.zeros(train_coo_record.shape[1], dtype = int)
all_item[unique_item] = counts
count = all_item
blen_pop = count / np.max(count)
blen_pop = torch.Tensor(blen_pop)


# calculate user propensity
sum_pop = blend_coo_record.dot(blen_pop)
count_inter = blend_coo_record.tocsr().sum(axis = 1).A1
count_inter_safe = np.where(count_inter == 0, 1, count_inter)
user_pop = sum_pop / count_inter_safe
user_pop = torch.Tensor(user_pop)
torch.min(user_pop)

#Dataloader
batch_size = 128
train_data = TrainDataset(train_lil, train_dok, train_skew_lil, train_skew_dok, popularity = popularity, skew_pop = skew_pop, device = device)
train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = False)

valid_data = CGDataset(test_data_source = 'val', data_root = valid_root, train_root = train_root, skew_train_root = skew_train_root, device = device)
val_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = True, drop_last = False)

test_data = CGDataset(test_data_source = 'test', data_root = test_root, train_root= train_root, skew_train_root = skew_train_root,device = device)
test_dataloader = DataLoader(test_data, batch_size = batch_size , shuffle = True, drop_last = False)
#%%
num_users = train_coo_record.shape[0]
num_items = train_coo_record.shape[1]


dice_lr = 0.006
disc_lr = 0.001
weight_decay = 5e-8
model = DICE(num_users= num_users, num_items = num_items, embedding_size = 64,dis_pen =0.01, int_weight = 0.1, pop_weight = 0.1, kl_weight = 0.1, device = device)
discriminator = Discriminator(input_dim = 1, hidden_dim = 20, device = device).to(device)
optimizer = optim.Adam(model.parameters(), lr = dice_lr, weight_decay = weight_decay, betas = (0.5, 0.99), amsgrad = True)
disc_optimizer = optim.Adam(discriminator.parameters(), lr = disc_lr)



#%%

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim//4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim//4), 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
    def forward(self, x):
        return self.model(x)
    def compute_adv_loss(self, total_score_input, total_labels):
        adv_loss = self.criterion(total_score_input, total_labels)
        return adv_loss



class DICE(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, dis_pen, int_weight, pop_weight, kl_weight, device):
        super(DICE, self).__init__()
        
        self.users_int = nn.Embedding(num_users, embedding_size).to(device)
        self.users_pop = nn.Embedding(num_users, embedding_size).to(device)
        self.items_int = nn.Embedding(num_items, embedding_size).to(device)
        self.items_pop = nn.Embedding(num_items, embedding_size).to(device)
        
        self.int_weight = int_weight
        self.pop_weight = pop_weight
        self.kl_weight = kl_weight
        
        self.criterion_discrepancy = nn.MSELoss()
        self.dis_pen = dis_pen
        self.device = device
    def adapt(self, epoch, decay):
        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay
        
    def compute_kl_loss(self, scores):
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True)
        
        # 표준편차가 0인 경우 작은 값을 더해줍니다.
        std = std + 1e-8
        
        norm_dist = torch.distributions.Normal(mean, std)
        standard_norm = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        
        kl_div = torch.distributions.kl.kl_divergence(norm_dist, standard_norm)
        return kl_div.mean()
    
    def dcor(self, x, y):
        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(x[:, None] - y, p=2, dim=2)
        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()
        n = x.size(0)
        
        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))
        
        return dcor

    # def bpr_loss(self, p_score, n_score):
    #     bpr_loss_value = -torch.mean(torch.log(torch.sigmoid(p_score-n_score)))
    #     l2_reg = sum(param.norm(2) ** 2 for param in self.parameters())
    #     total_loss = bpr_loss_value + 1e-4*l2_reg
    #     return total_loss
    
    # def mask_bpr_loss(self, p_score, n_score, mask):
    #     bpr_loss_value = -torch.mean(mask * torch.log(torch.sigmoid(p_score-n_score)))
    #     l2_reg = sum(param.norm(2) ** 2 for param in self.parameters())
    #     total_loss = bpr_loss_value + 1e-4*l2_reg
    #     return total_loss
    def bpr_loss(self, p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score-n_score)+1e-8))
    
    def mask_bpr_loss(self, p_score, n_score, mask):
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score-n_score)+1e-8))
    def forward(self, user, item_p, item_n, mask):
        users_int = self.users_int(user)
        users_pop = self.users_pop(user)
        
        items_int_weight = self.items_int.weight
        items_pop_weight = self.items_pop.weight
        
        items_p_int = self.items_int(item_p)
        items_p_pop = self.items_pop(item_p)
        items_n_int = self.items_int(item_n)
        items_n_pop = self.items_pop(item_n)
        
        # Compute all interactions
        score_int = torch.matmul(users_int[:,1,:], items_int_weight.t())  # (batch_size, num_items)
        score_pop = torch.matmul(users_pop[:,1,:], items_pop_weight.t())  # (batch_size, num_items)
        
        p_score_int = torch.sum(users_int * items_p_int, dim=2)
        n_score_int = torch.sum(users_int * items_n_int, dim=2)

        p_score_pop = torch.sum(users_pop * items_p_pop, dim=2)
        n_score_pop = torch.sum(users_pop * items_n_pop, dim=2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop
        
        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop,~mask) # mask : o_2, ~mask : o_1
        loss_total = self.bpr_loss(p_score_total, n_score_total)
        # Compute KL loss for all items
        kl_loss_int = self.compute_kl_loss(score_int)
        kl_loss_pop = self.compute_kl_loss(score_pop)
        
        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = self.items_int(item_all)
        item_pop = self.items_pop(item_all)
        user_all = torch.unique(user)
        user_int = self.users_int(user_all)
        user_pop = self.users_pop(user_all)
        discrepancy_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)
        
        loss = self.int_weight * loss_int + self.pop_weight * loss_pop + loss_total - self.dis_pen * discrepancy_loss + self.kl_weight * (kl_loss_int + kl_loss_pop)
        return loss, p_score_int, p_score_pop, n_score_int, n_score_pop
    
    def get_item_embeddings(self):
        item_embeddings = torch.cat((self.items_int.weight, self.items_pop.weight), 1)
        return item_embeddings.detach().cpu().numpy().astype('float32')
    
    def get_user_embeddings(self):
        user_embeddings = torch.cat((self.users_int.weight, self.users_pop.weight), 1)
        return user_embeddings.detach().cpu().numpy().astype('float32')

train_loader = train_dataloader
# %%
def train(model, discriminator, train_loader, optimizer,disc_optimizer, device):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    discriminator.train()
    total_loss = []
    total_DICE_loss= []
    total_adv_loss = []
    total_disc_loss = []
    
    for users, p_item, n_item, mask in tqdm(train_loader):
        users = users.to(device)
        p_item = p_item.to(device)
        n_item = n_item.to(device)
        mask = mask.to(device)
        
        loss, p_score_int, p_score_pop, n_score_int, n_score_pop = model(users, p_item, n_item, mask)
        # p_score_int = p_score_int[:,1].unsqueeze(-1)
        # p_score_pop = p_score_pop[:,1].unsqueeze(-1)
        p_score_int = p_score_int.unsqueeze(-1)
        p_score_pop = p_score_pop.unsqueeze(-1)
        n_score_int = n_score_int.unsqueeze(-1)
        n_score_pop = n_score_pop.unsqueeze(-1)
        
        p_total_score = torch.cat((p_score_int, p_score_pop), dim = 0)
        n_total_score = torch.cat((n_score_int, n_score_pop), dim = 0)
        
        p_total_input = discriminator(p_total_score)
        n_total_input = discriminator(n_total_score)
        
        
        # interest_labels = torch.tensor([1,0],dtype=torch.float32).repeat(n_score_int.size()).to(device)
        # conformity_labels = torch.tensor([0,1],dtype=torch.float32).repeat(n_score_pop.size()).to(device)

        # torch.ones와 torch.zeros로 변경
        interest_labels = torch.ones(n_score_int.size(), dtype=torch.float32).to(device)
        conformity_labels = torch.zeros(n_score_pop.size(), dtype=torch.float32).to(device)

        
        # p_labels_total = torch.concat((interest_labels[:,1,:], conformity_labels[:,1,:]), axis = 0)
        disc_labels_total = torch.cat((interest_labels, conformity_labels), dim = 0)
        adv_labels_total = torch.cat((conformity_labels, interest_labels), dim = 0)
        
        disc_loss_p = discriminator.compute_adv_loss(p_total_input, disc_labels_total)
        disc_loss_n = discriminator.compute_adv_loss(n_total_input, disc_labels_total)
        
        disc_loss = disc_loss_p + disc_loss_n
        
        disc_optimizer.zero_grad()
        disc_loss.backward(retain_graph=True)
        disc_optimizer.step() 
        
        p_total_input = discriminator(p_total_score)
        n_total_input = discriminator(n_total_score)
        
        adv_loss_p = discriminator.compute_adv_loss(p_total_input, adv_labels_total)
        adv_loss_n = discriminator.compute_adv_loss(n_total_input, adv_labels_total)
        adv_loss = adv_loss_p + adv_loss_n
        final_loss = loss + 0.5*adv_loss
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        total_DICE_loss.append(loss.item()) 
        total_loss.append(final_loss.item())
        total_adv_loss.append(adv_loss.item())
        total_disc_loss.append(disc_loss.item())
    return sum(total_loss) / len(total_loss), sum(total_DICE_loss) / len(total_DICE_loss), sum(total_adv_loss) / len(total_adv_loss), sum(total_disc_loss) / len(total_disc_loss)


def valid(model, val_dataloader, val_max_inter, top_k, device):
    real_num_val_users = 0
    cumulative_results = {metric: 0.0 for metric in ['recall','hit_ratio','ndcg']}
    with torch.no_grad():
        item_embeddings = model.get_item_embeddings()
        item_embeddings = torch.Tensor(item_embeddings).to(device)
        user_embeddings = model.get_user_embeddings()
        user_embeddings = torch.Tensor(user_embeddings).to(device)
        generator = FaissInnerProductMaximumSearchGenerator(item_embeddings, device = device)
        jud = Judger(topk = top_k)

        for data in tqdm(val_dataloader):
            users, train_pos, test_pos, num_test_pos = data
            users = users.squeeze().to(device)
            train_pos = train_pos.to(device)
            test_pos = test_pos.to(device)
            num_test_pos = num_test_pos.to(device)

            items = generator.generate(user_embeddings[users], top_k + val_max_inter)
            items = filter_history(items, top_k, train_pos, device)
            items = items.cpu()
            test_pos = test_pos.cpu()
            num_test_pos = num_test_pos.cpu()
            batch_results, valid_num_users = jud.judge(items = items, test_pos = test_pos, num_test_pos = num_test_pos)
            real_num_val_users = real_num_val_users + valid_num_users
            
            for metric, value in batch_results.items():
                cumulative_results[metric] += value
                
        average_results = {metric: value / real_num_val_users for metric, value in cumulative_results.items()}
    return average_results

def test(model, test_dataloader, test_max_inter, top_k, device):
    real_num_test_users = 0
    cumulative_results = {metric: 0.0 for metric in ['recall','hit_ratio','ndcg']}
    with torch.no_grad():
        item_embeddings = model.get_item_embeddings()
        item_embeddings = torch.Tensor(item_embeddings).to(device)
        user_embeddings = model.get_user_embeddings()
        user_embeddings = torch.Tensor(user_embeddings).to(device)
        generator = FaissInnerProductMaximumSearchGenerator(item_embeddings, device = device)
        jud = Judger(topk = top_k)
        
        for data in tqdm(test_dataloader):
            users, train_pos, test_pos, num_test_pos = data
            users = users.squeeze().to(device)
            train_pos = train_pos.to(device)
            test_pos = test_pos.to(device)
            num_test_pos = num_test_pos.to(device)
            
            items = generator.generate(user_embeddings[users], top_k + test_max_inter)
            items = filter_history(items, top_k, train_pos, device)
            items = items.cpu()
            test_pos = test_pos.cpu()
            batch_results, test_num_users = jud.judge(items = items, test_pos = test_pos, num_test_pos = num_test_pos)
            real_num_test_users = real_num_test_users + test_num_users
            
            for metric, value in batch_results.items():
                cumulative_results[metric] += value
                
        average_results = {metric: value / real_num_test_users for metric, value in cumulative_results.items()}
        
    return average_results
# %%