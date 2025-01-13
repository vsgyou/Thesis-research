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
#%%


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
    def forward(self, x):
        return self.model(x)
    def compute_adv_loss(self, total_score_input, total_labels):
        adv_loss = self.criterion(total_score_input, total_labels)
        return adv_loss



class DICE(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, dis_pen, int_weight, pop_weight, device):
        super(DICE, self).__init__()
        
        self.users_int = nn.Embedding(num_users, embedding_size).to(device)
        self.users_pop = nn.Embedding(num_users, embedding_size).to(device)
        self.items_int = nn.Embedding(num_items, embedding_size).to(device)
        self.items_pop = nn.Embedding(num_items, embedding_size).to(device)
        
        self.int_weight = int_weight
        self.pop_weight = pop_weight
        
        self.criterion_discrepancy = nn.MSELoss()
        self.dis_pen = dis_pen
    def adapt(self, epoch, decay):
        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay
        
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

    def bpr_loss(self, p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score-n_score)))
    
    def mask_bpr_loss(self, p_score, n_score, mask):
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score-n_score)))
    
    def forward(self, user, item_p, item_n, mask):
        users_int = self.users_int(user)
        users_pop = self.users_pop(user)
        items_p_int = self.items_int(item_p)
        items_p_pop = self.items_pop(item_p)
        items_n_int = self.items_int(item_n)
        items_n_pop = self.items_pop(item_n)
        
        p_score_int = torch.sum(users_int * items_p_int, dim=2)
        n_score_int = torch.sum(users_int * items_n_int, dim=2)

        p_score_pop = torch.sum(users_pop * items_p_pop, dim=2)
        n_score_pop = torch.sum(users_pop * items_n_pop, dim=2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop
        
        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop,~mask) # mask : o_2, ~mask : o_1
        loss_total = self.bpr_loss(p_score_total, n_score_total)
        
        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = self.items_int(item_all)
        item_pop = self.items_pop(item_all)
        user_all = torch.unique(user)
        user_int = self.users_int(user_all)
        user_pop = self.users_pop(user_all)
        discrepancy_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)
        
        loss = self.int_weight * loss_int + self.pop_weight * loss_pop + loss_total - self.dis_pen * discrepancy_loss
        return loss, p_score_int, p_score_pop, n_score_int, n_score_pop
    
    def get_item_embeddings(self):
        item_embeddings = torch.cat((self.items_int.weight, self.items_pop.weight), 1)
        return item_embeddings.detach().cpu().numpy().astype('float32')
    
    def get_user_embeddings(self):
        user_embeddings = torch.cat((self.users_int.weight, self.users_pop.weight), 1)
        return user_embeddings.detach().cpu().numpy().astype('float32')


# %%
def train(model, discriminator, train_loader, optimizer, disc_optimizer, device):
    model.train()
    discriminator.train()
    total_loss = []
    
    for users, p_item, n_item, mask in tqdm(train_loader):
        users = users.to(device)
        p_item = p_item.to(device)
        n_item = n_item.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        loss, p_score_int, p_score_pop, n_score_int, n_score_pop = model(users, p_item, n_item, mask)
        p_score_int = p_score_int.unsqueeze(-1)
        p_score_pop = p_score_pop.unsqueeze(-1)
        n_score_int = n_score_int.unsqueeze(-1)
        n_score_pop = n_score_pop.unsqueeze(-1)
        
        p_total_score = torch.concat((p_score_int, p_score_pop), axis = 2)
        n_total_score = torch.concat((n_score_int, n_score_pop), axis = 2)
        p_total_input = discriminator(p_total_score)
        n_total_input= discriminator(n_total_score)
        
        interest_labels = torch.ones_like(p_score_int).to(device)
        conformity_labels = torch.zeros_like(p_score_pop).to(device)
        labels_total = torch.concat((interest_labels, conformity_labels), axis = 2)
        
        adv_loss_p = discriminator.compute_adv_loss(p_total_input, labels_total)
        adv_loss_n = discriminator.compute_adv_loss(n_total_input, labels_total)
#        final_loss = loss -  0.1*(adv_loss_p + adv_loss_n)
        final_loss = loss -  0.1*(adv_loss_p)
        final_loss.backward(retain_graph = True)
        optimizer.step()
        
        total_loss.append(final_loss.item())
        
        disc_optimizer.zero_grad()
        disc_loss_p = discriminator.compute_adv_loss(p_total_input,labels_total)
        disc_loss_n = discriminator.compute_adv_loss(n_total_input,labels_total)
#        disc_loss = disc_loss_p + disc_loss_n
        disc_loss = disc_loss_p
        disc_loss.backward()
        disc_optimizer.step()
    return sum(total_loss) / len(total_loss)

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