#%%
import math
import numpy as np
import torch
import torch.nn as nn
# from torch.nn.parameter import Parameter
import torch.nn.functional as F
import random
from preprocess_10 import *
from metric_DPR import *

import torch.optim as optim

# %%

class BPR(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, blen_pop, device):
        super(BPR, self).__init__()
        
        self.users_emb = nn.Embedding(num_users, embedding_size).to(device)
        self.items_emb = nn.Embedding(num_items, embedding_size).to(device)
        self.blen_pop = blen_pop
        
    def adapt(self, epoch, decay):
        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay
        
    # def bpr_loss(self, p_score, n_score):
    #     return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))
    
    # def mask_bpr_loss(self, p_score, n_score, mask):
    #     return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))
    
    def bpr_loss(self, p_score, n_score, pop_item_p):
        epsilon = 1e-10
        weight = 1 / pop_item_p
        return -torch.mean(torch.log(torch.sigmoid(p_score-n_score) + epsilon))
    
    def mask_bpr_loss(self, p_score, n_score, mask):
        epsilon = 1e-10
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score-n_score)+epsilon))
    
    def forward(self, user, item_p, item_n, mask):
        users_emb = self.users_emb(user)
        items_p_emb = self.items_emb(item_p)
        items_n_emb = self.items_emb(item_n)
    
        p_score = torch.sum(users_emb * items_p_emb, dim=2)
        n_score = torch.sum(users_emb * items_n_emb, dim=2)

        pop_item_p = self.blen_pop[item_p]
        
        loss_total = self.bpr_loss(p_score, n_score, pop_item_p)
        
        loss = loss_total
        return loss
    
    def get_item_embeddings(self):
        item_embeddings = torch.Tensor(self.items_emb.weight)
        return item_embeddings.detach().cpu().numpy().astype('float32')
    
    def get_user_embeddings(self):
        user_embeddings = torch.Tensor(self.users_emb.weight)
        return user_embeddings.detach().cpu().numpy().astype('float32')


# %%
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = []
    for users, p_item, n_item, mask in tqdm(train_loader):
        users = users.to(device)
        p_item = p_item.to(device)
        n_item = n_item.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        loss = model(users, p_item, n_item, mask)
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss)
    return sum(total_loss) / len(total_loss)

# def valid(model, val_dataloader, val_max_inter, top_k, device):
#     real_num_val_users = 0
#     cumulative_results = {metric: 0.0 for metric in ['recall','hit_ratio','ndcg']}
#     with torch.no_grad():
#         item_embeddings = model.get_item_embeddings()
#         item_embeddings = torch.Tensor(item_embeddings).to(device)
#         user_embeddings = model.get_user_embeddings()
#         user_embeddings = torch.Tensor(user_embeddings).to(device)
#         generator = FaissInnerProductMaximumSearchGenerator(item_embeddings, device = device)
#         jud = Judger(topk = top_k)

#         for data in tqdm(val_dataloader):
#             users, train_pos, test_pos, num_test_pos = data
#             users = users.squeeze().to(device)
#             train_pos = train_pos.to(device)
#             test_pos = test_pos.to(device)
#             num_test_pos = num_test_pos.to(device)

#             items = generator.generate(user_embeddings[users], top_k + val_max_inter)
#             items = filter_history(items, top_k, train_pos, device)
#             items = items.cpu()
#             test_pos = test_pos.cpu()
#             num_test_pos = num_test_pos.cpu()
#             batch_results, valid_num_users = jud.judge(items = items, test_pos = test_pos, num_test_pos = num_test_pos)
#             real_num_val_users = real_num_val_users + valid_num_users
            
#             for metric, value in batch_results.items():
#                 cumulative_results[metric] += value
                
#         average_results = {metric: value / real_num_val_users for metric, value in cumulative_results.items()}
#     return average_results

# def test(model, test_dataloader, test_max_inter, top_k, device):
#     real_num_test_users = 0
#     cumulative_results = {metric: 0.0 for metric in ['recall','hit_ratio','ndcg']}
#     with torch.no_grad():
#         item_embeddings = model.get_item_embeddings()
#         item_embeddings = torch.Tensor(item_embeddings).to(device)
#         user_embeddings = model.get_user_embeddings()
#         user_embeddings = torch.Tensor(user_embeddings).to(device)
#         generator = FaissInnerProductMaximumSearchGenerator(item_embeddings, device = device)
#         jud = Judger(topk = top_k)
        
#         for data in tqdm(test_dataloader):
#             users, train_pos, test_pos, num_test_pos = data
#             users = users.squeeze().to(device)
#             train_pos = train_pos.to(device)
#             test_pos = test_pos.to(device)
#             num_test_pos = num_test_pos.to(device)
            
#             items = generator.generate(user_embeddings[users], top_k + test_max_inter)
#             items = filter_history(items, top_k, train_pos, device)
#             items = items.cpu()
#             test_pos = test_pos.cpu()
#             batch_results, test_num_users = jud.judge(items = items, test_pos = test_pos, num_test_pos = num_test_pos)
#             real_num_test_users = real_num_test_users + test_num_users
            
#             for metric, value in batch_results.items():
#                 cumulative_results[metric] += value
                
#         average_results = {metric: value / real_num_test_users for metric, value in cumulative_results.items()}
        
#     return average_results
# # %%

def valid(model, val_dataloader, val_max_inter, top_k, group_lists, device):
    real_num_val_users = 0
    cumulative_results = {metric: 0.0 for metric in ['recall','hit_ratio','ndcg','rsp','reo']}
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
            train_pos = train_pos.cpu()
            test_pos = test_pos.cpu()
            num_test_pos = num_test_pos.cpu()
            batch_results, valid_num_users = jud.judge(items = items, train_pos = train_pos, test_pos = test_pos, num_test_pos = num_test_pos, group_lists = group_lists)
            real_num_val_users = real_num_val_users + valid_num_users
            
            for metric, value in batch_results.items():
                cumulative_results[metric] += value
                
        average_results = {metric: value / real_num_val_users for metric, value in cumulative_results.items()}
    return average_results

def test(model, test_dataloader, test_max_inter, top_k, group_lists, device):
    real_num_test_users = 0
    cumulative_results = {metric: 0.0 for metric in ['recall','hit_ratio','ndcg','rsp','reo']}
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
            train_pos = train_pos.cpu()
            batch_results, test_num_users = jud.judge(items = items, train_pos = train_pos, test_pos = test_pos, num_test_pos = num_test_pos, group_lists = group_lists)
            real_num_test_users = real_num_test_users + test_num_users
            
            for metric, value in batch_results.items():
                cumulative_results[metric] += value
                
        average_results = {metric: value / real_num_test_users for metric, value in cumulative_results.items()}
        
    return average_results