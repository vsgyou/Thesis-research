#%%
import math
import numpy as np
import torch
import torch.nn as nn
# from torch.nn.parameter import Parameter
import torch.nn.functional as F
import random
from preprocess import *
from metric import *
import torch.optim as optim

# %%

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
        
    def pearson_correlation(self, x, y):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        mean_y = torch.mean(y, dim=1, keepdim=True)
        
        xm = x - mean_x
        ym = y - mean_y
        
        r_num = torch.sum(xm * ym, dim=1)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1))
        r = r_num / (r_den + 1e-8)
        
        return -torch.mean(r)
    # def bpr_loss(self, p_score, n_score):
    #     return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))
    
    # def mask_bpr_loss(self, p_score, n_score, mask):
    #     return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))
    def bpr_loss(self, p_score, n_score):
        epsilon = 1e-10
        return -torch.mean(torch.log(torch.sigmoid(p_score-n_score) + epsilon))
    
    def mask_bpr_loss(self, p_score, n_score, mask):
        epsilon = 1e-10
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score-n_score)+epsilon))
    
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
        # discrepancy_loss = self.pearson_correlation(item_int, item_pop) + self.pearson_correlation(user_int, user_pop)             
        loss = self.int_weight * loss_int + self.pop_weight * loss_pop + loss_total - self.dis_pen * discrepancy_loss
        return loss
    
    def get_item_embeddings(self):
        item_embeddings = torch.cat((self.items_int.weight, self.items_pop.weight), 1)
        return item_embeddings.detach().cpu().numpy().astype('float32')
    
    def get_user_embeddings(self):
        user_embeddings = torch.cat((self.users_int.weight, self.users_pop.weight), 1)
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