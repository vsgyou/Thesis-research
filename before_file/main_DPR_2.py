#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from tqdm import tqdm
import random
from metric import *
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader

from preprocess_10 import *
from model_DPR import DICE,Discriminator, train, valid, test
#%%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 이 부분을 수정하거나 제거합니다.
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # torch.backends.mps.manual_seed(seed) 이 줄은 오류가 발생하므로 주석 처리하거나 제거합니다.
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 설정 함수 호출
set_seed(44)
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


lr = 0.001
weight_decay = 5e-8
model = DICE(num_users= num_users, num_items = num_items, embedding_size = 64,dis_pen =0.01, int_weight = 0.1, pop_weight = 0.1, device = device)
discriminator = Discriminator(input_dim = 2, hidden_dim = 32, device = device)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay, betas = (0.5, 0.99), amsgrad = True)
disc_optimizer = optim.Adam(discriminator.parameters(), lr = lr)
epochs = 200
total_loss = 0.0

val_max_inter = valid_data.max_train_interaction
test_max_inter = test_data.max_train_interaction

# early stopping
best_val_result = float('-inf')
epochs_no_improve = 0
early_stop_patience = 5
best_model_path = "DPR_KL.pth"
with tqdm(range(1, epochs+1)) as tr:
    for epoch in tr:
        train_loss = train(model = model, train_loader = train_dataloader, optimizer = optimizer, device = device)
        val_result = valid(model = model, val_dataloader = val_dataloader, val_max_inter = val_max_inter, top_k = 20, device = device)

        print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
        print(f'epoch:{epoch}, valid_result: {val_result}')
        
        if val_result['recall'] > best_val_result:
            best_val_result = val_result['recall']
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break
model.load_state_dict(torch.load(best_model_path))
test_result_20 = test(model = model, test_dataloader = test_dataloader, test_max_inter = test_max_inter, top_k = 20, device = device)        
print(f'test_result_20:{test_result_20}')
test_result_50 = test(model = model, test_dataloader = test_dataloader, test_max_inter = test_max_inter, top_k = 50, device = device)  
print(f'test_result_50:{test_result_50}')

#%%
# 그래프그리기 
best_model_path = "DPR_KL_GPT.pth"

model.load_state_dict(torch.load(best_model_path))
item_embeddings = model.get_item_embeddings()
item_embeddings = torch.Tensor(item_embeddings).to(device)
user_embeddings = model.get_user_embeddings()
user_embeddings = torch.Tensor(user_embeddings).to(device)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

item_emb = item_embeddings.cpu()
user_emb = user_embeddings.cpu()

user_pop_top_values, user_pop_top_indices = torch.topk(user_pop, 100, largest = True)
user_pop_low_values, user_pop_low_indices = torch.topk(user_pop, 100, largest = False)
item_pop_top_values, item_pop_top_indices = torch.topk(blen_pop, 10, largest = True)
item_pop_low_values, item_pop_low_indices = torch.topk(blen_pop, 10, largest = False)
#%%


#%%
# 방법 1. 유저마다 정규화, 정규화 2번
top_user = torch.mm(user_emb[user_pop_top_indices], item_emb.T)
low_user = torch.mm(user_emb[user_pop_low_indices], item_emb.T)



# 각 행을 Min-Max 정규화하는 함수
def row_wise_min_max_normalize(tensor):
    min_vals = tensor.min(dim=1, keepdim=True)[0]  # 각 행의 최솟값
    max_vals = tensor.max(dim=1, keepdim=True)[0]  # 각 행의 최댓값
    return (tensor - min_vals) / (max_vals - min_vals)
top_user = row_wise_min_max_normalize(top_user)
low_user = row_wise_min_max_normalize(low_user)



# 네 가지 행렬 계산

# 데이터 플래튼 및 NumPy 변환
top_user_top_item = top_user[:,item_pop_top_indices].flatten().cpu().detach().numpy()
top_user_low_item = top_user[:,item_pop_low_indices].flatten().cpu().detach().numpy()
low_user_top_item = low_user[:,item_pop_top_indices].flatten().cpu().detach().numpy()
low_user_low_item = low_user[:,item_pop_low_indices].flatten().cpu().detach().numpy()
#%%
# 방법 2. 정규화도 4가지로
top_user_top_item = torch.mm(user_emb[user_pop_top_indices], item_emb[item_pop_top_indices].T)
top_user_low_item = torch.mm(user_emb[user_pop_top_indices], item_emb[item_pop_low_indices].T)
low_user_top_item = torch.mm(user_emb[user_pop_low_indices], item_emb[item_pop_top_indices].T)
low_user_low_item = torch.mm(user_emb[user_pop_low_indices], item_emb[item_pop_low_indices].T)

# minmiax 정규화
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

top_user_top_item = normalize(top_user_top_item.flatten().cpu().detach().numpy())
top_user_low_item = normalize(top_user_low_item.flatten().cpu().detach().numpy())
low_user_top_item = normalize(low_user_top_item.flatten().cpu().detach().numpy())
low_user_low_item = normalize(low_user_low_item.flatten().cpu().detach().numpy())

#%%

# 그래프를 그리는 함수 정의
def plot_density(data, title):
    kde = gaussian_kde(data)
    x_vals = np.linspace(0,1, 1000)
    y_vals = kde(x_vals)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label="Density")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.show()

# 각 데이터셋에 대해 밀도 함수 그래프 그리기
plot_density(top_user_top_item, "Density Function of top_user_top_item (Normalized Embeddings)")
plot_density(top_user_low_item, "Density Function of top_user_low_item (Normalized Embeddings)")
plot_density(low_user_top_item, "Density Function of low_user_top_item (Normalized Embeddings)")
plot_density(low_user_low_item, "Density Function of low_user_low_item (Normalized Embeddings)")
#%%
# 밀도 함수 그래프를 한 그래프에 그리기
def plot_all_densities(data_list, labels):
    plt.figure(figsize=(8, 6))
    
    for data, label in zip(data_list, labels):
        kde = gaussian_kde(data)
        x_vals = np.linspace(0, 1, 1000)
        y_vals = kde(x_vals)
        plt.plot(x_vals, y_vals, label=label)
    
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title("Density Functions")
    plt.legend()
    plt.show()

# 각 데이터셋에 대해 밀도 함수 그래프 그리기
data_list = [top_user_top_item, top_user_low_item, low_user_top_item, low_user_low_item]
labels = [
    "Density Function of top_user_top_item",
    "Density Function of top_user_low_item",
    "Density Function of low_user_top_item",
    "Density Function of low_user_low_item"
]

plot_all_densities(data_list, labels)
# %%
# 그래프그리기 
best_model_path = "DPR_only_KL_int_con_hap_5.0.pth"

model.load_state_dict(torch.load(best_model_path))
item_embeddings = model.get_item_embeddings()
item_embeddings = torch.Tensor(item_embeddings).to(device)
user_embeddings = model.get_user_embeddings()
user_embeddings = torch.Tensor(user_embeddings).to(device)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

item_emb = item_embeddings.cpu()
user_emb = user_embeddings.cpu()
total_user_emb_int = user_emb[:,:64]
total_user_emb_pop = user_emb[:,64:]
total_item_emb_int = item_emb[:,:64]
total_item_emb_pop = item_emb[:,64:]
total_int_score = torch.mm(total_user_emb_int,total_item_emb_int.T)
total_pop_score = torch.mm(total_user_emb_pop,total_item_emb_pop.T)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# normal_total_int_score = normalize(total_int_score.flatten().cpu().detach().numpy())
# normal_total_pop_score = normalize(total_pop_score.flatten().cpu().detach().numpy())

combined_scores = torch.cat([total_int_score.flatten(), total_pop_score.flatten()])

# 정규화를 진행 (전체 combined_scores에 대해)
normalized_combined_scores = normalize(combined_scores.cpu().detach().numpy())

# 정규화된 값을 다시 분리
split_point = total_int_score.numel()
normal_total_int_score = normalized_combined_scores[:split_point].reshape(total_int_score.shape)
normal_total_pop_score = normalized_combined_scores[split_point:].reshape(total_pop_score.shape)

from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

def plot_all_densities(data_list, labels):
    plt.figure(figsize=(8, 6))

    for data, label in zip(data_list, labels):
        data = data.flatten()  # 데이터를 1차원으로 변환
        if data.ndim > 1:  # 만약 아직 2차원 이상의 데이터를 갖고 있다면 오류 방지
            data = data.reshape(-1)
        
        kde = gaussian_kde(data)
        x_vals = np.linspace(0, 1, 1000)
        y_vals = kde(x_vals)
        plt.plot(x_vals, y_vals, label=label)

    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Density Functions")
    plt.show()

# Example usage:
data_list = [normal_total_int_score, normal_total_pop_score]
labels = ["Density Function of total_int_score", "Density Function of total_pop_score"]
plot_all_densities(data_list, labels)
# %%
