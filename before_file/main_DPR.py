#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from tqdm import tqdm
import math
from metric_DPR import *
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader

from preprocess_10 import *
from model_DPR import DICE, Discriminator, train, valid, test
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# device

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# train_root = 'netflix/train_coo_record.npz'
# valid_root = 'netflix/val_coo_record.npz'
# test_root = 'netflix/test_coo_record.npz'
# pop_root = 'netflix/popularity.npy'
# skew_train_root = 'netflix/train_skew_coo_record.npz'
# skew_pop_root = 'netflix/popularity_skew.npy'
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

all_item = np.ones(train_coo_record.shape[1], dtype = int)
all_item[unique_item] = counts
count = all_item
blen_pop = count / np.max(count)
blen_pop = torch.Tensor(blen_pop) + 1e-8
1/all_item
x_m = blen_pop.min()
weights = (1/all_item)**0.1
weights.min()
# calculate user propensity
sum_pop = blend_coo_record.dot(blen_pop)
count_inter = blend_coo_record.tocsr().sum(axis = 1).A1
count_inter_safe = np.where(count_inter == 0, 1, count_inter)
user_pop = sum_pop / count_inter_safe
user_pop = torch.Tensor(user_pop)
user_popularity = user_pop
#%%
plt.hist(popularity, bins=100, density=True, alpha=0.7, color='g')

# 파레토 분포 적합 (파라미터 추정)
param_pareto = pareto.fit(user_popularity)
x = np.linspace(min(user_popularity), max(user_popularity), 100)
pdf_fitted = pareto.pdf(x, *param_pareto)
plt.plot(x, pdf_fitted, 'r-', label="Pareto fit")

# 정규 분포 적합 (비교용)
param_norm = norm.fit(user_popularity)
pdf_fitted_norm = norm.pdf(x, *param_norm)
plt.plot(x, pdf_fitted_norm, 'b-', label="Normal fit")

plt.legend()
plt.title(f"User {user_id} Interaction Popularity Distribution")
plt.show()
#%%
user_popularity = user_popularity.numpy()
mean_popularity = np.mean(-user_popularity)

user_pop_plus = -user_popularity + (1-mean_popularity)
user_pop_plus_mean = user_pop_plus.mean()

plt.hist(weights, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Distribution of Scaled Weights (0.5 to 2.0)")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.show()
#%%
torch.max(user_pop)
ind = torch.argsort(user_pop,dim = -1,descending = True)
torch.mean(user_pop)
torch.median(user_pop)
sum(user_pop > torch.median(user_pop))

torch.max(np.log(1+1/user_pop))

sorted_indices = torch.argsort(blen_pop, descending = True)
pop_group = sorted_indices[:math.floor(len(sorted_indices)*0.2)]
unpop_group = sorted_indices[math.floor(len(sorted_indices)*0.2):]
group_lists = [pop_group.cpu().numpy(), unpop_group.cpu().numpy()]
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
discriminator = Discriminator(input_dim = 1, hidden_dim = 20, device = device)
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
best_model_path = "DPR_1.pth"
with tqdm(range(1, epochs+1)) as tr:
    for epoch in tr:
        train_loss = train(model = model, 
                           discriminator = discriminator, 
                           train_loader = train_dataloader, 
                           optimizer = optimizer, 
                           disc_optimizer = disc_optimizer, 
                           device = device)
        val_result = valid(model = model, 
                           val_dataloader = val_dataloader, 
                           val_max_inter = val_max_inter, 
                           top_k = 20, 
                           device = device)

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
best_model_path = 'DPR_plus_KL_weight_10_0.pth'
model.load_state_dict(torch.load(best_model_path))
test_result_20 = test(model = model, test_dataloader = test_dataloader, test_max_inter = test_max_inter, top_k = 20, device = device)        
print(f'test_result_20:{test_result_20}')
test_result_50 = test(model = model, test_dataloader = test_dataloader, test_max_inter = test_max_inter, top_k = 50, device = device)  
print(f'test_result_50:{test_result_50}')

# %%
#%%
# 그래프그리기 
best_model_path = "Adv_KL_divid_version_learn_0.02_0.001_weight_100_50_nonorm_pos4_dim20_8192.pth"

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

user_pop_top_values, user_pop_top_indices = torch.topk(user_pop, 10, largest = True)
user_pop_low_values, user_pop_low_indices = torch.topk(user_pop, 10, largest = False)
item_pop_top_values, item_pop_top_indices = torch.topk(blen_pop, 100, largest = True)
item_pop_low_values, item_pop_low_indices = torch.topk(blen_pop, 100, largest = False)
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

top_user_top_item_int = torch.mm(user_emb[user_pop_top_indices,:64],item_emb[item_pop_top_indices,:64].T)
top_user_top_item_pop = torch.mm(user_emb[user_pop_top_indices,64:],item_emb[item_pop_top_indices,64:].T)

top_user_low_item_int = torch.mm(user_emb[user_pop_top_indices,:64],item_emb[item_pop_low_indices,:64].T)
top_user_low_item_pop = torch.mm(user_emb[user_pop_top_indices,64:],item_emb[item_pop_low_indices,:64].T)

low_user_top_item_int = torch.mm(user_emb[user_pop_low_indices,:64],item_emb[item_pop_top_indices,:64].T)
low_user_top_item_pop = torch.mm(user_emb[user_pop_low_indices,64:],item_emb[item_pop_top_indices,64:].T)

low_user_low_item_int = torch.mm(user_emb[user_pop_low_indices,:64],item_emb[item_pop_low_indices,:64].T)
low_user_low_item_pop = torch.mm(user_emb[user_pop_low_indices,64:],item_emb[item_pop_low_indices,:64].T)


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
# 각 데이터셋에 대해 밀도 함수 그래프 그리기
data_list = [top_user_top_item_int, top_user_top_item_pop, top_user_low_item_int, top_user_low_item_pop]
labels = [
    "Density Function of top_user_top_item_int",
    "Density Function of top_user_top_item_pop",
    "Density Function of top_user_low_item_int",
    "Density Function of top_user_low_item_pop"
]

plot_all_densities(data_list, labels)
#%%
# 모델 로드 및 임베딩 준비
# best_model_path = 'Netflix_pretrain_DICE_KL_hap_no_opti_test_best_model_adv4000_kl100_dice_lr0.01_disc_lr0.001_batch8192_optimizer_conti.pth'
# best_model_path = "pretrain_DICE_KL_hap_no_opti_2_seed_45_conti_best_model_adv1000_kl100_dice_lr0.01_disc_lr0.001_batch8192_optimizer.pth"
best_model_path = "best_model_DICE_MSE_MSE_42_last_optimizer.pth"
# best_model_path = "Netflix_best_model_DICE_MSE_SEED_42.pth"
model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
optimizer.load_state_dict(torch.load(best_model_path)['optimizer_state_dict'])
# optimizer.param_groups[0]['lr'] = 0.01
item_embeddings = model.get_item_embeddings()
item_embeddings = torch.Tensor(item_embeddings).to(device)
user_embeddings = model.get_user_embeddings()
user_embeddings = torch.Tensor(user_embeddings).to(device)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# CPU로 데이터 변환
item_emb = item_embeddings.cpu()
user_emb = user_embeddings.cpu()
# 상위 및 하위 유저와 아이템 선택
user_pop_top_values, user_pop_top_indices = torch.topk(user_pop, 100, largest=True)
user_pop_low_values, user_pop_low_indices = torch.topk(user_pop, 100, largest=False)
item_pop_top_values, item_pop_top_indices = torch.topk(blen_pop, 500, largest=True)
item_pop_low_values, item_pop_low_indices = torch.topk(blen_pop, 500, largest=False)

# 스코어 계산: interest와 conformity의 합
top_user_top_item = torch.mm(user_emb[user_pop_top_indices], item_emb[item_pop_top_indices].T)
top_user_low_item = torch.mm(user_emb[user_pop_top_indices], item_emb[item_pop_low_indices].T)
low_user_top_item = torch.mm(user_emb[user_pop_low_indices], item_emb[item_pop_top_indices].T)
low_user_low_item = torch.mm(user_emb[user_pop_low_indices], item_emb[item_pop_low_indices].T)

# Flatten 및 정규화 없이 numpy 변환
top_user_top_item = top_user_top_item.flatten().cpu().detach().numpy()
top_user_low_item = top_user_low_item.flatten().cpu().detach().numpy()
low_user_top_item = low_user_top_item.flatten().cpu().detach().numpy()
low_user_low_item = low_user_low_item.flatten().cpu().detach().numpy()

# 밀도 함수 그리기 함수 정의
def plot_all_densities(data_list, labels):
    plt.figure(figsize=(8, 6))
    
    for data, label in zip(data_list, labels):
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 1000)
        y_vals = kde(x_vals)
        plt.plot(x_vals, y_vals, label=label,linewidth=2.0)
    
    plt.xlabel("Values",fontsize=12, fontweight='bold')
    plt.ylabel("Density",fontsize=12, fontweight='bold')
    plt.title("Density Functions", fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold')  # X축 눈금 글씨 크기 확대
    plt.yticks(fontsize=10, fontweight='bold')
    # 범례를 그래프 내부 우측 상단에 배치
    plt.legend(loc='upper left', frameon=True,prop={'size': 10, 'weight': 'bold'})
    plt.show()

# 합산된 스코어에 대한 밀도 함수 그래프 그리기
data_list = [top_user_top_item, top_user_low_item, low_user_top_item, low_user_low_item]
labels = [
    "Density Function of top_user_top_item",
    "Density Function of top_user_low_item",
    "Density Function of low_user_top_item",
    "Density Function of low_user_low_item"
]



plot_all_densities(data_list, labels)
#%%
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 밀도 함수 그리기 함수 정의
def plot_all_densities(data_list, labels):
    plt.figure(figsize=(12, 8))  # 그래프 크기 조정

    # 새로운 고급스러운 색상 팔레트
    colors = ['#E15759', '#F28E2B', '#4E79A7', '#76B7B2']  # 차분한 블루, 오렌지, 그린, 레드

    for data, label, color in zip(data_list, labels, colors):
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 1000)
        y_vals = kde(x_vals)
        plt.plot(
            x_vals, y_vals, label=label, linewidth=4.0, color=color
        )  # 밀도 선의 굵기를 더 두껍게 설정 (linewidth=4.0)

    # 그래프 디자인 개선
    plt.xlabel("Score", fontsize=25, fontweight='bold')  # X축 레이블 크기 확대
    plt.ylabel("Density", fontsize=25, fontweight='bold')  # Y축 레이블 크기 확대
    plt.title("DICE Model Density Plot", fontsize=25, fontweight='bold')  # 제목 크기 확대
    plt.xticks(fontsize=22, fontweight='bold')  # X축 눈금 글씨 크기 확대
    plt.yticks(fontsize=22, fontweight='bold')  # Y축 눈금 글씨 크기 확대

    # 범례 위치를 그래프 내부로 설정 (오른쪽 위)
    plt.legend(
        loc='upper left', frameon=True, shadow=False, borderpad=1.2,
        bbox_transform=plt.gca().transAxes, facecolor='white', edgecolor='gray', fancybox=True, framealpha=0.9,
        fontsize=20, prop={'size': 20, 'weight': 'bold'}  # 글씨 크기와 굵기 명시
    )

    # 그리드 추가
    plt.grid(alpha=0.4, linestyle='--')

    # 레이아웃 조정
    plt.tight_layout()
    plt.show()

# 합산된 스코어에 대한 밀도 함수 그래프 그리기
data_list = [top_user_top_item, top_user_low_item, low_user_top_item, low_user_low_item]
labels = [
    "Top User, Top Item",
    "Top User, Low Item",
    "Low User, Top Item",
    "Low User, Low Item"
]

plot_all_densities(data_list, labels)




#%%
# 여러개
# 모델 경로 리스트
best_model_paths = [
    'Netflix_pretrain_DICE_KL_hap_no_opti_test_best_model_adv1000_kl100_dice_lr0.01_disc_lr0.001_batch8192_optimizer_conti.pth', 
    'Netflix_pretrain_DICE_KL_hap_no_opti_test_best_model_adv2000_kl100_dice_lr0.01_disc_lr0.001_batch8192_optimizer_conti.pth',
    'Netflix_pretrain_DICE_KL_hap_no_opti_test_best_model_adv3000_kl100_dice_lr0.01_disc_lr0.001_batch8192_optimizer_conti.pth',
    'Netflix_pretrain_DICE_KL_hap_no_opti_test_best_model_adv4000_kl100_dice_lr0.01_disc_lr0.001_batch8192_optimizer_conti.pth',
    'Netflix_pretrain_DICE_KL_hap_no_opti_test_best_model_adv5000_kl100_dice_lr0.01_disc_lr0.001_batch8192_optimizer_conti.pth'
    ]# 밀도 함수 그리기 함수 정의
def plot_all_densities(data_list, labels, title_suffix):
    plt.figure(figsize=(8, 6))
    
    for data, label in zip(data_list, labels):
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 1000)
        y_vals = kde(x_vals)
        plt.plot(x_vals, y_vals, label=label)
    
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title(f"Density Functions {title_suffix}")
    plt.legend()
    plt.show()

# 각 모델에 대해 반복 수행
for best_model_path in best_model_paths:
    # 모델 로드 및 임베딩 추출
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    item_embeddings = model.get_item_embeddings()
    item_embeddings = torch.Tensor(item_embeddings).to(device)
    user_embeddings = model.get_user_embeddings()
    user_embeddings = torch.Tensor(user_embeddings).to(device)
    
    # CPU로 데이터 변환
    item_emb = item_embeddings.cpu()
    user_emb = user_embeddings.cpu()
    
    # 상위 및 하위 유저와 아이템 선택
    user_pop_top_values, user_pop_top_indices = torch.topk(user_pop, 100, largest=True)
    user_pop_low_values, user_pop_low_indices = torch.topk(user_pop, 100, largest=False)
    item_pop_top_values, item_pop_top_indices = torch.topk(blen_pop, 500, largest=True)
    item_pop_low_values, item_pop_low_indices = torch.topk(blen_pop, 500, largest=False)
    
    # 스코어 계산: interest와 conformity의 합
    top_user_top_item = torch.mm(user_emb[user_pop_top_indices], item_emb[item_pop_top_indices].T)
    top_user_low_item = torch.mm(user_emb[user_pop_top_indices], item_emb[item_pop_low_indices].T)
    low_user_top_item = torch.mm(user_emb[user_pop_low_indices], item_emb[item_pop_top_indices].T)
    low_user_low_item = torch.mm(user_emb[user_pop_low_indices], item_emb[item_pop_low_indices].T)
    
    # Flatten 및 정규화 없이 numpy 변환
    top_user_top_item = top_user_top_item.flatten().cpu().detach().numpy()
    top_user_low_item = top_user_low_item.flatten().cpu().detach().numpy()
    low_user_top_item = low_user_top_item.flatten().cpu().detach().numpy()
    low_user_low_item = low_user_low_item.flatten().cpu().detach().numpy()
    
    # 합산된 스코어에 대한 밀도 함수 그래프 그리기
    data_list = [top_user_top_item, top_user_low_item, low_user_top_item, low_user_low_item]
    labels = [
        "Density Function of top_user_top_item",
        "Density Function of top_user_low_item",
        "Density Function of low_user_top_item",
        "Density Function of low_user_low_item"
    ]
    # 모델 경로에 따라 그래프 제목에 구분을 두기 위해 경로 이름 일부를 추가
    plot_all_densities(data_list, labels, f"for {best_model_path.split('/')[-1]}")
# %%
# int, con 비교
top_user_int = torch.mm(user_emb[user_pop_top_indices,:64], item_emb[:,:64].T)
top_user_pop = torch.mm(user_emb[user_pop_top_indices,64:], item_emb[:,64:].T)
top_user_int = row_wise_min_max_normalize(top_user_int)
top_user_pop = row_wise_min_max_normalize(top_user_pop)

top_user_top_item_int = top_user_int[:,item_pop_top_indices].flatten().cpu().detach().numpy()
top_user_top_item_pop = top_user_pop[:,item_pop_top_indices].flatten().cpu().detach().numpy()
top_user_low_item_int = top_user_int[:,item_pop_low_indices].flatten().cpu().detach().numpy()
top_user_low_item_pop = top_user_pop[:,item_pop_low_indices].flatten().cpu().detach().numpy()

top_user_top_item_int = normalize(top_user_top_item_int.flatten().cpu().detach().numpy())
top_user_top_item_pop = normalize(top_user_top_item_pop.flatten().cpu().detach().numpy())
top_user_low_item_int = normalize(top_user_low_item_int.flatten().cpu().detach().numpy())
top_user_low_item_pop = normalize(top_user_low_item_pop.flatten().cpu().detach().numpy())

# %%
best_discriminator_path = 'Adv_KL_divid_version_learn_0.02_0.001_weight_10_1_nonorm_pos4_dim20_8192.pth'
discriminator.load_state_dict(torch.load(best_discriminator_path))

top_user_top_item_int = torch.mm(user_emb[user_pop_top_indices,:64],item_emb[item_pop_top_indices,:64].T).unsqueeze(-1)
top_user_top_item_pop = torch.mm(user_emb[user_pop_top_indices,64:],item_emb[item_pop_top_indices,64:].T).unsqueeze(-1)

top_user_low_item_int = torch.mm(user_emb[user_pop_top_indices,:64],item_emb[item_pop_low_indices,:64].T).unsqueeze(-1)
top_user_low_item_pop = torch.mm(user_emb[user_pop_top_indices,64:],item_emb[item_pop_low_indices,:64].T).unsqueeze(-1)

low_user_top_item_int = torch.mm(user_emb[user_pop_low_indices,:64],item_emb[item_pop_top_indices,:64].T).unsqueeze(-1)
low_user_top_item_pop = torch.mm(user_emb[user_pop_low_indices,64:],item_emb[item_pop_top_indices,64:].T).unsqueeze(-1)

low_user_low_item_int = torch.mm(user_emb[user_pop_low_indices,:64],item_emb[item_pop_low_indices,:64].T).unsqueeze(-1)
low_user_low_item_pop = torch.mm(user_emb[user_pop_low_indices,64:],item_emb[item_pop_low_indices,:64].T).unsqueeze(-1)


top_user_top_item_score = torch.concat((top_user_top_item_int,top_user_top_item_pop),axis=2)
top_user_low_item_score = torch.concat((top_user_low_item_int, top_user_low_item_pop),axis=2)
low_user_top_item_score = torch.concat((low_user_top_item_int,low_user_top_item_pop),axis=2)
low_user_low_item_score = torch.concat((low_user_low_item_int,low_user_low_item_pop),axis=2)

top_user_top_item_g = discriminator(top_user_top_item_score)
top_user_low_item_g = discriminator(top_user_low_item_score)
low_user_top_item_g = discriminator(low_user_top_item_score)
low_user_low_item_g = discriminator(low_user_low_item_score)

top_user_top_item_g = (top_user_top_item_g[:,:,0] + top_user_top_item_g[:,:,1]).squeeze()
top_user_low_item_g = (top_user_low_item_g[:,:,0] + top_user_low_item_g[:,:,1]).squeeze()
low_user_top_item_g = (low_user_top_item_g[:,:,0] + low_user_top_item_g[:,:,1]).squeeze()
low_user_low_item_g = (low_user_low_item_g[:,:,0] + low_user_low_item_g[:,:,1]).squeeze()

# def normalize(data):
#     min_val = np.min(data)
#     max_val = np.max(data)
    
#     # 최대값과 최소값이 같은 경우를 처리 (모든 데이터가 동일한 경우)
#     if max_val == min_val:
#         return np.zeros_like(data)  # 모든 값을 0으로 반환
#     else:
#         return (data - min_val) / (max_val - min_val + 1e-8)  # 작은 수를 더해 0으로 나누는 것을 방지

# top_user_top_item_g = normalize(top_user_top_item_g.flatten().cpu().detach().numpy())
# top_user_low_item_g = normalize(top_user_low_item_g.flatten().cpu().detach().numpy())
# low_user_top_item_g = normalize(low_user_top_item_g.flatten().cpu().detach().numpy())
# low_user_low_item_g = normalize(low_user_low_item_g.flatten().cpu().detach().numpy())
#%%


# %%
for users, p_item, n_item, mask in tqdm(train_dataloader):
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
    
    p_total_score = torch.cat((p_score_int,p_score_pop), axis=0)
    n_total_score = torch.cat((n_score_int, n_score_pop), axis =0)
    p_total_input = discriminator(p_total_score.cpu())
    p_total_input[:,1,:]
    n_total_input = discriminator(n_total_score.cpu())
    
    p_total_input[128:,1,:]
    interest_labels = torch.ones(n_score_int.size(), dtype=torch.float32).to(device)
    conformity_labels = torch.zeros(n_score_pop.size(), dtype=torch.float32).to(device)    
    p_labels_total = torch.concat((interest_labels, conformity_labels), axis = 0)

    n_labels_total = torch.concat((interest_labels, conformity_labels), axis = 0)
    
    labels_total = torch.concat((interest_labels, conformity_labels), axis = 2)
    
    adv_loss_p = discriminator.compute_adv_loss(p_total_input,p_labels_total.cpu())

for name, param in discriminator.named_parameters():
    print(f"Parameter {name}:")
    print(param.data) 

p_score_int
p_score_pop
(p_score_int + p_score_pop).shape
p_total_score = torch.concat((p_score_int, p_score_pop), axis = 0)
p_total_score
n_total_score = torch.concat((n_score_int, n_score_pop), axis = 0)
score_int = torch.matmul(users_int[:,1,:], items_int_weight.t())  # (batch_size, num_items)
score_pop = torch.matmul(users_pop[:,1,:], items_pop_weight.t())  # (batch_size, num_items)

p_total_input = discriminator(p_total_score)
n_total_input= discriminator(n_total_score)

interest_labels = torch.tensor([1,0],dtype=torch.float32).repeat(n_score_int.size()).to(device)
interest_labels.shape
conformity_labels = torch.tensor([0,1],dtype=torch.float32).repeat(n_score_pop.size()).to(device)
labels_total = torch.concat((interest_labels, conformity_labels), axis = 0)
p_total_score = torch.cat((p_score_int,p_score_pop),axis=0)

p_labels_total = torch.concat((interest_labels[:,1,:],conformity_labels[:,1,:]),axis=0)
n_labels_total = torch.concat((interest_labels, conformity_labels), axis=0) 
aa = nn.Sequential(nn.Linear(1,32),
                    nn.ReLU(),
                    nn.Linear(32,8),
                    nn.ReLU(),
                    nn.Linear(8,2),
                    nn.Sigmoid()
                    )
p_total_score_aa = aa(p_total_score.cpu())
p_total_score_aa.shape
n_total_score_aa = aa(n_total_score.cpu())
n_total_score_aa.shape
p_labels_total.shape
n_labels_total.shape
cri = nn.BCELoss()
cri(p_total_score_aa,p_labels_total.cpu())
cri(n_total_score_aa,n_labels_total.cpu())
a = user_emb[users.cpu(),:]
b = torch.matmul(a[:,1,:],item_emb.t())
mean = b.mean(dim=1,keepdim=True)
std = b.std(dim=1,keepdim=True)

users_pop = user_pop[users.cpu()]
users_pop = users_pop[:,1]

std = std + 1e-8

norm_dist = torch.distributions.Normal(mean, std)
user_pop_median = torch.median(user_pop)
target_mean_high = 0.7
target_mean_low = 0.3
target_std = torch.ones_like(std)

target_mean = torch.where(users_pop >= user_pop_median, target_mean_high, target_mean_low )
standard_norm = torch.distributions.Normal(target_mean, target_std)

kl_div = torch.distributions.kl.kl_divergence(norm_dist, standard_norm)
# %%

#%%

def gini_index(item_impression):
    impressions_sorted = np.sort(item_impression)
    n = len(item_impression)
    index = np.arange(1, n+1)
    gini_numerator = np.sum((2 * index -n -1) * impressions_sorted)
    gini_denominator = n*np.sum(impressions_sorted)
    return gini_numerator / gini_denominator if gini_denominator != 0 else 0
#%%
# best_model_path = 'parse_model/optim_pretrain_best_model_adv100_kl70_dice_lr0.01_disc_lr0.0005_batch8192.pth'
# model.load_state_dict(torch.load(best_model_path))
best_model_path = 'Netflix_pretrain_DICE_KL_test_no_opti_conti_best_model_adv1000_kl500_dice_lr0.01_disc_lr0.001_batch8192_optimizer.pth'
model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
item_embeddings = model.get_item_embeddings()
item_embeddings = torch.Tensor(item_embeddings).to(device)
user_embeddings = model.get_user_embeddings()
user_embeddings = torch.Tensor(user_embeddings).to(device)

top_k=20
top_k=50
def test(model, test_dataloader, test_max_inter, top_k, device):
    real_num_test_users = 0
    cumulative_results = {metric: 0.0 for metric in ['recall','hit_ratio','ndcg','rsp','reo']}
    item_impression = np.zeros(model.get_item_embeddings().shape[0])
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
            train_pos = train_pos.cpu()
            test_pos = test_pos.cpu()
# gene 지수 계산
            unique_items, counts = np.unique(items.numpy(), return_counts=True)
            item_impression[unique_items] += counts
            
            batch_results, test_num_users = jud.judge(items = items, train_pos = train_pos, test_pos = test_pos, num_test_pos = num_test_pos, group_lists = group_lists)
            real_num_test_users = real_num_test_users + test_num_users
            
            for metric, value in batch_results.items():
                cumulative_results[metric] += value
                
        average_results = {metric: value / real_num_test_users for metric, value in cumulative_results.items()}
        
    return average_results
gini_index(item_impression)
np.sum(item_impression[pop_group]) / np.sum(item_impression)

# %%
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


model.train()
model.load_state_dict(torch.load(best_model_path))

print("Loaded model parameters:", list(model.parameters())[0][:5])



adv_weight = 100
kl_weight = 1000
dice_lr = 0.01
disc_lr = 0.001
batch_size = 4096
pretrain_model_path = f'parse_best_model/pretrain_DICE_KL_hap_no_opti_best_model_adv{adv_weight}_kl{kl_weight}_dice_lr{dice_lr}_disc_lr{disc_lr}_batch{batch_size}_optimizer.pth'


#%%
# t-sne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 가정: item_embeddings는 이미 정의되어 있고 shape이 (N, 128)인 numpy 배열
# Interest와 Conformity 임베딩 분리
int_item_embeddings = item_embeddings[:, :64].cpu().numpy()  # Interest embeddings
con_item_embeddings = item_embeddings[:, 64:].cpu().numpy()  # Conformity embeddings

# TSNE 변환 수행
n_components = 3  # 저차원으로 변환 (2D)
tsne = TSNE(n_components=n_components, random_state=42)

# 두 임베딩을 TSNE로 변환
int_tsne_embeddings = tsne.fit_transform(int_item_embeddings)

con_tsne_embeddings = tsne.fit_transform(con_item_embeddings)

# 시각화
plt.figure(figsize=(10, 6))

# Interest embeddings 시각화
plt.scatter(
    int_tsne_embeddings[:, 0], int_tsne_embeddings[:, 1], int_tsne_embeddings[:, 2]
    c='blue', label='Interest Embeddings', alpha=0.6, edgecolor='k'
)

# Conformity embeddings 시각화
plt.scatter(
    con_tsne_embeddings[:, 0], con_tsne_embeddings[:, 1], int_tsne_embeddings[:, 2]
    c='red', label='Conformity Embeddings', alpha=0.6, edgecolor='k'
)

# 그래프 설정
plt.title("t-SNE Visualization of Item Embeddings", fontsize=16)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# %%
def plot_all_densities(data_list, labels):
    plt.figure(figsize=(8, 6))
    
    for data, label in zip(data_list, labels):
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 1000)
        y_vals = kde(x_vals)
        plt.plot(x_vals, y_vals, label=label)
    
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title("Density Functions")
    # 범례를 그래프 내부 우측 상단에 배치
    plt.legend(loc='upper left', frameon=True)
    plt.show()

# 합산된 스코어에 대한 밀도 함수 그래프 그리기
data_list = [torch.Tensor(int_item_embeddings).flatten(), torch.Tensor(con_item_embeddings).flatten()]
labels = [
    "Density Function of int_item_embeddings",
    "Density Function of con_item_emeddings",
]

plot_all_densities(data_list, labels)
# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 예제 데이터: Interest 및 Conformity 임베딩
int_item_embeddings = item_embeddings[:, :64].cpu().numpy()  # Interest embeddings
con_item_embeddings = item_embeddings[:, 64:].cpu().numpy()  # Conformity embeddings

# PCA 적용 (2D로 축소)
pca = PCA(n_components=2)
int_pca = pca.fit_transform(int_item_embeddings)
con_pca = pca.fit_transform(con_item_embeddings)

# 각 데이터의 평균 계산 (중심점)
int_centroid = int_pca.mean(axis=0)
con_centroid = con_pca.mean(axis=0)

# 시각화
fig, ax = plt.subplots(figsize=(8, 6))

# PCA 산점도 시각화
ax.scatter(int_pca[:, 0], int_pca[:, 1], color='blue', alpha=0.5, label='Interest Embeddings')
ax.scatter(con_pca[:, 0], con_pca[:, 1], color='green', alpha=0.5, label='Conformity Embeddings')

# 중심점 화살표 (벡터 방향)
ax.quiver(0, 0, int_centroid[0], int_centroid[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Interest Direction')
ax.quiver(0, 0, con_centroid[0], con_centroid[1], angles='xy', scale_units='xy', scale=1, color='green', label='Conformity Direction')

# 축 설정
ax.set_xlabel('PCA Dimension 1')
ax.set_ylabel('PCA Dimension 2')
ax.set_title('PCA Visualization of Interest and Conformity Embeddings')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.axvline(0, color='black', linewidth=0.5, linestyle='--')

# 범례 추가
ax.legend()

# 그래프 출력
plt.grid()
plt.show()

# %%
