#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from tqdm import tqdm

from metric import *
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader

from preprocess_10 import *
from model_DICE import DICE, train, valid, test
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
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay, betas = (0.5, 0.99), amsgrad = True)
epochs = 500
total_loss = 0.0

val_max_inter = valid_data.max_train_interaction
test_max_inter = test_data.max_train_interaction


with tqdm(range(1, epochs+1)) as tr:
    for epoch in tr:
        train_loss = train(model = model, train_loader = train_dataloader, optimizer = optimizer, device = device)
        val_result = valid(model = model, val_dataloader = val_dataloader, val_max_inter = val_max_inter, top_k = 20, device = device)
        
        if epoch % 10 == 0:
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
            print(f'epoch:{epoch}, valid_result: {val_result}')
            
test_result = test(model = model, test_dataloader = test_dataloader, test_max_inter = test_max_inter, top_k = 20, device = device)        
# %%
# test_data에서 인기에 민감도와 추천해준 아이템들의 인기도의 상관관계
top_k = 20
best_model_path = "best_model_DICE_COR.pth"
best_model_path = 'DPR_KL.pth'
model.load_state_dict(torch.load(best_model_path))

item_embeddings = model.get_item_embeddings()
item_embeddings = torch.Tensor(item_embeddings).to(device)
user_embeddings = model.get_user_embeddings()
user_embeddings = torch.Tensor(user_embeddings).to(device)
generator = FaissInnerProductMaximumSearchGenerator(item_embeddings, device = device)
jud = Judger(topk = top_k)
test_user_pop = []
test_item_pop_mean = []

max_recall = float('-inf')
min_recall = float('inf')
max_batch = None
min_batch = None

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
    
    batch_results, test_num_users = jud.judge(items=items, test_pos=test_pos, num_test_pos=num_test_pos)

    # 'recall' 값을 가져와서 정규화
    recall_value = batch_results['recall'] / test_num_users
        
    # 최고 recall 값을 가진 배치 확인 및 저장
    if recall_value > max_recall:
        max_recall = recall_value
        max_batch = (users.cpu(), train_pos.cpu(), test_pos, num_test_pos.cpu(), items, batch_results)
    
    # 최저 recall 값을 가진 배치 확인 및 저장
    if recall_value < min_recall:
        min_recall = recall_value
        min_batch = (users.cpu(), train_pos.cpu(), test_pos, num_test_pos.cpu(), items, batch_results)
        
    print(f"Recall: {recall_value}")

# 최종 결과 출력 (선택 사항)
print(f"Max Recall: {max_recall}")
print(f"Min Recall: {min_recall}")
#%%
values, indices = torch.topk(user_pop, 1, largest = False)
top_pop_user = indices
top_pop_user = torch.Tensor([2697]).long()
top_user_emb = user_embeddings[top_pop_user]
top_user_item = generator.generate(top_user_emb, 20 + test_max_inter)
top_user_item = np.array(top_user_item[:,:20]).flatten()
torch.mean(blen_pop[top_user_item])
# 그래프
plt.subplot(1, 2, 1)
sns.histplot(blen_pop[top_user_item].numpy(), bins=50, kde=False, color='blue')
plt.title('Histogram of DICE Popularity')


# conformity만으로 추천
item_int_emb = item_embeddings[:,:64]
item_con_emb = item_embeddings[:,64:]

user_int_emb = user_embeddings[:,:64]
user_con_emb = user_embeddings[:,64:]

con_generator = FaissInnerProductMaximumSearchGenerator(item_con_emb, device = device)
top_user_con_emb = user_con_emb[top_pop_user]
top_user_item_con = con_generator.generate(top_user_con_emb, 20 + test_max_inter)
top_user_item_con = np.array(top_user_item_con[:,:20]).flatten()
DICE_blen_pop_con = blen_pop[top_user_item_con].numpy()
np.mean(DICE_blen_pop_con)

plt.subplot(1, 2, 1)
sns.histplot(DICE_blen_pop_con, bins=50, kde=False, color='blue')
plt.title('Histogram of DICE Conformity Popularity')
# interest만으로 추천
int_generator = FaissInnerProductMaximumSearchGenerator(item_int_emb, device = device)
top_user_int_emb = user_int_emb[top_pop_user]
top_user_item_int = int_generator.generate(top_user_int_emb, 20 + test_max_inter)
top_user_item_int = np.array(top_user_item_int[:,:20]).flatten()
DICE_blen_pop_int = blen_pop[top_user_item_int].numpy()
np.mean(DICE_blen_pop_int)

plt.subplot(1, 2, 1)
sns.histplot(DICE_blen_pop_int, bins=50, kde=False, color='blue')
plt.title('Histogram of DICE Interest Popularity')


# %%
int_item = torch.matmul(top_user_con_emb,item_con_emb.T)
con_item = torch.matmul(top_user_int_emb,item_int_emb.T)

# numpy array로 변환
int_item_np = int_item.cpu().numpy().flatten()
con_item_np = con_item.cpu().numpy().flatten()

# 두 임베딩을 하나의 배열로 합침
combined = (int_item_np + con_item_np)/2
# 상위 20개의 값 식별
top_20_indices = np.argsort(combined)[-20:]
# 상위 20개의 인기도
torch.mean(blen_pop[top_20_indices]) 
# 시각화
plt.figure(figsize=(12, 6))

# 전체 데이터 시각화
plt.scatter(range(len(int_item_np)), int_item_np, color='red', label='int_item', s=10)
plt.scatter(range(len(int_item_np)), con_item_np, color='blue', label='con_item', s=10)
plt.legend()
plt.title('int_item and con_item scores')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score value')
plt.show()
#plt.scatter(range(len(int_item_np)),combined, color = 'green',label = 'combined',s=10)
# 상위 20개 점 강조 표시
for idx in top_20_indices:
    if idx < 4819:
        plt.scatter(idx, combined[idx], edgecolor='green', facecolor='none', linewidth=2, s=100)
        plt.scatter(idx, int_item_np[idx], edgecolor = 'pink',facecolor='none',linewidth = 2, s=100)
        plt.scatter(idx, con_item_np[idx], edgecolor = 'skyblue',facecolor='none',linewidth = 2, s=100)
    else:
        plt.scatter(idx, combined[idx], edgecolor='green', facecolor='none', linewidth=2, s=100)
        plt.scatter(idx, int_item_np[idx], edgecolor = 'pink',facecolor='none',linewidth = 2, s=100)
        plt.scatter(idx, con_item_np[idx], edgecolor = 'skyblue',facecolor='none',linewidth = 2, s=100)

plt.legend()
plt.title('int_item and con_item Values with Top 20 Points Highlighted')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

plt.figure(figsize = (12,6))
plt.scatter(range(len(int_item_np)),combined, color = 'green',label = 'combined',s=10)
#%%
plt.scatter(range(len(top_20_indices)), int_item_np[top_20_indices],color = 'red')
plt.scatter(range(len(top_20_indices)), con_item_np[top_20_indices],color = 'blue')
plt.scatter(range(len(top_20_indices)), combined[top_20_indices],color = 'green')



# %%
pop_large = np.argsort(blen_pop)[-20:]
plt.scatter(range(len(int_item_np)), int_item_np, color='red', label='int_item', s=10)
plt.scatter(range(len(int_item_np)), con_item_np, color='blue', label='con_item', s=10)
plt.scatter(range(len(int_item_np)),combined, color = 'green',label = 'combined',s=10)
# 상위 20개 점 강조 표시
for idx in pop_large:
    if idx < 4819:
        plt.scatter(idx, combined[idx], edgecolor='black', facecolor='none', linewidth=2, s=100)
        plt.scatter(idx, int_item_np[idx], edgecolor = 'red',facecolor='none',linewidth = 2, s=100)
        plt.scatter(idx, con_item_np[idx], edgecolor = 'blue',facecolor='none',linewidth = 2, s=100)
    else:
        plt.scatter(idx, combined[idx], edgecolor='black', facecolor='none', linewidth=2, s=100)
        plt.scatter(idx, int_item_np[idx], edgecolor = 'red',facecolor='none',linewidth = 2, s=100)
        plt.scatter(idx, con_item_np[idx], edgecolor = 'blue',facecolor='none',linewidth = 2, s=100)


plt.scatter(range(len(np.array(blen_pop))), np.array(blen_pop), color = 'skyblue',s=10)
for idx in top_20_indices:
    if idx < 4819:
        plt.scatter(idx, np.array(blen_pop[idx]), edgecolor='pink', facecolor='none', linewidth=2, s=100)
#%%

user_emb = user_embeddings[user]
user_item = generator.generate(user_emb, 20 + test_max_inter)
user_item = np.array(user_item[:,:20]).flatten()
unique_items, unique_counts = np.unique(user_item, return_counts = True)

unique_pop = blen_pop[unique_items]
plt.scatter(unique_pop, unique_counts,alpha=0.7, edgecolors='w', s=100)

plt.figure(figsize=(10, 6))
plt.hist(unique_pop, bins=100, weights=unique_counts, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(blen_pop.mean(), color='red', linestyle='dashed', linewidth=2)
np.quantile(blen_pop,0.5)
plt.xlabel('Item Popularity')
plt.ylabel('Sum of Interaction Counts')
plt.title('Histogram of Interaction Counts weighted by Item Popularity')
plt.grid(True)
plt.show()
plt.boxplot(unique_pop)
blen_pop[user_item].mean()
# %%
users_min, train_pos_min, test_pos_min, num_test_pos_min, items_min, batch_results_min = min_batch
users_min_emb = user_embeddings[users_min]
users_min_item = generator.generate(users_min_emb, 20 + test_max_inter)
users_min_item = filter_history(users_min_item, 20,train_pos_min, device)
users_min_item = users_min_item.cpu()
test_pos_min = test_pos_min.cpu()

users_min_item.shape
test_pos_min.shape

for i in range(users_min_item.shape[0]):
    hit_count = np.isin(users_min_item[i,:], test_pos_min[i,:]).sum()
    print(i,hit_count)
users_min_item[0,:]
hit_count = np.isin(items, test_pos).sum()

users_min_item[117,:]
test_pos_min[117,:]
2555 1306 215

# %%
blend_csr_record = blend_coo_record.tocsr()
test_csr_record = test_coo_record.tocsr()

item_int_emb = item_embeddings[:,:64]
item_con_emb = item_embeddings[:,64:]

user_int_emb = user_embeddings[:,:64]
user_con_emb = user_embeddings[:,64:]

#%%
# 가설 (1) 민감도가 제일 낮은 유저에게 추천한 아이템 인기도 비교
values, indices = torch.topk(user_pop, 1, largest = False)
lowest_pop_user = indices

# 해당 유저의 train에서 상호작용한 아이템
lowest_user_train = blend_csr_record[lowest_pop_user,:].indices
lowest_user_train_pop = blen_pop[lowest_user_train]
# 해당 유저의 test에서 상호작용한 아이템
lowest_user_test = test_csr_record[lowest_pop_user,:].indices
lowest_user_test_pop = blen_pop[lowest_user_test]

# DICE모델로 추천하는 아이템들의 인기도 분포
top_20_indices = np.argsort(combined)[-(20+len(lowest_user_train)):]
top_20_indices = np.setdiff1d(top_20_indices, lowest_user_train)
top_20_pop = blen_pop[top_20_indices]
top_20_pop_value, top_20_pop_indices = torch.topk(top_20_pop, k=20, largest = False)

# train, test, 추천아이템 아이템 분포 비교
plt.figure(figsize = (5,5))
plt.boxplot([lowest_user_train_pop,lowest_user_test_pop,top_20_pop_value], labels = ['Train','Test','Rec_20'], patch_artist = True)
plt.xlabel('Dataset')
plt.ylabel('Item Popularity')
plt.grid(True)
plt.show()

# 해당 유저의 score 분포
top_user_con_emb = user_con_emb[lowest_pop_user]
top_user_int_emb = user_int_emb[lowest_pop_user]

top_item_con_emb = item_con_emb[lowest_user_train]
top_item_int_emb = item_int_emb[lowest_user_train]

int_item = torch.matmul(top_user_con_emb,item_con_emb.T)
con_item = torch.matmul(top_user_int_emb,item_int_emb.T)

int_train_item = torch.matmul(top_user_con_emb, top_item_con_emb.T)
con_train_item = torch.matmul(top_user_int_emb, top_item_int_emb.T)

int_item_np = int_item.cpu().numpy().flatten()
con_item_np = con_item.cpu().numpy().flatten()

combined = (int_item_np + con_item_np)/2
# 시각화
normal_data = np.concatenate((int_item_np, con_item_np))
mean_val = np.mean(normal_data)
std_val = np.std(normal_data)
int_item_np_normal = (int_item_np - mean_val)/std_val
con_item_np_normal = (con_item_np - mean_val)/std_val

plt.figure(figsize=(12, 6))
plt.scatter(range(len(int_item_np_normal)), int_item_np_normal, color='red', label='int_item', s=10)
plt.scatter(range(len(int_item_np_normal)), con_item_np_normal, color='blue', label='con_item', s=10)
for idx in lowest_user_train:
    if idx < len(int_item_np_normal):
        plt.scatter(idx, int_item_np_normal[idx], color='black', edgecolor='white', s=100)
        plt.scatter(idx, con_item_np_normal[idx], color='gray', edgecolor='white', s=100)
# for idx in lowest_user_test:
#     if idx < len(int_item_np_normal):
#         plt.scatter(idx, int_item_np_normal[idx], color='green', edgecolor='white', s=100)
#         plt.scatter(idx, con_item_np_normal[idx], color='yellow', edgecolor='white', s=100)
plt.legend()
plt.title('int_item and con_item scores')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score value')
plt.show()

# 최종 score
combined = (int_item_np + con_item_np)
mean_com = np.mean(combined)
std_com = np.std(combined)
com_item_np_normal = (combined - mean_com) / std_com
plt.figure(figsize = (12,6))
plt.scatter(range(len(combined)), com_item_np_normal, color = 'green', label = 'combined_item', s=10)

for idx in top_20_indices:
    if idx < len(com_item_np_normal):
        plt.scatter(idx, com_item_np_normal[idx], color = 'skyblue',edgecolor = 'white', s=100)
plt.legend()
plt.title('combined scores')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score value')
plt.show()
#%%
# (2) 평소 인기없는 아이템만 사던 사람이 인기있는 아이템도 사는경우
pop_mean = blen_pop.mean()
values, indices = torch.topk(user_pop, 100, largest = False)
low_pop_user = indices
low_pop_user = indices[82]
# 해당 유저의 train에서 상호작용한 아이템
low_user_train = blend_csr_record[low_pop_user,:].indices
low_user_train_pop = blen_pop[low_user_train]
# 해당 유저의 test에서 상호작용한 아이템
low_user_test = test_csr_record[low_pop_user,:].indices
low_user_test_pop = blen_pop[low_user_test]
# 해당 유저의 score 분포
top_user_con_emb = user_con_emb[low_pop_user]
top_user_int_emb = user_int_emb[low_pop_user]

top_item_con_emb = item_con_emb[low_user_train]
top_item_int_emb = item_int_emb[low_user_train]

int_item = torch.matmul(top_user_con_emb,item_con_emb.T)
con_item = torch.matmul(top_user_int_emb,item_int_emb.T)

int_train_item = torch.matmul(top_user_con_emb, top_item_con_emb.T)
con_train_item = torch.matmul(top_user_int_emb, top_item_int_emb.T)

int_item_np = int_item.cpu().numpy().flatten()
con_item_np = con_item.cpu().numpy().flatten()

# DICE모델로 추천하는 아이템들의 인기도 분포
combined = (int_item_np + con_item_np)
top_20_indices = np.argsort(combined)[-(20+len(low_user_train)):]
top_20_indices = np.setdiff1d(top_20_indices, low_user_train)
top_20_pop = blen_pop[top_20_indices]
top_20_pop_value, top_20_pop_indices = torch.topk(top_20_pop, k=20, largest = False)

# train, test, 추천아이템 아이템 분포 비교
plt.figure(figsize = (5,5)) 
plt.boxplot([low_user_train_pop,low_user_test_pop,top_20_pop_value], labels = ['Train','Test','Rec_20'], patch_artist = True)
plt.boxplot([low_user_train_pop,low_user_test_pop], labels = ['Train','Test'], patch_artist = True)

plt.xlabel('Dataset')
plt.ylabel('Item Popularity')
plt.axhline(y=pop_mean, color='red', linestyle='--', linewidth=2)
plt.grid(True)
plt.show()

# 시각화
normal_data = np.concatenate((int_item_np, con_item_np))
mean_val = np.mean(normal_data)
std_val = np.std(normal_data)
int_item_np_normal = (int_item_np - mean_val)/std_val
con_item_np_normal = (con_item_np - mean_val)/std_val

plt.figure(figsize=(12, 6))
plt.scatter(range(len(int_item_np_normal)), int_item_np_normal, color='red', label='int_item', s=10)
plt.scatter(range(len(int_item_np_normal)), con_item_np_normal, color='blue', label='con_item', s=10)
for idx in low_user_train:
    if idx < len(int_item_np_normal):
        plt.scatter(idx, int_item_np_normal[idx], color='black', edgecolor='white', s=100)
        plt.scatter(idx, con_item_np_normal[idx], color='gray', edgecolor='white', s=100)
# for idx in lowest_user_test:
#     if idx < len(int_item_np_normal):
#         plt.scatter(idx, int_item_np_normal[idx], color='green', edgecolor='white', s=100)
#         plt.scatter(idx, con_item_np_normal[idx], color='yellow', edgecolor='white', s=100)
plt.legend()
plt.title('int_item and con_item scores')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score value')
plt.show()
# 최종 score
combined = (int_item_np + con_item_np)
mean_com = np.mean(combined)
std_com = np.std(combined)
com_item_np_normal = (combined - mean_com) / std_com
plt.figure(figsize = (12,6))
plt.scatter(range(len(combined)), com_item_np_normal, color = 'green', label = 'combined_item', s=10)

for idx in top_20_indices:
    if idx < len(com_item_np_normal):
        plt.scatter(idx, com_item_np_normal[idx], color = 'skyblue',edgecolor = 'white', s=100)
plt.legend()
plt.title('combined scores')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score value')
plt.show()
#%%
# (3) 제일 민감한 유저
values, indices = torch.topk(user_pop, 1)
high_pop_user = indices
# 해당 유저의 train에서 상호작용한 아이템
high_user_train = blend_csr_record[high_pop_user,:].indices
high_user_train_pop = blen_pop[high_user_train]
# 해당 유저의 test에서 상호작용한 아이템
high_user_test = test_csr_record[high_pop_user,:].indices
high_user_test_pop = blen_pop[high_user_test]

# 해당 유저의 score 분포
top_user_con_emb = user_con_emb[high_pop_user]
top_user_int_emb = user_int_emb[high_pop_user]

top_item_con_emb = item_con_emb[high_user_train]
top_item_int_emb = item_int_emb[high_user_train]

int_item = torch.matmul(top_user_con_emb,item_con_emb.T)
con_item = torch.matmul(top_user_int_emb,item_int_emb.T)

int_train_item = torch.matmul(top_user_con_emb, top_item_con_emb.T)
con_train_item = torch.matmul(top_user_int_emb, top_item_int_emb.T)

int_item_np = int_item.cpu().numpy().flatten()
con_item_np = con_item.cpu().numpy().flatten()

# DICE모델로 추천하는 아이템들의 인기도 분포
combined = np.array(int_item_np + con_item_np)
top_20_indices = np.argsort(combined)[-(20+len(high_user_train)):]
top_20_indices = np.setdiff1d(top_20_indices, high_user_train)
top_20_pop = blen_pop[top_20_indices]
top_20_pop_value, top_20_pop_indices = torch.topk(top_20_pop, k=20, largest = False)

# train, test, 추천아이템 아이템 분포 비교
plt.figure(figsize = (5,5)) 
plt.boxplot([high_user_train_pop,high_user_test_pop,top_20_pop_value], labels = ['Train','Test','Rec_20'], patch_artist = True)
plt.boxplot([high_user_train_pop,high_user_test_pop], labels = ['Train','Test'], patch_artist = True)

plt.xlabel('Dataset')
plt.ylabel('Item Popularity')
plt.axhline(y=pop_mean, color='red', linestyle='--', linewidth=2)
plt.grid(True)
plt.show()

# 시각화
normal_data = np.concatenate((int_item_np, con_item_np))
mean_val = np.mean(normal_data)
std_val = np.std(normal_data)
int_item_np_normal = (int_item_np - mean_val)/std_val
con_item_np_normal = (con_item_np - mean_val)/std_val

plt.figure(figsize=(12, 6))
plt.scatter(range(len(int_item_np_normal)), int_item_np_normal, color='red', label='int_item', s=10)
plt.scatter(range(len(int_item_np_normal)), con_item_np_normal, color='blue', label='con_item', s=10)
for idx in high_user_train:
    if idx < len(int_item_np_normal):
        plt.scatter(idx, int_item_np_normal[idx], color='black', edgecolor='white', s=100)
        plt.scatter(idx, con_item_np_normal[idx], color='gray', edgecolor='white', s=100)
# for idx in lowest_user_test:
#     if idx < len(int_item_np_normal):
#         plt.scatter(idx, int_item_np_normal[idx], color='green', edgecolor='white', s=100)
#         plt.scatter(idx, con_item_np_normal[idx], color='yellow', edgecolor='white', s=100)
plt.legend()
plt.title('int_item and con_item scores')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score value')
plt.show()
# 최종 score
combined = (int_item_np + con_item_np)
mean_com = np.mean(combined)
std_com = np.std(combined)
com_item_np_normal = (combined - mean_com) / std_com
plt.figure(figsize = (12,6))
plt.scatter(range(len(combined)), com_item_np_normal, color = 'green', label = 'combined_item', s=10)

for idx in top_20_indices:
    if idx < len(com_item_np_normal):
        plt.scatter(idx, com_item_np_normal[idx], color = 'skyblue',edgecolor = 'white', s=100)
plt.legend()
plt.title('combined scores')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score value')
plt.show()
#%%
# (4) 제일 민감한 유저 10명
values, indices = torch.topk(user_pop, 50)
high_pop_user_10 = indices
# 해당 유저의 train에서 상호작용한 아이템
high_user_train = blend_csr_record[high_pop_user_10,:].indices
high_user_train_pop = blen_pop[high_user_train]
# 해당 유저의 test에서 상호작용한 아이템
high_user_test = test_csr_record[high_pop_user,:].indices
high_user_test_pop = blen_pop[high_user_test]

# 해당 유저의 score 분포
top_user_con_emb = user_con_emb[high_pop_user_10]
top_user_int_emb = user_int_emb[high_pop_user_10]

top_item_con_emb = item_con_emb
top_item_int_emb = item_int_emb

int_item = torch.matmul(top_user_con_emb,item_con_emb.T)
con_item = torch.matmul(top_user_int_emb,item_int_emb.T)

int_train_item = torch.matmul(top_user_con_emb, top_item_con_emb.T).cpu()
con_train_item = torch.matmul(top_user_int_emb, top_item_int_emb.T).cpu()


# DICE모델로 추천하는 아이템들의 인기도 분포
combined = np.array(int_train_item + con_train_item)
top_20_indices = np.argsort(combined)[:,-(20):]
top_20_indices.flatten()
top_20_pop = blen_pop[top_20_indices.flatten()]

# train, test, 추천아이템 아이템 분포 비교
plt.figure(figsize = (5,5)) 
plt.boxplot([high_user_train_pop,high_user_test_pop,top_20_pop], labels = ['Train','Test','Rec_20'], patch_artist = True)
plt.boxplot([high_user_train_pop,high_user_test_pop], labels = ['Train','Test'], patch_artist = True)

plt.xlabel('Dataset')
plt.ylabel('Item Popularity')
plt.axhline(y=pop_mean, color='red', linestyle='--', linewidth=2)
plt.grid(True)
plt.show()

# 시각화
normal_data = np.concatenate((int_item_np, con_item_np))
mean_val = np.mean(normal_data)
std_val = np.std(normal_data)
int_item_np_normal = (int_item_np - mean_val)/std_val
con_item_np_normal = (con_item_np - mean_val)/std_val

plt.figure(figsize=(12, 6))
plt.scatter(range(len(int_item_np_normal)), int_item_np_normal, color='red', label='int_item', s=10)
plt.scatter(range(len(int_item_np_normal)), con_item_np_normal, color='blue', label='con_item', s=10)
for idx in high_user_train:
    if idx < len(int_item_np_normal):
        plt.scatter(idx, int_item_np_normal[idx], color='black', edgecolor='white', s=100)
        plt.scatter(idx, con_item_np_normal[idx], color='gray', edgecolor='white', s=100)
# for idx in lowest_user_test:
#     if idx < len(int_item_np_normal):
#         plt.scatter(idx, int_item_np_normal[idx], color='green', edgecolor='white', s=100)
#         plt.scatter(idx, con_item_np_normal[idx], color='yellow', edgecolor='white', s=100)
plt.legend()
plt.title('int_item and con_item scores')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score value')
plt.show()
# 최종 score
combined = (int_item_np + con_item_np)
mean_com = np.mean(combined)
std_com = np.std(combined)
com_item_np_normal = (combined - mean_com) / std_com
plt.figure(figsize = (12,6))
plt.scatter(range(len(combined)), com_item_np_normal, color = 'green', label = 'combined_item', s=10)

for idx in top_20_indices:
    if idx < len(com_item_np_normal):
        plt.scatter(idx, com_item_np_normal[idx], color = 'skyblue',edgecolor = 'white', s=100)
plt.legend()
plt.title('combined scores')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score value')
plt.show()


# 1. 데이터를 분리
embeddings_1 = item_embeddings[:, :64].cpu().numpy()  # 첫 번째 64차원
embeddings_2 = item_embeddings[:, 64:].cpu().numpy()  # 두 번째 64차원

# 2. 두 부분을 이어 붙여서 하나의 배열로 만들기
embeddings_combined = np.vstack([embeddings_1, embeddings_2])

# 3. TSNE 적용
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_combined)

# 4. 시각화: 첫 번째 64차원과 두 번째 64차원을 다른 색으로 구분
plt.figure(figsize=(10, 7))

# 첫 번째 64차원에 해당하는 부분을 파란색으로 시각화
plt.scatter(embeddings_tsne[:4819, 0], embeddings_tsne[:4819, 1], c='blue', label='First 64 dims', s=10, alpha=0.6)

# 두 번째 64차원에 해당하는 부분을 빨간색으로 시각화
plt.scatter(embeddings_tsne[4819:, 0], embeddings_tsne[4819:, 1], c='red', label='Second 64 dims', s=10, alpha=0.6)

# 그래프 제목과 레이블 설정
plt.title('TSNE Visualization of Item Embeddings')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.legend()
plt.show()

#%%
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
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# z 값의 범위 설정
z_values = np.linspace(-4, 4, 1000)

# 평균과 표준편차 설정
mean = 0
std_dev = 1

# 확률 밀도 함수 계산
pdf_values = norm.pdf(z_values, mean, std_dev)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(z_values, pdf_values, label='PDF of Z')
plt.title('Probability Density Function of Z')
plt.xlabel('Z')
plt.ylabel('Density')
plt.grid(True)
plt.legend()
plt.show()
# %%


# 그래프그리기 
best_model_path = "best_model_DICE_COR.pth"

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
