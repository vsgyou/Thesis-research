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
import random
from preprocess_10 import *
from model_user_IPS import DICE, train, valid, test
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
#%%
# seed
def set_seed(seed):
    import torch
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS 백엔드는 manual_seed 속성이 없으므로 이 부분을 생략합니다.
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
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

#%%
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


lr = 0.006
weight_decay = 5e-8
model = DICE(num_users= num_users, num_items = num_items, embedding_size = 64, blen_pop = blen_pop, user_pop = user_pop, dis_pen =0.01, int_weight = 0.1, pop_weight = 0.1, gamma = 0.02,device = device)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay, betas = (0.5, 0.99), amsgrad = True)
epochs = 200
total_loss = 0.0

val_max_inter = valid_data.max_train_interaction
test_max_inter = test_data.max_train_interaction

# early stopping
best_val_result = float('-inf')
epochs_no_improve = 0
early_stop_patience = 5
best_model_path = "MSE_1.pth"
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












# %%
# test_data에서 인기에 민감도와 추천해준 아이템들의 인기도의 상관관계
top_k = 20
best_model_path = 'COR_0_1.pth'

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
# 민감도가 높은 10명의 유저들의 추천된 아이템들의 인기도를 보자(높은 유저의 아이템들은 동일해, 근데 낮은 10명은 달라졌다)
#%%
a = blend_coo_record.tocsr()
print(a[36802,:])
b = np.array([908,915,3271,4697])
blen_pop[b]
c=test_coo_record.tocsr()
print(c[36802,:])
d = np.array([242,488,646,769,895,925,929,930,936,940,948,988,1030,1031,4094,4634,4690])
(blen_pop[d]).mean()
plt.plot(range(len(blen_pop)), np.array(blen_pop))
#%%

values, indices = torch.topk(user_pop, 2, largest = False)
top_pop_user = torch.reshape(indices[-1],(1,1))
top_pop_user = indices
top_user_emb = user_embeddings[top_pop_user]
top_user_item = generator.generate(top_user_emb, 20 + test_max_inter)
top_user_item = np.array(top_user_item[:,:20]).flatten()
UDIPS_blen_pop = blen_pop[top_user_item].numpy()
np.mean(UDIPS_blen_pop)
torch.mean(blen_pop[top_user_item])
# 그래프
plt.subplot(1, 2, 1)
sns.histplot(blen_pop[top_user_item].numpy(), bins=50, kde=False, color='blue')
plt.title('Histogram of UDIPS Popularity')


# conformity만으로 추천
item_int_emb = item_embeddings[:,:64]
item_con_emb = item_embeddings[:,64:]

user_int_emb = user_embeddings[:,:64]
user_con_emb = user_embeddings[:,64:]

con_generator = FaissInnerProductMaximumSearchGenerator(item_con_emb, device = device)
top_user_con_emb = user_con_emb[top_pop_user]
top_user_item_con = con_generator.generate(top_user_con_emb, 20 + test_max_inter)
top_user_item_con = np.array(top_user_item_con[:,:20]).flatten()
UDIPS_blen_pop_con = blen_pop[top_user_item_con].numpy()
np.mean(UDIPS_blen_pop_con)

plt.subplot(1, 2, 1)
sns.histplot(UDIPS_blen_pop_con, bins=50, kde=False, color='blue')
plt.title('Histogram of UDIPS Popularity')
# interest만으로 추천
int_generator = FaissInnerProductMaximumSearchGenerator(item_int_emb, device = device)
top_user_int_emb = user_int_emb[top_pop_user]
top_user_item_int = int_generator.generate(top_user_int_emb, 20 + test_max_inter)
top_user_item_int = np.array(top_user_item_int[:,:20]).flatten()
UDIPS_blen_pop_int = blen_pop[top_user_item_int].numpy()
np.mean(UDIPS_blen_pop_int)

plt.subplot(1, 2, 1)
sns.histplot(UDIPS_blen_pop_int, bins=50, kde=False, color='blue')
plt.title('Histogram of UDIPS Popularity')
#%%
# 합쳐진 추천
top_user_item
rec_total = np.sort(top_user_item)
# con 추천
top_user_item_con
rec_con = np.sort(top_user_item_con)
# int 추천
top_user_item_int
rec_int = np.sort(top_user_item_int)
# con_emb + int_emb = total_emb
con_emb = torch.matmul(top_user_con_emb, item_con_emb.T)
int_emb = torch.matmul(top_user_int_emb, item_int_emb.T)
total = con_emb + int_emb
torch.matmul(top_user_emb,item_embeddings.T)

#%%
int_item = torch.matmul(top_user_con_emb,item_con_emb.T)
con_item = torch.matmul(top_user_int_emb,item_int_emb.T)

# numpy array로 변환
int_item_np = int_item.cpu().numpy().flatten()
con_item_np = con_item.cpu().numpy().flatten()

# 두 임베딩을 하나의 배열로 합침
combined = (int_item_np + con_item_np)
#int에서 상위 20개 값 식별
np.argsort(int_item_np)[-20:]
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
#        plt.scatter(idx, combined[idx], edgecolor='green', facecolor='none', linewidth=2, s=100)
        plt.scatter(idx, int_item_np[idx], edgecolor = 'pink',facecolor='none',linewidth = 2, s=100)
        plt.scatter(idx, con_item_np[idx], edgecolor = 'skyblue',facecolor='none',linewidth = 2, s=100)
    else:
#        plt.scatter(idx, combined[idx], edgecolor='green', facecolor='none', linewidth=2, s=100)
        plt.scatter(idx, int_item_np[idx], edgecolor = 'pink',facecolor='none',linewidth = 2, s=100)
        plt.scatter(idx, con_item_np[idx], edgecolor = 'skyblue',facecolor='none',linewidth = 2, s=100)

plt.legend()
plt.title('int_item and con_item Values with Top 20 Points Highlighted')
plt.xlabel('Item index(0~4819)')
plt.ylabel('Score Value')
plt.show()

plt.figure(figsize = (12,6))
plt.scatter(range(len(int_item_np)),combined, color = 'green',label = 'combined',s=10)


for idx in top_20_indices:
    if idx < 4819:
        plt.scatter(idx, combined[idx], edgecolor='black', facecolor='none', linewidth=2, s=100)
    
#%%
plt.scatter(range(len(top_20_indices)), int_item_np[top_20_indices],color = 'red')
plt.scatter(range(len(top_20_indices)), con_item_np[top_20_indices],color = 'blue')
plt.scatter(range(len(top_20_indices)), combined[top_20_indices],color = 'green')
#%%


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

#%%

















#%%

# 데이터 파악
blend_item = np.hstack((train_coo_record.col, train_skew_coo_record.col))

train_item = np.hstack(train_coo_record.col)
train_user = np.hstack(train_coo_record.row)
unique_train_item, train_counts = np.unique(train_item, return_counts = True)
unique_user, counts = np.unique(train_user, return_counts = True)


all_item = np.zeros(train_coo_record.shape[1], dtype = int)
all_item[unique_item] = counts

total_data = sp.load_npz('/Users/jeongjiun/Documents/GitHub/my_DICE/coo_record.npz')

total_item = np.stack(total_data.col)
total_user = np.stack(total_data.row)
unique_total_item, item_counts = np.unique(total_item, return_counts = True)
unique_total_user, user_counts = np.unique(total_user, return_counts = True)

train_skew_item = np.stack(train_skew_coo_record.col)
train_skew_user = np.stack(train_skew_coo_record.row)
unique_train_skew_item, item_train_skew_counts = np.unique(train_skew_item, return_counts = True)
unique_train_skew_user, user_train_skew_counts = np.unique(train_skew_user, return_counts = True)


val_item = np.stack(val_coo_record.col)
val_user = np.stack(val_coo_record.row)
unique_val_item, item_val_counts = np.unique(val_item, return_counts = True)
unique_val_user, user_val_counts = np.unique(val_user, return_counts = True)

test_item = np.stack(test_coo_record.col)
test_user = np.stack(test_coo_record.row)
unique_test_item, item_test_counts = np.unique(test_item, return_counts = True)
unique_test_user, user_test_counts = np.unique(test_user, return_counts = True)

inter_item = np.hstack((train_skew_coo_record.col, val_coo_record.col, test_coo_record.col))
inter_user = np.hstack((train_skew_coo_record.row, val_coo_record.row, test_coo_record.row))

unique_inter_item, inter_item_counts = np.unique(inter_item, return_counts = True)
unique_inter_user, inter_user_counts = np.unique(inter_user, return_counts = True)

inter_prop = inter_item_counts / (len(train_skew_coo_record.row)+len(val_coo_record.row)+len(test_coo_record.row))
train_prop = train_counts / len(train_coo_record.row)

x = np.arange(len(inter_item_counts))
plt.bar(x,inter_prop)
x2 = np.arange(len(train_counts))
plt.bar(x2, train_prop)

x3 = np.arange(len(inter_user_counts))
plt.bar(x3,inter_user_counts)
plt.ylim(0,200)

x_val = np.arange(len(item_val_counts))
plt.bar(x_val, item_val_counts)

x_test = np.arange(len(item_test_counts))
plt.bar(x_test, item_test_counts)

#%%
total_item_inter = np.array(total_data.sum(axis = 0)).flatten()
train_item_inter = np.array(train_coo_record.sum(axis =0)).flatten()
item_index = np.arange(train_coo_record.shape[1])
plt.figure(figsize=(14, 7))
plt.bar(item_index, total_item_inter, color='blue',width = 1.0)
plt.bar(item_index, train_item_inter, color='red', width = 1.0)
plt.xlabel('Item Index')
plt.ylabel('Number of Interactions')
plt.title('Number of Interactions per Item')
plt.show()
# %%
total_csr = total_data.tocsr()
total_csr[:,6]
train_csr = train_coo_record.tocsr()
train_csr[:,6]


plt.figure(figsize=(14, 7))
plt.bar(item_index, total_item_inter, color='blue',width = 1.0)
plt.bar(item_index, inter_item_counts, color='red',width = 1.0)
plt.ylim(0,6000)
plt.xlabel('Item Index')
plt.ylabel('Number of Interactions')
plt.title('Number of Interactions per Item')
plt.show()

#%%
# 민감도 하위 0.01%유저가 상호작용한 아이템인기도 분포
blend_coo_record
user_index, user_inter = np.unique(blend_user, return_counts = True)
np.quantile(user_pop, 0.1)
thirty = np.array(np.where(user_pop < np.quantile(user_pop, 0.001))[0])
len(thirty)
blend_csr_record = blend_coo_record.tocsr()
thirty_item = blend_csr_record[thirty].indices
thirty_item_index, thirty_item_counts = np.unique(thirty_item, return_counts = True)
thirty_item_pop = blen_pop[thirty_item_index]

plt.scatter(thirty_item_pop, thirty_item_counts, alpha=0.7, edgecolors='w', s=100)
sns.histplot(thirty_item_counts)
#%%
# 중위값 0.00119으로 0.017인평균은 큰 인기도임
plt.figure(figsize=(10, 6))
plt.hist(thirty_item_pop, bins=100, weights=thirty_item_counts, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(blen_pop.mean(), color='red', linestyle='dashed', linewidth=2)
np.quantile(blen_pop,0.5)
plt.xlabel('Item Popularity')
plt.ylabel('Sum of Interaction Counts')
plt.title('Histogram of Interaction Counts weighted by Item Popularity')
plt.grid(True)
plt.show()
# %%
blen_pop[thirty_item].mean()
#%%
# 민감도 하위 10명
user = np.array([36802, 34736, 36814, 37959, 37938, 36322, 36189, 37331, 37903, 36294])
user = np.array([21975, 22039, 22371, 24732, 31263, 34736, 34878, 36000, 36026,
       36054, 36189, 36190, 36255, 36262, 36264, 36294, 36322, 36361,
       36362, 36802, 36814, 36843, 37302, 37331, 37774, 37806, 37833,
       37841, 37868, 37903, 37920, 37928, 37937, 37938, 37939, 37945,
       37947, 37959])
user = np.array([36294])
user = np.array([2697])
user_pop[user]
# 해당 10명에 대해 테스트데이터에서의 데이터확인
blen_indices, blen_counts = np.unique(blend_user, return_counts = True)
d = blen_indices[user]
normal_item = blend_csr_record[d,].indices
plt.hist(blen_pop[normal_item],bins = 100, alpha = 0.7, color = 'blue', edgecolor = 'black')


test_indices, test_counts  = np.unique(test_user, return_counts = True)
b = np.isin(test_indices,user)
c = test_indices[b]

test_csr_record = test_coo_record.tocsr()
real_item = test_csr_record[c,].indices
plt.hist(blen_pop[real_item],bins = 100, alpha = 0.7, color = 'blue', edgecolor = 'black')
blen_pop[real_item].mean()

#%%
# train과 test에서 유저가 산 아이템의 인기도 비교
plt.figure(figsize = (5,5))
plt.boxplot([blen_pop[normal_item],blen_pop[real_item]], labels = ['Train','Test'], patch_artist = True)
plt.xlabel('Dataset')
plt.ylabel('Item Popularity')
plt.grid(True)
plt.show()

#%%
my_pop = popularity
my_pop[popularity<17]=1
my_pop[(popularity>=17) & (popularity<71)]=2
my_pop[popularity >= 71] = 3


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

plt.boxplot(unique_pop,patch_artist = True)

blen_pop[user_item].mean()

# 1. 데이터를 분리
embeddings_1 = item_embeddings[:, :64].cpu().numpy()  # 첫 번째 64차원
embeddings_2 = item_embeddings[:, 64:].cpu().numpy()  # 두 번째 64차원

# 2. 두 부분을 이어 붙여서 하나의 배열로 만들기
embeddings_combined = np.vstack([embeddings_1, embeddings_2])

# 3. TSNE 적용
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_combined)
data=embeddings_tsne
# 4. 시각화: 첫 번째 64차원과 두 번째 64차원을 다른 색으로 구분
plt.figure(figsize=(10, 7))

# 첫 번째 64차원에 해당하는 부분을 파란색으로 시각화
plt.scatter(data[:4819,0], data[:4819,1],40,my_pop,'x',label = 'DICE-int')

# 두 번째 64차원에 해당하는 부분을 빨간색으로 시각화
plt.scatter(data[4819:,0], data[4819:,1],40,my_pop+3,'.',label ='DICE-con')

# 그래프 제목과 레이블 설정
plt.title('TSNE Visualization of Item Embeddings')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.legend()
plt.show()