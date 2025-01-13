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

from preprocess import *
from model import DICE, train, valid, test
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
blend_coo_record = sp.coo_matrix((blend_value, (blend_user, blend_item)))
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
blen_pop = count / np.sum(count)
nor_blen_pop = (blen_pop - min(blen_pop)) / (max(blen_pop) - min(blen_pop)) + 5e-8
nor_blen_pop = torch.tensor(nor_blen_pop)

# calculate user propensity

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
model = DICE(num_users= num_users, num_items = num_items, embedding_size = 64, blen_pop = nor_blen_pop, dis_pen =0.01, int_weight = 0.1, pop_weight = 0.1, gamma = 0.02, device = device)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay, betas = (0.5, 0.99), amsgrad = True)
epochs = 500
total_loss = 0.0

val_max_inter = valid_data.max_train_interaction
test_max_inter = test_data.max_train_interaction

best_val_result = float('-inf')
epochs_no_improve = 0
early_stop_patience = 5
best_model_path = "best_model_MY.pth"
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
total_params = sum(p.numel() for p in model.parameters())

blen_pop