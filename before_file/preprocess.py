#%%
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
#%%
#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#%%
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import random
#%%
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
#%%

class Sampler(object):
    def __init__(self, lil_record, dok_record, neg_sample_rate, device):
        self.lil_record = lil_record
        self.record = list(dok_record.keys())
        self.neg_sample_rate = neg_sample_rate
        self.n_user = lil_record.shape[0]
        self.n_item = lil_record.shape[1]
        self.device = device
        
    def sample(self, index, **kwargs):
        raise NotImplementedError
    
    def get_pos_user_item(self, index):
        user = self.record[index][0]
        pos_item = self.record[index][1]
        return user, pos_item

    def generate_negative_samples(self, user, **kwargs):
        user_pos = set(self.lil_record.rows[user])
        all_items = np.arange(self.n_item)
        negative_candidates = np.setdiff1d(all_items, list(user_pos), assume_unique=True)
        negative_samples = np.random.choice(negative_candidates, self.neg_sample_rate, replace=False)
        return negative_samples

class DICESampler(Sampler):
    def __init__(self, lil_record, dok_record, neg_sample_rate, popularity, device, margin=10, pool=10):
        super(DICESampler, self).__init__(lil_record, dok_record, neg_sample_rate, device)
        self.popularity = torch.tensor(popularity, device=device)
        self.margin = margin
        self.pool = pool
        
    def adapt(self, epoch, decay):
        self.margin = self.margin * decay
        
    def generate_negative_samples(self, user, pos_item):
        user_pos = set(self.lil_record.rows[user])
        item_pos_pop = self.popularity[pos_item].item()
        popularity_array = self.popularity.cpu().numpy()
        
        pop_items = np.where(popularity_array > item_pos_pop + self.margin)[0]
        unpop_items = np.where(popularity_array < item_pos_pop / 2)[0]
        
        pop_items = np.setdiff1d(pop_items, list(user_pos), assume_unique=True)
        unpop_items = np.setdiff1d(unpop_items, list(user_pos), assume_unique=True)
        
        if len(pop_items) < self.pool:
            chosen_items = np.random.choice(unpop_items, self.neg_sample_rate, replace=True)
            mask_type = np.zeros(self.neg_sample_rate, dtype=np.bool_)
        elif len(unpop_items) < self.pool:
            chosen_items = np.random.choice(pop_items, self.neg_sample_rate, replace=True)
            mask_type = np.ones(self.neg_sample_rate, dtype=np.bool_)
        else:
            half_rate = self.neg_sample_rate // 2
            chosen_items_pop = np.random.choice(pop_items, half_rate, replace=True)
            chosen_items_unpop = np.random.choice(unpop_items, self.neg_sample_rate - half_rate, replace=True)
            chosen_items = np.concatenate((chosen_items_pop, chosen_items_unpop))
            mask_type = np.concatenate((np.ones(half_rate, dtype=np.bool_), np.zeros(self.neg_sample_rate - half_rate, dtype=np.bool_)))
        
        return torch.tensor(chosen_items, device=self.device), torch.tensor(mask_type, device=self.device)
    
    def sample(self, index):
        user, pos_item = self.get_pos_user_item(index)
        users = torch.tensor(np.full(self.neg_sample_rate, user, dtype=np.int64)).to(self.device)
        items_pos = torch.tensor(np.full(self.neg_sample_rate, pos_item, dtype=np.int64)).to(self.device)
        items_neg, mask_type = self.generate_negative_samples(user, pos_item=pos_item)
        
        return users, items_pos, items_neg, mask_type
    
class TrainDataset(Dataset):
    def __init__(self, train_lil, train_dok, skew_train_lil, skew_train_dok, popularity, skew_pop, device):
        self.train_lil = train_lil
        self.train_dok = train_dok
        self.skew_train_lil = skew_train_lil
        self.skew_train_dok = skew_train_dok
        self.popularity = popularity
        self.skew_pop = skew_pop
        self.device = device
        self.sampler = DICESampler(self.train_lil, self.train_dok, neg_sample_rate=4, popularity=self.popularity, device=self.device)
        self.skew_sampler = DICESampler(self.skew_train_lil, self.skew_train_dok, neg_sample_rate=4, popularity=self.skew_pop, device=self.device)
        
    def __len__(self):
        return len(self.sampler.record) + len(self.skew_sampler.record)
    
    def __getitem__(self, index):
        if index < len(self.sampler.record):
            users, items_p, items_n, mask = self.sampler.sample(index)
        else:
            users, items_p, items_n, mask = self.skew_sampler.sample(index - len(self.sampler.record))
        return users, items_p, items_n, mask
    
    def adapt(self, epoch, decay):
        self.sampler.adapt(epoch, decay)
        self.skew_sampler.adapt(epoch, decay)


        
class FaissInnerProductMaximumSearchGenerator:
    def __init__(self, item_embeddings, device):
        self.items = item_embeddings
        self.embedding_size = self.items.shape[1]
        self.device = device
    def generate(self, users, k):
        users = users.to(self.device)
        scores = torch.matmul(users, self.items.T)
        indices = torch.argsort(scores, axis = 1, descending = True)[:,:k]
        return indices.cpu().numpy()
    
    def generate_with_distance(self, users, k):
        users = users.to(self.device)
        scores = torch.matmul(users, self.items.T)
        indices = torch.argsort(scores, axis = 1, descending = True)[:,:k]
        distances = torch.sort(scores, axis = 1, descending = True)[0][:,:k]
        return distances.cpu().numpy(), indices.cpu().numpy()
    
def filter_history(items, train_pos, device):
    items = torch.tensor(items)
    train_pos = train_pos.to('cpu')
    return torch.stack([items[i][~torch.isin(items[i], train_pos[i])][:50] for i in range(len(items))], dim = 0).to(device)

class CGDataset(Dataset):
    def __init__(self, test_data_source, data_root, train_root, skew_train_root, device):
        self.device = device
        self.test_data_source = test_data_source
        if self.test_data_source == 'val':
            self.valid_root = data_root
            coo_record = sp.load_npz(self.valid_root)
        elif self.test_data_source == 'test':
            self.test_root = data_root
            coo_record = sp.load_npz(self.test_root)
        self.lil_record = coo_record.tolil()
        self.train_root = train_root
        self.skew_train_root = skew_train_root
        train_coo_record = sp.load_npz(self.train_root)
        train_skew_coo_record = sp.load_npz(self.skew_train_root)

        blend_user = np.hstack((train_coo_record.row, train_skew_coo_record.row))
        blend_item = np.hstack((train_coo_record.col, train_skew_coo_record.col))
        blend_value = np.hstack((train_coo_record.data, train_skew_coo_record.data))
        blend_coo_record = sp.coo_matrix((blend_value, (blend_user, blend_item)))
        
        self.train_lil_record = blend_coo_record.tolil(copy = True)
        train_interaction_count = np.array([len(row) for row in self.train_lil_record.rows], dtype = np.int64)
        self.max_train_interaction = int(max(train_interaction_count))
        
        test_interaction_count = np.array([len(row) for row in self.lil_record.rows], dtype = np.int64)
        self.max_test_interaction = int(max(test_interaction_count))
        
    def __len__(self):
        return len(self.lil_record.rows)
    
    def __getitem__(self, index):
        unify_train_pos = np.full(self.max_train_interaction, -1, dtype = np.int64)
        unify_test_pos = np.full(self.max_test_interaction, -1, dtype = np.int64)
        train_pos = self.train_lil_record.rows[index]
        test_pos = self.lil_record.rows[index]
        
        unify_train_pos[:len(train_pos)] = train_pos
        unify_test_pos[:len(test_pos)] = test_pos
        
        return torch.LongTensor([index]).to(self.device), torch.LongTensor(unify_train_pos).to(self.device), torch.LongTensor(unify_test_pos).to(self.device), torch.LongTensor([len(test_pos)]).to(self.device)
# # %%






#%%