#%%
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from metric import *
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader
from making_plot import plot_density_scores
from preprocess import *
from model_DICA import DICE, Discriminator, train, valid, test

# %%
'''ARGPARSER'''
parser = argparse.ArgumentParser()
parser.add_argument('--adv-weight', type=int, default=1000, help='Adversarial weight')
parser.add_argument('--kl-weight', type=int, default=100, help='KL divergence weight')
parser.add_argument('--dice-lr', type=float, default=0.01, help='DICE learning rate')
parser.add_argument('--disc-lr', type=float, default=0.001, help='Discriminator learning rate')
parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
parser.add_argument('--batch-size', type=int, default=8192, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--early-stop-patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--weight-decay', type=float, default=5e-8, help='Weight decay for optimizer')
parser.add_argument('--data', type=str, default='movie', help='Set data, select movie or netfilx')

# Handling unrecognized arguments in Jupyter Notebook
try:
    args, unknown = parser.parse_known_args()
except SystemExit:
    # When running in Jupyter, avoid exit errors
    args, unknown = parser.parse_known_args([])
    
# Print parsed arguments for debugging
print("Parsed arguments:", args)
#%%
'''OPTIONS'''
adv_weight: int = args.adv_weight
kl_weight: int = args.kl_weight
dice_lr:float = args.dice_lr
disc_lr:float = args.disc_lr
embedding_size:int = args.embedding_size
batch_size:int = args.batch_size
epochs:int = args.epochs
early_stop_patience = args.early_stop_patience
weight_decay = args.weight_decay
data_select:str = args.data

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        print("MPS backend detected. Seed setting is not required for MPS.")  # MPS는 manual_seed 없음
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 설정 호출
set_seed(42)

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load data
train_root = f'../data/{data_select}/train_coo_record.npz'
valid_root = f'../data/{data_select}/val_coo_record.npz'
test_root = f'../data/{data_select}/test_coo_record.npz'
pop_root = f'../data/{data_select}/popularity.npy'
skew_train_root = f'../data/{data_select}/train_skew_coo_record.npz'
skew_pop_root = f'../data/{data_select}/popularity_skew.npy'

# Load training, validation, and test data
train_coo_record = sp.load_npz(train_root)
train_dok = train_coo_record.todok(copy=True)
train_lil = train_coo_record.tolil(copy=True)

train_skew_coo_record = sp.load_npz(skew_train_root)
train_skew_dok = train_skew_coo_record.todok(copy=True)
train_skew_lil = train_skew_coo_record.tolil(copy=True)

val_coo_record = sp.load_npz(valid_root)
valid_dok = val_coo_record.todok(copy=True)
valid_lil = val_coo_record.tolil(copy=True)

test_coo_record = sp.load_npz(test_root)
test_dok = test_coo_record.todok(copy=True)
test_lil = test_coo_record.tolil(copy=True)

popularity = np.load(pop_root)
skew_pop = np.load(skew_pop_root)

# Calculate blend_item_popularity
blend_user = np.hstack((train_coo_record.row, train_skew_coo_record.row))
blend_item = np.hstack((train_coo_record.col, train_skew_coo_record.col))
blend_value = np.hstack((train_coo_record.data, train_skew_coo_record.data))
blend_coo_record = sp.coo_matrix((blend_value, (blend_user, blend_item)), shape=train_coo_record.shape)

# Generate interaction counts and max values for training and validation
uni_train_lil_record = blend_coo_record.tolil(copy=True)
uni_train_interaction_count = np.array([len(row) for row in uni_train_lil_record.rows], dtype=np.int64)
max_train_interaction = int(max(uni_train_interaction_count)+1)

val_interaction_count = np.array([len(row) for row in valid_lil.rows], dtype=np.int64)
max_val_interaction = int(max(val_interaction_count)+1)

unify_train_pos = np.full(max_train_interaction, -1, dtype=np.int64)
unify_valid_pos = np.full(max_val_interaction, -1, dtype=np.int64)

# Calculate blend item popularity
unique_item, counts = np.unique(blend_item, return_counts=True)
all_item = np.zeros(train_coo_record.shape[1], dtype=int)
all_item[unique_item] = counts
count = all_item
blen_pop = count / np.max(count)
blen_pop = torch.Tensor(blen_pop)

# Calculate user propensity
sum_pop = blend_coo_record.dot(blen_pop)
count_inter = blend_coo_record.tocsr().sum(axis=1).A1
count_inter_safe = np.where(count_inter == 0, 1, count_inter)
user_pop = sum_pop / count_inter_safe
user_pop = torch.Tensor(user_pop)

# Compute groups based on popularity
sorted_indices = torch.argsort(blen_pop, descending=True)
pop_group = sorted_indices[:math.floor(len(sorted_indices) * 0.2)]
unpop_group = sorted_indices[math.floor(len(sorted_indices) * 0.2):]
group_lists = [pop_group.cpu().numpy(), unpop_group.cpu().numpy()]

# Dataloader
train_data = TrainDataset(train_lil, train_dok, train_skew_lil, train_skew_dok, popularity=popularity, skew_pop=skew_pop, device=device)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)

valid_data = CGDataset(test_data_source='val', data_root=valid_root, train_root=train_root, skew_train_root=skew_train_root, device=device)
val_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=False)

test_data = CGDataset(test_data_source='test', data_root=test_root, train_root=train_root, skew_train_root=skew_train_root, device=device)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

# Model and optimizer setup
num_users = train_coo_record.shape[0]
num_items = train_coo_record.shape[1]

model = DICE(num_users=num_users, num_items=num_items, embedding_size= embedding_size,
                dis_pen=0.1, int_weight=0.1, pop_weight=0.1, kl_weight= kl_weight, device=device)
discriminator = Discriminator(input_dim=1, hidden_dim=20, device=device).to(device)
#%%
if data_select == 'movie':
    pretrained_model_path = '../best_model/Movie_best_model_DICE_MSE_SEED_42.pth'
else:
    pretrained_model_path = '../best_model/Netflix_best_model_DICE_MSE_SEED_42.pth'

optimizer = optim.Adam(model.parameters(), lr=dice_lr, weight_decay=weight_decay, betas=(0.5, 0.99), amsgrad=True)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr)

# model.load_state_dict(torch.load(pretrained_model_path))
model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])
optimizer.load_state_dict(torch.load(pretrained_model_path)['optimizer_state_dict'])

# Training and validation loop
best_val_result = float('-inf')
epochs_no_improve = 0
best_model_path = "DICE_best_model.pth"
for epoch in tqdm(range(1, epochs + 1)):
    train_loss, DICE_loss, adv_loss, disc_loss, kl_loss = train(
        model=model,
        discriminator=discriminator,
        train_loader=train_dataloader,
        optimizer=optimizer,
        disc_optimizer=disc_optimizer,
        adv_weight=adv_weight,
        epoch=epoch,
        device=device
    )
    
    val_result = valid(
        model=model,
        val_dataloader=val_dataloader,
        val_max_inter=max_val_interaction,
        top_k=20,
        group_lists=group_lists,
        device=device
    )
    # # 그래프 그리기
    if epoch % 10 ==0:
        item_embeddings = model.get_item_embeddings()
        item_embeddings = torch.Tensor(item_embeddings).to(device)
        user_embeddings = model.get_user_embeddings()
        user_embeddings = torch.Tensor(user_embeddings).to(device)

        # CPU로 데이터 변환
        item_emb = item_embeddings.cpu()
        user_emb = user_embeddings.cpu()

        # 상위 및 하위 유저와 아이템 선택
        plot_density_scores(
            user_emb=user_emb,
            item_emb=item_emb,
            user_pop=user_pop,
            blen_pop=blen_pop,
            epoch=epoch,
            adv_weight=adv_weight,
            kl_weight=kl_weight,
            dice_lr=dice_lr,
            disc_lr=disc_lr,
            batch_size=batch_size,
            category="valid",
            device=device
        )
    
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.5f}, DICE Loss: {DICE_loss:.5f}, Adversarial Loss: {adv_loss:.5f},Discriminator Loss: {disc_loss:.5f}, KL Loss: {kl_loss:.5f}')
    print(f'Epoch: {epoch}, Validation Results: {val_result}')
    
    if val_result['recall'] > best_val_result:
        best_val_result = val_result['recall']
        epochs_no_improve = 0
        torch.save({
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        }, best_model_path)
    else:
        epochs_no_improve += 1
    
    if epoch > 100 and epochs_no_improve >= early_stop_patience:
        print("Early stopping triggered.")
        break

# Load best models and perform testing
model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
discriminator.load_state_dict(torch.load(best_model_path)['discriminator_state_dict'])

test_result_20 = test(
    model=model,
    test_dataloader=test_dataloader,
    test_max_inter=max_val_interaction,
    top_k=20,
    group_lists=group_lists,
    device=device
)
print(f'Test Results @20: {test_result_20}')

test_result_50 = test(
    model=model,
    test_dataloader=test_dataloader,
    test_max_inter=max_val_interaction,
    top_k=50,
    group_lists=group_lists,
    device=device
)
print(f'Test Results @50: {test_result_50}')

item_embeddings = model.get_item_embeddings()
item_embeddings = torch.Tensor(item_embeddings).to(device)
user_embeddings = model.get_user_embeddings()
user_embeddings = torch.Tensor(user_embeddings).to(device)

# CPU로 데이터 변환
item_emb = item_embeddings.cpu()
user_emb = user_embeddings.cpu()

# 상위 및 하위 유저와 아이템 선택
plot_density_scores(
    user_emb=user_emb,
    item_emb=item_emb,
    user_pop=user_pop,
    blen_pop=blen_pop,
    epoch=epoch,
    adv_weight=adv_weight,
    kl_weight=kl_weight,
    dice_lr=dice_lr,
    disc_lr=disc_lr,
    batch_size=batch_size,
    category="test",
    device=device
)
# %%
