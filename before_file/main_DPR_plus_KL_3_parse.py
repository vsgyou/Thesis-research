#%%
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import random
import math
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from metric_DPR import *
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader

from preprocess_10 import *
from model_DPR_plus_KL_3_parse import DICE, Discriminator, train, valid, test

#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument('--adv-weight', type = int, default = 70)
parser.add_argument('--kl-weight',type = int, default = 30)
parser.add_argument('--dice-lr',type = float, default = 0.05)
parser.add_argument('--disc-lr',type = float, default = 0.001)
parser.add_argument('--embedding_size', type = int, default = 64)
parser.add_argument('--batch-size', type = int, default = 8192)
parser.add_argument('--epochs',type = int, default=200)
parser.add_argument('--early-stop-patience',type = int, default = 30)
parser.add_argument('--weight-decay',type = float, default = 5e-8)
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


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

best_model_path = f"parse_best_model/optim_pretrain_best_model_adv{adv_weight}_kl{kl_weight}_dice_lr{dice_lr}_disc_lr{disc_lr}_batch{batch_size}.pth"
best_discriminator_model_path = f"parse_best_model/optim_pretrain_best_discriminator_adv{adv_weight}_kl{kl_weight}_dice_lr{dice_lr}_disc_lr{disc_lr}_batch{batch_size}.pth"

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # if torch.backends.mps.is_available():
        # torch.backends.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

wandb.init(
    project='DICA_experiments',
    config={
        "adv_weight": adv_weight,
        "kl_weight": kl_weight,
        "dice_lr": dice_lr,
        "disc_lr": disc_lr,
        "embedding_size": embedding_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
    }
)
# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load data
train_root = 'data/train_coo_record.npz'
valid_root = 'data/val_coo_record.npz'
test_root = 'data/test_coo_record.npz'
pop_root = 'data/popularity.npy'
skew_train_root = 'data/train_skew_coo_record.npz'
skew_pop_root = 'data/popularity_skew.npy'

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
max_train_interaction = int(max(uni_train_interaction_count))

val_interaction_count = np.array([len(row) for row in valid_lil.rows], dtype=np.int64)
max_val_interaction = int(max(val_interaction_count))

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
                dis_pen=0.01, int_weight=0.1, pop_weight=0.1, kl_weight= kl_weight, device=device)
discriminator = Discriminator(input_dim=1, hidden_dim=20, device=device).to(device)

optimizer = optim.Adam(model.parameters(), lr=dice_lr, weight_decay=weight_decay, betas=(0.5, 0.99), amsgrad=True)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr)

# Training and validation loop
best_val_result = float('-inf')
epochs_no_improve = 0
# best_model_path = "parse_best_model/best_model.pth"
# best_discriminator_model_path = "parse_best_model/best_discriminator.pth"

for epoch in tqdm(range(1, epochs + 1)):
    train_loss, DICE_loss, adv_loss, disc_loss, kl_loss = train(
        model=model,
        discriminator=discriminator,
        train_loader=train_dataloader,
        optimizer=optimizer,
        disc_optimizer=disc_optimizer,
        adv_weight=adv_weight,
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
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "DICE_loss": DICE_loss,
        "adv_loss": adv_loss,
        "disc_loss": disc_loss,
        "kl_loss": kl_loss,
        "val_recall": val_result['recall'],
        "val_ndcg": val_result['ndcg'],
        "val_hit_ratio": val_result['hit_ratio'],
        "val_rsp": val_result.get('rsp', 0),
        "val_reo": val_result.get('reo', 0)
    })
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.5f}, DICE Loss: {DICE_loss:.5f}, Adversarial Loss: {adv_loss:.5f},Discriminator Loss: {disc_loss:.5f}, KL Loss: {kl_loss:.5f}')
    print(f'Epoch: {epoch}, Validation Results: {val_result}')
    
    if val_result['recall'] > best_val_result:
        best_val_result = val_result['recall']
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        torch.save(discriminator.state_dict(), best_discriminator_model_path)
    else:
        epochs_no_improve += 1
    
    if epoch > 50 and epochs_no_improve >= early_stop_patience:
        print("Early stopping triggered.")
        break

# Load best models and perform testing
model.load_state_dict(torch.load(best_model_path))
discriminator.load_state_dict(torch.load(best_discriminator_model_path))

test_result_20 = test(
    model=model,
    test_dataloader=test_dataloader,
    test_max_inter=max_val_interaction,
    top_k=20,
    group_lists=group_lists,
    device=device
)
print(f'Test Results @20: {test_result_20}')
wandb.log({
    "Test Recall @20": test_result_20['recall'],
    "Test NDCG @20": test_result_20['ndcg'],
    "Test Hit Ratio @20": test_result_20['hit_ratio'],
    "Test RSP @20": test_result_20.get('rsp', 0),
    "Test REO @20": test_result_20.get('reo', 0)
})
test_result_50 = test(
    model=model,
    test_dataloader=test_dataloader,
    test_max_inter=max_val_interaction,
    top_k=50,
    group_lists=group_lists,
    device=device
)
print(f'Test Results @50: {test_result_50}')

wandb.log({
    "Test Recall @50": test_result_50['recall'],
    "Test NDCG @50": test_result_50['ndcg'],
    "Test Hit Ratio @50": test_result_50['hit_ratio'],
    "Test RSP @50": test_result_50.get('rsp', 0),
    "Test REO @50": test_result_50.get('reo', 0)
})

best_model_path = 'pretrain_DICE_KL_hap_no_opti_best_model_adv1000_kl100_dice_lr0.01_disc_lr0.001_batch8192_optimizer_conti.pth'
# best_model_path = "best_model_DICE_MSE_MSE_42_last_optimizer.pth"
model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
optimizer.load_state_dict(torch.load(best_model_path)['optimizer_state_dict'])
item_embeddings = model.get_item_embeddings()
item_embeddings = torch.Tensor(item_embeddings).to(device)
user_embeddings = model.get_user_embeddings()
user_embeddings = torch.Tensor(user_embeddings).to(device)

# CPU로 데이터 변환
item_emb = item_embeddings.cpu()
user_emb = user_embeddings.cpu()

# 상위 및 하위 유저와 아이템 선택
# user_pop_top_values, user_pop_top_indices = torch.topk(user_pop, 10, largest=True)
# user_pop_low_values, user_pop_low_indices = torch.topk(user_pop, 10, largest=False)
# item_pop_top_values, item_pop_top_indices = torch.topk(blen_pop, 1000, largest=True)
# item_pop_low_values, item_pop_low_indices = torch.topk(blen_pop, 1000, largest=False)
# 전체 유저 전체 아이템
int_score = torch.mm(user_embeddings[:,:64],item_embeddings[:,:64].T).cpu().numpy()
con_score = torch.mm(user_embeddings[:,64:],item_embeddings[:,64:].T).cpu().numpy()

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
data_list = [int_score, con_score]
labels = ["Density Function of total_int_score", "Density Function of total_pop_score"]
plot_all_densities(data_list, labels)
#%%
best_model_path = 'pretrain_DICE_KL_hap_no_opti_best_model_adv1000_kl100_dice_lr0.01_disc_lr0.001_batch8192_optimizer_conti.pth'
model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
optimizer.load_state_dict(torch.load(best_model_path)['optimizer_state_dict'])

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

data_list = [normal_total_int_score, normal_total_pop_score]
labels = ["Density Function of total_int_score", "Density Function of total_pop_score"]
plot_all_densities(data_list, labels)




#%%

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

# 밀도 함수 그리기 함수 정의 및 wandb 업로드
def plot_all_densities(data_list, labels, filename="density_plot.png"):
    plt.figure(figsize=(8, 6))
    for data, label in zip(data_list, labels):
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 1000)
        y_vals = kde(x_vals)
        plt.plot(x_vals, y_vals, label=label)
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title("Density Functions")
    plt.legend()
    
    # Save plot as image
    plt.savefig(filename)
    plt.close()
    
    # Upload image to wandb
    wandb.log({"density_plot": wandb.Image(filename)})

# 합산된 스코어에 대한 밀도 함수 그래프 그리기
data_list = [top_user_top_item, top_user_low_item, low_user_top_item, low_user_low_item]
labels = [
    "Density Function of top_user_top_item",
    "Density Function of top_user_low_item",
    "Density Function of low_user_top_item",
    "Density Function of low_user_low_item"
]

# Call the function to plot and upload to wandb
plot_all_densities(data_list, labels)

wandb.finish()
# %%
