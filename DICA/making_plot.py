import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_density_scores(
    user_emb,
    item_emb,
    user_pop,
    blen_pop,
    epoch,
    adv_weight,
    kl_weight,
    dice_lr,
    disc_lr,
    batch_size,
    category,
    device='cpu'
):
    # 상위 및 하위 유저와 아이템 선택
    user_pop_top_values, user_pop_top_indices = torch.topk(user_pop, 10, largest=True)
    user_pop_low_values, user_pop_low_indices = torch.topk(user_pop, 10, largest=False)
    item_pop_top_values, item_pop_top_indices = torch.topk(blen_pop, 100, largest=True)
    item_pop_low_values, item_pop_low_indices = torch.topk(blen_pop, 100, largest=False)

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

    def plot_all_densities(data_list, labels, filename=f"plot/dis_1_{category}_optim_adv{adv_weight}_kl{kl_weight}_dice_lr{dice_lr}_disc_lr{disc_lr}_batch{batch_size}_epoch{epoch}.png"):
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


    # 합산된 스코어에 대한 밀도 함수 그래프 그리기
    data_list = [top_user_top_item, top_user_low_item, low_user_top_item, low_user_low_item]
    labels = [
        "Density Function of top_user_top_item",
        "Density Function of top_user_low_item",
        "Density Function of low_user_top_item",
        "Density Function of low_user_low_item"
    ]
    plot_all_densities(data_list, labels)
