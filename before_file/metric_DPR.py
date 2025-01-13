# import numpy as np
# import torch
# import math

# class Judger(object):
#     def __init__(self, topk, blen_pop):
#         self.topk = topk
#         self.blen_pop = blen_pop  # RSP, REO 계산에 필요한 인기도 정보 추가
    
#     def judge(self, items, test_pos, num_test_pos, train_pos):
#         results = {}
#         items = items.cpu()
#         metrics = ['recall', 'hit_ratio', 'ndcg', 'rsp', 'reo']  # RSP와 REO도 포함

#         for metric in metrics:
#             f = Metrics.get_metrics(metric)
#             if metric in ['rsp', 'reo']:
#                 # RSP, REO도 sum으로 각 사용자의 값을 더함
#                 results[metric] = sum([f(items[i], blen_pop=self.blen_pop, train_pos=train_pos[i], test_pos=test_pos[i], num_groups=2) for i in range(len(items))])
#             else:
#                 # 기존 방식: 각 사용자에 대해 메트릭을 계산한 값을 sum으로 더함
#                 results[metric] = sum([f(items[i], test_pos=test_pos[i], num_test_pos=num_test_pos[i].item()) if num_test_pos[i] > 0 else 0 for i in range(len(items))])
        
#         # 유효한 사용자 수 계산
#         valid_num_users = sum([1 if t > 0 else 0 for t in num_test_pos])
        
#         return results, valid_num_users

    
    
# class Metrics(object):
#     @staticmethod
#     def get_metrics(metric):
#         metrics_map = {
#             'recall': Metrics.recall,
#             'hit_ratio': Metrics.hr,
#             'ndcg': Metrics.ndcg,
#             'rsp': Metrics.rsp,  # RSP 메트릭
#             'reo': Metrics.reo   # REO 메트릭
#         }
#         return metrics_map[metric]

#     @staticmethod
#     def recall(items, **kwargs):
#         test_pos = kwargs['test_pos']
#         num_test_pos = kwargs['num_test_pos']
#         hit_count = np.isin(items, test_pos).sum()
#         return hit_count / num_test_pos

#     @staticmethod
#     def hr(items, **kwargs):
#         test_pos = kwargs['test_pos']
#         hit_count = np.isin(items, test_pos).sum()
#         return 1.0 if hit_count > 0 else 0

#     @staticmethod
#     def ndcg(items, **kwargs):
#         test_pos = kwargs['test_pos']
#         num_test_pos = kwargs['num_test_pos']
#         index = np.arange(len(items))
#         k = min(len(items), num_test_pos)
#         idcg = (1 / np.log(2 + np.arange(k))).sum()
#         dcg = (1 / np.log(2 + index[np.isin(items, test_pos)])).sum()
#         return dcg / idcg

#     @staticmethod
#     def rsp(items, **kwargs):
#         """
#         RSP 계산 함수.
#         blen_pop : 인기도를 나타내는 텐서
#         items : 추천된 아이템 목록
#         train_pos : 훈련 데이터에서 선호 아이템 목록
#         num_groups : 그룹 개수 (기본값: 2)
#         """
#         blen_pop = kwargs['blen_pop']
#         train_pos = kwargs['train_pos']
#         num_groups = kwargs.get('num_groups', 2)
        
#         sorted_indices = torch.argsort(blen_pop, descending=True)
#         pop_group = sorted_indices[:math.floor(len(sorted_indices) * 0.2)]
#         unpop_group = sorted_indices[math.floor(len(sorted_indices) * 0.2):]
#         group_lists = [pop_group.cpu().numpy(), unpop_group.cpu().numpy()]

#         sum_rec = np.zeros(num_groups)
#         sum_pos = np.zeros(num_groups)
        
#         for j, group in enumerate(group_lists):
#             sum_rec[j] = np.sum(np.isin(items, group))
#             sum_pos[j] = np.sum(np.isin(np.setdiff1d(sorted_inidces.cpu().numpy(), train_pos.cpu().numpy()),group))
            
#         # for u in range(len(items)):
#         #     for j, group in enumerate(group_lists):
#         #         sum_rec[j] += np.sum(np.isin(items[u], group))
#         #         sum_pos[j] += np.sum(np.isin(np.setdiff1d(sorted_indices.cpu().numpy(), train_pos[u].cpu().numpy()), group))

#         rsp_prop = sum_rec / sum_pos
#         rsp = np.std(rsp_prop) / np.mean(rsp_prop) if np.mean(rsp_prop) != 0 else 0
#         return rsp

#     @staticmethod
#     def reo(items, **kwargs):
#         """
#         REO 계산 함수.
#         blen_pop : 인기도를 나타내는 텐서
#         items : 추천된 아이템 목록
#         test_pos : 테스트 데이터에서 선호 아이템 목록
#         num_groups : 그룹 개수 (기본값: 2)
#         """
#         blen_pop = kwargs['blen_pop']
#         test_pos = kwargs['test_pos']
#         num_groups = kwargs.get('num_groups', 2)
        
#         sorted_indices = torch.argsort(blen_pop, descending=True)
#         pop_group = sorted_indices[:math.floor(len(sorted_indices) * 0.2)]
#         unpop_group = sorted_indices[math.floor(len(sorted_indices) * 0.2):]
#         group_lists = [pop_group.cpu().numpy(), unpop_group.cpu().numpy()]

#         relevant_counts = np.zeros(num_groups)
#         total_counts = np.zeros(num_groups)

#         test_pos = test_pos.cpu().numpy()
#         for j, group in enumerate(group_lists):
#             group_items_in_recommendation = np.isin(items, group)
#             liked_items_in_test_set = np.isin(test_pos, group)
#             relavant_counts = np.sum(group_items_in_recommendation & np.isin(items, test_pos))
#             total_counts = np.sum(liked_items_in_test_set)
        
#         # for i in range(len(test_pos)):
#         #     test_pos_i = test_pos[i].cpu().numpy()
#         #     for j, group in enumerate(group_lists):
#         #         group_items_in_recommendation = np.isin(items[i], group)
#         #         liked_items_in_test_set = np.isin(test_pos_i, group)

#         #         relevant_counts[j] += np.sum(group_items_in_recommendation & np.isin(items[i], test_pos_i))
#         #         total_counts[j] += np.sum(liked_items_in_test_set)

#         reo_prop = np.divide(relevant_counts, total_counts, out=np.zeros_like(relevant_counts), where=total_counts != 0)
#         reo = np.std(reo_prop) / np.mean(reo_prop) if np.mean(reo_prop) != 0 else 0
#         return reo

import numpy as np
import torch
import torch.nn as nn

class Judger(object):
    def __init__(self, topk):
        self.topk = topk
    def judge(self, items, train_pos, test_pos, num_test_pos, group_lists):
        results = {}
        items = items.cpu()
        metrics = ['recall', 'hit_ratio', 'ndcg', 'rsp', 'reo']
        for metric in metrics:
            f = Metrics.get_metrics(metric)
            if metric in ['recall', 'hit_ratio', 'ndcg']:
                # Calculate recall, hit_ratio, ndcg
                results[metric] = sum([f(items[i], test_pos=test_pos[i], num_test_pos=num_test_pos[i].item()) if num_test_pos[i] > 0 else 0 for i in range(len(items))])
            elif metric == 'rsp':
                # Calculate rsp
                results[metric] = sum([f(items[i], train_pos = train_pos[i], group_lists = group_lists, num_test_pos = num_test_pos[i].item()) if num_test_pos[i] > 0 else 0 for i in range(len(items))])
            elif metric == 'reo':
                # Calculate reo
                results[metric] = sum([f(items[i], test_pos = test_pos[i], group_lists = group_lists, num_test_pos = num_test_pos[i].item()) if num_test_pos[i] > 0 else 0 for i in range(len(items))])

        valid_num_users = sum([1 if t > 0 else 0 for t in num_test_pos])
        
        return results, valid_num_users


class Metrics(object):
    @staticmethod
    def get_metrics(metric):
        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'ndcg': Metrics.ndcg,
            'rsp': Metrics.rsp,
            'reo': Metrics.reo
        }
        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):
        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()
        return hit_count / num_test_pos

    @staticmethod
    def hr(items, **kwargs):
        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()
        
        return 1.0 if hit_count > 0 else 0

    @staticmethod
    def ndcg(items, **kwargs):
        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        index = np.arange(len(items))
        k = min(len(items), num_test_pos)
        idcg = (1 / np.log(2 + np.arange(k))).sum()
        dcg = (1 / np.log(2 + index[np.isin(items, test_pos)])).sum()
        
        return dcg / idcg


    
    @staticmethod
    def rsp(items, **kwargs):
        train_pos = kwargs['train_pos']
        group_lists = kwargs['group_lists']
        sum_rec = np.zeros(2)
        sum_pos = np.zeros(2)
        for j, group in enumerate(group_lists):
            sum_rec[j] += np.sum(np.isin(items, group))
            sum_pos[j] += np.sum(np.isin(np.setdiff1d(np.concatenate(group_lists),train_pos),group))
        rsp_prop = sum_rec / sum_pos
        rsp = np.std(rsp_prop) / np.mean(rsp_prop)
        # group_size = len(blen_pop) // num_groups
        # groups = [blen_pop[i * group_size: (i + 1) * group_size] for i in range(num_groups)]
        # if len(blen_pop) % num_groups > 0:
        #     groups[-1] = torch.cat((groups[-1], blen_pop[-(len(blen_pop) % num_groups):]))

        # group_lists = [group.cpu().numpy() for group in groups]

        # batch_sums_rec = np.array([[np.isin(items[i], group).sum() for group in group_lists] for i in range(len(items))])
        # batch_sums_pos = np.array([[len(group) - np.isin(train_pos[i].cpu(), group).sum() for group in group_lists] for i in range(len(train_pos))])

        # sum_rec = np.sum(batch_sums_rec, axis=0)
        # sum_pos = np.sum(batch_sums_pos, axis=0)
        
        # rsp_prop = sum_rec / sum_pos
        
        # # Handle division by zero or invalid values
        # mean_rsp_prop = np.mean(rsp_prop)
        # rsp = np.std(rsp_prop) / mean_rsp_prop if mean_rsp_prop != 0 else 0  # Prevent division by zero
        
        return rsp

    @staticmethod
    # def reo(items, test_pos, blen_pop, num_groups):
    #     # Create group_lists similarly as in the rsp function
    #     group_size = len(blen_pop) // num_groups
    #     groups = [blen_pop[i * group_size: (i + 1) * group_size] for i in range(num_groups)]
    #     if len(blen_pop) % num_groups > 0:
    #         groups[-1] = torch.cat((groups[-1], blen_pop[-(len(blen_pop) % num_groups):]))

    #     group_lists = [group.cpu().numpy() for group in groups]

    #     relevant_counts = np.zeros(num_groups)
    #     total_counts = np.zeros(num_groups)

    #     for i in range(len(test_pos)):
    #         test_pos_i = test_pos[i].cpu().numpy()
    #         for j, group in enumerate(group_lists):
    #             relevant_counts[j] += np.sum(np.isin(items[i], group) & np.isin(items[i], test_pos_i))
    #             total_counts[j] += np.isin(items[i], group).sum()

    #     reo_prop = np.divide(relevant_counts, total_counts, out=np.zeros_like(relevant_counts), where=total_counts != 0)
    #     reo = np.std(reo_prop) / np.mean(reo_prop) if np.mean(reo_prop) != 0 else 0

    #     return reo
    def reo(items, **kwargs):
        # Create group_lists similarly as in the rsp function
        test_pos = kwargs['test_pos']
        group_lists = kwargs['group_lists']
        rel_counts = np.zeros(2)
        total_counts = np.zeros(2)
        
        for j,group in enumerate(group_lists):
            group_items_in_rec = np.isin(items,group)
            liked_items_in_test = np.isin(test_pos,group)
            rel_counts[j] = np.sum(group_items_in_rec & np.isin(items,test_pos))
            total_counts[j] = np.sum(liked_items_in_test)
        reo_prop = np.divide(rel_counts, total_counts, out = np.zeros_like(rel_counts), where = total_counts != 0)
        reo = np.std(reo_prop) / np.mean(reo_prop) if np.mean(reo_prop) != 0 else 0
        return reo