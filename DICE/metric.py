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