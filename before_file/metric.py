#%%
import numpy as np


class Judger(object):
    def __init__(self, topk):
        self.topk = topk
    def judge(self, items, test_pos, num_test_pos):
        results = {}
        items = items.cpu()
        metrics = ['recall','hit_ratio','ndcg']
        for metric in metrics:
            f = Metrics.get_metrics(metric)
            results[metric] = sum([f(items[i], test_pos = test_pos[i], num_test_pos = num_test_pos[i].item()) if num_test_pos[i] > 0 else 0 for i in range(len(items))])
            
        valid_num_users = sum([1 if t > 0 else 0 for t in num_test_pos])
        
        return results, valid_num_users
    
    
class Metrics(object):
    @staticmethod
    def get_metrics(metric):
        metrics_map = {
            'recall' : Metrics.recall,
            'hit_ratio' : Metrics.hr,
            'ndcg' : Metrics.ndcg 
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
        
        if hit_count > 0 :
            return 1.0
        else:
            return 0
    @staticmethod    
    def ndcg(items, **kwargs):
        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        index = np.arange(len(items))
        k = min(len(items), num_test_pos)
        idcg = (1/np.log(2 + np.arange(k))).sum()
        dcg = (1/np.log(2 + index[np.isin(items, test_pos)])).sum()
        
        return dcg/idcg

# %%
