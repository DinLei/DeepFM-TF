import numpy as np
from sklearn.metrics import precision_score


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all_mt = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all_mt = all_mt[np.lexsort((all_mt[:, 2], -1 * all_mt[:, 1]))]
    total_losses = all_mt[:, 0].sum()
    gini_sum = all_mt[:, 0].cumsum().sum() / total_losses

    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)


def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


def auc(actual, pred):
    assert (len(actual) == len(pred))
    eval_map = list(zip(pred, actual))
    rank = [a for p, a in sorted(eval_map, key=lambda x: x[0])]
    rank_list = [i for i in range(len(rank)) if rank[i] == 1]

    pos_num = np.sum(actual)
    neg_num = len(actual) - pos_num

    auc_val = (sum(rank_list) - (pos_num * (pos_num - 1)) / 2) / (pos_num * neg_num)
    return auc_val


def precision(actual, pred):
    pred = [1 if x > 0.5 else 0 for x in pred]
    return precision_score(actual, pred, average='micro')
