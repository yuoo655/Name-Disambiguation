
import codecs
import json
import numpy as np
from collections import Counter

def load_json(file_path):
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def dump_json(file_path, data):
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn

    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def cal_f1(prec, rec):
    return 2*prec*rec/(prec+rec)

def pred(c1, name):
    #predict
    clustered = []
    count = 0
    for i in c1:
        num = c1[i]
        clustered.extend([count]*num)
        count += 1
    print("predict num cluster {}".format(len(Counter(clustered))))

    #labels
    labels = label_data[name]
    tmp = []

    count = 0
    for i, elem in enumerate(labels):
        num = len(labels[elem])
        tmp.extend([i]*num)
        if num  == 1:
            count += 1
    labels = tmp
    c2 = Counter(labels)
    print("truth num cluster {}".format(len(Counter(labels))))

    prec, rec, f1 =  pairwise_precision_recall_f1(clustered, labels)
    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1))

class UnionSet(object):
    def __init__(self, num):
        self.count = num
        self.f = np.array([i for i in range(num)])
        self.rank = np.array([i for i in range(num)])

    def find(self, x):

        while self.f[x] != x:
            x = self.f[x]
        return x

    def merge(self, x, y):

        setx = self.find(x)
        sety = self.find(y)

        if setx == sety:
            return
        if setx != sety:
            if self.rank[setx] < self.rank[sety]:
                self.f[sety] = setx
            else:
                self.f[setx] = sety


