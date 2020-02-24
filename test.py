import codecs
import json
import numpy as np
from utils import pairwise_precision_recall_f1, load_json, dump_json, UnionSet, pred
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import fasttext
import re
from sklearn.cluster import DBSCAN, OPTICS
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from model import MYModel
from torch.utils.data import  DataLoader
from scipy.spatial.distance import cdist
from collections import Counter

r = '[^a-zA-Z0-9,.\'_!?]+'

author_data = load_json('data/test/author.json')
pud_data = load_json('data/test/pub.json')
paper_author_name = load_json('data/test/paper_author_name.json')
paper_org = load_json('data/test/paper_org.json')
paper_venue = load_json('data/test/paper_venue.json')
paper_keywords = load_json('data/test/paper_keywords.json')
paper_title = load_json('data/test/paper_title.json')
label_data = load_json('data/test/author_label.json')

word_emb_model = fasttext.load_model('word_embedding/emb_768.bin')
batch_size = 1
hidden_dim = 1024
num_layers = 2
emb_dim = 768
device = 'cuda'
model = MYModel(emb_dim, hidden_dim, batch_size, device, num_layers).to(device)
model.load_state_dict(torch.load('model/02-13 19_27_51'))
model.eval()
model.eval().to(device)

names = [i for i in author_data]

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

def get_cluster(sim, margin, num_paper):
    margin1 = margin[0]
    margin2 = margin[1]
    margin3 = margin[2]
    margin4 = margin[3]
    margin5 = margin[4]
    us = UnionSet(num_paper)
    for i in range(num_paper):
        for j in range(num_paper):
            if sim[i][j] > margin1:
                us.merge(i,j)
    for i in range(num_paper):
        us.f[i] = us.find(i)
    c1 = Counter(us.f)
    outer = [i for i in c1 if c1[i] == 1]
    print('num outer after step1 :{}'.format(len(outer)))
    most = []
    most.append(c1.most_common()[0][0])
    for i in outer:
        if us.find(i) in most:
            print('bug! ',i)
        for j in range(num_paper):
            if us.find(j) in most:
                continue
            assert(us.find(j) not in most)
            if sim[i][j] > margin2:
                us.merge(i, j)
    for i in range(num_paper):
        us.f[i] = us.find(i)
    c1 = Counter(us.f)
    outer = [i for i in c1 if c1[i] == 1]
    print('num outer after step2 :{}'.format(len(outer)))
    most.append(c1.most_common()[0][0])
    most.append(c1.most_common()[1][0])
    most.append(c1.most_common()[2][0])
    most.append(c1.most_common()[3][0])
    most.append(c1.most_common()[4][0])
    most.append(c1.most_common()[5][0])
    for i in outer:
        if us.find(i) in most:
            print('bug! ',i)
        for j in range(num_paper):
            if us.find(j) in most:
                continue
            assert(us.find(j) not in most)
            if sim[i][j] > margin3:
                us.merge(i, j)
    for i in range(num_paper):
        us.f[i] = us.find(i)
    c1 = Counter(us.f)
    outer = [i for i in c1 if c1[i] == 1]
    print('num outer after step3 :{}'.format(len(outer)))
    most.append(c1.most_common()[6][0])
    most.append(c1.most_common()[7][0])
    most.append(c1.most_common()[8][0])
    for i in outer:
        if us.find(i) in most:
            print('bug! ',i)
        for j in range(num_paper):
            if us.find(j) in most:
                continue
            assert(us.find(j) not in most)
            if sim[i][j] > margin4:
                us.merge(i, j)
    for i in range(num_paper):
        us.f[i] = us.find(i)
    c1 = Counter(us.f)
    outer = [i for i in c1 if c1[i] == 1]
    print('num outer after step4 :{}'.format(len(outer)))
    most.append(c1.most_common()[9][0])
    most.append(c1.most_common()[10][0])
    most.append(c1.most_common()[11][0])
    for i in outer:
        if us.find(i) in most:
            print('bug! ',i)
        for j in range(num_paper):
            if us.find(j) in most:
                continue
            assert(us.find(j) not in most)
            if sim[i][j] > margin5:
                us.merge(i, j)
    for i in range(num_paper):
        us.f[i] = us.find(i)
    c1 = Counter(us.f)
    outer = [i for i in c1 if c1[i] == 1]
    print('num outer after step5 :{}'.format(len(outer)))
    return c1, us, outer

for name in names:
    papers = author_data[name]
    num_paper = len(papers)
    print("{} papers {}".format(name, num_paper))
    #get sentence embedding
    data = []
    for i in papers:
        name_ = paper_author_name[i]
        keywords_ = paper_keywords[i]
        org_= paper_org[i]
        title_ = paper_title[i]
        venue_ = paper_venue[i]

        name_ = word_emb_model.get_sentence_vector(name_)
        keywords_ = word_emb_model.get_sentence_vector(keywords_)
        org_ = word_emb_model.get_sentence_vector(org_)
        title_ = word_emb_model.get_sentence_vector(title_)
        venue_ = word_emb_model.get_sentence_vector(venue_)
        data.append([name_,org_,venue_,keywords_, title_])
    data = np.array(data)

    #get embeddings from model
    embeddings = []
    for i in range(num_paper):
        x = torch.from_numpy(data[i]).view(1,5, emb_dim).to(device)
        out = model(x)
        out = out.view(5,hidden_dim)
        embeddings.append(out)

    #cul sim
    sim1 = np.ones((num_paper, num_paper))
    for i in range(num_paper):
        for j in range(num_paper):
            x = F.pairwise_distance(embeddings[i],embeddings[j],p=1)
            x = torch.exp(-x)
            sim1[i][j] = x.mean().item()

    margin = [0.95, 0.9 ,0.85, 0.8, 0.7]
    c1 ,us , outer = get_cluster(sim1, margin, num_paper)
    pred(c1, name)

c2

clu = DBSCAN(min_samples=2).fit_predict(sim1)
c1 = Counter(clu)
