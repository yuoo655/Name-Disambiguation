import codecs
import json
from utils import load_json, dump_json
import random
import numpy as np

author_data = load_json('data/train/author.json')
paper_data = load_json('data/train/pub.json')

author_names = author_data.keys()
paper_names = paper_data.keys()
print('author', len(author_names))
print('paper', len(paper_names))

anc = []
pos = []
neg = []
for name in author_names:
    papers_dict = author_data[name]
    authors = papers_dict.keys()
    
    for author_id in papers_dict:
        if len(papers_dict[author_id]) == 1:
            for i in range(5):
                paper_anc = papers_dict[author_id][0]
                paper_pos = papers_dict[author_id][0]
                neg_name = name
                while neg_name == name:
                    neg_name = random.choice(list(author_names))
                neg_author= random.choice(list(author_data[name].keys()))
                neg_paper = random.choice(author_data[name][neg_author])
                paper_neg = neg_paper
                anc.append(paper_anc)
                pos.append(paper_pos)
                neg.append(paper_neg)

        papers = papers_dict[author_id]
        #anc
        for i in papers:
            paper_anc = i
            for j in range(5):
                paper_pos = random.choice(papers)
                author_id_neg = author_id
                while author_id_neg == author_id:
                    author_id_neg = random.choice(list(papers_dict))
                paper_neg = random.choice(papers_dict[author_id_neg])
                anc.append(paper_anc)
                pos.append(paper_pos)
                neg.append(paper_neg)
            for k in range(5):
                paper_pos = random.choice(papers)
                while neg_name == name:
                    neg_name = random.choice(list(author_names))
                neg_author= random.choice(list(author_data[name].keys()))
                paper_neg = random.choice(author_data[name][neg_author])
                anc.append(paper_anc)
                pos.append(paper_pos)
                neg.append(paper_neg)

anc = np.array(anc)
pos = np.array(pos)
neg = np.array(neg)
np.save('data/train/anc_', anc)
np.save('data/train/pos_', pos)
np.save('data/train/neg_', neg)


