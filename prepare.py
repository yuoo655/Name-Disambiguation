import codecs
import fasttext
import numpy as np
import re
import pandas as pd
from utils import load_json
from tqdm import tqdm

embedding_model = fasttext.load_model('word_embedding/wiki.en/wiki.en.bin')

# embedding_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
embedding_model.to('cuda')
anc = np.load('data/train/anc_.npy')
pos = np.load('data/train/pos_.npy')
neg = np.load('data/train/neg_.npy')
paper_author_name = load_json('data/train/paper_author_name.json')
paper_keywords = load_json('data/train/paper_keywords.json')
paper_org = load_json('data/train/paper_org.json')
paper_title = load_json('data/train/paper_title.json')
paper_venue = load_json('data/train/paper_venue.json')

r = '[^a-zA-Z0-9,.\'_!?]+'

# anc data
anc_author_name = [paper_author_name[i].replace('\n','') for i in anc]
anc_keywords =  [paper_keywords[i].replace('\n','') for i in anc]
anc_org = [paper_org[i].replace('\n','') for i in anc]
anc_title = [paper_title[i].replace('\n','') for i in anc]
anc_venue = [paper_venue[i].replace('\n','') for i in anc]
# pos data
pos_author_name = [paper_author_name[i].replace('\n','') for i in pos]
pos_keywords =  [paper_keywords[i].replace('\n','') for i in pos]
pos_org = [paper_org[i].replace('\n','') for i in pos]
pos_title = [paper_title[i].replace('\n','') for i in pos]
pos_venue = [paper_venue[i].replace('\n','') for i in pos]
# neg data
neg_author_name = [paper_author_name[i].replace('\n','') for i in neg]
neg_keywords =  [paper_keywords[i].replace('\n','') for i in neg]
neg_org = [paper_org[i].replace('\n','') for i in neg]
neg_title = [paper_title[i].replace('\n','') for i in neg]
neg_venue = [paper_venue[i].replace('\n','') for i in neg]

anc_author_name = [embedding_model.get_sentence_vector(i) for i in anc_author_name]
anc_keywords = [embedding_model.get_sentence_vector(i) for i in anc_keywords]
anc_org = [embedding_model.get_sentence_vector(i) for i in anc_org]
anc_title = [embedding_model.get_sentence_vector(i) for i in anc_title]
anc_venue = [embedding_model.get_sentence_vector(i) for i in anc_venue]

pos_author_name = [embedding_model.get_sentence_vector(i) for i in pos_author_name]
pos_keywords = [embedding_model.get_sentence_vector(i) for i in pos_keywords]
pos_org = [embedding_model.get_sentence_vector(i) for i in pos_org]
pos_title = [embedding_model.get_sentence_vector(i) for i in pos_title]
pos_venue = [embedding_model.get_sentence_vector(i) for i in pos_venue]

neg_author_name = [embedding_model.get_sentence_vector(i) for i in neg_author_name]
neg_keywords = [embedding_model.get_sentence_vector(i) for i in neg_keywords]
neg_org = [embedding_model.get_sentence_vector(i) for i in neg_org]
neg_title = [embedding_model.get_sentence_vector(i) for i in neg_title]
neg_venue =[embedding_model.get_sentence_vector(i) for i in neg_venue]

anc = np.array((anc_author_name, anc_org, anc_venue, anc_keywords, anc_title)).reshape(len(anc_org), 5, 300)
pos = np.array((pos_author_name, pos_org, pos_venue, pos_keywords, pos_title)).reshape(len(anc_org), 5, 300)
neg = np.array((neg_author_name, neg_org, neg_venue, neg_keywords, neg_title)).reshape(len(anc_org), 5, 300)

np.save('data/train/emb_data/anc_300',anc)
np.save('data/train/emb_data/pos_300',pos)
np.save('data/train/emb_data/neg_300',neg)
