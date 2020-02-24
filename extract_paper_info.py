import re
import codecs
import json
from utils import load_json, dump_json

r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

stopword = ['university','univ','china','department','dept','laboratory','lab','school','al','et',
            'institute','inst','college','chinese','journal','science','international']


def dump_paper_feature(file_path, save_path):

    #提取所有论文信息方便以后查找
    # file_path = "data/test/pub.json"
    paper_data = load_json(file_path)

    paper_name ={}     
    paper_org ={}
    paper_title ={}
    paper_venue ={}
    paper_keywords ={}

    # 提取title venue keywords authors orgs
    for i in paper_data:
        if 'title' in paper_data[i]:
            title = paper_data[i]['title']
            title = title.lower().split()
            tmp = ''
            for word in title:
                tmp += re.sub(r, ' ', word).strip('\n').replace('\n', '') + ' '
            paper_title[i] = tmp
        else:
            paper_title[i] = ''

        if 'venue' in paper_data[i]:
            venue = paper_data[i]['venue']
            venue = venue.lower().split()
            tmp = ''
            for word in venue:
                tmp += re.sub(r, ' ', word).strip('\n').replace('\n', '') +' '
            paper_venue[i] = tmp
        else:
            paper_venue[i] = ''
        
        if 'keywords' in paper_data[i]:
            keywords = paper_data[i].get('keywords')
            tmp = ''
            for word in keywords:
                tmp += re.sub(r, ' ', word).strip('\n').replace('\n', '') +' '
            paper_keywords[i] = tmp
        else:
            paper_keywords[i] = ''

        if 'authors' in paper_data[i]:
            authors = paper_data[i].get('authors')
            name_tmp = ''
            org_tmp  = ''
            for j in range(len(authors)):
                if j>100:
                    break
                name_ = authors[j]['name']
                name_ = re.sub(r' ', '_', name_)
                # name_ = re.sub(r, '', name_)
                name_tmp += name_ +' '
                
                if 'org' in authors[j]:
                    org = authors[j].get('org')
                    org = re.sub(r, '', org)
                    org_tmp += org +' '

            paper_name[i] = name_tmp
            paper_org[i] = org_tmp

    #分别保存

    dump_json(save_path + '/paper_keywords.json', paper_keywords)
    dump_json(save_path + '/paper_author_name.json', paper_name)
    dump_json(save_path + '/paper_org.json', paper_org)
    dump_json(save_path + '/paper_title.json', paper_title)
    dump_json(save_path + '/paper_venue.json', paper_venue)

    print('dump_paper_feature done')


def merge_paper_info(mission):

    if mission == 'test':
        paper_keywords = load_json('data/test/paper_keywords.json')
        paper_title = load_json('data/test/paper_title.json')
        paper_org =  load_json('data/test/paper_org.json')

    papers = [i for i in paper_title]
    papers_info_merge = {}

    for i in papers:
        sentence = ''
        if paper_org[i] != []:
            for j in paper_org[i]:
                sentence += j
                sentence += ''
        if paper_keywords[i] != []:
            for j in paper_keywords[i]:
                sentence += j
                sentence += ' '
        sentence += paper_title[i]
        papers_info_merge[i] = sentence

    dump_json('data/test/paper_info_merge.json', papers_info_merge)
    print('paper info merge done')


if __name__ == "__main__":

    dump_paper_feature('data/test/pub.json', 'data/test')
    # merge_paper_info('test')

    

