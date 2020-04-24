from collections import Counter
import os
import pickle

import numpy as np

def ave(nums):
    return sum(nums)/len(nums)

def get_ave_length(sentences):
    return ave([len(s['text']) for s in sentences])

def get_ave_wordcount(sentences):
    return ave([len(s['words']) for s in sentences])

def get_ave_focusratio(sentences):
    return ave([sum(s['in_focus'])/len(s['words']) for s in sentences])

def get_ave_knownratio(sentences):
    return ave([sum(s['in_known'])/len(s['words']) for s in sentences])

def get_full_analysis(data):
    sentences = data['sentences']
    analysis = {
        #'ave_wordcount': get_ave_wordcount(sentences),
        #'ave_length': get_ave_length(sentences),
        'ave_focusratio': get_ave_focusratio(sentences),
        'ave_knownratio': get_ave_knownratio(sentences),
    }
    analysis['known+focus'] = analysis['ave_focusratio'] + analysis['ave_knownratio']
    return analysis

def get_datas():
    datas = []
    folder = 'outdata'
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'rb') as infile:
            data = pickle.load(infile)
            data['sentences'] = [s for s in data['sentences']
                                 if len(s['words']) > 0]
            datas.append(data)
    return datas

def get_sentence(datas, n):
    for d in datas:
        print(d['sentences'][n]['text'])

def data_get(datas, **kwargs):
    for d in datas:
        for k, v in kwargs.items():
            if d['options'].get(k) != v:
                break
        else:
            return d

def sweep(datas, **kwargs):
    ret = []
    for d in datas:
        for k, v in kwargs.items():
            if d['options'].get(k) != v:
                break
        else:
            ret.append(d)
    return ret

def select(datas, **kwargs):
    ret = []
    for d in datas:
        for k, v in kwargs.items():
            pass
