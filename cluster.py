#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Shawn
# Created Date: 2022/01/20
# version ='0.5'
# ---------------------------------------------------------------------------
""" labeling tourist attractions"""
import re
from sys import stdout
# ---------------------------------------------------------------------------
# pandas, numpy, re for data processing, jieba for word segmenting, word2vec for data quantizing
import jieba
import numpy as np
import pandas as pd
from gensim.corpora import WikiCorpus
from gensim.models import word2vec
from sklearn import preprocessing
#
dictionary_path = r'C:/VS_Workplace/graduate_project/data/dict.txt.big.txt'
#
data_pathes = [r'C:/VS_Workplace/graduate_project/data/hualien.xlsx']
#
fileSegWordDonePath = r'C:/VS_Workplace/graduate_project/data/descriptionSegDone.txt'
#
modelPath = r'C:/VS_Workplace/graduate_project/data/descriptionSegDone.bin'
#
results_path = r'C:/VS_Workplace/graduate_project/data/results.txt'
# stop word
stopPath = r'C:/VS_Workplace/graduate_project/data/ChineseStopWords.txt'
# seg


def seg():
    jieba.set_dictionary(dictionary_path)
    for data_path in data_pathes:
        data = pd.read_excel(data_path)
        with open(fileSegWordDonePath, mode='w', encoding="utf-8") as fW:
            for c in range(len(data['spot'])):
                line = str(data['description'][c])
                line = re.sub(r'[^\u4e00-\u9fa5]', '', line)
                phrases = jieba.cut(line, cut_all=False)
                line = list(phrases)
                fW.write(' '.join(line))
                fW.write('\n')


def load() -> word2vec.Word2Vec:
    try:
        model = word2vec.Word2Vec.load(fname=modelPath)
        return model
    except Exception as e:
        print(e)


def coef(s: str = u"隨便") -> np.array:
    model = load()
    try:
        return model.wv[s].shape
    except Exception as e:
        print(e)
        return np.zeros((model.vector_size,), dtype=np.float32)


def sim():
    stopwords = set()
    with open(stopPath, "r+") as f:
        for w in f.readlines():
            stopwords.add(w.rstrip())
    model = load()

    typ = [u'自然', u"戶外運動", u"藝文", u"風景名勝"]
    data = open(fileSegWordDonePath, 'r+')
    results = []
    for l in data.readlines():
        temp = []
        for t in typ:
            summary = 0
            length = 0
            for w in l.rstrip().split():
                w = re.sub(r"[^\u4e00-\uf9a5]", "", w)

                if w in stopwords:
                    continue
                try:
                    score = model.wv.similarity(t, w)
                    if score > 0.2:
                        summary += score
                        length += 1
                except Exception as e:
                    cut=jieba.cut(w, cut_all=True)
                    for w in cut:
                        try:
                            score = model.wv.similarity(t, w)
                            if score > 0.2:
                                summary += score
                                length += 1
                        except:
                            stdout.write(f"{w}: {str(e)}\n")
                            continue
                    stdout.write(f"{w}: {str(e)}\n")
                    continue
            avg = summary/length if length != 0 else (0., 0)

            temp.append((avg, length))
        results.append(' '.join([str(i) for i in temp]))

    data.close()
    with open(r'C:/VS_Workplace/graduate_project/data/results.txt', "wb") as f:
        for l in results:
            f.write(l.encode('utf-8'))
            f.write(b'\n')


def printFile(path: str):
    with open(path, "r+") as f:
        for i in f.readlines()[:100]:
            stdout.write(i)


if __name__ == '__main__':
    # seg()
    model = load()
    # model.wv.get_vector()
    # train()
    # sim()
    # printFile(results_path)
    # with open(fileSegWordDonePath, "r+") as f:
    #     for w in f.readline().rstrip().split():
    #         try:
    #             stdout.write(f'{w}: {model.wv.similarity(u"自然", w)}\n')
    #         except KeyError:
    #             stdout.write(f'{w}: no key, {0.}\n')
    for i in model.wv.most_similar("廖",topn = 100):
        print(i)