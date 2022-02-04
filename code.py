#!/usr/bin/python
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Shawn
# Created Date: 2022/01/20
# version ='0.5'
# ---------------------------------------------------------------------------
""" labeling tourist attractions"""
# ---------------------------------------------------------------------------
from itertools import count
from gensim.corpora import WikiCorpus
from opencc import OpenCC
from re import sub as re_sub
from jieba import set_dictionary as j_set_dict, cut as j_cut, setLogLevel as j_set_logl
from sys import stdout, stderr, exc_info
from os import cpu_count
from gensim.models import word2vec
from multiprocessing import Pool, RLock, Manager
from time import sleep, time
from traceback import extract_tb
from logging import INFO as log_info
from warnings import filterwarnings
# jieba dict
dictionary_path = r'C:/VS_Workplace/graduate_project/data/dict.txt.big.txt'
# wiki data for training w2v model
sources = r"C:/VS_Workplace/graduate_project/data/zhwiki-latest-pages-articles.xml.bz2"
# wiki sentences for training w2v model
output = r"C:/VS_Workplace/graduate_project/data/wiki_sentences.txt"
# w2v model
modelPath = r'C:/VS_Workplace/graduate_project/data/w2v.bin'
# stop word
stopPath = r'C:/VS_Workplace/graduate_project/data/ChineseStopWords.txt'
# sentences without stop words for training w2v model
sentences = r"C:/VS_Workplace/graduate_project/data/sentences.txt"


def train():
    """
    # architecture: skip-gram (slower, better for infrequent words) vs CBOW (fast)
    # the training algorithm: hierarchical softmax (better for infrequent words) vs 
    # negative sampling (better for frequent words, better with low dimensional vectors)
    # sub-sampling of frequent words: can improve both accuracy and speed for large 
    # data sets (useful values are in range 1e-3 to 1e-5)
    # dimensionality of the word vectors: usually more is better, but not always
    # context (window) size: for skip-gram usually around 10, for CBOW around 5
    """
    """
    # vector_size : int, optional
    #     Dimensionality of the word vectors.
    # window : int, optional
    #     Maximum distance between the current and predicted word within a sentence.
    # min_count : int, optional
    #     Ignores all words with total frequency lower than this.
    # workers : int, optional
    #     Use these many worker threads to train the model (=faster training with multicore machines).
    # sg : {0, 1}, optional
    #     Training algorithm: 1 for skip-gram; otherwise CBOW.
    # hs : {0, 1}, optional
    #     If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.
    # negative : int, optional
    #     If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
    # ns_exponent : float, optional
    #     The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
    #     More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that other values may perform better for recommendation applications.
    # cbow_mean : {0, 1}, optional
    #     If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    # alpha : float, optional
    #     The initial learning rate.
    # min_alpha : float, optional
    #     Learning rate will linearly drop to min_alpha as training progresses.
    # seed : int, optional
    #     Seed for the random number generator. Initial vectors for each word are seeded with a hash of the concatenation of word + str(seed). Note that for a fully deterministically-reproducible run, you must also limit the model to a single worker thread (workers=1), to eliminate ordering jitter from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED environment variable to control hash randomization).
    # max_vocab_size : int, optional
    #     Limits the RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM. Set to None for no limit.
    # max_final_vocab : int, optional
    #     Limits the vocab to a target vocab size by automatically picking a matching min_count. If the specified min_count is more than the calculated min_count, the specified min_count will be used. Set to None if not required.
    # sample : float, optional
    #     The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
    # hashfxn : function, optional
    #     Hash function to use to randomly initialize weights, for increased training reproducibility.
    # epochs : int, optional
    #     Number of iterations (epochs) over the corpus. (Formerly: iter)
    
    """
    # skip-gram
    sg = 1
    window_size = 10
    vector_size = 250
    min_count = 1
    workers = cpu_count()
    epochs = 5
    train_data = word2vec.LineSentence(sentences)
    model = word2vec.Word2Vec(
        train_data, epochs=epochs, vector_size=vector_size, sg=sg, window_size=window_size, workers=workers)
    model.save(modelPath)


def printFile(path: str):
    with open(path, "rb") as f:
        for read_f in f.readlines()[:100]:
            stdout.write(f'{read_f.decode("utf-8", "ignore").rstrip()}\n')


def print_exception(e: Exception):
    error_class = e.__class__.__name__  # 取得錯誤類型
    detail = e.args[0]  # 取得詳細內容
    cl, exc, tb = exc_info()  # 取得Call Stack
    lastCallStack = extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
    fileName = lastCallStack[0]  # 取得發生的檔案名稱
    lineNum = lastCallStack[1]  # 取得發生的行號
    funcName = lastCallStack[2]  # 取得發生的函數名稱
    errMsg = f"File \"{fileName}\", line {lineNum}, in {funcName}: [{error_class}] {detail}\n"
    stderr.write(f'{e}')


# read s2tw seg store

class preproccess():
    def __init__(self) -> None:
        pass


def job():
    start_time = time()
    # set seg dict
    j_set_dict(dictionary_path=dictionary_path)
    j_set_logl(log_info)
    # read articles from compressed wiki file ignore windows warning
    filterwarnings(
        'ignore', 'detected Windows; aliasing chunkize to chunkize_serial',)
    corpus = WikiCorpus(sources, dictionary={})
    data = corpus.get_texts()
    # simplified to traditional
    s2tw = OpenCC('s2tw')
    # format will be sentences seperate by '\n' with seg words sep by ' '
    f = open(sentences, mode='w', encoding='utf-8')
    count = 0
    for a in data:
        result = ''
        for line in a:
            # reserve only chinese
            l1 = re_sub(r'[^\u4e00-\uf9a5]', '', line)
            if l1 == '':
                continue
            # convert to tradtional, seg into word, append by sentence
            result += f'{" ".join(j_cut(s2tw.convert(l1), cut_all=False, HMM=True, use_paddle=True))}\n'
        f.write(result)
        stdout.write(f'{count} done\n')
        count += 1
        # if count > 32:
        #     break
    f.close()
    end_time = time()
    # 15685
    total_time = end_time - start_time
    (total_time, time_unit) = (total_time, 's') if total_time < 60 else (
        total_time / 60, 'm') if total_time < 3600 else (total_time / 3600, 'h')
    stdout.write(f'After {total_time}{time_unit} done\n')

# failed


def multi_preprocessing():
    def func(params):
        jobList = params['jobList']
        s2tw = OpenCC("s2tw")
        results = []
        # disable jieba log
        j_set_logl(log_info)
        for a in jobList:
            result = ''
            for line in a:
                l1 = re_sub(r'[^\u4e00-\uf9a5]', '', line)
                if l1 == '':
                    continue
                l2 = s2tw.convert(l1)
                l3 = j_cut(l2, cut_all=False)
                l4 = ' '.join(l3)
                result += ' ' + l4
            results.append(result[1:])
            # stdout.write(result[1:]+'\n')
        return results

    def multiRun(jobList, func, params, coreNum):
        # handle child error

        def handle_error(e):
            print_exception(e)

        def divJobList(jobList, coreNum):
            res = []
            length = len(jobList)
            left = length % coreNum
            right = coreNum-left
            step = int(length/coreNum)
            for _ in range(right):
                res.append(jobList[:step])
                jobList = jobList[step:]
            step += 1
            for _ in range(left):
                res.append(jobList[:step])
                jobList = jobList[step:]
            return res

        def subRun(idx, returnDict, func, params):
            returnDict[idx] = func(params)

        pool = Pool(coreNum)
        returnDict = Manager().dict()
        groupJobList = divJobList(jobList, coreNum)
        for idx in range(coreNum):
            newParams = params.copy()
            newParams['jobList'] = groupJobList[idx]
            pool.apply_async(func=subRun, args=(
                idx, returnDict, func, newParams))
        pool.close()
        pool.join()
        f = open(sentences, mode='a', encoding='utf-8')
        for idx in range(coreNum):
            for line in returnDict[idx]:
                f.write(line)
                f.write('\n')
        f.close()

    start_time = time()
    # set seg dict
    j_set_dict(dictionary_path=dictionary_path)

    # read articles from wiki file
    corpus = WikiCorpus(sources, dictionary={})
    data = corpus.get_texts()
    # set file empty
    with open(sentences, mode='w', encoding='utf-8') as f:
        pass
    # max parallel proccess
    max_worker = cpu_count()
    # subproccess has to be pickable which generator is not, so we have to recursively use list
    count = 0
    keep = True
    while keep:
        jobList = []
        for _ in range(max_worker):
            try:
                jobList.append(next(data))
                count += 1
            except StopIteration as stop:
                keep = False
                break
        multiRun(jobList, func, {}, coreNum=max_worker)
        if count > 16:
            break
    end_time = time()
    stdout.write(f'After {end_time-start_time}s done\n')


def remove_stop():
    # load stopwords set
    stopword_set = set()
    with open(stopPath, 'r', encoding='utf-8') as f:
        for w in f.readlines():
            w = re_sub(r"[^\u4e00-\uf9a5]", "", w)
            stopword_set.add(w)
    out_f = open(sentences, mode="w", encoding="utf-8")
    read_f = open(output, mode="r+")
    for l in read_f.readlines():
        out_f.write(" ".join(w for w in l.rstrip().split()
                    if w not in stopword_set))
        out_f.write("\n")
    read_f.close()
    out_f.close()


if __name__ == '__main__':
    # train()
    job()
    # printFile(sentences)
    pass
