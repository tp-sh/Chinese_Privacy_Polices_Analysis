"""
    输入 csv
    生成 tfidf embedding
"""

import pandas as pd
import numpy as np
import jieba
import joblib
import re
import torch
import os

from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

def get_svd(data_X, save_path="svd"):
    if (os.path.exists(save_path)):
        return joblib.load(save_path)
    svd = TruncatedSVD(300)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    data_X = lsa.fit_transform(data_X)
    joblib.dump(lsa, save_path)
    return lsa

def get_vectorizer(data_X, save_path="tfidf"):
    if (os.path.exists(save_path)):
        return joblib.load(save_path)
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", ngram_range=(1,3), norm="l2")
    vectorizer.fit(data_X)
    joblib.dump(vectorizer, save_path)
    return vectorizer


def cleanPunc(sentence):
    """删除符号"""
    sentence = str(sentence)
    cleaned = re.sub(r'[？|！|‘|“]', r" ", sentence)
    cleaned = re.sub(r'[?|!|\'|"|#]', r" ", sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/|<|>|;|:]', r" ", cleaned)
    cleaned = re.sub(r'[。|，|）|（|、|…|—|·|《|》|：|；|【|】]', r" ", cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

def text_clean(text):
    STOPWORDS = [line.strip() for line in open('classifier/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
    NonSTOPWORDS = []
    # 保留一些有意义的停止词
    text = jieba.cut(text)
    textlist = ''.join(text).split()
    text = ' '.join([ w for w in textlist if ((w not in STOPWORDS) or (w in NonSTOPWORDS))])
    return text

def get_tfidf_embedding(filepath="all_train_0411_clean.csv", save_path="tfidf_embeddings_train_0411data.pt"):
    data = pd.read_csv(filepath)
    data_X = data['text'].apply(cleanPunc).apply(text_clean)

    vectorizer = get_vectorizer(data_X)
    data_X = vectorizer.transform(data_X)
    lsa = get_svd(data_X)

    data_X = lsa.transform(data_X)
    # data_y = data.drop(labels=list(set(data.columns)&set(["label", "text","parents","siblings"])), axis=1)
    torch.save(data_X, save_path)
    return data_X

if __name__=="__main__":
    get_tfidf_embedding("datasets/all_train_0411_clean.csv", "classifier_data/tfidf_embeddings_train_0411data.pt")
    get_tfidf_embedding("datasets/all_test_0411_clean.csv", "classifier_data/tfidf_embeddings_test_0411data.pt")

    