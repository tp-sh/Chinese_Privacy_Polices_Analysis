#In[1]
"""
输入 embeddings
输出 各个 分类器分类结果

对每个二分类, 进行欠采样和超采样以维持正负样本1:2比例, 合并为多标签分类结果
"""
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

import pandas as pd
import numpy as np
import torch
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE 
from collections import Counter

import os
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from loguru import logger

logger.add("logs/tfidf_embedding_{time}.log", level="INFO")
#In[2]
train_features_path = ["datasets/bert_embeddings_train_0412model_0411data.pt"]
test_features_path = ["datasets/bert_embeddings_test_0412model_0411data.pt"]

Xtrains = list(map(torch.load, train_features_path))
# Xtrains[1] = Xtrains[1][:,:30]
Xtrain = np.concatenate(Xtrains, axis=1)
Xtests = list(map(torch.load, test_features_path))
# Xtests[1] = Xtests[1][:,:30]
Xtest = np.concatenate(Xtests, axis=1)

ytrain = pd.read_csv("datasets/train_0411_labels.csv")
ytest = pd.read_csv("datasets/test_0411_labels.csv")

save_path = "compare_models/compare_models" # 0406 训练10个epoch微调后的tfidf 训练集和测试集 与这里一致
if not os.path.isdir(save_path):
    os.makedirs(save_path)
LABEL_COLUMNS = [str(k) for k in range(1,89)]
#In[3]
# 数据集生成

seed = 42
# for t in LABEL_COLUMNS:
t = '1'

def resample_data(X_train, y_train, seed=42):
    dicty = Counter(y_train)
    if dicty[1] >= 1000:
        X_resampled2 = X_train
        y_resampled2 = y_train
    elif dicty[1] >= 500:
        samplerate = float(0.5)
        Rus = RandomUnderSampler(sampling_strategy=samplerate, random_state=seed)
        X_resampled2, y_resampled2 = Rus.fit_resample(X_train,y_train)
    elif dicty[1] >= 50:
        sampledict2 = {1:500}
        smt = SMOTE(sampling_strategy=sampledict2, random_state=seed, k_neighbors=5)
        X_resampled, y_resampled = smt.fit_resample(X_train,y_train)
        sampledict = {0:1000}
        Rus = RandomUnderSampler(sampling_strategy=sampledict, random_state=seed)
        X_resampled2, y_resampled2 = Rus.fit_resample(X_resampled, y_resampled)
    else:
        logger.warning("数据量太少")
        return None, None
    dicty = Counter(y_resampled2)
    logger.info(f"重采样后: 0:{dicty[0]}, 1:{dicty[1]}")
    return X_resampled2, y_resampled2

def my_cross_val(m,x,y,cv=5, seed=42):
    np.random.seed(seed)
    shuffled_index = np.random.permutation(len(x))
    split_cnt = len(x)//cv
    idxs = []
    for i in range(cv):
        idxs.append([i*split_cnt, (i+1)*split_cnt])
        
    idxs[-1][-1] = len(x)
    vals_x = []
    vals_y = []
    for i in range(cv):
        vals_x.append(x[idxs[i][0]:idxs[i][1]])
        vals_y.append(y[idxs[i][0]:idxs[i][1]])
    res = np.zeros(3)
    for i in range(cv):
        data_x = np.concatenate([vals_x[j] for j in range(cv) if j != i])
        val_x = vals_x[i]
        data_y = np.concatenate([vals_y[j] for j in range(cv) if j != i])
        val_y = vals_y[i]
        data_x, data_y = resample_data(data_x, data_y, seed)
        if (data_x is None): return res
        m.fit(data_x, data_y)
        p, r, f1, _ = precision_recall_fscore_support(val_y, m.predict(val_x), average='binary')
        res += [p, r, f1]
    res /= cv
    return res

def get_models_score(X_train, y_train, X_test, y_test, t="1",test_size=0.2, seed=42, n_jobs=None):

    dicty = Counter(y_train)
    logger.info(str(t)+f" 源数据类别 0: {dicty.get(0)}, 1: {dicty.get(1)}")

    x, y = resample_data(X_train, y_train, seed)
    if (x is None): return 

    # xgboost
    xgbc_model=XGBClassifier()
    # 随机森林
    rfc_model=RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
    # ET
    et_model=ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs)
    # 朴素贝叶斯
    gnb_model=GaussianNB()
    #K最近邻
    knn_model=KNeighborsClassifier(n_jobs=n_jobs)
    #逻辑回归
    lr_model=LogisticRegression(random_state=seed, n_jobs=n_jobs)
    #决策树
    dt_model=DecisionTreeClassifier(random_state=seed)
    #支持向量机
    svc_model=SVC(random_state=seed)


    # xgboost
    xgbc_model.fit(x,y)
    # 随机森林
    rfc_model.fit(x,y)
    # ET
    et_model.fit(x,y)
    # 朴素贝叶斯
    gnb_model.fit(x,y)
    # K最近邻
    knn_model.fit(x,y)
    # 逻辑回归
    lr_model.fit(x,y)
    # 决策树
    dt_model.fit(x,y)
    # 支持向量机
    svc_model.fit(x,y)
    models = {'xgboost': xgbc_model,
        'randomforest': rfc_model,
        'ET': et_model,
        'gnb': gnb_model,
        'knn': knn_model,
        'lr': lr_model,
        'dt': dt_model,
        'svc': svc_model}
    scores = {}
    for k, m in models.items():
        p, r, f1, _ = precision_recall_fscore_support(y_test, m.predict(X_test), average='binary')
        scores[k] = [p, r, f1]
        logger.info(f"{k}:\tp: {p}, r: {r}, f1: {f1}")
        joblib.dump(m, f"{save_path}/{t}_{k}.pkl")
    return scores

#In[4]

used_labels = ['1', '2', '3', '13', '15', '19', '20', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '34', '35', '36', '37', '38', '39', '40', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '56', '57', '59', '60', '62', '63', '66', '67', '79', '80', '81', '82', '83', '84', '85', '86', '88']

all_scores = {}
for t in used_labels:
    x = Xtrain
    y = ytrain[t]
    try:
        all_scores[t] = get_models_score(x, y, Xtest, ytest[t], "p"+t, seed=seed)
    except:
        all_scores[t] = None

df = pd.DataFrame(all_scores)
df = df.transpose()

df.to_csv(save_path+"/scores_0411data.csv")

from get_all_f1_roc_acc import get_all_f1_roc_acc

get_all_f1_roc_acc(Xtest, ytest, save_path, save_path+'/f1_roc_acc.csv')
