# 对已有的机器学习模型，计算测试集指标


import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import joblib
import pandas as pd
import os

def multi_label_metrics(y_pred, y_true, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    # sigmoid = torch.nn.Sigmoid()
    # probs = sigmoid(torch.Tensor(predictions))
    # # next, use threshold to turn them into integer predictions
    # y_pred = np.zeros(probs.shape)
    # y_pred[np.where(probs >= threshold)] = 1
    # # finally, compute metrics
    # y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

used_labels = ['1', '2', '3', '13', '15', '19', '20', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '34', '35', '36', '37', '38', '39', '40', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '56', '57', '59', '60', '62', '63', '66', '67', '79', '80', '81', '82', '83', '84', '85', '86', '88']
models_name = {'xgboost': 1,
    'randomforest': 1,
    'ET': 1,
    'gnb': 1,
    'knn': 1,
    'lr': 1,
    'dt': 1,
    'svc': 1}
model_path = "/home/user/zxh/classifier/compare_models_0406_cleanpunc"
Xtest = torch.load("/home/user/zxh/classifier_data/tfidf_embeddings_test_0411datav2.pt")
ytest = pd.read_csv("/home/user/zxh/classifier_data/test_0411_labels.csv").to_numpy()

def get_models(model_name, model_path):
    models = []
    for i in used_labels:
        models.append(os.path.join(model_path, f'p{i}_{model_name}.pkl'))
    return models

def get_metric(models, X, Y, model_name):
    Y_preds = []
    for m in models:
        m = joblib.load(m)
        Y_preds.append(m.predict(X))
    metric = multi_label_metrics(np.stack(Y_preds, axis=1), Y)
    metric['name'] = model_name
    return metric

def get_all_f1_roc_acc(Xtest=Xtest, ytest=ytest, model_path=model_path, save_path="tfidf_origin_embedding_final_results.csv"):
    metrics = []
    for k in models_name:
        models = get_models(k, model_path)
        metrics.append(get_metric(models, Xtest, ytest, k))
    df = pd.DataFrame(metrics)
    df.to_csv(save_path)

