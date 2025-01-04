"""
    输入 csv 原始文本数据, 预训练模型
    输出 bert 模型
"""
# step1 超参数配置

from datasets import load_dataset
import torch
pretrained_model_path = "bert-base-chinese"
save_path = "models/bert"
batch_size = 8
metric_name = "f1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("csv", data_files={"train": "data/all_train_0411_clean.csv", "test": "data/all_test_0411_clean.csv"})

labels = [label for label in dataset['train'].features.keys() if label not in ['text']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


# ## step2: 数据预处理
# 使用tokenizer分词, 并对label进行处理

from transformers import BertTokenizer
import numpy as np

def get_encoded_dataset(tokenizer):
    def preprocess_data(examples):
    # take a batch of texts
        text = examples["text"]
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=256)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        return encoding

    encoded_dataset = dataset.map(
        preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format("torch", device=device)
    return encoded_dataset

# ## step3: 构建模型
# 
# 在bert后加上一个全连接层, 用于多标签分类

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

def Bert_train(model_path, save_path, device=device, batch_size=8, epoch=20):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    encoded_dataset = get_encoded_dataset(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                problem_type="multi_label_classification", 
                                num_labels=len(labels),
                                id2label=id2label,
                                label2id=label2id)
    model = model.to(device)
    args = TrainingArguments(
        save_path,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

    trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )
    trainer_results = trainer.train()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__=='__main__':
    Bert_train(pretrained_model_path, save_path, device, batch_size)

