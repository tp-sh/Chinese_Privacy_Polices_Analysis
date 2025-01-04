
# In[1]:
"""
    输入 bert 微调模型路径
    输出 bert embedding
"""
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from transformers import BertModel
from tqdm import tqdm

model_path = "models/bert-checkpoint-95500"
device = "cuda" if torch.cuda.is_available() else "cpu"

def read_csv(file_path):
    # 返回 text, label, max_len
    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    df.fillna("",inplace=True)
    # df.drop(index=(df.loc[(df['pa_text']==0)].index))
    print("# of rows in data = {}".format(df.shape[0]))
    print("# of columns in data = {}".format(df.shape[1]))
    data = df
    data_X = data['text']
    data_y =  data.drop(labels=list(set(df.columns)&set(["filename", "rownum", "label", "text", "pa_text","parents","siblings"])), axis=1)
    return data_X.to_list(), data_y.to_numpy(), 256 # data_y.to_numpy(), 256

class MyDataset(Dataset):
    def __init__(self, texts, labels, max_length, model_path=model_path):
        self.all_text = texts
        self.all_label = labels
        self.max_len = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def __getitem__(self, index):
        # 取出一条数据并截断长度
        text = self.all_text[index]
        label = self.all_label[index]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
            )
        token_ids = encoding['input_ids'].flatten()
        mask = encoding['attention_mask'].flatten()
        return (token_ids, mask), label

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)

# In[3]:
def get_bert_embedding(file_path, save_path, model_path=model_path, device=device):
    model = BertModel.from_pretrained(model_path).to(device)
    all_text, all_label, max_len = read_csv(file_path)
    allDataset = MyDataset(all_text, all_label, max_len, model_path)
    allDataloader = DataLoader(allDataset, batch_size=4, shuffle=False)

    bert_embeddings = []
    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(allDataloader):
            input_ids, attention_mask = x[0].to(device), x[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                                    output_hidden_states=True)
            embedding = outputs.pooler_output
            bert_embeddings.append(embedding)

    bert_embeddings = torch.stack(bert_embeddings).reshape([-1, 768]).to('cpu').numpy()
    torch.save(bert_embeddings, save_path)
    
# %%
if __name__=="__main__":
    get_bert_embedding("datasets/all_train_0411_clean.csv", "classifier_data/bert_embeddings_train_0411data.pt")
    get_bert_embedding("datasets/all_test_0411_clean.csv", "classifier_data/bert_embeddings_test_0411data.pt")
