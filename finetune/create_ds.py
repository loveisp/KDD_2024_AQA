import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm
import math
from pathlib import Path


def faiss_query(xb_fn, xq_fn, k=None):
    xb = np.load(xb_fn).astype(np.float32)
    faiss.normalize_L2(xb)
    _, d = xb.shape
    index = faiss.IndexFlatIP(d)   # build the index
    #print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    xq = np.load(xq_fn).astype(np.float32)
    faiss.normalize_L2(xq)
    if k is not None:
        D, I = index.search(xq, k)
    else:
        D, I = index.search(xq, xb.shape[0])
    return D, I


xb_fn = '../embeds/sfr.npy'.format(model_name)
xq_fn = '../embeds/sfr_train.npy'.format(model_name, data_type)
D, I = faiss_query(xb_fn, xq_fn, 400)


with open('../AQA/pid_to_title_abs_new.json', 'r') as f:
    d_json = json.load(f)
paper_ids = np.array(list(d_json.keys()))


def format_query(title, body):
    task_description = 'Given a question including title and body, retrieve relevant papers that answer the question'
    query = f"<question_title> {title.strip()} </question_title>\n<question_body> {body.strip()} </question_body>"
    return f'Instruct: {task_description}\nQuery: {query}'


def format_passage(v):
    title = v['title'].strip() if v['title'] is not None else v['title']
    abstract = v['abstract'].strip()
    if title is None:
        passage = f"<abstract> {abstract} </abstract>"
    else:
        passage = f"<title> {title} </title>\n<abstract> {abstract} </abstract>"
    return passage


data = []
df = pd.read_json('../AQA/qa_train.txt', lines=True)
for i in tqdm(range(len(df))):
    row = df.iloc[i]
    # 如果不做np.unique(row.pids)，则得到的indices可能会越界，因为这里assume_unique=True
    _, indices, _ = np.intersect1d(paper_ids, np.unique(row.pids), assume_unique=True, return_indices=True)
    pos_indices, indices_for_reindex, _ = np.intersect1d(I[i], indices, assume_unique=True, return_indices=True)
    pos_indices = pos_indices[np.argsort(indices_for_reindex)][::-1]
    # 按相似度由小到大排序，而且只取前20个（最值得去学习的20个，本来相似度就高其实就不用训练了，而且提交结果只看20个）
    pos_indices = np.concatenate([np.setdiff1d(indices, pos_indices, assume_unique=True), pos_indices])[:20]
    neg_indices = np.setdiff1d(I[i], indices, assume_unique=True)  # 加了assume_unique以后这个就是保持原顺序的，否则会重新按大小排序
    for j, pos_index in enumerate(pos_indices):
        query_id = '{}_{}'.format(i, j)
        query = format_query(row.question, row.body)
        positive_passages = [{'docid': str(pos_index), 'title': '', 'text': format_passage(d_json[paper_ids[pos_index]])}]
        negative_passages = [{'docid': str(neg_index), 'title': '', 'text': format_passage(d_json[paper_ids[neg_index]])} \
                            for neg_index in neg_indices[j::len(pos_indices)][:15]]
        d_query = {
            'query_id': query_id, 
            'query': query, 
            'positive_passages': positive_passages, 
            'negative_passages': negative_passages, 
        }
        data.append(d_query)

Path('./ds').mkdir(parents=True, exist_ok=True)
with open('./ds/train.jsonl', 'w') as f:
    for d in tqdm(data):
        f.write(json.dumps(d) + "\n")