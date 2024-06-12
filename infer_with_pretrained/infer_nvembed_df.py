import json
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel
from functools import partial
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("data_type", type=str, help="train, valid, test, final")
args = parser.parse_args()


def get_model(model_path, device):
    model = AutoModel.from_pretrained(model_path, device_map=device, torch_dtype='auto', trust_remote_code=True)
    return model


class QuestionDataset(Dataset):
    """Passage dataset."""

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __get_question(self, title, body):
        title = title.strip()
        body = body.strip()
        question = f"{title}\n{body}"
        return question

    def __getitem__(self, idx):
        title = self.df.question[idx].strip()
        body = self.df.body[idx].strip()
        return self.__get_question(title, body)
    

print('Reading data...')
d_filename = {
    'train': '../AQA/qa_train.txt', 
    'valid': '../AQA/qa_valid_wo_ans.txt', 
    'test': '../AQA/qa_test_wo_ans.txt', 
    'final': '../AQA/AQA-test-public/qa_test_wo_ans_new.txt'
}
df = pd.read_json(d_filename[args.data_type], lines=True)
print('{} questions are read in total.'.format(len(df)))

print('Loading model...')
model_path = '/root/data/hf_data/models/nvidia/NV-Embed-v1'
device = 'cuda'
model = get_model(model_path, device)
model_name = 'nvembed'
print('{} is loaded.'.format(model_name))

bs = 8
fea_dim = 4096
num_workers = 0
ds = QuestionDataset(df)
dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
embeddings = np.zeros((len(ds), fea_dim), dtype=np.float16)

print('Infering...')
max_length = 4096
task_name_to_instruct = {"example": "Given a web search query, retrieve relevant passages that answer the query",}
passage_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
for batch_idx, batch in enumerate(tqdm(dl)):
    embeddings[(batch_idx*bs):((batch_idx+1)*bs)] = model.encode(batch, instruction=passage_prefix, max_length=max_length).cpu().numpy().astype(np.float16)
    
print('Writing results...')
save_path = Path('../embeds')
save_path.mkdir(parents=True, exist_ok=True)
np.save(save_path / '{}_{}.npy'.format(model_name, args.data_type), embeddings)