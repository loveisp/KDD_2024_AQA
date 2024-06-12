import json
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F
from torch import Tensor
from gritlm import GritLM
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path


DEBUG = False


# 为了测试显存占用，生成一些长样本
def generate_long_passages(length, n):
    return [' '.join(['apple']*length) for _ in range(n)]


def get_passage(v):
    title = v['title'].strip() if v['title'] is not None else v['title']
    abstract = v['abstract'].strip()
    if title is None:
        passage = f"<abstract> {abstract} </abstract>"
    else:
        passage = f"<title> {title} </title>\n<abstract> {abstract} </abstract>"
    return passage


def get_passages(json_path):
    with open(json_path, 'r') as f:
        d_json = json.load(f)
    passage_ids = pd.read_csv('../final_extra_ids.csv', header=None)[0].tolist()
    passages = [get_passage(d_json[passage_id]) for passage_id in passage_ids]
    return passage_ids, passages


def get_model(model_path, device):
    model = GritLM(model_path, torch_dtype="auto", mode="embedding", device_map=device)
    return model


class PassageDataset(Dataset):
    """Passage dataset."""

    def __init__(self, passages):
        """
        Arguments:
            passages
        """
        self.passages = passages

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        return self.passages[idx]
    
    
def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    
    
print('Reading data...')
if DEBUG:
    n = 1000
    length = 10000
    passages = generate_long_passages(length, n)
else:
    json_path = '../AQA/AQA-test-public/pid_to_title_abs_update_filter.json'
    _, passages = get_passages(json_path)
print('{} passages are read in total.'.format(len(passages)))

print('Loading model...')
model_path = '/root/workspace/dataset/hf_data/models/GritLM/GritLM-7B'
device = 'cuda'
model = get_model(model_path, device)
model.device = model.model.device
model_name = 'gritlm'
print('{} is loaded.'.format(model_name))

bs = 4
fea_dim = 4096
num_workers = 0
ds = PassageDataset(passages)
dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
embeddings = np.zeros((len(ds), fea_dim), dtype=np.float32)

print('Infering...')
max_length = 4096
instruction = gritlm_instruction("")
for batch_idx, batch in enumerate(tqdm(dl)):
    embeddings[(batch_idx*bs):((batch_idx+1)*bs)] = model.encode(batch, instruction=instruction, max_length=max_length)
    
print('Writing results...')
save_path = Path('../embeds')
save_path.mkdir(parents=True, exist_ok=True)
np.save(save_path / '{}_extra.npy'.format(model_name), embeddings)