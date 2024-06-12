import json
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from functools import partial
from tqdm.auto import tqdm
import numpy as np
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
    if (title is None) or (not title):
        passage = f"{abstract}"
    else:
        passage = f"{title}\n{abstract}"
    return passage


def get_passages(json_path):
    with open(json_path, 'r') as f:
        d_json = json.load(f)
    passage_ids = list(d_json.keys())
    passages = [get_passage(d_json[passage_id]) for passage_id in passage_ids]
    return passage_ids, passages


def get_model(model_path):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, device_map='auto', torch_dtype='auto', trust_remote_code=True)
    return tokenizer, model


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
    
    
def collate_fn(batches, tokenizer, max_length):
    return tokenizer(batches, max_length=max_length, padding=True, truncation=True, return_tensors="pt")


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    

print('Reading data...')
if DEBUG:
    n = 1000
    length = 10000
    passages = generate_long_passages(length, n)
else:
    json_path = '../AQA/pid_to_title_abs_new.json'
    _, passages = get_passages(json_path)
print('{} passages are read in total.'.format(len(passages)))

print('Loading model...')
model_path = '/home/ubuntu/data/huggingface/models/Linq-AI-Research/Linq-Embed-Mistral'
tokenizer, model = get_model(model_path)
device = model.device
model_name = 'linq'
print('{} is loaded.'.format(model_name))

bs = 8
max_length = 4096
fea_dim = 4096
ds = PassageDataset(passages)
dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=max_length))
embeddings = np.zeros((len(ds), fea_dim), dtype=np.float16)

print('Infering...')
for batch_idx, batch_dict in enumerate(tqdm(dl)):
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings[(batch_idx*bs):((batch_idx+1)*bs)] = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().numpy()
    
print('Writing results...')
save_path = Path('../embeds')
save_path.mkdir(parents=True, exist_ok=True)
np.save(save_path / '{}.npy'.format(model_name), embeddings)