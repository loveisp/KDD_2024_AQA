import json
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
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
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, device_map=device, torch_dtype='auto', trust_remote_code=True)
    return tokenizer, model


class QuestionDataset(Dataset):
    """Passage dataset."""

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __get_question(self, title, body):
        title = title.strip()
        body = body.strip()
        task_description = 'Given a question including title and body, retrieve relevant papers that answer the question'
        query = f"<question_title> {title} </question_title>\n<question_body> {body} </question_body>"
        return f'Instruct: {task_description}\nQuery: {query}'

    def __getitem__(self, idx):
        title = self.df.question[idx].strip()
        body = self.df.body[idx].strip()
        return self.__get_question(title, body)
    
    
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
d_filename = {
    'train': '../AQA/qa_train.txt', 
    'valid': '../AQA/qa_valid_wo_ans.txt', 
    'test': '../AQA/qa_test_wo_ans.txt', 
    'final': '../AQA/AQA-test-public/qa_test_wo_ans_new.txt', 
}
df = pd.read_json(d_filename[args.data_type], lines=True)
print('{} questions are read in total.'.format(len(df)))

print('Loading model...')
model_path = './sfr_merged/'
device = 'cuda'
tokenizer, model = get_model(model_path, device)
model_name = 'sfrm'
print('{} is loaded.'.format(model_name))

bs = 8
max_length = 4096
fea_dim = 4096
ds = QuestionDataset(df)
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
np.save(save_path / '{}_{}.npy'.format(model_name, args.data_type), embeddings)