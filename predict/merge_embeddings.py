import numpy as np
import json
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="model name")
args = parser.parse_args()

model_name = args.model_name

with open('../id2idx.json', 'r') as f:
    id2idx = json.load(f)
    
paper_ids_extra = pd.read_csv('../final_extra_ids.csv', header=None)[0].tolist()

with open('../AQA/pid_to_title_abs_new.json', 'r') as f:
    d_json_old = json.load(f)
paper_ids_old = list(d_json_old.keys())

embeds_old = np.load('../embeds/{}.npy'.format(model_name))
embeds_extra = np.load('../embeds/{}_extra.npy'.format(model_name))
embeds = np.zeros((len(id2idx), embeds_old.shape[1]), dtype=embeds_old.dtype)

paper_idx_old = [id2idx[id_] for id_ in paper_ids_old]
embeds[paper_idx_old] = embeds_old

paper_idx_extra = [id2idx[id_] for id_ in paper_ids_extra]
embeds[paper_idx_extra] = embeds_extra

np.save('../embeds/{}_all.npy'.format(model_name), embeds)