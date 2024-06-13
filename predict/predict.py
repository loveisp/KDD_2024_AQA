import numpy as np
from tqdm.auto import tqdm
import json
from pathlib import Path

with open('../AQA/AQA-test-public/pid_to_title_abs_update_filter.json', 'r') as f:
    d_json = json.load(f)
paper_ids = list(d_json.keys())

D = np.load('../faiss_results/D_final.npy')

weights = (0.65, 0.1, 0.05, 0.15, 0.05)
D_weighted = D[..., 0] * weights[0] + D[..., 1] * weights[1] + D[..., 2] * weights[2] + \
    D[..., 3] * weights[3] + D[..., 4] * weights[4]

result = ''
for i in tqdm(range(D_weighted.shape[0])):
    indices = np.argpartition(-D_weighted[i], 20)[:20]
    indices = indices[np.argsort(-D_weighted[i][indices])]
    row = ','.join([paper_ids[i] for i in indices]) + '\n'
    result += row
Path('../results').mkdir(parents=True, exist_ok=True)
with open('../results/result_final.txt', 'w') as f:
    f.write(result)