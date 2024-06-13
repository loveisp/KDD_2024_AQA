import numpy as np
from tqdm.auto import tqdm
import json

model_names = [
    'sfrm',
    'nvembed',
    'sfr',
    'gritlm',
    'linq',
]

D_normed = np.zeros((3000, 466387, 5), dtype=np.float32)
for model_idx, model_name in enumerate(tqdm(model_names)):
    I = np.load('../faiss_results/I_{}_final.npy'.format(model_name))
    D = np.load('../faiss_results/D_{}_final_normed.npy'.format(model_name))
    for i in tqdm(range(D.shape[0])):
        D_normed[i, I[i], model_idx] = D[i]
        
np.save('../faiss_results/D_final.npy', D_normed)