import numpy as np
import faiss
from tqdm.auto import tqdm
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="model name")
parser.add_argument("data_type", type=str, help="train, valid, test, final")
args = parser.parse_args()

model_name = args.model_name
data_type = args.data_type


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


xb_fn = '../embeds/{}_all.npy'.format(model_name)
xq_fn = '../embeds/{}_{}.npy'.format(model_name, data_type)
D, I = faiss_query(xb_fn, xq_fn)
Path('../faiss_results').mkdir(parents=True, exist_ok=True)
np.save('../faiss_results/D_{}_{}.npy'.format(model_name, data_type), D)
np.save('../faiss_results/I_{}_{}.npy'.format(model_name, data_type), I.astype(np.uint32))