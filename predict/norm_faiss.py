import numpy as np
from tqdm.auto import tqdm
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="model name")
parser.add_argument("data_type", type=str, help="train, valid, test, final")
args = parser.parse_args()

model_name = args.model_name
data_type = args.data_type


D = np.load('../faiss_results/D_{}_{}.npy'.format(model_name, data_type))
mean = D.mean(1)
std = D.std(1)
normed = (D - mean[:, None]) / std[:, None]
np.save('../faiss_results/D_{}_{}_normed.npy'.format(model_name, data_type), normed)