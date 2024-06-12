# 用预训练模型做推断

## NV-Embed-v1

该预训练模型来自：https://huggingface.co/nvidia/NV-Embed-v1

依次执行以下代码：

- python infer_nvembed.py ，可生成第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/nvembed.npy

- python infer_nvembed_extra.py ，可生成第二阶段的 passages 的 embedding （第一阶段之外的部分，为了节省时间，下同），生成后存储到 ../embeds/nvembed_extra.npy

- python infer_nvembed_df.py final ，可生成第二阶段的 queries 的 embedding ，生成后存储到 ../embeds/nvembed_final.npy

以上三个 py 文件可能需要修改 model_path 变量：如果没有下载到本地，将该变量改为 nvidia/NV-Embed-v1 即可，执行代码时会自动从 huggingface 下载；如果已下载到本地，则需要修改为本地模型对应的路径

- python merge_embeddings.py nvembed ，可将两个阶段的 passages 的 embedding 合起来，存储到 ../embeds/nvembed_all.npy

- python comp_faiss.py nvembed final ，用 faiss 计算 D 和 I ，存储到 ../faiss_results/D_nvembed_final.npy 和 ../faiss_results/I_nvembed_final.npy

- python norm_faiss.py nvembed final ，对 faiss 的 D 矩阵进行逐行归一化，存储到 ../faiss_results/D_nvembed_final_normed.npy

## SFR-Embedding-Mistral

该预训练模型来自：https://huggingface.co/Salesforce/SFR-Embedding-Mistral

依次执行以下代码：

- python infer_sfr.py ，可生成第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/sfr.npy

- python infer_sfr_extra.py ，可生成第二阶段的 passages 的 embedding ，生成后存储到 ../embeds/sfr_extra.npy

- python infer_sfr_df.py final ，可生成第二阶段的 queries 的 embedding ，生成后存储到 ../embeds/sfr_final.npy

以上三个 py 文件可能需要修改 model_path 变量：如果没有下载到本地，将该变量改为 Salesforce/SFR-Embedding-Mistral 即可，执行代码时会自动从 huggingface 下载；如果已下载到本地，则需要修改为本地模型对应的路径

- python merge_embeddings.py sfr ，可将两个阶段的 passages 的 embedding 合起来，存储到 ../embeds/sfr_all.npy

- python comp_faiss.py sfr final ，用 faiss 计算 D 和 I ，存储到 ../faiss_results/D_sfr_final.npy 和 ../faiss_results/I_sfr_final.npy

- python norm_faiss.py sfr final ，对 faiss 的 D 矩阵进行逐行归一化，存储到 ../faiss_results/D_sfr_final_normed.npy

## GritLM-7B

该预训练模型来自：https://huggingface.co/GritLM/GritLM-7B

依次执行以下代码：

- python infer_gritlm.py ，可生成第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/gritlm.npy

- python infer_gritlm_extra.py ，可生成第二阶段的 passages 的 embedding ，生成后存储到 ../embeds/gritlm_extra.npy

- python infer_gritlm_df.py final ，可生成第二阶段的 queries 的 embedding ，生成后存储到 ../embeds/gritlm_final.npy

以上三个 py 文件可能需要修改 model_path 变量：如果没有下载到本地，将该变量改为 GritLM/GritLM-7B 即可，执行代码时会自动从 huggingface 下载；如果已下载到本地，则需要修改为本地模型对应的路径

- python merge_embeddings.py gritlm ，可将两个阶段的 passages 的 embedding 合起来，存储到 ../embeds/gritlm_all.npy

- python comp_faiss.py gritlm final ，用 faiss 计算 D 和 I ，存储到 ../faiss_results/D_gritlm_final.npy 和 ../faiss_results/I_gritlm_final.npy

- python norm_faiss.py gritlm final ，对 faiss 的 D 矩阵进行逐行归一化，存储到 ../faiss_results/D_gritlm_final_normed.npy

## Linq-Embed-Mistral

该预训练模型来自：https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral

依次执行以下代码：

- python infer_linq.py ，可生成第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/linq.npy

- python infer_linq_extra.py ，可生成第二阶段的 passages 的 embedding ，生成后存储到 ../embeds/linq_extra.npy

- python infer_linq_df.py final ，可生成第二阶段的 queries 的 embedding ，生成后存储到 ../embeds/linq_final.npy

以上三个 py 文件可能需要修改 model_path 变量：如果没有下载到本地，将该变量改为 Linq-AI-Research/Linq-Embed-Mistral 即可，执行代码时会自动从 huggingface 下载；如果已下载到本地，则需要修改为本地模型对应的路径

- python merge_embeddings.py linq ，可将两个阶段的 passages 的 embedding 合起来，存储到 ../embeds/linq_all.npy

- python comp_faiss.py linq final ，用 faiss 计算 D 和 I ，存储到 ../faiss_results/D_linq_final.npy 和 ../faiss_results/I_linq_final.npy

- python norm_faiss.py linq final ，对 faiss 的 D 矩阵进行逐行归一化，存储到 ../faiss_results/D_linq_final_normed.npy