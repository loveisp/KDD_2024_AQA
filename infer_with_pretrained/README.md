# 用预训练模型做推断

## NV-Embed-v1

该预训练模型来自：https://huggingface.co/nvidia/NV-Embed-v1

执行以下代码：
- python infer_nvembed.py ，可生成第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/nvembed.npy
- python infer_nvembed_extra.py ，可生成第二阶段不包含在第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/nvembed_extra.npy
- python infer_nvembed_df.py final ，可生成第二阶段的 queries 的 embedding ，生成后存储到 ../embeds/nvembed_final.npy

以上三个 py 文件可能需要修改 model_path 变量：
- 如果没有下载到本地，将该变量改为 nvidia/NV-Embed-v1 即可，执行代码时会自动从 huggingface 下载
- 如果已下载到本地，则需要修改为本地模型对应的路径

这里 passages 的 embedding 是第一阶段和第二阶段的额外部分分开推断的，这样做是为了节省推断时间，下面的几个模型也是如此

## SFR-Embedding-Mistral

该预训练模型来自：https://huggingface.co/Salesforce/SFR-Embedding-Mistral

执行以下代码：
- python infer_sfr.py ，可生成第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/sfr.npy
- python infer_sfr_extra.py ，可生成第二阶段不包含在第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/sfr_extra.npy
- python infer_sfr_df.py final ，可生成第二阶段的 queries 的 embedding ，生成后存储到 ../embeds/sfr_final.npy

以上三个 py 文件可能需要修改 model_path 变量：
- 如果没有下载到本地，将该变量改为 Salesforce/SFR-Embedding-Mistral 即可，执行代码时会自动从 huggingface 下载
- 如果已下载到本地，则需要修改为本地模型对应的路径

## GritLM-7B

该预训练模型来自：https://huggingface.co/GritLM/GritLM-7B

执行以下代码：
- python infer_gritlm.py ，可生成第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/gritlm.npy
- python infer_gritlm_extra.py ，可生成第二阶段不包含在第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/gritlm_extra.npy
- python infer_gritlm_df.py final ，可生成第二阶段的 queries 的 embedding ，生成后存储到 ../embeds/gritlm_final.npy

以上三个 py 文件可能需要修改 model_path 变量：
- 如果没有下载到本地，将该变量改为 GritLM/GritLM-7B 即可，执行代码时会自动从 huggingface 下载
- 如果已下载到本地，则需要修改为本地模型对应的路径

## Linq-Embed-Mistral

该预训练模型来自：https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral

执行以下代码：
- python infer_linq.py ，可生成第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/linq.npy
- python infer_linq_extra.py ，可生成第二阶段不包含在第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/linq_extra.npy
- python infer_linq_df.py final ，可生成第二阶段的 queries 的 embedding ，生成后存储到 ../embeds/linq_final.npy

以上三个 py 文件可能需要修改 model_path 变量：
- 如果没有下载到本地，将该变量改为 Linq-AI-Research/Linq-Embed-Mistral 即可，执行代码时会自动从 huggingface 下载
- 如果已下载到本地，则需要修改为本地模型对应的路径