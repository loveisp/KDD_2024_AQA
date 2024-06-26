# 对预训练模型做微调

这次比赛使用 tevatron 对 SFR-Embedding-Mistral 做了微调

最好使用当前目录下的 tevatron，执行：cd ./tevatron && pip install -e . 进行安装

不建议 clone 最新的 tevatron

sfr_finetuned 目录已经存放了我微调之后的 lora，可以用这个 lora 去和 SFR-Embedding-Mistral （预训练模型，可从 huggingface 下载）合并

也可以直接从百度网盘下载我合并好的模型：
- 链接：https://pan.baidu.com/s/1LdKujPgDGkrFXrB6MFf8ZA?pwd=gf1j 
- 提取码：gf1j 

## 生成训练数据集

如果 ../embeds/sfr.npy 不存在，则需要先执行 cd ../infer_with_pretrained && python infer_sfr.py

同样，如果 ../embeds/sfr_train.npy 不存在，则需要先执行 cd ../infer_with_pretrained && python infer_sfr_df.py train

执行 python create_ds.py ，生成 tevatron 训练所需的数据集，生成的数据集文件会存到 ./ds/train.jsonl

## 微调

执行 bash train.sh

这里 localhost:0,1,2,3 需要根据本地 gpu 的数量进行修改

model_name_or_path 后的路径需要改为 SFR-Embedding-Mistral 所在的路径

微调后的 lora 会存储到 sfr_finetuned 目录下，我已经在该目录下放了之前训练好的文件

## 合并模型

合并之后，推断开销会小一些，所以最好还是合并

python merge_model.py ，合并后的模型会存到 ./sfr_merged/

执行以下命令，从本地模型路径复制一些文件过去（具体路径需要做修改）：
- cp /root/workspace/dataset/hf_data/models/Salesforce/SFR-Embedding-Mistral/tokenizer* ./sfr_merged/
- cp /root/workspace/dataset/hf_data/models/Salesforce/SFR-Embedding-Mistral/special_tokens_map.json ./sfr_merged/

## 用合并后的模型推断

和 infer_with_pretrained 里做的推断一样，这里也是用 ./sfr_merged/ 里的模型进行推断，得到 embedding

执行以下代码：
- python infer_sfrm.py ，可生成第一阶段的 passages 的 embedding ，生成后存储到 ../embeds/sfrm.npy
- python infer_sfrm_extra.py ，可生成第二阶段不包含在第一阶段的 passages 的 embedding，生成后存储到 ../embeds/sfrm_extra.npy
- python infer_sfrm_df.py final ，可生成第二阶段的 queries 的 embedding ，生成后存储到 ../embeds/sfrm_final.npy