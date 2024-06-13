# 预测

经过预训练模型的推断和微调模型的推断，现在 ../embeds 目录下，应该至少包含以下文件：
- {model_name}.npy
- {model_name}_extra.npy
- {model_name}_final.npy

这里 {model_name} 是 sfrm, nvembed, sfr, gritlm, linq

## 合并 embedding

之前为了节省推断时间，第二阶段的 passages 只是对和第一阶段相比多出来的那部分进行推断，现在需要把两部分合并起来，成为完整的所有 passages 的 embedding 文件

执行如下代码：
- python merge_embeddings.py sfrm ，将两个阶段的 passages 的 embedding 合起来，存储到 ../embeds/nvembed_all.npy ，下同
- python merge_embeddings.py nvembed
- python merge_embeddings.py sfr
- python merge_embeddings.py gritlm
- python merge_embeddings.py linq

也可以直接执行：bash merge_embeddings.sh

## 用 faiss 计算 D 和 I

用每个模型的 {model_name}_final.npy 作为 queries ， {model_name}_all.npy 作为 database ，进行 search 计算，得到每个模型的 D 和 I 矩阵

执行如下代码：
- python comp_faiss.py sfrm final ，用 faiss 计算 D 和 I ，存储到 ../faiss_results/D_sfrm_final.npy 和 ../faiss_results/I_sfrm_final.npy ，下同
- python comp_faiss.py nvembed final
- python comp_faiss.py sfr final
- python comp_faiss.py gritlm final
- python comp_faiss.py linq final

也可以直接执行：bash comp_faiss.sh

整个过程需要 20 到 30 分钟，而且需要较多内存

如果内存不够，可以在 comp_faiss.py 中，修改 D, I = faiss_query(xb_fn, xq_fn) 为 D, I = faiss_query(xb_fn, xq_fn, k) ，其中 k 可以取一个较小的整数，比如 1000 ，这样会占用少得多的内存，同时提交结果的分数也会略有下降

## 对 D 做归一化

对上一步用 faiss 计算得到的 D 矩阵进行逐行归一化

这么做的目的，是因为每个模型的单一 query 对所有 passages 计算的相似度分数的分布差异较大，有的分布较宽，而有的较窄，所以需要做逐行归一化，即每个 query 做归一化，这样后面做加权融合时效果会更好

执行如下代码：
- python norm_faiss.py sfrm final ，对 D 矩阵进行逐行归一化，存储到 ../faiss_results/D_sfrm_final_normed.npy ，下同
- python norm_faiss.py nvembed final
- python norm_faiss.py sfr final
- python norm_faiss.py gritlm final
- python norm_faiss.py linq final

也可以直接执行：bash norm_faiss.sh

整个过程可能需要 10 到 30 分钟，取决于硬盘速度和 cpu 速度等因素

## 加权融合，得到提交文件


