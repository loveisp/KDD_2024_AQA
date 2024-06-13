# KDD Cup 2024 OAG-Challenge AQA 3rd place solution

## Prerequisites

- Python 3.9
- PyTorch 2.1.2+cu118
- transformers 4.40.1
- faiss-gpu 1.7.2
- deepspeed 0.14.0
- einops 0.7.0
- gritlm 1.0.0
- pandas 2.1.4

第一阶段的数据集放入 ./AQA/ ，第二阶段的数据集放入 ./AQA/AQA-test-public/

## 方案

方案分为三个部分：
- 使用 4 个预训练模型进行推断，这部分的详细描述可参见 [infer_with_pretrained](https://github.com/loveisp/KDD_2024_AQA/tree/main/infer_with_pretrained)
- 对模型进行微调和推断，这部分的详细描述可参见 [finetune](https://github.com/loveisp/KDD_2024_AQA/tree/main/finetune)
- 对之前推断的结果，用 faiss 计算相似度矩阵，并进行加权融合，这部分的详细描述可参见 [predict](https://github.com/loveisp/KDD_2024_AQA/tree/main/predict)

依次执行以上三个部分的代码，即可得到最终的预测结果

有些代码是从 ipynb 中转移出来的，没有经过 py 代码的详尽测试，执行代码时可能会出错，如果有问题请及时反馈，谢谢！