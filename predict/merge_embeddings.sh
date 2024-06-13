for model_name in sfrm nvembed sfr gritlm linq
do
    python merge_embeddings.py $model_name
done