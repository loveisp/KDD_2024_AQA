for model_name in sfrm nvembed sfr gritlm linq
do
    python comp_faiss.py $model_name final
done