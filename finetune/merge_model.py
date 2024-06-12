from transformers import PreTrainedModel, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


model_path = "/root/workspace/dataset/hf_data/models/Salesforce/SFR-Embedding-Mistral/"
base_model = AutoModel.from_pretrained(model_path, torch_dtype='auto', device_map='cuda')

lora_name_or_path = './sfr_finetuned/'
lora_config = LoraConfig.from_pretrained(lora_name_or_path)
lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained('./sfr_merged/')