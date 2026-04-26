import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig



def get_modelo():
    print("Loading model...")
    modelo_id = 'zai-org/GLM-OCR'
    return modelo_id

def get_processor(modelo):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(modelo, trust_remote_code=True)
    
    return processor

def carrega_modelo(modelo_id):
    print("Loading model...")
    config_4bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    modelo = AutoModelForImageTextToText.from_pretrained(
        modelo_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        quantization_config=config_4bit
    )
    
    
    return modelo




    