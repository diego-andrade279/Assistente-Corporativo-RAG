import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from models.get_models import get_modelo, get_processor, carrega_modelo


print("iNICIANDO TESTE DE MODEL  OCR GLM")  
def extrair_text_doc(caminho_imagem):
    modelo_id = get_modelo()
    processador = get_processor(modelo_id)
    modelo = carrega_modelo(modelo_id)
    
    # carregando doc real
    try:
        imagem = Image.open(caminho_imagem)
    except Exception as e:
        print(f'falha ao processar doc {e}')
        return
    
    mensagem = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Analise esta imagem com atenção e descreva ou estraia os dados que conseguir identificar nela, seja o que for, tente extrair o máximo de informações possíveis, seja detalhado e minucioso."}
            ]
        }
    ]    
    
    text_processado = processador.apply_chat_template(mensagem, tokenize=False, add_generation_prompt=True)
    inputs = processador(images=imagem, text=text_processado, return_tensors="pt").to("cuda")
    
    print("LENDO IMAGEM")
    with torch.no_grad():
        saida = modelo.generate(**inputs, max_new_tokens=7000, num_beams=5, early_stopping=True)
    
    resultdo = processador.decode(saida[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    
    print("\n")

    return resultdo
    
    
