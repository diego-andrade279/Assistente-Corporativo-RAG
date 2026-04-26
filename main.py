import os

from models.test_models import extrair_text_doc
from teste_models.llms import criar_modelo_ollama
from DB.BD_BANCO_VETORES import criar_banco_memoria
import fitz



def main():
    while True:
        print("Iniciando processo de teste do modelo...")
        print("1. Converte PDF para imagens ")
        print("2. Carregando modelo Ollama...")
        
        op = int(input("Digite a opção (1-2): "))
        if op == 1:
            print("Iniciando conversão do PDF para imagens...")
            # Lógica para converter PDF para imagens
            pdf_path = rf'{input("Digite o caminho do PDF: ")}'  # Substitua pelo caminho do seu PDF
            doc = fitz.open(pdf_path)
            for i, pagina in enumerate(doc):
                imagem = pagina.get_pixmap()
                imagem.save(rf'C:\Users\iTech iFood\Desktop\IA_PROCESSO_INTERNOS\DOC\pagina_{i + 1}.png')
            print("Iniciando  teste de carregamento do modelo...")
            doc = os.listdir(r'C:\Users\iTech iFood\Desktop\IA_PROCESSO_INTERNOS\DOC')
            for i in doc:
                text_extraido = extrair_text_doc(rf'C:\Users\iTech iFood\Desktop\IA_PROCESSO_INTERNOS\DOC\{i}')
                criar_banco_memoria(texto=text_extraido)
                
        elif op == 2:
            print("Iniciando teste de carregamento do modelo Ollama...")
            modelo_ollama = criar_modelo_ollama()
            
        elif op != 0 and op != 1 and op != 2 and op != 3:
            print("Opção inválida. Por favor, digite um número entre 0 e 3.")

        
        print("")

if __name__ == "__main__":
    main()
