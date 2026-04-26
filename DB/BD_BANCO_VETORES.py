from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def criar_banco_memoria(**args):
    
    print("carregando texto extraido do doc")
    texto = args.get("texto", "")
    
    # fatiamento de texto
    # O chunk_size fatiador corta o texto em pedaços de 300 letras. 
    # O chunk_overlap de 50 letras garante que não vamos cortar uma frase no meio e perder o sentido.
    print("fatiando texto")
    texto_cortado = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    path_text = texto_cortado.split_text(texto)
    
    # criação de vetores
    print("Usando Embeddings para criar vetores")
    # Este modelo transforma frases em português em coordenadas numéricas.
    # Ele é pequeno, rápido e vai rodar direto no seu notebook.
    modelo_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Criando banco de vetores com Chroma")
    # O Chroma é tipo um banco de dados que guarda os vetores e permite fazer buscas rápidas.
    # Pega os blocos, converte em números e salva em uma pasta chamada "meu_banco_chroma"
    banco_vetores = Chroma.from_texts(texts=path_text, embedding=modelo_embeddings, persist_directory="./meu_banco_chroma")
    
    print("Banco de vetores criado e salvo em ./meu_banco_chroma")
    print("="*50)
    '''
        pergunta = input(' Faça uma pergunta: ')
        
        print("Buscando resposta no banco de vetores...")
        print(f"pergunta: {pergunta}")
        # Aqui a gente faz uma busca no banco de vetores usando a pergunta do usuário.
        resultado = banco_vetores.similarity_search(pergunta, k=1)
        
        for doc in resultado:
            print("\nTrecho encontrado:")
            print(doc.page_content)
            print("-"*30)
    '''    
    return banco_vetores
