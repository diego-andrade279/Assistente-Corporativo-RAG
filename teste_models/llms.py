from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from DB.BD_BANCO_VETORES import criar_banco_memoria
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic import PromptTemplate

def criar_modelo_ollama():
    print("Carregando modelo Ollama...")
    tradutor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    banco = Chroma(persist_directory="./meu_banco_chroma", embedding_function=tradutor_embeddings)
    modelo_ollama = Ollama(model="qwen3:32b")
    
    print("4. Configurando a Personalidade da IA...")
    # Aqui nós mudamos a regra do jogo. Damos liberdade para ele generalizar e dar comandos!
    template_especialista = """Você é um Engenheiro de TI especialista ajudando um usuário da empresa.
        Use o Contexto abaixo como base para a sua resposta.
        Como você é um especialista, você tem liberdade para expandir a resposta usando seu próprio conhecimento para deixar a explicação mais rica.
        Se a solução envolver o Windows, sempre forneça o caminho exato dos menus (ex: Iniciar > Configurações > Sistema) ou os comandos de terminal (CMD/PowerShell) necessários para resolver o problema.
        Dê uma resposta longa, detalhada e com o passo a passo em tópicos.

        Contexto da Empresa:
        {context}

        Pergunta do Usuário: {question}

        Resposta do Especialista de TI:"""
    # tradutor para o langchain entender o template
    prompt = PromptTemplate(template=template_especialista, input_variables=["context", "question"])
    print("Modelo Ollama carregado com sucesso!")
    
    sistema_rag = RetrievalQA.from_chain_type(
        llm=modelo_ollama,
        chain_type="stuff", # "stuff" significa "enfie todo o texto encontrado no prompt do LLM"
        retriever=banco.as_retriever(search_kwargs={"k": 10} ), # contexto para o modelo, ou seja, os 10 blocos mais relevantes encontrados no banco de vetores
        chain_type_kwargs={"prompt": prompt}
    ) 
    
    #print("Sistema RAG criado com sucesso!")
    #pergunta_usuario = input("Faça uma pergunta: ")
    #resposta = sistema_rag.invoke(pergunta_usuario)
    #print("Resposta: ", resposta)
    
    return modelo_ollama