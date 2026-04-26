import streamlit as st
import os
import fitz
from DB.BD_BANCO_VETORES import criar_banco_memoria
from models.test_models import extrair_text_doc
from teste_models.llms import criar_modelo_ollama


#1. configuraçao da site
st.set_page_config(page_title="Teste de MODELOS baseado em RAG", page_icon=":robot_face:", layout="centered")

# Botao de upload de pdf  para o banco de dados
with st.sidebar:
    st.header("📁 Upload de Documentos")
    st.markdown("Faça upload dos manuais, PDFs e fluxogramas da empresa para treinar a IA.")
    
    arquivo_pdf = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])
    if st.button("Iniciar processamento", use_container_width=True):
        if arquivo_pdf is not None:
            with st.spinner("Processando o documento..."):
                # Salvar o arquivo PDF em uma pasta temporária
                pdf_bytes = arquivo_pdf.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                texto_extraido = ""
                
                # Converter PDF para imagens e processar
                for i, pagina in enumerate(doc):
                    st.write(f"Processando página {i + 1}...")
                    imagem = pagina.get_pixmap()
                    caminho_imagem = f"DOC/pagina_{i + 1}.png"
                    imagem.save(caminho_imagem)
                    
                    # Aqui você pode chamar a função de extração de texto passando o caminho da imagem
                    st.write(f"Extraindo texto da página {i + 1}...")
                    text_extraido = extrair_text_doc(caminho_imagem)
                    texto_extraido += text_extraido
                    
                # Após extrair o texto de todas as páginas, criar o banco de vetores
                st.write("Criando banco de vetores...")
                criar_banco_memoria(texto=text_extraido)
                
                st.success("Documento processado com sucesso!")
        else:
            st.warning("Por favor, faça upload de um arquivo PDF antes de iniciar o processamento.")


#########################################################################
# 2 cabeçalho
st.title("💼 Assistente de TI base RAG")
st.markdown("Bem-vindo! Sou uma IA treinada com os manuais, PDFs e fluxogramas da empresa. Como posso te ajudar hoje?")
st.divider()

# 3. Carregar o Modelo na Memória (O cache impede que ele recarregue toda vez que clicar num botão)
@st.cache_resource
def carregar_modelo():
    modelo = criar_modelo_ollama()
    return modelo

sistema = carregar_modelo()

# MEMORIA DE CONVERSA - aqui a gente cria uma lista para guardar o histórico da conversa.
if 'mensagens' not in st.session_state:
    st.session_state.mensagens  = []
    
# desnha as mensagens antigas no ecra balao do chat
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"]) 
        
# Caixa de chat e processamento da resposta
if pergunta := st.chat_input("Faça sua pergunta aqui..."):
    # Passo 1: Guardar a pergurta do usuário na memória de conversa
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)
        
    with st.chat_message("assistant"):
        with st.spinner("Estou pensando..."):
            historyco = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.mensagens[-4:]]) # pega as ultimas 4 mensagens para dar contexto
            pergunta_com_contexto = f"Historico recente de conversa: \n{historyco}\n\nNova Pergunta do usuário: {pergunta}\nResposta do especialista de TI:"

            
            # enviar para o modelo o contexto da conversa + a nova pergunta
            resposta = sistema.invoke(pergunta_com_contexto)
            st.markdown(resposta)
        # Passo 2: Guardar a resposta do modelo na memória de conversa
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})
        

