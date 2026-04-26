# 💼 Assistente Corporativo de TI (RAG + OCR Multimodal)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-White?style=for-the-badge&logo=ollama)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Database-blueviolet?style=for-the-badge)

## 📌 Visão Geral
Este projeto é um sistema **RAG (Retrieval-Augmented Generation)** corporativo 100% local, focado em total privacidade de dados (Offline-first). Atua como um Engenheiro de TI Sénior, capaz de ingerir manuais técnicos e fluxogramas complexos da empresa e responder a perguntas de utilizadores com precisão contextual e histórico de chat.

## 🚀 Arquitetura e Tecnologias

1. **👀 Ingestão e Visão (GLM-OCR em Capacidade Nativa):**
   - Utilizado para extrair texto de PDFs e imagens. O GLM-OCR atua como um Vision Language Model (VLM), capaz de interpretar a lógica estrutural de **fluxogramas** e diagramas. O modelo corre na sua capacidade real, sem técnicas de redução, garantindo máxima fidelidade na leitura visual.
2. **🧠 Memória Permanente (ChromaDB + Sentence Transformers):**
   - **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` para suporte nativo a várias línguas (incluindo português).
   - Os documentos fatiados (*chunking* com *overlap*) são armazenados fisicamente em disco.
3. **⚙️ Motor de Inferência (Ollama + Qwen 3 32B):**
   - Orquestração híbrida utilizando **GPU Offloading**. O processamento é dividido entre a GPU (RTX 4070 8GB) e a memória do sistema (i9 + 64GB RAM DDR5) para sustentar o modelo massivo de 32B.
4. **🖥️ Interface de Utilizador (Streamlit):**
   - Frontend web interativo com suporte a upload de PDFs na barra lateral e sistema de memória de curto prazo (`session_state`).

## 🛠️ Como Executar Localmente

### Pré-requisitos
* Python 3.10
* Placa Gráfica NVIDIA (Recomendado 8GB+ VRAM) e ferramentas CUDA.
* Ollama instalado.

### Passos de Instalação

1. Clone o repositório:
   ```bash
   [git clone [(https://github.com/diego-andrade279/Assistente-Corporativo-RAG.git)]
