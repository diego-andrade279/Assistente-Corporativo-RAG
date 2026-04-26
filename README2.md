<div align="center">

<br/>

```
██████╗ ██████╗ ██╗██╗   ██╗ █████╗ ████████╗███████╗███╗   ███╗██╗███╗   ██╗██████╗ 
██╔══██╗██╔══██╗██║██║   ██║██╔══██╗╚══██╔══╝██╔════╝████╗ ████║██║████╗  ██║██╔══██╗
██████╔╝██████╔╝██║██║   ██║███████║   ██║   █████╗  ██╔████╔██║██║██╔██╗ ██║██║  ██║
██╔═══╝ ██╔══██╗██║╚██╗ ██╔╝██╔══██║   ██║   ██╔══╝  ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
██║     ██║  ██║██║ ╚████╔╝ ██║  ██║   ██║   ███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ 
```

### *Uma Arquitetura RAG Corporativa 100% Offline com Suporte Multimodal e GPU Offloading*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interface-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-LLM_Engine-white?style=for-the-badge&logo=ollama&logoColor=black)](https://ollama.ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-6B46C1?style=for-the-badge)](https://www.trychroma.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/Licença-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Pesquisa_Aplicada-0EA5E9?style=for-the-badge)]()

<br/>

> **Projeto de Pesquisa Aplicada** — Estudo de viabilidade de LLMs locais em escala corporativa.  
> Toda a inteligência roda dentro da sua rede. Nenhum dado sai.

<br/>

</div>

---

## 📌 Índice

- [Visão Geral](#-visão-geral)
- [O Problema que Resolve](#-o-problema-que-resolve)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Stack Tecnológica](#-stack-tecnológica)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação e Execução](#-instalação-e-execução)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Resultados e Desempenho](#-resultados-e-desempenho)
- [Roadmap de Escalabilidade](#-roadmap-de-escalabilidade)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)

---

## 🧠 Visão Geral

O **PrivateMind** é um sistema de *Retrieval-Augmented Generation* (RAG) corporativo desenvolvido como projeto de pesquisa aplicada, com foco em **total privacidade de dados** e **viabilidade de implantação em larga escala**.

O sistema atua como um **Engenheiro de TI Sênior virtual**, capaz de ingerir manuais técnicos, documentos internos e fluxogramas complexos para responder perguntas em linguagem natural — tudo rodando **100% offline**, sem dependência de APIs externas.

```
Você pergunta → O sistema busca no conhecimento da empresa → Qwen 3 32B responde com contexto real
```

A arquitetura foi projetada desde o início de forma **modular e substituível**, permitindo que o projeto evolua de um estudo acadêmico para uma plataforma corporativa de inteligência privada em produção.

---

## 🔐 O Problema que Resolve

Empresas que tentam adotar ferramentas de IA como ChatGPT ou Copilot esbarram em um bloqueio fundamental:

> *"Não podemos enviar nossos documentos internos para a nuvem de terceiros."*

Manuais de processos, fluxogramas de sistemas, políticas internas, dados de clientes — nenhum desses ativos pode trafegar para servidores externos sem violar políticas de privacidade e regulamentações como a **LGPD**.

O PrivateMind resolve isso criando um assistente de IA com o conhecimento específico da empresa, que **nunca se comunica com o mundo externo**.

| Solução Cloud (ChatGPT, Copilot) | PrivateMind |
|---|---|
| ☁️ Dados enviados para servidores externos | 🔒 100% local — dados nunca saem da rede |
| 💸 Custo recorrente por uso (tokens) | 🖥️ Custo fixo de hardware |
| 🌐 Dependência de internet | ✈️ Funciona em ambiente air-gapped |
| 📄 Contexto genérico | 🏢 Contexto específico da empresa |
| ⚠️ Risco de compliance (LGPD, ISO 27001) | ✅ Conformidade total |

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INTERFACE (Streamlit)                        │
│                    Upload de Docs │ Chat Interface                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │         PIPELINE DE INGESTÃO      │
          │                                   │
          │  PDF / Imagem → GLM-OCR (VLM)    │
          │  ↓ Extração de texto + diagramas  │
          │  ↓ Chunking com overlap           │
          │  ↓ Embeddings multilíngues        │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │         MEMÓRIA VETORIAL          │
          │                                   │
          │  ChromaDB (persistente em disco)  │
          │  paraphrase-multilingual-MiniLM   │
          │  Similaridade cossenoidal         │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │       MOTOR DE INFERÊNCIA         │
          │                                   │
          │  Ollama + Qwen 3 32B              │
          │  GPU Offloading Híbrido:          │
          │  ├─ RTX 4070 8GB (VRAM)          │
          │  └─ i9 + 64GB RAM DDR5 (RAM)     │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │            RESPOSTA               │
          │  Contexto recuperado + LLM        │
          │  Histórico de sessão (short-term) │
          └──────────────────────────────────┘
```

### Fluxo de uma Consulta

```
Usuário faz pergunta
        │
        ▼
Gera embedding da pergunta (MiniLM)
        │
        ▼
Busca top-K chunks mais similares no ChromaDB
        │
        ▼
Monta prompt: [Contexto recuperado] + [Histórico] + [Pergunta]
        │
        ▼
Qwen 3 32B gera resposta fundamentada nos documentos
        │
        ▼
Resposta exibida na interface com referência ao documento
```

---

## 🛠️ Stack Tecnológica

| Componente | Tecnologia | Função |
|---|---|---|
| **Visão / OCR** | GLM-OCR (VLM nativo) | Extração de texto, fluxogramas e diagramas |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` | Representação vetorial multilíngue |
| **Banco Vetorial** | ChromaDB | Armazenamento e busca por similaridade |
| **LLM** | Qwen 3 32B via Ollama | Geração de respostas contextuais |
| **Orquestração** | GPU Offloading (Ollama) | Distribuição entre VRAM + RAM |
| **Interface** | Streamlit | Frontend web para upload e chat |
| **Linguagem** | Python 3.10 | Backend e pipeline completa |

### Por que essas escolhas?

- **GLM-OCR:** diferente de OCR tradicional (Tesseract, etc.), o GLM atua como Vision Language Model — ele **entende a lógica** de um fluxograma, não apenas extrai pixels de texto.
- **Qwen 3 32B:** melhor relação capacidade/tamanho disponível para GPU Offloading híbrido. Supera modelos menores em raciocínio técnico sem exigir hardware de datacenter.
- **ChromaDB:** banco vetorial leve, sem servidor externo, com persistência nativa em disco — ideal para ambientes offline.
- **paraphrase-multilingual-MiniLM:** embeddings treinados para múltiplos idiomas, incluindo português — crítico para documentos corporativos em PT-BR.

---

## 💻 Pré-requisitos

### Hardware Mínimo (testado)
```
GPU:  NVIDIA RTX 4070 (8GB VRAM) ou equivalente
RAM:  64GB DDR5 (sistema) — o modelo excede a VRAM e usa RAM via offloading
CPU:  Intel Core i9 (12ª gen+) ou AMD Ryzen 9 equivalente
SSD:  ~50GB livres (modelo Qwen 3 32B ~20GB + ChromaDB)
```

### Hardware Recomendado para Produção
```
GPU:  NVIDIA RTX 4090 (24GB VRAM) ou A100/H100
RAM:  128GB+ ECC
CPU:  Xeon ou EPYC
SSD:  NVMe 1TB+
```

### Software
- Python 3.10
- CUDA Toolkit 12.x
- [Ollama](https://ollama.ai) instalado e rodando
- Drivers NVIDIA atualizados

---

## 🚀 Instalação e Execução

### 1. Clone o repositório
```bash
git clone https://github.com/diego-andrade279/Assistente-Corporativo-RAG.git
cd Assistente-Corporativo-RAG
```

### 2. Crie e ative o ambiente virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Baixe o modelo via Ollama
```bash
# Inicie o Ollama (se não estiver rodando)
ollama serve

# Baixe o Qwen 3 32B (em outro terminal)
ollama pull qwen3:32b
```

### 5. Execute a aplicação
```bash
# Interface principal (Streamlit)
streamlit run app.py

# Pipeline de ingestão (linha de comando)
python main.py
```

A interface estará disponível em `http://localhost:8501`

---

## 📖 Como Usar

### Ingerindo Documentos

1. Acesse a interface pelo navegador
2. Na barra lateral, clique em **"Upload de Documentos"**
3. Selecione um ou mais arquivos PDF ou imagens
4. Aguarde o processamento (OCR + chunking + embeddings)
5. O documento estará disponível para consulta imediatamente

### Fazendo Perguntas

```
Você: "Qual é o procedimento para abertura de chamado de rede?"
      ↓
PrivateMind busca nos documentos carregados
      ↓
PrivateMind: "Conforme o Manual de TI (pág. 23), o procedimento é..."
```

### Dicas de Uso

- **Seja específico:** perguntas diretas retornam contextos mais precisos
- **Use termos do documento:** o sistema busca por similaridade semântica
- **Fluxogramas:** o GLM-OCR consegue interpretar a lógica de processos visuais
- **Histórico:** o sistema mantém contexto da conversa na mesma sessão

---

## 📁 Estrutura do Projeto

```
Assistente-Corporativo-RAG/
│
├── app.py                    # Interface Streamlit — ponto de entrada principal
├── main.py                   # Pipeline de ingestão via linha de comando
├── requirements.txt          # Dependências do projeto
│
├── DB/                       # Documentos fonte para ingestão
│   └── ...                   # PDFs, imagens, manuais
│
├── meu_banco_chroma/         # Banco vetorial ChromaDB (persistente)
│   └── d9b84633-.../         # Coleção vetorial dos documentos
│
├── models/                   # Configurações e utilitários de modelos
│   └── ...
│
├── my_glm_treinado/          # Checkpoint do GLM-OCR ajustado
│   └── checkpoint-100/
│
├── teste_models/             # Scripts de teste e benchmarks
│   └── ...
│
└── .vscode/                  # Configurações do VS Code
```

---

## 📊 Resultados e Desempenho

> Testes realizados com documentos técnicos corporativos em português (manuais, fluxogramas de processo, políticas internas).

| Métrica | Resultado |
|---|---|
| Tempo médio de resposta | ~8-15 segundos (Qwen 3 32B offloading) |
| Precisão de recuperação (top-3) | Alta para documentos bem estruturados |
| Extração de fluxogramas (GLM-OCR) | Interpreta lógica de processos com fidelidade |
| Uso de VRAM (RTX 4070) | ~7.8GB (próximo ao limite) |
| Uso de RAM do sistema | ~35-40GB durante inferência |
| Suporte a idiomas | PT-BR, EN, ES e outros (embeddings multilíngues) |

### Observações de Viabilidade

- ✅ **Tecnicamente viável** rodar Qwen 3 32B em hardware consumer via GPU Offloading
- ✅ **GLM-OCR** supera OCR tradicional na interpretação de documentos estruturados
- ⚠️ **Latência** de 8-15s por resposta é aceitável para uso corporativo assíncrono
- ⚠️ **Hardware** é o principal gargalo para escala — GPUs com mais VRAM reduzem drasticamente a latência
- ✅ **Privacidade total** confirmada — zero chamadas externas durante operação

---

## 🗺️ Roadmap de Escalabilidade

O projeto foi arquitetado para crescer em fases. Cada componente é substituível de forma independente.

```
FASE 1 — Atual (Pesquisa / PoC)
├── ✅ Pipeline RAG offline funcional
├── ✅ OCR multimodal com GLM
├── ✅ Interface básica Streamlit
└── ✅ Validação de viabilidade técnica

FASE 2 — Produto Interno (3-6 meses)
├── 🔲 Autenticação e controle de acesso por usuário
├── 🔲 Multi-tenant — bases de conhecimento por departamento
├── 🔲 API REST (FastAPI) para integração com outros sistemas
├── 🔲 Dashboard de administração (upload, gestão de docs)
└── 🔲 Containerização com Docker + Docker Compose

FASE 3 — Escala Corporativa (6-12 meses)
├── 🔲 Integração com Active Directory / LDAP
├── 🔲 Deploy em cluster (Kubernetes)
├── 🔲 Suporte a múltiplos LLMs (troca de modelo sem código)
├── 🔲 Fine-tuning específico por setor/empresa
├── 🔲 Monitoramento e observabilidade (logs, métricas, alertas)
└── 🔲 Interface corporativa com branding customizável

FASE 4 — Plataforma SaaS Privada (12+ meses)
├── 🔲 Deploy on-premises automatizado (bare metal / VM)
├── 🔲 Marketplace de conectores (ERP, CRM, SharePoint)
├── 🔲 Avaliação automática de qualidade das respostas (RAG Eval)
└── 🔲 Certificações de segurança (ISO 27001, SOC 2)
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Este projeto é aberto para colaboração acadêmica e profissional.

```bash
# Fork o repositório
# Crie uma branch para sua feature
git checkout -b feature/minha-feature

# Commit suas mudanças
git commit -m "feat: adiciona suporte a documentos Word (.docx)"

# Push para sua branch
git push origin feature/minha-feature

# Abra um Pull Request
```

### Áreas onde contribuições são mais impactantes
- 🧪 Benchmarks e métricas de avaliação de qualidade (RAGAS, etc.)
- 📄 Suporte a novos formatos de documento (DOCX, XLSX, HTML)
- 🌐 Conectores com sistemas corporativos (SharePoint, Confluence)
- 🐳 Containerização e scripts de deploy
- 📊 Dashboard de administração

---

## 📚 Referências Acadêmicas

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Qwen Team (2025). *Qwen3 Technical Report*. Alibaba Cloud.
- Gao et al. (2023). *Precise Zero-Shot Dense Retrieval without Relevance Labels*. ACL 2023.
- Wang et al. (2024). *GLM: General Language Model Pretraining with Autoregressive Blank Infilling*.

---

## 📄 Licença

Distribuído sob a licença MIT. Veja [`LICENSE`](LICENSE) para mais informações.

---

<div align="center">

**Desenvolvido por [Diego Andrade](https://github.com/diego-andrade279)**

*Projeto de Pesquisa Aplicada — Arquitetura RAG Corporativa Offline*

⭐ Se este projeto foi útil para você, considere deixar uma estrela no repositório!

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-diego--andrade279-181717?style=for-the-badge&logo=github)](https://github.com/diego-andrade279)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/)

</div>
