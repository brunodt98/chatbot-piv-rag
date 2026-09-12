# Chatbot PI-V — RAG com HyDE (ICMS/SP)

Chatbot que responde perguntas sobre a documentação de ICMS/SP indexada
localmente, usando um pipeline RAG (Retrieval-Augmented Generation) com
HyDE (Hypothetical Document Embeddings) para melhorar a recuperação.

```
Pergunta do usuário
      │
      ▼
  HyDE (LLM)  ──► gera uma "hipótese documental" curta a partir da pergunta
      │
      ▼
 Embedding (HuggingFace, local, CPU)
      │
      ▼
   FAISS  ──► busca os k chunks mais similares nos documentos indexados
      │
      ▼
Contexto formatado + histórico da conversa
      │
      ▼
   LLM (OpenRouter) ──► gera a resposta final, fundamentada só no contexto
      │
      ▼
   Resposta ao usuário
```

## Índice

- [Como rodar](#como-rodar)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Arquitetura da solução](#arquitetura-da-solução)
- [Justificativa das escolhas de stack](#justificativa-das-escolhas-de-stack)
- [Custo de uso estimado](#custo-de-uso-estimado)
- [Latência](#latência)
- [Método de avaliação](#método-de-avaliação)
- [Limitações conhecidas / próximos passos](#limitações-conhecidas--próximos-passos)

## Como rodar

### 1. Pré-requisitos

- Python 3.11+ (testado com 3.13)
- Uma chave de API da [OpenRouter](https://openrouter.ai/keys)

> **Windows:** instale as dependências num ambiente virtual em um caminho
> **curto** (ex.: `C:\venv\hyde`), não dentro de `AppData\...`. O PyTorch
> tem arquivos internos com caminhos muito longos e o Windows tem um limite
> de 260 caracteres por caminho — instalar num caminho já longo causa
> `OSError: [WinError 206]` na instalação.

### 2. Criar o ambiente virtual e instalar dependências

PowerShell:

```powershell
python -m venv C:\venv\hyde
C:\venv\hyde\Scripts\python.exe -m pip install -r requirements.txt
```

(Se preferir um venv dentro do projeto em Linux/Mac, o caminho longo do
Windows não é um problema: `python -m venv .venv && source .venv/bin/activate
&& pip install -r requirements.txt`.)

### 3. Configurar a chave de API

```powershell
Copy-Item .env.example .env
notepad .env   # preencha OPENROUTER_API_KEY=sk-or-...
```

### 4. (Re)gerar o vector store — opcional

O repositório já vem com `vectorstore/index.faiss` e `vectorstore/index.pkl`
prontos (gerados a partir dos arquivos em `documentos/`). Só rode este passo
se adicionar/trocar documentos:

```powershell
C:\venv\hyde\Scripts\python.exe build_vectorstore.py
```

### 5. Rodar a interface web (Streamlit)

```powershell
C:\venv\hyde\Scripts\python.exe -m streamlit run streamlit_app.py
```

Abre em `http://localhost:8501`. Cole a `OPENROUTER_API_KEY` na barra
lateral (ou deixe em branco se já estiver no `.env`).

### 6. Rodar a versão CLI (opcional)

```powershell
C:\venv\hyde\Scripts\python.exe app.py
```

### 7. Rodar as avaliações

```powershell
# Avalia só a recuperação (FAISS), sem gastar tokens de API:
C:\venv\hyde\Scripts\python.exe eval/evaluate_retrieval.py

# Avalia o pipeline completo (HyDE + geração), requer OPENROUTER_API_KEY:
C:\venv\hyde\Scripts\python.exe eval/evaluate_generation.py
```

## Estrutura do projeto

```
hyde/
├── app.py                    # CLI interativa
├── streamlit_app.py          # Interface web (entregável principal)
├── rag_core.py                # Pipeline RAG compartilhado (CLI + web + eval)
├── prompts.py / prompts.yaml  # Prompts versionados (HyDE e resposta final)
├── build_vectorstore.py       # Gera o índice FAISS a partir de documentos/
├── documentos/                 # Fontes .docx (ICMS/SP)
├── vectorstore/                # Índice FAISS já construído (index.faiss/.pkl)
├── eval/
│   ├── dataset.json                # 12 perguntas com keywords esperadas
│   ├── evaluate_retrieval.py       # Recall@k do FAISS (sem custo de API)
│   ├── evaluate_generation.py      # Pipeline completo (com custo de API)
│   └── resultado_retrieval.json    # Saída da última avaliação de retrieval
├── .env.example
└── requirements.txt
```

## Arquitetura da solução

**`rag_core.py`** concentra toda a lógica do pipeline (carregamento de
embeddings/vectorstore, construção do LLM, geração da hipótese HyDE, busca
FAISS, formatação de contexto, geração da resposta e contabilização de
tokens/custo). Tanto `app.py` (CLI), `streamlit_app.py` (web) quanto os
scripts em `eval/` importam esse módulo — assim a lógica de recuperação e
geração não fica duplicada entre a CLI e a UI.

**Chunking:** `RecursiveCharacterTextSplitter` com `chunk_size=1000` e
`chunk_overlap=150` (parâmetros originais do projeto, mantidos — adequados
para texto legal em português, que tem frases/artigos longos).

**Prompts versionados:** `prompts.yaml` guarda os templates do HyDE e da
resposta final, cada um com uma chave de versão (`v1`, ...) e uma `default`.
`prompts.py` carrega e valida (os placeholders obrigatórios de cada tipo
precisam existir no template) — permite testar variações de prompt sem
mexer no código.

## Justificativa das escolhas de stack

- **RAG (em vez de fine-tuning ou só prompt):** a documentação de ICMS/SP
  muda com frequência (novas portarias, redações de artigos) e é extensa.
  RAG permite atualizar o conhecimento só trocando os documentos e
  reconstruindo o índice, sem retreinar nada, e permite citar a fonte —
  essencial para reduzir alucinação em contexto jurídico/fiscal.
- **HyDE:** a pergunta do usuário ("qual o prazo pra impugnar o preço
  publicado?") costuma ser mais curta e coloquial que a linguagem da
  legislação ("levantamento de preços... impugnação do percentual...").
  HyDE gera um trecho hipotético no "estilo" do documento antes de buscar,
  aproximando o vetor de busca do vocabulário real da fonte — mitiga o
  gap léxico entre pergunta e documento, comum em textos jurídicos.
- **Embeddings — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`:**
  multilíngue (mas com bom suporte a PT-BR), leve o suficiente para rodar
  em CPU sem custo de API, e é o mesmo modelo já usado para gerar o índice
  existente no repositório (trocar exigiria reindexar tudo).
- **FAISS (local, em disco):** volume de documentos pequeno (2 arquivos,
  poucas centenas de chunks) — não justifica um vector DB gerenciado
  (Pinecone/Qdrant/Weaviate). FAISS roda embarcado, sem infraestrutura
  extra, ideal para um projeto acadêmico com deploy local.
- **LLM — `openai/gpt-oss-20b` via OpenRouter:** modelo open-weight de baixo
  custo (ver seção de custo abaixo), acessível via uma única API key sem
  precisar de conta própria na OpenAI/Anthropic/Google, e a OpenRouter
  permite trocar de modelo mudando uma env var (`MODEL`) sem alterar código.

## Custo de uso estimado

Preço de referência do `openai/gpt-oss-20b` na OpenRouter em 11/09/2026:
**US$ 0,02 / milhão de tokens de entrada** e **US$ 0,10 / milhão de tokens
de saída** (varia um pouco entre os provedores que a OpenRouter agrega —
até ~US$ 0,03/US$ 0,13 em alguns). Fonte:
https://openrouter.ai/openai/gpt-oss-20b/pricing — **confirme o preço
atual antes da apresentação**, pois preços de LLM mudam com frequência.

Cada pergunta faz **2 chamadas** ao LLM (hipótese HyDE + resposta final).
`rag_core.py` contabiliza os tokens reais de cada chamada
(`response.usage_metadata`) e calcula o custo automaticamente — o valor
aparece no relatório de latência da CLI e no expander "Detalhes da
resposta" da interface web.

<!-- TODO (após rodar eval/evaluate_generation.py com uma API key real):
preencher aqui o custo médio real por pergunta e o custo total das
12 perguntas do dataset de avaliação, usando os números de
eval/resultado_generation.json. -->

## Latência

Cada resposta é decomposta em 3 etapas cronometradas
(`rag_core.RespostaRAG`): `tempo_hyde`, `tempo_recuperacao` (FAISS,
tipicamente < 0,1s pois é busca local em CPU) e `tempo_resposta` (a
geração final, normalmente a etapa mais lenta por ser o texto mais longo).

<!-- TODO (após rodar eval/evaluate_generation.py): preencher aqui a
latência média de ponta a ponta observada nas 12 perguntas do dataset,
e comentar qual etapa domina o tempo total. -->

## Método de avaliação

Duas avaliações complementares, ambas em `eval/`:

1. **`evaluate_retrieval.py` — qualidade da recuperação (sem custo de API).**
   Um dataset de 12 perguntas sobre ICMS/SP (`eval/dataset.json`), escrito
   manualmente a partir do conteúdo real de `documentos/`, cada uma com
   palavras-chave que **devem** aparecer nos chunks recuperados. Mede
   **recall@k**: para cada pergunta, roda a busca no FAISS (embedding
   direto da pergunta) e verifica se alguma keyword esperada aparece nos
   k chunks retornados. Não depende do LLM, então roda sem API key —
   útil para validar o índice/chunking isoladamente da qualidade do LLM.

   **Resultado já executado (11/09/2026, k=5): recall@5 = 91,7% (11/12).**
   Tempo médio de recuperação: 0,75s/pergunta. Única falha: **q05**
   ("Quando falta o valor da operação, quais critérios substitutos são
   usados para a base de cálculo?") — os chunks recuperados não continham
   "FOB" nem "mercado atacadista"; provavelmente a pergunta é genérica
   demais e compete com outros trechos sobre "base de cálculo" no top-5.
   É exatamente o tipo de caso em que o HyDE tende a ajudar (reescreve a
   pergunta num vocabulário mais próximo do texto legal antes de buscar)
   — vale comparar com `evaluate_generation.py` quando a API key estiver
   disponível. Ver `eval/resultado_retrieval.json` para o detalhe completo.
2. **`evaluate_generation.py` — pipeline completo (requer API key).**
   Roda o pipeline HyDE + FAISS + geração para as mesmas 12 perguntas,
   verifica se as keywords esperadas aparecem na **resposta final** (proxy
   de assertividade — não substitui revisão humana, mas dá um sinal
   objetivo e reproduzível), e agrega latência e custo reais por pergunta.

<!-- TODO: depois de rodar evaluate_retrieval.py e evaluate_generation.py,
colar aqui o recall@k obtido e a % de assertividade, com base nos JSONs
gerados em eval/resultado_retrieval.json e eval/resultado_generation.json. -->

## Limitações conhecidas / próximos passos

- Assertividade avaliada por presença de keyword na resposta é um proxy
  automático, não uma revisão jurídica — para a apresentação, vale revisar
  manualmente 3-4 respostas do `resultado_generation.json`.
- O índice FAISS atual foi gerado a partir de 2 documentos; se a base
  crescer bastante, vale reavaliar `chunk_size`/`k` e considerar
  metadados adicionais (ex.: número do artigo) para permitir citação mais
  precisa.
- Não há re-ranking pós-FAISS (ex.: cross-encoder) — para este volume de
  documentos o ganho tende a ser pequeno, mas é uma melhoria natural caso
  o corpus cresça.
