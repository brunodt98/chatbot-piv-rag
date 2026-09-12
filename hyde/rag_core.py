"""
Núcleo do pipeline RAG (HyDE + FAISS + OpenRouter).

Compartilhado entre a CLI (app.py) e a interface web (streamlit_app.py)
para evitar duplicar a lógica de recuperação/geração em dois lugares.
"""

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import prompts


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)

DEFAULT_MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")

DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_FAISS = VECTORSTORE_DIR / "index.faiss"
INDEX_PKL = VECTORSTORE_DIR / "index.pkl"

# Preço de referência (USD por 1 milhão de tokens) usado apenas para
# estimar custo na interface/relatórios. Ajuste conforme o modelo/
# provedor configurado — ver README para a fonte usada.
PRECO_USD_POR_MILHAO_INPUT = float(
    os.getenv("PRECO_USD_POR_MILHAO_INPUT", "0.02")
)
PRECO_USD_POR_MILHAO_OUTPUT = float(
    os.getenv("PRECO_USD_POR_MILHAO_OUTPUT", "0.10")
)


class VectorstoreNaoEncontrado(RuntimeError):
    pass


class ChaveApiAusente(RuntimeError):
    pass


# ============================================================
# EMBEDDINGS / VECTORSTORE (carregados uma única vez)
# ============================================================

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    if not INDEX_FAISS.exists() or not INDEX_PKL.exists():
        raise VectorstoreNaoEncontrado(
            f"Vectorstore não encontrado em {VECTORSTORE_DIR}. "
            f"Rode `python build_vectorstore.py` primeiro."
        )

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ============================================================
# LLM
# ============================================================

def build_llm(api_key, model=None, temperature=None, base_url=None):
    if not api_key:
        raise ChaveApiAusente(
            "Nenhuma OPENROUTER_API_KEY foi fornecida (.env ou UI)."
        )

    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=(
            DEFAULT_TEMPERATURE if temperature is None else temperature
        ),
        api_key=api_key,
        base_url=base_url or OPENROUTER_BASE_URL,
    )


@lru_cache(maxsize=8)
def _prompt_template(tipo, versao=None):
    _, template = prompts.carregar_prompt(tipo, versao)
    return ChatPromptTemplate.from_template(template)


# ============================================================
# ESTRUTURAS DE RESULTADO
# ============================================================

@dataclass
class UsoTokens:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def somar(self, usage_metadata):
        if not usage_metadata:
            return
        self.input_tokens += usage_metadata.get("input_tokens", 0) or 0
        self.output_tokens += usage_metadata.get("output_tokens", 0) or 0
        self.total_tokens += usage_metadata.get("total_tokens", 0) or 0

    def custo_usd(self):
        custo_input = (
            self.input_tokens / 1_000_000
        ) * PRECO_USD_POR_MILHAO_INPUT
        custo_output = (
            self.output_tokens / 1_000_000
        ) * PRECO_USD_POR_MILHAO_OUTPUT
        return custo_input + custo_output


@dataclass
class RespostaRAG:
    pergunta: str
    hipotese_hyde: str
    documentos: list
    contexto: str
    resposta: str
    tempo_hyde: float
    tempo_recuperacao: float
    tempo_resposta: float
    uso: UsoTokens = field(default_factory=UsoTokens)

    @property
    def tempo_total(self):
        return self.tempo_hyde + self.tempo_recuperacao + self.tempo_resposta


# ============================================================
# ETAPAS DO PIPELINE
# ============================================================

def _invocar_llm(llm, prompt_template, variaveis):
    mensagens = prompt_template.format_messages(**variaveis)
    resposta = llm.invoke(mensagens)
    texto = (resposta.content or "").strip()
    usage = getattr(resposta, "usage_metadata", None)
    return texto, usage


def gerar_hipotese_hyde(llm, pergunta, versao_prompt=None):
    prompt_template = _prompt_template("hyde", versao_prompt)

    inicio = time.perf_counter()
    hipotese, usage = _invocar_llm(
        llm, prompt_template, {"user_input": pergunta}
    )
    tempo = time.perf_counter() - inicio

    if not hipotese:
        raise ValueError("O modelo não gerou uma hipótese HyDE.")

    return hipotese, usage, tempo


def buscar_documentos(vectorstore, hipotese, k=5):
    inicio = time.perf_counter()
    documentos = vectorstore.similarity_search(hipotese, k=k)
    tempo = time.perf_counter() - inicio
    return documentos, tempo


def buscar_documentos_por_texto(vectorstore, texto, k=5):
    """Recuperação direta (sem HyDE) — usada na avaliação de retrieval
    e como baseline de comparação, já que não depende do LLM."""
    return buscar_documentos(vectorstore, texto, k=k)


def formatar_contexto(documentos):
    partes = []

    for i, documento in enumerate(documentos, start=1):
        texto = documento.page_content.strip()
        fonte = documento.metadata.get("source", "Documento não identificado")

        partes.append(f"\nDOCUMENTO {i}\nFonte: {fonte}\n\n{texto}\n")

    return "\n".join(partes)


def gerar_resposta_final(
    llm, pergunta, contexto, historico_texto, versao_prompt=None
):
    prompt_template = _prompt_template("answer", versao_prompt)

    inicio = time.perf_counter()
    resposta, usage = _invocar_llm(
        llm,
        prompt_template,
        {
            "user_input": pergunta,
            "context": contexto,
            "history": historico_texto,
        },
    )
    tempo = time.perf_counter() - inicio

    return resposta, usage, tempo


def formatar_historico(historico, max_turnos=5):
    if not historico:
        return "Nenhuma conversa anterior."

    return "\n".join(
        f"Usuário: {pergunta}\nAssistente: {resposta}"
        for pergunta, resposta in historico[-max_turnos:]
    )


# ============================================================
# ORQUESTRAÇÃO COMPLETA (usada pela CLI e pela UI)
# ============================================================

def responder_pergunta(
    llm,
    vectorstore,
    pergunta,
    historico=None,
    k=5,
    versao_prompt_hyde=None,
    versao_prompt_answer=None,
):
    uso = UsoTokens()

    hipotese, usage_hyde, tempo_hyde = gerar_hipotese_hyde(
        llm, pergunta, versao_prompt_hyde
    )
    uso.somar(usage_hyde)

    documentos, tempo_recuperacao = buscar_documentos(vectorstore, hipotese, k=k)

    contexto = formatar_contexto(documentos)
    historico_texto = formatar_historico(historico)

    resposta, usage_resposta, tempo_resposta = gerar_resposta_final(
        llm,
        pergunta,
        contexto,
        historico_texto,
        versao_prompt_answer,
    )
    uso.somar(usage_resposta)

    return RespostaRAG(
        pergunta=pergunta,
        hipotese_hyde=hipotese,
        documentos=documentos,
        contexto=contexto,
        resposta=resposta,
        tempo_hyde=tempo_hyde,
        tempo_recuperacao=tempo_recuperacao,
        tempo_resposta=tempo_resposta,
        uso=uso,
    )
