import os
import sys
import time

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
)

MODEL = os.getenv(
    "MODEL",
    "openai/gpt-oss-20b"
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.5")
)


if not OPENROUTER_API_KEY:
    print("❌ ERRO: OPENROUTER_API_KEY não foi encontrada no .env")
    sys.exit(1)


# ============================================================
# MODELO
# ============================================================

llm = ChatOpenAI(
    model=MODEL,
    temperature=TEMPERATURE,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    },
)


# ============================================================
# FAISS
# ============================================================

VECTORSTORE_DIR = os.path.join(
    BASE_DIR,
    "vectorstore"
)

INDEX_FAISS = os.path.join(
    VECTORSTORE_DIR,
    "index.faiss"
)

INDEX_PKL = os.path.join(
    VECTORSTORE_DIR,
    "index.pkl"
)


if not os.path.exists(INDEX_FAISS):
    print("❌ Arquivo index.faiss não encontrado.")
    print(f"Esperado em: {INDEX_FAISS}")
    sys.exit(1)


if not os.path.exists(INDEX_PKL):
    print("❌ Arquivo index.pkl não encontrado.")
    print(f"Esperado em: {INDEX_PKL}")
    sys.exit(1)


vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)


# ============================================================
# HYDE
# ============================================================

hyde_prompt = ChatPromptTemplate.from_template(
    """
Você é o componente HyDE de um sistema RAG.

Sua única função é criar uma pequena hipótese documental
para auxiliar uma busca semântica em documentos sobre PI-V.

NÃO responda à pergunta do usuário.

NÃO converse com o usuário.

NÃO explique sua resposta.

NÃO invente números, valores, percentuais, artigos de lei,
datas, fórmulas ou informações específicas que não estejam
presentes na pergunta.

Use somente os conceitos e termos identificáveis diretamente
na pergunta.

Transforme a pergunta em um pequeno trecho técnico,
contendo palavras-chave e conceitos que provavelmente
apareceriam na documentação relacionada ao assunto.

O texto deve ser curto e objetivo.

A saída será usada SOMENTE como consulta para o mecanismo
de recuperação semântica.

Pergunta do usuário:

{user_input}

Gere somente a hipótese documental.
"""
)

hyde_chain = (
    hyde_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# FUNÇÃO DE BUSCA HYDE + FAISS
# ============================================================

def buscar_documentos_hyde(pergunta, k=5):

    # ========================================================
    # MEDIÇÃO DO HYDE
    # ========================================================

    inicio_hyde = time.perf_counter()

    print("\n🧠 Gerando hipótese HyDE...")

    hipotese = hyde_chain.invoke({
        "user_input": pergunta
    })

    tempo_hyde = time.perf_counter() - inicio_hyde

    hipotese = hipotese.strip()

    if not hipotese:
        raise ValueError(
            "O modelo não gerou uma hipótese HyDE."
        )

    print("\n📄 Hipótese HyDE gerada:")
    print("----------------------------------------")
    print(hipotese)
    print("----------------------------------------")

    print(
        f"⏱️ Tempo HyDE: "
        f"{tempo_hyde:.3f} segundos"
    )


    # ========================================================
    # MEDIÇÃO DA RECUPERAÇÃO
    # ========================================================

    inicio_faiss = time.perf_counter()

    print("\n🔎 Buscando documentos no FAISS...")

    documentos = vectorstore.similarity_search(
        hipotese,
        k=k
    )

    tempo_faiss = time.perf_counter() - inicio_faiss

    print(
        f"⏱️ Tempo Recuperação: "
        f"{tempo_faiss:.3f} segundos"
    )


    return (
        documentos,
        tempo_hyde,
        tempo_faiss
    )


# ============================================================
# PROMPT DA RESPOSTA FINAL
# ============================================================

answer_prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado na documentação do PI-V.

Sua resposta deve ser baseada EXCLUSIVAMENTE
nos documentos recuperados pelo sistema.

REGRAS:

1. Responda à pergunta original do usuário.

2. Utilize somente as informações presentes no contexto.

3. NÃO invente informações.

4. NÃO utilize conhecimento externo.

5. NÃO utilize internet.

6. Se o contexto não possuir informação suficiente,
   diga claramente:

   "A informação não foi encontrada na documentação
   disponibilizada."

7. Não trate a hipótese HyDE como fonte de informação.
   Ela serve apenas para localizar documentos.

8. Quando houver informações relevantes no contexto,
   explique de maneira clara e objetiva.

9. Se houver conflito entre documentos,
   informe que existe uma divergência.

Pergunta original:

{user_input}


Contexto recuperado dos documentos:

{context}


Histórico da conversa:

{history}


Resposta:
"""
)

answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# FORMATAÇÃO DO CONTEXTO
# ============================================================

def formatar_contexto(documentos):

    partes = []

    for i, documento in enumerate(documentos, start=1):

        texto = documento.page_content.strip()

        fonte = documento.metadata.get(
            "source",
            "Documento não identificado"
        )

        partes.append(
            f"""
DOCUMENTO {i}
Fonte: {fonte}

{texto}
"""
        )

    return "\n".join(partes)


# ============================================================
# RELATÓRIO DE LATÊNCIA
# ============================================================

def mostrar_relatorio_latencia(
    tempo_hyde,
    tempo_faiss,
    tempo_resposta,
    tempo_total
):

    print("\n")
    print("=" * 70)
    print("              ⏱️ RELATÓRIO DE LATÊNCIA")
    print("=" * 70)

    print(
        f"🧠 Tempo HyDE:                    "
        f"{tempo_hyde:.3f} segundos"
    )

    print(
        f"🔎 Tempo Recuperação:             "
        f"{tempo_faiss:.3f} segundos"
    )

    print(
        f"🤖 Tempo Resposta Final:          "
        f"{tempo_resposta:.3f} segundos"
    )

    print("-" * 70)

    print(
        f"⏱️ TEMPO TOTAL DA CONSULTA:       "
        f"{tempo_total:.3f} segundos"
    )

    print("=" * 70)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("CHATBOT PI-V")
    print("=" * 60)

    print(f"\n🤖 Modelo: {MODEL}")
    print("🔗 Provedor: OpenRouter")
    print("📚 Banco vetorial: FAISS")
    print(f"🧠 Embeddings: {EMBEDDING_MODEL}")

    print("\nArquitetura:")
    print("Pergunta")
    print("   ↓")
    print("HyDE")
    print("   ↓")
    print("Embedding")
    print("   ↓")
    print("FAISS")
    print("   ↓")
    print("Documentos")
    print("   ↓")
    print(MODEL)
    print("   ↓")
    print("Resposta")

    print("\nDigite 'sair' para encerrar.")

    historico = []

    while True:

        # =====================================================
        # PERGUNTA
        # =====================================================

        pergunta = input("\nVocê: ").strip()

        if not pergunta:
            continue

        if pergunta.lower() == "sair":
            print("\nEncerrando...")
            break


        # =====================================================
        # INÍCIO DO TEMPO TOTAL
        # =====================================================

        inicio_total = time.perf_counter()

        tempo_hyde = 0.0
        tempo_faiss = 0.0
        tempo_resposta = 0.0

        resposta = None


        try:

            # =================================================
            # ETAPA 1 — HYDE + FAISS
            # =================================================

            print("\n" + "=" * 60)
            print("⏳ ETAPA 1/4 — HYDE + RECUPERAÇÃO")
            print("=" * 60)

            (
                documentos,
                tempo_hyde,
                tempo_faiss
            ) = buscar_documentos_hyde(
                pergunta,
                k=5
            )


            # =================================================
            # ETAPA 2 — CONTEXTO
            # =================================================

            print("\n" + "=" * 60)
            print("📚 ETAPA 2/4 — PREPARANDO CONTEXTO")
            print("=" * 60)

            inicio_contexto = time.perf_counter()

            contexto = formatar_contexto(
                documentos
            )

            tempo_contexto = (
                time.perf_counter()
                - inicio_contexto
            )

            print(
                f"✅ Contexto preparado em: "
                f"{tempo_contexto:.3f} segundos"
            )


            # =================================================
            # HISTÓRICO
            # =================================================

            if historico:

                historico_texto = "\n".join(
                    [
                        f"Usuário: {pergunta_usuario}\n"
                        f"Assistente: {resposta_anterior}"
                        for (
                            pergunta_usuario,
                            resposta_anterior
                        ) in historico[-5:]
                    ]
                )

            else:

                historico_texto = (
                    "Nenhuma conversa anterior."
                )


            # =================================================
            # ETAPA 3 — RESPOSTA FINAL
            # =================================================

            print("\n" + "=" * 60)
            print(
                f"🤖 ETAPA 3/4 — "
                f"{MODEL} GERANDO RESPOSTA"
            )
            print("=" * 60)

            inicio_resposta = time.perf_counter()

            resposta = answer_chain.invoke(
                {
                    "user_input": pergunta,
                    "context": contexto,
                    "history": historico_texto,
                }
            )

            tempo_resposta = (
                time.perf_counter()
                - inicio_resposta
            )

            print(
                f"\n✅ Resposta gerada em: "
                f"{tempo_resposta:.3f} segundos"
            )


            # =================================================
            # ETAPA 4 — EXIBIR RESPOSTA
            # =================================================

            print("\n" + "=" * 60)
            print("✅ ETAPA 4/4 — RESPOSTA CONCLUÍDA")
            print("=" * 60)

            print("\n🤖 Assistente:")
            print("----------------------------------------")
            print(resposta.strip())
            print("----------------------------------------")


        except KeyboardInterrupt:

            print("\n\nEncerrando...")
            break


        except Exception as e:

            print("\n❌ Ocorreu um erro:")
            print(
                f"{type(e).__name__}: {e}"
            )


        finally:

            # =================================================
            # TEMPO TOTAL
            # =================================================

            tempo_total = (
                time.perf_counter()
                - inicio_total
            )


            # =================================================
            # RELATÓRIO DE LATÊNCIA
            # =================================================

            mostrar_relatorio_latencia(
                tempo_hyde=tempo_hyde,
                tempo_faiss=tempo_faiss,
                tempo_resposta=tempo_resposta,
                tempo_total=tempo_total
            )


            # =================================================
            # HISTÓRICO
            # =================================================

            if resposta is not None:

                historico.append(
                    (
                        pergunta,
                        resposta.strip()
                    )
                )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()