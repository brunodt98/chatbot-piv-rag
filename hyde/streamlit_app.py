"""Interface web (Streamlit) do chatbot RAG (HyDE + FAISS + OpenRouter)."""

import os

import streamlit as st

import rag_core


st.set_page_config(page_title="Chatbot PI-V (RAG + HyDE)", layout="wide")


# ============================================================
# RECURSOS CACHEADOS (carregados uma vez por processo do servidor)
# ============================================================

@st.cache_resource(show_spinner="Carregando embeddings e vectorstore...")
def carregar_vectorstore():
    return rag_core.get_vectorstore()


@st.cache_resource(show_spinner=False)
def carregar_llm(api_key, model, temperature):
    return rag_core.build_llm(api_key, model=model, temperature=temperature)


# ============================================================
# SIDEBAR — CONFIGURAÇÃO
# ============================================================

st.sidebar.header("Configuração")

api_key_input = st.sidebar.text_input(
    "OPENROUTER_API_KEY",
    value=os.getenv("OPENROUTER_API_KEY", ""),
    type="password",
    help="Não é salva em disco. Também pode ser definida via .env.",
)

modelo = st.sidebar.text_input("Modelo (OpenRouter)", value=rag_core.DEFAULT_MODEL)

temperatura = st.sidebar.slider(
    "Temperatura", min_value=0.0, max_value=1.0,
    value=rag_core.DEFAULT_TEMPERATURE, step=0.05,
)

k_documentos = st.sidebar.slider(
    "Documentos recuperados (k)", min_value=1, max_value=10, value=5,
)

mostrar_detalhes = st.sidebar.checkbox(
    "Mostrar hipótese HyDE, contexto e métricas", value=True
)

if st.sidebar.button("Limpar conversa"):
    st.session_state["historico"] = []
    st.session_state["mensagens"] = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Embeddings: `{rag_core.EMBEDDING_MODEL}`\n\n"
    f"Vector store: FAISS local (`hyde/vectorstore`)"
)


# ============================================================
# CABEÇALHO
# ============================================================

st.title("🤖 Chatbot PI-V — RAG com HyDE")
st.caption(
    "Responde perguntas com base exclusivamente nos documentos indexados "
    "(FAISS). Pipeline: Pergunta → HyDE → Embedding → FAISS → Contexto → LLM."
)


# ============================================================
# CARREGAMENTO DO VECTORSTORE
# ============================================================

try:
    vectorstore = carregar_vectorstore()
except rag_core.VectorstoreNaoEncontrado as e:
    st.error(str(e))
    st.stop()


# ============================================================
# ESTADO DA CONVERSA
# ============================================================

if "mensagens" not in st.session_state:
    st.session_state["mensagens"] = []

if "historico" not in st.session_state:
    st.session_state["historico"] = []


for msg in st.session_state["mensagens"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("detalhes"):
            with st.expander("Detalhes da resposta (HyDE, contexto, métricas)"):
                st.markdown(msg["detalhes"])


pergunta = st.chat_input("Faça uma pergunta sobre a documentação...")

if pergunta:
    if not api_key_input:
        st.error(
            "Informe uma OPENROUTER_API_KEY na barra lateral (ou defina no .env) "
            "antes de perguntar."
        )
        st.stop()

    st.session_state["mensagens"].append({"role": "user", "content": pergunta})

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Gerando hipótese HyDE, buscando documentos e respondendo..."):
            try:
                llm = carregar_llm(api_key_input, modelo, temperatura)

                resultado = rag_core.responder_pergunta(
                    llm,
                    vectorstore,
                    pergunta,
                    historico=st.session_state["historico"],
                    k=k_documentos,
                )

            except rag_core.ChaveApiAusente as e:
                st.error(str(e))
                st.stop()

            except Exception as e:
                st.error(f"Ocorreu um erro ao consultar o modelo: {e}")
                st.stop()

        st.markdown(resultado.resposta)

        detalhes_md = None

        if mostrar_detalhes:
            fontes = sorted({
                doc.metadata.get("source", "desconhecida")
                for doc in resultado.documentos
            })

            detalhes_md = (
                f"**Hipótese HyDE:**\n\n> {resultado.hipotese_hyde}\n\n"
                f"**Fontes recuperadas ({len(resultado.documentos)}):** "
                f"{', '.join(os.path.basename(f) for f in fontes)}\n\n"
                f"**Latência:** HyDE `{resultado.tempo_hyde:.2f}s` · "
                f"Recuperação `{resultado.tempo_recuperacao:.2f}s` · "
                f"Resposta `{resultado.tempo_resposta:.2f}s` · "
                f"**Total `{resultado.tempo_total:.2f}s`**\n\n"
                f"**Tokens:** entrada `{resultado.uso.input_tokens}` · "
                f"saída `{resultado.uso.output_tokens}` · "
                f"total `{resultado.uso.total_tokens}` "
                f"(~US$ `{resultado.uso.custo_usd():.6f}`)"
            )

            with st.expander("Detalhes da resposta (HyDE, contexto, métricas)"):
                st.markdown(detalhes_md)

                for i, doc in enumerate(resultado.documentos, start=1):
                    fonte = os.path.basename(
                        doc.metadata.get("source", "desconhecida")
                    )
                    st.markdown(f"**Documento {i} — {fonte}**")
                    st.text(doc.page_content.strip()[:500])

        st.session_state["historico"].append((pergunta, resultado.resposta))
        st.session_state["mensagens"].append({
            "role": "assistant",
            "content": resultado.resposta,
            "detalhes": detalhes_md,
        })
