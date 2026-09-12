from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import rag_core


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).parent

DOCUMENTOS_DIR = BASE_DIR / "documentos"
VECTORSTORE_DIR = rag_core.VECTORSTORE_DIR.resolve()

EMBEDDING_MODEL = rag_core.EMBEDDING_MODEL


# ============================================================
# VERIFICA DOCUMENTOS
# ============================================================

arquivos_docx = list(DOCUMENTOS_DIR.glob("*.docx"))

if not arquivos_docx:
    raise SystemExit(
        f"Nenhum arquivo .docx encontrado em:\n{DOCUMENTOS_DIR}"
    )

print("Documentos encontrados:")

for arquivo in arquivos_docx:
    print(f"  - {arquivo.name}")


# ============================================================
# CARREGA OS DOCUMENTOS
# ============================================================

documentos = []

for arquivo in arquivos_docx:
    print(f"\nCarregando: {arquivo.name}")

    loader = Docx2txtLoader(str(arquivo))
    docs = loader.load()

    documentos.extend(docs)

print(f"\nTotal de documentos carregados: {len(documentos)}")


# ============================================================
# DIVISÃO EM CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

documentos_divididos = text_splitter.split_documents(documentos)

print(
    f"Total de chunks gerados: "
    f"{len(documentos_divididos)}"
)


# ============================================================
# EMBEDDINGS
# ============================================================

print("\nCarregando modelo de embeddings...")

embeddings = rag_core.get_embeddings()

print("Embeddings carregados.")


# ============================================================
# CRIA FAISS
# ============================================================

print("\nCriando banco vetorial FAISS...")

vectorstore = FAISS.from_documents(
    documentos_divididos,
    embeddings
)


# ============================================================
# SALVA BANCO VETORIAL
# ============================================================

VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

vectorstore.save_local(
    str(VECTORSTORE_DIR)
)


# ============================================================
# CONFIRMAÇÃO
# ============================================================

print("\n========================================")
print("BANCO VETORIAL CRIADO COM SUCESSO!")
print("========================================")

print(f"\nLocal: {VECTORSTORE_DIR}")

print("\nArquivos gerados:")

print(
    f"  - {VECTORSTORE_DIR / 'index.faiss'}"
)

print(
    f"  - {VECTORSTORE_DIR / 'index.pkl'}"
)