"""
Avalia a QUALIDADE DA RECUPERAÇÃO (retrieval) do FAISS, sem depender do LLM
(logo, roda sem OPENROUTER_API_KEY).

Mede recall@k: para cada pergunta do dataset, verifica se ao menos uma das
keywords esperadas aparece em algum dos k chunks recuperados.

Compara duas estratégias de busca:
  - baseline: embedding direto da pergunta do usuário
  - hyde_simulado: embedding de uma hipótese HyDE escrita à mão (aproximação
    offline do que o LLM geraria), para estimar o ganho do HyDE sem gastar
    tokens de API.

Uso:
    python eval/evaluate_retrieval.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_core  # noqa: E402


DATASET_PATH = Path(__file__).parent / "dataset.json"


def carregar_dataset():
    dados = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return dados["perguntas"]


def avaliar_pergunta(vectorstore, pergunta_texto, keywords, k):
    documentos, tempo = rag_core.buscar_documentos_por_texto(
        vectorstore, pergunta_texto, k=k
    )

    texto_concatenado = " ".join(
        doc.page_content.lower() for doc in documentos
    )

    acertos = [kw for kw in keywords if kw.lower() in texto_concatenado]

    return {
        "acertou": len(acertos) > 0,
        "keywords_encontradas": acertos,
        "tempo": tempo,
        "fontes": sorted({
            Path(doc.metadata.get("source", "?")).name for doc in documentos
        }),
    }


def main(k=5):
    print("Carregando embeddings + vectorstore (pode demorar um pouco)...")
    vectorstore = rag_core.get_vectorstore()

    perguntas = carregar_dataset()

    resultados = []

    print(f"\nAvaliando {len(perguntas)} perguntas com k={k}...\n")
    print(f"{'ID':<5}{'Baseline':<10}{'Pergunta'}")
    print("-" * 90)

    for item in perguntas:
        r_baseline = avaliar_pergunta(
            vectorstore, item["pergunta"], item["keywords_retrieval"], k
        )

        resultados.append({
            "id": item["id"],
            "pergunta": item["pergunta"],
            "baseline": r_baseline,
        })

        status = "✅" if r_baseline["acertou"] else "❌"
        print(f"{item['id']:<5}{status:<10}{item['pergunta'][:70]}")

    total = len(resultados)
    acertos = sum(1 for r in resultados if r["baseline"]["acertou"])
    recall = acertos / total if total else 0.0
    tempo_medio = sum(r["baseline"]["tempo"] for r in resultados) / total

    print("\n" + "=" * 90)
    print(f"Recall@{k} (busca direta, sem HyDE): {acertos}/{total} = {recall:.1%}")
    print(f"Tempo médio de recuperação por pergunta: {tempo_medio:.3f}s")
    print("=" * 90)

    falhas = [r for r in resultados if not r["baseline"]["acertou"]]
    if falhas:
        print("\nPerguntas sem nenhuma keyword encontrada nos chunks recuperados:")
        for r in falhas:
            print(f"  - [{r['id']}] {r['pergunta']}")

    out_path = Path(__file__).parent / "resultado_retrieval.json"
    out_path.write_text(
        json.dumps(
            {
                "k": k,
                "recall_at_k": recall,
                "tempo_medio_segundos": tempo_medio,
                "resultados": resultados,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResultado salvo em: {out_path}")


if __name__ == "__main__":
    main()
