"""
Avalia o pipeline COMPLETO (HyDE + FAISS + geração da resposta final).

Requer OPENROUTER_API_KEY válida no .env, pois cada pergunta do dataset
gera 2 chamadas ao LLM (hipótese HyDE + resposta final).

Para cada pergunta, registra: resposta gerada, se as keywords esperadas
aparecem na resposta (proxy de assertividade), tempos por etapa, tokens
consumidos e custo estimado (ver PRECO_USD_POR_MILHAO_* no rag_core.py —
ajuste esses valores para o preço real do modelo escolhido).

Uso:
    python eval/evaluate_generation.py
"""

import json
import os
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


def main(k=5):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("❌ ERRO: defina OPENROUTER_API_KEY no .env antes de rodar esta avaliação.")
        sys.exit(1)

    print("Carregando embeddings + vectorstore...")
    vectorstore = rag_core.get_vectorstore()
    llm = rag_core.build_llm(api_key)

    perguntas = carregar_dataset()
    resultados = []

    print(f"\nAvaliando {len(perguntas)} perguntas (pipeline completo, com custo real de API)...\n")

    for item in perguntas:
        print(f"[{item['id']}] {item['pergunta']}")

        try:
            r = rag_core.responder_pergunta(llm, vectorstore, item["pergunta"], k=k)
        except Exception as e:
            print(f"    ❌ erro: {e}")
            resultados.append({"id": item["id"], "erro": str(e)})
            continue

        resposta_lower = r.resposta.lower()
        keywords = item.get("keywords_retrieval", [])
        acertos = [kw for kw in keywords if kw.lower() in resposta_lower]

        print(f"    tempo_total={r.tempo_total:.2f}s  tokens={r.uso.total_tokens}  "
              f"custo=US${r.uso.custo_usd():.6f}  keywords_na_resposta={len(acertos)}/{len(keywords)}")

        resultados.append({
            "id": item["id"],
            "pergunta": item["pergunta"],
            "hipotese_hyde": r.hipotese_hyde,
            "resposta": r.resposta,
            "keywords_esperadas": keywords,
            "keywords_encontradas_na_resposta": acertos,
            "tempo_hyde": r.tempo_hyde,
            "tempo_recuperacao": r.tempo_recuperacao,
            "tempo_resposta": r.tempo_resposta,
            "tempo_total": r.tempo_total,
            "tokens_input": r.uso.input_tokens,
            "tokens_output": r.uso.output_tokens,
            "tokens_total": r.uso.total_tokens,
            "custo_usd": r.uso.custo_usd(),
        })

    validos = [r for r in resultados if "erro" not in r]
    total = len(validos)

    if total:
        assertividade = sum(
            1 for r in validos
            if r["keywords_esperadas"] and
            len(r["keywords_encontradas_na_resposta"]) == len(r["keywords_esperadas"])
        ) / total
        tempo_medio = sum(r["tempo_total"] for r in validos) / total
        custo_total = sum(r["custo_usd"] for r in validos)
        tokens_total = sum(r["tokens_total"] for r in validos)

        print("\n" + "=" * 90)
        print(f"Assertividade (todas as keywords presentes na resposta): {assertividade:.1%}")
        print(f"Tempo médio total por pergunta: {tempo_medio:.2f}s")
        print(f"Tokens totais consumidos: {tokens_total}")
        print(f"Custo total estimado: US$ {custo_total:.6f}")
        print(f"Custo médio por pergunta: US$ {custo_total / total:.6f}")
        print("=" * 90)

    out_path = Path(__file__).parent / "resultado_generation.json"
    out_path.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultado salvo em: {out_path}")


if __name__ == "__main__":
    main()
