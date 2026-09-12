"""CLI do chatbot RAG (HyDE + FAISS + OpenRouter) para consulta interativa."""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import rag_core


def mostrar_relatorio_latencia(resultado):
    print("\n")
    print("=" * 70)
    print("              ⏱️ RELATÓRIO DE LATÊNCIA")
    print("=" * 70)
    print(f"🧠 Tempo HyDE:                    {resultado.tempo_hyde:.3f} segundos")
    print(f"🔎 Tempo Recuperação:             {resultado.tempo_recuperacao:.3f} segundos")
    print(f"🤖 Tempo Resposta Final:          {resultado.tempo_resposta:.3f} segundos")
    print("-" * 70)
    print(f"⏱️ TEMPO TOTAL DA CONSULTA:       {resultado.tempo_total:.3f} segundos")
    print(
        f"💰 Tokens (in/out/total):         "
        f"{resultado.uso.input_tokens}/"
        f"{resultado.uso.output_tokens}/"
        f"{resultado.uso.total_tokens}  "
        f"(~US$ {resultado.uso.custo_usd():.6f})"
    )
    print("=" * 70)


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("❌ ERRO: OPENROUTER_API_KEY não foi encontrada no .env")
        sys.exit(1)

    try:
        vectorstore = rag_core.get_vectorstore()
    except rag_core.VectorstoreNaoEncontrado as e:
        print(f"❌ {e}")
        sys.exit(1)

    llm = rag_core.build_llm(api_key)

    print("=" * 60)
    print("CHATBOT PI-V")
    print("=" * 60)
    print(f"\n🤖 Modelo: {rag_core.DEFAULT_MODEL}")
    print("🔗 Provedor: OpenRouter")
    print("📚 Banco vetorial: FAISS")
    print(f"🧠 Embeddings: {rag_core.EMBEDDING_MODEL}")
    print("\nArquitetura:")
    print("Pergunta -> HyDE -> Embedding -> FAISS -> Documentos -> LLM -> Resposta")
    print("\nDigite 'sair' para encerrar.")

    historico = []

    while True:
        pergunta = input("\nVocê: ").strip()

        if not pergunta:
            continue

        if pergunta.lower() == "sair":
            print("\nEncerrando...")
            break

        try:
            print("\n🧠 Gerando hipótese HyDE, buscando documentos e respondendo...")

            resultado = rag_core.responder_pergunta(
                llm, vectorstore, pergunta, historico=historico, k=5
            )

            print("\n📄 Hipótese HyDE gerada:")
            print("-" * 40)
            print(resultado.hipotese_hyde)
            print("-" * 40)

            print("\n🤖 Assistente:")
            print("-" * 40)
            print(resultado.resposta)
            print("-" * 40)

            historico.append((pergunta, resultado.resposta))

            mostrar_relatorio_latencia(resultado)

        except KeyboardInterrupt:
            print("\n\nEncerrando...")
            break

        except Exception as e:
            print("\n❌ Ocorreu um erro:")
            print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
