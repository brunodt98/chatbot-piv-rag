# Notas para a apresentação (18/09/26)

Roteiro e pontos-chave para a defesa oral. Detalhes completos e números
atualizados estão no [README.md](README.md); este arquivo é o "script"
para falar sobre eles.

## 1. O problema

Responder perguntas sobre a legislação de ICMS/SP (RICMS/2000, Lei
6.374/1989) de forma **confiável** — sem inventar artigo, percentual ou
prazo que não esteja no material fornecido. Erro em conteúdo fiscal/legal
tem custo real, então o critério não é só "parecer uma resposta boa", é
"estar fundamentada no documento".

## 2. Por que RAG (e não só um LLM "cru")

- Um LLM genérico "sabe" ICMS de forma genérica, mas não conhece as
  redações específicas, datas de vigência e IVA-STs do material fornecido
  — e tende a **alucinar** artigo/número quando não sabe.
- RAG ancora a resposta no texto real: o LLM só responde com o que foi
  recuperado do índice, e o prompt instrui explicitamente a dizer "não
  encontrado" quando o contexto não tem a informação (ver `prompts.yaml`,
  regra 6 do prompt de resposta).
- Fine-tuning foi descartado: exigiria retreinar a cada atualização de
  legislação, dataset de treino muito maior que os 2 documentos
  disponíveis, e ainda não eliminaria alucinação (só reduz o problema de
  estilo, não de fato).

## 3. Por que HyDE

- Pergunta do usuário costuma ser coloquial ("qual o prazo pra reclamar
  do preço?"), enquanto o texto legal usa outro vocabulário
  ("impugnação do percentual ou preço... levantamento de preços").
- HyDE gera um trecho hipotético no estilo do documento **antes** de
  buscar, aproximando o vetor de busca do vocabulário da fonte.
- Trade-off explícito: HyDE adiciona uma chamada extra ao LLM (mais
  latência e custo) — por isso o projeto também mede e reporta o `recall@k`
  **sem** HyDE (`eval/evaluate_retrieval.py`) como baseline de comparação.
  <!-- TODO: depois de rodar a avaliação, citar aqui se o HyDE realmente
  ajudou nas perguntas do dataset, com o número de recall@k. -->

## 4. Por que este embedding, este vector store, este LLM

| Escolha | Motivo |
|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilíngue, leve, roda em CPU sem custo de API; já era o modelo usado para gerar o índice existente |
| FAISS local | Volume pequeno de documentos, não justifica um vector DB gerenciado; zero infraestrutura extra para o deploy local |
| `openai/gpt-oss-20b` via OpenRouter | Baixo custo (~US$0,02/US$0,10 por milhão tokens in/out — ver README), troca de modelo é só uma env var |

## 5. Como o pipeline evita alucinação (pontos a destacar)

1. O prompt de resposta lista regras explícitas: só usar o contexto,
   nunca conhecimento externo, e avisar quando a informação não está
   presente (`prompts.yaml`, seção `answer`).
2. A hipótese HyDE é usada **só** para buscar — o prompt final deixa
   explícito que ela não é fonte de informação (regra 7).
3. O contexto mostrado ao usuário na UI (expander "Detalhes da resposta")
   inclui as fontes recuperadas, permitindo auditar de onde veio a
   resposta.

## 6. Métricas para mostrar ao vivo (ou nos prints)

- **Latência por etapa** (HyDE / recuperação FAISS / geração) — aparece
  em todo request, na CLI e na UI.
- **Tokens e custo estimado** por pergunta — mesmo lugar.
- **Recall@k** do retrieval (`eval/resultado_retrieval.json`).
- **Assertividade** (keywords esperadas presentes na resposta final,
  `eval/resultado_generation.json`).

## 7. Perguntas que o professor pode fazer (e respostas prontas)

- **"E se o documento não tiver a resposta?"** → o prompt instrui o
  modelo a dizer explicitamente que a informação não foi encontrada,
  em vez de inventar (regra 6 do prompt de resposta).
- **"Por que não usar GPT-4 / Claude direto?"** → custo (`gpt-oss-20b`
  é ordens de grandeza mais barato) e porque o ponto do trabalho é o
  pipeline RAG, não qual LLM está por trás — a arquitetura é agnóstica
  ao modelo (troca por env var).
- **"O sistema erra?"** → sim, pode acontecer, por isso existe o dataset
  de avaliação com métricas objetivas (recall@k e assertividade) em vez
  de só "parece funcionar" — mostrar os números reais do
  `eval/resultado_*.json`.
