"""Carrega os prompts versionados de prompts.yaml."""

from pathlib import Path
from string import Formatter

import yaml


PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"

PLACEHOLDERS_ESPERADOS = {
    "hyde": {"user_input"},
    "answer": {"user_input", "context", "history"},
}


def carregar_prompt(tipo, versao=None):
    """
    Devolve (versao, template) do prompts.yaml para o `tipo` indicado
    ("hyde" ou "answer").

    Se `versao` for None, usa a chave `default` da seção do tipo.
    """

    if tipo not in PLACEHOLDERS_ESPERADOS:
        raise SystemExit(
            f"Tipo de prompt desconhecido: '{tipo}'. "
            f"Esperado: {', '.join(PLACEHOLDERS_ESPERADOS)}"
        )

    if not PROMPTS_FILE.exists():
        raise SystemExit(
            f"Arquivo de prompts não encontrado: {PROMPTS_FILE}"
        )

    dados = yaml.safe_load(
        PROMPTS_FILE.read_text(encoding="utf-8")
    ) or {}

    secao = dados.get(tipo) or {}
    prompts = secao.get("prompts") or {}

    if not prompts:
        raise SystemExit(
            f"Nenhum prompt definido para '{tipo}' em {PROMPTS_FILE.name}."
        )

    versao = versao or secao.get("default")

    if not versao:
        raise SystemExit(
            f"Defina a versão explicitamente ou a chave `default` "
            f"na seção '{tipo}' de {PROMPTS_FILE.name}."
        )

    if versao not in prompts:
        disponiveis = ", ".join(sorted(prompts))

        raise SystemExit(
            f"Versão de prompt '{versao}' não existe para '{tipo}' em "
            f"{PROMPTS_FILE.name}.\n"
            f"Disponíveis: {disponiveis}"
        )

    template = (prompts[versao] or {}).get("template")

    if not template:
        raise SystemExit(
            f"A versão '{tipo}/{versao}' não tem o campo `template`."
        )

    # Verifica se os placeholders obrigatórios existem no template
    campos = {
        nome
        for _, nome, _, _ in Formatter().parse(template)
        if nome
    }

    faltando = PLACEHOLDERS_ESPERADOS[tipo] - campos

    if faltando:
        raise SystemExit(
            f"O template '{tipo}/{versao}' não contém os placeholders "
            f"obrigatórios: {', '.join(sorted(faltando))}"
        )

    return versao, template
