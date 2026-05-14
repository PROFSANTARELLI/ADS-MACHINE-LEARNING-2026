"""
=============================================================================
PRÁTICA DE CHATBOT
EXERCÍCIO 02 — DETECÇÃO POR PALAVRAS-CHAVE (KEYWORD MATCHING)
=============================================================================

OBJETIVO:
  Superar a limitação do Exercício 01 (correspondência exata).
  O chatbot agora detecta a INTENÇÃO do usuário verificando se palavras-chave
  estão CONTIDAS na mensagem, não se a frase inteira bate.

EVOLUÇÃO EM RELAÇÃO AO EX-01:
  Ex-01: "o que é glicose" → reconhece
         "fala sobre glicose" → NÃO reconhece
  Ex-02: qualquer frase que contenha "glicose" → reconhece

CONCEITOS ABORDADOS:
  - Operador `in` para busca de substring
  - Dicionário de intenções (intent → keywords)
  - Prioridade de regras (ordem dos ifs importa)
  - Detecção de múltiplas palavras-chave por intenção
  - Remoção de pontuação (pré-processamento simples)

=============================================================================
"""

import re   # Módulo de expressões regulares — para limpar pontuação

# =============================================================================
# BASE DE CONHECIMENTO — DICIONÁRIO DE INTENÇÕES
# Cada intenção tem: lista de palavras-chave e resposta associada
# Esta estrutura substitui o grande bloco de if/elif do Exercício 01
# =============================================================================

BASE_CONHECIMENTO = {

    "glicose": {
        "keywords": ["glicose", "glicemia", "açúcar no sangue", "açucar no sangue",
                     "blood sugar", "diabetes glicose"],
        "resposta": (
            "GLICOSE (glicemia em jejum):\n"
            "  • Normal    : 70–99 mg/dL\n"
            "  • Pré-diab. : 100–125 mg/dL\n"
            "  • Diabetes  : ≥ 126 mg/dL\n"
            "Dica: sempre meça em jejum de pelo menos 8 horas."
        )
    },

    "pressao": {
        "keywords": ["pressão", "pressao", "hipertensão", "hipertensao",
                     "sistólica", "diastólica", "pa ", "mmhg"],
        "resposta": (
            "PRESSÃO ARTERIAL:\n"
            "  • Ótima       : < 120/80 mmHg\n"
            "  • Normal      : 120–129/80 mmHg\n"
            "  • Hipertensão : ≥ 140/90 mmHg\n"
            "Dica: meça sempre sentado, em repouso de 5 minutos."
        )
    },

    "imc": {
        "keywords": ["imc", "índice de massa", "indice de massa",
                     "peso ideal", "obesidade", "sobrepeso"],
        "resposta": (
            "IMC (Índice de Massa Corporal):\n"
            "  • Abaixo do peso : < 18,5\n"
            "  • Normal         : 18,5–24,9\n"
            "  • Sobrepeso      : 25,0–29,9\n"
            "  • Obesidade      : ≥ 30,0\n"
            "Fórmula: peso (kg) ÷ altura² (m²)"
        )
    },

    "colesterol": {
        "keywords": ["colesterol", "ldl", "hdl", "triglicerídeos",
                     "triglicerideos", "gordura no sangue", "lipídios"],
        "resposta": (
            "COLESTEROL TOTAL:\n"
            "  • Ótimo      : < 200 mg/dL\n"
            "  • Limítrofe  : 200–239 mg/dL\n"
            "  • Alto risco : ≥ 240 mg/dL\n"
            "Atenção: HDL alto é protetor; LDL alto é prejudicial."
        )
    },

    "risco": {
        "keywords": ["risco", "perigo", "classificação de risco",
                     "alto risco", "baixo risco", "risco clínico"],
        "resposta": (
            "CLASSIFICAÇÃO DE RISCO CLÍNICO:\n"
            "  • Baixo  : todos os indicadores normais\n"
            "  • Médio  : 1–2 indicadores alterados\n"
            "  • Alto   : 3 ou mais indicadores alterados\n"
            "Dica: o sistema de ML do projeto avalia todos juntos."
        )
    },

    "consulta": {
        "keywords": ["consulta", "médico", "medico", "doutor",
                     "clínica", "clinica", "agendamento", "marcar"],
        "resposta": (
            "SOBRE CONSULTAS:\n"
            "Este sistema é informativo. Para consultar um médico:\n"
            "  → UBS mais próxima (SUS): ligue 136\n"
            "  → Pronto-atendimento: ligue 192 (SAMU)\n"
            "  → Emergências: ligue 192 ou 193"
        )
    },

    "saudacao": {
        "keywords": ["olá", "ola", "oi", "bom dia", "boa tarde",
                     "boa noite", "hey", "hello", "e aí", "e ai"],
        "resposta": (
            "Olá! Sou o Assistente Clínico Virtual 🩺\n"
            "Posso informar sobre: glicose, pressão arterial, IMC e colesterol.\n"
            "Digite 'ajuda' para ver todos os tópicos."
        )
    },

    "ajuda": {
        "keywords": ["ajuda", "help", "tópicos", "topicos",
                     "o que você sabe", "o que voce sabe", "comandos"],
        "resposta": (
            "TÓPICOS DISPONÍVEIS:\n"
            "  → glicose / glicemia\n"
            "  → pressão arterial / hipertensão\n"
            "  → imc / obesidade / sobrepeso\n"
            "  → colesterol / ldl / hdl\n"
            "  → risco clínico\n"
            "  → consulta / médico\n"
            "  → 'sair' para encerrar"
        )
    }
}


# =============================================================================
# PRÉ-PROCESSAMENTO DE TEXTO
# Limpa a mensagem antes de buscar palavras-chave
# =============================================================================

def preprocessar(texto: str) -> str:
    """
    Normaliza o texto para comparação:
      1. Converte para minúsculas
      2. Remove pontuação (exceto letras, números e espaços)
      3. Remove espaços extras nas bordas

    Exemplos:
      "Olá!!"       → "olá"
      "O que é IMC?" → "o que é imc"
    """
    texto = texto.lower()
    # Remove qualquer caractere que não seja letra, número ou espaço
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.strip()


# =============================================================================
# MOTOR DE DETECÇÃO DE INTENÇÃO
# Percorre o dicionário e retorna a primeira intenção cujas keywords apareçam
# =============================================================================

def detectar_intencao(mensagem_limpa: str) -> str | None:
    """
    Verifica qual intenção do BASE_CONHECIMENTO é ativada pela mensagem.

    Retorna o nome da intenção (chave do dicionário) ou None se nenhuma
    palavra-chave for encontrada.
    """
    for intencao, dados in BASE_CONHECIMENTO.items():
        for keyword in dados["keywords"]:
            if keyword in mensagem_limpa:
                return intencao   # Primeira correspondência encontrada

    return None   # Nenhuma intenção detectada → fallback


# =============================================================================
# MOTOR DE RESPOSTA
# =============================================================================

def responder(mensagem: str) -> str:
    """
    Pipeline completo:
      1. Pré-processa o texto
      2. Detecta a intenção
      3. Retorna a resposta correspondente ou fallback

    Parâmetro:
        mensagem (str): entrada bruta do usuário

    Retorna:
        str: resposta do chatbot (ou sinal "ENCERRAR")
    """

    # Condição de saída — verificada antes do processamento
    if mensagem.lower().strip() in ["sair", "tchau", "exit", "quit", "até logo"]:
        return "ENCERRAR"

    # Pré-processa
    mensagem_limpa = preprocessar(mensagem)

    # Detecta intenção
    intencao = detectar_intencao(mensagem_limpa)

    # Retorna resposta ou mensagem de fallback
    if intencao:
        return BASE_CONHECIMENTO[intencao]["resposta"]
    else:
        return (
            f"Não encontrei informações sobre '{mensagem}'.\n"
            "Tente reformular ou digite 'ajuda' para ver os tópicos."
        )


# =============================================================================
# LOOP DE CONVERSA
# =============================================================================

def iniciar_chatbot():
    """Loop principal de conversa no terminal."""

    print("=" * 55)
    print("  ASSISTENTE CLÍNICO VIRTUAL — Exercício 02")
    print("  Detecção por Palavras-Chave")
    print("=" * 55)
    print("  Agora entendo frases variadas sobre o mesmo tema!")
    print("  Digite 'ajuda' ou 'sair'.\n")

    turno = 0

    while True:
        try:
            entrada = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Encerrado]")
            break

        if not entrada:
            continue

        turno += 1
        resposta = responder(entrada)

        if resposta == "ENCERRAR":
            print(f"Chatbot: Até logo! Sessão com {turno} turnos. Cuide-se! 🩺")
            break

        print(f"Chatbot: {resposta}\n")


# =============================================================================
# FUNÇÃO DE TESTE AUTOMATIZADO
# Demonstra que frases variadas sobre o mesmo tema são reconhecidas
# Execute esta função para ver o sistema em ação sem interação manual
# =============================================================================

def testar():
    """Roda um conjunto de frases de teste e exibe os resultados."""

    casos_de_teste = [
        # Devem ser reconhecidas
        ("oi",                          "saudacao"),
        ("Bom dia!!",                   "saudacao"),
        ("o que é glicemia?",           "glicose"),
        ("meu açúcar no sangue está alto", "glicose"),
        ("tenho hipertensão",           "pressao"),
        ("qual o valor normal da PA?",  "pressao"),
        ("estou com sobrepeso",         "imc"),
        ("como calcular meu IMC?",      "imc"),
        ("meu LDL está elevado",        "colesterol"),
        ("quero marcar uma consulta",   "consulta"),
        ("me dá uma ajuda aí",          "ajuda"),
        # Devem cair no fallback
        ("qual é o tempo hoje?",        None),
        ("recomende um remédio",        None),
    ]

    print("\n" + "=" * 55)
    print("  TESTES AUTOMATIZADOS — Exercício 02")
    print("=" * 55)

    acertos = 0
    for frase, esperado in casos_de_teste:
        limpa    = preprocessar(frase)
        detectou = detectar_intencao(limpa)
        ok       = "✓" if detectou == esperado else "✗"
        if detectou == esperado:
            acertos += 1
        print(f"  {ok} '{frase}' → detectou: {detectou} (esperado: {esperado})")

    print(f"\n  Resultado: {acertos}/{len(casos_de_teste)} corretos")
    print("=" * 55)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "teste":
        testar()
    else:
        iniciar_chatbot()

      
# =============================================================================
# EXERCÍCIO PARA O ALUNO:
#
# 1. Adicione a intenção "diabetes" com pelo menos 4 palavras-chave.
# 2. Todos os casos passaram? Algum keyword conflita com outra intenção?
# 3. Explique objetivamente o que aconteceu nesse chatbot
# 4. Execute e print o 3 exercícios acima e coloque no cartão do SPRINT-4
# =============================================================================
