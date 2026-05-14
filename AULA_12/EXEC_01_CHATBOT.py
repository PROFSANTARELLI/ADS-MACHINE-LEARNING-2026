"""
=============================================================================
EXERCÍCIO 01 — CHATBOT BASEADO EM REGRAS (IF/ELIF)
=============================================================================
OBJETIVO:
  Construir o chatbot simples com respostas fixas para perguntas fixas.
  Aprender o conceito de loop input/output e a estrutura básica de um chatbot.

CONCEITOS ABORDADOS:
  - Loop de conversa (while True)
  - Normalização básica de texto (lower/strip)
  - Correspondência exata por if/elif
  - Condição de saída

LIMITAÇÃO INTENCIONAL:
  Você vai perceber que frases ligeiramente diferentes (ex: "oi" vs "olá")
  não são reconhecidas. Isso será aplicado no Exercício 02.
=============================================================================
"""

# -----------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DO CHATBOT
# Recebe a mensagem do usuário e retorna a resposta adequada
# -----------------------------------------------------------------------------

def responder(mensagem: str) -> str:
    """
    Motor de resposta baseado em regras rígidas (if/elif).

    Parâmetro:
        mensagem (str): texto digitado pelo usuário

    Retorna:
        str: resposta do chatbot
    """

    # Normalização básica: tudo em minúsculas e sem espaços extras
    # Isso resolve "Olá" vs "olá" mas NÃO resolve "oi" vs "olá"
    msg = mensagem.lower().strip()

    # --- SAUDAÇÕES ---
    if msg == "olá" or msg == "ola":
        return "Olá! Sou o assistente clínico virtual. Como posso ajudar?"

    elif msg == "oi":
        return "Oi! Bem-vindo ao assistente de saúde. O que deseja saber?"

    elif msg == "bom dia":
        return "Bom dia! Em que posso ajudar você hoje?"

    elif msg == "boa tarde":
        return "Boa tarde! Pronto para auxiliar. Qual sua dúvida?"

    elif msg == "boa noite":
        return "Boa noite! Pode falar, estou aqui para ajudar."

    # --- DÚVIDAS CLÍNICAS COMUNS ---
    elif msg == "o que é glicose":
        return (
            "Glicose é o açúcar presente no sangue. "
            "O valor normal em jejum é entre 70 e 99 mg/dL. "
            "Acima de 126 mg/dL pode indicar diabetes."
        )

    elif msg == "o que é pressão arterial":
        return (
            "Pressão arterial é a força do sangue nas paredes das artérias. "
            "O valor ideal é abaixo de 120/80 mmHg. "
            "Acima de 140/90 mmHg configura hipertensão."
        )

    elif msg == "o que é imc":
        return (
            "IMC (Índice de Massa Corporal) mede a relação entre peso e altura. "
            "Fórmula: peso (kg) ÷ altura² (m). "
            "Entre 18,5 e 24,9 é considerado normal pela OMS."
        )

    elif msg == "o que é colesterol":
        return (
            "Colesterol é uma gordura essencial para o organismo. "
            "O nível total ideal é abaixo de 200 mg/dL. "
            "Acima de 240 mg/dL representa alto risco cardiovascular."
        )

    # --- PERGUNTAS SOBRE O SISTEMA ---
    elif msg == "o que você faz" or msg == "o que voce faz":
        return (
            "Sou um assistente clínico virtual. Posso informar sobre: "
            "glicose, pressão arterial, IMC e colesterol. "
            "Digite o nome do tema para saber mais."
        )

    elif msg == "ajuda" or msg == "help":
        return (
            "Tópicos disponíveis:\n"
            "  → 'o que é glicose'\n"
            "  → 'o que é pressão arterial'\n"
            "  → 'o que é imc'\n"
            "  → 'o que é colesterol'\n"
            "  → 'o que você faz'\n"
            "  → 'sair' para encerrar"
        )

    # --- ENCERRAMENTO ---
    elif msg == "sair" or msg == "tchau" or msg == "até logo":
        return "ENCERRAR"   # Sinal especial para fechar o loop

    # --- FALLBACK: mensagem não reconhecida ---
    else:
        return (
            f"Desculpe, não entendi '{mensagem}'. "
            "Digite 'ajuda' para ver os tópicos disponíveis."
        )


# ---------------------------------------------------------------
# LOOP PRINCIPAL DE CONVERSA
# Simula um terminal de chat: lê, processa, responde, repete
# ---------------------------------------------------------------

def iniciar_chatbot():
    """Inicia o loop de conversa no terminal."""

    print("=" * 55)
    print("  ASSISTENTE CLÍNICO VIRTUAL — Exercício 01")
    print("  Chatbot baseado em Regras")
    print("=" * 55)
    print("  Digite 'ajuda' para ver os comandos disponíveis.")
    print("  Digite 'sair' para encerrar.\n")

    # Contador de turnos — útil para análise didática
    turno = 0

    while True:
        # Lê a mensagem do usuário
        try:
            entrada = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Trata Ctrl+C ou fim de entrada sem travar
            print("\n[Sistema encerrado pelo usuário]")
            break

        # Ignora entradas vazias
        if not entrada:
            print("Chatbot: Por favor, digite alguma mensagem.")
            continue

        turno += 1

        # Obtém a resposta do motor de regras
        resposta = responder(entrada)

        # Verifica condição de encerramento
        if resposta == "ENCERRAR":
            print(f"Chatbot: Até logo! Foram {turno} turnos de conversa. Cuide-se!")
            break

        print(f"Chatbot: {resposta}\n")

if __name__ == "__main__":
    iniciar_chatbot()

# =============================================================================
# EXECUTE E ENTREGUE:
#
# 1. Adicione 3 novas perguntas sobre saúde (ex: "o que é diabetes")
# 2. Tente digitar "Oi!!" (com pontuação) — o que acontece? Por quê?
# 3. Tente digitar "o que é glicemia?" — o que acontece? Por quê?
# 4. Execute e print o 3 exercícios acima e coloque no cartão do SPRINT-4
# REFLEXÃO:
#   Quantas perguntas diferentes um usuário real poderia fazer sobre glicose?
#   Uma regra por frase é escalável? Siga para o Exercício 02.
# =============================================================================
