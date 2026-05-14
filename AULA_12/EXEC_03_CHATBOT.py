"""
=============================================================================
AULA PRÁTICA DE CHATBOT
EXERCÍCIO 03 — BASE DE DADOS SINTÉTICA + SIMILARIDADE TF-IDF
=============================================================================
OBJETIVO:
  Substituir o dicionário manual de palavras-chave por uma BASE DE DADOS de perguntas e respostas.
  O chatbot encontra a pergunta mais parecida usando TF-IDF + Similaridade do Cosseno — sem precisar definir keywords manualmente.

CONCEITOS ABORDADOS:
  - Geração de base de dados sintética (pandas)
  - TF-IDF (Term Frequency–Inverse Document Frequency)
  - Similaridade do Cosseno
  - Threshold de confiança (mínimo para responder)
  - scikit-learn: TfidfVectorizer + cosine_similarity

O QUE É TF-IDF?
  TF (Term Frequency): frequência de uma palavra no documento
  IDF (Inverse Document Frequency): penaliza palavras muito comuns
  TF-IDF alto → palavra importante naquele documento específico

O QUE É SIMILARIDADE DO COSSENO?
  Mede o ângulo entre dois vetores de palavras no espaço vetorial.
  Valor entre 0 (sem relação) e 1 (idênticos).
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# MÓDULO 1 — GERAÇÃO DA BASE DE DADOS SINTÉTICA
# =============================================================================

def gerar_base_faq() -> pd.DataFrame:
    """
    Gera um DataFrame com perguntas e respostas clínicas sintéticas.

    Estrutura:
        id         : identificador único
        categoria  : tema clínico
        pergunta   : pergunta do usuário (variações naturais de linguagem)
        resposta   : resposta informativa
        intencao   : rótulo de intenção (usado no Ex-04 e seguintes)

    Retorna:
        pd.DataFrame com 40 registros
    """

    dados = [
        # ── GLICOSE (8 perguntas) ────────────────────────────────────────
        {
            "id": 1, "categoria": "glicose", "intencao": "info_glicose",
            "pergunta": "qual é o valor normal de glicose no sangue",
            "resposta": "A glicemia normal em jejum está entre 70 e 99 mg/dL. Acima de 126 mg/dL pode indicar diabetes."
        },
        {
            "id": 2, "categoria": "glicose", "intencao": "info_glicose",
            "pergunta": "minha glicemia está alta o que fazer",
            "resposta": "Glicemia alta pode indicar pré-diabetes ou diabetes. Consulte um médico e evite açúcares e carboidratos refinados."
        },
        {
            "id": 3, "categoria": "glicose", "intencao": "info_glicose",
            "pergunta": "o que é pré-diabetes",
            "resposta": "Pré-diabetes ocorre quando a glicemia em jejum está entre 100 e 125 mg/dL. É reversível com dieta e exercício."
        },
        {
            "id": 4, "categoria": "glicose", "intencao": "info_glicose",
            "pergunta": "como medir a taxa de açúcar no sangue",
            "resposta": "Use um glicosímetro em jejum de 8 horas. Fure o dedo, aplique a gota na fita e leia o valor em segundos."
        },
        {
            "id": 5, "categoria": "glicose", "intencao": "info_glicose",
            "pergunta": "qual o perigo de glicose muito baixa",
            "resposta": "Hipoglicemia (glicose < 70 mg/dL) causa tontura, tremores e pode levar ao desmaio. Consuma açúcar ou suco imediatamente."
        },
        {
            "id": 6, "categoria": "glicose", "intencao": "info_glicose",
            "pergunta": "glicose 130 é perigoso",
            "resposta": "Glicemia de 130 mg/dL em jejum está acima do normal (≥ 126 mg/dL pode indicar diabetes). Procure avaliação médica."
        },
        {
            "id": 7, "categoria": "glicose", "intencao": "info_glicose",
            "pergunta": "alimentos que aumentam a glicose",
            "resposta": "Alimentos ricos em carboidratos simples (pão branco, arroz branco, doces, refrigerantes) elevam rapidamente a glicose."
        },
        {
            "id": 8, "categoria": "glicose", "intencao": "info_glicose",
            "pergunta": "diabetes tipo 2 tem cura",
            "resposta": "Diabetes tipo 2 não tem cura, mas é controlável com alimentação, exercício e medicação. Diagnóstico precoce é fundamental."
        },

        # ── PRESSÃO ARTERIAL (8 perguntas) ──────────────────────────────
        {
            "id": 9, "categoria": "pressao", "intencao": "info_pressao",
            "pergunta": "qual é a pressão arterial normal",
            "resposta": "A pressão ideal é abaixo de 120/80 mmHg. Entre 120–139/80–89 é normal limítrofe. Acima de 140/90 é hipertensão."
        },
        {
            "id": 10, "categoria": "pressao", "intencao": "info_pressao",
            "pergunta": "o que causa pressão alta",
            "resposta": "Causas: excesso de sal, sedentarismo, obesidade, estresse, genética e envelhecimento. A maioria é controlável."
        },
        {
            "id": 11, "categoria": "pressao", "intencao": "info_pressao",
            "pergunta": "sintomas de pressão alta",
            "resposta": "Hipertensão geralmente é silenciosa. Quando sintomática: dor de cabeça, visão turva, tontura e zumbido no ouvido."
        },
        {
            "id": 12, "categoria": "pressao", "intencao": "info_pressao",
            "pergunta": "como baixar a pressão naturalmente",
            "resposta": "Reduza sal, pratique exercícios aeróbicos, controle o estresse, evite álcool e cigarro, mantenha peso saudável."
        },
        {
            "id": 13, "categoria": "pressao", "intencao": "info_pressao",
            "pergunta": "pressão 160 por 100 é grave",
            "resposta": "Sim, 160/100 mmHg é Hipertensão Grau 2. Requer avaliação médica urgente e possivelmente medicação."
        },
        {
            "id": 14, "categoria": "pressao", "intencao": "info_pressao",
            "pergunta": "como medir a pressão corretamente",
            "resposta": "Sente-se, descanse 5 minutos, braço na altura do coração, manguito no braço nu. Repita 2x e tire a média."
        },
        {
            "id": 15, "categoria": "pressao", "intencao": "info_pressao",
            "pergunta": "pressão baixa tontura o que fazer",
            "resposta": "Hipotensão (< 90/60 mmHg) com tontura: deite e eleve as pernas, hidrate-se. Se persistir, procure atendimento médico."
        },
        {
            "id": 16, "categoria": "pressao", "intencao": "info_pressao",
            "pergunta": "hipertensão tem cura",
            "resposta": "A hipertensão essencial não tem cura, mas é controlada com medicação e mudanças no estilo de vida."
        },

        # ── IMC (6 perguntas) ────────────────────────────────────────────
        {
            "id": 17, "categoria": "imc", "intencao": "info_imc",
            "pergunta": "como calcular meu imc",
            "resposta": "Divida seu peso (kg) pela altura ao quadrado (m²). Exemplo: 70 kg ÷ (1,70)² = 70 ÷ 2,89 = IMC 24,2 (normal)."
        },
        {
            "id": 18, "categoria": "imc", "intencao": "info_imc",
            "pergunta": "imc 32 é considerado obeso",
            "resposta": "Sim. IMC entre 30 e 34,9 é Obesidade Grau I. Representa risco aumentado para diabetes, hipertensão e doenças cardiovasculares."
        },
        {
            "id": 19, "categoria": "imc", "intencao": "info_imc",
            "pergunta": "qual imc é considerado saudável",
            "resposta": "IMC entre 18,5 e 24,9 é considerado peso normal pela OMS. Entre 25 e 29,9 é sobrepeso."
        },
        {
            "id": 20, "categoria": "imc", "intencao": "info_imc",
            "pergunta": "perder peso melhora o imc",
            "resposta": "Sim. Cada kg perdido reduz o IMC em aproximadamente 0,35 pontos (para altura de 1,70 m). Pequenas reduções já trazem benefícios."
        },
        {
            "id": 21, "categoria": "imc", "intencao": "info_imc",
            "pergunta": "imc baixo é perigoso",
            "resposta": "IMC abaixo de 18,5 indica desnutrição ou magreza excessiva. Pode causar anemia, osteoporose e queda de imunidade."
        },
        {
            "id": 22, "categoria": "imc", "intencao": "info_imc",
            "pergunta": "imc funciona para atletas",
            "resposta": "O IMC tem limitações em atletas pois não distingue músculo de gordura. Medidas complementares como circunferência abdominal são recomendadas."
        },

        # ── COLESTEROL (6 perguntas) ─────────────────────────────────────
        {
            "id": 23, "categoria": "colesterol", "intencao": "info_colesterol",
            "pergunta": "qual o nível ideal de colesterol",
            "resposta": "Colesterol total: < 200 mg/dL (ótimo). LDL: < 130 mg/dL. HDL: > 60 mg/dL (protetor). Triglicerídeos: < 150 mg/dL."
        },
        {
            "id": 24, "categoria": "colesterol", "intencao": "info_colesterol",
            "pergunta": "alimentos que aumentam o colesterol ruim",
            "resposta": "Gorduras saturadas (carnes gordas, manteiga, queijos amarelos) e trans (biscoitos industrializados) elevam o LDL."
        },
        {
            "id": 25, "categoria": "colesterol", "intencao": "info_colesterol",
            "pergunta": "colesterol alto tem sintomas",
            "resposta": "Não. O colesterol alto é silencioso. Só é detectado por exame de sangue (lipidograma). Por isso exames periódicos são essenciais."
        },
        {
            "id": 26, "categoria": "colesterol", "intencao": "info_colesterol",
            "pergunta": "exercício físico baixa colesterol",
            "resposta": "Sim. Exercícios aeróbicos regulares (150 min/semana) elevam o HDL (bom) e reduzem triglicerídeos e LDL."
        },
        {
            "id": 27, "categoria": "colesterol", "intencao": "info_colesterol",
            "pergunta": "diferença entre ldl e hdl",
            "resposta": "LDL transporta colesterol para as artérias (ruim em excesso). HDL remove o excesso das artérias (protetor). Queremos LDL baixo e HDL alto."
        },
        {
            "id": 28, "categoria": "colesterol", "intencao": "info_colesterol",
            "pergunta": "colesterol 250 precisa de remédio",
            "resposta": "Colesterol 250 mg/dL está na faixa alto. O médico avaliará o risco cardiovascular global antes de indicar estatinas ou só dieta."
        },

        # ── RISCO CLÍNICO (6 perguntas) ──────────────────────────────────
        {
            "id": 29, "categoria": "risco", "intencao": "info_risco",
            "pergunta": "o que é risco cardiovascular",
            "resposta": "É a probabilidade de desenvolver infarto ou AVC em 10 anos. Fatores: idade, pressão, colesterol, diabetes, tabagismo e histórico familiar."
        },
        {
            "id": 30, "categoria": "risco", "intencao": "info_risco",
            "pergunta": "como sei se tenho alto risco clínico",
            "resposta": "Múltiplos fatores alterados simultaneamente (glicose + pressão + colesterol + IMC) indicam alto risco. Avaliação médica completa é necessária."
        },
        {
            "id": 31, "categoria": "risco", "intencao": "info_risco",
            "pergunta": "posso reduzir meu risco de doenças crônicas",
            "resposta": "Sim. 80% das doenças crônicas são preveníveis com alimentação saudável, exercício, não fumar e exames periódicos."
        },
        {
            "id": 32, "categoria": "risco", "intencao": "info_risco",
            "pergunta": "fatores de risco para infarto",
            "resposta": "Pressão alta, colesterol alto, diabetes, obesidade, tabagismo, sedentarismo, estresse crônico e histórico familiar."
        },
        {
            "id": 33, "categoria": "risco", "intencao": "info_risco",
            "pergunta": "exames preventivos que devo fazer",
            "resposta": "Anualmente: glicemia, lipidograma, pressão arterial, IMC. A cada 2 anos: ECG. Mulheres: mamografia e Papanicolau conforme idade."
        },
        {
            "id": 34, "categoria": "risco", "intencao": "info_risco",
            "pergunta": "com quantos anos aumenta o risco cardíaco",
            "resposta": "O risco cardiovascular aumenta progressivamente após 40 anos em homens e após 50 anos (menopausa) em mulheres."
        },

        # ── CONSULTA / SISTEMA (6 perguntas) ────────────────────────────
        {
            "id": 35, "categoria": "sistema", "intencao": "ajuda",
            "pergunta": "o que você sabe responder",
            "resposta": "Respondo sobre glicose, pressão arterial, IMC, colesterol e risco clínico. Para emergências, ligue 192 (SAMU)."
        },
        {
            "id": 36, "categoria": "sistema", "intencao": "ajuda",
            "pergunta": "quais tópicos estão disponíveis",
            "resposta": "Tópicos: glicose/glicemia, pressão arterial/hipertensão, IMC/obesidade, colesterol/LDL, risco cardiovascular."
        },
        {
            "id": 37, "categoria": "sistema", "intencao": "consulta",
            "pergunta": "quero marcar uma consulta médica",
            "resposta": "Para consultas: UBS (SUS): ligue 136. Pronto-atendimento: 192 (SAMU). Emergências: 192 ou 193."
        },
        {
            "id": 38, "categoria": "sistema", "intencao": "consulta",
            "pergunta": "onde posso fazer exames de sangue gratuitos",
            "resposta": "Pelo SUS, exames básicos são gratuitos nas UBS (Unidades Básicas de Saúde). Apresente cartão do SUS e pedido médico."
        },
        {
            "id": 39, "categoria": "sistema", "intencao": "saudacao",
            "pergunta": "olá bom dia tudo bem",
            "resposta": "Olá! Tudo bem sim, obrigado! Sou o assistente clínico virtual. Como posso ajudar você hoje?"
        },
        {
            "id": 40, "categoria": "sistema", "intencao": "despedida",
            "pergunta": "obrigado até logo tchau",
            "resposta": "Até logo! Cuide-se bem e não esqueça dos seus exames periódicos. Volte sempre! 🩺"
        },
    ]

    return pd.DataFrame(dados)


# =============================================================================
# MÓDULO 2 — MOTOR TF-IDF
# Vetoriza as perguntas e permite busca por similaridade
# =============================================================================

class ChatbotTFIDF:
    """
    Chatbot baseado em recuperação de informação via TF-IDF.

    Fluxo:
      1. Vetoriza todas as perguntas da base com TF-IDF
      2. Para cada mensagem do usuário, vetoriza também
      3. Calcula a similaridade do cosseno entre a mensagem e todas as perguntas
      4. Retorna a resposta da pergunta mais similar (se score ≥ threshold)
    """

    def __init__(self, base: pd.DataFrame, threshold: float = 0.25):
        """
        Inicializa e treina o vetorizador.

        Parâmetros:
            base      : DataFrame com colunas 'pergunta' e 'resposta'
            threshold : score mínimo de similaridade para responder (0 a 1)
        """
        self.base      = base
        self.threshold = threshold

        # TfidfVectorizer:
        #   - analyzer='word'     : analisa palavras (não caracteres)
        #   - ngram_range=(1,2)   : considera unigrams E bigrams ("pressão alta")
        #   - min_df=1            : inclui todos os termos
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1
        )

        # Vetoriza todas as perguntas da base
        # Resultado: matriz (n_perguntas × n_termos_únicos)
        self.matriz_tfidf = self.vectorizer.fit_transform(base["pergunta"])

        print(f"  ✓ TF-IDF treinado: {len(base)} perguntas × "
              f"{self.matriz_tfidf.shape[1]} termos únicos")

    def responder(self, mensagem: str) -> tuple[str, float, str]:
        """
        Encontra a resposta mais similar à mensagem do usuário.

        Retorna:
            (resposta, score, pergunta_mais_similar)
        """
        # Vetoriza a mensagem do usuário no mesmo espaço vetorial
        vetor_msg = self.vectorizer.transform([mensagem.lower()])

        # Calcula similaridade do cosseno entre a mensagem e cada pergunta
        scores = cosine_similarity(vetor_msg, self.matriz_tfidf).flatten()

        # Encontra o índice da pergunta com maior similaridade
        idx_melhor = scores.argmax()
        score      = scores[idx_melhor]

        if score >= self.threshold:
            resposta  = self.base.iloc[idx_melhor]["resposta"]
            pergunta  = self.base.iloc[idx_melhor]["pergunta"]
            return resposta, round(float(score), 4), pergunta
        else:
            return (
                " Não encontrei uma resposta precisa para isso. "
                "Tente reformular ou pergunte sobre glicose, pressão, IMC ou colesterol.",
                round(float(score), 4),
                "N/A"
            )


# =============================================================================
# LOOP DE CONVERSA
# =============================================================================

def iniciar_chatbot():
    print("=" * 60)
    print("  ASSISTENTE CLÍNICO VIRTUAL — Exercício 03")
    print("  TF-IDF + Similaridade do Cosseno")
    print("=" * 60)

    # Gera e exibe info da base
    print("\n  Gerando base de dados sintética...")
    base = gerar_base_faq()
    base.to_csv("base_faq_clinica.csv", index=False, encoding="utf-8")
    print(f"  ✓ Base salva: base_faq_clinica.csv ({len(base)} registros)")
    print(f"  Categorias: {base['categoria'].value_counts().to_dict()}\n")

    # Instancia o chatbot (treina o TF-IDF)
    bot = ChatbotTFIDF(base, threshold=0.20)

    print("\n  Faça perguntas em linguagem natural. Digite 'sair' para encerrar.\n")

    turno = 0
    while True:
        try:
            entrada = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Encerrado]")
            break

        if not entrada:
            continue

        if entrada.lower() in ["sair", "tchau", "exit"]:
            print("Chatbot: Até logo! Cuide-se! 🩺")
            break

        turno += 1
        resposta, score, pergunta_ref = bot.responder(entrada)

        print(f"Chatbot: {resposta}")
        # Exibe o score de confiança — didático para o aluno entender o modelo
        print(f"         [confiança: {score:.4f} | referência: '{pergunta_ref}']\n")


if __name__ == "__main__":
    iniciar_chatbot()

# =============================================================================
# EXERCÍCIO:
#
# 1. Execute e teste as frases:
#    a) "minha pressão está 150"
#    b) "açúcar no sangue muito alto"
#    c) "quanto deve ser o imc de uma pessoa normal"
#
# 2. Mude o threshold de 0.20 para 0.40. O que muda? E para 0.05?
#    Qual o risco de um threshold muito baixo? E muito alto?
#
# 3. Adicione 5 novas perguntas à base sobre "sedentarismo" ou "tabagismo".
#    O chatbot passa a responder sobre esses temas sem nenhum outro ajuste?
#
# 4. Execute e print o 3 exercícios acima e coloque no cartão do SPRINT-4
# =============================================================================
