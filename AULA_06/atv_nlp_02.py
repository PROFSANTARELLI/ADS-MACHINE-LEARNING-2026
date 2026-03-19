import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Realiza o download dos recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')

def etapa_01():
    """
    ETAPA 1: O NORMALIZADOR DE CHAT
    Objetivo: Criar uma função que recebe uma mensagem 'suja' e retorna 
    apenas o texto em minúsculo e sem nenhuma pontuação.
    """
    print("\n--- Exercício 1: Normalizador ---")
    mensagem_suja = "ERRO Crítico no Servidor!!! Código: #404... favor verificar AGORA."
    
    # 1. Transformar em minúsculo
    fase_1 = mensagem_suja.lower()
    
    # 2. Remover pontuação usando a biblioteca string
    # maketrans cria um mapa de substituição onde pontuação vira 'None'
    fase_2 = fase_1.translate(str.maketrans('', '', string.punctuation))
    
    print(f"Original: {mensagem_suja}")
    print(f"Limpo:    {fase_2}")


def etapa_02():
    """
    ETAPA 2: CONTADOR DE TOKENS RELEVANTES
    Objetivo: Dada uma lista de frases de suporte, o aluno deve tokenizar
    e remover as stopwords de português.
    """
    print("\n--- Exercício 2: Filtro de Relevância ---")
    frases = [
        "O meu boleto ainda não chegou no meu e-mail",
        "Gostaria de falar com um atendente humano por favor"
    ]
    
    stops = set(stopwords.words('portuguese'))
    
    for frase in frases:
        # Tokenização
        tokens = word_tokenize(frase.lower())
        
        # Filtro de Stopwords
        tokens_limpos = [t for t in tokens if t not in stops and t.isalnum()]
        
        print(f"Frase:  {frase}")
        print(f"Tokens: {tokens_limpos}")


def etapa_03():
    """
    ETAPA 3: DETECTOR DE URGÊNCIA (SCORE DE SENTIMENTO)
    Objetivo: Criar um sistema simples que atribui um 'Score de Urgência' 
    baseado na presença de palavras-chave após a limpeza do texto.
    """
    print("\n--- Exercício 3: Detector de Urgência ---")
    palavras_urgentes = ['urgente', 'erro', 'problema', 'quebrado', 'parado', 'falha']
    
    mensagens = [
        "Olá, estou com um problema urgente, meu sistema está parado!",
        "Bom dia, gostaria de saber o preço do plano premium.",
        "Erro fatal na integração, socorro!!!"
    ]
    
    for msg in mensagens:
        # Pipeline completa: Lower -> Clean -> Tokenize
        limpa = msg.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(limpa)
        
        # Cálculo do Score: Quantos tokens urgentes existem na mensagem?
        score = sum(1 for t in tokens if t in palavras_urgentes)
        
        status = "ALTA PRIORIDADE" if score > 0 else "NORMAL"
        print(f"[{status}] Mensagem: {msg}")


if __name__ == "__main__":
    etapa_01()
    etapa_02()
    etapa_03()
