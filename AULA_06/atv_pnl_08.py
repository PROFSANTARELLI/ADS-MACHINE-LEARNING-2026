# ==========================
# 1. IMPORTAÇÕES
# ==========================
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ==========================
# 2. BASE DE DADOS (INTENÇÕES)
# ==========================
frases = [
    "Quero cancelar meu pedido",
    "Preciso cancelar a compra",
    "Como faço para cancelar?",
    
    "Produto veio com defeito",
    "Minha entrega atrasou",
    "Estou insatisfeito com o produto",
    
    "Qual o status do meu pedido?",
    "Quando chega minha entrega?",
    "Quero saber o prazo de entrega"
]

labels = [
    "cancelamento", "cancelamento", "cancelamento",
    "reclamacao", "reclamacao", "reclamacao",
    "consulta", "consulta", "consulta"
]

# ==========================
# 3. STOPWORDS
# ==========================
stopwords = ["o", "a", "de", "do", "da", "meu", "minha", "com", "para", "e", "como"]

# ==========================
# 4. FUNÇÃO DE PRÉ-PROCESSAMENTO
# ==========================
def preprocessar_texto(texto):
    texto = texto.lower()  # normalização
    texto = re.sub(r'[^a-zà-ú\s]', '', texto)  # limpeza
    tokens = texto.split()  # tokenização
    tokens = [t for t in tokens if t not in stopwords]  # remoção de stopwords
    return " ".join(tokens)  # retorna string novamente (necessário para vetorizar)

# Aplica o pré-processamento em todas as frases
corpus = [preprocessar_texto(f) for f in frases]

# ==========================
# 5. VETORIZAÇÃO (BAG OF WORDS)
# ==========================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# ==========================
# 6. DIVISÃO TREINO / TESTE
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# ==========================
# 7. TREINAMENTO DO MODELO
# ==========================
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# ==========================
# 8. AVALIAÇÃO DO MODELO
# ==========================
y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.2f}")

# ==========================
# 9. SIMULAÇÃO DE CHATBOT
# ==========================
print("\n=== CHATBOT INTELIGENTE ===")
print("Digite 'sair' para encerrar\n")

while True:
    entrada = input("Usuário: ")

    if entrada.lower() == "sair":
        print("Chatbot: Encerrando atendimento...")
        break

    # pré-processa entrada
    entrada_proc = preprocessar_texto(entrada)

    # transforma em vetor
    entrada_vec = vectorizer.transform([entrada_proc])

    # prediz intenção
    intencao = modelo.predict(entrada_vec)[0]

    # respostas simples baseadas na intenção
    if intencao == "cancelamento":
        resposta = "Posso te ajudar a cancelar seu pedido."
    elif intencao == "reclamacao":
        resposta = "Sinto muito pelo problema. Vamos resolver isso."
    elif intencao == "consulta":
        resposta = "Vou verificar essa informação para você."
    else:
        resposta = "Não entendi sua solicitação."

    print(f"Chatbot ({intencao}): {resposta}\n")
