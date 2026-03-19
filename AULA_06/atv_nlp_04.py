from sklearn.feature_extraction.text import CountVectorizer

# Conjunto de frases
frases = [
    "eu quero comprar",
    "eu quero cancelar",
    "qual o horário",
]

# Cria vetor
vetorizador = CountVectorizer()

# Converte frases em matriz numérica
X = vetorizador.fit_transform(frases)

# Mostra vocabulário
print(vetorizador.get_feature_names_out())

# Mostra matriz numérica
print(X.toarray())
