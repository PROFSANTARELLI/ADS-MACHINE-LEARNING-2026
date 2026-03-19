import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixa base de stopwords
nltk.download('stopwords')

texto = "Eu quero aprender Machine Learning para criar chatbots inteligentes"

# Tokenização
tokens = word_tokenize(texto)

# Lista de stopwords em português
stop_words = stopwords.words('portuguese')

# Remove stopwords
palavras_filtradas = []

for palavra in tokens:
    if palavra.lower() not in stop_words:
        palavras_filtradas.append(palavra)

print(palavras_filtradas)
