import nltk

# Baixar pacote necessário para tokenização
nltk.download('punkt_tab')

# Importar tokenizador de palavras
from nltk.tokenize import word_tokenize

texto = "Machine Learning é uma das áreas mais importantes da Inteligência Artificial."

# Tokeniza o texto (divide em palavras)
tokens = word_tokenize(texto)

# Conta a quantidade de tokens
quantidade_tokens = len(tokens)

# Mostra os tokens encontrados
print(tokens)

# Mostra a quantidade de tokens
print(f"A quantidade de tokens na mensagem é: {quantidade_tokens}")
