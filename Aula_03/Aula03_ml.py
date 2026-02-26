# 1. IMPORTAÇÃO DE BIBLIOTECAS
import pandas as pd  # Importa o Pandas para manipulação de tabelas (DataFrames)
from sklearn.model_selection import train_test_split  # Importa a função para separar dados de treino e teste
from sklearn.tree import DecisionTreeClassifier  # Importa o algoritmo de Árvore de Decisão
from sklearn.metrics import accuracy_score  # Importa a métrica para calcular a taxa de acerto do modelo

# 2. CARREGAMENTO DOS DADOS
df = pd.read_csv('chatbot_data.csv')  # Lê o arquivo CSV e o transforma em uma estrutura de tabela (DataFrame)

# 3. SEPARAÇÃO DE VARIÁVEIS (FEATURES VS TARGET)
X = df.drop('label', axis=1)  # Define as Features (X): removemos a coluna 'label' do eixo das colunas (axis=1)
y = df['label']  # Define o Target (y): selecionamos apenas a coluna que o modelo deve aprender a prever

# 4. DIVISÃO DO DATASET EM TREINO E TESTE
# Reservamos 20% (test_size=0.2) para testar o modelo após o aprendizado e usamos a semente 42 (random_state=42) para reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# 5. INSTANCIAÇÃO E TREINAMENTO DO MODELO
model = DecisionTreeClassifier()  # Criamos o objeto do classificador (a "inteligência" ainda vazia)
model.fit(X_train, y_train)  # O comando fit faz o modelo "aprender" os padrões relacionando X_train com y_train

# 6. AVALIAÇÃO DO DESEMPENHO
predicoes = model.predict(X_test)  # O modelo tenta adivinhar os rótulos dos dados de teste que ele nunca viu antes
acuracia = accuracy_score(y_test, predicoes)  # Comparamos as previsões feitas com as respostas reais guardadas em y_test
print(f"Acurácia do Modelo: {acuracia * 100:.2f}%")  # Exibe a porcentagem de acerto formatada com duas casas decimais

# 7. INFERÊNCIA (TESTE COM UM NOVO DADO)
nova_msg = [[120, 1, 4]]  # Criamos um exemplo novo: mensagem de 120 caracteres, com exclamação (1) e 4 palavras negativas
resultado = model.predict(nova_msg)  # Solicitamos ao modelo treinado que classifique este novo cenário
print("Resultado da Previsão (0: Dúvida | 1: Reclamação):", resultado[0])  # Exibe o veredito final do chatbot
