import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CARREGAMENTO DOS DADOS
# O pandas lê o CSV gerado e cria o DataFrame na memória
df = pd.read_csv('telemetria_servidores.csv')

# 2. DEFINIÇÃO DE FEATURES E TARGET
# X: Dados de entrada (Telemetria) | y: O que queremos prever (Custo)
X = df[['msgs_per_sec', 'latencia_ms', 'uso_cpu_pct']]
y = df['custo_real']

# 3. DIVISÃO EM TREINO E TESTE
# Reservamos 20% para testar a qualidade do modelo após o treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. CONSTRUÇÃO DO PIPELINE
# Um Pipeline agrupa as etapas para que o tratamento de dados seja automático
# Passo 1: StandardScaler -> Coloca todos os dados na mesma escala (Média 0, Desvio 1)
# Passo 2: LinearRegression -> O algoritmo que fará a predição
etapas = [
    ('scaler', StandardScaler()), 
    ('modelo', LinearRegression())
]
pipe = Pipeline(etapas)

# 5. TREINAMENTO
# Ao dar o fit no pipeline, ele treina o Scaler e o Modelo simultaneamente
pipe.fit(X_train, y_train)

# 6. INFERÊNCIA E AVALIAÇÃO
# O modelo tenta prever os custos dos dados de teste
previsoes = pipe.predict(X_test)

# 7. CÁLCULO DAS MÉTRICAS (A parte teórica da aula)
mae = mean_absolute_error(y_test, previsoes)  # Erro Médio Absoluto (em Reais)
mse = mean_squared_error(y_test, previsoes)    # Erro Quadrático Médio (penaliza outliers)
rmse = np.sqrt(mse)                           # Raiz do Erro Quadrático (mesma unidade que y)
r2 = r2_score(y_test, previsoes)              # R-Quadrado (Acurácia da variância de 0 a 1)

# 8. EXIBIÇÃO DOS RESULTADOS
print("-" * 30)
print(f" RESULTADOS DO MODELO")
print("-" * 30)
print(f"MAE:  R$ {mae:.2f} (Erro médio bruto)")
print(f"RMSE: R$ {rmse:.2f} (Sensível a erros grandes)")
print(f"R²:   {r2:.4f} (Quanto mais próximo de 1.0, melhor)")
print("-" * 30)

# 9. TESTE REAL (Simulação de um novo servidor)
# Criamos um cenário com 1500 msgs/seg, 250ms de latência e 80% de CPU
novo_servidor = pd.DataFrame([[1500, 250, 80]], columns=['msgs_per_sec', 'latencia_ms', 'uso_cpu_pct'])
custo_previsto = pipe.predict(novo_servidor)
print(f" Custo previsto para novo cenário: R$ {custo_previsto[0]:.2f}")
