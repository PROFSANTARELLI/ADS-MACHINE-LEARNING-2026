import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. SETUP DO DESAFIO
# Vamos usar o mesmo dataset da atividade 1, mas o desafio agora é detectar o 'peso' do erro
df = pd.read_csv('telemetria_servidores.csv')

X = df[['msgs_per_sec', 'latencia_ms', 'uso_cpu_pct']]
y = df['custo_real']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. TREINAMENTO DO PIPELINE
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', LinearRegression())
])
pipe.fit(X_train, y_train)

# 3. ANÁLISE DE RESÍDUOS
# DICA: Resíduo = Valor Real - Valor Previsto
previsoes = pipe.predict(X_test)
residuos = y_test - previsoes

# 4. IDENTIFICANDO O MAIOR ERRO
# O aluno deve encontrar qual linha do teste o modelo errou mais 'feio'
maior_erro_idx = np.argmax(np.abs(residuos))
valor_real_problema = y_test.iloc[maior_erro_idx]
valor_previsto_problema = previsoes[maior_erro_idx]

print(f"--- Diagnóstico de Anomalias ---")
print(f"Maior erro encontrado: R$ {residuos.iloc[maior_erro_idx]:.2f}")
print(f"Valor Real: R$ {valor_real_problema:.2f} | Previsto: R$ {valor_previsto_problema:.2f}")

# 5. REFLEXÃO PARA TODOS:
# Como o Pipeline ajuda a garantir que o dado 'anômalo' também seja escalonado?
