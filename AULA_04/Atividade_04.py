import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Criando dados de exemplo para o tempo (Target Contínuo)
# Regra: Cada caractere adiciona ~0.5 min de análise + 5 min base
df['tempo_real_espera'] = (df['tamanho_msg'] * 0.5) + 5 + np.random.normal(0, 2, len(df))

X = df[['tamanho_msg']] # Feature: Comprimento
y = df['tempo_real_espera'] # Target: Minutos (Número)

# 2. Treinamento
reg = LinearRegression()
reg.fit(X, y)

# 3. Predição e Coeficientes
print(f"Intercepto (Tempo Base): {reg.intercept_:.2f} min")
print(f"Coeficiente (Aumento por Caractere): {reg.coef_[0]:.2f} min")

# Teste: Se uma mensagem tem 200 caracteres, quanto tempo demora?
previsao = reg.predict([[200]])
print(f"\nPrevisão para 200 caracteres: {previsao[0]:.2f} minutos")
