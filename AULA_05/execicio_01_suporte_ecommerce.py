import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. CARREGAMENTO DOS DADOS
df = #complete manualmente este trecho do código

# 2. SELEÇÃO DAS VARIÁVEIS
y_true = df['tempo_real']     # O que aconteceu de verdade na fila
y_pred = df['tempo_predito']  # O que o bot prometeu ao cliente

# 3. CÁLCULO DO MAE (Mean Absolute Error)
mae = mean_absolute_error(y_true, y_pred)
# Explicação: Média aritmética simples da diferença absoluta entre promessa e realidade.

# 4. CÁLCULO DO RMSE (Root Mean Squared Error)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# 5. EXIBIÇÃO E DIAGNÓSTICO
print("-" * 40)
print(f"RESUMO DE PERFORMANCE DO MODELO")
print("-" * 40)
print(f"MAE:  {mae:.2f} segundos")
print(f"RMSE: {rmse:.2f} segundos")
print("-" * 40)
