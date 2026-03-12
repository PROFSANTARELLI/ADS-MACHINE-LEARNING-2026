import pandas as pd  # Importa biblioteca para ler dados
import numpy as np   # Importa biblioteca para operações matemáticas
from sklearn.metrics import mean_squared_error  # Importa a função de métrica

# Lendo os dados gerados anteriormente
df = pd.read_csv('dados_metricas_aula04.csv')

# --- EXERCÍCIO: CÁLCULO E INTERPRETAÇÃO DO RMSE ---
# O RMSE penaliza erros grandes (outliers) elevando-os ao quadrado antes da média.
mse = mean_squared_error(df['custo_real'], df['previsao_modelo'])  # Calcula a média dos erros ao quadrado
rmse = np.sqrt(mse)  # Extrai a raiz quadrada do MSE para voltar à unidade original (Reais)
print(f"2. RMSE (Raiz do Erro Quadrático Médio): R$ {rmse:.2f}")  # Exibe o resultado
# Explicação: Note que o RMSE é maior que o MAE porque "sentiu" os 3 erros grandes.
