import pandas as pd  # Importa biblioteca para ler dados
import numpy as np   # Importa biblioteca para operações matemáticas
from sklearn.metrics import mean_absolute_error  # Importa a função de métrica

# Lendo os dados gerados anteriormente
df = pd.read_csv('COMPLETE ESSE PARÂMETRO MANUALMENTE')

# --- EXERCÍCIO: CÁLCULO E INTERPRETAÇÃO DO MAE ---
# O MAE foca no erro médio "bruto", sem dar peso extra a erros grandes.
mae = mean_absolute_error(df['custo_real'], df['previsao_modelo'])  # Calcula a média das distâncias absolutas
print(f"1. MAE (Erro Médio Absoluto): R$ {mae:.2f}")  # Exibe o erro médio em formato de moeda
# Explicação: Este valor indica que, na média, o bot erra o custo por esse valor fixo.
