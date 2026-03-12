import pandas as pd  # Importa a biblioteca para manipulação de tabelas
import numpy as np   # Importa a biblioteca para cálculos matemáticos e aleatórios

np.random.seed(42)

# Criando dados simulados de um Chatbot: Custo Real vs Custo Previsto pelo modelo
n_amostras = 100  # Define que teremos 100 linhas de dados
custo_real = np.random.uniform(10.0, 50.0, n_amostras)  # Gera custos reais entre 10 e 50 reais

# Simula a previsão do modelo com um pequeno erro (ruído)
# O erro médio aqui é propositalmente de aproximadamente 2 unidades
previsao_modelo = custo_real + np.random.normal(0, 2, n_amostras)

# Adiciona 3 "erros grosseiros" (outliers) para testar o peso no RMSE
previsao_modelo[0] += 50  # Erro de 50 reais a mais na primeira linha
previsao_modelo[10] -= 40 # Erro de 40 reais a menos na linha 10
previsao_modelo[50] += 60 # Erro de 60 reais a mais na linha 50

# Monta o DataFrame (tabela)
df_metricas = pd.DataFrame({
    'custo_real': custo_real,
    'previsao_modelo': previsao_modelo
})

# Salva o arquivo CSV para ser usado nos exercícios
df_metricas.to_csv('dados_metricas_aula04.csv', index=False)
print("✅ Arquivo 'dados_metricas_aula04.csv' gerado com sucesso!")
