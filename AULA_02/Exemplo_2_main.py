import pandas as pd
import matplotlib.pyplot as plt

# 1. Criação do Dataset Fictício
data = {
    'id_transacao': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'valor': [150, 450, 200, 800, 350, 120, 950, 600, 280, 720],
    'satisfacao': [5, 8, 6, 9, 7, 4, 10, 8, 6, 9]
}

# Converte o dicionário em um DataFrame do Pandas
df = pd.DataFrame(data)

# 2. Exploração de Dados
print("--- Primeiras 5 linhas do Dataset ---")
print(df.head())

print("\n--- Estatísticas Descritivas ---")
print(df.describe())

# 3. Visualização
plt.figure(figsize=(10, 6))
plt.scatter(df['valor'], df['satisfacao'], color='blue', edgecolors='black', alpha=0.7)

# Personalização do Gráfico
plt.title('Análise de Satisfação do Cliente vs Investimento')
plt.xlabel('Valor Gasto (R$)')
plt.ylabel('Nível de Satisfação (1-10)')
plt.grid(True, linestyle='--', alpha=0.6)

# Exibição
print("\n Gerando gráfico... Feche a janela da imagem para encerrar o script.")
plt.show()
