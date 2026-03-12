import pandas as pd  # Importa pandas para criar o arquivo
import numpy as np   # Importa numpy para gerar números aleatórios

# Simulando o comprimento das mensagens (número de caracteres) que o bot recebeu em um dia
# Queremos dados com alguns valores repetidos para testar a Moda e Mediana
comprimentos = [10, 15, 15, 20, 25, 30, 30, 30, 45, 50, 60, 100, 150, 200, 30, 15]

# Adicionando mais dados aleatórios para aumentar a base
np.random.seed(10)
dados_extras = np.random.randint(10, 100, 50).tolist()
comprimento_total = comprimentos + dados_extras

# Criando o DataFrame
df_estatistica = pd.DataFrame({'tamanho_mensagem': comprimento_total})

# Salvando o CSV
df_estatistica.to_csv('dados_estatistica_aula04.csv', index=False)
print(" Arquivo 'dados_estatistica_aula04.csv' gerado com sucesso!")
