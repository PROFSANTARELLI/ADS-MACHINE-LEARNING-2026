import pandas as pd
import numpy as np

def generate_chatbot_data(n=100):
    data = {
        'comprimento_msg': np.random.randint(10, 150, n),
        'contem_exclamacao': np.random.choice([0, 1], n),
        'palavras_negativas': np.random.randint(0, 5, n),
        'label': [] # 0: Dúvida, 1: Reclamação
    }
    
    for i in range(n):
        # Regra oculta para o modelo descobrir
        if data['palavras_negativas'][i] > 2 or (data['contem_exclamacao'][i] == 1 and data['comprimento_msg'][i] > 100):
            data['label'].append(1)
        else:
            data['label'].append(0)
            
    return pd.DataFrame(data)

# Gerando o arquivo para a aula
df = generate_chatbot_data(200)
df.to_csv('chatbot_data.csv', index=False)
print("Dataset 'chatbot_data.csv' gerado com sucesso!")
