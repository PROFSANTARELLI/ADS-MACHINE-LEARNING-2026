import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def gerar_dataset_atendimento(n=1000):
    np.random.seed(42)
    
    # 1. Geração de Datas (30 dias de logs)
    data_inicial = datetime(2024, 1, 1)
    datas = [data_inicial + timedelta(minutes=np.random.randint(0, 43200)) for _ in range(n)]
    
    # 2. Categorias e Canais
    canais = ['WhatsApp', 'Web', 'App iOS', 'App Android']
    categorias = ['Premium', 'Standard', 'Free']
    intencoes = ['Financeiro', 'Suporte Técnico', 'Dúvida Produto', 'Reclamação', 'Elogio']
    
    # 3. Construção das Colunas com Lógica de Correlação
    data = {
        'ticket_id': range(1000, 1000 + n),
        'timestamp': datas,
        'canal': np.random.choice(canais, n, p=[0.4, 0.2, 0.2, 0.2]),
        'categoria_cliente': np.random.choice(categorias, n, p=[0.2, 0.3, 0.5]),
        'intencao': np.random.choice(intencoes, n),
        'tamanho_msg': np.random.gamma(shape=2, scale=50, size=n).astype(int) + 10
    }
    
    df = pd.DataFrame(data)
    
    # 4. Lógica de Resposta (Tempo de Resposta depende da Categoria e Intenção)
    # Clientes Premium são atendidos mais rápido. Suporte Técnico demora mais.
    def calcular_tempo(row):
        base = 30 # segundos
        if row['categoria_cliente'] == 'Premium': base -= 15
        if row['categoria_cliente'] == 'Free': base += 60
        if row['intencao'] == 'Suporte Técnico': base *= 2.5
        return max(5, base + np.random.normal(0, base*0.2))

    df['tempo_resposta_seg'] = df.apply(calcular_tempo, axis=1)
    
    # 5. Score de Satisfação (Inversamente proporcional ao tempo de resposta)
    def calcular_satisfacao(row):
        score = 5 - (row['tempo_resposta_seg'] / 150)
        score += np.random.normal(0, 0.5)
        return np.clip(round(score), 1, 5)
    
    df['satisfacao'] = df.apply(calcular_satisfacao, axis=1)
    
    # 6. Injetando Ruído (Dados Ausentes e Outliers)
    # 5% de dados ausentes na satisfação
    df.loc[df.sample(frac=0.05).index, 'satisfacao'] = np.nan
    
    # Outliers: Mensagens gigantescas (bots de spam)
    df.loc[df.sample(10).index, 'tamanho_msg'] *= 15 
    
    df.to_csv('logs_chatbot_eda.csv', index=False)
    print(" Dataset 'logs_chatbot_eda.csv' gerado com sucesso para exploração!")

if __name__ == "__main__":
    gerar_dataset_atendimento()
