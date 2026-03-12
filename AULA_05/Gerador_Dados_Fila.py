import pandas as pd
import numpy as np   # Operações matemáticas e aleatórias

def gerar_dados_suporte(n=1000):
    np.random.seed(42)
    
    # Criando as Features (X)
    mensagens_anteriores = np.random.randint(1, 15, n)  # Quantidade de trocas com o bot
    complexidade_assunto = np.random.uniform(1.0, 5.0, n)  # Nota de 1 a 5 da dificuldade
    disponibilidade_atendentes = np.random.randint(1, 10, n)  # Quantos humanos online
    
    # Calculando o Tempo de Espera Real (y)
    # Regra: Base 120s + (complexidade * 60s) - (atendentes * 10s) + ruído
    tempo_real = 120 + (complexidade_assunto * 60) - (disponibilidade_atendentes * 10) + np.random.normal(0, 15, n)
    tempo_real = np.clip(tempo_real, 30, 600)  # Garante tempo entre 30s e 10 min
    
    # Gerando a Previsão do Modelo (ŷ)
    # O modelo é bom, mas erra em média 20 segundos
    tempo_predito = tempo_real + np.random.normal(0, 20, n)
    
    # Injetando Outliers Críticos (O "Cenário de Desastre")
    # Em 2% dos casos (20 linhas), o sistema trava e o tempo real explode, mas o modelo não percebe
    indices_falha = np.random.choice(n, 20, replace=False)
    tempo_real[indices_erro] += 400  # Adiciona quase 7 minutos de erro real
    
    # Criando coluna de Satisfação (CSAT) para os exercícios de estatística
    # 1 a 10, onde o tempo de espera influencia o humor do cliente
    satisfacao = 10 - (tempo_real / 80) + np.random.normal(0, 1, n)
    satisfacao = np.clip(satisfacao.astype(int), 1, 10)

    df = pd.DataFrame({
        'mensagens': mensagens_anteriores,
        'complexidade': complexidade_assunto,
        'atendentes': disponibilidade_atendentes,
        'tempo_real': tempo_real,
        'tempo_predito': tempo_predito,
        'satisfacao': satisfacao
    })
    
    df.to_csv('suporte_ecommerce.csv', index=False)
    print(" Arquivo 'suporte_ecommerce.csv' com 1000 registros gerado!")

if __name__ == "__main__":
    gerar_dados_suporte()
