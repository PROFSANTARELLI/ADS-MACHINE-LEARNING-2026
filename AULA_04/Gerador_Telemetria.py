import pandas as pd
import numpy as np

def gerar_dados_servidor(n=500):
    np.random.seed(42) # Garante que os dados sejam os mesmos para todos os alunos
    
    # Criando as Features (Características)
    msgs_per_sec = np.random.randint(100, 2000, n)  # De 100 a 2000 msgs/seg
    latencia_ms = (msgs_per_sec * 0.15) + np.random.normal(20, 10, n) # Latência sobe com o tráfego
    uso_cpu_pct = np.clip((msgs_per_sec / 20) + np.random.normal(5, 5, n), 0, 100) # CPU entre 0-100%

    # Criando o Target (Custo Operacional em R$)
    # Regra: Base de R$ 5.00 + pesos por feature + ruído (erro)
    custo = 5.0 + (msgs_per_sec * 0.05) + (uso_cpu_pct * 0.2) + np.random.normal(0, 5, n)
    
    # Adicionando 5 Outliers (Erros graves para testar o RMSE)
    indices_erro = np.random.choice(n, 5)
    custo[indices_erro] += 150 # Adiciona um custo exorbitante aleatório

    df = pd.DataFrame({
        'msgs_per_sec': msgs_per_sec,
        'latencia_ms': latencia_ms,
        'uso_cpu_pct': uso_cpu_pct,
        'custo_real': custo
    })
    
    df.to_csv('telemetria_servidores.csv', index=False)
    print("✅ Arquivo 'telemetria_servidores.csv' gerado com sucesso!")

if __name__ == "__main__":
    gerar_dados_servidor()
