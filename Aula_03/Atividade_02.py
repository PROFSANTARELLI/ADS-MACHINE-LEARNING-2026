import pandas as pd

# 1. CARREGAMENTO
df = pd.read_csv('mensagens_suporte.csv')

# 2. NORMALIZAÇÃO (Teoria aplicada: transformar tudo em minúsculo)
df['texto_limpo'] = df['texto'].str.lower()

# 3. CRIAÇÃO DE FEATURES (Engenharia de Features)
# Contar pontos de exclamação (indicador de urgência/emoção)
df['qtd_exclamacao'] = df['texto'].str.count('!')

# Verificar se contém palavras críticas (Retorna True/False, convertemos para 1/0)
palavras_alerta = ['atraso', 'cancelar', 'chegou', 'cadê', 'agora']
df['tem_palavra_alerta'] = df['texto_limpo'].apply(lambda x: 1 if any(p in x for p in palavras_alerta) else 0)

# Calcular o comprimento da mensagem
df['tamanho_msg'] = df['texto'].str.len()

# 4. RESULTADO DA ENGENHARIA
print("--- Tabela com Novas Features ---")
print(df[['texto', 'qtd_exclamacao', 'tem_palavra_alerta', 'tamanho_msg', 'rotulo']])

# 5. PREPARAÇÃO PARA O MODELO (Removendo o texto bruto)
# O modelo só entende números, então descartamos as colunas de texto
X = df.drop(['texto', 'texto_limpo', 'rotulo'], axis=1)
y = df['rotulo']

print("\n--- Dados prontos para o ML (X) ---")
print(X.head())
