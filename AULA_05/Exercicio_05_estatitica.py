import COMPLETE O CÓDIGO

# Lendo os dados de tamanho de mensagem
df = pd.read_csv('dados_estatistica_aula04.csv')

# --- MÉDIA (MEAN) ---
media = df['tamanho_mensagem'].mean()  # Soma todos os tamanhos e divide pela quantidade de mensagens
print(f"3. Média de caracteres: {media:.2f}")  # Exibe o valor médio das mensagens

# --- MEDIANA (MEDIAN) ---
mediana = df['tamanho_mensagem'].median()  # Ordena os dados e pega o valor que está exatamente no meio
print(f"4. Mediana de caracteres: {mediana:.2f}")  # Exibe o valor central
# Objetivo: Ver o centro sem a interferência de valores extremos (mensagens gigantes).

# --- MODA (MODE) ---
moda = df['tamanho_mensagem'].mode()[0]  # Identifica qual o valor que mais se repete na lista
print(f"5. Moda de caracteres: {moda}")  # Exibe o tamanho de mensagem mais frequente
# Objetivo: Identificar o padrão de comportamento mais comum dos usuários.

# --- DESVIO PADRÃO (STANDARD DEVIATION) ---
desvio = df['tamanho_mensagem'].std()  # Calcula o quanto os dados estão afastados da média
print(f"6. Desvio Padrão: {desvio:.2f}")  # Exibe a dispersão dos dados
# Objetivo: Saber se as mensagens têm tamanhos parecidos ou se variam drasticamente.
