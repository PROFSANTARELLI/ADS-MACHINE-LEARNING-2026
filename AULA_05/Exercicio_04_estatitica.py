import #COMPLETE ESTE TRECHO DO CÓDIGO

# Carregando o dataset
df = pd.read_csv(COMPLETE O CÓDIGO)

# --- ANÁLISE DE SATISFAÇÃO (CSAT) ---
# Objetivo: Identificar a nota mais comum e a média de humor do cliente.
media_sat = df['satisfacao'].mean()
moda_sat = df['satisfacao'].mode()[0]
print(f"3. Satisfação -> Média: {media_sat:.2f} | Moda: {moda_sat}")

# --- ANÁLISE DE TEMPO (MÉDIA vs MEDIANA) ---
# Objetivo: Descobrir se o tempo de espera está "viciado" por casos extremos.
media_tempo = df['tempo_real'].mean()
mediana_tempo = df['tempo_real'].median()
print(f"4. Tempo Real -> Média: {media_tempo:.2f}s | Mediana: {mediana_tempo:.2f}s")

# --- CONSISTÊNCIA DO ATENDIMENTO (DESVIO PADRÃO) ---
# Objetivo: O atendimento é previsível ou cada cliente tem uma experiência diferente?
desvio_tempo = df['tempo_real'].std()
print(f"5. Dispersão -> Desvio Padrão do Tempo: {desvio_tempo:.2f}s")
