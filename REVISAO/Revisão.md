Revisão Semestral de Machine Learning e Chatbots

1. Conteúdo

Etapa 1: Fundamentos de Processamento de Linguagem Natural (NLP)

O Processamento de Linguagem Natural estuda a transição de dados textuais não estruturados para representações formais que computadores possam manipular. Diferente de dados tabulares numéricos, o texto livre é inerentemente ambíguo, inconsistente e computacionalmente complexo, pois os computadores operam com álgebra linear e não com conceitos abstratos de linguagem escrita.

Para mitigar essa complexidade e padronizar o vocabulário, aplica-se um fluxo sequencial de refino linguístico:

- Normalização: Consiste na padronização da caixa de texto (conversão para letras minúsculas). Esta etapa impede que o sistema trate variações gráficas da mesma palavra (como Urgente, urgente e URGENTE) como termos distintos, reduzindo a variabilidade inicial do texto.

- Remoção de Ruído: Processo de eliminação de elementos textuais que não carregam valor de significado (semântico) para o modelo. Envolve a retirada de caracteres especiais, pontuações e símbolos matemáticos. A pontuação serve para a entonação humana, mas para algoritmos estatísticos comuns de classificação, ela atua apenas como ruído estocástico.

- Tokenização: Consiste na fragmentação de uma cadeia contínua de caracteres em unidades mínimas de significado denominadas tokens (geralmente palavras isoladas). É o processo que define as fronteiras físicas de cada termo no documento.

- Remoção de Stopwords: Filtração de conectivos gramaticais redundantes. Stopwords são termos de alta frequência que possuem apenas função sintática na estruturação de frases (artigos, preposições, conjunções, tais como "o", "a", "de", "com", "por"). Mantê-los incharia o espaço vetorial desnecessariamente, sem adicionar poder preditivo ao classificador de intenções.


Etapa 2: Vetorização de Texto e Agrupamento Não Supervisionado (Clusterização)

Uma vez limpo e tokenizado, o texto precisa ser mapeado em um espaço matemático n-dimensional, onde cada dimensão corresponde a um termo do vocabulário gerado pelo conjunto total de documentos.

- Bag of Words (BoW): Esta técnica constrói um vetor de frequência absoluta de ocorrências. Desconsidera-se totalmente a ordem sintática e o contexto das palavras. A principal limitação deste método é que termos de altíssima frequência em todo o dataset (mas pouco informativos) dominam os pesos dos vetores, obscurecendo palavras raras que definem o real assunto do texto.

- TF-IDF (Term Frequency – Inverse Document Frequency): É uma ponderação estatística que mede a importância relativa de um termo em um documento específico frente ao corpus completo. O TF (Frequência do Termo) mede o quão frequente uma palavra é no documento atual. O IDF (Frequência Inversa do Documento) penaliza termos que aparecem em quase todos os documentos do dataset, agindo como um filtro de relevância inteligente que valoriza termos de nicho e penaliza o ruído comum.

- K-Means (Clusterização): Algoritmo de aprendizado não supervisionado que agrupa documentos sem a necessidade de rótulos prévios. Ele opera calculando a distância euclidiana entre os vetores de texto gerados pelo TF-IDF, movendo iterativamente pontos centrais virtuais (centróides) até que a variância interna de cada grupo (cluster) seja minimizada. É utilizado para a descoberta automática de novos problemas e intenções em logs históricos de atendimento.


Etapa 3: Classificação Supervisionada de Intenções

No aprendizado supervisionado, o algoritmo é exposto a dados rotulados, contendo o texto vetorizado (variáveis independentes X) e suas respectivas intenções mapeadas (variável dependente y). O objetivo é aprender a função matemática que separa essas classes no espaço multidimensional.

- Regressão Logística: Algoritmo linear de alta performance para texto. Ele calcula a probabilidade de uma amostra pertencer a uma determinada classe utilizando a função sigmóide para mapear valores reais entre zero e um. É rápido e menos propenso a sobreajuste (overfitting) em matrizes esparsas de alta dimensão.

- Random Forest: Algoritmo baseado em um comitê de árvores de decisão que operam de forma independente (bagging). Cada árvore vota em uma classe e a decisão final é tomada por maioria. É robusto para lidar com características não lineares e complexas, embora apresente maior custo computacional de armazenamento e processamento.


Métricas de Avaliação de Classificação:

- Acurácia: Proporção de predições corretas em relação ao total. É uma métrica frágil e enganosa se o conjunto de dados for desbalanceado (por exemplo, se 95% das mensagens forem saudações e apenas 5% cancelamentos).

- Precisão: Mede a exatidão das predições positivas do modelo. Responde à proporção de casos classificados como positivos que eram realmente verdadeiros, minimizando a ocorrência de falsos positivos.

- Recall (Revocação): Mede a sensibilidade do modelo. Responde à proporção de casos reais positivos que o modelo foi capaz de capturar, minimizando falsos negativos.

- F1-Score: Média harmônica balanceada entre Precisão e Recall. É a métrica mais confiável para validar chatbots, pois penaliza severamente o desequilíbrio entre falsos positivos e falsos negativos.


Etapa 4: Persistência de Modelos e Arquitetura de APIs

Após o treinamento bem-sucedido de um modelo de Machine Learning, é necessário persistir o estado de seus pesos matemáticos e dicionários para que possam ser utilizados em sistemas externos de forma ágil.

Arquitetura FastAPI: Framework moderno de Python utilizado para construir microsserviços de alta performance. Atua como o intermediário que recebe requisições de texto via HTTP POST, encapsula os dados usando tipos estruturados validados (Pydantic), aciona o arquivo binário do modelo em disco para processar a inferência e devolve as predições em formato padronizado JSON para sistemas Web, Mobile ou de Games.


Etapa 5: Otimização de Modelos (Análise Exploratória de Dados)

A otimização de modelos envolve o ajuste de parâmetros configurados antes do treino (hiperparâmetros) e a análise estatística prévia do dataset para garantir a qualidade preditiva do sistema.

- Hiperparâmetros: Valores de controle que ditam o comportamento do treinamento do modelo (por exemplo, a profundidade máxima de uma árvore de decisão ou a quantidade de estimadores em uma floresta). O ajuste incorreto desses valores pode levar o modelo ao sobreajuste (decorar os dados de treino e falhar com dados novos) ou subajuste (não aprender a lógica básica do problema).

- Validação Cruzada (K-Fold): Técnica estatística que divide o dataset em K partes iguais. O modelo é treinado K vezes, utilizando K-1 partes para o treino e a parte restante para validação. Esse processo garante que todo o conjunto de dados seja testado pelo menos uma vez, eliminando o viés de divisão única de treino e teste.

- Grid Search: Algoritmo que executa a busca exaustiva por força bruta testando todas as combinações possíveis de uma lista de hiperparâmetros fornecida pelo desenvolvedor. Ele avalia cada combinação utilizando a validação cruzada e seleciona a configuração ideal que obteve o maior desempenho médio.

- Análise Exploratória de Dados (EDA): Etapa de diagnóstico estatístico realizada antes da modelagem. Envolve a identificação de correlações, tratamento de valores nulos, preenchimento por imputação (utilizando média ou mediana dependendo da assimetria da coluna) e remoção de anomalias (outliers) que possam distorcer a generalização do modelo.


Etapa 6: Ciclo de Vida do Modelo, Telemetria e MLOps

O ciclo de desenvolvimento de um modelo não termina com o deploy da API. A entrada em produção exige monitoramento constante para evitar a perda de qualidade do sistema ao longo do tempo.

MLOps (Machine Learning Operations): Conjunto de práticas que une o desenvolvimento de modelos com operações de infraestrutura. Trata do ciclo automatizado de monitoramento, coleta de novos dados e re-treinamento periódico.

Degradação de Desempenho e Drift: O comportamento dos usuários em produção é dinâmico. Mudanças no vocabulário, surgimento de novas gírias ou alteração nos padrões de uso geram o fenômeno de Data Drift (quando a distribuição estatística das entradas de produção diverge dos dados usados no treino original), degradando a acurácia do chatbot.

Estratégias de Telemetria e Logging: Armazenamento contínuo das mensagens enviadas pelos usuários em produção, das intenções previstas e, criticamente, do score de confiança de cada inferência. Se a confiança calculada for menor que um limite estabelecido pelo desenvolvedor (ex: menor que 65%), o sistema sinaliza que o atendimento deve ser encaminhado para revisão manual ou intervenção humana.


Etapa 7: IA Ética e Modelos Explicáveis (XAI)

À medida que algoritmos tomam decisões automatizadas que afetam diretamente a vida dos usuários, a explicabilidade e a mitigação de vieses tornam-se imperativos legais e éticos.

Explicabilidade (Explainable AI - XAI): Quebra do paradigma da "caixa-preta". Envolve o uso de ferramentas matemáticas que elucidam os motivos internos que levaram o algoritmo a tomar determinada decisão, permitindo a auditoria humana e garantindo a conformidade com as leis de proteção de dados.

Feature Importance: Técnica que calcula a contribuição relativa de cada variável de entrada para a diminuição da impureza dos nós nas árvores de decisão. Permite ao desenvolvedor e ao usuário final entender quais fatores (por exemplo, nível de glicose ou idade) ditaram o resultado do modelo.

Viés Algorítmico e Equidade: Os algoritmos herdam os preconceitos e desbalanceamentos contidos nos dados de treinamento histórico. Se um grupo demográfico específico estiver sub-representado no dataset de treino, as previsões para esse grupo serão menos precisas, gerando discriminação sistemática e falhas no atendimento.


Etapa 8: Sistemas Conversacionais Baseados em Regras e Heurísticas

Antes de enviar dados textuais para processamento por modelos preditivos complexos, sistemas profissionais implementam uma camada de controle determinístico baseada em regras e fluxogramas explícitos.

Heurísticas e Árvores de Decisão Explícitas: Regras lógicas rígidas codificadas pelo programador que controlam o fluxo básico da conversa. São altamente estáveis e previsíveis, mas difíceis de manter à medida que o escopo do chatbot cresce.

Filtros Determinísticos com Regex (Expressões Regulares): Mecanismos de busca de padrões textuais exatos que atuam como a primeira barreira de segurança da aplicação. Em chatbots clínicos, são utilizados para interceptar termos que indiquem situações de emergência de vida de forma rápida e segura, antes que a mensagem seja enviada para classificação probabilística por IA.


Etapa 9: Chatbots Semânticos e Representações Vetoriais (Embeddings)

A evolução das interfaces conversacionais substitui as matrizes de contagem esparsas por representações densas em espaços contínuos multidimensionais de baixa dimensão gerados por redes neurais profundas.

- Text Embeddings: Mapeamento de sentenças em vetores matemáticos densos, onde a proximidade geométrica das coordenadas reflete diretamente a proximidade de significado semântico. Frases que usam sinônimos ou palavras totalmente diferentes, mas que transmitem a mesma ideia clínica, são projetadas para pontos vizinhos neste espaço matemático.

- Similaridade de Cosseno: Operação geométrica que mede o cosseno do ângulo formado entre dois vetores no espaço multidimensional. Como a métrica avalia a similaridade angular e não a magnitude linear, ela é imune a diferenças no comprimento do texto, focando exclusivamente no alinhamento do sentido semântico das duas sentenças.


3. Exemplo Prático de Estudo

O código Python abaixo integra de forma lógica e sequencial as etapas de desenvolvimento abordadas ao longo do semestre. Ele realiza a preparação do texto, vetorização estatística por TF-IDF, otimização de parâmetros com validação cruzada, análise de importância de atributos para explicabilidade, logging simulado de telemetria e o controle conversacional com filtros de segurança.


import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Garantindo os downloads necessarios do NLTK para processamento de texto
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ==============================================================================
# ETAPA 1: CRIAÇÃO DO DATASET E LIMPEZA DE TEXTO (NLP FUNDAMENTOS)
# ==============================================================================
# Simulando logs de conversas em uma triagem clinica hospitalar
data = {
    'mensagem': [
        "Sinto dor no peito muito forte e falta de ar",  # Emergência
        "Estou com palpitacoes violentas no coracao",   # Emergência
        "Gostaria de saber se o exame de sangue esta pronto",  # Informação
        "Como funciona a consulta de retorno?",  # Informação
        "Quero marcar um check-up com o clinico geral",  # Agendamento
        "Favor agendar consulta com pediatra para amanha", # Agendamento
        "Acho que quebrei o braco apos queda de escada",  # Emergência
        "Onde vejo o resultado do meu colesterol?",  # Informação
    ] * 6,  # Multiplicando para gerar volume de treino estavel
    'classe': [
        'emergencia', 'emergencia', 'informacao', 'informacao',
        'agendamento', 'agendamento', 'emergencia', 'informacao'
    ] * 6
}

df = pd.DataFrame(data)
stops_pt = set(stopwords.words('portuguese'))

def pipeline_limpeza_nlp(texto):
    """
    Funcao de processamento para normalizar, limpar ruidos, tokenizar
    e filtrar stopwords.
    """
    # 1. Normalizacao
    texto = texto.lower()
    
    # 2. Remocao de Ruido (Pontuacao)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenizacao
    tokens = word_tokenize(texto)
    
    # 4. Remocao de Stopwords
    tokens_limpos = [t for t in tokens if t not in stops_pt and t.isalnum()]
    
    return " ".join(tokens_limpos)

# Executando a limpeza de texto no dataset
df['mensagem_limpa'] = df['mensagem'].apply(pipeline_limpeza_nlp)

print("Etapa: Limpeza NLP")
print(f"Original: '{df['mensagem'].iloc[0]}'")
print(f"Limpo:    '{df['mensagem_limpa'].iloc[0]}'\n")


# ==============================================================================
# ETAPA 2: VETORIZAÇÃO TF-IDF E CLASSIFICAÇÃO SUPERVISIONADA
# ==============================================================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['mensagem_limpa'])
y = df['classe']

# Divisao de validacao garantindo dados ineditos para simular producao
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Etapa: Vetorizacao")
print(f"Formato da matriz de treino: {X_train.shape}\n")


# ==============================================================================
# ETAPA 3: OTIMIZAÇÃO DE PARÂMETROS COM VALIDAÇÃO CRUZADA (GRID SEARCH)
# ==============================================================================
rf_model = RandomForestClassifier(random_state=42)

# Definimos a grelha de busca para o Grid Search
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 3, 5, 10]
}

# Configuracao do Grid Search com 3-Fold Cross-Validation (K-Fold)
print("Etapa: Grid Search em Execucao...")
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Melhor combinacao encontrada: {grid_search.best_params_}")
print(f"Melhor pontuacao media de F1-Score: {grid_search.best_score_:.4f}\n")


# ==============================================================================
# ETAPA 4: AVALIAÇÃO DE PERFORMANCE E EXPLICABILIDADE (XAI)
# ==============================================================================
predicoes = best_model.predict(X_test)

print("Etapa: Avaliacao de Métricas:")
print(classification_report(y_test, predicoes, zero_division=0))

# Extracao de Feature Importance
importancias = best_model.feature_importances_
nomes_palavras = vectorizer.get_feature_names_out()

df_importancia = pd.DataFrame({
    'Palavra-Chave': nomes_palavras,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False).head(5)

print("Etapa: Explicabilidade do Modelo (Feature Importance - Top 5):")
for idx, row in df_importancia.iterrows():
    print(f" -> Palavra '{row['Palavra-Chave']}' dita {row['Importancia']*100:.2f}% da tomada de decisao.")
print("\n")


# ==============================================================================
# ETAPA 5: MOTOR DE INFERÊNCIA, TELEMETRIA E REGRAS DE SEGURANÇA (REGEX)
# ==============================================================================
def processar_chat_conversacional(mensagem_usuario):
    """
    Funcao simula o tratamento do chatbot clinico, aplicando controle 
    de seguranca por Regex, inferencia da IA e telemetria.
    """
    msg_low = mensagem_usuario.lower()
    
    # 1. Filtro Deterministico de Seguranca: Regex para Emergencias Graves
    padrao_grave = r"(peito|ar|infarto|desmaio|quebrei|sangue|parada|tonto)"
    if re.search(padrao_grave, msg_low):
        return (
            "HealthBot: Alerta de Emergência Crítica. "
            "Os sintomas informados necessitam de intervencao presencial imediata. "
            "Por favor, dirija-se ao pronto-socorro mais proximo ou ligue 192."
        )
    
    # 2. Se nao for emergência critica, prossegue para a classificacao estatistica
    texto_limpo = pipeline_limpeza_nlp(mensagem_usuario)
    vetor_mensagem = vectorizer.transform([texto_limpo])
    
    # Executamos a predicao
    intencao = best_model.predict(vetor_mensagem)[0]
    confianca = best_model.predict_proba(vetor_mensagem).max()
    
    # 3. Telemetria e MLOps: Logging de logs de producao
    status_auditoria = "Critico para Revisao Manual" if confianca < 0.65 else "Estavel"
    
    response = (
        f"HealthBot: Identifiquei que voce deseja realizar um(a) [{intencao.upper()}]. "
        f"Encaminhando para o setor correspondente. (Confianca: {confianca*100:.2f}%)"
    )
    
    print(f"Log Telemetria | Input: '{mensagem_usuario}' | Predicao: {intencao} | Score: {confianca:.2f} | Status: {status_auditoria}")
    
    return response

# --- EXECUÇÃO DE TESTES CONVERSACIONAIS NO CONSOLE ---
print("Etapa: Execucao de Testes Clinicos:")

# Teste de Filtro Deterministico (Regex)
print(processar_chat_conversacional("Minha mae esta com falta de ar e tontura"))
print("-" * 50)

# Teste de Inferência Estatistica (Classificador)
print(processar_chat_conversacional("Desejo saber se o resultado do meu colesterol ja saiu"))
print("-" * 50)

# Teste de Inferencia de Agendamento
print(processar_chat_conversacional("Quero marcar um horario com o doutor amanhã"))
print("-" * 50)
