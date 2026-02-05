# "Previsão de Preços"

#Passo 1:
pip install pandas matplotlib scikit-learn


#Passo 2:
import pandas as pd
import matplotlib.pyplot as plt

# Criando dados fictícios na hora para não depender de arquivos externos
data = {'area': [40, 50, 60, 70, 80, 90, 100], 
        'preco': [200, 250, 280, 350, 390, 440, 500]}
df = pd.DataFrame(data)

plt.scatter(df.area, df.preco)
plt.xlabel('Área (m²)')
plt.ylabel('Preço (R$ mil)')
plt.show()


#Passo 3: Treinando a IA
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
X = df[['area']] # O que a IA usa para aprender
y = df['preco']  # O que ela deve prever

modelo.fit(X, y) # momento do aprendizado
print("IA Treinada com sucesso!")


#Passo 4: Predição
area_nova = [[110]]
previsao = modelo.predict(area_nova)
print(f"Para uma casa de 110m², o valor estimado é: R$ {previsao[0]:.2f} mil")
