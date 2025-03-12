import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Carregar dados
df = pd.read_csv('../inputs/dados_sorvetes.csv')

# Separar variáveis independentes e dependentes
X = df[['temperatura']]  # Temperatura
y = df['vendas']  # Vendas

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Previsão
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio: {mse}")

# Salvar o modelo treinado
with open('../models/modelo_sorvete.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo treinado e salvo com sucesso!")
