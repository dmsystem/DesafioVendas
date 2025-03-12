import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar dados
df = pd.read_csv('../inputs/dados_sorvetes.csv')

# Separar variáveis independentes e dependentes
X = df[['temperatura']]  # Temperatura
y = df['vendas']  # Vendas

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iniciar experimento no MLflow
mlflow.start_run()

# Treinamento do modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Previsão
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio: {mse}")

# Registrar modelo e métrica no MLflow
mlflow.sklearn.log_model(model, "modelo_sorvete")
mlflow.log_metric("mse", mse)

# Finalizar experimento
mlflow.end_run()

print(f"Modelo registrado no MLflow com erro MSE: {mse}")
