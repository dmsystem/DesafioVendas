# Previsão de Vendas de Sorvetes com Machine Learning

## 🌭 Sobre o Projeto
Este projeto tem como objetivo prever as vendas de sorvetes com base na temperatura ambiente utilizando um modelo de Regressão Linear. O modelo é treinado com dados fictícios e pode ser utilizado para prever a demanda de sorvetes em um determinado dia, otimizando a produção e evitando desperdícios.

## 🔮 Tecnologias Utilizadas
- **Python**: Linguagem de programação principal do projeto.
- **pandas**: Manipulação e análise de dados.
- **scikit-learn**: Construção e treinamento do modelo de Regressão Linear.
- **MLflow**: Gerenciamento do ciclo de vida do modelo, versionamento e monitoramento de métricas.
- **Flask**: Framework para criar a API de previsão em tempo real.

## 📚 Estrutura do Repositório
```
/previsao-vendas-sorvetes
├── /inputs
│   └── dados_sorvetes.csv  # Dados de entrada para treinamento
├── /models
│   └── modelo_sorvete.pkl  # Modelo treinado salvo
├── /src
│   ├── main.py             # Script para treinamento e avaliação do modelo
│   ├── mlflow_log.py       # Script para registrar o modelo no MLflow
│   └── predict_api.py      # API Flask para previsão em tempo real
└── README.md               # Documento de descrição do projeto
```

## 🚀 Como Rodar o Projeto
### 1. Instalar Dependências
Antes de executar o projeto, instale as dependências necessárias. Recomenda-se utilizar um ambiente virtual.
```bash
# Criar ambiente virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinar o Modelo
Para treinar o modelo de previsão de vendas de sorvetes com base na temperatura, execute:
```bash
python src/main.py
```
O modelo treinado será salvo na pasta `/models` como `modelo_sorvete.pkl`.

### 3. Registrar o Modelo no MLflow
Para registrar o modelo e acompanhar as métricas no MLflow, execute:
```bash
python src/mlflow_log.py
```
O servidor MLflow pode ser acessado localmente em:
```
http://127.0.0.1:5000
```

### 4. Executar a API de Previsão em Tempo Real
Para iniciar a API Flask e obter previsões em tempo real, execute:
```bash
python src/predict_api.py
```
A API estará rodando localmente e você pode fazer uma requisição POST para `/predict`:

#### Exemplo de Requisição (POST para /predict):
```json
{
  "temperatura": 30
}
```
#### Exemplo de Resposta:
```json
{
  "vendas_estimadas": 170.35
}
```

## 📊 Resultados
O modelo de regressão linear foi treinado utilizando dados de temperatura e vendas de sorvetes. A performance foi avaliada com o Erro Quadrático Médio (MSE). Exemplo de cálculo:
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio: {mse}")
```
Quanto menor o MSE, melhor a precisão do modelo.

## 🌟 Como Contribuir
Contribuições são bem-vindas! Para sugerir melhorias, abra uma *pull request* ou registre uma *issue*.

## 🛡️ Licença
Este projeto está licenciado sob a MIT License - veja o arquivo LICENSE para mais detalhes.

