from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Carregar o modelo treinado
with open('../models/modelo_sorvete.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receber dados da requisição
    temperatura = data['temperatura']
    prediction = model.predict([[temperatura]])  # Realizar previsão
    return jsonify({'vendas_estimadas': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
