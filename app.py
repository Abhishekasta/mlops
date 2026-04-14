from flask import Flask, request, jsonify
import pickle

import os
import subprocess

if not os.path.exists('model/model.pkl'):
    print("model not found, training...")
    subprocess.run(["python", "src/train.py"])

app = Flask(__name__)

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    input_data = [list(data.values())]

    prediction = model.predict(input_data)

    return jsonify({'price': prediction[0]})

app.run(host='0.0.0.0', port=5000)