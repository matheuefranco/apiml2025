import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import pickle
app = Flask(__name__)
CORS(app)
model = pickle.load(open("model.pkl", "rb"))
names = pickle.load(open("names.pkl", "rb"))


@app.route("/")
def home():
    return "API ITS WORKS!"


@app.route("/api/predict", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])
    output = names[pred[0]]
    return jsonify(output)