from flask import Flask, jsonify, request
from code import get_prediction
app=Flask(__name__)
@app.route("/predict-letter",methods=["POST"])
def predict_data():
    image=request.files.get("letter")
    prediction=get_prediction(image)
    return jsonify({
        "prediction":prediction
    }),200

app.run()