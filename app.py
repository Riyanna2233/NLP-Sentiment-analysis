from flask import Flask, request, jsonify, send_from_directory
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return send_from_directory('', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']
    scores = sia.polarity_scores(text)
    return jsonify(scores)

if __name__ == '__main__':
    app.run(debug=True)
