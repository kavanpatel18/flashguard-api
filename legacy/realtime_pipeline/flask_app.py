from flask import Flask, render_template, jsonify
import random
import time
import os

app = Flask(__name__)

# Basic Mock state for the UI
def get_mock_state():
    return {
        "timestamp": time.strftime('%H:%M:%S'),
        "crash_prob": round(random.uniform(0.01, 0.15), 4) if random.random() > 0.05 else round(random.uniform(0.7, 0.99), 4),
        "micro_price": round(22000 + random.uniform(-10, 10), 2),
        "ofi": round(random.uniform(-500, 500), 1),
        "depth_ratio": round(random.uniform(0.5, 2.0), 2),
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stream')
def stream():
    return jsonify(get_mock_state())

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, port=5000)
