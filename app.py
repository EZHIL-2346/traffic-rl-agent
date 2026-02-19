from flask import Flask, jsonify
import pickle
import numpy as np

app = Flask(__name__)

q_table = pickle.load(open("q_table.pkl", "rb"))

@app.route("/")
def home():
    return jsonify({"message": "Traffic RL Agent API is running"})

@app.route("/predict/<int:state>")
def predict(state):
    if state >= len(q_table) or state < 0:
        return jsonify({"error": "Invalid state"}), 400

    action = np.argmax(q_table[state])

    actions = {
        0: "Post Blog",
        1: "Share on Twitter",
        2: "Share on LinkedIn",
        3: "Do Nothing"
    }

    return jsonify({"best_action": actions[action]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
