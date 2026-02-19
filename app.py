from flask import Flask, jsonify, render_template, request
import numpy as np

app = Flask(__name__)

q_table = np.load("q_table.npy")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    content = request.form["content"]
    time = request.form["time"]

    # Simple mapping (can later be replaced with dataset logic)
    state_map = {
        ("blog", "morning"): 5,
        ("video", "evening"): 10,
        ("post", "afternoon"): 3
    }

    state = state_map.get((content, time), 0)
    action = np.argmax(q_table[state])

    actions = {
        0: "Post Blog",
        1: "Share on Twitter",
        2: "Share on LinkedIn",
        3: "Do Nothing"
    }

    return render_template("index.html", result=actions[action])

@app.route("/predict/<int:state>")
def predict(state):
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
