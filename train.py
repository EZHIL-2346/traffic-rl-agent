from environment import TrafficEnvironment
from rl_agent import QLearningAgent
import pickle

env = TrafficEnvironment()
agent = QLearningAgent()

for episode in range(1000):
    state = env.reset()

    for step in range(50):
        action = agent.choose_action(state)
        next_state, reward = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state

pickle.dump(agent.q_table, open("q_table.pkl", "wb"))
print("Training complete")
import numpy as np

np.save("q_table.npy", agent.q_table)

print("Model saved successfully!")

