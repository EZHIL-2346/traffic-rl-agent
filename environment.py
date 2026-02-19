import random

class TrafficEnvironment:
    def __init__(self):
        self.state = 100  # starting traffic

    def reset(self):
        self.state = 100
        return self.state

    def step(self, action):
        if action == 0:
            change = random.randint(5, 15)
        elif action == 1:
            change = random.randint(3, 10)
        elif action == 2:
            change = random.randint(4, 12)
        else:
            change = random.randint(-5, 2)

        self.state += change
        reward = change

        return self.state, reward
