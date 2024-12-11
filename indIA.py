import os
import random

# NumPy version: # 1.23.5
import numpy as np

# Gymnasium version: 0.29.1
from gymnasium import Env
from gymnasium import spaces

# Stable Baselines version: 2.3.2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

equations_txt_file_path = os.path.join(os.path.dirname(__file__), "links", "equations_nerdle.txt")

class indIA_Env(Env):
    def __init__(self):
        self.all_equations = self.load_equations(equations_txt_file_path)
        self.action_space = spaces.Discrete(len(self.all_equations))
        self.observation_space = spaces.MultiDiscrete([15] * 8)
        self.state = np.array([0, 10, 1, 2, 13, 3, 14, 4])
        self.attempts = 0
        self.correct_positions = np.zeros(8, dtype=bool)
        self.previous_states_set = set()
        self.max_steps = 6
        self.char_to_int = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '+': 10, '-': 11, '*': 12, '/': 13, '=': 14
        }
        
        self.equation = self.select_random_equation()

    def int_to_char(self, num):
        int_to_char_map = {v: k for k, v in self.char_to_int.items()}
        return int_to_char_map.get(num, '?')

    def load_equations(self, file_path):
        with open(file_path, 'r') as file:
            equations = file.readlines() 
        return [eq.strip() for eq in equations]

    def select_random_equation(self):
        equation = random.choice(self.all_equations)
        formatted_equation = [self.char_to_int[char] for char in equation if char in self.char_to_int]
        
        return np.array(formatted_equation)

    def step(self, action):
        selected_equation = self.all_equations[action]
        selected_equation_encoded = [self.char_to_int[char] for char in selected_equation]

        done = False
        reward = 0
        self.attempts += 1

        for position, new_value in enumerate(selected_equation_encoded):
            if self.correct_positions[position]:
                reward += 5
            else:
                self.state[position] = new_value
                if self.state[position] == self.equation[position]:
                    if not self.correct_positions[position]:
                        reward += 50
                        self.correct_positions[position] = True
                else:
                    reward -= 1

        state_hash = hash(tuple(self.state))
        if state_hash in self.previous_states_set:
            reward -= 20
        else:
            reward += 10
            self.previous_states_set.add(state_hash)

        if np.all(self.correct_positions):
            remaining_attempts = self.max_steps - self.attempts
            if remaining_attempts >= 4:
                reward += 900
            elif remaining_attempts == 3:
                reward += 400
            elif remaining_attempts == 2:
                reward += 300
            elif remaining_attempts == 1:
                reward += 200
            else:
                reward += 100

            done = True

        if self.attempts >= self.max_steps:
            reward -= 500
            done = True

        info = {}
        return self.state, reward, done, False, info


    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.array([0, 10, 1, 2, 13, 3, 14, 4])
        self.attempts = 0
        self.correct_positions = np.zeros(8, dtype=bool)
        self.previous_states_set = set()
        self.equation = self.select_random_equation()
        return self.state, {}

    def render(self, mode="human"):
        state_as_chars = [self.int_to_char(num) for num in self.state]
        print(f"State: {''.join(state_as_chars)}, Attempts: {self.attempts}, Correct positions: {self.correct_positions}")

env = indIA_Env()

check_env(env, warn=True)
eval_callback = EvalCallback(
    env, 
    best_model_save_path='./logs/', 
    log_path='./logs/', 
    eval_freq=10000,
    n_eval_episodes=10
)

policy_kwargs = dict(net_arch=[1024, 512, 256, 128, 64])

model_path = os.path.join(os.path.dirname(__file__), "links", "indIA.zip")
if os.path.exists(model_path):
    print(">> Loading pre-created agent <<")
    indIA = PPO.load(model_path, env=env)
else:
    print(">> Creating a new model <<")
    indIA = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-5,
        n_steps=4096,
        batch_size=256,
        n_epochs=30,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs
    )

indIA.learn(total_timesteps=100000, callback=eval_callback)

indIA.save(model_path)

episodes = 10000
results = []

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    score = 0
    attempts = 0
    while not done:
        action, _states = indIA.predict(obs)
        obs, reward, done, _, info = env.step(action)
        score += reward 
        attempts = env.attempts
    results.append((ep + 1, attempts, score))

print("\nResultats dels episodis:")
for episode, attempts, score in results:
    print(f"Episodi {episode} - {attempts+1} Intents - {score} Recompensa")


filtered_results = [(episode, attempts, score) for episode, attempts, score in results if attempts < 6]
print("\nEpisodis amb 6 intents o menys")
for episode, attempts, score in filtered_results:
    print(f"Episodi {episode} - {attempts+1} Intents - {score} Recompensa")