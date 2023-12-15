import numpy as np
import gymnasium as gym
env = gym.make("Taxi-v3")


class QLearningSolver:
    """Class containing the Q-learning algorithm that might be used for different discrete environments."""

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        q_table: np.ndarray = None
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        if q_table is None:
            self.q_table = np.zeros(shape=(observation_space, action_space))
        else:
            self.q_table = q_table

    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        """Return Q-value of given state and action."""
        return self.q_table[state][action]

    def update(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        """Update Q-value of given state and action."""
        self.q_table[state][action] += reward

    def get_best_action(self, state: np.ndarray) -> int:
        """Return action that maximizes Q-value for a given state."""
        return np.argmax(self.q_table[state])
    
    def get_best_move_evaluation(self, state: np.array) ->float:
        return np.max(self.q_table[state])

    def __repr__(self):
        """Elegant representation of Q-learning solver."""
        pass

    def __str__(self):
        return self.__repr__()



def run_episode(solver: QLearningSolver, environment, max_steps):
    state = environment.reset()[0]
    terminated, truncated = False, False
    number_of_steps = 0
    
    while not terminated and not truncated and number_of_steps < max_steps:
        if np.random.random() < solver.epsilon:
            action = environment.action_space.sample()
        else:
            action = solver.get_best_action(state)
            
        next_state, reward, terminated, truncated, info = environment.step(action)
        delta = reward + solver.gamma * solver.get_best_move_evaluation(next_state) - solver(state, action)
        solver.update(state, action, solver.learning_rate * delta)
        state = next_state
        number_of_steps += 1

def q_learning(environment, learning_rate = 0.1, epsilon = 0.1,  max_steps = 200, gamma = 0.9, number_of_episodes = 200):
    solver = QLearningSolver(environment.observation_space.n, environment.action_space.n, learning_rate, gamma, epsilon)
    for i in range(number_of_episodes):
        run_episode(solver, environment, max_steps)
        if i % 100 == 0:
          print(f"Episode: {i}")
    return solver
 
env = gym.make("Taxi-v3", render_mode="human")
solver = QLearningSolver(env.observation_space.n, env.action_space.n, q_table=np.load("solver.npy"))
 

observation, info = env.reset()
for _ in range(100):
   action = solver.get_best_action(observation)  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   if terminated or truncated:
      observation, info = env.reset()
env.close()



