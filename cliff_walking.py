from math import *
from numpy import *
from random import *
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
from gridworld import q_to_arrow


N_ROWS = 6
N_COLUMNS = 10

class State(object):
    def __init__(self, i, j, is_cliff=False, is_goal=False):
        self.i = i
        self.j = j
        self.is_cliff = is_cliff
        self.is_goal = is_goal
        #             north, east, south, west
        self.q_values = np.array([0.0, 0.0, 0.0, 0.0])

    def __str__(self):
        return '({}, {})'.format(self.i, self.j)

    def is_terminal(self):
        return self.is_goal or self.is_cliff

    def get_max_q_index(self):
        best_q_values = np.argwhere(self.q_values == np.max(self.q_values))
        if len(best_q_values) > 1:
            return best_q_values[randint(0, len(best_q_values) - 1)][0]
        else:
            _max_q = np.argmax(self.q_values)
            return _max_q

    def get_max_q_value(self):
        return np.max(self.q_values)


def initialize_states():
    # This is the set of states, all initialised with default values
    states = [[State(j, i) for i in range(N_COLUMNS)] for j in range(N_ROWS)]

    # make the cliff
    for j in range(1, N_COLUMNS - 1):
      states[-1][j].is_cliff = True

    states[-1][-1].is_goal = True
    return states


# The reward function defines what reward I get for transitioning between the first and second state
def reward(s_1, s_2):
  if (s_1.is_goal or s_1.is_cliff):
    return 0
  elif (s_2.is_goal):
    return 10
  elif (s_2.is_cliff):
    return -100
  else:
    return -1

""" the transition function takes state and action and results in a new state, depending on their attributes. The method takes the whole state-space as an argument (since the transition depends on the attributes of the states in the state-space), which could for example be the "states" matrix from above, the current state s from the state-space (with its attributes), and the current action, which takes the form of a "difference vector. For example, dx = 0, dy = 1 means: Move to the south. dx = -1, dy = 0 means: Move to the left"""
def transition(stsp, s, di, dj):
  if (s.is_cliff or s.is_goal):
    return s
  elif (s.j + dj not in range(N_COLUMNS) or s.i + di not in range(N_ROWS)):
    return s
  else:
    return stsp[s.i + di][s.j + dj]

gamma = 1
learning_rate = 0.01

def action_to_diff_vector(action):
    if action == 0:  # NORTH
        return -1, 0
    elif action == 1:  # EAST
        return 0, 1
    elif action == 2:  # SOUTH
        return 1, 0
    elif action == 3:  # WEST
        return 0, -1

def action_to_verbose(action):
    if action == 0:
        return 'NORTH'
    elif action == 1:
        return 'EAST'
    elif action == 2:
        return 'SOUTH'
    elif action == 3:
        return 'WEST'


def sarsa(state, next_state, action, next_state_action):
    return reward(state, next_state), state.q_values[action] +\
            learning_rate * (reward(state, next_state) + gamma * next_state.q_values[next_state_action] - state.q_values[action])


def q_learning(state, next_state, action, next_state_action):
    next_state_q_value = next_state.get_max_q_value()
    return reward(state, next_state), state.q_values[action] +\
            learning_rate * (reward(state, next_state) + gamma * next_state_q_value - state.q_values[action])

N_STEPS = 10000
METHOD = 'BOTH'
EPSILONS = [0.05, 0.1, 0.25]

def run_code(use_q_learning=False, _epsilon=0.01):
    states = initialize_states()
    decay = 1
    min_epsilon = 0.00001
    epsilon = _epsilon

    episode_rewards = []
    mistakes_array = []  # array which tracks error from convergence on each step
    for i in range(N_STEPS):
        # select a random starting state
        current_state = states[N_ROWS-1][0]

        # iterate until reaching a terminal state
        epsilon = max(min_epsilon, epsilon * decay)
        episode_reward = 0
        while not current_state.is_terminal():

            if random() < epsilon:
                next_action = randint(0, 3)
            else:
                next_action = current_state.get_max_q_index()

            di, dj = action_to_diff_vector(next_action)
            next_state = transition(states, current_state, di, dj)

            if random() < epsilon:
                next_state_action = randint(0, 3)
            else:
                next_state_action = next_state.get_max_q_index()

            if use_q_learning:
                reward, current_state.q_values[next_action] = q_learning(current_state, next_state, next_action, next_state_action)
            else:
                reward, current_state.q_values[next_action] = sarsa(current_state, next_state, next_action, next_state_action)

            # print(current_state, next_state, action_to_verbose(next_action), di, dj)
            episode_reward += reward
            current_state = next_state
        if len(episode_rewards):
            episode_rewards.append(episode_rewards[-1] + episode_reward)
        else:
            episode_rewards.append(episode_reward)

        '''
        if (i % 100 == 0):
            print(i)
        '''
        mistakes_array.append(check_accuracy(states))

    return np.array(mistakes_array), states, episode_rewards

def check_accuracy(states):
    correct_result = np.array([
        [-3, -2, -1, 0 , 1 , 2 , 3 , 4 , 5 , 6  ],
        [-2, -1, 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7  ],
        [-1, 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8  ],
        [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9  ],
        [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ],
        [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0  ],
    ])
    mistakes_delta = 0
    for i in range(N_ROWS):
        for j in range(N_COLUMNS):
            mistakes_delta += abs(correct_result[i][j] - max(states[i][j].q_values))

    return mistakes_delta

def plot_errors(mistakes_sarsa, mistakes_q_learning):
    plt.gca().invert_yaxis()
    legend = []
    for mistake_sarsa in mistakes_sarsa:
        plt.plot(mistake_sarsa[1])
        legend.append(r'SARSA $\epsilon={}$'.format(mistake_sarsa[0]))
    for mistake_q_learning in mistakes_q_learning:
        plt.plot(mistake_q_learning[1])
        legend.append(r'Q-learning $\epsilon={}$'.format(mistake_q_learning[0]))

    plt.grid(which='y')
    plt.legend(legend)

    plt.savefig('CLIFF_SARSA_VS_Q_LEARNING_{}.png'.format(N_STEPS))
    # plt.show()

def plot_best_q_values_states(states, method, epsilon, PLOTS, fig, ax):
    final_grid = np.array([[max(states[i][j].q_values) for j in range(N_COLUMNS)] for i in range(N_ROWS)])
    if PLOTS > 2:
        ax = ax[PLOTS % 3, 1]
    else:
        ax = ax[PLOTS, 0]
    ax.imshow(final_grid, aspect='auto', cmap='coolwarm')
    # fig, ax = plt.subplots()
    ax.set_xticks(np.arange(N_COLUMNS))
    ax.set_yticks(np.arange(N_ROWS))
    ax.set_xticklabels([i for i in range(N_COLUMNS)])
    ax.set_yticklabels([i for i in range(N_ROWS)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(N_ROWS):
        for j in range(N_COLUMNS):
            text = ax.text(j, i, '{:.2f}'.format(max(states[i][j].q_values)),
                           ha="center", va="center", color="w")

    fig.tight_layout()
    ax.set_title("{}; $\epsilon={}$".format(method, epsilon))
    for i in range(N_ROWS):
        str_ = ""
        for j in range(N_COLUMNS):
            str_ += str(int(final_grid[i][j])) + ", "
    PLOTS += 1
    # plt.savefig('CLIFF_WALKING: {}-{}-{}.png'.format(N_STEPS, epsilon, method))
    # plt.show()

def display_optimal_policy(states, method, epsilon):

    print("{}; ε = {}".format(method, epsilon))
    print('-' * 60)
    for i in range(len(states)):
        line_str = ''
        for j in range(len(states[0])):
            if j == 0:
                print('|', end='')
            if states[i][j].is_goal:
                print(Back.GREEN + '   ', end='')
                print(Style.RESET_ALL + ' | ', end='')
            elif states[i][j].is_cliff:
                print(Back.RED + '   ', end='')
                print(Style.RESET_ALL + ' | ', end='')
            else:
                print(' {}  | '.format(q_to_arrow(states[i][j].get_max_q_index())), end='')
        print(line_str)
        print('-' * 60)

if METHOD not in ['Q_LEARNING', 'SARSA', 'BOTH']:
    print('invalidt method. must be Q_LEARNING or SARSA or both')
    import sys; sys.exit()

mistakes_q_learning = []
mistakes_sarsa = []
PLOTS = 0
fig, axes = plt.subplots(3, 2)
rewards = []
for epsilon in EPSILONS:
    if METHOD == 'Q_LEARNING' or METHOD == 'BOTH':
        _mistakes_q_learning, end_states_q_learning, episode_rewards = run_code(use_q_learning=True, _epsilon=epsilon)
        plot_best_q_values_states(end_states_q_learning, 'Q_LEARNING', epsilon, PLOTS, fig, axes)
        display_optimal_policy(end_states_q_learning, 'Q LEARNING', epsilon)
        mistakes_q_learning.append((epsilon, _mistakes_q_learning))
        rewards.append(('Q_LEARNING', epsilon, episode_rewards))
    PLOTS += 1

for epsilon in EPSILONS:
    if METHOD == 'SARSA' or METHOD == 'BOTH':
        _mistakes_sarsa, end_states_sarsa, episode_rewards = run_code(use_q_learning=False, _epsilon=epsilon)
        plot_best_q_values_states(end_states_sarsa, 'SARSA', epsilon, PLOTS, fig, axes)
        display_optimal_policy(end_states_sarsa, 'SARSA', epsilon)
        mistakes_sarsa.append((epsilon, _mistakes_sarsa))
        rewards.append(('SARSA', epsilon, episode_rewards))
    PLOTS += 1


plt.savefig('all_runs.png')
plt.show()
# for i, j  in [(0, 3), (1, 4), (2, 5)]:
for reward in rewards:
    # plt.plot(rewards[i][2], 'o', label='{} ε = {} '.format(rewards[i][0], rewards[i][1]))
    # plt.plot(rewards[j][2], 'o', label='{} ε = {} '.format(rewards[j][0], rewards[j][1]))
    plt.plot(reward[2], label='{} ε = {} '.format(reward[0], reward[1]))
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
plt.legend()
plt.show()
plt.savefig('episode_rewards.png')

plot_errors(mistakes_sarsa, mistakes_q_learning)
