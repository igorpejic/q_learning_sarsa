from math import *
from numpy import *
from random import *
import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore, Back, Style


class State(object):
    def __init__(self, i, j, is_black=False, is_red=False, is_green=False):
        self.i = i
        self.j = j
        self.is_black = is_black
        self.is_red = is_red
        self.is_green = is_green
        #             north, east, south, west
        self.q_values = np.array([0.0, 0.0, 0.0, 0.0])

    def __str__(self):
        return '({}, {})'.format(self.i, self.j)

    def is_invalid_starting_space(self):
        if self.is_black or self.is_red or self.is_green:
            return True

    def is_terminal(self):
        return self.is_red or self.is_green

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
    states = [[State(j, i) for i in range(8)] for j in range(8)]

    # Now I make the walls black
    for j in range(2,6):
      states[1][j].is_black = True

    for j in range(1, 4):
      states[6][j].is_black = True

    for i in range(2, 5):
      states[i][5].is_black = True

    states[5][4].is_red = True
    states[7][7].is_green = True
    return states


# Valuable if we want to "look at" the state-space (next for methods accomplish that)
def project_black(s):
  result = [[int(s[i][j].is_black) for j in range(8)] for i in range(8)]
  return result

def project_red(s):
    result = [[int(s[i][j].is_red) for j in range(8)] for i in range(8)]
    return result

def project_green(s):
    result = [[int(s[i][j].is_green) for j in range(8)] for i in range(8)]
    return result

def project_complete(s):
  result = [["" for j in range(8)] for i in range(8)]
  for i in range(8):
    for j in range(8):
      if (s[i][j].is_black):
        result[i][j] = "b"
      elif (s[i][j].is_red):
        result[i][j] = "r"
      elif (s[i][j].is_green):
        result[i][j] = "g"
      else:
        result[i][j] = "-"
  return result

# The reward function defines what reward I get for transitioning between the first and second state
def reward(s_1, s_2):
  if (s_1.is_red or s_1.is_green):
    return 0
  elif (s_2.is_red):
    return -20
  elif (s_2.is_green):
    return 10
  else:
    return -1

""" the transition function takes state and action and results in a new state, depending on their attributes. The method takes the whole state-space as an argument (since the transition depends on the attributes of the states in the state-space), which could for example be the "states" matrix from above, the current state s from the state-space (with its attributes), and the current action, which takes the form of a "difference vector. For example, dx = 0, dy = 1 means: Move to the south. dx = -1, dy = 0 means: Move to the left"""
def transition(stsp, s, di, dj):
  if (s.is_red or s.is_green):
    return s
  elif (s.j + dj not in range(8) or s.i + di not in range(8)):
    return s
  elif (stsp[s.i + di][s.j + dj].is_black):
    return s
  else:
    return stsp[s.i + di][s.j + dj]

gamma = 1
learning_rate = 0.1

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
    return state.q_values[action] +\
            learning_rate * (reward(state, next_state) + gamma * next_state.q_values[next_state_action] - state.q_values[action])


def q_learning(state, next_state, action, next_state_action):
    next_state_q_value = next_state.get_max_q_value()
    return state.q_values[action] +\
            learning_rate * (reward(state, next_state) + gamma * next_state_q_value - state.q_values[action])
            

def run_code(use_q_learning=False):
    states = initialize_states()
    epsilon = 0.9
    decay = 0.999
    min_epsilon = 0.000000001

    mistakes_array = []  # array which tracks error from convergence on each step
    for i in range(10000):
        # select a random starting state
        current_state = states[randint(0, 7)][randint(0, 7)]
        while current_state.is_invalid_starting_space():
            current_state = states[randint(0, 7)][randint(0, 7)]

        # iterate until reaching a terminal state
        epsilon = max(min_epsilon, epsilon * decay)
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
                current_state.q_values[next_action] = q_learning(current_state, next_state, next_action, next_state_action)
            else:
                current_state.q_values[next_action] = sarsa(current_state, next_state, next_action, next_state_action)

            # print(current_state, next_state, action_to_verbose(next_action), di, dj)
            current_state = next_state

        if (i % 100 == 0):
            print(i)
        mistakes_array.append(check_accuracy(states))

    return np.array(mistakes_array), states

def check_accuracy(states):
    correct_result = np.array([
        [-3, -2, -1, 0, 1, 2, 3, 4],
        [-2, -3, 0, 0, 0, 0, 4, 5],
        [-1, -2, -3, -4, -5, 0, 5, 6],
        [0, -1, -2, -3, -4, 0, 6, 7],
        [1, 0, -1, -2, -3, 0, 7, 8],
        [2, 1, 0, -1, 0, 7, 8, 9],
        [3, 0, 0, 0, 7, 8, 9, 10],
        [4, 5, 6, 7, 8, 9, 10, 0],
    ])
    mistakes_delta = 0
    for i in range(8):
        for j in range(8):
            mistakes_delta += abs(correct_result[i][j] - max(states[i][j].q_values))

    return mistakes_delta

def plot_best_q_values_states(states):
    final_grid = np.array([[max(states[j][i].q_values) for i in range(8)] for j in range(8)])
    fig, ax = plt.subplots()
    im = ax.imshow(final_grid, cmap='coolwarm')
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels([i for i in range(8)])
    ax.set_yticklabels([i for i in range(8)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, '{:.2f}'.format(max(states[i][j].q_values)),
                           ha="center", va="center", color="w")

    fig.tight_layout()
    plt.title("Best q values for each cell")
    for i in range(8):
        str_ = ""
        for j in range(8):
            str_ += str(int(final_grid[i][j])) + ", "
    plt.show()


def q_to_arrow(index):
    if index == 0:
        return '▲'
    elif index == 1:
        return '►'
    elif index == 2:
        return '▼'
    else:
        return '◄'

def display_optimal_policy(states):

    print('-' * 40)
    for i in range(len(states)):
        line_str = ''
        for j in range(len(states[0])):
            if j == 0:
                print('|', end='')
            if states[i][j].is_black:
                print(Back.WHITE + '   ', end='')
                print(Style.RESET_ALL + ' |', end='')
            elif states[i][j].is_red:
                print(Back.RED + '   ', end='')
                print(Style.RESET_ALL + ' |', end='')
            elif states[i][j].is_green:
                print(Back.GREEN + '   ', end='')
                print(Style.RESET_ALL + ' |', end='')
            else:
                print(' {} | '.format(q_to_arrow(states[i][j].get_max_q_index())), end='')
        print(line_str)
        print('-' * 40)


if __name__ == '__main__':
    mistakes_q_learning, end_states_q_learning = run_code(use_q_learning=True)
    # mistakes_sarsa, end_states_sarsa = run_code(use_q_learning=False)

    # plot_best_q_values_states(end_states_sarsa)
    display_optimal_policy(end_states_q_learning)
    plt.gca().invert_yaxis()
    #plt.plot(mistakes_q_learning)
    # plt.plot(mistakes_sarsa)
    #plt.grid(which='y')
    #plt.legend(['mistakes Q-learning', 'mistakes SARSA'])
    #plt.show()
