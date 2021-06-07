from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from copy import deepcopy

import random
import numpy as np
import math
import matplotlib.pyplot as plt


class QTable(object):
    valid_actions = [None, 'forward', 'left', 'right']
    valid_traffic_light = ['red', 'green']
    valid_cars_directions = ['oncoming', 'right', 'left']
    valid_way_points = ['forward', 'left', 'right']

    def __init__(self, epsilon, gamma, learn_rate, lrate_decay=True, eps_decay=True):
        self.table = {}
        self.params = {'epsilon': epsilon, 'gamma': gamma, 'learn_rate': learn_rate,
                       'learn_rate_decay': lrate_decay, 'epsilon_decay': eps_decay}
        self.t = 0

    def provide_action(self, s):
        if str(s.state) not in self.table:
            self.table[str(s.state)] = {k: 0.0 for k in self.valid_actions}
            return random.choice(self.valid_actions)
        else:
            if random.uniform(0, 1) < self.params['epsilon']:
                return random.choice(self.valid_actions)
            else:
                actions = self.table[str(s.state)]
                return random.choice([a for a in actions if actions[a] == actions[max(actions, key=actions.get)]])

    def update(self, s, a, r, sp):
        assert str(s.state) in self.table
        assert str(sp.state) in self.table
        action_sub_table = self.table[str(sp.state)]
        maxQ = action_sub_table[max(action_sub_table, key=action_sub_table.get)]

        self.table[str(s.state)][a] = (1-self.params['learn_rate'])*self.table[str(s.state)][a]\
                                               + self.params['learn_rate'] *(r + self.params['gamma'] * maxQ)

        self.t += 1
        self.learn_rate_decay()
        self.epsilon_decay()

    def learn_rate_decay(self):
        # self.params['learn_rate'] *= float(self.t + 1)/(self.t + 2)
        self.params['learn_rate'] *= math.exp(-0.1)
        # self.params['learn_rate'] *= math.log(1+self.t)/math.log(2+self.t)

    def epsilon_decay(self):
        # self.params['epsilon'] *= float(self.t + 1) / (self.t + 2)
        self.params['epsilon'] *= math.exp(-0.1)



class State(object):
    def __init__(self):
        self.state = None

    def update(self, inputs, way_point, deadline):
        self.check_validity()
        self.state = inputs
        self.state['way_point'] = way_point
        #self.state['deadline'] = deadline

    def check_validity(self):
        pass

    def reset(self):
        self.state = None

    def __str__(self):
        return str(self.state)


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)
        self.color = 'red'
        self.planner = RoutePlanner(self.env, self)

        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None

        self.state = State()
        self.first_step = True

        self.table = None

        # Variables to monitor success/failure
        self.reached_destination = []
        self.trial = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.state.reset()

        self.trial += 1
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None

        self.state = State()
        self.first_step = True

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.previous_state = deepcopy(self.state)
        self.state.update(inputs, self.next_waypoint, deadline)

        # Select action according to your policy
        action = self.table.provide_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        if reward > 9:
            self.reached_destination.append(self.trial)

        # Learn policy based on state, action, reward
        if not self.first_step:
            self.table.update(self.previous_state, self.previous_action, self.previous_reward, self.state)

        self.first_step = False
        self.previous_action = action
        self.previous_reward = reward
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def set_Qtable(self, epsilon, gamma, learn_rate, learn_rate_decay, epsilon_decay):
        self.table = QTable(epsilon, gamma, learn_rate, learn_rate_decay, epsilon_decay)


def run(n_trials, params):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    a.set_Qtable(params['epsilon'], params['gamma'], params['learn_rate'],
                 params['learn_rate_decay'], params['epsilon_decay'])
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.000001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=n_trials)  # run for a specified number of trials

    return float(len(a.reached_destination)) / n_trials



if __name__ == '__main__':
    n_simulations = 200
    n_trials = 40

    counts = []
    params = {'epsilon': 0.1,'gamma': 0.1, 'learn_rate': 0.005, 'learn_rate_decay': True, 'epsilon_decay': True}
    for j in range(n_simulations):
        counts.append(run(n_trials, params))

    plt.title("Late stage learning")
    plt.xlabel("Fraction of times destination is reached")
    plt.ylabel("Count")
    plt.hist(counts, bins=6)
    plt.savefig("../Report/Q_learning_late_nodeadline_histogram_reached_destination.pdf")
    plt.show()

