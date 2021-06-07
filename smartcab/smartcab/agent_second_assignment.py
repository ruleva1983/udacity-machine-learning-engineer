import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator



class State(object):
    def __init__(self):
        self.traffic_light = None
        self.cars = None
        self.way_point = None
    def update(self, inputs, way_point):
        self.traffic_light = inputs['light']
        self.cars = {k: v for k, v in inputs.items() if k != 'lights'}
        self.way_point = way_point
    def reset(self):
        self.traffic_light = None
        self.cars = None
        self.way_point = None
    def __str__(self):
        return "Current system state: {} light, {} oncoming, {} from left, {} from right. Preferred next" \
               " way-point: {}".format(self.traffic_light, self.cars['oncoming'], self.cars['left'],
                                      self.cars['right'], self.way_point)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)
        self.color = 'red'
        self.planner = RoutePlanner(self.env, self)
        self.state = State()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.state.reset()

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state.update(inputs, self.next_waypoint)
        print self.state

        # TODO: Select action according to your policy

        action = self.next_waypoint

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track


    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1)  # run for a specified number of trials


    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
