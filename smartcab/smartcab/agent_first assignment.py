import random
import matplotlib.pyplot as plt
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.visited_locations = []
        self.times_reached_destination = 0

        self.reached_destination = []
        self.trial = 0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.trial += 1
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)

        deadline = self.env.get_deadline(self)

        # Update state
        self.visited_locations.append(list(self.env.agent_states[self]['location']))

        if self.env.agent_states[self]['location'] == self.planner.destination:
            self.times_reached_destination += 1

        # TODO: Select action according to your policy
        action = random.choice([None, 'forward', 'left', 'right'])

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward > 9:
            self.reached_destination.append(self.trial)



def run(n_trials):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()
    a = e.create_agent(LearningAgent)
    e.set_primary_agent(a, enforce_deadline=False)

    sim = Simulator(e, update_delay=0.00001, display=False)  # create simulator (uses pygame when display=True, if available)

    sim.run(n_trials=n_trials)

    #import matplotlib.pyplot as plt
    #plt.figure(1)
    #plt.title("Relative number of visits in vertical streets")
    #plt.xlabel("Street number")
    #plt.ylabel("Relative number of visits")
    #plt.xlim([1, 8])
    #plt.hist(np.array(a.visited_locations)[:,0], bins=8, normed=1)
    #plt.savefig("../Report/random_walk_histogram_X_visits.pdf")

    #plt.figure(2)
    #plt.title("Relative number of visits in horizontal streets")
    #plt.xlabel("Street number")
    #plt.ylabel("Relative number of visits")
    #plt.xlim([1, 6])
    #plt.hist(np.array(a.visited_locations)[:, 1], bins=6, normed=1)
    #plt.savefig("../Report/random_walk_histogram_Y_visits.pdf")

    #plt.show()

    return float(len(a.reached_destination))/n_trials


if __name__ == '__main__':
    counts = []
    for i in range(200):
        print "Run number ", i
        counts.append(run(n_trials=30))

    plt.xlabel("Fraction of times destination is reached")
    plt.ylabel("Count")
    plt.hist(counts, bins=10)
    #plt.savefig("../Report/random_walk_histogram_reached_destination.pdf")
    #plt.show()
