from enum import Enum
import random

import mesa
import networkx as nx

class OpinionState(Enum):
    NEGATIVE = 0
    NEUTRAL = 0.5
    POSITIVE = 1

class OpinionAgent(mesa.Agent):
    """
    Agent with an opinion and interactions based on tolerance to opinion differences.
    """
    def __init__(self, unique_id, model, opinion, tolerance):
        '''Each agent has an ID, an opinion state, and a tolerance'''
        super().__init__(unique_id, model)
        self.opinion = opinion
        self.tolerance = tolerance

    def step(self):
        '''At each step, each agent updates its opinion based on neighbors,
        and make new or break connections.'''
        print(self.pos)
        print(self.model.grid.get_neighbors(self.pos))
        self.update_opinion()
        self.break_connections()
        self.make_connections_neighbors()
        self.make_connections_recommended()

    def update_opinion(self):
        '''next opinion state is determined by the accumulation of pairwise
            interactions of agent x and all neighbors.
        '''
        neighbors = self.model.grid.get_neighbors(self.pos)
        
        for neighbor in neighbors:
            # if abs(neighbor.opinion - self.opinion) <= self.tolerance:
            self.opinion += self.tolerance * (neighbor.opinion - self.opinion)
            self.opinion = round(self.opinion, 1)

    def break_connections(self):
        '''break connections if pairwise opinion differences are larger than tolerance'''
        current_neighbors = self.model.grid.get_neighbors(self.pos)
        for neighbor in current_neighbors:
            if abs(neighbor.opinion - self.opinion) > self.tolerance:
                self.model.G.remove_edge(self.pos, neighbor.pos)

    def make_connections_neighbors(self):
        '''make new connections: neighbors' neighbors'''
        current_neighbors = self.model.grid.get_neighbors(self.pos)
        potential_connections = [a for a in self.model.schedule.agents if a not in current_neighbors]
        for agent in random.sample(potential_connections, min(len(potential_connections), self.model.num_neighbor_conn)):
            if abs(agent.opinion - self.opinion) < self.tolerance:
                self.model.G.add_edge(self.pos, agent.pos)

    def make_connections_recommended(self):
        '''make new connections: recommender system
        '''
        current_neighbors = self.model.grid.get_neighbors(self.pos)
        if self.opinion == 0.5:
            direction = OpinionState.POSITIVE if sum(1 for n in current_neighbors if n.opinion > 0.5) >= len(current_neighbors)/2 else OpinionState.NEGATIVE
        else:
            direction = OpinionState.POSITIVE if self.opinion > 0.5 else OpinionState.NEGATIVE

        potential_connections = [a for a in self.model.schedule.agents if a not in current_neighbors and a.opinion == direction]
        for agent in random.sample(potential_connections, min(len(potential_connections), self.model.num_recommended)):
            if abs(agent.opinion - self.opinion) < self.tolerance:
                self.model.G.add_edge(self.pos, agent.pos)


class EchoChamberModel(mesa.Model):
    def __init__(self, num_agents=20, avg_degree=3, tolerance=0.3, num_recommended=5, num_neighbor_conn=1):
        super().__init__()
        self.num_agents = num_agents
        self.avg_degree = avg_degree
        self.tolerance = tolerance
        self.num_recommended = num_recommended
        self.num_neighbor_conn = num_neighbor_conn
        self.G = nx.erdos_renyi_graph(n=self.num_agents, p=self.avg_degree/(self.num_agents-1))
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)
        self.init_agents()

    def init_agents(self):
        for i, node in enumerate(self.G.nodes()):
            opinion = random.uniform(0, 1)
            a = OpinionAgent(i, self, opinion, self.tolerance)
            self.schedule.add(a)
            self.grid.place_agent(a, node)

    def step(self):
        self.schedule.step()

