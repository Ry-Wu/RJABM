from enum import Enum
import random
import numpy as np

import mesa
import networkx as nx
from networkx.algorithms.components import connected_components


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

        agents_rec_probs = {}
        prob_sum = 0
        for a in self.model.schedule.agents:
            if a not in current_neighbors:
                rec_prob = 1 / (abs(a.opinion - self.opinion) + 1)
                agents_rec_probs[a] = rec_prob
                prob_sum += rec_prob

        
        agents_select_probs =  {agent: prob / prob_sum for agent, prob in agents_rec_probs.items()}
        
        selected_agents = np.random.choice(list(agents_select_probs.keys()), 
                                                size=min(len(agents_select_probs), self.model.num_recommended), 
                                                p=list(agents_select_probs.values()), 
                                                replace=False)

        for agent in selected_agents:
            if abs(agent.opinion - self.opinion) < self.tolerance:
                self.model.G.add_edge(self.pos, agent.pos)


        # potential_connections = [a for a in self.model.schedule.agents if a not in current_neighbors and a.opinion == direction]
        # for agent in random.sample(potential_connections, min(len(potential_connections), self.model.num_recommended)):
        #     if abs(agent.opinion - self.opinion) < self.tolerance:
        #         self.model.G.add_edge(self.pos, agent.pos)


class EchoChamberModel(mesa.Model):

    schedule_types = {
        "Sequential": mesa.time.BaseScheduler,
        "Random": mesa.time.RandomActivation,
        "Simultaneous": mesa.time.SimultaneousActivation,
    }
    
    def __init__(self, num_agents=20, avg_degree=3, tolerance=0.3, 
                 num_recommended=5, num_neighbor_conn=1, schedule_type="Random"):
        super().__init__()
        self.num_agents = num_agents
        self.avg_degree = avg_degree
        self.tolerance = tolerance
        self.num_recommended = num_recommended
        self.num_neighbor_conn = num_neighbor_conn
        self.G = nx.erdos_renyi_graph(n=self.num_agents, p=self.avg_degree/(self.num_agents-1))
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)
        self.num_clusters = nx.number_connected_components(self.G)
        self.schedule_type = schedule_type
        self.schedule = self.schedule_types[self.schedule_type](self)
        self.init_agents()

        self.datacollector = mesa.DataCollector(
            {
                "Number_of_Clusters": lambda m: nx.number_connected_components(m.G)
            }
        )
        self.running = True
        self.datacollector.collect(self)

    def init_agents(self):
        for i, node in enumerate(self.G.nodes()):
            opinion = random.uniform(0, 1)
            a = OpinionAgent(i, self, opinion, self.tolerance)
            self.schedule.add(a)
            self.grid.place_agent(a, node)

    def step(self):
        
        self.schedule.step()
        self.num_clusters = nx.number_connected_components(self.G)
        self.datacollector.collect(self)
        print(f"Model Step: {self._steps} at Model Time: {self._time}")

