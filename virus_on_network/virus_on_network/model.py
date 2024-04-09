import math
from enum import Enum

import mesa
import networkx as nx


class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RESISTANT = 2


def number_state(model, state):
    return sum(1 for a in model.grid.get_all_cell_contents() \
    if not isinstance(a, HospitalAgent) and a.state is state)


def number_infected(model):
    return number_state(model, State.INFECTED)


def number_susceptible(model):
    return number_state(model, State.SUSCEPTIBLE)


def number_resistant(model):
    return number_state(model, State.RESISTANT)


class VirusOnNetwork(mesa.Model):
    """
    A virus model with some number of agents
    """

    def __init__(
        self,
        num_nodes=10,
        avg_node_degree=3,
        initial_outbreak_size=1,
        virus_spread_chance=0.4,
        virus_check_frequency=0.4,
        recovery_chance=0.3,
        gain_resistance_chance=0.5,

        #hospitals
        num_hospital = 1,
        range_hospital = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_hospital = num_hospital
        prob = avg_node_degree / (self.num_nodes + self.num_hospital)
        self.G = nx.erdos_renyi_graph(n=self.num_nodes + self.num_hospital, p=prob)
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)
        self.initial_outbreak_size = (
            initial_outbreak_size if initial_outbreak_size <= num_nodes else num_nodes
        )
        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

        self.range_hospital = range_hospital

        self.datacollector = mesa.DataCollector(
            {
                "Infected": number_infected,
                "Susceptible": number_susceptible,
                "Resistant": number_resistant,
            }
        )

        # Create agents
        for i, node in enumerate(self.G.nodes()):
            if i < self.num_nodes:
                a = VirusAgent(
                    i,
                    self,
                    State.SUSCEPTIBLE,
                    self.virus_spread_chance,
                    self.virus_check_frequency,
                    self.recovery_chance,
                    self.gain_resistance_chance,
                    self.range_hospital,
                )
                self.schedule.add(a)
                # Add the agent to the node
                self.grid.place_agent(a, node)
            else:
                b = HospitalAgent('H' + str(i), self, self.num_hospital, self.range_hospital)
                self.schedule.add(b)
                self.grid.place_agent(b, node)


        # Infect some nodes
        infected_nodes = self.random.sample(list(self.G), self.initial_outbreak_size)
        for a in self.grid.get_cell_list_contents(infected_nodes):
            a.state = State.INFECTED

        self.running = True
        self.datacollector.collect(self)

        # # Create Hospital Agents
        # existing_nodes = set(self.grid.get_all_cell_contents())
        # available_nodes = [node for node in self.G.nodes() if node not in existing_nodes]

        # for _ in range(self.num_hospital):
        #     if available_nodes:
        #         chosen_node = self.random.choice(available_nodes)
        #         hospital_agent = HospitalAgent('H' + str(chosen_node), self, 
        #                                         self.num_hospital, self.range_hospital)
        #         self.schedule.add(hospital_agent)
        #         self.grid.place_agent(hospital_agent, chosen_node)
        #         available_nodes.remove(chosen_node)
        #     else:
        #         # Handle case where there are no available nodes
        #         pass

    def resistant_susceptible_ratio(self):
        try:
            return number_state(self, State.RESISTANT) / number_state(
                self, State.SUSCEPTIBLE
            )
        except ZeroDivisionError:
            return math.inf

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()

class HospitalAgent(mesa.Agent):
    """
    Individual Agent definition and its properties/interaction methods
    """

    def __init__(
        self,
        unique_id,
        model,
        num_hospital,
        range_hospital
    ):
        super().__init__(unique_id, model)

        self.num_hospital = num_hospital
        self.range_hospital = range_hospital


    def step(self):
        pass

class VirusAgent(mesa.Agent):
    """
    Individual Agent definition and its properties/interaction methods
    """

    def __init__(
        self,
        unique_id,
        model,
        initial_state,
        virus_spread_chance,
        virus_check_frequency,
        recovery_chance,
        gain_resistance_chance,
        range_hospital,
    ):
        super().__init__(unique_id, model)

        self.state = initial_state

        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance
        self.range_hospital = range_hospital

    def try_to_infect_neighbors(self):
        neighbors_nodes = self.model.grid.get_neighborhood(self.pos, include_center=False)
        susceptible_neighbors = [a for a in self.model.grid.get_cell_list_contents(neighbors_nodes) \
                                 if isinstance(a, VirusAgent) and a.state is State.SUSCEPTIBLE]
        for a in susceptible_neighbors:
            if self.random.random() < self.virus_spread_chance:
                a.state = State.INFECTED

    def try_gain_resistance(self):
        if self.random.random() < self.gain_resistance_chance:
            self.state = State.RESISTANT

    def try_remove_infection(self):
        neighbors_nodes = self.model.grid.get_neighborhood(self.pos, 
                                                           include_center=False, 
                                                           radius=self.range_hospital)

        for a in self.model.grid.get_cell_list_contents(neighbors_nodes):
            if isinstance(a, HospitalAgent):
                self.state = State.SUSCEPTIBLE
                self.try_gain_resistance()
                return

        if self.random.random() < self.recovery_chance:
            self.state = State.SUSCEPTIBLE
            self.try_gain_resistance()
        else:
            self.state = State.INFECTED


    def try_check_situation(self):
        if (self.random.random() < self.virus_check_frequency) and (
            self.state is State.INFECTED
        ):
            return self.try_remove_infection()


    def treat_in_hospital(self):
        # Logic when treated by a hospital
        # Example: the agent recovers and gains resistance
        self.state = State.SUSCEPTIBLE
        self.try_gain_resistance()

    def step(self):
        if self.state is State.INFECTED:
            self.try_to_infect_neighbors()
        self.try_check_situation()


