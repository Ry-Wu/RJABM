import mesa

from model import EchoChamberModel

def network_portrayal(G):
    def node_color(agent):
        if agent.opinion > 0.5:
            return "#FF0000"  # Red for positive opinions
        elif agent.opinion == 0.5:
            return "#808080"  # Grey for neutral opinions
        else:
            return "#008000"  # Green for negative opinions

    def edge_color(source, target):
        return "#e8e8e8"  # Standard color for all edges

    def edge_width(source, target):
        return 2

    def get_tooltip(agent):
        return f"Agent id: {agent.unique_id}<br>Opinion: {agent.opinion:.2f}<br>Tolerance: {agent.tolerance:.2f}"

    portrayal = {
        "nodes": [{"size": 6, "color": node_color(agent), "tooltip": get_tooltip(agent)} for (_, [agent]) in G.nodes.data('agent')],
        "edges": [{"source": source, "target": target, "color": edge_color(source, target), "width": edge_width(source, target)} for (source, target) in G.edges],
    }
    return portrayal

def get_num_of_clusters(model):
    num_of_clusters = model.num_clusters

    return "Number of Clusters: {}".format(
        num_of_clusters
    )
def get_rate_of_uniform(model):
    rate_uniform = model.rate_uniform_opinion_neighbors()

    return "Proportion of Agents Connected to Uniform Neighbors: {}".format(
        rate_uniform
    )
def get_opinion_homophily(model):
    opinion_homophily = model.datacollector.get_model_vars_dataframe()["Opinion_Homophily"].iloc[-1]
    return "Opinion Homophily: {:.2f}".format(opinion_homophily)

def get_opinion_modularity(model):
    opinion_modularity = model.datacollector.get_model_vars_dataframe()["Opinion_Modularity"].iloc[-1]
    return "Opinion Modularity: {:.2f}".format(opinion_modularity)

def get_clustering_coef(model):
    clustering_coef = model.datacollector.get_model_vars_dataframe()["Opinion_Clustering_Coefficient"].iloc[-1]
    return "Clustering Coefficient: {:.2f}".format(clustering_coef)

def get_radicalization(model):
    radicalization = model.datacollector.get_model_vars_dataframe()["Average_Radicalization"].iloc[-1]
    return "Average Radicalization: {:.2f}".format(radicalization)


network = mesa.visualization.NetworkModule(portrayal_method=network_portrayal,
                                           canvas_height=500, canvas_width=500)


model_params = {
    "num_agents": mesa.visualization.Slider("Number of agents", 15, 10, 200, 1),
    "avg_degree": mesa.visualization.Slider("Average Node Degree", 3, 1, 10, 1),
    "tolerance": mesa.visualization.Slider("Tolerance for Opinion Difference", 0.3, 0.0, 1.0, 0.05),
    "num_recommended": mesa.visualization.Slider("Number of Recommended Agents", 5, 0, 15, 1),
    "radical": mesa.visualization.Choice("Radical Recommendation?",value=False,
                                         choices=list([False, True])),
    "num_neighbor_conn": mesa.visualization.Slider("Number of Neighbor's Connections", 1, 1, 5, 1),
    "schedule_type": mesa.visualization.Choice("Scheduler type",value="Random",
                                                choices=list(EchoChamberModel.schedule_types.keys()))
}

server = mesa.visualization.ModularServer(model_cls=EchoChamberModel, 
                                          visualization_elements=[network, get_num_of_clusters,
                                                                  get_opinion_homophily,
                                                                  get_opinion_modularity,
                                                                  get_clustering_coef,
                                                                  get_rate_of_uniform,
                                                                  get_radicalization],
                                          name="Echo Chamber Model", 
                                          model_params=model_params)
server.port = 8521
