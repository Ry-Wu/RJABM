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

network = mesa.visualization.NetworkModule(portrayal_method=network_portrayal,
                                           canvas_height=500, canvas_width=500)

# chart = mesa.visualization.ChartModule([
#     {"Label": "Positive Opinion", "Color": "#FF0000"},
#     {"Label": "Negative Opinion", "Color": "#008000"},
# ])

model_params = {
    "num_agents": mesa.visualization.Slider("Number of agents", 15, 10, 200, 1),
    "avg_degree": mesa.visualization.Slider("Average Node Degree", 3, 1, 10, 1),
    "tolerance": mesa.visualization.Slider("Tolerance for Opinion Difference", 0.3, 0.0, 1.0, 0.05),
    "num_recommended": mesa.visualization.Slider("Number of Recommended Agents", 5, 1, 15, 1),
    "num_neighbor_conn": mesa.visualization.Slider("Number of Neighbor's Connections", 1, 1, 5, 1),
    "schedule_type": mesa.visualization.Choice("Scheduler type",value="Random",
                                                choices=list(EchoChamberModel.schedule_types.keys()))
}

server = mesa.visualization.ModularServer(model_cls=EchoChamberModel, 
                                          visualization_elements=[network, get_num_of_clusters],
                                          name="Echo Chamber Model", 
                                          model_params=model_params)
server.port = 8521
