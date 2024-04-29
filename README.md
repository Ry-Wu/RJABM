# RJABM - ABM on Echo Chambers

## Summary

We are building a model that investigates how the tolerance for opinion difference and the recommender system influence the formation of opinion clusters and the emergence of echo chambers over time.

JavaScript library used in this example to render the network: [d3.js](https://d3js.org/).

## Installation

To install the dependencies use pip and the requirements.txt in this directory. e.g.

```
    pip install -r requirements.txt
```

## How to Run

To run the model interactively, run ``mesa runserver`` in this directory. e.g.

```
    mesa runserver
```

Then open your browser to [http://127.0.0.1:8521/](http://127.0.0.1:8521/) and press Reset, then Run.

or

Directly run the file ``run.py`` in the terminal. e.g.

```
    python run.py
```


## Files

* ``run.py``: Launches a model visualization server.
* ``model.py``: Contains the agent class, and the overall model class.
* ``server.py``: Defines classes for visualizing the model (network layout) in the browser via Mesa's modular server, and instantiates a visualization server.

## Further Reading

Deffuant, G., Neau, D., Amblard, F., and Weisbuch, G. (2000). Mixing beliefs among interacting agents. Adv. Comp. Syst., 03(01n04):87–98.
