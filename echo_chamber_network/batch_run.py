from model import EchoChamberModel
from mesa import batch_run
import numpy as np
import pandas as pd

# NOTE: You do not need this as a separate file BUT it can be nice to track
# can also call the file and it makes things a little cleaner as it runs

# Here you will have elements that you want to sweep, eg:
# parameters that will remain constant
# parameters you want to vary
parameters = {"num_agents": 20,
              "avg_degree": 3,
              "tolerance": 0,
              "num_recommended": 1,
              "num_neighbor_conn": 1,
              "schedule_type": "Random"
              } 

# what to run and what to collect
# iterations is how many runs per parameter value
# max_steps is how long to run the model
results = batch_run(EchoChamberModel, 
                    parameters,
                    iterations=3,  
                    max_steps=5, 
                    data_collection_period = 1) #how often do you want to pull the data





## NOTE: to do data collection, you need to be sure your pathway is correct to save this!
# Data collection
# extract data as a pandas Data Frame
pd.DataFrame(results).to_csv("batch_data.csv")