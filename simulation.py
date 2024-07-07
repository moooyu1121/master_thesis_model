import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import pymarket as pm
from tqdm import tqdm
import glob
import warnings
warnings.simplefilter('ignore', FutureWarning)
import visualize
from preprocess import Preprocess
from market import Market, UniformPrice
from agent import Agent
from q import Q
import logging
logger = logging.getLogger('Logging')
logger.setLevel(10)
fh = logging.FileHandler('main.log')
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s: line %(lineno)d: %(levelname)s: %(message)s')
fh.setFormatter(formatter)

class Simulation:
    def __init__(self, num_agent, parent_dir, **kwargs) -> None:
        self.num_agent = num_agent
        self.episode = 1
        # Adding the new mechanism to the list of available mechanism of the market
        pm.market.MECHANISM['uniform'] = UniformPrice # type: ignore
        # Update market and uniform parameters
        params = {'thread_num': -1,
                  'price_max': 120,
                  'price_min': 0,
                  'wheeling_charge': 10,
                  'battery_charge_efficiency': 0.9,
                  'battery_discharge_efficiency': 0.9,
                  'ev_charge_efficiency': 0.9,
                  'ev_discharge_efficiency': 0.9,
        }
        params.update(kwargs)
        self.thread_num = params['thread_num']
        self.price_max = params['price_max']
        self.price_min = params['price_min']
        self.wheeling_charge = params['wheeling_charge']
        self.battery_charge_efficiency = params['battery_charge_efficiency']
        self.battery_discharge_efficiency = params['battery_discharge_efficiency']
        self.ev_charge_efficiency = params['ev_charge_efficiency']
        self.ev_discharge_efficiency = params['ev_discharge_efficiency']

        # Initialize Q table
        self.q = Q(params, agent_num=num_agent, num_dizitized_pv_ratio=20, num_dizitized_soc=20)
    
    def load_existing_q_table(self, path):
        self.q.load_q_table(folder_path=path)
