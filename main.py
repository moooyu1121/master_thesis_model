import numpy as np 
import pandas as pd 
import pymarket as pm
import matplotlib.pyplot as plt
import pprint


if __name__ == "__main__":
    df = pd.read_csv('data/electricity.csv', parse_dates=[0], index_col=0)
    