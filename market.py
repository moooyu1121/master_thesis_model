import numpy as np 
import pandas as pd 
import pymarket as pm
import matplotlib.pyplot as plt
import pprint


if __name__ == "__main__":
    mar = pm.Market() # Creates a new market

    buyers_names=['CleanRetail','El4You','EVcharge','QualiWatt','IntelliWatt']
    mar.accept_bid(250,200,0,True) # CleanRetail 0 
    mar.accept_bid(300,110,1,True) #El4You 1 
    mar.accept_bid(120,100,2,True) # EVcharge 2 
    mar.accept_bid(80, 90,3,True) #QualiWatt 3 
    mar.accept_bid(40, 85,4,True) # IntelliWatt 4 
    mar.accept_bid(70, 75,1,True) #El4You 5 
    mar.accept_bid(60, 65,0,True) # CleanRetail 6 
    mar.accept_bid(45, 40,4,True) #IntelliWatt 7 
    mar.accept_bid(30, 38,3,True) # QualiWatt 8 
    mar.accept_bid(35, 31,4,True) #IntelliWatt 9 
    mar.accept_bid(25, 24,0,True) # CleanRetail 10 
    mar.accept_bid(10, 21,1,True) #El4You 11

    sellers_names=['RT','WeTrustInWind','BlueHydro','KøbenhavnCHP','DirtyPower','SafePeak']
    mar.accept_bid(120, 0,5,False) #RT 12 
    mar.accept_bid(50, 0,6,False) #WeTrustInWind 13 
    mar.accept_bid(200, 15,7,False) #BlueHydro 14 
    mar.accept_bid(400, 30,5,False) #RT 15 
    mar.accept_bid(60,32.5,8,False) #KøbenhavnCHP 16 
    mar.accept_bid(50, 34,8,False) #KøbenhavnCHP 17 
    mar.accept_bid(60, 36,8,False) #KøbenhavnCHP 18 
    mar.accept_bid(100,37.5,9,False) #DirtyPower 19 
    mar.accept_bid(70, 39,9,False) #DirtyPower 20 
    mar.accept_bid(50, 40,9,False) #DirtyPower 21 
    mar.accept_bid(70, 60,5,False) #RT 22 
    mar.accept_bid(45, 70,5,False) #RT 23 
    mar.accept_bid(50, 100,10,False) #SafePeak 24 
    mar.accept_bid(60, 150,10,False) #SafePeak 25 
    mar.accept_bid(50, 200,10,False) #SafePeak 26

    bids = mar.bm.get_df()
    print(bids)

    transactions, extras = mar.run('huang') # run the huang mechanism
    transactions_df = transactions.get_df()
    print(transactions_df)
    pprint.pprint(extras)
    stats=mar.statistics()
    # # pprint.pprint(statistics)
    # mar.plot()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = mar.plot_method('huang', ax=ax)
    fig.savefig("graph")