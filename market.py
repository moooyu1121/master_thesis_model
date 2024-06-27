import numpy as np 
import pandas as pd 
import pymarket as pm
from pymarket.bids.demand_curves import demand_curve_from_bids, supply_curve_from_bids
import matplotlib.pyplot as plt
import os
import logging
logger = logging.getLogger('Logging')
logger.setLevel(10)
fh = logging.FileHandler('market.log')
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s: line %(lineno)d: %(levelname)s: %(message)s')
fh.setFormatter(formatter)


class ModifiedPyMarket(pm.Market):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(self, ax=None):
        """
        Plots the demand and supply curves.
        """
        if ax is None:
            fig, ax = plt.subplots()

        df = self.bm.get_df()
        ax = plot_demand_curves(df, ax=ax)
        return ax
    

def plot_demand_curves(bids, ax=None, margin_X=1.2, margin_Y=1.2):
    """Plots the demand curves.
    If ax is none, creates a new figure

    Parameters
    ----------
    bids
          Collection of bids to be used

    ax : TODO, optional
         (Default value = None)
    margin_X :
         (Default value = 1.2)
    margin_Y :
         (Default value = 1.2)

    Returns
    -------


    """

    if ax is None:
        fig, ax = plt.subplots()

    extra_X = 3
    extra_Y = 1

    dc = demand_curve_from_bids(bids)[0]
    sp = supply_curve_from_bids(bids)[0]

    x_dc = dc[:, 0]
    x_dc = np.concatenate([[0], x_dc])
    x_sp = np.concatenate([[0], sp[:, 0]])

    y_sp = sp[:, 1]
    y_dc = dc[:, 1]
    max_x = max(x_dc[-2], x_sp[-2])
    extra_X = max_x * margin_X

    x_dc[-1] = extra_X
    y_dc = np.concatenate([y_dc, [0]])
    max_point = y_dc.max() * margin_Y

    x_sp[-1] = extra_X
    y_sp[-1] = max_point
    y_sp = np.concatenate([y_sp, [y_sp[-1]]])

    ax.step(x_dc, y_dc, where='post', c='r', label='Demand')
    ax.step(x_sp, y_sp, where='post', c='b', label='Supply')
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    ax.legend()
    return ax


def uniform_price_mechanism(bids: pd.DataFrame) -> (pm.TransactionManager, dict): # type: ignore
    """
    pymarketにはuniform price mechanismが実装されていないので、自前で実装する
    """
    trans = pm.TransactionManager()
    
    # Add some noise to the prices to find the intersection point
    bids['price'] = bids['price'] + np.random.uniform(-0.000001, 0.000001, bids.shape[0])
    
    buy, _ = pm.bids.demand_curve_from_bids(bids) # type: ignore # Creates demand curve from bids
    sell, _ = pm.bids.supply_curve_from_bids(bids) # type: ignore # Creates supply curve from bids

    # q_ is the quantity at which supply and demand meet
    # price is the price at which that happens
    # b_ is the index of the buyer in that position
    # s_ is the index of the seller in that position
    q_, b_, s_, price = pm.bids.intersect_stepwise(buy, sell, k=0) # type: ignore
    count = 0
    while b_ is None or s_ is None:
        logger.warning('intersection not found, adding bigger noise to the prices and trying again.')
        bids['price'] = bids['price'] + np.random.uniform(-0.00001, 0.00001, bids.shape[0])
        buy, _ = pm.bids.demand_curve_from_bids(bids) # type: ignore # Creates demand curve from bids
        sell, _ = pm.bids.supply_curve_from_bids(bids) # type: ignore # Creates supply curve from bids
        q_, b_, s_, price = pm.bids.intersect_stepwise(buy, sell, k=0) # type: ignore
        count += 1
        if count > 10:
            logger.error('intersection not found after 10 tries.')
            raise ValueError('intersection not found after 10 tries.')

    buying_bids  = bids.loc[bids['buying']].sort_values('price', ascending=False)
    selling_bids = bids.loc[~bids['buying']].sort_values('price', ascending=True)

    ## Filter only the trading bids.
    buying_bids = buying_bids.iloc[: b_ + 1, :]
    selling_bids = selling_bids.iloc[: s_ + 1, :]

    # Find the long side of the market
    buying_quantity = buying_bids.quantity.sum()
    selling_quantity = selling_bids.quantity.sum()


    if buying_quantity > selling_quantity:
        long_side = buying_bids
        short_side = selling_bids
    else:
        long_side = selling_bids
        short_side = buying_bids

    traded_quantity = short_side.quantity.sum()

    ## All the short side will trade at `price`
    ## The -1 is there because there is no clear 1 to 1 trade.
    for i, x in short_side.iterrows():
        t = (i, x.quantity, price, -1, False)
        trans.add_transaction(*t)

    ## The long side has to trade only up to the short side
    quantity_added = 0
    for i, x in long_side.iterrows():

        if x.quantity + quantity_added <= traded_quantity:
            x_quantity = x.quantity
        else:
            x_quantity = traded_quantity - quantity_added
        t = (i, x_quantity, price, -1, False)
        trans.add_transaction(*t)
        quantity_added += x.quantity

    extra = {
        'clearing quantity': q_,
        'clearing price': price
    }

    return trans, extra


class UniformPrice(pm.Mechanism):
    """
    Interface for our new uniform price mechanism.

    Parameters
    -----------
    bids
        Collection of bids to run the mechanism
        with.
    """

    def __init__(self, bids, *args, **kwargs):
        """TODO: to be defined1. """
        pm.Mechanism.__init__(self, uniform_price_mechanism, bids, *args, **kwargs)


class Market:
    def __init__(self, demand_list, supply_list, wholesale_price, BID_SAVE=False) -> None:
        """
        [[quantity1, price1, user1, buying], [quantity2, price2, user2, buying]...]
        の形式で需要と供給をリストで受け取る
        """
        self.demand_list = demand_list
        self.supply_list = supply_list
        self.whoelsale_price = wholesale_price
        # self.market = pm.Market()
        self.market = ModifiedPyMarket()
        self.BID_SAVE = BID_SAVE

    def bid(self):
        for i in range(len(self.demand_list)):
            if self.demand_list[i][0] > 0:
                self.market.accept_bid(self.demand_list[i][0], self.demand_list[i][1], self.demand_list[i][2], True)
        for i in range(len(self.supply_list)):
            if self.supply_list[i][0] > 0:
                self.market.accept_bid(self.supply_list[i][0], self.supply_list[i][1], self.supply_list[i][2], False)
        # import from grid and export to grid
        if self.BID_SAVE:
            self.market.accept_bid(0.2*len(self.demand_list), self.whoelsale_price, 99999, False)
            # self.market.accept_bid(0.1*len(self.supply_list), 0.01, 99999, True)
            pass
        else:
            self.market.accept_bid(1*len(self.demand_list), self.whoelsale_price, 99999, False)
            # self.market.accept_bid(1*len(self.supply_list), 0.01, 99999, True)
            pass

    def run(self, mechanism='uniform'):
        transactions, extras = self.market.run(mechanism)
        transactions_df = transactions.get_df()
        bids = self.market.bm.get_df()
        return transactions_df, extras
    
    def plot(self, title, episode, number, ax=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = self.market.plot(ax=ax)
        ax.set_title(title)
        os.makedirs('output/episode' + str(episode) + '/bid_image', exist_ok=True)
        fig.savefig(f"output/episode{episode}/bid_image/{number}.png")
        plt.close(fig)
        plt.clf()



if __name__ == "__main__":
    # Adding the new mechanism to the list of available mechanism of the market
    pm.market.MECHANISM['uniform'] = UniformPrice # type: ignore

    mar = ModifiedPyMarket() # Creates a new market
    # mar.accept_bid(quantity, price, user, buying)
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

    transactions, extras = mar.run('uniform') # run the uniform mechanism
    transactions_df = transactions.get_df()
    print(transactions_df)
    # pprint.pprint(extras)
    # stats=mar.statistics()
    # # pprint.pprint(statistics)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = mar.plot(ax=ax)
    fig.savefig("graph.png")
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax = mar.plot_method('huang', ax=ax)
    # fig.savefig("graph_huang.png")

    