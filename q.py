from agent import Agent
import pandas as pd
import numpy as np


class Q:
    def __init__(self, params, agents_params_df, num_dizitized_pv_ratio, num_dizitized_soc):
        self.agents_params_df = agents_params_df
        agent_num = agents_params_df.shape[0]
        self.possible_params = Agent(agent_num).generate_params()
        self.params = params
        self.num_dizitized_pv_ratio = num_dizitized_pv_ratio
        self.num_dizitized_soc = num_dizitized_soc

    def reset_qtb(self):
        dr_buy_rows = self.num_dizitized_pv_ratio * len(self.possible_params['elastic_ratio_list']) * \
                len(self.possible_params['dr_price_threshold_list']) * \
                len(self.possible_params['alpha_list']) * \
                len(self.possible_params['beta_list'])
        battery_buy_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc * \
                len(self.possible_params['battery_capacity_list']) * \
                len(self.possible_params['gamma_list']) * \
                len(self.possible_params['epsilon_list'])
        battery_sell_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc * \
                len(self.possible_params['battery_capacity_list']) * \
                len(self.possible_params['gamma_list']) * \
                len(self.possible_params['epsilon_list'])
        ev_battery_buy_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc * \
                len(self.possible_params['ev_capacity_list']) * \
                len(self.possible_params['psi_list']) * \
                len(self.possible_params['omega_list'])
        ev_battery_sell_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc * \
                len(self.possible_params['ev_capacity_list']) * \
                len(self.possible_params['psi_list']) * \
                len(self.possible_params['omega_list'])
        cols = int(self.params['price_max']) - int(self.params['price_min']) + 1
        self.dr_buy_qtb = np.full((dr_buy_rows, cols), 1000.0)
        self.battery_buy_qtb = np.full((battery_buy_rows, cols), 1000.0)
        self.battery_sell_qtb = np.full((battery_sell_rows, cols), 1000.0)
        self.ev_battery_buy_qtb = np.full((ev_battery_buy_rows, cols), 1000.0)
        self.ev_battery_sell_qtb = np.full((ev_battery_sell_rows, cols), 1000.0)

    def get_qtbs(self):
        return self.dr_buy_qtb, self.battery_buy_qtb, self.battery_sell_qtb, self.ev_battery_buy_qtb, self.ev_battery_sell_qtb

    def battery_bids(self, agent_num, price_min, price_max):
        battery_capacity = self.agents_params_df.at[agent_num, 'battery_capacity']
        sell_price = (price_min + price_max) / 2
        buy_price = price_min + 0.01
        return 1
    

if __name__ == '__main__':
    agents_params_df = pd.DataFrame(1.0, index=np.arange(10), columns=['shift_limit', 
                                                                        'elastic_ratio', 
                                                                        'dr_price_threshold', 
                                                                        'battery_capacity', 
                                                                        'ev_capacity', 
                                                                        'alpha', 
                                                                        'beta',
                                                                        'gamma', 
                                                                        'epsilon',
                                                                        'psi',
                                                                        'omega',])
    params = {'price_max': 120,
              'price_min': 5,
              'wheeling_charge': 10,
              'battery_charge_efficiency': 0.9,
              'battery_discharge_efficiency': 0.9,
              'ev_charge_efficiency': 0.9,
              'ev_discharge_efficiency': 0.9,
    }
    q = Q(params, agents_params_df, num_dizitized_pv_ratio=20, num_dizitized_soc=20)
    q.reset_qtb()
    qtbs = q.get_qtbs()
    print(qtbs[4].shape)