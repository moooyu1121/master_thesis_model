import numpy as np 
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import pymarket as pm
from tqdm import tqdm
import random
import warnings
warnings.simplefilter('ignore', FutureWarning)
import visualize
from preprocess import Preprocess
from market import Market, UniformPrice
from agent import Agent
from q import Q
import capex_opex
import logging
logger = logging.getLogger('Logging')
logger.setLevel(10)
fh = logging.FileHandler('main.log')
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s: line %(lineno)d: %(levelname)s: %(message)s')
fh.setFormatter(formatter)

class Simulation:
    def __init__(self, num_agent, parent_dir, episode, train, **kwargs) -> None:
        os.makedirs(parent_dir, exist_ok=True)
        self.num_agent = num_agent
        self.parent_dir = parent_dir
        self.episode = episode
        self.train = train
        # Adding the new mechanism to the list of available mechanism of the market
        pm.market.MECHANISM['uniform'] = UniformPrice # type: ignore
        # Update market and uniform parameters
        params = {'thread_num': -1,
                  'BID_SAVE': False,
                  'price_max': 110,
                  'price_min': 10,
                  'wheeling_charge': 0,
                  'battery_charge_efficiency': 0.9,
                  'battery_discharge_efficiency': 0.9,
                  'ev_charge_efficiency': 0.9,
                  'ev_discharge_efficiency': 0.9,
                  'battery_capacity_list': [0, 10, 15, 20],
                  'ev_capacity_list': [40],
                  'pv_capacity_list': [0, 5, 10],
                  'discount_rate': 0.99,
                  'learning_rate': 0.1,
                  'shift_limit_list': [6.0, 12.0, 18.0, 24.0],  # hours
                  'max_battery_charge_speed': [3.0],  # kW
                  'max_battery_discharge_speed': [3.0],  # kW
                  'max_ev_charge_speed': [6.0],  # kW
                  'max_ev_discharge_speed': [3.0],  # kW
                  'dr_boolean_list': [True, False],
                  'alpha_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
                  'beta_list': [1, 1.5, 2, 2.5, 3, 3.5, 4]
                #   'gamma_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
                #   'epsilon_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
                #   'psi_list': [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
                #   'omega_list': [1, 1.5, 2, 2.5, 3, 3.5, 4]
        }
        params.update(kwargs)
        self.params = params
        self.thread_num = params['thread_num']
        self.BID_SAVE = params['BID_SAVE']
        self.price_max = params['price_max']
        self.price_min = params['price_min']
        self.wheeling_charge = params['wheeling_charge']
        self.battery_charge_efficiency = params['battery_charge_efficiency']
        self.battery_discharge_efficiency = params['battery_discharge_efficiency']
        self.ev_charge_efficiency = params['ev_charge_efficiency']
        self.ev_discharge_efficiency = params['ev_discharge_efficiency']
        self.battery_capacity_list = params['battery_capacity_list']
        self.ev_capacity_list = params['ev_capacity_list']
        self.pv_capacity_list = params['pv_capacity_list']
        self.discount_rate = params['discount_rate']
        self.learning_rate = params['learning_rate']

        # Initialize Q table
        self.q = Q(params, agent_num=num_agent, num_dizitized_pv_ratio=5, num_dizitized_soc=5, num_elastic_ratio_pattern=3)
    
    def load_existing_q_table(self, folder_path):
        self.q.load_q_table(folder_path=folder_path)

    def remove_existing_q_table(self, folder_path):
        self.q.remove_q_table_saved_data(folder_path=folder_path)

    def preprocess(self):
        # Generate agent parameters
        self.agents = Agent(self.num_agent)
        self.agents.generate_params(self.params, seed=self.thread_num)
        for agent_id in range(self.num_agent):
            battery_capacity, ev_capacity, pv_capacity = self.q.get_facility_capacities(agent_id, episode=self.episode-1, is_train=self.train)
            self.agents.set_one_agent(agent_id, battery_capacity=battery_capacity, ev_capacity=ev_capacity, pv_capacity=pv_capacity)
        self.agents.save(self.parent_dir)
        agent_params_df = self.agents.get_agents_params_df_

        # Preprocess and generate demand, price, and car_movement(boolean) data
        preprocess = Preprocess(seed=self.thread_num)
        preprocess.set(
            pd.read_csv('data/demand.csv'),
            pd.read_csv('data/supply.csv'),
            pd.read_csv('data/price.csv'),
            pd.read_csv('data/ev_charging_bool.csv'),
            pd.read_csv('data/ev_move_consumption.csv'),
            pd.read_csv('data/elastic_ratio.csv')
        )
        # preprocess.generate_d_s(self.num_agent)
        preprocess.generate_demand(self.num_agent)
        pv_capacity_list = agent_params_df['pv_capacity'].values
        # Generate supply data
        preprocess.generate_supply_flex_pv_size(self.num_agent, pv_capacity_list)
        # Generate car charge data
        preprocess.generate_car_charge(self.num_agent)
        preprocess.save(self.parent_dir)
        preprocess.drop_index_  # drop timestamp index
        self.demand_df, self.supply_df, self.price_df, self.car_charge_df, self.car_move_consumption_df, self.elastic_ratio_df = preprocess.get_dfs_

        # get average pv production ratio to get state in Q table, indicating the solar radiation
        # data is stored as kWh/kW, which means, the values are within 0~1
        pv_ratio_df = pd.read_csv('data/supply.csv', index_col=0)
        pv_ratio_df['mean'] = pv_ratio_df.mean(axis=1)
        self.pv_ratio_arr = pv_ratio_df['mean'].values

        # Initialize record arrays
        self.grid_import_record_arr = np.full(len(self.price_df), 0.0)
        self.microgrid_price_record_arr = np.full(len(self.price_df), 0.0)
        self.ev_battery_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.ev_battery_soc_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.battery_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.battery_soc_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_inelastic_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_elastic_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_shifted_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_battery_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_ev_battery_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.sell_pv_record_arr = np.full((len(self.supply_df), self.num_agent), 0.0)
        self.sell_battery_record_arr = np.full((len(self.supply_df), self.num_agent), 0.0)
        self.sell_ev_battery_record_arr = np.full((len(self.supply_df), self.num_agent), 0.0)
        self.reward_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.electricity_cost_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.potential_demand_arr = np.full(len(self.demand_df), 0.0)
        self.potential_supply_arr = np.full(len(self.supply_df), 0.0)

        # set initial ev battery state to 50% of its capacity
        for i in range(self.num_agent):
            self.ev_battery_record_arr[0, i] = self.agents[i]['ev_capacity'] / 2
            if self.agents[i]['ev_capacity'] != 0:
                self.ev_battery_soc_record_arr[0, i] = self.ev_battery_record_arr[0, i] / self.agents[i]['ev_capacity']
            else:
                self.ev_battery_soc_record_arr[0, i] = 0.0
        
        # Generate elastic and inelastic demand according to the elastic ratio of each agent
        self.demand_elastic_arr = self.demand_df.values.copy()
        self.demand_inelastic_arr = self.demand_df.values.copy()
        for i in range(self.num_agent):
            if self.agents[i]['dr_boolean'] == False:
                self.demand_elastic_arr[:, i] = 0
                self.demand_inelastic_arr[:, i] = self.demand_df[f'{i}']
            elif self.agents[i]['dr_boolean'] == True:
                self.demand_elastic_arr[:, i] = self.demand_df[f'{i}'] * self.elastic_ratio_df["elastic_ratio"]
                self.demand_inelastic_arr[:, i] = self.demand_df[f'{i}'] * (1 - self.elastic_ratio_df["elastic_ratio"])
            else:
                raise ValueError("DR boolean key is invalid.")

        # Prepare dataframe to record shifted demand
        # shift_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        self.shift_arr = np.full((len(self.demand_df), self.num_agent), 0.0)

    def run(self):
        for t in tqdm(range(len(self.demand_df))):
            demand_list = []
            supply_list = []
            potential_demand = 0
            potential_supply = 0
            wholesale_price = self.price_df.at[t, 'Price'] + self.wheeling_charge
            self.q.reset_all_digitized_states()
            self.q.reset_all_actions()
            for i in range(self.num_agent):
                #============================================================================================================================================================
                self.q.set_digitized_states(agent_id=i, agent_params=self.agents[i], 
                                            pv_ratio=self.pv_ratio_arr[t], 
                                            battery_soc=self.battery_soc_record_arr[t, i], 
                                            ev_battery_soc=self.ev_battery_soc_record_arr[t, i], 
                                            elastic_ratio=self.elastic_ratio_df.at[t, "elastic_ratio"])
                # Qテーブルから行動を取得, ε-greedy法で徐々に最適行動を選択する式が、エピソード0から始まるように定義されているので、エピソード-1を引数に渡す
                self.q.set_actions(agent_id=i, episode=self.episode-1, is_train=self.train)
                # 時刻tでのバッテリー残量を時刻t+1にコピー、取引が行われる場合あとでバッテリー残量をさらに更新
                # t+1でのcar_charge_dfがFalseのとき、car_move_consumption_dfの値を引く
                # EVバッテリー残量が負の値になる場合もここではそのままにして、報酬を計算するフェーズで対応、0に更新するとともに-1000を報酬に反映
                if t+1 != len(self.demand_df):
                    self.battery_record_arr[t+1, i] = self.battery_record_arr[t, i]
                    if self.agents[i]['battery_capacity'] != 0:
                        self.battery_soc_record_arr[t+1, i] = self.battery_record_arr[t+1, i] / self.agents[i]['battery_capacity']
                    else:
                        self.battery_soc_record_arr[t+1, i] = 0.0
                    if self.agents[i]['ev_capacity'] != 0:
                        self.ev_battery_record_arr[t+1, i] = self.ev_battery_record_arr[t, i]
                        if ~self.car_charge_df.at[t+1, f'{i}']:
                            self.ev_battery_record_arr[t+1, i] = self.ev_battery_record_arr[t, i] - self.car_move_consumption_df.at[t+1, f'{i}']
                        self.ev_battery_soc_record_arr[t+1, i] = self.ev_battery_record_arr[t+1, i] / self.agents[i]['ev_capacity']
                    else:
                        self.ev_battery_soc_record_arr[t+1, i] = 0.0
                # ユーザIDはデマンドレスポンスによる移動を考慮して1エージェントごとに
                # リアルタイム(inelas, elas)，バッテリー充放電，ev充放電，PV発電供給，シフトリミット時間ステップ分の数IDを保有する
                # シフトリミットが24時間なら，31個IDを保有する
                # agentのIDは0～, 100～, 200～, 300～, ...として，101にagent1のinelas，102にagent1のelas...のように割り当てる
                id_base = i * 100
                # デマンドレスポンス不可の需要
                d_inelas = self.demand_inelastic_arr[t, i]
                demand_list.append([d_inelas, self.price_max, id_base+0, True])
                potential_demand += d_inelas

                # デマンドレスポンス可能の需要
                d_elas_max = self.demand_elastic_arr[t, i]
                price_elas = self.q.get_actions_[i, 0]
                # d_elas = d_elas_max * max((agents[i]['dr_price_threshold'] - price_elas)/(agents[i]['dr_price_threshold'] - price_min), 0)
                if price_elas == self.price_min:
                    # To avoid missing intersection point of supply and demand curve
                    price_elas += 0.00001
                # デマンドレスポンス可の需要はid_base+1に割り当てる
                demand_list.append([d_elas_max, price_elas, id_base+1, True])
                potential_demand += d_elas_max

                # バッテリー充放電価格の取得
                price_buy_battery = self.q.get_actions_[i, 1]
                price_sell_battery = self.q.get_actions_[i, 2]
                # バッテリー充放電可能量の取得
                battery_amount = self.battery_record_arr[t, i]
                if (self.agents[i]['battery_capacity'] - battery_amount) < (self.agents[i]['max_battery_charge_speed'] * self.battery_charge_efficiency):
                    charge_amount = (self.agents[i]['battery_capacity'] - battery_amount) / self.battery_charge_efficiency
                else:
                    charge_amount = self.agents[i]['max_battery_charge_speed']
                if battery_amount < (self.agents[i]['max_battery_discharge_speed'] / self.battery_discharge_efficiency):
                    discharge_amount = battery_amount * self.battery_discharge_efficiency
                else:
                    discharge_amount = self.agents[i]['max_battery_discharge_speed']
                if price_buy_battery == self.price_min:
                    # To avoid missing intersection point of supply and demand curve
                    price_buy_battery += 0.00001
                # バッテリー充電はid_base+2, 放電はid_base+3に割り当てる
                demand_list.append([charge_amount, price_buy_battery, id_base+2, True])
                supply_list.append([discharge_amount, price_sell_battery, id_base+3, False])
                potential_demand += charge_amount
                potential_supply += discharge_amount

                # EV充放電価格の取得 
                price_buy_ev_battery = self.q.get_actions_[i, 3]
                price_sell_ev_battery = self.q.get_actions_[i, 4]
                # EV充放電可能量の取得
                ev_battery_amount = self.ev_battery_record_arr[t, i]
                if (self.agents[i]['ev_capacity'] - ev_battery_amount) < (self.agents[i]['max_ev_charge_speed'] * self.ev_charge_efficiency):
                    ev_charge_amount = (self.agents[i]['ev_capacity'] - ev_battery_amount) / self.ev_charge_efficiency
                else:
                    ev_charge_amount = self.agents[i]['max_ev_charge_speed']
                if ev_battery_amount < (self.agents[i]['max_ev_discharge_speed'] / self.ev_discharge_efficiency):
                    ev_discharge_amount = ev_battery_amount * self.ev_discharge_efficiency
                else:
                    ev_discharge_amount = self.agents[i]['max_ev_discharge_speed']
                if ~self.car_charge_df.at[t, f'{i}']:
                    ev_charge_amount = 0
                    ev_discharge_amount = 0

                if price_buy_ev_battery == self.price_min:
                    # To avoid missing intersection point of supply and demand curve
                    price_buy_ev_battery += 0.00001
                # EVバッテリー充電はid_base+4, 放電はid_base+5に割り当てる
                demand_list.append([ev_charge_amount, price_buy_ev_battery, id_base+4, True])
                supply_list.append([ev_discharge_amount, price_sell_ev_battery, id_base+5, False])
                potential_demand += ev_charge_amount
                potential_supply += ev_discharge_amount

                # 供給
                s = self.supply_df.at[t, f'{i}']
                # 供給はid_base+6に割り当てる
                price_pv = self.q.get_actions_[i, 5]
                # 学習を安定させいい感じのところに導くためにPVの入札価格は最低価格で固定する
                # supply_list.append([s, price_pv, id_base+6, False])
                supply_list.append([s, self.price_min, id_base+6, False])
                potential_supply += s 

                # 後ろの時間にシフトさせる需要量の最大値を記録
                # マーケット取引をした後実際の取引があった場合，その分shiftする需要量を差し引くことで更新する
                self.shift_arr[t, i] = d_elas_max

                # 過去からシフトした需要の入札
                for k in range(t-int(self.agents[i]['shift_limit']), t):
                    if k >= 0:
                        d_shift = self.shift_arr[k, i]
                        # シフトした需要の入札価格は，デマンドレスポンス可能の需要の入札価格と同じ
                        price_shift = price_elas
                        if k == t-int(self.agents[i]['shift_limit']):
                            # シフトリミットでの価格は最高価格
                            price_shift = self.price_max
                        # 過去からのシフトはj+7から割り当てる
                        demand_list.append([d_shift, price_shift, id_base+7+t-k-1, True])
                        potential_demand += d_shift

            self.potential_demand_arr[t] = potential_demand
            self.potential_supply_arr[t] = potential_supply
            
            market = Market(demand_list, supply_list, wholesale_price)
            market.bid()
            bids_df = market.market.bm.get_df()
            # print(bids_df)
            
            # if episode == 0 or episode == num_episode-1 or episode%10 == 9:
            if self.BID_SAVE:
                timestamp = pd.read_csv('data/demand.csv').iat[t, 0]
                market.plot(title=timestamp, number=t, parent_dir=self.parent_dir)
            transactions_df, _ = market.run(mechanism='uniform')
            # print(transactions_df)
            # input()
            
            # マーケット取引の結果を記録、報酬を計算
            reward = np.full(self.num_agent, 0.0)
            cost = np.full(self.num_agent, 0.0)
            for bid_num in transactions_df['bid']:
                id = bids_df.at[bid_num, 'user']
                if id == 99999:
                    # record import from grid
                    self.grid_import_record_arr[t] = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]

                if id != 99999:
                    # 100の位以降の数字を取り出す->agentID
                    user = id // 100
                    # リアルタイム(inelas, elas)@2，バッテリー充放電@2，ev充放電@2，シフトリミット@shift_limit，供給@1
                    item = id % 100
                    price = transactions_df[transactions_df['bid']==bid_num]['price'].values[0]
                    if item == 0:
                        # リアルタイム(inelas)の取引量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.buy_inelastic_record_arr[t, user] = value
                        reward[user] -= value * price / 100  # reward cost in dollar, not cents
                        cost[user] += value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: inelastic, {value}, {price}')

                    elif item == 1:
                        # リアルタイム(elas)の取引量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.buy_elastic_record_arr[t, user] = value
                        # 時刻tでのDRの分だけ後ろの時間にシフトさせる需要量を減らす
                        self.shift_arr[t, user] -= value
                        reward[user] -= value * price / 100  # reward cost in dollar, not cents
                        reward[user] -= (self.agents[int(user)]['alpha']/2 * (self.demand_elastic_arr[t, user] - value)**2 + 
                                        self.agents[int(user)]['beta']*(self.demand_elastic_arr[t, user] - value))
                        cost[user] += value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: elastic, {value}, {price}, {self.demand_elastic_arr[t, user]}')

                    elif item == 2:
                        # バッテリー充電の取引量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.buy_battery_record_arr[t, user] = value
                        if t+1 != len(self.demand_df):
                            self.battery_record_arr[t+1, user] += value * self.battery_charge_efficiency
                            self.battery_soc_record_arr[t+1, user] = self.battery_record_arr[t+1, user] / self.agents[user]['battery_capacity']
                            reward[user] -= value * price / 100  # reward cost in dollar, not cents
                            # reward[user] -= ((self.agents[int(user)]['max_battery_charge_speed'] - value) * 
                            #                 (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t, user]))**2 + 
                            #                 self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t, user]))))
                            # reward[user] -= (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t+1, user]))**2 + 
                            #                 self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t+1, user])))
                        cost[user] += value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: battery charge, {value}, {price}, {self.battery_soc_record_arr[t, user]}')

                    elif item == 3:
                        # バッテリー放電の取引量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.sell_battery_record_arr[t, user] = value
                        if t+1 !=len(self.demand_df):
                            self.battery_record_arr[t+1, user] -= value / self.battery_discharge_efficiency
                            self.battery_soc_record_arr[t+1, user] = self.battery_record_arr[t+1, user] / self.agents[user]['battery_capacity']
                            reward[user] += value * price / 100  # reward cost in dollar, not cents
                            # reward[user] -= ((self.agents[int(user)]['max_battery_charge_speed'] + value) * 
                            #                 (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t, user]))**2 + 
                            #                 self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t, user]))))
                            # reward[user] -= (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t+1, user]))**2 + 
                            #                 self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t+1, user])))
                        cost[user] -= -value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: battery discharge, {value}, {price}, {self.battery_soc_record_arr[t, user]}')

                    elif item == 4:
                        # EVバッテリー充電の取引量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.buy_ev_battery_record_arr[t, user] = value
                        if t+1 !=len(self.demand_df):
                            self.ev_battery_record_arr[t+1, user] += value * self.ev_charge_efficiency
                            self.ev_battery_soc_record_arr[t+1, user] = self.ev_battery_record_arr[t+1, user] / self.agents[user]['ev_capacity']
                            reward[user] -= value * price / 100  # reward cost in dollar, not cents
                            # reward[user] -= ((self.agents[int(user)]['max_ev_charge_speed'] - value) *
                            #                 (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t, user]))**2 + 
                            #                 self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t, user]))))
                            # reward[user] -= (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t+1, user]))**2 + 
                            #                 self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t+1, user])))
                        cost[user] += value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: ev charge, {value}, {price}, {self.ev_battery_soc_record_arr[t, user]}')

                    elif item == 5:
                        # EVバッテリー放電の取引量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.sell_ev_battery_record_arr[t, user] = value
                        if t+1 !=len(self.demand_df):
                            self.ev_battery_record_arr[t+1, user] -= value / self.ev_discharge_efficiency
                            self.ev_battery_soc_record_arr[t+1, user] = self.ev_battery_record_arr[t+1, user] / self.agents[user]['ev_capacity']
                            reward[user] += value * price / 100  # reward cost in dollar, not cents
                            # reward[user] -= ((self.agents[int(user)]['max_ev_charge_speed'] + value) *
                            #                 (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t, user]))**2 + 
                            #                 self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t, user]))))
                            # reward[user] -= (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t+1, user]))**2 + 
                            #                 self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t+1, user])))
                        cost[user] -= value * price 
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: ev discharge, {value}, {price}, {self.ev_battery_soc_record_arr[t, user]}')

                    elif item == 6:
                        # PV発電供給量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.sell_pv_record_arr[t, user] = value
                        reward[user] += value * price / 100  # reward cost in dollar, not cents
                        cost[user] -= value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: pv, {value}, {price}')

                    else:
                        # buy_shifted_record_dfに足し上げながらshift_dfを更新
                        for k in range(7, 7+int(self.agents[user]['shift_limit'])):
                            if item == k:
                                value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                                self.buy_shifted_record_arr[t, user] += value
                                reward[user] -= value * price / 100  # reward cost in dollar, not cents
                                cost[user] += value * price
                                if t-k+7-1>= 0:
                                    self.shift_arr[t-k+7-1, user] -= value
            
            # EV SoCが0未満になっている場合は0にする、報酬に-1000を反映
            # capex, opexを1時間あたりの値にしてrewardから差し引く(より大きい設備を導入するとcapex, opexが増える)
            for i in range(self.num_agent):
                if t+1 != len(self.demand_df):
                    if self.ev_battery_soc_record_arr[t+1, i] < 0:
                        self.ev_battery_soc_record_arr[t+1, i] = 0
                        self.ev_battery_record_arr[t+1, i] = 0
                        reward[i] -= 1000
                pv_size = self.agents[i]['pv_capacity']
                battery_size = self.agents[i]['battery_capacity']
                pv_capex = capex_opex.pv_capex_func(pv_size)
                pv_opex = capex_opex.pv_opex_func(pv_size)
                battery_capex = capex_opex.battery_capex_func(battery_size, pv_size)
                # CAPEX of PV and BES are calculated by Straight Line Method. (定額法)
                # PVの法定耐用年数は17年、BESの法定耐用年数は6年.
                # The statutory useful life of the depreciable assets for PV is 17 years.
                # The statutory useful life of the depreciable assets for BES is 6 years.
                reward[i] -= (pv_capex / 17 + pv_opex + battery_capex / 6) / 8784  # reward cost in dollar, not cents

            self.microgrid_price_record_arr[t] = transactions_df['price'].values[0]

            # Q学習
            dr_states, battery_states, ev_battery_states, pv_states, battery_patterns, ev_battery_patterns, pv_patterns  = self.q.get_states_
            actions_arr = self.q.get_actions_
            if t == 0:
                previous_states = []
                previous_actions = []
                previous_rewards = []
            for i in range(self.num_agent):
                self.reward_arr[t, i] = reward[i]
                self.electricity_cost_arr[t, i] = cost[i] / 100  # record cost in dollar, not cents
                # バッテリーの充放電、EVバッテリーの充放電はそれぞれ同じstateで管理できるため重複している
                states = [int(dr_states[i]), int(battery_states[i]), int(battery_states[i]), int(ev_battery_states[i]), int(ev_battery_states[i]), int(pv_states[i])]
                actions = [actions_arr[i, 0], actions_arr[i, 1], actions_arr[i, 2], actions_arr[i, 3], actions_arr[i, 4], actions_arr[i, 5]]
                rewards = [reward[i], reward[i], reward[i], reward[i], reward[i], reward[i]]   # rewardは共通の値(すべての要素からのrewardの合計)
                if t == 0:
                    previous_states.append(states)
                    previous_actions.append(actions)
                    previous_rewards.append(rewards)
                else:
                    if self.train:
                        self.q.update_q_table(agent_id=i,
                                            states=previous_states[i],
                                            actions=previous_actions[i], 
                                            rewards=previous_rewards[i],
                                            next_states=states)
                    previous_states[i] = states
                    previous_actions[i] = actions
                    previous_rewards[i] = rewards

    def save(self):
        timestamp = pd.read_csv('data/demand.csv').iloc[:, 0]
        # parent_dir = 'output/episode' + str(episode)

        if not self.train:
            grid_import_record_df = pd.DataFrame(self.grid_import_record_arr, index=timestamp, columns=['Grid import'])
            grid_import_record_df.to_csv(self.parent_dir + '/grid_import_record.csv', index=True)
            microgrid_price_record_df = pd.DataFrame(self.microgrid_price_record_arr, index=timestamp, columns=['Price'])
            microgrid_price_record_df.to_csv(self.parent_dir + '/price_record.csv', index=True)
            battery_record_df = pd.DataFrame(self.battery_record_arr, index=timestamp, columns=self.demand_df.columns)
            battery_record_df.to_csv(self.parent_dir + '/battery_record.csv', index=True)
            ev_battery_record_df = pd.DataFrame(self.ev_battery_record_arr, index=timestamp, columns=self.demand_df.columns)
            ev_battery_record_df.to_csv(self.parent_dir + '/ev_battery_record.csv', index=True)
            battery_soc_record_df = pd.DataFrame(self.battery_soc_record_arr, index=timestamp, columns=self.demand_df.columns)
            battery_soc_record_df.to_csv(self.parent_dir + '/battery_soc_record.csv', index=True)
            ev_battery_soc_record_df = pd.DataFrame(self.ev_battery_soc_record_arr, index=timestamp, columns=self.demand_df.columns)
            ev_battery_soc_record_df.to_csv(self.parent_dir + '/ev_battery_soc_record.csv', index=True)

            buy_inelastic_record_df = pd.DataFrame(self.buy_inelastic_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_inelastic_record_df.to_csv(self.parent_dir + '/buy_inelastic_record.csv', index=True)
            buy_elastic_record_df = pd.DataFrame(self.buy_elastic_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_elastic_record_df.to_csv(self.parent_dir + '/buy_elastic_record.csv', index=True)
            buy_shifted_record_df = pd.DataFrame(self.buy_shifted_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_shifted_record_df.to_csv(self.parent_dir + '/buy_shifted_record.csv', index=True)
            sell_pv_record_df = pd.DataFrame(self.sell_pv_record_arr, index=timestamp, columns=self.supply_df.columns)
            sell_pv_record_df.to_csv(self.parent_dir + '/sell_pv_record.csv', index=True)
            
            buy_battery_record_df = pd.DataFrame(self.buy_battery_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_battery_record_df.to_csv(self.parent_dir + '/buy_battery_record.csv', index=True)
            buy_ev_battery_record_df = pd.DataFrame(self.buy_ev_battery_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_ev_battery_record_df.to_csv(self.parent_dir + '/buy_ev_battery_record.csv', index=True)
            sell_battery_record_df = pd.DataFrame(self.sell_battery_record_arr, index=timestamp, columns=self.supply_df.columns)
            sell_battery_record_df.to_csv(self.parent_dir + '/sell_battery_record.csv', index=True)
            sell_ev_battery_record_df = pd.DataFrame(self.sell_ev_battery_record_arr, index=timestamp, columns=self.supply_df.columns)
            sell_ev_battery_record_df.to_csv(self.parent_dir + '/sell_ev_battery_record.csv', index=True)

            shift_df = pd.DataFrame(self.shift_arr, index=timestamp, columns=self.demand_df.columns)
            shift_df.to_csv(self.parent_dir + '/shift_record.csv', index=True)

            potential_demand_df = pd.DataFrame(self.potential_demand_arr, index=timestamp, columns=['Potential demand'])
            potential_demand_df.to_csv(self.parent_dir + '/potential_demand.csv', index=True)
            potential_supply_df = pd.DataFrame(self.potential_supply_arr, index=timestamp, columns=['Potential supply'])
            potential_supply_df.to_csv(self.parent_dir + '/potential_supply.csv', index=True)

            reward_df = pd.DataFrame(self.reward_arr, index=timestamp, columns=self.demand_df.columns)
            reward_df.to_csv(self.parent_dir + '/reward.csv', index=True)
            # This data is recorded as net cost
            net_electricity_cost_df = pd.DataFrame(self.electricity_cost_arr, index=timestamp, columns=self.demand_df.columns)
            net_electricity_cost_df.to_csv(self.parent_dir + '/net_electricity_cost.csv', index=True)
            self.car_charge_df.to_csv(self.parent_dir + '/car_charge_bool.csv', index=True)
            self.car_move_consumption_df.to_csv(self.parent_dir + '/car_move_consumption.csv', index=True)

            # JSON形式でパラメータを保存
            file_name = self.parent_dir + "/params.json"
            with open(file_name, 'w') as file:
                json.dump(self.params, file, indent=4)

            vis = visualize.Visualize(folder_path=self.parent_dir)
            vis.plot_consumption()

        self.q.save_q_table(folder_path = self.parent_dir, train=self.train)
        logger.info(f'Q table is saved to {self.parent_dir}')


class SimulationNoP2P:
    def __init__(self, num_agent, parent_dir, episode, train, **kwargs) -> None:
        os.makedirs(parent_dir, exist_ok=True)
        self.num_agent = num_agent
        self.parent_dir = parent_dir
        self.episode = episode
        self.train = train
        # Update market and uniform parameters
        params = {'thread_num': -1,
                  'price_max': 110,
                  'price_min': 10,
                  'wheeling_charge': 0,
                  'battery_charge_efficiency': 0.9,
                  'battery_discharge_efficiency': 0.9,
                  'ev_charge_efficiency': 0.9,
                  'ev_discharge_efficiency': 0.9,
                  'battery_capacity_list': [0, 10, 15, 20],
                  'ev_capacity_list': [40],
                  'pv_capacity_list': [0, 5, 10],
                  'discount_rate': 1.0,
                  'learning_rate': 0.01,
                  'shift_limit_list': [6.0, 12.0, 18.0, 24.0],  # hours
                  'max_battery_charge_speed': [3.0],  # kW
                  'max_battery_discharge_speed': [3.0],  # kW
                  'max_ev_charge_speed': [6.0],  # kW
                  'max_ev_discharge_speed': [3.0],  # kW
                  'dr_boolean_list': [True, False],
                  'alpha_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
                  'beta_list': [1, 1.5, 2, 2.5, 3, 3.5, 4]
                #   'gamma_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
                #   'epsilon_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
                #   'psi_list': [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
                #   'omega_list': [1, 1.5, 2, 2.5, 3, 3.5, 4]
        }
        params.update(kwargs)
        self.params = params
        self.thread_num = params['thread_num']
        self.price_max = params['price_max']
        self.price_min = params['price_min']
        self.wheeling_charge = params['wheeling_charge']
        self.battery_charge_efficiency = params['battery_charge_efficiency']
        self.battery_discharge_efficiency = params['battery_discharge_efficiency']
        self.ev_charge_efficiency = params['ev_charge_efficiency']
        self.ev_discharge_efficiency = params['ev_discharge_efficiency']
        self.battery_capacity_list = params['battery_capacity_list']
        self.ev_capacity_list = params['ev_capacity_list']
        self.pv_capacity_list = params['pv_capacity_list']
        self.discount_rate = params['discount_rate']
        self.learning_rate = params['learning_rate']

        # Initialize Q table
        self.q = Q(params, agent_num=num_agent, num_dizitized_pv_ratio=5, num_dizitized_soc=5, num_elastic_ratio_pattern=3)
    
    def load_existing_q_table(self, folder_path):
        self.q.load_q_table(folder_path=folder_path)

    def remove_existing_q_table(self, folder_path):
        self.q.remove_q_table_saved_data(folder_path=folder_path)

    def preprocess(self):
        # Generate agent parameters
        self.agents = Agent(self.num_agent)
        self.agents.generate_params(self.params, seed=self.thread_num)
        for agent_id in range(self.num_agent):
            battery_capacity, ev_capacity, pv_capacity = self.q.get_facility_capacities(agent_id, episode=self.episode-1, is_train=self.train)
            self.agents.set_one_agent(agent_id, battery_capacity=battery_capacity, ev_capacity=ev_capacity, pv_capacity=pv_capacity)
        self.agents.save(self.parent_dir)
        agent_params_df = self.agents.get_agents_params_df_
        
        # Preprocess and generate demand, price, and car_movement(boolean) data
        preprocess = Preprocess(seed=self.thread_num)
        preprocess.set(
            pd.read_csv('data/demand.csv'),
            pd.read_csv('data/supply.csv'),
            pd.read_csv('data/price.csv'),
            pd.read_csv('data/ev_charging_bool.csv'),
            pd.read_csv('data/ev_move_consumption.csv'),
            pd.read_csv('data/elastic_ratio.csv')
        )
        # preprocess.generate_d_s(self.num_agent)
        preprocess.generate_demand(self.num_agent)
        pv_capacity_list = agent_params_df['pv_capacity'].values
        # Generate supply data
        preprocess.generate_supply_flex_pv_size(self.num_agent, pv_capacity_list)
        # Generate car charge data
        preprocess.generate_car_charge(self.num_agent)
        preprocess.save(self.parent_dir)
        preprocess.drop_index_  # drop timestamp index
        self.demand_df, self.supply_df, self.price_df, self.car_charge_df, self.car_move_consumption_df, self.elastic_ratio_df = preprocess.get_dfs_

        # get average pv production ratio to get state in Q table
        # data is stored as kWh/kW, which means, the values are within 0~1
        pv_ratio_df = pd.read_csv('data/supply.csv', index_col=0)
        pv_ratio_df['mean'] = pv_ratio_df.mean(axis=1)
        self.pv_ratio_arr = pv_ratio_df['mean'].values

        # Initialize record arrays
        self.grid_import_record_arr = np.full(len(self.price_df), 0.0)
        self.microgrid_price_record_arr = np.full(len(self.price_df), 0.0)
        self.ev_battery_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.battery_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.battery_soc_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.ev_battery_soc_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_inelastic_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_elastic_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_shifted_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_battery_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.buy_ev_battery_record_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.sell_pv_record_arr = np.full((len(self.supply_df), self.num_agent), 0.0)
        self.sell_battery_record_arr = np.full((len(self.supply_df), self.num_agent), 0.0)
        self.sell_ev_battery_record_arr = np.full((len(self.supply_df), self.num_agent), 0.0)
        self.reward_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.electricity_cost_arr = np.full((len(self.demand_df), self.num_agent), 0.0)
        self.potential_demand_arr = np.full(len(self.demand_df), 0.0)
        self.potential_supply_arr = np.full(len(self.supply_df), 0.0)

        # set initial ev battery state to 50% of its capacity
        for i in range(self.num_agent):
            self.ev_battery_record_arr[0, i] = self.agents[i]['ev_capacity'] / 2
            if self.agents[i]['ev_capacity'] != 0:
                self.ev_battery_soc_record_arr[0, i] = self.ev_battery_record_arr[0, i] / self.agents[i]['ev_capacity']
            else:
                self.ev_battery_soc_record_arr[0, i] = 0.0
        
        # Generate elastic and inelastic demand according to the elastic ratio of each agent
        self.demand_elastic_arr = self.demand_df.values.copy()
        self.demand_inelastic_arr = self.demand_df.values.copy()
        for i in range(self.num_agent):
            if self.agents[i]['dr_boolean'] == False:
                self.demand_elastic_arr[:, i] = 0
                self.demand_inelastic_arr[:, i] = self.demand_df[f'{i}']
            elif self.agents[i]['dr_boolean'] == True:
                self.demand_elastic_arr[:, i] = self.demand_df[f'{i}'] * self.elastic_ratio_df["elastic_ratio"]
                self.demand_inelastic_arr[:, i] = self.demand_df[f'{i}'] * (1 - self.elastic_ratio_df["elastic_ratio"])
            else:
                raise ValueError("DR boolean key is invalid.")

        # Prepare dataframe to record shifted demand
        # shift_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        self.shift_arr = np.full((len(self.demand_df), self.num_agent), 0.0)

    def run(self):
        for t in tqdm(range(len(self.demand_df))):
            potential_demand = 0
            potential_supply = 0
            wholesale_price = self.price_df.at[t, 'Price'] + self.wheeling_charge
            self.microgrid_price_record_arr[t] = wholesale_price
            self.q.reset_all_digitized_states()
            self.q.reset_all_actions()
            reward = np.full(self.num_agent, 0.0)
            cost = np.full(self.num_agent, 0.0)
            for i in range(self.num_agent): 
                #============================================================================================================================================================
                self.q.set_digitized_states(agent_id=i, agent_params=self.agents[i], 
                                            pv_ratio=self.pv_ratio_arr[t], 
                                            battery_soc=self.battery_soc_record_arr[t, i], 
                                            ev_battery_soc=self.ev_battery_soc_record_arr[t, i], 
                                            elastic_ratio=self.elastic_ratio_df.at[t, "elastic_ratio"])
                # Qテーブルから行動を取得, ε-greedy法で徐々に最適行動を選択する式が、エピソード0から始まるように定義されているので、エピソード-1を引数に渡す
                self.q.set_actions(agent_id=i, episode=self.episode-1, is_train=self.train)
                # 時刻tでのバッテリー残量を時刻t+1にコピー、取引が行われる場合あとでバッテリー残量をさらに更新
                # t+1でのcar_charge_dfがFalseのとき、car_move_consumption_dfの値を引く
                # EVバッテリー残量が負の値になる場合もここではそのままにして、報酬を計算するフェーズで対応、0に更新するとともに-1000を報酬に反映
                if t+1 != len(self.demand_df):
                    self.battery_record_arr[t+1, i] = self.battery_record_arr[t, i]
                    if self.agents[i]['battery_capacity'] != 0:
                        self.battery_soc_record_arr[t+1, i] = self.battery_record_arr[t+1, i] / self.agents[i]['battery_capacity']
                    else:
                        self.battery_soc_record_arr[t+1, i] = 0.0
                    if self.agents[i]['ev_capacity'] != 0:
                        self.ev_battery_record_arr[t+1, i] = self.ev_battery_record_arr[t, i]
                        if ~self.car_charge_df.at[t+1, f'{i}']:
                            self.ev_battery_record_arr[t+1, i] = self.ev_battery_record_arr[t, i] - self.car_move_consumption_df.at[t+1, f'{i}']
                        self.ev_battery_soc_record_arr[t+1, i] = self.ev_battery_record_arr[t+1, i] / self.agents[i]['ev_capacity']
                    else:
                        self.ev_battery_soc_record_arr[t+1, i] = 0.0
                # リアルタイム(inelas, elas)，バッテリー充放電，ev充放電，PV発電供給，シフトリミット時間ステップ分の種類の需要と供給がある
                # 供給
                s = self.supply_df.at[t, f'{i}']
                # price_pv = self.q.get_actions_[i, 5]
                potential_supply += s
                
                # デマンドレスポンス不可の需要
                d_inelas = self.demand_inelastic_arr[t, i]
                potential_demand += d_inelas

                # デマンドレスポンス可能の需要
                d_elas_max = self.demand_elastic_arr[t, i]
                # デマンドレスポンスするかのしきい価格の取得
                price_elas = self.q.get_actions_[i, 0]
                if price_elas == self.price_min:
                    # To make the same situation as the case with P2P
                    price_elas += 0.00001
                potential_demand += d_elas_max
                # 後ろの時間にシフトさせる需要量の最大値を記録
                # 実際の取引があった場合，その分shiftする需要量を差し引くことで更新する
                self.shift_arr[t, i] = d_elas_max

                # バッテリー充放電しきい価格の取得
                price_buy_battery = self.q.get_actions_[i, 1]
                price_sell_battery = self.q.get_actions_[i, 2]
                # バッテリー充放電可能量の取得
                battery_amount = self.battery_record_arr[t, i]
                if (self.agents[i]['battery_capacity'] - battery_amount) < (self.agents[i]['max_battery_charge_speed'] * self.battery_charge_efficiency):
                    charge_amount = (self.agents[i]['battery_capacity'] - battery_amount) / self.battery_charge_efficiency
                else:
                    charge_amount = self.agents[i]['max_battery_charge_speed']
                if battery_amount < (self.agents[i]['max_battery_discharge_speed'] / self.battery_discharge_efficiency):
                    discharge_amount = battery_amount * self.battery_discharge_efficiency
                else:
                    discharge_amount = self.agents[i]['max_battery_discharge_speed']
                if price_buy_battery == self.price_min:
                    # To make the same situation as the case with P2P
                    price_buy_battery += 0.00001
                potential_demand += charge_amount
                potential_supply += discharge_amount

                # EV充放電しきい価格の取得 
                price_buy_ev_battery = self.q.get_actions_[i, 3]
                price_sell_ev_battery = self.q.get_actions_[i, 4]
                # EV充放電可能量の取得
                ev_battery_amount = self.ev_battery_record_arr[t, i]
                if (self.agents[i]['ev_capacity'] - ev_battery_amount) < (self.agents[i]['max_ev_charge_speed'] * self.ev_charge_efficiency):
                    ev_charge_amount = (self.agents[i]['ev_capacity'] - ev_battery_amount) / self.ev_charge_efficiency
                else:
                    ev_charge_amount = self.agents[i]['max_ev_charge_speed']
                if ev_battery_amount < (self.agents[i]['max_ev_discharge_speed'] / self.ev_discharge_efficiency):
                    ev_discharge_amount = ev_battery_amount * self.ev_discharge_efficiency
                else:
                    ev_discharge_amount = self.agents[i]['max_ev_discharge_speed']
                if ~self.car_charge_df.at[t, f'{i}']:
                    ev_charge_amount = 0
                    ev_discharge_amount = 0

                if price_buy_ev_battery == self.price_min:
                    # To make the same situation as the case with P2P
                    price_buy_ev_battery += 0.00001
                potential_demand += ev_charge_amount
                potential_supply += ev_discharge_amount

                # print(i)
                # print(price_elas, price_buy_battery, price_sell_battery, price_buy_ev_battery, price_sell_ev_battery)
                # print(d_inelas, d_elas_max, charge_amount, discharge_amount, ev_charge_amount, ev_discharge_amount, s)


                # Check if the PV of the agent is enough to supply the inelastic demand
                if s >= d_inelas:
                    s_residue = s - d_inelas
                    d_inelas_residue = 0
                    self.buy_inelastic_record_arr[t, i] += d_inelas
                    self.sell_pv_record_arr[t, i] += d_inelas
                else:
                    s_residue = 0
                    d_inelas_residue = d_inelas - s
                    self.buy_inelastic_record_arr[t, i] += s
                    self.sell_pv_record_arr[t, i] += s
                
                # Check if the residue PV of the agent is enough to supply the elastic demand
                if s_residue > d_elas_max:
                    s_residue -= d_elas_max
                    d_elas_residue = 0
                    self.buy_elastic_record_arr[t, i] += d_elas_max
                    self.sell_pv_record_arr[t, i] += d_elas_max
                    self.shift_arr[t, i] = 0
                else:
                    d_elas_residue = d_elas_max - s_residue
                    self.buy_elastic_record_arr[t, i] += s_residue
                    self.sell_pv_record_arr[t, i] += s_residue
                    self.shift_arr[t, i] = d_elas_residue
                    s_residue = 0

                # Check if the residue PV of the agent is enough to supply the shifted demand
                for k in range(t-int(self.agents[i]['shift_limit']), t):
                    if k >= 0:
                        d_shift = self.shift_arr[k, i]
                        potential_demand += d_shift
                        if s_residue > d_shift:
                            self.buy_shifted_record_arr[t, i] += d_shift
                            self.shift_arr[k, i] -= d_shift
                            s_residue -= d_shift
                        else:
                            self.buy_shifted_record_arr[t, i] += s_residue
                            self.shift_arr[k, i] -= s_residue
                            s_residue = 0                            

                # Check if the residue PV of the agent is enough to supply the EV battery charge demand
                if s_residue > ev_charge_amount:
                    s_residue -= ev_charge_amount
                    ev_charge_residue = 0
                    self.buy_ev_battery_record_arr[t, i] += ev_charge_amount
                    if t+1 != len(self.demand_df):
                        self.ev_battery_record_arr[t+1, i] += ev_charge_amount * self.ev_charge_efficiency
                        if self.agents[i]['ev_capacity'] != 0:
                            self.ev_battery_soc_record_arr[t+1, i] = self.ev_battery_record_arr[t+1, i] / self.agents[i]['ev_capacity']
                        else:
                            self.ev_battery_soc_record_arr[t+1, i] = 0.0
                    self.sell_pv_record_arr[t, i] += ev_charge_amount
                else:
                    ev_charge_residue = ev_charge_amount - s_residue
                    self.buy_ev_battery_record_arr[t, i] += s_residue
                    if t+1 != len(self.demand_df):
                        self.ev_battery_record_arr[t+1, i] += s_residue * self.ev_charge_efficiency
                        if self.agents[i]['ev_capacity'] != 0:
                            self.ev_battery_soc_record_arr[t+1, i] = self.ev_battery_record_arr[t+1, i] / self.agents[i]['ev_capacity']
                        else:
                            self.ev_battery_soc_record_arr[t+1, i] = 0.0
                    self.sell_pv_record_arr[t, i] += s_residue
                    s_residue = 0
                
                # Check if the residue PV of the agent is enough to supply the battery charge demand
                if s_residue >= charge_amount:
                    s_residue -= charge_amount
                    charge_residue = 0
                    self.buy_battery_record_arr[t, i] += charge_amount
                    if t+1 != len(self.demand_df):
                        self.battery_record_arr[t+1, i] += charge_amount * self.battery_charge_efficiency
                        if self.agents[i]['battery_capacity'] != 0:
                            self.battery_soc_record_arr[t+1, i] = self.battery_record_arr[t+1, i] / self.agents[i]['battery_capacity']
                        else:
                            self.battery_soc_record_arr[t+1, i] = 0.0
                    self.sell_pv_record_arr[t, i] += charge_amount
                else:
                    charge_residue = charge_amount - s_residue
                    self.buy_battery_record_arr[t, i] += s_residue
                    if t+1 != len(self.demand_df):
                        self.battery_record_arr[t+1, i] += s_residue * self.battery_charge_efficiency
                        if self.agents[i]['battery_capacity'] != 0:
                            self.battery_soc_record_arr[t+1, i] = self.battery_record_arr[t+1, i] / self.agents[i]['battery_capacity']
                        else:
                            self.battery_soc_record_arr[t+1, i] = 0.0
                    self.sell_pv_record_arr[t, i] += s_residue
                    s_residue = 0

                # Check if the battery discharge is available according to the wholesale price
                # and discharge amount is enough to supply to residue of the inelastic demand
                if price_sell_battery < wholesale_price:
                    if discharge_amount >= d_inelas_residue:
                        self.buy_inelastic_record_arr[t, i] += d_inelas_residue
                        discharge_residue = discharge_amount - d_inelas_residue
                        d_inelas_residue = 0
                        self.sell_battery_record_arr[t, i] += d_inelas_residue
                        if t+1 != len(self.demand_df):
                            self.battery_record_arr[t+1, i] -= d_inelas_residue / self.battery_discharge_efficiency
                            if self.agents[i]['battery_capacity'] != 0:
                                self.battery_soc_record_arr[t+1, i] = self.battery_record_arr[t+1, i] / self.agents[i]['battery_capacity']
                            else:
                                self.battery_soc_record_arr[t+1, i] = 0.0
                    else:
                        self.buy_inelastic_record_arr[t, i] += discharge_amount
                        d_inelas_residue -= discharge_amount
                        discharge_residue = 0
                        self.sell_battery_record_arr[t, i] += discharge_amount
                        if t+1 != len(self.demand_df):
                            self.battery_record_arr[t+1, i] -= discharge_amount / self.battery_discharge_efficiency
                            if self.agents[i]['battery_capacity'] != 0:
                                self.battery_soc_record_arr[t+1, i] = self.battery_record_arr[t+1, i] / self.agents[i]['battery_capacity']
                            else:
                                self.battery_soc_record_arr[t+1, i] = 0.0

                # Check if the EV battery discharge is available according to the wholesale price
                # and discharge amount is enough to supply to residue of the inelastic demand
                if price_sell_ev_battery < wholesale_price:
                    if ev_discharge_amount >= d_inelas_residue:
                        self.buy_inelastic_record_arr[t, i] += d_inelas_residue
                        ev_discharge_residue = ev_discharge_amount - d_inelas_residue
                        d_inelas_residue = 0
                        self.sell_ev_battery_record_arr[t, i] += d_inelas_residue
                        if t+1 != len(self.demand_df):
                            self.ev_battery_record_arr[t+1, i] -= d_inelas_residue / self.ev_discharge_efficiency
                            if self.agents[i]['ev_capacity'] != 0:
                                self.ev_battery_soc_record_arr[t+1, i] = self.ev_battery_record_arr[t+1, i] / self.agents[i]['ev_capacity']
                            else:
                                self.ev_battery_soc_record_arr[t+1, i] = 0.0
                    else:
                        self.buy_inelastic_record_arr[t, i] += ev_discharge_amount
                        d_inelas_residue -= ev_discharge_amount
                        ev_discharge_residue = 0
                        self.sell_ev_battery_record_arr[t, i] += ev_discharge_amount
                        if t+1 != len(self.demand_df):
                            self.ev_battery_record_arr[t+1, i] -= ev_discharge_amount / self.ev_discharge_efficiency
                            if self.agents[i]['ev_capacity'] != 0:
                                self.ev_battery_soc_record_arr[t+1, i] = self.ev_battery_record_arr[t+1, i] / self.agents[i]['ev_capacity']
                            else:
                                self.ev_battery_soc_record_arr[t+1, i] = 0.0

                # import the rest of the inelastic demand from the grid
                self.grid_import_record_arr[t] += d_inelas_residue
                self.buy_inelastic_record_arr[t, i] += d_inelas_residue
                cost[i] += d_inelas_residue * wholesale_price
                reward[i] -= d_inelas_residue * wholesale_price / 100  # reward cost in dollar, not cents

                # Check if the DR is available according to the wholesale price
                if price_elas >= wholesale_price:
                    self.buy_elastic_record_arr[t, i] += d_elas_residue
                    self.grid_import_record_arr[t] += d_elas_residue
                    cost[i] += d_elas_residue * wholesale_price
                    reward[i] -= d_elas_residue * wholesale_price / 100  # reward cost in dollar, not cents
                    d_elas_residue = 0
                    self.shift_arr[t, i] = 0
                else:
                    reward[i] -= (self.agents[int(i)]['alpha']/2 * (d_elas_max - d_elas_residue)**2 + 
                                        self.agents[int(i)]['beta']*(d_elas_max - d_elas_residue))
                
                # Check if the EV battery charge is available according to the wholesale price
                if price_buy_ev_battery >= wholesale_price:
                    self.grid_import_record_arr[t] += ev_charge_residue
                    cost[i] += ev_charge_residue * wholesale_price
                    self.buy_ev_battery_record_arr[t, i] += ev_charge_residue
                    if t+1 != len(self.demand_df):
                        self.ev_battery_record_arr[t+1, i] += ev_charge_residue * self.ev_charge_efficiency
                        if self.agents[i]['ev_capacity'] != 0:
                            self.ev_battery_soc_record_arr[t+1, i] = self.ev_battery_record_arr[t+1, i] / self.agents[i]['ev_capacity']
                        else:
                            self.ev_battery_soc_record_arr[t+1, i] = 0.0
                        reward[i] -= ev_charge_residue * wholesale_price / 100  # reward cost in dollar, not cents
                    ev_charge_residue = 0

                # Check if the battery charge is available according to the wholesale price
                if price_buy_battery >= wholesale_price:
                    self.grid_import_record_arr[t] += charge_residue
                    cost[i] += charge_residue * wholesale_price
                    self.buy_battery_record_arr[t, i] += charge_residue
                    if t+1 != len(self.demand_df):
                        self.battery_record_arr[t+1, i] += charge_residue * self.battery_charge_efficiency
                        if self.agents[i]['battery_capacity'] != 0:
                            self.battery_soc_record_arr[t+1, i] = self.battery_record_arr[t+1, i] / self.agents[i]['battery_capacity']
                        else:
                            self.battery_soc_record_arr[t+1, i] = 0.0
                        reward[i] -= charge_residue * wholesale_price / 100  # reward cost in dollar, not cents
                    charge_residue = 0
                
                # Add reward for the battery and EV battery SOC
                # if t+1 != len(self.demand_df):
                #     reward[i] -= (self.agents[int(i)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t+1, i]))**2 + 
                #                         self.agents[int(i)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t+1, i])))
                #     reward[i] -= (self.agents[int(i)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t+1, i]))**2 + 
                #                       self.agents[int(i)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t+1, i])))
                    

                # Check if the shifted demand is available according to the wholesale price and handle one by one
                for k in range(t-int(self.agents[i]['shift_limit']), t):
                    if k >= 0:
                        d_shift = self.shift_arr[k, i]
                        potential_demand += d_shift
                        # シフトした需要のしきい価格は，デマンドレスポンス可能の需要のしきい価格と同じ
                        price_shift = price_elas
                        if k == t-int(self.agents[i]['shift_limit']):
                            # シフトリミットでの価格しきい値は最高価格
                            price_shift = self.price_max
                        if price_shift >= wholesale_price:
                            self.grid_import_record_arr[t] += d_shift
                            cost[i] += d_shift * wholesale_price
                            reward[i] -= d_shift * wholesale_price / 100  # reward cost in dollar, not cents
                            self.buy_shifted_record_arr[t, i] += d_shift
                            self.shift_arr[k, i] -= d_shift


            self.potential_demand_arr[t] = potential_demand
            self.potential_supply_arr[t] = potential_supply
            
            # print(f'potential_demand: {potential_demand}, potential_supply: {potential_supply}')
            # print(f'grid_import: {self.grid_import_record_arr[t]}')
            # print(self.shift_arr)
            # input()
            
            # EV SoCが0未満になっている場合は0にする、報酬に-1000を反映
            # capex, opexを1時間あたりの値にしてrewardから差し引く(より大きい設備を導入するとcapex, opexが増える)
            for i in range(self.num_agent):
                if t+1 != len(self.demand_df):
                    if self.ev_battery_soc_record_arr[t+1, i] < 0:
                        self.ev_battery_soc_record_arr[t+1, i] = 0
                        self.ev_battery_record_arr[t+1, i] = 0
                        reward[i] -= 1000
                pv_size = self.agents[i]['pv_capacity']
                battery_size = self.agents[i]['battery_capacity']
                pv_capex = capex_opex.pv_capex_func(pv_size)
                pv_opex = capex_opex.pv_opex_func(pv_size)
                battery_capex = capex_opex.battery_capex_func(battery_size, pv_size)
                # CAPEX of PV and BES are calculated by Straight Line Method. (定額法)
                # PVの法定耐用年数は17年、BESの法定耐用年数は6年.
                # The statutory useful life of the depreciable assets for PV is 17 years.
                # The statutory useful life of the depreciable assets for BES is 6 years.
                reward[i] -= (pv_capex / 17 + pv_opex + battery_capex / 6) / 8784  # reward cost in dollar, not cents


            # Q学習
            dr_states, battery_states, ev_battery_states, pv_states, battery_patterns, ev_battery_patterns, pv_patterns  = self.q.get_states_
            actions_arr = self.q.get_actions_
            if t == 0:
                previous_states = []
                previous_actions = []
                previous_rewards = []
            for i in range(self.num_agent):
                self.reward_arr[t, i] = reward[i]
                self.electricity_cost_arr[t, i] = cost[i] / 100  # record cost in dollar, not cents
                # バッテリーの充放電、EVバッテリーの充放電はそれぞれ同じstateで管理できるため重複している
                states = [int(dr_states[i]), int(battery_states[i]), int(battery_states[i]), int(ev_battery_states[i]), int(ev_battery_states[i]), int(pv_states[i])]
                actions = [actions_arr[i, 0], actions_arr[i, 1], actions_arr[i, 2], actions_arr[i, 3], actions_arr[i, 4], actions_arr[i, 5]]
                rewards = [reward[i], reward[i], reward[i], reward[i], reward[i], reward[i]]   # rewardは共通の値(すべての要素からのrewardの合計)
                if t == 0:
                    previous_states.append(states)
                    previous_actions.append(actions)
                    previous_rewards.append(rewards)
                else:
                    if self.train:
                        self.q.update_q_table(agent_id=i,
                                            states=previous_states[i],
                                            actions=previous_actions[i], 
                                            rewards=previous_rewards[i],
                                            next_states=states)
                    previous_states[i] = states
                    previous_actions[i] = actions
                    previous_rewards[i] = rewards

    def save(self):
        timestamp = pd.read_csv('data/demand.csv').iloc[:, 0]
        # parent_dir = 'output/episode' + str(episode)

        if not self.train:
            grid_import_record_df = pd.DataFrame(self.grid_import_record_arr, index=timestamp, columns=['Grid import'])
            grid_import_record_df.to_csv(self.parent_dir + '/grid_import_record.csv', index=True)
            microgrid_price_record_df = pd.DataFrame(self.microgrid_price_record_arr, index=timestamp, columns=['Price'])
            microgrid_price_record_df.to_csv(self.parent_dir + '/price_record.csv', index=True)
            battery_record_df = pd.DataFrame(self.battery_record_arr, index=timestamp, columns=self.demand_df.columns)
            battery_record_df.to_csv(self.parent_dir + '/battery_record.csv', index=True)
            ev_battery_record_df = pd.DataFrame(self.ev_battery_record_arr, index=timestamp, columns=self.demand_df.columns)
            ev_battery_record_df.to_csv(self.parent_dir + '/ev_battery_record.csv', index=True)
            battery_soc_record_df = pd.DataFrame(self.battery_soc_record_arr, index=timestamp, columns=self.demand_df.columns)
            battery_soc_record_df.to_csv(self.parent_dir + '/battery_soc_record.csv', index=True)
            ev_battery_soc_record_df = pd.DataFrame(self.ev_battery_soc_record_arr, index=timestamp, columns=self.demand_df.columns)
            ev_battery_soc_record_df.to_csv(self.parent_dir + '/ev_battery_soc_record.csv', index=True)

            buy_inelastic_record_df = pd.DataFrame(self.buy_inelastic_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_inelastic_record_df.to_csv(self.parent_dir + '/buy_inelastic_record.csv', index=True)
            buy_elastic_record_df = pd.DataFrame(self.buy_elastic_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_elastic_record_df.to_csv(self.parent_dir + '/buy_elastic_record.csv', index=True)
            buy_shifted_record_df = pd.DataFrame(self.buy_shifted_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_shifted_record_df.to_csv(self.parent_dir + '/buy_shifted_record.csv', index=True)
            sell_pv_record_df = pd.DataFrame(self.sell_pv_record_arr, index=timestamp, columns=self.supply_df.columns)
            sell_pv_record_df.to_csv(self.parent_dir + '/sell_pv_record.csv', index=True)
            
            buy_battery_record_df = pd.DataFrame(self.buy_battery_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_battery_record_df.to_csv(self.parent_dir + '/buy_battery_record.csv', index=True)
            buy_ev_battery_record_df = pd.DataFrame(self.buy_ev_battery_record_arr, index=timestamp, columns=self.demand_df.columns)
            buy_ev_battery_record_df.to_csv(self.parent_dir + '/buy_ev_battery_record.csv', index=True)
            sell_battery_record_df = pd.DataFrame(self.sell_battery_record_arr, index=timestamp, columns=self.supply_df.columns)
            sell_battery_record_df.to_csv(self.parent_dir + '/sell_battery_record.csv', index=True)
            sell_ev_battery_record_df = pd.DataFrame(self.sell_ev_battery_record_arr, index=timestamp, columns=self.supply_df.columns)
            sell_ev_battery_record_df.to_csv(self.parent_dir + '/sell_ev_battery_record.csv', index=True)

            shift_df = pd.DataFrame(self.shift_arr, index=timestamp, columns=self.demand_df.columns)
            shift_df.to_csv(self.parent_dir + '/shift_record.csv', index=True)

            potential_demand_df = pd.DataFrame(self.potential_demand_arr, index=timestamp, columns=['Potential demand'])
            potential_demand_df.to_csv(self.parent_dir + '/potential_demand.csv', index=True)
            potential_supply_df = pd.DataFrame(self.potential_supply_arr, index=timestamp, columns=['Potential supply'])
            potential_supply_df.to_csv(self.parent_dir + '/potential_supply.csv', index=True)

            reward_df = pd.DataFrame(self.reward_arr, index=timestamp, columns=self.demand_df.columns)
            reward_df.to_csv(self.parent_dir + '/reward.csv', index=True)
            # This data is recorded as net cost
            net_electricity_cost_df = pd.DataFrame(self.electricity_cost_arr, index=timestamp, columns=self.demand_df.columns)
            net_electricity_cost_df.to_csv(self.parent_dir + '/net_electricity_cost.csv', index=True)
            self.car_charge_df.to_csv(self.parent_dir + '/car_charge_bool.csv', index=True)
            self.car_move_consumption_df.to_csv(self.parent_dir + '/car_move_consumption.csv', index=True)

            # JSON形式でパラメータを保存
            file_name = self.parent_dir + "/params.json"
            with open(file_name, 'w') as file:
                json.dump(self.params, file, indent=4)

            vis = visualize.Visualize(folder_path=self.parent_dir)
            vis.plot_consumption()

        self.q.save_q_table(folder_path = self.parent_dir, train=self.train)
        logger.info(f'Q table is saved to {self.parent_dir}')


if __name__ == '__main__':
    world = SimulationNoP2P(num_agent=10, parent_dir='output/no_p2p/debug', episode=1, train=True, thread_num=0)
    world.preprocess()
    world.run()