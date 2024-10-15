import numpy as np 
import pandas as pd
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
                  'price_max': 120,
                  'price_min': 10,
                  'wheeling_charge': 10,
                  'battery_charge_efficiency': 0.9,
                  'battery_discharge_efficiency': 0.9,
                  'ev_charge_efficiency': 0.9,
                  'ev_discharge_efficiency': 0.9,
                  'ev_efficiency': 7,  # km/kWh
                  'car_movement_speed': 30,  # km/h
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
        self.ev_efficiency = params['ev_efficiency']
        self.car_movement_speed = params['car_movement_speed']

        # Initialize Q table
        self.q = Q(params, agent_num=num_agent, num_dizitized_pv_ratio=20, num_dizitized_soc=20, num_elastic_ratio_pattern=3)
    
    def load_existing_q_table(self, folder_path):
        self.q.load_q_table(folder_path=folder_path)

    def preprocess(self):
        # Generate agent parameters
        self.agents = Agent(self.num_agent)
        self.agents.generate_params(seed=self.thread_num)
        
        # Preprocess and generate demand, price, and car_movement(boolean) data
        preprocess = Preprocess(seed=self.thread_num)
        preprocess.set(
            pd.read_csv('data/demand.csv'),
            pd.read_csv('data/supply.csv'),
            pd.read_csv('data/price.csv'),
            pd.read_csv('data/car_movement.csv')
        )
        # preprocess.generate_d_s(self.num_agent)
        preprocess.generate_demand(self.num_agent)
        _, agent_car_categories = preprocess.generate_car_movement(self.num_agent)

        # Save car movement categories data to agent_params_df
        self.agents.set_car_movement_categories(agent_car_categories)
        self.agents.save(self.parent_dir)
        agent_params_df = self.agents.get_agents_params_df_

        pv_capacity_list = agent_params_df['pv_capacity'].values
        # Generate supply data
        preprocess.generate_supply_flex_pv_size(self.num_agent, pv_capacity_list)
        preprocess.save(self.parent_dir)
        preprocess.drop_index_  # drop timestamp index
        self.demand_df, self.supply_df, self.price_df, self.car_movement_df, self.elastic_ratio_df = preprocess.get_dfs_

        # get average pv production ratio to get state in Q table
        # data is stored as kWh/kW, which means, the values are within 0~1
        pv_ratio_df = pd.read_csv('data/supply.csv', index_col=0)
        pv_ratio_df['mean'] = pv_ratio_df.mean(axis=1)
        self.pv_ratio_arr = pv_ratio_df['mean'].values

        # Initialize record arrays
        self.grid_import_record_arr = np.full(len(self.price_df), 0.0)
        self.microgrid_price_record_arr = np.full(len(self.price_df), 999.0)
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
            if self.agents[i]['dr_boolean_list'] == False:
                self.demand_elastic_arr[:, i] = 0
                self.demand_inelastic_arr[:, i] = self.demand_df[f'{i}']
            elif self.agents[i]['dr_boolean_list'] == True:
                self.demand_elastic_arr[:, i] = self.demand_df[f'{i}'] * self.elastic_ratio_df["elastic_ratio"]
                self.demand_inelastic_arr[:, i] = self.demand_df[f'{i}'] * (1 - self.elastic_ratio_df["elastic_ratio"])
            else:
                raise ValueError("DR boolean key is invalid.")

        # Prepare dataframe to record shifted demand
        # shift_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        self.shift_arr = np.full((len(self.demand_df), self.num_agent), 0.0)

    def run(self, BID_SAVE=False):
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
                dr_state, battery_state, ev_battery_state = self.q.set_digitized_states(agent_id=i,
                                                            pv_ratio=self.pv_ratio_arr[t],
                                                            battery_soc=self.battery_soc_record_arr[t, i],
                                                            ev_battery_soc=self.ev_battery_soc_record_arr[t, i],
                                                            elastic_ratio=self.elastic_ratio_df.at[t, "elastic_ratio"])
                # Qテーブルから行動を取得, ε-greedy法で徐々に最適行動を選択する式が、エピソード0から始まるように定義されているので、エピソード-1を引数に渡す
                self.q.set_actions(agent_id=i, episode=self.episode-1, is_train=self.train)
                # 時刻tでのバッテリー残量を時刻t+1にコピー、取引が行われる場合あとでバッテリー残量をさらに更新
                # car_movement_dfがTrueの場合は1時間走行したとして消費したバッテリー量を時刻t+1に記録
                # EVバッテリー残量が負の値になる場合もここではそのままにして、報酬を計算するフェーズで対応、0に更新するとともに-10000を報酬に反映
                if t+1 != len(self.demand_df):
                    self.battery_record_arr[t+1, i] = self.battery_record_arr[t, i]
                    if self.agents[i]['battery_capacity'] != 0:
                        self.battery_soc_record_arr[t+1, i] = self.battery_record_arr[t+1, i] / self.agents[i]['battery_capacity']
                    else:
                        self.battery_soc_record_arr[t+1, i] = 0.0

                    if self.car_movement_df.at[t, f'{i}'] and self.agents[i]['ev_capacity'] != 0:
                        self.ev_battery_record_arr[t+1, i] = self.ev_battery_record_arr[t, i] - self.car_movement_speed / self.ev_efficiency
                    else:
                        self.ev_battery_record_arr[t+1, i] = self.ev_battery_record_arr[t, i]
                    if self.agents[i]['ev_capacity'] != 0:
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
                if self.car_movement_df.at[t, f'{i}']:
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
                supply_list.append([s, self.price_min, id_base+6, False])
                potential_supply += s

                # 後ろの時間にシフトさせる需要量の最大値を記録
                # マーケット取引をした後実際の取引があった場合，その分shiftする需要量を差し引くことで更新する
                self.shift_arr[t, i] = d_elas_max

                # 過去からシフトした需要の入札
                for k in range(t-int(self.agents[i]['shift_limit']), t):
                    if k >= 0:
                        d_shift = self.shift_arr[k, i]
                        # シフトした需要の価格は，最低価格からしきい価格までシフトリミット時間ステップ分で線形に変化
                        price_shift = self.price_min + (self.agents[i]['dr_price_threshold'] - self.price_min) * (t-k) / self.agents[i]['shift_limit']
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
            
            # if episode == 0 or episode == num_episode-1 or episode%10 == 9:
            if BID_SAVE:
                timestamp = pd.read_csv('data/demand.csv').iat[t, 0]
                market.plot(title=timestamp, number=t, parent_dir=self.parent_dir)
            transactions_df, _ = market.run(mechanism='uniform')
            # print(bids_df)
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
                        reward[user] -= value * price
                        cost[user] += value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: inelastic, {value}, {price}')

                    elif item == 1:
                        # リアルタイム(elas)の取引量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.buy_elastic_record_arr[t, user] = value
                        # 時刻tでのDRの分だけ後ろの時間にシフトさせる需要量を減らす
                        self.shift_arr[t, user] -= value
                        reward[user] -= value * price
                        reward[user] -= (self.agents[int(user)]['alpha']/2 * (self.demand_elastic_arr[t, i] - value)**2 + 
                                        self.agents[int(user)]['beta']*(self.demand_elastic_arr[t, i] - value))
                        cost[user] += value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: elastic, {value}, {price}, {self.demand_elastic_arr[t, i]}')

                    elif item == 2:
                        # バッテリー充電の取引量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.buy_battery_record_arr[t, user] = value
                        if t+1 != len(self.demand_df):
                            self.battery_record_arr[t+1, user] += value * self.battery_charge_efficiency
                            self.battery_soc_record_arr[t+1, user] = self.battery_record_arr[t+1, user] / self.agents[user]['battery_capacity']
                        reward[user] -= value * price
                        # reward[user] -= ((self.agents[int(user)]['max_battery_charge_speed'] - value) * 
                        #                 (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t, user]))**2 + 
                        #                 self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t, user]))))
                        reward[user] -= (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t, user]))**2 + 
                                        self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t, user])))
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
                        reward[user] += value * price
                        # reward[user] -= ((self.agents[int(user)]['max_battery_charge_speed'] + value) * 
                        #                 (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t, user]))**2 + 
                        #                 self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t, user]))))
                        reward[user] -= (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t, user]))**2 + 
                                        self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t, user])))
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
                        reward[user] -= value * price
                        # reward[user] -= ((self.agents[int(user)]['max_ev_charge_speed'] - value) *
                        #                 (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t, user]))**2 + 
                        #                 self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t, user]))))
                        reward[user] -= (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t, user]))**2 + 
                                        self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t, user])))
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
                        reward[user] += value * price
                        # reward[user] -= ((self.agents[int(user)]['max_ev_charge_speed'] + value) *
                        #                 (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t, user]))**2 + 
                        #                 self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t, user]))))
                        reward[user] -= (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t, user]))**2 + 
                                        self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t, user])))
                        cost[user] -= value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: ev discharge, {value}, {price}, {self.ev_battery_soc_record_arr[t, user]}')

                    elif item == 6:
                        # PV発電供給量を記録
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.sell_pv_record_arr[t, user] = value
                        reward[user] += value * price
                        cost[user] -= value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: pv, {value}, {price}')

                    else:
                        # buy_shifted_record_dfに足し上げながらshift_dfを更新
                        for k in range(7, 7+int(self.agents[user]['shift_limit'])):
                            if item == k:
                                value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                                self.buy_shifted_record_arr[t, user] += value
                                reward[user] -= value * price
                                cost[user] += value * price
                                if t-k+7-1>= 0:
                                    self.shift_arr[t-k+7-1, user] -= value
            
            # EV SoCが0未満になっている場合は0にする、報酬に-10000を反映
            for i in range(self.num_agent):
                if t+1 != len(self.demand_df):
                    if self.ev_battery_soc_record_arr[t+1, i] < 0:
                        self.ev_battery_soc_record_arr[t+1, i] = 0
                        reward[i] -= 10000

            self.microgrid_price_record_arr[t] = transactions_df['price'].values[0]

            # Q学習
            dr_states, battery_states, ev_battery_states = self.q.get_states_
            actions_arr = self.q.get_actions_
            if t == 0:
                previous_states = []
                previous_actions = []
                previous_rewards = []
            for i in range(self.num_agent):
                self.reward_arr[t, i] = reward[i]
                self.electricity_cost_arr[t, i] = cost[i] / 100  # record cost in dollar, not cents
                # バッテリーの充放電、EVバッテリーの充放電はそれぞれ同じstateで管理できるため重複している
                states = [int(dr_states[i]), int(battery_states[i]), int(battery_states[i]), int(ev_battery_states[i]), int(ev_battery_states[i])]
                actions = [actions_arr[i, 0], actions_arr[i, 1], actions_arr[i, 2], actions_arr[i, 3], actions_arr[i, 4]]
                rewards = [reward[i], reward[i], reward[i], reward[i], reward[i]]   # rewardは共通の値(すべての要素からのrewardの合計)
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
        self.q.save_q_table(folder_path = self.parent_dir)
        self.car_movement_df.to_csv(self.parent_dir + '/car_movement.csv', index=True)

        vis = visualize.Visualize(folder_path=self.parent_dir)
        vis.plot_consumption()
