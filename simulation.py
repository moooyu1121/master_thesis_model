import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import pymarket as pm
from tqdm import tqdm
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
        os.makedirs(parent_dir, exist_ok=True)
        self.num_agent = num_agent
        self.parent_dir = parent_dir
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

    def preprocess(self):
        # Preprocess and generate demand, supply, and price data
        preprocess = Preprocess(seed=self.thread_num)
        preprocess.set(
            pd.read_csv('data/demand.csv'),
            pd.read_csv('data/supply.csv'),
            pd.read_csv('data/price.csv')
        )
        preprocess.generate_d_s(self.num_agent)
        preprocess.save(self.parent_dir)
        preprocess.drop_index_  # drop timestamp index
        self.demand_df, self.supply_df, self.price_df = preprocess.get_dfs_

        # Generate agent parameters
        self.agents = Agent(self.num_agent)
        self.agents.generate_params(seed=self.thread_num)
        self.agents.save(self.parent_dir)
        agent_params_df = self.agents.get_agents_params_df_
        self.q.set_agent_params(agent_params_df)

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
        
        # Generate elastic and inelastic demand according to the elastic ratio of each agent
        self.demand_elastic_arr = self.demand_df.values.copy()
        self.demand_inelastic_arr = self.demand_df.values.copy()
        for i in range(self.num_agent):
            self.demand_elastic_arr[:, i] = self.demand_df[f'{i}'] * self.agents[i]['elastic_ratio']
            self.demand_inelastic_arr[:, i] = self.demand_df[f'{i}'] * (1 - self.agents[i]['elastic_ratio'])

        # Prepare dataframe to record shifted demand
        # shift_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        self.shift_arr = np.full((len(self.demand_df), self.num_agent), 0.0)

    def run(self, BID_SAVE=False):
        for t in tqdm(range(len(self.demand_df))):
            demand_list = []
            supply_list = []
            wholesale_price = self.price_df.at[t, 'Price'] + self.wheeling_charge
            self.q.reset_all_digitized_states()
            self.q.reset_all_actions()
            for i in range(self.num_agent):
                self.q.set_digitized_states(agent_id=i,
                                    pv_ratio=self.pv_ratio_arr[t],
                                    battery_soc=self.battery_soc_record_arr[t, i],
                                    ev_battery_soc=self.ev_battery_soc_record_arr[t, i])
                self.q.set_actions(agent_id=i, episode=self.episode)
                # 時刻tでのバッテリー残量を時刻t+1にコピー、取引が行われる場合あとでバッテリー残量をさらに更新
                if t+1 != len(self.demand_df):
                    self.battery_record_arr[t+1, i] = self.battery_record_arr[t, i]
                    if self.agents[i]['battery_capacity'] != 0:
                        self.battery_soc_record_arr[t+1, i] = self.battery_record_arr[t+1, i] / self.agents[i]['battery_capacity']
                    else:
                        self.battery_soc_record_arr[t+1, i] = 0.0
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

                # デマンドレスポンス可能の需要
                d_elas_max = self.demand_elastic_arr[t, i]
                price_elas = self.q.get_actions[i, 0]
                # d_elas = d_elas_max * max((agents[i]['dr_price_threshold'] - price_elas)/(agents[i]['dr_price_threshold'] - price_min), 0)
                if price_elas == self.price_min:
                    # To avoid missing intersection point of supply and demand curve
                    price_elas += 0.00001
                # デマンドレスポンス可の需要はid_base+1に割り当てる
                demand_list.append([d_elas_max, price_elas, id_base+1, True])

                # バッテリー充放電価格の取得
                price_buy_battery = self.q.get_actions[i, 1]
                price_sell_battery = self.q.get_actions[i, 2]
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

                # EV充放電価格の取得 
                price_buy_ev_battery = self.q.get_actions[i, 3]
                price_sell_ev_battery = self.q.get_actions[i, 4]
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
                if price_buy_ev_battery == self.price_min:
                    # To avoid missing intersection point of supply and demand curve
                    price_buy_ev_battery += 0.00001
                # EVバッテリー充電はid_base+4, 放電はid_base+5に割り当てる
                demand_list.append([ev_charge_amount, price_buy_ev_battery, id_base+4, True])
                supply_list.append([ev_discharge_amount, price_sell_ev_battery, id_base+5, False])

                # 供給
                s = self.supply_df.at[t, f'{i}']
                # 供給はid_base+6に割り当てる
                supply_list.append([s, self.price_min, id_base+6, False])

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
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.buy_inelastic_record_arr[t, user] = value
                        reward[user] -= value * price
                        cost[user] += value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: inelastic, {value}, {price}')

                    elif item == 1:
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
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.buy_battery_record_arr[t, user] = value
                        if t+1 != len(self.demand_df):
                            self.battery_record_arr[t+1, user] += value * self.battery_charge_efficiency
                            self.battery_soc_record_arr[t+1, user] = self.battery_record_arr[t+1, user] / self.agents[user]['battery_capacity']
                        reward[user] -= value * price
                        reward[user] -= ((self.agents[int(user)]['max_battery_charge_speed'] - value) * 
                                        (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t, user]))**2 + 
                                        self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t, user]))))
                        cost[user] += value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: battery charge, {value}, {price}, {self.battery_soc_record_arr[t, user]}')

                    elif item == 3:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.sell_battery_record_arr[t, user] = value
                        if t+1 !=len(self.demand_df):
                            self.battery_record_arr[t+1, user] -= value / self.battery_discharge_efficiency
                            self.battery_soc_record_arr[t+1, user] = self.battery_record_arr[t+1, user] / self.agents[user]['battery_capacity']
                        reward[user] += value * price
                        reward[user] -= ((self.agents[int(user)]['max_battery_charge_speed'] + value) * 
                                        (self.agents[int(user)]['gamma']/2 * (1 * (1-self.battery_soc_record_arr[t, user]))**2 + 
                                        self.agents[int(user)]['epsilon']*(1 * (1-self.battery_soc_record_arr[t, user]))))
                        cost[user] -= -value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: battery discharge, {value}, {price}, {self.battery_soc_record_arr[t, user]}')

                    elif item == 4:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.buy_ev_battery_record_arr[t, user] = value
                        if t+1 !=len(self.demand_df):
                            self.ev_battery_record_arr[t+1, user] += value * self.ev_charge_efficiency
                            self.ev_battery_soc_record_arr[t+1, user] = self.ev_battery_record_arr[t+1, user] / self.agents[user]['ev_capacity']
                        reward[user] -= value * price
                        reward[user] -= ((self.agents[int(user)]['max_ev_charge_speed'] - value) *
                                        (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t, user]))**2 + 
                                        self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t, user]))))
                        cost[user] += value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: ev charge, {value}, {price}, {self.ev_battery_soc_record_arr[t, user]}')

                    elif item == 5:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.sell_ev_battery_record_arr[t, user] = value
                        if t+1 !=len(self.demand_df):
                            self.ev_battery_record_arr[t+1, user] -= value / self.ev_discharge_efficiency
                            self.ev_battery_soc_record_arr[t+1, user] = self.ev_battery_record_arr[t+1, user] / self.agents[user]['ev_capacity']
                        reward[user] += value * price
                        reward[user] -= ((self.agents[int(user)]['max_ev_charge_speed'] + value) *
                                        (self.agents[int(user)]['psi']/2 * (1 * (1-self.ev_battery_soc_record_arr[t, user]))**2 + 
                                        self.agents[int(user)]['omega']*(1 * (1-self.ev_battery_soc_record_arr[t, user]))))
                        cost[user] -= -value * price
                        if np.isnan(reward[user]):
                            logger.error(f'Numpy nan is detected: ev discharge, {value}, {price}, {self.ev_battery_soc_record_arr[t, user]}')

                    elif item == 6:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        self.sell_pv_record_arr[t, user] = value
                        reward[user] += value * price
                        cost[user] -= -value * price
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
            self.microgrid_price_record_arr[t] = transactions_df['price'].values[0]

            # Q学習
            dr_states, battery_states, ev_battery_states = self.q.get_states
            actions_arr = self.q.get_actions
            for i in range(self.num_agent):
                self.reward_arr[t, i] = reward[i]
                self.electricity_cost_arr[t, i] = cost[i]
                # バッテリーの充放電、EVバッテリーの充放電はそれぞれ同じstateで管理できるため重複している
                states = [int(dr_states[i]), int(battery_states[i]), int(battery_states[i]), int(ev_battery_states[i]), int(ev_battery_states[i])]
                actions = [actions_arr[i, 0], actions_arr[i, 1], actions_arr[i, 2], actions_arr[i, 3], actions_arr[i, 4]]
                rewards = [reward[i], reward[i], reward[i], reward[i], reward[i]]   # rewardは共通の値(すべての要素からのrewardの合計)
                if t == 0:
                    previous_states = states
                    previous_actions = actions
                    previous_rewards = rewards
                else:
                    self.q.update_q_table(states=previous_states, 
                                          actions=previous_actions, 
                                          rewards=previous_rewards, 
                                          next_states=states)
                    previous_states = states
                    previous_actions = actions
                    previous_rewards = rewards

    def save(self):
        timestamp = pd.read_csv('data/demand.csv').iloc[:, 0]
        # parent_dir = 'output/episode' + str(episode)

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

        reward_df = pd.DataFrame(self.reward_arr, index=timestamp, columns=self.demand_df.columns)
        reward_df.to_csv(self.parent_dir + '/reward.csv', index=True)
        electricity_cost_df = pd.DataFrame(self.electricity_cost_arr, index=timestamp, columns=self.demand_df.columns)
        electricity_cost_df.to_csv(self.parent_dir + '/electricity_cost.csv', index=True)
        self.q.save_q_table(folder_path = self.parent_dir)

        vis = visualize.Visualize(folder_path=self.parent_dir)
        vis.plot_consumption()