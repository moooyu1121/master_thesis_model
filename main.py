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
from multiprocessing import Pool
import logging
logger = logging.getLogger('Logging')
logger.setLevel(10)
fh = logging.FileHandler('main.log')
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s: line %(lineno)d: %(levelname)s: %(message)s')
fh.setFormatter(formatter)
np.seterr(all='raise')  # これでエラー発生時に例外が発生する

def main(num_agent, episode, BID_SAVE=False, **kwargs):
    # Adding the new mechanism to the list of available mechanism of the market
    pm.market.MECHANISM['uniform'] = UniformPrice # type: ignore
    # Update market and uniform parameters
    params = {'thread_num': -1,
              'load_q': False,
              'price_max': 120,
              'price_min': 5,
              'wheeling_charge': 10,
              'battery_charge_efficiency': 0.9,
              'battery_discharge_efficiency': 0.9,
              'ev_charge_efficiency': 0.9,
              'ev_discharge_efficiency': 0.9,
    }
    params.update(kwargs)
    thread_num = params['thread_num']
    load_q = params['load_q']
    price_max = params['price_max']
    price_min = params['price_min']
    wheeling_charge = params['wheeling_charge']
    battery_charge_efficiency = params['battery_charge_efficiency']
    battery_discharge_efficiency = params['battery_discharge_efficiency']
    ev_charge_efficiency = params['ev_charge_efficiency']
    ev_discharge_efficiency = params['ev_discharge_efficiency']

    # Initialize Q table
    q = Q(params, agent_num=num_agent, num_dizitized_pv_ratio=20, num_dizitized_soc=20)
    # Load Q table from previous episode, get average Q table
    if load_q:
        q.load_q_table(folder_path='output/average_q/' + str(episode-1))
    
    # ====================================================================================================
    # for episode in range(num_episode):
    if thread_num == -1:
        os.makedirs('output/episode' + str(episode) + '/', exist_ok=True)
        parent_dir = 'output/episode' + str(episode)
    else:
        os.makedirs('output/thread' + str(thread_num) + '/episode' + str(episode)  + '/', exist_ok=True)
        parent_dir = 'output/thread' + str(thread_num) + '/episode' + str(episode)
    
    # Preprocess and generate demand, supply, and price data
    preprocess = Preprocess()
    preprocess.set(
        pd.read_csv('data/demand.csv'),
        pd.read_csv('data/supply.csv'),
        pd.read_csv('data/price.csv')
    )
    preprocess.generate_d_s(num_agent)
    preprocess.save(parent_dir)
    preprocess.drop_index_  # drop timestamp index
    demand_df, supply_df, price_df = preprocess.get_dfs_

    # Generate agent parameters
    agents = Agent(num_agent)
    agents.generate_params()
    agents.save(parent_dir)
    agent_params_df = agents.get_agents_params_df_
    q.set_agent_params(agent_params_df)

    # get average pv production ratio to get state in Q table
    # data is stored as kWh/kW, which means, the values are within 0~1
    pv_ratio_df = pd.read_csv('data/supply.csv', index_col=0)
    pv_ratio_df['mean'] = pv_ratio_df.mean(axis=1)
    pv_ratio_arr = pv_ratio_df['mean'].values

    # Initialize record arrays
    grid_import_record_arr = np.full(len(price_df), 0.0)
    microgrid_price_record_arr = np.full(len(price_df), 999.0)
    battery_record_arr = np.full((len(demand_df), num_agent), 0.0)
    ev_battery_record_arr = np.full((len(demand_df), num_agent), 0.0)
    battery_soc_record_arr = np.full((len(demand_df), num_agent), 0.0)
    ev_battery_soc_record_arr = np.full((len(demand_df), num_agent), 0.0)
    buy_inelastic_record_arr = np.full((len(demand_df), num_agent), 0.0)
    buy_elastic_record_arr = np.full((len(demand_df), num_agent), 0.0)
    buy_shifted_record_arr = np.full((len(demand_df), num_agent), 0.0)
    buy_battery_record_arr = np.full((len(demand_df), num_agent), 0.0)
    buy_ev_battery_record_arr = np.full((len(demand_df), num_agent), 0.0)
    sell_pv_record_arr = np.full((len(supply_df), num_agent), 0.0)
    sell_battery_record_arr = np.full((len(supply_df), num_agent), 0.0)
    sell_ev_battery_record_arr = np.full((len(supply_df), num_agent), 0.0)
    reward_arr = np.full((len(demand_df), num_agent), 0.0)
    electricity_cost_arr = np.full((len(demand_df), num_agent), 0.0)
    
    # Generate elastic and inelastic demand according to the elastic ratio of each agent
    demand_elastic_arr = demand_df.values.copy()
    demand_inelastic_arr = demand_df.values.copy()
    for i in range(num_agent):
        demand_elastic_arr[:, i] = demand_df[f'{i}'] * agents[i]['elastic_ratio']
        demand_inelastic_arr[:, i] = demand_df[f'{i}'] * (1 - agents[i]['elastic_ratio'])

    # Prepare dataframe to record shifted demand
    # shift_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
    shift_arr = np.full((len(demand_df), num_agent), 0.0)
    
    # for t in tqdm(range(100)):
    for t in tqdm(range(len(demand_df))):
        demand_list = []
        supply_list = []
        wholesale_price = price_df.at[t, 'Price'] + wheeling_charge
        q.reset_all_digitized_states()
        q.reset_all_actions()
        for i in range(num_agent):
            q.set_digitized_states(agent_id=i,
                                pv_ratio=pv_ratio_arr[t],
                                battery_soc=battery_soc_record_arr[t, i],
                                ev_battery_soc=ev_battery_soc_record_arr[t, i])
            q.set_actions(agent_id=i, episode=episode)
            # 時刻tでのバッテリー残量を時刻t+1にコピー、取引があった場合バッテリー残量をさらに更新
            if t+1 != len(demand_df):
                battery_record_arr[t+1, i] = battery_record_arr[t, i]
                if agents[i]['battery_capacity'] != 0:
                    battery_soc_record_arr[t+1, i] = battery_record_arr[t+1, i] / agents[i]['battery_capacity']
                else:
                    battery_soc_record_arr[t+1, i] = 0.0
                ev_battery_record_arr[t+1, i] = ev_battery_record_arr[t, i]
                if agents[i]['ev_capacity'] != 0:
                    ev_battery_soc_record_arr[t+1, i] = ev_battery_record_arr[t+1, i] / agents[i]['ev_capacity']
                else:
                    ev_battery_soc_record_arr[t+1, i] = 0.0
            # ユーザIDはデマンドレスポンスによる移動を考慮して1エージェントごとに
            # リアルタイム(inelas, elas)，バッテリー充放電，ev充放電，PV発電供給，シフトリミット時間ステップ分の数IDを保有する
            # シフトリミットが24時間なら，31個IDを保有する
            # agentのIDは0～, 100～, 200～, 300～, ...として，101にagent1のinelas，102にagent1のelas...のように割り当てる
            id_base = i * 100
            # デマンドレスポンス不可の需要
            d_inelas = demand_inelastic_arr[t, i]
            demand_list.append([d_inelas, price_max, id_base+0, True])

            # デマンドレスポンス可能の需要
            d_elas_max = demand_elastic_arr[t, i]
            price_elas = q.get_actions[i, 0]
            # d_elas = d_elas_max * max((agents[i]['dr_price_threshold'] - price_elas)/(agents[i]['dr_price_threshold'] - price_min), 0)
            if price_elas == price_min:
                # To avoid missing intersection point of supply and demand curve
                price_elas += 0.00001
            # デマンドレスポンス可の需要はid_base+1に割り当てる
            demand_list.append([d_elas_max, price_elas, id_base+1, True])

            # バッテリー充放電価格の取得
            price_buy_battery = q.get_actions[i, 1]
            price_sell_battery = q.get_actions[i, 2]
            # バッテリー充放電可能量の取得
            battery_amount = battery_record_arr[t, i]
            if (agents[i]['battery_capacity'] - battery_amount) < (agents[i]['max_battery_charge_speed'] * battery_charge_efficiency):
                charge_amount = (agents[i]['battery_capacity'] - battery_amount) / battery_charge_efficiency
            else:
                charge_amount = agents[i]['max_battery_charge_speed']
            if battery_amount < (agents[i]['max_battery_discharge_speed'] / battery_discharge_efficiency):
                discharge_amount = battery_amount * battery_discharge_efficiency
            else:
                discharge_amount = agents[i]['max_battery_discharge_speed']
            if price_buy_battery == price_min:
                # To avoid missing intersection point of supply and demand curve
                price_buy_battery += 0.00001
            # バッテリー充電はid_base+2, 放電はid_base+3に割り当てる
            demand_list.append([charge_amount, price_buy_battery, id_base+2, True])
            supply_list.append([discharge_amount, price_sell_battery, id_base+3, False])

            # EV充放電価格の取得 
            price_buy_ev_battery = q.get_actions[i, 3]
            price_sell_ev_battery = q.get_actions[i, 4]
            # EV充放電可能量の取得
            ev_battery_amount = ev_battery_record_arr[t, i]
            if (agents[i]['ev_capacity'] - ev_battery_amount) < (agents[i]['max_ev_charge_speed'] * ev_charge_efficiency):
                ev_charge_amount = (agents[i]['ev_capacity'] - ev_battery_amount) / ev_charge_efficiency
            else:
                ev_charge_amount = agents[i]['max_ev_charge_speed']
            if ev_battery_amount < (agents[i]['max_ev_discharge_speed'] / ev_discharge_efficiency):
                ev_discharge_amount = ev_battery_amount * ev_discharge_efficiency
            else:
                ev_discharge_amount = agents[i]['max_ev_discharge_speed']
            if price_buy_ev_battery == price_min:
                # To avoid missing intersection point of supply and demand curve
                price_buy_ev_battery += 0.00001
            # EVバッテリー充電はid_base+4, 放電はid_base+5に割り当てる
            demand_list.append([ev_charge_amount, price_buy_ev_battery, id_base+4, True])
            supply_list.append([ev_discharge_amount, price_sell_ev_battery, id_base+5, False])

            # 供給
            s = supply_df.at[t, f'{i}']
            # 供給はid_base+6に割り当てる
            supply_list.append([s, price_min, id_base+6, False])

            # 後ろの時間にシフトさせる需要量の最大値を記録
            # マーケット取引をした後実際の取引があった場合，その分shiftする需要量を差し引くことで更新する
            shift_arr[t, i] = d_elas_max

            # 過去からシフトした需要の入札
            for k in range(t-int(agents[i]['shift_limit']), t):
                if k >= 0:
                    d_shift = shift_arr[k, i]
                    # シフトした需要の価格は，最低価格からしきい価格までシフトリミット時間ステップ分で線形に変化
                    price_shift = price_min + (agents[i]['dr_price_threshold'] - price_min) * (t-k) / agents[i]['shift_limit']
                    if k == t-int(agents[i]['shift_limit']):
                        # シフトリミットでの価格は最高価格
                        price_shift = price_max
                    # 過去からのシフトはj+7から割り当てる
                    demand_list.append([d_shift, price_shift, id_base+7+t-k-1, True])

            
            
        market = Market(demand_list, supply_list, wholesale_price)
        market.bid()
        bids_df = market.market.bm.get_df()
        
        # if episode == 0 or episode == num_episode-1 or episode%10 == 9:
        if BID_SAVE:
            timestamp = pd.read_csv('data/demand.csv').iat[t, 0]
            market.plot(title=timestamp, number=t, parent_dir=parent_dir)
        transactions_df, _ = market.run(mechanism='uniform')
        # print(bids_df)
        # print(transactions_df)
        # input()
        
        # マーケット取引の結果を記録、報酬を計算
        reward = np.full(num_agent, 0.0)
        cost = np.full(num_agent, 0.0)
        for bid_num in transactions_df['bid']:
            id = bids_df.at[bid_num, 'user']
            if id == 99999:
                # record import from grid
                grid_import_record_arr[t] = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]

            if id != 99999:
                # 100の位以降の数字を取り出す->agentID
                user = id // 100
                # リアルタイム(inelas, elas)@2，バッテリー充放電@2，ev充放電@2，シフトリミット@shift_limit，供給@1
                item = id % 100
                price = transactions_df[transactions_df['bid']==bid_num]['price'].values[0]
                if item == 0:
                    value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                    buy_inelastic_record_arr[t, user] = value
                    reward[user] -= value * price
                    cost[user] += value * price
                    if np.isnan(reward[user]):
                        print(item, value, price)
                        input()

                elif item == 1:
                    value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                    buy_elastic_record_arr[t, user] = value
                    # 時刻tでのDRの分だけ後ろの時間にシフトさせる需要量を減らす
                    shift_arr[t, user] -= value
                    reward[user] -= value * price
                    reward[user] -= (agents[int(user)]['alpha']/2 * (demand_elastic_arr[t, i] - value)**2 + 
                                    agents[int(user)]['beta']*(demand_elastic_arr[t, i] - value))
                    cost[user] += value * price
                    if np.isnan(reward[user]):
                        print(item, value, price, demand_elastic_arr[t, i])
                        input()

                elif item == 2:
                    value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                    buy_battery_record_arr[t, user] = value
                    if t+1 != len(demand_df):
                        battery_record_arr[t+1, user] += value * battery_charge_efficiency
                        battery_soc_record_arr[t+1, user] = battery_record_arr[t+1, user] / agents[user]['battery_capacity']
                    reward[user] -= value * price
                    reward[user] -= ((agents[int(user)]['max_battery_charge_speed'] - value) * 
                                    (agents[int(user)]['gamma']/2 * (1 * (1-battery_soc_record_arr[t, user]))**2 + 
                                    agents[int(user)]['epsilon']*(1 * (1-battery_soc_record_arr[t, user]))))
                    cost[user] += value * price
                    if np.isnan(reward[user]):
                        print(item, value, price, battery_soc_record_arr[t, user])
                        input()

                elif item == 3:
                    value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                    sell_battery_record_arr[t, user] = value
                    if t+1 !=len(demand_df):
                        battery_record_arr[t+1, user] -= value / battery_discharge_efficiency
                        battery_soc_record_arr[t+1, user] = battery_record_arr[t+1, user] / agents[user]['battery_capacity']
                    reward[user] += value * price
                    reward[user] -= ((agents[int(user)]['max_battery_charge_speed'] + value) * 
                                    (agents[int(user)]['gamma']/2 * (1 * (1-battery_soc_record_arr[t, user]))**2 + 
                                    agents[int(user)]['epsilon']*(1 * (1-battery_soc_record_arr[t, user]))))
                    cost[user] -= -value * price
                    if np.isnan(reward[user]):
                        print(item, value, price, battery_soc_record_arr[t, user])
                        input()

                elif item == 4:
                    value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                    buy_ev_battery_record_arr[t, user] = value
                    if t+1 !=len(demand_df):
                        ev_battery_record_arr[t+1, user] += value * ev_charge_efficiency
                        ev_battery_soc_record_arr[t+1, user] = ev_battery_record_arr[t+1, user] / agents[user]['ev_capacity']
                    reward[user] -= value * price
                    reward[user] -= ((agents[int(user)]['max_ev_charge_speed'] - value) *
                                    (agents[int(user)]['psi']/2 * (1 * (1-ev_battery_soc_record_arr[t, user]))**2 + 
                                    agents[int(user)]['omega']*(1 * (1-ev_battery_soc_record_arr[t, user]))))
                    cost[user] += value * price
                    if np.isnan(reward[user]):
                        print(item, value, price, ev_battery_soc_record_arr[t, user])
                        input()

                elif item == 5:
                    value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                    sell_ev_battery_record_arr[t, user] = value
                    if t+1 !=len(demand_df):
                        ev_battery_record_arr[t+1, user] -= value / ev_discharge_efficiency
                        ev_battery_soc_record_arr[t+1, user] = ev_battery_record_arr[t+1, user] / agents[user]['ev_capacity']
                    reward[user] += value * price
                    reward[user] -= ((agents[int(user)]['max_ev_charge_speed'] + value) *
                                    (agents[int(user)]['psi']/2 * (1 * (1-ev_battery_soc_record_arr[t, user]))**2 + 
                                    agents[int(user)]['omega']*(1 * (1-ev_battery_soc_record_arr[t, user]))))
                    cost[user] -= -value * price
                    if np.isnan(reward[user]):
                        print(item, value, price, ev_battery_soc_record_arr[t, user])
                        input()

                elif item == 6:
                    value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                    sell_pv_record_arr[t, user] = value
                    reward[user] += value * price
                    cost[user] -= -value * price
                    if np.isnan(reward[user]):
                        print(item, value, price)
                        input()

                else:
                    # buy_shifted_record_dfに足し上げながらshift_dfを更新
                    for k in range(7, 7+int(agents[user]['shift_limit'])):
                        if item == k:
                            value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                            buy_shifted_record_arr[t, user] += value
                            reward[user] -= value * price
                            cost[user] += value * price
                            if t-k+7-1>= 0:
                                shift_arr[t-k+7-1, user] -= value
        microgrid_price_record_arr[t] = transactions_df['price'].values[0]

        # Q学習
        dr_states, battery_states, ev_battery_states = q.get_states
        actions_arr = q.get_actions
        for i in range(num_agent):
            reward_arr[t, i] = reward[i]
            electricity_cost_arr[t, i] = cost[i]
            # バッテリー充放電とEVバッテリー充放電は同じstateで管理できるため重複している
            states = [int(dr_states[i]), int(battery_states[i]), int(battery_states[i]), int(ev_battery_states[i]), int(ev_battery_states[i])]
            actions = [actions_arr[i, 0], actions_arr[i, 1], actions_arr[i, 2], actions_arr[i, 3], actions_arr[i, 4]]
            rewards = [reward[i], reward[i], reward[i], reward[i], reward[i]]   # rewardは共通の値(すべての要素からのrewardの合計)
            if t == 0:
                previous_states = states
                previous_actions = actions
                previous_rewards = rewards
            else:
                q.update_q_table(states=previous_states, 
                                actions=previous_actions, 
                                rewards=previous_rewards, 
                                next_states=states)
                previous_states = states
                previous_actions = actions
                previous_rewards = rewards
        # print(reward_arr[t,:])
        # print(electricity_cost_arr[t,:])
        # input()
    # ====================================================================================================
    timestamp = pd.read_csv('data/demand.csv').iloc[:, 0]
    # parent_dir = 'output/episode' + str(episode)

    grid_import_record_df = pd.DataFrame(grid_import_record_arr, index=timestamp, columns=['Grid import'])
    grid_import_record_df.to_csv(parent_dir + '/grid_import_record.csv', index=True)
    microgrid_price_record_df = pd.DataFrame(microgrid_price_record_arr, index=timestamp, columns=['Price'])
    microgrid_price_record_df.to_csv(parent_dir + '/price_record.csv', index=True)
    battery_record_df = pd.DataFrame(battery_record_arr, index=timestamp, columns=demand_df.columns)
    battery_record_df.to_csv(parent_dir + '/battery_record.csv', index=True)
    ev_battery_record_df = pd.DataFrame(ev_battery_record_arr, index=timestamp, columns=demand_df.columns)
    ev_battery_record_df.to_csv(parent_dir + '/ev_battery_record.csv', index=True)
    battery_soc_record_df = pd.DataFrame(battery_soc_record_arr, index=timestamp, columns=demand_df.columns)
    battery_soc_record_df.to_csv(parent_dir + '/battery_soc_record.csv', index=True)
    ev_battery_soc_record_df = pd.DataFrame(ev_battery_soc_record_arr, index=timestamp, columns=demand_df.columns)
    ev_battery_soc_record_df.to_csv(parent_dir + '/ev_battery_soc_record.csv', index=True)

    buy_inelastic_record_df = pd.DataFrame(buy_inelastic_record_arr, index=timestamp, columns=demand_df.columns)
    buy_inelastic_record_df.to_csv(parent_dir + '/buy_inelastic_record.csv', index=True)
    buy_elastic_record_df = pd.DataFrame(buy_elastic_record_arr, index=timestamp, columns=demand_df.columns)
    buy_elastic_record_df.to_csv(parent_dir + '/buy_elastic_record.csv', index=True)
    buy_shifted_record_df = pd.DataFrame(buy_shifted_record_arr, index=timestamp, columns=demand_df.columns)
    buy_shifted_record_df.to_csv(parent_dir + '/buy_shifted_record.csv', index=True)
    sell_pv_record_df = pd.DataFrame(sell_pv_record_arr, index=timestamp, columns=supply_df.columns)
    sell_pv_record_df.to_csv(parent_dir + '/sell_pv_record.csv', index=True)
    
    buy_battery_record_df = pd.DataFrame(buy_battery_record_arr, index=timestamp, columns=demand_df.columns)
    buy_battery_record_df.to_csv(parent_dir + '/buy_battery_record.csv', index=True)
    buy_ev_battery_record_df = pd.DataFrame(buy_ev_battery_record_arr, index=timestamp, columns=demand_df.columns)
    buy_ev_battery_record_df.to_csv(parent_dir + '/buy_ev_battery_record.csv', index=True)
    sell_battery_record_df = pd.DataFrame(sell_battery_record_arr, index=timestamp, columns=supply_df.columns)
    sell_battery_record_df.to_csv(parent_dir + '/sell_battery_record.csv', index=True)
    sell_ev_battery_record_df = pd.DataFrame(sell_ev_battery_record_arr, index=timestamp, columns=supply_df.columns)
    sell_ev_battery_record_df.to_csv(parent_dir + '/sell_ev_battery_record.csv', index=True)

    shift_df = pd.DataFrame(shift_arr, index=timestamp, columns=demand_df.columns)
    shift_df.to_csv(parent_dir + '/shift_record.csv', index=True)

    reward_df = pd.DataFrame(reward_arr, index=timestamp, columns=demand_df.columns)
    reward_df.to_csv(parent_dir + '/reward.csv', index=True)
    electricity_cost_df = pd.DataFrame(electricity_cost_arr, index=timestamp, columns=demand_df.columns)
    electricity_cost_df.to_csv(parent_dir + '/electricity_cost.csv', index=True)
    q.save_q_table(folder_path = parent_dir)

    vis = visualize.Visualize(folder_path=parent_dir)
    vis.plot_consumption()


def main_wrapper(args):
    return main(**args)


if __name__ == "__main__":
    # Adding the new mechanism to the list of available mechanism of the market
    pm.market.MECHANISM['uniform'] = UniformPrice # type: ignore
    # pd.set_option('display.max_rows', None)  # 全行表示

    # main(num_agent=50, num_episode=51, BID_SAVE=True, price_min=10)
    
    # 並列処理で10エピソード行って，qテーブルを平均値で更新，また並列処理で10エピソード行う
    # 100エピソードまで行う．

    # max_workers = os.cpu_count()
    max_workers = 10
    
    p = Pool(max_workers)
    values = [{'num_agent': 50, 'episode': 1, 'BID_SAVE': False, 'price_min': 10, 'thread_num': x} for x in range(max_workers)]
    p.map(main_wrapper, values)

    p.close()
    p.join()

    os.makedirs('output/average_q/1' + '/', exist_ok=True)
    dr_buy_qtb_list = glob.glob('output/*/episode1/dr_buy_qtb.npy')
    battery_buy_qtb_list = glob.glob('output/*/episode1/battery_buy_qtb.npy')
    battery_sell_qtb_list = glob.glob('output/*/episode1/battery_sell_qtb.npy')
    ev_battery_buy_qtb_list = glob.glob('output/*/episode1/ev_battery_buy_qtb.npy')
    ev_battery_sell_qtb_list = glob.glob('output/*/episode1/ev_battery_sell_qtb.npy')
    for thread in range(max_workers):
        dr_buy_qtb = np.load(dr_buy_qtb_list[thread])
        battery_buy_qtb = np.load(battery_buy_qtb_list[thread])
        battery_sell_qtb = np.load(battery_sell_qtb_list[thread])
        ev_battery_buy_qtb = np.load(ev_battery_buy_qtb_list[thread])
        ev_battery_sell_qtb = np.load(ev_battery_sell_qtb_list[thread])
        if thread == 0:
            average_dr_buy_qtb = dr_buy_qtb
            average_battery_buy_qtb = battery_buy_qtb
            average_battery_sell_qtb = battery_sell_qtb
            average_ev_battery_buy_qtb = ev_battery_buy_qtb
            average_ev_battery_sell_qtb = ev_battery_sell_qtb
        else:
            average_dr_buy_qtb += dr_buy_qtb
            average_battery_buy_qtb += battery_buy_qtb
            average_battery_sell_qtb += battery_sell_qtb
            average_ev_battery_buy_qtb += ev_battery_buy_qtb
            average_ev_battery_sell_qtb += ev_battery_sell_qtb
    average_dr_buy_qtb /= max_workers
    average_battery_buy_qtb /= max_workers
    average_battery_sell_qtb /= max_workers
    average_ev_battery_buy_qtb /= max_workers
    average_ev_battery_sell_qtb /= max_workers
    np.save('output/average_q/1/dr_buy_qtb.npy', average_dr_buy_qtb)
    np.save('output/average_q/1/battery_buy_qtb.npy', average_battery_buy_qtb)
    np.save('output/average_q/1/battery_sell_qtb.npy', average_battery_sell_qtb)
    np.save('output/average_q/1/ev_battery_buy_qtb.npy', average_ev_battery_buy_qtb)
    np.save('output/average_q/1/ev_battery_sell_qtb.npy', average_ev_battery_sell_qtb)
    df = pd.DataFrame(average_dr_buy_qtb)
    df.to_csv('output/average_q/1/dr_buy_qtb.csv')
    df = pd.DataFrame(average_battery_buy_qtb)
    df.to_csv('output/average_q/1/battery_buy_qtb.csv')
    df = pd.DataFrame(average_battery_sell_qtb)
    df.to_csv('output/average_q/1/battery_sell_qtb.csv')
    df = pd.DataFrame(average_ev_battery_buy_qtb)
    df.to_csv('output/average_q/1/ev_battery_buy_qtb.csv')
    df = pd.DataFrame(average_ev_battery_sell_qtb)
    df.to_csv('output/average_q/1/ev_battery_sell_qtb.csv')

    print('episode 1 finished.')

    for episode in range(2, 101):
        p = Pool(max_workers)
        values = [{'num_agent': 50, 'episode': episode, 'BID_SAVE': False, 'price_min': 10, 'thread_num': x, 'load_q': True} for x in range(max_workers)]
        p.map(main_wrapper, values)

        p.close()
        p.join()

        os.makedirs('output/average_q/' + str(episode) + '/', exist_ok=True)
        dr_buy_qtb_list = glob.glob('output/*/episode' + str(episode) + '/dr_buy_qtb.npy')
        battery_buy_qtb_list = glob.glob('output/*/episode' + str(episode) + '/battery_buy_qtb.npy')
        battery_sell_qtb_list = glob.glob('output/*/episode' + str(episode) + '/battery_sell_qtb.npy')
        ev_battery_buy_qtb_list = glob.glob('output/*/episode' + str(episode) + '/ev_battery_buy_qtb.npy')
        ev_battery_sell_qtb_list = glob.glob('output/*/episode' + str(episode) + '/ev_battery_sell_qtb.npy')

        for thread in range(max_workers):
            dr_buy_qtb = np.load(dr_buy_qtb_list[thread])
            battery_buy_qtb = np.load(battery_buy_qtb_list[thread])
            battery_sell_qtb = np.load(battery_sell_qtb_list[thread])
            ev_battery_buy_qtb = np.load(ev_battery_buy_qtb_list[thread])
            ev_battery_sell_qtb = np.load(ev_battery_sell_qtb_list[thread])
            if thread == 0:
                average_dr_buy_qtb = dr_buy_qtb
                average_battery_buy_qtb = battery_buy_qtb
                average_battery_sell_qtb = battery_sell_qtb
                average_ev_battery_buy_qtb = ev_battery_buy_qtb
                average_ev_battery_sell_qtb = ev_battery_sell_qtb
            else:
                average_dr_buy_qtb += dr_buy_qtb
                average_battery_buy_qtb += battery_buy_qtb
                average_battery_sell_qtb += battery_sell_qtb
                average_ev_battery_buy_qtb += ev_battery_buy_qtb
                average_ev_battery_sell_qtb += ev_battery_sell_qtb
        average_dr_buy_qtb /= max_workers
        average_battery_buy_qtb /= max_workers
        average_battery_sell_qtb /= max_workers
        average_ev_battery_buy_qtb /= max_workers
        average_ev_battery_sell_qtb /= max_workers
        np.save('output/average_q/' + str(episode) + '/dr_buy_qtb.npy', average_dr_buy_qtb)
        np.save('output/average_q/' + str(episode) + '/battery_buy_qtb.npy', average_battery_buy_qtb)
        np.save('output/average_q/' + str(episode) + '/battery_sell_qtb.npy', average_battery_sell_qtb)
        np.save('output/average_q/' + str(episode) + '/ev_battery_buy_qtb.npy', average_ev_battery_buy_qtb)
        np.save('output/average_q/' + str(episode) + '/ev_battery_sell_qtb.npy', average_ev_battery_sell_qtb)
        df = pd.DataFrame(average_dr_buy_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/dr_buy_qtb.csv')
        df = pd.DataFrame(average_battery_buy_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/battery_buy_qtb.csv')
        df = pd.DataFrame(average_battery_sell_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/battery_sell_qtb.csv')
        df = pd.DataFrame(average_ev_battery_buy_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/ev_battery_buy_qtb.csv')
        df = pd.DataFrame(average_ev_battery_sell_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/ev_battery_sell_qtb.csv')
        print(f'episode {episode} finished.')
    print('All episodes finished.')
