import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import pymarket as pm
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore', FutureWarning)
import market
import visualize
from preprocess import Preprocess
from market import Market
from agent import Agent
from q import Q
import logging
logger = logging.getLogger('Logging')
logger.setLevel(10)
fh = logging.FileHandler('main.log')
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s: line %(lineno)d: %(levelname)s: %(message)s')
fh.setFormatter(formatter)


def main(num_agent, num_episode, BID_SAVE=False, **kwargs):
    # Update market and uniform parameters
    params = {'price_max': 120,
              'price_min': 5,
              'wheeling_charge': 10,
              'battery_charge_efficiency': 0.9,
              'battery_discharge_efficiency': 0.9,
              'ev_charge_efficiency': 0.9,
              'ev_discharge_efficiency': 0.9,
    }
    params.update(kwargs)
    price_max = params['price_max']
    price_min = params['price_min']
    wheeling_charge = params['wheeling_charge']
    battery_charge_efficiency = params['battery_charge_efficiency']
    battery_discharge_efficiency = params['battery_discharge_efficiency']
    ev_charge_efficiency = params['ev_charge_efficiency']
    ev_discharge_efficiency = params['ev_discharge_efficiency']

    # Initialize Q table
    q = Q(params, agent_num=num_agent, num_dizitized_pv_ratio=20, num_dizitized_soc=20)
    
    # ====================================================================================================
    for episode in range(num_episode):
        os.makedirs('output/episode' + str(episode) + '/', exist_ok=True)
        
        # Preprocess and generate demand, supply, and price data
        preprocess = Preprocess()
        preprocess.set(
            pd.read_csv('data/demand.csv'),
            pd.read_csv('data/supply.csv'),
            pd.read_csv('data/price.csv')
        )
        preprocess.generate_d_s(num_agent)
        preprocess.save('output/episode' + str(episode))
        preprocess.drop_index_  # drop timestamp index
        demand_df, supply_df, price_df = preprocess.get_dfs_

        # Generate agent parameters
        agents = Agent(num_agent)
        agents.generate_params()
        agents.save('output/episode' + str(episode))
        agent_params_df = agents.get_agents_params_df_
        q.set_agent_params(agent_params_df)

        # get average pv production ratio to get state in Q table
        # data is stored as kWh/kW, which means, the values are within 0~1
        pv_ratio_df = pd.read_csv('data/supply.csv', index_col=0)
        pv_ratio_df['mean'] = pv_ratio_df.mean(axis=1)
        pv_ratio_df.reset_index(inplace=True, drop=True)

        # Initialize record dataframes
        microgrid_price_record_df = pd.DataFrame(999.0, index=price_df.index, columns=['Price'])
        battery_record_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        ev_battery_record_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        battery_soc_record_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        ev_battery_soc_record_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        buy_inelastic_record_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        buy_elastic_record_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        buy_shifted_record_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        buy_battery_record_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        buy_ev_battery_record_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        sell_pv_record_df = pd.DataFrame(0.0, index=supply_df.index, columns=supply_df.columns)
        sell_battery_record_df = pd.DataFrame(0.0, index=supply_df.index, columns=supply_df.columns)
        sell_ev_battery_record_df = pd.DataFrame(0.0, index=supply_df.index, columns=supply_df.columns)
        reward_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        
        # Generate elastic and inelastic demand according to the elastic ratio of each agent
        demand_elastic_df = demand_df.copy()
        demand_inelastic_df = demand_df.copy()
        for i in range(num_agent):
            demand_elastic_df[f'{i}'] = demand_df[f'{i}'] * agents[i]['elastic_ratio']
            demand_inelastic_df[f'{i}'] = demand_df[f'{i}'] * (1 - agents[i]['elastic_ratio'])
        # Prepare dataframe to record shifted demand
        shift_df = pd.DataFrame(0.0, index=demand_df.index, columns=demand_df.columns)
        

        for t in tqdm(range(len(demand_df))):
            demand_list = []
            supply_list = []
            wholesale_price = price_df.at[t, 'Price'] + wheeling_charge
            q.reset_all_digitized_states()
            q.reset_all_actions()
            reward = np.full(num_agent, 0.0)
            for i in range(num_agent):
                q.set_digitized_states(agent_id=i, 
                                    pv_ratio=pv_ratio_df.at[t, 'mean'], 
                                    battery_soc=battery_soc_record_df.at[t, f'{i}'],
                                    ev_battery_soc=ev_battery_soc_record_df.at[t, f'{i}'])
                q.set_actions(agent_id=i, episode=episode)
                # ユーザIDはデマンドレスポンスによる移動を考慮して1エージェントごとに
                # リアルタイム(inelas, elas)，バッテリー充放電，ev充放電，PV発電供給，シフトリミット時間ステップ分の数IDを保有する
                # シフトリミットが24時間なら，31個IDを保有する
                # agentのIDは0～, 100～, 200～, 300～, ...として，101にagent1のinelas，102にagent1のelas...のように割り当てる
                id_base = i * 100
                # デマンドレスポンス不可の需要
                d_inelas = demand_inelastic_df.at[t, f'{i}']
                demand_list.append([d_inelas, price_max, id_base+0, True])

                # デマンドレスポンス可能の需要
                d_elas_max = demand_elastic_df.at[t, f'{i}']
                price_elas = q.get_actions[i, 0]
                d_elas = d_elas_max * max((agents[i]['dr_price_threshold'] - price_elas)/(agents[i]['dr_price_threshold'] - price_min), 0)
                if price_elas == price_min:
                    # To avoid missing intersection point of supply and demand curve
                    price_elas += 0.00001
                # デマンドレスポンス可の需要はid_base+1に割り当てる
                demand_list.append([d_elas, price_elas, id_base+1, True])

                # バッテリー充放電価格の取得
                price_buy_battery = q.get_actions[i, 1]
                price_sell_battery = q.get_actions[i, 2]
                # バッテリー充放電可能量の取得
                battery_amount = battery_record_df.at[t, f'{i}']
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
                ev_battery_amount = ev_battery_record_df.at[t, f'{i}']
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
                shift_df.at[t, f'{i}'] = d_elas_max

                # 過去からシフトした需要の入札
                for k in range(t-int(agents[i]['shift_limit']), t):
                    if k >= 0:
                        d_shift = shift_df.at[k, f'{i}']
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
            # print(bids_df)
            if episode == 0 or episode == num_episode-1:
                if BID_SAVE:
                    timestamp = pd.read_csv('data/demand.csv').iat[t, 0]
                    market.plot(title=timestamp, episode=episode, number=t)
            transactions_df, _ = market.run(mechanism='uniform')
            
            
            # マーケット取引の結果を記録、報酬を計算
            for bid_num in transactions_df['bid']:
                id = bids_df.at[bid_num, 'user']
                if id != 99999:
                    # 100の位以降の数字を取り出す->agentID
                    user = id // 100
                    # リアルタイム(inelas, elas)@2，バッテリー充放電@2，ev充放電@2，シフトリミット@shift_limit，供給@1
                    item = id % 100
                    price = transactions_df[transactions_df['bid']==bid_num]['price'].values[0]
                    if item == 0:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        buy_inelastic_record_df.at[t, f'{user}'] = value
                        reward[user] -= value * price

                    elif item == 1:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        buy_elastic_record_df.at[t, f'{user}'] = value
                        # 時刻tでのDRの分だけ後ろの時間にシフトさせる需要量を減らす
                        shift_df.at[t, f'{user}'] -= value
                        reward[user] -= value * price
                        reward[user] -= (agents[int(user)]['alpha']/2 * (demand_elastic_df.at[t, f'{i}'] - value)**2 + 
                                        agents[int(user)]['beta']*(demand_elastic_df.at[t, f'{i}'] - value))

                    elif item == 2:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        buy_battery_record_df.at[t, f'{user}'] = value
                        if t+1 !=len(demand_df):
                            battery_record_df.at[t+1, f'{user}'] += value * battery_charge_efficiency
                            battery_soc_record_df.at[t+1, f'{user}'] = battery_record_df.at[t+1, f'{user}'] / agents[user]['battery_capacity']
                        reward[user] -= value * price
                        reward[user] -= ((agents[int(user)]['max_battery_charge_speed'] - value) * 
                                        (agents[int(user)]['gamma']/2 * (100 * (1-battery_soc_record_df.at[t, f'{user}']))**2 + 
                                        agents[int(user)]['epsilon']*(100 * (1-battery_soc_record_df.at[t, f'{user}']))))

                    elif item == 3:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        sell_battery_record_df.at[t, f'{user}'] = value
                        if t+1 !=len(demand_df):
                            battery_record_df.at[t+1, f'{user}'] -= value / battery_discharge_efficiency
                            battery_soc_record_df.at[t+1, f'{user}'] = battery_record_df.at[t+1, f'{user}'] / agents[user]['battery_capacity']
                        reward[user] += value * price
                        reward[user] -= ((agents[int(user)]['max_battery_charge_speed'] + value) * 
                                        (agents[int(user)]['gamma']/2 * (100 * (1-battery_soc_record_df.at[t, f'{user}']))**2 + 
                                        agents[int(user)]['epsilon']*(100 * (1-battery_soc_record_df.at[t, f'{user}']))))

                    elif item == 4:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        buy_ev_battery_record_df.at[t, f'{user}'] = value
                        if t+1 !=len(demand_df):
                            ev_battery_record_df.at[t+1, f'{user}'] += value * ev_charge_efficiency
                            ev_battery_soc_record_df.at[t+1, f'{user}'] = ev_battery_record_df.at[t+1, f'{user}'] / agents[user]['ev_capacity']
                        reward[user] -= value * price
                        reward[user] -= ((agents[int(user)]['max_ev_charge_speed'] - value) *
                                        (agents[int(user)]['psi']/2 * (100 * (1-ev_battery_soc_record_df.at[t, f'{user}']))**2 + 
                                        agents[int(user)]['omega']*(100 * (1-ev_battery_soc_record_df.at[t, f'{user}']))))

                    elif item == 5:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        sell_ev_battery_record_df.at[t, f'{user}'] = value
                        if t+1 !=len(demand_df):
                            ev_battery_record_df.at[t+1, f'{user}'] -= value / ev_discharge_efficiency
                            ev_battery_soc_record_df.at[t+1, f'{user}'] = ev_battery_record_df.at[t+1, f'{user}'] / agents[user]['ev_capacity']
                        reward[user] += value * price
                        reward[user] -= ((agents[int(user)]['max_ev_charge_speed'] + value) *
                                        (agents[int(user)]['psi']/2 * (100 * (1-ev_battery_soc_record_df.at[t, f'{user}']))**2 + 
                                        agents[int(user)]['omega']*(100 * (1-ev_battery_soc_record_df.at[t, f'{user}']))))

                    elif item == 6:
                        value = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        sell_pv_record_df.at[t, f'{user}'] = value
                        reward[user] += value * price

                    else:
                        # buy_shifted_record_dfに足し上げながらshift_dfを更新
                        for k in range(7, 7+int(agents[user]['shift_limit'])):
                            if item == k:
                                buy_shifted_record_df.at[t, f'{user}'] += transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                                if t-k+7-1>= 0:
                                    shift_df.at[t-k+7-1, f'{user}'] -= transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
            microgrid_price_record_df.at[t, 'Price'] = transactions_df['price'].values[0]

            # Q学習
            dr_states, battery_states, ev_battery_states = q.get_states
            actions_arr = q.get_actions
            for i in range(num_agent):
                reward_df.at[t, f'{i}'] = reward[i]
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
        # ====================================================================================================
        microgrid_price_record_df.to_csv('output/episode' + str(episode) + '/price_record.csv', index=True)
        battery_record_df.to_csv('output/episode' + str(episode) + '/battery_record.csv', index=True)
        ev_battery_record_df.to_csv('output/episode' + str(episode) + '/ev_battery_record.csv', index=True)
        battery_soc_record_df.to_csv('output/episode' + str(episode) + '/battery_soc_record.csv', index=True)
        ev_battery_soc_record_df.to_csv('output/episode' + str(episode) + '/ev_battery_soc_record.csv', index=True)

        buy_inelastic_record_df.to_csv('output/episode' + str(episode) + '/buy_inelastic_record.csv', index=True)
        buy_elastic_record_df.to_csv('output/episode' + str(episode) + '/buy_elastic_record.csv', index=True)
        buy_shifted_record_df.to_csv('output/episode' + str(episode) + '/buy_shifted_record.csv', index=True)
        sell_pv_record_df.to_csv('output/episode' + str(episode) + '/sell_record.csv', index=True)
        
        buy_battery_record_df.to_csv('output/episode' + str(episode) + '/buy_battery_record.csv', index=True)
        buy_ev_battery_record_df.to_csv('output/episode' + str(episode) + '/buy_ev_battery_record.csv', index=True)
        sell_battery_record_df.to_csv('output/episode' + str(episode) + '/sell_battery_record.csv', index=True)
        sell_ev_battery_record_df.to_csv('output/episode' + str(episode) + '/sell_ev_battery_record.csv', index=True)

        shift_df.to_csv('output/episode' + str(episode) + '/shift_record.csv', index=True)

        reward_df.to_csv('output/episode' + str(episode) + '/reward.csv', index=True)
        q.save_q_table(folder_path = 'output/episode'+str(episode))


if __name__ == "__main__":
    # Adding the new mechanism to the list of available mechanism of the market
    pm.market.MECHANISM['uniform'] = market.UniformPrice # type: ignore
    pd.set_option('display.max_rows', None)  # 全行表示

    main(num_agent=10, num_episode=100, BID_SAVE=True)
    vis = visualize.Visualize(folder_path='output/episode0')
    vis.plot_consumption()
