import numpy as np 
import pandas as pd
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
import logging
logger = logging.getLogger('Logging')
logger.setLevel(10)
fh = logging.FileHandler('main.log')
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s: line %(lineno)d: %(levelname)s: %(message)s')
fh.setFormatter(formatter)


def main(num_agent, BID_SAVE = False, **kwargs):
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

    # Preprocess and generate demand, supply, and price data
    preprocess = Preprocess()
    preprocess.set(
        pd.read_csv('data/demand.csv'),
        pd.read_csv('data/supply.csv'),
        pd.read_csv('data/price.csv')
    )
    preprocess.generate_d_s(num_agent)
    preprocess.save('output')
    preprocess.drop_index_  # drop timestamp index
    demand_df, supply_df, price_df = preprocess.get_dfs_

    # Generate agent parameters
    agents = Agent(num_agent)
    agents.generate_params()
    agents.save('output')

    # get average pv production ratio to get state in Q table
    # data is stored as kWh/kW, which means, the values are within 0~1
    pv_ratio_df = pd.read_csv('data/supply.csv', index_col=0)
    pv_ratio_df['mean'] = pv_ratio_df.mean(axis=1)
    pv_ratio_df.reset_index(inplace=True, drop=True)
    print(pv_ratio_df.at[10,'mean'])

    # Initialize record dataframes
    microgrid_price_record_df = pd.DataFrame(999.0, index=price_df.index, columns=['Price'])
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
        for i in range(num_agent):
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
            # 報酬の期待値を最大にするDRの量と価格を求める
            denominator = agents[i]['alpha'] * d_elas_max * price_min - (agents[i]['dr_price_threshold'] - price_min) * agents[i]['beta']
            numerator = 2 * (agents[i]['dr_price_threshold'] - price_min) - (agents[i]['alpha'] * d_elas_max)
            price_elas = denominator / numerator
            if price_elas > price_max:
                price_elas = price_max
                d_elas = d_elas_max * (agents[i]['dr_price_threshold']-price_elas)/(agents[i]['dr_price_threshold']-price_min)
            elif price_elas < price_min:
                price_elas = price_min + 0.000001
                # price_elas = price_min
                d_elas = d_elas_max * (agents[i]['dr_price_threshold']-price_elas)/(agents[i]['dr_price_threshold']-price_min)
            else:
                d_elas = d_elas_max * (agents[i]['dr_price_threshold']-price_elas)/(agents[i]['dr_price_threshold']-price_min)
            
            if d_elas < 0:
                d_elas = 0
            # デマンドレスポンス可の需要はid_base+1に割り当てる
            demand_list.append([d_elas, price_elas, id_base+1, True])

            # バッテリー充放電 id_base+2, id_base+3

            # EV充放電 id_base+4, id_base+5

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
        if BID_SAVE:
            timestamp = pd.read_csv('data/demand.csv').iat[t, 0]
            market.plot(title=timestamp, number=t)
        transactions_df, _ = market.run(mechanism='uniform')
        
        

        for bid_num in transactions_df['bid']:
            id = bids_df.at[bid_num, 'user']
            if id != 99999:
                # 100の位以降の数字を取り出す->agentID
                user = id // 100
                # リアルタイム(inelas, elas)@2，バッテリー充放電@2，ev充放電@2，シフトリミット@shift_limit，供給@1
                item = id % 100
                if item == 0:
                    buy_inelastic_record_df.at[t, f'{user}'] = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                elif item == 1:
                    buy_elastic_record_df.at[t, f'{user}'] = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                    # 時刻tでのDRの分だけ後ろの時間にシフトさせる需要量を減らす
                    shift_df.at[t, f'{user}'] -= transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                # buy_shifted_record_dfに足し上げながらshift_dfを更新
                for k in range(7, 7+int(agents[user]['shift_limit'])):
                    if item == k:
                        buy_shifted_record_df.at[t, f'{user}'] += transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                        if t-k+7-1>= 0:
                            shift_df.at[t-k+7-1, f'{user}'] -= transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
                





                if item == 6:
                    sell_pv_record_df.at[t, f'{user}'] = transactions_df[transactions_df['bid']==bid_num]['quantity'].values[0]
        microgrid_price_record_df.at[t, 'Price'] = transactions_df['price'].values[0]
        # print(bids_df)
        # print(transactions_df)
        # print(shift_df.head(10))
        # print(buy_shifted_record_df.head(5))
        # input()
    microgrid_price_record_df.to_csv('output/price_record.csv', index=True)
    battery_soc_record_df.to_csv('output/battery_soc_record.csv', index=True)
    ev_battery_soc_record_df.to_csv('output/ev_battery_soc_record.csv', index=True)

    buy_inelastic_record_df.to_csv('output/buy_inelastic_record.csv', index=True)
    buy_elastic_record_df.to_csv('output/buy_elastic_record.csv', index=True)
    buy_shifted_record_df.to_csv('output/buy_shifted_record.csv', index=True)
    sell_pv_record_df.to_csv('output/sell_record.csv', index=True)
    
    buy_battery_record_df.to_csv('output/buy_battery_record.csv', index=True)
    buy_ev_battery_record_df.to_csv('output/buy_ev_battery_record.csv', index=True)
    sell_battery_record_df.to_csv('output/sell_battery_record.csv', index=True)
    sell_ev_battery_record_df.to_csv('output/sell_ev_battery_record.csv', index=True)


if __name__ == "__main__":
    # Adding the new mechanism to the list of available mechanism of the market
    pm.market.MECHANISM['uniform'] = market.UniformPrice # type: ignore
    pd.set_option('display.max_rows', None)  # 全行表示

    main(num_agent=10, BID_SAVE=False, price_min=5)
    vis = visualize.Visualize()
    vis.plot_consumption()
