import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import capex_opex


def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return list(map(int, numbers))


def reward_history_plot_4_4_powerplot(reward_sorted_file_paths_list, agent_num, folder_path):
    """
    expected to receive 16 lists of reward file paths(e.g. 16 threads)
    """
    print('Plotting reward history powerplot...')
    start_time = datetime.datetime.now()
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(4):
        for j in range(4):
            reward_list_list = []
            for agent in range(agent_num-1):  # start from agent 0
                reward_list = []
                for path in reward_sorted_file_paths_list[i*4+j]:
                    df = pd.read_csv(path, index_col=0)
                    reward_df = df.loc[:, f'{agent}']
                    reward = reward_df.sum(axis=0)
                    reward_list.append(reward)
                reward_list_list.append(reward_list)
                axs[i, j].plot(reward_list, linewidth=0.5)
            axs[i, j].set_title(f'Thread {i*4+j}')
            axs[i, j].set_xlabel('Episode')
            axs[i, j].set_ylabel('Reward')
            # axs[i, j].set_yscale('log')  # log scale
            print(f'Thread {i*4+j} done.')
    plt.tight_layout()
    # plt.show()
    fig.savefig(folder_path + '/insight/reward_history_powerplot.png', dpi=600)
    fig.savefig(folder_path + '/insight/reward_history_powerplot.svg')
    print('Reward history powerplot saved.')
    print(f'Execution time: {datetime.datetime.now()-start_time}')


def reward_history_plot_4_4(reward_sorted_file_paths_list, agent_num, folder_path):
    """
    expected to receive 16 lists of reward file paths(e.g. 16 threads)
    """
    print('Plotting reward history...')
    start_time = datetime.datetime.now()
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(4):
        for j in range(4):
            reward_list_list = []
            for agent in range(agent_num):  # start from agent 0
                reward_list = []
                for path in reward_sorted_file_paths_list[i*4+j]:
                    df = pd.read_csv(path, index_col=0)
                    reward_df = df.loc[:, f'{agent}']
                    reward = reward_df.sum(axis=0)
                    reward_list.append(reward)
                reward_list_list.append(reward_list)
            
            # Calculate mean and standard deviation of rewards for each episode
            reward_array = np.array(reward_list_list)
            mean_rewards = reward_array.mean(axis=0)
            std_rewards = reward_array.std(axis=0)
            
            # plot with mean and error bars
            axs[i, j].errorbar(range(len(mean_rewards)), mean_rewards, yerr=std_rewards, linewidth=0.5, fmt='-o', ecolor='r', capsize=2)
            axs[i, j].set_title(f'Thread {i*4+j}')
            axs[i, j].set_xlabel('Episode')
            axs[i, j].set_ylabel('Reward')
            # axs[i, j].set_yscale('log')  # log scale
            print(f'Thread {i*4+j} done.')
    plt.tight_layout()
    # plt.show()
    fig.savefig(folder_path + '/insight/reward_history.png', dpi=600)
    fig.savefig(folder_path + '/insight/reward_history.svg')
    print('Reward history plot saved.')
    print(f'Execution time: {datetime.datetime.now()-start_time}')


def buy_sell_amount_cost_by_battery_pv_dr_exist_plot(thread_num, folder_path, 
                                                     include_capex_opex=True):
    """
    CAPEX of PV and BES are calculated by Straight Line Method. (定額法)
    PVの法定耐用年数は17年、BESの法定耐用年数は6年.
    The statutory useful life of the depreciable assets for PV is 17 years.
    The statutory useful life of the depreciable assets for BES is 6 years.
    """
    pv_lifetime = 17
    bes_lifetime = 6
    buy_inelastic_file_path_list = []
    for i in range(thread_num):
        buy_inelastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_inelastic_record.csv')
        buy_inelastic_sorted_file_paths = sorted(buy_inelastic_file_paths, key=numerical_sort)
        buy_inelastic_file_path_list.append(buy_inelastic_sorted_file_paths[-1])  # get the last episode
    buy_elastic_file_path_list = []
    for i in range(thread_num):
        buy_elastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_elastic_record.csv')
        buy_elastic_sorted_file_paths = sorted(buy_elastic_file_paths, key=numerical_sort)
        buy_elastic_file_path_list.append(buy_elastic_sorted_file_paths[-1])  # get the last episode
    buy_shifted_file_path_list = []
    for i in range(thread_num):
        buy_shifted_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_shifted_record.csv')
        buy_shifted_sorted_file_paths = sorted(buy_shifted_file_paths, key=numerical_sort)
        buy_shifted_file_path_list.append(buy_shifted_sorted_file_paths[-1])  # get the last episode
    buy_battery_file_path_list = []
    for i in range(thread_num):
        buy_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_battery_record.csv')
        buy_battery_sorted_file_paths = sorted(buy_battery_file_paths, key=numerical_sort)
        buy_battery_file_path_list.append(buy_battery_sorted_file_paths[-1])  # get the last episode
    buy_ev_battery_file_path_list = []
    for i in range(thread_num):
        buy_ev_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_ev_battery_record.csv')
        buy_ev_battery_sorted_file_paths = sorted(buy_ev_battery_file_paths, key=numerical_sort)
        buy_ev_battery_file_path_list.append(buy_ev_battery_sorted_file_paths[-1])  # get the last episode

    sell_pv_file_path_list = []
    for i in range(thread_num):
        sell_pv_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_pv_record.csv')
        sell_pv_sorted_file_paths = sorted(sell_pv_file_paths, key=numerical_sort)
        sell_pv_file_path_list.append(sell_pv_sorted_file_paths[-1])  # get the last episode
    sell_battery_file_path_list = []
    for i in range(thread_num):
        sell_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_battery_record.csv')
        sell_battery_sorted_file_paths = sorted(sell_battery_file_paths, key=numerical_sort)
        sell_battery_file_path_list.append(sell_battery_sorted_file_paths[-1])  # get the last episode
    sell_ev_battery_file_path_list = []
    for i in range(thread_num):
        sell_ev_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_ev_battery_record.csv')
        sell_ev_battery_sorted_file_paths = sorted(sell_ev_battery_file_paths, key=numerical_sort)
        sell_ev_battery_file_path_list.append(sell_ev_battery_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/agent_params.csv')
        agent_params_sorted_file_paths = sorted(agent_params_file_paths, key=numerical_sort)
        agent_params_file_path_list.append(agent_params_sorted_file_paths[-1])  # get the last episode

    microgrid_price_file_path_list = []
    for i in range(thread_num):
        microgrid_price_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/price_record.csv')
        microgrid_price_sorted_file_paths = sorted(microgrid_price_file_paths, key=numerical_sort)
        microgrid_price_file_path_list.append(microgrid_price_sorted_file_paths[-1])  # get the last episode
 
    item_dict = {
        'w/battery_w/pv_w/dr': {'amount': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                'cost': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                'cost/kWh': []},
        'w/battery_w/pv_wo/dr': {'amount': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost/kWh': []},
        'w/battery_wo/pv_w/dr': {'amount': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost/kWh': []},
        'w/battery_wo/pv_wo/dr': {'amount': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost/kWh': []},
        'wo/battery_w/pv_w/dr': {'amount': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost/kWh': []},
        'wo/battery_w/pv_wo/dr': {'amount': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost/kWh': []},
        'wo/battery_wo/pv_w/dr': {'amount': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost/kWh': []},
        'wo/battery_wo/pv_wo/dr': {'amount': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': [], 'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}, 
                                 'cost/kWh': []},
    }

    for i in range(len(agent_params_file_path_list)):
        agent_params_file_path = agent_params_file_path_list[i]
        agent_params_df = pd.read_csv(agent_params_file_path, index_col=0)
        microgrid_price = pd.read_csv(microgrid_price_file_path_list[i], index_col=0)
        buy_inelastic = pd.read_csv(buy_inelastic_file_path_list[i], index_col=0)
        buy_elastic = pd.read_csv(buy_elastic_file_path_list[i], index_col=0)
        buy_shifted = pd.read_csv(buy_shifted_file_path_list[i], index_col=0)
        buy_battery = pd.read_csv(buy_battery_file_path_list[i], index_col=0)
        buy_ev_battery = pd.read_csv(buy_ev_battery_file_path_list[i], index_col=0)
        sell_pv = pd.read_csv(sell_pv_file_path_list[i], index_col=0)
        sell_battery = pd.read_csv(sell_battery_file_path_list[i], index_col=0)
        sell_ev_battery = pd.read_csv(sell_ev_battery_file_path_list[i], index_col=0)
        for j in range(agent_params_df.shape[0]):
            battery_capacity = agent_params_df.loc[j, 'battery_capacity']
            ev_capacity = agent_params_df.loc[j, 'ev_capacity']
            pv_capacity = agent_params_df.loc[j, 'pv_capacity']
            dr_boolean = agent_params_df.loc[j, 'dr_boolean']
            if battery_capacity > 0 and pv_capacity > 0 and dr_boolean:
                target = 'w/battery_w/pv_w/dr'
            elif battery_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                target = 'w/battery_w/pv_wo/dr'
            elif battery_capacity > 0 and pv_capacity == 0 and dr_boolean:
                target = 'w/battery_wo/pv_w/dr'
            elif battery_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                target = 'w/battery_wo/pv_wo/dr'
            elif battery_capacity == 0 and pv_capacity > 0 and dr_boolean:
                target = 'wo/battery_w/pv_w/dr'
            elif battery_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                target = 'wo/battery_w/pv_wo/dr'
            elif battery_capacity == 0 and pv_capacity == 0 and dr_boolean:
                target = 'wo/battery_wo/pv_w/dr'
            elif battery_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                target = 'wo/battery_wo/pv_wo/dr'
            else:
                raise ValueError('Invalid combination of battery_capacity, pv_capacity, and dr_boolean.')
            
            item_dict[target]['amount']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
            item_dict[target]['amount']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
            item_dict[target]['amount']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
            item_dict[target]['amount']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
            item_dict[target]['amount']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            item_dict[target]['amount']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
            item_dict[target]['amount']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
            item_dict[target]['amount']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            item_dict[target]['cost']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            item_dict[target]['cost']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            item_dict[target]['cost']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            item_dict[target]['cost']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            item_dict[target]['cost']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            item_dict[target]['cost']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            item_dict[target]['cost']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            item_dict[target]['cost']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            if include_capex_opex:
                item_dict[target]['cost']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                item_dict[target]['cost']['battery_capex'].append(capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                item_dict[target]['cost']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            else:
                item_dict[target]['cost']['pv_capex'].append(0)
                item_dict[target]['cost']['battery_capex'].append(0)
                item_dict[target]['cost']['pv_opex'].append(0)
            cost = ((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 + 
                    (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 + 
                    (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 + 
                    (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 + 
                    (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 - 
                    (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 - 
                    (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 - 
                    (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            if include_capex_opex:
                cost += (capex_opex.pv_capex_func(pv_capacity) / pv_lifetime + 
                         capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime + 
                         capex_opex.pv_opex_func(pv_capacity))
            else:
                cost += 0
            amount = (buy_inelastic.loc[:, f'{j}'].sum() + 
                        buy_elastic.loc[:, f'{j}'].sum() + 
                        buy_shifted.loc[:, f'{j}'].sum()  
                    #   + buy_battery.loc[:, f'{j}'].sum() 
                        + buy_ev_battery.loc[:, f'{j}'].sum()
                    #   - sell_battery.loc[:, f'{j}'].sum()
                        - sell_ev_battery.loc[:, f'{j}'].sum()
                    #  - sell_pv.loc[:, f'{j}'].sum()
                        )
            item_dict[target]['cost/kWh'].append(cost / amount if amount > 0 else 0)
                
    # draw graph
    buy_amount_composition = ['buy_inelastic', 'buy_elastic', 'buy_shifted', 'buy_battery', 'buy_ev_battery']
    buy_cost_composition = ['pv_capex', 'battery_capex', 'pv_opex', 'buy_inelastic', 'buy_elastic', 'buy_shifted', 'buy_battery', 'buy_ev_battery']
    sell_composition = ['sell_pv', 'sell_battery', 'sell_ev_battery']
    categories = ['w/battery_w/pv_w/dr', 'w/battery_w/pv_wo/dr', 'w/battery_wo/pv_w/dr', 'w/battery_wo/pv_wo/dr',
                  'wo/battery_w/pv_w/dr', 'wo/battery_w/pv_wo/dr', 'wo/battery_wo/pv_w/dr', 'wo/battery_wo/pv_wo/dr']
    # labels = ['pv_capex', 'battery_capex', 'pv_opex', 'buy_inelastic', 'buy_elastic', 'buy_shifted', 
    #           'buy_battery', 'buy_ev_battery', 'sell_pv', 'sell_battery', 'sell_ev_battery']
    # blue, deepskyblue, skyblue, red, purple
    buy_colors = ['#0000ff', '#00bfff', '#87ceeb', '#d62728', '#9467bd']
    sell_colors = ['#ffd700', '#d62728', '#9467bd']  # gold, red, purple

    # make stacked bar graph of amount
    fig, ax = plt.subplots(figsize=(10, 8))
    # bar_width = 0.35
    r = np.arange(len(categories))
    buy_bottom = np.zeros(len(categories))
    sell_bottom = np.zeros(len(categories))
    for i, label in enumerate(buy_amount_composition):
        values = [np.mean(item_dict[category]['amount'][label]) if len(item_dict[category]['amount'][label]) > 0 else 0 for category in categories]
        ax.bar(r, values, bottom=buy_bottom, label=label, color=buy_colors[i])  # width=bar_width,
        buy_bottom += np.array(values)
    for i, label in enumerate(sell_composition):
        values = [-np.mean(item_dict[category]['amount'][label]) if len(item_dict[category]['amount'][label]) > 0 else 0 for category in categories]
        ax.bar(r, values, bottom=sell_bottom, label=label, color=sell_colors[i])  # width=bar_width,
        sell_bottom += np.array(values)

    # Show the number of agents on each bar
    for i, count in enumerate([len(item_dict[category]['amount']['buy_inelastic']) for category in categories]):
        ax.text(r[i], buy_bottom[i], f'n={count}', ha='center', va='bottom')

    # Add labels, title, and grid
    ax.set_xticks(r)
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                    rotation=45)
    ax.set_ylabel('Electricity Amount [kWh]')
    if include_capex_opex:
        ax.set_title('Average Electricity Amount Buy/Sell Composition')
    else:
        ax.set_title('Average Electricity Amount Buy/Sell Composition (excluding CAPEX and OPEX)')
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(-15000, 20000)

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/01_buy_sell_amount_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig(folder_path + '/insight/01_buy_sell_amount_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Electricity amount buy&sell composition plot saved.')
    plt.close('all')

    # darkgrey, mediumgrey, lightgrey, blue, deepskyblue, skyblue, red, purple
    buy_colors = ['#696969', '#A9A9A9', '#D3D3D3', '#0000ff', '#00bfff', '#87ceeb', '#d62728', '#9467bd']
    sell_colors = ['#ffd700', '#d62728', '#9467bd']  # gold, red, purple
    # make stacked bar graph of cost
    fig, ax = plt.subplots(figsize=(10, 8))
    # bar_width = 0.35
    r = np.arange(len(categories))
    buy_bottom = np.zeros(len(categories))
    sell_bottom = np.zeros(len(categories))
    for i, label in enumerate(buy_cost_composition):
        values = [np.mean(item_dict[category]['cost'][label]) if len(item_dict[category]['cost'][label]) > 0 else 0 for category in categories]
        ax.bar(r, values, bottom=buy_bottom, label=label, color=buy_colors[i])  # width=bar_width,
        buy_bottom += np.array(values)
    for i, label in enumerate(sell_composition):
        values = [-np.mean(item_dict[category]['cost'][label]) if len(item_dict[category]['cost'][label]) > 0 else 0 for category in categories]
        ax.bar(r, values, bottom=sell_bottom, label=label, color=sell_colors[i])  # width=bar_width,
        sell_bottom += np.array(values)

    # Show the number of agents on each bar
    for i, count in enumerate([len(item_dict[category]['cost']['buy_inelastic']) for category in categories]):
        ax.text(r[i], buy_bottom[i], f'n={count}', ha='center', va='bottom')

    # Add labels, title, and grid
    ax.set_xticks(r)
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                    rotation=45)
    ax.set_ylabel('Electricity Cost [$]')
    if include_capex_opex:
        ax.set_title('Average Electricity Cost Buy/Sell Composition')
    else:
        ax.set_title('Average Electricity Cost Buy/Sell Composition (excluding CAPEX and OPEX)')
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if include_capex_opex:
        ax.set_ylim(-4000, 12000)
        plt.savefig(folder_path + '/insight/01_buy_sell_cost_by_battery_pv_dr_include_capex_opex.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_buy_sell_cost_by_battery_pv_dr_include_capex_opex.svg')
    else:
        ax.set_ylim(-3000, 6000)
        plt.savefig(folder_path + '/insight/01_buy_sell_cost_by_battery_ev_pv_dr.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_buy_sell_cost_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Electricity cost buy&sell composition plot saved.')
    plt.close('all')

    # make box plot of cost/kWh
    fig, ax = plt.subplots(figsize=(10, 8))
    # Create a boxplot for each category
    boxprops = dict(color='black', linewidth=1.5)
    medianprops = dict(color='red', linewidth=2)
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='blue', markersize=8)
    data = [item_dict[category]['cost/kWh'] for category in categories]
    bplot = ax.boxplot(data, boxprops=boxprops, medianprops=medianprops, meanprops=meanpointprops,
                       showmeans=True, patch_artist=True)
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                        rotation=45)
    # Set all boxplot colors to gray
    gray_color = '#808080'
    for patch in bplot['boxes']:
        patch.set_facecolor(gray_color)
        
    ax.set_ylabel('Electricity Cost per kWh [$]')
    if include_capex_opex:
        ax.set_title('Electricity Cost per kWh Distribution')
    else:
        ax.set_title('Electricity Cost per kWh Distribution (excluding CAPEX and OPEX)')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    if include_capex_opex:
        ax.set_ylim(0, 1.5)
        plt.savefig(folder_path + '/insight/01_cost_per_kWh_by_battery_pv_dr_include_capex_opex.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_cost_per_kWh_by_battery_pv_dr_include_capex_opex.svg')
    else:
        ax.set_ylim(-0.5, 0.5)
        plt.savefig(folder_path + '/insight/01_cost_per_kWh_by_battery_ev_pv_dr.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_cost_per_kWh_by_battery_ev_pv_dr.svg')
    # plt.show()

    # Add mean, standard deviation, and median text
    # Calculate mean, standard deviation, and median costs for each category
    mean_costs = [np.mean(item_dict[category]['cost/kWh']) for category in categories]
    std_costs = [np.std(item_dict[category]['cost/kWh']) for category in categories]
    median_costs = [np.median(item_dict[category]['cost/kWh']) for category in categories]
    for i, category in enumerate(categories):
        ax.text(i+1, median_costs[i], 
                f'Mean: {median_costs[i]:.2f}\nStd: {std_costs[i]:.2f}\nMed: {median_costs[i]:.2f}', 
                ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))
    if include_capex_opex:
        ax.set_ylim(0, 1.5)
        plt.savefig(folder_path + '/insight/01_cost_per_kWh_by_battery_pv_dr_include_capex_opex_with_values.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_cost_per_kWh_by_battery_pv_dr_include_capex_opex_with_values.svg')
    else:
        ax.set_ylim(-0.5, 0.5)
        plt.savefig(folder_path + '/insight/01_cost_per_kWh_by_battery_ev_pv_dr_with_values.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_cost_per_kWh_by_battery_ev_pv_dr_with_values.svg')
    # plt.show()
    print('Electricity cost per kWh distribution plot saved.')
    plt.close()

    # make box plot of net cost
    fig, ax = plt.subplots(figsize=(10, 8))
    # Create a boxplot for each category
    boxprops = dict(color='black', linewidth=1.5)
    medianprops = dict(color='red', linewidth=2)
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='blue', markersize=8)
    data = [np.array(item_dict[category]['cost']['buy_inelastic']) 
            + np.array(item_dict[category]['cost']['buy_elastic']) 
            + np.array(item_dict[category]['cost']['buy_shifted']) 
            + np.array(item_dict[category]['cost']['buy_battery']) 
            + np.array(item_dict[category]['cost']['buy_ev_battery']) 
            - np.array(item_dict[category]['cost']['sell_pv']) 
            - np.array(item_dict[category]['cost']['sell_battery']) 
            - np.array(item_dict[category]['cost']['sell_ev_battery']) for category in categories]
    if include_capex_opex:
        data = [np.array(data[i]) + np.array(item_dict[category]['cost']['pv_capex']) 
                + np.array(item_dict[category]['cost']['battery_capex']) 
                + np.array(item_dict[category]['cost']['pv_opex']) for i, category in enumerate(categories)]
    else:
        data = data
    bplot = ax.boxplot(data, boxprops=boxprops, medianprops=medianprops, meanprops=meanpointprops,
                       showmeans=True, patch_artist=True)
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                        rotation=45)
    # Set all boxplot colors to gray
    gray_color = '#808080'
    for patch in bplot['boxes']:
        patch.set_facecolor(gray_color)
        
    ax.set_ylabel('Electricity Cost [$]')
    if include_capex_opex:
        ax.set_title('Electricity Cost Distribution')
    else:
        ax.set_title('Electricity Cost Distribution (excluding CAPEX and OPEX)')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    if include_capex_opex:
        ax.set_ylim(0, 14000)
        plt.savefig(folder_path + '/insight/01_cost_by_battery_pv_dr_include_capex_opex.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_cost_by_battery_pv_dr_include_capex_opex.svg')
    else:
        ax.set_ylim(0, 10000)
        plt.savefig(folder_path + '/insight/01_cost_by_battery_ev_pv_dr.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_cost_by_battery_ev_pv_dr.svg')
    # plt.show()

    # Add mean, standard deviation, and median text
    # Calculate mean, standard deviation, and median costs for each category
    mean_costs = [np.mean(data[i]) for i in range(len(data))]
    std_costs = [np.std(data[i]) for i in range(len(data))]
    median_costs = [np.median(data[i]) for i in range(len(data))]
    for i, category in enumerate(categories):
        ax.text(i+1, median_costs[i], 
                f'Mean: {mean_costs[i]:.2f}\nStd: {std_costs[i]:.2f}\nMed: {median_costs[i]:.2f}', 
                ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))
    if include_capex_opex:
        ax.set_ylim(0, 14000)
        plt.savefig(folder_path + '/insight/01_cost_by_battery_pv_dr_include_capex_opex_with_values.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_cost_by_battery_pv_dr_include_capex_opex_with_values.svg')
    else:
        ax.set_ylim(0, 10000)
        plt.savefig(folder_path + '/insight/01_cost_by_battery_ev_pv_dr_with_values.png', dpi=600)
        plt.savefig(folder_path + '/insight/01_cost_by_battery_ev_pv_dr_with_values.svg')
    # plt.show()
    print('Electricity cost distribution plot saved.')
    plt.close()


def net_cost_by_battery_ev_pv_size_plot(thread_num, folder_path, include_capex_opex=True):
    """
    CAPEX of PV and BES are calculated by Straight Line Method. (定額法)
    PVの法定耐用年数は17年、BESの法定耐用年数は6年.
    The statutory useful life of the depreciable assets for PV is 17 years.
    The statutory useful life of the depreciable assets for BES is 6 years.
    """
    pv_lifetime = 17
    bes_lifetime = 6
    buy_inelastic_file_path_list = []
    for i in range(thread_num):
        buy_inelastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_inelastic_record.csv')
        buy_inelastic_sorted_file_paths = sorted(buy_inelastic_file_paths, key=numerical_sort)
        buy_inelastic_file_path_list.append(buy_inelastic_sorted_file_paths[-1])  # get the last episode
    buy_elastic_file_path_list = []
    for i in range(thread_num):
        buy_elastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_elastic_record.csv')
        buy_elastic_sorted_file_paths = sorted(buy_elastic_file_paths, key=numerical_sort)
        buy_elastic_file_path_list.append(buy_elastic_sorted_file_paths[-1])  # get the last episode
    buy_shifted_file_path_list = []
    for i in range(thread_num):
        buy_shifted_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_shifted_record.csv')
        buy_shifted_sorted_file_paths = sorted(buy_shifted_file_paths, key=numerical_sort)
        buy_shifted_file_path_list.append(buy_shifted_sorted_file_paths[-1])  # get the last episode
    buy_battery_file_path_list = []
    for i in range(thread_num):
        buy_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_battery_record.csv')
        buy_battery_sorted_file_paths = sorted(buy_battery_file_paths, key=numerical_sort)
        buy_battery_file_path_list.append(buy_battery_sorted_file_paths[-1])  # get the last episode
    buy_ev_battery_file_path_list = []
    for i in range(thread_num):
        buy_ev_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_ev_battery_record.csv')
        buy_ev_battery_sorted_file_paths = sorted(buy_ev_battery_file_paths, key=numerical_sort)
        buy_ev_battery_file_path_list.append(buy_ev_battery_sorted_file_paths[-1])  # get the last episode

    sell_pv_file_path_list = []
    for i in range(thread_num):
        sell_pv_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_pv_record.csv')
        sell_pv_sorted_file_paths = sorted(sell_pv_file_paths, key=numerical_sort)
        sell_pv_file_path_list.append(sell_pv_sorted_file_paths[-1])  # get the last episode
    sell_battery_file_path_list = []
    for i in range(thread_num):
        sell_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_battery_record.csv')
        sell_battery_sorted_file_paths = sorted(sell_battery_file_paths, key=numerical_sort)
        sell_battery_file_path_list.append(sell_battery_sorted_file_paths[-1])  # get the last episode
    sell_ev_battery_file_path_list = []
    for i in range(thread_num):
        sell_ev_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_ev_battery_record.csv')
        sell_ev_battery_sorted_file_paths = sorted(sell_ev_battery_file_paths, key=numerical_sort)
        sell_ev_battery_file_path_list.append(sell_ev_battery_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/agent_params.csv')
        agent_params_sorted_file_paths = sorted(agent_params_file_paths, key=numerical_sort)
        agent_params_file_path_list.append(agent_params_sorted_file_paths[-1])  # get the last episode

    microgrid_price_file_path_list = []
    for i in range(thread_num):
        microgrid_price_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/price_record.csv')
        microgrid_price_sorted_file_paths = sorted(microgrid_price_file_paths, key=numerical_sort)
        microgrid_price_file_path_list.append(microgrid_price_sorted_file_paths[-1])  # get the last episode


    buy_composition = ['buy_inelastic', 'buy_elastic', 'buy_shifted', 'buy_battery', 'buy_ev_battery']
    sell_composition = ['sell_pv', 'sell_battery', 'sell_ev_battery']

    master_list = []

    for i in range(len(agent_params_file_path_list)):
        agent_params_file_path = agent_params_file_path_list[i]
        agent_params_df = pd.read_csv(agent_params_file_path, index_col=0)
        microgrid_price = pd.read_csv(microgrid_price_file_path_list[i], index_col=0)
        buy_inelastic = pd.read_csv(buy_inelastic_file_path_list[i], index_col=0)
        buy_elastic = pd.read_csv(buy_elastic_file_path_list[i], index_col=0)
        buy_shifted = pd.read_csv(buy_shifted_file_path_list[i], index_col=0)
        buy_battery = pd.read_csv(buy_battery_file_path_list[i], index_col=0)
        buy_ev_battery = pd.read_csv(buy_ev_battery_file_path_list[i], index_col=0)
        sell_pv = pd.read_csv(sell_pv_file_path_list[i], index_col=0)
        sell_battery = pd.read_csv(sell_battery_file_path_list[i], index_col=0)
        sell_ev_battery = pd.read_csv(sell_ev_battery_file_path_list[i], index_col=0)
        for j in range(agent_params_df.shape[0]):
            battery_capacity = agent_params_df.loc[j, 'battery_capacity']
            ev_capacity = agent_params_df.loc[j, 'ev_capacity']
            pv_capacity = agent_params_df.loc[j, 'pv_capacity']
            dr_boolean = agent_params_df.loc[j, 'dr_boolean']

            if include_capex_opex:
                cost = ((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 
                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                + capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime
                + capex_opex.pv_opex_func(pv_capacity))
            else:
                cost = ((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 
                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            
            amount = (buy_inelastic.loc[:, f'{j}'].sum()
            + buy_elastic.loc[:, f'{j}'].sum()
            + buy_shifted.loc[:, f'{j}'].sum()
            # + buy_battery.loc[:, f'{j}'].sum()
            + buy_ev_battery.loc[:, f'{j}'].sum()
            # - sell_pv.loc[:, f'{j}'].sum()
            # - sell_battery.loc[:, f'{j}'].sum()
            - sell_ev_battery.loc[:, f'{j}'].sum())

            cost_per_kWh = cost / amount
            # print(f'cost: {cost}, amount: {amount}, cost_per_kWh: {cost_per_kWh}')

            master_list.append({'battery_capacity': battery_capacity, 'ev_capacity': ev_capacity, 'pv_capacity': pv_capacity, 'dr_boolean': dr_boolean,
                                'cost': cost, 'amount': amount, 'cost_per_kWh': cost_per_kWh})

    master_df = pd.DataFrame(master_list)

    battery_capacity_list = np.unique(master_df['battery_capacity'])[::-1]
    pv_capacity_list = np.unique(master_df['pv_capacity'])
    ev_capacity_list = np.unique(master_df['ev_capacity'])

    heat_map_net_total_cost_df = pd.DataFrame(index=battery_capacity_list, columns=pv_capacity_list)
    heat_map_net_cost_per_kWh_df = pd.DataFrame(index=battery_capacity_list, columns=pv_capacity_list)

    for battery_capacity in battery_capacity_list:
        for pv_capacity in pv_capacity_list:
            heat_map_net_total_cost_df.loc[battery_capacity, pv_capacity] = master_df[(master_df['battery_capacity'] == battery_capacity) & (master_df['pv_capacity'] == pv_capacity)]['cost'].mean()
            heat_map_net_cost_per_kWh_df.loc[battery_capacity, pv_capacity] = master_df[(master_df['battery_capacity'] == battery_capacity) & (master_df['pv_capacity'] == pv_capacity)]['cost_per_kWh'].mean()

    heat_map_net_total_cost_df = heat_map_net_total_cost_df.apply(pd.to_numeric, errors='coerce')
    heat_map_net_cost_per_kWh_df = heat_map_net_cost_per_kWh_df.apply(pd.to_numeric, errors='coerce')
    # print(heat_map_net_total_cost_df)
    # print(heat_map_net_cost_per_kWh_df)

    fig, ax = plt.subplots(figsize=(10, 8))
    if include_capex_opex:
        sns.heatmap(heat_map_net_total_cost_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, cbar_kws={'label': 'Net Total Cost [$]'}, vmin=2000, vmax=8000)
    else:
        sns.heatmap(heat_map_net_total_cost_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, cbar_kws={'label': 'Net Total Cost [$]'}, vmin=0, vmax=5000)
    ax.set_title('Net Total Cost Heatmap by Battery and PV Capacity')
    ax.set_xlabel('PV Capacity [kW]')
    ax.set_ylabel('Battery Capacity [kWh]')
    plt.tight_layout()
    if include_capex_opex:
        plt.savefig(folder_path + '/insight/net_total_cost_heatmap_by_battery_pv_include_capex_opex.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_total_cost_heatmap_by_battery_pv_include_capex_opex.svg')
    else:
        plt.savefig(folder_path + '/insight/net_total_cost_heatmap_by_battery_pv.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_total_cost_heatmap_by_battery_pv.svg')

    fig, ax = plt.subplots(figsize=(10, 8))
    if include_capex_opex:
        sns.heatmap(heat_map_net_cost_per_kWh_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, cbar_kws={'label': 'Net Cost per kWh [$]'}, vmin=0.2, vmax=0.8)
    else:
        sns.heatmap(heat_map_net_cost_per_kWh_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, cbar_kws={'label': 'Net Cost per kWh [$]'}, vmin=0, vmax=0.4)
    ax.set_title('Net Cost per kWh Heatmap by Battery and PV Capacity')
    ax.set_xlabel('PV Capacity [kW]')
    ax.set_ylabel('Battery Capacity [kWh]')
    plt.tight_layout()
    if include_capex_opex:
        plt.savefig(folder_path + '/insight/net_cost_per_kWh_heatmap_by_battery_pv_include_capex_opex.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_cost_per_kWh_heatmap_by_battery_pv_include_capex_opex.svg')
    else:
        plt.savefig(folder_path + '/insight/net_cost_per_kWh_heatmap_by_battery_pv.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_cost_per_kWh_heatmap_by_battery_pv.svg')

    print('Net cost by battery, PV capacity heatmap saved.')

    # plot how many agents are in each category, using heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    heat_map_agent_count_df = pd.DataFrame(index=battery_capacity_list, columns=pv_capacity_list)
    for battery_capacity in battery_capacity_list:
        for pv_capacity in pv_capacity_list:
            heat_map_agent_count_df.loc[battery_capacity, pv_capacity] = master_df[(master_df['battery_capacity'] == battery_capacity) & (master_df['pv_capacity'] == pv_capacity)].shape[0]
    heat_map_agent_count_df = heat_map_agent_count_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(heat_map_agent_count_df, annot=True, fmt=".0f", cmap="coolwarm", cbar=True, cbar_kws={'label': 'Prosumer Count'}, vmin=0, vmax=400)
    ax.set_title('Prosumer Count Heatmap by Battery and PV Capacity')
    ax.set_xlabel('PV Capacity [kW]')
    ax.set_ylabel('Battery Capacity [kWh]')
    plt.tight_layout()
    plt.savefig(folder_path + '/insight/prosumer_count_heatmap_by_battery_pv.png', dpi=600)
    plt.savefig(folder_path + '/insight/prosumer_count_heatmap_by_battery_pv.svg')
    print('Prosumer count by battery, PV capacity heatmap saved.')


def sor_per_month_plot(thread_num, folder_path):
    """
    Solar Operation Ratio (SOR) per month plot with error bars
    """
    pv_gen_file_path_list = []
    pv_sell_file_path_list = []
    
    # Collect the file paths for all threads
    for i in range(thread_num):
        pv_gen_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/supply.csv')
        pv_gen_sorted_file_paths = sorted(pv_gen_file_paths, key=numerical_sort)
        pv_gen_file_path_list.append(pv_gen_sorted_file_paths[-1])  # get the last episode
        
        pv_sell_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_pv_record.csv')
        pv_sell_sorted_file_paths = sorted(pv_sell_file_paths, key=numerical_sort)
        pv_sell_file_path_list.append(pv_sell_sorted_file_paths[-1])  # get the last episode

    all_ratios = []
    
    # Calculate the ratios for each thread
    for i in range(len(pv_gen_file_path_list)):
        pv_gen = pd.read_csv(pv_gen_file_path_list[i], index_col=0)
        pv_sell = pd.read_csv(pv_sell_file_path_list[i], index_col=0)
        ratio_df = pv_sell / pv_gen
        ratio_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ratio_df.index = pd.to_datetime(ratio_df.index)
        monthly_avg = ratio_df.resample('ME').mean()
        all_ratios.append(monthly_avg)

    # Concatenate all monthly averages
    all_ratios_df = pd.concat(all_ratios)
    
    # Group by month and calculate mean and standard deviation
    monthly_avg = all_ratios_df.groupby(all_ratios_df.index.month).mean()
    monthly_std = all_ratios_df.groupby(all_ratios_df.index.month).std()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = range(1, 13)

    ax.errorbar(x, monthly_avg.mean(axis=1), yerr=monthly_std.mean(axis=1), marker='o', ecolor='red', linestyle='-', linewidth=1, markersize=8, capsize=4)
    # ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel('Solar Operation Ratio [-]')
    ax.set_title('Solar Operation Ratio per Month')
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/sor_per_month.png', dpi=600)
    plt.savefig(folder_path + '/insight/sor_per_month.svg')
    # plt.show()

    # calculate annual average of SOR
    yearly_avg = all_ratios_df.groupby(all_ratios_df.index.year).mean().mean().mean()
    # save annual average of SOR as a text file
    with open(folder_path + '/insight/yearly_avg_sor.txt', 'w') as f:
        f.write(f'Yearly average SOR: {yearly_avg}')
    print('Annual average SOR:' + str(yearly_avg))
    print('SOR per month plot with error bars saved.')
    return yearly_avg


def ssr_per_month_plot(thread_num, folder_path):
    """
    Self Sufficiency Ratio (SSR) per month plot with error bars
    """
    grid_import_file_path_list = []
    # Collect the file paths for all threads
    for i in range(thread_num):
        grid_import_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/grid_import_record.csv')
        grid_import_sorted_file_paths = sorted(grid_import_file_paths, key=numerical_sort)
        grid_import_file_path_list.append(grid_import_sorted_file_paths[-1])  # get the last episode

    buy_inelastic_file_path_list = []
    for i in range(thread_num):
        buy_inelastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_inelastic_record.csv')
        buy_inelastic_sorted_file_paths = sorted(buy_inelastic_file_paths, key=numerical_sort)
        buy_inelastic_file_path_list.append(buy_inelastic_sorted_file_paths[-1])  # get the last episode
    buy_elastic_file_path_list = []
    for i in range(thread_num):
        buy_elastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_elastic_record.csv')
        buy_elastic_sorted_file_paths = sorted(buy_elastic_file_paths, key=numerical_sort)
        buy_elastic_file_path_list.append(buy_elastic_sorted_file_paths[-1])  # get the last episode
    buy_shifted_file_path_list = []
    for i in range(thread_num):
        buy_shifted_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_shifted_record.csv')
        buy_shifted_sorted_file_paths = sorted(buy_shifted_file_paths, key=numerical_sort)
        buy_shifted_file_path_list.append(buy_shifted_sorted_file_paths[-1])  # get the last episode
    buy_battery_file_path_list = []
    for i in range(thread_num):
        buy_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_battery_record.csv')
        buy_battery_sorted_file_paths = sorted(buy_battery_file_paths, key=numerical_sort)
        buy_battery_file_path_list.append(buy_battery_sorted_file_paths[-1])  # get the last episode
    buy_ev_battery_file_path_list = []
    for i in range(thread_num):
        buy_ev_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_ev_battery_record.csv')
        buy_ev_battery_sorted_file_paths = sorted(buy_ev_battery_file_paths, key=numerical_sort)
        buy_ev_battery_file_path_list.append(buy_ev_battery_sorted_file_paths[-1])  # get the last episode

    all_ratios = []
    
    # Calculate the ratios for each thread
    for i in range(len(grid_import_file_path_list)):
        grid_import = pd.read_csv(grid_import_file_path_list[i], index_col=0)
        total_buy_amount = (pd.read_csv(buy_inelastic_file_path_list[i], index_col=0).sum(axis=1) +
                            pd.read_csv(buy_elastic_file_path_list[i], index_col=0).sum(axis=1) +
                            pd.read_csv(buy_shifted_file_path_list[i], index_col=0).sum(axis=1) +
                            pd.read_csv(buy_battery_file_path_list[i], index_col=0).sum(axis=1) +
                            pd.read_csv(buy_ev_battery_file_path_list[i], index_col=0).sum(axis=1))
        ratio_series = 1 - grid_import['Grid import'] / total_buy_amount
        ratio_series.replace([np.inf, -np.inf], np.nan, inplace=True)
        ratio_series.index = pd.to_datetime(ratio_series.index)
        monthly_avg = ratio_series.resample('ME').mean()
        all_ratios.append(monthly_avg)

    # Concatenate all monthly averages
    all_ratios_df = pd.concat(all_ratios)
    
    # Group by month and calculate mean and standard deviation
    monthly_avg = all_ratios_df.groupby(all_ratios_df.index.month).mean()
    monthly_std = all_ratios_df.groupby(all_ratios_df.index.month).std()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = range(1, 13)

    ax.errorbar(x, monthly_avg, yerr=monthly_std, marker='o', ecolor='red', linestyle='-', linewidth=1, markersize=8, capsize=4)
    # ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel('Self Sufficiency Ratio [-]')
    ax.set_title('Self Sufficiency Ratio per Month')
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/ssr_per_month.png', dpi=600)
    plt.savefig(folder_path + '/insight/ssr_per_month.svg')
    # plt.show()

    # calculate annual average of SSR
    yearly_avg = all_ratios_df.groupby(all_ratios_df.index.year).mean()
    # save annual average of SSR as a text file
    with open(folder_path + '/insight/yearly_avg_ssr.txt', 'w') as f:
        f.write(f'Yearly average SSR: {yearly_avg}')
    print('Annual average SSR:' + str(yearly_avg))
    print('SSR per month plot with error bars saved.')

    return yearly_avg


def supply_demand_margin_plot(thread_num, folder_path):
    """
    Supply-Demand Margin plot by boxplot, aggregated in 0-24 hours
    """
    # Collect the file paths for all threads
    potential_supply_file_path_list = []
    for i in range(thread_num):
        potential_supply_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/potential_supply.csv')
        potential_supply_sorted_file_paths = sorted(potential_supply_file_paths, key=numerical_sort)
        potential_supply_file_path_list.append(potential_supply_sorted_file_paths[-1])
    potential_demand_file_path_list = []
    for i in range(thread_num):
        potential_demand_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/potential_demand.csv')
        potential_demand_sorted_file_paths = sorted(potential_demand_file_paths, key=numerical_sort)
        potential_demand_file_path_list.append(potential_demand_sorted_file_paths[-1])

    buy_inelastic_file_path_list = []
    for i in range(thread_num):
        buy_inelastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_inelastic_record.csv')
        buy_inelastic_sorted_file_paths = sorted(buy_inelastic_file_paths, key=numerical_sort)
        buy_inelastic_file_path_list.append(buy_inelastic_sorted_file_paths[-1])  # get the last episode
    buy_elastic_file_path_list = []
    for i in range(thread_num):
        buy_elastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_elastic_record.csv')
        buy_elastic_sorted_file_paths = sorted(buy_elastic_file_paths, key=numerical_sort)
        buy_elastic_file_path_list.append(buy_elastic_sorted_file_paths[-1])  # get the last episode
    buy_shifted_file_path_list = []
    for i in range(thread_num):
        buy_shifted_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_shifted_record.csv')
        buy_shifted_sorted_file_paths = sorted(buy_shifted_file_paths, key=numerical_sort)
        buy_shifted_file_path_list.append(buy_shifted_sorted_file_paths[-1])  # get the last episode
    buy_battery_file_path_list = []
    for i in range(thread_num):
        buy_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_battery_record.csv')
        buy_battery_sorted_file_paths = sorted(buy_battery_file_paths, key=numerical_sort)
        buy_battery_file_path_list.append(buy_battery_sorted_file_paths[-1])  # get the last episode
    buy_ev_battery_file_path_list = []
    for i in range(thread_num):
        buy_ev_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_ev_battery_record.csv')
        buy_ev_battery_sorted_file_paths = sorted(buy_ev_battery_file_paths, key=numerical_sort)
        buy_ev_battery_file_path_list.append(buy_ev_battery_sorted_file_paths[-1])  # get the last episode
    sell_pv_file_path_list = []
    for i in range(thread_num):
        sell_pv_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_pv_record.csv')
        sell_pv_sorted_file_paths = sorted(sell_pv_file_paths, key=numerical_sort)
        sell_pv_file_path_list.append(sell_pv_sorted_file_paths[-1])
    sell_battery_file_path_list = []
    for i in range(thread_num):
        sell_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_battery_record.csv')
        sell_battery_sorted_file_paths = sorted(sell_battery_file_paths, key=numerical_sort)
        sell_battery_file_path_list.append(sell_battery_sorted_file_paths[-1])
    sell_ev_battery_file_path_list = []
    for i in range(thread_num):
        sell_ev_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_ev_battery_record.csv')
        sell_ev_battery_sorted_file_paths = sorted(sell_ev_battery_file_paths, key=numerical_sort)
        sell_ev_battery_file_path_list.append(sell_ev_battery_sorted_file_paths[-1])

    all_ratios = []
    
    # Calculate the total demand and supply for each thread
    surplus_demand_list = []
    surplus_supply_list = []
    for i in range(len(buy_inelastic_file_path_list)):
        total_demand = (pd.read_csv(buy_inelastic_file_path_list[i], index_col=0).sum(axis=1) +
                        pd.read_csv(buy_elastic_file_path_list[i], index_col=0).sum(axis=1) +
                        pd.read_csv(buy_shifted_file_path_list[i], index_col=0).sum(axis=1) +
                        pd.read_csv(buy_battery_file_path_list[i], index_col=0).sum(axis=1) +
                        pd.read_csv(buy_ev_battery_file_path_list[i], index_col=0).sum(axis=1))
        total_supply = (pd.read_csv(sell_pv_file_path_list[i], index_col=0).sum(axis=1) +
                        pd.read_csv(sell_battery_file_path_list[i], index_col=0).sum(axis=1) +
                        pd.read_csv(sell_ev_battery_file_path_list[i], index_col=0).sum(axis=1))
        surplus_demand = (pd.read_csv(potential_demand_file_path_list[i], index_col=0).sum(axis=1) - total_demand).clip(lower=0)
        surplus_supply = (pd.read_csv(potential_supply_file_path_list[i], index_col=0).sum(axis=1) - total_supply).clip(lower=0)
        surplus_demand_list.append(surplus_demand)
        surplus_supply_list.append(surplus_supply)
    # Concatenate all surplus data into single Series by getting average of each timeslot
    # Reset index to avoid potential duplicates or conflicts
    surplus_demand_all = pd.concat(surplus_demand_list, axis=1)
    surplus_supply_all = pd.concat(surplus_supply_list, axis=1)
    surplus_demand_average = surplus_demand_all.mean(axis=1)
    surplus_supply_average = surplus_supply_all.mean(axis=1)

    # Ensure the index is in datetime format
    surplus_demand_average.index = pd.to_datetime(surplus_demand_average.index)
    surplus_supply_average.index = pd.to_datetime(surplus_supply_average.index)
    
    surplus_demand_average_by_hour = surplus_demand.groupby(surplus_demand_average.index.hour)
    surplus_supply_average_by_hour = surplus_supply.groupby(surplus_supply_average.index.hour)

    # Calculate statistics
    demand_mean_by_hour = surplus_demand_average_by_hour.mean()
    supply_mean_by_hour = surplus_supply_average_by_hour.mean()
    demand_std_by_hour = surplus_demand_average_by_hour.std()
    supply_std_by_hour = surplus_supply_average_by_hour.std()
    demand_median_by_hour = surplus_demand_average_by_hour.median()
    supply_median_by_hour = surplus_supply_average_by_hour.median()
    
    fig, ax = plt.subplots(figsize=(20, 10))
    boxprops = dict(color='black', linewidth=1.5)
    medianprops = dict(color='red', linewidth=2)
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='blue', markersize=8)
    bplot = ax.boxplot([surplus_demand_average_by_hour.get_group(i).values for i in range(24)], patch_artist=True, showmeans=True,
                          boxprops=boxprops, medianprops=medianprops, meanprops=meanpointprops)
    # Set all boxplot colors to gray
    gray_color = '#808080'
    for patch in bplot['boxes']:
        patch.set_facecolor(gray_color)
    ax.set_xticklabels([f'{i:02}:00' for i in range(24)])
    ax.set_ylabel('Surplus Demand [kWh]')
    ax.set_title('Surplus Demand by Hour')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 500)
    plt.tight_layout()
    plt.savefig(folder_path + '/insight/surplus_demand_by_hour.png', dpi=600)
    plt.savefig(folder_path + '/insight/surplus_demand_by_hour.svg')
    
    for i in range(24):
        ax.text(i + 1, demand_median_by_hour[i], f'Mean: {demand_mean_by_hour[i]:.2f}\nStd: {demand_std_by_hour[i]:.2f}\nMed: {demand_median_by_hour[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(folder_path + '/insight/surplus_demand_by_hour_with_values.png', dpi=600)
    plt.savefig(folder_path + '/insight/surplus_demand_by_hour_with_values.svg')
    # plt.show()
    # calculate annual average of surplus demand
    yearly_avg_demand = surplus_demand_average.groupby(surplus_demand_average.index.year).mean()
    # save annual average of surplus demand as a text file
    with open(folder_path + '/insight/yearly_avg_surplus_demand.txt', 'w') as f:
        f.write(f'Yearly average surplus demand: {yearly_avg_demand}')
    print('Annual average surplus demand:' + str(yearly_avg_demand))
    print('Surplus Demand by Hour plot saved.')
    plt.close()

    fig, ax = plt.subplots(figsize=(20, 10))
    boxprops = dict(color='black', linewidth=1.5)
    medianprops = dict(color='red', linewidth=2)
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='blue', markersize=8)
    bplot = ax.boxplot([surplus_supply_average_by_hour.get_group(i).values for i in range(24)], patch_artist=True, showmeans=True,
                          boxprops=boxprops, medianprops=medianprops, meanprops=meanpointprops)
    # Set all boxplot colors to gray
    gray_color = '#808080'
    for patch in bplot['boxes']:
        patch.set_facecolor(gray_color)
    ax.set_xticklabels([f'{i:02}:00' for i in range(24)])
    ax.set_ylabel('Surplus Supply [kWh]')
    ax.set_title('Surplus Supply by Hour')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 900)
    plt.tight_layout()
    plt.savefig(folder_path + '/insight/surplus_supply_by_hour.png', dpi=600)
    plt.savefig(folder_path + '/insight/surplus_supply_by_hour.svg')

    for i in range(24):
        ax.text(i + 1, supply_median_by_hour[i], f'Mean: {supply_mean_by_hour[i]:.2f}\nStd: {supply_std_by_hour[i]:.2f}\nMed: {supply_median_by_hour[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(folder_path + '/insight/surplus_supply_by_hour_with_values.png', dpi=600)
    plt.savefig(folder_path + '/insight/surplus_supply_by_hour_with_values.svg')
    # plt.show()
    # calculate annual average of surplus supply
    yearly_avg_supply = surplus_supply_average.groupby(surplus_supply_average.index.year).mean()
    # save annual average of surplus supply as a text file
    with open(folder_path + '/insight/yearly_avg_surplus_supply.txt', 'w') as f:
        f.write(f'Yearly average surplus supply: {yearly_avg_supply}')
    print('Annual average surplus supply:' + str(yearly_avg_supply))
    print('Surplus Supply by Hour plot saved.')
    plt.close()


def bes_pv_installed_capacity(thread_num, folder_path):
    """
    BES and PV installed capacity plot
    """
    # Collect the file paths for all threads
    agent_file_path_list = []
    for i in range(thread_num):
        agent_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/agent_params.csv')
        agent_sorted_file_paths = sorted(agent_file_paths, key=numerical_sort)
        agent_file_path_list.append(agent_sorted_file_paths[-1])  # get the last episode

    bes_capacity_list = []
    pv_capacity_list = []
    for i in range(len(agent_file_path_list)):
        df = pd.read_csv(agent_file_path_list[i], index_col=0)
        bes_capacities = df['battery_capacity']
        pv_capacities = df['pv_capacity']
        bes_capacity = bes_capacities.sum()
        pv_capacity = pv_capacities.sum()
        bes_capacity_list.append(bes_capacity)
        pv_capacity_list.append(pv_capacity)

    # Get average of BES and PV capacities
    bes_capacity_avg = np.mean(bes_capacity_list)
    pv_capacity_avg = np.mean(pv_capacity_list)

    # save as txt file
    with open(folder_path + '/insight/bes_pv_installed_capacity.txt', 'w') as f:
        f.write(f'BES installed capacity: {bes_capacity_avg:.2f} kWh\n')
        f.write(f'PV installed capacity: {pv_capacity_avg:.2f} kW\n')

    return bes_capacity_avg, pv_capacity_avg


def get_master_df(thread_num, folder_path, include_capex_opex=True):
    pv_lifetime = 17
    bes_lifetime = 6
    buy_inelastic_file_path_list = []
    for i in range(thread_num):
        buy_inelastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_inelastic_record.csv')
        buy_inelastic_sorted_file_paths = sorted(buy_inelastic_file_paths, key=numerical_sort)
        buy_inelastic_file_path_list.append(buy_inelastic_sorted_file_paths[-1])  # get the last episode
    buy_elastic_file_path_list = []
    for i in range(thread_num):
        buy_elastic_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_elastic_record.csv')
        buy_elastic_sorted_file_paths = sorted(buy_elastic_file_paths, key=numerical_sort)
        buy_elastic_file_path_list.append(buy_elastic_sorted_file_paths[-1])  # get the last episode
    buy_shifted_file_path_list = []
    for i in range(thread_num):
        buy_shifted_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_shifted_record.csv')
        buy_shifted_sorted_file_paths = sorted(buy_shifted_file_paths, key=numerical_sort)
        buy_shifted_file_path_list.append(buy_shifted_sorted_file_paths[-1])  # get the last episode
    buy_battery_file_path_list = []
    for i in range(thread_num):
        buy_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_battery_record.csv')
        buy_battery_sorted_file_paths = sorted(buy_battery_file_paths, key=numerical_sort)
        buy_battery_file_path_list.append(buy_battery_sorted_file_paths[-1])  # get the last episode
    buy_ev_battery_file_path_list = []
    for i in range(thread_num):
        buy_ev_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/buy_ev_battery_record.csv')
        buy_ev_battery_sorted_file_paths = sorted(buy_ev_battery_file_paths, key=numerical_sort)
        buy_ev_battery_file_path_list.append(buy_ev_battery_sorted_file_paths[-1])  # get the last episode

    sell_pv_file_path_list = []
    for i in range(thread_num):
        sell_pv_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_pv_record.csv')
        sell_pv_sorted_file_paths = sorted(sell_pv_file_paths, key=numerical_sort)
        sell_pv_file_path_list.append(sell_pv_sorted_file_paths[-1])  # get the last episode
    sell_battery_file_path_list = []
    for i in range(thread_num):
        sell_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_battery_record.csv')
        sell_battery_sorted_file_paths = sorted(sell_battery_file_paths, key=numerical_sort)
        sell_battery_file_path_list.append(sell_battery_sorted_file_paths[-1])  # get the last episode
    sell_ev_battery_file_path_list = []
    for i in range(thread_num):
        sell_ev_battery_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/sell_ev_battery_record.csv')
        sell_ev_battery_sorted_file_paths = sorted(sell_ev_battery_file_paths, key=numerical_sort)
        sell_ev_battery_file_path_list.append(sell_ev_battery_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/agent_params.csv')
        agent_params_sorted_file_paths = sorted(agent_params_file_paths, key=numerical_sort)
        agent_params_file_path_list.append(agent_params_sorted_file_paths[-1])  # get the last episode

    microgrid_price_file_path_list = []
    for i in range(thread_num):
        microgrid_price_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/price_record.csv')
        microgrid_price_sorted_file_paths = sorted(microgrid_price_file_paths, key=numerical_sort)
        microgrid_price_file_path_list.append(microgrid_price_sorted_file_paths[-1])  # get the last episode

    master_list = []

    for i in range(len(agent_params_file_path_list)):
        agent_params_file_path = agent_params_file_path_list[i]
        agent_params_df = pd.read_csv(agent_params_file_path, index_col=0)
        microgrid_price = pd.read_csv(microgrid_price_file_path_list[i], index_col=0)
        buy_inelastic = pd.read_csv(buy_inelastic_file_path_list[i], index_col=0)
        buy_elastic = pd.read_csv(buy_elastic_file_path_list[i], index_col=0)
        buy_shifted = pd.read_csv(buy_shifted_file_path_list[i], index_col=0)
        buy_battery = pd.read_csv(buy_battery_file_path_list[i], index_col=0)
        buy_ev_battery = pd.read_csv(buy_ev_battery_file_path_list[i], index_col=0)
        sell_pv = pd.read_csv(sell_pv_file_path_list[i], index_col=0)
        sell_battery = pd.read_csv(sell_battery_file_path_list[i], index_col=0)
        sell_ev_battery = pd.read_csv(sell_ev_battery_file_path_list[i], index_col=0)
        for j in range(agent_params_df.shape[0]):
            battery_capacity = agent_params_df.loc[j, 'battery_capacity']
            ev_capacity = agent_params_df.loc[j, 'ev_capacity']
            pv_capacity = agent_params_df.loc[j, 'pv_capacity']
            dr_boolean = agent_params_df.loc[j, 'dr_boolean']
            # print(f'battery_capacity: {battery_capacity}, ev_capacity: {ev_capacity}, pv_capacity: {pv_capacity}, dr_boolean: {dr_boolean}')

            if include_capex_opex:
                cost = ((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 
                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                + (capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                + capex_opex.pv_opex_func(pv_capacity))
            else:
                cost = ((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100 
                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            
            amount = (buy_inelastic.loc[:, f'{j}'].sum()
            + buy_elastic.loc[:, f'{j}'].sum()
            + buy_shifted.loc[:, f'{j}'].sum()
            + buy_battery.loc[:, f'{j}'].sum()
            + buy_ev_battery.loc[:, f'{j}'].sum()
            # - sell_pv.loc[:, f'{j}'].sum()
            - sell_battery.loc[:, f'{j}'].sum()
            - sell_ev_battery.loc[:, f'{j}'].sum())

            cost_per_kWh = cost / amount
            # print(f'cost: {cost}, amount: {amount}, cost_per_kWh: {cost_per_kWh}')

            master_list.append({'battery_capacity': battery_capacity, 'ev_capacity': ev_capacity, 'pv_capacity': pv_capacity, 'dr_boolean': dr_boolean,
                                'cost': cost, 'amount': amount, 'cost_per_kWh': cost_per_kWh})

    master_df = pd.DataFrame(master_list)
    master_df.to_csv(folder_path + '/insight/master_df.csv')
    return master_df


if __name__ == '__main__':
    max_workers = 16
    
    if os.path.exists('output/no_p2p/test/thread0/episode10/agent_params.csv'): 
        os.makedirs('output/no_p2p/insight', exist_ok=True)   
        print('Start plotting analysis figures for no_p2p...')
        # agent_num = pd.read_csv('output/thread0/episode1/agent_params.csv', index_col=0).shape[0]
        agent_num = pd.read_csv('output/no_p2p/test/thread0/episode10/agent_params.csv', index_col=0).shape[0]
        print(f'Detected number of agents: {agent_num}')

        reward_sorted_file_paths_list = []
        for i in range(max_workers):
            # reward_file_paths = glob.glob(f'output/thread{i}/episode*/reward.csv')
            reward_file_paths = glob.glob(f'output/no_p2p/test/thread{i}/episode*/reward.csv') 
            reward_sorted_file_paths = sorted(reward_file_paths, key=numerical_sort)
            reward_sorted_file_paths_list.append(reward_sorted_file_paths)
        # print(reward_sorted_file_paths_list)
        if os.path.exists('output/no_p2p/insight/reward_history_powerplot.png'):
            print('Reward history powerplot already exists. Skip plotting.')
        else:
            reward_history_plot_4_4_powerplot(reward_sorted_file_paths_list, agent_num=agent_num, folder_path='output/no_p2p')

        if os.path.exists('output/no_p2p/insight/reward_history.png'):
            print('Reward history already exists. Skip plotting.')
        else:
            reward_history_plot_4_4(reward_sorted_file_paths_list, agent_num=agent_num, folder_path='output/no_p2p')

    # ==================================================================================================
        buy_sell_amount_cost_by_battery_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/no_p2p', include_capex_opex=False)
        buy_sell_amount_cost_by_battery_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/no_p2p', include_capex_opex=True)

    # ==================================================================================================
        sor_per_month_plot(thread_num=max_workers, folder_path='output/no_p2p')
        ssr_per_month_plot(thread_num=max_workers, folder_path='output/no_p2p')
        supply_demand_margin_plot(thread_num=max_workers, folder_path='output/no_p2p')
    # ==================================================================================================
        bes_capacity_avg, pv_capacity_avg = bes_pv_installed_capacity(thread_num=max_workers, folder_path='output/no_p2p')
        print(f'BES installed capacity: {bes_capacity_avg:.2f} kWh')
        print(f'PV installed capacity: {pv_capacity_avg:.2f} kW')
        net_cost_by_battery_ev_pv_size_plot(thread_num=max_workers, folder_path='output/no_p2p', include_capex_opex=True)
        net_cost_by_battery_ev_pv_size_plot(thread_num=max_workers, folder_path='output/no_p2p', include_capex_opex=False)
        get_master_df(thread_num=max_workers, folder_path='output/no_p2p', include_capex_opex=True)

# ==================================================================================================
    if os.path.exists('output/p2p/test/thread0/episode10/agent_params.csv'): 
        os.makedirs('output/p2p/insight', exist_ok=True)
        print('Start plotting analysis figures for p2p...')
        agent_num = pd.read_csv('output/p2p/test/thread0/episode10/agent_params.csv', index_col=0).shape[0]
        print(f'Detected number of agents: {agent_num}')

        reward_sorted_file_paths_list = []
        for i in range(max_workers):
            # reward_file_paths = glob.glob(f'output/thread{i}/episode*/reward.csv')
            reward_file_paths = glob.glob(f'output/p2p/test/thread{i}/episode*/reward.csv') 
            reward_sorted_file_paths = sorted(reward_file_paths, key=numerical_sort)
            reward_sorted_file_paths_list.append(reward_sorted_file_paths)
        # print(reward_sorted_file_paths_list)
        if os.path.exists('output/p2p/insight/reward_history_powerplot.png'):
            print('Reward history powerplot already exists. Skip plotting.')
        else:
            reward_history_plot_4_4_powerplot(reward_sorted_file_paths_list, agent_num=agent_num, folder_path='output/p2p')

        if os.path.exists('output/p2p/insight/reward_history.png'):
            print('Reward history already exists. Skip plotting.')
        else:
            reward_history_plot_4_4(reward_sorted_file_paths_list, agent_num=agent_num, folder_path='output/p2p')

    # ==================================================================================================
        buy_sell_amount_cost_by_battery_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/p2p', include_capex_opex=False)
        buy_sell_amount_cost_by_battery_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/p2p', include_capex_opex=True)

    # ==================================================================================================
        sor_per_month_plot(thread_num=max_workers, folder_path='output/p2p')
        ssr_per_month_plot(thread_num=max_workers, folder_path='output/p2p')
        supply_demand_margin_plot(thread_num=max_workers, folder_path='output/p2p')
        
    # ==================================================================================================
        bes_capacity_avg, pv_capacity_avg = bes_pv_installed_capacity(thread_num=max_workers, folder_path='output/p2p')
        print(f'BES installed capacity: {bes_capacity_avg:.2f} kWh')
        print(f'PV installed capacity: {pv_capacity_avg:.2f} kW')
        net_cost_by_battery_ev_pv_size_plot(thread_num=max_workers, folder_path='output/p2p', include_capex_opex=True)
        net_cost_by_battery_ev_pv_size_plot(thread_num=max_workers, folder_path='output/p2p', include_capex_opex=False)
        get_master_df(thread_num=max_workers, folder_path='output/p2p', include_capex_opex=True)
        