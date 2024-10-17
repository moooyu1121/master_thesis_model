import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
# import seaborn as sns
import glob
import os
import re
import capex_opex
max_workers = 16

os.makedirs('output/insight', exist_ok=True)


def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return list(map(int, numbers))


def reward_history_plot_4_4_powerplot(reward_sorted_file_paths_list, agent_num):
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
    fig.savefig('output/insight/reward_history_powerplot.png', dpi=600)
    fig.savefig('output/insight/reward_history_powerplot.svg')
    print('Reward history powerplot saved.')
    print(f'Execution time: {datetime.datetime.now()-start_time}')


def reward_history_plot_4_4(reward_sorted_file_paths_list, agent_num):
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
    fig.savefig('output/insight/reward_history.png', dpi=600)
    fig.savefig('output/insight/reward_history.svg')
    print('Reward history plot saved.')
    print(f'Execution time: {datetime.datetime.now()-start_time}')


def buy_amount_by_battery_ev_pv_dr_exist_plot(thread_num):
    buy_inelastic_file_path_list = []
    for i in range(thread_num):
        buy_inelastic_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_inelastic_record.csv')
        buy_inelastic_sorted_file_paths = sorted(buy_inelastic_file_paths, key=numerical_sort)
        buy_inelastic_file_path_list.append(buy_inelastic_sorted_file_paths[-1])  # get the last episode
    buy_elastic_file_path_list = []
    for i in range(thread_num):
        buy_elastic_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_elastic_record.csv')
        buy_elastic_sorted_file_paths = sorted(buy_elastic_file_paths, key=numerical_sort)
        buy_elastic_file_path_list.append(buy_elastic_sorted_file_paths[-1])  # get the last episode
    buy_shifted_file_path_list = []
    for i in range(thread_num):
        buy_shifted_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_shifted_record.csv')
        buy_shifted_sorted_file_paths = sorted(buy_shifted_file_paths, key=numerical_sort)
        buy_shifted_file_path_list.append(buy_shifted_sorted_file_paths[-1])  # get the last episode
    buy_battery_file_path_list = []
    for i in range(thread_num):
        buy_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_battery_record.csv')
        buy_battery_sorted_file_paths = sorted(buy_battery_file_paths, key=numerical_sort)
        buy_battery_file_path_list.append(buy_battery_sorted_file_paths[-1])  # get the last episode
    buy_ev_battery_file_path_list = []
    for i in range(thread_num):
        buy_ev_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_ev_battery_record.csv')
        buy_ev_battery_sorted_file_paths = sorted(buy_ev_battery_file_paths, key=numerical_sort)
        buy_ev_battery_file_path_list.append(buy_ev_battery_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(f'output/test/thread{i}/episode*/agent_params.csv')
        agent_params_sorted_file_paths = sorted(agent_params_file_paths, key=numerical_sort)
        agent_params_file_path_list.append(agent_params_sorted_file_paths[-1])  # get the last episode
 
    buy_dict = {
        'w/battery_w/ev_w/pv_w/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_w/ev_w/pv_wo/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_w/ev_wo/pv_w/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_w/ev_wo/pv_wo/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_wo/ev_w/pv_w/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_wo/ev_w/pv_wo/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_wo/ev_wo/pv_w/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_wo/ev_wo/pv_wo/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_w/ev_w/pv_w/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_w/ev_w/pv_wo/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_w/ev_wo/pv_w/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_w/ev_wo/pv_wo/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_wo/ev_w/pv_w/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_wo/ev_w/pv_wo/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_wo/ev_wo/pv_w/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_wo/ev_wo/pv_wo/dr': {'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []}        
    }

    for i in range(len(agent_params_file_path_list)):
        agent_params_file_path = agent_params_file_path_list[i]
        agent_params_df = pd.read_csv(agent_params_file_path, index_col=0)
        buy_inelastic = pd.read_csv(buy_inelastic_file_path_list[i], index_col=0)
        buy_elastic = pd.read_csv(buy_elastic_file_path_list[i], index_col=0)
        buy_shifted = pd.read_csv(buy_shifted_file_path_list[i], index_col=0)
        buy_battery = pd.read_csv(buy_battery_file_path_list[i], index_col=0)
        buy_ev_battery = pd.read_csv(buy_ev_battery_file_path_list[i], index_col=0)
        for j in range(agent_params_df.shape[0]):
            battery_capacity = agent_params_df.loc[j, 'battery_capacity']
            ev_capacity = agent_params_df.loc[j, 'ev_capacity']
            pv_capacity = agent_params_df.loc[j, 'pv_capacity']
            dr_boolean = agent_params_df.loc[j, 'dr_boolean']
            if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_inelastic'].append(buy_inelastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_elastic'].append(buy_elastic.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_shifted'].append(buy_shifted.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_battery'].append(buy_battery.loc[:, f'{j}'].sum())
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_ev_battery'].append(buy_ev_battery.loc[:, f'{j}'].sum())

    # draw graph
    categories = ['w/battery_w/ev_w/pv_w/dr', 'w/battery_w/ev_w/pv_wo/dr', 'w/battery_w/ev_wo/pv_w/dr', 'w/battery_w/ev_wo/pv_wo/dr',
                    'w/battery_wo/ev_w/pv_w/dr', 'w/battery_wo/ev_w/pv_wo/dr', 'w/battery_wo/ev_wo/pv_w/dr', 'w/battery_wo/ev_wo/pv_wo/dr',
                    'wo/battery_w/ev_w/pv_w/dr', 'wo/battery_w/ev_w/pv_wo/dr', 'wo/battery_w/ev_wo/pv_w/dr', 'wo/battery_w/ev_wo/pv_wo/dr',
                    'wo/battery_wo/ev_w/pv_w/dr', 'wo/battery_wo/ev_w/pv_wo/dr', 'wo/battery_wo/ev_wo/pv_w/dr', 'wo/battery_wo/ev_wo/pv_wo/dr']
    labels = ['buy_inelastic', 'buy_elastic', 'buy_shifted', 'buy_battery', 'buy_ev_battery']
    colors = ['#0000ff', '#00bfff', '#87ceeb', '#d62728', '#9467bd']  # blue, deepskyblue, skyblue, red, purple

    data_means = []
    data_counts = []
    for category in categories:
        means = [np.mean(buy_dict[category][key]) if len(buy_dict[category][key]) > 0 else 0 for key in labels]
        data_means.append(means)
        data_counts.append(len(buy_dict[category]['buy_inelastic']))

    # make stacked bar graph
    fig, ax = plt.subplots(figsize=(20, 16))
    # bar_width = 0.35
    r = np.arange(len(categories))

    bottom = np.zeros(len(categories))
    for i, label in enumerate(['Inelastic', 'Elastic', 'Shifted', 'Battery', 'EV Battery']):
        values = [data_means[j][i] for j in range(len(categories))]
        ax.bar(r, values, bottom=bottom, label=label, color=colors[i])  # width=bar_width,

        # Show percentage in the middle of each bar
        for j in range(len(categories)):
            if values[j] > 0:
                percentage = values[j] / sum([data_means[j][i] for i in range(len(labels))]) * 100
                ax.text(r[j], bottom[j] + values[j]/2, f'{percentage:.1f}%', ha='center', va='center')

        bottom += np.array(values)

    # Show the number of agents on each bar
    for i, count in enumerate(data_counts):
        ax.text(r[i], bottom[i], f'n={count}', ha='center', va='bottom')

    ax.set_xticks(r)
    ax.set_xticklabels(['w/ BES, w/ EV, w/ PV, w/ DR', 'w/ BES, w/ EV, w/ PV, w/o DR', 'w/ BES, w/ EV, w/o PV, w/ DR', 'w/ BES, w/ EV, w/o PV, w/o DR',
                        'w/ BES, w/o EV, w/ PV, w/ DR', 'w/ BES, w/o EV, w/ PV, w/o DR', 'w/ BES, w/o EV, w/o PV, w/ DR', 'w/ BES, w/o EV, w/o PV, w/o DR',
                        'w/o BES, w/ EV, w/ PV, w/ DR', 'w/o BES, w/ EV, w/ PV, w/o DR', 'w/o BES, w/ EV, w/o PV, w/ DR', 'w/o BES, w/ EV, w/o PV, w/o DR',
                        'w/o BES, w/o EV, w/ PV, w/ DR', 'w/o BES, w/o EV, w/ PV, w/o DR', 'w/o BES, w/o EV, w/o PV, w/ DR', 'w/o BES, w/o EV, w/o PV, w/o DR'],
                    rotation=90)
    ax.set_ylabel('Energy Amount [kWh]')
    ax.set_title('Average Energy Amount Buy Composition')
    ax.legend()

    plt.tight_layout()
    plt.savefig('output/insight/buy_amount_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig('output/insight/buy_amount_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Energy amount buy composition plot saved.')


def buy_cost_by_battery_ev_pv_dr_exist_plot(thread_num):
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
        buy_inelastic_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_inelastic_record.csv')
        buy_inelastic_sorted_file_paths = sorted(buy_inelastic_file_paths, key=numerical_sort)
        buy_inelastic_file_path_list.append(buy_inelastic_sorted_file_paths[-1])  # get the last episode
    buy_elastic_file_path_list = []
    for i in range(thread_num):
        buy_elastic_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_elastic_record.csv')
        buy_elastic_sorted_file_paths = sorted(buy_elastic_file_paths, key=numerical_sort)
        buy_elastic_file_path_list.append(buy_elastic_sorted_file_paths[-1])  # get the last episode
    buy_shifted_file_path_list = []
    for i in range(thread_num):
        buy_shifted_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_shifted_record.csv')
        buy_shifted_sorted_file_paths = sorted(buy_shifted_file_paths, key=numerical_sort)
        buy_shifted_file_path_list.append(buy_shifted_sorted_file_paths[-1])  # get the last episode
    buy_battery_file_path_list = []
    for i in range(thread_num):
        buy_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_battery_record.csv')
        buy_battery_sorted_file_paths = sorted(buy_battery_file_paths, key=numerical_sort)
        buy_battery_file_path_list.append(buy_battery_sorted_file_paths[-1])  # get the last episode
    buy_ev_battery_file_path_list = []
    for i in range(thread_num):
        buy_ev_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_ev_battery_record.csv')
        buy_ev_battery_sorted_file_paths = sorted(buy_ev_battery_file_paths, key=numerical_sort)
        buy_ev_battery_file_path_list.append(buy_ev_battery_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(f'output/test/thread{i}/episode*/agent_params.csv')
        agent_params_sorted_file_paths = sorted(agent_params_file_paths, key=numerical_sort)
        agent_params_file_path_list.append(agent_params_sorted_file_paths[-1])  # get the last episode

    microgrid_price_file_path_list = []
    for i in range(thread_num):
        microgrid_price_file_paths = glob.glob(f'output/test/thread{i}/episode*/price_record.csv')
        microgrid_price_sorted_file_paths = sorted(microgrid_price_file_paths, key=numerical_sort)
        microgrid_price_file_path_list.append(microgrid_price_sorted_file_paths[-1])  # get the last episode
 
    buy_dict = {
        'w/battery_w/ev_w/pv_w/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_w/ev_w/pv_wo/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_w/ev_wo/pv_w/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_w/ev_wo/pv_wo/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_wo/ev_w/pv_w/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_wo/ev_w/pv_wo/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_wo/ev_wo/pv_w/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'w/battery_wo/ev_wo/pv_wo/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_w/ev_w/pv_w/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_w/ev_w/pv_wo/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_w/ev_wo/pv_w/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_w/ev_wo/pv_wo/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_wo/ev_w/pv_w/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_wo/ev_w/pv_wo/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_wo/ev_wo/pv_w/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []},
        'wo/battery_wo/ev_wo/pv_wo/dr': {'pv_capex': [], 'battery_capex': [], 'pv_opex': [], 'buy_inelastic': [], 'buy_elastic': [], 'buy_shifted': [], 'buy_battery': [], 'buy_ev_battery': []}        
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
        for j in range(agent_params_df.shape[0]):
            battery_capacity = agent_params_df.loc[j, 'battery_capacity']
            ev_capacity = agent_params_df.loc[j, 'ev_capacity']
            pv_capacity = agent_params_df.loc[j, 'pv_capacity']
            dr_boolean = agent_params_df.loc[j, 'dr_boolean']
            if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100) # convert from cents to dollars
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_w/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['w/battery_w/ev_w/pv_w/dr']['battery_capex'].append(capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_w/ev_w/pv_w/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['battery_capex'].append(capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['pv_capex'].append(0)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['battery_capex'].append(capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['pv_opex'].append(0)
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['pv_capex'].append(0)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['battery_capex'].append(capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['pv_opex'].append(0)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['battery_capex'].append(capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['battery_capex'].append(capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['pv_capex'].append(0)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['battery_capex'].append(capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['pv_opex'].append(0)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['pv_capex'].append(0)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['battery_capex'].append(capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['pv_opex'].append(0)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['battery_capex'].append(0)
                buy_dict['wo/battery_w/ev_w/pv_w/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['battery_capex'].append(0)
                buy_dict['wo/battery_w/ev_w/pv_wo/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['pv_capex'].append(0)
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['battery_capex'].append(0)
                buy_dict['wo/battery_w/ev_wo/pv_w/dr']['pv_opex'].append(0)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['pv_capex'].append(0)
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['battery_capex'].append(0)
                buy_dict['wo/battery_w/ev_wo/pv_wo/dr']['pv_opex'].append(0)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['battery_capex'].append(0)
                buy_dict['wo/battery_wo/ev_w/pv_w/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['battery_capex'].append(0)
                buy_dict['wo/battery_wo/ev_w/pv_wo/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['pv_capex'].append(0)
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['battery_capex'].append(0)
                buy_dict['wo/battery_wo/ev_wo/pv_w/dr']['pv_opex'].append(0)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['pv_capex'].append(0)
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['battery_capex'].append(0)
                buy_dict['wo/battery_wo/ev_wo/pv_wo/dr']['pv_opex'].append(0)

    # draw graph
    categories = ['w/battery_w/ev_w/pv_w/dr', 'w/battery_w/ev_w/pv_wo/dr', 'w/battery_w/ev_wo/pv_w/dr', 'w/battery_w/ev_wo/pv_wo/dr',
                    'w/battery_wo/ev_w/pv_w/dr', 'w/battery_wo/ev_w/pv_wo/dr', 'w/battery_wo/ev_wo/pv_w/dr', 'w/battery_wo/ev_wo/pv_wo/dr',
                    'wo/battery_w/ev_w/pv_w/dr', 'wo/battery_w/ev_w/pv_wo/dr', 'wo/battery_w/ev_wo/pv_w/dr', 'wo/battery_w/ev_wo/pv_wo/dr',
                    'wo/battery_wo/ev_w/pv_w/dr', 'wo/battery_wo/ev_w/pv_wo/dr', 'wo/battery_wo/ev_wo/pv_w/dr', 'wo/battery_wo/ev_wo/pv_wo/dr']
    labels = ['pv_capex', 'battery_capex', 'pv_opex', 'buy_inelastic', 'buy_elastic', 'buy_shifted', 'buy_battery', 'buy_ev_battery']
    colors = ['#696969', '#A9A9A9', '#D3D3D3', '#0000ff', '#00bfff', '#87ceeb', '#d62728', '#9467bd']  # darkgrey, mediumgrey, lightgrey, blue, deepskyblue, skyblue, red, purple

    data_means = []
    data_counts = []
    for category in categories:
        means = [np.mean(buy_dict[category][key]) if len(buy_dict[category][key]) > 0 else 0 for key in labels]
        data_means.append(means)
        data_counts.append(len(buy_dict[category]['buy_inelastic']))

    # make stacked bar graph
    fig, ax = plt.subplots(figsize=(20, 16))
    # bar_width = 0.35
    r = np.arange(len(categories))

    bottom = np.zeros(len(categories))
    for i, label in enumerate(['PV Capex', 'Battery Capex', 'PV Opex', 'Inelastic', 'Elastic', 'Shifted', 'Battery', 'EV Battery']):
        values = [data_means[j][i] for j in range(len(categories))]
        ax.bar(r, values, bottom=bottom, label=label, color=colors[i])  # width=bar_width, 

        # Show percentage in the middle of each bar
        for j in range(len(categories)):
            if values[j] > 0:
                percentage = values[j] / sum([data_means[j][i] for i in range(len(labels))]) * 100
                ax.text(r[j], bottom[j] + values[j]/2, f'{percentage:.1f}%', ha='center', va='center')

        bottom += np.array(values)

    # Show the number of agents on each bar
    for i, count in enumerate(data_counts):
        ax.text(r[i], bottom[i], f'n={count}', ha='center', va='bottom')

    ax.set_xticks(r)
    ax.set_xticklabels(['w/ BES, w/ EV, w/ PV, w/ DR', 'w/ BES, w/ EV, w/ PV, w/o DR', 'w/ BES, w/ EV, w/o PV, w/ DR', 'w/ BES, w/ EV, w/o PV, w/o DR',
                        'w/ BES, w/o EV, w/ PV, w/ DR', 'w/ BES, w/o EV, w/ PV, w/o DR', 'w/ BES, w/o EV, w/o PV, w/ DR', 'w/ BES, w/o EV, w/o PV, w/o DR',
                        'w/o BES, w/ EV, w/ PV, w/ DR', 'w/o BES, w/ EV, w/ PV, w/o DR', 'w/o BES, w/ EV, w/o PV, w/ DR', 'w/o BES, w/ EV, w/o PV, w/o DR',
                        'w/o BES, w/o EV, w/ PV, w/ DR', 'w/o BES, w/o EV, w/ PV, w/o DR', 'w/o BES, w/o EV, w/o PV, w/ DR', 'w/o BES, w/o EV, w/o PV, w/o DR'],
                    rotation=90)
    ax.set_ylabel('Energy Cost [$]')
    ax.set_title('Average Energy Cost Buy Composition')
    ax.legend()

    plt.tight_layout()
    plt.savefig('output/insight/buy_cost_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig('output/insight/buy_cost_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Energy cost buy composition plot saved.')


def sell_amount_by_battery_ev_pv_dr_exist_plot(thread_num):
    sell_pv_file_path_list = []
    for i in range(thread_num):
        sell_pv_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_pv_record.csv')
        sell_pv_sorted_file_paths = sorted(sell_pv_file_paths, key=numerical_sort)
        sell_pv_file_path_list.append(sell_pv_sorted_file_paths[-1])  # get the last episode
    sell_battery_file_path_list = []
    for i in range(thread_num):
        sell_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_battery_record.csv')
        sell_battery_sorted_file_paths = sorted(sell_battery_file_paths, key=numerical_sort)
        sell_battery_file_path_list.append(sell_battery_sorted_file_paths[-1])  # get the last episode
    sell_ev_battery_file_path_list = []
    for i in range(thread_num):
        sell_ev_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_ev_battery_record.csv')
        sell_ev_battery_sorted_file_paths = sorted(sell_ev_battery_file_paths, key=numerical_sort)
        sell_ev_battery_file_path_list.append(sell_ev_battery_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(f'output/test/thread{i}/episode*/agent_params.csv')
        agent_params_sorted_file_paths = sorted(agent_params_file_paths, key=numerical_sort)
        agent_params_file_path_list.append(agent_params_sorted_file_paths[-1])  # get the last episode
 
    sell_dict = {
        'w/battery_w/ev_w/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_w/ev_w/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_w/ev_wo/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},  
        'w/battery_w/ev_wo/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_wo/ev_w/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_wo/ev_w/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_wo/ev_wo/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_wo/ev_wo/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_w/ev_w/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_w/ev_w/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_w/ev_wo/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_w/ev_wo/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_wo/ev_w/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_wo/ev_w/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_wo/ev_wo/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_wo/ev_wo/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}
    }

    for i in range(len(agent_params_file_path_list)):
        agent_params_file_path = agent_params_file_path_list[i]
        agent_params_df = pd.read_csv(agent_params_file_path, index_col=0)
        sell_pv = pd.read_csv(sell_pv_file_path_list[i], index_col=0)
        sell_battery = pd.read_csv(sell_battery_file_path_list[i], index_col=0)
        sell_ev_battery = pd.read_csv(sell_ev_battery_file_path_list[i], index_col=0)
        for j in range(agent_params_df.shape[0]):
            battery_capacity = agent_params_df.loc[j, 'battery_capacity']
            ev_capacity = agent_params_df.loc[j, 'ev_capacity']
            pv_capacity = agent_params_df.loc[j, 'pv_capacity']
            dr_boolean = agent_params_df.loc[j, 'dr_boolean']
            if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                sell_dict['w/battery_w/ev_w/pv_w/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['w/battery_w/ev_w/pv_w/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['w/battery_w/ev_w/pv_w/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                sell_dict['w/battery_w/ev_w/pv_wo/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['w/battery_w/ev_w/pv_wo/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['w/battery_w/ev_w/pv_wo/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                sell_dict['w/battery_w/ev_wo/pv_w/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['w/battery_w/ev_wo/pv_w/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['w/battery_w/ev_wo/pv_w/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                sell_dict['w/battery_w/ev_wo/pv_wo/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['w/battery_w/ev_wo/pv_wo/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['w/battery_w/ev_wo/pv_wo/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                sell_dict['w/battery_wo/ev_w/pv_w/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['w/battery_wo/ev_w/pv_w/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['w/battery_wo/ev_w/pv_w/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                sell_dict['w/battery_wo/ev_w/pv_wo/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['w/battery_wo/ev_w/pv_wo/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['w/battery_wo/ev_w/pv_wo/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                sell_dict['w/battery_wo/ev_wo/pv_w/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['w/battery_wo/ev_wo/pv_w/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['w/battery_wo/ev_wo/pv_w/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                sell_dict['w/battery_wo/ev_wo/pv_wo/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['w/battery_wo/ev_wo/pv_wo/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['w/battery_wo/ev_wo/pv_wo/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                sell_dict['wo/battery_w/ev_w/pv_w/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_w/ev_w/pv_w/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_w/ev_w/pv_w/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                sell_dict['wo/battery_w/ev_w/pv_wo/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_w/ev_w/pv_wo/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_w/ev_w/pv_wo/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                sell_dict['wo/battery_w/ev_wo/pv_w/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_w/ev_wo/pv_w/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_w/ev_wo/pv_w/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                sell_dict['wo/battery_w/ev_wo/pv_wo/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_w/ev_wo/pv_wo/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_w/ev_wo/pv_wo/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                sell_dict['wo/battery_wo/ev_w/pv_w/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_wo/ev_w/pv_w/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_wo/ev_w/pv_w/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                sell_dict['wo/battery_wo/ev_w/pv_wo/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_wo/ev_w/pv_wo/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_wo/ev_w/pv_wo/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                sell_dict['wo/battery_wo/ev_wo/pv_w/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_wo/ev_wo/pv_w/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_wo/ev_wo/pv_w/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                sell_dict['wo/battery_wo/ev_wo/pv_wo/dr']['sell_pv'].append(sell_pv.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_wo/ev_wo/pv_wo/dr']['sell_battery'].append(sell_battery.loc[:, f'{j}'].sum())
                sell_dict['wo/battery_wo/ev_wo/pv_wo/dr']['sell_ev_battery'].append(sell_ev_battery.loc[:, f'{j}'].sum())

    # draw graph
    categories = ['w/battery_w/ev_w/pv_w/dr', 'w/battery_w/ev_w/pv_wo/dr', 'w/battery_w/ev_wo/pv_w/dr', 'w/battery_w/ev_wo/pv_wo/dr',
                    'w/battery_wo/ev_w/pv_w/dr', 'w/battery_wo/ev_w/pv_wo/dr', 'w/battery_wo/ev_wo/pv_w/dr', 'w/battery_wo/ev_wo/pv_wo/dr',
                    'wo/battery_w/ev_w/pv_w/dr', 'wo/battery_w/ev_w/pv_wo/dr', 'wo/battery_w/ev_wo/pv_w/dr', 'wo/battery_w/ev_wo/pv_wo/dr',
                    'wo/battery_wo/ev_w/pv_w/dr', 'wo/battery_wo/ev_w/pv_wo/dr', 'wo/battery_wo/ev_wo/pv_w/dr', 'wo/battery_wo/ev_wo/pv_wo/dr']
    labels = ['sell_pv', 'sell_battery', 'sell_ev_battery']
    colors = ['#ffd700', '#d62728', '#9467bd']  # gold, red, purple

    data_means = []
    data_counts = []
    for category in categories:
        means = [np.mean(sell_dict[category][key]) if len(sell_dict[category][key]) > 0 else 0 for key in labels]
        data_means.append(means)
        data_counts.append(len(sell_dict[category]['sell_pv']))

    # make stacked bar graph
    fig, ax = plt.subplots(figsize=(20, 16))
    # bar_width = 0.35
    r = np.arange(len(categories))

    bottom = np.zeros(len(categories))
    for i, label in enumerate(['PV', 'Battery', 'EV Battery']):
        values = [data_means[j][i] for j in range(len(categories))]
        ax.bar(r, values, bottom=bottom, label=label, color=colors[i])  # width=bar_width, 

        # Show percentage in the middle of each bar
        for j in range(len(categories)):
            if values[j] > 0:
                percentage = values[j] / sum([data_means[j][i] for i in range(len(labels))]) * 100
                ax.text(r[j], bottom[j] + values[j]/2, f'{percentage:.1f}%', ha='center', va='center')

        bottom += np.array(values)

    # Show the number of agents on each bar
    for i, count in enumerate(data_counts):
        ax.text(r[i], bottom[i], f'n={count}', ha='center', va='bottom')

    ax.set_xticks(r)
    ax.set_xticklabels(['w/ BES, w/ EV, w/ PV, w/ DR', 'w/ BES, w/ EV, w/ PV, w/o DR', 'w/ BES, w/ EV, w/o PV, w/ DR', 'w/ BES, w/ EV, w/o PV, w/o DR',
                        'w/ BES, w/o EV, w/ PV, w/ DR', 'w/ BES, w/o EV, w/ PV, w/o DR', 'w/ BES, w/o EV, w/o PV, w/ DR', 'w/ BES, w/o EV, w/o PV, w/o DR',
                        'w/o BES, w/ EV, w/ PV, w/ DR', 'w/o BES, w/ EV, w/ PV, w/o DR', 'w/o BES, w/ EV, w/o PV, w/ DR', 'w/o BES, w/ EV, w/o PV, w/o DR',
                        'w/o BES, w/o EV, w/ PV, w/ DR', 'w/o BES, w/o EV, w/ PV, w/o DR', 'w/o BES, w/o EV, w/o PV, w/ DR', 'w/o BES, w/o EV, w/o PV, w/o DR'],
                    rotation=90)
    ax.set_ylabel('Energy Amount [kWh]')
    ax.set_title('Average Energy Amount Sell Composition')
    ax.legend()

    plt.tight_layout()
    plt.savefig('output/insight/sell_amount_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig('output/insight/sell_amount_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Energy amount sell composition plot saved.')


def sell_cost_by_battery_ev_pv_dr_exist_plot(thread_num):
    sell_pv_file_path_list = []
    for i in range(thread_num):
        sell_pv_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_pv_record.csv')
        sell_pv_sorted_file_paths = sorted(sell_pv_file_paths, key=numerical_sort)
        sell_pv_file_path_list.append(sell_pv_sorted_file_paths[-1])  # get the last episode
    sell_battery_file_path_list = []
    for i in range(thread_num):
        sell_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_battery_record.csv')
        sell_battery_sorted_file_paths = sorted(sell_battery_file_paths, key=numerical_sort)
        sell_battery_file_path_list.append(sell_battery_sorted_file_paths[-1])  # get the last episode
    sell_ev_battery_file_path_list = []
    for i in range(thread_num):
        sell_ev_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_ev_battery_record.csv')
        sell_ev_battery_sorted_file_paths = sorted(sell_ev_battery_file_paths, key=numerical_sort)
        sell_ev_battery_file_path_list.append(sell_ev_battery_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(f'output/test/thread{i}/episode*/agent_params.csv')
        agent_params_sorted_file_paths = sorted(agent_params_file_paths, key=numerical_sort)
        agent_params_file_path_list.append(agent_params_sorted_file_paths[-1])  # get the last episode

    microgrid_price_file_path_list = []
    for i in range(thread_num):
        microgrid_price_file_paths = glob.glob(f'output/test/thread{i}/episode*/price_record.csv')
        microgrid_price_sorted_file_paths = sorted(microgrid_price_file_paths, key=numerical_sort)
        microgrid_price_file_path_list.append(microgrid_price_sorted_file_paths[-1])  # get the last episode
 
    sell_dict = {
        'w/battery_w/ev_w/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_w/ev_w/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_w/ev_wo/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},  
        'w/battery_w/ev_wo/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_wo/ev_w/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_wo/ev_w/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_wo/ev_wo/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'w/battery_wo/ev_wo/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_w/ev_w/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_w/ev_w/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_w/ev_wo/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_w/ev_wo/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_wo/ev_w/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_wo/ev_w/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_wo/ev_wo/pv_w/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []},
        'wo/battery_wo/ev_wo/pv_wo/dr': {'sell_pv': [], 'sell_battery': [], 'sell_ev_battery': []}
    }

    for i in range(len(agent_params_file_path_list)):
        agent_params_file_path = agent_params_file_path_list[i]
        agent_params_df = pd.read_csv(agent_params_file_path, index_col=0)
        microgrid_price = pd.read_csv(microgrid_price_file_path_list[i], index_col=0)
        sell_pv = pd.read_csv(sell_pv_file_path_list[i], index_col=0)
        sell_battery = pd.read_csv(sell_battery_file_path_list[i], index_col=0)
        sell_ev_battery = pd.read_csv(sell_ev_battery_file_path_list[i], index_col=0)
        for j in range(agent_params_df.shape[0]):
            battery_capacity = agent_params_df.loc[j, 'battery_capacity']
            ev_capacity = agent_params_df.loc[j, 'ev_capacity']
            pv_capacity = agent_params_df.loc[j, 'pv_capacity']
            dr_boolean = agent_params_df.loc[j, 'dr_boolean']
            if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                sell_dict['w/battery_w/ev_w/pv_w/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_w/ev_w/pv_w/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_w/ev_w/pv_w/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                sell_dict['w/battery_w/ev_w/pv_wo/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_w/ev_w/pv_wo/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_w/ev_w/pv_wo/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                sell_dict['w/battery_w/ev_wo/pv_w/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_w/ev_wo/pv_w/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_w/ev_wo/pv_w/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                sell_dict['w/battery_w/ev_wo/pv_wo/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_w/ev_wo/pv_wo/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_w/ev_wo/pv_wo/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                sell_dict['w/battery_wo/ev_w/pv_w/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_wo/ev_w/pv_w/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_wo/ev_w/pv_w/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                sell_dict['w/battery_wo/ev_w/pv_wo/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_wo/ev_w/pv_wo/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_wo/ev_w/pv_wo/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                sell_dict['w/battery_wo/ev_wo/pv_w/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_wo/ev_wo/pv_w/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_wo/ev_wo/pv_w/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                sell_dict['w/battery_wo/ev_wo/pv_wo/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_wo/ev_wo/pv_wo/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['w/battery_wo/ev_wo/pv_wo/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                sell_dict['wo/battery_w/ev_w/pv_w/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_w/ev_w/pv_w/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_w/ev_w/pv_w/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                sell_dict['wo/battery_w/ev_w/pv_wo/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_w/ev_w/pv_wo/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_w/ev_w/pv_wo/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                sell_dict['wo/battery_w/ev_wo/pv_w/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_w/ev_wo/pv_w/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_w/ev_wo/pv_w/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                sell_dict['wo/battery_w/ev_wo/pv_wo/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_w/ev_wo/pv_wo/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_w/ev_wo/pv_wo/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                sell_dict['wo/battery_wo/ev_w/pv_w/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_wo/ev_w/pv_w/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_wo/ev_w/pv_w/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                sell_dict['wo/battery_wo/ev_w/pv_wo/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_wo/ev_w/pv_wo/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_wo/ev_w/pv_wo/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                sell_dict['wo/battery_wo/ev_wo/pv_w/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_wo/ev_wo/pv_w/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_wo/ev_wo/pv_w/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                sell_dict['wo/battery_wo/ev_wo/pv_wo/dr']['sell_pv'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_wo/ev_wo/pv_wo/dr']['sell_battery'].append((sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                sell_dict['wo/battery_wo/ev_wo/pv_wo/dr']['sell_ev_battery'].append((sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)

    # draw graph
    categories = ['w/battery_w/ev_w/pv_w/dr', 'w/battery_w/ev_w/pv_wo/dr', 'w/battery_w/ev_wo/pv_w/dr', 'w/battery_w/ev_wo/pv_wo/dr',
                    'w/battery_wo/ev_w/pv_w/dr', 'w/battery_wo/ev_w/pv_wo/dr', 'w/battery_wo/ev_wo/pv_w/dr', 'w/battery_wo/ev_wo/pv_wo/dr',
                    'wo/battery_w/ev_w/pv_w/dr', 'wo/battery_w/ev_w/pv_wo/dr', 'wo/battery_w/ev_wo/pv_w/dr', 'wo/battery_w/ev_wo/pv_wo/dr',
                    'wo/battery_wo/ev_w/pv_w/dr', 'wo/battery_wo/ev_w/pv_wo/dr', 'wo/battery_wo/ev_wo/pv_w/dr', 'wo/battery_wo/ev_wo/pv_wo/dr']
    labels = ['sell_pv', 'sell_battery', 'sell_ev_battery']
    colors = ['#ffd700', '#d62728', '#9467bd']  # gold, red, purple

    data_means = []
    data_counts = []
    for category in categories:
        means = [np.mean(sell_dict[category][key]) if len(sell_dict[category][key]) > 0 else 0 for key in labels]
        data_means.append(means)
        data_counts.append(len(sell_dict[category]['sell_pv']))

    # make stacked bar graph
    fig, ax = plt.subplots(figsize=(20, 16))
    # bar_width = 0.35
    r = np.arange(len(categories))

    bottom = np.zeros(len(categories))
    for i, label in enumerate(['PV', 'Battery', 'EV Battery']):
        values = [data_means[j][i] for j in range(len(categories))]
        ax.bar(r, values, bottom=bottom, label=label, color=colors[i])  # width=bar_width,

        # Show percentage in the middle of each bar
        for j in range(len(categories)):
            if values[j] > 0:
                percentage = values[j] / sum([data_means[j][i] for i in range(len(labels))]) * 100
                ax.text(r[j], bottom[j] + values[j]/2, f'{percentage:.1f}%', ha='center', va='center')

        bottom += np.array(values)

    # Show the number of agents on each bar
    for i, count in enumerate(data_counts):
        ax.text(r[i], bottom[i], f'n={count}', ha='center', va='bottom')

    ax.set_xticks(r)
    ax.set_xticklabels(['w/ BES, w/ EV, w/ PV, w/ DR', 'w/ BES, w/ EV, w/ PV, w/o DR', 'w/ BES, w/ EV, w/o PV, w/ DR', 'w/ BES, w/ EV, w/o PV, w/o DR',
                        'w/ BES, w/o EV, w/ PV, w/ DR', 'w/ BES, w/o EV, w/ PV, w/o DR', 'w/ BES, w/o EV, w/o PV, w/ DR', 'w/ BES, w/o EV, w/o PV, w/o DR',
                        'w/o BES, w/ EV, w/ PV, w/ DR', 'w/o BES, w/ EV, w/ PV, w/o DR', 'w/o BES, w/ EV, w/o PV, w/ DR', 'w/o BES, w/ EV, w/o PV, w/o DR',
                        'w/o BES, w/o EV, w/ PV, w/ DR', 'w/o BES, w/o EV, w/ PV, w/o DR', 'w/o BES, w/o EV, w/o PV, w/ DR', 'w/o BES, w/o EV, w/o PV, w/o DR'],
                    rotation=90)
    ax.set_ylabel('Energy Cost [$]')
    ax.set_title('Average Energy Cost Sell Composition')
    ax.legend()

    plt.tight_layout()
    plt.savefig('output/insight/sell_cost_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig('output/insight/sell_cost_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Energy cost sell composition plot saved.')


def net_cost_by_battery_ev_pv_dr_exist_plot(thread_num):
    net_cost_file_path_list = []
    for i in range(thread_num):
        # Change it later
        net_cost_file_paths = glob.glob(f'output/test/thread{i}/episode*/net_electricity_cost.csv')
        net_cost_sorted_file_paths = sorted(net_cost_file_paths, key=numerical_sort)
        net_cost_file_path_list.append(net_cost_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(f'output/thread{i}/episode*/agent_params.csv')
        agent_params_sorted_file_paths = sorted(agent_params_file_paths, key=numerical_sort)
        agent_params_file_path_list.append(agent_params_sorted_file_paths[-1])  # get the last episode
    
    net_dict = {
        'w/battery_w/ev_w/pv_w/dr': [],
        'w/battery_w/ev_w/pv_wo/dr': [],
        'w/battery_w/ev_wo/pv_w/dr': [],
        'w/battery_w/ev_wo/pv_wo/dr': [],
        'w/battery_wo/ev_w/pv_w/dr': [],
        'w/battery_wo/ev_w/pv_wo/dr': [],
        'w/battery_wo/ev_wo/pv_w/dr': [],
        'w/battery_wo/ev_wo/pv_wo/dr': [],
        'wo/battery_w/ev_w/pv_w/dr': [],
        'wo/battery_w/ev_w/pv_wo/dr': [],
        'wo/battery_w/ev_wo/pv_w/dr': [],
        'wo/battery_w/ev_wo/pv_wo/dr': [],
        'wo/battery_wo/ev_w/pv_w/dr': [],
        'wo/battery_wo/ev_w/pv_wo/dr': [],
        'wo/battery_wo/ev_wo/pv_w/dr': [],
        'wo/battery_wo/ev_wo/pv_wo/dr': []
    }

    for i in range(len(agent_params_file_path_list)):
        agent_params_file_path = agent_params_file_path_list[i]
        agent_params_df = pd.read_csv(agent_params_file_path, index_col=0)
        net_cost_df = pd.read_csv(net_cost_file_path_list[i], index_col=0)  # already recorded as dollars
        for j in range(agent_params_df.shape[0]):
            battery_capacity = agent_params_df.loc[j, 'battery_capacity']
            ev_capacity = agent_params_df.loc[j, 'ev_capacity']
            pv_capacity = agent_params_df.loc[j, 'pv_capacity']
            dr_boolean = agent_params_df.loc[j, 'dr_boolean']
            if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_w/ev_w/pv_w/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_w/ev_w/pv_wo/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_w/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_wo/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_w/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_wo/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_w/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_wo/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_w/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_wo/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_w/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_wo/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_w/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_wo/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_w/dr'].append(net_cost_df.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_wo/dr'].append(net_cost_df.loc[:, f'{j}'].sum())

    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]) for key in net_dict.keys()]

    # Plotting
    categories = ['w/battery_w/ev_w/pv_w/dr', 'w/battery_w/ev_w/pv_wo/dr', 'w/battery_w/ev_wo/pv_w/dr', 'w/battery_w/ev_wo/pv_wo/dr',
                    'w/battery_wo/ev_w/pv_w/dr', 'w/battery_wo/ev_w/pv_wo/dr', 'w/battery_wo/ev_wo/pv_w/dr', 'w/battery_wo/ev_wo/pv_wo/dr',
                    'wo/battery_w/ev_w/pv_w/dr', 'wo/battery_w/ev_w/pv_wo/dr', 'wo/battery_w/ev_wo/pv_w/dr', 'wo/battery_w/ev_wo/pv_wo/dr',
                    'wo/battery_wo/ev_w/pv_w/dr', 'wo/battery_wo/ev_w/pv_wo/dr', 'wo/battery_wo/ev_wo/pv_w/dr', 'wo/battery_wo/ev_wo/pv_wo/dr']

    fig, ax = plt.subplots(figsize=(20, 16))

    # Create a boxplot for each category
    boxprops = dict(color='black', linewidth=1.5)
    medianprops = dict(color='red', linewidth=2)
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='blue', markersize=8)

    bplot = ax.boxplot([net_dict[cat] for cat in categories], patch_artist=True, showmeans=True,
                       boxprops=boxprops, medianprops=medianprops, meanprops=meanpointprops)

    # Set boxplot colors
    colors = ['#1f77b4', '#0000ff', '#66c2a5', '#d62728']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # Add labels, title, and grid
    ax.set_xticklabels(['w/ BES, w/ EV, w/ PV, w/ DR', 'w/ BES, w/ EV, w/ PV, w/o DR', 'w/ BES, w/ EV, w/o PV, w/ DR', 'w/ BES, w/ EV, w/o PV, w/o DR',
                        'w/ BES, w/o EV, w/ PV, w/ DR', 'w/ BES, w/o EV, w/ PV, w/o DR', 'w/ BES, w/o EV, w/o PV, w/ DR', 'w/ BES, w/o EV, w/o PV, w/o DR',
                        'w/o BES, w/ EV, w/ PV, w/ DR', 'w/o BES, w/ EV, w/ PV, w/o DR', 'w/o BES, w/ EV, w/o PV, w/ DR', 'w/o BES, w/ EV, w/o PV, w/o DR',
                        'w/o BES, w/o EV, w/ PV, w/ DR', 'w/o BES, w/o EV, w/ PV, w/o DR', 'w/o BES, w/o EV, w/o PV, w/ DR', 'w/o BES, w/o EV, w/o PV, w/o DR'],
                    rotation=90)
    ax.set_ylabel('Net Electricity Cost [$]')
    ax.set_title('Net Electricity Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, mean_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('output/insight/net_cost_by_battery_ev_pv_dr_misstake.png', dpi=600)
    plt.savefig('output/insight/net_cost_by_battery_ev_pv_dr_misstake.svg')
    # plt.show()

    print('Net cost by battery, EV, PV, and DR plot saved. Maybe mistake in the data.')


def net_cost_by_battery_ev_pv_dr_exist_plot_2(thread_num):
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
        buy_inelastic_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_inelastic_record.csv')
        buy_inelastic_sorted_file_paths = sorted(buy_inelastic_file_paths, key=numerical_sort)
        buy_inelastic_file_path_list.append(buy_inelastic_sorted_file_paths[-1])  # get the last episode
    buy_elastic_file_path_list = []
    for i in range(thread_num):
        buy_elastic_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_elastic_record.csv')
        buy_elastic_sorted_file_paths = sorted(buy_elastic_file_paths, key=numerical_sort)
        buy_elastic_file_path_list.append(buy_elastic_sorted_file_paths[-1])  # get the last episode
    buy_shifted_file_path_list = []
    for i in range(thread_num):
        buy_shifted_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_shifted_record.csv')
        buy_shifted_sorted_file_paths = sorted(buy_shifted_file_paths, key=numerical_sort)
        buy_shifted_file_path_list.append(buy_shifted_sorted_file_paths[-1])  # get the last episode
    buy_battery_file_path_list = []
    for i in range(thread_num):
        buy_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_battery_record.csv')
        buy_battery_sorted_file_paths = sorted(buy_battery_file_paths, key=numerical_sort)
        buy_battery_file_path_list.append(buy_battery_sorted_file_paths[-1])  # get the last episode
    buy_ev_battery_file_path_list = []
    for i in range(thread_num):
        buy_ev_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_ev_battery_record.csv')
        buy_ev_battery_sorted_file_paths = sorted(buy_ev_battery_file_paths, key=numerical_sort)
        buy_ev_battery_file_path_list.append(buy_ev_battery_sorted_file_paths[-1])  # get the last episode

    sell_pv_file_path_list = []
    for i in range(thread_num):
        sell_pv_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_pv_record.csv')
        sell_pv_sorted_file_paths = sorted(sell_pv_file_paths, key=numerical_sort)
        sell_pv_file_path_list.append(sell_pv_sorted_file_paths[-1])  # get the last episode
    sell_battery_file_path_list = []
    for i in range(thread_num):
        sell_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_battery_record.csv')
        sell_battery_sorted_file_paths = sorted(sell_battery_file_paths, key=numerical_sort)
        sell_battery_file_path_list.append(sell_battery_sorted_file_paths[-1])  # get the last episode
    sell_ev_battery_file_path_list = []
    for i in range(thread_num):
        sell_ev_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_ev_battery_record.csv')
        sell_ev_battery_sorted_file_paths = sorted(sell_ev_battery_file_paths, key=numerical_sort)
        sell_ev_battery_file_path_list.append(sell_ev_battery_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(f'output/test/thread{i}/episode*/agent_params.csv')
        agent_params_sorted_file_paths = sorted(agent_params_file_paths, key=numerical_sort)
        agent_params_file_path_list.append(agent_params_sorted_file_paths[-1])  # get the last episode

    microgrid_price_file_path_list = []
    for i in range(thread_num):
        microgrid_price_file_paths = glob.glob(f'output/test/thread{i}/episode*/price_record.csv')
        microgrid_price_sorted_file_paths = sorted(microgrid_price_file_paths, key=numerical_sort)
        microgrid_price_file_path_list.append(microgrid_price_sorted_file_paths[-1])  # get the last episode

    net_dict = {
        'w/battery_w/ev_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_w/ev_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_w/ev_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_w/ev_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/ev_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/ev_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/ev_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/ev_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/ev_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/ev_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/ev_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/ev_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/ev_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/ev_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/ev_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/ev_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []}
    }

    buy_composition = ['buy_inelastic', 'buy_elastic', 'buy_shifted', 'buy_battery', 'buy_ev_battery']
    sell_composition = ['sell_pv', 'sell_battery', 'sell_ev_battery']

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
            if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_w/ev_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                            + capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime
                                                            + capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_w/ev_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                            + capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime
                                                            + capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                            + capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime
                                                            + capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                            + capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime
                                                            + capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.battery_capex(battery_capacity, pv_capacity) / bes_lifetime)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                            + capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                            + capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                            + capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                            + capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                
            if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_w/ev_w/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
                                                                            
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_w/ev_w/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
                
    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]['cost']) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]['cost']) for key in net_dict.keys()]

    # Plotting
    categories = ['w/battery_w/ev_w/pv_w/dr', 'w/battery_w/ev_w/pv_wo/dr', 'w/battery_w/ev_wo/pv_w/dr', 'w/battery_w/ev_wo/pv_wo/dr',
                    'w/battery_wo/ev_w/pv_w/dr', 'w/battery_wo/ev_w/pv_wo/dr', 'w/battery_wo/ev_wo/pv_w/dr', 'w/battery_wo/ev_wo/pv_wo/dr',
                    'wo/battery_w/ev_w/pv_w/dr', 'wo/battery_w/ev_w/pv_wo/dr', 'wo/battery_w/ev_wo/pv_w/dr', 'wo/battery_w/ev_wo/pv_wo/dr',
                    'wo/battery_wo/ev_w/pv_w/dr', 'wo/battery_wo/ev_w/pv_wo/dr', 'wo/battery_wo/ev_wo/pv_w/dr', 'wo/battery_wo/ev_wo/pv_wo/dr']

    fig, ax = plt.subplots(figsize=(20, 16))

    # Create a boxplot for each category
    boxprops = dict(color='black', linewidth=1.5)
    medianprops = dict(color='red', linewidth=2)
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='blue', markersize=8)

    bplot = ax.boxplot([net_dict[cat]['cost'] for cat in categories], patch_artist=True, showmeans=True,
                       boxprops=boxprops, medianprops=medianprops, meanprops=meanpointprops)

    # Set all boxplot colors to gray
    gray_color = '#808080'
    for patch in bplot['boxes']:
        patch.set_facecolor(gray_color)

    # Add labels, title, and grid
    ax.set_xticklabels(['w/ BES, w/ EV, w/ PV, w/ DR', 'w/ BES, w/ EV, w/ PV, w/o DR', 'w/ BES, w/ EV, w/o PV, w/ DR', 'w/ BES, w/ EV, w/o PV, w/o DR',
                        'w/ BES, w/o EV, w/ PV, w/ DR', 'w/ BES, w/o EV, w/ PV, w/o DR', 'w/ BES, w/o EV, w/o PV, w/ DR', 'w/ BES, w/o EV, w/o PV, w/o DR',
                        'w/o BES, w/ EV, w/ PV, w/ DR', 'w/o BES, w/ EV, w/ PV, w/o DR', 'w/o BES, w/ EV, w/o PV, w/ DR', 'w/o BES, w/ EV, w/o PV, w/o DR',
                        'w/o BES, w/o EV, w/ PV, w/ DR', 'w/o BES, w/o EV, w/ PV, w/o DR', 'w/o BES, w/o EV, w/o PV, w/ DR', 'w/o BES, w/o EV, w/o PV, w/o DR'],
                    rotation=90)
    ax.set_ylabel('Net Electricity Cost [$]')
    ax.set_title('Net Electricity Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, mean_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('output/insight/net_cost_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig('output/insight/net_cost_by_battery_ev_pv_dr.svg')
    # plt.show()

    print('Net cost by battery, EV, PV, and DR plot saved.')
    plt.close()

    # Calculate cost per kWh for each category
    for key in net_dict.keys():
        for i in range(len(net_dict[key]['cost'])):
            net_dict[key]['cost/kWh'].append(net_dict[key]['cost'][i]/net_dict[key]['amount'][i])
    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]['cost/kWh']) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]['cost/kWh']) for key in net_dict.keys()]

    fig, ax = plt.subplots(figsize=(20, 16))

    # Create a boxplot for each category
    boxprops = dict(color='black', linewidth=1.5)
    medianprops = dict(color='red', linewidth=2)
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='blue', markersize=8)

    bplot = ax.boxplot([net_dict[cat]['cost/kWh'] for cat in categories], patch_artist=True, showmeans=True,
                       boxprops=boxprops, medianprops=medianprops, meanprops=meanpointprops)

    # Set all boxplot colors to gray
    gray_color = '#808080'
    for patch in bplot['boxes']:
        patch.set_facecolor(gray_color)

    # Add labels, title, and grid
    ax.set_xticklabels(['w/ BES, w/ EV, w/ PV, w/ DR', 'w/ BES, w/ EV, w/ PV, w/o DR', 'w/ BES, w/ EV, w/o PV, w/ DR', 'w/ BES, w/ EV, w/o PV, w/o DR',
                        'w/ BES, w/o EV, w/ PV, w/ DR', 'w/ BES, w/o EV, w/ PV, w/o DR', 'w/ BES, w/o EV, w/o PV, w/ DR', 'w/ BES, w/o EV, w/o PV, w/o DR',
                        'w/o BES, w/ EV, w/ PV, w/ DR', 'w/o BES, w/ EV, w/ PV, w/o DR', 'w/o BES, w/ EV, w/o PV, w/ DR', 'w/o BES, w/ EV, w/o PV, w/o DR',
                        'w/o BES, w/o EV, w/ PV, w/ DR', 'w/o BES, w/o EV, w/ PV, w/o DR', 'w/o BES, w/o EV, w/o PV, w/ DR', 'w/o BES, w/o EV, w/o PV, w/o DR'],
                    rotation=90)
    ax.set_ylabel('Net Electricity Unit Cost [$/kWh]')
    ax.set_title('Net Electricity Unit Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(-10, 10)

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, mean_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('output/insight/net_cost_per_kWh_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig('output/insight/net_cost_per_kWh_by_battery_ev_pv_dr.svg')
    # plt.show()

    print('Net unit cost by battery, EV, PV, and DR plot saved.')
    plt.close()


def sor_per_month_plot(thread_num):
    """
    Solar Operation Ratio (SOR) per month plot with error bars
    """
    pv_gen_file_path_list = []
    pv_sell_file_path_list = []
    
    # Collect the file paths for all threads
    for i in range(thread_num):
        pv_gen_file_paths = glob.glob(f'output/test/thread{i}/episode*/supply.csv')
        pv_gen_sorted_file_paths = sorted(pv_gen_file_paths, key=numerical_sort)
        pv_gen_file_path_list.append(pv_gen_sorted_file_paths[-1])  # get the last episode
        
        pv_sell_file_paths = glob.glob(f'output/test/thread{i}/episode*/sell_pv_record.csv')
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

    plt.tight_layout()
    plt.savefig('output/insight/sor_per_month.png', dpi=600)
    plt.savefig('output/insight/sor_per_month.svg')
    # plt.show()

    print('SOR per month plot with error bars saved.')


def ssr_per_month_plot(thread_num):
    """
    Self Sufficiency Ratio (SSR) per month plot with error bars
    """
    grid_import_file_path_list = []
    # Collect the file paths for all threads
    for i in range(thread_num):
        grid_import_file_paths = glob.glob(f'output/test/thread{i}/episode*/grid_import_record.csv')
        grid_import_sorted_file_paths = sorted(grid_import_file_paths, key=numerical_sort)
        grid_import_file_path_list.append(grid_import_sorted_file_paths[-1])  # get the last episode

    buy_inelastic_file_path_list = []
    for i in range(thread_num):
        buy_inelastic_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_inelastic_record.csv')
        buy_inelastic_sorted_file_paths = sorted(buy_inelastic_file_paths, key=numerical_sort)
        buy_inelastic_file_path_list.append(buy_inelastic_sorted_file_paths[-1])  # get the last episode
    buy_elastic_file_path_list = []
    for i in range(thread_num):
        buy_elastic_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_elastic_record.csv')
        buy_elastic_sorted_file_paths = sorted(buy_elastic_file_paths, key=numerical_sort)
        buy_elastic_file_path_list.append(buy_elastic_sorted_file_paths[-1])  # get the last episode
    buy_shifted_file_path_list = []
    for i in range(thread_num):
        buy_shifted_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_shifted_record.csv')
        buy_shifted_sorted_file_paths = sorted(buy_shifted_file_paths, key=numerical_sort)
        buy_shifted_file_path_list.append(buy_shifted_sorted_file_paths[-1])  # get the last episode
    buy_battery_file_path_list = []
    for i in range(thread_num):
        buy_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_battery_record.csv')
        buy_battery_sorted_file_paths = sorted(buy_battery_file_paths, key=numerical_sort)
        buy_battery_file_path_list.append(buy_battery_sorted_file_paths[-1])  # get the last episode
    buy_ev_battery_file_path_list = []
    for i in range(thread_num):
        buy_ev_battery_file_paths = glob.glob(f'output/test/thread{i}/episode*/buy_ev_battery_record.csv')
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

    plt.tight_layout()
    plt.savefig('output/insight/ssr_per_month.png', dpi=600)
    plt.savefig('output/insight/ssr_per_month.svg')
    # plt.show()

    print('SSR per month plot with error bars saved.')
    


if __name__ == '__main__':
    # agent_num = pd.read_csv('output/thread0/episode1/agent_params.csv', index_col=0).shape[0]
    agent_num = pd.read_csv('output/test/thread0/episode10/agent_params.csv', index_col=0).shape[0]
    print(f'Detected number of agents: {agent_num}')

    reward_sorted_file_paths_list = []
    for i in range(max_workers):
        # reward_file_paths = glob.glob(f'output/thread{i}/episode*/reward.csv')
        reward_file_paths = glob.glob(f'output/test/thread{i}/episode*/reward.csv') 
        reward_sorted_file_paths = sorted(reward_file_paths, key=numerical_sort)
        reward_sorted_file_paths_list.append(reward_sorted_file_paths)
    # print(reward_sorted_file_paths_list)
    if os.path.exists('output/insight/reward_history_powerplot.png'):
        print('Reward history powerplot already exists. Skip plotting.')
    else:
        reward_history_plot_4_4_powerplot(reward_sorted_file_paths_list, agent_num=agent_num)

    if os.path.exists('output/insight/reward_history.png'):
        print('Reward history already exists. Skip plotting.')
    else:
        reward_history_plot_4_4(reward_sorted_file_paths_list, agent_num=agent_num)

# ==================================================================================================
    buy_amount_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers)
    buy_cost_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers)
    sell_amount_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers)
    sell_cost_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers)
    # net_cost_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers)
    net_cost_by_battery_ev_pv_dr_exist_plot_2(thread_num=max_workers)

# ==================================================================================================
    sor_per_month_plot(thread_num=max_workers)
    ssr_per_month_plot(thread_num=max_workers)
    