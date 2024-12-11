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


def buy_amount_by_battery_ev_pv_dr_exist_plot(thread_num, folder_path):
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

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/agent_params.csv')
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
                    rotation=45)
    ax.set_ylabel('Energy Amount [kWh]')
    ax.set_title('Average Energy Amount Buy Composition')
    ax.legend()

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/buy_amount_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig(folder_path + '/insight/buy_amount_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Energy amount buy composition plot saved.')


def buy_cost_by_battery_ev_pv_dr_exist_plot(thread_num, folder_path):
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
                buy_dict['w/battery_w/ev_w/pv_w/dr']['battery_capex'].append(capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_w/ev_w/pv_w/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['battery_capex'].append(capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_w/ev_w/pv_wo/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['pv_capex'].append(0)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['battery_capex'].append(capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_w/ev_wo/pv_w/dr']['pv_opex'].append(0)
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['pv_capex'].append(0)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['battery_capex'].append(capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_w/ev_wo/pv_wo/dr']['pv_opex'].append(0)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['battery_capex'].append(capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_wo/ev_w/pv_w/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['pv_capex'].append(capex_opex.pv_capex_func(pv_capacity) / pv_lifetime)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['battery_capex'].append(capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_wo/ev_w/pv_wo/dr']['pv_opex'].append(capex_opex.pv_opex_func(pv_capacity))
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['pv_capex'].append(0)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['battery_capex'].append(capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                buy_dict['w/battery_wo/ev_wo/pv_w/dr']['pv_opex'].append(0)
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_inelastic'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_elastic'].append((buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_shifted'].append((buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_battery'].append((buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['buy_ev_battery'].append((buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['pv_capex'].append(0)
                buy_dict['w/battery_wo/ev_wo/pv_wo/dr']['battery_capex'].append(capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
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
                    rotation=45)
    ax.set_ylabel('Energy Cost [$]')
    ax.set_title('Average Energy Cost Buy Composition')
    ax.legend()

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/buy_cost_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig(folder_path + '/insight/buy_cost_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Energy cost buy composition plot saved.')


def sell_amount_by_battery_ev_pv_dr_exist_plot(thread_num, folder_path):
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
                    rotation=45)
    ax.set_ylabel('Energy Amount [kWh]')
    ax.set_title('Average Energy Amount Sell Composition')
    ax.legend()

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/sell_amount_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig(folder_path + '/insight/sell_amount_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Energy amount sell composition plot saved.')


def sell_cost_by_battery_ev_pv_dr_exist_plot(thread_num, folder_path):
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
                    rotation=45)
    ax.set_ylabel('Energy Cost [$]')
    ax.set_title('Average Energy Cost Sell Composition')
    ax.legend()

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/sell_cost_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig(folder_path + '/insight/sell_cost_by_battery_ev_pv_dr.svg')
    # plt.show()
    print('Energy cost sell composition plot saved.')


def net_cost_by_battery_ev_pv_dr_exist_plot(thread_num, folder_path):
    net_cost_file_path_list = []
    for i in range(thread_num):
        # Change it later
        net_cost_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/net_electricity_cost.csv')
        net_cost_sorted_file_paths = sorted(net_cost_file_paths, key=numerical_sort)
        net_cost_file_path_list.append(net_cost_sorted_file_paths[-1])  # get the last episode

    agent_params_file_path_list = []
    for i in range(thread_num):
        agent_params_file_paths = glob.glob(folder_path + f'/test/thread{i}/episode*/agent_params.csv')
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
    med_costs = [np.median(net_dict[key]) for key in net_dict.keys()]

    # Plotting
    categories = ['w/battery_w/ev_w/pv_w/dr', 'w/battery_w/ev_w/pv_wo/dr', 'w/battery_w/ev_wo/pv_w/dr', 'w/battery_w/ev_wo/pv_wo/dr',
                    'w/battery_wo/ev_w/pv_w/dr', 'w/battery_wo/ev_w/pv_wo/dr', 'w/battery_wo/ev_wo/pv_w/dr', 'w/battery_wo/ev_wo/pv_wo/dr',
                    'wo/battery_w/ev_w/pv_w/dr', 'wo/battery_w/ev_w/pv_wo/dr', 'wo/battery_w/ev_wo/pv_w/dr', 'wo/battery_w/ev_wo/pv_wo/dr',
                    'wo/battery_wo/ev_w/pv_w/dr', 'wo/battery_wo/ev_w/pv_wo/dr', 'wo/battery_wo/ev_wo/pv_w/dr', 'wo/battery_wo/ev_wo/pv_wo/dr']

    fig, ax = plt.subplots(figsize=(12, 8))

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
                    rotation=45)
    ax.set_ylabel('Net Electricity Cost [$]')
    ax.set_title('Net Electricity Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()

    plt.savefig(folder_path + '/insight/net_cost_by_battery_ev_pv_dr_misstake.png', dpi=600)
    plt.savefig(folder_path + '/insight/net_cost_by_battery_ev_pv_dr_misstake.svg')

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, med_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}\nMed: ${med_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(folder_path + '/insight/net_cost_by_battery_ev_pv_dr_misstake_with_values.png', dpi=600)
    plt.savefig(folder_path + '/insight/net_cost_by_battery_ev_pv_dr_misstake_with_values.svg')
    # plt.show()

    print('Net cost by battery, EV, PV, and DR plot saved. Maybe mistake in the data.')


def net_cost_by_battery_ev_pv_dr_exist_plot_2(thread_num, folder_path, include_capex_opex=True):
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

    net_dict = {
        'w/battery_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []}
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
                if include_capex_opex:
                    net_dict['w/battery_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
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
                    net_dict['w/battery_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                if include_capex_opex:
                    net_dict['w/battery_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
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
                    net_dict['w/battery_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                if include_capex_opex:
                    net_dict['w/battery_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                else:
                    net_dict['w/battery_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                if include_capex_opex:
                    net_dict['w/battery_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + capex_opex.battery_capex_func(battery_capacity, pv_capacity) / bes_lifetime)
                else:
                    net_dict['w/battery_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                if include_capex_opex:
                    net_dict['wo/battery_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                                + capex_opex.pv_opex_func(pv_capacity))
                else:
                    net_dict['wo/battery_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                if include_capex_opex:
                    net_dict['wo/battery_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + capex_opex.pv_capex_func(pv_capacity) / pv_lifetime
                                                                + capex_opex.pv_opex_func(pv_capacity))
                else:
                    net_dict['wo/battery_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                if include_capex_opex:
                    net_dict['wo/battery_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                else:
                    net_dict['wo/battery_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                if include_capex_opex:
                    net_dict['wo/battery_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                else:
                    net_dict['wo/battery_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                                - (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            
            if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_w/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
                                                                            
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_w/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_wo/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_wo/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_w/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_w/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_wo/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            - sell_pv.loc[:, f'{j}'].sum()
                                                            - sell_battery.loc[:, f'{j}'].sum()
                                                            - sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_wo/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
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
    med_costs = [np.median(net_dict[key]['cost']) for key in net_dict.keys()]

    # Plotting
    categories = ['w/battery_w/pv_w/dr', 'w/battery_w/pv_wo/dr', 'w/battery_wo/pv_w/dr', 'w/battery_wo/pv_wo/dr',
                    'wo/battery_w/pv_w/dr', 'wo/battery_w/pv_wo/dr', 'wo/battery_wo/pv_w/dr', 'wo/battery_wo/pv_wo/dr']

    fig, ax = plt.subplots(figsize=(12, 8))

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
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                    rotation=45)
    ax.set_ylabel('Net Electricity Cost [$]')
    if include_capex_opex:
        ax.set_title('Net Electricity Cost Distribution (Including CAPEX and OPEX)')
    else:
        ax.set_title('Net Electricity Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    # ax.set_ylim(-4000, 14000)
    plt.tight_layout()

    if include_capex_opex:
        plt.savefig(folder_path + '/insight/net_cost_by_battery_pv_dr_include_capex_opex.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_cost_by_battery_pv_dr_include_capex_opex.svg')
    else:
        plt.savefig(folder_path + '/insight/net_cost_by_battery_pv_dr.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_cost_by_battery_pv_dr.svg')

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, med_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}\nMed: ${med_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    if include_capex_opex:
        plt.savefig(folder_path + '/insight/net_cost_by_battery_pv_dr_include_capex_opex_with_values.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_cost_by_battery_pv_dr_include_capex_opex_with_values.svg')
    else:
        plt.savefig(folder_path + '/insight/net_cost_by_battery_pv_dr_with_values.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_cost_by_battery_pv_dr_with_values.svg')
    # plt.show()

    print('Net cost by battery, PV, and DR plot saved.')
    plt.close()

    # Calculate cost per kWh for each category
    for key in net_dict.keys():
        for i in range(len(net_dict[key]['cost'])):
            net_dict[key]['cost/kWh'].append(net_dict[key]['cost'][i]/net_dict[key]['amount'][i])
    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]['cost/kWh']) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]['cost/kWh']) for key in net_dict.keys()]
    med_costs = [np.median(net_dict[key]['cost/kWh']) for key in net_dict.keys()]

    fig, ax = plt.subplots(figsize=(12, 8))

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
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                    rotation=45)
    ax.set_ylabel('Net Electricity Unit Cost [$/kWh]')
    if include_capex_opex:
        ax.set_title('Net Electricity Unit Cost Distribution (Including CAPEX and OPEX)')
    else:
        ax.set_title('Net Electricity Unit Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(-10, 10)

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, med_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}\nMed: ${med_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    if include_capex_opex:
        plt.savefig(folder_path + '/insight/net_cost_per_kWh_by_battery_pv_dr_include_capex_opex.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_cost_per_kWh_by_battery_pv_dr_include_capex_opex.svg')
    else:
        plt.savefig(folder_path + '/insight/net_cost_per_kWh_by_battery_pv_dr.png', dpi=600)
        plt.savefig(folder_path + '/insight/net_cost_per_kWh_by_battery_pv_dr.svg')
    # plt.show()

    print('Net unit cost by battery, PV, and DR plot saved.')
    plt.close()


def net_cost_by_battery_ev_pv_dr_exist_plot_3(thread_num, folder_path, include_capex_opex=True):
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
            - sell_pv.loc[:, f'{j}'].sum()
            - sell_battery.loc[:, f'{j}'].sum()
            - sell_ev_battery.loc[:, f'{j}'].sum())

            cost_per_kWh = cost / amount
            print(f'cost: {cost}, amount: {amount}, cost_per_kWh: {cost_per_kWh}')

            master_list.append({'battery_capacity': battery_capacity, 'ev_capacity': ev_capacity, 'pv_capacity': pv_capacity, 'dr_boolean': dr_boolean,
                                'cost': cost, 'amount': amount, 'cost_per_kWh': cost_per_kWh})

    master_df = pd.DataFrame(master_list)
    # Plotting
    categories = ['w/battery_w/pv_w/dr', 'w/battery_w/pv_wo/dr', 'w/battery_wo/pv_w/dr', 'w/battery_wo/pv_wo/dr',
                    'wo/battery_w/pv_w/dr', 'wo/battery_w/pv_wo/dr', 'wo/battery_wo/pv_w/dr', 'wo/battery_wo/pv_wo/dr']

    net_dict = {
        'w/battery_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []}
    }

    for i in range(master_df.shape[0]):
        battery_capacity = master_df.loc[i, 'battery_capacity']
        ev_capacity = master_df.loc[i, 'ev_capacity']
        pv_capacity = master_df.loc[i, 'pv_capacity']
        dr_boolean = master_df.loc[i, 'dr_boolean']
        cost = master_df.loc[i, 'cost']
        amount = master_df.loc[i, 'amount']
        cost_per_kWh = master_df.loc[i, 'cost_per_kWh']

        if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
            net_dict['w/battery_w/pv_w/dr']['amount'].append(amount)
            net_dict['w/battery_w/pv_w/dr']['cost'].append(cost)
            net_dict['w/battery_w/pv_w/dr']['cost/kWh'].append(cost_per_kWh)
        elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
            net_dict['w/battery_w/pv_wo/dr']['amount'].append(amount)
            net_dict['w/battery_w/pv_wo/dr']['cost'].append(cost)
            net_dict['w/battery_w/pv_wo/dr']['cost/kWh'].append(cost_per_kWh)
        elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
            net_dict['w/battery_wo/pv_w/dr']['amount'].append(amount)
            net_dict['w/battery_wo/pv_w/dr']['cost'].append(cost)
            net_dict['w/battery_wo/pv_w/dr']['cost/kWh'].append(cost_per_kWh)
        elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
            net_dict['w/battery_wo/pv_wo/dr']['amount'].append(amount)
            net_dict['w/battery_wo/pv_wo/dr']['cost'].append(cost)
            net_dict['w/battery_wo/pv_wo/dr']['cost/kWh'].append(cost_per_kWh)
        elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
            net_dict['wo/battery_w/pv_w/dr']['amount'].append(amount)
            net_dict['wo/battery_w/pv_w/dr']['cost'].append(cost)
            net_dict['wo/battery_w/pv_w/dr']['cost/kWh'].append(cost_per_kWh)
        elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
            net_dict['wo/battery_w/pv_wo/dr']['amount'].append(amount)
            net_dict['wo/battery_w/pv_wo/dr']['cost'].append(cost)
            net_dict['wo/battery_w/pv_wo/dr']['cost/kWh'].append(cost_per_kWh)
        elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
            net_dict['wo/battery_wo/pv_w/dr']['amount'].append(amount)
            net_dict['wo/battery_wo/pv_w/dr']['cost'].append(cost)
            net_dict['wo/battery_wo/pv_w/dr']['cost/kWh'].append(cost_per_kWh)
        elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
            net_dict['wo/battery_wo/pv_wo/dr']['amount'].append(amount)
            net_dict['wo/battery_wo/pv_wo/dr']['cost'].append(cost)
            net_dict['wo/battery_wo/pv_wo/dr']['cost/kWh'].append(cost_per_kWh)

    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]['cost']) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]['cost']) for key in net_dict.keys()]
    med_costs = [np.median(net_dict[key]['cost']) for key in net_dict.keys()]

    # Plotting
    categories = ['w/battery_w/pv_w/dr', 'w/battery_w/pv_wo/dr', 'w/battery_wo/pv_w/dr', 'w/battery_wo/pv_wo/dr',
                    'wo/battery_w/pv_w/dr', 'wo/battery_w/pv_wo/dr', 'wo/battery_wo/pv_w/dr', 'wo/battery_wo/pv_wo/dr']

    fig, ax = plt.subplots(figsize=(12, 8))

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
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                    rotation=45)
    ax.set_ylabel('Net Electricity Cost [$]')
    if include_capex_opex:
        ax.set_title('Net Electricity Cost Distribution (Including CAPEX and OPEX)')
    else:
        ax.set_title('Net Electricity Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    # ax.set_ylim(-4000, 14000)
    plt.tight_layout()

    if include_capex_opex:
        plt.savefig(folder_path + '/insight/test_net_cost_by_battery_pv_dr_include_capex_opex.png', dpi=600)
        plt.savefig(folder_path + '/insight/test_net_cost_by_battery_pv_dr_include_capex_opex.svg')
    else:
        plt.savefig(folder_path + '/insight/test_net_cost_by_battery_pv_dr.png', dpi=600)
        plt.savefig(folder_path + '/insight/test_net_cost_by_battery_pv_dr.svg')

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, med_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}\nMed: ${med_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    if include_capex_opex:
        plt.savefig(folder_path + '/insight/test_net_cost_by_battery_pv_dr_include_capex_opex_with_values.png', dpi=600)
        plt.savefig(folder_path + '/insight/test_net_cost_by_battery_pv_dr_include_capex_opex_with_values.svg')
    else:
        plt.savefig(folder_path + '/insight/test_net_cost_by_battery_pv_dr_with_values.png', dpi=600)
        plt.savefig(folder_path + '/insight/test_net_cost_by_battery_pv_dr_with_values.svg')
    # plt.show()

    print('Net cost by battery, PV, and DR plot saved.')
    plt.close()

    # Calculate cost per kWh for each category
    for key in net_dict.keys():
        for i in range(len(net_dict[key]['cost'])):
            net_dict[key]['cost/kWh'].append(net_dict[key]['cost'][i]/net_dict[key]['amount'][i])
    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]['cost/kWh']) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]['cost/kWh']) for key in net_dict.keys()]
    med_costs = [np.median(net_dict[key]['cost/kWh']) for key in net_dict.keys()]

    fig, ax = plt.subplots(figsize=(12, 8))

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
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                    rotation=45)
    ax.set_ylabel('Net Electricity Unit Cost [$/kWh]')
    if include_capex_opex:
        ax.set_title('Net Electricity Unit Cost Distribution (Including CAPEX and OPEX)')
    else:
        ax.set_title('Net Electricity Unit Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(-10, 10)

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, med_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}\nMed: ${med_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    if include_capex_opex:
        plt.savefig(folder_path + '/insight/test_net_cost_per_kWh_by_battery_pv_dr_include_capex_opex.png', dpi=600)
        plt.savefig(folder_path + '/insight/test_net_cost_per_kWh_by_battery_pv_dr_include_capex_opex.svg')
    else:
        plt.savefig(folder_path + '/insight/test_net_cost_per_kWh_by_battery_pv_dr.png', dpi=600)
        plt.savefig(folder_path + '/insight/test_net_cost_per_kWh_by_battery_pv_dr.svg')
    # plt.show()

    print('Net unit cost by battery, PV, and DR plot saved.')
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
            + buy_battery.loc[:, f'{j}'].sum()
            + buy_ev_battery.loc[:, f'{j}'].sum()
            - sell_pv.loc[:, f'{j}'].sum()
            - sell_battery.loc[:, f'{j}'].sum()
            - sell_ev_battery.loc[:, f'{j}'].sum())

            cost_per_kWh = cost / amount
            print(f'cost: {cost}, amount: {amount}, cost_per_kWh: {cost_per_kWh}')

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
    print(heat_map_net_total_cost_df)
    print(heat_map_net_cost_per_kWh_df)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heat_map_net_total_cost_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, cbar_kws={'label': 'Net Total Cost [$]'})
    ax.set_title('Net Total Cost Heatmap by Battery and PV Capacity')
    ax.set_xlabel('PV Capacity [kW]')
    ax.set_ylabel('Battery Capacity [kWh]')
    plt.tight_layout()
    plt.savefig(folder_path + '/insight/net_total_cost_heatmap_by_battery_pv_include_capex_opex.png', dpi=600)
    plt.savefig(folder_path + '/insight/net_total_cost_heatmap_by_battery_pv_include_capex_opex.svg')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heat_map_net_cost_per_kWh_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, cbar_kws={'label': 'Net Cost per kWh [$]'})
    ax.set_title('Net Cost per kWh Heatmap by Battery and PV Capacity')
    ax.set_xlabel('PV Capacity [kW]')
    ax.set_ylabel('Battery Capacity [kWh]')
    plt.tight_layout()
    plt.savefig(folder_path + '/insight/net_cost_per_kWh_heatmap_by_battery_pv_include_capex_opex.png', dpi=600)
    plt.savefig(folder_path + '/insight/net_cost_per_kWh_heatmap_by_battery_pv_include_capex_opex.svg')

    print('Net cost by battery, PV capacity heatmap saved.')

    # plot how many agents are in each category, using heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    heat_map_agent_count_df = pd.DataFrame(index=battery_capacity_list, columns=pv_capacity_list)
    for battery_capacity in battery_capacity_list:
        for pv_capacity in pv_capacity_list:
            heat_map_agent_count_df.loc[battery_capacity, pv_capacity] = master_df[(master_df['battery_capacity'] == battery_capacity) & (master_df['pv_capacity'] == pv_capacity)].shape[0]
    heat_map_agent_count_df = heat_map_agent_count_df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(heat_map_agent_count_df, annot=True, fmt=".0f", cmap="coolwarm", cbar=True, cbar_kws={'label': 'Prosumer Count'})
    ax.set_title('Prosumer Count Heatmap by Battery and PV Capacity')
    ax.set_xlabel('PV Capacity [kW]')
    ax.set_ylabel('Battery Capacity [kWh]')
    plt.tight_layout()
    plt.savefig(folder_path + '/insight/prosumer_count_heatmap_by_battery_pv.png', dpi=600)
    plt.savefig(folder_path + '/insight/prosumer_count_heatmap_by_battery_pv.svg')
    print('Prosumer count by battery, PV capacity heatmap saved.')


def buy_cost_per_kwh_by_battery_ev_pv_dr_exist_plot(thread_num, folder_path):
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

    net_dict = {
        'w/battery_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'w/battery_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_w/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/pv_w/dr': {'amount': [], 'cost': [], 'cost/kWh': []},
        'wo/battery_wo/pv_wo/dr': {'amount': [], 'cost': [], 'cost/kWh': []}
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
            if battery_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_w/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_w/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_wo/pv_w/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_wo/pv_wo/dr']['cost'].append((buy_inelastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_elastic.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_shifted.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (buy_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
                
            if battery_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_w/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            )
                                                                            
            elif battery_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_w/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            )
            elif battery_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_wo/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            )
            elif battery_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_wo/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            )
            elif battery_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_w/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            )
            elif battery_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_w/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            )
            elif battery_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_wo/pv_w/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            )
            elif battery_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_wo/pv_wo/dr']['amount'].append(buy_inelastic.loc[:, f'{j}'].sum()
                                                            + buy_elastic.loc[:, f'{j}'].sum()
                                                            + buy_shifted.loc[:, f'{j}'].sum()
                                                            + buy_battery.loc[:, f'{j}'].sum()
                                                            + buy_ev_battery.loc[:, f'{j}'].sum()
                                                            )

                
    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]['cost']) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]['cost']) for key in net_dict.keys()]
    med_costs = [np.median(net_dict[key]['cost']) for key in net_dict.keys()]

    # Plotting
    categories = ['w/battery_w/pv_w/dr', 'w/battery_w/pv_wo/dr', 'w/battery_wo/pv_w/dr', 'w/battery_wo/pv_wo/dr',
                    'wo/battery_w/pv_w/dr', 'wo/battery_w/pv_wo/dr', 'wo/battery_wo/pv_w/dr', 'wo/battery_wo/pv_wo/dr']

    fig, ax = plt.subplots(figsize=(12, 8))

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
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                    rotation=45)
    ax.set_ylabel('Buy Electricity Cost [$]')
    ax.set_title('Buy Electricity Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, med_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}\nMed: ${med_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/buy_cost_by_battery_pv_dr_2.png', dpi=600)
    plt.savefig(folder_path + '/insight/buy_cost_by_battery_pv_dr_2.svg')
    # plt.show()

    print('Net cost by battery, PV, and DR plot saved.')
    plt.close()

    # Calculate cost per kWh for each category
    for key in net_dict.keys():
        for i in range(len(net_dict[key]['cost'])):
            net_dict[key]['cost/kWh'].append(net_dict[key]['cost'][i]/net_dict[key]['amount'][i])
    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]['cost/kWh']) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]['cost/kWh']) for key in net_dict.keys()]

    fig, ax = plt.subplots(figsize=(12, 8))

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
    ax.set_xticklabels(['w/ BES, w/ PV, w/ DR', 'w/ BES, w/ PV, w/o DR', 'w/ BES, w/o PV, w/ DR', 'w/ BES, w/o PV, w/o DR',
                        'w/o BES, w/ PV, w/ DR', 'w/o BES, w/ PV, w/o DR', 'w/o BES, w/o PV, w/ DR', 'w/o BES, w/o PV, w/o DR'],
                    rotation=45)
    ax.set_ylabel('Buy Electricity Unit Cost [$/kWh]')
    ax.set_title('Buy Electricity Unit Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    # ax.set_ylim(-10, 10)
    plt.tight_layout()

    plt.savefig(folder_path + '/insight/buy_cost_per_kWh_by_battery_pv_dr.png', dpi=600)
    plt.savefig(folder_path + '/insight/buy_cost_per_kWh_by_battery_pv_dr.svg')

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, mean_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(folder_path + '/insight/buy_cost_per_kWh_by_battery_pv_dr_with_values.png', dpi=600)
    plt.savefig(folder_path + '/insight/buy_cost_per_kWh_by_battery_pv_dr_with_values.svg')
    # plt.show()

    print('Net unit cost by battery, PV, and DR plot saved.')
    plt.close()


def sell_cost_per_kwh_by_battery_ev_pv_dr_exist_plot(thread_num, folder_path):
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
                net_dict['w/battery_w/ev_w/pv_w/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_w/ev_w/pv_wo/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_w/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_wo/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_w/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_wo/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_w/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_wo/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_w/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_wo/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_w/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_wo/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_w/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_wo/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            )
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_w/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_wo/dr']['cost'].append((sell_pv.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100
                                                            + (sell_ev_battery.loc[:, f'{j}']*microgrid_price.loc[:, 'Price']).sum()/100)
                
            if battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_w/ev_w/pv_w/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
                                                                            
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_w/ev_w/pv_wo/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_w/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_w/ev_wo/pv_wo/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_w/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_w/pv_wo/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_w/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity > 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['w/battery_wo/ev_wo/pv_wo/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_w/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_w/pv_wo/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_w/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity > 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_w/ev_wo/pv_wo/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_w/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity > 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_w/pv_wo/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_w/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
            elif battery_capacity == 0 and ev_capacity == 0 and pv_capacity == 0 and not dr_boolean:
                net_dict['wo/battery_wo/ev_wo/pv_wo/dr']['amount'].append(sell_pv.loc[:, f'{j}'].sum()
                                                            + sell_battery.loc[:, f'{j}'].sum()
                                                            + sell_ev_battery.loc[:, f'{j}'].sum())
                
    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]['cost']) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]['cost']) for key in net_dict.keys()]
    med_costs = [np.median(net_dict[key]['cost']) for key in net_dict.keys()]

    # Plotting
    categories = ['w/battery_w/ev_w/pv_w/dr', 'w/battery_w/ev_w/pv_wo/dr', 'w/battery_w/ev_wo/pv_w/dr', 'w/battery_w/ev_wo/pv_wo/dr',
                    'w/battery_wo/ev_w/pv_w/dr', 'w/battery_wo/ev_w/pv_wo/dr', 'w/battery_wo/ev_wo/pv_w/dr', 'w/battery_wo/ev_wo/pv_wo/dr',
                    'wo/battery_w/ev_w/pv_w/dr', 'wo/battery_w/ev_w/pv_wo/dr', 'wo/battery_w/ev_wo/pv_w/dr', 'wo/battery_w/ev_wo/pv_wo/dr',
                    'wo/battery_wo/ev_w/pv_w/dr', 'wo/battery_wo/ev_w/pv_wo/dr', 'wo/battery_wo/ev_wo/pv_w/dr', 'wo/battery_wo/ev_wo/pv_wo/dr']

    fig, ax = plt.subplots(figsize=(12, 8))

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
                    rotation=45)
    ax.set_ylabel('Net Electricity Cost [$]')
    ax.set_title('Net Electricity Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, med_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}\nMed: ${med_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/sell_cost_by_battery_ev_pv_dr_2.png', dpi=600)
    plt.savefig(folder_path + '/insight/sell_cost_by_battery_ev_pv_dr_2.svg')
    # plt.show()

    print('Net cost by battery, EV, PV, and DR plot saved.')
    plt.close()

    # Calculate cost per kWh for each category
    for key in net_dict.keys():
        for i in range(len(net_dict[key]['cost'])):
            if net_dict[key]['amount'][i] == 0:
                net_dict[key]['cost/kWh'].append(0)
            else:
                net_dict[key]['cost/kWh'].append(net_dict[key]['cost'][i]/net_dict[key]['amount'][i])
    # Calculate mean and standard deviation of costs for each category
    mean_costs = [np.mean(net_dict[key]['cost/kWh']) for key in net_dict.keys()]
    std_costs = [np.std(net_dict[key]['cost/kWh']) for key in net_dict.keys()]

    fig, ax = plt.subplots(figsize=(12, 8))

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
                    rotation=45)
    ax.set_ylabel('Net Electricity Unit Cost [$/kWh]')
    ax.set_title('Net Electricity Unit Cost Distribution')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    # ax.set_ylim(-10, 10)

    # Adding mean and standard deviation text
    for i in range(len(categories)):
        ax.text(i + 1, mean_costs[i], f'Mean: ${mean_costs[i]:.2f}\nStd: ${std_costs[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(folder_path + '/insight/sell_cost_per_kWh_by_battery_ev_pv_dr.png', dpi=600)
    plt.savefig(folder_path + '/insight/sell_cost_per_kWh_by_battery_ev_pv_dr.svg')
    # plt.show()

    print('Net unit cost by battery, EV, PV, and DR plot saved.')
    plt.close()



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

    print('SOR per month plot with error bars saved.')


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

    print('SSR per month plot with error bars saved.')


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
    plt.tight_layout()
    plt.savefig(folder_path + '/insight/surplus_demand_by_hour.png', dpi=600)
    plt.savefig(folder_path + '/insight/surplus_demand_by_hour.svg')
    
    for i in range(24):
        ax.text(i + 1, demand_median_by_hour[i], f'Mean: {demand_mean_by_hour[i]:.2f}\nStd: {demand_std_by_hour[i]:.2f}\nMed: {demand_median_by_hour[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(folder_path + '/insight/surplus_demand_by_hour_with_values.png', dpi=600)
    plt.savefig(folder_path + '/insight/surplus_demand_by_hour_with_values.svg')
    # plt.show()
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
    plt.tight_layout()
    plt.savefig(folder_path + '/insight/surplus_supply_by_hour.png', dpi=600)
    plt.savefig(folder_path + '/insight/surplus_supply_by_hour.svg')

    for i in range(24):
        ax.text(i + 1, supply_median_by_hour[i], f'Mean: {supply_mean_by_hour[i]:.2f}\nStd: {supply_std_by_hour[i]:.2f}\nMed: {supply_median_by_hour[i]:.2f}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(folder_path + '/insight/surplus_supply_by_hour_with_values.png', dpi=600)
    plt.savefig(folder_path + '/insight/surplus_supply_by_hour_with_values.svg')
    # plt.show()
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


def net_consumption_vs_bes_pv_size_scatter(thread_num, folder_path):
    """
    Net consumption vs BES and PV size scatter plot
    """
    # Collect the file paths for all threads
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


    net_consumption_list = []
    bes_capacity_list = []
    pv_capacity_list = []
    for i in range(len(agent_params_file_path_list)):
        buy_inelastic = pd.read_csv(buy_inelastic_file_path_list[i], index_col=0)
        buy_elastic = pd.read_csv(buy_elastic_file_path_list[i], index_col=0)
        buy_shifted = pd.read_csv(buy_shifted_file_path_list[i], index_col=0)
        buy_battery = pd.read_csv(buy_battery_file_path_list[i], index_col=0)
        buy_ev_battery = pd.read_csv(buy_ev_battery_file_path_list[i], index_col=0)
        
        sell_pv = pd.read_csv(sell_pv_file_path_list[i], index_col=0)
        sell_battery = pd.read_csv(sell_battery_file_path_list[i], index_col=0)
        sell_ev_battery = pd.read_csv(sell_ev_battery_file_path_list[i], index_col=0)

        agent_params_df = pd.read_csv(agent_params_file_path_list[i], index_col=0)
        net_consumptions = (buy_inelastic.sum(axis=0)+buy_elastic.sum(axis=0)+buy_shifted.sum(axis=0)+buy_battery.sum(axis=0)+buy_ev_battery.sum(axis=0) -
                           sell_pv.sum(axis=0)-sell_battery.sum(axis=0)-sell_ev_battery.sum(axis=0))
        bes_capacities = agent_params_df['battery_capacity']
        pv_capacities = agent_params_df['pv_capacity']
        net_consumption_list.extend(net_consumptions)
        bes_capacity_list.extend(bes_capacities)
        pv_capacity_list.extend(pv_capacities)

    fig, ax = plt.subplots(figsize=(20, 15))
    ax.scatter(bes_capacity_list, pv_capacity_list, c=net_consumption_list, cmap='coolwarm', s=100, alpha=0.7)
    ax.set_xlabel('BES Capacity [kWh]')
    ax.set_ylabel('PV Capacity [kW]')
    ax.set_title('Net Consumption vs BES and PV Size')
    cbar = plt.colorbar(ax.scatter(bes_capacity_list, pv_capacity_list, c=net_consumption_list, cmap='coolwarm', s=100, alpha=0.7))
    cbar.set_label('Net Consumption [kWh]')
    plt.tight_layout()
    plt.savefig(folder_path + '/insight/net_consumption_vs_bes_pv_size_scatter.png', dpi=600)
    plt.savefig(folder_path + '/insight/net_consumption_vs_bes_pv_size_scatter.svg')
    # plt.show()

    print('Net consumption vs BES and PV size scatter plot saved.')
    plt.close()


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
        buy_amount_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/no_p2p')
        buy_cost_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/no_p2p')
        sell_amount_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/no_p2p')
        sell_cost_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/no_p2p')
        net_cost_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/no_p2p')
        net_cost_by_battery_ev_pv_dr_exist_plot_3(thread_num=max_workers, folder_path='output/no_p2p', include_capex_opex=False)
        net_cost_by_battery_ev_pv_dr_exist_plot_3(thread_num=max_workers, folder_path='output/no_p2p', include_capex_opex=True)
        buy_cost_per_kwh_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/no_p2p')
        sell_cost_per_kwh_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/no_p2p')

    # ==================================================================================================
        sor_per_month_plot(thread_num=max_workers, folder_path='output/no_p2p')
        ssr_per_month_plot(thread_num=max_workers, folder_path='output/no_p2p')
        supply_demand_margin_plot(thread_num=max_workers, folder_path='output/no_p2p')
    # ==================================================================================================
        bes_capacity_avg, pv_capacity_avg = bes_pv_installed_capacity(thread_num=max_workers, folder_path='output/no_p2p')
        print(f'BES installed capacity: {bes_capacity_avg:.2f} kWh')
        print(f'PV installed capacity: {pv_capacity_avg:.2f} kW')
        net_cost_by_battery_ev_pv_size_plot(thread_num=max_workers, folder_path='output/no_p2p', include_capex_opex=True)
        # net_consumption_vs_bes_pv_size_scatter(thread_num=max_workers, folder_path='output/no_p2p')

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
        buy_amount_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/p2p')
        buy_cost_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/p2p')
        sell_amount_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/p2p')
        sell_cost_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/p2p')
        net_cost_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/p2p')
        net_cost_by_battery_ev_pv_dr_exist_plot_3(thread_num=max_workers, folder_path='output/p2p', include_capex_opex=True)
        net_cost_by_battery_ev_pv_dr_exist_plot_3(thread_num=max_workers, folder_path='output/p2p', include_capex_opex=False)
        buy_cost_per_kwh_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/p2p')
        sell_cost_per_kwh_by_battery_ev_pv_dr_exist_plot(thread_num=max_workers, folder_path='output/p2p')

    # ==================================================================================================
        sor_per_month_plot(thread_num=max_workers, folder_path='output/p2p')
        ssr_per_month_plot(thread_num=max_workers, folder_path='output/p2p')
        supply_demand_margin_plot(thread_num=max_workers, folder_path='output/p2p')
        
    # ==================================================================================================
        bes_capacity_avg, pv_capacity_avg = bes_pv_installed_capacity(thread_num=max_workers, folder_path='output/p2p')
        print(f'BES installed capacity: {bes_capacity_avg:.2f} kWh')
        print(f'PV installed capacity: {pv_capacity_avg:.2f} kW')
        net_cost_by_battery_ev_pv_size_plot(thread_num=max_workers, folder_path='output/p2p', include_capex_opex=True)
        # net_consumption_vs_bes_pv_size_scatter(thread_num=max_workers, folder_path='output/p2p')
        