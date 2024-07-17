from agent import Agent
import pandas as pd
import numpy as np
import os


# 観測した状態を離散値にデジタル変換する
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num+1)[0:-1]


class Q:
    def __init__(self, params, agent_num, num_dizitized_pv_ratio, num_dizitized_soc, num_elastic_ratio_pattern):
        self.agent_num = agent_num
        self.possible_params = Agent(self.agent_num).generate_params()
        self.params = params
        self.num_dizitized_pv_ratio = num_dizitized_pv_ratio
        self.num_dizitized_soc = num_dizitized_soc
        self.num_elastic_ratio_pattern = num_elastic_ratio_pattern

        dr_buy_rows = self.num_dizitized_pv_ratio * self.num_elastic_ratio_pattern
        battery_buy_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc
        battery_sell_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc
        ev_battery_buy_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc
        ev_battery_sell_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc
        cols = int(self.params['price_max']) - int(self.params['price_min']) + 1
        # MARLのため、エージェントごとにQテーブルを用意する
        self.dr_buy_qtb_list = [np.full((dr_buy_rows, cols), 100.0) for _ in range(self.agent_num)]
        self.battery_buy_qtb_list = [np.full((battery_buy_rows, cols), 100.0) for _ in range(self.agent_num)]
        self.battery_sell_qtb_list = [np.full((battery_sell_rows, cols), 100.0) for _ in range(self.agent_num)]
        self.ev_battery_buy_qtb_list = [np.full((ev_battery_buy_rows, cols), 100.0) for _ in range(self.agent_num)]
        self.ev_battery_sell_qtb_list = [np.full((ev_battery_sell_rows, cols), 100.0) for _ in range(self.agent_num)]

    @property
    def get_qtbs_(self):
        return self.dr_buy_qtb_list, self.battery_buy_qtb_list, self.battery_sell_qtb_list, self.ev_battery_buy_qtb_list, self.ev_battery_sell_qtb_list

    def get_agent_qtbs(self, agent_id):
        return self.dr_buy_qtb_list[agent_id], self.battery_buy_qtb_list[agent_id], self.battery_sell_qtb_list[agent_id], self.ev_battery_buy_qtb_list[agent_id], self.ev_battery_sell_qtb_list[agent_id]
    
    def reset_all_digitized_states(self):
        self.dr_states = np.full(self.agent_num, np.nan)
        self.battery_states = np.full(self.agent_num, np.nan)
        self.ev_battery_states = np.full(self.agent_num, np.nan)
    
    def set_digitized_states(self, agent_id, pv_ratio, battery_soc, ev_battery_soc, elastic_ratio):
        """
        agent_idごとに離散化した状態(番号)を格納していく
        """
        digitized_pv_ratio = np.digitize(pv_ratio, bins=bins(0, 1, self.num_dizitized_pv_ratio))-1
        digitized_battery_soc = np.digitize(battery_soc, bins=bins(0, 1, self.num_dizitized_soc))-1
        digitized_ev_battery_soc = np.digitize(ev_battery_soc, bins=bins(0, 1, self.num_dizitized_soc))-1
        digitized_elastic_ratio = np.digitize(elastic_ratio, bins=bins(0.1, 0.5, self.num_elastic_ratio_pattern))-1
        # print(digitized_pv_ratio)
        # print(digitized_battery_soc)
        # print(digitized_ev_battery_soc)
        # print(digitized_elastic_ratio)

        dr_state = (digitized_pv_ratio + 
                    digitized_elastic_ratio * self.num_dizitized_pv_ratio)
            
        battery_state = (digitized_pv_ratio +
                         digitized_battery_soc * self.num_dizitized_soc)
                                
        ev_battery_state = (digitized_pv_ratio +
                            digitized_ev_battery_soc * self.num_dizitized_soc)
                         
        self.dr_states[agent_id] = dr_state
        self.battery_states[agent_id] = battery_state
        self.ev_battery_states[agent_id] = ev_battery_state
        return dr_state, battery_state, ev_battery_state
    
    @property
    def get_states_(self):
        return self.dr_states, self.battery_states, self.ev_battery_states
    
    def get_agent_states(self, agent_id):
        return self.dr_states[agent_id], self.battery_states[agent_id], self.ev_battery_states[agent_id]
    
    def reset_all_actions(self):
        self.next_actions = np.full((self.agent_num, 5), np.nan)
    
    def set_actions(self, agent_id, episode):
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.5 * (1 / (episode + 1))
        next_action_list = []
        if epsilon <= np.random.uniform(0, 1):
            next_action_list.append(np.argmax(self.dr_buy_qtb_list[agent_id][int(self.dr_states[agent_id])]) + int(self.params['price_min']))
            next_action_list.append(np.argmax(self.battery_buy_qtb_list[agent_id][int(self.battery_states[agent_id])]) + int(self.params['price_min']))
            next_action_list.append(np.argmax(self.battery_sell_qtb_list[agent_id][int(self.battery_states[agent_id])]) + int(self.params['price_min']))
            next_action_list.append(np.argmax(self.ev_battery_buy_qtb_list[agent_id][int(self.ev_battery_states[agent_id])]) + int(self.params['price_min']))
            next_action_list.append(np.argmax(self.ev_battery_sell_qtb_list[agent_id][int(self.ev_battery_states[agent_id])]) + int(self.params['price_min']))
        else:
            for i in range(5):
                next_action_list.append(np.random.choice(
                    range(int(self.params['price_min']), int(self.params['price_max']) + 1)
                    ))
        self.next_actions[agent_id] = next_action_list
        return next_action_list
    
    @property
    def get_actions_(self):
        """
        エージェントiの行動がi行目
        カラムにはdr_buy, battery_buy, battery_sell, ev_battery_buy, ev_battery_sellの順で格納されている
        """
        return self.next_actions
    
    def update_q_table(self, agent_id, states, actions, rewards, next_states):
        """
        states, actions, rewards, next_statesは全てリスト
        それぞれのリストについて、0番目にdr_buy、1番目にbattery_buy、2番目にbattery_sell、3番目にev_battery_buy、4番目にev_battery_sellの情報が格納されている
        """
        # if agent_id == 3:
        #     print('states:', states)
        #     print('actions:', actions)
        #     print('rewards:', rewards)
        #     print('next_states:', next_states)
        gamma = 0.99
        alpha = 0.1
        dr_buy_td_error = rewards[0] + gamma * np.max(self.dr_buy_qtb_list[agent_id][next_states[0], :]) - self.dr_buy_qtb_list[agent_id][states[0], 
                                                                                                            int(actions[0]-self.params['price_min'])]
        battery_buy_td_error = rewards[1] + gamma * np.max(self.battery_buy_qtb_list[agent_id][next_states[1], :]) - self.battery_buy_qtb_list[agent_id][states[1], 
                                                                                                                           int(actions[1]-self.params['price_min'])]
        battery_sell_td_error = rewards[2] + gamma * np.max(self.battery_sell_qtb_list[agent_id][next_states[2], :]) - self.battery_sell_qtb_list[agent_id][states[2], 
                                                                                                                              int(actions[2]-self.params['price_min'])]
        ev_battery_buy_td_error = rewards[3] + gamma * np.max(self.ev_battery_buy_qtb_list[agent_id][next_states[3], :]) - self.ev_battery_buy_qtb_list[agent_id][states[3], 
                                                                                                                                    int(actions[3]-self.params['price_min'])]
        ev_battery_sell_td_error = rewards[4] + gamma * np.max(self.ev_battery_sell_qtb_list[agent_id][next_states[4], :]) - self.ev_battery_sell_qtb_list[agent_id][states[4], 
                                                                                                                                       int(actions[4]-self.params['price_min'])]

        self.dr_buy_qtb_list[agent_id][states[0], int(actions[0] - self.params['price_min'])] += alpha * dr_buy_td_error
        self.battery_buy_qtb_list[agent_id][states[1], int(actions[1] - self.params['price_min'])] += alpha * battery_buy_td_error
        self.battery_sell_qtb_list[agent_id][states[2], int(actions[2] - self.params['price_min'])] += alpha * battery_sell_td_error
        self.ev_battery_buy_qtb_list[agent_id][states[3], int(actions[3] - self.params['price_min'])] += alpha * ev_battery_buy_td_error
        self.ev_battery_sell_qtb_list[agent_id][states[4], int(actions[4] - self.params['price_min'])] += alpha * ev_battery_sell_td_error

    def save_q_table(self, folder_path):
        os.makedirs(folder_path + '/q_table', exist_ok=True)
        for i in range(len(self.dr_buy_qtb_list)):
            np.save(folder_path + f'/q_table/dr_buy_qtb_{i}.npy', self.dr_buy_qtb_list[i])
            np.save(folder_path + f'/q_table/battery_buy_qtb_{i}.npy', self.battery_buy_qtb_list[i])
            np.save(folder_path + f'/q_table/battery_sell_qtb_{i}.npy', self.battery_sell_qtb_list[i])
            np.save(folder_path + f'/q_table/ev_battery_buy_qtb_{i}.npy', self.ev_battery_buy_qtb_list[i])
            np.save(folder_path + f'/q_table/ev_battery_sell_qtb_{i}.npy', self.ev_battery_sell_qtb_list[i])
            df = pd.DataFrame(self.dr_buy_qtb_list[i])
            df.to_csv(folder_path + f'/q_table/dr_buy_qtb_{i}.csv')
            df = pd.DataFrame(self.battery_buy_qtb_list[i])
            df.to_csv(folder_path + f'/q_table/battery_buy_qtb_{i}.csv')
            df = pd.DataFrame(self.battery_sell_qtb_list[i])
            df.to_csv(folder_path + f'/q_table/battery_sell_qtb_{i}.csv')
            df = pd.DataFrame(self.ev_battery_buy_qtb_list[i])
            df.to_csv(folder_path + f'/q_table/ev_battery_buy_qtb_{i}.csv')
            df = pd.DataFrame(self.ev_battery_sell_qtb_list[i])
            df.to_csv(folder_path + f'/q_table/ev_battery_sell_qtb_{i}.csv')
            
    def load_q_table(self, folder_path):
        self.dr_buy_qtb_list = []
        self.battery_buy_qtb_list = []
        self.battery_sell_qtb_list = []
        self.ev_battery_buy_qtb_list = []
        self.ev_battery_sell_qtb_list = []
        for i in range(self.agent_num):
            self.dr_buy_qtb_list.append(np.load(folder_path + f'/dr_buy_qtb_{i}.npy'))
            self.battery_buy_qtb_list.append(np.load(folder_path + f'/battery_buy_qtb_{i}.npy'))
            self.battery_sell_qtb_list.append(np.load(folder_path + f'/battery_sell_qtb_{i}.npy'))
            self.ev_battery_buy_qtb_list.append(np.load(folder_path + f'/ev_battery_buy_qtb_{i}.npy'))
            self.ev_battery_sell_qtb_list.append(np.load(folder_path + f'/ev_battery_sell_qtb_{i}.npy'))
        # print('Q table loaded.')
    

if __name__ == '__main__':
    agent_num = 10
    params = {'price_max': 120,
              'price_min': 5,
              'wheeling_charge': 10,
              'battery_charge_efficiency': 0.9,
              'battery_discharge_efficiency': 0.9,
              'ev_charge_efficiency': 0.9,
              'ev_discharge_efficiency': 0.9,
    }
    q = Q(params, agent_num=agent_num, num_dizitized_pv_ratio=20, num_dizitized_soc=20, num_elastic_ratio_pattern=3)
    dr_buy_qtb, battery_buy_qtb, battery_sell_qtb, ev_battery_buy_qtb, ev_battery_sell_qtb = q.get_qtbs_
    q.reset_all_digitized_states()
    for n in range(agent_num):
        q.set_digitized_states(agent_id=n, pv_ratio=0.9, battery_soc=0.53, ev_battery_soc=0.67, elastic_ratio=0.5)
    # q.set_digitized_states(agent_id=0, pv_ratio=0.21, battery_soc=0.43, ev_battery_soc=0.67, elastic_ratio=0.3)
    dr_states, battery_states, ev_battery_states = q.get_states_
    print(dr_states)
    print(battery_states)
    print(ev_battery_states)


    # print(np.digitize(100, bins=bins(0, 100, 20)))

    dr_price_threshold_list = [20, 25, 30, 35, 40]
    # print(dr_price_threshold_list.index(25))

    q.reset_all_actions()
    q.set_actions(agent_id=0, episode=1)
    