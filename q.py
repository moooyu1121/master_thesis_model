from agent import Agent
import pandas as pd
import numpy as np


# 観測した状態を離散値にデジタル変換する
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


class Q:
    def __init__(self, params, agent_num, num_dizitized_pv_ratio, num_dizitized_soc):
        self.agent_num = agent_num
        self.possible_params = Agent(self.agent_num).generate_params()
        self.params = params
        self.num_dizitized_pv_ratio = num_dizitized_pv_ratio
        self.num_dizitized_soc = num_dizitized_soc

        dr_buy_rows = self.num_dizitized_pv_ratio * len(self.possible_params['elastic_ratio_list']) * \
                len(self.possible_params['dr_price_threshold_list']) * \
                len(self.possible_params['alpha_list']) * \
                len(self.possible_params['beta_list'])
        battery_buy_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc * \
                len(self.possible_params['battery_capacity_list']) * \
                len(self.possible_params['gamma_list']) * \
                len(self.possible_params['epsilon_list'])
        battery_sell_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc * \
                len(self.possible_params['battery_capacity_list']) * \
                len(self.possible_params['gamma_list']) * \
                len(self.possible_params['epsilon_list'])
        ev_battery_buy_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc * \
                len(self.possible_params['ev_capacity_list']) * \
                len(self.possible_params['psi_list']) * \
                len(self.possible_params['omega_list'])
        ev_battery_sell_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc * \
                len(self.possible_params['ev_capacity_list']) * \
                len(self.possible_params['psi_list']) * \
                len(self.possible_params['omega_list'])
        cols = int(self.params['price_max']) - int(self.params['price_min']) + 1
        self.dr_buy_qtb = np.full((dr_buy_rows, cols), 1000.0)
        self.battery_buy_qtb = np.full((battery_buy_rows, cols), 1000.0)
        self.battery_sell_qtb = np.full((battery_sell_rows, cols), 1000.0)
        self.ev_battery_buy_qtb = np.full((ev_battery_buy_rows, cols), 1000.0)
        self.ev_battery_sell_qtb = np.full((ev_battery_sell_rows, cols), 1000.0)

    def set_agent_params(self, agent_params_df):
        self.agents_params_df = agent_params_df

    @property
    def get_qtbs_(self):
        return self.dr_buy_qtb, self.battery_buy_qtb, self.battery_sell_qtb, self.ev_battery_buy_qtb, self.ev_battery_sell_qtb
    
    def reset_all_digitized_states(self):
        self.dr_states = np.full(self.agent_num, np.nan)
        self.battery_states = np.full(self.agent_num, np.nan)
        self.ev_battery_states = np.full(self.agent_num, np.nan)
    
    def set_digitized_states(self, agent_id, pv_ratio, battery_soc, ev_battery_soc):
        """
        agent_idごとに離散化した状態(番号)を格納していく
        """
        digitized_pv_ratio = np.digitize(pv_ratio, bins=bins(0, 1, self.num_dizitized_pv_ratio))
        digitized_battery_soc = np.digitize(battery_soc, bins=bins(0, 1, self.num_dizitized_soc))
        digitized_ev_battery_soc = np.digitize(ev_battery_soc, bins=bins(0, 1, self.num_dizitized_soc))
        elastic_ratio = self.agents_params_df.at[agent_id, 'elastic_ratio']
        dr_price_threshold = self.agents_params_df.at[agent_id, 'dr_price_threshold']
        battery_capacity = self.agents_params_df.at[agent_id, 'battery_capacity']
        ev_battery_capacity = self.agents_params_df.at[agent_id, 'ev_capacity']
        alpha = self.agents_params_df.at[agent_id, 'alpha']
        beta = self.agents_params_df.at[agent_id, 'beta']
        gamma = self.agents_params_df.at[agent_id, 'gamma']
        epsilon = self.agents_params_df.at[agent_id, 'epsilon']
        psi = self.agents_params_df.at[agent_id, 'psi']
        omega = self.agents_params_df.at[agent_id, 'omega']
        
        elastic_ratio_len = len(self.possible_params['elastic_ratio_list'])
        dr_price_threshold_len = len(self.possible_params['dr_price_threshold_list'])
        battery_capacity_len = len(self.possible_params['battery_capacity_list'])
        ev_battery_capacity_len = len(self.possible_params['ev_capacity_list'])
        alpha_len = len(self.possible_params['alpha_list'])
        beta_len = len(self.possible_params['beta_list'])
        gamma_len = len(self.possible_params['gamma_list'])
        epsilon_len = len(self.possible_params['epsilon_list'])
        psi_len = len(self.possible_params['psi_list'])
        omega_len = len(self.possible_params['omega_list'])

        elastic_ratio_index = self.possible_params['elastic_ratio_list'].index(elastic_ratio)
        dr_price_threshold_index = self.possible_params['dr_price_threshold_list'].index(dr_price_threshold)
        battery_capacity_index = self.possible_params['battery_capacity_list'].index(battery_capacity)
        ev_battery_capacity_index = self.possible_params['ev_capacity_list'].index(ev_battery_capacity)
        alpha_index = self.possible_params['alpha_list'].index(alpha)
        beta_index = self.possible_params['beta_list'].index(beta)
        gamma_index = self.possible_params['gamma_list'].index(gamma)
        epsilon_index = self.possible_params['epsilon_list'].index(epsilon)
        psi_index = self.possible_params['psi_list'].index(psi)
        omega_index = self.possible_params['omega_list'].index(omega)

        dr_state = (digitized_pv_ratio + 
                         elastic_ratio_index * elastic_ratio_len + 
                         dr_price_threshold_index * elastic_ratio_len * dr_price_threshold_len + 
                         alpha_index * elastic_ratio_len * dr_price_threshold_len * alpha_len + 
                         beta_index * elastic_ratio_len * dr_price_threshold_len * alpha_len * beta_len)
            
        battery_state = (digitized_pv_ratio +
                              digitized_battery_soc * self.num_dizitized_soc + 
                              battery_capacity_index * self.num_dizitized_soc * battery_capacity_len +
                              gamma_index * self.num_dizitized_soc * battery_capacity_len * gamma_len +
                              epsilon_index * self.num_dizitized_soc * battery_capacity_len * gamma_len * epsilon_len)
        
        ev_battery_state = (digitized_pv_ratio +
                                 digitized_ev_battery_soc * self.num_dizitized_soc +
                                 ev_battery_capacity_index * self.num_dizitized_soc * ev_battery_capacity_len +
                                 psi_index * self.num_dizitized_soc * ev_battery_capacity_len * psi_len +
                                 omega_index * self.num_dizitized_soc * ev_battery_capacity_len * psi_len * omega_len)
        
        self.dr_states[agent_id] = dr_state
        self.battery_states[agent_id] = battery_state
        self.ev_battery_states[agent_id] = ev_battery_state
        return dr_state, battery_state, ev_battery_state
    
    @property
    def get_states(self):
        return self.dr_states, self.battery_states, self.ev_battery_states
    
    def reset_all_actions(self):
        self.next_actions = np.full((self.agent_num, 5), np.nan)
    
    def set_actions(self, agent_id, episode):
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.5 * (1 / (episode + 1))
        next_action_list = []
        if epsilon <= np.random.uniform(0, 1):
            next_action_list.append(np.argmax(self.dr_buy_qtb[int(self.dr_states[agent_id])]) + int(self.params['price_min']))
            next_action_list.append(np.argmax(self.battery_buy_qtb[int(self.battery_states[agent_id])]) + int(self.params['price_min']))
            next_action_list.append(np.argmax(self.battery_sell_qtb[int(self.battery_states[agent_id])]) + int(self.params['price_min']))
            next_action_list.append(np.argmax(self.ev_battery_buy_qtb[int(self.ev_battery_states[agent_id])]) + int(self.params['price_min']))
            next_action_list.append(np.argmax(self.ev_battery_sell_qtb[int(self.ev_battery_states[agent_id])]) + int(self.params['price_min']))
        else:
            for i in range(5):
                next_action_list.append(np.random.choice(
                    range(int(self.params['price_min']), int(self.params['price_max']) + 1)
                    ))
        self.next_actions[agent_id] = next_action_list
        return next_action_list
    
    @property
    def get_actions(self):
        """
        エージェントiの行動がi行目
        カラムにはdr_buy, battery_buy, battery_sell, ev_battery_buy, ev_battery_sellの順で格納されている
        """
        return self.next_actions
    
    def update_q_table(self, states, actions, rewards, next_states):
        """
        states, actions, rewards, next_statesは全てリスト
        それぞれのリストについて、0番目にdr_buy、1番目にbattery_buy、2番目にbattery_sell、3番目にev_battery_buy、4番目にev_battery_sellの情報が格納されている
        """
        gamma = 0.999
        alpha = 0.1
        dr_buy_td_error = rewards[0] + gamma * np.max(self.dr_buy_qtb[next_states[0], :]) - self.dr_buy_qtb[states[0], 
                                                                                                            int(actions[0]-self.params['price_min'])]
        battery_buy_td_error = rewards[1] + gamma * np.max(self.battery_buy_qtb[next_states[1], :]) - self.battery_buy_qtb[states[1], 
                                                                                                                           int(actions[1]-self.params['price_min'])]
        battery_sell_td_error = rewards[2] + gamma * np.max(self.battery_sell_qtb[next_states[2], :]) - self.battery_sell_qtb[states[2], 
                                                                                                                              int(actions[2]-self.params['price_min'])]
        ev_battery_buy_td_error = rewards[3] + gamma * np.max(self.ev_battery_buy_qtb[next_states[3], :]) - self.ev_battery_buy_qtb[states[3], 
                                                                                                                                    int(actions[3]-self.params['price_min'])]
        ev_battery_sell_td_error = rewards[4] + gamma * np.max(self.ev_battery_sell_qtb[next_states[4], :]) - self.ev_battery_sell_qtb[states[4], 
                                                                                                                                       int(actions[4]-self.params['price_min'])]

        self.dr_buy_qtb[states[0], int(actions[0] - self.params['price_min'])] += alpha * dr_buy_td_error
        self.battery_buy_qtb[states[1], int(actions[1] - self.params['price_min'])] += alpha * battery_buy_td_error
        self.battery_sell_qtb[states[2], int(actions[2] - self.params['price_min'])] += alpha * battery_sell_td_error
        self.ev_battery_buy_qtb[states[3], int(actions[3] - self.params['price_min'])] += alpha * ev_battery_buy_td_error
        self.ev_battery_sell_qtb[states[4], int(actions[4] - self.params['price_min'])] += alpha * ev_battery_sell_td_error

    def save_q_table(self, folder_path):
        np.save(folder_path + '/dr_buy_qtb.npy', self.dr_buy_qtb)
        np.save(folder_path + '/battery_buy_qtb.npy', self.battery_buy_qtb)
        np.save(folder_path + '/battery_sell_qtb.npy', self.battery_sell_qtb)
        np.save(folder_path + '/ev_battery_buy_qtb.npy', self.ev_battery_buy_qtb)
        np.save(folder_path + '/ev_battery_sell_qtb.npy', self.ev_battery_sell_qtb)

    def load_q_table(self, folder_path):
        self.dr_buy_qtb = np.load(folder_path + '/dr_buy_qtb.npy')
        self.battery_buy_qtb = np.load(folder_path + '/battery_buy_qtb.npy')
        self.battery_sell_qtb = np.load(folder_path + '/battery_sell_qtb.npy')
        self.ev_battery_buy_qtb = np.load(folder_path + '/ev_battery_buy_qtb.npy')
        self.ev_battery_sell_qtb = np.load(folder_path + '/ev_battery_sell_qtb.npy')
    

if __name__ == '__main__':
    agents_params_df = pd.read_csv("output/agent_params.csv")
    params = {'price_max': 120,
              'price_min': 5,
              'wheeling_charge': 10,
              'battery_charge_efficiency': 0.9,
              'battery_discharge_efficiency': 0.9,
              'ev_charge_efficiency': 0.9,
              'ev_discharge_efficiency': 0.9,
    }
    q = Q(params, agent_num=10, num_dizitized_pv_ratio=20, num_dizitized_soc=20)
    dr_buy_qtb, battery_buy_qtb, battery_sell_qtb, ev_battery_buy_qtb, ev_battery_sell_qtb = q.get_qtbs_
    q.set_agent_params(agents_params_df)
    q.reset_all_digitized_states()
    # for n in range(agents_params_df.shape[0]):
    #     q.set_digitized_states(agent_id=n, pv_ratio=0.21, battery_soc=0.43, ev_battery_soc=0.67)
    q.set_digitized_states(agent_id=0, pv_ratio=0.21, battery_soc=0.43, ev_battery_soc=0.67)
    dr_states, battery_states, ev_battery_states = q.get_states
    print(dr_states)

    # print(np.digitize(100, bins=bins(0, 100, 20)))

    dr_price_threshold_list = [20, 25, 30, 35, 40]
    # print(dr_price_threshold_list.index(25))

    q.reset_all_actions()
    q.set_actions(agent_id=0, episode=1)
    print(q.get_actions)
    print(q.get_actions[0][0])

    print(np.full(10, np.nan))