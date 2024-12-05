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
        self.possible_params = Agent(self.agent_num).generate_params(params)
        self.params = params
        self.num_dizitized_pv_ratio = num_dizitized_pv_ratio
        self.num_dizitized_soc = num_dizitized_soc
        self.num_elastic_ratio_pattern = num_elastic_ratio_pattern
        self.discount_rate = params['discount_rate']
        self.learning_rate = params['learning_rate']

        dr_buy_rows = self.num_dizitized_pv_ratio * self.num_elastic_ratio_pattern
        battery_buy_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc
        battery_sell_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc
        ev_battery_buy_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc
        ev_battery_sell_rows = self.num_dizitized_pv_ratio * self.num_dizitized_soc
        pv_sell_rows = self.num_dizitized_pv_ratio
        cols = (int(self.params['price_max']) - int(self.params['price_min']) + 1)//2
        self.cols_num = cols
        self.battery_pattern_list = self.possible_params['battery_capacity_list']
        self.ev_battery_pattern_list = self.possible_params['ev_capacity_list']
        self.pv_pattern_list = self.possible_params['pv_capacity_list']
        # MARLのため、エージェントごとにQテーブルを用意する
        self.dr_buy_qtb_list = [np.full((dr_buy_rows, cols), 0.0) for _ in range(self.agent_num)]
        self.battery_buy_qtb_list = [np.full((battery_buy_rows, cols, len(self.battery_pattern_list)), 0.0) for _ in range(self.agent_num)]
        self.battery_sell_qtb_list = [np.full((battery_sell_rows, cols, len(self.battery_pattern_list)), 0.0) for _ in range(self.agent_num)]
        self.ev_battery_buy_qtb_list = [np.full((ev_battery_buy_rows, cols, len(self.ev_battery_pattern_list)), 0.0) for _ in range(self.agent_num)]
        self.ev_battery_sell_qtb_list = [np.full((ev_battery_sell_rows, cols, len(self.ev_battery_pattern_list)), 0.0) for _ in range(self.agent_num)]
        self.pv_sell_qtb_list = [np.full((pv_sell_rows, cols, len(self.pv_pattern_list)), 0.0) for _ in range(self.agent_num)]

    @property
    def get_qtbs_(self):
        return self.dr_buy_qtb_list, self.battery_buy_qtb_list, self.battery_sell_qtb_list, self.ev_battery_buy_qtb_list, self.ev_battery_sell_qtb_list, self.pv_sell_qtb_list

    def get_agent_qtbs(self, agent_id):
        return self.dr_buy_qtb_list[agent_id], self.battery_buy_qtb_list[agent_id], self.battery_sell_qtb_list[agent_id], self.ev_battery_buy_qtb_list[agent_id], self.ev_battery_sell_qtb_list[agent_id], self.pv_sell_qtb_list[agent_id]
    
    def reset_all_digitized_states(self):
        self.dr_states = np.full(self.agent_num, np.nan)
        self.battery_states = np.full(self.agent_num, np.nan)
        self.ev_battery_states = np.full(self.agent_num, np.nan)
        self.pv_states = np.full(self.agent_num, np.nan)
        self.battery_patterns = np.full(self.agent_num, np.nan)
        self.ev_battery_patterns = np.full(self.agent_num, np.nan)
        self.pv_patterns = np.full(self.agent_num, np.nan)
    
    def set_digitized_states(self, agent_id, agent_params, pv_ratio, battery_soc, ev_battery_soc, elastic_ratio):
        """
        agent_idごとに離散化した状態(番号)を格納していく
        """
        digitized_pv_ratio = np.digitize(pv_ratio, bins=bins(0, 1, self.num_dizitized_pv_ratio))-1
        digitized_battery_soc = np.digitize(battery_soc, bins=bins(0, 1, self.num_dizitized_soc))-1
        digitized_ev_battery_soc = np.digitize(ev_battery_soc, bins=bins(0, 1, self.num_dizitized_soc))-1
        digitized_elastic_ratio = np.digitize(elastic_ratio, bins=bins(0.1, 0.5, self.num_elastic_ratio_pattern))-1
        pv_pattern = self.pv_pattern_list.index(agent_params['pv_capacity'])
        battery_pattern = self.battery_pattern_list.index(agent_params['battery_capacity'])
        ev_battery_pattern = self.ev_battery_pattern_list.index(agent_params['ev_capacity'])

        dr_state = (digitized_pv_ratio + 
                    digitized_elastic_ratio * self.num_dizitized_pv_ratio)
            
        battery_state = (digitized_pv_ratio +
                         digitized_battery_soc * self.num_dizitized_soc)
                                
        ev_battery_state = (digitized_pv_ratio +
                            digitized_ev_battery_soc * self.num_dizitized_soc)
        
        pv_states = digitized_pv_ratio
                         
        self.dr_states[agent_id] = dr_state
        self.battery_states[agent_id] = battery_state
        self.ev_battery_states[agent_id] = ev_battery_state
        self.pv_states[agent_id] = pv_states
        self.battery_patterns[agent_id] = battery_pattern
        self.ev_battery_patterns[agent_id] = ev_battery_pattern
        self.pv_patterns[agent_id] = pv_pattern
        return dr_state, battery_state, ev_battery_state, pv_states, battery_pattern, ev_battery_pattern, pv_pattern
    
    @property
    def get_states_(self):
        return self.dr_states, self.battery_states, self.ev_battery_states, self.pv_states, self.battery_patterns, self.ev_battery_patterns, self.pv_patterns
    
    def get_agent_states(self, agent_id):
        return self.dr_states[agent_id], self.battery_states[agent_id], self.ev_battery_states[agent_id], self.pv_states[agent_id], self.battery_patterns[agent_id], self.ev_battery_patterns[agent_id], self.pv_patterns[agent_id] 
    def reset_all_actions(self):
        self.next_actions = np.full((self.agent_num, 6), np.nan)
    
    def set_actions(self, agent_id, episode, is_train):
        """
        dr_buy, battery_buy, battery_sell, ev_battery_buy, ev_battery_sell, pv_sellの順でsetする
        battery_sell, ev_battery_sellはそれぞれのbuyよりも高い価格で入札させるため、buy入札価格からの差分を学習対象とする
        """
        # The seed must be set to None to avoid the same random numbers being generated
        np.random.seed(None)
        if is_train:
            # 徐々に最適行動のみをとる、ε-greedy法
            epsilon = 0.5 * (1 / (episode + 1))
            next_action_list = []
            if epsilon <= np.random.uniform(0, 1):
                dr_buy_price = np.argmax(self.dr_buy_qtb_list[agent_id][int(self.dr_states[agent_id])]) + int(self.params['price_min'])
                next_action_list.append(dr_buy_price)
                sliced_battery_buy_qtb = self.battery_buy_qtb_list[agent_id][:, :, int(self.battery_patterns[agent_id])]
                battery_buy_price = np.argmax(sliced_battery_buy_qtb[int(self.battery_states[agent_id])]) + int(self.params['price_min'])
                next_action_list.append(battery_buy_price)
                sliced_battery_sell_qtb = self.battery_sell_qtb_list[agent_id][:, :, int(self.battery_patterns[agent_id])]
                reversed_battery_sell_qtb = sliced_battery_sell_qtb[int(self.battery_states[agent_id])][::-1]
                max_index = len(sliced_battery_sell_qtb[int(self.battery_states[agent_id])]) - 1 - np.argmax(reversed_battery_sell_qtb)
                battery_sell_price = max_index + + int(self.params['price_min'])
                # if battery_sell_price > int(self.params['price_max']):
                #     battery_sell_price = int(self.params['price_max'])
                next_action_list.append(battery_sell_price)
                # next_action_list.append(np.argmax(self.battery_sell_qtb_list[agent_id][int(self.battery_states[agent_id])]) + int(self.params['price_min']))
                sliced_ev_battery_buy_qtb = self.ev_battery_buy_qtb_list[agent_id][:, :, int(self.ev_battery_patterns[agent_id])]
                ev_battery_buy_price = np.argmax(sliced_ev_battery_buy_qtb[int(self.ev_battery_states[agent_id])]) + int(self.params['price_min'])
                next_action_list.append(ev_battery_buy_price)
                sliced_ev_battery_sell_qtb = self.ev_battery_sell_qtb_list[agent_id][:, :, int(self.ev_battery_patterns[agent_id])]
                reversed_ev_battery_sell_qtb = sliced_ev_battery_sell_qtb[int(self.ev_battery_states[agent_id])][::-1]
                max_index = len(sliced_ev_battery_sell_qtb[int(self.ev_battery_states[agent_id])]) - 1 - np.argmax(reversed_ev_battery_sell_qtb)
                ev_battery_sell_price = max_index + + int(self.params['price_min'])
                # if ev_battery_sell_price > int(self.params['price_max']):
                #     ev_battery_sell_price = int(self.params['price_max'])
                next_action_list.append(ev_battery_sell_price)
                # next_action_list.append(np.argmax(self.ev_battery_sell_qtb_list[agent_id][int(self.ev_battery_states[agent_id])]) + int(self.params['price_min']))
                sliced_pv_sell_qtb = self.pv_sell_qtb_list[agent_id][:, :, int(self.pv_patterns[agent_id])]
                pv_sell_price = np.argmax(sliced_pv_sell_qtb[int(self.pv_states[agent_id])]) + int(self.params['price_min'])
                next_action_list.append(pv_sell_price)
            else:
                dr_buy_price = np.random.choice(range(int(self.params['price_min']), self.cols_num + 1))
                next_action_list.append(dr_buy_price)
                battery_buy_price = np.random.choice(range(int(self.params['price_min']), self.cols_num + 1))
                next_action_list.append(battery_buy_price)
                battery_sell_price = np.random.choice(range(battery_buy_price, self.cols_num + 1))
                next_action_list.append(battery_sell_price)
                ev_battery_buy_price = np.random.choice(range(int(self.params['price_min']), self.cols_num + 1))
                next_action_list.append(ev_battery_buy_price)
                ev_battery_sell_price = np.random.choice(range(ev_battery_buy_price, self.cols_num + 1))
                next_action_list.append(ev_battery_sell_price)
                pv_sell_price = np.random.choice(range(int(self.params['price_min']), self.cols_num + 1))
                next_action_list.append(pv_sell_price)
            self.next_actions[agent_id] = next_action_list
            return next_action_list
        # テスト時は最適行動のみをとる
        else:
            next_action_list = []
            next_action_list.append(np.argmax(self.dr_buy_qtb_list[agent_id][int(self.dr_states[agent_id])]) + int(self.params['price_min']))
            sliced_battery_buy_qtb = self.battery_buy_qtb_list[agent_id][:, :, int(self.battery_patterns[agent_id])]
            next_action_list.append(np.argmax(sliced_battery_buy_qtb[int(self.battery_states[agent_id])]) + int(self.params['price_min']))
            sliced_battery_sell_qtb = self.battery_sell_qtb_list[agent_id][:, :, int(self.battery_patterns[agent_id])]
            reversed_battery_sell_qtb = sliced_battery_sell_qtb[int(self.battery_states[agent_id])][::-1]
            max_index = len(sliced_battery_sell_qtb[int(self.battery_states[agent_id])]) - 1 - np.argmax(reversed_battery_sell_qtb)
            next_action_list.append(max_index + int(self.params['price_min']))
            # next_action_list.append(np.argmax(self.battery_sell_qtb_list[agent_id][int(self.battery_states[agent_id])]) + int(self.params['price_min']))
            sliced_ev_battery_buy_qtb = self.ev_battery_buy_qtb_list[agent_id][:, :, int(self.ev_battery_patterns[agent_id])]
            next_action_list.append(np.argmax(sliced_ev_battery_buy_qtb[int(self.ev_battery_states[agent_id])]) + int(self.params['price_min']))
            sliced_ev_battery_sell_qtb = self.ev_battery_sell_qtb_list[agent_id][:, :, int(self.ev_battery_patterns[agent_id])]
            reversed_ev_battery_sell_qtb = sliced_ev_battery_sell_qtb[int(self.ev_battery_states[agent_id])][::-1]
            max_index = len(sliced_ev_battery_sell_qtb[int(self.ev_battery_states[agent_id])]) - 1 - np.argmax(reversed_ev_battery_sell_qtb)
            next_action_list.append(max_index + int(self.params['price_min']))
            # next_action_list.append(np.argmax(self.ev_battery_sell_qtb_list[agent_id][int(self.ev_battery_states[agent_id])]) + int(self.params['price_min']))
            sliced_pv_sell_qtb = self.pv_sell_qtb_list[agent_id][:, :, int(self.pv_patterns[agent_id])]
            next_action_list.append(np.argmax(sliced_pv_sell_qtb[int(self.pv_states[agent_id])]) + int(self.params['price_min']))
            self.next_actions[agent_id] = next_action_list
            return next_action_list

    def get_facility_capacities(self, agent_id, episode, is_train):
        np.random.seed(None)
        if is_train:
            epsilon = (1 / (episode + 1))
            if epsilon <= np.random.uniform(0, 1):
                # 各スライスごとに平均値を計算
                battery_buy_avg_values = [np.mean(self.battery_buy_qtb_list[agent_id][:, :, i]) for i in range(self.battery_buy_qtb_list[agent_id].shape[2])]
                battery_sell_avg_values = [np.mean(self.battery_sell_qtb_list[agent_id][:, :, i]) for i in range(self.battery_sell_qtb_list[agent_id].shape[2])]
                sums = np.array(battery_buy_avg_values) + np.array(battery_sell_avg_values)
                # Qテーブルの平均値が最大となるindexを取得
                battery_capacity_index = np.argmax(sums)
                battery_capacity = self.possible_params['battery_capacity_list'][battery_capacity_index]

                ev_battery_buy_avg_values = [np.mean(self.ev_battery_buy_qtb_list[agent_id][:, :, i]) for i in range(self.ev_battery_buy_qtb_list[agent_id].shape[2])]
                ev_battery_sell_avg_values = [np.mean(self.ev_battery_sell_qtb_list[agent_id][:, :, i]) for i in range(self.ev_battery_sell_qtb_list[agent_id].shape[2])]
                sums = np.array(ev_battery_buy_avg_values) + np.array(ev_battery_sell_avg_values)
                ev_capacity_index = np.argmax(sums)
                ev_capacity = self.possible_params['ev_capacity_list'][ev_capacity_index]

                pv_sell_avg_values = [np.mean(self.pv_sell_qtb_list[agent_id][:, :, i]) for i in range(self.pv_sell_qtb_list[agent_id].shape[2])]
                pv_capacity_index = np.argmax(pv_sell_avg_values)
                pv_capacity = self.possible_params['pv_capacity_list'][pv_capacity_index]
            else:
                battery_capacity = np.random.choice(self.possible_params['battery_capacity_list'])
                ev_capacity = np.random.choice(self.possible_params['ev_capacity_list'])
                pv_capacity = np.random.choice(self.possible_params['pv_capacity_list'])
            return battery_capacity, ev_capacity, pv_capacity
        else:
            # 各スライスごとに平均値を計算(初期値のままのセルは平均値計算から除外)
            exclude_value = 0
            battery_buy_avg_values = [
                np.mean(
                    np.ma.masked_where(
                        self.battery_buy_qtb_list[agent_id][:, :, i] == exclude_value,
                        self.battery_buy_qtb_list[agent_id][:, :, i]
                    )
                )
                for i in range(self.battery_buy_qtb_list[agent_id].shape[2])
            ]
            battery_sell_avg_values = [
                np.mean(
                    np.ma.masked_where(
                        self.battery_sell_qtb_list[agent_id][:, :, i] == exclude_value,
                        self.battery_sell_qtb_list[agent_id][:, :, i]
                    )
                )
                for i in range(self.battery_sell_qtb_list[agent_id].shape[2])
            ]
            # battery_buy_avg_values = [np.mean(self.battery_buy_qtb_list[agent_id][:, :, i]) for i in range(self.battery_buy_qtb_list[agent_id].shape[2])]
            # battery_sell_avg_values = [np.mean(self.battery_sell_qtb_list[agent_id][:, :, i]) for i in range(self.battery_sell_qtb_list[agent_id].shape[2])]
            sums = np.array(battery_buy_avg_values) + np.array(battery_sell_avg_values)
            # Qテーブルの平均値が最大となるindexを取得
            battery_capacity_index = np.argmax(sums)
            battery_capacity = self.possible_params['battery_capacity_list'][battery_capacity_index]

            ev_battery_buy_avg_values = [
                np.mean(
                    np.ma.masked_where(
                        self.ev_battery_buy_qtb_list[agent_id][:, :, i] == exclude_value,
                        self.ev_battery_buy_qtb_list[agent_id][:, :, i]
                    )
                )
                for i in range(self.ev_battery_buy_qtb_list[agent_id].shape[2])
            ]
            # ev_battery_buy_avg_values = [np.mean(self.ev_battery_buy_qtb_list[agent_id][:, :, i]) for i in range(self.ev_battery_buy_qtb_list[agent_id].shape[2])]
            # ev_battery_sell_avg_values = [np.mean(self.ev_battery_sell_qtb_list[agent_id][:, :, i]) for i in range(self.ev_battery_sell_qtb_list[agent_id].shape[2])]
            sums = np.array(ev_battery_buy_avg_values) + np.array(ev_battery_sell_avg_values)
            ev_capacity_index = np.argmax(sums)
            ev_capacity = self.possible_params['ev_capacity_list'][ev_capacity_index]

            pv_sell_avg_values = [
                np.mean(
                    np.ma.masked_where(
                        self.pv_sell_qtb_list[agent_id][:, :, i] == exclude_value,
                        self.pv_sell_qtb_list[agent_id][:, :, i]
                    )
                )
                for i in range(self.pv_sell_qtb_list[agent_id].shape[2])
            ]
            # pv_sell_avg_values = [np.mean(self.pv_sell_qtb_list[agent_id][:, :, i]) for i in range(self.pv_sell_qtb_list[agent_id].shape[2])]
            pv_capacity_index = np.argmax(pv_sell_avg_values)
            pv_capacity = self.possible_params['pv_capacity_list'][pv_capacity_index]
            return battery_capacity, ev_capacity, pv_capacity

    
    @property
    def get_actions_(self):
        """
        エージェントiの行動がi行目
        カラムにはdr_buy, battery_buy, battery_sell, ev_battery_buy, ev_battery_sell, pv_sellの順で格納されている
        """
        return self.next_actions
    
    def update_q_table(self, agent_id, states, actions, rewards, next_states):
        """
        states, actions, rewards, next_statesは全てリスト
        それぞれのリストについて、0番目にdr_buy、1番目にbattery_buy、2番目にbattery_sell、3番目にev_battery_buy、4番目にev_battery_sell, 5番目にpv_sell の情報が格納されている
        """
        # if agent_id == 3:
        #     print('states:', states)
        #     print('actions:', actions)
        #     print('rewards:', rewards)
        #     print('next_states:', next_states)
        gamma = self.discount_rate
        alpha = self.learning_rate
        dr_buy_td_error = rewards[0] + gamma * np.max(self.dr_buy_qtb_list[agent_id][next_states[0], :]) - self.dr_buy_qtb_list[agent_id][states[0], 
                                                                                                            int(actions[0]-self.params['price_min'])]
        battery_buy_td_error = rewards[1] + gamma * np.max(self.battery_buy_qtb_list[agent_id][next_states[1], :, int(self.battery_patterns[agent_id])]) - self.battery_buy_qtb_list[agent_id][states[1], 
                                                                                                                           int(actions[1]-self.params['price_min']), int(self.battery_patterns[agent_id])]
        battery_sell_td_error = rewards[2] + gamma * np.max(self.battery_sell_qtb_list[agent_id][next_states[2], :, int(self.battery_patterns[agent_id])]) - self.battery_sell_qtb_list[agent_id][states[2], 
                                                                                                                              int(actions[2]-self.params['price_min']), int(self.battery_patterns[agent_id])]
        ev_battery_buy_td_error = rewards[3] + gamma * np.max(self.ev_battery_buy_qtb_list[agent_id][next_states[3], :, int(self.ev_battery_patterns[agent_id])]) - self.ev_battery_buy_qtb_list[agent_id][states[3], 
                                                                                                                                    int(actions[3]-self.params['price_min']), int(self.ev_battery_patterns[agent_id])]
        ev_battery_sell_td_error = rewards[4] + gamma * np.max(self.ev_battery_sell_qtb_list[agent_id][next_states[4], :, int(self.ev_battery_patterns[agent_id])]) - self.ev_battery_sell_qtb_list[agent_id][states[4], 
                                                                                                                                       int(actions[4]-self.params['price_min']), int(self.ev_battery_patterns[agent_id])]
        pv_sell_td_error = rewards[5] + gamma * np.max(self.pv_sell_qtb_list[agent_id][next_states[5], :, int(self.pv_patterns[agent_id])]) - self.pv_sell_qtb_list[agent_id][states[5], 
                                                                                                                              int(actions[5]-self.params['price_min']), int(self.pv_patterns[agent_id])]

        self.dr_buy_qtb_list[agent_id][states[0], int(actions[0] - self.params['price_min'])] += alpha * dr_buy_td_error
        self.battery_buy_qtb_list[agent_id][states[1], int(actions[1] - self.params['price_min']), int(self.battery_patterns[agent_id])] += alpha * battery_buy_td_error
        self.battery_sell_qtb_list[agent_id][states[2], int(actions[2] - self.params['price_min']), int(self.battery_patterns[agent_id])] += alpha * battery_sell_td_error
        self.ev_battery_buy_qtb_list[agent_id][states[3], int(actions[3] - self.params['price_min']), int(self.ev_battery_patterns[agent_id])] += alpha * ev_battery_buy_td_error
        self.ev_battery_sell_qtb_list[agent_id][states[4], int(actions[4] - self.params['price_min']), int(self.ev_battery_patterns[agent_id])] += alpha * ev_battery_sell_td_error
        self.pv_sell_qtb_list[agent_id][states[5], int(actions[5] - self.params['price_min']), int(self.pv_patterns[agent_id])] += alpha * pv_sell_td_error

    def save_q_table(self, folder_path, train=True):
        os.makedirs(folder_path + '/q_table', exist_ok=True)
        for i in range(len(self.dr_buy_qtb_list)):
            np.save(folder_path + f'/q_table/dr_buy_qtb_{i}.npy', self.dr_buy_qtb_list[i])
            np.save(folder_path + f'/q_table/battery_buy_qtb_{i}.npy', self.battery_buy_qtb_list[i])
            np.save(folder_path + f'/q_table/battery_sell_qtb_{i}.npy', self.battery_sell_qtb_list[i])
            np.save(folder_path + f'/q_table/ev_battery_buy_qtb_{i}.npy', self.ev_battery_buy_qtb_list[i])
            np.save(folder_path + f'/q_table/ev_battery_sell_qtb_{i}.npy', self.ev_battery_sell_qtb_list[i])
            np.save(folder_path + f'/q_table/pv_sell_qtb_{i}.npy', self.pv_sell_qtb_list[i])
            if not train:
                df = pd.DataFrame(self.dr_buy_qtb_list[i])
                df.to_csv(folder_path + f'/q_table/dr_buy_qtb_{i}.csv')
                for j in range(self.battery_buy_qtb_list[i].shape[2]):
                    sliced_battery_buy_qtb = self.battery_buy_qtb_list[i][:, :, j]
                    np.savetxt(folder_path + f'/q_table/battery_buy_qtb_{i}_{j}.csv', sliced_battery_buy_qtb, delimiter=',', fmt="%.5f")
                for j in range(self.battery_sell_qtb_list[i].shape[2]):
                    sliced_battery_sell_qtb = self.battery_sell_qtb_list[i][:, :, j]
                    np.savetxt(folder_path + f'/q_table/battery_sell_qtb_{i}_{j}.csv', sliced_battery_sell_qtb, delimiter=',', fmt="%.5f")
                for j in range(self.ev_battery_buy_qtb_list[i].shape[2]):
                    sliced_ev_battery_buy_qtb = self.ev_battery_buy_qtb_list[i][:, :, j]
                    np.savetxt(folder_path + f'/q_table/ev_battery_buy_qtb_{i}_{j}.csv', sliced_ev_battery_buy_qtb, delimiter=',', fmt="%.5f")
                for j in range(self.ev_battery_sell_qtb_list[i].shape[2]):
                    sliced_ev_battery_sell_qtb = self.ev_battery_sell_qtb_list[i][:, :, j]
                    np.savetxt(folder_path + f'/q_table/ev_battery_sell_qtb_{i}_{j}.csv', sliced_ev_battery_sell_qtb, delimiter=',', fmt="%.5f")
                for j in range(self.pv_sell_qtb_list[i].shape[2]):
                    sliced_pv_sell_qtb = self.pv_sell_qtb_list[i][:, :, j]
                    np.savetxt(folder_path + f'/q_table/pv_sell_qtb_{i}_{j}.csv', sliced_pv_sell_qtb, delimiter=',', fmt="%.5f")
            
    def load_q_table(self, folder_path):
        self.dr_buy_qtb_list = []
        self.battery_buy_qtb_list = []
        self.battery_sell_qtb_list = []
        self.ev_battery_buy_qtb_list = []
        self.ev_battery_sell_qtb_list = []
        self.pv_sell_qtb_list = []
        for i in range(self.agent_num):
            self.dr_buy_qtb_list.append(np.load(folder_path + f'/dr_buy_qtb_{i}.npy'))
            self.battery_buy_qtb_list.append(np.load(folder_path + f'/battery_buy_qtb_{i}.npy'))
            self.battery_sell_qtb_list.append(np.load(folder_path + f'/battery_sell_qtb_{i}.npy'))
            self.ev_battery_buy_qtb_list.append(np.load(folder_path + f'/ev_battery_buy_qtb_{i}.npy'))
            self.ev_battery_sell_qtb_list.append(np.load(folder_path + f'/ev_battery_sell_qtb_{i}.npy'))
            self.pv_sell_qtb_list.append(np.load(folder_path + f'/pv_sell_qtb_{i}.npy'))
        # print('Q table loaded.')

    def remove_q_table_saved_data(self, folder_path):
        for i in range(self.agent_num):
            os.remove(folder_path + f'/dr_buy_qtb_{i}.npy')
            os.remove(folder_path + f'/battery_buy_qtb_{i}.npy')
            os.remove(folder_path + f'/battery_sell_qtb_{i}.npy')
            os.remove(folder_path + f'/ev_battery_buy_qtb_{i}.npy')
            os.remove(folder_path + f'/ev_battery_sell_qtb_{i}.npy')
            os.remove(folder_path + f'/pv_sell_qtb_{i}.npy')


if __name__ == '__main__':
    agent_num = 10
    params = {'price_max': 120,
              'price_min': 10,
              'wheeling_charge': 10,
              'battery_charge_efficiency': 0.9,
              'battery_discharge_efficiency': 0.9,
              'ev_charge_efficiency': 0.9,
              'ev_discharge_efficiency': 0.9,
              'battery_capacity_list': [0, 5, 10, 20],
              'ev_capacity_list': [0, 20, 40, 80],
              'pv_capacity_list': [0, 5, 10, 20],
              'discount_rate': 0.99,
              'learning_rate': 0.1,
              'shift_limit_list': [6.0, 12.0, 18.0, 24.0],  # hours
              'max_battery_charge_speed': [3.0],  # kW
              'max_battery_discharge_speed': [3.0],  # kW
              'max_ev_charge_speed': [6.0],  # kW
              'max_ev_discharge_speed': [3.0],  # kW
              'dr_boolean_list': [True, False],
              'alpha_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
              'beta_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
              'gamma_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
              'epsilon_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
              'psi_list': [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
              'omega_list': [1, 1.5, 2, 2.5, 3, 3.5, 4]
    }
    agents = Agent(agent_num)
    agents.generate_params(params)
    agents_params_df = agents.get_agents_params_df_
    q = Q(params, agent_num=agent_num, num_dizitized_pv_ratio=20, num_dizitized_soc=20, num_elastic_ratio_pattern=3)
    dr_buy_qtb, battery_buy_qtb, battery_sell_qtb, ev_battery_buy_qtb, ev_battery_sell_qtb, pv_sell_qtv = q.get_qtbs_
    q.reset_all_digitized_states()
    q.reset_all_actions()
    for n in range(agent_num):
        q.set_digitized_states(agent_id=n, agent_params=agents[n], pv_ratio=0.9, battery_soc=0.53, ev_battery_soc=0.67, elastic_ratio=0.5)
    # q.set_digitized_states(agent_id=0, pv_ratio=0.21, battery_soc=0.43, ev_battery_soc=0.67, elastic_ratio=0.3)
    dr_states, battery_states, ev_battery_states, pv_states, battery_patterns, ev_battery_patterns, pv_patterns = q.get_states_
    print(dr_states)
    print(battery_states)
    print(ev_battery_states)
    print(pv_states)
    print(battery_patterns)
    print(ev_battery_patterns)
    print(pv_patterns)

    next_action_list = q.set_actions(agent_id=0, episode=0, is_train=True)
    print(next_action_list)
    # q.save_q_table(folder_path='output/', train=False)
    battery_capacity, ev_capacity, pv_capacity = q.get_facility_capacities(agent_id=0, episode=1, is_train=True)
    print(battery_capacity)
    print(ev_capacity)
    print(pv_capacity)
    