import pandas as pd
import numpy as np


class Agent:
    def __init__(self, num_agent):
        self.num_agent = num_agent
        self.agent_params_df = pd.DataFrame(0.0, index=np.arange(num_agent), columns=['shift_limit', 
                                                                                      'elastic_ratio', 
                                                                                      'dr_price_threshold', 
                                                                                      'battery_capacity', 
                                                                                      'ev_capacity', 
                                                                                      'alpha', 
                                                                                      'beta',
                                                                                      'gamma', 
                                                                                      'epsilon',
                                                                                      'psi',
                                                                                      'omega',])
        
        
    def __getitem__(self, index):
        if 0 <= index < self.num_agent:
            return self.agent_params_df.loc[index]
        else:
            raise IndexError("Agent index out of range")

    def set_uniform(self, **kwargs):
        self.agent_params_df['shift_limit'] = kwargs['shift_limit']
        self.agent_params_df['elastic_ratio'] = kwargs['elastic_ratio']
        self.agent_params_df['dr_price_threshold'] = kwargs['dr_price_threshold']
        self.agent_params_df['battery_capacity'] = kwargs['battery_capacity']
        self.agent_params_df['ev_capacity'] = kwargs['ev_capacity']
        self.agent_params_df['alpha'] = kwargs['alpha']
        self.agent_params_df['beta'] = kwargs['beta']
        self.agent_params_df['gamma'] = kwargs['gamma']
        self.agent_params_df['epsilon'] = kwargs['epsilon']
        self.agent_params_df['psi'] = kwargs['psi']
        self.agent_params_df['omega'] = kwargs['omega']

    def set_one_agent(self, agent_id, **kwargs):
        self.agent_params_df.at[agent_id, 'shift_limit'] = kwargs['shift_limit']
        self.agent_params_df.at[agent_id, 'elastic_ratio'] = kwargs['elastic_ratio']
        self.agent_params_df.at[agent_id, 'dr_price_threshold'] = kwargs['dr_price_threshold']
        self.agent_params_df.at[agent_id, 'battery_capacity'] = kwargs['battery_capacity']
        self.agent_params_df.at[agent_id, 'ev_capacity'] = kwargs['ev_capacity']
        self.agent_params_df.at[agent_id, 'alpha'] = kwargs['alpha']
        self.agent_params_df.at[agent_id, 'beta'] = kwargs['beta']
        self.agent_params_df.at[agent_id, 'gamma'] = kwargs['gamma']
        self.agent_params_df.at[agent_id, 'epsilon'] = kwargs['epsilon']
        self.agent_params_df.at[agent_id, 'psi'] = kwargs['psi']
        self.agent_params_df.at[agent_id, 'omega'] = kwargs['omega']

    def generate_params(self, **kwargs):
        params = {'shift_limit_list': [6.0, 12.0, 18.0, 24.0],
                  'elastic_ratio_list': [0.3, 0.4, 0.5],
                  'dr_price_threshold_list': [20, 25, 30, 35, 40], 
                  'battery_capacity_list': [10, 15, 20],
                  'ev_capacity_list': [20, 40, 60, 80],
                  'alpha_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
                  'beta_list': [1],
                  'gamma_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
                  'epsilon_list': [1],
                  'psi_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
                  'omega_list': [1],}
        params.update(kwargs)
        shift_limit_list = params['shift_limit_list']
        elastic_ratio_list = params['elastic_ratio_list']
        dr_price_threshold_list = params['dr_price_threshold_list']
        battery_capacity_list = params['battery_capacity_list']
        ev_capacity_list = params['ev_capacity_list']
        alpha_list = params['alpha_list']
        beta_list = params['beta_list']
        gamma_list = params['gamma_list']
        epsilon_list = params['epsilon_list']
        psi_list = params['psi_list']
        omega_list = params['omega_list']

        self.agent_params_df['shift_limit'] = self.agent_params_df.apply(
            lambda row: shift_limit_list[np.random.randint(0, len(shift_limit_list))], axis=1)
        self.agent_params_df['elastic_ratio'] = self.agent_params_df.apply(
            lambda row: elastic_ratio_list[np.random.randint(0, len(elastic_ratio_list))], axis=1)
        self.agent_params_df['dr_price_threshold'] = self.agent_params_df.apply(
            lambda row: dr_price_threshold_list[np.random.randint(0, len(dr_price_threshold_list))], axis=1)
        self.agent_params_df['battery_capacity'] = self.agent_params_df.apply(
            lambda row: battery_capacity_list[np.random.randint(0, len(battery_capacity_list))], axis=1)
        self.agent_params_df['ev_capacity'] = self.agent_params_df.apply(
            lambda row: ev_capacity_list[np.random.randint(0, len(ev_capacity_list))], axis=1)
        self.agent_params_df['alpha'] = self.agent_params_df.apply(
            lambda row: alpha_list[np.random.randint(0, len(alpha_list))], axis=1)
        self.agent_params_df['beta'] = self.agent_params_df.apply(
            lambda row: beta_list[np.random.randint(0, len(beta_list))], axis=1)
        self.agent_params_df['gamma'] = self.agent_params_df.apply(
            lambda row: gamma_list[np.random.randint(0, len(gamma_list))], axis=1)
        self.agent_params_df['epsilon'] = self.agent_params_df.apply(
            lambda row: epsilon_list[np.random.randint(0, len(epsilon_list))], axis=1)
        self.agent_params_df['psi'] = self.agent_params_df.apply(
            lambda row: psi_list[np.random.randint(0, len(psi_list))], axis=1)
        self.agent_params_df['omega'] = self.agent_params_df.apply(
            lambda row: omega_list[np.random.randint(0, len(omega_list))], axis=1)
        return params
    
    def get_agent_params(self, agent_id):
        return self.agent_params_df.loc[agent_id]
    
    @property
    def get_agents_params_df_(self):
        return self.agent_params_df
    
    def save(self, folder_path):
        self.agent_params_df.to_csv(f'{folder_path}/agent_params.csv', index=True)
        

if __name__ == '__main__':
    agents = Agent(20)
    agents.generate_params()
    params_df = agents.get_agents_params_df_
    print(params_df)
    
    for i in range(5):
        print(f"Agent {i} parameters:")
        print(agents[i])
        print()
