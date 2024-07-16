import pandas as pd
import random


class Preprocess:
    def __init__(self, seed=42) -> None:
        self.demand_df = pd.DataFrame()
        self.supply_df = pd.DataFrame()
        self.price_df = pd.DataFrame()
        self.seed = seed

    def set(self, demand_df, supply_df, price_df):
        self.demand_df = demand_df
        self.demand_df['timestamp'] = pd.to_datetime(self.demand_df['timestamp'])
        self.demand_df.set_index('timestamp', inplace=True)
        self.supply_df = supply_df
        self.supply_df['timestamp'] = pd.to_datetime(self.supply_df['timestamp'])
        self.supply_df.set_index('timestamp', inplace=True)
        self.price_df = price_df
        self.price_df['timestamp'] = pd.to_datetime(self.price_df['timestamp'])
        self.price_df.set_index('timestamp', inplace=True)

    @property
    def get_dfs_(self):
        return self.demand_df, self.supply_df, self.price_df
    
    @property
    def get_nparrays_(self):
        return self.demand_df.to_numpy(), self.supply_df.to_numpy(), self.price_df.to_numpy()
    
    def generate_demand(self, n):
        num_columns = self.demand_df.shape[1]
        columns = generate_random_integers(num_columns, n, seed=self.seed)
        # とりあえず100m^2の広さとして需要を生成する&WをkWに変換する
        self.demand_df = self.demand_df.iloc[:, columns] * 100 / 1000
        self.demand_df.columns = [f'{i}' for i in range(n)]
        return self.demand_df
    
    def generate_supply(self, n):
        num_columns = self.supply_df.shape[1]
        columns = generate_random_integers(num_columns, n, seed=self.seed)
        # とりあえず10kWの容量のPVを導入するとして供給を生成する
        self.supply_df = self.supply_df.iloc[:, columns] * 10
        self.supply_df.columns = [f'{i}' for i in range(n)]
        return self.supply_df
    
    def generate_d_s(self, n):
        self.generate_demand(n)
        self.generate_supply(n)
        return self.demand_df, self.supply_df
    
    @property
    def drop_index_(self):
        self.demand_df.reset_index(inplace=True, drop=True)
        self.supply_df.reset_index(inplace=True, drop=True)
        self.price_df.reset_index(inplace=True, drop=True)
        return self.demand_df, self.supply_df, self.price_df
    
    def save(self, folder_path):
        self.demand_df.to_csv(f'{folder_path}/demand.csv', index=True)
        self.supply_df.to_csv(f'{folder_path}/supply.csv', index=True)
        self.price_df.to_csv(f'{folder_path}/price.csv', index=True)


# ランダムな整数を生成する関数
def generate_random_integers(num_columns, n, seed=42):
    random.seed(seed)
    return random.sample(range(num_columns), n)


if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.set(pd.read_csv('data/demand.csv'), pd.read_csv('data/supply.csv'), pd.read_csv('data/price.csv'))
    d_df = preprocess.generate_demand(20)
    s_df = preprocess.generate_supply(20)
    print(d_df)
    print(s_df)