import pandas as pd
import random


class Preprocess:
    def __init__(self, seed=42) -> None:
        self.demand_df = pd.DataFrame()
        self.supply_df = pd.DataFrame()
        self.price_df = pd.DataFrame()
        self.car_movement_df = pd.DataFrame()
        self.seed = seed

    def set(self, demand_df:pd.DataFrame, supply_df:pd.DataFrame, price_df:pd.DataFrame, car_movement_df:pd.DataFrame):
        self.demand_df = demand_df
        self.demand_df['timestamp'] = pd.to_datetime(self.demand_df['timestamp'])
        self.demand_df.set_index('timestamp', inplace=True)
        self.supply_df = supply_df
        self.supply_df['timestamp'] = pd.to_datetime(self.supply_df['timestamp'])
        self.supply_df.set_index('timestamp', inplace=True)
        self.price_df = price_df
        self.price_df['timestamp'] = pd.to_datetime(self.price_df['timestamp'])
        self.price_df.set_index('timestamp', inplace=True)
        self.car_movement_df = car_movement_df
        self.car_movement_df['timestamp'] = pd.to_datetime(self.car_movement_df['timestamp'])
        self.car_movement_df.set_index('timestamp', inplace=True)
        self.elastic_ratio_df = pd.read_csv('data/elastic_ratio.csv')
        self.elastic_ratio_df['timestamp'] = pd.to_datetime(self.elastic_ratio_df['timestamp'])
        self.elastic_ratio_df.set_index('timestamp', inplace=True)

    @property
    def get_dfs_(self):
        return self.demand_df, self.supply_df, self.price_df, self.car_movement_df, self.elastic_ratio_df
    
    @property
    def get_nparrays_(self):
        return self.demand_df.to_numpy(), self.supply_df.to_numpy(), self.price_df.to_numpy(), self.car_movement_df.to_numpy(), self.elastic_ratio_df.to_numpy()
    
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
        # とりあえず8kWの容量のPVを導入するとして供給を生成する
        self.supply_df = self.supply_df.iloc[:, columns] * 8
        self.supply_df.columns = [f'{i}' for i in range(n)]
        return self.supply_df
    
    def generate_supply_flex_pv_size(self, n, pv_capacity_list):
        num_columns = self.supply_df.shape[1]
        columns = generate_random_integers(num_columns, n, seed=self.seed)
        self.supply_df = self.supply_df.iloc[:, columns] * pv_capacity_list
        self.supply_df.columns = [f'{i}' for i in range(n)]
        return self.supply_df
    
    def generate_d_s(self, n):
        self.generate_demand(n)
        self.generate_supply(n)
        return self.demand_df, self.supply_df
    
    def generate_car_movement(self, n):
        mileage_categories = ["-3000", "3000-5000", "5000-7000", "7000-9000", "9000-11000", "11000-16000", "16000-"]
        num_columns = self.car_movement_df.shape[1]
        columns = generate_random_integers(num_columns, n, seed=self.seed)
        self.car_movement_df = self.car_movement_df.iloc[:, columns]
        self.car_movement_df.columns = [f'{i}' for i in range(n)]
        agent_car_categories = []
        for column in columns:
            if column < 129:
                agent_car_categories.append(mileage_categories[0])
            elif column < 387:
                agent_car_categories.append(mileage_categories[1])
            elif column < 612:
                agent_car_categories.append(mileage_categories[2])
            elif column < 760:
                agent_car_categories.append(mileage_categories[3])
            elif column < 898:
                agent_car_categories.append(mileage_categories[4])
            elif column < 958:
                agent_car_categories.append(mileage_categories[5])
            else:
                agent_car_categories.append(mileage_categories[6])
        return self.car_movement_df, agent_car_categories
    
    @property
    def drop_index_(self):
        self.demand_df.reset_index(inplace=True, drop=True)
        self.supply_df.reset_index(inplace=True, drop=True)
        self.price_df.reset_index(inplace=True, drop=True)
        self.car_movement_df.reset_index(inplace=True, drop=True)
        self.elastic_ratio_df.reset_index(inplace=True, drop=True)
        return self.demand_df, self.supply_df, self.price_df, self.car_movement_df, self.elastic_ratio_df
    
    def save(self, folder_path):
        self.demand_df.to_csv(f'{folder_path}/demand.csv', index=True)
        self.supply_df.to_csv(f'{folder_path}/supply.csv', index=True)
        self.price_df.to_csv(f'{folder_path}/price.csv', index=True)
        self.car_movement_df.to_csv(f'{folder_path}/car_movement.csv', index=True)


# ランダムな整数を生成する関数(重複あり)
def generate_random_integers(num_columns, n, seed=42):
    random.seed(seed)
    # return random.sample(range(num_columns), n)
    return random.choices(range(num_columns), k=n)


if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.set(pd.read_csv('data/demand.csv'), pd.read_csv('data/supply.csv'), pd.read_csv('data/price.csv'), pd.read_csv('data/car_movement.csv'))
    d_df = preprocess.generate_demand(20)
    s_df = preprocess.generate_supply(20)
    print(d_df)
    print(s_df)
    car_movement_df, agent_car_categories = preprocess.generate_car_movement(20)
    print(car_movement_df)
    print(agent_car_categories)
    s_df = preprocess.generate_supply_flex_pv_size(5, [0, 10, 12, 14, 16])
    print(s_df.head(50))