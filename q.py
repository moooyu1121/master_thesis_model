class Q:
    def __init__(self, agents_params_df):
        self.agents_params_df = agents_params_df



    def battery_bids(self, agent_num, price_min, price_max):
        battery_capacity = self.agent_params_df.at[agent_num, 'battery_capacity']
        sell_price = (price_min + price_max) / 2
        buy_price = price_min + 0.01
        return 