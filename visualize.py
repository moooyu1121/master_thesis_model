import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np 


class Visualize:
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path
        self.grid_price_df = pd.read_csv(folder_path + '/price.csv', index_col=0)
        self.microgrid_price_df = pd.read_csv(folder_path + '/price_record.csv', index_col=0)
        self.battery_soc_df = pd.read_csv(folder_path + '/battery_soc_record.csv', index_col=0)
        self.ev_battery_soc_df = pd.read_csv(folder_path + '/ev_battery_soc_record.csv', index_col=0)

        self.buy_inelastic_df = pd.read_csv(folder_path + '/buy_inelastic_record.csv', index_col=0)
        self.buy_elastic_df = pd.read_csv(folder_path + '/buy_elastic_record.csv', index_col=0)
        self.buy_shifted_df = pd.read_csv(folder_path + '/buy_shifted_record.csv', index_col=0)
        self.sell_pv_df = pd.read_csv(folder_path + '/sell_record.csv', index_col=0)

        self.buy_battery_df = pd.read_csv(folder_path + '/buy_battery_record.csv', index_col=0)
        self.buy_ev_battery_df = pd.read_csv(folder_path + '/buy_ev_battery_record.csv', index_col=0)
        self.sell_battery_df = pd.read_csv(folder_path + '/sell_battery_record.csv', index_col=0)
        self.sell_ev_battery_df = pd.read_csv(folder_path + '/sell_ev_battery_record.csv', index_col=0)

        timestamp_df = pd.read_csv('data/demand.csv')
        timestamp_df['timestamp'] = pd.to_datetime(timestamp_df['timestamp'])
        self.timestamps = timestamp_df['timestamp']
        self.original_demand_df = pd.read_csv(folder_path + '/demand.csv', index_col=0)
        self.original_pv_supply_df = pd.read_csv(folder_path + '/supply.csv', index_col=0)

    def plot_consumption(self):
        # fig = go.Figure()
        # Figureオブジェクトを作成し、2行1列のサブプロットを設定
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)

        # Plot original demand line
        fig.add_trace(go.Scatter(
        x=self.timestamps, y=self.original_demand_df.sum(axis=1),
        mode='lines',
        name='Original demand',
        line=dict(dash='dash'),
        visible=True
        ), row=1, col=1)
        # Plot original pv supply line
        fig.add_trace(go.Scatter(
        x=self.timestamps, y=-self.original_pv_supply_df.sum(axis=1),
        mode='lines',
        name='PV generation',
        line=dict(dash='dash'),
        visible=True
        ), row=1, col=1)

        # Stack plot for lower part -> sell
        fig.add_trace(go.Scatter(
        x=self.timestamps, y=-self.sell_pv_df.sum(axis=1), 
        mode='lines', 
        name='PV supply',
        stackgroup='down',
        visible=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
        x=self.timestamps, y=-self.sell_ev_battery_df.sum(axis=1),
        mode='lines',
        name='EV discharge',
        stackgroup='down',
        visible=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
        x=self.timestamps, y=-self.sell_battery_df.sum(axis=1),
        mode='lines',
        name='Battery discharge',
        stackgroup='down',
        visible=True
        ), row=1, col=1)

        # Stack plot for upper part -> buy
        fig.add_trace(go.Scatter(
        x=self.timestamps, y=self.buy_inelastic_df.sum(axis=1), 
        mode='lines', 
        name='Inelastic consumption',
        stackgroup='up',
        visible=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
        x=self.timestamps, y=self.buy_elastic_df.sum(axis=1), 
        mode='lines', 
        name='Elastic consumption',
        stackgroup='up',
        visible=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
        x=self.timestamps, y=self.buy_shifted_df.sum(axis=1),
        mode='lines',
        name='Shifted consumption',
        stackgroup='up',
        visible=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
        x=self.timestamps, y=self.buy_ev_battery_df.sum(axis=1),
        mode='lines',
        name='EV charge',
        stackgroup='up',
        visible=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
        x=self.timestamps, y=self.buy_battery_df.sum(axis=1),
        mode='lines',
        name='Battery charge',
        stackgroup='up',
        visible=True
        ), row=1, col=1)

        for agent in self.buy_inelastic_df.columns:
            # Plot original demand line
            fig.add_trace(go.Scatter(
            x=self.timestamps, y=self.original_demand_df[agent],
            mode='lines',
            name='Original demand',
            line=dict(dash='dash'),
            visible=False
            ), row=1, col=1)
            # Plot original pv supply line
            fig.add_trace(go.Scatter(
            x=self.timestamps, y=-self.original_pv_supply_df[agent],
            mode='lines',
            name='PV generation',
            line=dict(dash='dash'),
            visible=False
            ), row=1, col=1)

            # Stack plot for lower part -> sell
            fig.add_trace(go.Scatter(
            x=self.timestamps, y=-self.sell_pv_df[agent], 
            mode='lines', 
            name='PV supply',
            stackgroup='down',
            visible=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
            x=self.timestamps, y=-self.sell_ev_battery_df[agent],
            mode='lines',
            name='EV discharge',
            stackgroup='down',
            visible=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
            x=self.timestamps, y=-self.sell_battery_df[agent],
            mode='lines',
            name='Battery discharge',
            stackgroup='down',
            visible=False
            ), row=1, col=1)

            # Stack plot for upper part -> buy
            fig.add_trace(go.Scatter(
            x=self.timestamps, y=self.buy_inelastic_df[agent], 
            mode='lines', 
            name='Inelastic consumption',
            stackgroup='up',
            visible=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
            x=self.timestamps, y=self.buy_elastic_df[agent], 
            mode='lines', 
            name='Elastic consumption',
            stackgroup='up',
            visible=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
            x=self.timestamps, y=self.buy_shifted_df[agent],
            mode='lines',
            name='Shifted consumption',
            stackgroup='up',
            visible=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
            x=self.timestamps, y=self.buy_ev_battery_df[agent],
            mode='lines',
            name='EV charge',
            stackgroup='up',
            visible=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
            x=self.timestamps, y=self.buy_battery_df[agent],
            mode='lines',
            name='Battery charge',
            stackgroup='up',
            visible=False
            ), row=1, col=1)

        
        fig.add_trace(go.Scatter(
            x=self.timestamps, y=self.microgrid_price_df['Price'],
            mode='lines',
            name='Microgrid price',
            visible=True
            ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.timestamps, y=self.grid_price_df['Price'],
            mode='lines',
            name='Grid price',
            line=dict(dash='dash'),
            visible=True
            ), row=2, col=1)


        # ドロップダウンボタンの作成
        dropdown_buttons = []
        dropdown_buttons.append(
                dict(
                    args=[{'visible': [False] * len(fig.data)}], # type: ignore
                    label='sum',
                    method='update'
                )
            )
        for j in range(10):
            dropdown_buttons[-1]['args'][0]['visible'][j] = True
        dropdown_buttons[-1]['args'][0]['visible'][-2] = True  # show microgrid price
        dropdown_buttons[-1]['args'][0]['visible'][-1] = True  # show grid price

        
        for i, user in enumerate(self.buy_inelastic_df.columns):
            dropdown_buttons.append(
                dict(
                    args=[{'visible': [False] * len(fig.data)}], # type: ignore
                    label='agent ' + user,
                    method='update'
                )
            )
            for j in range(10):
                dropdown_buttons[-1]['args'][0]['visible'][10*(i+1) + j] = True
            dropdown_buttons[-1]['args'][0]['visible'][-2] = True  # show microgrid price
            dropdown_buttons[-1]['args'][0]['visible'][-1] = True  # show grid price
        

        # レイアウトの設定
        fig.update_layout(
            title='Consumption and generation',
            xaxis_title='Time',
            yaxis_title='kWh',
            xaxis=dict(
                tickformat='%Y-%m-%d %H',
                rangeslider=dict(
                    visible=True
                ),
                type='date'
            ),
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    x=0.5,
                    xanchor='center',
                    y=1.2,
                    yanchor='top'
                )
            ]
        )
        fig.update_xaxes(title_text='Time', row=1, col=1)
        fig.update_yaxes(title_text='kWh', row=1, col=1)

        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='cents/kWh', row=2, col=1)
        
        # HTMLファイルとして保存
        fig.write_html(self.folder_path + "/consumption_generation_plot.html")
                      

if __name__ == '__main__':
    Visualize(folder_path='output/避難').plot_consumption()
    # print(pd.read_csv('output/buy_inelastic_record.csv', index_col=0))