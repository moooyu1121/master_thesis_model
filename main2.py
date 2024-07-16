import numpy as np 
import pandas as pd
import os
import glob
import warnings
warnings.simplefilter('ignore', FutureWarning)
from multiprocessing import Pool
from simulation import Simulation


def main(num_agent, parent_dir, load_q=False, **kwargs):
    world = Simulation(num_agent, parent_dir)
    if load_q:
        world.load_q()
    

def main_wrapper(args):
    return main(**args)


if __name__ == "__main__":
    max_workers = 16
    
    p = Pool(max_workers)
    values = [{'num_agent': 50, 'episode': 1, 'price_min': 10, 'thread_num': x, 'parent_dir': 'output/thread'+str(x)+'/episode1/'} for x in range(max_workers)]
    p.map(main_wrapper, values)

    p.close()
    p.join()

    os.makedirs('output/average_q/1' + '/', exist_ok=True)
    dr_buy_qtb_list = glob.glob('output/*/episode1/dr_buy_qtb.npy')
    battery_buy_qtb_list = glob.glob('output/*/episode1/battery_buy_qtb.npy')
    battery_sell_qtb_list = glob.glob('output/*/episode1/battery_sell_qtb.npy')
    ev_battery_buy_qtb_list = glob.glob('output/*/episode1/ev_battery_buy_qtb.npy')
    ev_battery_sell_qtb_list = glob.glob('output/*/episode1/ev_battery_sell_qtb.npy')
    for thread in range(max_workers):
        dr_buy_qtb = np.load(dr_buy_qtb_list[thread])
        battery_buy_qtb = np.load(battery_buy_qtb_list[thread])
        battery_sell_qtb = np.load(battery_sell_qtb_list[thread])
        ev_battery_buy_qtb = np.load(ev_battery_buy_qtb_list[thread])
        ev_battery_sell_qtb = np.load(ev_battery_sell_qtb_list[thread])
        if thread == 0:
            average_dr_buy_qtb = dr_buy_qtb
            average_battery_buy_qtb = battery_buy_qtb
            average_battery_sell_qtb = battery_sell_qtb
            average_ev_battery_buy_qtb = ev_battery_buy_qtb
            average_ev_battery_sell_qtb = ev_battery_sell_qtb
        else:
            average_dr_buy_qtb += dr_buy_qtb
            average_battery_buy_qtb += battery_buy_qtb
            average_battery_sell_qtb += battery_sell_qtb
            average_ev_battery_buy_qtb += ev_battery_buy_qtb
            average_ev_battery_sell_qtb += ev_battery_sell_qtb
    average_dr_buy_qtb /= max_workers
    average_battery_buy_qtb /= max_workers
    average_battery_sell_qtb /= max_workers
    average_ev_battery_buy_qtb /= max_workers
    average_ev_battery_sell_qtb /= max_workers
    np.save('output/average_q/1/dr_buy_qtb.npy', average_dr_buy_qtb)
    np.save('output/average_q/1/battery_buy_qtb.npy', average_battery_buy_qtb)
    np.save('output/average_q/1/battery_sell_qtb.npy', average_battery_sell_qtb)
    np.save('output/average_q/1/ev_battery_buy_qtb.npy', average_ev_battery_buy_qtb)
    np.save('output/average_q/1/ev_battery_sell_qtb.npy', average_ev_battery_sell_qtb)
    df = pd.DataFrame(average_dr_buy_qtb)
    df.to_csv('output/average_q/1/dr_buy_qtb.csv')
    df = pd.DataFrame(average_battery_buy_qtb)
    df.to_csv('output/average_q/1/battery_buy_qtb.csv')
    df = pd.DataFrame(average_battery_sell_qtb)
    df.to_csv('output/average_q/1/battery_sell_qtb.csv')
    df = pd.DataFrame(average_ev_battery_buy_qtb)
    df.to_csv('output/average_q/1/ev_battery_buy_qtb.csv')
    df = pd.DataFrame(average_ev_battery_sell_qtb)
    df.to_csv('output/average_q/1/ev_battery_sell_qtb.csv')

    print('episode 1 finished.')

    for episode in range(2, 101):
        p = Pool(max_workers)
        values = [{'num_agent': 50, 'episode': episode, 'price_min': 10, 'thread_num': x, 'load_q': True} for x in range(max_workers)]
        p.map(main_wrapper, values)

        p.close()
        p.join()

        os.makedirs('output/average_q/' + str(episode) + '/', exist_ok=True)
        dr_buy_qtb_list = glob.glob('output/*/episode' + str(episode) + '/dr_buy_qtb.npy')
        battery_buy_qtb_list = glob.glob('output/*/episode' + str(episode) + '/battery_buy_qtb.npy')
        battery_sell_qtb_list = glob.glob('output/*/episode' + str(episode) + '/battery_sell_qtb.npy')
        ev_battery_buy_qtb_list = glob.glob('output/*/episode' + str(episode) + '/ev_battery_buy_qtb.npy')
        ev_battery_sell_qtb_list = glob.glob('output/*/episode' + str(episode) + '/ev_battery_sell_qtb.npy')

        for thread in range(max_workers):
            dr_buy_qtb = np.load(dr_buy_qtb_list[thread])
            battery_buy_qtb = np.load(battery_buy_qtb_list[thread])
            battery_sell_qtb = np.load(battery_sell_qtb_list[thread])
            ev_battery_buy_qtb = np.load(ev_battery_buy_qtb_list[thread])
            ev_battery_sell_qtb = np.load(ev_battery_sell_qtb_list[thread])
            if thread == 0:
                average_dr_buy_qtb = dr_buy_qtb
                average_battery_buy_qtb = battery_buy_qtb
                average_battery_sell_qtb = battery_sell_qtb
                average_ev_battery_buy_qtb = ev_battery_buy_qtb
                average_ev_battery_sell_qtb = ev_battery_sell_qtb
            else:
                average_dr_buy_qtb += dr_buy_qtb
                average_battery_buy_qtb += battery_buy_qtb
                average_battery_sell_qtb += battery_sell_qtb
                average_ev_battery_buy_qtb += ev_battery_buy_qtb
                average_ev_battery_sell_qtb += ev_battery_sell_qtb
        average_dr_buy_qtb /= max_workers
        average_battery_buy_qtb /= max_workers
        average_battery_sell_qtb /= max_workers
        average_ev_battery_buy_qtb /= max_workers
        average_ev_battery_sell_qtb /= max_workers
        np.save('output/average_q/' + str(episode) + '/dr_buy_qtb.npy', average_dr_buy_qtb)
        np.save('output/average_q/' + str(episode) + '/battery_buy_qtb.npy', average_battery_buy_qtb)
        np.save('output/average_q/' + str(episode) + '/battery_sell_qtb.npy', average_battery_sell_qtb)
        np.save('output/average_q/' + str(episode) + '/ev_battery_buy_qtb.npy', average_ev_battery_buy_qtb)
        np.save('output/average_q/' + str(episode) + '/ev_battery_sell_qtb.npy', average_ev_battery_sell_qtb)
        df = pd.DataFrame(average_dr_buy_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/dr_buy_qtb.csv')
        df = pd.DataFrame(average_battery_buy_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/battery_buy_qtb.csv')
        df = pd.DataFrame(average_battery_sell_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/battery_sell_qtb.csv')
        df = pd.DataFrame(average_ev_battery_buy_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/ev_battery_buy_qtb.csv')
        df = pd.DataFrame(average_ev_battery_sell_qtb)
        df.to_csv('output/average_q/' + str(episode) + '/ev_battery_sell_qtb.csv')
        print(f'episode {episode} finished.')
    print('All episodes finished.')
