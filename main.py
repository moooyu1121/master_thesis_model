import numpy as np 
import pandas as pd
import os
import glob
import warnings
warnings.simplefilter('ignore', FutureWarning)
from multiprocessing import Pool
import simulation


def main(num_agent, parent_dir, episode, load_q=False, train=True, **kwargs):
    params = {'thread_num': -1}  # dummy initial declaration
    params.update(kwargs)  # get the actual thread_num
    thread_num = params['thread_num']
    world = simulation.Simulation(num_agent, parent_dir, episode, train, **kwargs)
    if load_q and train:
        world.load_existing_q_table(folder_path=f'output/p2p/thread{thread_num}/episode{episode-1}/q_table')
    elif load_q and not train:
        # test simulation
        world.load_existing_q_table(folder_path=f'output/p2p/thread{thread_num}/episode{episode}/q_table')
    world.preprocess()
    world.run()
    world.save()
    if load_q and episode % 10 != 0:
        world.remove_existing_q_table(folder_path=f'output/p2p/thread{thread_num}/episode{episode-1}/q_table')


def main_no_p2p(num_agent, parent_dir, episode, load_q=False, train=True, **kwargs):
    world = simulation.SimulationNoP2P(num_agent, parent_dir, episode, train, **kwargs)
    params = {'thread_num': -1}
    params.update(kwargs)
    thread_num = params['thread_num']
    if load_q and train:        
        world.load_existing_q_table(folder_path=f'output/no_p2p/thread{thread_num}/episode{episode-1}/q_table')
    elif load_q and not train:
        # test simulation
        world.load_existing_q_table(folder_path=f'output/no_p2p/thread{thread_num}/episode{episode}/q_table')
    world.preprocess()
    world.run()
    world.save()
    if load_q and episode % 10 != 0:
        world.remove_existing_q_table(folder_path=f'output/no_p2p/thread{thread_num}/episode{episode-1}/q_table')
    

def main_wrapper(args):
    return main(**args)


def main_no_p2p_wrapper(args):
    return main_no_p2p(**args)


if __name__ == "__main__":
    max_workers = 16
    simulation_p2p = True
    simulation_no_p2p = True

    # params = {'thread_num': -1,
    #           'BID_SAVE': False,
    #           'price_max': 110,
    #           'price_min': 10,
    #           'wheeling_charge': 0,
    #           'battery_charge_efficiency': 0.9,
    #           'battery_discharge_efficiency': 0.9,
    #           'ev_charge_efficiency': 0.9,
    #           'ev_discharge_efficiency': 0.9,
    #           'battery_capacity_list': [0, 10, 15, 20],
    #           'ev_capacity_list': [40],
    #           'pv_capacity_list': [0, 5, 10],
    #           'discount_rate': 1.0,
    #           'learning_rate': 0.01,
    #           'shift_limit_list': [6.0, 12.0, 18.0, 24.0],  # hours
    #           'max_battery_charge_speed': [3.0],  # kW
    #           'max_battery_discharge_speed': [3.0],  # kW
    #           'max_ev_charge_speed': [6.0],  # kW
    #           'max_ev_discharge_speed': [3.0],  # kW
    #           'dr_boolean_list': [True, False],
    #           'alpha_list': [1, 1.5, 2, 2.5, 3, 3.5, 4],
    #           'beta_list': [1, 1.5, 2, 2.5, 3, 3.5, 4]
    #         }

    print('p2p:', simulation_p2p)
    print('no_p2p:', simulation_no_p2p)

    p = Pool(max_workers)

    # P2P <--- This is the main scenario
    if simulation_p2p:
        if not os.path.exists('output/p2p'):
            values = [{'num_agent': 100, 'episode': 1,'BID_SAVE': False, 'train': True, 'thread_num': x,  'load_q': False, 'parent_dir': 'output/p2p/thread'+str(x)+'/episode1'} for x in range(max_workers)]
            p.map(main_wrapper, values)

            p.close()
            p.join()

            print('p2p episode 1 finished.')

            for episode in range(2, 101):
                p = Pool(max_workers)
                values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': False, 'train': True, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/p2p/thread{x}/episode{episode}'} for x in range(max_workers)]
                p.map(main_wrapper, values)

                p.close()
                p.join()

                print(f'p2p episode {episode} finished.')

                if episode % 10 == 0:
                    print('Running test...')
                    p = Pool(max_workers)
                    if episode == 100:
                        values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': True, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/p2p/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                    else:
                        values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': False, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/p2p/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                    p.map(main_wrapper, values)

                    p.close()
                    p.join()

                    print(f'Test @ episode {episode} finished.')
            print('All episodes finished.')
        else:
            # Search for existing episodes
            existing_episodes = set()
            for folder in glob.glob('output/p2p/thread*/episode*'):
                episode = int(folder.split('episode')[-1])
                existing_episodes.add(episode)
            # remove the last episode, because it has not finished yet. 
            max_number = max(existing_episodes)
            existing_episodes.remove(max_number)
            print('Existing episodes:')
            print(existing_episodes)
            for episode in range(1, 101):
                if episode not in existing_episodes:
                    episide = episode -1  # Load the previous episode, because the current episode has not finished yet.
                    p = Pool(max_workers)
                    values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': False, 'train': True, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/p2p/thread{x}/episode{episode}'} for x in range(max_workers)]
                    p.map(main_wrapper, values)

                    p.close()
                    p.join()

                    print(f'p2p episode {episode} finished.')

                    if episode % 10 == 0:
                        print('Running test...')
                        p = Pool(max_workers)
                        if episode == 100:
                            values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': True, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/p2p/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                        else:
                            values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': False, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/p2p/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                        p.map(main_wrapper, values)

                        p.close()
                        p.join()

                        print(f'Test @ episode {episode} finished.')       
            print('All episodes finished.')
        print('P2P scenario finished.')

        
    # No P2P <--- This is the BAU scenario
    if simulation_no_p2p:
        if not os.path.exists('output/no_p2p'):
            values = [{'num_agent': 100, 'episode': 1, 'BID_SAVE': False, 'train': True, 'thread_num': x,  'load_q': False, 'parent_dir': 'output/no_p2p/thread'+str(x)+'/episode1'} for x in range(max_workers)]
            p.map(main_no_p2p_wrapper, values)

            p.close()
            p.join()

            print('no_p2p episode 1 finished.')

            for episode in range(2, 101):
                p = Pool(max_workers)
                values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': False, 'train': True, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/no_p2p/thread{x}/episode{episode}'} for x in range(max_workers)]
                p.map(main_no_p2p_wrapper, values)

                p.close()
                p.join()

                print(f'no_p2p episode {episode} finished.')

                if episode % 10 == 0:
                    print('Running test...')
                    p = Pool(max_workers)
                    if episode == 100:
                        values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': True, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/no_p2p/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                    else:
                        values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': False, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/no_p2p/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                    p.map(main_no_p2p_wrapper, values)

                    p.close()
                    p.join()

                    print(f'Test @ episode {episode} finished.')
            print('All episodes finished.')
        else:
            # Search for existing episodes
            existing_episodes = set()
            for folder in glob.glob('output/no_p2p/thread*/episode*'):
                episode = int(folder.split('episode')[-1])
                existing_episodes.add(episode)
            # remove the last episode, because it has not finished yet.
            max_number = max(existing_episodes)
            existing_episodes.remove(max_number)
            print('Existing episodes:')
            print(existing_episodes)
            for episode in range(1, 101):
                if episode not in existing_episodes:
                    episide = episode -1  # Load the previous episode, because the current episode has not finished yet.
                    p = Pool(max_workers)
                    values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': False, 'train': True, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/no_p2p/thread{x}/episode{episode}'} for x in range(max_workers)]
                    p.map(main_no_p2p_wrapper, values)

                    p.close()
                    p.join()

                    print(f'no_p2p episode {episode} finished.')

                    if episode % 10 == 0:
                        print('Running test...')
                        p = Pool(max_workers)
                        if episode == 100:
                            values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': True, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/no_p2p/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                        else:
                            values = [{'num_agent': 100, 'episode': episode, 'BID_SAVE': False, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/no_p2p/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                        p.map(main_no_p2p_wrapper, values)

                        p.close()
                        p.join()

                        print(f'Test @ episode {episode} finished.')
            print('All episodes finished.')

        print('No-P2P scenario finished.')
