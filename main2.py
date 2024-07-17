import numpy as np 
import pandas as pd
import os
import glob
import warnings
warnings.simplefilter('ignore', FutureWarning)
from multiprocessing import Pool
from simulation import Simulation


def main(num_agent, parent_dir, episode, load_q=False, **kwargs):
    world = Simulation(num_agent, parent_dir, episode, **kwargs)
    if load_q:
        params = {'thread_num': -1}
        params.update(kwargs)
        thread_num = params['thread_num']
        world.load_existing_q_table(folder_path=f'output/thread{thread_num}/episode{episode}/q_table')
    world.preprocess()
    world.run()
    world.save()
    

def main_wrapper(args):
    return main(**args)


if __name__ == "__main__":
    max_workers = 16
    
    p = Pool(max_workers)
    values = [{'num_agent': 50, 'episode': 1, 'price_min': 10, 'BID_SAVE': False, 'thread_num': x, 'parent_dir': 'output/thread'+str(x)+'/episode1/'} for x in range(max_workers)]
    p.map(main_wrapper, values)

    p.close()
    p.join()

    print('episode 1 finished.')

    for episode in range(2, 101):
        p = Pool(max_workers)
        values = [{'num_agent': 50, 'episode': episode, 'price_min': 10, 'BID_SAVE': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/thread{x}/episode{episode}/'} for x in range(max_workers)]
        p.map(main_wrapper, values)

        p.close()
        p.join()

        print(f'episode {episode} finished.')
    print('All episodes finished.')
