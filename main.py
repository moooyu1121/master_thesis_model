import numpy as np 
import pandas as pd
import os
import glob
import warnings
warnings.simplefilter('ignore', FutureWarning)
from multiprocessing import Pool
from simulation import Simulation


def main(num_agent, parent_dir, episode, load_q=False, train=True, **kwargs):
    world = Simulation(num_agent, parent_dir, episode, train, **kwargs)
    if load_q:
        params = {'thread_num': -1}
        params.update(kwargs)
        thread_num = params['thread_num']
        world.load_existing_q_table(folder_path=f'output/thread{thread_num}/episode{episode-1}/q_table')
    world.preprocess()
    world.run()
    world.save()
    

def main_wrapper(args):
    return main(**args)


if __name__ == "__main__":
    max_workers = 16
    p = Pool(max_workers)
    if not os.path.exists('output'):
        values = [{'num_agent': 100, 'episode': 1, 'price_min': 10, 'BID_SAVE': False, 'train': True, 'thread_num': x,  'load_q': False, 'parent_dir': 'output/thread'+str(x)+'/episode1'} for x in range(max_workers)]
        p.map(main_wrapper, values)

        p.close()
        p.join()

        print('episode 1 finished.')

        for episode in range(2, 101):
            p = Pool(max_workers)
            values = [{'num_agent': 100, 'episode': episode, 'price_min': 10, 'BID_SAVE': False, 'train': True, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/thread{x}/episode{episode}'} for x in range(max_workers)]
            p.map(main_wrapper, values)

            p.close()
            p.join()

            print(f'episode {episode} finished.')

            if episode % 10 == 0:
                print('Running test...')
                p = Pool(max_workers)
                values = [{'num_agent': 100, 'episode': episode, 'price_min': 10, 'BID_SAVE': True, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                p.map(main_wrapper, values)

                p.close()
                p.join()

                print(f'Test @ episode {episode} finished.')
        print('All episodes finished.')
    else:
        # Search for existing episodes
        existing_episodes = set()
        for folder in glob.glob('output/thread*/episode*'):
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
                values = [{'num_agent': 100, 'episode': episode, 'price_min': 10, 'BID_SAVE': False, 'train': True, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/thread{x}/episode{episode}'} for x in range(max_workers)]
                p.map(main_wrapper, values)

                p.close()
                p.join()

                print(f'episode {episode} finished.')

                if episode % 10 == 0:
                    print('Running test...')
                    p = Pool(max_workers)
                    values = [{'num_agent': 100, 'episode': episode, 'price_min': 10, 'BID_SAVE': True, 'train': False, 'thread_num': x, 'load_q': True, 'parent_dir': f'output/test/thread{x}/episode{episode}'} for x in range(max_workers)]
                    p.map(main_wrapper, values)

                    p.close()
                    p.join()

                    print(f'Test @ episode {episode} finished.')
        print('All episodes finished.')

