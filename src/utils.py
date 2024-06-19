from os import path
from glob import glob
from pathlib import Path
from datetime import datetime
import re

from stable_baselines3.common.utils import set_random_seed
from red_gym_env import RedGymEnv

# read step count from filename
def previous_step_count(filename):
    pattern = r'poke_(.*?)_steps\.zip'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)

# read last checkpoint file in given path
def get_last_checkpoint(dir):
    search_pattern = path.join(dir, "poke_*_steps.zip")
    files = glob(search_pattern)
    if not files:
        return None
    latest_file = max(files, key=path.getctime)
    return latest_file

# try to load session in given path or create a new session
def get_session_path(dir = None):
    filename = get_last_checkpoint(dir) if dir else None
    if filename is not None:
        sess_path = Path(dir)
        load_checkpoint = True
    else:
        sess_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        sess_path = Path(f'sessions/{sess_id}')
        load_checkpoint = False
    return sess_path, load_checkpoint, filename

# default environment configuration
def get_default_env_config(sess_path, filename = None):
    return {
        'headless': True,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'gb_path': '../game/PokemonRed.gb',
        'init_state': '../game/has_pokedex_nballs.state',
        'max_steps': 0, # to define in main script 
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True,
        'reward_scale': 4,
        'extra_buttons': False,
        'log_every_n_steps': 1000,
        'frame_every_n_steps': 200,
        'explore_weight': 3, # 2.5
        'step_count': previous_step_count(filename) if filename else 0,
        'clean_images_on_reset': True,
    }

# returns true if agent_enabled.txt file exists and contains 1
def is_agent_enabled():
    try:
        with open('agent_enabled.txt', 'r') as f:
            return f.read().strip() == '1'
    except:
        return False

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init