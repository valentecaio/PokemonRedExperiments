from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

from stable_baselines3.common.policies import ActorCriticCnnPolicy
import torch as th
import sys

# local imports
from utils import get_session_path, get_default_env_config, make_env
from scheduler import sched_exp_inc, sched_log_dec, sched_linear_inc, sched_exp_dec


# used to change optimizer parameters
class PokeCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(PokeCnnPolicy, self).__init__(
            optimizer_class=th.optim.Adam,
            # optimizer_kwargs={'weight_decay': 1e-4},
            *args, **kwargs)


if __name__ == '__main__':
    ######################### HYPERPARAMETERS #########################

    # constants
    n_cpus = 12                         # how many episodes in parallel. Also sets the number of episodes per training iteration.
    len_episodes = 2048 * 12            # episode length, number of steps per episode. Each CPU resets its game after this many steps.
    n_steps = len_episodes // 12        # epoch length (or batch_size per cpu), number of steps per training iteration (train 12x per episode), smaller value = train more often, with less data
    total_ts = len_episodes*n_cpus*233  # global total number of training steps. summing all cpus
    print(f'n_cpus: {n_cpus}')
    print(f'len_episodes: {len_episodes}')
    print(f'n_steps: {n_steps}')
    print(f'total_ts: {total_ts}')

    # dynamic minibatch size and learning rate
    batch_size_initial = 32
    batch_size_final = 2048 * 3 # the max between: your GPU memory, rollout_buffer_size (= n_steps*n_cpus)
    lr_initial = 0.005 # 5e-3
    lr_final = 0.00005 # 5e-5
    # print(f'batch_size: {batch_size_initial} to {batch_size_final} increasing exponentially')
    # print(f'learning_rate: {lr_initial:.2e} to {lr_final:.2e} decreasing logarithmically')



    ######################### SESSION LOAD / ENV CREATION #########################

    # if you want to start from a checkpoint, pass it as arg or put it here
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else ''
    # checkpoint = 'sessions/20240531-135214' # QDN lr 1e-4 100k buffer 10k start 1.0 tau 10000 target 0.1 explore 1.0-0.01
    # checkpoint = 'sessions/20240531-163421' # QDN lr 1e-4 100k buffer 10k start 1.0 tau 10000 target 0.1 explore 1.0-0.01 NO PASS
    # checkpoint = 'sessions/20240531-222646' # PPO CNN 128,64,64, n_epochs=3, ent_coef=0.02, lr=1e-3
    # checkpoint = 'sessions/4da05e87_main_good/' # pretrained model, needs extra_buttons = True
    # checkpoint = 'sessions/20240528-205506' # adamW ep_length 2048*16, 'weight_decay': 1e-4, batch_size=512, ent_coef=0.02, NO PASS # 110k steps - passed gym 1
    # checkpoint = 'sessions/20240601-140035' # big batch (2048*6)
    # checkpoint = 'sessions/20240602-115912' # original (lr 3e-4, batch 128, ep_length 2048*12)
    # checkpoint = 'sessions/20240603-215221' # dynamic lr, dynamic batch
    # checkpoint = 'sessions/20240604-235742' # dynamic lr, small batch (128)
    # checkpoint = 'sessions/20240605-213933' # dynamic lr, big batch (2048*6)
    # checkpoint = "sessions/20240606-191654" # AdamW weight_decay 1e-4
    # checkpoint = "sessions/20240607-170833" # big batch (2048*6), ent_coef 0.02

    sess_path, load_checkpoint, checkpoint_filename = get_session_path(checkpoint)
    print(f'sess_path: {sess_path}')
    print(f'load_checkpoint: {load_checkpoint}')

    env_config = get_default_env_config(sess_path, checkpoint_filename)
    env_config['max_steps'] = len_episodes
    print(f'env_config: {env_config}')

    # create envs
    env = SubprocVecEnv([make_env(i, env_config) for i in range(n_cpus)])
    checkpoint_callback = CheckpointCallback(save_freq=len_episodes//4, save_path=sess_path, name_prefix='poke', save_replay_buffer=True, save_vecnormalize=True)
    callbacks = [checkpoint_callback, TensorboardCallback()]



    ######################### MODEL LOAD / CREATION #########################

    # model creation / loading
    if load_checkpoint:
        print(f'\nloading checkpoint {checkpoint_filename}')
        # model = DQN.load(checkpoint_filename, env=env, print_system_info=True)
        model = PPO.load(checkpoint_filename, env=env, print_system_info=True)
        total_ts = total_ts - model.num_timesteps
        # model.rollout_buffer.buffer_size = model.n_steps = len_episodes
        # model.rollout_buffer.n_envs      = model.n_envs  = n_cpus
        # model.rollout_buffer.reset()
    else:
        # original code
        # model = PPO('CnnPolicy', env, verbose=1, n_steps=n_steps, batch_size=128,
        #             n_epochs=3, gamma=0.998, tensorboard_log=sess_path, device='auto')

        model = PPO(
            PokeCnnPolicy,
            env,
            verbose=2,
            n_steps=n_steps, # define training frequency, as rollout buffer size is n_steps*n_cpus
            n_epochs=3,      # original, smaller than default for faster training
            gamma=0.998,     # original
            tensorboard_log=sess_path,
            device='auto',

            # ent_coef=0.02, # entropy coefficient, higher = more exploration

            # batch_size=n_steps*6, # size of batch subset to use for training
            # batch_size=sched_exp_inc(batch_size_initial, batch_size_final, total_ts),

            # learning_rate=sched_log_dec(lr_initial, lr_final, total_ts),

            # policy_kwargs=dict(
            #     # future tests
            #     net_arch = dict(pi=[512, 256, 128], vf=[512, 256, 128]),
            # ),
        )


    ######################### TRAIN #########################

    model.learn(total_timesteps=total_ts, callback=CallbackList(callbacks), log_interval=1, reset_num_timesteps=False)
