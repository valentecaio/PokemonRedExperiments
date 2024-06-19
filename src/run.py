from stable_baselines3 import A2C, PPO, DQN
import sys
import time

# local imports
from utils import get_session_path, get_default_env_config, make_env, is_agent_enabled


if __name__ == '__main__':
    # to load a model, pass it as arg or put it here
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else ''
    # checkpoint = 'models/Baseline 439M'               # 0: original pre trained model (needs extra_buttons = True in utils.py)
    # checkpoint = 'models/Baseline 68M'                # 1: original (lr 3e-4, batch 128, ep_length 2048*12)
    # checkpoint = "models/AdamW"                       # 2: AdamW weight_decay 1e-4
    # checkpoint = 'models/Dynamic LR'                  # 3: dynamic lr, small batch (128)
    # checkpoint = 'models/Large Batch'                 # 4: big batch (2048*6)
    # checkpoint = 'models/Large Batch + Dynamic LR'    # 5: dynamic lr, big batch (2048*6)
    # checkpoint = "models/Large Batch + Entropy"       # 6: big batch (2048*6), ent_coef 0.02
    # checkpoint = 'models/Dynamic LR + Dynamic batch'  # 7: dynamic lr, dynamic batch (needs modified version of PPO)

    sess_path, load_checkpoint, checkpoint_filename = get_session_path(checkpoint)
    if not load_checkpoint:
        print(f'Model not found! (checkpoint = "{checkpoint}")\nPass model as argument or edit file:\n{__file__}')
        exit(1)

    env_config = get_default_env_config(sess_path, checkpoint_filename)
    env_config['headless'] = False              # show game window
    env_config['clean_images_on_reset'] = False # avoid deleting training images
    env_config['log_every_n_steps'] = 1         # verbosity, log every step
    env_config['max_steps'] = 2e23              # run forever
    env = make_env(0, env_config)()

    print(f'Loading checkpoint {checkpoint_filename}')
    model = PPO.load(checkpoint_filename, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
    # model = DQN.load(checkpoint_filename, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    obs, info = env.reset()
    while True:
        # edit agent_enabled.txt to enable/disable agent
        if is_agent_enabled():
            # agent plays the game
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, terminated, truncated, info = env.step(action)
            env.render()
            if truncated:
                break
        else:
            # human plays the game
            env.step(None)
            env.render()
            # time.sleep(0.5)
    env.close()
