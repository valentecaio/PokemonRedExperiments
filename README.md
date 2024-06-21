# Speeding Up PPO Training for Pokémon Red

This repository is a fork of Peter Whidden's [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) that I used for a research about Deep RL for a master's degree project at PUC-Rio in 2024.  

My research aimed to improve the training speed of the PPO algorithm proposed originally by Peter Whidden. I was able to propose different model setups that could be trained in less than one day using only 12 cores and 16GB of RAM and were still able to beat the first gym of the game.  

More information can be found in [Speeding Up PPO Training for Pokémon Red](paper/Speeding%20Up%20PPO%20Training%20for%20Pokémon%20Red.pdf), and a video comparing models can be found [here](https://drive.google.com/file/d/1w_vO8Vj63JmTXHN6-RTA0cTdcWv71IGS/view?usp=sharing).

![models-comparison1.gif](https://github.com/valentecaio/PokemonRedExperiments/blob/master/assets/models-comparison-1.gif?raw=true)

---

After some time playing:  

![models-comparison2.gif](https://github.com/valentecaio/PokemonRedExperiments/blob/master/assets/models-comparison-2.gif?raw=true)

---

### Repository Structure
```
├── assets         # Images used in README files
├── game           # PokemonRed.rb ROM and initial game file state
├── original       # Code from original repository, with a separate README
├── paper          # LaTeX files for the paper
└── src            # Code for running and training models from this research
    ├── models     # Pretrained models
    └── sessions   # Training sessions (models being trained are saved here)
```

---

### Pokémon Red ROM
You need to have a legally obtained Pokémon Red ROM to run or train the models. The ROM should be placed in the `game/` directory and named `PokemonRed.gb`. The SHA1 sum of the ROM should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. 

---

### Dependencies
Python 3.10 is recommended. The dependencies can be installed with Conda by running:
```
cd src/
conda create --name poke python=3.10
conda activate poke
pip install -r requirements.txt
conda install -c conda-forge libffi libstdcxx-ng # extra dependencies
```

---

### Run models
To run the pretrained models available at `models/` run:
```
python run.py <model_directory>
python run.py 'models/Large Batch'  # example
```

---

### Train models
To train a model, run:  
```
python train.py
```
A new session will be created in the `sessions/` directory with the new model being trained.

Checkpoints are saved every once, which means you can pause model training at any time and resume later:
```
python train.py <session_directory>
python train.py sessions/20240621-100714/   # example
```

---

### Monitor training
To monitor the training of a model, move into the model's session directory and run:  
```
tensorboard --logdir .
```
You can then navigate to `localhost:6006` in your browser to view them.  

---

### Compare metrics of pretrained models
The metrics of pretrained models can be seen by running:  
```
cd models/ && tensorboard --logdir .
```

---

### Notes about models

1) The original pre-trained model (Baseline 439M) has a different action space. To run this model, you need to add the extra buttons in the `utils.py` file, function `get_default_env_config()`.  
```
  'extra_buttons': True,
```

2) The models that use dynamic batch size need a modified version of stable-baselines3 library that is not yet available in the official repository. You will not be able to run these models without it.

---

### Supporting Libraries
This work was made possible by the following awesome projects!
* [PyBoy emulator](https://github.com/Baekalfen/PyBoy)
* [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
* [Pokémon Red Disassembly Project](https://github.com/pret/pokered)
* And, of course, the original work from Peter Whidden: [Train RL agents to play Pokemon Red](https://github.com/PWhiddy/PokemonRedExperiments) 