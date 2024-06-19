### Dependencies
Python 3.10 is recommended. The dependencies can be installed by running the following commands:
```
  cd src/
  conda create --name poke python=3.10
  conda activate poke
  pip install -r requirements.txt
  conda install -c conda-forge libffi libstdcxx-ng # extra dependencies
```

### Run models
To run the pretrained models available at models/, run the following command:  
```python3 run.py <model_dir>```
Where `<model_dir>` is the directory of the model you would like to run.
Example:  
```python3 run.py models/AdamW```

### Train models
To train a model, run the following command:  
```python3 train.py```
A new session will be created in the sessions/ directory with the current date and time.

To resume training a model, run the following command:  
```python3 train.py <session_dir>```
Where `<session_dir>` is the directory of the session you would like to resume training.

### Monitor training
To monitor the training of a model, move into the session directory and run:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view them.  

### Compare metrics of pretrained models
The metrics of pretrained models can be seen by running:  
```cd models/ && tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view them.  

### Note about models

The pre-trained model has a different action space. To run it, you need to add the extra buttons in the `utils.py` file, function `get_default_env_config()`.  
```
  'extra_buttons': True,
```

The models that use dynamic batch size need a modified version of stable-baselines3 library that is not yet available in the official repository. 
You will not be able to run these models without it.

