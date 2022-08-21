# Smort-Snake

## ------- TRAINING -------

For training with NEAT run
```
python neatAgent.py
```

For training with DQN run
```
python agent.py
```


## ------- TESTING TRAINED NETS -------

For testing already trained DQN network run
```
python trainedNetTest.py best_instance_dqn.pth
```

For testing already trained NEAT network run
```
python trainedNetTest.py best_instance_neat.pickle
```


## ------- SAVE FILES -------

Graphs from training are saved to ./Graphs folder under appropriate name (dqn_scores.png or neat_scores.png)   
Trained networks are saved to ./neural-net folder. DQN net is saved in pth format while NEAT is saved in pickle format.
