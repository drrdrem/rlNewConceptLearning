# Usage - 
* a) Given a new concept, the agent will keep recommending a reference to read until reaching an unknown target.
* b) This can be seen as the shortest path from an initial reference to an unknown target reference.
    * b.1) The initial reference (state) is recommended based on a global recommendation system.
* c) The graph is constructed with a global reference embedding. (Veteices: references; Edges: k nearest neighbors of a reference)

If you are interested in or use this code, please cite the following paper:  
[Learning to Construct Knowledge through Sparse Reference Selection with Reinforcement Learning](https://your-url-here.com)

# Train a model - 
## Baseline
* Build a dataset instance with torchtext (Please See Baseline_model.ipynb).
* import baseline/torch/train.py to train a Baseline Model.

## RL Agent - 
* Build a dataset instance with torchtext (Please See RL_model.ipynb).
* import RL/agent/REINFORCE or RL/agent/A2C to create an agent.
* import RUN.py and set train=True to train a model.

# Inference - 
## Baseline
* load a dataset instance with torchtext.
* import baseline/torch/train.py to run a Baseline Model.
* Please see Baseline_model.ipynb for the pre-trained model results.

## RL Agent - 
* load a dataset instance with torchtext.
* load a pre-trained agent.
* import RUN.py to run an agent.
* Please see RL_model.ipynb for pre-trained models results.