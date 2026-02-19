# Exclusively Penalized Q-learning for Offline Reinforcement Learning
This repository considers the implementation of the paper "Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent Reinforcement Learning" which has been accepted to ICLR 2026.

# Installation Guide

1. Create conda env 
```
conda create -n S2Q python=3.9
```
2. Activate conda env
```
conda activate S2Q
```

6. Install other packages
```
pip install h5py tqdm pyyaml python-dateutil matplotlib gtimer scikit-learn
  numba==0.56.2 path.py==12.5.0 patchelf==0.15.0.0 joblib==1.2.0 gtimer python-dateutil matplotlib scikit-learn wandb
```

# Run EPQ

This codebase is built on rlkit (https://github.com/vitchyr/rlkit/), and implements CQL (https://github.com/aviralkumar2907/CQL). To run our code, follow the installation instructions for rlkit as shown below, then install D4RL(https://github.com/rail-berkeley/d4rl).

Then we can run EPQ with an example as follow :

1. First, train VAE for the behavior policy
```
python behavior_cloning.py --env=halfcheetah-medium-v2
```

2. After training the behavior model, we can run EPQ by executing :
```
python EPQ_main.py --env=halfcheetah-medium-v2
```
