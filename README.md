# Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent Reinforcement Learning
This repository considers the implementation of the paper "Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent Reinforcement Learning" which has been accepted to ICLR 2026.

# Installation Guide

1. Create conda env 
```
conda create -n S2Q python=3.8
```
2. Activate conda env
```
conda activate S2Q
```

3. Install Python packages
```
bash install_dependencies.sh
```

4. Install SMAC
```
bash install_sc2.sh
```

6. Install Google Research Football:
```
bash install_gfootball.sh
```

## Command Line Tool

**Run experiments**

```shell

# SMAC-Hard+
python3 s2q_smac/src/main.py --config=qmix_att --env-config=sc2 with env_args.map_name="5m_vs_6m"

# SMAC-Comm
python3 s2q_smac/src/main.py --config=s2q_comm --env-config=sc2 with env_args.map_name="1o_2r_vs_4r"

# SMACv2
python3 s2q_smac/src/main.py --config=s2q --env-config=sc2v2 with env_args.map_name="terran_5_vs_5"

# GRF
python3 s2q_grf/src/main.py --config=s2q --env-config=grf with env_args.map_name=academy_3_vs_2

```

The default setups for an algorithm or environment is represented as config files located in `src/config`.

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`


To enable experiment tracking with WandB, enter following configurations in `src/config/default.yaml`:

use_wandb: # Log results to W&B
wandb_team: # W&B team name
wandb_project: W&B project name
