## Command Line Tool

**Run experiments**

```shell

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
