## Command Line Tool

**Run experiments**

```shell

# SMAC-Hard+
python3 s2q_smac/src/main.py --config=qmix_att --env-config=sc2 with env_args.map_name="5m_vs_6m"

# SMAC-Comm
python3 s2q_smac/src/main.py --config=s2q_comm --env-config=sc2 with env_args.map_name="1o_2r_vs_4r"

# SMACv2
python3 s2q_smac/src/main.py --config=s2q --env-config=sc2v2 with env_args.map_name="terran_5_vs_5"

```

The default setups for an algorithm or environment is represented as config files located in `src/config`.

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`
