from functools import partial
import sys
import os

from .grf.grf import GRF
from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
# from .mpe.push_box import PushBox


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["grf"] = partial(env_fn, env=GRF)
# REGISTRY["push_box"] = partial(env_fn, env=PushBox)



if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
