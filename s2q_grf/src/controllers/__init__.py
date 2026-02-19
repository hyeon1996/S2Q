REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .basic_central_controller import CentralBasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["n_mac"] = NMAC
