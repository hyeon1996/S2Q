from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .central_rnn_agent import CentralRNNAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent
