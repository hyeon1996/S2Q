import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class Encoder(nn.Module):
    """
    Encoder E_ψ that maps concatenated agent local histories to latent representation z_t
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)


    def forward(self, inputs):

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        z = self.fc3(x)

        return z


class Decoder(nn.Module):
    """
    Decoder D_ψ that reconstructs global state and approximate distribution from latent representation
    """

    def __init__(self, state_dim, K, hidden_dim=64, latent_dim=32):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.prob_dim = K + 1

        # Shared hidden layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # State reconstruction head
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.prob_head = nn.Linear(hidden_dim, self.prob_dim)

    def forward(self, inputs):
        """
        Args:
            z_t: Latent representation [batch_size, latent_dim]
        Returns:
            hat_s_t: Reconstructed global state [batch_size, state_dim]
            hat_P_t: Approximate probability distribution [batch_size, prob_dim]
        """
        # Shared processing
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))

        # Reconstruct state and probability distribution
        hat_s = self.state_head(x)
        hat_P = self.prob_head(x)
        hat_P = F.softmax(hat_P, dim=-1)

        return hat_s, hat_P


class EncoderDecoder(nn.Module):
    """
    Complete encoder-decoder architecture for approximating P_t and global state
    """

    def __init__(self, input_dim, state_dim, K, hidden_dim=64, latent_dim=32):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(state_dim, K)

    def forward(self, inputs):
        """
        Args:
            tau_t: Concatenated local histories [batch_size, seq_len, input_dim]
        Returns:
            hat_s_t: Reconstructed global state [batch_size, state_dim]
            hat_P_t: Approximate probability distribution [batch_size, prob_dim]
        """
        # Encode to latent representation
        z = self.encoder(inputs)

        # Decode to state and probability distribution
        hat_s, hat_P = self.decoder(z)

        return hat_s, hat_P