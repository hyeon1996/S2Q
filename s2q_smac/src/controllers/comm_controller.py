from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class CommMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(CommMAC, self).__init__(scheme, groups, args)

        self.hidden_states_2 = None
        self.hidden_states_3 = None

        self.encoder_decoder = None
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        state = ep_batch["state"][:, t_ep]
        qvals, qvals2, qvals3, history = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], qvals2[bs], qvals3[bs], history[bs], avail_actions[bs], t_env, t_ep, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        b, n, e = agent_inputs.size()
        if self.encoder_decoder is None:
            z = th.zeros(b, n, 32, device=ep_batch.device)
        else:
            z = self.encoder_decoder.encoder(self.hidden_states.detach().reshape(b, -1))
            z = z.unsqueeze(1).repeat(1,n,1)

        agent_inputs = th.cat((agent_inputs, z),-1)

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        agent_outs_2, self.hidden_states_2 = self.agent_2(agent_inputs, self.hidden_states_2)
        agent_outs_3, self.hidden_states_3 = self.agent_3(agent_inputs, self.hidden_states_3)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), agent_outs_2.view(ep_batch.batch_size, self.n_agents, -1), agent_outs_3.view(ep_batch.batch_size, self.n_agents, -1), self.hidden_states.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.hidden_states_2 = self.agent_2.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.hidden_states_3 = self.agent_3.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return list(self.agent.parameters()) + list(self.agent_2.parameters()) + list(self.agent_3.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.agent_2.load_state_dict(other_mac.agent_2.state_dict())
        self.agent_3.load_state_dict(other_mac.agent_3.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.agent_2.cuda()
        self.agent_3.cuda()


    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.agent_2.state_dict(), "{}/agent_2.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_2.load_state_dict(th.load("{}/agent_2.th".format(path), map_location = lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.agent_2 = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.agent_3 = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _build_state_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["state"][:, t].unsqueeze(1).repeat(1, self.n_agents, 1))  # b1av
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape + self.args.latent_dim