from matplotlib.pyplot import xcorr
import torch as th
from torch.distributions import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from .epsilon_schedules import DecayThenFlatSchedule
import torch.nn.functional as F


class GumbelSoftmax(OneHotCategorical):

    def __init__(self, logits, probs=None, temperature=1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -th.log( -th.log( U + self.eps))

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return th.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (th.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()

def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy()

REGISTRY = {}

class GumbelSoftmaxMultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, 'save_probs', False)

    def select_action(self, agent_logits, avail_actions, t_env, test_mode=False):
        masked_policies = agent_logits.clone()
        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = GumbelSoftmax(logits=masked_policies).sample()
            picked_actions = th.argmax(picked_actions, dim=-1).long()

        if self.save_probs:
            return picked_actions, masked_policies
        else:
            return picked_actions


REGISTRY["gumbel"] = GumbelSoftmaxMultinomialActionSelector


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, 'save_probs', False)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0] = 0
        masked_policies = masked_policies / (masked_policies.sum(-1, keepdim=True) + 1e-8)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            self.epsilon = self.schedule.eval(t_env)

            epsilon_action_num = (avail_actions.sum(-1, keepdim=True) + 1e-8)
            masked_policies = ((1 - self.epsilon) * masked_policies
                        + avail_actions * self.epsilon/epsilon_action_num)
            masked_policies[avail_actions == 0] = 0
            
            picked_actions = Categorical(masked_policies).sample().long()

        if self.save_probs:
            return picked_actions, masked_policies
        else:
            return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector

def categorical_entropy(probs):
    assert probs.size(-1) > 1
    return Categorical(probs=probs).entropy()


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

        self.encoder_decoder = None
        self.strategy_selector = None


    def select_action(self, agent_inputs, agent_inputs_2, agent_inputs_3, history, avail_actions,
                      t_env, t_ep,
                      test_mode=False):
        self.epsilon = self.schedule.eval(t_env)
        if test_mode:
            self.epsilon = 0.0

        # --- Select strategy at the start of each episode ---
        if t_ep == 0 and not test_mode:
            if th.rand(1).item() < 0.5:
                self.strategy_selector = "action_1"
            else:
                self.strategy_selector = "mixed"

        # Mask unavailable actions
        masked_q1 = agent_inputs.clone()
        masked_q1[avail_actions == 0.0] = -float("inf")
        q_1, action_1 = masked_q1.max(dim = 2)

        masked_q2 = agent_inputs_2.clone()
        masked_q2[avail_actions == 0.0] = -float("inf")
        q_2, action_2 = masked_q2.max(dim = 2)

        masked_q3 = agent_inputs_3.clone()
        masked_q3[avail_actions == 0.0] = -float("inf")
        q_3, action_3 = masked_q3.max(dim = 2)

        batch_size, n_agents = action_1.shape

        if (self.encoder_decoder is not None and not test_mode and self.strategy_selector == "mixed"):
            with th.no_grad():
                _, probs = self.encoder_decoder(history.reshape(batch_size, 1, -1))
                dist = Categorical(probs.squeeze(1))
                selector_per_batch = dist.sample()  # [batch]
                selector = selector_per_batch.unsqueeze(-1).expand(-1, n_agents)  # [batch, n_agents]
        else:
            selector = th.zeros((batch_size, n_agents), dtype = th.long, device = agent_inputs.device)  # use action_1

        all_actions = th.stack([action_1, action_2, action_3], dim = 2)
        selected_actions = th.gather(all_actions, dim = 2, index = selector.unsqueeze(-1)).squeeze(-1)

        # Epsilon-greedy exploration
        random_mask = (th.rand(batch_size, n_agents, device = agent_inputs.device) < self.epsilon)
        random_actions = Categorical(avail_actions.float()).sample()
        final_actions = th.where(random_mask, random_actions, selected_actions)

        return final_actions

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class GaussianActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, mu, sigma, test_mode=False):
        # Expects the following input dimensions:
        # mu: [b x a x u]
        # sigma: [b x a x u x u]
        assert mu.dim() == 3, "incorrect input dim: mu"
        assert sigma.dim() == 3, "incorrect input dim: sigma"
        sigma = sigma.view(-1, self.args.n_agents, self.args.n_actions, self.args.n_actions)

        if test_mode and self.test_greedy:
            picked_actions = mu
        else:
            dst = th.distributions.MultivariateNormal(mu.view(-1,
                                                              mu.shape[-1]),
                                                      sigma.view(-1,
                                                                 mu.shape[-1],
                                                                 mu.shape[-1]))
            try:
                picked_actions = dst.sample().view(*mu.shape)
            except Exception as e:
                a = 5
                pass
        return picked_actions


REGISTRY["gaussian"] = GaussianActionSelector