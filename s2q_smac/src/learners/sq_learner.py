import copy
from components.episode_buffer import EpisodeBatch
from controllers import REGISTRY as mac_REGISTRY
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from modules.mixers.nmix import Mixer
from modules.comm import EncoderDecoder
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
import torch.nn.functional as F
import torch.nn as nn


class SQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        self.last_reset_episode = 0
        self.mixer = Mixer(args)

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.mixer_2 = Mixer(args)
        self.mixer_3 = Mixer(args)

        self.params += list(self.mixer_2.parameters()) + list(self.mixer_3.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        self.central_mixer = QMixerCentralFF(args)
        self.target_central_mixer = copy.deepcopy(self.central_mixer)

        self.central_mac = mac_REGISTRY[args.central_mac](scheme, args)
        self.target_central_mac = copy.deepcopy(self.central_mac)

        self.params += list(self.central_mixer.parameters()) + list(self.central_mac.parameters())

        self.encoder_decoder = EncoderDecoder(input_dim=64 * self.args.n_agents, state_dim=self.args.state_shape, K = self.args.K)

        self.params += list(self.encoder_decoder.parameters())

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params = self.params, lr = args.lr, weight_decay = getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params = self.params, lr = args.lr, alpha = args.optim_alpha, eps = args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

        self.w_c = self.args.w_c

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.mac.encoder_decoder = self.encoder_decoder
        self.mac.action_selector.encoder_decoder = self.encoder_decoder

        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out, mac_out_2, mac_out_3, history = [], [], [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, agent_outs_2, agent_outs_3, histories = self.mac.forward(batch, t = t)
            mac_out.append(agent_outs)
            mac_out_2.append(agent_outs_2)
            mac_out_3.append(agent_outs_3)
            history.append(histories)
        mac_out = th.stack(mac_out, dim = 1)  # Concat over time
        mac_out_2 = th.stack(mac_out_2, dim = 1)
        mac_out_3 = th.stack(mac_out_3, dim = 1)
        history = th.stack(history, dim= 1)

        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t = t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim = 1)  # Concat over time
        central_chosen_action_qvals_agents = th.gather(central_mac_out, dim = 3,
                                                       index = batch["actions"].unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                                    self.args.central_action_embed)).squeeze(
            3)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim = 3, index = actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_2 = th.gather(mac_out_2[:, :-1], dim = 3, index = actions).squeeze(3)
        chosen_action_qvals_3 = th.gather(mac_out_3[:, :-1], dim = 3, index = actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim = 3, keepdim = True)[1]

            mac_out_2_detach = mac_out_2.clone().detach()
            mac_out_2_detach[avail_actions == 0] = -9999999
            cur_max_actions_2 = mac_out_2_detach.max(dim = 3, keepdim = True)[1]

            mac_out_3_detach = mac_out_3.clone().detach()
            mac_out_3_detach[avail_actions == 0] = -9999999
            cur_max_actions_3 = mac_out_3_detach.max(dim = 3, keepdim = True)[1]

            central_target_mac_out = []
            self.target_central_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                central_target_agent_outs = self.target_central_mac.forward(batch, t = t)
                central_target_mac_out.append(central_target_agent_outs)
            central_target_mac_out = th.stack(central_target_mac_out[:], dim = 1)  # Concat across time

            central_target_max_agent_qvals = th.gather(central_target_mac_out[:, :], 3,
                                                       cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                                 self.args.central_action_embed)).squeeze(
                3)

            central_target_max_agent_qvals_2 = th.gather(central_target_mac_out[:, :], 3,
                                                       cur_max_actions_2[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                                 self.args.central_action_embed)).squeeze(
                3)

            central_target_max_agent_qvals_3 = th.gather(central_target_mac_out[:, :], 3,
                                                       cur_max_actions_3[:, :].unsqueeze(4).repeat(1, 1, 1, 1,
                                                                                                 self.args.central_action_embed)).squeeze(
                3)

            central_target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals, batch["state"])
            central_target_max_qvals_2 = self.target_central_mixer(central_target_max_agent_qvals_2, batch["state"])
            central_target_max_qvals_3 = self.target_central_mixer(central_target_max_agent_qvals_3, batch["state"])

        targets = build_td_lambda_targets(rewards, terminated, mask, central_target_max_qvals, self.args.n_agents, self.args.gamma, self.args.td_lambda)

        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        chosen_action_qvals_2 = self.mixer_2(chosen_action_qvals_2, batch["state"][:, :-1])
        chosen_action_qvals_3 = self.mixer_3(chosen_action_qvals_3, batch["state"][:, :-1])
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"])

        is_max_action = (actions == cur_max_actions[:, :-1]).min(dim = 2)[0]
        is_max_action_2 = (actions == cur_max_actions_2[:, :-1]).min(dim = 2)[0]

        td_error = chosen_action_qvals - targets.clone().detach()
        central_td_error = (central_chosen_action_qvals[:, :-1] - targets.clone().detach())

        subtract_mask = (is_max_action | is_max_action_2).float()

        td_error_2 = chosen_action_qvals_2 - (targets.clone().detach() - central_chosen_action_qvals[:, :-1].clone().detach() * is_max_action.float())
        td_error_3 = chosen_action_qvals_3 - (targets.clone().detach() - central_chosen_action_qvals[:, :-1].clone().detach() * subtract_mask)

        ws = th.ones_like(td_error)

        central_max_diff = (central_chosen_action_qvals[:, :-1] >= central_target_max_qvals[:, :-1]).clone().detach()
        central_sub_diff = (central_chosen_action_qvals[:, :-1] >= central_target_max_qvals_2[:, :-1]).clone().detach()
        central_sub_diff_2 = (central_chosen_action_qvals[:, :-1] >= central_target_max_qvals_3[:, :-1]).clone().detach()

        w_mask = (central_max_diff & central_sub_diff & central_sub_diff_2)
        w_mask_2 = (td_error_2 < 0)
        w_mask_3 = (td_error_3 < 0)

        ws_1 = th.where(w_mask, th.ones_like(td_error) * 1, ws * self.args.w_c)  # Target is greater than current max
        ws_2 = th.where(w_mask_2, th.ones_like(td_error) * 1, ws * self.args.w_c)
        ws_3 = th.where(w_mask_3, th.ones_like(td_error) * 1, ws * self.args.w_c)

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        masked_td_error_2 = td_error_2 * mask
        masked_td_error_3 = td_error_3 * mask
        masked_central_td_error = central_td_error * mask

        hat_s, hat_P = self.encoder_decoder(history[:, :-1].clone().detach().reshape(batch.batch_size, batch.max_seq_length-1, -1))

        stacked_qvals = th.stack([
            central_target_max_qvals[:, :-1],
            central_target_max_qvals_2[:, :-1],
            central_target_max_qvals_3[:, :-1]
        ], dim = -1)

        P_K = th.softmax(stacked_qvals / self.args.T, dim=-1).squeeze(2)

        ce_loss = F.kl_div(th.log(hat_P + 1e-8), P_K.detach(), reduction='mean')
        mse_loss = F.mse_loss(hat_s, batch["state"][:, :-1])
        latent_loss = ce_loss + mse_loss

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device = self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        L_td = (ws_1.detach() * masked_td_error ** 2).sum() / mask.sum()
        L_td_2 = (ws_2.detach() * masked_td_error_2 ** 2).sum() / mask.sum()
        L_td_3 = (ws_3.detach() * masked_td_error_3 ** 2).sum() / mask.sum()

        central_td_loss = (masked_central_td_error ** 2).sum() / mask.sum()
        loss = L_td + central_td_loss + L_td_2 + L_td_3 + latent_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                        / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                         / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_central_mac.load_state(self.central_mac)

        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())

            # self.V_target.load_state_dict(self.V_net.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            self.mixer_2.cuda()
            self.mixer_3.cuda()
        # self.state_prediction.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()
        self.central_mac.cuda()
        self.target_central_mac.cuda()
        self.encoder_decoder.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        self.central_mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
            th.save(self.mixer_2.state_dict(), "{}/mixer_2.th".format(path))
            th.save(self.mixer_3.state_dict(), "{}/mixer_3.th".format(path))
            th.save(self.central_mixer.state_dict(), "{}/central_mixer.th".format(path))

        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        print("load models")
        self.mac.load_models(path)
        self.central_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            print("load models is none")
            self.mac.central_mac = self.central_mac
            self.mac.action_selector.mixer = self.central_mixer

            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location = lambda storage, loc: storage))
            self.mixer_2.load_state_dict(th.load("{}/mixer_2.th".format(path), map_location = lambda storage, loc: storage))
            self.mixer_3.load_state_dict(th.load("{}/mixer_3.th".format(path), map_location = lambda storage, loc: storage))
            self.central_mixer.load_state_dict(th.load("{}/central_mixer.th".format(path), map_location = lambda storage, loc: storage))

        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location = lambda storage, loc: storage))


