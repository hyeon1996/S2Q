from smac.env.multiagentenv import MultiAgentEnv
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np


class GRF(MultiAgentEnv):

    def __init__(
            self,
            dense_reward=False,
            write_full_episode_dumps=False,
            write_goal_dumps=False,
            dump_freq=1000,
            render=False,
            time_step=0,
            map_name='academy_3_vs_1_with_keeper',
            stacked=False,
            representation="simple115",
            rewards='scoring',
            logdir='football_dumps',
            write_video=False,
            number_of_right_players_agent_controls=0,
            seed=0,
    ):
        if map_name == 'academy_3_vs_2':
            map_name = 'academy_3_vs_1_with_keeper'
            n_agents = 3
            n_enemies = 2
            obs_dim = 26
            time_limit = 150
            pos_limit = 0
        elif map_name == 'academy_3_vs_2_full_field':
            map_name = 'academy_3_vs_1_with_keeper_full_field'
            n_agents = 3
            n_enemies = 2
            obs_dim = 26
            time_limit = 300
            pos_limit = -0.75
        elif map_name == 'academy_3_vs_3':
            map_name = 'academy_3_vs_2_with_keeper'
            n_agents = 3
            n_enemies = 3
            obs_dim = 30
            time_limit = 150
            pos_limit = 0
        elif map_name == 'academy_3_vs_3_full_field':
            map_name = 'academy_3_vs_2_with_keeper_full_field'
            n_agents = 3
            n_enemies = 3
            obs_dim = 30
            time_limit = 300
            pos_limit = -0.75
        elif map_name == 'academy_4_vs_3':
            map_name = 'academy_4_vs_2_with_keeper'
            n_agents = 4
            n_enemies = 3
            obs_dim = 34
            time_limit = 150
            pos_limit = 0
        elif map_name == 'academy_4_vs_3_full_field':
            map_name = 'academy_4_vs_2_with_keeper_full_field'
            n_agents = 4
            n_enemies = 3
            obs_dim = 34
            time_limit = 300
            pos_limit = -0.75
        elif map_name == 'academy_counterattack':
            map_name = 'academy_counterattack_hard'
            n_agents = 4
            n_enemies = 3
            obs_dim = 34
            time_limit = 150
            pos_limit = 0
        elif map_name == 'academy_counterattack_full_field':
            map_name = 'academy_counterattack_hard_full_field'
            n_agents = 4
            n_enemies = 3
            obs_dim = 34
            time_limit = 300
            pos_limit = -0.4
        elif map_name == 'academy_corner':
            map_name = 'academy_corner'
            n_agents = 10
            n_enemies = 11
            obs_dim = 90
            time_limit = 150
            pos_limit = -0.1
        elif map_name == 'academy_2_vs_2':
            map_name = 'academy_run_pass_and_shoot_with_keeper'
            n_agents = 2
            n_enemies = 2
            obs_dim = 22
            time_limit = 150
            pos_limit = 0
        elif map_name == 'academy_2_vs_2_full_field':
            map_name = 'academy_run_pass_and_shoot_with_keeper_full_field'
            n_agents = 2
            n_enemies = 2
            obs_dim = 22
            time_limit = 300
            pos_limit = -0.75
        elif map_name == '5_vs_5':
            map_name = '5_vs_5'
            n_agents = 4
            n_enemies = 5
            obs_dim = 42
            time_limit = 300
            pos_limit = -0.15
        else:
            raise NotImplementedError

        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.episode_limit = time_limit
        self.time_step = time_step
        self.obs_dim = obs_dim
        self.map_name = map_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed
        self.pos_limit = pos_limit

        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.map_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
        self.env.seed(self.seed)

        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in
            range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n

        self.unit_dim = self.obs_dim

    def get_simple_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []

        if index == -1:
            # global state, absolute position
            simple_obs.append(full_obs['left_team'][-self.n_agents:].reshape(-1))
            simple_obs.append(full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

            simple_obs.append(full_obs['right_team'][:self.n_enemies].reshape(-1))
            simple_obs.append(full_obs['right_team_direction'][:self.n_enemies].reshape(-1))

            simple_obs.append(full_obs['ball'])
            simple_obs.append(full_obs['ball_direction'])

        else:
            # local state, relative position
            ego_position = full_obs['left_team'][-self.n_agents + index].reshape(-1)
            simple_obs.append(ego_position)
            simple_obs.append((np.delete(
                full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1))

            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
            simple_obs.append(np.delete(
                full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1))

            simple_obs.append((full_obs['right_team'][:self.n_enemies] - ego_position).reshape(-1))
            simple_obs.append(full_obs['right_team_direction'][:self.n_enemies].reshape(-1))

            simple_obs.append(full_obs['ball'][:2] - ego_position)
            simple_obs.append(full_obs['ball'][-1].reshape(-1))
            simple_obs.append(full_obs['ball_direction'])

        simple_obs = np.concatenate(simple_obs)
        return simple_obs

    def get_global_state(self):
        return self.get_simple_obs(-1)

    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < self.pos_limit or any(ours_loc[:, 0] < self.pos_limit):
            return True

        return False

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1
        _, original_rewards, done, infos = self.env.step(actions.tolist())
        rewards = list(original_rewards)
        # obs = np.array([self.get_obs(i) for i in range(self.n_agents)])

        if self.time_step >= self.episode_limit:
            done = True

        if self.check_if_done():
            done = True

        # if sum(rewards) <= 0:
        #     return obs, self.get_global_state(), -int(done), done, infos
            # return -int(done), done, infos

        # return obs, self.get_global_state(), 100, done, infos
        # return 100, done, infos

        return sum(rewards), done, infos

    def get_obs(self):
        """Returns all agent observations in a list."""
        # obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])
        obs = [self.get_simple_obs(i) for i in range(self.n_agents)]
        return obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_simple_obs(agent_id)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        # TODO: in wrapper_grf_3vs1.py, author set state_shape=obs_shape
        return self.obs_dim

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.env.reset()
        obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])

        return obs, self.get_global_state()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_native_state(self):
        return self.get_state()

    def get_native_state_size(self):
        return self.obs_dim

    def get_native_state_idx(self):
        return list(range(self.obs_dim))

    def get_alive_state(self):
        return np.ones(self.n_agents + self.n_enemies)

    def get_alive_state_size(self):
        return self.n_agents + self.n_enemies

    def get_n_enemies(self):
        return self.n_enemies

    def get_native_state_summary(self):
        offset = 0
        summary = {}
        for state in ['ally_position', 'ally_direction']:
            summary[state] = np.arange(offset, offset + self.n_agents * 2)
            offset += self.n_agents * 2
        for state in ['enemy_position', 'enemy_direction']:
            summary[state] = np.arange(offset, offset + self.n_enemies * 2)
            offset += self.n_enemies * 2
        for state in ['ball_position', 'ball_direction']:
            summary[state] = np.arange(offset, offset + 3)
            offset += 3

        return summary