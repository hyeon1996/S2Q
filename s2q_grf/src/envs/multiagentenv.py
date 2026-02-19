class MultiAgentEnv(object):

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def get_native_state(self):
        """ Returns the native state """
        raise NotImplementedError

    def get_native_state_size(self):
        """ Returns the shape of the native state """
        raise NotImplementedError

    def get_alive_state(self):
        """ Returns the alive state """
        raise NotImplementedError

    def get_alive_state_size(self):
        """ Returns the shape of the alive state """
        raise NotImplementedError

    def get_n_enemies(self):
        """ Returns the total number of enemies """
        raise NotImplementedError

    def get_native_state_summary(self):
        """ Returns the summary of native state """
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
