import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import os


class GOEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self,
                 healthy_z_range=(0.15, 0.5),
                 reset_noise_scale=1e-2,
                 terminate_when_unhealthy=True,
                 exclude_current_positions_from_observation=False,
                 frame_skip=40,
                 **kwargs,
                 ):
        if exclude_current_positions_from_observation:
            self.obs_dim = 17 + 18
        else:
            self.obs_dim = 19 + 18

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), 'go/scene.xml'),
                           frame_skip=frame_skip,
                           observation_space=observation_space,
                           **kwargs
                           )
        self.action_dim = 12
        self.action_space = Box(
            low=self.lower_limits, high=self.upper_limits, shape=(self.action_dim,), dtype=np.float64
        )

        self._reset_noise_scale = reset_noise_scale
        self._healthy_z_range = healthy_z_range
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

    @property
    def lower_limits(self):
        return np.array([-0.863, -0.686, -2.818]*4)

    @property
    def upper_limits(self):
        return np.array([0.863, 4.501, -0.888]*4)

    @property
    def init_joints(self):
        return np.array([0, 0, 0.37, 1, 0, 0, 0] + [0, 0.7, -1.4]*4)

    @property
    def base_rotation(self):
        """
        compute root (base) rotation of the robot. The rotation can be used for rewards
        :return: rotation of root in xyz direction
        """
        q = self.data.qpos[3:7].copy()
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        x = np.arctan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = np.arcsin(np.clip(2 * (q1 * q3 + q0 * q2), -1, 1))
        z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))

        return [x, y, z]

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        return terminated

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return np.concatenate([qpos, qvel])

    # ------------ reward functions----------------
    def _reward_healthy(self):
        return (self.is_healthy - 1) * 5

    def _reward_lin_vel(self, before_pos, after_pos):
        target_vel = np.array([0.5, 0, 0])
        lin_vel = (after_pos - before_pos) / self.dt
        return np.exp(-np.linalg.norm(target_vel - lin_vel))
    
    def _reward_position(self):
        target_position = np.array([10, 0, 0])
        position = self.data.qpos[:3]
        return np.exp(-np.linalg.norm(target_position - position))
    
    def _reward_goal(self):
        target_position = np.array([10., 0, 0])
        position = self.data.qpos[:3]
        if (np.linalg.norm(target_position - position) < 0.2):
            return 300
        else:   
            return 0
    
    
    def _reward_orientation(self):
        target_orientation = np.array([0, 0, 0])
        orientation = self.base_rotation
        return np.exp(-np.linalg.norm(target_orientation - orientation))

    def _reward_closer(self, before_pos, after_pos):
        target_position = np.array([10, 0, 0])
        distance_before = np.linalg.norm(target_position - before_pos)
        distance_after = np.linalg.norm(target_position - after_pos)
        if np.abs(distance_after - distance_before) < 0.01:
            return 0
        if distance_after < distance_before:
            return 20
        else:
            return -10
        
    def _reward_side_bounds(self):
        position = self.data.qpos[:3]
        if (abs(position[1]) > 0.5 or abs(position[2]) > 0.5):
            return -100
        else:   
            return 1
    
    def _reward_alive(self):
        return 1
    
    def _reward_delta_q(self, delta_q):
        return -np.linalg.norm(delta_q)


    def step(self, delta_q):
        action = delta_q + self.data.qpos[-12:]
        action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)

        before_pos = self.data.qpos[:3].copy()
        self.do_simulation(action, self.frame_skip)
        after_pos = self.data.qpos[:3].copy()

        lin_v_track_reward = self._reward_lin_vel(before_pos, after_pos)
        healthy_reward = self._reward_healthy()
        orientation_reward = self._reward_orientation()
        position_reward = self._reward_position()
        closer_reward = self._reward_closer(before_pos, after_pos)
        alive_reward = self._reward_alive()
        goal_reward = self._reward_goal()
        delta_q_reward = self._reward_delta_q(delta_q)
        side_bounds_reward = self._reward_side_bounds()

        #total_rewards = orientation_reward + position_reward + 4.0*healthy_reward + 2.0*lin_v_track_reward + closer_reward
        #total_rewards =  4.0*healthy_reward + closer_reward + 2*alive_reward + 0.1*orientation_reward
        total_rewards = 50*healthy_reward + goal_reward + delta_q_reward + closer_reward + side_bounds_reward

        terminate = self.terminated
        observation = self._get_obs()
        info = {
            'total_reward': total_rewards,
            'lin_v_track_reward': lin_v_track_reward,
            "healthy_reward": healthy_reward,
            "orientation_reward": orientation_reward,
            "position_reward": position_reward,
            "closer_reward": closer_reward,
            "alive_reward": alive_reward,
            "goal_reward": goal_reward,
            "delta_q_reward": delta_q_reward,
            "side_bounds_reward": side_bounds_reward,
            "traverse": self.data.qpos[0],
        }

        if self.render_mode == "human":
            self.render()
        return observation, total_rewards, terminate, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_joints + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
