import numpy as np
import torch
from torch import nn

from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0
        
        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
    
    def normalize_reward(self) -> None:
        min, max = self.rewards.min(), self.rewards.max()
        self.rewards = (self.rewards - min) / (max - min)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }


class ReplayTrajBuffer(ReplayBuffer):
    """Replay buffer that stores mini trajectories of horizon length"""
    def __init__(self, buffer_size: int, obs_shape: Tuple, obs_dtype: np.dtype, action_dim: int, action_dtype: np.dtype, max_context_horizon: int, context_dim: int, device: str = "cpu") -> None:
        super().__init__(buffer_size, obs_shape, obs_dtype, action_dim, action_dtype, device)
        self.obs_dim = np.prod(obs_shape)
        self.observations = np.zeros((self._max_size, max_context_horizon) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size, max_context_horizon) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, max_context_horizon, self.action_dim), dtype=action_dtype)
        self.last_actions = np.zeros((self._max_size, max_context_horizon, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, max_context_horizon, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, max_context_horizon, 1), dtype=np.float32)
        self.actor_context = np.zeros((self._max_size, max_context_horizon, context_dim))
        self.critic_context = np.zeros((self._max_size, max_context_horizon, context_dim))
        self.valid = np.zeros((self._max_size, max_context_horizon, 1))

        self.max_context_horizon = max_context_horizon
        self.context_dim = context_dim

    def _set_index_mapping(
        self, data_size: int, mask: np.ndarray
    ) -> int:
        """Index mapping from flat"""
        ### Split data into mini trajectories
        traj_target_ind = 0
        ### mini_target_ind indicates the index inside the mini trajectory,
        ### which is separated by horizon
        mini_target_ind = 0
        ### Record the number of mini trajectories
        mini_traj_cum_num = 0
        self.index_mapping = [None] * data_size
        for i in range(data_size):
            self.index_mapping[i] = (traj_target_ind, mini_target_ind)
            mini_target_ind += 1
            if mask[i] or mini_target_ind == self.max_context_horizon:
                ### A new slot
                traj_target_ind = (traj_target_ind + 1) % self._max_size
                mini_traj_cum_num += 1
                mini_target_ind = 0

        return mini_traj_cum_num

    def _update_buffer_by_keys(
        self, data_dict: Dict[str, np.ndarray],
    ) -> None:
        for k in data_dict:
            target = getattr(self, k)
            data_size = len(data_dict[k])
            for i in range(data_size):
                index = self.index_mapping[i]
                target[index] = data_dict[k][i]
    
    def _get_context_update(
        self,
        actor_context_extractor: nn.Module,
        critic_context_extractor: nn.Module,
        traj_num_to_infer: int = 400,
    ) -> None:
        """Get new context with updated extractors"""
        total_traj_num = len(self.trajectory_lengths)
        last_traj_start_index = 0
        for i_ter in range(int(np.ceil(total_traj_num / traj_num_to_infer))):
            traj_lens_it = self.trajectory_lengths[traj_num_to_infer * i_ter : min(traj_num_to_infer * (i_ter + 1), total_traj_num)]
            num_traj = len(traj_lens_it)
            ### obs and last_act are placeholders that includes multiple trajectories of observations and last_actions
            obs = np.zeros((num_traj, self.max_trajectory_length, self.obs_dim), dtype=np.float32)
            last_act = np.zeros((len(traj_lens_it), self.max_trajectory_length, self.action_dim), dtype=np.float32)
            start_index = last_traj_start_index
            for ind, item in enumerate(traj_lens_it):
                ### Fill data in the placeholders
                obs[ind, :item] = self.observations_flat[start_index:(start_index + item)]
                last_act[ind, :item] = self.last_actions_flat[start_index:(start_index + item)]
                start_index += item

            obs = torch.from_numpy(obs).to(self.device)
            last_act = torch.from_numpy(last_act).to(self.device)
            lens = torch.IntTensor(traj_lens_it)
            actor_context_out = actor_context_extractor(obs, last_act, lens)
            critic_context_out = critic_context_extractor(obs, last_act, lens)
            traj_actor_context = np.concatenate(
                (np.zeros((num_traj, 1, self.context_dim)), actor_context_out[:, :-1].cpu().detach().numpy()),
                axis=1
            )
            traj_critic_context = np.concatenate(
                (np.zeros((num_traj, 1, self.context_dim)), critic_context_out[:, :-1].cpu().detach().numpy()),
                axis=1
            )
            data_size = len(self.observations_flat)
            actor_context = np.zeros((data_size, self.context_dim))
            critic_context = np.zeros((data_size, self.context_dim))
            start_index = last_traj_start_index
            for ind, item in enumerate(traj_lens_it):
                ### The context in the buffer is of shape (data_size, context_dim)
                actor_context[start_index:(start_index + item)] = traj_actor_context[ind, :item]
                critic_context[start_index:(start_index + item)] = traj_critic_context[ind, :item]
                start_index += item
            last_traj_start_index = start_index

        ### Flat context
        return actor_context, critic_context

    def load_dataset(
        self, dataset: Dict[str, np.ndarray],
        actor_context_extractor: nn.Module,
        critic_context_extractor: nn.Module
    ) -> None:
        observations = dataset["observations"]
        next_observations = dataset["next_observations"]
        actions = dataset["actions"] 
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        data_size = len(rewards)
        obs_mask = np.abs(observations[1:] - next_observations[:-1]) < 1e-5
        mask = np.all(obs_mask, axis=-1)
        mask = np.concatenate(
            (np.zeros(1, dtype=bool), ~mask)
        )
        mask = mask | terminals
        last_actions = np.concatenate(
            (np.zeros((1, self.action_dim), dtype=self.action_dtype),
            actions[:-1, :]), axis=0,
        )
        last_actions[mask] = 0
        
        ### For context update
        self.observations_flat = observations.copy()
        self.last_actions_flat = last_actions.copy()
        
        ### Get lengths of trajectories
        l = 0
        self.trajectory_lengths = []
        self.max_trajectory_length = 0
        for i in range(data_size):
            l += 1
            if mask[i]:
                self.trajectory_lengths.append(l)
                if l > self.max_trajectory_length:
                    self.max_trajectory_length = l
                l = 0

        self.actor_context_flat, self.critic_context_flat = self._get_context_update(actor_context_extractor, critic_context_extractor,)

        mini_traj_cum_num = self._set_index_mapping(data_size, mask)
        data_dict = {
            "observations": observations,
            "next_observations": next_observations,
            "actions": actions,
            "last_actions": last_actions,
            "rewards": rewards,
            "terminals": terminals,
            "valid": ~mask,
            "actor_context": self.actor_context_flat,
            "critic_context": self.critic_context_flat,
        }
        self._update_buffer_by_keys(data_dict)
        ### Add all mini_traj_cum_num mini trajectories to the buffer, move the buffer pointer
        self._size = int(min(mini_traj_cum_num + self._size, self._max_size))
        self._ptr = (self._ptr + mini_traj_cum_num) % self._max_size

    def update_context(
        self,
        actor_context_extractor: nn.Module,
        critic_context_extractor: nn.Module,
    ):
        self.actor_context_flat, self.critic_context_flat = self._get_context_update(actor_context_extractor, critic_context_extractor)
        data_dict = {
            "actor_context": self.actor_context_flat,
            "critic_context": self.critic_context_flat,
        }
        self._update_buffer_by_keys(data_dict)
    
    def add_batch(self, batch) -> None:
        """Add rollout samples to buffer"""
        batch_size = len(batch["valid"])
        indices = np.arange(self._ptr, self._ptr + batch_size) % self._max_size
        for k in batch:
            target = getattr(self, k)
            target[indices] = np.array(batch[k])

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def sample_rollout(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample initial observations and context for rollout"""
        data_size = len(self.observations_flat)
        indices = np.random.randint(data_size, size=batch_size)
        batch = {
            "observations": self.observations_flat[indices],
            "last_actions": self.last_actions_flat[indices],
            "actor_context": self.actor_context_flat[indices],
            "critic_context": self.critic_context_flat[indices]
        }
        return batch

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(self._size, size=batch_size)
        sample_keys = (
            "observations", "next_observations", "actions", "last_actions",
            "rewards", "terminals", "valid", "actor_context", "critic_context"
        )
        batch = {}
        for k in sample_keys:
            value = getattr(self, k)[indices]
            batch[k] = torch.as_tensor(value, device=self.device)
        return batch