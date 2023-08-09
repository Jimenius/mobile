import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Union, Tuple, Optional
from copy import deepcopy
from collections import defaultdict
from policies import BasePolicy
from dynamics import BaseDynamics


class MAPLEPolicy(BasePolicy):
    """
    Offline Model-based Adaptable Policy Learning
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor_context_extractor: nn.Module,
        actor: nn.Module,
        critic_context_extractor: nn.Module,
        critics: nn.ModuleList,
        actor_optim: torch.optim.Optimizer,
        critics_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        penalty_coef: float = 1.0,
        penalty_type: str = 'lcb',
        num_samples: int = 10,
        deterministic_backup: bool = False,
        max_q_backup: bool = False
    ) -> None:

        super().__init__()
        self.dynamics = dynamics
        self.actor_context_extractor = actor_context_extractor
        self.actor = actor
        self.critic_context_extractor = critic_context_extractor
        self.critics = critics
        self.critics_old = deepcopy(critics)
        self.critics_old.eval()

        self.actor_optim = actor_optim
        self.critics_optim = critics_optim

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._penalty_coef = penalty_coef
        assert penalty_type in ("lcb", "disagreement", "aleatoric")
        self._penalty_type = penalty_type
        self._num_samples = num_samples
        self._deteterministic_backup = deterministic_backup
        self._max_q_backup = max_q_backup

    def train(self, mode: bool = True) -> "MAPLEPolicy":
        self.training = mode
        self.actor.train(mode)
        self.actor_context_extractor.train(mode)
        self.critics.train(mode)
        self.critic_context_extractor.train(mode)
        return self

    def eval(self) -> None:
        self.actor.eval()
        self.actor_context_extractor.eval()
        self.critics.eval()
        self.critic_context_extractor.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critics_old.parameters(), self.critics.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def actforward(
        self,
        obs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs, context)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob,

    def select_action(
        self,
        obs: np.ndarray,
        last_action: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> np.ndarray:
        batch_size = len(obs)
        seq_len = [1] * batch_size
        if last_action is None:
            last_action = torch.zeros(
                (batch_size, 1, self.actor_context_extractor.action_dim)
            )
        if context is None:
            context = torch.zeros(
                (batch_size, 1, self.actor_context_extractor.hidden_units),
                dtype=torch.float32,
            )
        if len(obs.shape) == 2:
            obs = obs[:, None, :]
        if len(last_action.shape) == 2:
            last_action = last_action[:, None, :]
        with torch.no_grad():
            new_context = self.actor_context_extractor(obs, last_action, seq_len, pre_hidden=context)
            action, _ = self.actforward(obs, new_context, deterministic)
        new_context = new_context.squeeze(1).cpu().numpy()
        action = action.squeeze(1).cpu().numpy()
        return action, new_context

    @torch.no_grad()
    def rollout(
        self,
        batch: Dict[str, np.ndarray],
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)
        observations = batch["observations"]
        last_actions = batch["last_actions"]
        actor_context = batch["actor_context"]
        critic_context = batch["critic_context"]

        rollout_batch_size = observations.shape[0]
        nonterm_seq = np.ones(rollout_batch_size, dtype=bool)

        # rollout
        for _ in range(rollout_length):
            actions, next_actor_context = self.select_action(
                observations, last_actions, actor_context
            )
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            next_critic_context = self.critic_context_extractor(
                observations, last_actions, [1] * rollout_batch_size, critic_context
            )
            next_critic_context = next_critic_context.squeeze(1).cpu().numpy()

            rollout_transitions["observations"].append(observations)
            rollout_transitions["next_observations"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)
            rollout_transitions["valid"].append(nonterm_seq[:, None])
            rollout_transitions["last_actions"].append(last_actions)
            rollout_transitions["actor_context"].append(actor_context)
            rollout_transitions["critic_context"].append(critic_context)

            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            nonterm_seq = nonterm_seq & nonterm_mask
            observations = next_observations
            last_actions = actions
            actor_context = next_actor_context
            critic_context = next_critic_context

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.stack(v, axis=1)

        return rollout_transitions, \
            {"reward_mean": rewards_arr.mean()}

    @torch.no_grad()
    def compute_penalty(self, obss: torch.Tensor, actions: torch.Tensor):
        # compute next q std
        pred_next_obss, mean, std = self.dynamics.predict_next_obs(obss, actions, self._num_samples)
        num_samples, num_ensembles, batch_size, obs_dim = pred_next_obss.shape
        if self._penalty_type == "lcb":
            pred_next_obss = pred_next_obss.reshape(-1, obs_dim)
            pred_next_actions, _ = self.actforward(pred_next_obss)

            pred_next_qs =  torch.cat([critic_old(pred_next_obss, pred_next_actions) for critic_old in self.critics_old], 1)
            pred_next_qs = torch.min(pred_next_qs, 1)[0].reshape(num_samples, num_ensembles, batch_size, 1)
            penalty = pred_next_qs.mean(0).std(0)
        elif self._penalty_type == "disagreement":
            mode = mean[:, :, :-1]
            next_mean = torch.mean(mode, dim=0)
            diff = mode - next_mean
            penalty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
        elif self._penalty_type == "aleatoric":
            penalty = torch.max(torch.norm(std, dim=-1, keepdim=True), dim=0)[0]

        return penalty
    
    def update_context(
        self, buffer
    ) -> None:
        """Update context stored in the buffer"""
        buffer.update_context(
            self.actor_context_extractor,
            self.critic_context_extractor,
        )

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        ### s_1 ~ s_t, s_2 ~ s_{t+1}, a_1 ~ a_t, a_0 ~ a_{t-1}
        obss, actions, next_obss, rewards, terminals, last_actions, actor_context, critic_context, valid = mix_batch["observations"], mix_batch["actions"], mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"], mix_batch["last_actions"], mix_batch["actor_context"], mix_batch["critic_context"], mix_batch["valid"]
        batch_size = obss.shape[0]
        ### z_0
        actor_context_init = actor_context[:, 0]
        critic_context_init = critic_context[:, 0]

        seq_lens = torch.sum(valid, axis=1).squeeze(-1).cpu()
        valid_num = torch.sum(valid)

        ### z_1 ~ z_t
        actor_context = self.actor_context_extractor(obss, last_actions, seq_lens, actor_context_init)
        critic_context = self.critic_context_extractor(obss, last_actions, seq_lens, critic_context_init)
        seq_lens_next = torch.ones(batch_size).int()
        ### z_{t+1}
        actor_context_next = self.actor_context_extractor(next_obss[:, -1:], actions[:, -1:], seq_lens_next, actor_context[:, -1])
        critic_context_next = self.critic_context_extractor(next_obss[:, -1:], actions[:, -1:], seq_lens_next, critic_context[:, -1])
        ### z_2 ~ z_{t+1}
        actor_context_next = torch.cat((actor_context[:, 1:], actor_context_next), dim=1)
        critic_context_next = torch.cat((critic_context[:, 1:], critic_context_next), dim=1)

        # update critic
        qs = torch.cat([critic(obss, critic_context, actions) for critic in self.critics], -1)
        with torch.no_grad():
            # penalty = self.compute_penalty(obss, actions)
            # penalty[:len(real_batch["rewards"])] = 0.0
            ### Penalty is disabled now
            penalty = 0
            ###

            if self._max_q_backup:
                ### max_q_backup is enabled by default in Adroit
                raise Exception("Not supported yet.")
            else:
                next_actions, next_log_probs = self.actforward(next_obss, actor_context_next)
                ### critic output shape (batch_size, horizon, 1)
                next_qs = torch.cat([critic_old(next_obss, critic_context_next, next_actions) for critic_old in self.critics_old], -1)
                next_q = torch.min(next_qs, -1, keepdim=True)[0]
                if not self._deteterministic_backup:
                    next_q -= self._alpha * next_log_probs
            target_q = (rewards - self._penalty_coef * penalty) + self._gamma * (1 - terminals) * next_q
            target_q = torch.clamp(target_q, 0, None)

        # critic_loss = ((qs - target_q) ** 2).mean()
        ### qs shape (batch_size, horizon, num_critics)
        ### target_q and valid shape (batch_size, horizon, 1)
        critic_loss = torch.sum((qs - target_q) ** 2 * valid) / valid_num
        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # update actor
        a, log_probs = self.actforward(obss, actor_context)
        qas = torch.cat([critic(obss, critic_context, a).detach() for critic in self.critics], -1)
        actor_loss = self._alpha * log_probs.mean() - torch.min(qas, -1)[0]
        actor_loss = torch.sum(actor_loss * valid) / valid_num
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result