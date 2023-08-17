import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional


# for SAC
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist


class Critic(nn.Module):
    def __init__(self, backbone: nn.Module, device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values


class MAPLEActor(ActorProb):
    def __init__(
        self,
        backbone: nn.Module,
        preprocess_net: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__(backbone, dist_net, device)
        self.context_preprocess_net = preprocess_net

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        context: Union[np.ndarray, torch.Tensor],
    ) -> torch.distributions.Distribution:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        context = torch.as_tensor(context, device=self.device, dtype=torch.float32)
        context = self.context_preprocess_net(context)
        obs = torch.cat((obs, context), dim=-1)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist
    

class MAPLECritic(Critic):
    def __init__(
        self,
        backbone: nn.Module,
        preprocess_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__(backbone, device)
        self.context_preprocess_net = preprocess_net

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        context: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        context = torch.as_tensor(context, device=self.device, dtype=torch.float32)
        context = self.context_preprocess_net(context)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        obs = torch.cat([obs, context, actions], dim=-1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values