from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.flatten(1)), dim=1)


def clone_momentum_module(module: nn.Module) -> nn.Module:
    momentum_module = copy.deepcopy(module)
    for p in momentum_module.parameters():
        p.requires_grad_(False)
    return momentum_module


@torch.no_grad()
def momentum_update(online: nn.Module, momentum: nn.Module, m: float = 0.999) -> None:
    online_state = online.state_dict()
    momentum_state = momentum.state_dict()
    for key, value in online_state.items():
        if not torch.is_floating_point(value):
            momentum_state[key].copy_(value)
            continue
        momentum_state[key].mul_(m).add_(value.detach(), alpha=1.0 - m)


class DegradationContrastiveQueue(nn.Module):
    def __init__(self, embedding_dim: int = 128, prior_dim: int = 28, queue_size: int = 4096) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.register_buffer("embeddings", torch.zeros(queue_size, embedding_dim))
        self.register_buffer("priors", torch.zeros(queue_size, prior_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, priors: torch.Tensor) -> None:
        embeddings = embeddings.detach()
        priors = priors.detach().to(device=embeddings.device, dtype=embeddings.dtype)
        batch = embeddings.shape[0]
        if batch >= self.queue_size:
            self.embeddings.copy_(embeddings[-self.queue_size :])
            self.priors.copy_(priors[-self.queue_size :])
            self.ptr.zero_()
            self.filled.fill_(self.queue_size)
            return

        ptr = int(self.ptr.item())
        end = ptr + batch
        if end <= self.queue_size:
            self.embeddings[ptr:end].copy_(embeddings)
            self.priors[ptr:end].copy_(priors)
        else:
            first = self.queue_size - ptr
            self.embeddings[ptr:].copy_(embeddings[:first])
            self.priors[ptr:].copy_(priors[:first])
            self.embeddings[: end - self.queue_size].copy_(embeddings[first:])
            self.priors[: end - self.queue_size].copy_(priors[first:])
        self.ptr.fill_(end % self.queue_size)
        self.filled.fill_(min(self.queue_size, int(self.filled.item()) + batch))

    def valid(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(self.filled.item())
        return self.embeddings[:n], self.priors[:n]


def prior_weighted_moco_loss(
    queries: torch.Tensor,
    keys: torch.Tensor,
    priors: torch.Tensor,
    queue: DegradationContrastiveQueue,
    temperature: float = 0.07,
    prior_temperature: float = 0.25,
) -> torch.Tensor:
    queries = F.normalize(queries, dim=1)
    keys = F.normalize(keys.detach(), dim=1)
    priors = priors.to(device=queries.device, dtype=queries.dtype)

    queue_embeddings, queue_priors = queue.valid()
    queue_embeddings = queue_embeddings.to(device=queries.device, dtype=queries.dtype)
    queue_priors = queue_priors.to(device=queries.device, dtype=queries.dtype)
    if queue_embeddings.numel() == 0:
        logits = queries @ keys.t() / temperature
        labels = torch.arange(queries.shape[0], device=queries.device)
        return F.cross_entropy(logits, labels)

    candidates = torch.cat([keys, queue_embeddings], dim=0)
    candidate_priors = torch.cat([priors, queue_priors], dim=0)
    logits = queries @ candidates.t() / temperature

    prior_dist = torch.cdist(
        F.normalize(priors, dim=1),
        F.normalize(candidate_priors, dim=1),
        p=2,
    )
    target_logits = -prior_dist / prior_temperature
    labels = torch.arange(queries.shape[0], device=queries.device)
    target_logits[:, : keys.shape[0]].scatter_(1, labels[:, None], 1.0 / prior_temperature)
    targets = F.softmax(target_logits, dim=1).detach()
    log_prob = F.log_softmax(logits, dim=1)
    return -(targets * log_prob).sum(dim=1).mean()
