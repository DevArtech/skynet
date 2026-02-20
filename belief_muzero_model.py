from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from muzero_model import MLP, MuZeroConfig, MuZeroDynamics, ScalarSupport, SkyjoObservationEncoder


@dataclass
class BeliefMuZeroInferenceOutput:
    hidden_state: Tensor
    policy_logits: Tensor
    value_logits: Tensor
    winner_logits: Tensor
    rank_logits: Tensor
    reward_logits: Tensor | None = None


class BeliefAwarePrediction(nn.Module):
    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.policy_head = MLP(config.latent_dim, config.ff_hidden_dim, config.action_space_size, dropout=config.dropout)
        self.value_head = MLP(
            config.latent_dim,
            config.ff_hidden_dim,
            (2 * config.value_support_max) + 1,
            dropout=config.dropout,
        )
        # Belief-aware auxiliary heads for multiplayer first-place objective.
        self.winner_head = MLP(config.latent_dim, config.ff_hidden_dim, config.max_players, dropout=config.dropout)
        self.rank_head = MLP(config.latent_dim, config.ff_hidden_dim, config.max_players, dropout=config.dropout)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        policy_logits = self.policy_head(hidden)
        value_logits = self.value_head(hidden)
        winner_logits = self.winner_head(hidden)
        rank_logits = self.rank_head(hidden)
        return policy_logits, value_logits, winner_logits, rank_logits


class BeliefAwareMuZeroNet(nn.Module):
    """
    MuZero with ego-conditioned belief heads for variable player counts.
    """

    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.config = config
        self.representation = SkyjoObservationEncoder(config)
        self.dynamics = MuZeroDynamics(config)
        self.prediction = BeliefAwarePrediction(config)
        self.value_support = ScalarSupport(config.value_support_max)
        self.reward_support = ScalarSupport(config.reward_support_max)

        self.ego_player_emb = nn.Embedding(config.max_players + 1, config.latent_dim)
        self.current_player_emb = nn.Embedding(config.max_players + 1, config.latent_dim)
        self.num_players_emb = nn.Embedding(config.max_players + 1, config.latent_dim)
        self.condition_norm = nn.LayerNorm(config.latent_dim)

    def _condition_hidden(
        self,
        hidden_state: Tensor,
        ego_player_id: Tensor,
        current_player_id: Tensor,
        num_players: Tensor,
    ) -> Tensor:
        ego = ego_player_id.long().clamp(0, self.config.max_players)
        current = current_player_id.long().clamp(0, self.config.max_players)
        nplayers = num_players.long().clamp(1, self.config.max_players)
        conditioned = (
            hidden_state
            + self.ego_player_emb(ego)
            + self.current_player_emb(current)
            + self.num_players_emb(nplayers)
        )
        return self.condition_norm(conditioned)

    def initial_inference(
        self,
        observation_tokens: dict[str, Tensor],
        ego_player_id: Tensor,
        current_player_id: Tensor,
        num_players: Tensor,
    ) -> BeliefMuZeroInferenceOutput:
        hidden_state = self.representation(observation_tokens)
        conditioned = self._condition_hidden(hidden_state, ego_player_id, current_player_id, num_players)
        policy_logits, value_logits, winner_logits, rank_logits = self.prediction(conditioned)
        return BeliefMuZeroInferenceOutput(
            hidden_state=hidden_state,
            policy_logits=policy_logits,
            value_logits=value_logits,
            winner_logits=winner_logits,
            rank_logits=rank_logits,
            reward_logits=None,
        )

    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        ego_player_id: Tensor,
        current_player_id: Tensor,
        num_players: Tensor,
    ) -> BeliefMuZeroInferenceOutput:
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        conditioned = self._condition_hidden(next_hidden_state, ego_player_id, current_player_id, num_players)
        policy_logits, value_logits, winner_logits, rank_logits = self.prediction(conditioned)
        return BeliefMuZeroInferenceOutput(
            hidden_state=next_hidden_state,
            policy_logits=policy_logits,
            value_logits=value_logits,
            winner_logits=winner_logits,
            rank_logits=rank_logits,
            reward_logits=reward_logits,
        )


def build_default_belief_muzero_config(action_space_size: int = 36) -> MuZeroConfig:
    return MuZeroConfig(action_space_size=action_space_size)
