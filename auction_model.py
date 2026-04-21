"""
MuZero model for the competitive auction environment.

Architecture mirrors the Skyjo MuZero (transformer encoder → latent → MLP
dynamics/prediction) with a domain-specific observation encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from muzero_model import MLP, ScalarSupport

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuctionConfig:
    # Slots over sampled continuous raises + one pass slot.
    action_space_size: int = 17
    max_players: int = 2
    num_item_types: int = 5
    num_items: int = 15
    history_length: int = 16
    upcoming_length: int = 5
    d_model: int = 256
    latent_dim: int = 512
    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    ff_hidden_dim: int = 1024
    value_support_max: int = 50
    reward_support_max: int = 20
    dropout: float = 0.1

    @property
    def max_sequence_length(self) -> int:
        # [CLS] + global + self + opponent + valuations + history + upcoming
        return 1 + 1 + 1 + 1 + self.num_item_types + self.history_length + self.upcoming_length


# ---------------------------------------------------------------------------
# Observation Encoder
# ---------------------------------------------------------------------------

class AuctionObservationEncoder(nn.Module):
    """Tokenised auction observation → latent vector via transformer."""

    def __init__(self, config: AuctionConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.small_shift = 128
        self.small_vocab = 256
        self.large_vocab = 512

        # Global token: [round, item_type, base_value, current_bid, leader_is_me, is_my_turn, remaining, num_raises, progress]
        self.glob_round_emb = nn.Embedding(self.large_vocab, d)
        self.glob_item_type_emb = nn.Embedding(config.num_item_types + 1, d)
        self.glob_base_value_emb = nn.Embedding(self.small_vocab, d)
        self.glob_current_bid_emb = nn.Embedding(self.large_vocab, d)
        self.glob_leader_emb = nn.Embedding(3, d)  # me=1, opp=0, nobody=2
        self.glob_my_turn_emb = nn.Embedding(2, d)
        self.glob_remaining_emb = nn.Embedding(self.large_vocab, d)
        self.glob_num_raises_emb = nn.Embedding(32, d)
        self.glob_progress_emb = nn.Embedding(3, d)

        # Self token: [player_id, budget, items_won, profit, spent, my_value]
        self.self_player_emb = nn.Embedding(config.max_players + 1, d)
        self.self_budget_emb = nn.Embedding(self.large_vocab, d)
        self.self_items_won_emb = nn.Embedding(config.num_items + 1, d)
        self.self_profit_emb = nn.Embedding(self.small_vocab, d)
        self.self_spent_emb = nn.Embedding(self.large_vocab, d)
        self.self_value_emb = nn.Embedding(self.small_vocab, d)

        # Opponent token: [player_id, items_won, spent]
        self.opp_player_emb = nn.Embedding(config.max_players + 1, d)
        self.opp_items_won_emb = nn.Embedding(config.num_items + 1, d)
        self.opp_spent_emb = nn.Embedding(self.large_vocab, d)

        # Valuation tokens: [type_id, modifier]
        self.val_type_emb = nn.Embedding(config.num_item_types + 1, d)
        self.val_mod_emb = nn.Embedding(self.small_vocab, d)

        # History tokens: [round, i_won, opp_won, unsold, price, item_type, base_value]
        self.hist_round_emb = nn.Embedding(self.large_vocab, d)
        self.hist_i_won_emb = nn.Embedding(2, d)
        self.hist_opp_won_emb = nn.Embedding(2, d)
        self.hist_unsold_emb = nn.Embedding(2, d)
        self.hist_price_emb = nn.Embedding(self.large_vocab, d)
        self.hist_item_type_emb = nn.Embedding(config.num_item_types + 1, d)
        self.hist_base_value_emb = nn.Embedding(self.small_vocab, d)

        # Upcoming tokens: [distance, item_type, base_value, my_value]
        self.up_dist_emb = nn.Embedding(config.upcoming_length + 1, d)
        self.up_type_emb = nn.Embedding(config.num_item_types + 1, d)
        self.up_base_emb = nn.Embedding(self.small_vocab, d)
        self.up_value_emb = nn.Embedding(self.small_vocab, d)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, config.max_sequence_length, d)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_attention_heads,
            dim_feedforward=config.ff_hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_transformer_layers
        )
        self.to_latent = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, config.latent_dim),
            nn.Tanh(),
        )

    @staticmethod
    def _clamp(t: Tensor, hi: int) -> Tensor:
        return t.long().clamp(0, hi)

    def _shift(self, t: Tensor) -> Tensor:
        return (t.long() + self.small_shift).clamp(0, self.small_vocab - 1)

    def _large(self, t: Tensor) -> Tensor:
        return t.long().clamp(0, self.large_vocab - 1)

    def _encode_global(self, g: Tensor) -> Tensor:
        return (
            self.glob_round_emb(self._large(g[:, 0]))
            + self.glob_item_type_emb(self._clamp(g[:, 1], self.config.num_item_types))
            + self.glob_base_value_emb(self._shift(g[:, 2]))
            + self.glob_current_bid_emb(self._large(g[:, 3]))
            + self.glob_leader_emb(self._clamp(g[:, 4], 2))
            + self.glob_my_turn_emb(self._clamp(g[:, 5], 1))
            + self.glob_remaining_emb(self._large(g[:, 6]))
            + self.glob_num_raises_emb(self._clamp(g[:, 7], 31))
            + self.glob_progress_emb(self._clamp(g[:, 8], 2))
        )

    def _encode_self(self, s: Tensor) -> Tensor:
        return (
            self.self_player_emb(self._clamp(s[:, 0], self.config.max_players))
            + self.self_budget_emb(self._large(s[:, 1]))
            + self.self_items_won_emb(self._clamp(s[:, 2], self.config.num_items))
            + self.self_profit_emb(self._shift(s[:, 3]))
            + self.self_spent_emb(self._large(s[:, 4]))
            + self.self_value_emb(self._shift(s[:, 5]))
        )

    def _encode_opponent(self, o: Tensor) -> Tensor:
        return (
            self.opp_player_emb(self._clamp(o[:, 0], self.config.max_players))
            + self.opp_items_won_emb(self._clamp(o[:, 1], self.config.num_items))
            + self.opp_spent_emb(self._large(o[:, 2]))
        )

    def _encode_valuations(self, v: Tensor) -> Tensor:
        return (
            self.val_type_emb(self._clamp(v[:, :, 0], self.config.num_item_types))
            + self.val_mod_emb(self._shift(v[:, :, 1]))
        )

    def _encode_history(self, h: Tensor) -> Tensor:
        return (
            self.hist_round_emb(self._large(h[:, :, 0]))
            + self.hist_i_won_emb(self._clamp(h[:, :, 1], 1))
            + self.hist_opp_won_emb(self._clamp(h[:, :, 2], 1))
            + self.hist_unsold_emb(self._clamp(h[:, :, 3], 1))
            + self.hist_price_emb(self._large(h[:, :, 4]))
            + self.hist_item_type_emb(self._clamp(h[:, :, 5], self.config.num_item_types))
            + self.hist_base_value_emb(self._shift(h[:, :, 6]))
        )

    def _encode_upcoming(self, u: Tensor) -> Tensor:
        return (
            self.up_dist_emb(self._clamp(u[:, :, 0], self.config.upcoming_length))
            + self.up_type_emb(self._clamp(u[:, :, 1], self.config.num_item_types))
            + self.up_base_emb(self._shift(u[:, :, 2]))
            + self.up_value_emb(self._shift(u[:, :, 3]))
        )

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        glob_emb = self._encode_global(tokens["global_token"]).unsqueeze(1)
        self_emb = self._encode_self(tokens["self_token"]).unsqueeze(1)
        opp_emb = self._encode_opponent(tokens["opponent_token"]).unsqueeze(1)
        val_emb = self._encode_valuations(tokens["valuation_tokens"])
        seq_parts = [glob_emb, self_emb, opp_emb, val_emb]

        hist = tokens.get("history_tokens")
        if hist is not None and hist.numel() > 0:
            seq_parts.append(self._encode_history(hist))

        upcoming = tokens.get("upcoming_tokens")
        if upcoming is not None and upcoming.numel() > 0:
            seq_parts.append(self._encode_upcoming(upcoming))

        seq = torch.cat(seq_parts, dim=1)
        B = seq.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, seq], dim=1)
        seq = seq + self.positional_embedding[:, : seq.size(1), :]
        encoded = self.transformer(seq)
        return self.to_latent(encoded[:, 0])


# ---------------------------------------------------------------------------
# Dynamics & Prediction
# ---------------------------------------------------------------------------

class AuctionDynamics(nn.Module):
    def __init__(self, config: AuctionConfig) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(config.action_space_size, config.latent_dim)
        self.transition = MLP(config.latent_dim * 2, config.ff_hidden_dim, config.latent_dim, config.dropout)
        self.reward_head = MLP(
            config.latent_dim * 2, config.ff_hidden_dim,
            2 * config.reward_support_max + 1, config.dropout,
        )
        self.latent_norm = nn.LayerNorm(config.latent_dim)

    def forward(self, latent: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        a_emb = self.action_embedding(action.long())
        x = torch.cat([latent, a_emb], dim=-1)
        delta = self.transition(x)
        next_latent = self.latent_norm(latent + delta)
        return next_latent, self.reward_head(x)


class AuctionPrediction(nn.Module):
    def __init__(self, config: AuctionConfig) -> None:
        super().__init__()
        self.policy_head = MLP(config.latent_dim, config.ff_hidden_dim, config.action_space_size, config.dropout)
        self.value_head = MLP(
            config.latent_dim, config.ff_hidden_dim,
            2 * config.value_support_max + 1, config.dropout,
        )

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        return self.policy_head(latent), self.value_head(latent)


# ---------------------------------------------------------------------------
# Inference output & Net
# ---------------------------------------------------------------------------

@dataclass
class AuctionInferenceOutput:
    hidden_state: Tensor
    policy_logits: Tensor
    value_logits: Tensor
    reward_logits: Tensor | None = None


class AuctionMuZeroNet(nn.Module):
    def __init__(self, config: AuctionConfig) -> None:
        super().__init__()
        self.config = config
        self.representation = AuctionObservationEncoder(config)
        self.dynamics = AuctionDynamics(config)
        self.prediction = AuctionPrediction(config)
        self.value_support = ScalarSupport(config.value_support_max)
        self.reward_support = ScalarSupport(config.reward_support_max)

    def initial_inference(self, tokens: dict[str, Tensor]) -> AuctionInferenceOutput:
        h = self.representation(tokens)
        p, v = self.prediction(h)
        return AuctionInferenceOutput(hidden_state=h, policy_logits=p, value_logits=v)

    def recurrent_inference(self, hidden: Tensor, action: Tensor) -> AuctionInferenceOutput:
        h, r = self.dynamics(hidden, action)
        p, v = self.prediction(h)
        return AuctionInferenceOutput(hidden_state=h, policy_logits=p, value_logits=v, reward_logits=r)

    @staticmethod
    def masked_policy_logits(logits: Tensor, mask: Tensor) -> Tensor:
        m = logits.clone()
        m[mask <= 0] = -1e9
        return m


# ---------------------------------------------------------------------------
# Belief-aware variant
# ---------------------------------------------------------------------------

class AuctionBeliefPrediction(nn.Module):
    def __init__(self, config: AuctionConfig) -> None:
        super().__init__()
        self.policy_head = MLP(config.latent_dim, config.ff_hidden_dim, config.action_space_size, config.dropout)
        self.value_head = MLP(
            config.latent_dim, config.ff_hidden_dim,
            2 * config.value_support_max + 1, config.dropout,
        )
        self.winner_head = MLP(config.latent_dim, config.ff_hidden_dim, config.max_players, config.dropout)
        self.rank_head = MLP(config.latent_dim, config.ff_hidden_dim, config.max_players, config.dropout)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.policy_head(hidden), self.value_head(hidden), self.winner_head(hidden), self.rank_head(hidden)


@dataclass
class AuctionBeliefInferenceOutput:
    hidden_state: Tensor
    policy_logits: Tensor
    value_logits: Tensor
    winner_logits: Tensor
    rank_logits: Tensor
    reward_logits: Tensor | None = None


class AuctionBeliefMuZeroNet(nn.Module):
    def __init__(self, config: AuctionConfig) -> None:
        super().__init__()
        self.config = config
        self.representation = AuctionObservationEncoder(config)
        self.dynamics = AuctionDynamics(config)
        self.prediction = AuctionBeliefPrediction(config)
        self.value_support = ScalarSupport(config.value_support_max)
        self.reward_support = ScalarSupport(config.reward_support_max)

        self.ego_player_emb = nn.Embedding(config.max_players + 1, config.latent_dim)
        self.current_player_emb = nn.Embedding(config.max_players + 1, config.latent_dim)
        self.num_players_emb = nn.Embedding(config.max_players + 1, config.latent_dim)
        self.condition_norm = nn.LayerNorm(config.latent_dim)

    def _condition_hidden(
        self, hidden: Tensor, ego: Tensor, current: Tensor, num_players: Tensor,
    ) -> Tensor:
        e = ego.long().clamp(0, self.config.max_players)
        c = current.long().clamp(0, self.config.max_players)
        n = num_players.long().clamp(1, self.config.max_players)
        return self.condition_norm(
            hidden + self.ego_player_emb(e) + self.current_player_emb(c) + self.num_players_emb(n)
        )

    def initial_inference(
        self, tokens: dict[str, Tensor],
        ego_player_id: Tensor, current_player_id: Tensor, num_players: Tensor,
    ) -> AuctionBeliefInferenceOutput:
        hidden = self.representation(tokens)
        conditioned = self._condition_hidden(hidden, ego_player_id, current_player_id, num_players)
        p, v, w, r = self.prediction(conditioned)
        return AuctionBeliefInferenceOutput(
            hidden_state=hidden, policy_logits=p, value_logits=v,
            winner_logits=w, rank_logits=r,
        )

    def recurrent_inference(
        self, hidden: Tensor, action: Tensor,
        ego_player_id: Tensor, current_player_id: Tensor, num_players: Tensor,
    ) -> AuctionBeliefInferenceOutput:
        next_hidden, reward_logits = self.dynamics(hidden, action)
        conditioned = self._condition_hidden(next_hidden, ego_player_id, current_player_id, num_players)
        p, v, w, r = self.prediction(conditioned)
        return AuctionBeliefInferenceOutput(
            hidden_state=next_hidden, policy_logits=p, value_logits=v,
            winner_logits=w, rank_logits=r, reward_logits=reward_logits,
        )


# ---------------------------------------------------------------------------
# Observation → tensor batching
# ---------------------------------------------------------------------------

_HIST_TOKEN_DIM = 7
_UPCOMING_TOKEN_DIM = 4
_VAL_TOKEN_DIM = 2
_PAD_HIST = [0] * _HIST_TOKEN_DIM
_PAD_UPCOMING = [0] * _UPCOMING_TOKEN_DIM


def auction_observation_batch_to_tensors(
    observations: list[dict[str, Any]],
    config: AuctionConfig | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    cfg = config or AuctionConfig()
    glob_batch, self_batch, opp_batch, val_batch = [], [], [], []
    hist_batch, up_batch = [], []

    for obs in observations:
        t = obs["tokens"]
        glob_batch.append(t["global_token"])
        self_batch.append(t["self_token"])
        opp_batch.append(t["opponent_token"])
        val_batch.append(t["valuation_tokens"])

        history = t.get("history_tokens", [])
        if len(history) >= cfg.history_length:
            history = history[-cfg.history_length:]
        else:
            pad = [list(_PAD_HIST) for _ in range(cfg.history_length - len(history))]
            history = pad + history
        hist_batch.append(history)

        upcoming = t.get("upcoming_tokens", [])
        if len(upcoming) >= cfg.upcoming_length:
            upcoming = upcoming[: cfg.upcoming_length]
        else:
            pad = [list(_PAD_UPCOMING) for _ in range(cfg.upcoming_length - len(upcoming))]
            upcoming = upcoming + pad
        up_batch.append(upcoming)

    return {
        "global_token": torch.tensor(glob_batch, dtype=torch.long, device=device),
        "self_token": torch.tensor(self_batch, dtype=torch.long, device=device),
        "opponent_token": torch.tensor(opp_batch, dtype=torch.long, device=device),
        "valuation_tokens": torch.tensor(val_batch, dtype=torch.long, device=device),
        "history_tokens": torch.tensor(hist_batch, dtype=torch.long, device=device),
        "upcoming_tokens": torch.tensor(up_batch, dtype=torch.long, device=device),
    }


def build_default_auction_config() -> AuctionConfig:
    return AuctionConfig()
