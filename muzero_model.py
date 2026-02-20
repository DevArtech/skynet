from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MuZeroConfig:
    action_space_size: int = 36
    max_players: int = 8
    board_size: int = 12
    history_length: int = 16
    d_model: int = 256
    latent_dim: int = 512
    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    ff_hidden_dim: int = 1024
    value_support_max: int = 200
    reward_support_max: int = 50
    dropout: float = 0.1

    @property
    def max_board_tokens(self) -> int:
        return self.max_players * self.board_size

    @property
    def max_sequence_length(self) -> int:
        # [CLS] + board + discard + global + action-history
        return 1 + self.max_board_tokens + 1 + 1 + self.history_length


class ScalarSupport:
    """Converts scalar targets to support distributions and back."""

    def __init__(self, max_value: int) -> None:
        if max_value <= 0:
            raise ValueError("max_value must be > 0")
        self.max_value = max_value
        self.size = (2 * max_value) + 1

    def scalar_to_logits_target(self, scalars: Tensor) -> Tensor:
        clipped = scalars.clamp(-self.max_value, self.max_value)
        floor = torch.floor(clipped)
        prob_high = clipped - floor
        low_idx = floor + self.max_value
        high_idx = (low_idx + 1).clamp(0, self.size - 1)
        low_idx = low_idx.long()
        high_idx = high_idx.long()

        target = torch.zeros((*scalars.shape, self.size), device=scalars.device, dtype=scalars.dtype)
        target.scatter_add_(-1, low_idx.unsqueeze(-1), (1.0 - prob_high).unsqueeze(-1))
        target.scatter_add_(-1, high_idx.unsqueeze(-1), prob_high.unsqueeze(-1))
        return target

    def logits_to_scalar(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        support = torch.arange(-self.max_value, self.max_value + 1, device=logits.device, dtype=logits.dtype)
        return torch.sum(probs * support, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SkyjoObservationEncoder(nn.Module):
    """
    Encodes the environment token schema into a latent state for MuZero.

    Expected token dictionary keys:
    - board_tokens: [B, N_board, 4] -> [owner, pos, visible, value]
    - discard_token: [B, 2] -> [discard_top, discard_size]
    - global_token: [B, 6] -> [deck_size, step, current_player, phase, phase_bucket, round_index]
    - action_tokens: [B, N_hist, 6] -> public action tokens
    - decision_token: [B, 2] -> [decision_phase_id, current_drawn_value]
    """

    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.config = config

        # Ranges include unknown/placeholder values from the environment.
        self.small_shift = 128
        self.small_vocab = 256
        self.large_shift = 256
        self.large_vocab = 1024

        # Board token feature embeddings.
        self.board_owner_emb = nn.Embedding(config.max_players + 1, config.d_model)
        self.board_pos_emb = nn.Embedding(config.board_size, config.d_model)
        self.board_visible_emb = nn.Embedding(2, config.d_model)
        self.board_value_emb = nn.Embedding(self.small_vocab, config.d_model)

        # Discard/global/action feature embeddings.
        self.discard_top_emb = nn.Embedding(self.small_vocab, config.d_model)
        self.discard_size_emb = nn.Embedding(self.large_vocab, config.d_model)

        self.deck_size_emb = nn.Embedding(self.large_vocab, config.d_model)
        self.step_emb = nn.Embedding(self.large_vocab, config.d_model)
        self.current_player_emb = nn.Embedding(config.max_players + 1, config.d_model)
        self.phase_emb = nn.Embedding(4, config.d_model)
        self.phase_bucket_emb = nn.Embedding(3, config.d_model)
        self.round_index_emb = nn.Embedding(self.large_vocab, config.d_model)
        self.decision_phase_emb = nn.Embedding(8, config.d_model)
        self.current_drawn_emb = nn.Embedding(self.small_vocab, config.d_model)

        self.action_actor_emb = nn.Embedding(config.max_players + 1, config.d_model)
        self.action_type_emb = nn.Embedding(8, config.d_model)
        self.action_source_emb = nn.Embedding(self.small_vocab, config.d_model)
        self.action_target_pos_emb = nn.Embedding(config.board_size, config.d_model)
        self.action_a_value_emb = nn.Embedding(self.small_vocab, config.d_model)
        self.action_b_value_emb = nn.Embedding(self.small_vocab, config.d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.positional_embedding = nn.Parameter(torch.zeros(1, config.max_sequence_length, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_attention_heads,
            dim_feedforward=config.ff_hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_transformer_layers)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.latent_dim),
            nn.Tanh(),
        )

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        board_tokens = tokens["board_tokens"]
        discard_token = tokens["discard_token"]
        global_token = tokens["global_token"]
        action_tokens = tokens.get("action_tokens")
        decision_token = tokens.get("decision_token")

        board_emb = self._encode_board_tokens(board_tokens)
        discard_emb = self._encode_discard_token(discard_token).unsqueeze(1)
        global_emb = self._encode_global_token(global_token).unsqueeze(1)
        seq_parts = [board_emb, discard_emb, global_emb]

        if action_tokens is not None and action_tokens.numel() > 0:
            seq_parts.append(self._encode_action_tokens(action_tokens))
        if decision_token is not None and decision_token.numel() > 0:
            seq_parts.append(self._encode_decision_token(decision_token).unsqueeze(1))

        seq = torch.cat(seq_parts, dim=1)
        batch_size = seq.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, seq], dim=1)

        if seq.size(1) > self.config.max_sequence_length:
            raise ValueError(
                f"Encoded sequence length {seq.size(1)} exceeds maximum "
                f"{self.config.max_sequence_length}. Increase config limits."
            )

        seq = seq + self.positional_embedding[:, : seq.size(1), :]
        encoded = self.transformer(seq)
        return self.to_latent(encoded[:, 0])

    def _encode_board_tokens(self, board_tokens: Tensor) -> Tensor:
        owner = self._clamp_index(board_tokens[:, :, 0], self.config.max_players)
        pos = self._clamp_index(board_tokens[:, :, 1], self.config.board_size - 1)
        visible = self._clamp_index(board_tokens[:, :, 2], 1)
        value = self._shift_clamp(board_tokens[:, :, 3], self.small_shift, self.small_vocab)
        return (
            self.board_owner_emb(owner)
            + self.board_pos_emb(pos)
            + self.board_visible_emb(visible)
            + self.board_value_emb(value)
        )

    def _encode_discard_token(self, discard_token: Tensor) -> Tensor:
        top = self._shift_clamp(discard_token[:, 0], self.small_shift, self.small_vocab)
        size = self._clamp_index(discard_token[:, 1], self.large_vocab - 1)
        return self.discard_top_emb(top) + self.discard_size_emb(size)

    def _encode_global_token(self, global_token: Tensor) -> Tensor:
        deck_size = self._clamp_index(global_token[:, 0], self.large_vocab - 1)
        step = self._clamp_index(global_token[:, 1], self.large_vocab - 1)
        current_player = self._clamp_index(global_token[:, 2], self.config.max_players)
        phase = self._clamp_index(global_token[:, 3], 3)
        phase_bucket = self._clamp_index(global_token[:, 4], 2)
        round_index = self._clamp_index(global_token[:, 5], self.large_vocab - 1)
        return (
            self.deck_size_emb(deck_size)
            + self.step_emb(step)
            + self.current_player_emb(current_player)
            + self.phase_emb(phase)
            + self.phase_bucket_emb(phase_bucket)
            + self.round_index_emb(round_index)
        )

    def _encode_action_tokens(self, action_tokens: Tensor) -> Tensor:
        actor = self._clamp_index(action_tokens[:, :, 0], self.config.max_players)
        action_type = self._clamp_index(action_tokens[:, :, 1], 7)
        source = self._shift_clamp(action_tokens[:, :, 2], self.small_shift, self.small_vocab)
        target_pos = self._clamp_index(action_tokens[:, :, 3], self.config.board_size - 1)
        a_value = self._shift_clamp(action_tokens[:, :, 4], self.small_shift, self.small_vocab)
        b_value = self._shift_clamp(action_tokens[:, :, 5], self.small_shift, self.small_vocab)
        return (
            self.action_actor_emb(actor)
            + self.action_type_emb(action_type)
            + self.action_source_emb(source)
            + self.action_target_pos_emb(target_pos)
            + self.action_a_value_emb(a_value)
            + self.action_b_value_emb(b_value)
        )

    def _encode_decision_token(self, decision_token: Tensor) -> Tensor:
        phase_id = self._clamp_index(decision_token[:, 0], 7)
        drawn_value = self._shift_clamp(decision_token[:, 1], self.small_shift, self.small_vocab)
        return self.decision_phase_emb(phase_id) + self.current_drawn_emb(drawn_value)

    @staticmethod
    def _clamp_index(index_tensor: Tensor, max_index: int) -> Tensor:
        return index_tensor.long().clamp(0, max_index)

    @staticmethod
    def _shift_clamp(index_tensor: Tensor, shift: int, vocab_size: int) -> Tensor:
        return (index_tensor.long() + shift).clamp(0, vocab_size - 1)


class MuZeroDynamics(nn.Module):
    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(config.action_space_size, config.latent_dim)
        self.transition = MLP(
            in_dim=config.latent_dim * 2,
            hidden_dim=config.ff_hidden_dim,
            out_dim=config.latent_dim,
            dropout=config.dropout,
        )
        self.reward_head = MLP(
            in_dim=config.latent_dim * 2,
            hidden_dim=config.ff_hidden_dim,
            out_dim=(2 * config.reward_support_max) + 1,
            dropout=config.dropout,
        )
        self.latent_norm = nn.LayerNorm(config.latent_dim)

    def forward(self, latent: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        action_emb = self.action_embedding(action.long())
        x = torch.cat([latent, action_emb], dim=-1)
        delta = self.transition(x)
        next_latent = self.latent_norm(latent + delta)
        reward_logits = self.reward_head(x)
        return next_latent, reward_logits


class MuZeroPrediction(nn.Module):
    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.policy_head = MLP(
            in_dim=config.latent_dim,
            hidden_dim=config.ff_hidden_dim,
            out_dim=config.action_space_size,
            dropout=config.dropout,
        )
        self.value_head = MLP(
            in_dim=config.latent_dim,
            hidden_dim=config.ff_hidden_dim,
            out_dim=(2 * config.value_support_max) + 1,
            dropout=config.dropout,
        )

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        policy_logits = self.policy_head(latent)
        value_logits = self.value_head(latent)
        return policy_logits, value_logits


@dataclass
class MuZeroInferenceOutput:
    hidden_state: Tensor
    policy_logits: Tensor
    value_logits: Tensor
    reward_logits: Tensor | None = None


class MuZeroNet(nn.Module):
    """
    Classical MuZero network split into representation, dynamics, prediction.
    """

    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.config = config
        self.representation = SkyjoObservationEncoder(config)
        self.dynamics = MuZeroDynamics(config)
        self.prediction = MuZeroPrediction(config)
        self.value_support = ScalarSupport(config.value_support_max)
        self.reward_support = ScalarSupport(config.reward_support_max)

    def initial_inference(self, observation_tokens: dict[str, Tensor]) -> MuZeroInferenceOutput:
        hidden_state = self.representation(observation_tokens)
        policy_logits, value_logits = self.prediction(hidden_state)
        return MuZeroInferenceOutput(
            hidden_state=hidden_state,
            policy_logits=policy_logits,
            value_logits=value_logits,
            reward_logits=None,
        )

    def recurrent_inference(self, hidden_state: Tensor, action: Tensor) -> MuZeroInferenceOutput:
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden_state)
        return MuZeroInferenceOutput(
            hidden_state=next_hidden_state,
            policy_logits=policy_logits,
            value_logits=value_logits,
            reward_logits=reward_logits,
        )

    def unroll(self, hidden_state: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Unrolls K recurrent steps.

        Args:
            hidden_state: [B, latent_dim]
            actions: [B, K]

        Returns:
            policy_logits: [B, K, action_space_size]
            value_logits: [B, K, value_support_size]
            reward_logits: [B, K, reward_support_size]
        """
        policy_steps: list[Tensor] = []
        value_steps: list[Tensor] = []
        reward_steps: list[Tensor] = []

        state = hidden_state
        for k in range(actions.size(1)):
            state, reward_logits = self.dynamics(state, actions[:, k])
            policy_logits, value_logits = self.prediction(state)
            policy_steps.append(policy_logits)
            value_steps.append(value_logits)
            reward_steps.append(reward_logits)

        return torch.stack(policy_steps, dim=1), torch.stack(value_steps, dim=1), torch.stack(reward_steps, dim=1)

    @staticmethod
    def masked_policy_logits(policy_logits: Tensor, legal_action_mask: Tensor) -> Tensor:
        """
        Applies a legal-action mask before sampling or argmax.

        legal_action_mask shape: [B, action_space_size], values in {0,1}
        """
        masked = policy_logits.clone()
        masked[legal_action_mask <= 0] = -1e9
        return masked


def observation_batch_to_tensors(
    observations: list[dict[str, Any]],
    history_length: int = 16,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    """
    Converts environment observations into batched token tensors.
    """
    board_tokens = []
    discard_tokens = []
    global_tokens = []
    action_tokens = []
    decision_tokens = []

    for obs in observations:
        tokens = obs["tokens"]
        board_tokens.append(tokens["board_tokens"])
        discard_tokens.append(tokens["discard_token"])
        global_tokens.append(tokens["global_token"])

        history = tokens.get("action_tokens", [])
        if len(history) >= history_length:
            history = history[-history_length:]
        else:
            pad = [[0, 0, 0, 0, 0, 0] for _ in range(history_length - len(history))]
            history = pad + history
        action_tokens.append(history)
        decision_tokens.append(tokens.get("decision_token", [0, 0]))

    return {
        "board_tokens": torch.tensor(board_tokens, dtype=torch.long, device=device),
        "discard_token": torch.tensor(discard_tokens, dtype=torch.long, device=device),
        "global_token": torch.tensor(global_tokens, dtype=torch.long, device=device),
        "action_tokens": torch.tensor(action_tokens, dtype=torch.long, device=device),
        "decision_token": torch.tensor(decision_tokens, dtype=torch.long, device=device),
    }


def build_default_skyjo_muzero_config(action_space_size: int = 36) -> MuZeroConfig:
    return MuZeroConfig(action_space_size=action_space_size)
