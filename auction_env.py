"""
Competitive Sequential Auction — environment for MuZero-style training.

Two players compete across multiple ascending-bid auction rounds.  Each round
features one item with a public base value; each player has private valuation
modifiers (drawn per-episode) that make the same item worth different amounts.
Budgets are private.

Per-round flow:
  1. Item revealed (type + base value) — public
  2. Players alternate with continuous raises or PASS
  3. PASS while opponent leads → opponent wins the item at their bid
  4. PASS when nobody has bid → item unsold, next round
  5. Winner pays bid, gains personal item value (base + private modifier)
  6. Next round; starting bidder alternates
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_PLAYERS = 2
NUM_ITEMS = 15
NUM_ITEM_TYPES = 5
HISTORY_LENGTH = 16
UNKNOWN_VALUE = -99
CONTINUOUS_PASS = -1.0
DEFAULT_RAISE_SAMPLES = 16

BASE_VALUES_BY_TYPE = [8, 12, 15, 10, 20]
BASE_VALUE_NOISE = 3
BUDGET_MIN = 80
BUDGET_MAX = 120
VALUATION_MOD_MIN = 0
VALUATION_MOD_MAX = 15
SIGNAL_NOISE = 3
VALUATION_PROFILE_TEMPLATES = [
    [14, 10, 6, 2, 0],
    [0, 14, 10, 6, 2],
    [2, 0, 14, 10, 6],
    [6, 2, 0, 14, 10],
    [10, 6, 2, 0, 14],
]

@dataclass
class ItemInfo:
    item_type: int
    base_value: int


@dataclass
class RoundResult:
    item: ItemInfo
    winner: int | None   # None if unsold
    price: float
    round_idx: int


class AuctionEnv:
    """Two-player competitive ascending-bid auction environment."""

    def __init__(
        self,
        seed: int = 42,
        num_items: int = NUM_ITEMS,
        num_item_types: int = NUM_ITEM_TYPES,
        profile_variation: float = 0.0,
        env_variant: str = "v2",
        min_raise_increment: float = 0.1,
    ) -> None:
        self.num_items = num_items
        self.num_item_types = num_item_types
        self.num_players = NUM_PLAYERS
        self._base_seed = seed
        self.rng = np.random.RandomState(seed)
        self.env_variant = str(env_variant)
        if self.env_variant != "v2":
            raise ValueError("AuctionEnv now supports only env_variant='v2'.")
        self.min_raise_increment = float(min_raise_increment)
        if self.min_raise_increment <= 0:
            raise ValueError("min_raise_increment must be > 0.")

        self.items: list[ItemInfo] = []
        self.budgets: list[float] = [0.0, 0.0]
        self.valuations: list[list[int]] = [[], []]  # [player][item_type]
        self.round_signals: list[list[int]] = []  # [round_idx][player] in {0,1,2}
        self.round_idx: int = 0
        self.current_bid: float = 0.0
        self.bid_leader: int = -1  # who placed the last raise (-1 = nobody)
        self.active_player: int = 0
        self.first_bidder_this_round: int = 0
        self.num_raises_this_round: int = 0
        self.round_history: list[RoundResult] = []
        self.items_won: list[int] = [0, 0]
        self.total_profit: list[float] = [0.0, 0.0]
        self.total_spent: list[float] = [0.0, 0.0]
        self.terminated: bool = False
        self.scores: list[float] = [0.0, 0.0]  # alias for compatibility
        self.global_step: int = 0

    @property
    def current_player(self) -> int:
        return self.active_player

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------
    def _generate_items(self) -> list[ItemInfo]:
        items: list[ItemInfo] = []
        items_per_type = max(1, self.num_items // self.num_item_types)
        for t in range(self.num_item_types):
            for _ in range(items_per_type):
                noise = int(self.rng.randint(-BASE_VALUE_NOISE, BASE_VALUE_NOISE + 1))
                bv = max(1, BASE_VALUES_BY_TYPE[t] + noise)
                items.append(ItemInfo(item_type=t, base_value=bv))
        while len(items) < self.num_items:
            t = int(self.rng.randint(0, self.num_item_types))
            noise = int(self.rng.randint(-BASE_VALUE_NOISE, BASE_VALUE_NOISE + 1))
            bv = max(1, BASE_VALUES_BY_TYPE[t] + noise)
            items.append(ItemInfo(item_type=t, base_value=bv))
        self.rng.shuffle(items)
        return items[: self.num_items]

    def _generate_budgets(self) -> list[float]:
        return [
            float(int(self.rng.randint(BUDGET_MIN, BUDGET_MAX + 1)))
            for _ in range(NUM_PLAYERS)
        ]

    def _generate_valuations(self) -> list[list[int]]:
        profile_ids = [int(self.rng.randint(0, len(VALUATION_PROFILE_TEMPLATES))) for _ in range(NUM_PLAYERS)]
        if profile_ids[0] == profile_ids[1]:
            profile_ids[1] = (profile_ids[1] + int(self.rng.randint(1, len(VALUATION_PROFILE_TEMPLATES)))) % len(
                VALUATION_PROFILE_TEMPLATES
            )
        vals: list[list[int]] = []
        for p in range(NUM_PLAYERS):
            base_profile = VALUATION_PROFILE_TEMPLATES[profile_ids[p]]
            player_vals = []
            for t in range(self.num_item_types):
                noise = int(self.rng.randint(-2, 3))
                v = int(max(VALUATION_MOD_MIN, min(VALUATION_MOD_MAX, base_profile[t] + noise)))
                player_vals.append(v)
            vals.append(player_vals)
        return vals

    def _generate_round_signals(self) -> list[list[int]]:
        """
        Build a public, noisy signal bucket (0/1/2) for each player and round.
        In v2 this encodes a coarse clue about each player's private value modifier
        on the current item's type.
        """
        signals: list[list[int]] = []
        for item in self.items:
            row: list[int] = []
            for player in range(NUM_PLAYERS):
                true_mod = int(self.valuations[player][item.item_type])
                noisy = true_mod + int(self.rng.randint(-SIGNAL_NOISE, SIGNAL_NOISE + 1))
                if noisy <= 4:
                    bucket = 0
                elif noisy <= 9:
                    bucket = 1
                else:
                    bucket = 2
                row.append(bucket)
            signals.append(row)
        return signals

    def personal_value(self, player: int, item: ItemInfo) -> int:
        return item.base_value + self.valuations[player][item.item_type]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._base_seed = seed
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(self._base_seed)

        self.items = self._generate_items()
        self.budgets = self._generate_budgets()
        self.valuations = self._generate_valuations()
        self.round_signals = self._generate_round_signals()
        self.round_idx = 0
        self.current_bid = 0.0
        self.bid_leader = -1
        self.first_bidder_this_round = int(self.rng.randint(0, NUM_PLAYERS))
        self.active_player = self.first_bidder_this_round
        self.num_raises_this_round = 0
        self.round_history = []
        self.items_won = [0, 0]
        self.total_profit = [0.0, 0.0]
        self.total_spent = [0.0, 0.0]
        self.terminated = False
        self.scores = [0.0, 0.0]
        self.global_step = 0

        return self.observe(self.active_player)

    def legal_raise_range(self) -> tuple[float, float] | None:
        if self.terminated or self.round_idx >= self.num_items:
            return None
        max_raise = float(self.budgets[self.active_player] - self.current_bid)
        if max_raise < self.min_raise_increment:
            return None
        return (self.min_raise_increment, max_raise)

    def _is_legal_continuous_raise(self, raise_amount: float) -> bool:
        if not math.isfinite(raise_amount):
            return False
        if raise_amount < self.min_raise_increment:
            return False
        rr = self.legal_raise_range()
        if rr is None:
            return False
        _, hi = rr
        return raise_amount <= hi + 1e-9

    def _encode_amount(self, value: float) -> int:
        return int(round(value / self.min_raise_increment))

    def sample_action_candidates(self, num_raise_samples: int = DEFAULT_RAISE_SAMPLES) -> list[float]:
        """
        Return legal continuous actions for planners/policies:
        [sampled_raise_1, ..., sampled_raise_k, CONTINUOUS_PASS].
        """
        rr = self.legal_raise_range()
        if rr is None:
            return [CONTINUOUS_PASS]
        lo, hi = rr
        if hi <= lo + 1e-9:
            return [round(lo, 6), CONTINUOUS_PASS]
        k = max(1, int(num_raise_samples))
        if k == 1:
            raise_samples = [lo]
        else:
            raise_samples = np.linspace(lo, hi, num=k).tolist()
        # Deduplicate numerically-close samples and quantize to stable values.
        dedup: list[float] = []
        for a in raise_samples:
            q = round(float(a), 6)
            if not dedup or abs(dedup[-1] - q) > 1e-9:
                dedup.append(q)
        dedup.append(CONTINUOUS_PASS)
        return dedup

    def step(self, action: float) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("Game terminated. Call reset().")

        self.global_step += 1
        actor = self.active_player
        rewards: dict[str, float] = {f"player_{p}": 0.0 for p in range(NUM_PLAYERS)}

        action_float = float(action)
        if math.isclose(action_float, CONTINUOUS_PASS, abs_tol=1e-9):
            self._resolve_round(rewards)
        else:
            if not self._is_legal_continuous_raise(action_float):
                rr = self.legal_raise_range()
                raise ValueError(
                    f"Illegal continuous raise {action_float}. "
                    f"pass={CONTINUOUS_PASS}, legal_raise_range={rr}"
                )
            self.current_bid += action_float
            self.bid_leader = actor
            self.num_raises_this_round += 1
            self.active_player = 1 - actor

        if self.round_idx >= self.num_items:
            self.terminated = True
            self.scores = list(self.total_profit)
            winner = 0 if self.total_profit[0] > self.total_profit[1] else (
                1 if self.total_profit[1] > self.total_profit[0] else -1
            )
            for p in range(NUM_PLAYERS):
                if winner == -1:
                    pass  # tie: 0 reward
                elif p == winner:
                    rewards[f"player_{p}"] += 1.0
                else:
                    rewards[f"player_{p}"] -= 1.0

        obs = self.observe(self.active_player)
        return obs, rewards, self.terminated, {
            "round_idx": self.round_idx,
            "continuous_pass_action": CONTINUOUS_PASS,
            "legal_raise_range": self.legal_raise_range(),
        }

    def _resolve_round(self, rewards: dict[str, float]) -> None:
        item = self.items[self.round_idx]
        if self.bid_leader >= 0:
            winner = self.bid_leader
            price = float(self.current_bid)
            value = self.personal_value(winner, item)
            surplus = value - price
            self.budgets[winner] -= price
            self.total_spent[winner] += price
            self.total_profit[winner] += surplus
            self.items_won[winner] += 1
            rewards[f"player_{winner}"] += surplus / 50.0
            self.round_history.append(
                RoundResult(item=item, winner=winner, price=price, round_idx=self.round_idx)
            )
        else:
            self.round_history.append(
                RoundResult(item=item, winner=None, price=0.0, round_idx=self.round_idx)
            )

        self.round_idx += 1
        self.current_bid = 0.0
        self.bid_leader = -1
        self.num_raises_this_round = 0
        self.first_bidder_this_round = 1 - self.first_bidder_this_round
        self.active_player = self.first_bidder_this_round

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observe(self, observer: int) -> dict[str, Any]:
        opp = 1 - observer
        item = self.items[self.round_idx] if self.round_idx < self.num_items else ItemInfo(0, 0)
        rounds_remaining = max(0, self.num_items - self.round_idx)
        bid_leader_is_me = 1 if self.bid_leader == observer else (0 if self.bid_leader == opp else 2)
        is_my_turn = int(self.active_player == observer)
        if self.round_idx < self.num_items and self.round_signals:
            round_signal = int(self.round_signals[self.round_idx][self.active_player])
        else:
            round_signal = 0

        my_value_for_item = self.personal_value(observer, item) if self.round_idx < self.num_items else 0

        global_token = [
            self.round_idx,
            item.item_type,
            item.base_value,
            self._encode_amount(self.current_bid),
            bid_leader_is_me,
            is_my_turn,
            rounds_remaining,
            min(self.num_raises_this_round, 31),
            round_signal,
        ]

        self_token = [
            observer,
            self._encode_amount(self.budgets[observer]),
            self.items_won[observer],
            int(round(self.total_profit[observer])),
            self._encode_amount(self.total_spent[observer]),
            my_value_for_item,
        ]

        opponent_token = [
            opp,
            self.items_won[opp],
            self._encode_amount(self.total_spent[opp]),
        ]

        valuation_tokens: list[list[int]] = []
        for t in range(self.num_item_types):
            valuation_tokens.append([t, self.valuations[observer][t]])

        history_tokens: list[list[int]] = []
        for rr in self.round_history:
            i_won = int(rr.winner == observer) if rr.winner is not None else 0
            opp_won = int(rr.winner == opp) if rr.winner is not None else 0
            unsold = int(rr.winner is None)
            history_tokens.append([
                rr.round_idx,
                i_won,
                opp_won,
                unsold,
                self._encode_amount(rr.price),
                rr.item.item_type,
                rr.item.base_value,
            ])

        upcoming_tokens: list[list[int]] = []
        for i in range(self.round_idx + 1, min(self.round_idx + 6, self.num_items)):
            fut = self.items[i]
            upcoming_tokens.append([
                i - self.round_idx,
                fut.item_type,
                fut.base_value,
                self.personal_value(observer, fut),
            ])

        tokens = {
            "global_token": global_token,
            "self_token": self_token,
            "opponent_token": opponent_token,
            "valuation_tokens": valuation_tokens,
            "history_tokens": history_tokens,
            "upcoming_tokens": upcoming_tokens,
        }

        return {
            "current_player": self.active_player,
            "observer": observer,
            "round_idx": self.round_idx,
            "active_player": self.active_player,
            "current_bid": self.current_bid,
            "bid_leader": self.bid_leader,
            "num_raises_this_round": self.num_raises_this_round,
            "min_raise_increment": self.min_raise_increment,
            "scores": list(self.total_profit),
            "env_variant": self.env_variant,
            "tokens": tokens,
        }

    # ------------------------------------------------------------------
    # Ground truth (for belief training targets)
    # ------------------------------------------------------------------
    def opponent_budget(self, observer: int) -> float:
        return self.budgets[1 - observer]

    def opponent_valuations(self, observer: int) -> list[int]:
        return list(self.valuations[1 - observer])


# ---------------------------------------------------------------------------
# Heuristic agents
# ---------------------------------------------------------------------------

def random_action(obs: dict[str, Any], legal: list[float], rng: np.random.RandomState | None = None) -> float:
    if not legal:
        return CONTINUOUS_PASS
    if rng is None:
        return float(np.random.choice(legal))
    return float(rng.choice(legal))


def value_bidder_action(obs: dict[str, Any], legal: list[float]) -> float:
    """Bid up to personal value for the current item, then pass."""
    tokens = obs["tokens"]
    my_value = tokens["self_token"][5]
    current_bid = tokens["global_token"][3]
    raises = sorted([a for a in legal if not math.isclose(a, CONTINUOUS_PASS, abs_tol=1e-9)], reverse=True)
    for a in raises:
        if current_bid + a <= my_value:
            return float(a)
    return CONTINUOUS_PASS


def conservative_bidder_action(obs: dict[str, Any], legal: list[float]) -> float:
    """Bid up to 70% of personal value."""
    tokens = obs["tokens"]
    my_value = tokens["self_token"][5]
    current_bid = tokens["global_token"][3]
    threshold = float(my_value * 0.7)
    raises = sorted([a for a in legal if not math.isclose(a, CONTINUOUS_PASS, abs_tol=1e-9)])
    for a in raises:
        if current_bid + a <= threshold:
            return float(a)
    return CONTINUOUS_PASS


def aggressive_bidder_action(obs: dict[str, Any], legal: list[float]) -> float:
    """Bid up to 120% of personal value (willing to overpay)."""
    tokens = obs["tokens"]
    my_value = tokens["self_token"][5]
    current_bid = tokens["global_token"][3]
    threshold = float(my_value * 1.2)
    raises = sorted([a for a in legal if not math.isclose(a, CONTINUOUS_PASS, abs_tol=1e-9)], reverse=True)
    for a in raises:
        if current_bid + a <= threshold:
            return float(a)
    return CONTINUOUS_PASS
