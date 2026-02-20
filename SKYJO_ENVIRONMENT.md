# Skyjo RL Environment (CLI, Deterministic, Seedable)

This project includes a full Skyjo environment suitable for reinforcement learning and self-play.

## Goals Covered

The environment was designed to satisfy:

1. **Official player count support**: 2-8 players.
2. **Determinism**: explicit RNG seed controls shuffling and reproducible simulations.
3. **Full state tracking**:
   - full remaining draw deck order
   - each player’s 12 cards
   - face-up / face-down flags
   - discard pile contents + top
   - current player, phase, round and turn metadata
4. **Agent observation + tokenization**:
   - own and opponents' visible information with `UNKNOWN` masking
   - discard top, deck size, phase features
   - public action history window
   - transformer-friendly token structure
5. **Discrete action IDs over macro-turn actions**:
   - `TAKE_DISCARD_AND_REPLACE(pos)`
   - `DRAW_DECK_KEEP_AND_REPLACE(pos)`
   - `DRAW_DECK_DISCARD_AND_FLIP(pos)`
   - plus setup reveal actions for initial 2 flips per player

No GUI is used; everything is CLI and Python API based.

---

## File Layout

- `skyjo_env.py`: complete environment implementation + CLI commands.
- `main.py`: entrypoint, delegates to CLI parser.

---

## Rules Implementation Summary

### Deck

The deck uses standard Skyjo values:

- `-2` x5
- `-1` x10
- `0` x15
- `1..12` x10 each

### Board

Each player has 12 cards in a `3x4` layout (index mapping row-major: `0..11`).

Column definitions for clear checks:

- column 0 => positions `(0, 4, 8)`
- column 1 => positions `(1, 5, 9)`
- column 2 => positions `(2, 6, 10)`
- column 3 => positions `(3, 7, 11)`

When all 3 cards in a column are visible and equal, that column is removed and no longer contributes to score.

### Round flow

1. Deal 12 cards to each player.
2. Start discard pile with one deck card.
3. Setup: each player reveals 2 cards (manual or deterministic auto mode).
4. Starting player is chosen by highest sum of revealed setup cards (deterministic tie-break by lowest player index).
5. Main turns follow action rules.
6. When a player has all active cards revealed after their turn, round end is triggered:
   - each other player gets one final turn.
7. Round scores are summed from non-removed cards.
8. Ender penalty: if round-ending player does **not** have strictly lowest round score, that player’s round score is doubled.
9. Cumulative scoring continues until any player reaches `score_limit` (default 100). Lowest cumulative score wins.

### Draw deck refill

If draw deck empties, discard pile (except top) is shuffled with the same deterministic RNG and reused as draw deck.

---

## Determinism

All randomness is controlled by `random.Random(seed)` inside `SkyjoEnv`.

Deterministic components:

- deck shuffling
- auto setup reveals (if `manual_initial_reveals=False`)
- random simulation helper when you pass deterministic seeds

Setup mode control:

- `setup_mode="auto"`: initial two reveals per player are chosen randomly, seeded and reproducible.
- `setup_mode="manual"`: players choose setup reveal positions through setup actions.

To reproduce runs:

- use same seed
- same player count
- same sequence of action IDs

---

## State Model

Call:

```python
state = env.get_full_state()
```

State includes:

- `deck_order_remaining`
- `discard_pile`
- each board’s raw `cards`, `visible`, `removed`
- turn metadata: `current_player`, `phase`, `round_index`, `global_step`, `turns_in_round`
- setup bookkeeping
- round-end bookkeeping
- scores

This is suitable for exact replay and debugging.

---

## Observation Model

Call:

```python
obs = env.observe(observer_player=0, history_k=16)
```

Returns:

- `board_views[owner]["values"]`: 12 values where hidden cards become `UNKNOWN_VALUE (-99)`
- `board_views[owner]["visible_mask"]`: 12 binary flags
- `discard_top`, `deck_size`
- phase/current player metadata
- `history`: last `K` public actions
- `history_since_last_turn`: public actions since observer’s previous turn context window
- `tokens` payload for transformer input

### Token schema

`tokens["board_tokens"]`:

- one token per board slot (`12 * num_players`)
- each token:
  - `[owner_id, pos_id, visible_flag, value_or_unknown]`

`tokens["discard_token"]`:

- `[discard_top, discard_size]`

`tokens["global_token"]`:

- `[deck_size, global_step, current_player, phase_id, round_phase_bucket, round_index]`
- round phase bucket: `0=early`, `1=mid`, `2=late`

`tokens["action_tokens"]`:

- up to `K` recent public actions
- each token:
  - `[actor, action_type, source, target_pos, a_value, b_value]`

`tokens["action_tokens_since_last_turn"]`:

- same structure, but filtered for context since observer’s last turn window.

---

## Action Space

Action IDs are fixed and discrete:

- `0..11`: `TAKE_DISCARD_AND_REPLACE(pos)`
- `12..23`: `DRAW_DECK_KEEP_AND_REPLACE(pos)`
- `24..35`: `DRAW_DECK_DISCARD_AND_FLIP(pos)`
- `36..47`: `SETUP_FLIP(pos)` (used only during setup phase)

Helpers:

```python
action_id = macro_to_action("DRAW_DECK_KEEP_AND_REPLACE", 5)
macro, pos = action_to_macro(action_id)
```

Always check legal actions:

```python
legal = env.legal_actions()
```

---

## Gym-style stepping

```python
next_obs, rewards, terminated, info = env.step(action_id)
```

- `next_obs`: observation for `env.current_player` after transition
- `rewards`:
  - per-player sparse terminal win/loss reward on game end
  - negative round scores when rounds end
- `terminated`: true when game over
- `info`: action and bookkeeping metadata

---

## CLI Usage

Run through `main.py`.

### 1) Manual play

```bash
python main.py play --players 3 --seed 42
```

Optional: force automatic setup reveals even in play mode:

```bash
python main.py play --players 3 --seed 42 --setup-mode auto
```

### 2) Self-play simulation

```bash
python main.py simulate --players 4 --games 20 --seed 100 --setup-mode auto
```

### 3) Inspect state and tokens

```bash
python main.py inspect --players 2 --seed 7 --steps 5 --setup-mode auto
```

---

## Python API Example

```python
from skyjo_env import SkyjoEnv

env = SkyjoEnv(num_players=4, seed=123, history_window_k=24, setup_mode="auto")
obs = env.reset()

done = False
while not done:
    legal = env.legal_actions()
    action = legal[0]
    obs, rewards, done, info = env.step(action)
```

---

## RL integration tips

- Use `get_full_state()` for deterministic replay datasets.
- Use `observe(player)` for partial-information training.
- Feed `tokens` directly into transformer pipelines.
- Keep action masking from `legal_actions()` during policy sampling.
- For large-scale self-play, run many env instances with deterministic seed schedules.
