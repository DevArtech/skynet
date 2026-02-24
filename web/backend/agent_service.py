from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from belief_muzero_mcts import run_belief_mcts
from belief_muzero_model import BeliefAwareMuZeroNet, build_default_belief_muzero_config
from muzero_mcts import MCTSConfig, run_mcts
from muzero_model import MuZeroNet, build_default_skyjo_muzero_config
from skyjo_decision_env import DecisionAction, DecisionPhase, PendingTurnState, SkyjoDecisionEnv
from skyjo_env import SkyjoEnv, action_to_macro, macro_to_action
from heuristic_bots import make_heuristic_bot

from .schemas import AIActionResult, AgentConfig, AgentType
from .checkpoints import find_latest_checkpoint


class AgentService:
    def __init__(self) -> None:
        self._baseline_cache: dict[tuple[str, str], MuZeroNet] = {}
        self._belief_cache: dict[tuple[str, str], BeliefAwareMuZeroNet] = {}
        self._project_root = Path(__file__).resolve().parents[2]

    def _heuristic_seed(self, observation: dict[str, Any]) -> int:
        # Keep heuristic behavior deterministic for a given game state while
        # still varying decisions across different states.
        parts = (
            int(observation.get("current_player", 0)),
            int(observation.get("global_step", 0)),
            int(observation.get("round_step", 0)),
            int(observation.get("deck_size", 0)),
            int(observation.get("discard_top", 0)),
            str(observation.get("decision_phase", "")),
        )
        return hash(parts) & 0x7FFFFFFF

    def _resolve_checkpoint(self, agent_type: AgentType, configured_path: str | None) -> Path:
        if agent_type == AgentType.HEURISTIC:
            raise ValueError("Heuristic agents do not use checkpoints.")
        if configured_path:
            path = Path(configured_path)
            if not path.is_absolute():
                path = (self._project_root / path).resolve()
            if not path.exists():
                raise ValueError(f"Checkpoint does not exist: {path}")
            return path

        default_dir = (
            self._project_root / "runs" / "muzero_baseline" / "checkpoints"
            if agent_type == AgentType.BASELINE
            else self._project_root / "runs" / "muzero_belief" / "checkpoints"
        )
        latest = find_latest_checkpoint(default_dir)
        if latest is None:
            raise ValueError(
                f"No checkpoint configured and none found in {default_dir}. "
                "Provide agent.checkpoint_path in create game request."
            )
        return latest

    def _load_baseline(self, checkpoint: Path, device: str) -> MuZeroNet:
        cache_key = (str(checkpoint), device)
        cached = self._baseline_cache.get(cache_key)
        if cached is not None:
            return cached
        dev = torch.device(device)
        ckpt = torch.load(checkpoint, map_location=dev)
        state_dict = ckpt["model_state_dict"]
        action_space_size = state_dict["prediction.policy_head.net.6.weight"].shape[0]
        model = MuZeroNet(build_default_skyjo_muzero_config(action_space_size=action_space_size)).to(dev)
        model.load_state_dict(state_dict)
        model.eval()
        self._baseline_cache[cache_key] = model
        return model

    def _load_belief(self, checkpoint: Path, device: str) -> BeliefAwareMuZeroNet:
        cache_key = (str(checkpoint), device)
        cached = self._belief_cache.get(cache_key)
        if cached is not None:
            return cached
        dev = torch.device(device)
        ckpt = torch.load(checkpoint, map_location=dev)
        state_dict = ckpt["model_state_dict"]
        action_space_size = state_dict["prediction.policy_head.net.6.weight"].shape[0]
        model = BeliefAwareMuZeroNet(build_default_belief_muzero_config(action_space_size=action_space_size)).to(dev)
        model.load_state_dict(state_dict)
        model.eval()
        self._belief_cache[cache_key] = model
        return model

    def _select_macro_action_with_decision_model(
        self,
        env: SkyjoEnv,
        agent: AgentConfig,
        model: MuZeroNet | BeliefAwareMuZeroNet,
        is_belief: bool,
        cfg: MCTSConfig,
    ) -> tuple[int, dict[int, int], dict[int, float], dict[int, float], float]:
        decision_env = SkyjoDecisionEnv(
            num_players=env.num_players,
            seed=env.initial_seed,
            history_window_k=env.history_window_k,
            score_limit=env.score_limit,
            setup_mode=env.setup_mode,
        )
        decision_env.base = copy.deepcopy(env)
        decision_env.pending = PendingTurnState()
        decision_env._sync_decision_phase()

        final_stats_visits: dict[int, int] = {}
        final_stats_q: dict[int, float] = {}
        final_stats_policy: dict[int, float] = {}
        final_root_value = 0.0

        for _ in range(4):
            phase = decision_env.decision_phase
            obs = decision_env.observe(decision_env.current_player)
            legal = decision_env.legal_actions()
            if is_belief:
                stats = run_belief_mcts(
                    model=model,  # type: ignore[arg-type]
                    observation=obs,
                    legal_action_ids=legal,
                    ego_player_id=int(obs["current_player"]),
                    config=cfg,
                    ablate_belief_head=agent.ablate_belief_head,
                    device=agent.device,
                )
            else:
                stats = run_mcts(
                    model=model,  # type: ignore[arg-type]
                    observation=obs,
                    legal_action_ids=legal,
                    config=cfg,
                    device=agent.device,
                )

            decision_action = int(stats.action)
            final_stats_visits = {int(k): int(v) for k, v in stats.visit_counts.items()}
            final_stats_q = {int(k): float(v) for k, v in stats.q_values.items()}
            final_stats_policy = {int(k): float(v) for k, v in stats.policy_target.items()}
            final_root_value = float(stats.root_value)

            if phase == DecisionPhase.SETUP_REVEAL:
                pos = decision_action - int(DecisionAction.CHOOSE_POS_BASE)
                return (
                    macro_to_action("SETUP_FLIP", pos),
                    final_stats_visits,
                    final_stats_q,
                    final_stats_policy,
                    final_root_value,
                )

            if phase == DecisionPhase.CHOOSE_POSITION:
                pos = decision_action - int(DecisionAction.CHOOSE_POS_BASE)
                if decision_env.pending.source == "DISCARD":
                    macro = "TAKE_DISCARD_AND_REPLACE"
                elif decision_env.pending.keep_drawn:
                    macro = "DRAW_DECK_KEEP_AND_REPLACE"
                else:
                    macro = "DRAW_DECK_DISCARD_AND_FLIP"
                return (
                    macro_to_action(macro, pos),
                    final_stats_visits,
                    final_stats_q,
                    final_stats_policy,
                    final_root_value,
                )

            decision_env.step(decision_action)

        raise RuntimeError("Failed to derive macro action from decision-model rollout.")

    def select_action(
        self,
        agent: AgentConfig,
        observation: dict[str, Any],
        legal_actions: list[int],
        env: SkyjoEnv | None = None,
    ) -> AIActionResult:
        if not legal_actions:
            raise ValueError("No legal actions available for AI turn.")
        actor = int(observation["current_player"])
        if agent.type == AgentType.HUMAN:
            raise ValueError("Cannot select AI action for human agent.")
        if agent.type == AgentType.HEURISTIC:
            bot = make_heuristic_bot(
                name=agent.heuristic_bot_name,
                seed=self._heuristic_seed(observation),
                epsilon=agent.heuristic_bot_epsilon,
            )
            action_id = int(bot.select_action(observation, legal_actions))
            macro_action, position = action_to_macro(action_id)
            return AIActionResult(
                actor=actor,
                action_id=action_id,
                macro_action=macro_action,
                position=position,
                root_value=None,
                visit_counts=None,
                q_values=None,
                policy_target=None,
            )

        cfg = MCTSConfig(
            num_simulations=agent.simulations,
            temperature=1e-8,
            add_exploration_noise=False,
            root_exploration_fraction=0.0,
        )

        checkpoint = self._resolve_checkpoint(agent.type, agent.checkpoint_path)
        if agent.type == AgentType.BASELINE:
            model = self._load_baseline(checkpoint, device=agent.device)
            if model.config.action_space_size == 16 and max(legal_actions) >= 16 and env is not None:
                action_id, visits, q_values, policy_target, root_value = self._select_macro_action_with_decision_model(
                    env=env,
                    agent=agent,
                    model=model,
                    is_belief=False,
                    cfg=cfg,
                )
                macro_action, position = action_to_macro(action_id)
                return AIActionResult(
                    actor=actor,
                    action_id=action_id,
                    macro_action=macro_action,
                    position=position,
                    root_value=root_value,
                    visit_counts=visits,
                    q_values=q_values,
                    policy_target=policy_target,
                )
            with torch.no_grad():
                stats = run_mcts(
                    model=model,
                    observation=observation,
                    legal_action_ids=legal_actions,
                    config=cfg,
                    device=agent.device,
                )
        else:
            model = self._load_belief(checkpoint, device=agent.device)
            if model.config.action_space_size == 16 and max(legal_actions) >= 16 and env is not None:
                action_id, visits, q_values, policy_target, root_value = self._select_macro_action_with_decision_model(
                    env=env,
                    agent=agent,
                    model=model,
                    is_belief=True,
                    cfg=cfg,
                )
                macro_action, position = action_to_macro(action_id)
                return AIActionResult(
                    actor=actor,
                    action_id=action_id,
                    macro_action=macro_action,
                    position=position,
                    root_value=root_value,
                    visit_counts=visits,
                    q_values=q_values,
                    policy_target=policy_target,
                )
            with torch.no_grad():
                stats = run_belief_mcts(
                    model=model,
                    observation=observation,
                    legal_action_ids=legal_actions,
                    ego_player_id=actor,
                    config=cfg,
                    ablate_belief_head=agent.ablate_belief_head,
                    device=agent.device,
                )
        action_id = int(stats.action)
        macro_action, position = action_to_macro(action_id)
        return AIActionResult(
            actor=actor,
            action_id=action_id,
            macro_action=macro_action,
            position=position,
            root_value=float(stats.root_value),
            visit_counts={int(k): int(v) for k, v in stats.visit_counts.items()},
            q_values={int(k): float(v) for k, v in stats.q_values.items()},
            policy_target={int(k): float(v) for k, v in stats.policy_target.items()},
        )

    def select_decision_action(
        self,
        agent: AgentConfig,
        observation: dict[str, Any],
        legal_actions: list[int],
    ) -> AIActionResult:
        if not legal_actions:
            raise ValueError("No legal decision actions available.")
        actor = int(observation["current_player"])
        if agent.type == AgentType.HUMAN:
            raise ValueError("Cannot select AI action for human agent.")
        if agent.type == AgentType.HEURISTIC:
            bot = make_heuristic_bot(
                name=agent.heuristic_bot_name,
                seed=self._heuristic_seed(observation),
                epsilon=agent.heuristic_bot_epsilon,
            )
            action_id = int(bot.select_action(observation, legal_actions))
            return AIActionResult(
                actor=actor,
                action_id=action_id,
                macro_action="DECISION_STEP",
                position=-1,
                root_value=None,
                visit_counts=None,
                q_values=None,
                policy_target=None,
            )

        cfg = MCTSConfig(
            num_simulations=agent.simulations,
            temperature=1e-8,
            add_exploration_noise=False,
            root_exploration_fraction=0.0,
        )
        checkpoint = self._resolve_checkpoint(agent.type, agent.checkpoint_path)
        if agent.type == AgentType.BASELINE:
            model = self._load_baseline(checkpoint, device=agent.device)
            with torch.no_grad():
                stats = run_mcts(
                    model=model,
                    observation=observation,
                    legal_action_ids=legal_actions,
                    config=cfg,
                    device=agent.device,
                )
        else:
            model = self._load_belief(checkpoint, device=agent.device)
            with torch.no_grad():
                stats = run_belief_mcts(
                    model=model,
                    observation=observation,
                    legal_action_ids=legal_actions,
                    ego_player_id=actor,
                    config=cfg,
                    ablate_belief_head=agent.ablate_belief_head,
                    device=agent.device,
                )

        action_id = int(stats.action)
        return AIActionResult(
            actor=actor,
            action_id=action_id,
            macro_action="DECISION_STEP",
            position=-1,
            root_value=float(stats.root_value),
            visit_counts={int(k): int(v) for k, v in stats.visit_counts.items()},
            q_values={int(k): float(v) for k, v in stats.q_values.items()},
            policy_target={int(k): float(v) for k, v in stats.policy_target.items()},
        )
