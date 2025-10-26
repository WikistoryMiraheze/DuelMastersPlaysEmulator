#!/usr/bin/env python3
"""Self-play training pipeline for the Duel Masters Play's emulator.

For step-by-step CLI instructions see ``docs/train_self_play_usage.md``.

This script demonstrates how to combine the emulator's public APIs with a
simple learning loop:

1. Run CPU vs CPU self-play matches and record the board state that
   :class:`~emulator.Game` exposes via
   :meth:`~emulator.Game.get_turn_end_zone_state`.
2. Track stack usage through :meth:`~emulator.Game.get_effect_stack` and
   :meth:`~emulator.Game.get_shield_trigger_stack` to include meta-features.
3. Train a lightweight logistic-regression baseline on the collected samples.
4. (Optional) Sample decks from user-provided pools so the model experiences
   a wider variety of opponents.

The produced model is not meant to be competitive – it merely acts as a
starting point for experimentation (data collection, feature engineering or
plugging in a stronger model).  Use ``--help`` to see available options.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from card_effects import SPELL_TARGET_PROMPTS
from cpu_ai import SimpleCpuAgent
from model_cpu_agent import PHASE_NAMES, PHASE_TO_INDEX
from emulator import Card, Game, Player, build_enemy_deck, build_my_deck, create_card_by_name


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainingConfig:
    """User-provided knobs for the self-play training run."""

    games: int = 100
    learning_rate: float = 0.1
    epochs: int = 200
    validation_split: float = 0.2
    l2: float = 1e-4
    seed: int = 1234
    log_interval: int = 20
    save_model: Optional[Path] = None
    save_dataset: Optional[Path] = None
    metrics_output: Optional[Path] = None
    agent: str = "safe-simple"
    deck_a_specs: Tuple[Tuple[str, ...], ...] = ()
    deck_b_specs: Tuple[Tuple[str, ...], ...] = ()


def _coerce_card_names(raw: Any, source: Path) -> List[str]:
    if isinstance(raw, dict):
        for key in ("cards", "deck", "list"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raise ValueError(f"{source}: 辞書形式のデッキには 'cards' キーが必要です")

    if isinstance(raw, (str, bytes)):
        raise ValueError(
            f"{source}: JSON で単一の文字列は解釈できません。リストで指定してください"
        )

    if not isinstance(raw, Iterable):
        raise ValueError(f"{source}: デッキ形式を解釈できませんでした ({type(raw)!r})")

    names: List[str] = []
    for item in raw:
        name = str(item).strip()
        if not name:
            continue
        names.append(name)

    if not names:
        raise ValueError(f"{source}: カード名が 1 枚も見つかりませんでした")
    return names


def load_deck_specs(paths: Sequence[Path]) -> Tuple[Tuple[str, ...], ...]:
    specs: List[Tuple[str, ...]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"{path}: ファイルを読み込めませんでした: {exc}") from exc
        if path.suffix.lower() in {".json", ".jsn"}:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: JSON の読み込みに失敗しました: {exc}") from exc
            names = _coerce_card_names(payload, path)
        else:
            names = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                names.append(stripped)
            if not names:
                raise ValueError(f"{path}: 有効なカード名が見つかりませんでした")
        specs.append(tuple(names))
    return tuple(specs)


def make_deck_factory(
    specs: Tuple[Tuple[str, ...], ...],
    fallback: Callable[[], List[Card]],
) -> DeckFactory:
    if not specs:
        return fallback

    def factory() -> List[Card]:
        template = random.choice(specs)
        return [create_card_by_name(name) for name in template]

    return factory


# ---------------------------------------------------------------------------
# CPU agent helpers
# ---------------------------------------------------------------------------


class SafeSimpleCpuAgent(SimpleCpuAgent):
    """A :class:`SimpleCpuAgent` that avoids interactive spell prompts."""

    def choose_card_to_play(
        self,
        game: Game,
        player: Player,
        playable_cards: Sequence["Card"],
    ) -> Optional[Tuple["Card", Optional[Sequence["Card"]]]]:
        from emulator import Card  # Local import to avoid circular hints

        for card in playable_cards:
            if card.name in SPELL_TARGET_PROMPTS:
                # Skip spells that would prompt for manual target selection.
                continue
            targets = game.auto_spell_targets(card, player)
            if targets is None:
                continue
            return card, targets
        return None


def build_agent(name: str) -> SimpleCpuAgent:
    """Construct the CPU agent requested by ``--agent``."""

    lowered = name.lower().strip()
    if lowered in {"simple", "baseline"}:
        return SimpleCpuAgent()
    if lowered in {"safe", "safe-simple", "safe_simple"}:
        return SafeSimpleCpuAgent()
    raise ValueError(f"Unknown agent preset: {name}")


# ---------------------------------------------------------------------------
# Stack depth tracking utilities
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StackTracker:
    """Keep track of effect-stack statistics for each turn."""

    game_max_effect_stack_depth: int = 0
    game_max_shield_stack_depth: int = 0
    current_turn_max_effect_stack_depth: int = 0
    current_turn_max_shield_stack_depth: int = 0
    turn_effect_depths: List[int] = field(default_factory=list)
    turn_shield_depths: List[int] = field(default_factory=list)

    def reset_turn(self) -> None:
        self.current_turn_max_effect_stack_depth = 0
        self.current_turn_max_shield_stack_depth = 0

    def observe_depths(self, effect_depth: int, shield_depth: int) -> None:
        if effect_depth > self.game_max_effect_stack_depth:
            self.game_max_effect_stack_depth = effect_depth
        if shield_depth > self.game_max_shield_stack_depth:
            self.game_max_shield_stack_depth = shield_depth
        if effect_depth > self.current_turn_max_effect_stack_depth:
            self.current_turn_max_effect_stack_depth = effect_depth
        if shield_depth > self.current_turn_max_shield_stack_depth:
            self.current_turn_max_shield_stack_depth = shield_depth

    def finalize_turn(self) -> None:
        self.turn_effect_depths.append(self.current_turn_max_effect_stack_depth)
        self.turn_shield_depths.append(self.current_turn_max_shield_stack_depth)


def install_stack_tracker(game: Game) -> StackTracker:
    """Monkey-patch ``game`` so we can observe effect-stack depth changes."""

    tracker = StackTracker()

    original_push = game._push_effect_stack_entry  # type: ignore[attr-defined]
    original_pop = game._pop_effect_stack_entry  # type: ignore[attr-defined]

    def update_depths() -> None:
        effect_depth = len(game.get_effect_stack())
        shield_depth = len(game.get_shield_trigger_stack())
        tracker.observe_depths(effect_depth, shield_depth)

    def wrapped_push(self: Game, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        entry = original_push(*args, **kwargs)
        update_depths()
        return entry

    def wrapped_pop(self: Game, entry: Dict[str, Any]) -> None:
        original_pop(entry)
        update_depths()

    game._push_effect_stack_entry = types.MethodType(  # type: ignore[attr-defined]
        wrapped_push,
        game,
    )
    game._pop_effect_stack_entry = types.MethodType(  # type: ignore[attr-defined]
        wrapped_pop,
        game,
    )
    return tracker


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------


def _safe_power(card_snapshot: Dict[str, Any]) -> float:
    power = card_snapshot.get("power")
    if isinstance(power, (int, float)):
        return float(power)
    return 0.0


def _summarize_player(snapshot: Dict[str, Any]) -> np.ndarray:
    battle_zone: List[Dict[str, Any]] = snapshot.get("battle_zone", [])  # type: ignore[assignment]
    mana_zone: List[Dict[str, Any]] = snapshot.get("mana_zone", [])  # type: ignore[assignment]
    shield_zone: List[Dict[str, Any]] = snapshot.get("shields", [])  # type: ignore[assignment]
    graveyard: List[Dict[str, Any]] = snapshot.get("graveyard", [])  # type: ignore[assignment]
    hand_cards: List[Dict[str, Any]] = snapshot.get("hand", [])  # type: ignore[assignment]

    tapped = sum(1 for card in battle_zone if card.get("tapped"))
    total_power = sum(_safe_power(card) for card in battle_zone)
    max_power = max((_safe_power(card) for card in battle_zone), default=0.0)
    mana_pips = sum(card.get("mana_number", 0) or 0 for card in mana_zone)
    civilizations = {
        civ for card in mana_zone for civ in card.get("civilizations", [])
    }

    def _types(card: Dict[str, Any]) -> set:
        return set(card.get("types", []))

    creature_types = {"クリーチャー", "進化クリーチャー"}
    spell_types = {"呪文"}

    hand_costs = [float(card.get("cost") or 0.0) for card in hand_cards]
    hand_creatures = sum(1 for card in hand_cards if _types(card) & creature_types)
    hand_spells = sum(1 for card in hand_cards if _types(card) & spell_types)
    hand_cost_total = float(sum(hand_costs))
    hand_cost_max = max(hand_costs, default=0.0)

    return np.array(
        [
            float(len(battle_zone)),
            float(tapped),
            float(len(battle_zone) - tapped),
            float(total_power),
            float(max_power),
            float(len(mana_zone)),
            float(mana_pips),
            float(len(civilizations)),
            float(len(shield_zone)),
            float(len(graveyard)),
            float(len(hand_cards)),
            float(hand_creatures),
            float(hand_spells),
            float(hand_cost_total),
            float(hand_cost_max),
            float(snapshot.get("deck_size", 0)),
            float(snapshot.get("available_mana", 0)),
            float(snapshot.get("max_available_mana", 0)),
            float(snapshot.get("max_mana", len(mana_zone))),
        ],
        dtype=np.float32,
    )


def build_feature_vector(
    snapshot: Dict[str, Any],
    *,
    turn_number: int,
    phase: str,
    effect_depth: float,
    shield_depth: float,
    player1_name: str,
    player2_name: str,
) -> np.ndarray:
    player1_snapshot = snapshot[player1_name]
    player2_snapshot = snapshot[player2_name]

    p1_summary = _summarize_player(player1_snapshot)
    p2_summary = _summarize_player(player2_snapshot)
    diff = p1_summary - p2_summary

    phase_vector = np.zeros(len(PHASE_NAMES), dtype=np.float32)
    phase_index = PHASE_TO_INDEX.get(phase, PHASE_TO_INDEX["turn_end"])
    phase_vector[phase_index] = 1.0

    return np.concatenate(
        [
            np.array([float(turn_number)], dtype=np.float32),
            phase_vector,
            p1_summary,
            p2_summary,
            diff,
            np.array([float(effect_depth), float(shield_depth)], dtype=np.float32),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------------


def _dummy_output(_message: str) -> None:
    """Discard emulator output during automated training."""


DeckFactory = Callable[[], List[Card]]


def collect_game_samples(
    game_index: int,
    agent: SimpleCpuAgent,
    deck_factory_a: DeckFactory,
    deck_factory_b: DeckFactory,
) -> Tuple[List[np.ndarray], Optional[int], Dict[str, Any]]:
    """Play a single game and return per-turn feature vectors."""

    player1 = Player(name=f"学習プレイヤーA#{game_index}", deck=deck_factory_a())
    player2 = Player(name=f"学習プレイヤーB#{game_index}", deck=deck_factory_b())

    game = Game(
        player1,
        player2,
        human_player_index=None,
        input_func=lambda prompt: (_raise_unexpected_input(prompt)),
        output_func=_dummy_output,
        board_window=None,
        cpu_agent=agent,
    )

    tracker = install_stack_tracker(game)
    phase_counts = {phase: 0 for phase in PHASE_NAMES}

    game.start_game()
    if game.game_over:
        tracker.finalize_turn()
        winner_label = _label_for_winner(game, player1)
        return [], winner_label, _summarize_game(game_index, game, tracker, phase_counts)

    previous_turn = game.turn_number
    while not game.game_over:
        tracker.reset_turn()
        game.run_turn()
        if game.turn_number == previous_turn:
            # Turn did not advance (game probably ended mid-step).
            continue
        tracker.finalize_turn()
        previous_turn = game.turn_number

    winner_label = _label_for_winner(game, player1)
    samples: List[np.ndarray] = []
    for turn_number in sorted(game.turn_phase_snapshots):
        phase_entries = game.turn_phase_snapshots[turn_number]
        for phase in PHASE_NAMES:
            entry = phase_entries.get(phase)
            if not entry:
                continue
            snapshot = entry.get("zones")
            if snapshot is None:
                continue
            features = build_feature_vector(
                snapshot,
                turn_number=turn_number,
                phase=phase,
                effect_depth=float(entry.get("effect_depth", 0)),
                shield_depth=float(entry.get("shield_depth", 0)),
                player1_name=player1.name,
                player2_name=player2.name,
            )
            samples.append(features)
            phase_counts[phase] += 1

    return samples, winner_label, _summarize_game(game_index, game, tracker, phase_counts)


def _raise_unexpected_input(prompt: str) -> str:
    raise RuntimeError(
        "Unexpected input request during CPU self-play: {prompt}".format(prompt=prompt)
    )


def _label_for_winner(game: Game, player1: Player) -> Optional[int]:
    if game.winner is None:
        return None
    return 1 if game.winner is player1 else 0


def _summarize_game(
    game_index: int,
    game: Game,
    tracker: StackTracker,
    phase_counts: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "game_index": game_index,
        "turns": game.turn_number,
        "winner": None if game.winner is None else game.winner.name,
        "max_effect_stack": tracker.game_max_effect_stack_depth,
        "max_shield_stack": tracker.game_max_shield_stack_depth,
        "phase_counts": dict(phase_counts),
    }


# ---------------------------------------------------------------------------
# Model definition (logistic regression)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LogisticModel:
    weights: np.ndarray
    bias: float

    @classmethod
    def create(cls, feature_dim: int) -> "LogisticModel":
        return cls(weights=np.zeros(feature_dim, dtype=np.float32), bias=0.0)

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        return self._sigmoid(logits)

    def loss(self, features: np.ndarray, labels: np.ndarray, l2: float) -> float:
        preds = self.predict_proba(features)
        eps = 1e-7
        loss = -np.mean(
            labels * np.log(preds + eps)
            + (1.0 - labels) * np.log(1.0 - preds + eps)
        )
        if l2:
            loss += 0.5 * l2 * float(np.dot(self.weights, self.weights))
        return float(loss)

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        preds = self.predict_proba(features) >= 0.5
        return float(np.mean((preds.astype(int)) == labels.astype(int)))

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        epochs: int,
        l2: float,
        log_interval: int,
    ) -> None:
        if features.size == 0:
            return
        for epoch in range(1, epochs + 1):
            logits = features @ self.weights + self.bias
            preds = self._sigmoid(logits)
            errors = preds - labels
            grad_w = (features.T @ errors) / features.shape[0]
            if l2:
                grad_w += l2 * self.weights
            grad_b = float(np.mean(errors))

            self.weights -= learning_rate * grad_w.astype(np.float32)
            self.bias -= learning_rate * grad_b

            if log_interval and epoch % log_interval == 0:
                loss_value = self.loss(features, labels, l2)
                print(
                    f"Epoch {epoch:4d} | loss={loss_value:.4f}",
                    file=sys.stderr,
                )


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def collect_dataset(config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = random.Random(config.seed)
    agent = build_agent(config.agent)

    deck_factory_a = make_deck_factory(config.deck_a_specs, build_my_deck)
    deck_factory_b = make_deck_factory(config.deck_b_specs, build_enemy_deck)

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    per_game_metrics: List[Dict[str, Any]] = []
    results = {"wins": 0, "losses": 0, "draws": 0}
    aggregate_phase_counts = {phase: 0 for phase in PHASE_NAMES}

    for game_index in range(config.games):
        random_seed = rng.randrange(1 << 30)
        random.seed(random_seed)

        samples, label, metrics = collect_game_samples(
            game_index,
            agent,
            deck_factory_a,
            deck_factory_b,
        )
        per_game_metrics.append(metrics)
        counts = metrics.get("phase_counts", {})
        for phase, value in counts.items():
            if phase in aggregate_phase_counts:
                aggregate_phase_counts[phase] += int(value)

        if label is None:
            results["draws"] += 1
            continue
        if label == 1:
            results["wins"] += 1
        else:
            results["losses"] += 1

        if samples:
            all_features.extend(samples)
            all_labels.extend([label] * len(samples))

    if not all_features:
        raise RuntimeError("No training samples collected – try increasing --games")

    feature_matrix = np.stack(all_features).astype(np.float32)
    label_vector = np.array(all_labels, dtype=np.float32)

    summary = {
        "config": {
            "games": config.games,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "validation_split": config.validation_split,
            "l2": config.l2,
            "seed": config.seed,
            "agent": config.agent,
            "phases": list(PHASE_NAMES),
        },
        "samples": int(feature_matrix.shape[0]),
        "feature_dim": int(feature_matrix.shape[1]),
        "results": results,
        "phase_counts": aggregate_phase_counts,
        "per_game": per_game_metrics,
    }

    return feature_matrix, label_vector, summary


def normalise_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    normalised = (features - mean) / std
    return normalised.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    validation_split: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if not 0.0 < validation_split < 1.0:
        return features, labels, None, None

    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    split_index = int(features.shape[0] * (1.0 - validation_split))
    if split_index <= 0 or split_index >= features.shape[0]:
        return features, labels, None, None

    train_features = features[:split_index]
    train_labels = labels[:split_index]
    val_features = features[split_index:]
    val_labels = labels[split_index:]
    return train_features, train_labels, val_features, val_labels


def train(config: TrainingConfig) -> None:
    features, labels, summary = collect_dataset(config)
    features, mean, std = normalise_features(features)

    train_features, train_labels, val_features, val_labels = split_dataset(
        features, labels, config.validation_split, config.seed
    )

    model = LogisticModel.create(train_features.shape[1])
    model.fit(
        train_features,
        train_labels,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        l2=config.l2,
        log_interval=config.log_interval,
    )

    train_loss = model.loss(train_features, train_labels, config.l2)
    train_acc = model.accuracy(train_features, train_labels)
    print(f"Training samples: {train_features.shape[0]} | loss={train_loss:.4f} | accuracy={train_acc:.3f}")

    if val_features is not None and val_labels is not None:
        val_loss = model.loss(val_features, val_labels, config.l2)
        val_acc = model.accuracy(val_features, val_labels)
        print(f"Validation samples: {val_features.shape[0]} | loss={val_loss:.4f} | accuracy={val_acc:.3f}")

    if config.save_model is not None:
        payload = {
            "weights": model.weights,
            "bias": np.array([model.bias], dtype=np.float32),
            "feature_mean": mean,
            "feature_std": std,
        }
        np.savez(config.save_model, **payload)
        print(f"Saved model parameters to {config.save_model}")

    if config.save_dataset is not None:
        np.savez(
            config.save_dataset,
            features=features,
            labels=labels,
            feature_mean=mean,
            feature_std=std,
        )
        print(f"Saved dataset to {config.save_dataset}")

    if config.metrics_output is not None:
        config.metrics_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Wrote metrics to {config.metrics_output}")


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train a baseline evaluator from emulator self-play data",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of self-play games to simulate")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Gradient-descent learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs of gradient descent to run")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation (0-1)",
    )
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularisation strength")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for simulation and shuffling")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="How often (in epochs) to report loss to stderr",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path to save the trained logistic model (NumPy .npz)",
    )
    parser.add_argument(
        "--save-dataset",
        type=Path,
        default=None,
        help="Optional path to persist the raw feature/label arrays (NumPy .npz)",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional path to dump a JSON summary of the simulation runs",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="safe-simple",
        help="CPU agent preset to use (simple, safe-simple)",
    )
    parser.add_argument(
        "--deck",
        dest="deck_shared",
        type=Path,
        action="append",
        default=[],
        help=(
            "Deck list file applied to both players. Accepts JSON lists or simple "
            "newline-separated text files. Option can be repeated."
        ),
    )
    parser.add_argument(
        "--deck-a",
        dest="deck_a",
        type=Path,
        action="append",
        default=[],
        help="Deck list file for player A (learning side). Option can be repeated.",
    )
    parser.add_argument(
        "--deck-b",
        dest="deck_b",
        type=Path,
        action="append",
        default=[],
        help="Deck list file for player B (opponent side). Option can be repeated.",
    )

    args = parser.parse_args(argv)
    try:
        shared_specs = load_deck_specs(args.deck_shared)
        deck_a_specs = shared_specs + load_deck_specs(args.deck_a)
        deck_b_specs = shared_specs + load_deck_specs(args.deck_b)
    except ValueError as exc:
        parser.error(str(exc))

    return TrainingConfig(
        games=args.games,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        validation_split=args.validation_split,
        l2=args.l2,
        seed=args.seed,
        log_interval=args.log_interval,
        save_model=args.save_model,
        save_dataset=args.save_dataset,
        metrics_output=args.metrics_output,
        agent=args.agent,
        deck_a_specs=deck_a_specs,
        deck_b_specs=deck_b_specs,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    train(config)


if __name__ == "__main__":
    main()
