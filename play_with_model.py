#!/usr/bin/env python3
"""Run games that leverage the logistic model guided CPU agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from emulator import (
    Card,
    Game,
    Player,
    build_enemy_deck,
    build_my_deck,
    create_card_by_name,
    run_cli_game,
)

from model_cpu_agent import LogisticEvaluator, ModelGuidedCpuAgent


def _load_deck_names(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ".jsn"}:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("cards", "deck", "list"):
                if key in payload:
                    payload = payload[key]
                    break
        if isinstance(payload, (str, bytes)):
            raise ValueError(f"{path}: JSON deck must be a list, not a string")
        if not isinstance(payload, Iterable):
            raise ValueError(f"{path}: JSON deck must be a list or contain a 'cards' list")
        names = [str(item).strip() for item in payload if str(item).strip()]
    else:
        names = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            names.append(stripped)
    if not names:
        raise ValueError(f"{path}: デッキに有効なカード名が見つかりませんでした")
    return names


def _make_deck_factory(
    path: Optional[Path],
    fallback: Callable[[], List[Card]],
) -> Callable[[], List[Card]]:
    if path is None:
        return fallback
    names = tuple(_load_deck_names(path))

    def factory() -> List[Card]:
        return [create_card_by_name(name) for name in names]

    return factory


def _simulate_headless(
    *,
    evaluator: LogisticEvaluator,
    games: int,
    ai_factory: Callable[[], List[Card]],
    opponent_factory: Callable[[], List[Card]],
    min_delta: float,
    verbose: bool,
) -> None:
    agent = ModelGuidedCpuAgent(
        evaluator,
        controlled_players={"モデルAI"},
        min_delta=min_delta,
    )

    wins = losses = draws = 0
    for game_index in range(games):
        player_ai = Player(name="モデルAI", deck=ai_factory())
        player_cpu = Player(name="シンプルCPU", deck=opponent_factory())
        game = Game(
            player_ai,
            player_cpu,
            human_player_index=None,
            input_func=lambda prompt: (_raise_unexpected_input(prompt)),
            output_func=(print if verbose else (lambda *_: None)),
            board_window=None,
            cpu_agent=agent,
        )
        game.start_game()
        if not game.game_over:
            game.run_until_game_over()
        if game.winner is player_ai:
            wins += 1
        elif game.winner is player_cpu:
            losses += 1
        else:
            draws += 1
        if verbose:
            print(f"Game {game_index + 1}: winner = {game.winner.name if game.winner else 'draw'}")

    print("==== 結果 ====")
    print(f"勝利: {wins}")
    print(f"敗北: {losses}")
    print(f"引き分け: {draws}")


def _raise_unexpected_input(prompt: str) -> str:
    raise RuntimeError(f"Unexpected input request during headless execution: {prompt}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run games with the model-guided CPU agent")
    parser.add_argument("model", type=Path, help="Path to the logistic model .npz file")
    parser.add_argument(
        "--mode",
        choices=["headless", "interactive"],
        default="headless",
        help="Headless simulation against SimpleCpuAgent or interactive CLI vs human",
    )
    parser.add_argument("--games", type=int, default=10, help="Number of headless self-play games")
    parser.add_argument("--ai-deck", type=Path, default=None, help="Deck list for the model-controlled player")
    parser.add_argument("--opponent-deck", type=Path, default=None, help="Deck list for the Simple CPU opponent")
    parser.add_argument("--human-deck", type=Path, default=None, help="Deck list for the human player in interactive mode")
    parser.add_argument(
        "--starting-player",
        choices=["human_first", "ai_first", "random"],
        default="human_first",
        help="Who should start first in interactive mode",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-3,
        help="Minimum probability improvement required before switching from the baseline action",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-game results in headless mode")
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("--games must be at least 1")
    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    evaluator = LogisticEvaluator.from_npz(str(args.model))
    ai_factory = _make_deck_factory(args.ai_deck, build_my_deck)
    opponent_factory = _make_deck_factory(args.opponent_deck, build_enemy_deck)

    if args.mode == "headless":
        _simulate_headless(
            evaluator=evaluator,
            games=args.games,
            ai_factory=ai_factory,
            opponent_factory=opponent_factory,
            min_delta=args.min_delta,
            verbose=args.verbose,
        )
        return

    human_factory = _make_deck_factory(args.human_deck, build_my_deck)
    agent = ModelGuidedCpuAgent(
        evaluator,
        controlled_players={"モデルAI"},
        min_delta=args.min_delta,
    )
    player_human = Player(name="あなた", deck=human_factory())
    player_ai = Player(name="モデルAI", deck=ai_factory())
    run_cli_game(
        player_human,
        player_ai,
        cpu_agent=agent,
        starting_player=args.starting_player,
    )


if __name__ == "__main__":
    main()
