"""Model-guided CPU agent powered by a saved logistic regression model."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

import numpy as np

from cpu_ai import SimpleCpuAgent
from emulator import Card, Game, Player, has_w_breaker


PHASE_NAMES: Tuple[str, ...] = (
    "pre_mana",
    "post_mana",
    "pre_main",
    "post_main",
    "pre_attack",
    "post_attack",
    "turn_end",
)
PHASE_TO_INDEX = {name: index for index, name in enumerate(PHASE_NAMES)}


@dataclass(frozen=True)
class PlayerSummary:
    """Compact numerical description of a player's visible board state."""

    battle_creatures: int
    tapped_creatures: int
    total_power: float
    max_power: float
    mana_cards: int
    mana_pips: float
    mana_civilizations: FrozenSet[str]
    shields: int
    graveyard: int
    hand_cards: int
    hand_creatures: int
    hand_spells: int
    hand_cost_total: float
    hand_cost_max: float
    deck_cards: int
    available_mana: float
    max_available_mana: float
    max_mana: float
    hand_costs: Tuple[float, ...] = field(default_factory=tuple, repr=False)

    def untapped_creatures(self) -> int:
        return max(0, self.battle_creatures - self.tapped_creatures)

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                float(self.battle_creatures),
                float(self.tapped_creatures),
                float(self.untapped_creatures()),
                float(self.total_power),
                float(self.max_power),
                float(self.mana_cards),
                float(self.mana_pips),
                float(len(self.mana_civilizations)),
                float(self.shields),
                float(self.graveyard),
                float(self.hand_cards),
                float(self.hand_creatures),
                float(self.hand_spells),
                float(self.hand_cost_total),
                float(self.hand_cost_max),
                float(self.deck_cards),
                float(self.available_mana),
                float(self.max_available_mana),
                float(self.max_mana),
            ],
            dtype=np.float32,
        )

    def without_hand_card(self, card: Card) -> "PlayerSummary":
        costs = list(self.hand_costs)
        cost_value = float(card.cost or 0)
        try:
            costs.remove(cost_value)
        except ValueError:
            pass

        hand_cards = max(0, self.hand_cards - 1)
        hand_creatures = max(0, self.hand_creatures - (1 if card.is_creature() else 0))
        hand_spells = max(0, self.hand_spells - (1 if card.is_spell() else 0))
        hand_cost_total = max(0.0, self.hand_cost_total - cost_value)
        hand_cost_max = max(costs) if costs else 0.0

        return dataclasses.replace(
            self,
            hand_cards=hand_cards,
            hand_creatures=hand_creatures,
            hand_spells=hand_spells,
            hand_cost_total=hand_cost_total,
            hand_cost_max=hand_cost_max,
            hand_costs=tuple(costs),
        )

    def with_mana_card(self, card: Card) -> "PlayerSummary":
        base = self.without_hand_card(card)
        civilizations = set(base.mana_civilizations)
        civilizations.update(card.civilizations)
        mana_number = float(card.mana_number or 0)
        available_mana = base.available_mana
        max_available_mana = base.max_available_mana
        if card.mana_number == 1:
            max_available_mana += 1.0
            if not card.is_multicolored():
                available_mana += 1.0
        return dataclasses.replace(
            base,
            mana_cards=base.mana_cards + 1,
            mana_pips=base.mana_pips + mana_number,
            mana_civilizations=frozenset(civilizations),
            max_mana=base.max_mana + 1.0,
            max_available_mana=max_available_mana,
            available_mana=available_mana,
        )

    def with_creature_added(self, card: Card, *, tapped: bool = False) -> "PlayerSummary":
        base = self.without_hand_card(card)
        power = float(card.power or 0)
        tapped_creatures = base.tapped_creatures + (1 if tapped else 0)
        return dataclasses.replace(
            base,
            battle_creatures=base.battle_creatures + 1,
            tapped_creatures=tapped_creatures,
            total_power=base.total_power + power,
            max_power=max(base.max_power, power),
            available_mana=max(0.0, base.available_mana - float(card.cost or 0)),
        )

    def with_creature_destroyed(self, power: float, *, was_tapped: bool = True) -> "PlayerSummary":
        battle_creatures = max(0, self.battle_creatures - 1)
        tapped_creatures = max(0, self.tapped_creatures - (1 if was_tapped else 0))
        total_power = max(0.0, self.total_power - power)
        if battle_creatures == 0:
            max_power = 0.0
        elif power >= self.max_power - 1e-6:
            # Fall back to an average power estimate when the strongest creature was removed.
            average_power = total_power / max(battle_creatures, 1)
            max_power = max(average_power, total_power)
        else:
            max_power = self.max_power
        return dataclasses.replace(
            self,
            battle_creatures=battle_creatures,
            tapped_creatures=tapped_creatures,
            total_power=total_power,
            max_power=max_power,
            graveyard=self.graveyard + 1,
        )

    def with_shields_removed(self, amount: int) -> "PlayerSummary":
        return dataclasses.replace(
            self,
            shields=max(0, self.shields - max(0, amount)),
        )

    def with_tapped_increment(self, amount: int = 1) -> "PlayerSummary":
        tapped = max(0, self.tapped_creatures + amount)
        return dataclasses.replace(self, tapped_creatures=min(self.battle_creatures, tapped))


def summarize_player(player: Player) -> PlayerSummary:
    battle_creatures = 0
    tapped_creatures = 0
    total_power = 0.0
    max_power = 0.0
    for creature in player.battle_zone:
        if not creature.is_creature():
            continue
        battle_creatures += 1
        if player.is_creature_tapped(creature):
            tapped_creatures += 1
        power = float(creature.power or 0)
        total_power += power
        if power > max_power:
            max_power = power

    mana_cards = len(player.mana_zone)
    mana_pips = sum(float(card.mana_number or 0) for card in player.mana_zone)
    civilizations = {
        civilization for card in player.mana_zone for civilization in card.civilizations
    }

    hand_cards = len(player.hand)
    hand_costs = [float(card.cost or 0) for card in player.hand]
    hand_creatures = sum(1 for card in player.hand if card.is_creature())
    hand_spells = sum(1 for card in player.hand if card.is_spell())
    hand_cost_total = float(sum(hand_costs))
    hand_cost_max = max(hand_costs, default=0.0)

    return PlayerSummary(
        battle_creatures=battle_creatures,
        tapped_creatures=tapped_creatures,
        total_power=total_power,
        max_power=max_power,
        mana_cards=mana_cards,
        mana_pips=mana_pips,
        mana_civilizations=frozenset(civilizations),
        shields=len(player.shields),
        graveyard=len(player.graveyard),
        hand_cards=hand_cards,
        hand_creatures=hand_creatures,
        hand_spells=hand_spells,
        hand_cost_total=hand_cost_total,
        hand_cost_max=hand_cost_max,
        deck_cards=len(player.deck),
        available_mana=float(player.available_mana),
        max_available_mana=float(player.max_available_mana),
        max_mana=float(player.max_mana),
        hand_costs=tuple(hand_costs),
    )


@dataclass
class LogisticEvaluator:
    """Wrapper around the learned logistic regression model."""

    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray

    @classmethod
    def from_npz(cls, path: str) -> "LogisticEvaluator":
        payload = np.load(path)
        weights = payload["weights"].astype(np.float32)
        bias = float(np.squeeze(payload["bias"]))
        feature_mean = payload["feature_mean"].astype(np.float32)
        feature_std = payload["feature_std"].astype(np.float32)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        return cls(weights=weights, bias=bias, feature_mean=feature_mean, feature_std=feature_std)

    def score(
        self,
        *,
        turn_number: int,
        phase: str,
        our_summary: PlayerSummary,
        opponent_summary: PlayerSummary,
        effect_depth: int,
        shield_depth: int,
    ) -> float:
        our_array = our_summary.to_array()
        opponent_array = opponent_summary.to_array()
        diff = our_array - opponent_array
        phase_vector = np.zeros(len(PHASE_NAMES), dtype=np.float32)
        phase_index = PHASE_TO_INDEX.get(phase, PHASE_TO_INDEX["turn_end"])
        phase_vector[phase_index] = 1.0
        features = np.concatenate(
            [
                np.array([float(max(1, turn_number))], dtype=np.float32),
                phase_vector,
                our_array,
                opponent_array,
                diff,
                np.array([float(effect_depth), float(shield_depth)], dtype=np.float32),
            ]
        )
        normalised = (features - self.feature_mean) / self.feature_std
        logits = normalised @ self.weights + self.bias
        clipped = np.clip(logits, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-clipped))
        return float(probability)


class ModelGuidedCpuAgent(SimpleCpuAgent):
    """CPU agent that uses a logistic model score to guide greedy decisions."""

    def __init__(
        self,
        evaluator: LogisticEvaluator,
        *,
        controlled_players: Optional[Iterable[str]] = None,
        min_delta: float = 1e-3,
    ) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.controlled_players = set(controlled_players) if controlled_players else None
        self.min_delta = min_delta

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------
    def _is_controlled(self, player: Player) -> bool:
        if self.controlled_players is None:
            return True
        return player.name in self.controlled_players

    def _opponent(self, game: Game, player: Player) -> Player:
        for candidate in game.players:
            if candidate is not player:
                return candidate
        return game.players[0]

    def _current_summaries(
        self, game: Game, player: Player
    ) -> Tuple[PlayerSummary, PlayerSummary]:
        opponent = self._opponent(game, player)
        return summarize_player(player), summarize_player(opponent)

    def _score_from_summaries(
        self,
        game: Game,
        our_summary: PlayerSummary,
        opponent_summary: PlayerSummary,
        *,
        phase: str,
    ) -> float:
        effect_depth = len(game.get_effect_stack())
        shield_depth = len(game.get_shield_trigger_stack())
        return self.evaluator.score(
            turn_number=game.turn_number,
            phase=phase,
            our_summary=our_summary,
            opponent_summary=opponent_summary,
            effect_depth=effect_depth,
            shield_depth=shield_depth,
        )

    # ------------------------------------------------------------------
    # Mana charging
    # ------------------------------------------------------------------
    def choose_mana_charge(
        self,
        game: Game,
        player: Player,
        options: Sequence[Card],
    ) -> Optional[Card]:
        if not self._is_controlled(player) or not options:
            return super().choose_mana_charge(game, player, options)

        our_summary, opponent_summary = self._current_summaries(game, player)
        base_score = self._score_from_summaries(
            game,
            our_summary,
            opponent_summary,
            phase="pre_mana",
        )
        best_card: Optional[Card] = None
        best_score = base_score

        for card in options:
            candidate_summary = our_summary.with_mana_card(card)
            score = self._score_from_summaries(
                game,
                candidate_summary,
                opponent_summary,
                phase="post_mana",
            )
            if score > best_score + self.min_delta:
                best_score = score
                best_card = card

        return best_card

    # ------------------------------------------------------------------
    # Playing cards
    # ------------------------------------------------------------------
    def choose_card_to_play(
        self,
        game: Game,
        player: Player,
        playable_cards: Sequence[Card],
    ) -> Optional[Tuple[Card, Optional[Sequence[Card]]]]:
        if not self._is_controlled(player):
            return super().choose_card_to_play(game, player, playable_cards)

        our_summary, opponent_summary = self._current_summaries(game, player)
        base_score = self._score_from_summaries(
            game,
            our_summary,
            opponent_summary,
            phase="pre_main",
        )
        best_choice: Optional[Tuple[Card, Optional[Sequence[Card]]]] = None
        best_score = base_score

        for card in playable_cards:
            targets = game.auto_spell_targets(card, player)
            if targets is None:
                continue
            if card.is_creature():
                candidate_summary = our_summary.with_creature_added(card, tapped=False)
                score = self._score_from_summaries(
                    game,
                    candidate_summary,
                    opponent_summary,
                    phase="pre_main",
                )
                if score > best_score + self.min_delta:
                    best_score = score
                    best_choice = (card, targets)
            else:
                # Fallback to the baseline behaviour for spells and other effects.
                continue

        if best_choice is not None:
            return best_choice

        return super().choose_card_to_play(game, player, playable_cards)

    # ------------------------------------------------------------------
    # Attacking decisions
    # ------------------------------------------------------------------
    def choose_attack(
        self,
        game: Game,
        attacker_owner: Player,
        defender: Player,
        attackers: Sequence[Card],
    ) -> Optional[Tuple[Card, Tuple[str, Optional[Card]]]]:
        if not self._is_controlled(attacker_owner):
            return super().choose_attack(game, attacker_owner, defender, attackers)

        if not attackers:
            return None

        our_summary, opponent_summary = self._current_summaries(game, attacker_owner)
        base_score = self._score_from_summaries(
            game,
            our_summary,
            opponent_summary,
            phase="pre_attack",
        )
        best_score = base_score
        best_action: Optional[Tuple[Card, Tuple[str, Optional[Card]]]] = None

        for attacker in attackers:
            targets = game.available_attack_targets(attacker_owner, defender, attacker)
            if not targets:
                continue
            attacker_power = float(game._calculate_card_power(attacker_owner, attacker, is_attacking=True))
            for target in targets:
                score = self._evaluate_attack(
                    game,
                    our_summary,
                    opponent_summary,
                    attacker_owner,
                    defender,
                    attacker,
                    attacker_power,
                    target,
                )
                if score > best_score + self.min_delta:
                    best_score = score
                    best_action = (attacker, target)

        if best_action is not None:
            return best_action

        return None

    def _evaluate_attack(
        self,
        game: Game,
        our_summary: PlayerSummary,
        opponent_summary: PlayerSummary,
        attacker_owner: Player,
        defender: Player,
        attacker: Card,
        attacker_power: float,
        target: Tuple[str, Optional[Card]],
    ) -> float:
        post_our = our_summary.with_tapped_increment(1)
        post_opponent = opponent_summary
        target_type, target_creature = target

        if target_type == "opponent":
            breaks = 2 if has_w_breaker(attacker) else 1
            post_opponent = post_opponent.with_shields_removed(breaks)
        elif target_type == "creature" and target_creature is not None:
            defender_power = float(game._calculate_card_power(defender, target_creature))
            if attacker_power > defender_power:
                post_opponent = post_opponent.with_creature_destroyed(defender_power, was_tapped=True)
            elif attacker_power < defender_power:
                post_our = post_our.with_creature_destroyed(attacker_power, was_tapped=True)
            else:
                post_opponent = post_opponent.with_creature_destroyed(defender_power, was_tapped=True)
                post_our = post_our.with_creature_destroyed(attacker_power, was_tapped=True)
        else:
            return float("-inf")

        return self._score_from_summaries(
            game,
            post_our,
            post_opponent,
            phase="post_attack",
        )


__all__ = [
    "PHASE_NAMES",
    "PHASE_TO_INDEX",
    "LogisticEvaluator",
    "ModelGuidedCpuAgent",
    "PlayerSummary",
    "summarize_player",
]
