"""CPU logic helpers for the Duel Masters Play's emulator."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from emulator import Card, Game, Player

AttackTarget = Tuple[str, Optional["Card"]]


class CpuAgent:
    """Interface for CPU decision making."""

    def choose_mana_charge(
        self,
        game: "Game",
        player: "Player",
        options: Sequence["Card"],
    ) -> Optional["Card"]:
        raise NotImplementedError

    def choose_card_to_play(
        self,
        game: "Game",
        player: "Player",
        playable_cards: Sequence["Card"],
    ) -> Optional[Tuple["Card", Optional[Sequence["Card"]]]]:
        raise NotImplementedError

    def choose_attack(
        self,
        game: "Game",
        attacker_owner: "Player",
        defender: "Player",
        attackers: Sequence["Card"],
    ) -> Optional[Tuple["Card", AttackTarget]]:
        raise NotImplementedError

    def choose_blocker(
        self,
        game: "Game",
        defender: "Player",
        blockers: Sequence["Card"],
    ) -> Optional["Card"]:
        raise NotImplementedError


class SimpleCpuAgent(CpuAgent):
    """A naive CPU agent that performs the first available legal action."""

    def choose_mana_charge(
        self,
        game: "Game",
        player: "Player",
        options: Sequence["Card"],
    ) -> Optional["Card"]:
        return options[0] if options else None

    def choose_card_to_play(
        self,
        game: "Game",
        player: "Player",
        playable_cards: Sequence["Card"],
    ) -> Optional[Tuple["Card", Optional[Sequence["Card"]]]]:
        for card in playable_cards:
            targets = game.auto_spell_targets(card, player)
            if targets is None:
                continue
            return card, targets
        return None

    def choose_attack(
        self,
        game: "Game",
        attacker_owner: "Player",
        defender: "Player",
        attackers: Sequence["Card"],
    ) -> Optional[Tuple["Card", AttackTarget]]:
        for attacker in attackers:
            targets = game.available_attack_targets(attacker_owner, defender, attacker)
            if not targets:
                continue
            preferred = None
            for option in targets:
                if option[0] == "opponent":
                    preferred = option
                    break
            if preferred is None:
                preferred = targets[0]
            return attacker, preferred
        return None

    def choose_blocker(
        self,
        game: "Game",
        defender: "Player",
        blockers: Sequence["Card"],
    ) -> Optional["Card"]:
        return blockers[0] if blockers else None


__all__ = ["CpuAgent", "SimpleCpuAgent"]
