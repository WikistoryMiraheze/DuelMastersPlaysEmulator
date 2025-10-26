from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "炎獣兵マグナム・ブルース"


def attack_bonus(game: "Game", owner: "Player", card: "Card") -> int:
    if card.name != CARD_NAME:
        return 0
    has_other_fire = any(
        other is not card
        and other.is_creature()
        and other.has_civilization("火")
        for other in owner.battle_zone
    )
    return 3000 if has_other_fire else 0


def static_bonus(game: "Game", owner: "Player", card: "Card") -> int:
    return 0
