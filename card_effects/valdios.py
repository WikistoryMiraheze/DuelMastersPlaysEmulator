from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "機神装甲ヴァルディオス"


def static_bonus(game: "Game", owner: "Player", card: "Card") -> int:
    if card.name == CARD_NAME:
        return 0
    if not card.has_race("ヒューマノイド"):
        return 0
    bonus = 0
    for creature in owner.battle_zone:
        if creature.name == CARD_NAME and creature is not card:
            bonus += 1000
    return bonus
