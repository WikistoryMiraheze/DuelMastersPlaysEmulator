from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "ボルシャック・ドラゴン"


def attack_bonus(game: "Game", owner: "Player", card: "Card") -> int:
    if card.name != CARD_NAME:
        return 0
    fire_cards = sum(1 for grave_card in owner.graveyard if "火" in grave_card.civilizations)
    return fire_cards * 1000
