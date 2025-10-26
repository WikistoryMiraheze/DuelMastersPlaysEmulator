from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "磁力の使徒マグリス"


def enter_battle_zone(game: "Game", player: "Player", card: "Card") -> None:
    game._draw_cards_from_effect(player, 1, card)
