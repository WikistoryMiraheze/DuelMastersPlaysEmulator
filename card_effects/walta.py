from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "ウォルタ"


def on_attack(game: "Game", attacker_owner: "Player", defender: "Player", card: "Card") -> None:
    game._draw_cards_from_effect(attacker_owner, 1, card)
