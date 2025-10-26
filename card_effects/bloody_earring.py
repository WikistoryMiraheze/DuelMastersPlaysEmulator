from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "ブラッディ・イヤリング"


def after_battle(game: "Game", owner: "Player", card: "Card") -> None:
    if card in owner.battle_zone:
        game._destroy_creature(owner, card)
