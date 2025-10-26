from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "浄化の精霊ウルス"


def on_end_step(game: "Game", player: "Player", card: "Card") -> None:
    if player.is_creature_tapped(card):
        player.untap_creature(card)
        game.log_creature_effect_detail(player, "《浄化の精霊ウルス》アンタップ")
        game._update_board_window()
