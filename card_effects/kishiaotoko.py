from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "汽車男"


def enter_battle_zone(game: "Game", player: "Player", card: "Card") -> None:
    opponent = game.non_turn_player if player is game.turn_player else game.turn_player
    if not opponent.hand:
        return
    discarded = random.choice(opponent.hand)
    opponent.hand.remove(discarded)
    opponent.graveyard.append(discarded)
    game.log_creature_effect_detail(player, f"相手の《{discarded.name}》を捨てさせた")
    game._update_board_window()
