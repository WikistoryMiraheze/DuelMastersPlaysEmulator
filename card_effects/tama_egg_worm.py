from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "卵胞虫ゼリー・ワーム"


def on_attack(game: "Game", attacker_owner: "Player", defender: "Player", card: "Card") -> None:
    if not defender.hand:
        return
    discarded = random.choice(defender.hand)
    defender.hand.remove(discarded)
    defender.graveyard.append(discarded)
    game.log_creature_effect_detail(attacker_owner, f"《{card.name}》の効果で相手の《{discarded.name}》を捨てさせた")
    game._update_board_window()
