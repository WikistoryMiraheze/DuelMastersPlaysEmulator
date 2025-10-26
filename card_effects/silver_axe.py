from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "銀の戦斧"


def enter_battle_zone(game: "Game", player: "Player", card: "Card") -> None:
    # This creature has no entry effect; the function exists for registry completeness.
    return


def on_attack(game: "Game", attacker_owner: "Player", defender: "Player", card: "Card") -> None:
    game._move_top_deck_to_mana(attacker_owner, card)
    game._update_board_window()
