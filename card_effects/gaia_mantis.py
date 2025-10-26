from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "大昆虫ガイアマンティス"


def filter_blockers(
    game: "Game", defender: "Player", attacker: "Card", blockers: Sequence["Card"]
) -> Sequence["Card"]:
    filtered = []
    for creature in blockers:
        power = game._calculate_card_power(defender, creature)
        if power > 8000:
            filtered.append(creature)
    return filtered
