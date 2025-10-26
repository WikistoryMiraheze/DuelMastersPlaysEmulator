from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "フェアリー・ライフ"
TARGETS_REQUIRED = False


def spell_effect(
    game: "Game", caster: "Player", opponent: "Player", targets: Sequence["Card"]
) -> None:
    game._move_top_deck_to_mana(caster, "フェアリー・ライフ")
    game._update_board_window()


def prompt_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    return ()


def auto_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    return ()
