from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "ネオ・ブレイン"
TARGETS_REQUIRED = False


def spell_effect(
    game: "Game", caster: "Player", opponent: "Player", targets: Sequence["Card"]
) -> None:
    game._draw_cards_from_effect(caster, 2, "ネオ・ブレイン")
    game._update_board_window()


def prompt_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    return ()


def auto_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    return ()
