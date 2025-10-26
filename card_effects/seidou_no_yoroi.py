from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "青銅の鎧"


def enter_battle_zone(game: "Game", player: "Player", card: "Card") -> None:
    if not player.deck:
        game.output("青銅の鎧の能力は山札がないため何も起こりませんでした。")
        game.log_spell_effect_detail(player, "山札がないため効果なし")
        return

    top_card = player.deck.pop(0)
    from emulator import Zone

    player._add_card_to_zone(top_card, Zone.MANA)
    game.output(f"青銅の鎧の能力でマナゾーンに置かれました：{top_card.name}")
    game.log_spell_effect_detail(player, f"《{top_card.name}》マナゾーンへ")
    if not player.deck and not game.game_over:
        game._handle_deck_out(player)
