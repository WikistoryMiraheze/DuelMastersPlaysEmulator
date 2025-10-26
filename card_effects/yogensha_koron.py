from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "予言者コロン"


def enter_battle_zone(game: "Game", player: "Player", card: "Card") -> None:
    opponent = game.non_turn_player if player is game.turn_player else game.turn_player
    if not opponent.battle_zone:
        return

    agent = game._cpu_agent_for(player)
    if agent is None and game._player_index(player) == game.human_player_index:
        game.output("予言者コロンの能力：相手のクリーチャー1体をタップしてもよい。")
        for idx, creature in enumerate(opponent.battle_zone):
            state = "タップ" if opponent.is_creature_tapped(creature) else "アンタップ"
            game.output(f"  {idx}: {creature.name}（{state}）")
        from emulator import SINGLE_DIGIT_CHOICES

        choice = game.input("タップするクリーチャーの番号、または他のキーでスキップ：").strip()
        if choice in SINGLE_DIGIT_CHOICES:
            index = int(choice)
            if index < len(opponent.battle_zone):
                target = opponent.battle_zone[index]
                opponent.tap_creature(target)
                game.log_creature_effect_detail(player, f"《{target.name}》をタップ")
                game._update_board_window()
        return

    target: Optional["Card"] = None
    for creature in opponent.battle_zone:
        if not opponent.is_creature_tapped(creature):
            target = creature
            break
    if target is None:
        return
    opponent.tap_creature(target)
    game.log_creature_effect_detail(player, f"《{target.name}》をタップ")
    game._update_board_window()
