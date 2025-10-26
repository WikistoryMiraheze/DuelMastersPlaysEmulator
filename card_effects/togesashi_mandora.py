from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "トゲ刺しマンドラ"


def enter_battle_zone(game: "Game", player: "Player", card: "Card") -> None:
    candidates = [grave_card for grave_card in player.graveyard if grave_card.is_creature()]
    if not candidates:
        return

    unique_names: List[str] = []
    for grave_card in candidates:
        if grave_card.name not in unique_names:
            unique_names.append(grave_card.name)
        if len(unique_names) == 3:
            break

    agent = game._cpu_agent_for(player)
    selected_name: Optional[str] = None
    if agent is None and game._player_index(player) == game.human_player_index:
        game.output("トゲ刺しマンドラの能力：墓地から探索するカードを選んでください。")
        for idx, name in enumerate(unique_names):
            game.output(f"  {idx}: {name}")
        from emulator import SINGLE_DIGIT_CHOICES

        choice = game.input("番号を入力、または他のキーで選ばない：").strip()
        if choice in SINGLE_DIGIT_CHOICES:
            index = int(choice)
            if index < len(unique_names):
                selected_name = unique_names[index]
    else:
        selected_name = unique_names[0]

    if not selected_name:
        return

    from emulator import Zone

    for grave_card in list(player.graveyard):
        if grave_card.name == selected_name:
            player.graveyard.remove(grave_card)
            player._add_card_to_zone(grave_card, Zone.MANA)
            game.log_creature_effect_detail(player, f"《{grave_card.name}》をマナゾーンへ")
            game._update_board_window()
            break
