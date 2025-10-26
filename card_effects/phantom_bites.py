from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "ファントム・バイツ"
TARGETS_REQUIRED = True


def spell_effect(
    game: "Game", caster: "Player", opponent: "Player", targets: Sequence["Card"]
) -> None:
    if not targets:
        return
    creature = targets[0]
    if creature not in opponent.battle_zone or not creature.is_creature():
        from emulator import GameError

        raise GameError("対象は相手のクリーチャーでなければなりません。")
    game.turn_power_modifiers[creature] = game.turn_power_modifiers.get(creature, 0) - 2000
    game.log_spell_effect_detail(opponent, f"《{creature.name}》パワー-2000")
    game._update_board_window()


def prompt_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    opponent = game.non_turn_player if caster is game.turn_player else game.turn_player
    if not opponent.battle_zone:
        game.output("相手のバトルゾーンにクリーチャーがいません。")
        return ()

    game.output("パワーを下げる対象を選んでください：")
    for idx, creature in enumerate(opponent.battle_zone):
        game.output(f"  {idx}: {creature.name}")

    from emulator import SINGLE_DIGIT_CHOICES

    choice = game.input("対象の番号（0-9）を入力、他のキーでキャンセル：").strip()
    if choice not in SINGLE_DIGIT_CHOICES:
        return ()
    index = int(choice)
    if index >= len(opponent.battle_zone):
        game.output("不正な選択です。対象は選ばれませんでした。")
        return ()
    return (opponent.battle_zone[index],)


def auto_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    opponent = game.non_turn_player if caster is game.turn_player else game.turn_player
    for creature in opponent.battle_zone:
        if creature.is_creature():
            return (creature,)
    return ()
