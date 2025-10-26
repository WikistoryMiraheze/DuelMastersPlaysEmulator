from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for static typing only
    from emulator import Card, Game, Player

CARD_NAME = "デス・スモーク"
TARGETS_REQUIRED = True


def spell_effect(
    game: "Game", caster: "Player", opponent: "Player", targets: Sequence["Card"]
) -> None:
    if not targets:
        return

    if len(targets) != 1:
        from emulator import GameError

        raise GameError("デス・スモークの対象は1体までです。")

    creature = targets[0]
    if creature not in opponent.battle_zone or not creature.is_creature():
        from emulator import GameError

        raise GameError("対象は相手のバトルゾーンにいるクリーチャーでなければなりません。")

    if opponent.is_creature_tapped(creature):
        game.output("選択されたクリーチャーはすでにタップしています。効果は不発です。")
        game.log_spell_effect_detail(opponent, f"《{creature.name}》はタップ状態のため不発")
        return

    game._destroy_creature(opponent, creature)
    game.log_spell_effect_detail(opponent, f"《{creature.name}》破壊")
    game._update_board_window()


def prompt_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    opponent = game.non_turn_player if caster is game.turn_player else game.turn_player
    valid_targets = [
        (index, creature)
        for index, creature in enumerate(opponent.battle_zone)
        if creature.is_creature() and not opponent.is_creature_tapped(creature)
    ]

    if not valid_targets:
        game.output("相手のアンタップしているクリーチャーはいません。対象は選択されません。")
        return ()

    game.output("相手のアンタップしているクリーチャー：")
    for index, creature in valid_targets:
        game.output(f"  {index}: {creature.name}")

    choice = game.input(
        "破壊するクリーチャーの番号を選択してください（0-9、空入力で対象なし、cでキャンセル）："
    ).strip()

    if not choice:
        game.output("対象を選択しませんでした。")
        return ()

    if choice.lower() == "c":
        game.output("デス・スモークのプレイをキャンセルしました。")
        return None

    from emulator import SINGLE_DIGIT_CHOICES

    if choice not in SINGLE_DIGIT_CHOICES:
        game.output("無効な入力です。対象は選択されません。")
        return ()

    index = int(choice)
    for valid_index, creature in valid_targets:
        if valid_index == index:
            return (creature,)

    game.output("指定した番号のアンタップクリーチャーは存在しません。対象は選択されません。")
    return ()


def auto_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    opponent = game.non_turn_player if caster is game.turn_player else game.turn_player
    for creature in opponent.battle_zone:
        if not creature.is_creature():
            continue
        if opponent.is_creature_tapped(creature):
            continue
        return (creature,)
    return ()
