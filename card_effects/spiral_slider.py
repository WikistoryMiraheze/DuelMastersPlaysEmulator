from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from emulator import Card, Game, Player

CARD_NAME = "スパイラル・スライダー"
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
    cost = creature.cost or 0
    if cost > 6:
        from emulator import GameError

        raise GameError("対象のコストは6以下である必要があります。")
    from emulator import Zone

    game._move_card_with_evolution(opponent, creature, Zone.BATTLE, Zone.HAND)
    if creature in opponent.hand:
        game.log_spell_effect_detail(opponent, f"《{creature.name}》を手札に戻した")
    else:
        game.log_spell_effect_detail(opponent, f"《{creature.name}》は手札に戻れず墓地へ")
    game._update_board_window()


def prompt_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    opponent = game.non_turn_player if caster is game.turn_player else game.turn_player
    valid_targets = [
        (idx, creature)
        for idx, creature in enumerate(opponent.battle_zone)
        if creature.is_creature() and (creature.cost or 0) <= 6
    ]
    if not valid_targets:
        game.output("条件を満たすクリーチャーがいません。")
        return ()

    game.output("手札に戻す対象を選んでください：")
    for idx, creature in valid_targets:
        game.output(f"  {idx}: {creature.name}（コスト：{creature.cost}）")

    from emulator import SINGLE_DIGIT_CHOICES

    choice = game.input("対象の番号（0-9）を入力、他のキーでキャンセル：").strip()
    if choice not in SINGLE_DIGIT_CHOICES:
        return ()
    index = int(choice)
    for original_index, creature in valid_targets:
        if original_index == index:
            return (creature,)
    game.output("不正な選択です。対象は選ばれませんでした。")
    return ()


def auto_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    opponent = game.non_turn_player if caster is game.turn_player else game.turn_player
    for creature in opponent.battle_zone:
        if creature.is_creature() and (creature.cost or 0) <= 6:
            return (creature,)
    return ()
