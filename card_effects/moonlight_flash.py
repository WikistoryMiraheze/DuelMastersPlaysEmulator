from __future__ import annotations

from typing import List, Optional, Sequence, TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from emulator import Card, Game, Player

CARD_NAME = "ムーンライト・フラッシュ"
TARGETS_REQUIRED = True


def spell_effect(
    game: "Game", caster: "Player", opponent: "Player", targets: Sequence["Card"]
) -> None:
    if len(targets) > 2:
        from emulator import GameError  # local import to avoid circular dependency

        raise GameError("ムーンライト・フラッシュの対象は最大2体です。")

    seen: set[int] = set()
    for creature in targets:
        identifier = id(creature)
        if identifier in seen:
            from emulator import GameError

            raise GameError("同じクリーチャーを複数回選ぶことはできません。")
        seen.add(identifier)

        if creature not in opponent.battle_zone or not creature.is_creature():
            from emulator import GameError

            raise GameError("対象は相手のバトルゾーンにいるクリーチャーでなければなりません。")

    for creature in targets:
        opponent.tap_creature(creature)
        game.log_spell_effect_detail(opponent, f"《{creature.name}》タップ")


def prompt_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    opponent = game.non_turn_player if caster is game.turn_player else game.turn_player
    if not opponent.battle_zone:
        game.output("相手のバトルゾーンに対象となるクリーチャーはいません。")
        return ()

    game.output("相手のバトルゾーン：")
    for index, creature in enumerate(opponent.battle_zone):
        state = "タップ" if opponent.is_creature_tapped(creature) else "アンタップ"
        game.output(f"  {index}: {creature.name}（{state}）")

    selection = game.input(
        "対象とする番号をスペース区切りで最大2つ入力してください（空欄でスキップ）："
    ).strip()

    if not selection:
        return ()

    from emulator import SINGLE_DIGIT_CHOICES

    chosen: List["Card"] = []
    for token in selection.split():
        if token in SINGLE_DIGIT_CHOICES:
            idx = int(token)
            if idx < len(opponent.battle_zone):
                card = opponent.battle_zone[idx]
                if card not in chosen:
                    chosen.append(card)
        if len(chosen) == 2:
            break

    if not chosen:
        game.output("有効な対象が選択されませんでした。")

    return tuple(chosen)


def auto_targets(game: "Game", caster: "Player") -> Optional[Sequence["Card"]]:
    opponent = game.non_turn_player if caster is game.turn_player else game.turn_player
    choices: List["Card"] = []
    for creature in opponent.battle_zone:
        if not creature.is_creature():
            continue
        choices.append(creature)
        if len(choices) == 2:
            break
    return tuple(choices)
