"""外部ファイルでデッキを構築してエミュレータを起動するサンプル。"""

from __future__ import annotations

from typing import List, Sequence, Tuple, TYPE_CHECKING

from cpu_ai import SimpleCpuAgent
from emulator import Player, create_card_by_name, run_cli_game

if TYPE_CHECKING:
    from emulator import Card

CardEntry = Tuple[str, int]
DeckDefinition = Sequence[CardEntry]

MY_DECK_LIST: DeckDefinition = (
    ("汽車男", 20),
    ("スパイラル・スライダー", 20),
)

ENEMY_DECK_LIST: DeckDefinition = (
    ("ボルシャック・ドラゴン", 40),
)


# 先攻設定（以下のいずれかの文字列を指定してください）
#   "自分が先行" : 常に自分が先攻
#   "自分が後攻" : 常に自分が後攻
#   "ランダム"   : ランダムに先攻を決定
TURN_ORDER_SETTING = "自分が先行"

_TURN_ORDER_MAP = {
    "自分が先行": "human_first",
    "自分が後攻": "human_second",
    "ランダム": "random",
}


def _build_deck(definition: DeckDefinition) -> List["Card"]:
    """Expand ``(カード名, 枚数)`` の組を ``Card`` のリストに変換します。"""

    deck: List["Card"] = []
    for card_name, count in definition:
        deck.extend(create_card_by_name(card_name) for _ in range(count))
    return deck


def build_my_deck() -> List["Card"]:
    """自分用デッキを生成します。"""

    return _build_deck(MY_DECK_LIST)


def build_enemy_deck() -> List["Card"]:
    """相手用デッキを生成します。"""

    return _build_deck(ENEMY_DECK_LIST)


def main() -> None:
    """デッキを構築して CLI エミュレータを起動します。"""

    player1 = Player(name="自分", deck=build_my_deck())
    player2 = Player(name="相手", deck=build_enemy_deck())

    try:
        starting_option = _TURN_ORDER_MAP[TURN_ORDER_SETTING]
    except KeyError as exc:
        valid = "、".join(_TURN_ORDER_MAP.keys())
        raise ValueError(
            f"TURN_ORDER_SETTING には {valid} のいずれかを指定してください。"
        ) from exc

    run_cli_game(
        player1,
        player2,
        cpu_agent=SimpleCpuAgent(),
        starting_player=starting_option,
    )


if __name__ == "__main__":
    main()
