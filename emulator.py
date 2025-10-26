"""Duel Masters Play's emulator core logic.

This module provides a lightweight representation of the basic rules of
"Duel Masters Play's" as described in the accompanying specification.  The
focus of this implementation is tracking zones, mana, turn structure and
fundamental actions such as drawing cards, charging mana and playing simple
cards.  Combat handling and card specific abilities are intentionally left as
extension points.

The code is intentionally explicit and verbose so that it can be expanded in
future work – for example by wiring it to a user interface or scripting
system, or by injecting detailed card behaviours.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
import random
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from card_database import CARD_DATA_BY_ID, CARD_DATA_BY_NAME
from card_effects import (
    ATTACK_POWER_BONUS_HANDLERS,
    ATTACK_TRIGGER_EFFECTS,
    BLOCKER_FILTERS,
    ENTER_BATTLE_ZONE_EFFECTS,
    END_STEP_EFFECTS,
    POST_BATTLE_EFFECTS,
    SPELL_AUTO_TARGETS,
    SPELL_EFFECTS,
    SPELL_TARGET_PROMPTS,
    STATIC_POWER_BONUS_CALCULATORS,
    TARGETED_SPELL_NAMES,
)

try:  # pragma: no cover - GUI availability depends on runtime environment
    import tkinter as tk
    from tkinter import scrolledtext
except Exception:  # pragma: no cover - fall back to console-only mode
    tk = None  # type: ignore[assignment]
    scrolledtext = None  # type: ignore[assignment]


class GameError(Exception):
    """Base class for game related exceptions."""


class DeckOut(GameError):
    """Raised when a player attempts to draw from an empty deck."""


class Zone(Enum):
    """Zones that a card can occupy for a player."""

    DECK = "deck"
    HAND = "hand"
    MANA = "mana"
    SHIELD = "shield"
    BATTLE = "battle"
    GRAVEYARD = "graveyard"
    SUPER_DIMENSION = "super_dimension"
    SUPER_GR = "super_gr"
    FINAL_PS = "final_ps"


ZONE_CAPACITIES: Dict[Zone, int] = {
    Zone.HAND: 10,
    Zone.SHIELD: 10,
    Zone.BATTLE: 7,
}


ZONE_LABELS: Dict[Zone, str] = {
    Zone.DECK: "山札",
    Zone.HAND: "手札",
    Zone.MANA: "マナゾーン",
    Zone.SHIELD: "シールドゾーン",
    Zone.BATTLE: "バトルゾーン",
    Zone.GRAVEYARD: "墓地",
    Zone.SUPER_DIMENSION: "超次元ゾーン",
    Zone.SUPER_GR: "超GRゾーン",
    Zone.FINAL_PS: "最終P'S封印ゾーン",
}


class TurnStep(Enum):
    """Ordered steps for a player's turn."""

    START = auto()
    DRAW = auto()
    MANA_CHARGE = auto()
    MAIN = auto()
    ATTACK = auto()
    END = auto()


STEP_LABELS: Dict[TurnStep, str] = {
    TurnStep.START: "ターン開始ステップ",
    TurnStep.DRAW: "ドローステップ",
    TurnStep.MANA_CHARGE: "マナチャージステップ",
    TurnStep.MAIN: "メインステップ",
    TurnStep.ATTACK: "攻撃ステップ",
    TurnStep.END: "ターン終了ステップ",
}


CREATURE_TYPES = {"クリーチャー", "進化クリーチャー"}
SPELL_TYPES = {"呪文"}


SINGLE_DIGIT_CHOICES = {str(i) for i in range(10)}


StartingPlayerOption = Literal["human_first", "human_second", "random"]


if TYPE_CHECKING:
    from cpu_ai import CpuAgent


def _coerce_power(value: Optional[str]) -> Optional[int]:
    """Convert a display power string into an integer where possible."""

    if value is None:
        return None

    stripped = value.strip()
    if not stripped or stripped in {"なし", "-", "null"}:
        return None

    digits = "".join(ch for ch in stripped if ch.isdigit())
    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def _parse_races(data: Dict[str, Any]) -> Tuple[str, ...]:
    races: List[str] = []
    for key in ("race1", "race2", "race3", "race4"):
        value = data.get(key)
        if value and value != "なし":
            races.append(value)
    return tuple(races)


def _parse_abilities(text: Optional[str]) -> Tuple[str, ...]:
    if not text:
        return ()

    abilities: List[str] = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line[0] in {"◇", "■"}:
            line = line[1:].strip()
        abilities.append(line)
    return tuple(abilities)


def _extract_evolution_requirement(abilities: Sequence[str]) -> Optional[str]:
    for ability in abilities:
        if ability.startswith("進化－"):
            return ability.replace("進化－", "", 1).strip()
    return None


def _card_types_from_entry(entry: Dict[str, Any]) -> Set[str]:
    card_type = entry.get("card_type")
    return {card_type} if card_type else set()


def _civilizations_from_entry(entry: Dict[str, Any]) -> Set[str]:
    culture = entry.get("culture")
    return {culture} if culture else set()


def create_card_from_data(entry: Dict[str, Any]) -> "Card":
    abilities = _parse_abilities(entry.get("body_text"))
    races = _parse_races(entry)
    power = _coerce_power(entry.get("power_disp"))

    return Card(
        name=entry.get("card_name", ""),
        civilizations=_civilizations_from_entry(entry),
        cost=entry.get("cost"),
        mana_number=entry.get("mana"),
        types=_card_types_from_entry(entry),
        power=power,
        races=set(races),
        abilities=abilities,
        card_id=entry.get("card_id"),
        card_type=entry.get("card_type"),
        culture=entry.get("culture"),
        race_names=races,
        power_display=entry.get("power_disp"),
        body_text=entry.get("body_text", ""),
        evolution_requirement=_extract_evolution_requirement(abilities),
    )


def create_card_by_name(name: str) -> "Card":
    try:
        entry = CARD_DATA_BY_NAME[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise GameError(f"不明なカード名です：{name}") from exc
    return create_card_from_data(entry)


def create_card_by_id(card_id: int) -> "Card":
    try:
        entry = CARD_DATA_BY_ID[card_id]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise GameError(f"不明なカードIDです：{card_id}") from exc
    return create_card_from_data(entry)


def card_has_ability(card: Card, keyword: str) -> bool:
    """Return True if any of the card's abilities contains the keyword."""

    return any(keyword in ability for ability in card.abilities)


POWER_ATTACKER_PATTERN = re.compile(r"パワーアタッカー\s*\+?(\d+)")


def extract_power_attacker_bonus(card: Card) -> int:
    bonus = 0
    for ability in card.abilities:
        if "得る" in ability:
            continue
        match = POWER_ATTACKER_PATTERN.search(ability)
        if match:
            bonus += int(match.group(1))
    return bonus


def has_s_trigger(card: Card) -> bool:
    return card_has_ability(card, "S・トリガー")


def has_w_breaker(card: Card) -> bool:
    return card_has_ability(card, "W・ブレイカー")


class BoardWindow:
    """Background thread that renders board and log information in windows."""

    def __init__(self, title: str = "盤面情報") -> None:
        self.enabled = tk is not None
        self._title = title
        self._queue: "queue.Queue[Optional[Tuple[str, str]]]" = queue.Queue()
        self._closed = False
        self._root: Optional["tk.Tk"] = None
        self._text_widget: Optional["tk.Text"] = None
        self._log_window: Optional["tk.Toplevel"] = None
        self._log_text_widget: Optional["tk.Text"] = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:  # pragma: no cover - UI loop is not unit tested
        self._root = tk.Tk()
        self._root.title(self._title)
        board_widget: "tk.Text"
        if scrolledtext is not None:
            board_widget = scrolledtext.ScrolledText(
                self._root,
                width=80,
                height=30,
                state="disabled",
                font=("Consolas", 10),
            )
        else:
            board_widget = tk.Text(
                self._root, width=80, height=30, state="disabled", font=("Consolas", 10)
            )
        self._text_widget = board_widget
        board_widget.pack(fill="both", expand=True)

        self._log_window = tk.Toplevel(self._root)
        self._log_window.title("バトルログ")
        if scrolledtext is not None:
            log_widget: "tk.Text" = scrolledtext.ScrolledText(
                self._log_window,
                width=80,
                height=30,
                state="disabled",
                font=("Consolas", 10),
            )
        else:
            log_widget = tk.Text(
                self._log_window, width=80, height=30, state="disabled", font=("Consolas", 10)
            )
        self._log_text_widget = log_widget
        log_widget.pack(fill="both", expand=True)

        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._log_window.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_updates()
        self._root.mainloop()

    def _poll_updates(self) -> None:  # pragma: no cover - UI loop is not unit tested
        if self._closed:
            return

        updated = False
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break

            if message is None:
                self._closed = True
                if self._root is not None:
                    self._root.after(0, self._root.destroy)
                if self._log_window is not None:
                    try:
                        self._log_window.after(0, self._log_window.destroy)
                    except Exception:
                        pass
                return

            target, payload = message
            if target == "board" and self._text_widget is not None:
                self._text_widget.configure(state="normal")
                self._text_widget.delete("1.0", "end")
                self._text_widget.insert("1.0", payload)
                self._text_widget.configure(state="disabled")
                updated = True
            elif target == "log" and self._log_text_widget is not None:
                self._log_text_widget.configure(state="normal")
                self._log_text_widget.delete("1.0", "end")
                self._log_text_widget.insert("1.0", payload)
                self._log_text_widget.configure(state="disabled")
                updated = True

        if self._root is not None and not self._closed:
            delay = 100 if updated else 200
            self._root.after(delay, self._poll_updates)

    def update(self, message: str) -> None:
        if not self.enabled or self._closed:
            return
        self._queue.put(("board", message))

    def update_log(self, message: str) -> None:
        if not self.enabled or self._closed:
            return
        self._queue.put(("log", message))

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
        if self._log_window is not None:
            try:
                self._log_window.after(0, self._log_window.destroy)
            except Exception:
                pass
        self._queue.put(None)


@dataclass(eq=False)
class Card:
    """Representation of a Duel Masters card."""

    name: str
    civilizations: Set[str] = field(default_factory=set)
    cost: Optional[int] = None
    mana_number: Optional[int] = None
    types: Set[str] = field(default_factory=set)
    power: Optional[int] = None
    races: Set[str] = field(default_factory=set)
    abilities: Sequence[str] = field(default_factory=list)
    card_id: Optional[int] = None
    card_type: Optional[str] = None
    culture: Optional[str] = None
    race_names: Tuple[str, ...] = field(default_factory=tuple)
    power_display: Optional[str] = None
    body_text: str = ""
    evolution_requirement: Optional[str] = None

    def is_creature(self) -> bool:
        return bool(self.types & CREATURE_TYPES)

    def is_spell(self) -> bool:
        return bool(self.types & SPELL_TYPES)

    def is_evolution(self) -> bool:
        return "進化クリーチャー" in self.types

    def is_multicolored(self) -> bool:
        """Return True when the card has two or more civilizations."""

        return len(self.civilizations) >= 2

    def has_race(self, race: str) -> bool:
        return race in self.races

    def has_civilization(self, civilization: str) -> bool:
        return civilization in self.civilizations


@dataclass
class Player:
    """State container for an individual player."""

    name: str
    deck: List[Card]
    hand: List[Card] = field(default_factory=list)
    mana_zone: List[Card] = field(default_factory=list)
    shields: List[Card] = field(default_factory=list)
    battle_zone: List[Card] = field(default_factory=list)
    graveyard: List[Card] = field(default_factory=list)
    super_dimension_zone: List[Card] = field(default_factory=list)
    super_gr_zone: List[Card] = field(default_factory=list)
    final_ps_zone: List[Card] = field(default_factory=list)
    max_mana: int = 0
    max_available_mana: int = 0
    available_mana: int = 0
    unlocked_civilizations: Set[str] = field(default_factory=set)
    battle_zone_tapped: Dict[Card, bool] = field(default_factory=dict)
    battle_zone_entry_turn: Dict[Card, int] = field(default_factory=dict)
    evolution_sources: Dict[Card, List[Card]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._zone_mapping: Dict[Zone, List[Card]] = {
            Zone.DECK: self.deck,
            Zone.HAND: self.hand,
            Zone.MANA: self.mana_zone,
            Zone.SHIELD: self.shields,
            Zone.BATTLE: self.battle_zone,
            Zone.GRAVEYARD: self.graveyard,
            Zone.SUPER_DIMENSION: self.super_dimension_zone,
            Zone.SUPER_GR: self.super_gr_zone,
            Zone.FINAL_PS: self.final_ps_zone,
        }

    # ------------------------------------------------------------------
    # Card movement helpers
    # ------------------------------------------------------------------
    def draw_card(self) -> Card:
        """Draw a card from the top of the deck into the hand."""

        if not self.deck:
            raise DeckOut(f"{self.name}はドローするカードが残っていません。")

        card = self.deck.pop(0)
        self._add_card_to_zone(card, Zone.HAND)
        if not self.deck:
            raise DeckOut(f"{self.name}の山札が尽きました。")
        return card

    def draw_cards(self, amount: int) -> List[Card]:
        return [self.draw_card() for _ in range(amount)]

    def move_card(self, card: Card, from_zone: Zone, to_zone: Zone) -> None:
        self._remove_card_from_zone(card, from_zone)
        self._add_card_to_zone(card, to_zone)

    def _remove_card_from_zone(self, card: Card, zone: Zone) -> None:
        zone_cards = self._zone_mapping[zone]
        try:
            zone_cards.remove(card)
        except ValueError as exc:  # pragma: no cover - safety guard
            zone_name = ZONE_LABELS.get(zone, zone.value)
            raise GameError(f"カード《{card.name}》は{zone_name}に存在しません。") from exc

        self._on_leave_zone(zone, card)

    def _add_card_to_zone(self, card: Card, zone: Zone) -> None:
        if zone is Zone.GRAVEYARD:
            self.graveyard.append(card)
            return

        zone_cards = self._zone_mapping[zone]
        limit = ZONE_CAPACITIES.get(zone)
        if limit is not None and len(zone_cards) >= limit:
            # Overflow goes directly to the graveyard.
            self.graveyard.append(card)
            return

        zone_cards.append(card)
        self._on_enter_zone(zone, card)

    def send_to_graveyard(self, card: Card, from_zone: Zone) -> None:
        self._remove_card_from_zone(card, from_zone)
        self.graveyard.append(card)

    # ------------------------------------------------------------------
    # Zone bookkeeping
    # ------------------------------------------------------------------
    def _on_enter_zone(self, zone: Zone, card: Card) -> None:
        if zone is Zone.MANA:
            self.max_mana += 1
            if card.mana_number == 1:
                self.max_available_mana += 1
                if not card.is_multicolored():
                    self.available_mana += 1
            self._recalculate_unlocked_civilizations()
            self._clamp_available_mana()
        elif zone is Zone.BATTLE and card.is_creature():
            # Creatures enter the battle zone untapped unless card-specific
            # effects specify otherwise (such effects can override this hook).
            self.battle_zone_tapped[card] = False

    def _on_leave_zone(self, zone: Zone, card: Card) -> None:
        if zone is Zone.MANA:
            self.max_mana = max(0, self.max_mana - 1)
            if card.mana_number == 1:
                self.max_available_mana = max(0, self.max_available_mana - 1)
            self._recalculate_unlocked_civilizations()
            self._clamp_available_mana()
        elif zone is Zone.BATTLE:
            if card in self.battle_zone_tapped:
                # Remove any tap tracking when the creature leaves the battle zone.
                self.battle_zone_tapped.pop(card, None)
            if card in self.battle_zone_entry_turn:
                self.battle_zone_entry_turn.pop(card, None)

    def _recalculate_unlocked_civilizations(self) -> None:
        civs: Set[str] = set()
        for mana_card in self.mana_zone:
            civs.update(mana_card.civilizations)
        self.unlocked_civilizations = civs

    def _clamp_available_mana(self) -> None:
        if self.available_mana > self.max_available_mana:
            self.available_mana = self.max_available_mana

    # ------------------------------------------------------------------
    # Helpers for gameplay actions
    # ------------------------------------------------------------------
    def add_shield_from_top_of_deck(self) -> Card:
        if not self.deck:
            raise DeckOut(f"{self.name}はシールドに置くカードが残っていません。")

        card = self.deck.pop(0)
        self._add_card_to_zone(card, Zone.SHIELD)
        if not self.deck:
            raise DeckOut(f"{self.name}の山札が尽きました。")
        return card

    def charge_mana_from_hand(self, card: Card) -> None:
        self._remove_card_from_zone(card, Zone.HAND)
        self._add_card_to_zone(card, Zone.MANA)

    def available_creature_slots(self) -> int:
        return max(0, ZONE_CAPACITIES[Zone.BATTLE] - len(self.battle_zone))

    def has_required_civilizations(self, civilizations: Iterable[str]) -> bool:
        return set(civilizations).issubset(self.unlocked_civilizations)

    # ------------------------------------------------------------------
    # Battle zone tap handling
    # ------------------------------------------------------------------
    def tap_creature(self, creature: Card) -> None:
        if creature not in self.battle_zone:
            raise GameError("そのクリーチャーはバトルゾーンに存在しません。")
        if not creature.is_creature():
            raise GameError("タップできるのはクリーチャーだけです。")
        self.battle_zone_tapped[creature] = True

    def untap_creature(self, creature: Card) -> None:
        if creature not in self.battle_zone:
            raise GameError("そのクリーチャーはバトルゾーンに存在しません。")
        if not creature.is_creature():
            raise GameError("アンタップできるのはクリーチャーだけです。")
        self.battle_zone_tapped[creature] = False

    def is_creature_tapped(self, creature: Card) -> bool:
        if creature not in self.battle_zone:
            raise GameError("そのクリーチャーはバトルゾーンに存在しません。")
        if not creature.is_creature():
            raise GameError("バトルゾーンでタップ状態を持つのはクリーチャーだけです。")
        return self.battle_zone_tapped.get(creature, False)

    def untap_all_battle_creatures(self) -> None:
        for creature in list(self.battle_zone_tapped):
            self.battle_zone_tapped[creature] = False

    def mark_creature_entry_turn(self, creature: Card, turn_number: int) -> None:
        if creature.is_creature():
            self.battle_zone_entry_turn[creature] = turn_number

    def creature_entry_turn(self, creature: Card) -> Optional[int]:
        return self.battle_zone_entry_turn.get(creature)

    def set_evolution_sources(self, evolution: Card, bases: Sequence[Card]) -> None:
        self.evolution_sources[evolution] = list(bases)

    def clear_evolution_sources(self, evolution: Card) -> List[Card]:
        return self.evolution_sources.pop(evolution, [])

    def get_evolution_sources(self, evolution: Card) -> List[Card]:
        return list(self.evolution_sources.get(evolution, []))


class Game:
    """Controller for the overall flow of a Duel Masters Play's game."""

    def __init__(
        self,
        player1: Player,
        player2: Player,
        *,
        human_player_index: Optional[int] = 0,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
        board_window: Optional[BoardWindow] = None,
        cpu_agent: Optional["CpuAgent"] = None,
    ) -> None:
        self.players = [player1, player2]
        self.turn_player_index = 0
        self.turn_number = 0
        self.current_step = TurnStep.START
        self.current_turn_is_extra = False
        self.extra_turns: List[int] = [0, 0]
        self.game_over = False
        self.winner: Optional[Player] = None
        self.mana_charged_this_turn = False
        self.human_player_index = human_player_index
        self.input = input_func
        self.output = output_func
        self.result_announced = False
        self.board_window = board_window
        self.cpu_agent = cpu_agent
        self.turn_counts = [0 for _ in self.players]
        self.battle_log_lines: List[str] = []
        self.turn_power_modifiers: Dict[Card, int] = {}
        self.pending_creature_entry_triggers: List[Tuple[Player, Card]] = []

    # ------------------------------------------------------------------
    # Game setup
    # ------------------------------------------------------------------
    def start_game(self) -> None:
        """Initialises the game by placing shields and drawing cards."""

        for player in self.players:
            random.shuffle(player.deck)
            try:
                for _ in range(5):
                    player.add_shield_from_top_of_deck()
                player.draw_cards(5)
            except DeckOut:
                self._handle_deck_out(player)
                return

        self._update_board_window()
        self._update_battle_log_window()

    # ------------------------------------------------------------------
    # Turn flow
    # ------------------------------------------------------------------
    @property
    def turn_player(self) -> Player:
        return self.players[self.turn_player_index]

    @property
    def non_turn_player(self) -> Player:
        return self.players[1 - self.turn_player_index]

    def _player_index(self, player: Player) -> int:
        try:
            return self.players.index(player)
        except ValueError as exc:  # pragma: no cover - safety guard
            raise GameError("このプレイヤーはこの対戦の参加者ではありません。") from exc

    def _cpu_agent_for(self, player: Player) -> Optional["CpuAgent"]:
        if self.cpu_agent is None:
            return None

        if self.human_player_index is not None and self._player_index(player) == self.human_player_index:
            return None

        return self.cpu_agent

    def grant_extra_turn(self, player: Player, amount: int = 1) -> None:
        index = self._player_index(player)
        self.extra_turns[index] += amount

    def run_turn(self) -> None:
        if self.game_over:
            return

        player = self.turn_player
        opponent = self.non_turn_player

        self.current_turn_is_extra = self._consume_extra_turn(player)
        self.turn_number += 1
        self.mana_charged_this_turn = False

        self._log_turn_start(player)

        extra_turn_label = "（追加ターン）" if self.current_turn_is_extra else ""
        self.output(f"==== ターン{self.turn_number}：{player.name}{extra_turn_label} ====")

        self._turn_start_step(player)
        if self.game_over:
            return

        self._draw_step(player)
        if self.game_over:
            return

        self._mana_charge_step(player)
        if self.game_over:
            return

        self._main_step(player)
        if self.game_over:
            return

        self._attack_step(player, opponent)
        if self.game_over:
            return

        self._end_step(player)

        self._determine_next_player(player, opponent)

    def _consume_extra_turn(self, player: Player) -> bool:
        index = self._player_index(player)
        remaining = self.extra_turns[index]
        if remaining > 0:
            self.extra_turns[index] = remaining - 1
            return True
        return False

    # ------------------------------------------------------------------
    # Individual turn steps
    # ------------------------------------------------------------------
    def _set_step(self, step: TurnStep) -> None:
        self.current_step = step
        step_label = STEP_LABELS.get(step, step.name)
        self.output(f"現在のステップ：{step_label}")
        self._display_public_information()

    def _display_public_information(self) -> None:
        """Refresh any auxiliary displays without spamming the console."""
        self._update_board_window()

    def _update_battle_log_window(self) -> None:
        if self.board_window is None:
            return

        text = "\n".join(self.battle_log_lines)
        self.board_window.update_log(text)

    def _log_line(self, line: str) -> None:
        self.battle_log_lines.append(line)
        self._update_battle_log_window()

    def _is_human_player(self, player: Player) -> bool:
        if self.human_player_index is None:
            return False
        return self._player_index(player) == self.human_player_index

    def _player_prefix(self, player: Player) -> str:
        if self.human_player_index is None:
            return player.name
        return "自分" if self._is_human_player(player) else "相手"

    def _log_turn_start(self, player: Player) -> None:
        try:
            index = self._player_index(player)
        except GameError:
            return

        self.turn_counts[index] += 1
        prefix = self._player_prefix(player)
        header = f"＝＝＝{prefix}の{self.turn_counts[index]}ターン目＝＝＝"
        if self.current_turn_is_extra:
            header += "（追加ターン）"
        self._log_line(header)

    def _log_draw(self, player: Player, card: Card, before: int, after: int) -> None:
        prefix = self._player_prefix(player)
        if self._is_human_player(player):
            line = f"{prefix}《{card.name}》　手札：{before}→{after}"
        else:
            line = f"{prefix}手札：{before}→{after}"
        self._log_line(line)

    def _log_mana_charge(self, player: Player, card: Card, before: int, after: int) -> None:
        prefix = self._player_prefix(player)
        line = f"{prefix}《{card.name}》　マナ：{before}→{after}"
        self._log_line(line)

    def _log_summon(self, player: Player, card: Card) -> None:
        prefix = self._player_prefix(player)
        self._log_line(f"{prefix}《{card.name}》　召喚")

    def _log_spell_cast(self, player: Player, card: Card) -> None:
        prefix = self._player_prefix(player)
        self._log_line(f"{prefix}《{card.name}》　唱えた")

    def _log_spell_resolution(self, player: Player, card: Card) -> None:
        prefix = self._player_prefix(player)
        self._log_line(f"{prefix}《{card.name}》　能力解決")

    def _log_shield_overflow(self, player: Player, card: Card) -> None:
        prefix = self._player_prefix(player)
        self._log_line(f"{prefix}《{card.name}》　手札上限のため墓地へ")

    def log_spell_effect_detail(self, owner: Player, detail: str) -> None:
        prefix = self._player_prefix(owner)
        self._log_line(f"┗{prefix}{detail}")

    def log_creature_effect_detail(self, owner: Player, detail: str) -> None:
        prefix = self._player_prefix(owner)
        self._log_line(f"┗{prefix}{detail}")

    def _log_attack_declaration(
        self,
        attacker_owner: Player,
        defender: Player,
        attacker: Card,
        target_type: str,
        target_creature: Optional[Card],
    ) -> None:
        prefix = self._player_prefix(attacker_owner)
        if target_type == "creature" and target_creature is not None:
            target_label = target_creature.name
        else:
            target_label = f"{self._player_prefix(defender)}プレイヤー"
        self._log_line(f"{prefix}《{attacker.name}》　{target_label}に攻撃")

    def _log_block(self, player: Player, blocker: Card) -> None:
        prefix = self._player_prefix(player)
        self._log_line(f"{prefix}《{blocker.name}》　ブロック")

    def _log_shield_break(
        self,
        attacker_owner: Player,
        defender: Player,
        attacker: Card,
        shields_before: int,
        shields_after: int,
        hand_before: int,
        hand_after: int,
    ) -> None:
        attacker_prefix = self._player_prefix(attacker_owner)
        defender_prefix = self._player_prefix(defender)
        self._log_line(f"{attacker_prefix}《{attacker.name}》　シールドブレイク")
        self._log_line(f"{defender_prefix}シールド：{shields_before}→{shields_after}")
        self._log_line(f"{defender_prefix}手札：{hand_before}→{hand_after}")

    def _log_direct_attack(self, attacker_owner: Player, attacker: Card) -> None:
        prefix = self._player_prefix(attacker_owner)
        self._log_line(f"{prefix}《{attacker.name}》　ダイレクトアタック")

    def _log_battle_result(
        self, attacker: Card, defender: Card, outcome: str
    ) -> None:
        if outcome == "attacker":
            line = f"バトル{attacker.name}（勝利） VS {defender.name}（敗北）"
        elif outcome == "defender":
            line = f"バトル{attacker.name}（敗北） VS {defender.name}（勝利）"
        else:
            line = f"バトル{attacker.name}（引き分け） VS {defender.name}（引き分け）"
        self._log_line(line)

    def _build_board_window_text(self) -> str:
        lines: List[str] = []

        turn_label = self.turn_number if self.turn_number > 0 else 0
        lines.append("デュエル・マスターズ プレイス エミュレータ")
        if self.game_over:
            result = self._result_message()
            lines.append(f"結果：{result}")
        else:
            extra_label = "（追加ターン）" if self.current_turn_is_extra else ""
            lines.append(f"ターン：{turn_label}{extra_label}")
            step_label = STEP_LABELS.get(self.current_step, self.current_step.name)
            lines.append(f"ステップ：{step_label}")

        lines.append("")

        for index, player in enumerate(self.players):
            label = player.name
            if index == self.human_player_index:
                label += "（自分）"
            lines.append(label)
            lines.append(f"  山札：{len(player.deck)}枚")
            lines.append(f"  手札：{len(player.hand)}枚")
            lines.append(
                "  マナゾーン："
                f"{len(player.mana_zone)}枚（最大マナ {player.max_mana}／"
                f"使用可能 {player.available_mana}/{player.max_available_mana}）"
            )
            if player.mana_zone:
                for mana_card in player.mana_zone:
                    civs = "/".join(sorted(mana_card.civilizations)) or "無色"
                    types = "/".join(sorted(mana_card.types)) or "タイプなし"
                    mana_number = mana_card.mana_number if mana_card.mana_number is not None else "-"
                    lines.append(
                        "    - "
                        f"{mana_card.name} | 文明：{civs} | マナ数：{mana_number} | "
                        f"カードタイプ：{types}"
                    )
            lines.append(f"  シールド：{len(player.shields)}枚")
            if player.shields:
                lines.append("    - （非公開）")
            lines.append(
                "  バトルゾーン："
                f"{len(player.battle_zone)}/{ZONE_CAPACITIES[Zone.BATTLE]}体"
            )
            if player.battle_zone:
                for creature in player.battle_zone:
                    if creature.is_creature():
                        state = "タップ" if player.is_creature_tapped(creature) else "アンタップ"
                    else:
                        state = "-"
                    civs = "/".join(sorted(creature.civilizations)) or "無色"
                    if creature.is_creature():
                        power_value = self._calculate_card_power(player, creature)
                    else:
                        power_value = creature.power if creature.power is not None else "-"
                    lines.append(
                        "    - "
                        f"{creature.name} | 文明：{civs} | パワー：{power_value} | "
                        f"状態：{state}"
                    )
            lines.append(f"  墓地：{len(player.graveyard)}枚")
            if player.graveyard:
                lines.append("    - 墓地のカード（上から順）")
                for index, grave_card in enumerate(player.graveyard, 1):
                    civs = "/".join(sorted(grave_card.civilizations)) or "無色"
                    types = "/".join(sorted(grave_card.types)) or "タイプなし"
                    lines.append(
                        "      "
                        f"{index}: {grave_card.name} | 文明：{civs} | カードタイプ：{types}"
                    )
            lines.append("")

        return "\n".join(lines).strip()

    def _update_board_window(self) -> None:
        if self.board_window is None:
            return

        text = self._build_board_window_text()
        self.board_window.update(text)

    def _handle_creature_entry(
        self, player: Player, card: Card, *, defer_triggers: bool = False
    ) -> None:
        """Handle bookkeeping when a creature enters the battle zone."""

        entry_turn = self.turn_number
        if card.is_evolution():
            entry_turn = max(0, self.turn_number - 1)
        player.mark_creature_entry_turn(card, entry_turn)

        self.pending_creature_entry_triggers.append((player, card))
        if not defer_triggers:
            self._resolve_pending_creature_entries()

    def _resolve_pending_creature_entries(self) -> None:
        while self.pending_creature_entry_triggers:
            owner, creature = self.pending_creature_entry_triggers.pop(0)
            if creature not in owner.battle_zone:
                continue
            self._execute_creature_entry_trigger(owner, creature)

    def _execute_creature_entry_trigger(self, player: Player, card: Card) -> None:
        effect = ENTER_BATTLE_ZONE_EFFECTS.get(card.name)
        if effect is not None:
            effect(self, player, card)

    def _draw_cards_from_effect(
        self, player: Player, amount: int, source: Union[Card, str]
    ) -> None:
        source_name = source.name if isinstance(source, Card) else str(source)
        for _ in range(amount):
            hand_before = len(player.hand)
            try:
                drawn = player.draw_card()
            except DeckOut:
                self._handle_deck_out(player)
                return
            hand_after = len(player.hand)
            self.log_creature_effect_detail(
                player,
                f"《{source_name}》で{drawn.name}をドロー",
            )
            self._log_draw(player, drawn, hand_before, hand_after)

    def _move_top_deck_to_mana(
        self, player: Player, source: Union[Card, str]
    ) -> None:
        source_name = source.name if isinstance(source, Card) else str(source)
        if not player.deck:
            self.log_creature_effect_detail(player, f"《{source_name}》の能力は山札がないため不発")
            return
        mana_card = player.deck.pop(0)
        player._add_card_to_zone(mana_card, Zone.MANA)
        self.log_creature_effect_detail(player, f"《{mana_card.name}》をマナゾーンへ")
        if not player.deck and not self.game_over:
            self._handle_deck_out(player)

    def _force_random_discard(
        self, victim: Player, source_owner: Player, source: Card
    ) -> None:
        if not victim.hand:
            return
        discarded = random.choice(victim.hand)
        victim.hand.remove(discarded)
        victim.graveyard.append(discarded)
        self.log_creature_effect_detail(
            source_owner,
            f"《{source.name}》の効果で相手の《{discarded.name}》を捨てさせた",
        )
        self._update_board_window()

    def _turn_start_step(self, player: Player) -> None:
        self._set_step(TurnStep.START)
        # Trigger handling would be inserted here.
        if self.turn_power_modifiers:
            self.turn_power_modifiers.clear()
        player.untap_all_battle_creatures()
        player.available_mana = player.max_available_mana

    def _draw_step(self, player: Player) -> None:
        self._set_step(TurnStep.DRAW)
        is_first_player = self.turn_number == 1 and not self.current_turn_is_extra
        if is_first_player and player is self.players[0]:
            return

        hand_before = len(player.hand)
        try:
            drawn = player.draw_card()
        except DeckOut:
            self._handle_deck_out(player)
        else:
            hand_after = len(player.hand)
            self._log_draw(player, drawn, hand_before, hand_after)

    def _mana_charge_step(self, player: Player) -> None:
        self._set_step(TurnStep.MANA_CHARGE)
        agent = self._cpu_agent_for(player)
        if agent is not None:
            options = self.available_mana_charge_cards(player)
            if not options:
                self.output(f"{player.name}はマナチャージを行いません。")
                return

            selection = agent.choose_mana_charge(self, player, options)
            if selection is None:
                self.output(f"{player.name}はマナチャージを行いません。")
                return

            if selection not in options:
                raise GameError("CPUが不正なマナチャージ対象を選択しました。")

            self.output(f"{player.name}は《{selection.name}》をマナチャージしました。")
            self.charge_mana(player, selection)
            return

        if not self._player_can_charge_mana(player):
            if not player.hand:
                self.output("手札が空のためマナチャージステップを終了します。")
            elif self.mana_charged_this_turn:
                self.output("このターンは既にマナチャージ済みです。")
            else:
                self.output("マナチャージ可能なカードがありません。")
            return

        self._display_hand(player)
        choice = self.input(
            "マナチャージするカードの番号（0-9）を入力してください。その他の入力でスキップします："
        ).strip()

        if choice in SINGLE_DIGIT_CHOICES:
            index = int(choice)
            if index < len(player.hand):
                card = player.hand[index]
                try:
                    self.charge_mana(player, card)
                except GameError as exc:
                    self.output(f"マナチャージできません：{exc}")
                else:
                    self.output(f"《{card.name}》をマナゾーンに置きました。")
                    self.output(
                        f"使用可能マナ：{player.available_mana}/{player.max_available_mana}"
                    )
            else:
                self.output("無効な番号です。マナチャージをスキップします。")
        else:
            self.output("マナチャージを行いませんでした。")

    def charge_mana(self, player: Player, card: Card) -> None:
        if player is not self.turn_player:
            raise GameError("マナをチャージできるのはターンプレイヤーだけです。")
        if self.current_step is not TurnStep.MANA_CHARGE:
            raise GameError("マナチャージはマナチャージステップでのみ行えます。")
        if self.mana_charged_this_turn:
            raise GameError("このターンは既にマナチャージ済みです。")

        if card not in player.hand:
            raise GameError("選択したカードは手札にありません。")

        before_max = player.max_mana
        player.charge_mana_from_hand(card)
        self.mana_charged_this_turn = True
        self._log_mana_charge(player, card, before_max, player.max_mana)
        self._update_board_window()

    def _main_step(self, player: Player) -> None:
        self._set_step(TurnStep.MAIN)
        agent = self._cpu_agent_for(player)
        if agent is not None:
            while True:
                playable_cards = self.playable_cards(player)
                if not playable_cards:
                    self.output(f"{player.name}はメインステップを行動せずに終了します。")
                    break

                decision = agent.choose_card_to_play(self, player, playable_cards)
                if decision is None:
                    self.output(f"{player.name}はメインステップを行動せずに終了します。")
                    break

                card, targets = decision
                if card not in playable_cards:
                    raise GameError("CPUが不正なカードを選択しました。")

                try:
                    self.play_card_from_hand(player, card, targets)
                except GameError as exc:
                    self.output(f"{player.name}は《{card.name}》をプレイできませんでした：{exc}")
                    break
                else:
                    self.output(f"{player.name}は《{card.name}》をプレイしました。")

                if self.game_over:
                    break

            return

        while True:
            if not player.hand:
                self.output("手札がないためメインステップを終了します。")
                break

            if not self._player_has_playable_card(player):
                self.output("プレイ可能なカードがないため自動的にメインステップを終了します。")
                break

            self._display_hand(player)
            choice = self.input(
                "プレイするカードの番号（0-9）を入力してください。その他の入力でメインステップを終了します："
            ).strip()

            if choice not in SINGLE_DIGIT_CHOICES:
                self.output("メインステップを終了します。")
                break

            index = int(choice)
            if index >= len(player.hand):
                self.output("無効な番号です。手札にあるカードの番号を入力してください。")
                continue

            card = player.hand[index]
            targets = self._gather_spell_targets(card, player)
            if targets is None:
                continue
            try:
                self.play_card_from_hand(player, card, targets)
            except GameError as exc:
                self.output(f"カードをプレイできません：{exc}")
            else:
                self.output(f"《{card.name}》をプレイしました。")

            if self.game_over:
                break

    def play_card_from_hand(
        self, player: Player, card: Card, targets: Optional[Sequence[Card]] = None
    ) -> None:
        self._validate_card_play(player, card)

        evolution_base: Optional[Card] = None
        if card.is_creature() and card.is_evolution():
            selected = self._select_evolution_base(player, card)
            if selected is None:
                raise GameError("進化がキャンセルされました。")
            evolution_base = selected

        player.hand.remove(card)
        player.available_mana -= card.cost

        if card.is_creature():
            if evolution_base is not None:
                self._apply_evolution(player, card, evolution_base)
            else:
                player._add_card_to_zone(card, Zone.BATTLE)
            self._handle_creature_entry(player, card, defer_triggers=True)
            self._log_summon(player, card)
            self._resolve_pending_creature_entries()
        elif card.is_spell():
            # Spells resolve immediately.  Known spells can execute bespoke
            # behaviour before they are moved to the graveyard.
            resolved_targets: Sequence[Card] = tuple(targets) if targets else ()
            self._log_spell_cast(player, card)
            self._log_spell_resolution(player, card)
            self._resolve_spell(card, player, resolved_targets)
            player.graveyard.append(card)
        else:
            # Other card types are placed into the battle zone by default.  This
            # can be refined as the emulator grows.
            player._add_card_to_zone(card, Zone.BATTLE)

        self._update_board_window()

    def _attack_step(self, player: Player, opponent: Player) -> None:
        self._set_step(TurnStep.ATTACK)
        agent = self._cpu_agent_for(player)
        if agent is not None:
            while True:
                attackers = self.available_attackers(player)
                if not attackers:
                    self.output(f"{player.name}は攻撃ステップをスキップします。")
                    break

                decision = agent.choose_attack(self, player, opponent, attackers)
                if decision is None:
                    self.output(f"{player.name}は攻撃ステップを終了します。")
                    break

                attacker, target_info = decision
                if attacker not in attackers:
                    raise GameError("CPUが不正な攻撃クリーチャーを選択しました。")

                valid_targets = self.available_attack_targets(player, opponent, attacker)
                if target_info not in valid_targets:
                    raise GameError("CPUが不正な攻撃対象を選択しました。")

                self._execute_attack(player, opponent, attacker, target_info)

                if self.game_over:
                    break

            return

        while True:
            attackers = self.available_attackers(player)
            if not attackers:
                self.output("攻撃可能なクリーチャーがいないため攻撃ステップを終了します。")
                break

            self._display_attackers(player, attackers)
            choice = self.input(
                "攻撃するクリーチャーの番号（0-9）を入力してください。その他の入力で攻撃ステップを終了します："
            ).strip()

            if choice not in SINGLE_DIGIT_CHOICES:
                self.output("攻撃ステップを終了します。")
                break

            index = int(choice)
            if index >= len(attackers):
                self.output("無効な番号です。攻撃可能なクリーチャーの番号を入力してください。")
                continue

            attacker = attackers[index]
            target_info = self._choose_attack_target(player, opponent, attacker)
            if target_info is None:
                self.output("攻撃を取りやめました。")
                continue

            self._execute_attack(player, opponent, attacker, target_info)

            if self.game_over:
                break

    def available_attackers(self, player: Player) -> List[Card]:
        attackers: List[Card] = []
        for creature in player.battle_zone:
            if not creature.is_creature():
                continue
            if player.is_creature_tapped(creature):
                continue
            if card_has_ability(creature, "攻撃できない"):
                continue

            entry_turn = player.creature_entry_turn(creature)
            if (
                entry_turn is not None
                and entry_turn >= self.turn_number
                and not creature.is_evolution()
                and not card_has_ability(creature, "スピードアタッカー")
            ):
                continue

            attackers.append(creature)

        return attackers

    def available_attack_targets(
        self, player: Player, opponent: Player, attacker: Card
    ) -> List[Tuple[str, Optional[Card]]]:
        valid_targets: List[Tuple[str, Optional[Card]]] = []

        if not card_has_ability(attacker, "相手プレイヤーを攻撃できない"):
            valid_targets.append(("opponent", None))

        for creature in opponent.battle_zone:
            if not creature.is_creature():
                continue
            if not opponent.is_creature_tapped(creature):
                continue
            if card_has_ability(creature, "攻撃されない"):
                continue
            valid_targets.append(("creature", creature))

        return valid_targets

    def available_blockers(self, defender: Player, attacker: Card) -> List[Card]:
        blockers: List[Card] = []

        if attacker.name in {"キャンディ・ドロップ", "キング・オリオン"}:
            return blockers

        for creature in defender.battle_zone:
            if not creature.is_creature():
                continue
            if defender.is_creature_tapped(creature):
                continue
            if not card_has_ability(creature, "ブロッカー"):
                continue
            if attacker.name == "大昆虫ガイアマンティス":
                power = self._calculate_card_power(defender, creature)
                if power <= 8000:
                    continue
            blockers.append(creature)
        return blockers

    def _display_attackers(self, player: Player, attackers: Sequence[Card]) -> None:
        self.output("攻撃可能なクリーチャー：")
        for idx, creature in enumerate(attackers):
            power = creature.power if creature.power is not None else "-"
            self.output(f"  {idx}: {creature.name}（パワー：{power}）")

    def _choose_attack_target(
        self, player: Player, opponent: Player, attacker: Card
    ) -> Optional[Tuple[str, Optional[Card]]]:
        valid_targets = self.available_attack_targets(player, opponent, attacker)

        if not valid_targets:
            self.output("攻撃可能な対象がありません。")
            return None

        self.output("攻撃対象を選んでください：")
        for idx, (target_type, creature) in enumerate(valid_targets):
            if target_type == "opponent":
                self.output(f"  {idx}: {opponent.name}を攻撃")
            elif creature is not None:
                power = creature.power if creature.power is not None else "-"
                self.output(f"  {idx}: {creature.name}（パワー：{power}、タップ）")

        choice = self.input(
            "攻撃対象の番号（0-9）を入力してください。その他の入力で攻撃を取り消します："
        ).strip()

        if choice not in SINGLE_DIGIT_CHOICES:
            return None

        index = int(choice)
        if index >= len(valid_targets):
            self.output("無効な対象が選択されました。")
            return None

        return valid_targets[index]

    def _execute_attack(
        self,
        attacker_owner: Player,
        defender: Player,
        attacker: Card,
        target_info: Tuple[str, Optional[Card]],
    ) -> None:
        if attacker not in attacker_owner.battle_zone:
            self.output("攻撃クリーチャーがバトルゾーンに存在しないため攻撃は中止されました。")
            return

        if attacker_owner.is_creature_tapped(attacker):
            self.output("攻撃しようとしたクリーチャーはすでにタップしています。")
            return

        target_type, target_creature = target_info

        self._log_attack_declaration(attacker_owner, defender, attacker, target_type, target_creature)

        attacker_owner.tap_creature(attacker)
        self._update_board_window()

        self._handle_attack_triggers(attacker_owner, defender, attacker)

        blocker = self._choose_blocker(defender, attacker)
        if blocker is not None:
            target_type = "creature"
            target_creature = blocker

        if attacker not in attacker_owner.battle_zone:
            self.output("攻撃は中止されました（攻撃クリーチャーがバトルゾーンを離れました）。")
            return

        if target_type == "creature":
            if target_creature is None or target_creature not in defender.battle_zone:
                self.output("攻撃は中止されました（対象のクリーチャーが不在です）。")
                return

            self._resolve_battle(attacker_owner, defender, attacker, target_creature)
        else:
            if target_type == "opponent":
                if defender.shields:
                    self._handle_shield_break(attacker_owner, defender, attacker)
                else:
                    self._log_direct_attack(attacker_owner, attacker)
                    self._set_game_over(attacker_owner)

        # Placeholder for "攻撃の終わり" triggers.
        # Placeholder for "攻撃の後" triggers.

        self._update_board_window()

    def _handle_attack_triggers(
        self, attacker_owner: Player, defender: Player, attacker: Card
    ) -> None:
        effect = ATTACK_TRIGGER_EFFECTS.get(attacker.name)
        if effect is not None:
            effect(self, attacker_owner, defender, attacker)

    def _choose_blocker(self, defender: Player, attacker: Card) -> Optional[Card]:
        blockers = self.available_blockers(defender, attacker)

        if not blockers:
            return None

        agent = self._cpu_agent_for(defender)

        if agent is None and self._player_index(defender) == self.human_player_index:
            self.output(
                "ブロックする場合はブロッカーの番号を入力してください。しない場合は他のキーを押してください："
            )
            for idx, creature in enumerate(blockers):
                power = creature.power if creature.power is not None else "-"
                self.output(f"  {idx}: {creature.name}（パワー：{power}）")
            choice = self.input("ブロッカーの番号（0-9）を入力、他のキーでブロックしない：").strip()
            if choice not in SINGLE_DIGIT_CHOICES:
                return None
            index = int(choice)
            if index >= len(blockers):
                self.output("無効な選択です。ブロックは行われませんでした。")
                return None
            blocker = blockers[index]
            defender.tap_creature(blocker)
            self.output(f"{defender.name}は《{blocker.name}》でブロックしました。")
            self._update_board_window()
            self._log_block(defender, blocker)
            return blocker

        if agent is not None:
            selection = agent.choose_blocker(self, defender, blockers)
            if selection is None:
                return None
            if selection not in blockers:
                raise GameError("CPUが不正なブロッカーを選択しました。")
            blocker = selection
        else:
            blocker = blockers[0]

        defender.tap_creature(blocker)
        self.output(f"{defender.name}は《{blocker.name}》でブロックしました。")
        self._update_board_window()
        self._log_block(defender, blocker)
        return blocker

    def _resolve_battle(
        self,
        attacker_owner: Player,
        defender: Player,
        attacker: Card,
        defender_creature: Card,
    ) -> None:
        attacker_power = self._calculate_card_power(attacker_owner, attacker, is_attacking=True)
        defender_power = self._calculate_card_power(defender, defender_creature)

        if attacker_power > defender_power:
            self.output(
                f"{attacker.name}（パワー{attacker_power}）が{defender_creature.name}（パワー{defender_power}）に勝利しました。"
            )
            self._destroy_creature(defender, defender_creature)
            self._log_battle_result(attacker, defender_creature, "attacker")
        elif attacker_power < defender_power:
            self.output(
                f"{defender_creature.name}（パワー{defender_power}）が{attacker.name}（パワー{attacker_power}）に勝利しました。"
            )
            self._destroy_creature(attacker_owner, attacker)
            self._log_battle_result(attacker, defender_creature, "defender")
        else:
            self.output(
                f"{attacker.name}と{defender_creature.name}はパワー{attacker_power}で相討ちになりました。"
            )
            self._destroy_creature(defender, defender_creature)
            self._destroy_creature(attacker_owner, attacker)
            self._log_battle_result(attacker, defender_creature, "draw")

        if card_has_ability(attacker, "スレイヤー") and defender_creature in defender.battle_zone:
            self._destroy_creature(defender, defender_creature)
        if card_has_ability(defender_creature, "スレイヤー") and attacker in attacker_owner.battle_zone:
            self._destroy_creature(attacker_owner, attacker)
        for card, owner in ((attacker, attacker_owner), (defender_creature, defender)):
            effect = POST_BATTLE_EFFECTS.get(card.name)
            if effect is not None:
                effect(self, owner, card)

    def _handle_shield_break(
        self, attacker_owner: Player, defender: Player, attacker: Card
    ) -> None:
        if not defender.shields:
            self._update_board_window()
            return

        breaks = 2 if has_w_breaker(attacker) else 1
        to_break = min(breaks, len(defender.shields))
        attacker_is_human = self._player_index(attacker_owner) == self.human_player_index

        if to_break == 0:
            self._update_board_window()
            return

        shields_before_total = len(defender.shields)
        selected_indices: List[int] = []
        for _ in range(to_break):
            available = [i for i in range(len(defender.shields)) if i not in selected_indices]
            if not available:
                break

            if attacker_is_human:
                remaining = to_break - len(selected_indices)
                self.output(
                    f"ブレイクするシールドを選んでください（残り{remaining}枚、0～{len(defender.shields) - 1}）："
                )
                choice = self.input("シールドの番号（0-9）を入力：").strip()
                if choice in SINGLE_DIGIT_CHOICES:
                    idx = int(choice)
                    if idx in available:
                        selected_indices.append(idx)
                        continue
                    self.output("範囲外または既に選択済みの番号です。先頭の未選択シールドをブレイクします。")
                else:
                    self.output("無効な入力です。先頭の未選択シールドをブレイクします。")
                selected_indices.append(available[0])
            else:
                selected_indices.append(available[0])

        if not selected_indices:
            self._update_board_window()
            return

        chosen_entries = [(index, defender.shields[index]) for index in selected_indices]

        if len(chosen_entries) > 1:
            self.output("シールドが同時にブレイクされました。")
        else:
            self.output("シールドがブレイクされました。")

        for index, _ in sorted(chosen_entries, key=lambda pair: pair[0], reverse=True):
            defender.shields.pop(index)

        running_shields_before = shields_before_total
        broken_cards = [shield_card for _, shield_card in chosen_entries]
        trigger_positions = [
            idx for idx, card in enumerate(broken_cards) if has_s_trigger(card)
        ]
        trigger_decisions: Dict[int, bool] = {}
        trigger_order_positions: List[int] = []

        if trigger_positions:
            if self._player_index(defender) == self.human_player_index:
                eligible_options: List[Tuple[int, Card]] = []
                for pos in trigger_positions:
                    card = broken_cards[pos]
                    if self._can_use_shield_trigger(defender, card):
                        eligible_options.append((pos, card))
                    else:
                        trigger_decisions[pos] = False
                if eligible_options:
                    selection_order = self._prompt_shield_trigger_order(eligible_options)
                    for choice in selection_order:
                        pos, _ = eligible_options[choice]
                        trigger_decisions[pos] = True
                        trigger_order_positions.append(pos)
                    for pos, _ in eligible_options:
                        trigger_decisions.setdefault(pos, False)
            else:
                for pos in trigger_positions:
                    card = broken_cards[pos]
                    if not self._can_use_shield_trigger(defender, card):
                        trigger_decisions[pos] = False
                        continue
                    use_trigger = self._should_use_shield_trigger(defender, card)
                    trigger_decisions[pos] = use_trigger
                    if use_trigger:
                        trigger_order_positions.append(pos)

        for pos in trigger_positions:
            trigger_decisions.setdefault(pos, False)

        use_trigger_positions = [
            pos for pos in trigger_order_positions if trigger_decisions.get(pos, False)
        ]

        pending_trigger_info: Dict[int, Tuple[int, int]] = {}

        for idx, shield_card in enumerate(broken_cards):
            shields_before = running_shields_before
            shields_after = shields_before - 1
            if idx in use_trigger_positions:
                pending_trigger_info[idx] = (shields_before, shields_after)
            else:
                hand_before = len(defender.hand)
                self._move_shield_card_to_hand(defender, shield_card)
                hand_after = len(defender.hand)
                self._log_shield_break(
                    attacker_owner,
                    defender,
                    attacker,
                    shields_before,
                    shields_after,
                    hand_before,
                    hand_after,
                )
            running_shields_before = shields_after

        remaining_spell_triggers = sum(
            1
            for pos in use_trigger_positions
            if broken_cards[pos].is_spell()
        )

        for pos in use_trigger_positions:
            shield_card = broken_cards[pos]
            shields_before, shields_after = pending_trigger_info[pos]
            hand_before = len(defender.hand)
            used = self._resolve_shield_trigger(attacker_owner, defender, shield_card)
            if not used:
                self._move_shield_card_to_hand(defender, shield_card)
            hand_after = len(defender.hand)
            self._log_shield_break(
                attacker_owner,
                defender,
                attacker,
                shields_before,
                shields_after,
                hand_before,
                hand_after,
            )
            if used and shield_card.is_spell():
                remaining_spell_triggers = max(0, remaining_spell_triggers - 1)
                if remaining_spell_triggers == 0:
                    self._resolve_pending_creature_entries()

        self._resolve_pending_creature_entries()
        self._update_board_window()

    def _move_shield_card_to_hand(self, defender: Player, card: Card) -> None:
        defender._add_card_to_zone(card, Zone.HAND)
        if card not in defender.hand and card in defender.graveyard:
            owner_label = "自分" if self._is_human_player(defender) else "相手"
            self.output(
                f"{owner_label}の手札が上限のため《{card.name}》は墓地に置かれました。"
            )
            self._log_shield_overflow(defender, card)

    def _can_use_shield_trigger(self, defender: Player, card: Card) -> bool:
        if card.is_evolution():
            return bool(self._valid_evolution_bases(defender, card))
        return True
 
    def _should_use_shield_trigger(self, defender: Player, card: Card) -> bool:
        if not self._can_use_shield_trigger(defender, card):
            return False

        if self._player_index(defender) == self.human_player_index:
            response = self.input(f"《{card.name}》のシールドトリガーを使いますか？ (y/N)：").strip()
            return response.lower().startswith("y")

        if card.name in TARGETED_SPELL_NAMES:
            targets = self.auto_spell_targets(card, defender)
            if not targets:
                return False
        return True

    def _prompt_shield_trigger_order(
        self, options: Sequence[Tuple[int, Card]]
    ) -> List[int]:
        self.output("どのS・トリガーを使いますか？　使用したい順番で番号を入力してください。")
        for idx, (_, card) in enumerate(options):
            self.output(f"  {idx}: 《{card.name}》")
        response = self.input(
            "番号をカンマ区切りで入力（例：0,1）。入力しない場合は使用しません："
        ).strip()
        if not response:
            return []

        tokens = [token.strip() for token in response.split(",") if token.strip()]
        selections: List[int] = []
        seen: Set[int] = set()
        for token in tokens:
            if not token.isdigit():
                self.output("入力が無効なため、S・トリガーは使用しません。")
                return []
            index = int(token)
            if index < 0 or index >= len(options) or index in seen:
                self.output("入力が無効なため、S・トリガーは使用しません。")
                return []
            selections.append(index)
            seen.add(index)

        return selections

    def _resolve_shield_trigger(
        self, attacker_owner: Player, defender: Player, card: Card
    ) -> bool:
        if card.is_spell():
            if self._player_index(defender) == self.human_player_index:
                targets = self._gather_spell_targets(card, defender)
                if targets is None:
                    return False
            else:
                targets = self.auto_spell_targets(card, defender)
                if targets is None:
                    targets = ()
            self._log_spell_cast(defender, card)
            self._log_spell_resolution(defender, card)
            self._resolve_spell(card, defender, tuple(targets))
            defender.graveyard.append(card)
            self._update_board_window()
            return True

        if card.is_creature():
            if card.is_evolution():
                bases = self._valid_evolution_bases(defender, card)
                if not bases:
                    return False
                base = bases[0]
                self._apply_evolution(defender, card, base)
            else:
                defender._add_card_to_zone(card, Zone.BATTLE)
            self._handle_creature_entry(defender, card, defer_triggers=True)
            self._log_summon(defender, card)
            self._update_board_window()
            if card not in defender.battle_zone and card not in defender.graveyard:
                defender.graveyard.append(card)
            return True

        return False

    def _end_step(self, player: Player) -> None:
        self._set_step(TurnStep.END)
        # Trigger handling would be inserted here.
        for creature in list(player.battle_zone):
            effect = END_STEP_EFFECTS.get(creature.name)
            if effect is not None:
                effect(self, player, creature)

    def _resolve_spell(
        self, card: Card, caster: Player, targets: Sequence[Card]
    ) -> None:
        opponent = self.non_turn_player if caster is self.turn_player else self.turn_player
        effect = SPELL_EFFECTS.get(card.name)
        if effect is None:
            return

        effect(self, caster, opponent, targets)

    def _move_card_with_evolution(
        self, player: Player, card: Card, from_zone: Zone, to_zone: Zone
    ) -> None:
        player._remove_card_from_zone(card, from_zone)
        if to_zone is Zone.GRAVEYARD:
            player.graveyard.append(card)
        else:
            player._add_card_to_zone(card, to_zone)

        self.turn_power_modifiers.pop(card, None)
        for base in player.clear_evolution_sources(card):
            self.turn_power_modifiers.pop(base, None)
            if to_zone is Zone.GRAVEYARD:
                player.graveyard.append(base)
            else:
                player._add_card_to_zone(base, to_zone)

    def _static_power_bonus(self, owner: Player, card: Card) -> int:
        bonus = 0
        for calculator in STATIC_POWER_BONUS_CALCULATORS:
            bonus += calculator(self, owner, card)
        return bonus

    def _calculate_card_power(
        self, owner: Player, card: Card, *, is_attacking: bool = False
    ) -> int:
        power = card.power or 0
        power += self.turn_power_modifiers.get(card, 0)
        power += self._static_power_bonus(owner, card)

        if is_attacking:
            bonus = extract_power_attacker_bonus(card)
            power += bonus
            for handler in ATTACK_POWER_BONUS_HANDLERS:
                power += handler(self, owner, card)

        return max(0, power)

    def _destroy_creature(self, owner: Player, creature: Card, cause: str = "battle") -> None:
        if card_has_ability(creature, "破壊される時、かわりに手札に戻す"):
            self._move_card_with_evolution(owner, creature, Zone.BATTLE, Zone.HAND)
            if creature in owner.hand:
                self.log_creature_effect_detail(owner, f"《{creature.name}》を手札に戻した")
            else:
                self.log_creature_effect_detail(
                    owner,
                    f"《{creature.name}》は手札がいっぱいのため墓地へ",
                )
            return

        self._move_card_with_evolution(owner, creature, Zone.BATTLE, Zone.GRAVEYARD)

    # ------------------------------------------------------------------
    # Evolution helpers
    # ------------------------------------------------------------------
    def _valid_evolution_bases(self, player: Player, evolution: Card) -> List[Card]:
        requirement = evolution.evolution_requirement
        if not requirement:
            return []

        valid: List[Card] = []
        for candidate in player.battle_zone:
            if not candidate.is_creature():
                continue
            if requirement in candidate.races:
                valid.append(candidate)
        return valid

    def _select_evolution_base(
        self, player: Player, evolution: Card
    ) -> Optional[Card]:
        valid_bases = self._valid_evolution_bases(player, evolution)
        if not valid_bases:
            return None

        agent = self._cpu_agent_for(player)
        if agent is not None:
            return valid_bases[0]

        if self._player_index(player) != self.human_player_index:
            return valid_bases[0]

        self.output(
            f"{evolution.name}に進化させる進化元を選んでください。"
        )
        for idx, creature in enumerate(valid_bases):
            power = creature.power if creature.power is not None else "-"
            state = "タップ" if player.is_creature_tapped(creature) else "アンタップ"
            self.output(f"  {idx}: {creature.name}（パワー：{power}、{state}）")
        choice = self.input("進化元の番号（0-9）を入力、他のキーでキャンセル：").strip()
        if choice not in SINGLE_DIGIT_CHOICES:
            return None
        index = int(choice)
        if index >= len(valid_bases):
            self.output("無効な選択です。進化は行われません。")
            return None
        return valid_bases[index]

    def _apply_evolution(
        self, player: Player, evolution: Card, base_creature: Card
    ) -> None:
        was_tapped = player.is_creature_tapped(base_creature)
        player._remove_card_from_zone(base_creature, Zone.BATTLE)
        player._add_card_to_zone(evolution, Zone.BATTLE)
        player.set_evolution_sources(evolution, [base_creature])
        player.battle_zone_tapped[evolution] = was_tapped
        if base_creature in self.turn_power_modifiers:
            modifier = self.turn_power_modifiers.pop(base_creature)
            self.turn_power_modifiers[evolution] = (
                self.turn_power_modifiers.get(evolution, 0) + modifier
            )

    # ------------------------------------------------------------------
    # Game end handling
    # ------------------------------------------------------------------
    def can_play_card(self, player: Player, card: Card) -> bool:
        try:
            self._validate_card_play(player, card)
        except GameError:
            return False
        return True

    def _validate_card_play(self, player: Player, card: Card) -> None:
        if player is not self.turn_player:
            raise GameError("カードをプレイできるのはターンプレイヤーだけです。")
        if self.current_step is not TurnStep.MAIN:
            raise GameError("カードをプレイできるのはメインステップ中のみです。")
        if card not in player.hand:
            raise GameError("カードは手札にある必要があります。")

        if card.cost is None:
            raise GameError("コストが未定義のカードはプレイできません。")
        if card.cost == float("inf"):
            raise GameError("コスト∞のカードはプレイできません。")
        if player.available_mana < card.cost:
            raise GameError("使用可能マナが不足しています。")
        if not player.has_required_civilizations(card.civilizations):
            raise GameError("必要な文明が解放されていません。")
        if card.is_creature():
            if card.is_evolution():
                if not self._valid_evolution_bases(player, card):
                    raise GameError("進化元となるクリーチャーが存在しません。")
            elif player.available_creature_slots() <= 0:
                raise GameError("バトルゾーンに空きがありません。")

    def _player_can_charge_mana(self, player: Player) -> bool:
        if self.mana_charged_this_turn:
            return False
        return bool(player.hand)

    def available_mana_charge_cards(self, player: Player) -> List[Card]:
        if not self._player_can_charge_mana(player):
            return []
        return list(player.hand)

    def _player_has_playable_card(self, player: Player) -> bool:
        for candidate in player.hand:
            if self.can_play_card(player, candidate):
                return True
        return False

    def playable_cards(self, player: Player) -> List[Card]:
        return [card for card in player.hand if self.can_play_card(player, card)]

    def _gather_spell_targets(
        self, card: Card, caster: Player
    ) -> Optional[Sequence[Card]]:
        prompt = SPELL_TARGET_PROMPTS.get(card.name)
        if prompt is None:
            return ()
        return prompt(self, caster)

    def auto_spell_targets(self, card: Card, caster: Player) -> Optional[Sequence[Card]]:
        auto = SPELL_AUTO_TARGETS.get(card.name)
        if auto is None:
            return ()
        return auto(self, caster)

    def prompt_power_limited_target(
        self, caster: Player, power_limit: int
    ) -> Sequence[Card]:
        opponent = self.non_turn_player if caster is self.turn_player else self.turn_player
        valid_targets = [
            (idx, creature)
            for idx, creature in enumerate(opponent.battle_zone)
            if creature.is_creature()
            and self._calculate_card_power(opponent, creature) <= power_limit
        ]

        if not valid_targets:
            self.output("条件を満たすクリーチャーがいません。")
            return ()

        self.output(f"パワー{power_limit}以下の対象：")
        for idx, creature in valid_targets:
            power = self._calculate_card_power(opponent, creature)
            self.output(f"  {idx}: {creature.name}（パワー：{power}）")
        choice = self.input("対象の番号（0-9）を入力、他のキーでキャンセル：").strip()
        if choice not in SINGLE_DIGIT_CHOICES:
            return ()
        index = int(choice)
        for original_index, creature in valid_targets:
            if original_index == index:
                return (creature,)
        self.output("不正な選択です。対象は選ばれませんでした。")
        return ()

    def _display_hand(self, player: Player) -> None:
        self.output("手札：")
        for index, card in enumerate(player.hand):
            self.output(
                f"  {index}: {card.name}（コスト：{card.cost}、マナ数：{card.mana_number}）"
            )

    def _handle_deck_out(self, player: Player) -> None:
        winner = self.non_turn_player if player is self.turn_player else self.turn_player
        self._set_game_over(winner)

    def _determine_next_player(self, player: Player, opponent: Player) -> None:
        if self.game_over:
            return

        if self.extra_turns[self._player_index(player)] > 0:
            # The current player will take their queued extra turn(s).
            self.turn_player_index = self._player_index(player)
            return

        if self.extra_turns[self._player_index(opponent)] > 0:
            # Opponent takes all of their queued extra turns before their normal one.
            self.turn_player_index = self._player_index(opponent)
            return

        # Normal alternating turn order.
        self.turn_player_index = self._player_index(opponent)

    def _set_game_over(self, winner: Optional[Player]) -> None:
        self.game_over = True
        self.winner = winner
        self._announce_result()

    def _announce_result(self) -> None:
        if self.result_announced:
            return

        message = self._result_message()
        self.output(message)
        self._log_line(message)
        self.result_announced = True
        self._update_board_window()

    def _result_message(self) -> str:
        if self.winner is None:
            return "引き分けです。"

        try:
            winner_index = self._player_index(self.winner)
        except GameError:
            winner_index = None

        if winner_index is not None and winner_index == self.human_player_index:
            return "あなたの勝利です"
        return "あなたの敗北です"

    def run_until_game_over(self) -> None:
        while not self.game_over:
            self.run_turn()

        if not self.result_announced:
            self._announce_result()


def _build_default_list() -> List[Card]:
    deck: List[Card] = []
    deck.extend(create_card_by_name("晴天の守護者レゾ・パコス") for _ in range(8))
    deck.extend(create_card_by_name("雲霧の守護者ク・ラウド") for _ in range(8))
    deck.extend(create_card_by_name("アクア・ビークル") for _ in range(8))
    deck.extend(create_card_by_name("腐卵虫ハングワーム") for _ in range(8))
    deck.extend(create_card_by_name("臓裂虫テンタイク・ワーム") for _ in range(8))
    return deck


def build_my_deck() -> List[Card]:
    """Create the default deck configuration for the human player."""

    return _build_default_list()


def build_enemy_deck() -> List[Card]:
    """Create the default deck configuration for the CPU opponent."""

    return _build_default_list()


def _resolve_starting_player_index(game: Game, option: StartingPlayerOption) -> int:
    """Return the index of the player who should take the first turn."""

    total_players = len(game.players)
    if total_players == 0:
        raise GameError("対戦参加者が存在しません。")

    if option == "random":
        return random.randrange(total_players)

    human_index = game.human_player_index

    if option == "human_first":
        if human_index is not None:
            return min(human_index, total_players - 1)
        return 0

    if option == "human_second":
        if total_players <= 1:
            return 0 if human_index is None else human_index
        if human_index is None:
            return 1 % total_players
        for idx in range(total_players):
            if idx != human_index:
                return idx
        return human_index

    raise ValueError(f"未対応の先攻指定です：{option}")


def run_cli_game(
    player1: Player,
    player2: Player,
    *,
    cpu_agent: Optional["CpuAgent"] = None,
    starting_player: StartingPlayerOption = "human_first",
) -> None:
    """Run the interactive CLI loop for the provided players.

    This helper centralises the common startup/teardown flow so that external
    scripts can customise deck construction while reusing the same runtime
    environment provided by :mod:`emulator`.
    """

    board_window = BoardWindow()
    if not board_window.enabled:
        print("盤面ウインドウを初期化できませんでした（tkinterが利用できません）。")

    game = Game(
        player1,
        player2,
        board_window=board_window if board_window.enabled else None,
        cpu_agent=cpu_agent,
    )
    game.output("デュエル・マスターズ プレイス エミュレータを開始します…")

    first_player_index = _resolve_starting_player_index(game, starting_player)
    game.turn_player_index = first_player_index
    starting_player_name = game.players[first_player_index].name
    message = f"先攻は{starting_player_name}です。"
    game.output(message)
    game._log_line(message)

    try:
        game.start_game()

        if not game.game_over:
            game.run_until_game_over()
    finally:
        if board_window.enabled:
            try:
                input("終了するにはEnterキーを押してください…")
            except EOFError:
                pass
        board_window.close()


def main() -> None:
    """Run a basic command-line game loop until the game ends."""

    from cpu_ai import SimpleCpuAgent

    player1 = Player(name="自分", deck=build_my_deck())
    player2 = Player(name="相手", deck=build_enemy_deck())

    run_cli_game(player1, player2, cpu_agent=SimpleCpuAgent())


if __name__ == "__main__":
    main()


__all__ = [
    "create_card_by_id",
    "create_card_by_name",
    "build_my_deck",
    "build_enemy_deck",
    "run_cli_game",
    "StartingPlayerOption",
    "BoardWindow",
    "Card",
    "DeckOut",
    "Game",
    "GameError",
    "Player",
    "TurnStep",
    "Zone",
]

