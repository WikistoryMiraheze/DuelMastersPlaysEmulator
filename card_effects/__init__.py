"""Registries for card-specific behaviour implementations."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from emulator import Card, Game, Player

SpellEffect = Callable[["Game", "Player", "Player", Sequence["Card"]], None]
SpellTargetPrompt = Callable[["Game", "Player"], Optional[Sequence["Card"]]]
SpellAutoTargets = Callable[["Game", "Player"], Optional[Sequence["Card"]]]
EnterBattleZoneEffect = Callable[["Game", "Player", "Card"], None]
AttackTriggerEffect = Callable[["Game", "Player", "Player", "Card"], None]
EndStepEffect = Callable[["Game", "Player", "Card"], None]
BlockerFilter = Callable[["Game", "Player", "Card", Sequence["Card"]], Sequence["Card"]]
AttackPowerBonus = Callable[["Game", "Player", "Card"], int]
StaticPowerBonusCalculator = Callable[["Game", "Player", "Card"], int]
PostBattleEffect = Callable[["Game", "Player", "Card"], None]

from . import (
    aqua_hulcus,
    bloody_earring,
    bolshack_dragon,
    death_smoke,
    fairy_life,
    gaia_mantis,
    genryuho,
    kishiaotoko,
    madou_scrum,
    magnum_blues,
    maguris,
    moonlight_flash,
    neo_brain,
    phantom_bites,
    seidou_no_yoroi,
    silver_axe,
    solar_ray,
    spiral_slider,
    tama_egg_worm,
    togesashi_mandora,
    tornado_frame,
    urusu,
    valdios,
    walta,
    yogensha_koron,
)

# Spells -----------------------------------------------------------------
SPELL_EFFECTS = {
    moonlight_flash.CARD_NAME: moonlight_flash.spell_effect,
    death_smoke.CARD_NAME: death_smoke.spell_effect,
    solar_ray.CARD_NAME: solar_ray.spell_effect,
    spiral_slider.CARD_NAME: spiral_slider.spell_effect,
    neo_brain.CARD_NAME: neo_brain.spell_effect,
    phantom_bites.CARD_NAME: phantom_bites.spell_effect,
    genryuho.CARD_NAME: genryuho.spell_effect,
    tornado_frame.CARD_NAME: tornado_frame.spell_effect,
    fairy_life.CARD_NAME: fairy_life.spell_effect,
    madou_scrum.CARD_NAME: madou_scrum.spell_effect,
}

SPELL_TARGET_PROMPTS = {
    moonlight_flash.CARD_NAME: moonlight_flash.prompt_targets,
    death_smoke.CARD_NAME: death_smoke.prompt_targets,
    solar_ray.CARD_NAME: solar_ray.prompt_targets,
    spiral_slider.CARD_NAME: spiral_slider.prompt_targets,
    phantom_bites.CARD_NAME: phantom_bites.prompt_targets,
    genryuho.CARD_NAME: genryuho.prompt_targets,
    tornado_frame.CARD_NAME: tornado_frame.prompt_targets,
    madou_scrum.CARD_NAME: madou_scrum.prompt_targets,
}

SPELL_AUTO_TARGETS = {
    moonlight_flash.CARD_NAME: moonlight_flash.auto_targets,
    death_smoke.CARD_NAME: death_smoke.auto_targets,
    solar_ray.CARD_NAME: solar_ray.auto_targets,
    spiral_slider.CARD_NAME: spiral_slider.auto_targets,
    phantom_bites.CARD_NAME: phantom_bites.auto_targets,
    genryuho.CARD_NAME: genryuho.auto_targets,
    tornado_frame.CARD_NAME: tornado_frame.auto_targets,
    madou_scrum.CARD_NAME: madou_scrum.auto_targets,
}

TARGETED_SPELL_NAMES = {
    name
    for name, module in (
        (moonlight_flash.CARD_NAME, moonlight_flash),
        (death_smoke.CARD_NAME, death_smoke),
        (solar_ray.CARD_NAME, solar_ray),
        (spiral_slider.CARD_NAME, spiral_slider),
        (phantom_bites.CARD_NAME, phantom_bites),
        (genryuho.CARD_NAME, genryuho),
        (tornado_frame.CARD_NAME, tornado_frame),
        (madou_scrum.CARD_NAME, madou_scrum),
    )
    if getattr(module, "TARGETS_REQUIRED", True)
}

# Creatures ---------------------------------------------------------------
ENTER_BATTLE_ZONE_EFFECTS = {
    seidou_no_yoroi.CARD_NAME: seidou_no_yoroi.enter_battle_zone,
    maguris.CARD_NAME: maguris.enter_battle_zone,
    aqua_hulcus.CARD_NAME: aqua_hulcus.enter_battle_zone,
    yogensha_koron.CARD_NAME: yogensha_koron.enter_battle_zone,
    togesashi_mandora.CARD_NAME: togesashi_mandora.enter_battle_zone,
    kishiaotoko.CARD_NAME: kishiaotoko.enter_battle_zone,
}

ATTACK_TRIGGER_EFFECTS = {
    walta.CARD_NAME: walta.on_attack,
    silver_axe.CARD_NAME: silver_axe.on_attack,
    tama_egg_worm.CARD_NAME: tama_egg_worm.on_attack,
}

END_STEP_EFFECTS = {
    urusu.CARD_NAME: urusu.on_end_step,
}

BLOCKER_FILTERS = {
    gaia_mantis.CARD_NAME: gaia_mantis.filter_blockers,
}

ATTACK_POWER_BONUS_HANDLERS = [
    magnum_blues.attack_bonus,
    bolshack_dragon.attack_bonus,
]

STATIC_POWER_BONUS_CALCULATORS = [
    valdios.static_bonus,
]

POST_BATTLE_EFFECTS = {
    bloody_earring.CARD_NAME: bloody_earring.after_battle,
}

__all__ = [
    "SpellEffect",
    "SpellTargetPrompt",
    "SpellAutoTargets",
    "EnterBattleZoneEffect",
    "AttackTriggerEffect",
    "EndStepEffect",
    "BlockerFilter",
    "AttackPowerBonus",
    "StaticPowerBonusCalculator",
    "PostBattleEffect",
    "SPELL_EFFECTS",
    "SPELL_TARGET_PROMPTS",
    "SPELL_AUTO_TARGETS",
    "TARGETED_SPELL_NAMES",
    "ENTER_BATTLE_ZONE_EFFECTS",
    "ATTACK_TRIGGER_EFFECTS",
    "END_STEP_EFFECTS",
    "BLOCKER_FILTERS",
    "ATTACK_POWER_BONUS_HANDLERS",
    "STATIC_POWER_BONUS_CALCULATORS",
    "POST_BATTLE_EFFECTS",
]
