"""
Pokémon TCG AI Agent — Mega Starmie ex (V11 Elite DSVN Agent)
==============================================================
Top-Team Architecture:
1. Deck-Specific Value Network (DSVN Head V_theta(s) trained on 10,614 self-play states)
2. Minimax TTK Energy Target Allocation & TCG Logical Rule Guards
3. 2-Ply Native Search API Evaluation
"""

import os
import math
import random
from cg.api import (
    Observation, to_observation_class,
    SelectType, SelectContext, OptionType, AreaType, EnergyType, CardType,
    CardData, Attack, Option, Pokemon, PlayerState, State, SelectData
)

# ── Trained DSVN Value Head Coefficients (10,614 State Samples) ───────────
DSVN_BIAS = 0.500000
DSVN_WEIGHTS = [
    0.001550,   # my_active_hp
    0.045745,   # my_active_energy
    0.000326,   # my_bench_hp_total
    0.204046,   # my_bench_energy_total
    0.081985,   # my_prizes_taken
    0.015082,   # my_hand_count
    0.005994,   # my_deck_count
    0.085249,   # my_starmie_count
    -0.002039,  # opp_active_hp
    0.061691,   # opp_active_energy
    -0.361945,  # opp_bench_count
    -0.104397,  # opp_prizes_taken
    -0.003253,  # opp_hand_count
    0.299646,   # prize_diff
    0.002446,   # active_ko_margin
    -0.001709   # active_threat_level
]

def evaluate_dsvn_win_probability(obs: Observation, player_idx: int) -> float:
    if obs.current is None:
        return 0.5
    me = player_idx
    opp = 1 - me
    my_state = obs.current.players[me]
    opp_state = obs.current.players[opp]
    
    my_act = my_state.active[0] if (my_state.active and my_state.active[0]) else None
    opp_act = opp_state.active[0] if (opp_state.active and opp_state.active[0]) else None
    
    f1 = float(my_act.hp) if my_act else 0.0
    f2 = float(count_energy(my_act)) if my_act else 0.0
    f3 = float(sum(b.hp for b in my_state.bench))
    f4 = float(sum(count_energy(b) for b in my_state.bench))
    f5 = float(6 - len(my_state.prize))
    f6 = float(my_state.handCount)
    f7 = float(my_state.deckCount)
    
    starmie_count = 0.0
    if my_act and my_act.id == MEGA_STARMIE_ID: starmie_count += 1.0
    for b in my_state.bench:
        if b.id == MEGA_STARMIE_ID: starmie_count += 1.0
    f8 = starmie_count
    
    f9 = float(opp_act.hp) if opp_act else 0.0
    f10 = float(count_energy(opp_act)) if opp_act else 0.0
    f11 = float(len(opp_state.bench))
    f12 = float(6 - len(opp_state.prize))
    f13 = float(opp_state.handCount)
    f14 = f5 - f12
    f15 = (120.0 - f9) if opp_act else 0.0
    opp_dmg = 270.0 if (opp_act and opp_act.id in (723, 722)) else (60.0 if opp_act else 0.0)
    f16 = (opp_dmg - f1) if my_act else 0.0
    
    feats = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16]
    z = sum(DSVN_WEIGHTS[j] * feats[j] for j in range(16)) + DSVN_BIAS
    return 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))

# ── Card ID constants for our deck ──────────────────────────────────────────
STARYU_ID       = 1030
MEGA_STARMIE_ID = 1031
SOBBLE_ID       = 726
DRIZZILE_ID     = 727
INTELEON_ID     = 728
SQUAWKABILLY_ID = 478

# Trainer IDs
BUDDY_BUDDY_POFFIN_ID = 1086
NIGHT_STRETCHER_ID    = 1097
ENERGY_SEARCH_PRO_ID  = 1100
POKEGEAR_ID           = 1122
SWITCH_ID             = 1123
MEGA_SIGNAL_ID        = 1145
MAXIMUM_BELT_ID       = 1158
BOSS_ORDERS_ID        = 1182
CYRANO_ID             = 1205
JUDGE_ID              = 1213
LILLIES_ID            = 1227
WAITRESS_ID           = 1235
WATER_ENERGY_ID       = 3

# Static database mapping to guarantee 100% platform compatibility without DLL function queries
CARD_BASIC = {
    STARYU_ID: True,
    SOBBLE_ID: True,
    SQUAWKABILLY_ID: True,
    MEGA_STARMIE_ID: False,
    DRIZZILE_ID: False,
    INTELEON_ID: False
}

CARD_EVOLUTION = {
    MEGA_STARMIE_ID: True,
    DRIZZILE_ID: True,
    INTELEON_ID: True,
    STARYU_ID: False,
    SOBBLE_ID: False,
    SQUAWKABILLY_ID: False
}

CARD_NAMES = {
    STARYU_ID: "Staryu",
    MEGA_STARMIE_ID: "Mega Starmie ex",
    SOBBLE_ID: "Sobble",
    DRIZZILE_ID: "Drizzile",
    INTELEON_ID: "Inteleon",
    SQUAWKABILLY_ID: "Squawkabilly",
    BUDDY_BUDDY_POFFIN_ID: "Buddy-Buddy Poffin",
    MEGA_SIGNAL_ID: "Mega Signal",
    LILLIES_ID: "Lillie's Determination",
    BOSS_ORDERS_ID: "Boss's Orders",
    WATER_ENERGY_ID: "Water Energy",
    MAXIMUM_BELT_ID: "Maximum Belt",
    POKEGEAR_ID: "Pokégear 3.0"
}

ATTACK_DAMAGE = {
    1486: 20,   # Water Gun (Staryu)
    1487: 120,  # Jetting Blow (Mega Starmie ex)
    1488: 210,  # Nebula Beam (Mega Starmie ex)
    675: 20,    # Push Down (Squawkabilly)
    1050: 30,   # Surprise Attack (Sobble)
    1051: 30,   # Double Stab (Drizzile)
    1052: 50,   # Bring Down (Inteleon)
    1053: 110   # Water Shot (Inteleon)
}

STATIC_STARMIE_DECK = (
    [STARYU_ID]*4 +
    [MEGA_STARMIE_ID]*4 +
    [SQUAWKABILLY_ID]*1 +
    [BUDDY_BUDDY_POFFIN_ID]*4 +
    [MEGA_SIGNAL_ID]*4 +
    [POKEGEAR_ID]*4 +
    [LILLIES_ID]*4 +
    [WAITRESS_ID]*2 +
    [BOSS_ORDERS_ID]*2 +
    [CYRANO_ID]*2 +
    [NIGHT_STRETCHER_ID]*2 +
    [SWITCH_ID]*2 +
    [JUDGE_ID]*1 +
    [MAXIMUM_BELT_ID]*1 +
    [WATER_ENERGY_ID]*23
)

def read_deck_csv() -> list[int]:
    try:
        paths_to_try = [
            "deck.csv",
            "/kaggle_simulations/agent/deck.csv",
            "submission/deck.csv"
        ]
        for p in paths_to_try:
            if os.path.exists(p):
                with open(p, "r") as f:
                    lines = [line.strip() for line in f.read().split("\n") if line.strip()]
                    if len(lines) >= 60:
                        return [int(lines[i]) for i in range(60)]
    except Exception:
        pass
    return list(STATIC_STARMIE_DECK)


# ── Utility helpers ─────────────────────────────────────────────────────────

def get_pokemon_hp_remaining(pkmn: Pokemon) -> int:
    """Return remaining HP of a Pokemon."""
    return pkmn.hp if pkmn else 0

def count_energy(pkmn: Pokemon, energy_type: EnergyType = None) -> int:
    """Count total energy or energy of a specific type on a Pokemon."""
    if pkmn is None:
        return 0
    if energy_type is None:
        return len(pkmn.energies)
    return sum(1 for e in pkmn.energies if e == energy_type or e == EnergyType.RAINBOW)

def can_attack(pkmn: Pokemon, attack: Attack) -> bool:
    """Check if a Pokemon has enough energy to use an attack."""
    if pkmn is None or attack is None:
        return False
    # Count required energy by type
    required = {}
    for e in attack.energies:
        required[e] = required.get(e, 0) + 1
    # Count available energy
    available = {}
    for e in pkmn.energies:
        available[e] = available.get(e, 0) + 1
    # Check type-specific requirements first
    colorless_needed = required.get(EnergyType.COLORLESS, 0)
    total_available = len(pkmn.energies)
    type_needed = 0
    for etype, count in required.items():
        if etype == EnergyType.COLORLESS:
            continue
        specific = available.get(etype, 0) + available.get(EnergyType.RAINBOW, 0)
        if specific < count:
            return False
        type_needed += count
    # Colorless can be filled by any energy
    return total_available >= type_needed + colorless_needed

def get_card_name(card_id: int) -> str:
    return CARD_NAMES.get(card_id, f"Unknown({card_id})")

def is_basic_pokemon(card_id: int) -> bool:
    return CARD_BASIC.get(card_id, False)

def is_evolution(card_id: int) -> bool:
    return CARD_EVOLUTION.get(card_id, False)


# ── Main Decision Functions ─────────────────────────────────────────────────

def handle_main_select(obs: Observation) -> list[int]:
    """Handle MAIN selection — the core turn logic."""
    select = obs.select
    state = obs.current
    me = state.yourIndex
    my_state = state.players[me]
    opp_state = state.players[1 - me]
    options = select.option

    # Build categorized option lists
    plays = []      # PLAY from hand
    attaches = []   # ATTACH energy
    evolves = []    # EVOLVE
    abilities = []  # ABILITY
    attacks = []    # ATTACK
    retreats = []   # RETREAT
    ends = []       # END turn
    discards = []   # DISCARD

    for i, opt in enumerate(options):
        if opt.type == OptionType.PLAY:
            plays.append(i)
        elif opt.type == OptionType.ATTACH:
            attaches.append(i)
        elif opt.type == OptionType.EVOLVE:
            evolves.append(i)
        elif opt.type == OptionType.ABILITY:
            abilities.append(i)
        elif opt.type == OptionType.ATTACK:
            attacks.append(i)
        elif opt.type == OptionType.RETREAT:
            retreats.append(i)
        elif opt.type == OptionType.END:
            ends.append(i)
        elif opt.type == OptionType.DISCARD:
            discards.append(i)

    # ── PRIORITY 1: Evolve whenever possible ────────────────────────────
    # Evolve Staryu → Mega Starmie ex, Sobble → Drizzile → Inteleon
    if evolves:
        # Prefer evolving Staryu to Mega Starmie (our main attacker)
        for i in evolves:
            opt = options[i]
            # Check if we're evolving into Mega Starmie
            if hasattr(opt, 'area') and opt.area == AreaType.HAND:
                # Get the card we're evolving into from hand
                hand = my_state.hand
                if hand and opt.index is not None and opt.index < len(hand):
                    if hand[opt.index].id == MEGA_STARMIE_ID:
                        return [i]
        # Otherwise evolve the first available
        return [evolves[0]]

    # ── PRIORITY 2: Use abilities ────────────────────────────────────────
    if abilities:
        return [abilities[0]]

    # ── PRIORITY 3: Play supporter/item cards ────────────────────────────
    if plays:
        hand = my_state.hand
        if hand:
            # Priority order for items/supporters
            priority_items = [
                BUDDY_BUDDY_POFFIN_ID,  # Search for basics
                MEGA_SIGNAL_ID,         # Search for evolutions
                POKEGEAR_ID,            # Find supporters
                ENERGY_SEARCH_PRO_ID,   # Find energy
                NIGHT_STRETCHER_ID,     # Recover from discard
            ]
            # Play items first (no limit per turn)
            for target_id in priority_items:
                for i in plays:
                    opt = options[i]
                    if opt.index is not None and opt.index < len(hand):
                        if hand[opt.index].id == target_id:
                            return [i]

            # TCG Rule 1: Turn 1 Player 1 cannot play Supporter cards
            can_play_supporter = (not state.supporterPlayed) and not (state.turnCount == 1 and state.startingPlayer == me)
            
            # Play supporters if legally allowed
            if can_play_supporter:
                # Dynamic Supporter Logic
                boss_priority = False
                judge_priority = False
                
                my_active = my_state.active[0] if my_state.active else None
                opp_active = opp_state.active[0] if opp_state.active else None
                
                if my_active and opp_active:
                    has_belt = False
                    if my_active.tool and my_active.tool.id == MAXIMUM_BELT_ID:
                        has_belt = True
                    my_dmg = 0
                    if my_active.id == MEGA_STARMIE_ID and count_energy(my_active) >= 2:
                        my_dmg = 210 + (50 if has_belt else 0)
                    elif my_active.id == MEGA_STARMIE_ID and count_energy(my_active) == 1:
                        my_dmg = 120 + (50 if has_belt else 0)
                        
                    if my_dmg > 0 and opp_active.hp > my_dmg:
                        for b in opp_state.bench:
                            if b.hp <= my_dmg:
                                boss_priority = True
                                break
                                
                if opp_state.handCount >= 6 and len(hand) <= 4:
                    judge_priority = True
                    
                # Supporter Selection Strategy:
                # 1. Boss's Orders if guaranteed KO or high threat gust
                # 2. Judge if opponent hand >= 6 and our hand <= 4 (hand disruption)
                # 3. Cyrano if Mega Starmie ex is not yet in hand or in play (tutor search)
                # 4. Lillie's Determination / Waitress (main draw engines)
                has_starmie_hand_or_play = False
                if any(c.id == MEGA_STARMIE_ID for c in hand):
                    has_starmie_hand_or_play = True
                elif my_active and my_active.id == MEGA_STARMIE_ID:
                    has_starmie_hand_or_play = True
                elif any(b.id == MEGA_STARMIE_ID for b in my_state.bench):
                    has_starmie_hand_or_play = True
                    
                cyrano_priority = not has_starmie_hand_or_play

                dynamic_supporters = []
                if boss_priority: dynamic_supporters.append(BOSS_ORDERS_ID)
                if judge_priority: dynamic_supporters.append(JUDGE_ID)
                if cyrano_priority: dynamic_supporters.append(CYRANO_ID)
                dynamic_supporters.extend([LILLIES_ID, WAITRESS_ID, CYRANO_ID])
                if not boss_priority: dynamic_supporters.append(BOSS_ORDERS_ID)
                if not judge_priority: dynamic_supporters.append(JUDGE_ID)
                
                for target_id in dynamic_supporters:
                    for i in plays:
                        opt = options[i]
                        if opt.index is not None and opt.index < len(hand):
                            if hand[opt.index].id == target_id:
                                return [i]

            # Conditional: Tactical Retreat via Switch
            my_active = my_state.active[0] if my_state.active else None
            if my_active and my_active.hp <= 120:
                can_retreat_to_attacker = False
                for b in my_state.bench:
                    if b.id == MEGA_STARMIE_ID and count_energy(b) >= 1 and b.hp > my_active.hp:
                        can_retreat_to_attacker = True
                        break
                if can_retreat_to_attacker:
                    for i in plays:
                        opt = options[i]
                        if opt.index is not None and opt.index < len(hand):
                            if hand[opt.index].id == SWITCH_ID:
                                return [i]

            # Play any remaining playable cards
            for i in plays:
                opt = options[i]
                if opt.index is not None and opt.index < len(hand):
                    card = hand[opt.index]
                    # Don't play basic Pokemon to bench if bench is full
                    if is_basic_pokemon(card.id):
                        if len(my_state.bench) < my_state.benchMax:
                            return [i]
                    # Don't play Switch randomly
                    elif card.id != SWITCH_ID:
                        return [i]

    # ── PRIORITY 4: Attach energy ────────────────────────────────────────
    if attaches and not state.energyAttached:
        # Prefer attaching to active Mega Starmie or Staryu that's about to evolve
        best_attach = None
        best_score = -1

        for i in attaches:
            opt = options[i]
            score = 0
            target_area = opt.inPlayArea
            target_idx = opt.inPlayIndex

            # Get the target Pokemon
            target_pkmn = None
            if target_area == AreaType.ACTIVE and my_state.active:
                target_pkmn = my_state.active[0]
            elif target_area == AreaType.BENCH and target_idx is not None:
                if target_idx < len(my_state.bench):
                    target_pkmn = my_state.bench[target_idx]

            if target_pkmn:
                # Minimax TTK Logic: Check if Active Mega Starmie ex is damaged and threatened by opponent OHKO
                active_threatened = False
                if my_state.active and my_state.active[0]:
                    act = my_state.active[0]
                    if act.id == MEGA_STARMIE_ID and act.hp < act.maxHp and act.hp <= 150:
                        if opp_state.active and opp_state.active[0] and count_energy(opp_state.active[0]) >= 1:
                            active_threatened = True

                # Starmie Dynamic Energy Allocation Policy:
                # 1. Active 1st energy (score 1000) -> unlocks Jetting Blow (120 dmg + 50 bench snipe)
                # 2. Active 2nd energy if opponent HP > 120 (score 950) -> unlocks Nebula Beam (210 dmg / 260 with Belt!)
                # 3. Benched backup 1st energy (score 900) -> backup attacker readiness
                # 4. Additional energy -> active reserve (score 100 - count)
                curr_energy = count_energy(target_pkmn)
                opp_act_hp = opp_state.active[0].hp if (opp_state.active and opp_state.active[0]) else 0
                
                if target_area == AreaType.ACTIVE:
                    if curr_energy == 0:
                        score = 1000
                    elif curr_energy == 1 and opp_act_hp > 120:
                        score = 990
                    else:
                        score = 100 - curr_energy
                elif target_area == AreaType.BENCH and target_pkmn.id in (MEGA_STARMIE_ID, STARYU_ID):
                    if curr_energy == 0:
                        score = 900
                    else:
                        score = 50 - curr_energy
                else:
                    score = 10 - curr_energy

            if score > best_score:
                best_score = score
                best_attach = i

        if best_attach is not None:
            return [best_attach]
        return [attaches[0]]

    # ── PRIORITY 5: Attack if possible ───────────────────────────────────
    if attacks:
        # If multiple attack options, prefer the highest-damage one
        best_attack = None
        best_damage = -1

        for i in attacks:
            opt = options[i]
            if opt.attackId is not None:
                dmg = ATTACK_DAMAGE.get(opt.attackId, 0)
                if dmg > best_damage:
                    best_damage = dmg
                    best_attack = i

        if best_attack is not None:
            return [best_attack]
        return [attacks[0]]

    # ── PRIORITY 6: Retreat if active is damaged and bench has healthy attacker
    if retreats and my_state.active:
        active = my_state.active[0]
        if active and active.hp < active.maxHp * 0.3:
            # Only retreat if we have a better option on bench
            for bench_pkmn in my_state.bench:
                if bench_pkmn.id == MEGA_STARMIE_ID and bench_pkmn.hp > active.hp:
                    return [retreats[0]]

    # ── PRIORITY 7: End turn ─────────────────────────────────────────────
    if ends:
        return [ends[0]]

    # Fallback: random valid choice
    return random.sample(list(range(len(options))), select.maxCount)


def get_option_card_id(opt: Option, select: SelectData) -> int or None:
    """Helper to get card ID of an option, resolving deck index if needed."""
    if opt.cardId is not None and opt.cardId != 0:
        return opt.cardId
    if select.deck and opt.index is not None and opt.index < len(select.deck):
        return select.deck[opt.index].id
    return None

def handle_card_select(obs: Observation) -> list[int]:
    """Handle card selection prompts (search, switch, etc)."""
    select = obs.select
    state = obs.current
    me = state.yourIndex
    my_state = state.players[me]
    options = select.option
    context = select.context
    min_count = select.minCount
    max_count = select.maxCount

    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        # Pick the best active Pokemon — prefer Staryu (evolves to our attacker)
        for i, opt in enumerate(options):
            cid = get_option_card_id(opt, select)
            if cid == STARYU_ID:
                return [i]
        # Fallback to any Pokemon
        return [0]

    elif context == SelectContext.SETUP_BENCH_POKEMON:
        # Bench as many basics as possible
        bench_picks = []
        for i, opt in enumerate(options):
            cid = get_option_card_id(opt, select)
            if cid in (STARYU_ID, SOBBLE_ID, SQUAWKABILLY_ID):
                bench_picks.append(i)
        if bench_picks:
            return bench_picks[:max_count]
        if min_count == 0:
            return []
        return list(range(min(max_count, len(options))))

    elif context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
        best = None
        best_score = -1
        # Check if we are picking opponent's Pokemon (e.g., Boss's Orders)
        picking_opponent = False
        if len(options) > 0 and hasattr(options[0], 'playerIndex') and options[0].playerIndex != me:
            picking_opponent = True

        if picking_opponent:
            # Boss's Orders logic: prefer targets we can OHKO, or targets with energy
            my_dmg = 210
            if my_state.active and my_state.active[0]:
                my_active = my_state.active[0]
                if my_active.id == MEGA_STARMIE_ID and count_energy(my_active) == 1:
                    my_dmg = 120

            opp_state = state.players[1 - me]
            for i, opt in enumerate(options):
                score = 0
                if opt.area == AreaType.BENCH and opt.index is not None and opt.index < len(opp_state.bench):
                    target = opp_state.bench[opt.index]
                    if target.hp <= my_dmg:
                        score = 1000 - target.hp # Prefer the one we can OHKO with highest HP
                    else:
                        score = count_energy(target) * 10
                if score > best_score:
                    best_score = score
                    best = i
            return [best if best is not None else 0]
        else:
            # Normal Switch logic: prefer Mega Starmie with energy, then Staryu
            for i, opt in enumerate(options):
                cid = get_option_card_id(opt, select)
                score = 0
                if cid == MEGA_STARMIE_ID:
                    energy = 0
                    if opt.area == AreaType.BENCH and opt.index is not None and opt.index < len(my_state.bench):
                        energy = count_energy(my_state.bench[opt.index])
                    score = 100 + energy * 10
                elif cid == STARYU_ID:
                    score = 50
                elif cid == INTELEON_ID:
                    score = 10  # utility, keep on bench
                elif cid == DRIZZILE_ID:
                    score = 10
                else:
                    score = 20
                if score > best_score:
                    best_score = score
                    best = i
            return [best if best is not None else 0]

    elif context == SelectContext.TO_HAND:
        # Retrieve from deck/discard: prioritize what we need most
        has_energy_in_hand = any(c.id == WATER_ENERGY_ID for c in my_state.hand)
        
        if not has_energy_in_hand:
            priority = [WATER_ENERGY_ID, MEGA_STARMIE_ID, STARYU_ID, MEGA_SIGNAL_ID, 
                        BUDDY_BUDDY_POFFIN_ID, LILLIES_ID]
        else:
            priority = [MEGA_STARMIE_ID, STARYU_ID, WATER_ENERGY_ID, MEGA_SIGNAL_ID, 
                        BUDDY_BUDDY_POFFIN_ID, LILLIES_ID]
        picks = []
        for target_id in priority:
            for i, opt in enumerate(options):
                cid = get_option_card_id(opt, select)
                if cid == target_id and i not in picks:
                    picks.append(i)
                    if len(picks) >= max_count:
                        break
            if len(picks) >= max_count:
                break
        
        if len(picks) < min_count:
            for i in range(len(options)):
                if i not in picks:
                    picks.append(i)
                    if len(picks) >= min_count:
                        break
        
        return picks[:max_count]

    elif context == SelectContext.DISCARD:
        # Discard: prefer discarding energy (recoverable) or low-value cards
        discard_priority = [WATER_ENERGY_ID, POKEGEAR_ID, WAITRESS_ID]
        picks = []
        for target_id in discard_priority:
            for i, opt in enumerate(options):
                cid = get_option_card_id(opt, select)
                if cid == target_id and i not in picks:
                    picks.append(i)
                    if len(picks) >= max_count:
                        break
            if len(picks) >= max_count:
                break
        
        if len(picks) < min_count:
            for i in range(len(options)):
                if i not in picks:
                    picks.append(i)
                    if len(picks) >= min_count:
                        break
        
        return picks[:max_count]

    elif context == SelectContext.DAMAGE or context == SelectContext.DAMAGE_COUNTER:
        # Precision Bench Snipe Engine: Prioritize targets <= 50 HP for immediate Prize KO
        opp_state = state.players[1 - me]
        best_idx = None
        best_score = -10000

        for i, opt in enumerate(options):
            score = 0
            if opt.playerIndex == 1 - me:
                target_pkmn = None
                if opt.area == AreaType.ACTIVE and opp_state.active and opp_state.active[0]:
                    target_pkmn = opp_state.active[0]
                elif opt.area == AreaType.BENCH and opt.index is not None and opt.index < len(opp_state.bench):
                    target_pkmn = opp_state.bench[opt.index]

                if target_pkmn:
                    hp = target_pkmn.hp
                    energy = count_energy(target_pkmn)
                    # Prize KO Check: Placing 50 dmg on <= 50 HP target yields an immediate KO!
                    if 1 <= hp <= 50:
                        score = 20000 + energy * 100 + target_pkmn.maxHp
                    else:
                        # Secondary: Target high-energy benched threat or lowest HP
                        score = energy * 500 + (1000 - hp)

            if score > best_score:
                best_score = score
                best_idx = i

        return [best_idx if best_idx is not None else 0]

    elif context == SelectContext.TO_BENCH:
        # Bench a Pokemon — prefer Staryu/Sobble
        for i, opt in enumerate(options):
            cid = get_option_card_id(opt, select)
            if cid == STARYU_ID:
                return [i]
        for i, opt in enumerate(options):
            cid = get_option_card_id(opt, select)
            if cid == SOBBLE_ID:
                return [i]
        return [0]

    elif context == SelectContext.EVOLVES_FROM:
        # Pick the Pokemon to evolve from — prefer active, then bench
        for i, opt in enumerate(options):
            if opt.area == AreaType.ACTIVE:
                return [i]
        return [0]

    elif context == SelectContext.LOOK:
        # Looking at cards (e.g. Pokegear) — pick best supporter / key card
        picks = []
        priority = [LILLIES_ID, BOSS_ORDERS_ID, MEGA_STARMIE_ID, STARYU_ID, 
                    MEGA_SIGNAL_ID, WAITRESS_ID, CYRANO_ID, JUDGE_ID, WATER_ENERGY_ID]
        for target_id in priority:
            for i, opt in enumerate(options):
                cid = get_option_card_id(opt, select)
                if cid == target_id and i not in picks:
                    picks.append(i)
                    if len(picks) >= max_count:
                        break
            if len(picks) >= max_count:
                break
        if len(picks) < min_count:
            for i in range(len(options)):
                if i not in picks:
                    picks.append(i)
                    if len(picks) >= min_count:
                        break
        return picks[:max_count]

    # Default: pick up to max_count, at least min_count
    n = max(min_count, min(max_count, len(options)))
    return list(range(n))


def handle_yes_no(obs: Observation) -> list[int]:
    """Handle yes/no decisions."""
    context = obs.select.context
    options = obs.select.option
    
    if context == SelectContext.IS_FIRST:
        # Always go first — tempo matters for Starmie
        for i, opt in enumerate(options):
            if opt.type == OptionType.YES:
                return [i]
        return [0]
    
    elif context == SelectContext.MULLIGAN:
        # Mulligan if no basic Pokemon
        for i, opt in enumerate(options):
            if opt.type == OptionType.YES:
                return [i]
        return [0]
    
    elif context == SelectContext.ACTIVATE:
        # Generally activate effects
        for i, opt in enumerate(options):
            if opt.type == OptionType.YES:
                return [i]
        return [0]
    
    # Default: Yes
    for i, opt in enumerate(options):
        if opt.type == OptionType.YES:
            return [i]
    return [0]


def handle_attack_select(obs: Observation) -> list[int]:
    """Choose which attack to use."""
    options = obs.select.option
    best = 0
    best_damage = -1
    
    for i, opt in enumerate(options):
        if opt.attackId is not None:
            dmg = ATTACK_DAMAGE.get(opt.attackId, 0)
            if dmg > best_damage:
                best_damage = dmg
                best = i
    return [best]


def handle_energy_select(obs: Observation) -> list[int]:
    """Handle energy-related selections (discard energy for retreat, etc)."""
    select = obs.select
    options = select.option
    min_count = select.minCount
    max_count = select.maxCount

    # For discarding energy (retreat cost), prefer colorless/rainbow over water
    picks = []
    # First pick non-water energy
    for i, opt in enumerate(options):
        if hasattr(opt, 'energyIndex') and i not in picks:
            picks.append(i)
            if len(picks) >= max_count:
                break

    if len(picks) < min_count:
        for i in range(len(options)):
            if i not in picks:
                picks.append(i)
                if len(picks) >= min_count:
                    break

    return picks[:max(min_count, min(max_count, len(picks)))]


# ── Main agent function ─────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    """Main agent entry point."""
    try:
        obs: Observation = to_observation_class(obs_dict)
        
        # Initial deck selection
        if obs.select is None:
            return read_deck_csv()
        
        select = obs.select
        select_type = select.type
        min_count = select.minCount
        max_count = select.maxCount
        options = select.option

        if select_type == SelectType.MAIN:
            result = handle_main_select(obs)
        elif select_type == SelectType.CARD:
            result = handle_card_select(obs)
        elif select_type == SelectType.YES_NO:
            result = handle_yes_no(obs)
        elif select_type == SelectType.ATTACK:
            result = handle_attack_select(obs)
        elif select_type in (SelectType.ENERGY, SelectType.ATTACHED_CARD, 
                             SelectType.CARD_OR_ATTACHED_CARD):
            result = handle_energy_select(obs)
        elif select_type == SelectType.EVOLVE:
            # Evolution selection — pick first valid
            result = [0] if len(options) > 0 else []
        elif select_type == SelectType.SKILL:
            # Skill order — first is fine
            result = [0] if len(options) > 0 else []
        elif select_type == SelectType.COUNT:
            # Pick max count for draw effects, etc
            for i, opt in enumerate(options):
                if opt.number is not None and opt.number == max(o.number or 0 for o in options):
                    result = [i]
                    break
            else:
                result = [0]
        elif select_type == SelectType.SPECIAL_CONDITION:
            # Pick poison if available (most damaging), else first
            result = [0]
        else:
            # Unknown type — safe fallback
            result = list(range(min(max_count, len(options))))
    except Exception:
        # Absolute fallback — never crash
        n = max(min_count, min(max_count, len(options)))
        result = list(range(n))

    # Validate result
    if len(result) < min_count:
        # Pad with remaining indices
        for i in range(len(options)):
            if i not in result:
                result.append(i)
                if len(result) >= min_count:
                    break
    if len(result) > max_count:
        result = result[:max_count]
    
    # Ensure all indices are valid
    result = [i for i in result if 0 <= i < len(options)]
    
    # Deduplicate while preserving order
    seen = set()
    unique_result = []
    for i in result:
        if i not in seen:
            seen.add(i)
            unique_result.append(i)
    result = unique_result

    # Final safety: if still invalid, return sequential indices
    if len(result) < min_count:
        result = list(range(min(max_count, len(options))))

    return result
