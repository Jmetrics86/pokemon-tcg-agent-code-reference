# Strategy & Learnings Report: 700+ ELO Milestone

In our evaluation on the Kaggle PTCG AI Battle simulation, two strategies successfully broke the 700 ELO barrier: **Treatment 05 (Fast Evolve)** at 740.3 ELO and **Treatment 04 (Energy Hoarder)** at 734.6 ELO.

To ensure we can build upon these successes tomorrow, here is a detailed breakdown of the exact mechanics that drove their performance compared to the baseline and failed treatments.

---

## 🏆 Top Performer: Treatment 05 - Fast Evolve (740.3 ELO)

**The Core Concept:**
The baseline agent heavily prioritizes playing basics and searching for basics (like Buddy-Buddy Poffin) in the early game. Treatment 05 shifts this priority strictly toward finding and evolving into Starmie ex as fast as possible.

**Code Changes from Baseline:**
```python
# 1. Force the agent to skip early basic checks and prioritize evolution pieces
- has_starmie_hand_or_play = False
+ has_starmie_hand_or_play = True

# 2. Modify item priority lists to favor high-tier search/evolution items before Poffins
- priority_items = [
-     BUDDY_BUDDY_POFFIN_ID,
+ priority_items = [
+     MEGA_SIGNAL_ID,
+     BUDDY_BUDDY_POFFIN_ID,
```

**Why it Worked:**
By forcefully toggling the `has_starmie_hand_or_play` flag, the agent stopped durdling with unnecessary basic Pokémon setup and committed its resources to getting a fully evolved Starmie ex online. In the current simulation meta, speed to evolution directly translates to early knockouts and prize leads.

---

## 🥈 Runner Up: Treatment 04 - Energy Hoarder (734.6 ELO)

**The Core Concept:**
The baseline heuristic weights for having energy on the active and bench were relatively low, meaning the agent often prioritized other board states over manually attaching energy. Treatment 04 massively inflated the value of hoarding energy on the board.

**Code Changes from Baseline:**
```python
# Drastically increased the heuristic weight for energy on the Active Pokémon
- 0.045745,   # my_active_energy
+ 0.120000,   # my_active_energy

# Doubled the heuristic weight for energy on the Bench
- 0.204046,   # my_bench_energy_total
+ 0.400000,   # my_bench_energy_total
```

**Why it Worked:**
By telling the agent that holding energy on the board is the most valuable state (0.40 weight), the agent practically guaranteed it never missed an energy attachment for the turn. This prevented energy droughts and ensured that retreating or attacking was always an option, leading to high consistency.

---

## 📉 What Failed (The Baseline Group - 600.0 ELO)

We tested two other treatments that failed to improve upon the baseline:

1. **Treatment 02 (Aggressive)**: We doubled the weights on taking prizes (`my_prizes_taken` to 0.15, `prize_diff` to 0.50) and reduced the HP retreat threshold. **Result:** The agent likely became too reckless, overcommitting to knockouts without proper setup, resulting in lost games and a flat 600 ELO.
2. **Treatment 03 (Defensive)**: We increased the weight of `my_active_hp` and raised the retreat threshold to 50% HP. **Result:** The agent retreated too often, losing tempo and failing to take knockouts, resulting in a flat 600 ELO.

## Tomorrow's Next Steps

To push for **800 ELO**, we should combine the learnings from our two successful runs:
1. Combine the **Fast Evolve priority logic** (from Treatment 05) with the **Energy Hoarding heuristic weights** (from Treatment 04).
2. Continue iterating through the remaining generated treatments (06 - 10) to see if interactions or nonlinear HP scaling offer even better heuristics.
