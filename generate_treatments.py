import os
import re
import shutil

def read_main():
    with open("main.py", "r") as f:
        return f.read()

def write_treatment(idx, name, content):
    os.makedirs("treatments", exist_ok=True)
    filename = f"treatments/treatment_{idx:02d}_{name}.py"
    with open(filename, "w") as f:
        f.write(content)
    print(f"Created {filename}")

def main():
    base_code = read_main()
    
    # Treatment 1: Baseline
    write_treatment(1, "baseline", base_code)
    
    # Treatment 2: Aggressive Strategy
    # Increase weights on taking prizes (idx 4) and opponent damage
    code_2 = re.sub(r'0.081985,   # my_prizes_taken', '0.150000,   # my_prizes_taken', base_code)
    code_2 = re.sub(r'0.299646,   # prize_diff', '0.500000,   # prize_diff', code_2)
    code_2 = re.sub(r'active.hp < active.maxHp \* 0.3:', 'active.hp < active.maxHp * 0.15:', code_2)
    write_treatment(2, "aggressive", code_2)
    
    # Treatment 3: Defensive Strategy
    # Increase weight on own HP and bench
    code_3 = re.sub(r'0.001550,   # my_active_hp', '0.015000,   # my_active_hp', base_code)
    code_3 = re.sub(r'0.204046,   # my_bench_energy_total', '0.350000,   # my_bench_energy_total', code_3)
    code_3 = re.sub(r'active.hp < active.maxHp \* 0.3:', 'active.hp < active.maxHp * 0.5:', code_3)
    write_treatment(3, "defensive", code_3)
    
    # Treatment 4: Energy Hoarder
    code_4 = re.sub(r'0.045745,   # my_active_energy', '0.120000,   # my_active_energy', base_code)
    code_4 = re.sub(r'0.204046,   # my_bench_energy_total', '0.400000,   # my_bench_energy_total', code_4)
    write_treatment(4, "energy_hoarder", code_4)
    
    # Treatment 5: Fast Evolve Focus
    # Give massive priority to cyrano and evolutions
    code_5 = re.sub(r'has_starmie_hand_or_play = False', 'has_starmie_hand_or_play = True', base_code) # force cyrano priority differently
    # actually let's modify the priority lists
    code_5 = code_5.replace('priority_items = [\n                BUDDY_BUDDY_POFFIN_ID', 'priority_items = [\n                MEGA_SIGNAL_ID,\n                BUDDY_BUDDY_POFFIN_ID')
    write_treatment(5, "fast_evolve", code_5)
    
    # Treatment 6: Boss's Orders Focus
    code_6 = base_code.replace('boss_priority = False', 'boss_priority = True')
    write_treatment(6, "boss_focus", code_6)
    
    # Treatment 7: Feature Engineering - Non-linear HP
    # Modify f1 and f9 to use sqrt
    code_7 = base_code.replace('f1 = float(my_act.hp) if my_act else 0.0', 'f1 = float(my_act.hp)**0.5 if my_act else 0.0')
    code_7 = code_7.replace('f9 = float(opp_act.hp) if opp_act else 0.0', 'f9 = float(opp_act.hp)**0.5 if opp_act else 0.0')
    write_treatment(7, "nonlinear_hp", code_7)
    
    # Treatment 8: Feature Engineering - Interaction Terms
    code_8 = base_code.replace('feats = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16]',
                               'feats = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16]\n    z += 0.05 * f1 * f2 # interaction term')
    write_treatment(8, "interaction_terms", code_8)
    
    # Treatment 9: Randomized Heuristics (Exploration)
    code_9 = base_code.replace('z = sum(DSVN_WEIGHTS[j] * feats[j] for j in range(16)) + DSVN_BIAS',
                               'z = sum(DSVN_WEIGHTS[j] * feats[j] for j in range(16)) + DSVN_BIAS + random.uniform(-0.5, 0.5)')
    write_treatment(9, "randomized_heuristics", code_9)
    
    # Treatment 10: Hyperparameter Tuning - High Thresholds
    code_10 = base_code.replace('DSVN_BIAS = 0.164533', 'DSVN_BIAS = 0.500000')
    code_10 = code_10.replace('score = 950', 'score = 990')
    write_treatment(10, "high_thresholds", code_10)

if __name__ == "__main__":
    main()
