"""
Bot Ultra-Optimizado para 1000 Rondas - Estrategia Ganadora

Mejoras clave:
1. Análisis de competencia en tiempo real
2. Gestión dinámica del pool con ROI
3. Adaptación basada en historial de prev_auctions
4. Bids inteligentes considerando comportamiento de oponentes
5. Maximización del interés compuesto a largo plazo
"""

import random


def evaluate_auction_value(auction):
    """Calcula valor esperado, máximo y mínimo de una subasta"""
    num = auction.get("num", 1)
    die = auction.get("die", 6)
    bonus = auction.get("bonus", 0)
    
    expected = num * (die + 1) / 2.0 + bonus
    max_val = num * die + bonus
    min_val = num + bonus
    
    # Calcular varianza (riesgo)
    variance = num * ((die * die - 1) / 12.0)
    std_dev = variance ** 0.5
    
    return expected, max_val, min_val, std_dev


def get_game_phase(current_round, total_rounds=1000):
    """Determina fase del juego"""
    progress = current_round / total_rounds
    
    if progress < 0.15:  # Rondas 1-150
        return "early"
    elif progress < 0.50:  # Rondas 151-500
        return "growth"
    elif progress < 0.85:  # Rondas 501-850
        return "peak"
    else:  # Rondas 851-1000
        return "endgame"


def analyze_competition(agent_id, states, prev_auctions):
    """
    Analiza la competencia y retorna métricas clave.
    
    Returns:
        dict: {
            'my_rank': int,
            'avg_opponent_gold': float,
            'avg_opponent_points': float,
            'total_opponents': int,
            'am_leading': bool,
            'avg_winning_bid': float
        }
    """
    my_state = states.get(agent_id, {"gold": 1000, "points": 0})
    
    opponents = [s for aid, s in states.items() if aid != agent_id]
    
    if not opponents:
        return {
            'my_rank': 1,
            'avg_opponent_gold': 1000,
            'avg_opponent_points': 0,
            'total_opponents': 0,
            'am_leading': True,
            'avg_winning_bid': 300
        }
    
    avg_opp_gold = sum(o["gold"] for o in opponents) / len(opponents)
    avg_opp_points = sum(o["points"] for o in opponents) / len(opponents)
    
    # Calcular ranking por puntos
    all_points = [(aid, s["points"]) for aid, s in states.items()]
    all_points.sort(key=lambda x: x[1], reverse=True)
    my_rank = next(i for i, (aid, _) in enumerate(all_points, 1) if aid == agent_id)
    
    am_leading = my_state["points"] >= avg_opp_points
    
    # Analizar bids ganadores del historial
    winning_bids = []
    if prev_auctions:
        for auction_data in prev_auctions.values():
            bids = auction_data.get("bids", [])
            if bids:
                # El primer bid es el ganador
                winning_bids.append(bids[0]["gold"])
    
    avg_winning_bid = sum(winning_bids) / len(winning_bids) if winning_bids else 300
    
    return {
        'my_rank': my_rank,
        'avg_opponent_gold': avg_opp_gold,
        'avg_opponent_points': avg_opp_points,
        'total_opponents': len(opponents),
        'am_leading': am_leading,
        'avg_winning_bid': avg_winning_bid
    }


def calculate_bank_info(my_gold, bank_state):
    """Obtiene información del banco"""
    bank_interest = bank_state.get("bank_interest_per_round", [1.10])[0]
    bank_limit = bank_state.get("bank_limit_per_round", [2000])[0]
    gold_income = bank_state.get("gold_income_per_round", [1000])[0]
    
    effective_gold = min(my_gold, bank_limit)
    gold_over_limit = max(0, my_gold - bank_limit)
    
    return effective_gold, bank_limit, gold_over_limit, bank_interest, gold_income


def calculate_pool_roi(pool, my_points, phase):
    """
    Calcula si vale la pena invertir puntos en el pool.
    
    Returns:
        int: Puntos a invertir (0 si no vale la pena)
    """
    if pool < 200 or my_points < 12:
        return 0
    
    # En early/growth, necesitamos puntos - no invertir
    if phase in ["early", "growth"]:
        return 0
    
    # En peak, considerar si tenemos exceso de puntos
    if phase == "peak":
        if my_points > 30 and pool > 1000:
            # Invertir máximo 10% de puntos excedentes
            excess_points = my_points - 25
            return min(int(excess_points * 0.10), 5)
        return 0
    
    # En endgame, si vamos muy adelante, invertir más
    if phase == "endgame":
        if my_points > 40:
            excess_points = my_points - 35
            return min(int(excess_points * 0.20), 10)
        return 0
    
    return 0


def calculate_optimal_spending(my_gold, my_points, phase, bank_limit, gold_over_limit, 
                               competition_analysis, current_round):
    """
    Calcula cuánto oro gastar considerando múltiples factores.
    """
    productive_gold = min(my_gold, bank_limit)
    
    # ENDGAME: Gastar TODO
    if phase == "endgame":
        return my_gold
    
    # Base spending rate por fase
    if phase == "early":
        # Early: ser más agresivo de lo usual para acumular puntos
        # El interés compuesto es importante pero necesitamos base de puntos
        base_rate = 0.55  # 55% del oro productivo
    elif phase == "growth":
        # Growth: balance - ya tenemos puntos, aprovechar interés
        base_rate = 0.68  # 68%
    else:  # peak
        # Peak: muy agresivo
        base_rate = 0.82  # 82%
    
    # AJUSTE 1: Si vamos perdiendo, ser más agresivo
    if not competition_analysis['am_leading']:
        rank_penalty = (competition_analysis['my_rank'] - 1) * 0.03
        base_rate += min(rank_penalty, 0.15)  # Máximo +15%
    
    # AJUSTE 2: Si tenemos mucho oro vs oponentes, ser más agresivo
    if my_gold > competition_analysis['avg_opponent_gold'] * 1.3:
        base_rate += 0.08
    
    # AJUSTE 3: Si tenemos poco oro, conservar más
    if my_gold < competition_analysis['avg_opponent_gold'] * 0.7:
        base_rate -= 0.10
    
    # Limitar rate
    base_rate = max(0.40, min(0.95, base_rate))
    
    # Calcular gasto
    spending = int(productive_gold * base_rate + gold_over_limit)
    
    return spending


def calculate_participation_rate(phase, my_gold, avg_opponent_gold, am_leading):
    """
    Calcula tasa de participación dinámica.
    """
    # Base por fase
    if phase == "early":
        base_rate = 0.72
    elif phase == "growth":
        base_rate = 0.78
    elif phase == "peak":
        base_rate = 0.88
    else:  # endgame
        return 1.0
    
    # Ajustes dinámicos
    if not am_leading:
        base_rate += 0.08  # Ser más agresivo si vamos perdiendo
    
    if my_gold > avg_opponent_gold * 1.5:
        base_rate += 0.05  # Aprovechar ventaja de oro
    
    return min(1.0, base_rate)


def calculate_dynamic_bid_intensity(auction_score, expected, min_val, max_val, 
                                    phase, avg_winning_bid, gold_per_auction):
    """
    Calcula intensidad del bid basándose en múltiples factores.
    """
    # Base intensity por fase
    if phase == "early":
        base = 0.85
    elif phase == "growth":
        base = 0.92
    elif phase == "peak":
        base = 1.05
    else:  # endgame
        base = 1.18
    
    # Ajuste por valor esperado
    if expected > 18:
        base *= 1.22
    elif expected > 15:
        base *= 1.14
    elif expected > 12:
        base *= 1.06
    elif expected < 8:
        base *= 0.88
    
    # Ajuste por riesgo
    risk_factor = min_val / max(max_val, 1)
    if risk_factor > 0.65:  # Bajo riesgo
        base *= 1.08
    elif risk_factor < 0.35:  # Alto riesgo
        base *= 0.90
    
    # Ajuste competitivo: superar el avg_winning_bid
    if avg_winning_bid > 0:
        competitive_ratio = gold_per_auction / avg_winning_bid
        if competitive_ratio < 0.8:  # Estamos bideando bajo
            base *= 1.12  # Compensar
        elif competitive_ratio > 1.5:  # Estamos bideando alto
            base *= 0.95  # Reducir un poco
    
    return base


def make_bid(agent_id, round, states, auctions, prev_auctions, pool, prev_pool_buys, bank_state):
    """
    Función principal del bot - OPTIMIZADA para 1000 rondas.
    """
    # Estado propio
    my_state = states.get(agent_id, {"gold": 1000, "points": 0})
    my_gold = my_state["gold"]
    my_points = my_state["points"]
    
    # Análisis del banco
    effective_gold, bank_limit, gold_over_limit, bank_interest, gold_income = \
        calculate_bank_info(my_gold, bank_state)
    
    # Fase del juego
    phase = get_game_phase(round, 1000)
    
    # Análisis de competencia
    comp_analysis = analyze_competition(agent_id, states, prev_auctions)
    
    # Decidir inversión en pool
    pool_investment = calculate_pool_roi(pool, my_points, phase)
    
    if not auctions:
        return {"bids": {}, "pool": pool_investment}
    
    # Evaluar subastas con análisis avanzado
    auction_values = []
    for aid, data in auctions.items():
        expected, max_val, min_val, std_dev = evaluate_auction_value(data)
        
        # Score sofisticado
        risk_factor = min_val / max(max_val, 1)
        stability = 1.0 - (std_dev / max(expected, 1))
        
        score = (
            expected * 1.0 +              # Valor esperado
            min_val * 0.35 +              # Piso garantizado
            risk_factor * 3.0 +           # Bajo riesgo
            stability * 2.0               # Estabilidad
        )
        
        auction_values.append({
            "id": aid,
            "score": score,
            "expected": expected,
            "min": min_val,
            "max": max_val,
            "std_dev": std_dev,
            "risk": risk_factor
        })
    
    # Ordenar por score
    auction_values.sort(key=lambda x: x["score"], reverse=True)
    
    # Determinar participación
    participation_rate = calculate_participation_rate(
        phase, my_gold, comp_analysis['avg_opponent_gold'], comp_analysis['am_leading']
    )
    
    num_to_bid = max(1, int(len(auction_values) * participation_rate))
    selected_auctions = auction_values[:num_to_bid]
    
    # Calcular oro a gastar
    gold_to_spend = calculate_optimal_spending(
        my_gold, my_points, phase, bank_limit, gold_over_limit, comp_analysis, round
    )
    
    # Distribuir oro entre subastas
    gold_per_auction = gold_to_spend / len(selected_auctions) if selected_auctions else 0
    
    bids = {}
    for auc in selected_auctions:
        # Calcular intensidad dinámica
        intensity = calculate_dynamic_bid_intensity(
            auc["score"], auc["expected"], auc["min"], auc["max"],
            phase, comp_analysis['avg_winning_bid'], gold_per_auction
        )
        
        bid = int(gold_per_auction * intensity)
        
        # Mínimos por fase
        min_bid = 35 if phase == "early" else (55 if phase == "growth" else 75)
        bid = max(bid, min_bid)
        
        # Variación aleatoria para romper empates
        bid += random.randint(1, 28)
        
        bids[auc["id"]] = bid
    
    # Ajuste final
    total_bid = sum(bids.values())
    if total_bid > my_gold:
        factor = (my_gold * 0.97) / total_bid
        for k in bids:
            bids[k] = max(25, int(bids[k] * factor))
    
    # ESTRATEGIA ESPECIAL: Si oro muy por encima del límite en peak/endgame
    if phase in ["peak", "endgame"] and gold_over_limit > bank_limit * 0.4:
        # Redistribuir ese oro extra en top 3 subastas
        top_3 = list(bids.keys())[:min(3, len(bids))]
        bonus_per_auction = int((gold_over_limit * 0.7) / max(len(top_3), 1))
        
        for auction_id in top_3:
            bids[auction_id] += bonus_per_auction
        
        # Re-validar
        total_bid = sum(bids.values())
        if total_bid > my_gold:
            factor = (my_gold * 0.96) / total_bid
            for k in bids:
                bids[k] = int(bids[k] * factor)
    
    return {"bids": bids, "pool": pool_investment}


# Wrapper de clase
class AggressiveBot:
    def __init__(self):
        self.name = "UltraWarrior"
        self.version = "2.0-Ultimate"
    
    def __call__(self, agent_id, round, states, auctions, prev_auctions, pool, prev_pool_buys, bank_state):
        return make_bid(agent_id, round, states, auctions, prev_auctions, pool, prev_pool_buys, bank_state)


# Testing
if __name__ == "__main__":
    print("🎲 Testing UltraWarrior Bot (1000 Rounds Optimized)\n")
    
    # Simular diferentes escenarios
    scenarios = [
        (50, 2500, 12, "Early - Building Capital"),
        (300, 8500, 85, "Growth - Balanced"),
        (650, 22000, 220, "Peak - Domination"),
        (950, 45000, 480, "Endgame - ALL IN")
    ]
    
    for round_num, gold, points, desc in scenarios:
        print(f"{'='*70}")
        print(f"📊 Scenario: {desc} (Round {round_num}/1000)")
        print(f"{'='*70}")
        
        test_states = {
            "me": {"gold": gold, "points": points},
            "opp1": {"gold": int(gold * 0.85), "points": int(points * 1.15)},
            "opp2": {"gold": int(gold * 1.05), "points": int(points * 0.92)},
            "opp3": {"gold": int(gold * 0.95), "points": int(points * 1.05)}
        }
        
        test_auctions = {
            "a1": {"die": 6, "num": 3, "bonus": 5},
            "a2": {"die": 10, "num": 2, "bonus": 3},
            "a3": {"die": 20, "num": 1, "bonus": 0},
            "a4": {"die": 8, "num": 2, "bonus": 4},
            "a5": {"die": 12, "num": 2, "bonus": -1},
        }
        
        test_prev = {
            "prev1": {"bids": [{"a_id": "opp1", "gold": 320}]},
            "prev2": {"bids": [{"a_id": "me", "gold": 410}]}
        }
        
        test_bank = {
            "gold_income_per_round": [1000],
            "bank_interest_per_round": [1.10],
            "bank_limit_per_round": [2000]
        }
        
        result = make_bid("me", round_num, test_states, test_auctions, 
                         test_prev, 500, {}, test_bank)
        
        total = sum(result['bids'].values())
        phase = get_game_phase(round_num, 1000)
        comp = analyze_competition("me", test_states, test_prev)
        
        print(f"\n  Gold: {gold} | Points: {points} | Phase: {phase}")
        print(f"  Rank: {comp['my_rank']}/4 | Leading: {comp['am_leading']}")
        print(f"  Participation: {len(result['bids'])}/{len(test_auctions)} ({len(result['bids'])/len(test_auctions)*100:.0f}%)")
        print(f"  Total bid: {total} ({total/gold*100:.1f}% of gold)")
        print(f"  Pool investment: {result['pool']} points")
        print(f"  Gold reserved: {gold - total}\n")
    
    print(f"{'='*70}")
    print("✅ Bot ready for 1000-round domination!")
    print("\n🎯 Key Features:")
    print("  • Dynamic competition analysis")
    print("  • Adaptive spending based on rank")
    print("  • Smart pool ROI calculation")
    print("  • Phase-aware aggression levels")
    print("  • Historical bid analysis")