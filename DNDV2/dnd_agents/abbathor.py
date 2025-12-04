"""
Value Shark Agent v2 (Inflation Corrected)
------------------------------------------
Counter to "Wolf" agents in high-concurrency games.

Fixes from v1:
1. POOL FIX: 'pool' return value is points-to-sell. We now return 0 to hoard points.
2. INFLATION FIX: Removed CP_MAX caps. Adapts to hyper-inflation (1M+ gold bids).
3. AGGRESSION: Dynamically scales bids if we are hoarding too much gold.
"""

import os
import random
import statistics
from typing import Dict, Any, Tuple, Optional

from dnd_auction_game import AuctionGameClient

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Expected Value per die type (Average roll)
AVG = {
    1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 6: 3.5,
    8: 4.5, 10: 5.5, 12: 6.5, 20: 10.5,
}

# Market Analysis
EMA_ALPHA = 0.2           # Reactivity to price changes
STARTING_CP = 10.0        # Initial guess for Gold per Point
# CP_MAX is removed to allow handling hyper-inflation

# ------------------------------------------------------------
# Logic Class
# ------------------------------------------------------------

class ValueSharkAgent:
    def __init__(self):
        self.market_cp = STARTING_CP
        self.round_count = 0
        self.total_rounds = int(os.environ.get("ASSUMED_TOTAL_ROUNDS", "500"))
        
        # Track our win rate to adjust aggression
        self.auctions_seen = 0
        self.auctions_won = 0

    def update_market_cp(self, prev_auctions: Dict[str, Any], my_id: str):
        """
        Update the estimated Cost Per Point (CP) based on recent winning bids.
        Allows for massive inflation.
        """
        ratios = []
        for a in prev_auctions.values():
            self.auctions_seen += 1
            
            reward = float(a.get("reward", 0))
            bids = a.get("bids", [])
            
            if bids:
                winner_id = bids[0]["a_id"]
                if winner_id == my_id:
                    self.auctions_won += 1
                
                winning_bid = float(bids[0]["gold"])
                
                # Only analyze positive rewards to avoid skewing data with penalties
                if reward > 0:
                    # Cost Per Point = Bid / Reward
                    ratios.append(winning_bid / reward)

        if not ratios:
            return

        # Use median to filter out extreme outliers (both low and high)
        round_median_cp = statistics.median(ratios)
        
        # Smooth update: New_CP = (Old * 0.8) + (New * 0.2)
        self.market_cp = (1.0 - EMA_ALPHA) * self.market_cp + (EMA_ALPHA * round_median_cp)
        
        # Safety floor only, no ceiling
        self.market_cp = max(1.0, self.market_cp)

    def decide(
        self,
        agent_id: str,
        round_number: int,
        states: Dict[str, Any],
        auctions: Dict[str, Any],
        prev_auctions: Dict[str, Any],
        pool_gold: int,
        prev_pool_buys: Dict[str, Any],
        bank_state: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, int], int]:

        if isinstance(round_number, int) and round_number > 0:
            self.round_count = round_number
        else:
            self.round_count += 1
        
        # 1. Update Market Knowledge
        if prev_auctions:
            self.update_market_cp(prev_auctions, agent_id)

        me = states[agent_id]
        my_gold = int(me["gold"])
        
        if my_gold <= 0:
            return {}, 0

        # 2. Calculate Resource Pressure
        # How much gold do we need to dump per round to finish empty?
        rounds_left = max(1, self.total_rounds - self.round_count)
        if isinstance(bank_state, dict):
            income_schedule = bank_state.get("gold_income_per_round")
            if isinstance(income_schedule, list) and income_schedule:
                rounds_left = max(1, len(income_schedule))
        
        # Estimate total EV on board this round
        total_ev_on_board = 0
        scored_auctions = []
        
        for a_id, a in auctions.items():
            die = int(a["die"])
            num = int(a["num"])
            bonus = int(a["bonus"])
            ev = (num * AVG.get(die, 0)) + bonus
            if ev <= 0: continue
            
            total_ev_on_board += ev
            scored_auctions.append((ev, a_id))

        if total_ev_on_board == 0:
            return {}, 0

        # 3. Determine Spending Aggression
        # "Fair" spend per round to drain account exactly by end of game
        target_burn_rate = my_gold / rounds_left
        
        # Current Market Value of the board
        board_market_value = total_ev_on_board * self.market_cp
        
        # Inflation Factor:
        # If we have WAY more gold than the market implies we need, we must inflate our bids
        # to guarantee wins. We are rich, so pay more.
        if board_market_value > 0:
            wealth_ratio = target_burn_rate / (board_market_value / len(auctions))
        else:
            wealth_ratio = 1.0

        # Aggression multiplier:
        # 1.0 = Bid Market Value
        # 2.0 = Bid Double Market Value (if we are super rich compared to prices)
        aggression = max(1.0, min(5.0, wealth_ratio * 1.2))

        # 4. Bidding Logic
        bids = {}
        
        # Sort by highest EV first
        scored_auctions.sort(reverse=True)
        
        # Budget for this specific round (can exceed burn rate if opportunities are good)
        round_budget = target_burn_rate * 2.0 
        current_spend = 0
        
        for ev, a_id in scored_auctions:
            if current_spend >= round_budget:
                break

            # Calculate base valuation
            valuation = ev * self.market_cp * aggression
            
            # Random jitter to prevent ties
            bid = int(valuation * random.uniform(1.0, 1.1))
            
            # Hard Limit: Don't spend more than 40% of TOTAL holding on one item 
            # (unless it's the very last round)
            if rounds_left > 10:
                bid = min(bid, int(my_gold * 0.40))
            
            # Clamp to actual gold
            bid = min(bid, my_gold - current_spend)
            
            if bid > 0:
                bids[a_id] = bid
                current_spend += bid

        # 5. Pool Strategy (FIXED)
        # We return 0. The API 'pool' value is POINTS TO SELL.
        # We never want to sell points. We want to win.
        points_to_sell = 0

        return bids, points_to_sell

# ------------------------------------------------------------
# API Hook
# ------------------------------------------------------------

def make_bid(
    agent_id: str,
    round_number: int,
    states: Dict[str, Any],
    auctions: Dict[str, Any],
    prev_auctions: Dict[str, Any],
    pool_gold: int,
    prev_pool_buys: Dict[str, Any],
    bank_state: Dict[str, Any],
) -> Dict[str, Any]:
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = ValueSharkAgent()

    bids, pool_points = make_bid._agent.decide(
        agent_id,
        round_number,
        states,
        auctions,
        prev_auctions,
        pool_gold,
        prev_pool_buys,
        bank_state,
    )
    return {"bids": bids, "pool": pool_points}

if __name__ == "__main__":
    host = "localhost"
    agent_name = "abbathor"
    player_id = "Ole Hogsnes Haugholt"
    port = 8000

    game = AuctionGameClient(
        host=host,
        agent_name=agent_name,
        player_id=player_id,
        port=port,
    )
    try:
        game.run(make_bid)
    except KeyboardInterrupt:
        print("<interrupt>")
    print("<game done>")