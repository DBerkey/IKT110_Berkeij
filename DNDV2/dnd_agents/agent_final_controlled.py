import random
import statistics
import math
import json
import os
from dnd_auction_game import AuctionGameClient

# --- CONFIG LOADER ---
# This allows you to control the bot live from the Dashboard
DEFAULT_CONFIG = {
    "aggression": 1.0,      # Multiplier for bid prices
    "safety_cap": 15,       # Max points to spend on pool
    "mode_override": "AUTO" # Options: AUTO, SAVE, SPEND, DUMP
}

def load_config():
    try:
        if os.path.exists("bot_config.json"):
            with open("bot_config.json", "r") as f:
                return json.load(f)
    except:
        pass
    return DEFAULT_CONFIG

class PoolShark:
    def __init__(self):
        self.history = [] 
    
    def update(self, total_spent_by_others, pool_size):
        self.history.append({
            'spent': total_spent_by_others,
            'pool': pool_size
        })
        if len(self.history) > 20:
            self.history.pop(0)

    def get_competition_level(self):
        if not self.history:
            return 0
        ratios = []
        for h in self.history:
            if h['pool'] > 500:
                ratios.append(h['spent'] / h['pool'])
        if not ratios:
            return 0.001 
        return statistics.mean(ratios)

class HedgeFundBrain:
    def __init__(self):
        self.price_history = []
        self.window = 30
        self.shark = PoolShark()
    
    def update(self, prev_auctions, prev_pool_buys, current_pool):
        round_spent = 0
        round_points = 0
        for info in prev_auctions.values():
            if "bids" in info and info["bids"]:
                win_gold = info["bids"][0]["gold"]
                pts = info.get("reward", 0)
                if pts > 0:
                    round_spent += win_gold
                    round_points += pts
        
        if round_points > 0:
            self.price_history.append(round_spent / round_points)
        if len(self.price_history) > self.window:
            self.price_history.pop(0)

        total_pool_spent = sum(prev_pool_buys.values())
        self.shark.update(total_pool_spent, current_pool)

    def get_market_price(self):
        if not self.price_history:
            return 50.0
        return statistics.median(self.price_history)

brain = HedgeFundBrain()

def get_ev(auction):
    return ((auction['die'] + 1) / 2 * auction['num']) + auction['bonus']

def make_bid(agent_id: str,
             round: int,
             states: dict,
             auctions: dict,
             prev_auctions: dict,
             pool: int,
             prev_pool_buys: dict,
             bank_state: dict) -> dict:

    try:
        # LOAD LIVE CONTROLS
        config = load_config()
        aggression_factor = float(config.get("aggression", 1.0))
        safety_cap_setting = int(config.get("safety_cap", 15))
        mode_override = config.get("mode_override", "AUTO")

        # 1. Update Intel
        brain.update(prev_auctions, prev_pool_buys, pool)
        market_price = brain.get_market_price()
        
        my_gold = states[agent_id]['gold']
        my_points = states[agent_id]['points']
        
        # 2. Bank Strategy
        limits = bank_state['bank_limit_per_round']
        current_limit = limits[0] if limits else 0
        next_limit = limits[1] if len(limits) > 1 else 0
        
        target_savings = current_limit
        if next_limit < current_limit:
            target_savings = next_limit

        disposable_gold = max(0, my_gold - target_savings)
        
        # 3. Pool Shark
        buy_pool = 0
        if pool > 500:
            competitor_ratio = brain.shark.get_competition_level()
            predicted_others_spend = max(1, pool * competitor_ratio)
            
            # Dynamic Cap based on Config
            safety_cap = max(1, int(my_points * 0.05)) 
            safety_cap = min(safety_cap, safety_cap_setting) # User Control Limiter
            
            if my_points < 20: safety_cap = 1

            best_profit = 0
            best_bid = 0
            
            for bid in range(1, safety_cap + 1):
                share = bid / (predicted_others_spend + bid)
                gold_gained = pool * share
                cost_in_gold = bid * market_price
                profit = gold_gained - cost_in_gold
                
                if profit > best_profit:
                    best_profit = profit
                    best_bid = bid
            
            if best_profit > 200:
                buy_pool = best_bid

        # 4. Auction Strategy
        bids = {}
        
        # Mode Logic
        if mode_override != "AUTO":
            mode = mode_override
        else:
            mode = "SAVE"
            if disposable_gold > 0: mode = "SPEND"
            rounds_left = len(bank_state['gold_income_per_round'])
            if rounds_left < 50: mode = "DUMP"
            
        potential_items = []
        for aid, auc in auctions.items():
            ev = get_ev(auc)
            is_small = ev < 12
            if my_gold > 20000: is_small = False 
                
            estimated_value = ev * market_price
            
            if mode == "SAVE":
                willing_to_pay = estimated_value * 0.6
            elif mode == "SPEND":
                willing_to_pay = estimated_value * 1.2 
            else: # DUMP
                willing_to_pay = estimated_value * 5.0
            
            # Apply Manual Aggression Control
            willing_to_pay *= aggression_factor

            if is_small and mode != "DUMP":
                willing_to_pay *= 1.1
            
            potential_items.append({
                "id": aid,
                "val": int(willing_to_pay),
                "ev": ev,
                "is_small": is_small
            })
            
        if mode == "DUMP":
            potential_items.sort(key=lambda x: x['ev'], reverse=True)
        else:
            potential_items.sort(key=lambda x: (x['is_small'], x['ev']), reverse=True)
            
        current_spent = 0
        for item in potential_items:
            bid_val = item['val']
            
            if bid_val < 1: continue
            if mode == "SAVE" and (current_spent + bid_val > disposable_gold): continue
            
            if bid_val % 10 == 0: bid_val += 3
            elif bid_val % 5 == 0: bid_val += 2
            
            if bid_val > my_gold: bid_val = my_gold
            
            bids[item['id']] = bid_val
            current_spent += bid_val
            
            if current_spent > (my_gold * 0.7) and mode != "DUMP":
                break
                
        return {"bids": bids, "pool": buy_pool}

    except Exception as e:
        return {}

if __name__ == "__main__":
    
    host = "localhost"
    agent_name = "abbathor_"
    player_id = "id_of_human_player"
    port = 8000

    game = AuctionGameClient(host=host,
                                agent_name=agent_name,
                                player_id=player_id,
                                port=port)
    try:
        game.run(make_bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")