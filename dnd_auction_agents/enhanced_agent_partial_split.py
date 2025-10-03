import random
import os
import json
from typing import Any, Dict
from dnd_auction_game import AuctionGameClient

# Global variables to track agent state across rounds
previous_gold = 0
accumulated_interest = 0
config_file = "agent_config.json"

# Default configuration
default_config = {
    "gold_multiplier": 1.0,
    "minimum_reserve": 5000,
    "wait_rounds": 4,
    "timestamp": ""
}

def load_config():
    """Load configuration from file, return defaults if file doesn't exist"""
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults in case some keys are missing
                merged_config = default_config.copy()
                merged_config.update(config)
                return merged_config
    except Exception as e:
        print(f"Error loading config: {e}")
    
    return default_config.copy()

def save_agent_metrics(round_num, gold, points, interest, bids_made, config_used):
    """Save agent performance metrics for dashboard"""
    metrics = {
        "round": round_num,
        "gold": gold,
        "points": points,
        "interest_available": interest,
        "bids_made": len(bids_made),
        "total_bid_amount": sum(bids_made.values()),
        "config": config_used,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    try:
        # Append to metrics log
        with open("agent_metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
    except Exception as e:
        print(f"Error saving metrics: {e}")

def enhanced_interest_split_strategy(agent_id: str, current_round: int, states: dict[str, dict[str, Any]], 
                                   auctions: dict[str, Dict[str, Any]], prev_auctions: dict[str, Dict[str, Any]], 
                                   bank_state: dict[str, Any]) -> Dict[str, int]:
    """
    Enhanced strategy: Dynamically adjustable interest-split strategy with configurable parameters
    """
    global previous_gold, accumulated_interest
    
    # Load current configuration
    config = load_config()
    gold_multiplier = config["gold_multiplier"]
    minimum_reserve = config["minimum_reserve"]
    wait_rounds = config["wait_rounds"]
    
    agent_state = states[agent_id]
    current_gold = agent_state["gold"]
    current_points = agent_state["points"]
    
    print(f"\n=== Round {current_round} Enhanced Strategy ===")
    print(f"Config: {gold_multiplier}x multiplier, {minimum_reserve:,} reserve, wait {wait_rounds} rounds")
    print(f"Current: {current_gold:,} gold, {current_points:,} points")
    
    # Calculate interest earned this round
    if current_round > 0:
        gold_difference = current_gold - previous_gold
        
        # Estimate current round's income (1000 base + any bonuses)
        current_income = 1000
        if 'remainder_gold_income' in bank_state and len(bank_state['remainder_gold_income']) > 0:
            try:
                # Try to get actual income for this round
                current_income = bank_state['remainder_gold_income'][current_round % len(bank_state['remainder_gold_income'])]
            except:
                current_income = 1000
        
        # Interest is the gold difference minus the base income
        interest_this_round = max(0, gold_difference - current_income)
        accumulated_interest += interest_this_round
        
        print(f"Gold change: {gold_difference:+}, Income: {current_income}, Interest earned: {interest_this_round}")
        print(f"Total accumulated interest: {accumulated_interest}")
    
    bids = {}
    
    # Strategy phase: Wait for specified rounds, then bid with enhanced logic
    if current_round >= wait_rounds and accumulated_interest > 0 and len(auctions) > 0:
        # Calculate available gold for bidding
        usable_gold = max(0, current_gold - minimum_reserve)
        
        # Base bid per auction from accumulated interest
        base_bid_per_auction = max(1, int(accumulated_interest / len(auctions)))
        
        # Enhanced bid with multiplier, but constrained by available gold
        enhanced_bid_per_auction = int(base_bid_per_auction * gold_multiplier)
        
        # Ensure we don't exceed available gold
        total_enhanced_bids = enhanced_bid_per_auction * len(auctions)
        if total_enhanced_bids > usable_gold:
            # Scale down bids to fit available gold
            enhanced_bid_per_auction = max(1, int(usable_gold / len(auctions)))
        
        print(f"Bidding strategy:")
        print(f"  Usable gold: {usable_gold:,} (total: {current_gold:,} - reserve: {minimum_reserve:,})")
        print(f"  Base bid per auction: {base_bid_per_auction}")
        print(f"  Enhanced bid per auction: {enhanced_bid_per_auction} (multiplier: {gold_multiplier}x)")
        
        remaining_gold = usable_gold
        successful_bids = 0
        
        # Sort auctions by potential value (die * num + bonus) for smarter bidding
        auction_items = list(auctions.items())
        auction_items.sort(key=lambda x: x[1]['die'] * x[1]['num'] + x[1]['bonus'], reverse=True)
        
        for auction_id, auction in auction_items:
            if remaining_gold <= 0:
                break
                
            # Make sure we don't bid more than we have
            actual_bid = min(enhanced_bid_per_auction, remaining_gold)
            
            if actual_bid > 0:
                bids[auction_id] = actual_bid
                remaining_gold -= actual_bid
                successful_bids += 1
                
                expected_value = auction['die'] * auction['num'] + auction['bonus']
                print(f"  Bidding {actual_bid} on {auction_id} (expected value: {expected_value})")
        
        print(f"Total bids placed: {successful_bids}/{len(auctions)}, Total amount: {sum(bids.values()):,}")
        
        # Reset accumulated interest after using it
        accumulated_interest = 0
    else:
        reason = []
        if current_round < wait_rounds:
            reason.append(f"waiting (round {current_round} < {wait_rounds})")
        if accumulated_interest <= 0:
            reason.append(f"no interest ({accumulated_interest})")
        if len(auctions) == 0:
            reason.append("no auctions")
        
        print(f"Not bidding: {', '.join(reason)}")
    
    # Save metrics for dashboard
    save_agent_metrics(current_round, current_gold, current_points, 
                      accumulated_interest, bids, config)
    
    # Update for next round
    previous_gold = current_gold
    
    print(f"=== End Round {current_round} ===\n")
    return bids

if __name__ == "__main__":
    
    host = "localhost"
    agent_name = "enhanced_{}_{}".format(os.path.basename(__file__), random.randint(1, 1000))
    player_id = "id_of_human_player"
    port = 8000

    print("🚀 Starting Enhanced Interest-Split Agent")
    print("📊 This agent supports dynamic configuration via agent_config.json")
    print("🎯 Use the dashboard to adjust parameters in real-time!")
    
    game: AuctionGameClient = AuctionGameClient(host=host,
                            agent_name=agent_name,
                            player_id=player_id,
                            port=port)
    try:
        game.run(enhanced_interest_split_strategy)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")