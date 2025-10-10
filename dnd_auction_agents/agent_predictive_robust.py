import random
import os
import json
from typing import Any, List, Dict
from dnd_auction_game import AuctionGameClient

# Global variables for learning (kept minimal)
auction_history: List[Dict[str, Any]] = []
agent_performance: Dict[str, Dict[str, int]] = {}
def predictive_bidding(agent_id: str, current_round: int, states: Dict[str, Dict[str, Any]], 
                       auctions: Dict[str, Dict[str, Any]], prev_auctions: Dict[str, Dict[str, Any]], bank_state: Dict[str, Any]) -> Dict[str, int]:
    """
    Params:
    - agent_id: str - Unique identifier for this agent
    - current_round: int - Current round number (0-indexed)
    - states: dict - Current states of all agents, keyed by agent_id
    - auctions: dict - Current auctions available, keyed by auction_id
    - prev_auctions: dict - Previous auctions with results, keyed by auction_id
    - bank_state: dict - Current state of the bank (gold income, etc.)
    Returns:
    - bids: dict - Bids to place, keyed by auction_id with bid amount as value
    """
    global auction_history, agent_performance

    # Randomize the order of auctions to avoid predictable patterns
    auction_items = list(auctions.items())
    random.shuffle(auction_items)
    auctions = dict(auction_items)

    # Randomize auction history to avoid overfitting to recent patterns
    random.shuffle(auction_history)
    
    # Get current state 
    agent_state = states[agent_id]
    current_gold = agent_state["gold"]
    
    # Get next round income 
    next_round_gold_income = 0
    if len(bank_state["gold_income_per_round"]) > 0:
        next_round_gold_income = bank_state["gold_income_per_round"][0]
    
    # Update historical data
    try:
        if prev_auctions:
            for auction_id, auction_data in prev_auctions.items():
                if 'bids' in auction_data:
                    auction_history.append(auction_data)
                    # Keep only recent history to prevent memory issues
                    if len(auction_history) > 50:
                        auction_history = auction_history[-50:]
    except:
        # If historical update fails, continue without it
        pass
    
    # Calculate base bidding parameters
    base_bid = 15
    max_bid = 80 
    
    # Adjust based on income
    if next_round_gold_income > 1050:
        max_bid = 200
        base_bid = 40
    elif next_round_gold_income > 800:
        max_bid = 120
        base_bid = 25
    
    
    with open("dnd_auction_agents/config.json") as config_file:
        config = json.load(config_file)

    # Use historical data to be smarter about bidding
    if len(auction_history) > 5:
        try:
            # Calculate average winning bids from recent history
            recent_winners: List[int] = []
            for auction in auction_history[-10:]:
                auction_bids: list[dict[str, Any]] = auction.get('bids', [])
                if auction_bids:
                    gold_bids: list[int] = [int(bid.get('gold', 0)) for bid in auction_bids if 'gold' in bid]
                    if gold_bids:
                        winning_bid: int = max(gold_bids)
                        if winning_bid > 0:
                            recent_winners.append(winning_bid)
            
            if recent_winners:
                avg_winning_bid = sum(recent_winners) / len(recent_winners)
                # Adjust our max bid based on what typically wins
                max_bid = min(max_bid * config.get("max_multiplier", 1.5), int(avg_winning_bid * 1.4))
        except:
            # If calculation fails, stick with defaults
            pass
    
    # Progressive bidding strategy based on round
    round_multiplier = 1.0 + (current_round * config.get("round_multiplier", 0.03))  # Get more aggressive over time
    max_bid = int(max_bid * round_multiplier)
    
    # Bid on auctions
    bids: dict[str, int] = {}

    random_int = random.randint(1, 100)
    if random_int <= 3 and current_gold > 20000: 
        top_auctions = list(auctions.keys())[:5]
        if top_auctions:
            chosen_auction = random.choice(top_auctions)
            bid_amount = int(current_gold * 0.5)
            bid_amount = max(base_bid, min(bid_amount, current_gold - len(auctions) + len(bids)))
            if bid_amount >= 1 and current_gold >= bid_amount:
                bids[chosen_auction] = bid_amount
                current_gold -= bid_amount
    else:
        for auction_id, auction_data in auctions.items():
            try:
                # Calculate expected value for this auction
                die_size = auction_data.get('die', 6)
                num_dice = auction_data.get('num', 1)
                bonus = auction_data.get('bonus', 0)
                
                expected_value = ((die_size + 1) / 2) * num_dice + bonus
                
                # More aggressive bidding strategy
                if expected_value > 25:  # Very high value - bid aggressively
                    bid = random.randint(int(max_bid * 0.8), max_bid)
                elif expected_value > 15:  # High value auction
                    bid = random.randint(int(max_bid * 0.6), int(max_bid * 0.7))
                elif expected_value > 8:  # Medium value auction
                    bid = random.randint(int(max_bid * 0.4), int(max_bid * 0.6))
                elif expected_value > 0:  # Low value auction
                    bid = random.randint(base_bid, int(max_bid * 0.2))
                else:  # negative or zero expected value
                    bid = 0  
                
                # Ensure minimum bid of base_bid and maximum of what we can afford
                bid = max(base_bid, min(bid, current_gold - len(auctions) + len(bids)))  # Reserve gold for remaining auctions
                
                # Only skip if we can't afford ANY bid
                if bid >= 1 and current_gold >= bid:
                    bids[auction_id] = bid
                    current_gold -= bid
                
            except:
                # If auction processing fails, make a safe tiny bid
                if current_gold > 1:
                    safe_bid = random.randint(1, min(5, current_gold - 1))
                    bids[auction_id] = safe_bid
                    current_gold -= safe_bid
    
    # Track our performance
    try:
        if agent_id not in agent_performance:
            agent_performance[agent_id] = {'rounds': 0, 'total_bids': 0}
        
        agent_performance[agent_id]['rounds'] += 1
        agent_performance[agent_id]['total_bids'] += len(bids)
        
        # Print performance info
        if current_round % 10 == 0 or current_round < 5:
            perf: Dict[str, int] = agent_performance[agent_id]
            print(f"Round {current_round}: Bids={len(bids)}, Total={sum(bids.values())}, Gold={current_gold}")
            print(f"  Performance: {perf['rounds']} rounds, avg {perf['total_bids']/perf['rounds']:.1f} bids/round")
    except:
        # If performance tracking fails, just print basic info
        print(f"Round {current_round}: Made {len(bids)} bids, total {sum(bids.values())}, remaining {current_gold}")
    
    return bids

if __name__ == "__main__":
    host = "opentsetlin.com"
    player_id = "Douwe Berkeij"
    port = 8000
    agent_name = 'Dutch Courage                                                  ☠'

    game: AuctionGameClient = AuctionGameClient(host=host,
                                agent_name=agent_name,
                                player_id=player_id,
                                port=port)
    try:
        print(f"Starting robust predictive agent: {agent_name}")
        game.run(predictive_bidding)
        print(f"Game completed! Processed {len(auction_history)} auctions")
        if agent_performance:
            for agent_id, perf in agent_performance.items():
                print(f"Final performance - {perf['rounds']} rounds, {perf['total_bids']} total bids")
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")
    except Exception as e:
        print(f"Game failed: {e}")

    print("<game is done>")