import random
import os
from typing import Any, Dict
from dnd_auction_game import AuctionGameClient

# Global variables to track agent state across rounds
previous_gold = 0

def interest_split_strategy(agent_id: str, current_round: int, states: Dict[str, Dict[str, Any]], 
                                   auctions: Dict[str, Dict[str, Any]], prev_auctions: Dict[str, Dict[str, Any]], 
                                   bank_state: Dict[str, Any]) -> Dict[str, int]:
    """
    params:
    - agent_id: str - Unique identifier for this agent
    - current_round: int - Current round number (0-indexed)
    - states: dict - Current states of all agents, keyed by agent_id
    - auctions: dict - Current auctions available, keyed by auction_id
    - prev_auctions: dict - Previous auctions with results, keyed by auction_id
    - bank_state: dict - Current state of the bank (gold income, etc.)
    returns:
    - bids: dict - Bids to place, keyed by auction_id with bid amount as value
    """
    global previous_gold
    
    minimum_reserve = 5000
    wait_rounds = 4

    agent_state = states[agent_id]
    current_gold = agent_state["gold"]
    current_points = agent_state["points"]
    
    print(f"\n=== Round {current_round} Enhanced Strategy ===")
    print(f"Current: {current_gold:,} gold, {current_points:,} points")
    
    # Calculate interest earned this round
    gold_difference = current_gold - previous_gold

    bids: Dict[str, int] = {}

    # Strategy phase: Wait for specified rounds, then bid with enhanced logic
    if current_round >= wait_rounds and len(auctions) > 0:
        # Get current bank limit for this round
        current_bank_limit = 0
        try:
            if 'bank_limit_per_round' in bank_state and len(bank_state['bank_limit_per_round']) > 0:
                current_bank_limit = bank_state['bank_limit_per_round'][0]
        except:
            current_bank_limit = 2000  # Default fallback
        
        # Calculate expected values for all auctions
        auction_values: list[tuple[str, dict[str, Any], int]] = []
        for auction_id, auction in auctions.items():
            expected_value = auction['die'] * auction['num'] + auction['bonus']
            auction_values.append((auction_id, auction, expected_value))
        
        # Sort by expected value (lowest first for bottom 70%)
        auction_values.sort(key=lambda x: x[2])
        
        # Split into bottom 70% and top 30%
        split_index = int(len(auction_values) * 0.7)
        bottom_70_percent = auction_values[:split_index]
        top_30_percent = auction_values[split_index:]
        
        print(f"Auction analysis:")
        print(f"  Total auctions: {len(auction_values)}")
        print(f"  Bottom 70% auctions: {len(bottom_70_percent)} (conservative bidding)")
        print(f"  Top 30% auctions: {len(top_30_percent)} (aggressive bidding)")
        print(f"  Current bank limit: {current_bank_limit:,}")
        
        # Calculate available gold above bank limit
        gold_above_limit = max(0, current_gold - current_bank_limit)
        usable_gold = max(0, current_gold - minimum_reserve)
        
        print(f"Gold analysis:")
        print(f"  Total gold: {current_gold:,}")
        print(f"  Bank limit: {current_bank_limit:,}")
        print(f"  Gold above limit: {gold_above_limit:,}")
        print(f"  Reserve: {minimum_reserve:,}")
        print(f"  Usable for bidding: {usable_gold:,}")
        
        # Bidding strategy
        remaining_gold = usable_gold
        successful_bids = 0
        
        # For bottom 70%: Use conservative interest-based bidding
        if bottom_70_percent:
            base_bid_per_auction = max(1, int(gold_difference / len(bottom_70_percent)))
            enhanced_bid_per_auction = int(base_bid_per_auction)
            
            print(f"Bottom 70% strategy:")
            print(f"  Base bid per auction: {base_bid_per_auction}")
            print(f"  Enhanced bid per auction: {enhanced_bid_per_auction}")
            
            for auction_id, auction, expected_value in bottom_70_percent:
                if remaining_gold <= 0:
                    break
                    
                actual_bid = min(enhanced_bid_per_auction, remaining_gold)
                
                if actual_bid > 0 and expected_value > 0:
                    bids[auction_id] = int(actual_bid)
                    remaining_gold -= actual_bid
                    successful_bids += 1
                    print(f"  Conservative bid {actual_bid} on {auction_id} (EV: {expected_value})")
        
        # For top 30%: Bid everything above bank limit
        if top_30_percent and gold_above_limit > 0:
            # Use all gold above bank limit for top-value auctions
            aggressive_gold = min(gold_above_limit, remaining_gold)
            bid_per_top_auction = max(1, int(aggressive_gold / len(top_30_percent)))
            
            print(f"Top 30% strategy:")
            print(f"  Aggressive gold available: {aggressive_gold:,}")
            print(f"  Bid per top auction: {bid_per_top_auction}")
            
            for auction_id, auction, expected_value in top_30_percent:
                if aggressive_gold <= 0:
                    break
                    
                actual_bid = min(bid_per_top_auction, aggressive_gold, remaining_gold)
                
                if actual_bid > 0:
                    bids[auction_id] = int(actual_bid)
                    aggressive_gold -= actual_bid
                    remaining_gold -= actual_bid
                    successful_bids += 1
                    print(f"  Aggressive bid {actual_bid} on {auction_id} (EV: {expected_value})")
        
        print(f"Total bids placed: {successful_bids}/{len(auctions)}, Total amount: {sum(int(v) for v in bids.values()):,}")

    # Update for next round
    previous_gold = current_gold
    
    print(f"=== End Round {current_round} ===\n")
    return bids

if __name__ == "__main__":
    
    host = "localhost"
    agent_name = "enhanced_{}_{}".format(os.path.basename(__file__), random.randint(1, 1000))
    player_id = "id_of_human_player1"
    port = 8000

    print("Starting Enhanced Interest-Split Agent")
    print("This agent supports dynamic configuration via agent_config.json")
    print("Use the dashboard to adjust parameters in real-time!")
    print()

    
    game: AuctionGameClient = AuctionGameClient(host=host,
                            agent_name=agent_name,
                            player_id=player_id,
                            port=port)
    try:
        game.run(interest_split_strategy)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")