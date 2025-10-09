import sys
from typing import Any, Dict, List, Tuple
from dnd_auction_game import AuctionGameClient

def rank_auctions_by_value(auctions: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any], float]]:
    """Rank auctions by expected value, best first"""
    auction_values = []
    
    for auction_id, auction in auctions.items():
        # Calculate expected value: E(dice) + bonus
        die_size = auction.get('die', 6)
        num_dice = auction.get('num', 1)
        bonus = auction.get('bonus', 0)
        
        # Expected value of a die is (1 + max_value) / 2
        expected_dice_value = ((1 + die_size) / 2) * num_dice
        total_expected_value = expected_dice_value + bonus
        
        auction_values.append((auction_id, auction, total_expected_value))
    
    # Sort by expected value (highest first)
    auction_values.sort(key=lambda x: x[2], reverse=True)
    return auction_values

def individual_bot_strategy(agent_id: str, current_round: int, states: Dict[str, Dict[str, Any]], 
                           auctions: Dict[str, Dict[str, Any]], prev_auctions: Dict[str, Dict[str, Any]], 
                           bank_state: Dict[str, Any]) -> Dict[str, int]:
    """
    Individual bot strategy with configurable auction rank and bid timing
    
    Args:
        agent_id: Unique identifier for this agent
        current_round: Current round number
        states: Current states of all agents
        auctions: Current auctions available
        prev_auctions: Previous auction results (unused but required by interface)
        bank_state: Current bank state (unused but required by interface)
    
    Returns:
        Dictionary of bids keyed by auction_id
    """
    # Suppress unused argument warnings - these are required by the interface
    _ = prev_auctions
    _ = bank_state
    
    # Get bot configuration from global variables (set by command line args)
    bot_target_rank = getattr(individual_bot_strategy, 'target_auction_rank', 0)
    bot_bid_turn = getattr(individual_bot_strategy, 'bid_turn_in_cycle', 1)
    bot_identifier = getattr(individual_bot_strategy, 'bot_id', 0)
    
    # Get agent state
    agent_state = states[agent_id]
    current_gold = agent_state["gold"]
    current_points = agent_state["points"]
    
    # Check if this is our turn to bid (every 10 rounds starting from our assigned turn)
    is_our_turn = (current_round % 10) == (bot_bid_turn - 1)
    
    if current_round % 20 == 0 or current_round < 5:  # Log periodically
        print(f"Bot {bot_identifier} (Rank {bot_target_rank+1}, Turn {bot_bid_turn}): Round {current_round}")
        print(f"  Gold: {current_gold:,}, Points: {current_points:,}")
        print(f"  My turn to bid: {is_our_turn}")
    
    # If it's not our turn, don't bid
    if not is_our_turn:
        return {}
    
    # If no auctions available, return empty bids
    if not auctions:
        return {}
    
    # Rank auctions by expected value
    ranked_auctions = rank_auctions_by_value(auctions)
    
    # Check if we have enough auctions to target our assigned rank
    if len(ranked_auctions) <= bot_target_rank:
        if current_round % 20 == 0:
            print(f"  Bot {bot_identifier}: Not enough auctions ({len(ranked_auctions)}) for rank {bot_target_rank+1}")
        return {}
    
    # Get our target auction (the one at our assigned rank)
    target_auction_id, _, expected_value = ranked_auctions[bot_target_rank]
    
    # Calculate bid amount based on expected value and available gold
    # Be more conservative with bidding to avoid running out of gold
    base_bid = max(1, int(expected_value * 0.8))  # Bid 0.8x expected value (more conservative)
    
    # Ensure we don't bid more than we can afford (keep larger reserve)
    reserve_gold = max(500, current_gold // 10)  # Keep at least 500 gold or 10% of current gold
    max_affordable_bid = max(1, current_gold - reserve_gold)
    
    # Also limit bid to a reasonable percentage of current gold
    max_percentage_bid = max(1, current_gold // 20)  # Never bid more than 5% of current gold
    
    bid_amount = min(base_bid, max_affordable_bid, max_percentage_bid)
    
    # Ensure we have enough gold
    if current_gold >= bid_amount and bid_amount > 0:
        print(f"  Bot {bot_identifier} BIDDING: {bid_amount} on {target_auction_id} (Rank {bot_target_rank+1}, EV: {expected_value:.1f})")
        return {target_auction_id: bid_amount}
    else:
        print(f"  Bot {bot_identifier}: Not enough gold ({current_gold}) for bid on {target_auction_id}")
        return {}

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python individual_bot.py <bot_id> <target_auction_rank> <bid_turn_in_cycle>")
        print("  bot_id: Unique identifier for this bot (0-99)")
        print("  target_auction_rank: Which auction rank to target (0-9, where 0=best)")
        print("  bid_turn_in_cycle: When in the 10-round cycle to bid (1-10)")
        sys.exit(1)
    
    try:
        bot_id = int(sys.argv[1])
        target_auction_rank = int(sys.argv[2])
        bid_turn_in_cycle = int(sys.argv[3])
        
        # Validate arguments
        if not (0 <= bot_id <= 99):
            raise ValueError("bot_id must be between 0 and 99")
        if not (0 <= target_auction_rank <= 9):
            raise ValueError("target_auction_rank must be between 0 and 9")
        if not (1 <= bid_turn_in_cycle <= 10):
            raise ValueError("bid_turn_in_cycle must be between 1 and 10")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set configuration as function attributes
    individual_bot_strategy.bot_id = bot_id
    individual_bot_strategy.target_auction_rank = target_auction_rank
    individual_bot_strategy.bid_turn_in_cycle = bid_turn_in_cycle
    
    # Configure connection - each bot needs completely unique identifiers
    host = "localhost"
    agent_name = f"bot_{bot_id}_rank{target_auction_rank+1}_turn{bid_turn_in_cycle}"
    player_id = f"unique_bot_player_{bot_id}_{target_auction_rank}_{bid_turn_in_cycle}"
    port = 8000

    print(f"🤖 Starting Individual Bot #{bot_id}")
    print(f"  Target auction rank: {target_auction_rank+1} (1=best, 10=worst)")
    print(f"  Bid turn in cycle: {bid_turn_in_cycle} (every 10 rounds)")
    print(f"  Agent name: {agent_name}")
    print()

    game = AuctionGameClient(host=host,
                            agent_name=agent_name,
                            player_id=player_id,
                            port=port)
    try:
        game.run(individual_bot_strategy)
    except KeyboardInterrupt:
        print(f"<Bot {bot_id} interrupted - shutting down>")
    except (ConnectionError, RuntimeError) as e:
        print(f"<Bot {bot_id} error: {e}>")

    print(f"<Bot {bot_id} game is done>")