import random
from typing import Any, Dict, List, Tuple
from dnd_auction_game import AuctionGameClient

class BotManager:
    """Manages 100 bots with 10-round cooldown periods"""
    
    def __init__(self):
        # Initialize 100 bots, 10 for each of the top 10 auctions
        self.bots = []
        for i in range(100):
            bot = {
                'id': i,
                'last_bid_round': -10,  # Start with -10 so they can bid immediately
                'assigned_auction_rank': i // 10,  # 0-9 for top 10 auctions
                'cooldown_rounds': 10
            }
            self.bots.append(bot)
        
        self.current_round = 0
        print(f"Initialized {len(self.bots)} bots:")
        print("  - 10 bots for each of the top 10 auctions")
        print("  - Each bot has a 10-round cooldown between bids")
    
    def get_available_bots_for_rank(self, auction_rank: int) -> List[Dict]:
        """Get bots available to bid on auctions of a specific rank"""
        available_bots = []
        for bot in self.bots:
            if (bot['assigned_auction_rank'] == auction_rank and 
                self.current_round - bot['last_bid_round'] >= bot['cooldown_rounds']):
                available_bots.append(bot)
        return available_bots
    
    def mark_bot_as_used(self, bot_id: int):
        """Mark a bot as having bid this round"""
        for bot in self.bots:
            if bot['id'] == bot_id:
                bot['last_bid_round'] = self.current_round
                break
    
    def get_bot_status(self) -> str:
        """Get status information about all bots"""
        available_by_rank = {}
        for rank in range(10):
            available_by_rank[rank] = len(self.get_available_bots_for_rank(rank))
        
        total_available = sum(available_by_rank.values())
        return f"Round {self.current_round}: {total_available}/100 bots available"

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

def multi_bot_strategy(agent_id: str, current_round: int, states: Dict[str, Dict[str, Any]], 
                      auctions: Dict[str, Dict[str, Any]], prev_auctions: Dict[str, Dict[str, Any]], 
                      bank_state: Dict[str, Any]) -> Dict[str, int]:
    """
    Strategy that manages 100 bots bidding on top 10 auctions with cooldown periods
    
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
    # Initialize global bot manager if it doesn't exist
    if not hasattr(multi_bot_strategy, 'bot_manager'):
        multi_bot_strategy.bot_manager = BotManager()
    
    bot_manager = multi_bot_strategy.bot_manager
    bot_manager.current_round = current_round
    
    # Get agent state
    agent_state = states[agent_id]
    current_gold = agent_state["gold"]
    current_points = agent_state["points"]
    
    print(f"\n=== Multi-Bot Strategy Round {current_round} ===")
    print(f"Agent gold: {current_gold:,}, points: {current_points:,}")
    print(f"Available auctions: {len(auctions)}")
    print(bot_manager.get_bot_status())
    
    # If no auctions available, return empty bids
    if not auctions:
        return {}
    
    # Rank auctions by expected value
    ranked_auctions = rank_auctions_by_value(auctions)
    
    # Take only top 10 auctions (or all if fewer than 10)
    top_10_auctions = ranked_auctions[:10]
    
    print(f"Top {len(top_10_auctions)} auctions by expected value:")
    for i, (auction_id, auction, expected_value) in enumerate(top_10_auctions):
        print(f"  {i+1}. {auction_id}: EV={expected_value:.1f} (d{auction['die']}x{auction['num']}+{auction['bonus']})")
    
    bids = {}
    total_bid_amount = 0
    bids_placed = 0
    
    # For each of the top 10 auction ranks, try to place one bid
    for rank, (auction_id, auction, expected_value) in enumerate(top_10_auctions):
        # Get available bots for this auction rank
        available_bots = bot_manager.get_available_bots_for_rank(rank)
        
        if not available_bots:
            print(f"  Rank {rank+1}: No bots available (all on cooldown)")
            continue
        
        # Select one bot randomly from available bots for this rank
        selected_bot = random.choice(available_bots)
        
        # Calculate bid amount based on expected value and available gold
        base_bid = max(1, int(expected_value * 2))  # Bid roughly 2x expected value
        
        # Adjust bid based on how much gold we have left
        remaining_gold = current_gold - total_bid_amount
        max_affordable_bid = max(1, remaining_gold // (len(top_10_auctions) - rank))  # Reserve gold for remaining auctions
        
        bid_amount = min(base_bid, max_affordable_bid)
        
        # Ensure we have enough gold
        if remaining_gold >= bid_amount and bid_amount > 0:
            bids[auction_id] = bid_amount
            total_bid_amount += bid_amount
            bids_placed += 1
            
            # Mark bot as used
            bot_manager.mark_bot_as_used(selected_bot['id'])
            
            print(f"  Rank {rank+1}: Bot {selected_bot['id']} bids {bid_amount} on {auction_id} (EV: {expected_value:.1f})")
        else:
            print(f"  Rank {rank+1}: Not enough gold ({remaining_gold}) for minimum bid on {auction_id}")
    
    print(f"Total bids placed: {bids_placed}/{len(top_10_auctions)}")
    print(f"Total gold spent: {total_bid_amount:,}/{current_gold:,}")
    print(f"=== End Round {current_round} ===\n")
    
    return bids

if __name__ == "__main__":
    host = "localhost"
    agent_name = f"multi_bot_agent_{random.randint(1, 1000)}"
    player_id = "multi_bot_player"
    port = 8000

    print("🤖 Starting Multi-Bot Auction Agent")
    print("=" * 50)
    print("Features:")
    print("• Manages 100 bots total")
    print("• 10 bots assigned to each of the top 10 auctions")
    print("• Each bot can only bid once every 10 rounds")
    print("• Automatically ranks auctions by expected value")
    print("• Places one bid per round on each top auction")
    print("=" * 50)
    print()

    game = AuctionGameClient(host=host,
                            agent_name=agent_name,
                            player_id=player_id,
                            port=port)
    try:
        game.run(multi_bot_strategy)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")
    except (ConnectionError, RuntimeError) as e:
        print(f"<error: {e}>")

    print("<game is done>")