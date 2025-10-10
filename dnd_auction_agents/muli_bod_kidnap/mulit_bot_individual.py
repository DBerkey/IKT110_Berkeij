#!/usr/bin/env python3
import sys
from typing import Any, Dict, List, Tuple
from dnd_auction_game import AuctionGameClient


def rank_auctions_by_value(auctions: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any], float]]:
    """Bewertet Auktionen nach erwartetem Wert und sortiert sie absteigend."""
    auction_values = []
    for aid, a in auctions.items():
        die = a.get("die", 6)
        num = a.get("num", 1)
        bonus = a.get("bonus", 0)
        ev = ((1 + die) / 2.0) * num + bonus
        auction_values.append((aid, a, ev))
    auction_values.sort(key=lambda x: x[2], reverse=True)
    return auction_values


def individual_bot_strategy(agent_id: str, current_round: int, states: Dict[str, Dict[str, Any]],
                            auctions: Dict[str, Dict[str, Any]], prev_auctions: Dict[str, Dict[str, Any]],
                            bank_state: Dict[str, Any]) -> Dict[str, int]:
    """Strategie: Wenn dran → setze ALLES auf die Zielauktion mit gegebener Rangposition."""
    bot_id = getattr(individual_bot_strategy, "bot_id", 0)
    target_rank = getattr(individual_bot_strategy, "target_auction_rank", 0)
    bid_turn = getattr(individual_bot_strategy, "bid_turn_in_cycle", 1)

    # Prüfen, ob diese Runde "unsere" ist (1,11,21,... für Turn 1)
    is_turn = ((current_round - 1) % 10) == (bid_turn - 1)
    if not is_turn:
        return {}

    # Kein Auktionsangebot → nix tun
    if not auctions:
        return {}

    agent_state = states.get(agent_id, {})
    gold = int(agent_state.get("gold", 0))
    if gold <= 0:
        return {}

    ranked = rank_auctions_by_value(auctions)
    if len(ranked) <= target_rank:
        return {}

    target_auction_id, _, ev = ranked[target_rank]

    # ALLES setzen – kein Limit, keine Reserve
    bid_amount = gold
    print(f"[Bot {bot_id}] 🔥 Round {current_round}: ALL-IN {bid_amount} auf {target_auction_id} (Rank {target_rank+1}, EV={ev:.1f})")

    return {target_auction_id: bid_amount}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python individual_bot.py <bot_id> <target_auction_rank> <bid_turn_in_cycle>")
        sys.exit(1)

    bot_id = int(sys.argv[1])
    target_rank = int(sys.argv[2])
    bid_turn = int(sys.argv[3])

    individual_bot_strategy.bot_id = bot_id
    individual_bot_strategy.target_auction_rank = target_rank
    individual_bot_strategy.bid_turn_in_cycle = bid_turn

    host = "opentsetlin.com"
    port = 8000
    agent_name = f"bot_{bot_id}_rank{target_rank+1}_turn{bid_turn}"
    player_id = f"bot_player_{bot_id}"

    print(f"🤖 Starting ALL-IN Bot #{bot_id}")
    print(f"  ➜ Target auction rank: {target_rank+1}")
    print(f"  ➜ Bid every 10 rounds, on turn: {bid_turn}")
    print()

    game = AuctionGameClient(host=host, agent_name=agent_name, player_id=player_id, port=port)

    try:
        game.run(individual_bot_strategy)
    except KeyboardInterrupt:
        print(f"<Bot {bot_id} interrupted>")
    except Exception as e:
        print(f"<Bot {bot_id} error: {e}>")

    print(f"<Bot {bot_id} finished>")
