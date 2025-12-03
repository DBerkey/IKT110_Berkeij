import random
from dnd_auction_game import AuctionGameClient

class AggressiveDebtBot:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def make_bid(self, agent_id, round, states, auctions, prev_auctions, pool,
                 prev_pool_buys, bank_state):
        my_state = states[agent_id]
        my_gold = my_state["gold"]
        my_points = my_state["points"]

        bids = {}
        pool_buys = {}

        # === AUCTION BIDS ===
        if auctions:
            # Aggressive: push beyond available gold to leverage debt
            target_spend = int(my_gold * 1.2) if my_gold > 0 else 10
            max_spend = max(target_spend, my_gold)

            # Estimate expected value for each auction
            evs = {}
            total_ev = 0
            for aid, a in auctions.items():
                ev = (a["die"] + 1) / 2 * a["num"] + a.get("bonus", 0)
                evs[aid] = ev
                total_ev += ev

            remaining = max_spend
            n = len(auctions)
            for idx, (aid, _) in enumerate(auctions.items()):
                if idx == n - 1:
                    bid = remaining
                else:
                    share = (evs[aid] / total_ev) if total_ev > 0 else (1 / n)
                    bid = int(share * max_spend)
                    bid = max(bid, 1)
                    bid = min(bid, remaining)
                bids[aid] = bid
                remaining -= bid
                if remaining <= 0:
                    break

        # === POOL BIDS ===
        if isinstance(pool, dict) and pool and my_points > 0:
            # Only spend up to available points (no negative points)
            max_points_to_spend = int(my_points * 0.8)  # aggressive pool spend
            total_pool_cost = sum(item.get("cost", 0) for item in pool.values()) or 1
            for pid, item in pool.items():
                share = item.get("cost", 0) / total_pool_cost
                spend = int(share * max_points_to_spend)
                spend = max(spend, 1)
                # Cannot spend more than we have
                spend = min(spend, my_points)
                pool_buys[pid] = spend

        return {"bids": bids, "pool_buys": pool_buys}
