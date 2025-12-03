"""Balanced bot tuned for the DND auction launcher."""

import random


ESTIMATED_TOTAL_ROUNDS = 1000


class BalancedAuctionBot:
    def __init__(self, agent_id: str, estimated_rounds: int = ESTIMATED_TOTAL_ROUNDS):
        self.agent_id = agent_id
        self.estimated_rounds = estimated_rounds
        self.rounds_played = 0
        self.items_won = []

    def _auction_score(self, auction: dict) -> float:
        """Compute a rough attractiveness score for an auction item."""
        if "dice" in auction:
            base = sum((sides + 1) / 2 for sides in auction["dice"])
        else:
            die_size = auction.get("die", 0)
            num_dice = max(auction.get("num", 1), 1)
            base = ((die_size + 1) / 2) * num_dice
        bonus = auction.get("bonus", 0)
        rarity = auction.get("rarity", 1)
        return base + bonus + rarity

    def make_bid(self,
                 agent_id: str,
                 round: int,
                 states: dict,
                 auctions: dict,
                 prev_auctions: dict,
                 pool: dict,
                 prev_pool_buys: dict,
                 bank_state: dict) -> dict:
        self.rounds_played += 1
        my_state = states.get(agent_id, {})
        gold = my_state.get("gold", 0)
        points = my_state.get("points", 0)

        bids = {}
        pool_buys = {}

        if auctions and gold > 0:
            phase_ratio = min(round / max(self.estimated_rounds, 1), 1.0)
            reserve_fraction = max(0.05, 0.25 * (1 - phase_ratio))
            spendable_gold = max(int(gold * (1 - reserve_fraction)), 1)

            scores = {aid: self._auction_score(data) for aid, data in auctions.items()}
            total_score = sum(scores.values())
            if total_score <= 0:
                total_score = len(scores)

            remaining = min(spendable_gold, gold)
            auction_items = list(auctions.items())
            for idx, (aid, _) in enumerate(auction_items):
                if idx == len(auction_items) - 1:
                    bid_amount = remaining
                else:
                    share = scores.get(aid, 1) / total_score
                    bid_amount = int(share * spendable_gold)
                    bid_amount = max(bid_amount, 1)
                    jitter = random.randint(0, 2)
                    bid_amount = min(bid_amount + jitter, remaining)

                if bid_amount <= 0:
                    continue
                bids[aid] = bid_amount
                remaining -= bid_amount
                if remaining <= 0:
                    break

        if isinstance(pool, dict) and pool and points > 0:
            pool_budget = max(int(points * 0.4), 1)
            total_cost = sum(item.get("cost", 0) for item in pool.values()) or 1
            remaining_points = min(pool_budget, points)

            pool_items = list(pool.items())
            for idx, (pid, item) in enumerate(pool_items):
                if idx == len(pool_items) - 1:
                    spend = remaining_points
                else:
                    share = item.get("cost", 0) / total_cost
                    spend = int(share * pool_budget)
                    spend = max(spend, 1)
                    spend = min(spend, remaining_points)
                if spend <= 0:
                    continue
                pool_buys[pid] = spend
                remaining_points -= spend
                if remaining_points <= 0:
                    break

        return {"bids": bids, "pool_buys": pool_buys}

    def win_item(self, item, price):
        self.items_won.append(item)

