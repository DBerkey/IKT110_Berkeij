import random
from dnd_auction_game import AuctionGameClient

class AggressiveBot:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def make_bid(self, agent_id, round, states, auctions, prev_auctions, pool,
                 prev_pool_buys, bank_state):
        my_state = states[agent_id]
        my_gold = my_state["gold"]
        my_points = my_state["points"]

        bids = {}
        # Strategy: bid on all auctions, allocate a fixed fraction of available gold
        # E.g. spend up to 50–70% of gold this round (aggressive).
        max_spend = int(my_gold * 0.6)

        # Split evenly (or weighted by expected value) among auctions
        n = len(auctions)
        if n == 0 or max_spend <= 0:
            return {}

        # Estimate expected value for each auction: EV ≈ (die + 1)/2 * num + bonus
        evs = {}
        total_ev = 0
        for aid, a in auctions.items():
            ev = (a["die"] + 1) / 2 * a["num"] + a.get("bonus", 0)
            evs[aid] = ev
            total_ev += ev

        for aid, a in auctions.items():
            ev = evs[aid]
            # proportionally allocate gold based on EV
            share = ev / total_ev
            bid = int(share * max_spend)
            # At least 1 gold
            bid = max(bid, 1)
            # But no more than we have
            if bid > my_gold:
                bid = my_gold
            bids[aid] = bid

        return {"bids": bids}

