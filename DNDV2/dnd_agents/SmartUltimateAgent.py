"""SmartUltimateAgent implementation compatible with the launcher API."""


class SmartUltimateAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.name = "SmartUltimateAgent"
        self.rounds_played = 0

    def _expected_points(self, auction: dict) -> float:
        """Estimate auction value using whichever schema the server provides."""
        if "dice" in auction:
            base = sum((sides + 1) / 2 for sides in auction["dice"])
        else:
            die_size = auction.get("die", 0)
            num_dice = auction.get("num", 1)
            base = ((die_size + 1) / 2) * num_dice
        return base + auction.get("bonus", 0)

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
        if gold <= 0 or not auctions:
            return {"bids": {}}

        bank_fraction = 0.2 if bank_state else 0
        spendable_gold = max(int(gold * (1 - bank_fraction)), 1)

        aggression = min(1.0, 0.2 + 0.1 * self.rounds_played)
        auction_values = {
            aid: self._expected_points(data) * aggression
            for aid, data in auctions.items()
        }
        total_value = sum(auction_values.values())
        if total_value <= 0:
            return {"bids": {}}

        bids = {}
        remaining_gold = gold
        for aid, value in auction_values.items():
            portion = value / total_value
            bid_amount = max(1, int(spendable_gold * portion))
            bid_amount = min(bid_amount, remaining_gold)
            if bid_amount <= 0:
                continue
            bids[aid] = bid_amount
            remaining_gold -= bid_amount
            if remaining_gold <= 0:
                break

        return {"bids": bids}
