
import random
import math
from typing import Any, Dict, List

class CompoundCrusherAgent:
    def __init__(self):
        self.round_history: List[float] = []
        # Standardverdier (oppdateres av serveren)
        self.interest_rate = 0.10
        self.interest_cap = 1000
        self.rounds_remaining = 1000

    def _update_target_stats(self, prev_auctions: Dict[str, Any]):
        """Analyserer forrige runde for å se hva ting gikk for."""
        wins = []
        for auction in prev_auctions.values():
            bids = auction.get("bids", [])
            if bids:
                wins.append(float(bids[0].get("gold", 0)))

        # Holder historikk kort for å se trender
        self.round_history.extend(wins)
        if len(self.round_history) > 20:
            self.round_history = self.round_history[-20:]

    def calculate_ev(self, auction: Dict[str, Any]) -> float:
        """Regner ut forventet poengverdi (Expected Value)."""
        die = int(auction.get("die", 6))
        num = int(auction.get("num", 1))
        bonus = int(auction.get("bonus", 0))
        avg_roll = (die + 1) / 2
        return (avg_roll * num) + bonus

    def decide(self,
               agent_id: str,
               states: Dict[str, Any],
               auctions: Dict[str, Any],
               prev_auctions: Dict[str, Any],
               pool_gold: int,
               bank_state: Dict[str, Any]) -> (Dict[str, int], int):

        # --- 1. Oppdater Kunnskap ---
        me = states.get(agent_id, {})
        my_gold = int(me.get("gold", 0))

        if prev_auctions:
            self._update_target_stats(prev_auctions)

        # Hent bank-info
        rates = bank_state.get("bank_interest_per_round", [])
        limits = bank_state.get("bank_limit_per_round", [])

        if rates:
            self.interest_rate = float(rates[0])
            self.rounds_remaining = len(rates)
        if limits:
            self.interest_cap = int(limits[0])

        if my_gold <= 0:
            return {}, 0

        # --- 2. Beregn "Opportunity Cost" (Matte-fiksen) ---

        effective_rounds = self.rounds_remaining
        compound_factor = 1.0

        # Hvis vi har plass til å vokse (under taket) og rente > 0
        if my_gold < self.interest_cap and self.interest_rate > 0:
            try:
                # Hvor mange ganger kan pengene doble seg før taket?
                # Formel: t = ln(Cap / Gold) / ln(1 + r)
                growth_room = self.interest_cap / max(1, my_gold)
                rounds_to_cap = math.log(growth_room) / math.log(1 + self.interest_rate)

                # Vi planlegger bare frem til vi treffer taket.
                effective_rounds = min(self.rounds_remaining, rounds_to_cap)

                # Beregn faktoren: Hvor mye blir 1 gull verdt når vi når taket?
                compound_factor = (1 + self.interest_rate) ** effective_rounds

            except (ValueError, ZeroDivisionError):
                compound_factor = 1.0
        else:
            # Vi er over taket (eller rente er 0). Ingen vits å spare.
            compound_factor = 1.0
            effective_rounds = 0

        # --- 3. Vurder Auksjoner ---
        bids = {}
        remaining_gold = my_gold

        sorted_auctions = []
        for aid, a in auctions.items():
            ev = self.calculate_ev(a)
            sorted_auctions.append((ev, aid))
        sorted_auctions.sort(reverse=True)

        for ev, aid in sorted_auctions:
            if remaining_gold <= 1:
                break

            base_valuation = ev * 3.5

            # Kalkuler maks bud
            if my_gold < self.interest_cap:
                # SPAREMODUS:
                max_bid = base_valuation / (compound_factor * 0.7)
            else:
                # OVERSKUDDSMODUS:
                surplus = my_gold - self.interest_cap
                max_bid = base_valuation + (surplus * 0.4)

            bid = int(max_bid)

            if bid < 1 and max_bid > 0.6:
                bid = 1

            noise = random.uniform(0.9, 1.1)
            bid = int(bid * noise)
            bid = max(0, min(bid, remaining_gold))

            if bid > 0:
                bids[aid] = bid
                remaining_gold -= bid

        # --- 4. Pool Strategi ---
        pool_pts = 0
        if my_gold > (self.interest_cap * 1.5) and pool_gold > 200:
             pool_pts = 10

        return bids, pool_pts


# --- KOBLING MOT SPILLMOTOR ---
AGENT = CompoundCrusherAgent()

def make_bid(agent_id, round, states, auctions, prev_auctions, pool, prev_pool_buys, bank_state):
    # Fikset syntax-feil her:
    bids, pool_pts = AGENT.decide(
        agent_id,
        states,
        auctions,
        prev_auctions,
        pool,
        bank_state
    )
    return {"bids": bids, "pool": pool_pts}


if __name__ == "__main__":
    from dnd_auction_game import AuctionGameClient

    HOST = "127.0.0.1"
    PORT = 8000
    NAME = "Compound_Crusher_v2"

    print(f"[STARTUP] Connecting {NAME} to {HOST}:{PORT}")
    client = AuctionGameClient(host=HOST, port=PORT, agent_name=NAME, player_id=NAME)

    try:
        client.run(make_bid)
    except Exception as e:
        print(f"[ERROR] {e}")