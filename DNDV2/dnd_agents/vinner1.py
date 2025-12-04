
from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List

# Standard spillkonstanter (kan justeres hvis serveren sender andre verdier)
DEFAULT_INTEREST_RATE = 0.10
DEFAULT_INTEREST_CAP = 1000  # Etter dette tjener man ikke mer renter
TOTAL_ROUNDS_ESTIMATE = 1000


class CompoundCrusherAgent:
    def __init__(self):
        self.round_history: List[float] = []
        self.target_win_avg = 0.0  # Estimat av Targets "recent_win_avg"
        self.my_gold = 0.0
        self.my_points = 0.0
        self.median_points = 0.0
        self.interest_cap = DEFAULT_INTEREST_CAP
        self.interest_rate = DEFAULT_INTEREST_RATE

    def _update_target_stats(self, prev_auctions: Dict[str, Any]):
        """Sporer hva ting går for, for å forutsi Targets 'bid floor'."""
        wins = []
        for auction in prev_auctions.values():
            bids = auction.get("bids", [])
            if bids:
                wins.append(float(bids[0].get("gold", 0)))

        # Target bruker en 30-runders buffer, vi prøver å emulere det
        self.round_history.extend(wins)
        if len(self.round_history) > 30:
            self.round_history = self.round_history[-30:]

        if self.round_history:
            self.target_win_avg = sum(self.round_history) / len(self.round_history)
        else:
            self.target_win_avg = 0.0

    def _calculate_ev(self, auction: Dict[str, int]) -> float:
        """Beregner faktisk verdi (Expected Value)."""
        die = auction.get("die", 0)
        num = auction.get("num", 0)
        bonus = auction.get("bonus", 0)
        avg_roll = (die + 1) / 2
        return (avg_roll * num) + bonus

    def _calculate_variance(self, auction: Dict[str, int]) -> float:
        """Beregner varians (risiko)."""
        die = auction.get("die", 0)
        num = auction.get("num", 0)
        if die <= 1: return 0.0
        return num * ((die ** 2 - 1) / 12)

    def decide(self,
               agent_id: str,
               round_number: int,
               states: Dict[str, Dict[str, float]],
               auctions: Dict[str, Dict[str, int]],
               prev_auctions: Dict[str, Any],
               info: Any) -> Dict[str, Any]:

        # 1. Oppdater tilstand
        self.my_gold = float(states[agent_id].get("gold", 0))
        self.my_points = float(states[agent_id].get("points", 0))

        all_points = [s.get("points", 0) for s in states.values()]
        self.median_points = statistics.median(all_points) if all_points else 0

        self._update_target_stats(prev_auctions)

        # Hent bank-info hvis tilgjengelig, ellers bruk defaults
        # info er ofte remainder_random_info fra serveren
        if isinstance(info, dict) and "bank" in info:
            self.interest_rate = info["bank"].get("rate", self.interest_rate)
            self.interest_cap = info["bank"].get("limit", self.interest_cap)

        # 2. Bestem Fase (Basert på gjenværende runder, ikke faste tall)
        rounds_left = max(1, TOTAL_ROUNDS_ESTIMATE - round_number)

        # End-game trigger: Når vi bør begynne å tømme kontoen
        # Vi vil tømme kontoen gradvis over de siste 10-15% av spillet
        is_endgame = rounds_left < 150

        bids: Dict[str, int] = {}

        # Sorter auksjoner etter EV (verdi)
        auction_list = []
        for aid, item in auctions.items():
            ev = self._calculate_ev(item)
            var = self._calculate_variance(item)
            auction_list.append((aid, item, ev, var))

        auction_list.sort(key=lambda x: x[2], reverse=True)  # Høyest EV først

        # 3. Kjerne-strategi
        for aid, item, ev, var in auction_list:

            # --- A. VARIANS JUSTERING ---
            # Hvis vi ligger bak, elsker vi varians (Hail Mary).
            # Hvis vi leder, hater vi varians (Play Safe).
            adjusted_value = ev
            if self.my_points < self.median_points - 20:
                adjusted_value += (var * 0.1)  # Bonus for varians
            elif self.my_points > self.median_points + 20:
                adjusted_value -= (var * 0.05)  # Straff for varians

            # --- B. BUD-KALKULASJON ---

            if is_endgame:
                # MODUS: KILLER (Tøm banken)
                # Vi har spart hele spillet, nå skal vi utby Targets reserve-logikk.
                # Vi fordeler gullet på gjenværende runder, men er aggressive.
                budget_per_round = self.my_gold / max(1, rounds_left / 2)  # Tøm dobbelt så fort som nødvendig
                bid = min(self.my_gold, budget_per_round + (adjusted_value * 2))

            else:
                # MODUS: COMPOUND (Spar og Snip)

                # SJEKK FOR FELLEN: Er Target tvunget til å overby?
                # Target har en 'bid floor' på ca 90% av recent_win_avg.
                # Hvis EV er lav (f.eks 5), men recent_win_avg er høy (f.eks 20),
                # vil Target by ca 18. Dette er idioti. Vi byr 0.
                if ev < (self.target_win_avg * 0.7):
                    # Dette er en felle for Target. La dem kjøpe søpla dyrt.
                    continue

                    # Hva er dette verdt for oss?
                # Hver gullmynt brukt NÅ koster oss renter for alltid.
                # Opportunity cost: 1 gull i dag = (1+rente)^runder gull i sluttspillet.
                # Vi byr DERFOR KUN hvis prisen er latterlig lav (Sniping).

                if self.my_gold < self.interest_cap:
                    # Vi er i "Oppbyggingsfase".
                    # Vi byr ekstremt konservativt. Kun røverkjøp.
                    valuation = adjusted_value * 1.5  # Gullkonvertering
                    max_willing_to_pay = valuation * 0.4  # Vil kun betale 40% av verdi
                else:
                    # Vi har nådd rentetaket! Overskytende gull brenner i lomma.
                    # Bruk overskuddet ("Surplus") til å plage Target.
                    surplus = self.my_gold - self.interest_cap
                    valuation = adjusted_value * 2.5
                    max_willing_to_pay = min(surplus, valuation)

                bid = max_willing_to_pay

            # Legg på støy for å ikke være forutsigbar
            bid = int(bid * random.uniform(0.95, 1.05))

            # Sikkerhetssjekk
            if bid > self.my_gold:
                bid = self.my_gold

            if bid > 1:
                bids[aid] = int(bid)

        # 4. Pool Strategi (Aggressivt kjøp ved overskudd)
        # Target kjøper bare hvis den leder ("Win-More").
        # Vi kjøper hvis det er matematisk lønnsomt ELLER vi trenger å tømme overskudd.
        pool_bid = 0

        # Hvis vi har makset renten, konverter overskuddsgull til poeng hvis poolen er billig
        if self.my_gold > self.interest_cap:
            pool_bid = 1  # Token bid to participate if logic allows
            # (Enklere pool logikk her for å holde koden ren -
            # hovedpoenget er auksjonsdominans)

        return {"bids": bids, "pool": pool_bid}


# Global instans
AGENT = CompoundCrusherAgent()


def make_bid(
    agent_id: str,
    round: int,
    states: Dict[str, Any],
    auctions: Dict[str, Any],
    prev_auctions: Dict[str, Any],
    pool: Any,
    prev_pool_buys: Any,
    bank_state: Any,
    remainder_random_info: Any = None,
) -> Dict[str, Any]:
    info_dict: Dict[str, Any] = {}

    if isinstance(bank_state, dict):
        info_dict["bank"] = bank_state

    # Preserve backwards compatibility if remainder info is passed when testing locally
    if isinstance(remainder_random_info, dict):
        info_dict.update(remainder_random_info)
    elif isinstance(remainder_random_info, list):
        for item in remainder_random_info:
            if isinstance(item, dict):
                info_dict.update(item)

    return AGENT.decide(
        agent_id=agent_id,
        round_number=round,
        states=states,
        auctions=auctions,
        prev_auctions=prev_auctions,
        info=info_dict,
    )


if __name__ == "__main__":
    # Standard tilkoblingskode for testing
    try:
        from dnd_auction_game import AuctionGameClient
    except ImportError:
        print("Mangler spillmotor bibliotek.")
        exit()

    host = "127.0.0.1"
    agent_name = "thewinner"
    player_id = "maksymilian dunajski"
    port = 8000

    game = AuctionGameClient(host=host, agent_name=agent_name, player_id=player_id, port=port)
    game.run(make_bid)