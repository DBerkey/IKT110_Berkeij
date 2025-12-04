"""
AggressiveSpenderAgent:
- Spends most available gold on top EV targets
- Rarely uses pool; prefers auctions
- Strong jitter to compete
"""
import random
from typing import Dict, Any, Tuple

AVG = {2:1.5,3:2.0,4:2.5,6:3.5,8:4.5,10:5.5,12:6.5,20:10.5}

class AggressiveSpenderAgent:
    def __init__(self):
        self.cp = 28.0
        self.alpha = 0.3

    def update_cp(self, prev_auctions: Dict[str,Any]):
        vals = []
        for a in prev_auctions.values():
            r = int(a.get("reward",0))
            bids = a.get("bids", [])
            if r>0 and bids:
                vals.append(int(bids[0]["gold"])/r)
        if vals:
            med = sorted(vals)[len(vals)//2]
            self.cp = (1-self.alpha)*self.cp + self.alpha*med

    def decide(self, agent_id: str, states: Dict[str,Any], auctions: Dict[str,Any], prev_auctions: Dict[str,Any], pool_gold: int, prev_pool_buys: Dict[str,Any]) -> Tuple[Dict[str,int], int]:
        if prev_auctions:
            self.update_cp(prev_auctions)
        me = states.get(agent_id, {})
        gold = int(me.get("gold",0))
        points = int(me.get("points",0))
        try:
            print(f"[AGG][INFO] gold={gold} points={points} cp={self.cp:.1f} auctions={len(auctions)}")
        except Exception:
            pass
        if gold<=0:
            return {}, 0
        scored = []
        for a_id, a in auctions.items():
            die = int(a.get("die",6)); num=int(a.get("num",1)); bonus=int(a.get("bonus",0))
            ev = num*AVG.get(die,3.5)+bonus
            if ev>0:
                scored.append((ev, a_id))
        scored.sort(reverse=True)
        bids: Dict[str,int] = {}
        remaining = gold
        top = scored[:8]
        for ev, a_id in top:
            fair = ev*self.cp
            target = int(max(1, fair*0.9))   # aggressive 90%
            per = max(1, remaining//(len(top) or 1))
            bid = min(target, per)
            bid = int(bid * random.uniform(0.85, 1.25))
            bid = max(1, min(bid, remaining))
            bids[a_id] = bid
            try:
                print(f"[AGG][BID] {a_id} bid={bid} EV={ev:.2f} fair≈{fair:.2f} rem={remaining-bid}")
            except Exception:
                pass
            remaining -= bid
            if remaining<=0:
                break
        # Pool: only if we have huge point lead
        pool_pts = 0
        if isinstance(prev_pool_buys, dict) and pool_gold>0:
            others = [int(s.get("points",0)) for aid,s in states.items() if aid!=agent_id]
            lead = points - (max(others) if others else 0)
            if lead>400 and points>500:
                pool_pts = min(50, (points-500)//10)
        return bids, pool_pts


def make_bid(
    agent_id: str,
    current_round: int,
    states: Dict[str, Any],
    auctions: Dict[str, Any],
    prev_auctions: Dict[str, Any],
    pool_gold: Any,
    prev_pool_buys: Any,
    bank_state: Any,
) -> Dict[str, Any]:
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = AggressiveSpenderAgent()

    safe_pool_gold = int(pool_gold) if isinstance(pool_gold, (int, float)) else 0
    safe_prev_pool_buys: Dict[str, Any] = {}
    if isinstance(prev_pool_buys, dict):
        safe_prev_pool_buys = prev_pool_buys

    bids, pool_points = make_bid._agent.decide(
        agent_id,
        states,
        auctions,
        prev_auctions,
        safe_pool_gold,
        safe_prev_pool_buys,
    )
    return {"bids": bids, "pool": pool_points}


if __name__ == "__main__":
    from dnd_auction_game import AuctionGameClient
    host = "127.0.0.1"
    agent_name = "Aggressive_Spender"
    player_id = "Aggressive_Spender_Player"
    port = 8000
    print(f"[STARTUP] Connecting to {host}:{port}")
    print(f"[STARTUP] Agent: {agent_name}")
    print(f"[STARTUP] Player: {player_id}")
    game = AuctionGameClient(host=host, agent_name=agent_name, player_id=player_id, port=port)
    try:
        print("[STARTUP] Starting agent...")
        game.run(make_bid)
    except KeyboardInterrupt:
        print("<interrupt>")
    except Exception as e:
        print(f"[ERROR] Agent crashed: {e}")
        import traceback; traceback.print_exc()
    print("<game done>")
