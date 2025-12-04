# runner.py
from moha import make_bid  # Import the separated agent logic
from dnd_auction_game import AuctionGameClient

host = "localhost"
agent_name = "moha_ml"
player_id = "moha_ml"
port = 8000


# Callback that the game client calls every round
def _entry(
    agent_id,
    round_no,
    states,
    auctions,
    prev_auctions,
    pool,
    prev_pool_buys,
    bank_state,
):
    return make_bid(
        agent_id,
        round_no,
        states,
        auctions,
        prev_auctions,
        pool,
        prev_pool_buys,
        bank_state,
    )


if __name__ == "__main__":
    game = AuctionGameClient(
        host=host, agent_name=agent_name, player_id=player_id, port=port
    )
    try:
        game.run(_entry)  # Run the game loop; calls _entry each round
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")
