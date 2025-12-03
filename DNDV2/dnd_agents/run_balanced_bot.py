#!/usr/bin/env python3
"""Entry point for running the BalancedAuctionBot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dnd_auction_game import AuctionGameClient  # type: ignore[import]


def _load_agent_class():
    agent_dir = Path(__file__).resolve().parent
    project_root = agent_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from dnd_agents.balanced_bot import BalancedAuctionBot  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Unable to import balanced_bot. Ensure the project root is on PYTHONPATH."
        ) from exc
    return BalancedAuctionBot


def run_agent(host: str, port: int, agent_name: str, player_id: str) -> None:
    BalancedBotClass = _load_agent_class()
    client = AuctionGameClient(
        host=host,
        port=port,
        agent_name=agent_name,
        player_id=player_id,
    )
    agent = BalancedBotClass(client.agent_id)

    try:
        client.run(agent.make_bid)
    except KeyboardInterrupt:
        print("<BalancedAuctionBot interrupted>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BalancedAuctionBot")
    parser.add_argument("--host", default="localhost", help="Auction server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Auction server port (default: 8000)")
    parser.add_argument("--agent-name", default="balanced_bot", help="Display name for this agent")
    parser.add_argument("--player-id", default="balanced_bot_player", help="Player id used for tracking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_agent(args.host, args.port, args.agent_name, args.player_id)


if __name__ == "__main__":
    main()
