#!/usr/bin/env python3
"""Launcher for RandomSplashBidderAgent."""

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
        from dnd_agents.RandomSplashBidderAgent import RandomSplashBidderAgent  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Unable to import RandomSplashBidderAgent") from exc
    return RandomSplashBidderAgent


def run_agent(host: str, port: int, agent_name: str, player_id: str) -> None:
    AgentClass = _load_agent_class()
    client = AuctionGameClient(host=host, port=port, agent_name=agent_name, player_id=player_id)
    agent = AgentClass(client.agent_id)
    try:
        client.run(agent.make_bid)
    except KeyboardInterrupt:
        print("<RandomSplashBidderAgent interrupted>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RandomSplashBidderAgent")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--agent-name", default="random_splash_bidder")
    parser.add_argument("--player-id", default="random_splash_bidder_player")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_agent(args.host, args.port, args.agent_name, args.player_id)


if __name__ == "__main__":
    main()
