#!/usr/bin/env python3
"""Launcher for BetterMeanBidderAgent."""

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
        from dnd_agents.BetterMeanBidderAgent import BetterMeanBidderAgent  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError("Unable to import BetterMeanBidderAgent") from exc
    return BetterMeanBidderAgent


def run_agent(host: str, port: int, agent_name: str, player_id: str) -> None:
    AgentClass = _load_agent_class()
    client = AuctionGameClient(host=host, port=port, agent_name=agent_name, player_id=player_id)
    agent = AgentClass(client.agent_id)
    try:
        client.run(agent.make_bid)
    except KeyboardInterrupt:
        print("<BetterMeanBidderAgent interrupted>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BetterMeanBidderAgent")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--agent-name", default="better_mean_bidder")
    parser.add_argument("--player-id", default="better_mean_bidder_player")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_agent(args.host, args.port, args.agent_name, args.player_id)


if __name__ == "__main__":
    main()
