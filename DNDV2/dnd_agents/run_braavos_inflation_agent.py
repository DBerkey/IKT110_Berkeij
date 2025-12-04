#!/usr/bin/env python3
"""Launcher for the BraavosInflationAgent."""

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
        from dnd_agents.BraavosInflationAgent import BraavosInflationAgent  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Unable to import BraavosInflationAgent. Ensure the project root is on PYTHONPATH"
        ) from exc
    return BraavosInflationAgent


def run_agent(host: str, port: int, agent_name: str, player_id: str) -> None:
    AgentClass = _load_agent_class()
    client = AuctionGameClient(host=host, port=port, agent_name=agent_name, player_id=player_id)
    agent = AgentClass()

    try:
        client.run(agent.make_bid)
    except KeyboardInterrupt:  # pragma: no cover - runtime guard
        print("<BraavosInflationAgent interrupted>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BraavosInflationAgent")
    parser.add_argument("--host", default="localhost", help="Auction server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Auction server port (default: 8000)")
    parser.add_argument(
        "--agent-name",
        default="braavos_inflation_agent",
        help="Display name shown on the leaderboard",
    )
    parser.add_argument(
        "--player-id",
        default="braavos_inflation_player",
        help="Player id recorded in logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_agent(args.host, args.port, args.agent_name, args.player_id)


if __name__ == "__main__":  # pragma: no cover
    main()
