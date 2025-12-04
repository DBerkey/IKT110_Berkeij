import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot point totals and rolling gains for a specific agent from an auction log."
    )
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help="Path to the auction_house_log_*.jsonln file to analyze.",
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="Agent identifier (e.g., local_rand_id_720650) whose points should be plotted.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling window (in rounds) for measuring point increases (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the resulting plot. If omitted, the plot is displayed interactively.",
    )
    return parser.parse_args()


def load_points(log_path: Path, agent_id: str) -> List[Tuple[int, float]]:
    points_by_round: List[Tuple[int, float]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            round_no = entry.get("round")
            if round_no is None:
                continue
            states = entry.get("states", {})
            state = states.get(agent_id)
            if not state:
                continue
            points = float(state.get("points", 0) or 0)
            points_by_round.append((int(round_no), points))
    points_by_round.sort(key=lambda item: item[0])
    return points_by_round


def compute_rolling(points: List[Tuple[int, float]], window: int) -> List[float]:
    if window <= 0:
        raise ValueError("Window size must be positive")
    gains: List[float] = []
    for idx, (_, pts) in enumerate(points):
        base_idx = max(0, idx - window)
        base_pts = points[base_idx][1]
        gains.append(pts - base_pts)
    return gains


def plot_points(
    rounds: List[int],
    totals: List[float],
    rolling: List[float],
    window: int,
    agent_id: str,
    output_path: Path | None,
) -> None:
    fig, (ax_total, ax_gain) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    ax_total.plot(rounds, totals, color="tab:blue", label="Points")
    ax_total.set_ylabel("Points")
    ax_total.set_title(f"Point trajectory for {agent_id}")
    ax_total.grid(True, alpha=0.3)

    ax_gain.plot(rounds, rolling, color="tab:orange", label=f"Δ points (last {window} rounds)")
    ax_gain.set_ylabel("Rolling gain")
    ax_gain.set_xlabel("Round")
    ax_gain.grid(True, alpha=0.3)

    ax_total.legend(loc="upper left")
    ax_gain.legend(loc="upper left")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    points = load_points(args.log, args.agent_id)
    if not points:
        raise SystemExit(f"No states found for agent_id '{args.agent_id}' in {args.log}")

    rounds = [round_no for round_no, _ in points]
    totals = [pts for _, pts in points]
    rolling = compute_rolling(points, args.window)
    plot_points(rounds, totals, rolling, args.window, args.agent_id, args.output)


if __name__ == "__main__":
    main()
