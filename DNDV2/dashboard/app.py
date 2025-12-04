#!/usr/bin/env python3
"""Dash dashboard for steering the Braavos Optimal agent in real time."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence

import dash
import dash_daq as daq
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots
import plotly.graph_objects as go

DEFAULT_CONFIG: Dict[str, float | bool] = {
    "force_optimal": False,
    "spend_fraction_scale": 1.0,
    "pool_spend_multiplier": 1.0,
    "auction_aggression_scale": 1.0,
    "reserve_padding_scale": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Braavos Optimal control dashboard")
    parser.add_argument("--log", required=True, type=Path, help="Path to an auction_house_log_*.jsonln file")
    parser.add_argument("--agent-id", required=True, help="Agent identifier to track (e.g. local_rand_id_799865)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "braavos_optimal_runtime.json",
        help="Runtime override config file (defaults to DNDV2/braavos_optimal_runtime.json)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dash server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Dash server port (default: 8050)")
    parser.add_argument("--refresh-ms", type=int, default=4000, help="Graph refresh interval in milliseconds")
    return parser.parse_args()


def ensure_config_file(path: Path) -> None:
    if path.exists():
        return
    path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")


def read_runtime_config(path: Path) -> Dict[str, float | bool]:
    ensure_config_file(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_CONFIG.copy()
    config: Dict[str, float | bool] = DEFAULT_CONFIG.copy()
    config.update({
        "force_optimal": bool(data.get("force_optimal", False)),
        "spend_fraction_scale": _clamp(float(data.get("spend_fraction_scale", 1.0)), 0.2, 2.0),
        "pool_spend_multiplier": _clamp(float(data.get("pool_spend_multiplier", 1.0)), 0.0, 3.0),
        "auction_aggression_scale": _clamp(float(data.get("auction_aggression_scale", 1.0)), 0.3, 2.5),
        "reserve_padding_scale": _clamp(float(data.get("reserve_padding_scale", 1.0)), 0.2, 3.0),
    })
    return config


def write_runtime_config(path: Path, config: Dict[str, float | bool]) -> None:
    payload = {
        "force_optimal": bool(config["force_optimal"]),
        "spend_fraction_scale": float(config["spend_fraction_scale"]),
        "pool_spend_multiplier": float(config["pool_spend_multiplier"]),
        "auction_aggression_scale": float(config["auction_aggression_scale"]),
        "reserve_padding_scale": float(config["reserve_padding_scale"]),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_log(log_path: Path, agent_id: str) -> Dict[str, List[float]] | None:
    if not log_path.exists():
        return None
    rounds: List[int] = []
    points: List[float] = []
    gold: List[float] = []
    medians: List[float] = []
    pool_spend: List[float] = []

    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                round_no = entry.get("round")
                states = entry.get("states", {})
                agent_state = states.get(agent_id)
                if round_no is None or not agent_state:
                    continue
                rounds.append(int(round_no))
                points.append(float(agent_state.get("points", 0) or 0))
                gold.append(float(agent_state.get("gold", 0) or 0))
                snapshot = [float(state.get("points", 0) or 0) for state in states.values()]
                medians.append(median(snapshot) if snapshot else 0.0)
                pool_delta = entry.get("prev_pool_buys", {}).get(agent_id, 0)
                pool_spend.append(float(pool_delta))
    except OSError:
        return None

    if not rounds:
        return None
    return {
        "rounds": rounds,
        "points": points,
        "gold": gold,
        "median": medians,
        "pool": pool_spend,
    }


def build_points_figure(series: Dict[str, Sequence[float]]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["rounds"],
            y=series["points"],
            name="Braavos points",
            mode="lines",
            line=dict(color="#2563eb", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["rounds"],
            y=series["median"],
            name="Table median",
            mode="lines",
            line=dict(color="#94a3b8", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
        yaxis_title="Points",
        xaxis_title="Round",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_gold_figure(series: Dict[str, Sequence[float]]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=series["rounds"],
            y=series["gold"],
            name="Gold",
            mode="lines",
            line=dict(color="#f59e0b", width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=series["rounds"],
            y=series["pool"],
            name="Pool spend (points)",
            marker_color="#0ea5e9",
            opacity=0.45,
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Gold", secondary_y=False)
    fig.update_yaxes(title_text="Pool spend", secondary_y=True)
    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_placeholder_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_layout(template="plotly_white", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def build_app(log_path: Path, agent_id: str, config_path: Path, refresh_ms: int) -> Dash:
    ensure_config_file(config_path)
    initial_config = read_runtime_config(config_path)
    app = Dash(__name__)
    app.title = "Braavos Optimal Dashboard"

    app.layout = html.Div(
        [
            html.H1("Braavos Optimal Control Center"),
            html.Div(
                id="run-info",
                className="run-info",
                children="Waiting for log updates...",
            ),
            html.Div(
                className="controls",
                children=[
                    html.Div(
                        [
                            html.Label("Force Optimal Logic"),
                            daq.BooleanSwitch(id="force-optimal", on=bool(initial_config["force_optimal"])),
                        ],
                        className="control-block",
                    ),
                    html.Div(
                        [
                            html.Label("Gold spend aggressiveness"),
                            dcc.Slider(
                                id="spend-slider",
                                min=0.4,
                                max=1.6,
                                step=0.05,
                                value=float(initial_config["spend_fraction_scale"]),
                                marks={0.4: "0.4×", 1.0: "1.0×", 1.6: "1.6×"},
                            ),
                            html.Div(id="spend-label", className="slider-value"),
                        ],
                        className="control-block",
                    ),
                    html.Div(
                        [
                            html.Label("Pool spend multiplier"),
                            dcc.Slider(
                                id="pool-slider",
                                min=0.0,
                                max=2.0,
                                step=0.05,
                                value=float(initial_config["pool_spend_multiplier"]),
                                marks={0.0: "0", 1.0: "1×", 2.0: "2×"},
                            ),
                            html.Div(id="pool-label", className="slider-value"),
                        ],
                        className="control-block",
                    ),
                    html.Div(
                        [
                            html.Label("Auction aggression scale"),
                            dcc.Slider(
                                id="aggression-slider",
                                min=0.6,
                                max=1.8,
                                step=0.05,
                                value=float(initial_config["auction_aggression_scale"]),
                                marks={0.6: "0.6×", 1.0: "1.0×", 1.8: "1.8×"},
                            ),
                            html.Div(id="aggression-label", className="slider-value"),
                        ],
                        className="control-block",
                    ),
                    html.Div(
                        [
                            html.Label("Reserve padding scale"),
                            dcc.Slider(
                                id="reserve-slider",
                                min=0.5,
                                max=2.0,
                                step=0.05,
                                value=float(initial_config["reserve_padding_scale"]),
                                marks={0.5: "0.5×", 1.0: "1.0×", 2.0: "2.0×"},
                            ),
                            html.Div(id="reserve-label", className="slider-value"),
                        ],
                        className="control-block",
                    ),
                ],
            ),
            html.Div(
                className="control-buttons",
                children=[
                    html.Button("Save overrides", id="save-btn", n_clicks=0),
                    html.Button("Reset to defaults", id="reset-btn", n_clicks=0),
                    html.Div(id="save-status", className="save-status"),
                ],
            ),
            dcc.Graph(id="points-graph"),
            dcc.Graph(id="gold-graph"),
            dcc.Interval(id="refresh-interval", interval=max(1000, refresh_ms), n_intervals=0),
        ]
    )

    @app.callback(
        Output("force-optimal", "on"),
        Output("spend-slider", "value"),
        Output("pool-slider", "value"),
        Output("aggression-slider", "value"),
        Output("reserve-slider", "value"),
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        State("force-optimal", "on"),
        State("spend-slider", "value"),
        State("pool-slider", "value"),
        State("aggression-slider", "value"),
        State("reserve-slider", "value"),
        prevent_initial_call=True,
    )
    def handle_controls(
        save_clicks: int | None,
        reset_clicks: int | None,
        force: bool,
        spend: float,
        pool_mult: float,
        aggression: float,
        reserve_scale: float,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        timestamp = datetime.now().strftime("%H:%M:%S")

        if trigger == "save-btn":
            config = {
                "force_optimal": force,
                "spend_fraction_scale": spend,
                "pool_spend_multiplier": pool_mult,
                "auction_aggression_scale": aggression,
                "reserve_padding_scale": reserve_scale,
            }
            write_runtime_config(config_path, config)
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                f"Saved overrides at {timestamp}.",
            )

        write_runtime_config(config_path, DEFAULT_CONFIG)
        return (
            bool(DEFAULT_CONFIG["force_optimal"]),
            float(DEFAULT_CONFIG["spend_fraction_scale"]),
            float(DEFAULT_CONFIG["pool_spend_multiplier"]),
            float(DEFAULT_CONFIG["auction_aggression_scale"]),
            float(DEFAULT_CONFIG["reserve_padding_scale"]),
            f"Reset to defaults at {timestamp}.",
        )

    @app.callback(
        Output("points-graph", "figure"),
        Output("gold-graph", "figure"),
        Output("run-info", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def refresh_graphs(_: int):
        series = parse_log(log_path, agent_id)
        current_config = read_runtime_config(config_path)
        if not series:
            return (
                build_placeholder_figure("Waiting for data..."),
                build_placeholder_figure("Waiting for data..."),
                "No log data yet. Ensure the log path is correct and rounds are running.",
            )
        points_fig = build_points_figure(series)
        gold_fig = build_gold_figure(series)
        info = html.Div(
            [
                html.Span(f"Rounds tracked: {len(series['rounds'])}"),
                html.Span(f"Latest round: {int(series['rounds'][-1])}"),
                html.Span(f"Current points: {series['points'][-1]:.0f}"),
                html.Span(f"Gold: {series['gold'][-1]:.0f}"),
                html.Span(
                    "Bootstrap override: ON" if current_config["force_optimal"] else "Bootstrap override: OFF"
                ),
            ],
            className="run-info-grid",
        )
        return points_fig, gold_fig, info

    @app.callback(Output("spend-label", "children"), Input("spend-slider", "value"))
    def show_spend_label(value: float) -> str:
        return f"{value:.2f}×"

    @app.callback(Output("pool-label", "children"), Input("pool-slider", "value"))
    def show_pool_label(value: float) -> str:
        return f"{value:.2f}×"

    @app.callback(Output("aggression-label", "children"), Input("aggression-slider", "value"))
    def show_aggression_label(value: float) -> str:
        return f"{value:.2f}×"

    @app.callback(Output("reserve-label", "children"), Input("reserve-slider", "value"))
    def show_reserve_label(value: float) -> str:
        return f"{value:.2f}×"

    return app


def main() -> None:
    args = parse_args()
    log_path = args.log.resolve()
    config_path = args.config.resolve()
    app = build_app(log_path, args.agent_id, config_path, args.refresh_ms)
    try:
        app.run_server(host=args.host, port=args.port, debug=False)
    except OSError as exc:
        print(f"[ERROR] Dash failed to start: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
