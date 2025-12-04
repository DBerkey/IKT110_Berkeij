#!/usr/bin/env python3
"""Utility launcher for spinning up local DND auction game sessions."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ROUNDS = 1000
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
PLAY_TOKEN = "play123"
GAME_TOKEN = "play123"

# Customize which agents to launch. Paths are relative to this file's folder.
AGENT_CONFIGS: List[Dict[str, object]] = [
    {"name": "Adaptive v1", "path": "dnd_agents/run_adaptive_v1.py", "enabled": True},
    {"name": "Aggressive Bot", "path": "dnd_agents/run_aggressive_bot.py", "enabled": True},
    {"name": "Aggressive Debt Bot", "path": "dnd_agents/run_aggressive_debt_bot.py", "enabled": True},
    {"name": "Smart Ultimate", "path": "dnd_agents/run_smart_ultimate_agent.py", "enabled": True},
    {"name": "Balanced Bot", "path": "dnd_agents/run_balanced_bot.py", "enabled": True},
    {"name": "Random Single", "path": "dnd_auction_game/example_agents/agent_random_single.py", "enabled": True},
    {"name": "Tiny Bid", "path": "dnd_auction_game/example_agents/agent_tiny_bid.py", "enabled": True},
    {"name": "Interest Split", "path": "dnd_agents/run_interest_split_agent.py", "enabled": True},
    {"name": "Braavos Optimal", "path": "dnd_agents/run_braavos_optimal_agent.py", "enabled": True},
    {"name": "Braavos Sentinel", "path": "dnd_agents/run_braavos_sentinel_agent.py", "enabled": True},
    {"name": "Braavos Phoenix", "path": "dnd_agents/run_braavos_phoenix_agent.py", "enabled": True},
    {"name": "Braavos Learning", "path": "dnd_agents/run_braavos_learning_agent.py", "enabled": True},
    {"name": "Braavos lockdown", "path": "dnd_agents/run_braavos_lockdown_agent.py", "enabled": True},
    {"name": "Braavos Centurion", "path": "dnd_agents/run_braavos_centurion_agent.py", "enabled": True},
    {"name": "Braavos Legion", "path": "dnd_agents/run_braavos_legion_agent.py", "enabled": True},
    {"name": "Simple Mean Bidder", "path": "dnd_agents/run_simple_mean_bidder_agent.py", "enabled": True},
    {"name": "Pool Mean Bidder", "path": "dnd_agents/run_pool_mean_bidder_agent.py", "enabled": True},
    {"name": "Fixed Spread Bidder", "path": "dnd_agents/run_fixed_spread_bidder_agent.py", "enabled": True},
    {"name": "Random Splash Bidder", "path": "dnd_agents/run_random_splash_bidder_agent.py", "enabled": True},
    {"name": "Better mean Bidder", "path": "dnd_agents/run_better_mean_bidder_agent.py", "enabled": True},
    {"name": "Neural Bidder", "path": "dnd_agents/run_neural_bidder_agent.py", "enabled": True},
    {"name": "agent_maxi", "path": "dnd_agents/agent_maxi.py", "enabled": True},
    {"name": "agent_victor", "path": "dnd_agents/agent_run_victor.py", "enabled": True},
    {"name": "agent_third", "path": "dnd_agents/agent_third.py", "enabled": True},
    {"name": "agent_final_controlled", "path": "dnd_agents/agent_final_controlled.py", "enabled": True},
    {"name": "agent_aggressive_spender", "path": "dnd_agents/agent_aggressive_spender.py", "enabled": True},
    {"name": "vinner1", "path": "dnd_agents/vinner1.py", "enabled": True},
    {"name": "run_logic", "path": "dnd_agents/run.py", "enabled": True},
    {"name": "Compound Crusher", "path": "dnd_agents/run_compound_crusher_agent.py", "enabled": True},
    {"name": "Compound Crusher v2", "path": "dnd_agents/melk.py", "enabled": True},
    {"name": "Moha_2", "path": "dnd_agents/run_2.py", "enabled": True},
]


@dataclass
class AgentProcess:
    name: str
    process: subprocess.Popen


class GameLauncher:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.server_process: Optional[subprocess.Popen] = None
        self.agents: List[AgentProcess] = []

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _resolve_agent_configs(self) -> List[Dict[str, object]]:
        resolved: List[Dict[str, object]] = []
        for cfg in AGENT_CONFIGS:
            if not cfg.get("enabled", False):
                continue
            agent_path = (self.base_dir / str(cfg["path"]))
            if not agent_path.exists():
                print(f"[WARN] Missing agent path: {agent_path}")
                continue
            resolved.append({
                "name": cfg["name"],
                "path": agent_path,
                "is_file": agent_path.is_file(),
            })
        return resolved

    def _start_server(self, host: str, port: int) -> bool: 
        if self.server_process and self.server_process.poll() is None:
            print("[INFO] Server already running")
            return True

        env = os.environ.copy()
        env["AH_GAME_TOKEN"] = GAME_TOKEN
        env["AH_PLAY_TOKEN"] = PLAY_TOKEN

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "dnd_auction_game.server:app",
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "warning",
        ]

        print(f"[INFO] Starting server on {host}:{port}")
        self.server_process = subprocess.Popen(cmd, cwd=self.base_dir, env=env)

        if not self._wait_for_port(host, port, timeout=10):
            print("[ERROR] Server failed to bind within timeout")
            self._stop_process(self.server_process)
            self.server_process = None
            return False

        print("[INFO] Server is live")
        return True

    def _wait_for_port(self, host: str, port: int, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with socket.socket() as sock:
                sock.settimeout(1.0)
                try:
                    sock.connect((host, port))
                except OSError:
                    time.sleep(0.3)
                    continue
                return True
        return False

    def _start_agent(self, cfg: Dict[str, object]) -> Optional[subprocess.Popen]:
        path: Path = cfg["path"]  # type: ignore[assignment]
        if path.is_dir():
            script = path / "main.py"
            if not script.exists():
                print(f"[WARN] No entry point for directory agent {path}")
                return None
        else:
            script = path

        cmd = [sys.executable, str(script)]
        try:
            proc = subprocess.Popen(cmd, cwd=self.base_dir)
            print(f"[INFO] Agent '{cfg['name']}' launched")
            return proc
        except OSError as exc:
            print(f"[ERROR] Unable to launch agent {cfg['name']}: {exc}")
            return None

    def _launch_agents(self) -> bool:
        configs = self._resolve_agent_configs()
        if not configs:
            print("[ERROR] No enabled agents could be resolved")
            return False

        for cfg in configs:
            proc = self._start_agent(cfg)
            if proc:
                self.agents.append(AgentProcess(name=str(cfg["name"]), process=proc))

        if not self.agents:
            print("[ERROR] Failed to start any agents")
            return False

        time.sleep(1.5)  # allow agents to connect
        return True

    def _run_game(self, rounds: int, host: str) -> bool:
        time.sleep(15) 
        cmd = [
            sys.executable,
            "-m",
            "dnd_auction_game.play",
            str(rounds),
            PLAY_TOKEN,
        ]
        print(f"[INFO] Running game for {rounds} rounds on {host}")
        try:
            subprocess.check_call(cmd, cwd=self.base_dir)
            print("[INFO] Game completed")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Game runner returned {exc.returncode}")
            return False

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------
    def _stop_process(self, proc: Optional[subprocess.Popen]) -> None:
        if not proc:
            return
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def stop_agents(self) -> None:
        for agent in self.agents:
            self._stop_process(agent.process)
        self.agents.clear()

    def stop_server(self) -> None:
        self._stop_process(self.server_process)
        self.server_process = None

    def cleanup(self) -> None:
        self.stop_agents()
        self.stop_server()

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------
    def launch(self, *, rounds: int, host: str, port: int, keep_server: bool) -> bool:
        try:
            if not self._start_server(host, port):
                return False
            if not self._launch_agents():
                return False
            if not self._run_game(rounds, host):
                return False

            print("[INFO] Session finished. Check http://{}:{}/ for leaderboard.".format(host, port))
            self._wait_for_user_exit(keep_server)
        finally:
            if not keep_server:
                self.cleanup()
            else:
                self.stop_agents()
        return True

    def _wait_for_user_exit(self, keep_server: bool) -> None:
        if keep_server:
            prompt = "Press Enter to stop the server and exit..."
        else:
            prompt = "Press Enter to shut everything down..."

        try:
            input(prompt)
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt detected; shutting down.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DND auction test launcher")
    parser.add_argument("--rounds", "-r", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--server-only", action="store_true")
    parser.add_argument("--list-agents", action="store_true")
    return parser.parse_args()


def list_agents(launcher: GameLauncher) -> None:
    print("Configured agents:")
    for idx, cfg in enumerate(AGENT_CONFIGS, 1):
        absolute = (launcher.base_dir / str(cfg["path"]))
        exists = "OK" if absolute.exists() else "MISSING"
        enabled = "ENABLED" if cfg.get("enabled", False) else "DISABLED"
        print(f" {idx:02d}. {cfg['name']:<16} {enabled:<8} {exists:<8} -> {absolute}")


def main() -> None:
    args = parse_args()
    launcher = GameLauncher()

    if args.list_agents:
        list_agents(launcher)
        return

    if args.server_only:
        try:
            launcher._start_server(args.host, args.port)
            print(f"Server at http://{args.host}:{args.port}/ running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[INFO] Server-only mode stopped.")
        finally:
            launcher.cleanup()
        return

    success = launcher.launch(
        rounds=args.rounds,
        host=args.host,
        port=args.port,
        keep_server=args.keep_server,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
