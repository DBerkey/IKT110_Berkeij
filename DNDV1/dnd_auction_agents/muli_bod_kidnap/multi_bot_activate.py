#!/usr/bin/env python3
"""
Central Bot Manager - startet 100 einzelne Auction Bots
Speichert stdout/stderr jedes Bots in logs/.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

class BotManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent.resolve()
        # Name der Individual-Bot-Datei (must exist)
        self.individual_bot_script = self.base_dir / "mulit_bot_individual.py"
        self.bot_processes = []

    def generate_bot_configs(self):
        configs = []
        for auction_rank in range(10):          # 0..9 (0 = beste Auktion)
            for turn_in_cycle in range(1, 11):  # 1..10
                bot_id = auction_rank * 10 + (turn_in_cycle - 1)
                configs.append({
                    "bot_id": bot_id,
                    "target_auction_rank": auction_rank,
                    "bid_turn_in_cycle": turn_in_cycle
                })
        return configs

    def start_individual_bot(self, cfg, delay=0.0, extra_args=None):
        if delay:
            time.sleep(delay)
        bot_id = cfg["bot_id"]
        target = cfg["target_auction_rank"]
        turn = cfg["bid_turn_in_cycle"]

        cmd = [
            sys.executable,
            str(self.individual_bot_script),
            str(bot_id),
            str(target),
            str(turn)
        ]
        if extra_args:
            cmd += extra_args

        # ensure logs dir
        logs_dir = self.base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        stdout_path = logs_dir / f"bot_{bot_id:03d}.out.log"
        stderr_path = logs_dir / f"bot_{bot_id:03d}.err.log"

        out_f = open(stdout_path, "a", encoding="utf-8")
        err_f = open(stderr_path, "a", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdout=out_f,
            stderr=err_f,
            encoding='utf-8',
            errors='replace'
        )
        print(f"Started bot {bot_id:3d} (rank {target+1}, turn {turn}) -> logs/{stdout_path.name}")
        # keep file handles so they don't get garbage-collected
        self.bot_processes.append((proc, out_f, err_f))
        return proc

    def start_all_bots(self, stagger_delay=0.02, dry_run=False, extra_args=None):
        configs = self.generate_bot_configs()
        if dry_run:
            print("Dry run: would start these bots:")
            for c in configs[:20]:
                print(c)
            return True

        if not self.individual_bot_script.exists():
            print(f"ERROR: individual bot script not found: {self.individual_bot_script}")
            return False

        for i, cfg in enumerate(configs):
            self.start_individual_bot(cfg, delay=stagger_delay, extra_args=extra_args)
            if (i+1) % 20 == 0:
                print(f"Progress: {i+1}/{len(configs)}")

        print("All start commands issued.")
        return True

    def cleanup(self):
        print("Stopping all child processes...")
        for proc, out_f, err_f in self.bot_processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
            # close filehandles
            try:
                out_f.close()
                err_f.close()
            except Exception:
                pass
        print("Cleanup done.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--stagger", type=float, default=0.02, help="Sekunden zwischen Start der Bots")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-in", action="store_true", help="Übergebe --all-in an die Bots (setzen alles)")
    parser.add_argument("--max-rounds", type=int, default=None, help="(optional) Anzahl Runden, in denen Bots aktiv sind")
    args = parser.parse_args()

    extra = []
    if args.all_in:
        extra.append("--all-in")
    if args.max_rounds:
        extra += ["--max-rounds", str(args.max_rounds)]

    manager = BotManager()
    try:
        manager.start_all_bots(stagger_delay=args.stagger, dry_run=args.dry_run, extra_args=extra)
        if not args.no_monitor:
            print("Press Ctrl+C to stop and cleanup.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        manager.cleanup()

if __name__ == "__main__":
    main()
